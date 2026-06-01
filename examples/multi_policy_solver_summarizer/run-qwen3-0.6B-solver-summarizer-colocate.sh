#!/bin/bash

# Colocate variant of run-qwen3-0.6B-solver-summarizer-nocolocate.sh.
#
# Same example, same task, same policies (solver + summarizer). The only
# difference is `--colocate`: each policy's Megatron actor and SGLang
# engine share GPUs via fractional Ray resources, with offload/onload
# swapping between train and rollout phases.
#
# Cluster sizing (with --colocate), tuned for a single 8xH200 node:
#   actor_gpus   = sum(megatron_num_nodes * num_gpus_per_node) = 2 × 1 × 4 = 8
#   rollout_gpus = sum(sglang_num_nodes   * num_gpus_per_node) = 2 × 1 × 4 = 8
#   total        = max(actor_gpus, rollout_gpus)               = 8
# Layout: GPUs 0-3 host solver     (TP-2×DP-2 Megatron + 4 SGLang engines, timeshared)
#         GPUs 4-7 host summarizer (TP-2×DP-2 Megatron + 4 SGLang engines, timeshared)
#
# H200 tuning lives in config-colocate.yaml: mem_fraction_static 0.5 (sglang and
# the trainer timeshare each policy's 4 GPUs — sglang is paused during the train
# step, the trainer is offloaded during rollout, so 0.5 is safe on 141GB),
# max_tokens_per_gpu 32768 (fits a full-context sample + better packing),
# max_running_requests 256 and cuda_graph_bs up to 64 for the wider rollout.

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-0.6B.sh"

# Per-policy fields (parallel, recompute, batching, optimizer, loss, paths,
# Megatron numerical / dropout, log_probs_chunk_size) all live in config.yaml.
# Run-level orchestration (rollout cadence, wandb) stays as CLI args.

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.multi_policy_solver_summarizer.rollout_with_multi_agents.generate_with_multi_agents
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --disable-rollout-trim-samples
   --rollout-max-context-len 32768
   --rollout-max-response-len 32768
   --rollout-temperature 1
   --balance-data
)

NUM_GPUS=8

TRAIN_ARGS=(
   --config "${SCRIPT_DIR}/config-colocate.yaml"
   --save-interval 5
   # Per-role rollout/train data dumps land under
   #   <dump-details>/<policy_name>/rollout_data/<rollout_id>.pt
   #   <dump-details>/<policy_name>/train_data/<rollout_id>_<rank>.pt
   #   <dump-details>/<policy_name>/packed_data/<rollout_id>_<rank>.pt
   --dump-details /tmp/multi_policy_solver_summarizer/dump_details_colocate
)
# Note: --colocate auto-enables offload_train + offload_rollout
# (slime/utils/arguments.py:1777-1781).

EVAL_ARGS=(
   # AIME-2024 via eval_config.yaml. Custom eval function emits four
   # per-prompt aggregates per dataset: best-of-4 and mean for each
   # role. --log-passrate intentionally not set; it would also trigger
   # train-side pass-rate logging whose group_size assertion does not
   # hold when the chain emits num_parallel samples per call.
   --eval-interval 2
   --eval-config "${SCRIPT_DIR}/eval_config.yaml"
   --eval-function-path examples.multi_policy_solver_summarizer.eval_fn.eval_with_multi_agents
   --eval-max-response-len 32768
   --eval-top-p 1
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-0.6B-solver-summarizer-colocate
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here —
# torch_memory_saver (which sglang uses for colocate-mode pause/resume)
# refuses to initialize when expandable_segments is enabled. Choose one:
# colocate offload OR expandable segments. We need offload, so we skip
# expandable segments and rely on the other knobs (smaller chunk sizes,
# zero margin, smaller mem_fraction) for fragmentation control.

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_multi_policy.py \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${TRAIN_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${EVAL_ARGS[@]}
