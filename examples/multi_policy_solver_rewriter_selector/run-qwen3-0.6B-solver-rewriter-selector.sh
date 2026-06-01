#!/bin/bash

# clean up before rerun
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# prevent ray from buffering stdout/stderr
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

# Per-policy fields live in config.yaml; globals (rollout cadence, wandb) stay CLI args.

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.multi_policy_solver_rewriter_selector.rollout_with_multi_agents.generate_with_multi_agents
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
   --rollout-max-response-len 16384
   --rollout-temperature 1
   --balance-data
)
# n_samples_per_prompt / global_batch_size come from config.yaml's first policy
# (via _set_multi_policy_global_defaults); passing them on CLI would shadow per-policy values.

# Cluster sizing from config.yaml: 3 actor + 3 sglang GPUs.
#   colocate = max(3, 3) = 3; separate = 6. (4-L40S box: 1 spare.)
NUM_GPUS=3

TRAIN_ARGS=(
   --config "${SCRIPT_DIR}/config.yaml"
   --save-interval 5
   --train-memory-margin-bytes 0
   # Per-role dumps: <dump-details>/<policy_name>/{rollout,train,packed}_data/...
   --dump-details /tmp/multi_policy_solver_rewriter_selector/dump_details
)
# rollout_num_gpus is derived from config.yaml, so no --rollout-num-gpus needed.
# config.yaml is the single source of truth (one entry per policy).

EVAL_ARGS=(
   --eval-interval 2
   --eval-config "${SCRIPT_DIR}/eval_config.yaml"
   --eval-function-path examples.multi_policy_solver_rewriter_selector.eval_fn.eval_with_multi_agents
   --eval-max-response-len 16384
   --eval-top-p 1
)

# Perf/optimizer flags are per-policy in config.yaml.

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-0.6B-solver-rewriter-selector
)

# sglang server args are per-policy in config.yaml (sglang sub-block).
# Megatron numerical/dropout flags are per-policy too (RL-correctness invariants
# are policy-level, not run-level).

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — torch_memory_saver
# (sglang colocate pause/resume) won't init with it. We need offload, so skip it.

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_multi_policy.py \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${TRAIN_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${EVAL_ARGS[@]}
