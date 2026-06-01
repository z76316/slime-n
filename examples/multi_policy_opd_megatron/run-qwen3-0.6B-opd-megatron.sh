#!/bin/bash

# Multi-policy OPD with a Megatron-backend teacher.
#
#   student          : trainable, paired Megatron + SGLang (m✓ s✓)
#   teacher_megatron : frozen, standalone Megatron (m✓ s✗ trainable=false)
#                      runs forward-only, returns teacher_log_probs as
#                      external_data for the student's KL term.
#
# Cluster: 3 GPUs total — 1 student megatron + 1 teacher megatron + 1 student sglang.

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
# Student and teacher share Megatron architecture (both Qwen3-0.6B); the
# teacher is a different fine-tune of the same base, loaded from a different
# hf_checkpoint. MODEL_ARGS is sourced once and applies to both policies.
source "${SCRIPT_DIR}/../../scripts/models/qwen3-0.6B.sh"

# Per-policy fields (parallel, recompute, batching, optimizer, loss, paths,
# Megatron numerical / dropout, log_probs_chunk_size, OPD flags) all live in
# config.yaml. Run-level orchestration (rollout cadence, wandb) stays as CLI args.

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 300
   --rollout-batch-size 16
   --rollout-max-context-len 32768
   --rollout-max-response-len 32768
   --rollout-temperature 1
   --balance-data
)
# n_samples_per_prompt / global_batch_size are projected onto manager-global
# args from config.yaml's first trainable policy by _set_multi_policy_global_defaults.

# Cluster sizing — derived from config.yaml (NO colocate):
#   actor_gpus   = sum(megatron_num_nodes * num_gpus_per_node) = 2 × 1 × 1 = 2
#                  (student + teacher both have a Megatron actor)
#   rollout_gpus = sum(sglang_num_nodes   * num_gpus_per_node) = 1 × 1 × 1 = 1
#                  (only student has a paired SGLang engine; teacher.sglang_num_nodes=0)
#   total        = actor_gpus + rollout_gpus                   = 3
# Layout: GPU 0 = student megatron, GPU 1 = teacher megatron, GPU 2 = student sglang.
NUM_GPUS=3

TRAIN_ARGS=(
   --config "${SCRIPT_DIR}/config.yaml"
   --save-interval 20
   # Per-role rollout / train / packed data dumps land under
   #   <dump-details>/<policy_name>/rollout_data/<rollout_id>.pt
   #   <dump-details>/<policy_name>/train_data/<rollout_id>_<rank>.pt
   #   <dump-details>/<policy_name>/packed_data/<rollout_id>_<rank>.pt
   --dump-details /tmp/multi_policy_opd_megatron/dump_details
)

EVAL_ARGS=(
   # --eval-interval 20
   # --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 16
   # --eval-max-response-len 32768
   # --eval-top-p 1
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-0.6B-opd-megatron
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

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_multi_policy.py \
   ${MODEL_ARGS[@]} \
   ${TRAIN_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${EVAL_ARGS[@]}


####clear after training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
