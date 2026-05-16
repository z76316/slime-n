#!/bin/bash

# Multi-policy PPO — asymmetric actor + critic.
#
#   actor   : trainable, paired Megatron + SGLang (m✓ s✓), Qwen3-1.7B
#   critic  : trainable, standalone Megatron (m✓ s✗), Qwen3-0.6B
#             Runs train_critic (forward + value-loss + backward), returns
#             per-token `values` as external_data for the actor.
#
# Cluster: 3 GPUs total — 1 actor megatron + 1 critic megatron + 1 actor sglang.

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
# CLI-global MODEL_ARGS is sourced from the *actor* (Qwen3-1.7B). Per-policy
# arch overrides for the critic (Qwen3-0.6B) live in config.yaml's
# `extra_megatron_args` (auto-captured from the `megatron:` YAML block) —
# config_to_namespace projects them onto the
# critic's per-policy args namespace after CLI parse.
source "${SCRIPT_DIR}/../../scripts/models/qwen3-1.7B.sh"

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
   --rollout-max-response-len 16384
   --rollout-temperature 1
   --balance-data
)

# Cluster sizing — derived from config.yaml (NO colocate):
#   actor_gpus   = sum(megatron_num_nodes * num_gpus_per_node) = 2 × 1 × 1 = 2
#   rollout_gpus = sum(sglang_num_nodes   * num_gpus_per_node) = 1 × 1 × 1 = 1
#   total                                                      = 3
NUM_GPUS=3

TRAIN_ARGS=(
   --config "${SCRIPT_DIR}/config.yaml"
   --save-interval 20
   --dump-details /tmp/multi_policy_ppo/dump_details
)

EVAL_ARGS=(
   # --eval-interval 20
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-1.7B-0.6B-ppo
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
