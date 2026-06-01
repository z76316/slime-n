#!/bin/bash

# usage:
#   On head node (NODE_RANK=0):
#     MASTER_ADDR=<head-ip> NODE_RANK=0 NUM_NODES=17 NUM_GPUS_PER_NODE=6 \
#       bash examples/multi_policy_exam_swarm_rl/run-qwen3-0.6B-exam-swarm-colocate-102.sh
#   On every worker node (NODE_RANK=1..NUM_NODES-1):
#     MASTER_ADDR=<head-ip> NODE_RANK=$RANK NUM_NODES=17 NUM_GPUS_PER_NODE=6 \
#       bash examples/multi_policy_exam_swarm_rl/run-qwen3-0.6B-exam-swarm-colocate-102.sh

# 102 homogeneous agents, multi-node 102-GPU --colocate.
#
# Cluster: 102 GPUs total. NUM_GPUS_PER_NODE must divide 102.
# Divisors of 102: 1, 2, 3, 6, 17, 34, 51, 102. Common shapes:
#   - 17 nodes × 6 GPUs   (default)
#   - 34 nodes × 3 GPUs
#   - 51 nodes × 2 GPUs
# 8-GPU nodes do NOT divide 102 (102 % 8 = 6). Use config-102.yaml only
# on clusters whose per-node GPU count is a divisor of 102; otherwise
# edit N to 96 (12×8) or 104 (13×8) in this script + config-102.yaml +
# agent_system.py:N_AGENTS.
#
# Per-trajectory advantage is composed in agent_system.py and stored as
# Sample.reward. --disable-rewards-normalization is required so slime
# does not re-normalize.

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

# Multi-node config — set via environment, override defaults if needed.
# Defaults to 17 nodes × 6 GPUs/node = 102 GPUs total.
NUM_NODES=${NUM_NODES:-17}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-6}
NODE_RANK=${NODE_RANK:-0}
NUM_GPUS_TOTAL=$((NUM_NODES * NUM_GPUS_PER_NODE))

if [ "$NUM_GPUS_TOTAL" -ne 102 ]; then
    echo "WARNING: NUM_NODES × NUM_GPUS_PER_NODE = $NUM_GPUS_TOTAL, not 102"
    echo "         Adjust config-102.yaml + agent_system.py:N_AGENTS to match."
fi

if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: MASTER_ADDR must be set to the head node's IP"
    exit 1
fi

# Ray cluster setup: head on NODE_RANK=0, workers on the rest.
if [ "$NODE_RANK" -eq 0 ]; then
    ray start --head --node-ip-address ${MASTER_ADDR} \
        --num-gpus ${NUM_GPUS_PER_NODE} \
        --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
else
    ray start --address="${MASTER_ADDR}:6379" \
        --num-gpus ${NUM_GPUS_PER_NODE}
    # Worker nodes wait for the head to submit the job and run.
    sleep infinity
fi

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.multi_policy_exam_swarm_rl.rollout_with_swarm.generate_with_swarm
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 8                  # 8 prompts × K=8 = 64 per-agent samples = global_batch_size
   --disable-rollout-trim-samples
   --rollout-max-context-len 32768
   --rollout-max-response-len 32768
   --rollout-temperature 1.0
   --rollout-top-p 0.95
   --balance-data
   # Required: agent_system.py already composed self_adv via GRPO group-norm.
   --disable-rewards-normalization
)

TRAIN_ARGS=(
   --config "${SCRIPT_DIR}/config-102.yaml"
   --num-gpus-per-node ${NUM_GPUS_PER_NODE}
   --save-interval 50
   --train-memory-margin-bytes 0
   --dump-details /tmp/multi_policy_exam_swarm_rl/dump_details_102
)

EVAL_ARGS=(
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 32768
   --eval-top-p 1
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-0.6B-exam-swarm-colocate-102
)

# Wait for all worker nodes to join the Ray cluster before submitting.
echo "Waiting for ${NUM_GPUS_TOTAL} GPUs to join the Ray cluster..."
until ray status 2>/dev/null | grep -q "${NUM_GPUS_TOTAL}\.0/${NUM_GPUS_TOTAL}\.0 GPU"; do
    sleep 5
    echo "  ray status: $(ray status 2>/dev/null | grep GPU | head -1)"
done
echo "All ${NUM_GPUS_TOTAL} GPUs joined."

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
   --colocate \
   ${MODEL_ARGS[@]} \
   ${TRAIN_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${EVAL_ARGS[@]}
