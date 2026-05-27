#!/bin/bash

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

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.multi_policy_shared_state.rollout_with_shared_state.generate_with_shared_state
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

NUM_GPUS=4

TRAIN_ARGS=(
   --config "${SCRIPT_DIR}/config.yaml"
   --save-interval 5
   --rollout-data-postprocess-path examples.multi_policy_shared_state.postprocess.rollout_data_postprocess
   --dump-details /tmp/multi_policy_shared_state/dump_details
)

EVAL_ARGS=(
   --eval-interval 2
   --eval-config "${SCRIPT_DIR}/eval_config.yaml"
   --eval-function-path examples.multi_policy_shared_state.eval_fn.eval_with_shared_state
   --eval-max-response-len 16384
   --eval-top-p 1
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-0.6B-shared-state
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
