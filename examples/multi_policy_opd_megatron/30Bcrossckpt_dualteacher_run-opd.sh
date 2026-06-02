#!/bin/bash

# Multi-policy OPD — 30B-A3B cross-checkpoint, dual teacher (Megatron + SGLang, same teacher weights).
#   student          : trainable, Qwen3-30B-A3B-Instruct-2507 (paired Megatron + SGLang)
#   teacher_megatron : frozen Megatron, Qwen3-30B-A3B-Thinking-2507 — drives OPD (opd_type=megatron_actor)
#   teacher_sglang   : frozen SGLang,   Qwen3-30B-A3B-Thinking-2507 — diagnostic logp (Sample.teacher_sglang_log_probs)
#
# Cluster: 4 nodes × 8 GPU = 32 GPU, NO colocate. Fits via student/teacher
# megatron EP8 DP1 (8 GPU each) — see 30Bcrossckpt_dualteacher_config.yaml
# header for the GPU accounting and the DP1-vs-DP2 memory trade-off.
#   actor(student M 8 + teacher_megatron M 8) + rollout(student S 8 + teacher_sglang S 8) = 32
#
# Schedule: --num-rollout 1000, save-interval 50.

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

# Both student and teacher are Qwen3-30B-A3B (Instruct vs Thinking 2507), identical MoE
# architecture, rope_theta=10000000 (verified from HF configs).
export MODEL_ARGS_ROTARY_BASE=10000000
source "${SCRIPT_DIR}/../../scripts/models/qwen3-30B-A3B.sh"

NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

echo "Checking Ray cluster status..."
ray status 2>&1 | head -20

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 1000
   --rollout-batch-size 32
   --rollout-max-context-len 32768
   --rollout-max-response-len 16384
   --rollout-temperature 1
   --balance-data
   # Custom rollout: student SGLang generation + POST teacher_sglang for per-token
   # logprobs (stamps Sample.teacher_sglang_log_probs); OPD signal itself comes
   # from teacher_megatron via external_data.
   --custom-generate-function-path examples.multi_policy_opd_sglang.rollout_with_teacher_sglang.generate_with_teacher_sglang
)

TRAIN_ARGS=(
   --config "${SCRIPT_DIR}/30Bcrossckpt_dualteacher_config.yaml"
   --num-gpus-per-node ${NUM_GPUS_PER_NODE}
   --save-interval 50
   --dump-details /tmp/multi_policy_opd_dualteacher_30Bcrossckpt/dump_details
)

EVAL_ARGS=(
   --eval-interval 20
   # aime-2024, student-only eval (eval_student.generate_eval_student, set via
   # the eval-config's per-dataset custom_generate_function_path) — overrides the
   # run-level generate_with_teacher_sglang at eval, so no teacher_sglang POST.
   # No `policies:` field in the yaml → legacy resolver routes to the student engine.
   --eval-config "${SCRIPT_DIR}/30Bcrossckpt_dualteacher_eval_aime_config.yaml"
   --eval-max-response-len 16384
   --eval-top-p 1
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-n-opd-mega-dual
   --wandb-group qwen3-30b-a3b-instruct-from-think
)

RUNTIME_ENV_JSON="{
  \"working_dir\": \"/root/slime-n\",
  \"excludes\": [\".git\", \"docs\", \"imgs\", \"docker\", \"slime_plugins\", \"*.egg-info\", \"__pycache__\"],
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TRITON_CACHE_DIR\": \"/tmp/triton_cache_30Bcrossckpt_dualteacher\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 /root/slime-n/train_multi_policy.py \
   ${MODEL_ARGS[@]} \
   ${TRAIN_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${EVAL_ARGS[@]}
