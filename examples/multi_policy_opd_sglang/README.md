# Multi-Policy OPD (frozen SGLang teacher)

On-policy distillation where the teacher is served by **SGLang** (inference engine) instead of being loaded into a Megatron training actor. The student generates rollouts; for each generated sample, the rollout function POSTs the response tokens to the teacher's SGLang server with `return_logprobs=True` (prefix-only scoring, no new generation) and stamps `sample.teacher_log_probs`. The student's loss adds a reverse-KL term against those logprobs.

![architecture: student pair + frozen SGLang teacher](./imgs/arch.png)

*Trainable **student** pair (Megatron + SGLang) plus a frozen **teacher** standalone SGLang engine — no Megatron actor. The rollout function POSTs response tokens to the teacher's SGLang URL and receives logprobs back.*

## Why this variant

- **Cheaper teacher deployment**: SGLang inference is faster and supports quantization. A larger teacher can fit on the same hardware.
- **Architecture independence** (vs Megatron-backed teacher): the teacher only needs to run inference, so it can be a completely different model family from the student.

**Trade-off vs `multi_policy_opd_megatron`**: SGLang's inference kernels disagree with Megatron's training-time forward on the same weights. The resulting kernel mismatch is a noise floor on the KL term — distillation still works, just slightly noisier than the kernel-clean Megatron-backed variant. Pick this when teacher size/cost matters more than KL fidelity.

## Files

* `config.yaml`: student (paired) + teacher_sglang (standalone engine, `m✗ s✓ trainable=false`).
* `run-qwen3-0.6B-opd-sglang.sh`: launch script.
* `rollout_with_teacher_sglang.py`: custom RM that queries the teacher's framework-managed SGLang URL via `get_model_url(args, "teacher_sglang")`.
