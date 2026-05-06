# Multi-Policy OPD (frozen Megatron teacher)

Multi-policy on-policy distillation: a trainable **student** generates rollouts; a frozen **teacher_megatron** runs forward-only on those rollouts and emits per-token logprobs that feed a reverse-KL term (`KL(student ‖ teacher)`) into the student's loss. Because the teacher is its own Megatron Ray actor (separate weights, separate GPU), its forward kernels match the student's — keeping the KL noise floor low.

## Files

* `config.yaml`: student + teacher_megatron policy schema.
* `run-qwen3-0.6B-opd-megatron.sh`: launch script (ray start + train_multi_policy.py).

## Quick Start

```bash
cd slime
bash examples/multi_policy_opd_megatron/run-qwen3-0.6B-opd-megatron.sh
```

Place a Qwen3-0.6B HF checkpoint at `/root/Qwen3-0.6B`, a different fine-tune at `/root/Qwen3-0.6B-teacher`, and `dapo-math-17k.jsonl` at `/root/dapo-math-17k/`.

## How It Works

* `teacher.train()` runs forward-only via the existing `compute_log_prob` primitive and returns `{"teacher_log_probs": ...}`.
* The driver runs frozen producers first on the trainable policy's rollout data, then merges all producer outputs into the student's `external_data`.
* `train_actor` writes `external_data["teacher_log_probs"]` into `rollout_data` before `compute_advantages_and_returns`, where `apply_opd_kl_to_advantages` consumes it.
* Per-role rollout / train / packed-data dumps land at `/tmp/multi_policy_opd_megatron/dump_details/<policy_name>/...`.

## Policies

| policy | megatron | sglang | trainable | role |
|---|---|---|---|---|
| `student` | ✓ | ✓ | ✓ | paired pipeline; generates rollouts |
| `teacher_megatron` | ✓ | ✗ | ✗ | standalone actor; per-token logprobs |

Cluster: 3 GPUs (1 student megatron + 1 teacher megatron + 1 student sglang, no colocate).

## Compared to legacy single-policy OPD

`examples/on_policy_distillation/` loads the teacher as an in-process tag inside a single Megatron actor (`_switch_model("teacher")`). This multi-policy version puts the teacher in its own Ray actor — scaling teacher size independently of the student and providing the foundation for multi-teacher setups.
