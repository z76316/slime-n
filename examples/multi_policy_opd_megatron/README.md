# Multi-Policy On-Policy Distillation (Megatron teacher)

Trainable **student** generates rollouts; frozen **teacher** runs forward-only
on the student's tokens to produce per-token log-probs. The student's loss
adds a reverse-KL term against those teacher log-probs (`KL(student ‖ teacher)`).

This example uses a **Megatron-backend teacher** — the teacher is a separate
Megatron actor (its own Ray actor, its own GPU, its own weights), not an
in-process tag teacher. Numerically the teacher's forward kernels match the
student's, so the KL noise floor is much lower than the SGLang-teacher
alternative.

## Cluster layout

```
┌─── student (trainable, paired) ──────┐    ┌─── teacher_megatron (frozen) ──┐
│  Megatron actor   ⇆   SGLang engine  │    │       Megatron actor             │
│       1 GPU              1 GPU       │    │            1 GPU                 │
│                                      │    │      (no SGLang engine)          │
└───────────┬──────────────────────────┘    └──────────────┬───────────────────┘
            │ rollouts (student generates only)            │ forward-only on
            │                                              │ the student's tokens
            ▼                                              ▼
   train (KL loss)  ◀──── external_data["teacher_log_probs"] ──────┘
```

3 GPUs total. The teacher returns `{"teacher_log_probs": ...}` from its
forward-only train; the multi-policy driver routes that dict as `external_data`
to the student, which merges it into `rollout_data` and the reverse-KL term
folds into the student's advantages.

## Files

- `config.yaml` — policy schema for student + teacher_megatron.
- `run-qwen3-0.6B-opd-megatron.sh` — Ray + `train_multi_policy.py` launcher.

## Running

1. **Checkpoints**: place a Qwen3-0.6B HF checkpoint at `/root/Qwen3-0.6B`,
   and a different (e.g., more-trained) Qwen3-0.6B at `/root/Qwen3-0.6B-teacher`.
2. **Data**: place `dapo-math-17k.jsonl` at `/root/dapo-math-17k/`.
3. **Run**:
   ```bash
   bash examples/multi_policy_opd_megatron/run-qwen3-0.6B-opd-megatron.sh
   ```

Per-role rollout / train / packed dumps land at
`/tmp/multi_policy_opd_megatron/dump_details/<policy_name>/...`.

Useful checks after the first rollout:

- `train/student/opd_reverse_kl` shows up in metrics (non-NaN).
- Timer logs show `teacher_megatron.train` running before `student.train`.
- No checkpoint dir is written for `teacher_megatron`.
- No weight-push happens to the teacher's actor.

For a **kernel-consistency** sanity check, pick one sample from
`/tmp/multi_policy_opd_megatron/dump_details/student/packed_data/<rollout_id>_<rank>.pt`,
load the dict, compute `|teacher_log_probs - student_log_probs|` element-wise,
and compare against `train_rollout_logprob_abs_diff` (the sglang-vs-Megatron
noise floor for the student's own logprobs).

## Comparison vs. legacy single-policy OPD

| | Legacy `examples/on_policy_distillation/` | This example |
|---|---|---|
| Teacher placement | In-process tag in same Megatron actor | Separate Megatron actor (own GPU, own weights) |
| Driver | `train.py` | `train_multi_policy.py` |
| Teacher load | `--opd-teacher-load /path/to/torch_dist` | `hf_checkpoint` per-policy in YAML |
| KL kernels | Identical (single actor) | Identical (separate actor; same Megatron forward) |
| Teacher size | Bound by single-actor memory budget | Independent budget — own GPU(s) |
| Scales to multi-policy | No | Yes |

The legacy path stays as-is; this directory does **not** replace it. Use
this version when you want the teacher in its own Ray actor — e.g., to scale
the teacher independently of the student, or as the building block for
future multi-teacher setups.

## Limitations

- **Same architecture for student and teacher**: `MODEL_ARGS` is a CLI-global
  source. Cross-architecture distillation (e.g., 0.6B student + 1.7B teacher)
  needs Megatron architecture fields moved into per-policy YAML.
- **TP / PP / CP must equal 1 on both policies**: per-token alignment in the
  KL term assumes identical layouts.
- **No SGLang-backend teacher option here**: that's a separate example
  (frozen `m✗ s✓ trainable=false` engine; KL noise floor is higher because
  of inference/training kernel mismatch).
