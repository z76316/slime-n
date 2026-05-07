# Multi-Policy OPD (frozen SGLang teacher)

> **Status: design spec — not runnable today.**
> Requires framework changes F1/F2/F3 (tracked in
> `slimen_workspace/plan_colocate.md`) and a custom rollout function
> that queries the teacher's SGLang engine. Without these, the teacher
> policy isn't sized/spawned correctly. The companion
> `examples/multi_policy_opd_megatron/` is the runnable variant today.

Multi-policy on-policy distillation with an **SGLang-backend** teacher: a
trainable **student** generates rollouts; a frozen **teacher** SGLang
engine serves per-token logprobs at rollout time. The student's loss
adds a reverse-KL term against those teacher logprobs (`KL(student ‖ teacher)`).

Compared to the Megatron-backend teacher (`multi_policy_opd_megatron`),
this variant trades **kernel consistency for cheaper deployment**: SGLang
inference is faster and supports quantization, but its kernels disagree
with Megatron's training-time forward — so the KL noise floor is higher.
Use this when teacher size or inference efficiency matters more than
kernel-clean KL.

## Files

* `config.yaml`: student + teacher_sglang policy schema.
* `run-qwen3-0.6B-opd-sglang.sh`: launch script (won't run today; see Status).
* `rollout_with_teacher_sglang.py`: spec for the custom rollout function
  (not yet implemented).

## How It Works

```
┌─── student (trainable, paired) ──────┐    ┌─── teacher_sglang (frozen) ────┐
│  Megatron actor   ⇆   SGLang engine  │    │       SGLang engine             │
│       1 GPU              1 GPU       │    │            1 GPU                │
│                                      │    │      (no Megatron actor)        │
└───────────┬──────────────────────────┘    └──────────────┬──────────────────┘
            │ rollout (student generates)                  │
            │                                              │
            │  for each generated sample:                  │
            │   POST /generate to teacher_sglang with     │
            │   return_logprobs=True on response tokens   │
            │   ──────────────────────────────────────────▶│
            │   ◀─────────────────────────────────────────
            │   sample.teacher_log_probs = [...]
            ▼
   train (KL loss via apply_opd_kl_to_advantages)
```

3 GPUs total under no-colocate. The teacher's SGLang engine spawns alongside
the student's, both managed by the rollout manager. The student's custom
rollout function queries the teacher per sample.

## Data Path

* **Rollout phase**: student's SGLang engine generates the response token
  sequence. Custom rollout fn then sends `POST /generate` to
  `teacher_sglang` with `return_logprobs=True` over the response tokens
  (no new generation — just logprob scoring on the student's output).
  Sets `sample.teacher_log_probs = [list of float]`.
* **Train phase**: `_convert_samples_to_train_data` (already in
  `slime/ray/rollout.py:984-985`) hoists per-sample `teacher_log_probs`
  into `train_data["teacher_log_probs"]`. Student's `train_actor` reads
  this through `rollout_data` and `apply_opd_kl_to_advantages` applies
  the KL term.

This re-uses the existing single-policy SGLang-OPD path
(`slime/rollout/on_policy_distillation.py:reward_func`,
`slime/ray/rollout.py:_convert_samples_to_train_data`). The only new
piece is the custom rollout fn that knows to query `teacher_sglang`'s
engine via the rollout manager (rather than an external `--rm-url`).

## Policies

| policy | megatron | sglang | trainable | role |
|---|---|---|---|---|
| `student` | ✓ | ✓ | ✓ | paired pipeline; generates rollouts |
| `teacher_sglang` | ✗ | ✓ | ✗ | standalone engine; per-token logprobs |

Cluster: 3 GPUs (1 student megatron + 1 student sglang + 1 teacher sglang, no colocate).
Under `--colocate`: 2 GPUs (student.M + student.S share GPU 0; teacher.S on GPU 1).

## Compared to other OPD variants

| Variant | Where teacher runs | Kernels | KL noise floor | Setup |
|---|---|---|---|---|
| `examples/on_policy_distillation/` (legacy, single-policy) | External SGLang server, user-managed | sglang-vs-megatron mismatch | High | Manual server launch + `--rm-url` |
| `examples/multi_policy_opd_megatron/` | Megatron actor in multi-policy schema | Identical (same Megatron forward) | **Low** ★ | Single config.yaml; framework manages |
| **This** (`multi_policy_opd_sglang/`) | SGLang engine in multi-policy schema | sglang-vs-megatron mismatch | High | Single config.yaml; framework manages |

★ Recommended when kernel-consistent KL matters.

## Required framework work (before this example runs)

These are tracked in
[`slimen_workspace/plan_colocate.md`](../../../plan_colocate.md):

### F1 — `derive_cluster_sizing` predicate

`policy_config.py:300` filters rollout_gpus by `cfg.trainable`. The
frozen `teacher_sglang` has `trainable=false`, so its
`sglang_num_nodes * num_gpus_per_node` contribution is excluded from the
rollout budget — cluster underallocates.

**Fix**: predicate should be "hosts an engine"
(`bool(cfg.sglang.get("server_groups"))`), not `cfg.trainable`.

### F2 — `build_sglang_config_from_policies` filter

`policy_config.py:262` skips `not cfg.trainable`. The teacher_sglang is
skipped → no engine spawned. Same fix: gate on engine-hosting, not
trainability.

### F3 — `_policy_relative_engine_gpu_offsets` for non-Megatron policies

`slime/ray/rollout.py:554-559` asserts engine GPUs within
`[actor_offset, actor_offset + actor_gpus)`. For an `m✗ s✓` judge with
`megatron_num_nodes=0` → `actor_gpus=0`, the assertion fires for any
positive engine offset.

**Fix**: early-return for policies with `actor_gpus == 0` (their engines
have no Megatron actor slice to relativize against; relative offset =
absolute offset).

### Custom rollout function

`rollout_with_teacher_sglang.py` (not yet written) needs to:

1. Run the student's normal rollout (produces `sample.response`).
2. Look up the teacher's SGLang URL via the rollout manager (something
   like `rollout_manager.get_server("teacher_sglang").endpoint`).
3. For each sample, `POST /generate` to that URL with the response
   tokens and `return_logprobs=True` to get per-token logprobs **without
   regenerating** (use the existing prefix-only scoring mode).
4. Set `sample.teacher_log_probs` to the per-token logprob list aligned
   with the response span.
5. Return the samples.

The existing `slime/rollout/on_policy_distillation.py:32-67` does step
3 against `args.rm_url`; the new rollout fn substitutes the
multi-policy-managed teacher URL.

## Today's runnable alternative

If you want SGLang-teacher OPD running TODAY (without framework work),
use the legacy single-policy variant:

```bash
# Launch external SGLang server for teacher manually:
python -m sglang.launch_server --model-path /root/Qwen3-0.6B-teacher --port 30000 &

# Then run the legacy single-policy OPD-SGLang script:
bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
# (with --rm-url http://127.0.0.1:30000 already set in that script)
```

The legacy version loses the "framework-managed teacher" property but
otherwise produces the same training signal.
