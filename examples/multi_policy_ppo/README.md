# Multi-Policy PPO (asymmetric actor + critic)

Multi-policy PPO with a **smaller critic than actor**: a 1.7B trainable **actor** generates rollouts; a 0.6B trainable **critic** runs `train_critic` (forward + value-loss + backward) on those rollouts and emits per-token `values` that feed PPO advantages into the actor's loss. Because the critic is its own Megatron Ray actor (separate weights, separate optimizer, separate GPU), it can be a different architecture from the actor — the use case legacy single-policy PPO (`train.py + --critic-config-path`) cannot express.

## Files

* `config.yaml`: actor (Qwen3-1.7B, paired) + critic (Qwen3-0.6B, standalone) policy schema. Each policy declares its architecture via `megatron.model_args_path` pointing at an upstream `scripts/models/<name>.sh`.
* `run-qwen3-1.7B-0.6B-ppo.sh`: launch script (ray start + `train_multi_policy.py`). No `source` step, no `${MODEL_ARGS[@]}` interpolation — per-policy arch comes from the YAML.

## Quick Start

```bash
cd slime-n
bash examples/multi_policy_ppo/run-qwen3-1.7B-0.6B-ppo.sh
```

Place a Qwen3-1.7B HF checkpoint at `/root/Qwen3-1.7B`, a Qwen3-0.6B HF checkpoint at `/root/Qwen3-0.6B`, and `dapo-math-17k.jsonl` at `/root/dapo-math-17k/`.

## How It Works

* **Critic dispatch (shape-derived).** The critic is identified by shape: `trainable + sglang_num_nodes == 0 + advantage_estimator == "ppo"` → `is_critic_shape(cfg)` returns True. `placement_group.py`'s `create_training_models_multi` derives `role_eff="critic"` for it and passes that to `allocate_train_group` and `async_init`. The model provider then attaches a 1-dim value head (`LinearForLastLayer`) on the critic, and `actor.__init__` skips `weight_updater` / `weights_backuper` setup. The YAML's `role: actor` is kept intact for logs/saves.

* **Asymmetric architecture (`model_args_path`).** Each policy's `megatron:` block declares one line: `model_args_path: scripts/models/<name>.sh`. The parser reads each referenced `.sh`'s `MODEL_ARGS=(...)` bash array, converts kebab→snake (e.g. `--num-layers 28` → `num_layers: 28`, bare `--swiglu` → `swiglu: True`, `--no-rope-fusion` → `rope_fusion: False`, `${VAR:-default}` → `default`), splits the parsed keys into "matches a PolicyConfig field" vs "extra Megatron arg", and merges each behind its inline counterpart (inline always wins on conflict). `config_to_namespace` projects the merged result onto the policy's args namespace. Actor reads `qwen3-1.7B.sh`; critic reads `qwen3-0.6B.sh`. No bash `source` step required.

* **Parse-flow (multi-policy).** `train_multi_policy.py` runs `parse_multi_policy_args()` instead of plain `parse_args()`. The flow: pre-parse `--config` → load `policy_configs` → `parse_args(skip_megatron_model_validation=True)` (defers HF + Megatron structural validation) → `_set_multi_policy_global_defaults` populates `base_args` → for each policy, `config_to_namespace(cfg, base_args)` produces a fully-defaulted namespace (internally calls `_apply_megatron_defaults` and re-applies extras). Megatron-hosting policies (`cfg.megatron_num_nodes > 0`) then run `_validate_hf_per_policy` + `megatron_validate_args` against their own arch — so the actor and critic each validate independently.

* **Train order.** The driver partitions handles by shape into `frozen / trainable_standalone / trainable_pair`. The critic lands in `trainable_standalone` (no engine) and runs before the actor every rollout. Its `{"values": ...}` return value merges into `merged_external` and reaches the actor as `external_data`.

* **Advantage merge.** `train_actor` (`actor.py:509-515`) folds `external_data["values"]` into `rollout_data["values"]` on the actor's last PP rank, gated on `args.use_critic` (which the driver flips True on the actor's namespace when any sibling is critic-shaped — before `register_policy` snapshots the args). `compute_advantages_and_returns` then sees the critic-provided values when computing PPO advantages for the actor.

* **Weight push.** Refined gate `cfg.trainable and has_sglang_engine(cfg)` skips the critic (no engine to push to).

## Compared to legacy single-policy PPO

`train.py + --critic-config-path` ties actor and critic to the same CLI-global `MODEL_ARGS` (one bash `source` provides one global arch). Multi-policy PPO lifts that constraint via per-policy `model_args_path:` — each policy points at its own `scripts/models/<name>.sh`, enabling:

- A **smaller critic** (1.7B + 0.6B as shown here) — saves a GPU's worth of value estimation without throttling the actor's rollout engine.
- A **larger critic** for harder value-estimation problems — same plumbing, different YAML.

## Known limitations (v1)

- **TP/PP/CP/DP must match between actor and critic.** `external_data["values"]` is routed per-rank via the driver's `list[dict]` pass-through, so the producer's effective DP world (non-empty last-PP-stage ranks, in DP order under Megatron's default `tp-cp-ep-dp-pp` mesh) must equal the consumer's `world_size`. This example uses 1 GPU on each side, which satisfies the constraint trivially.
- **Single rollout arch when MoE / routed-experts are in use.** `slime/rollout/sglang_rollout.py` reads `args.num_layers` for routed-expert reshape. `parse_multi_policy_args` populates this from engine-hosting policies and raises on mixed engine arch + `use_rollout_routing_replay`. (Not triggered by this example, which is dense.)
- **Cold-start critic.** Conventional PPO critics initialize from the actor's weights with a value head bolted on; that's impossible with asymmetric architecture. The critic loads from its own (smaller) base LM checkpoint, so expect a flat-then-rising critic loss for the first many rollouts and weak advantages on the actor side until the critic warms up.

## Implementation status

The `model_args_path` field, the `.sh` parser, the parse-flow refactor (`parse_multi_policy_args` with per-policy HF/Megatron validation), the rollout-arch guard, and the multi-value flag handling are all specified in `plan_model_field.md`. This example reflects the target end-state — running it requires those framework pieces to land first.
