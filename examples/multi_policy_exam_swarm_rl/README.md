# Multi-Policy Exam Swarm RL Example

This example trains 8 homogeneous LLM agents that independently take the same exam. Each agent's gradient signal blends three pressures: how it improves against itself (individual), how the swarm progresses against its history (cooperative), and how it ranks against its peers on this question (competitive).

## Key Features

- **N=8 homogeneous agents**, all initialized from the same checkpoint, each its own trainable Megatron actor + sglang engine pair.
- **No agent-to-agent communication** during rollout. Cooperative-competitive coupling is purely in the per-trajectory advantage.
- **Three-channel advantage** composed at rollout time, stored as `Sample.reward` (single float). Slime broadcasts to per-token via the GRPO advantage estimator path.
- **Two `self_adv` modes**: standard GRPO group-norm (default) or adversarial baseline against the best peer's mean (head-to-head competition).

## Key Constants (in `agent_system.py`)

| Constant | Default | Role |
|----------|---------|------|
| `ALPHA` | 0.5 | individual self_adv weight |
| `BETA` | 0.3 | cooperative swarm_adv weight |
| `GAMMA` | 0.2 | competitive peer_adv weight |
| `BASELINE_MODE` | `"grpo"` | `"grpo"` or `"adversarial"` |
| `ADV_CLIP` | 5.0 | final advantage clip magnitude |
| `N_AGENTS` | 8 | must match `policies` count in `config.yaml` |

Edit and re-launch — no CLI flag, no YAML edit.

## Components

- `agent_system.py`: `SwarmBaseline` (cross-question EMA), `rank_advantage`, `self_advantage_grpo`, `self_advantage_adversarial`, `run_agent_system` (top-level rollout).
- `rollout_with_swarm.py`: slime `--custom-generate-function-path` entrypoint.
- `config.yaml`: 8 byte-identical paired policies (Megatron + sglang per agent, colocate-friendly sizing).
- `run-qwen3-0.6B-exam-swarm-colocate.sh`: 8-GPU single-node colocate launch.
- `prompts.py`: pass-through math prompt.

## Reward Composition

Per outer prompt:

1. Sample one math question from DAPO-math-17k.
2. Dispatch to all 8 agents in parallel; each generates K=8 independent answers (8×8=64 trajectories).
3. RLVR-score every answer (deepscaler boxed-answer match) → per-trajectory `c ∈ {0, 1}`.
4. Compose three advantage channels:

```text
self_adv  = (c - mean_K) / (std_K + ε)            # GRPO mode
            (c - max_peer_mean) / (std_K + ε)     # adversarial mode
swarm_adv = (g - μ_g) / (σ_g + ε)                 # g = pass rate over all 64 traj
                                                   # μ_g, σ_g = EMA over PAST questions
peer_adv  = (N + 1 - 2·rank_i) / (N - 1)          # zero-mean across agents per question

final = α · self_adv + β · swarm_adv + γ · peer_adv   (clipped to ±5)
```

5. Stored as `Sample.reward`. Slime's GRPO path broadcasts to per-token. **`--disable-rewards-normalization` is required** (the run script sets it) — slime would otherwise re-normalize the per-trajectory advantage and erase the swarm/peer signals.

6. Per-agent buffer routing via `Sample.policy_name = "agent_i"`. Each agent gradient-steps on its own slice.

## Why Each Channel Survives Composition

GRPO group-norm subtracts the mean within an agent's K samples. Per-question constants (`swarm_adv`, `peer_adv` per agent) would cancel inside that group-norm. The trick: compose advantages at rollout time *before* slime's normalization, then disable slime's normalization.

- `swarm_adv` survives because its EMA baseline lives across questions; it varies as the swarm improves.
- `peer_adv` survives because it's zero-mean across agents *by construction* (sums to 0 each question).
- `self_adv` adversarial-mode survives because `max_peer_mean_i` differs per agent; the resulting advantage is deliberately not zero-mean within K (that's the competitive pull).

## Running the Example

```bash
cd slime
bash examples/multi_policy_exam_swarm_rl/run-qwen3-0.6B-exam-swarm-colocate.sh
```

Cluster: 8 GPUs, single node, `--colocate`. Each GPU hosts one agent's Megatron + sglang (offload-swap between rollout and train phases). Without colocate, 8 Megatron + 8 sglang = 16 GPUs (exceeds a single node).

Per GPU: Qwen3-0.6B fp16 (~1.2 GB) + activations + sglang KV cache + (CPU-offloaded) optimizer state. Comfortable on 80 GB H100, fits on 40 GB A100 with care.

## Knobs and Knob Effects

The cooperative-competitive dial is the **ratio γ/β**:

- `γ ≪ β` — cooperative dominant; agents pull together, pairwise KL stays small.
- `γ ≈ β` — balanced tension (default).
- `γ ≫ β` — competitive dominant; agents diverge fighting for top rank.

`BASELINE_MODE = "adversarial"` strengthens competition further: every agent's gradient is directly tied to beating the swarm leader on every trajectory.

## Edge Cases

- All-correct or all-wrong question: zero variance everywhere, no signal — standard GRPO failure mode for trivial / hard prompts.
- Cold start: first 20 questions return `swarm_adv = 0` while the EMA stabilizes (`SwarmBaseline.WARMUP`).
- Rollout failure for one agent: padded to K with zero-reward placeholders so the per-policy split-buffer count invariant holds.

## Diagnostics

Every `Sample.metadata` carries `raw_reward` (= c), `self_adv`, `swarm_adv`, `peer_adv`, `peer_max`, `g`, `agent_idx`, `baseline_mode`. Slime's standard `raw_reward` plumbing surfaces these to wandb.

## FAQ

1. **Why is `--disable-rewards-normalization` required?**
   Slime's default GRPO post-processing subtracts a per-prompt mean and divides by std. Our `Sample.reward` already encodes the full 3-channel advantage. Re-normalizing would (a) cancel the swarm and peer constants, and (b) double-normalize the self-component.

2. **Can I drop the cooperative or competitive channel?**
   Yes — set `BETA = 0` (drop swarm) or `GAMMA = 0` (drop rank). With `ALPHA = 1`, `BETA = GAMMA = 0`, the example reduces to vanilla GRPO per agent (the null hypothesis).

3. **Why `BASELINE_MODE = "adversarial"`?**
   Standard GRPO compares each trajectory to the agent's own K others (self-improvement). Adversarial mode replaces that baseline with the best other agent's mean, making every gradient step a head-to-head competition with the swarm leader.

4. **Can I scale beyond 8 agents?**
   Bump `N_AGENTS` in `agent_system.py` and add policy entries to `config.yaml`. At N>8 you need more GPUs (multi-node) since slime requires ≥1 GPU per policy under colocate.
