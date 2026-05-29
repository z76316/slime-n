# Multi-Policy Solver-Rewriter-Selector

Three trainable paired policies cooperate on math problems (DAPO-math-17k). The **solver** generates N candidate solutions; the **rewriter** sees all N solver candidates and synthesizes a refined solution per worker; the **selector** then sees the N rewriter candidates and emits `Judgment: IDX` picking one as best. The selector inherits the rewriter's correctness reward on the picked candidate.

| schema | slime<sup>n</sup> |
|:---:|:---:|
| ![solver-rewriter-selector schema](./imgs/schema.png) | ![solver-rewriter-selector framework](./imgs/arch.png) |

*Left: the solver emits N candidates, the rewriter refines each after seeing all N, the selector picks one with `Judgment: IDX`. Right: three trainable pairs in a chain (solver → rewriter → selector), each Megatron + SGLang with its own buffer and optimizer.*

## Files

* `config.yaml`: solver + rewriter + selector policy schema (all three trainable).
* `run-qwen3-0.6B-solver-rewriter-selector.sh`: launch script (ray start + train_multi_policy.py, `--colocate` to fit on a 3-4 GPU box).
* `agent_system.py`: per-prompt rollout orchestration (solver → rewriter → selector dispatch).
* `rollout_with_multi_agents.py`: top-level multi-agent rollout entrypoint.
* `prompts.py`: solver / rewriter / selector prompt templates.

## Quick Start

```bash
cd slime-n
bash examples/multi_policy_solver_rewriter_selector/run-qwen3-0.6B-solver-rewriter-selector.sh
```

## How It Works

* Pipeline (N=4 trajectories per prompt): N solvers → N rewriters (each sees all N solver candidates) → N selectors (each sees all N rewriter candidates and picks one).
* Reward: solver and rewriter get direct RLVR correctness on their own responses; selector inherits the reward of its picked rewriter. Group shaping multiplies all three roles by 1.2 if mean parsed-selector reward > 0.5, else by 0.8. If all selectors fail to parse, raw rewards are preserved (anti-train guard — broken selector must not penalize correct upstream).
* Each policy has its own buffer (`buffer_mode: split`), routed by `Sample.policy_name`. `n_samples_per_prompt = num_parallel = 4` for GRPO group-norm.

## Policies

| policy | megatron | sglang | trainable | role |
|---|---|---|---|---|
| `solver` | ✓ | ✓ | ✓ | candidate solution generator |
| `rewriter` | ✓ | ✓ | ✓ | refines solver candidates |
| `selector` | ✓ | ✓ | ✓ | picks best of N rewriter candidates |

Cluster: 3 GPUs with `--colocate` (max(3 megatron, 3 sglang) = 3). Without colocate, this would be 6 GPUs (3 + 3).
