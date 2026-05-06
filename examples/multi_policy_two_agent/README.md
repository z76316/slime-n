# Multi-Policy Two-Agent (solver + selector)

Two trainable paired policies cooperate on math problems (DAPO-math-17k). The **solver** generates N candidate solutions in parallel; the **selector** then sees ALL N candidates and emits `Judgment: IDX` picking one as best. The selector inherits the solver's correctness reward on the picked candidate.

## Files

* `config.yaml`: solver + selector policy schema (both trainable, paired with their own SGLang engines).
* `run-qwen3-0.6B-two-policy-two-agent.sh`: launch script (ray start + train_multi_policy.py).
* `agent_system.py`: per-prompt rollout orchestration (solver → selector dispatch).
* `rollout_with_multi_agents.py`: top-level multi-agent rollout entrypoint.
* `prompts.py`: solver / selector prompt templates.

## Quick Start

```bash
cd slime
bash examples/multi_policy_two_agent/run-qwen3-0.6B-two-policy-two-agent.sh
```

## How It Works

* Pipeline (N=4 trajectories per prompt): N solvers run in parallel → N selectors that each see all N solver candidates and pick one.
* Reward: solver gets direct RLVR correctness reward; selector inherits the reward of its picked solver candidate. Group shaping multiplies both roles by 1.2 if mean parsed-selector reward > 0.5, else by 0.8. If all selectors fail to parse, raw rewards are preserved (anti-train guard — broken selector must not penalize correct solvers).
* Each policy has its own buffer (`buffer_mode: split`), routed by `Sample.policy_name`. `n_samples_per_prompt = num_parallel = 4` for GRPO group-norm.

## Policies

| policy | megatron | sglang | trainable | role |
|---|---|---|---|---|
| `solver` | ✓ | ✓ | ✓ | candidate solution generator |
| `selector` | ✓ | ✓ | ✓ | picks best of N solver candidates |

Cluster: 4 GPUs (2 megatron + 2 sglang, no colocate).
