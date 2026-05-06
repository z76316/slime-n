# Multi-Policy Solver + Summarizer

Two trainable paired policies cooperate on math problems (DAPO-math-17k). The **solver** generates N candidate solutions in parallel; the **summarizer** then sees ALL N candidates and synthesizes one final answer in the standard `Answer: \boxed{...}` format. Both policies receive direct correctness rewards on their own completions.

## Files

* `config.yaml`: solver + summarizer policy schema (both trainable, paired with their own SGLang engines).
* `run-qwen3-0.6B-solver-summarizer.sh`: launch script (ray start + train_multi_policy.py).
* `agent_system.py`: per-prompt rollout orchestration (solver → summarizer dispatch).
* `rollout_with_multi_agents.py`: top-level multi-agent rollout entrypoint.
* `prompts.py`: solver / summarizer prompt templates.

## Quick Start

```bash
cd slime
bash examples/multi_policy_solver_summarizer/run-qwen3-0.6B-solver-summarizer.sh
```

## How It Works

* Pipeline (N=4 trajectories per prompt): N solvers run in parallel → N summarizers that each see all N solver candidates and synthesize a final answer.
* Reward: solver gets RLVR correctness on its own response; summarizer gets correctness on its synthesized answer (graded directly, no index lookup). Group shaping multiplies both roles by 1.2 if mean parsed-summarizer reward > 0.5, else by 0.8. If the summarizer phase fails entirely, raw rewards are preserved (anti-train guard).
* Each policy has its own buffer (`buffer_mode: split`), routed by `Sample.policy_name`. `n_samples_per_prompt = num_parallel = 4` for GRPO group-norm.

## Policies

| policy | megatron | sglang | trainable | role |
|---|---|---|---|---|
| `solver` | ✓ | ✓ | ✓ | candidate solution generator |
| `summarizer` | ✓ | ✓ | ✓ | synthesizes final answer from N candidates |

Cluster: 4 GPUs (2 megatron + 2 sglang, no colocate).
