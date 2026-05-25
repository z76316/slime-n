# Multi-Policy Solver + Summarizer

Two trainable paired policies cooperate on math problems (DAPO-math-17k). The **solver** generates N candidate solutions in parallel; the **summarizer** then sees ALL N candidates and synthesizes one final answer in the standard `Answer: \boxed{...}` format. Both policies receive direct correctness rewards on their own completions.

![architecture: two trainable pairs (solver, summarizer)](./imgs/arch.png)

*Two trainable pairs. The chain is owned by the custom rollout function: solver SGLang produces N candidates → those candidates become the summarizer's prompt → summarizer SGLang emits the final boxed answer. Each policy has its own optimizer, buffer, and RLVR reward.*

## Files

* `config.yaml`: solver + summarizer policy schema (both trainable, paired with their own SGLang engines).
* `eval_config.yaml`: AIME-2024 eval-dataset config (rm_type, n_samples). Consumed via `--eval-config`.
* `run-qwen3-0.6B-solver-summarizer.sh`: launch script (ray start + train_multi_policy.py).
* `agent_system.py`: per-prompt rollout orchestration (solver → summarizer dispatch).
* `rollout_with_multi_agents.py`: top-level multi-agent rollout entrypoint.
* `eval_fn.py`: custom eval function — aggregates chain samples into summarizer / solver-mean / solver-max metrics.
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


## Eval

Every `--eval-interval` rollouts, the full chain runs on AIME-2024 (30 prompts; 30 × 8 = 240 generations per eval). The custom eval function (`eval_fn.eval_with_multi_agents`, wired via `--eval-function-path`) emits per-attempt **raw** RM rewards (unscaled by the 0.8/1.2 training weights) under two logged datasets:

* `eval/aime_summarizer/score` — per-attempt accuracy = `pass@1`.
* `eval/aime_solver/score` — per-attempt accuracy = `pass@1`.

With `--log-passrate --n-samples-per-eval-prompt 4`, the default logger adds `pass@1`, `pass@2`, `pass@4` for both. The headline final-answer-quality metric is `eval/aime_summarizer-pass@4` (= 1 if any of the 4 summarizer attempts is correct). `eval/aime_solver-pass@4` is the solver best-of-4 skyline. Their difference diagnoses whether the summarizer is synthesizing nontrivially or just aggregating (or destroying) signal the solver produced.

Two limitations: eval generation flows through the first-listed policy's SGLang engine (the solver's), and metrics are split by role name, not by per-policy namespace.


## Results

1873-step run on Qwen3-0.6B.

**Per-role raw reward** — both policies trend up; summarizer mean
~0.54, solver mean ~0.43. The summarizer benefits from seeing all N
solver candidates so its peak (~0.78) exceeds the solver's (~0.65).

![reward](imgs/reward.png)
