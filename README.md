<div align="center">

# $\huge\color{#1E293B}{\textsf{\textbf{slime}}}^{\color{#3B82F6}{\textsf{\textbf{n}}}}$

</div>

## A Multi-Policy, Multi-Agent RL Training Framework

slime<sup>n</sup> extends [slime](https://github.com/THUDM/slime) into a flexible RL training framework for multi-policy and multi-agent workloads. A run is composed of any mix of three policy shapes:

- **Trainable policy pair** — a Megatron training actor paired 1:1 with an SGLang engine. The default workhorse for RL.
- **Standalone Megatron actor** — Megatron only, no SGLang engine. Trainable variant trains its own loss without rolling out (e.g. a PPO critic); frozen variant runs forward-only and emits per-token logprobs (e.g. an OPD teacher).
- **Standalone SGLang engine (frozen)** — SGLang only, no trainer; serves inference for judges, reward models, or verifier-style scoring.

![multi-policy architecture](./imgs/arch_2.png)

## Multi-policy

- **`train_multi_policy.py`** — driver for n≥1 trainable policies. Replaces `train.py` for multi-policy runs.
- **YAML-driven configs** — `--config <path>.yaml`. Per-policy fields (parallelism, batching, optimizer, loss, paths, Megatron numerical / dropout, `log_probs_chunk_size`) live in the YAML; cluster sizing is derived from policies. See [`slime/utils/policy_config.py`](slime/utils/policy_config.py).
- **Per-policy buffers (split mode)** — each policy trains on its own samples, tagged via `Sample.policy_name`.
- **Per-policy weight sync** — serialized push from each Megatron actor to its paired sglang engine.

## Examples

Three workloads exercise the multi-policy schema — two paired-pipeline cooperations (debate, solver+summarizer) and a frozen standalone Megatron actor (OPD teacher). Standalone SGLang engines (judge / reward model variants) are supported by the same schema; example pending.

### 1. Multi-Policy Multi-Agent debate — generator + critic

Two trainable paired policies implement a paper-aligned debate workflow. In round 0, N `generator` agents propose independent answers. In later rounds, an untracked summarize subroutine summarizes the other agents' previous responses, and each `critic` agent updates its own answer from that summary plus its own prior response.

Rewards are computed from the final critic responses: the system majority-votes a final answer `ŷ`; round-0 generator samples are rewarded for matching `ŷ`, and each critic trajectory receives reward 1 when that agent's final critic answer matches `ŷ`. The dataset gold label is intentionally ignored in this example.

| policy | megatron | sglang | trainable | role |
|---|---|---|---|---|
| `generator` | ✓ | ✓ | ✓ | round-0 answer generator |
| `critic` | ✓ | ✓ | ✓ | round-1+ answer updater |

The summarize step is routed through the generator SGLang engine, but its samples are not added to a training buffer. Code: [`examples/multi_policy_multiagent_debate`](examples/multi_policy_multiagent_debate).

### 2. Multi-Policy Solver + Summarizer — candidate generation + final answer synthesis

Two trainable paired policies cooperate on math problems. The `solver` policy generates N candidate solutions in parallel. The `summarizer` policy then sees all N solver candidates and synthesizes a final answer in the standard `Answer: \boxed{...}` format.

Both policies train on split buffers and receive direct RLVR correctness rewards on their own completions. The example also applies group reward shaping from the summarizer phase: if the mean summarizer reward is high, both roles are upweighted; otherwise both roles are downweighted.

| policy | megatron | sglang | trainable | role |
|---|---|---|---|---|
| `solver` | ✓ | ✓ | ✓ | candidate solution generator |
| `summarizer` | ✓ | ✓ | ✓ | final answer synthesizer |

Code: [`examples/multi_policy_solver_summarizer`](examples/multi_policy_solver_summarizer).

### 3. OPD — on-policy distillation (student + frozen teacher)

Trainable **student** generates rollouts; frozen **teacher** runs forward-only on those rollouts and returns per-token logprobs that feed a reverse-KL term into the student's loss. The schema admits two teacher backends:

| policy | megatron | sglang | trainable | role |
|---|---|---|---|---|
| `student` | ✓ | ✓ | ✓ | paired pipeline; generates rollouts |
| `teacher` *(Megatron — recommended, kernel-consistent)* | ✓ | ✗ | ✗ | standalone actor; per-token logprobs |
| `teacher` *(SGLang — cheaper, kernel mismatch)* | ✗ | ✓ | ✗ | standalone engine; per-token logprobs |

The teacher's `train()` returns `{"teacher_log_probs": ...}`. The driver merges all frozen-policy outputs into the student's `external_data`, which `train_actor` writes into `rollout_data` so `apply_opd_kl_to_advantages` can consume it.

Code: [`examples/multi_policy_opd_megatron`](examples/multi_policy_opd_megatron) (Megatron-backend teacher). The SGLang-backend variant uses the same schema; example pending.

## Run

```bash
bash examples/multi_policy_two_agent/run-qwen3-0.6B-two-policy-two-agent.sh
```

Which boils down to:

```bash
ray job submit ... -- python3 train_multi_policy.py --config examples/multi_policy_two_agent/config.yaml
```

See [`train_multi_policy.py`](train_multi_policy.py) for the train-loop body and the architecture figure above (source: `../figure/arch_2.typ`) for the runtime layout.
