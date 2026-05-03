# $$\huge\color{#1E293B}{\textsf{\textbf{slime}}}^{\color{#3B82F6}{\textsf{\textbf{N}}}}$$

## A Multi-Policy, Multi-Agent RL Training Framework

$\text{slime}^{N}$ extends [slime](https://github.com/THUDM/slime) into a flexible RL training framework for multi-policy and multi-agent workloads. It can compose arbitrary combinations of trainable policy pairs, standalone trainable Megatron actors, and standalone frozen models.

- **Trainable policy pairs**: a Megatron training actor paired with an SGLang engine.
- **Standalone trainable Megatron actors**: roles such as a PPO critic.
- **Standalone frozen models**: Megatron teachers for OPD, or SGLang judges and verifiers.

![multi-policy architecture](./imgs/arch_2.png)

## Multi-policy

- **`train_multi_policy.py`** — driver for N≥1 trainable policies. Replaces `train.py` for multi-policy runs.
- **YAML-driven configs** — `--config <path>.yaml`. Per-policy fields (parallelism, batching, optimizer, loss, paths, Megatron numerical / dropout, `log_probs_chunk_size`) live in the YAML; cluster sizing is derived from policies. See [`slime/utils/policy_config.py`](slime/utils/policy_config.py).
- **Per-policy buffers (split mode)** — each policy trains on its own samples, tagged via `Sample.policy_name`.
- **Per-policy weight sync** — serialized push from each Megatron actor to its paired sglang engine.

## Examples

Three workloads exercise the full multi-policy pattern space — paired pipelines, standalone actors, and standalone engines (see the figure above):

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

Trainable **student** generates rollouts; frozen **teacher** provides per-token logprobs that feed a KL term in the student's loss. Two backends for the teacher, both supported in the same schema:

| variant | student | teacher | teacher slot |
|---|---|---|---|
| **Megatron-backend teacher** (recommended — kernel-consistent) | paired pipeline (`m✓ s✓ trainable`) | `m✓ s✗ trainable=false` | standalone Megatron actor |
| **SGLang-backend teacher** (cheaper, kernel mismatch) | paired pipeline | `m✗ s✓ trainable=false` | standalone SGLang engine |

Teacher is forward-only — `train()` returns `{"teacher_log_probs": ...}` as `external_data`, which is routed to the student through producer→consumer plumbing.

See [`plan_opd.md`](../plan_opd.md) for the full design.

### Other examples

| Directory | Pattern |
|---|---|
| [`examples/multi_policy_two_agent`](examples/multi_policy_two_agent) | Solver + selector — two trainable paired policies. |
| [`examples/multi_policy_multi_agent`](examples/multi_policy_multi_agent) | General multi-agent setup with N trainable policies. |

Each example contains: `config.yaml`, `agent_system.py`, `prompts.py`, `rollout_with_multi_agents.py`, `run-*.sh`.

## Run

```bash
bash examples/multi_policy_two_agent/run-qwen3-0.6B-two-policy-two-agent.sh
```

Which boils down to:

```bash
ray job submit ... -- python3 train_multi_policy.py --config examples/multi_policy_two_agent/config.yaml
```

See the module docstring in [`train_multi_policy.py`](train_multi_policy.py) and the architecture figure above (source: `../figure/arch_2.typ`) for the runtime architecture.
