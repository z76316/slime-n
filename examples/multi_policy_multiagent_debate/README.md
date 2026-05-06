# Multi-Policy Multi-Agent Debate (generator + critic)

Paper-aligned implementation of [Subramaniam et al. 2025, "Multiagent Finetuning of Language Models"](https://arxiv.org/abs/2501.05707) Algorithm 1, on math problems (DAPO-math-17k). N=3 generator agents propose initial answers; in subsequent rounds an untracked summarize subroutine summarizes the OTHER agents' previous responses, and each critic agent updates its own answer from that summary. The dataset's ground-truth label is **intentionally ignored** — rewards come from a majority vote over the agents' own final critic responses (the paper's self-improvement-without-ground-truth setup).

## Files

* `config.yaml`: generator + critic policy schema (both trainable, paired with their own SGLang engines).
* `run-qwen3-0.6B-multiagent-debate.sh`: launch script (ray start + train_multi_policy.py).
* `agent_system.py`: paper-aligned debate orchestration (round 0 generators, summarize subroutine, critic rounds, ŷ majority vote, reward propagation).
* `rollout_with_multi_agents.py`: top-level multi-agent rollout entrypoint.
* `prompts.py`: generator / summarize / critic prompt templates.

## Quick Start

```bash
cd slime
bash examples/multi_policy_multiagent_debate/run-qwen3-0.6B-multiagent-debate.sh
```

## How It Works

* Pipeline per prompt (N=3 agents, M=3 rounds):
  * **m=0**: N parallel generators (`A^G`) propose initial responses → trained as `generator`.
  * **m=1..M-1**: for each agent i, an untracked summarize step (paper's `A^S`, routed through the generator engine but NOT trained as a separate policy) summarizes the OTHER N-1 agents' round-(m-1) responses; agent i then runs a critic step on summary + its own prior response → trained as `critic`.
* Reward (Algorithm 1 lines 23-26):
  * `ŷ` = majority vote over the FINAL critic responses (per prompt).
  * **generator** (round 0): `1` if its boxed answer = ŷ, else `0`.
  * **critic** (any round): trajectory-level — `1` if THIS agent's FINAL critic response = ŷ, propagated to all of that agent's critic rounds.

## Policies

| policy | megatron | sglang | trainable | role |
|---|---|---|---|---|
| `generator` | ✓ | ✓ | ✓ | round-0 answer generator |
| `critic` | ✓ | ✓ | ✓ | round-1+ answer updater |

Cluster: 4 GPUs (2 megatron + 2 sglang, no colocate).
