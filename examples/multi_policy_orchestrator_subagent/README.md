# Multi-Policy Orchestrator + Subagent

This example trains two paired policies on a fan-out/fan-in math chain.

```
prompt --> orchestrator round-1 --> plan (3 dispatch prompts)
                                        |
                         +--------------+--------------+
                         v              v              v
                    subagent(d1)   subagent(d2)   subagent(d3)
                         |              |              |
                         +--------------+--------------+
                                        v
           orchestrator round-2 <-- all 3 subagent outputs
                    |
                    v
              final answer
```

The orchestrator decomposes a problem into 3 approaches (round 1), dispatches
each to a shared-weight subagent policy, then synthesizes all 3 results into a
final answer (round 2).

## Reward

- `orchestrator` round 1 + round 2: RM(final_answer) (chain-outcome)
- `subagent`: RM(own_answer) (individual competence)

## Buffer Shape

- `num_parallel = 4` chains per outer prompt
- Orchestrator: 8 samples (4 round-1 + 4 round-2), `n_samples_per_prompt = 8`
- Subagent: 12 samples (4 chains x 3 approaches), `n_samples_per_prompt = 12`
- Total: 20 samples per outer prompt

The coordinator pattern ensures only one outer clone per `group_index` runs the
chain; the rest return `[]`.

## Run

No-colocate (4 GPUs):

```bash
bash examples/multi_policy_orchestrator_subagent/run-qwen3-0.6B-orchestrator-subagent.sh
```

Colocate (2 GPUs):

```bash
bash examples/multi_policy_orchestrator_subagent/run-qwen3-0.6B-orchestrator-subagent-colocate.sh
```

## Eval

The custom eval function runs the chain internally and logs:

- `eval/aime_final_pass{1,2,4}` -- synthesized final answer
- `eval/aime_subagent_pass{1,2,4}` -- individual subagent answers
- `eval/aime_best_subagent_pass{1,2,4}` -- best of 3 subagents per chain
- `eval/aime_plan_parse_failure_rate`
- `eval/aime_synthesis_lift` -- RM(final) - max(RM(subagents))
- `eval/aime_subagent_answer_agreement`
- `eval/aime_round1_truncated_ratio`
- `eval/aime_round2_truncated_ratio`
- `eval/aime_subagent_truncated_ratio`
