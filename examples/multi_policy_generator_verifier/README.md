# Multi-Policy Generator + Verifier

This example trains two paired policies on a verify-and-revise math chain.

Per prompt, the generator produces 8 first-round answers. The verifier critiques
each answer independently. The generator then produces 8 second-round answers,
one per critique.

| schema | slime<sup>n</sup> |
|:---:|:---:|
| ![generator+verifier schema](./imgs/schema.png) | ![generator+verifier framework](./imgs/arch.png) |

*Left: answer → critique → revise, with the round-1 answer carried into round 2. Right: two trainable pairs (generator, verifier), each Megatron + SGLang with its own buffer.*

## Reward

- `generator` round 1: rule-based RM on `answer_1`
- `generator` round 2: rule-based RM on `answer_2`
- `verifier`: same-chain rule-based RM on `answer_2`

The verifier verdict is diagnostic-only in v1. It is logged as accuracy against
the round-1 RM label, but it is not used as the training reward.

## Buffer Shape

The first policy has `n_samples_per_prompt=16`, so the default slime data source
creates 16 outer clones per prompt. This example does not change slime core.
Instead, `rollout_with_verifier.py` treats one deterministic clone per
`group_index` as the coordinator and returns `[]` for the other clones.

After rollout flattening, each original prompt contributes:

- 16 generator samples: 8 round 1 + 8 round 2
- 8 verifier samples: one critique per chain

Zero-response placeholders are used only for failed stages. They set
`response_length=0`, `loss_mask=[]`, `rollout_log_probs=[]`, and
`remove_sample=True`.

## Run

No-colocate:

```bash
bash examples/multi_policy_generator_verifier/run-qwen3-0.6B-generator-verifier.sh
```

Colocate:

```bash
bash examples/multi_policy_generator_verifier/run-qwen3-0.6B-generator-verifier-colocate.sh
```

## Eval

The custom eval function runs the 8-chain loop internally and logs:

- `eval/aime_round1_pass{1,2,4,8}`
- `eval/aime_round2_pass{1,2,4,8}`
- `eval/aime_revise_lift`
- `eval/aime_verifier_accuracy`
- `eval/aime_verifier_accuracy_on_correct`
- `eval/aime_verifier_accuracy_on_incorrect`
- `eval/aime_verifier_parse_failure_rate`
- `eval/aime_round1_truncated_ratio`
- `eval/aime_round2_truncated_ratio`

## Smoke Checks

With `--dump-details`, verify:

- each original prompt has 16 generator samples and 8 verifier samples;
- generator samples split 8:8 across `round_number=1` and `round_number=2`;
- round-2 prompts contain the matching round-1 answer and verifier critique;
- placeholder samples have zero response length and zero loss mask;
- neither policy shows steadily growing `train_rollout_logprob_abs_diff`.
