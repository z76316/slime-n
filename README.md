<div align="center">

# $\huge\color{#1E293B}{\textsf{\textbf{slime}}}^{\color{#3B82F6}{\textsf{\textbf{n}}}}$

### A Multi-Policy, Multi-Agent RL Framework

### Scale Multi-Agent RL from 1 Policy to 100+ Policies
</div>


slime<sup>n</sup> extends [slime](https://github.com/THUDM/slime) into a flexible multi-policy and multi-agent RL training framework. Each run can be composed of any combination of three policy types:



- **Trainable policy pair** — a Megatron training actor paired with an SGLang rollout engine. The default workhorse for single policy RL.
- **Standalone Megatron actor (frozen/trainable)** — Megatron only, no SGLang engine. Trainable variant trains its own loss without rolling out (e.g. a PPO critic); frozen variant runs forward-only and emits per-token logprobs (e.g. an OPD teacher).
- **Standalone SGLang engine (frozen)** — SGLang only, no trainer; serves inference for reward models, verifier or OPD teacher.

![multi-policy architecture](./imgs/arch_2.png)

## Features

- **`train_multi_policy.py`** — driver for n≥1 trainable policies. Replaces `train.py` for multi-policy runs.
- **YAML-driven configs** — `--config <path>.yaml`. Per-policy fields (parallelism, batching, optimizer, loss, paths, Megatron numerical / dropout, `log_probs_chunk_size`) live in the YAML; cluster sizing is derived from policies. See [`slime/utils/policy_config.py`](slime/utils/policy_config.py).
- **Per-policy buffers (split mode)** — each policy trains on its own samples, tagged via `Sample.policy_name`.
- **Per-policy weight sync** — serialized push from each Megatron actor to its paired sglang engine.

## Multi-Policy YAML Schema

Multi-policy runs are defined by a single YAML file passed with `--config`. The top-level `policies` list is the source of truth for the run composition: each entry declares one policy's identity, trainability, checkpoints, buffer routing, GPU slice, Megatron training settings, and optional SGLang engine settings. Policy names must be unique, and each paired policy gets a 1:1 SGLang server with the same name.

```yaml
policies:
  - name: student                 # unique policy name; also the paired SGLang server name
    role: actor                   # only actor is supported today
    trainable: true               # false for frozen producers such as OPD teachers

    hf_checkpoint: /root/Qwen3-0.6B
    load: /ckpt/student           # optional Megatron torch_dist resume path
    save: /ckpt/student           # optional per-policy save path
    ref_load: /ckpt/ref           # optional KL reference checkpoint

    buffer_mode: split            # split | shared

    num_gpus_per_node: 1
    megatron_num_nodes: 1         # Megatron actor GPUs = nodes * num_gpus_per_node
    sglang_num_nodes: 1           # SGLang engine GPUs = nodes * num_gpus_per_node

    megatron:
      megatron_to_hf_mode: bridge
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      context_parallel_size: 1
      micro_batch_size: 1
      global_batch_size: 64
      lr: 1.0e-6
      advantage_estimator: grpo
      n_samples_per_prompt: 4
      log_probs_chunk_size: 512

    sglang:
      update_weights: true
      num_gpus_per_engine: 1
      mem_fraction_static: 0.85
      server_groups:
        - worker_type: regular
          num_gpus: 1
```

The `megatron:` block is flattened into the per-policy Megatron argument namespace, so parallelism, recompute, batching, optimizer, loss, KL, and OPD fields can differ by policy. The `sglang:` block is projected into the SGLang model/server config; `model_path` defaults to `hf_checkpoint`, and server arguments such as `mem_fraction_static`, `cuda_graph_bs`, and `max_total_tokens` are passed through to each server group.

Cluster sizing is derived from the YAML. Without `--colocate`, total GPUs are `sum(megatron_num_nodes * num_gpus_per_node) + sum(sglang_num_nodes * num_gpus_per_node)` across active policies. With `--colocate`, slime uses the larger of the Megatron and SGLang sides. For trainable policies with an engine, `sglang_num_nodes * num_gpus_per_node` must match the sum of `sglang.server_groups[].num_gpus`. A frozen standalone Megatron teacher sets `trainable: false` and `sglang_num_nodes: 0`.

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

See [`train_multi_policy.py`](train_multi_policy.py) for the train-loop body and the architecture figure above (source: `../fig_arch_2.typ`) for the runtime layout.
