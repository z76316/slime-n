import random

from transformers import AutoTokenizer

from slime.utils.misc import load_function
from slime.utils.types import Sample

MULTI_AGENT_CONFIGS = {
    "custom_multi_agent_function_path": "examples.multi_policy_solver_summarizer.agent_system.run_agent_system",
    # Must match n_samples_per_prompt (solver/summarizer) in config.yaml to keep
    # GRPO group-norm reshape on the fast path.
    "num_parallel": 4,
    "incorrect_reward_weight": 0.8,
    "correct_reward_weight": 1.2,
}


async def generate_with_multi_agents(args, sample: Sample, sampling_params, evaluation=False) -> list[Sample]:
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    max_context_length = args.rollout_max_context_len if not evaluation else args.eval_max_context_len

    args.sampling_params = sampling_params
    args.rollout_max_context_len = max_context_length
    args.tokenizer = tokenizer

    for key, value in MULTI_AGENT_CONFIGS.items():
        setattr(args, key, value)

    custom_multi_agent_func = load_function(args.custom_multi_agent_function_path)
    samples = await custom_multi_agent_func(args, sample)

    # All samples from one prompt share that prompt's group id: the rollout
    # validator requires group_id on every sibling of a compact rollout, and
    # after split-buffer routing each policy's n_samples_per_prompt samples
    # then form one GRPO group (instead of falling back to per-sample index).
    group_id = sample.group_id if sample.group_id is not None else sample.index
    for s in samples:
        s.group_id = group_id

    random.shuffle(samples)
    return samples
