import random
from copy import copy

from transformers import AutoTokenizer

from slime.utils.misc import load_function
from slime.utils.types import Sample

MULTI_AGENT_CONFIGS = {
    "custom_multi_agent_function_path": "examples.multi_policy_orchestrator_subagent.agent_system.run_agent_system",
    "num_parallel": 4,
    "num_subagents": 3,
    "outer_group_size": 1,
}


def _is_coordinator_clone(args, sample: Sample, evaluation: bool) -> bool:
    if evaluation:
        return True
    n = getattr(args, "n_samples_per_prompt", 1) or 1
    return (sample.index or 0) % n == 0


async def generate_with_orchestrator(args, sample: Sample, sampling_params, evaluation=False) -> list[Sample]:
    if not _is_coordinator_clone(args, sample, evaluation):
        return []

    local_args = copy(args)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    max_context_length = args.rollout_max_context_len if not evaluation else args.eval_max_context_len

    local_args.sampling_params = sampling_params
    local_args.rollout_max_context_len = max_context_length
    local_args.tokenizer = tokenizer

    for key, value in MULTI_AGENT_CONFIGS.items():
        setattr(local_args, key, value)

    custom_multi_agent_func = load_function(local_args.custom_multi_agent_function_path)
    samples = await custom_multi_agent_func(local_args, sample)

    # Share the prompt's group id across all siblings (rollout validator
    # requires it; each policy's n_samples_per_prompt then form one GRPO group).
    group_id = sample.group_id if sample.group_id is not None else sample.index
    for s in samples:
        s.group_id = group_id

    random.shuffle(samples)
    return samples
