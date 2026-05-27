import random
from copy import copy

from transformers import AutoTokenizer

from slime.utils.misc import load_function
from slime.utils.types import Sample

MULTI_AGENT_CONFIGS = {
    "custom_multi_agent_function_path": "examples.multi_policy_solver_rewriter_selector.agent_system.run_agent_system",
    # num_parallel must match n_samples_per_prompt for solver / rewriter / selector
    # in config.yaml so GRPO group-norm reshape stays on the fast path.
    "num_parallel": 4,
    "incorrect_reward_weight": 0.8,
    "correct_reward_weight": 1.2,
}


async def generate_with_multi_agents(args, sample: Sample, sampling_params, evaluation=False) -> list[Sample]:

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

    random.shuffle(samples)

    return samples
