"""Top-level rollout entrypoint. Wires slime's
--custom-generate-function-path to run_agent_system in agent_system.py."""

import random

from transformers import AutoTokenizer

from slime.utils.misc import load_function
from slime.utils.types import Sample

# num_parallel = K (samples per agent); must equal n_samples_per_prompt in config.yaml.
SWARM_CONFIGS = {
    "custom_multi_agent_function_path": "examples.multi_policy_exam_swarm_rl.agent_system.run_agent_system",
    "num_parallel": 8,
}


async def generate_with_swarm(args, sample: Sample, sampling_params, evaluation: bool = False) -> list[Sample]:
    """Per outer prompt, fan out to all 8 agents (K responses each). Returns
    8 × K samples flat, tagged with policy_name for split-buffer routing."""
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    max_context_length = args.rollout_max_context_len if not evaluation else args.eval_max_context_len

    args.sampling_params = sampling_params
    args.rollout_max_context_len = max_context_length
    args.tokenizer = tokenizer

    for key, value in SWARM_CONFIGS.items():
        setattr(args, key, value)

    custom_fn = load_function(args.custom_multi_agent_function_path)
    samples = await custom_fn(args, sample)

    random.shuffle(samples)
    return samples
