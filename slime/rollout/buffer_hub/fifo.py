from typing import Any

from slime.utils.types import Sample


# Simple operations
def push_end(args, buffer: list[list[Sample]], samples: list[Sample], rollout_info: dict[str, Any]):
    """
    Simply append the samples to the end of the buffer.
    """
    buffer.append(samples)


def pop_first(args, buffer: list[list[Sample]], num_samples: int, rollout_info: dict[str, Any]):
    """
    Try to pop the first `num_samples` from the buffer.
    """
    num_to_pop = min(len(buffer), num_samples // args.n_samples_per_prompt)
    return_samples = []
    for i in range(num_to_pop):
        return_samples.extend(buffer[i])
    del buffer[:num_to_pop]
    return return_samples
