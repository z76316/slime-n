from slime.utils.types import Sample
from typing import Any


# partial filters
def clear_sample_response(sample: Sample):
    """
    Clear the response, start_rollout_id, and status of a sample.
    """
    sample.response = None
    sample.metadata["start_rollout_id"] = None
    sample.status = Sample.Status.PENDING

def valid_partial_sample(args, sample: Sample):
    """
    Check if a sample is a valid partial sample.
    """
    if (sample.response and 
        len(sample.response.strip()) < args.partial_rollout_min_response_length):
        return False
    if (sample.response_length and sample.response_length < args.partial_rollout_min_tokens):
        return False
    return True

def filter_partial_samples(args, samples: list[Sample], rollout_info: dict[str, Any]):
    """
    Update the partial samples.
    If partial samples are too short, we clear the response and status.
    """
    for sample in samples:
        if sample.status != Sample.Status.PENDING and not valid_partial_sample(args, sample):
            clear_sample_response(sample)
    # TODO(jiajun): Staleness may be handled here using rollout_info.

def partial_push_end(args, buffer: list[list[Sample]], samples: list[Sample], rollout_info: dict[str, Any]):
    """
    Push the samples to the end of the buffer.
    """
    filter_partial_samples(args, samples, rollout_info)
    for sample in samples:
        # reset partial sample's index without start_rollout_id
        if sample.status != Sample.Status.PENDING and sample.metadata.get("start_rollout_id", None) == None:
            sample.metadata["start_rollout_id"] = rollout_info["rollout_id"]
    buffer.append(samples)

def partial_pop_first(args, buffer: list[list[Sample]], num_samples: int, rollout_info: dict[str, Any]):
    """
    Filter for partial rollout.
    This function pops the front `num_samples` from the buffer.
    Partial samples are prioritized.
    """
    for samples in buffer:
        filter_partial_samples(args, samples, rollout_info)
    
    # Group samples by n_samples_per_prompt
    partial_groups_idx = []
    new_groups_idx = []
    
    for i in range(0, len(buffer)):
        samples = buffer[i]
        # Check if all samples in the group are PENDING
        all_pending = all(sample.status == Sample.Status.PENDING for sample in samples)
        if all_pending:
            new_groups_idx.append(i)
        else:
            partial_groups_idx.append(i)
    
    # Calculate how many partial groups we can take
    num_groups = num_samples // args.n_samples_per_prompt
    num_partial_groups = min(int(num_groups * args.partial_rollout_mix_ratio), len(partial_groups_idx))
    selected_partial_groups_idx = partial_groups_idx[:num_partial_groups]
    
    # Select new groups to fill the remaining quota
    num_new_groups = min(num_groups - num_partial_groups, len(new_groups_idx))
    selected_new_groups_idx = new_groups_idx[:num_new_groups]
    
    # Collect all selected samples
    selected_samples = []
    selected_idx = [False] * len(buffer)
    new_buffer = []
    for i in selected_partial_groups_idx:
        selected_samples.extend(buffer[i])
        selected_idx[i] = True
    for i in selected_new_groups_idx:
        selected_samples.extend(buffer[i])
        selected_idx[i] = True
    for i in range(len(buffer)):
        if not selected_idx[i]:
            new_buffer.append(buffer[i])

    buffer = new_buffer
    
    return selected_samples

