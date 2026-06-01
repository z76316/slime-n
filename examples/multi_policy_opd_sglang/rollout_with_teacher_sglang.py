"""Custom rollout for multi-policy OPD with an SGLang-backend teacher.

Student generates → POST to teacher_sglang for prefix-only logprob scoring → dual-write to:
  - sample.teacher_log_probs        (OPD signal; overwritten by teacher_megatron when present)
  - sample.teacher_sglang_log_probs (diagnostic copy; never overwritten)
"""

from __future__ import annotations

import torch

from slime.rollout.sglang_rollout import generate, get_model_url
from slime.utils.http_utils import post
from slime.utils.types import Sample


async def generate_with_teacher_sglang(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    sample = await generate(args, sample, sampling_params)

    if sample.status == Sample.Status.ABORTED or sample.response_length == 0:
        return sample

    teacher_url = get_model_url(args, "teacher_sglang")
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    resp = await post(teacher_url, payload)

    # input_token_logprobs = [logprob, token_id] pairs; skip index 0 (no
    # preceding context), keep the logprob, then slice to response tail.
    all_logprobs = torch.tensor(
        [item[0] for item in resp["meta_info"]["input_token_logprobs"][1:]],
        dtype=torch.float32,
    )
    response_logprobs = all_logprobs[-sample.response_length :]
    sample.teacher_log_probs = response_logprobs  # OPD signal; overwritten by teacher_megatron when present
    sample.teacher_sglang_log_probs = response_logprobs  # diagnostic copy; never overwritten
    return sample
