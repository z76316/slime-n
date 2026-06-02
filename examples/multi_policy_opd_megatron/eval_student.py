"""Eval-only rollout for the OPD dualteacher run — route each eval prompt to the
trainable `student` policy's SGLang engine and return a single Sample.

Why this exists — the dualteacher train rollout uses
``--custom-generate-function-path generate_with_teacher_sglang`` (student gen +
a POST to the frozen `teacher_sglang` engine for offline logprob diff). At EVAL
time slime resolves the per-sample generate as
``sample.generate_function_path or args.custom_generate_function_path``
(slime/rollout/sglang_rollout.py), so WITHOUT a per-dataset override eval would
re-run ``generate_with_teacher_sglang`` — adding a hard dependency on the
teacher_sglang engine and pointless teacher scoring (eval only uses
``sample.reward``). This function generates from the student only.

Targets the fixed policy name ``student`` and POSTs once to the student engine
via ``get_model_url(args, "student")`` — robust to policy order in the YAML,
unlike the plain ``generate`` which hardcodes the global (models[0]) router.

Keep the eval-config WITHOUT a ``policies:`` field so the legacy resolver hands
the dataset to the first trainable-paired policy (student) only — a ``policies:``
field trips the not-yet-wired per-policy-eval guard in train_multi_policy.
"""

import logging
from copy import deepcopy

from transformers import AutoTokenizer

from slime.rollout.sglang_rollout import get_model_url
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_EVAL_POLICY = "student"  # the only trainable+sglang policy in the dualteacher config
_TOKENIZER = None


def _tokenizer(args):
    global _TOKENIZER
    if _TOKENIZER is None:
        # Instruct/Thinking share the Qwen3 tokenizer/vocab, so the input_ids
        # produced here match whatever the student engine expects.
        _TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    return _TOKENIZER


async def generate_eval_student(args, sample: Sample, sampling_params, evaluation: bool = True) -> Sample:
    tokenizer = _tokenizer(args)
    url = get_model_url(args, _EVAL_POLICY)
    max_context_len = getattr(args, "eval_max_context_len", None) or args.rollout_max_context_len

    prompt_token_ids = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
    sample.tokens = prompt_token_ids
    prompt_length = len(prompt_token_ids)

    sp = deepcopy(sampling_params)
    sp["max_new_tokens"] = min(sampling_params["max_new_tokens"], max_context_len - prompt_length)
    if sp["max_new_tokens"] <= 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    output = await post(url, {"input_ids": prompt_token_ids, "sampling_params": sp, "return_logprob": True})

    meta = output["meta_info"]
    new_tokens = [item[1] for item in meta.get("output_token_logprobs", [])]
    sample.tokens = sample.tokens + new_tokens
    sample.response_length = len(new_tokens)
    sample.response = output["text"]
    sample.policy_name = _EVAL_POLICY
    match meta["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "stop":
            sample.status = Sample.Status.COMPLETED
    return sample
