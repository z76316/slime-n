"""Custom rollout for multi-policy OPD with an SGLang-backend teacher.

STATUS: SPEC. Not yet implemented end-to-end. The shape below describes
what `--custom-generate-function-path` should resolve to once F1/F2/F3
land; until then this module raises NotImplementedError.

Pipeline:
    1. Generate the student's response via the student's SGLang engine
       (identical to slime's default rollout).
    2. For each generated sample, POST to teacher_sglang's /generate
       endpoint with `return_logprobs=True` over the response tokens —
       prefix-only scoring, no new generation. Get per-token logprobs.
    3. Stamp sample.teacher_log_probs = [...].
    4. Return the samples; downstream
       `_convert_samples_to_train_data` (slime/ray/rollout.py:984-985)
       hoists per-sample teacher_log_probs into
       train_data["teacher_log_probs"], which the student's train_actor
       reads via rollout_data and apply_opd_kl_to_advantages applies as
       reverse-KL.

Key dependency: looking up teacher_sglang's URL from the rollout manager.
The rollout manager's `_policy_to_server` map (populated by
`register_policy`) lets us go policy_name → SGLang server endpoint.
"""

from __future__ import annotations

from slime.utils.types import Sample


async def generate_with_teacher_sglang(
    args, sample: Sample, sampling_params, evaluation: bool = False
) -> list[Sample]:
    """Student rollout + teacher logprob scoring.

    Skeleton:

        # 1. Student generation (same as slime's default rollout)
        from slime.rollout.sglang_rollout import generate_default
        samples = await generate_default(args, sample, sampling_params, evaluation)

        # 2. Look up the teacher's SGLang URL via the rollout manager.
        #    Requires F1/F2/F3 to have landed so teacher_sglang's engine
        #    is correctly spawned and registered.
        teacher_url = _resolve_teacher_url(args, "teacher_sglang")

        # 3. For each sample, score the response tokens under the teacher.
        for s in samples:
            tokens = _extract_response_tokens(s)
            tlp = await _query_logprobs(teacher_url, tokens)
            s.teacher_log_probs = tlp

        return samples

    The pieces marked _placeholder need wiring:

      * _resolve_teacher_url: read rollout_manager._policy_to_server to
        find teacher_sglang's server name, then read its endpoint. Today
        rollout_manager is in a different ray actor, so this needs
        either an env-var hand-off at startup or a small RPC.

      * _query_logprobs: mirror slime/rollout/on_policy_distillation.py:
        reward_func's POST shape — `return_logprobs=True`, with the
        student's response tokens as the prefix.

      * _extract_response_tokens: convert sample.response (text) into
        the token-id sequence the teacher's tokenizer expects. Same
        tokenizer as the student in v1 (validated by the cross-policy
        validator in plan_opd.md §B2).
    """
    raise NotImplementedError(
        "rollout_with_teacher_sglang.generate_with_teacher_sglang is a "
        "design spec. Implementing end-to-end requires F1/F2/F3 from "
        "plan_colocate.md (engine-hosting predicate, slice fix, "
        "non-Megatron-actor offset handling) plus the teacher-URL "
        "lookup wiring described in the module docstring."
    )
