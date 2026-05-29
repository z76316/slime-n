import asyncio
import itertools
import logging
import re
import time
import traceback
from copy import deepcopy

from slime.rollout.rm_hub import batched_async_rm
from slime.rollout.sglang_rollout import get_model_url
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .prompts import GENERATOR_ROUND1_TEMPLATE, GENERATOR_ROUND2_TEMPLATE, VERIFIER_TEMPLATE

logger = logging.getLogger(__name__)

_INNER_SAMPLE_ID = itertools.count(start=1_000_000_000)
_CHAT_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>(?:user|assistant|system)?\s*")
_VERDICT_RE = re.compile(r"<verdict>\s*(approve|reject)\s*</verdict>", re.IGNORECASE)


def _strip_chat_tokens(text: str) -> str:
    return _CHAT_TOKEN_RE.sub("", text or "").strip()


def _wrap_user_turn(tokenizer, user_content: str) -> str:
    if getattr(tokenizer, "chat_template", None) is None:
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _split_reason_response(text: str) -> tuple[str | None, str]:
    cleaned = (text or "").replace("<|user|>", "").strip()
    if "</think>" not in cleaned:
        return None, _strip_chat_tokens(cleaned)
    reason, response = cleaned.rsplit("</think>", 1)
    return reason.strip(), _strip_chat_tokens(response)


def _visible_response(sample: Sample | None) -> str:
    if sample is None:
        return ""
    return (sample.response_content or _strip_chat_tokens(sample.response)).strip()


def _parse_verdict(text: str) -> str:
    matches = _VERDICT_RE.findall(text or "")
    return matches[-1].lower() if matches else "unparseable"


async def generate_response(args, prompt: str, key: str) -> Sample | None:
    try:
        sample = deepcopy(args.sample)
        sample.prompt = prompt
        sample.policy_name = key
        sample.index = next(_INNER_SAMPLE_ID)
        sample.metadata = dict(sample.metadata or {})
        sample.response = ""
        sample.response_length = 0
        sample.rollout_log_probs = []

        prompt_token_ids = args.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        sample.tokens = prompt_token_ids
        prompt_length = len(prompt_token_ids)

        sampling_params = deepcopy(args.sampling_params)
        sampling_params["max_new_tokens"] = min(
            sampling_params["max_new_tokens"], args.rollout_max_context_len - prompt_length
        )
        if sampling_params["max_new_tokens"] <= 0:
            return None

        payload = {"input_ids": prompt_token_ids, "sampling_params": sampling_params, "return_logprob": True}
        output = await post(get_model_url(args, key), payload)

        token_logprobs = output.get("meta_info", {}).get("output_token_logprobs") or []
        response_tokens = [item[1] for item in token_logprobs]
        response_log_probs = [item[0] for item in token_logprobs]
        sample.tokens = sample.tokens + response_tokens
        sample.response_length = len(response_tokens)
        sample.rollout_log_probs = response_log_probs
        sample.response = output.get("text", "")

        finish_type = output.get("meta_info", {}).get("finish_reason", {}).get("type")
        if finish_type == "length":
            sample.status = Sample.Status.TRUNCATED
        elif finish_type == "stop":
            sample.status = Sample.Status.COMPLETED
        else:
            sample.status = Sample.Status.FAILED

        sample.reason_content, sample.response_content = _split_reason_response(sample.response)
        args.results_dict[key].append(sample)
        return sample
    except Exception:
        logger.warning("generate_response failed for %s:\n%s", key, traceback.format_exc())
        return None


class Agent:
    async def run(self, args, prompt: str, key: str, max_retries: int = 1) -> Sample | None:
        for attempt in range(max_retries):
            sample = await generate_response(args, prompt, key=key)
            if sample is not None:
                return sample
            if attempt + 1 < max_retries:
                time.sleep(1)
        return None


class GeneratorAgent(Agent):
    async def generate_round1(self, args, problem_statement: str) -> Sample | None:
        body = GENERATOR_ROUND1_TEMPLATE.format(problem_statement=problem_statement)
        return await self.run(args, _wrap_user_turn(args.tokenizer, body), key="generator", max_retries=3)

    async def generate_round2(
        self, args, problem_statement: str, candidate_solution: str, critique: str
    ) -> Sample | None:
        body = GENERATOR_ROUND2_TEMPLATE.format(
            problem_statement=problem_statement,
            candidate_solution=_strip_chat_tokens(candidate_solution),
            critique=_strip_chat_tokens(critique),
        )
        return await self.run(args, _wrap_user_turn(args.tokenizer, body), key="generator", max_retries=3)


class VerifierAgent(Agent):
    async def critique(self, args, problem_statement: str, candidate_solution: str) -> Sample | None:
        body = VERIFIER_TEMPLATE.format(
            problem_statement=problem_statement,
            candidate_solution=_strip_chat_tokens(candidate_solution),
        )
        return await self.run(args, _wrap_user_turn(args.tokenizer, body), key="verifier", max_retries=3)


async def _generator_round1_worker(args, problem_statement: str, chain_id: int) -> Sample | None:
    try:
        sample = await GeneratorAgent().generate_round1(args, problem_statement)
        if sample is not None:
            sample.metadata["round_number"] = 1
            sample.metadata["chain_id"] = chain_id
        return sample
    except Exception:
        logger.warning("round1 worker %s failed:\n%s", chain_id, traceback.format_exc())
        return None


async def _verifier_worker(args, problem_statement: str, round1_sample: Sample, chain_id: int) -> Sample | None:
    try:
        sample = await VerifierAgent().critique(args, problem_statement, _visible_response(round1_sample))
        if sample is not None:
            sample.metadata["chain_id"] = chain_id
        return sample
    except Exception:
        logger.warning("verifier worker %s failed:\n%s", chain_id, traceback.format_exc())
        return None


async def _generator_round2_worker(
    args,
    problem_statement: str,
    round1_sample: Sample,
    verifier_sample: Sample,
    chain_id: int,
) -> Sample | None:
    try:
        sample = await GeneratorAgent().generate_round2(
            args,
            problem_statement,
            _visible_response(round1_sample),
            _visible_response(verifier_sample),
        )
        if sample is not None:
            sample.metadata["round_number"] = 2
            sample.metadata["chain_id"] = chain_id
        return sample
    except Exception:
        logger.warning("round2 worker %s failed:\n%s", chain_id, traceback.format_exc())
        return None


def _prompt_tokens(args, source: Sample) -> list[int]:
    tokens = list(getattr(source, "tokens", []) or [])
    response_length = getattr(source, "response_length", 0) or 0
    if tokens and response_length > 0:
        return tokens[:-response_length]
    if tokens:
        return tokens

    prompt = getattr(source, "prompt", "")
    if isinstance(prompt, str) and getattr(args, "tokenizer", None) is not None:
        return args.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return []


def _donor_pool(args, role: str, donor_role: str | None):
    samples = args.results_dict[role]
    donor_pool = samples if samples else (args.results_dict.get(donor_role) or [])
    if not donor_pool:
        donor_pool = [args.sample]
    return donor_pool


def _append_placeholder(args, role: str, donor: Sample, metadata_overrides: dict | None = None):
    placeholder = deepcopy(donor)
    placeholder.policy_name = role
    placeholder.index = next(_INNER_SAMPLE_ID)
    placeholder.reward = 0.0
    metadata = dict(placeholder.metadata or {})
    for key in ("verdict", "round1_correct", "chain_id", "round_number"):
        metadata.pop(key, None)
    placeholder.metadata = {
        **metadata,
        "raw_reward": 0.0,
        "is_padding_placeholder": True,
        "padding_donor_policy": getattr(donor, "policy_name", None),
    }
    if metadata_overrides:
        placeholder.metadata.update(metadata_overrides)
    if role == "generator" and "round_number" not in placeholder.metadata:
        placeholder.metadata["round_number"] = 0
    placeholder.status = Sample.Status.FAILED
    placeholder.response = ""
    placeholder.response_length = 0
    placeholder.loss_mask = []
    placeholder.remove_sample = True
    placeholder.response_content = None
    placeholder.reason_content = None
    placeholder.tokens = _prompt_tokens(args, donor)
    placeholder.rollout_log_probs = []
    placeholder.rollout_routed_experts = None
    placeholder.teacher_log_probs = None
    args.results_dict[role].append(placeholder)


def _pad_generator_buffer(args, target_count: int):
    samples = args.results_dict["generator"]
    target_per_round = target_count // 2
    donor = _donor_pool(args, "generator", None)[0]

    for round_number in (1, 2):
        count = sum(1 for s in samples if (s.metadata or {}).get("round_number") == round_number)
        while count < target_per_round:
            _append_placeholder(args, "generator", donor, {"round_number": round_number})
            count += 1
    if len(samples) > target_count:
        del samples[target_count:]


def _pad_role_buffer(args, role: str, target_count: int, donor_role: str | None = None):
    if role == "generator":
        _pad_generator_buffer(args, target_count)
        return

    samples = args.results_dict[role]
    if len(samples) >= target_count:
        del samples[target_count:]
        return

    donor_pool = _donor_pool(args, role, donor_role)

    logger.warning(
        "padding role=%s count=%s donor_policy=%s outer_index=%s",
        role,
        target_count - len(samples),
        getattr(donor_pool[0], "policy_name", None),
        getattr(args.sample, "index", None),
    )

    while len(samples) < target_count:
        _append_placeholder(args, role, donor_pool[0])


async def run_agent_system(args, sample: Sample) -> list[Sample]:
    args = deepcopy(args)
    args.sample = sample
    args.results_dict = {"generator": [], "verifier": []}

    raw_problem = _strip_chat_tokens(sample.prompt)
    n = args.num_parallel

    round1_by_chain = await asyncio.gather(
        *[_generator_round1_worker(args, raw_problem, chain_id) for chain_id in range(n)],
        return_exceptions=False,
    )
    round1_samples = [s for s in round1_by_chain if s is not None]
    if round1_samples:
        round1_rewards = await batched_async_rm(args, round1_samples)
        for s, r in zip(round1_samples, round1_rewards, strict=False):
            s.reward = r
            s.metadata["raw_reward"] = r

    if not round1_samples:
        _pad_role_buffer(args, "generator", 2 * n)
        _pad_role_buffer(args, "verifier", n, donor_role="generator")
        return args.results_dict["generator"] + args.results_dict["verifier"]

    verifier_tasks = []
    verifier_chain_ids = []
    for chain_id, round1_sample in enumerate(round1_by_chain):
        if round1_sample is None:
            continue
        verifier_tasks.append(_verifier_worker(args, raw_problem, round1_sample, chain_id))
        verifier_chain_ids.append(chain_id)
    verifier_results = await asyncio.gather(*verifier_tasks, return_exceptions=False)
    verifier_by_chain = dict(zip(verifier_chain_ids, verifier_results, strict=False))

    for chain_id, verifier_sample in verifier_by_chain.items():
        if verifier_sample is None:
            continue
        verdict = _parse_verdict(_visible_response(verifier_sample))
        round1_sample = round1_by_chain[chain_id]
        verifier_sample.metadata["verdict"] = verdict
        verifier_sample.metadata["round1_correct"] = (round1_sample.metadata or {}).get("raw_reward", 0.0)

    round2_tasks = []
    round2_chain_ids = []
    for chain_id, round1_sample in enumerate(round1_by_chain):
        verifier_sample = verifier_by_chain.get(chain_id)
        if round1_sample is None or verifier_sample is None:
            continue
        round2_tasks.append(_generator_round2_worker(args, raw_problem, round1_sample, verifier_sample, chain_id))
        round2_chain_ids.append(chain_id)
    round2_results = await asyncio.gather(*round2_tasks, return_exceptions=False)
    round2_by_chain = dict(zip(round2_chain_ids, round2_results, strict=False))

    round2_samples = [s for s in round2_by_chain.values() if s is not None]
    if round2_samples:
        round2_rewards = await batched_async_rm(args, round2_samples)
        for s, r in zip(round2_samples, round2_rewards, strict=False):
            s.reward = r
            s.metadata["raw_reward"] = r

    for chain_id, verifier_sample in verifier_by_chain.items():
        if verifier_sample is None:
            continue
        round2_sample = round2_by_chain.get(chain_id)
        if round2_sample is None or "raw_reward" not in (round2_sample.metadata or {}):
            verifier_sample.reward = 0.0
            verifier_sample.remove_sample = True
            verifier_sample.metadata["raw_reward"] = 0.0
            continue
        reward = round2_sample.metadata["raw_reward"]
        verifier_sample.reward = reward
        verifier_sample.metadata["raw_reward"] = reward

    _pad_role_buffer(args, "generator", 2 * n)
    _pad_role_buffer(args, "verifier", n, donor_role="generator")
    return args.results_dict["generator"] + args.results_dict["verifier"]
