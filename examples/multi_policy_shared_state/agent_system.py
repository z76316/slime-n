import asyncio
import itertools
import logging
import re
import traceback
from copy import deepcopy

from slime.rollout.rm_hub import batched_async_rm
from slime.rollout.sglang_rollout import get_model_url
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .prompts import PEER_ROUND1_TEMPLATE, PEER_ROUND2_TEMPLATE, PEER_ROUND3_TEMPLATE

logger = logging.getLogger(__name__)

_INNER_SAMPLE_ID = itertools.count(start=1_000_000_000)
_CHAT_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>(?:user|assistant|system)?\s*")


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
                await asyncio.sleep(1)
        return None


class PeerAgent(Agent):
    async def round1(self, args, problem_statement: str, key: str) -> Sample | None:
        body = PEER_ROUND1_TEMPLATE.format(problem_statement=problem_statement)
        return await self.run(args, _wrap_user_turn(args.tokenizer, body), key=key, max_retries=3)

    async def round2(
        self, args, problem_statement: str, own_round1: str, other_round1: str, key: str
    ) -> Sample | None:
        body = PEER_ROUND2_TEMPLATE.format(
            problem_statement=problem_statement,
            own_round1_solution=_strip_chat_tokens(own_round1),
            other_round1_solution=_strip_chat_tokens(other_round1),
        )
        return await self.run(args, _wrap_user_turn(args.tokenizer, body), key=key, max_retries=3)

    async def round3(
        self,
        args,
        problem_statement: str,
        own_round1: str,
        other_round1: str,
        own_round2: str,
        other_round2: str,
        key: str,
    ) -> Sample | None:
        body = PEER_ROUND3_TEMPLATE.format(
            problem_statement=problem_statement,
            own_round1_solution=_strip_chat_tokens(own_round1),
            other_round1_solution=_strip_chat_tokens(other_round1),
            own_round2_solution=_strip_chat_tokens(own_round2),
            other_round2_solution=_strip_chat_tokens(other_round2),
        )
        return await self.run(args, _wrap_user_turn(args.tokenizer, body), key=key, max_retries=3)


async def _round1_worker(args, problem_statement: str, key: str, chain_id: int) -> Sample | None:
    try:
        sample = await PeerAgent().round1(args, problem_statement, key=key)
        if sample is not None:
            sample.metadata["round_number"] = 1
            sample.metadata["chain_id"] = chain_id
        return sample
    except Exception:
        logger.warning("round1 worker %s/%s failed:\n%s", key, chain_id, traceback.format_exc())
        return None


async def _round2_worker(
    args, problem_statement: str, own_r1: str, other_r1: str, key: str, chain_id: int
) -> Sample | None:
    try:
        sample = await PeerAgent().round2(args, problem_statement, own_r1, other_r1, key=key)
        if sample is not None:
            sample.metadata["round_number"] = 2
            sample.metadata["chain_id"] = chain_id
        return sample
    except Exception:
        logger.warning("round2 worker %s/%s failed:\n%s", key, chain_id, traceback.format_exc())
        return None


async def _round3_worker(
    args,
    problem_statement: str,
    own_r1: str,
    other_r1: str,
    own_r2: str,
    other_r2: str,
    key: str,
    chain_id: int,
) -> Sample | None:
    try:
        sample = await PeerAgent().round3(args, problem_statement, own_r1, other_r1, own_r2, other_r2, key=key)
        if sample is not None:
            sample.metadata["round_number"] = 3
            sample.metadata["chain_id"] = chain_id
        return sample
    except Exception:
        logger.warning("round3 worker %s/%s failed:\n%s", key, chain_id, traceback.format_exc())
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
    for key in ("chain_id", "round_number"):
        metadata.pop(key, None)
    placeholder.metadata = {
        **metadata,
        "raw_reward": 0.0,
        "is_padding_placeholder": True,
        "padding_donor_policy": getattr(donor, "policy_name", None),
    }
    if metadata_overrides:
        placeholder.metadata.update(metadata_overrides)
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


def _pad_peer_buffer(args, role: str, target_count: int):
    samples = args.results_dict[role]
    target_per_round = target_count // 3
    donor = _donor_pool(args, role, None)[0]

    for round_number in (1, 2, 3):
        count = sum(1 for s in samples if (s.metadata or {}).get("round_number") == round_number)
        while count < target_per_round:
            _append_placeholder(args, role, donor, {"round_number": round_number})
            count += 1
    if len(samples) > target_count:
        del samples[target_count:]


def _pad_role_buffer(args, role: str, target_count: int, donor_role: str | None = None):
    _pad_peer_buffer(args, role, target_count)
    _fixup_logprobs(args, role)


def _fixup_logprobs(args, role: str):
    samples = args.results_dict[role]
    if not any(getattr(s, "rollout_log_probs", None) is not None for s in samples):
        return

    fixed = 0
    fixed_indices = []
    for s in samples:
        response_length = getattr(s, "response_length", 0) or 0
        rollout_log_probs = getattr(s, "rollout_log_probs", None)
        if rollout_log_probs is not None and len(rollout_log_probs) == response_length:
            continue

        fixed += 1
        fixed_indices.append(getattr(s, "index", None))
        s.remove_sample = True
        s.reward = 0.0
        s.loss_mask = [0] * response_length
        s.rollout_log_probs = [0.0] * response_length
        s.metadata = dict(s.metadata or {})
        s.metadata["raw_reward"] = 0.0
        s.metadata["missing_rollout_log_probs"] = True

    if fixed:
        logger.warning(
            "role=%s fixed %s samples with missing/mismatched rollout_log_probs; indices=%s",
            role,
            fixed,
            fixed_indices[:8],
        )


async def run_agent_system(args, sample: Sample) -> list[Sample]:
    args = deepcopy(args)
    args.sample = sample
    args.results_dict = {"peer_a": [], "peer_b": []}

    raw_problem = _strip_chat_tokens(sample.prompt)
    n = args.num_parallel

    # Phase 1: round-1, both peers solve independently
    r1_tasks = []
    for chain_id in range(n):
        r1_tasks.append(_round1_worker(args, raw_problem, "peer_a", chain_id))
        r1_tasks.append(_round1_worker(args, raw_problem, "peer_b", chain_id))
    r1_results = await asyncio.gather(*r1_tasks, return_exceptions=False)

    r1_a_by_chain: dict[int, Sample | None] = {}
    r1_b_by_chain: dict[int, Sample | None] = {}
    for i, chain_id in enumerate(range(n)):
        r1_a_by_chain[chain_id] = r1_results[2 * i]
        r1_b_by_chain[chain_id] = r1_results[2 * i + 1]

    r1_real = [s for s in r1_results if s is not None]
    if r1_real:
        r1_rewards = await batched_async_rm(args, r1_real)
        for s, r in zip(r1_real, r1_rewards, strict=False):
            s.reward = r
            s.metadata["raw_reward"] = r

    if not r1_real:
        _pad_role_buffer(args, "peer_a", 3 * n)
        _pad_role_buffer(args, "peer_b", 3 * n)
        return args.results_dict["peer_a"] + args.results_dict["peer_b"]

    # Phase 2: round-2, both peers see shared state v1
    r2_tasks = []
    r2_keys = []
    for chain_id in range(n):
        a1 = r1_a_by_chain[chain_id]
        b1 = r1_b_by_chain[chain_id]
        vis_a1 = _visible_response(a1) if a1 is not None else "[no response]"
        vis_b1 = _visible_response(b1) if b1 is not None else "[no response]"

        r2_tasks.append(_round2_worker(args, raw_problem, vis_a1, vis_b1, "peer_a", chain_id))
        r2_keys.append(("peer_a", chain_id))
        r2_tasks.append(_round2_worker(args, raw_problem, vis_b1, vis_a1, "peer_b", chain_id))
        r2_keys.append(("peer_b", chain_id))

    r2_results = await asyncio.gather(*r2_tasks, return_exceptions=False)

    r2_a_by_chain: dict[int, Sample | None] = {}
    r2_b_by_chain: dict[int, Sample | None] = {}
    for (key, chain_id), sample_result in zip(r2_keys, r2_results, strict=False):
        if key == "peer_a":
            r2_a_by_chain[chain_id] = sample_result
        else:
            r2_b_by_chain[chain_id] = sample_result

    r2_real = [s for s in r2_results if s is not None]
    if r2_real:
        r2_rewards = await batched_async_rm(args, r2_real)
        for s, r in zip(r2_real, r2_rewards, strict=False):
            s.reward = r
            s.metadata["raw_reward"] = r

    # Phase 3: round-3, both peers see shared state v2
    r3_tasks = []
    r3_keys = []
    for chain_id in range(n):
        a1 = r1_a_by_chain[chain_id]
        b1 = r1_b_by_chain[chain_id]
        a2 = r2_a_by_chain.get(chain_id)
        b2 = r2_b_by_chain.get(chain_id)
        vis_a1 = _visible_response(a1) if a1 is not None else "[no response]"
        vis_b1 = _visible_response(b1) if b1 is not None else "[no response]"
        vis_a2 = _visible_response(a2) if a2 is not None else "[no response]"
        vis_b2 = _visible_response(b2) if b2 is not None else "[no response]"

        r3_tasks.append(_round3_worker(args, raw_problem, vis_a1, vis_b1, vis_a2, vis_b2, "peer_a", chain_id))
        r3_keys.append(("peer_a", chain_id))
        r3_tasks.append(_round3_worker(args, raw_problem, vis_b1, vis_a1, vis_b2, vis_a2, "peer_b", chain_id))
        r3_keys.append(("peer_b", chain_id))

    r3_results = await asyncio.gather(*r3_tasks, return_exceptions=False)

    r3_a_by_chain: dict[int, Sample | None] = {}
    r3_b_by_chain: dict[int, Sample | None] = {}
    for (key, chain_id), sample_result in zip(r3_keys, r3_results, strict=False):
        if key == "peer_a":
            r3_a_by_chain[chain_id] = sample_result
        else:
            r3_b_by_chain[chain_id] = sample_result

    r3_real = [s for s in r3_results if s is not None]
    if r3_real:
        r3_rewards = await batched_async_rm(args, r3_real)
        for s, r in zip(r3_real, r3_rewards, strict=False):
            s.reward = r
            s.metadata["raw_reward"] = r

    # Snapshot direct RM reward before chain-outcome overwrite: eval uses
    # "direct_raw_reward" for per-round lift; training uses "raw_reward".
    for s in r1_real + r2_real:
        s.metadata["direct_raw_reward"] = s.metadata.get("raw_reward", 0.0)

    # Chain-outcome: round-1/2 inherit round-3 reward; if round-3 failed, drop them.
    for chain_id in range(n):
        r3_a = r3_a_by_chain.get(chain_id)
        if r3_a is not None and "raw_reward" in (r3_a.metadata or {}):
            final_reward_a = r3_a.metadata["raw_reward"]
        else:
            final_reward_a = None

        for prior in (r1_a_by_chain.get(chain_id), r2_a_by_chain.get(chain_id)):
            if prior is None:
                continue
            if final_reward_a is not None:
                prior.reward = final_reward_a
                prior.metadata["raw_reward"] = final_reward_a
            else:
                prior.reward = 0.0
                prior.remove_sample = True
                prior.metadata["raw_reward"] = 0.0

        r3_b = r3_b_by_chain.get(chain_id)
        if r3_b is not None and "raw_reward" in (r3_b.metadata or {}):
            final_reward_b = r3_b.metadata["raw_reward"]
        else:
            final_reward_b = None

        for prior in (r1_b_by_chain.get(chain_id), r2_b_by_chain.get(chain_id)):
            if prior is None:
                continue
            if final_reward_b is not None:
                prior.reward = final_reward_b
                prior.metadata["raw_reward"] = final_reward_b
            else:
                prior.reward = 0.0
                prior.remove_sample = True
                prior.metadata["raw_reward"] = 0.0

    # Pad each peer buffer to 3 rounds x num_parallel = 12
    _pad_role_buffer(args, "peer_a", 3 * n)
    _pad_role_buffer(args, "peer_b", 3 * n)

    return args.results_dict["peer_a"] + args.results_dict["peer_b"]
