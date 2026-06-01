import asyncio
import dataclasses
import itertools
import logging
import re
import traceback
from copy import deepcopy

from slime.rollout.rm_hub import batched_async_rm
from slime.rollout.sglang_rollout import get_model_url
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .prompts import (
    ORCHESTRATOR_PLAN_TEMPLATE,
    ORCHESTRATOR_SYNTHESIZE_TEMPLATE,
    SUBAGENT_FALLBACK_DISPATCH,
    SUBAGENT_TEMPLATE,
)

logger = logging.getLogger(__name__)

_INNER_SAMPLE_ID = itertools.count(start=1_000_000_000)
_CHAT_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>(?:user|assistant|system)?\s*")
_APPROACH_RE = re.compile(r"<approach_(\d+)>(.*?)</approach_\1>", re.DOTALL)

NUM_SUBAGENTS = 3


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


@dataclasses.dataclass
class PlanParseResult:
    dispatches: list[str]
    missing_tags: list[int]
    duplicate_tags: list[int]
    failed: bool


def _parse_plan(text: str, num_subagents: int = NUM_SUBAGENTS) -> PlanParseResult:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    matches = _APPROACH_RE.findall(cleaned)
    by_id: dict[int, str] = {}
    seen_counts: dict[int, int] = {}
    for tag_id_str, content in matches:
        tag_id = int(tag_id_str)
        seen_counts[tag_id] = seen_counts.get(tag_id, 0) + 1
        if tag_id not in by_id:
            by_id[tag_id] = content.strip()
    dispatches = []
    missing_tags = []
    for i in range(1, num_subagents + 1):
        if i in by_id:
            dispatches.append(by_id[i])
        else:
            dispatches.append(SUBAGENT_FALLBACK_DISPATCH)
            missing_tags.append(i)
    duplicate_tags = [tid for tid, cnt in seen_counts.items() if cnt > 1 and 1 <= tid <= num_subagents]
    failed = bool(missing_tags or duplicate_tags)
    return PlanParseResult(dispatches, missing_tags, duplicate_tags, failed)


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


class OrchestratorAgent(Agent):
    async def plan(self, args, problem_statement: str) -> Sample | None:
        body = ORCHESTRATOR_PLAN_TEMPLATE.format(problem_statement=problem_statement)
        return await self.run(args, _wrap_user_turn(args.tokenizer, body), key="orchestrator", max_retries=3)

    async def synthesize(self, args, problem_statement: str, plan_text: str, results: list[str]) -> Sample | None:
        body = ORCHESTRATOR_SYNTHESIZE_TEMPLATE.format(
            problem_statement=problem_statement,
            plan=_strip_chat_tokens(plan_text),
            result_1=_strip_chat_tokens(results[0]) if len(results) > 0 else "[no response]",
            result_2=_strip_chat_tokens(results[1]) if len(results) > 1 else "[no response]",
            result_3=_strip_chat_tokens(results[2]) if len(results) > 2 else "[no response]",
        )
        return await self.run(args, _wrap_user_turn(args.tokenizer, body), key="orchestrator", max_retries=3)


class SubagentAgent(Agent):
    async def solve(self, args, problem_statement: str, dispatch_instruction: str) -> Sample | None:
        body = SUBAGENT_TEMPLATE.format(
            problem_statement=problem_statement,
            dispatch_instruction=dispatch_instruction,
        )
        return await self.run(args, _wrap_user_turn(args.tokenizer, body), key="subagent", max_retries=3)


async def _plan_worker(args, problem_statement: str, chain_id: int) -> Sample | None:
    try:
        sample = await OrchestratorAgent().plan(args, problem_statement)
        if sample is not None:
            sample.metadata["round_number"] = 1
            sample.metadata["chain_id"] = chain_id
        return sample
    except Exception:
        logger.warning("plan worker %s failed:\n%s", chain_id, traceback.format_exc())
        return None


async def _subagent_worker(
    args, problem_statement: str, dispatch_instruction: str, chain_id: int, approach_index: int
) -> Sample | None:
    try:
        sample = await SubagentAgent().solve(args, problem_statement, dispatch_instruction)
        if sample is not None:
            sample.metadata["chain_id"] = chain_id
            sample.metadata["approach_index"] = approach_index
        return sample
    except Exception:
        logger.warning(
            "subagent worker chain=%s approach=%s failed:\n%s", chain_id, approach_index, traceback.format_exc()
        )
        return None


async def _synthesize_worker(
    args,
    problem_statement: str,
    plan_sample: Sample,
    subagent_responses: list[str],
    chain_id: int,
) -> Sample | None:
    try:
        sample = await OrchestratorAgent().synthesize(
            args, problem_statement, _visible_response(plan_sample), subagent_responses
        )
        if sample is not None:
            sample.metadata["round_number"] = 2
            sample.metadata["chain_id"] = chain_id
        return sample
    except Exception:
        logger.warning("synthesize worker %s failed:\n%s", chain_id, traceback.format_exc())
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
    for key in (
        "chain_id",
        "round_number",
        "approach_index",
        "plan_parse_failed",
        "plan_missing_tags",
        "plan_duplicate_tags",
    ):
        metadata.pop(key, None)
    placeholder.metadata = {
        **metadata,
        "raw_reward": 0.0,
        "is_padding_placeholder": True,
        "padding_donor_policy": getattr(donor, "policy_name", None),
    }
    if metadata_overrides:
        placeholder.metadata.update(metadata_overrides)
    if role == "orchestrator" and "round_number" not in placeholder.metadata:
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


def _pad_orchestrator_buffer(args, target_count: int):
    samples = args.results_dict["orchestrator"]
    target_per_round = target_count // 2
    donor = _donor_pool(args, "orchestrator", None)[0]

    for round_number in (1, 2):
        count = sum(1 for s in samples if (s.metadata or {}).get("round_number") == round_number)
        while count < target_per_round:
            _append_placeholder(args, "orchestrator", donor, {"round_number": round_number})
            count += 1
    if len(samples) > target_count:
        del samples[target_count:]


def _pad_role_buffer(args, role: str, target_count: int, donor_role: str | None = None):
    if role == "orchestrator":
        _pad_orchestrator_buffer(args, target_count)
        _fixup_logprobs(args, role)
        return

    samples = args.results_dict[role]
    if len(samples) >= target_count:
        del samples[target_count:]
        _fixup_logprobs(args, role)
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
    args.results_dict = {"orchestrator": [], "subagent": []}

    raw_problem = _strip_chat_tokens(sample.prompt)
    n = args.num_parallel

    # Phase 1: plan (orchestrator round-1)
    plan_by_chain = await asyncio.gather(
        *[_plan_worker(args, raw_problem, chain_id) for chain_id in range(n)],
        return_exceptions=False,
    )

    if not any(s is not None for s in plan_by_chain):
        _pad_role_buffer(args, "orchestrator", 2 * n)
        _pad_role_buffer(args, "subagent", n * NUM_SUBAGENTS, donor_role="orchestrator")
        return args.results_dict["orchestrator"] + args.results_dict["subagent"]

    # Parse plans into dispatches
    parse_results: dict[int, PlanParseResult] = {}
    for chain_id, plan_sample in enumerate(plan_by_chain):
        if plan_sample is None:
            continue
        result = _parse_plan(_visible_response(plan_sample))
        parse_results[chain_id] = result
        plan_sample.metadata["plan_parse_failed"] = result.failed
        plan_sample.metadata["plan_missing_tags"] = result.missing_tags
        plan_sample.metadata["plan_duplicate_tags"] = result.duplicate_tags

    # Phase 2: subagents (all chains × approaches in parallel)
    subagent_tasks = []
    subagent_keys = []  # (chain_id, approach_index)
    for chain_id, plan_sample in enumerate(plan_by_chain):
        if plan_sample is None:
            continue
        dispatches = parse_results[chain_id].dispatches
        for approach_index, dispatch in enumerate(dispatches):
            subagent_tasks.append(_subagent_worker(args, raw_problem, dispatch, chain_id, approach_index))
            subagent_keys.append((chain_id, approach_index))

    subagent_results = await asyncio.gather(*subagent_tasks, return_exceptions=False)
    # chain_id -> [sample_or_None per approach]
    subagent_by_chain: dict[int, list[Sample | None]] = {}
    for (chain_id, approach_index), sub_sample in zip(subagent_keys, subagent_results, strict=False):
        if chain_id not in subagent_by_chain:
            subagent_by_chain[chain_id] = [None] * NUM_SUBAGENTS
        subagent_by_chain[chain_id][approach_index] = sub_sample

    # Score subagents
    real_subagent_samples = [s for s in subagent_results if s is not None]
    if real_subagent_samples:
        subagent_rewards = await batched_async_rm(args, real_subagent_samples)
        for s, r in zip(real_subagent_samples, subagent_rewards, strict=False):
            s.reward = r
            s.metadata["raw_reward"] = r

    # Mark infra-failed subagents
    for s in subagent_results:
        if s is not None and "raw_reward" not in s.metadata:
            s.reward = 0.0
            s.remove_sample = True
            s.metadata["raw_reward"] = 0.0

    # Phase 3: synthesis (orchestrator round-2)
    synth_tasks = []
    synth_chain_ids = []
    for chain_id, plan_sample in enumerate(plan_by_chain):
        if plan_sample is None:
            continue
        subs = subagent_by_chain.get(chain_id, [None] * NUM_SUBAGENTS)
        sub_responses = []
        for sub in subs:
            sub_responses.append(_visible_response(sub) if sub is not None else "[no response]")
        synth_tasks.append(_synthesize_worker(args, raw_problem, plan_sample, sub_responses, chain_id))
        synth_chain_ids.append(chain_id)

    synth_results = await asyncio.gather(*synth_tasks, return_exceptions=False)
    synth_by_chain = dict(zip(synth_chain_ids, synth_results, strict=False))

    # Score synthesis
    real_synth_samples = [s for s in synth_by_chain.values() if s is not None]
    if real_synth_samples:
        synth_rewards = await batched_async_rm(args, real_synth_samples)
        for s, r in zip(real_synth_samples, synth_rewards, strict=False):
            s.reward = r
            s.metadata["raw_reward"] = r

    # Plan reward = its chain's synthesis (final) reward
    for chain_id, plan_sample in enumerate(plan_by_chain):
        if plan_sample is None:
            continue
        synth_sample = synth_by_chain.get(chain_id)
        if synth_sample is None or "raw_reward" not in (synth_sample.metadata or {}):
            plan_sample.reward = 0.0
            plan_sample.remove_sample = True
            plan_sample.metadata["raw_reward"] = 0.0
            continue
        final_reward = synth_sample.metadata["raw_reward"]
        plan_sample.reward = final_reward
        plan_sample.metadata["raw_reward"] = final_reward

    # Pad buffers
    _pad_role_buffer(args, "orchestrator", 2 * n)
    _pad_role_buffer(args, "subagent", n * NUM_SUBAGENTS, donor_role="orchestrator")

    return args.results_dict["orchestrator"] + args.results_dict["subagent"]
