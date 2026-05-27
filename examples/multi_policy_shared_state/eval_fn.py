import asyncio
import copy
import re
from argparse import Namespace
from typing import Any

from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

from .rollout_with_shared_state import generate_with_shared_state

_EVAL_DATASET_CACHE: dict[Any, Dataset] = {}
_PASSK_KS = (1, 2, 4)
_BOXED_RE = re.compile(r"\\boxed\{")  # sentinel for _extract_boxed


def _load_eval_dataset(args: Namespace, dataset_cfg) -> Dataset:
    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in _EVAL_DATASET_CACHE:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        _EVAL_DATASET_CACHE[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    return _EVAL_DATASET_CACHE[cache_key]


def eval_with_shared_state(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = True
) -> RolloutFnEvalOutput:
    assert evaluation, "eval_with_shared_state is the eval-only entry point"
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    eval_datasets = getattr(args, "eval_datasets", None) or []
    assert eval_datasets, "eval_with_shared_state requires --eval-config with at least one dataset"

    results: dict[str, dict[str, list[Any]]] = {}
    for dataset_cfg in eval_datasets:
        results.update(_eval_one_dataset(args, dataset_cfg))
    return RolloutFnEvalOutput(data=results)


def _eval_one_dataset(args: Namespace, dataset_cfg) -> dict[str, dict[str, list[Any]]]:
    dataset = _load_eval_dataset(args, dataset_cfg)
    sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    async def run_one(prompt_sample):
        sample = copy.deepcopy(prompt_sample)
        sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
        chain = await generate_with_shared_state(args, sample, sampling_params, evaluation=True)
        peer_a = [s for s in chain if s.policy_name == "peer_a" and not _is_placeholder(s)]
        peer_b = [s for s in chain if s.policy_name == "peer_b" and not _is_placeholder(s)]
        return peer_a, peer_b

    async def gather_all():
        tasks = [asyncio.create_task(run_one(ps)) for ps in dataset.samples]
        out = []
        pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}")
        for coro in asyncio.as_completed(tasks):
            out.append(await coro)
            pbar.update(1)
        pbar.close()
        return out

    # Per-prompt accumulators
    peer_a_r3_rewards_pp: list[list[float]] = []
    peer_b_r3_rewards_pp: list[list[float]] = []
    round1_pooled_pp: list[list[float]] = []
    round2_pooled_pp: list[list[float]] = []
    combined_r3_pp: list[list[float]] = []

    lift_r1_to_r2_a: list[float] = []
    lift_r2_to_r3_a: list[float] = []
    lift_r1_to_r2_b: list[float] = []
    lift_r2_to_r3_b: list[float] = []
    total_lift_a: list[float] = []
    total_lift_b: list[float] = []

    cross_peer_agreement_r1: list[float] = []
    cross_peer_agreement_r3: list[float] = []

    peer_a_r1_truncated: list[float] = []
    peer_a_r2_truncated: list[float] = []
    peer_a_r3_truncated: list[float] = []
    peer_b_r1_truncated: list[float] = []
    peer_b_r2_truncated: list[float] = []
    peer_b_r3_truncated: list[float] = []

    all_peer_a: list[Sample] = []
    all_peer_b: list[Sample] = []

    for peer_a, peer_b in run(gather_all()):
        all_peer_a.extend(peer_a)
        all_peer_b.extend(peer_b)

        a_by_round = _group_by_round(peer_a)
        b_by_round = _group_by_round(peer_b)
        a_by_chain = _group_by_chain(peer_a)
        b_by_chain = _group_by_chain(peer_b)

        # peer_a round-3 pass@k
        a_r3 = [_raw_reward(s) for s in a_by_round.get(3, [])]
        if a_r3:
            peer_a_r3_rewards_pp.append(a_r3)

        # peer_b round-3 pass@k
        b_r3 = [_raw_reward(s) for s in b_by_round.get(3, [])]
        if b_r3:
            peer_b_r3_rewards_pp.append(b_r3)

        # pooled round-1 pass@k (direct RM reward, not chain-outcome)
        r1_pool = [_direct_reward(s) for s in a_by_round.get(1, [])] + [_direct_reward(s) for s in b_by_round.get(1, [])]
        if r1_pool:
            round1_pooled_pp.append(r1_pool)

        # pooled round-2 pass@k (direct RM reward, not chain-outcome)
        r2_pool = [_direct_reward(s) for s in a_by_round.get(2, [])] + [_direct_reward(s) for s in b_by_round.get(2, [])]
        if r2_pool:
            round2_pooled_pp.append(r2_pool)

        # combined round-3 pass@k (both peers pooled)
        combined = a_r3 + b_r3
        if combined:
            combined_r3_pp.append(combined)

        # Per-chain lifts for peer_a (direct RM reward per round)
        for chain_samples in a_by_chain.values():
            rds = {(s.metadata or {}).get("round_number"): _direct_reward(s) for s in chain_samples}
            if 1 in rds and 2 in rds:
                lift_r1_to_r2_a.append(rds[2] - rds[1])
            if 2 in rds and 3 in rds:
                lift_r2_to_r3_a.append(rds[3] - rds[2])
            if 1 in rds and 3 in rds:
                total_lift_a.append(rds[3] - rds[1])

        # Per-chain lifts for peer_b (direct RM reward per round)
        for chain_samples in b_by_chain.values():
            rds = {(s.metadata or {}).get("round_number"): _direct_reward(s) for s in chain_samples}
            if 1 in rds and 2 in rds:
                lift_r1_to_r2_b.append(rds[2] - rds[1])
            if 2 in rds and 3 in rds:
                lift_r2_to_r3_b.append(rds[3] - rds[2])
            if 1 in rds and 3 in rds:
                total_lift_b.append(rds[3] - rds[1])

        # Cross-peer agreement
        a_r1_by_chain = {_chain_id(s): s for s in a_by_round.get(1, []) if _chain_id(s) is not None}
        b_r1_by_chain = {_chain_id(s): s for s in b_by_round.get(1, []) if _chain_id(s) is not None}
        for cid in set(a_r1_by_chain) & set(b_r1_by_chain):
            ans_a = _extract_boxed(_visible_text(a_r1_by_chain[cid]))
            ans_b = _extract_boxed(_visible_text(b_r1_by_chain[cid]))
            if ans_a is not None and ans_b is not None:
                cross_peer_agreement_r1.append(1.0 if ans_a == ans_b else 0.0)

        a_r3_by_chain = {_chain_id(s): s for s in a_by_round.get(3, []) if _chain_id(s) is not None}
        b_r3_by_chain = {_chain_id(s): s for s in b_by_round.get(3, []) if _chain_id(s) is not None}
        for cid in set(a_r3_by_chain) & set(b_r3_by_chain):
            ans_a = _extract_boxed(_visible_text(a_r3_by_chain[cid]))
            ans_b = _extract_boxed(_visible_text(b_r3_by_chain[cid]))
            if ans_a is not None and ans_b is not None:
                cross_peer_agreement_r3.append(1.0 if ans_a == ans_b else 0.0)

        # Truncated ratios
        for s in a_by_round.get(1, []):
            peer_a_r1_truncated.append(float(s.status == Sample.Status.TRUNCATED))
        for s in a_by_round.get(2, []):
            peer_a_r2_truncated.append(float(s.status == Sample.Status.TRUNCATED))
        for s in a_by_round.get(3, []):
            peer_a_r3_truncated.append(float(s.status == Sample.Status.TRUNCATED))
        for s in b_by_round.get(1, []):
            peer_b_r1_truncated.append(float(s.status == Sample.Status.TRUNCATED))
        for s in b_by_round.get(2, []):
            peer_b_r2_truncated.append(float(s.status == Sample.Status.TRUNCATED))
        for s in b_by_round.get(3, []):
            peer_b_r3_truncated.append(float(s.status == Sample.Status.TRUNCATED))

    base = dataset_cfg.name
    out: dict[str, dict[str, list[Any]]] = {}

    for k in _PASSK_KS:
        out[f"{base}_peer_a_pass{k}"] = _ds(_pass_at_k_per_prompt(peer_a_r3_rewards_pp, k), all_peer_a)
        out[f"{base}_peer_b_pass{k}"] = _ds(_pass_at_k_per_prompt(peer_b_r3_rewards_pp, k), all_peer_b)
        out[f"{base}_round1_pass{k}"] = _ds(_pass_at_k_per_prompt(round1_pooled_pp, k), all_peer_a + all_peer_b)
        out[f"{base}_round2_pass{k}"] = _ds(_pass_at_k_per_prompt(round2_pooled_pp, k), all_peer_a + all_peer_b)
        out[f"{base}_combined_pass{k}"] = _ds(_pass_at_k_per_prompt(combined_r3_pp, k), all_peer_a + all_peer_b)

    out[f"{base}_lift_r1_to_r2_a"] = _ds(_nonempty(lift_r1_to_r2_a), all_peer_a)
    out[f"{base}_lift_r2_to_r3_a"] = _ds(_nonempty(lift_r2_to_r3_a), all_peer_a)
    out[f"{base}_lift_r1_to_r2_b"] = _ds(_nonempty(lift_r1_to_r2_b), all_peer_b)
    out[f"{base}_lift_r2_to_r3_b"] = _ds(_nonempty(lift_r2_to_r3_b), all_peer_b)
    out[f"{base}_total_lift_a"] = _ds(_nonempty(total_lift_a), all_peer_a)
    out[f"{base}_total_lift_b"] = _ds(_nonempty(total_lift_b), all_peer_b)

    out[f"{base}_cross_peer_agreement_r1"] = _ds(_nonempty(cross_peer_agreement_r1), all_peer_a + all_peer_b)
    out[f"{base}_cross_peer_agreement_r3"] = _ds(_nonempty(cross_peer_agreement_r3), all_peer_a + all_peer_b)

    out[f"{base}_peer_a_round1_truncated_ratio"] = _ds(_nonempty(peer_a_r1_truncated), all_peer_a)
    out[f"{base}_peer_a_round2_truncated_ratio"] = _ds(_nonempty(peer_a_r2_truncated), all_peer_a)
    out[f"{base}_peer_a_round3_truncated_ratio"] = _ds(_nonempty(peer_a_r3_truncated), all_peer_a)
    out[f"{base}_peer_b_round1_truncated_ratio"] = _ds(_nonempty(peer_b_r1_truncated), all_peer_b)
    out[f"{base}_peer_b_round2_truncated_ratio"] = _ds(_nonempty(peer_b_r2_truncated), all_peer_b)
    out[f"{base}_peer_b_round3_truncated_ratio"] = _ds(_nonempty(peer_b_r3_truncated), all_peer_b)
    return out


def _raw_reward(sample: Sample) -> float:
    return float((sample.metadata or {}).get("raw_reward", sample.reward or 0.0))


def _direct_reward(sample: Sample) -> float:
    md = sample.metadata or {}
    return float(md.get("direct_raw_reward", md.get("raw_reward", sample.reward or 0.0)))


def _chain_id(sample: Sample):
    return (sample.metadata or {}).get("chain_id")


def _is_placeholder(sample: Sample) -> bool:
    return bool((sample.metadata or {}).get("is_padding_placeholder"))


def _visible_text(sample: Sample) -> str:
    return (sample.response_content or (sample.response or "")).strip()


def _extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} content with balanced braces."""
    last = None
    for m in _BOXED_RE.finditer(text or ""):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[start : i - 1].strip()
    return last


def _group_by_round(samples: list[Sample]) -> dict[int, list[Sample]]:
    out: dict[int, list[Sample]] = {}
    for s in samples:
        rn = (s.metadata or {}).get("round_number")
        if rn is not None:
            out.setdefault(rn, []).append(s)
    return out


def _group_by_chain(samples: list[Sample]) -> dict[int, list[Sample]]:
    out: dict[int, list[Sample]] = {}
    for s in samples:
        cid = _chain_id(s)
        if cid is not None:
            out.setdefault(cid, []).append(s)
    return out


def _nonempty(values: list[float]) -> list[float]:
    return values if values else [0.0]


def _pass_at_k_per_prompt(rewards_per_prompt: list[list[float]], k: int) -> list[float]:
    out = []
    for rewards in rewards_per_prompt:
        n = len(rewards)
        c = sum(1 for r in rewards if r == 1)
        if k > n:
            continue
        if n - c < k:
            out.append(1.0)
        else:
            p = 1.0
            for i in range(n - c + 1, n + 1):
                p *= 1.0 - k / i
            out.append(1.0 - p)
    return _nonempty(out)


def _ds(rewards: list[float], samples: list[Sample]) -> dict[str, list[Any]]:
    if not samples:
        samples = [Sample(response="", response_length=0, reward=0.0, status=Sample.Status.FAILED)]
    return {
        "rewards": rewards,
        "truncated": [s.status == Sample.Status.TRUNCATED for s in samples],
        "samples": samples,
    }
