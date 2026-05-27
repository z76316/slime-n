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

from .rollout_with_orchestrator import generate_with_orchestrator

_EVAL_DATASET_CACHE: dict[Any, Dataset] = {}
_PASSK_KS = (1, 2, 4)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


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


def eval_with_orchestrator(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = True
) -> RolloutFnEvalOutput:
    assert evaluation, "eval_with_orchestrator is the eval-only entry point"
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    eval_datasets = getattr(args, "eval_datasets", None) or []
    assert eval_datasets, "eval_with_orchestrator requires --eval-config with at least one dataset"

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
        chain = await generate_with_orchestrator(args, sample, sampling_params, evaluation=True)
        orchestrator = [s for s in chain if s.policy_name == "orchestrator"]
        subagent = [s for s in chain if s.policy_name == "subagent"]
        round1 = [s for s in orchestrator if (s.metadata or {}).get("round_number") == 1]
        round2 = [s for s in orchestrator if (s.metadata or {}).get("round_number") == 2]
        return round1, round2, subagent

    async def gather_all():
        tasks = [asyncio.create_task(run_one(ps)) for ps in dataset.samples]
        out = []
        pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}")
        for coro in asyncio.as_completed(tasks):
            out.append(await coro)
            pbar.update(1)
        pbar.close()
        return out

    final_rewards_per_prompt: list[list[float]] = []
    subagent_rewards_per_prompt: list[list[float]] = []
    best_subagent_rewards_per_prompt: list[list[float]] = []
    round1_samples_flat: list[Sample] = []
    round2_samples_flat: list[Sample] = []
    subagent_samples_flat: list[Sample] = []
    synthesis_lift: list[float] = []
    plan_parse_failure: list[float] = []
    subagent_agreement: list[float] = []
    round1_truncated: list[float] = []
    round2_truncated: list[float] = []
    subagent_truncated: list[float] = []

    for round1, round2, subagent in run(gather_all()):
        round1_samples_flat.extend(round1)
        round2_samples_flat.extend(round2)
        subagent_samples_flat.extend(subagent)

        # final_pass@k: based on round-2 (synthesis) rewards
        r2_rewards = [_raw_reward(s) for s in round2 if not _is_placeholder(s)]
        if r2_rewards:
            final_rewards_per_prompt.append(r2_rewards)

        # subagent_pass@k: pooled across all subagent answers
        sub_rewards = [_raw_reward(s) for s in subagent if not _is_placeholder(s)]
        if sub_rewards:
            subagent_rewards_per_prompt.append(sub_rewards)

        # best_subagent_pass@k: per chain, correct if any of 3 subagents correct
        by_chain_sub: dict[int, list[float]] = {}
        for s in subagent:
            if _is_placeholder(s):
                continue
            cid = _chain_id(s)
            if cid is not None:
                by_chain_sub.setdefault(cid, []).append(_raw_reward(s))
        best_per_chain = [max(rs) for rs in by_chain_sub.values() if rs]
        if best_per_chain:
            best_subagent_rewards_per_prompt.append(best_per_chain)

        # synthesis_lift: RM(final) - max(RM(subagents)) per chain
        by_chain_r2 = {
            _chain_id(s): _raw_reward(s) for s in round2 if not _is_placeholder(s) and _chain_id(s) is not None
        }
        for cid, final_r in by_chain_r2.items():
            sub_rs = by_chain_sub.get(cid, [])
            if sub_rs:
                synthesis_lift.append(final_r - max(sub_rs))

        # plan_parse_failure_rate
        for s in round1:
            if _is_placeholder(s):
                continue
            plan_parse_failure.append(1.0 if (s.metadata or {}).get("plan_parse_failed") else 0.0)

        # subagent_answer_agreement: per chain, check if all 3 extracted answers match
        by_chain_answers: dict[int, list[str]] = {}
        for s in subagent:
            if _is_placeholder(s):
                continue
            cid = _chain_id(s)
            if cid is None:
                continue
            answer = _extract_boxed(_visible_text(s))
            if answer is not None:
                by_chain_answers.setdefault(cid, []).append(answer)
        for answers in by_chain_answers.values():
            if len(answers) >= 2:
                subagent_agreement.append(1.0 if len(set(answers)) == 1 else 0.0)

        round1_truncated.extend([float(s.status == Sample.Status.TRUNCATED) for s in round1 if not _is_placeholder(s)])
        round2_truncated.extend([float(s.status == Sample.Status.TRUNCATED) for s in round2 if not _is_placeholder(s)])
        subagent_truncated.extend(
            [float(s.status == Sample.Status.TRUNCATED) for s in subagent if not _is_placeholder(s)]
        )

    base = dataset_cfg.name
    out: dict[str, dict[str, list[Any]]] = {}
    for k in _PASSK_KS:
        out[f"{base}_final_pass{k}"] = _ds(_pass_at_k_per_prompt(final_rewards_per_prompt, k), round2_samples_flat)
        out[f"{base}_subagent_pass{k}"] = _ds(
            _pass_at_k_per_prompt(subagent_rewards_per_prompt, k), subagent_samples_flat
        )
        out[f"{base}_best_subagent_pass{k}"] = _ds(
            _pass_at_k_per_prompt(best_subagent_rewards_per_prompt, k), subagent_samples_flat
        )

    out[f"{base}_plan_parse_failure_rate"] = _ds(_nonempty(plan_parse_failure), round1_samples_flat)
    out[f"{base}_synthesis_lift"] = _ds(_nonempty(synthesis_lift), round2_samples_flat)
    out[f"{base}_subagent_answer_agreement"] = _ds(_nonempty(subagent_agreement), subagent_samples_flat)
    out[f"{base}_round1_truncated_ratio"] = _ds(_nonempty(round1_truncated), round1_samples_flat)
    out[f"{base}_round2_truncated_ratio"] = _ds(_nonempty(round2_truncated), round2_samples_flat)
    out[f"{base}_subagent_truncated_ratio"] = _ds(_nonempty(subagent_truncated), subagent_samples_flat)
    return out


def _raw_reward(sample: Sample) -> float:
    return float((sample.metadata or {}).get("raw_reward", sample.reward or 0.0))


def _chain_id(sample: Sample):
    return (sample.metadata or {}).get("chain_id")


def _is_placeholder(sample: Sample) -> bool:
    return bool((sample.metadata or {}).get("is_padding_placeholder"))


def _visible_text(sample: Sample) -> str:
    return (sample.response_content or (sample.response or "")).strip()


def _extract_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text or "")
    return matches[-1].strip() if matches else None


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
