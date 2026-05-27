import asyncio
import copy
from argparse import Namespace
from typing import Any

from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

from .rollout_with_verifier import generate_with_verifier

_EVAL_DATASET_CACHE: dict[Any, Dataset] = {}
_PASSK_KS = (1, 2, 4, 8)


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


def eval_with_verifier(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = True
) -> RolloutFnEvalOutput:
    assert evaluation, "eval_with_verifier is the eval-only entry point"
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    eval_datasets = getattr(args, "eval_datasets", None) or []
    assert eval_datasets, "eval_with_verifier requires --eval-config with at least one dataset"

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
        chain = await generate_with_verifier(args, sample, sampling_params, evaluation=True)
        generator = [s for s in chain if s.policy_name == "generator"]
        verifier = [s for s in chain if s.policy_name == "verifier"]
        round1 = [s for s in generator if (s.metadata or {}).get("round_number") == 1]
        round2 = [s for s in generator if (s.metadata or {}).get("round_number") == 2]
        return round1, round2, verifier

    async def gather_all():
        tasks = [asyncio.create_task(run_one(ps)) for ps in dataset.samples]
        out = []
        pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}")
        for coro in asyncio.as_completed(tasks):
            out.append(await coro)
            pbar.update(1)
        pbar.close()
        return out

    round1_rewards_per_prompt: list[list[float]] = []
    round2_rewards_per_prompt: list[list[float]] = []
    round1_samples_flat: list[Sample] = []
    round2_samples_flat: list[Sample] = []
    verifier_samples_flat: list[Sample] = []
    revise_lift: list[float] = []
    verifier_accuracy: list[float] = []
    verifier_accuracy_correct: list[float] = []
    verifier_accuracy_incorrect: list[float] = []
    verifier_parse_failure: list[float] = []
    round1_truncated: list[float] = []
    round2_truncated: list[float] = []

    for round1, round2, verifier in run(gather_all()):
        round1_samples_flat.extend(round1)
        round2_samples_flat.extend(round2)
        verifier_samples_flat.extend(verifier)

        r1_rewards = [_raw_reward(s) for s in round1]
        r2_rewards = [_raw_reward(s) for s in round2]
        if r1_rewards:
            round1_rewards_per_prompt.append(r1_rewards)
        if r2_rewards:
            round2_rewards_per_prompt.append(r2_rewards)

        by_chain_r1 = {_chain_id(s): s for s in round1 if _chain_id(s) is not None}
        by_chain_r2 = {_chain_id(s): s for s in round2 if _chain_id(s) is not None}
        for chain_id, s1 in by_chain_r1.items():
            s2 = by_chain_r2.get(chain_id)
            if s2 is not None:
                revise_lift.append(_raw_reward(s2) - _raw_reward(s1))

        for s in verifier:
            verdict = (s.metadata or {}).get("verdict", "unparseable")
            if verdict == "unparseable":
                verifier_parse_failure.append(1.0)
            else:
                verifier_parse_failure.append(0.0)
            if verdict not in {"approve", "reject"}:
                continue
            correct = float((s.metadata or {}).get("round1_correct", 0.0)) == 1.0
            expected = "approve" if correct else "reject"
            hit = 1.0 if verdict == expected else 0.0
            verifier_accuracy.append(hit)
            if correct:
                verifier_accuracy_correct.append(hit)
            else:
                verifier_accuracy_incorrect.append(hit)

        round1_truncated.extend([float(s.status == Sample.Status.TRUNCATED) for s in round1])
        round2_truncated.extend([float(s.status == Sample.Status.TRUNCATED) for s in round2])

    base = dataset_cfg.name
    out: dict[str, dict[str, list[Any]]] = {}
    for k in _PASSK_KS:
        out[f"{base}_round1_pass{k}"] = _ds(
            _pass_at_k_per_prompt(round1_rewards_per_prompt, k),
            round1_samples_flat,
        )
        out[f"{base}_round2_pass{k}"] = _ds(
            _pass_at_k_per_prompt(round2_rewards_per_prompt, k),
            round2_samples_flat,
        )

    out[f"{base}_revise_lift"] = _ds(_nonempty(revise_lift), round2_samples_flat)
    out[f"{base}_verifier_accuracy"] = _ds(_nonempty(verifier_accuracy), verifier_samples_flat)
    out[f"{base}_verifier_accuracy_on_correct"] = _ds(_nonempty(verifier_accuracy_correct), verifier_samples_flat)
    out[f"{base}_verifier_accuracy_on_incorrect"] = _ds(_nonempty(verifier_accuracy_incorrect), verifier_samples_flat)
    out[f"{base}_verifier_parse_failure_rate"] = _ds(_nonempty(verifier_parse_failure), verifier_samples_flat)
    out[f"{base}_round1_truncated_ratio"] = _ds(_nonempty(round1_truncated), round1_samples_flat)
    out[f"{base}_round2_truncated_ratio"] = _ds(_nonempty(round2_truncated), round2_samples_flat)
    return out


def _raw_reward(sample: Sample) -> float:
    return float((sample.metadata or {}).get("raw_reward", sample.reward or 0.0))


def _chain_id(sample: Sample):
    return (sample.metadata or {}).get("chain_id")


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
