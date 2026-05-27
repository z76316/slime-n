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

from .rollout_with_multi_agents import generate_with_multi_agents

_EVAL_DATASET_CACHE: dict[Any, Dataset] = {}
_PASSK_KS = (1, 2, 4)
_JUDGMENT_RE = re.compile(r"Judgment:\s*(?:IDX|Solution)?\s*#?(\d+)", re.IGNORECASE)


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


def eval_with_multi_agents(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = True
) -> RolloutFnEvalOutput:
    assert evaluation, "eval_with_multi_agents is the eval-only entry point"
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    eval_datasets = getattr(args, "eval_datasets", None) or []
    assert eval_datasets, "eval_with_multi_agents requires --eval-config with at least one dataset"

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
        chain = await generate_with_multi_agents(args, sample, sampling_params, evaluation=True)
        solver = [s for s in chain if s.policy_name == "solver" and not _is_placeholder(s)]
        rewriter = [s for s in chain if s.policy_name == "rewriter" and not _is_placeholder(s)]
        selector = [s for s in chain if s.policy_name == "selector" and not _is_placeholder(s)]
        return solver, rewriter, selector

    async def gather_all():
        tasks = [asyncio.create_task(run_one(ps)) for ps in dataset.samples]
        out = []
        pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}")
        for coro in asyncio.as_completed(tasks):
            out.append(await coro)
            pbar.update(1)
        pbar.close()
        return out

    solver_rewards_pp: list[list[float]] = []
    rewriter_rewards_pp: list[list[float]] = []
    selector_rewards_pp: list[list[float]] = []
    solver_samples_flat: list[Sample] = []
    rewriter_samples_flat: list[Sample] = []
    selector_samples_flat: list[Sample] = []
    rewrite_lift: list[float] = []
    selector_parse_failure: list[float] = []
    selector_accuracy: list[float] = []
    solver_truncated: list[float] = []
    rewriter_truncated: list[float] = []
    selector_truncated: list[float] = []

    for solver, rewriter, selector in run(gather_all()):
        solver_samples_flat.extend(solver)
        rewriter_samples_flat.extend(rewriter)
        selector_samples_flat.extend(selector)

        s_rewards = [_raw_reward(s) for s in solver]
        if s_rewards:
            solver_rewards_pp.append(s_rewards)

        r_rewards = [_raw_reward(s) for s in rewriter]
        if r_rewards:
            rewriter_rewards_pp.append(r_rewards)

        sel_rewards = [_raw_reward(s) for s in selector]
        if sel_rewards:
            selector_rewards_pp.append(sel_rewards)

        if s_rewards and r_rewards:
            rewrite_lift.append(sum(r_rewards) / len(r_rewards) - sum(s_rewards) / len(s_rewards))

        for s in selector:
            response = (s.response_content or (s.response or "")).strip()
            matched = _JUDGMENT_RE.findall(response)
            if not matched:
                selector_parse_failure.append(1.0)
            else:
                selector_parse_failure.append(0.0)
                best_rewriter_reward = max(r_rewards) if r_rewards else 0.0
                selected_reward = _raw_reward(s)
                selector_accuracy.append(1.0 if selected_reward >= best_rewriter_reward and best_rewriter_reward > 0 else 0.0)

        solver_truncated.extend([float(s.status == Sample.Status.TRUNCATED) for s in solver])
        rewriter_truncated.extend([float(s.status == Sample.Status.TRUNCATED) for s in rewriter])
        selector_truncated.extend([float(s.status == Sample.Status.TRUNCATED) for s in selector])

    base = dataset_cfg.name
    out: dict[str, dict[str, list[Any]]] = {}

    for k in _PASSK_KS:
        out[f"{base}_solver_pass{k}"] = _ds(_pass_at_k_per_prompt(solver_rewards_pp, k), solver_samples_flat)
        out[f"{base}_rewriter_pass{k}"] = _ds(_pass_at_k_per_prompt(rewriter_rewards_pp, k), rewriter_samples_flat)
        out[f"{base}_selector_pass{k}"] = _ds(_pass_at_k_per_prompt(selector_rewards_pp, k), selector_samples_flat)

    out[f"{base}_rewrite_lift"] = _ds(_nonempty(rewrite_lift), rewriter_samples_flat)
    out[f"{base}_selector_parse_failure_rate"] = _ds(_nonempty(selector_parse_failure), selector_samples_flat)
    out[f"{base}_selector_accuracy"] = _ds(_nonempty(selector_accuracy), selector_samples_flat)
    out[f"{base}_solver_truncated_ratio"] = _ds(_nonempty(solver_truncated), solver_samples_flat)
    out[f"{base}_rewriter_truncated_ratio"] = _ds(_nonempty(rewriter_truncated), rewriter_samples_flat)
    out[f"{base}_selector_truncated_ratio"] = _ds(_nonempty(selector_truncated), selector_samples_flat)
    return out


def _raw_reward(sample: Sample) -> float:
    return float((sample.metadata or {}).get("raw_reward", sample.reward or 0.0))


def _is_placeholder(sample: Sample) -> bool:
    return bool((sample.metadata or {}).get("is_padding_placeholder"))


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
