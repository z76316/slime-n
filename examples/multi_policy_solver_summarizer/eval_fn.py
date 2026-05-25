"""Eval for the solver+summarizer chain.

Runs the chain once per AIME prompt and emits two logged datasets,
each carrying all 4 attempt-level (raw RM) rewards per prompt in
prompt order so the default eval logger can compute pass@k when
--log-passrate is set:

  eval/aime_summarizer/score        per-attempt accuracy (= pass@1)
  eval/aime_summarizer-pass@{1,2,4} best-of-k summarizer (any-correct)
  eval/aime_solver/score            per-attempt accuracy (= pass@1)
  eval/aime_solver-pass@{1,2,4}     best-of-k solver (any-correct)

The headline final-answer-quality metric is `aime_summarizer-pass@4`
(= 1 if any of the 4 summarizer attempts is correct). The skyline
ceiling is `aime_solver-pass@4`. Their difference diagnoses whether
the summarizer is synthesizing nontrivially or just aggregating (or
destroying) signal the solver produced.
"""

import asyncio
import copy
from argparse import Namespace
from typing import Any

from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.processing_utils import load_processor, load_tokenizer

from .rollout_with_multi_agents import generate_with_multi_agents

_EVAL_DATASET_CACHE: dict[Any, Dataset] = {}


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
    """Custom eval function: --eval-function-path points here.

    For each eval dataset listed in args.eval_datasets, runs the chain
    once per prompt and emits one logged dataset per role with all
    per-attempt rewards in prompt order (so --log-passrate can compute
    pass@k with group_size=args.n_samples_per_eval_prompt).

    Mirrors the (sync outer, async inner via run()) shape used by
    `generate_rollout` in slime.rollout.sglang_rollout.
    """
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
        # Stable per-role order so the flat per-prompt block lines up with
        # group_size in compute_pass_rate.
        solver = [s for s in chain if s.policy_name == "solver"]
        summarizer = [s for s in chain if s.policy_name == "summarizer"]
        return solver, summarizer

    async def gather_in_prompt_order():
        tasks = [asyncio.create_task(run_one(ps)) for ps in dataset.samples]
        pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}")
        for task in tasks:
            await task
            pbar.update(1)
        pbar.close()
        return [task.result() for task in tasks]

    solver_samples: list = []
    summarizer_samples: list = []
    for solver, summarizer in run(gather_in_prompt_order()):
        solver_samples.extend(solver)
        summarizer_samples.extend(summarizer)

    base = dataset_cfg.name
    return {
        f"{base}_summarizer": _flat_dataset(summarizer_samples),
        f"{base}_solver": _flat_dataset(solver_samples),
    }


def _flat_dataset(samples) -> dict[str, list[Any]]:
    from slime.utils.types import Sample

    # Use the unscaled RM verdict that agent_system.run_agent_system stashes
    # in metadata before applying the 0.8/1.2 training reward weights;
    # s.reward at this point is the scaled value, useless for pass-rate.
    raw_rewards = [s.metadata.get("raw_reward", s.reward) for s in samples]
    out: dict[str, list[Any]] = {
        "rewards": raw_rewards,
        "truncated": [s.status == Sample.Status.TRUNCATED for s in samples],
    }
    # Only include samples if non-empty; the eval logger calls
    # compute_metrics_from_samples on `samples is not None`, which would
    # crash on np.max([]) when no samples were collected.
    if samples:
        out["samples"] = samples
    return out
