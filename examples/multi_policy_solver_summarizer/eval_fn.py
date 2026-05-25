"""Eval for the solver+summarizer chain.

Runs the chain once per AIME prompt and emits per-prompt pass@k
aggregates (k = 1, 2, 4) for both roles, computed from the raw
(unscaled) RM rewards via the unbiased pass@k estimator. Pass@k is
computed inside this function rather than via --log-passrate because
that global flag also triggers train-side pass-rate logging, whose
assertion (`len(rewards) == rollout_batch_size * n_samples_per_prompt`)
does not hold in this multi-agent setup: slime calls the chain
n_samples_per_prompt times per outer prompt and each call returns
num_parallel samples per role, so the per-role buffer carries
n_samples_per_prompt * num_parallel samples per prompt.

Emitted metrics (per dataset NAME in eval_config.yaml):

  eval/<NAME>_summarizer_pass1/score   summarizer pass@1
  eval/<NAME>_summarizer_pass2/score   summarizer pass@2
  eval/<NAME>_summarizer_pass4/score   summarizer pass@4 (best-of-4)
  eval/<NAME>_solver_pass1/score       solver pass@1
  eval/<NAME>_solver_pass2/score       solver pass@2
  eval/<NAME>_solver_pass4/score       solver pass@4 (best-of-4, skyline)

Headline: `pass4` for both roles. Their difference diagnoses whether
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
    once per prompt and emits per-prompt pass@k aggregates (k=1,2,4) for
    both solver and summarizer using the raw RM rewards.

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
        # Return real Sample objects (not just rewards) so we can attach
        # them to each emitted dataset; _save_debug_rollout_data and
        # compute_metrics_from_samples both require info["samples"] to
        # exist and be a non-empty list of Samples.
        solver = [s for s in chain if s.policy_name == "solver"]
        summarizer = [s for s in chain if s.policy_name == "summarizer"]
        return solver, summarizer

    async def gather_all():
        tasks = [asyncio.create_task(run_one(ps)) for ps in dataset.samples]
        out = []
        pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}")
        for coro in asyncio.as_completed(tasks):
            out.append(await coro)
            pbar.update(1)
        pbar.close()
        return out

    solver_rewards_per_prompt: list[list[float]] = []
    summarizer_rewards_per_prompt: list[list[float]] = []
    solver_samples_flat: list[Any] = []
    summarizer_samples_flat: list[Any] = []
    for solver, summarizer in run(gather_all()):
        if solver:
            solver_rewards_per_prompt.append([s.metadata.get("raw_reward", s.reward) for s in solver])
            solver_samples_flat.extend(solver)
        if summarizer:
            summarizer_rewards_per_prompt.append([s.metadata.get("raw_reward", s.reward) for s in summarizer])
            summarizer_samples_flat.extend(summarizer)

    base = dataset_cfg.name
    out: dict[str, dict[str, list[Any]]] = {}
    for k in _PASSK_KS:
        out[f"{base}_summarizer_pass{k}"] = _ds(
            _pass_at_k_per_prompt(summarizer_rewards_per_prompt, k),
            summarizer_samples_flat,
        )
        out[f"{base}_solver_pass{k}"] = _ds(
            _pass_at_k_per_prompt(solver_rewards_per_prompt, k),
            solver_samples_flat,
        )
    return out


_PASSK_KS = (1, 2, 4)


def _pass_at_k_per_prompt(rewards_per_prompt: list[list[float]], k: int) -> list[float]:
    """Unbiased pass@k estimator (Chen et al. 2021) computed per prompt.

    For each prompt's n raw 0/1 rewards with c correct,
      pass@k = 1 - C(n - c, k) / C(n, k)  if (n - c) >= k else 1.
    """
    out = []
    for rewards in rewards_per_prompt:
        n = len(rewards)
        c = sum(1 for r in rewards if r == 1)
        if k > n:
            # Not enough attempts to evaluate pass@k; skip the prompt.
            continue
        if n - c < k:
            out.append(1.0)
        else:
            # 1 - prod_{i=n-c+1..n} (1 - k/i)
            p = 1.0
            for i in range(n - c + 1, n + 1):
                p *= 1.0 - k / i
            out.append(1.0 - p)
    return out


def _ds(rewards: list[float], samples: list) -> dict[str, list[Any]]:
    from slime.utils.types import Sample

    # Same `samples` list (real Sample objects) attached to every pass@k
    # variant per role. _save_debug_rollout_data requires the "samples"
    # key to exist on every emitted dataset, and compute_metrics_from_samples
    # crashes on an empty list — so attach the full per-role sample list
    # rather than slicing/omitting it. The 3x duplication in the debug
    # dump is acceptable given the small per-eval sample count.
    return {
        "rewards": rewards,
        "truncated": [s.status == Sample.Status.TRUNCATED for s in samples],
        "samples": samples,
    }
