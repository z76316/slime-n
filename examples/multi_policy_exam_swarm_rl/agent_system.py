"""Exam Swarm RL — 8 homogeneous agents take the same exam independently.

Per-trajectory advantage is composed at rollout time as
  final = α·self_adv + β·swarm_adv + γ·peer_adv  (clipped to ±5)
and stored as Sample.reward (single float). Slime broadcasts to per-token
via the GRPO advantage estimator path. Run script must pass
--disable-rewards-normalization so slime does not re-normalize.
"""

import asyncio
import itertools
import logging
import math
import traceback
from copy import deepcopy

from slime.rollout.rm_hub import batched_async_rm
from slime.rollout.sglang_rollout import get_model_url
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .prompts import EXAM_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


# Edit for ablations; no YAML round-trip.
ALPHA = 0.5
BETA = 0.3
GAMMA = 0.2
BASELINE_MODE = "grpo"   # "grpo" | "adversarial"
ADV_CLIP = 5.0
N_AGENTS = 8


class SwarmBaseline:
    """Running EMA of swarm pass rate g across questions, with warmup
    (returns 0.0 for the first WARMUP calls so cold-start does not inject
    a systematic bias) and z-score clipping (bounds runaway when σ_g is
    very small)."""

    WARMUP = 20

    def __init__(self, momentum: float = 0.95, clip: float = 3.0):
        self.mean = 0.0
        self.var = 1.0
        self.m = momentum
        self.clip = clip
        self.calls = 0

    def norm(self, g: float) -> float:
        if self.calls < self.WARMUP:
            return 0.0
        z = (g - self.mean) / (math.sqrt(self.var) + 1e-6)
        return max(-self.clip, min(self.clip, z))

    def update(self, g: float) -> None:
        self.calls += 1
        self.mean = self.m * self.mean + (1 - self.m) * g
        self.var = self.m * self.var + (1 - self.m) * (g - self.mean) ** 2


# Module-level singleton — persists across run_agent_system calls within
# the rollout-manager actor process.
_SWARM_BASELINE = SwarmBaseline()


def rank_advantage(per_agent_scores: list[list[float]]) -> list[float]:
    """Per-agent rank, normalized to mean 0 across agents and range [-1, +1].
    Average-rank tie-breaking keeps the per-question sum exactly 0."""
    s = [sum(ks) / len(ks) if ks else 0.0 for ks in per_agent_scores]
    n = len(s)
    if n <= 1:
        return [0.0] * n
    order = sorted(range(n), key=lambda i: -s[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return [(n + 1 - 2 * r) / (n - 1) for r in ranks]


def self_advantage_grpo(per_agent_scores: list[list[float]]) -> list[list[float]]:
    """Standard GRPO group-norm: (c - mean_K) / (std_K + ε), within each agent."""
    out = []
    for ks in per_agent_scores:
        if not ks:
            out.append([])
            continue
        m = sum(ks) / len(ks)
        var = sum((c - m) ** 2 for c in ks) / len(ks)
        std = math.sqrt(var) + 1e-6
        out.append([(c - m) / std for c in ks])
    return out


def self_advantage_adversarial(per_agent_scores: list[list[float]]) -> list[list[float]]:
    """Adversarial baseline: (c - max_peer_mean_i) / (std_i + ε). Per-agent
    peer baseline doesn't get cancelled by per-question normalization, so
    the resulting advantage is non-zero-mean within agent i's K — positive
    iff agent i beats the swarm leader on this question."""
    n = len(per_agent_scores)
    means = [(sum(ks) / len(ks) if ks else 0.0) for ks in per_agent_scores]
    out = []
    for i, ks in enumerate(per_agent_scores):
        if not ks:
            out.append([])
            continue
        peer_max = max((means[j] for j in range(n) if j != i), default=0.0)
        var = sum((c - means[i]) ** 2 for c in ks) / len(ks)
        std = math.sqrt(var) + 1e-6
        out.append([(c - peer_max) / std for c in ks])
    return out


# Inner-sample id: K-deepcopies of an outer Sample share its index, which
# trips slime's get_data_iterator uniqueness assertion. High-offset counter
# keeps inner indices clear of slime's data_source counter.
_INNER_SAMPLE_ID = itertools.count(start=1_000_000_000)


async def generate_response(args, prompt: str, agent_name: str) -> Sample | None:
    """Dispatch one inference call to agent_name's sglang engine. Mirrors
    solver_summarizer.generate_response."""
    try:
        sampling_params = args.sampling_params
        tokenizer = args.tokenizer
        max_context_length = args.rollout_max_context_len

        sample = deepcopy(args.sample)
        url = get_model_url(args, agent_name)

        sample.prompt = prompt
        prompt_token_ids = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        sample.tokens = prompt_token_ids
        prompt_length = len(prompt_token_ids)
        current_sampling_params = deepcopy(sampling_params)
        current_sampling_params["max_new_tokens"] = min(
            sampling_params["max_new_tokens"], max_context_length - prompt_length
        )

        if current_sampling_params["max_new_tokens"] <= 0:
            return None

        payload = {
            "input_ids": prompt_token_ids,
            "sampling_params": current_sampling_params,
            "return_logprob": True,
        }
        output = await post(url, payload)

        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_response_tokens = []
            new_response_log_probs = []

        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response = output["text"]
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "stop":
                sample.status = Sample.Status.COMPLETED

        sample.policy_name = agent_name
        sample.index = next(_INNER_SAMPLE_ID)
        if sample.metadata is None:
            sample.metadata = {}
        args.results_dict[agent_name].append(sample)
        return sample
    except Exception as e:
        logger.warning(f"[{agent_name}] generate_response failed: {e}")
        return None


async def agent_worker(args, prompt: str, agent_name: str, worker_id: int) -> Sample | None:
    try:
        return await generate_response(args, prompt, agent_name)
    except Exception:
        logger.warning(f"[{agent_name}/{worker_id}] worker exception:\n{traceback.format_exc()}")
        return None


def _pad_agent_buffer(args, agent_name: str, target_count: int, donor_pool=None):
    """Split-buffer routing requires each policy's per-rollout buffer to
    equal global_batch_size. Pad with placeholders when an agent's
    inference fails."""
    samples = args.results_dict[agent_name]
    if len(samples) >= target_count:
        del samples[target_count:]
        return
    pool = samples if samples else (donor_pool or [])
    if not pool:
        return
    while len(samples) < target_count:
        placeholder = deepcopy(pool[0])
        placeholder.policy_name = agent_name
        placeholder.index = next(_INNER_SAMPLE_ID)
        placeholder.reward = 0.0
        if placeholder.metadata is None:
            placeholder.metadata = {}
        placeholder.metadata["raw_reward"] = 0.0
        placeholder.metadata["is_pad"] = True
        samples.append(placeholder)


async def run_agent_system(args, sample: Sample) -> list[Sample]:
    """Per outer prompt: dispatch to all N agents in parallel (K each),
    score by RLVR, compose 3-channel per-trajectory advantage, return
    flat list tagged for split-buffer routing.

    args.num_parallel = K (samples per agent for GRPO group-norm).
    Returns N_AGENTS × K samples.
    """
    args = deepcopy(args)
    args.sample = sample
    args.results_dict = {f"agent_{i}": [] for i in range(N_AGENTS)}

    k = args.num_parallel
    prompt = (
        EXAM_PROMPT_TEMPLATE.format(problem_statement=sample.prompt)
        if "{problem_statement}" in EXAM_PROMPT_TEMPLATE
        else sample.prompt
    )

    # Phase 1 — fan out N×K parallel sglang requests.
    tasks = [
        agent_worker(args, prompt, f"agent_{i}", w)
        for i in range(N_AGENTS)
        for w in range(k)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Phase 2 — RLVR score every collected response.
    all_samples: list[Sample] = []
    for i in range(N_AGENTS):
        all_samples.extend(args.results_dict[f"agent_{i}"])
    if all_samples:
        try:
            raw_rewards = await batched_async_rm(args, all_samples)
            for s, r in zip(all_samples, raw_rewards, strict=False):
                if s.metadata is None:
                    s.metadata = {}
                s.metadata["raw_reward"] = float(r) if r is not None else 0.0
        except Exception as e:
            logger.warning(f"batched_async_rm failed; defaulting all rewards to 0: {e}")
            for s in all_samples:
                if s.metadata is None:
                    s.metadata = {}
                s.metadata["raw_reward"] = 0.0

    # Phase 3 — pad each agent to exactly K (split-buffer invariant).
    donor_pool = next(
        (args.results_dict[f"agent_{i}"] for i in range(N_AGENTS) if args.results_dict[f"agent_{i}"]),
        [],
    )
    for i in range(N_AGENTS):
        _pad_agent_buffer(args, f"agent_{i}", k, donor_pool=donor_pool)

    # Phase 4 — compose advantages.
    per_agent_scores = [
        [float(s.metadata.get("raw_reward", 0.0)) for s in args.results_dict[f"agent_{i}"]]
        for i in range(N_AGENTS)
    ]
    flat_c = [c for ks in per_agent_scores for c in ks]
    g = sum(flat_c) / len(flat_c) if flat_c else 0.0
    swarm_adv = _SWARM_BASELINE.norm(g)
    _SWARM_BASELINE.update(g)
    peer_adv = rank_advantage(per_agent_scores)

    if BASELINE_MODE == "grpo":
        self_adv = self_advantage_grpo(per_agent_scores)
    elif BASELINE_MODE == "adversarial":
        self_adv = self_advantage_adversarial(per_agent_scores)
    else:
        raise ValueError(f"unknown BASELINE_MODE {BASELINE_MODE}")

    means = [(sum(ks) / len(ks) if ks else 0.0) for ks in per_agent_scores]

    # Phase 5 — write final advantage to Sample.reward, diagnostics to metadata.
    for i in range(N_AGENTS):
        agent_samples = args.results_dict[f"agent_{i}"]
        peer_max_i = max((means[j] for j in range(N_AGENTS) if j != i), default=0.0)
        for k_idx, s in enumerate(agent_samples):
            sa = self_adv[i][k_idx] if k_idx < len(self_adv[i]) else 0.0
            pa = peer_adv[i]
            final = ALPHA * sa + BETA * swarm_adv + GAMMA * pa
            final = max(-ADV_CLIP, min(ADV_CLIP, final))
            s.reward = float(final)
            s.metadata.update({
                "self_adv": float(sa),
                "swarm_adv": float(swarm_adv),
                "peer_adv": float(pa),
                "peer_max": float(peer_max_i),
                "g": float(g),
                "agent_idx": i,
                "baseline_mode": BASELINE_MODE,
            })

    return [s for sub in args.results_dict.values() for s in sub]
