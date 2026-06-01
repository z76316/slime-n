"""Multi-agent debate, aligned with Subramaniam et al. 2025
"Multiagent Finetuning of Language Models" (arxiv 2501.05707), Algorithm 1.

Two trainable policies:
  - generator (A^G): round-0 independent responses.
  - critic    (A^C): round-m≥1 updates, given a summary of the OTHER N-1
                     agents' prior responses plus the agent's OWN prior
                     response (so it iterates on its own reasoning).

Non-trained subroutine:
  - summarize (A^S): summarizes others' responses. Routed through the
                     generator engine; its Sample is NOT added to
                     results_dict (subroutine, not a trained policy).

REWARD (Algorithm 1 lines 23-26): ŷ = majority vote over FINAL critic
responses (per slot).
  - Generator: 1 if round-0 boxed answer == ŷ, else 0 (per-sample).
  - Critic: 1 if agent's FINAL critic response == ŷ, else 0 (trajectory-
    level; same reward to all of the agent's critic rounds).
Grading mirrors the paper's parse_answer(...) == parse_answer(answer):
string equality on extracted boxed answers (stripped). Majority vote is
used in place of ground truth to stay faithful to the no-ground-truth
self-improvement premise; the dataset gold label is intentionally ignored.

AGENT IDENTITY: agents are tracked by `wid` (worker id) on sample.metadata,
because asyncio.gather completion order is non-deterministic and would
otherwise scramble per-agent pairing across rounds. Samples are sorted by
wid after each round so trajectory_i is the same logical agent every round.
"""

import asyncio
import itertools
import logging
import random
import re
import time
import traceback
from collections import Counter
from copy import deepcopy

from slime.rollout.rm_hub import extract_boxed_answer  # slime's canonical grader
from slime.rollout.sglang_rollout import get_model_url
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .prompts import GENERATOR_INITIAL_TEMPLATE, GENERATOR_UPDATE_TEMPLATE, generate_summarize_template

logger = logging.getLogger(__name__)

_INNER_SAMPLE_ID = itertools.count(start=1_000_000_000)
_CHAT_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>(?:user|assistant|system)?\s*")


def _strip_chat_tokens(text: str) -> str:
    return _CHAT_TOKEN_RE.sub("", text).strip()


def _wrap_user_turn(tokenizer, user_content: str) -> str:
    if getattr(tokenizer, "chat_template", None) is None:
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


async def generate_response(args, prompt, key, track: bool = True, wid: int | None = None):
    """Call the sglang engine for `key` ∈ {"generator", "critic"}.

    track=False: sample is discarded (summarize subroutine; A^S not trained).
    wid: worker id on sample.metadata, for per-agent pairing across rounds
    (asyncio.gather order is non-deterministic).
    """
    try:
        sampling_params = args.sampling_params
        tokenizer = args.tokenizer
        max_context_length = args.rollout_max_context_len
        sample = deepcopy(args.sample)

        url = get_model_url(args, key)

        sample.prompt = prompt
        prompt_token_ids = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        sample.tokens = prompt_token_ids
        prompt_length = len(prompt_token_ids)
        current_sampling_params = deepcopy(sampling_params)
        current_sampling_params["max_new_tokens"] = min(
            sampling_params["max_new_tokens"], max_context_length - prompt_length
        )
        if current_sampling_params["max_new_tokens"] <= 0:
            logger.warning(
                f"prompt exceeds context budget — role={key}, wid={wid}, "
                f"prompt_length={prompt_length}, max_context_length={max_context_length}; "
                f"skipping generation (no sample produced)."
            )
            return None

        payload = {"input_ids": prompt_token_ids, "sampling_params": current_sampling_params, "return_logprob": True}
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

        final = output["text"].replace("<|user|>", "")
        if "</think>" in final:
            contents = final.split("</think>")
            if len(contents) == 2 and contents[1] != "":
                sample.reason_content = contents[0].strip()
                sample.response_content = contents[1].strip()
            else:
                sample.reason_content = None
                sample.response_content = output["text"]
        else:
            sample.reason_content = None
            sample.response_content = output["text"]

        if track:
            sample.policy_name = key
            sample.index = next(_INNER_SAMPLE_ID)
            # Stamp wid for per-agent pairing.
            if wid is not None:
                if sample.metadata is None:
                    sample.metadata = {}
                sample.metadata["wid"] = wid
            args.results_dict[key].append(sample)

        return sample.response_content
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


# --- agent classes ---


class _Agent:
    async def _run(self, args, prompt, key, max_retries: int = 3, track: bool = True, wid: int | None = None) -> str:
        for _ in range(max_retries):
            try:
                response = await generate_response(args, prompt, key=key, track=track, wid=wid)
                return response
            except Exception as e:
                print(f"Error querying LLM: {e}")
                time.sleep(1)
        return None


class GeneratorAgent(_Agent):
    async def propose(self, args, problem_statement, wid: int) -> str:
        prompt = GENERATOR_INITIAL_TEMPLATE.format(problem_statement=problem_statement)
        return await self._run(args, prompt, key="generator", max_retries=3, track=True, wid=wid)


class CriticAgent(_Agent):
    async def update(self, args, problem_statement, prior_response: str, summary: str, wid: int) -> str:
        body = GENERATOR_UPDATE_TEMPLATE.format(
            problem_statement=problem_statement,
            prior_response=_strip_chat_tokens(prior_response or ""),
            summary=summary,
        )
        prompt = _wrap_user_turn(args.tokenizer, body)
        return await self._run(args, prompt, key="critic", max_retries=3, track=True, wid=wid)


async def summarize_subroutine(args, other_responses: list[str], wid: int = 0) -> str:
    """Untracked summarize via generator engine; sample discarded (A^S)."""
    cleaned = [_strip_chat_tokens(r) for r in other_responses if r and r.strip()]
    if not cleaned:
        return ""
    template = generate_summarize_template(len(cleaned))
    format_params = {f"solution{i+1}": s for i, s in enumerate(cleaned)}
    body = template.format(**format_params)
    prompt = _wrap_user_turn(args.tokenizer, body)
    try:
        response = await generate_response(args, prompt, key="generator", track=False)
        return response or ""
    except Exception as e:
        print(f"[summarize-{wid}] {e}\n{traceback.format_exc()}")
        return ""


# --- workers ---


async def round0_worker(args, problem_statement, wid):
    try:
        return await GeneratorAgent().propose(args, problem_statement, wid=wid)
    except Exception as e:
        print(f"[round0-{wid}] {e}\n{traceback.format_exc()}")
        return None


async def critic_worker(args, problem_statement, prior_response, summary, wid):
    try:
        return await CriticAgent().update(args, problem_statement, prior_response, summary, wid=wid)
    except Exception as e:
        print(f"[critic-{wid}] {e}\n{traceback.format_exc()}")
        return None


# --- buffer-count invariant ---


def _pad_role_buffer(args, role: str, target_count: int, donor_role: str | None = None):
    """Pad to exactly `target_count` so per-role buffer counts stay constant
    for GRPO reshape. Last-resort fallback uses `args.sample`."""
    samples = args.results_dict[role]
    if len(samples) >= target_count:
        del samples[target_count:]
        return
    donor_pool = samples if samples else (args.results_dict.get(donor_role) or [])
    used_args_sample = False
    if not donor_pool:
        donor_pool = [args.sample]
        used_args_sample = True
    while len(samples) < target_count:
        placeholder = deepcopy(donor_pool[0])
        placeholder.policy_name = role
        placeholder.index = next(_INNER_SAMPLE_ID)
        placeholder.reward = 0.0
        placeholder.response = ""
        placeholder.response_length = 0
        placeholder.loss_mask = []
        placeholder.remove_sample = True
        placeholder.response_content = None
        placeholder.reason_content = None
        placeholder.tokens = list(getattr(args.sample, "tokens", []) or [])
        placeholder.rollout_log_probs = []
        placeholder.rollout_routed_experts = None
        placeholder.teacher_log_probs = None
        placeholder.metadata = {
            **(placeholder.metadata or {}),
            "raw_reward": 0.0,
            "is_padding_placeholder": True,
            "padding_donor_policy": getattr(donor_pool[0], "policy_name", None),
        }
        # Placeholders are not logical agents: drop any wid.
        if placeholder.metadata is not None and "wid" in placeholder.metadata:
            placeholder.metadata = {k: v for k, v in placeholder.metadata.items() if k != "wid"}
        samples.append(placeholder)
    if used_args_sample:
        outer_idx = getattr(args.sample, "index", "?")
        logger.warning(
            f"_pad_role_buffer: no-donor fallback fired for role={role} " f"(outer prompt index={outer_idx})."
        )


# --- per-agent ordering helper ---


def _by_wid(samples: list[Sample], n: int) -> list[Sample | None]:
    """Length-n list with position i = sample whose metadata["wid"] == i
    (None if missing). Recovers per-agent pairing after non-deterministic
    asyncio.gather ordering."""
    by_wid: dict[int, Sample] = {}
    for s in samples:
        if s.metadata and "wid" in s.metadata:
            by_wid[s.metadata["wid"]] = s
    return [by_wid.get(i) for i in range(n)]


# --- paper-aligned grading: extracted-boxed string equality ---


def _normalize_answer(ans: str | None) -> str | None:
    if ans is None:
        return None
    return ans.strip()


def _matches(sample: Sample | None, y_hat: str) -> bool:
    """True iff the LAST `\\boxed{...}` in sample.response equals y_hat
    (whitespace-normalized). Uses `.response` (full output), not
    `.response_content`: the latter strips the pre-`</think>` portion in
    Qwen3 thinking mode, but the paper extracts the LAST boxed answer
    anywhere in the text."""
    if sample is None or not sample.response:
        return False
    extracted = extract_boxed_answer(sample.response)
    return _normalize_answer(extracted) == _normalize_answer(y_hat)


def _majority_vote(answers: list[str | None]) -> str | None:
    """ŷ = majority vote over boxed answers (None/empty skipped). Ties broken
    randomly, per the paper."""
    valid = [a for a in answers if a is not None and a.strip() != ""]
    if not valid:
        return None
    counter = Counter(valid)
    max_count = max(counter.values())
    top = [a for a, c in counter.items() if c == max_count]
    return random.choice(top)


# --- orchestration ---


async def run_agent_system(args, sample):
    """One debate slot. Reward computed after all rounds: majority-vote ŷ
    over final critic responses, then grade each sample against ŷ
    (Algorithm 1)."""
    args = deepcopy(args)
    args.sample = sample
    args.results_dict = {"generator": [], "critic": []}

    raw_problem = _strip_chat_tokens(sample.prompt)
    n = args.num_parallel  # 3
    rounds = args.rounds  # 3 (m=0 generator, m=1,2 critic)

    target_gen = n  # 3
    target_critic = n * (rounds - 1)  # 6

    # ---- Round 0: N parallel generators (grading deferred) ----
    gen_count_before = len(args.results_dict["generator"])
    r0_tasks = [round0_worker(args, raw_problem, wid) for wid in range(n)]
    await asyncio.gather(*r0_tasks, return_exceptions=True)
    gen_samples_unsorted = args.results_dict["generator"][gen_count_before:]
    gen_samples_by_wid = _by_wid(gen_samples_unsorted, n)

    # prev_responses[i] = agent-i's prior-round response (or "")
    prev_responses = [(s.response_content if s and s.response_content is not None else "") for s in gen_samples_by_wid]

    # Abort if round 0 lacks N usable responses.
    if sum(1 for r in prev_responses if r) < n:
        for s in args.results_dict["generator"]:
            s.reward = 0.0
        _pad_role_buffer(args, "generator", target_gen)
        _pad_role_buffer(args, "critic", target_critic, donor_role="generator")
        return args.results_dict["generator"] + args.results_dict["critic"]

    # critic_trajectory[i] = agent-i's critic samples for rounds m=1..M-1
    critic_trajectory: list[list[Sample]] = [[] for _ in range(n)]

    # ---- Rounds 1..M-1: summarize + critic ----
    for _round_idx in range(1, rounds):
        # Summarize: agent-i sees OTHERS' prior responses.
        summary_tasks = []
        for i in range(n):
            others = [prev_responses[j] for j in range(n) if j != i and prev_responses[j]]
            summary_tasks.append(summarize_subroutine(args, others, wid=i))
        summaries = await asyncio.gather(*summary_tasks, return_exceptions=True)
        summaries = [s if isinstance(s, str) else "" for s in summaries]

        # Critic: paired summary + own prior response.
        critic_count_before = len(args.results_dict["critic"])
        critic_tasks = [critic_worker(args, raw_problem, prev_responses[i], summaries[i], wid=i) for i in range(n)]
        await asyncio.gather(*critic_tasks, return_exceptions=True)
        round_critic_unsorted = args.results_dict["critic"][critic_count_before:]
        round_critic_by_wid = _by_wid(round_critic_unsorted, n)

        # Append to per-agent trajectory; stop if too many slots failed.
        any_landed = False
        next_prev = [""] * n
        for i in range(n):
            s = round_critic_by_wid[i]
            if s is None:
                continue
            critic_trajectory[i].append(s)
            if s.response_content:
                next_prev[i] = s.response_content
                any_landed = True

        if not any_landed:
            break

        prev_responses = next_prev
        if sum(1 for r in prev_responses if r) < n:
            # Not enough material for next round; finalize.
            break

    # ---- ŷ from FINAL critic responses (per slot) ----
    # Extract from .response (full text), not .response_content: the paper's
    # last-boxed scan is more robust than slicing post-</think>, esp. for
    # truncated outputs that never close the think block.
    final_critic_responses: list[str | None] = []
    for traj in critic_trajectory:
        if traj and traj[-1].response:
            final_critic_responses.append(traj[-1].response)
        else:
            final_critic_responses.append(None)

    final_answers = [extract_boxed_answer(r) if r else None for r in final_critic_responses]
    y_hat = _majority_vote(final_answers)

    if y_hat is None:
        # No ŷ → no signal: zero all real samples and pad (paper drops
        # un-graded trajectories).
        for s in args.results_dict["generator"]:
            s.reward = 0.0
        for traj in critic_trajectory:
            for s in traj:
                s.reward = 0.0
        _pad_role_buffer(args, "generator", target_gen)
        _pad_role_buffer(args, "critic", target_critic, donor_role="generator")
        return args.results_dict["generator"] + args.results_dict["critic"]

    # ---- Generator reward: per-sample, 1 if round-0 answer == ŷ ----
    for s in gen_samples_by_wid:
        if s is None:
            continue
        s.reward = 1.0 if _matches(s, y_hat) else 0.0

    # ---- Critic reward: trajectory-level. Agent i's FINAL response sets
    #      the reward for ALL of its critic samples. ----
    for traj in critic_trajectory:
        if not traj:
            continue
        final_match = _matches(traj[-1], y_hat)
        traj_reward = 1.0 if final_match else 0.0
        for s in traj:
            s.reward = traj_reward

    _pad_role_buffer(args, "generator", target_gen)
    _pad_role_buffer(args, "critic", target_critic, donor_role="generator")

    return args.results_dict["generator"] + args.results_dict["critic"]
