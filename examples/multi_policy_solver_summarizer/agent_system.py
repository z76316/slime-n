"""Solver → summarizer chain. N parallel solvers per prompt, then N
summarizers that each see all N solver candidates. Both roles train
on their own buffers (split-buffer mode); see README for the reward shape."""

import asyncio
import itertools
import logging
import re
import time
import traceback
from copy import deepcopy

from slime.rollout.rm_hub import batched_async_rm
from slime.rollout.sglang_rollout import get_model_url
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .prompts import SOLVER_PROMPT_TEMPLATE, generate_summarize_template

logger = logging.getLogger(__name__)

# Unique indices for inner samples: deep-copied siblings all inherit the outer
# index, so a high offset keeps them clear of slime's data_source counter and
# avoids get_data_iterator's uniqueness assertion.
_INNER_SAMPLE_ID = itertools.count(start=1_000_000_000)


# Qwen chat-control tokens, stripped from the solver prompt before embedding it
# as plain problem text inside the summarizer template.
_CHAT_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>(?:user|assistant|system)?\s*")


def _strip_chat_tokens(text: str) -> str:
    """Strip chat-control tokens so inner text can be embedded in another prompt
    without nested chat structures (which confuse downstream models)."""
    return _CHAT_TOKEN_RE.sub("", text).strip()


def _wrap_user_turn(tokenizer, user_content: str) -> str:
    """Render `user_content` as a single user turn via the chat template;
    fall back to manual wrap if the tokenizer has none (e.g. test stubs)."""
    if getattr(tokenizer, "chat_template", None) is None:
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


async def generate_response(args, prompt, key):
    """Call the policy's paired sglang engine; tag the Sample with
    policy_name=key so the manager routes it to the right buffer."""
    try:
        sampling_params = args.sampling_params
        tokenizer = args.tokenizer
        max_context_length = args.rollout_max_context_len
        sample = deepcopy(args.sample)

        # Route to the sglang engine paired with this role.
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
        # Keep per-token logprobs for train-side train_rollout_logprob_abs_diff
        # (and tis_* under --use-tis).
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "stop":
                sample.status = Sample.Status.COMPLETED

        # Buffer-routing tag + unique inner index (deepcopy inherited the outer
        # index, which would collide across num_parallel siblings).
        sample.policy_name = key
        sample.index = next(_INNER_SAMPLE_ID)
        args.results_dict[key].append(sample)

        final = output["text"].replace("<|user|>", "")
        if "</think>" in final:
            contents = final.split("</think>")
            if len(contents) == 2 and contents[1] != "":
                sample.reason_content = contents[0].strip()
                sample.response_content = contents[1].strip()
                return sample.response_content
        sample.reason_content = None
        sample.response_content = None
        return None
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


class Agent:
    async def run(self, args, prompt, max_retries: int = 1, key: str = None) -> str:
        for _ in range(max_retries):
            try:
                response = await generate_response(args, prompt, key=key)
                return response
            except Exception as e:
                print(f"Error querying LLM: {e}")
                time.sleep(1)
        return None


class SolverAgent(Agent):
    async def generate_initial_solution(self, args, problem_statement) -> str:
        prompt = SOLVER_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
        return await self.run(args, prompt, max_retries=3, key="solver")


class SummarizerAgent(Agent):
    async def summarize(self, args, problem_statement, candidate_solutions: list[str]) -> str:
        template = generate_summarize_template(len(candidate_solutions))
        # problem_statement arrives raw (see run_agent_system); candidate
        # solutions are model outputs that may carry stray chat tokens — strip.
        format_params = {"problem_statement": problem_statement}
        for i, solution in enumerate(candidate_solutions):
            format_params[f"solution{i+1}"] = _strip_chat_tokens(solution)
        body = template.format(**format_params)
        # Wrap as a proper user turn (avoids bare text with nested tokens).
        prompt = _wrap_user_turn(args.tokenizer, body)
        return await self.run(args, prompt, max_retries=3, key="summarizer")


async def solver_worker(args, problem_statement, worker_id):
    try:
        solver = SolverAgent()
        return await solver.generate_initial_solution(args, problem_statement)
    except Exception as e:
        print(f"[Solver-{worker_id}] exception: {e}\n{traceback.format_exc()}")
        return None


async def summarize_worker(args, problem_statement, candidate_solutions, worker_id):
    """Run one summarizer. Running num_parallel in parallel gives the
    summarizer policy enough trajectories per problem for GRPO group-norm."""
    try:
        summarizer = SummarizerAgent()
        return await summarizer.summarize(args, problem_statement, candidate_solutions)
    except Exception as e:
        print(f"[Summarizer-{worker_id}] exception: {e}\n{traceback.format_exc()}")
        return None


def _pad_role_buffer(args, role: str, target_count: int, donor_role: str | None = None):
    """Top up `args.results_dict[role]` to exactly `target_count` samples.

    Each rollout must contribute exactly num_parallel samples per role or
    get_data_iterator's num_steps_per_rollout assertion trips. Pad with
    zero-response placeholders when a phase fails.
    """
    samples = args.results_dict[role]
    if len(samples) >= target_count:
        del samples[target_count:]  # trim if longer than expected
        return
    donor_pool = samples if samples else (args.results_dict.get(donor_role) or [])
    if not donor_pool:
        donor_pool = [args.sample]
    donor_policy = getattr(donor_pool[0], "policy_name", None)
    logger.warning(
        "_pad_role_buffer: role=%s padding %d zero-response placeholder(s) from donor_policy=%s outer_index=%s",
        role,
        target_count - len(samples),
        donor_policy,
        getattr(args.sample, "index", None),
    )

    def _prompt_tokens(source: Sample) -> list[int]:
        tokens = list(getattr(source, "tokens", []) or [])
        response_length = getattr(source, "response_length", 0) or 0
        if tokens and response_length > 0:
            return tokens[:-response_length]
        if tokens:
            return tokens

        prompt = getattr(source, "prompt", "")
        tokenizer = getattr(args, "tokenizer", None)
        if isinstance(prompt, str) and tokenizer is not None:
            return tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return []

    while len(samples) < target_count:
        donor = donor_pool[0]
        prompt_tokens = _prompt_tokens(donor)
        placeholder = deepcopy(donor)
        placeholder.policy_name = role
        placeholder.index = next(_INNER_SAMPLE_ID)
        placeholder.reward = 0.0
        placeholder.metadata = {
            **(placeholder.metadata or {}),
            "raw_reward": 0.0,
            "is_padding_placeholder": True,
            "padding_donor_policy": getattr(donor, "policy_name", None),
        }
        placeholder.status = Sample.Status.FAILED
        placeholder.response = ""
        placeholder.response_length = 0
        placeholder.loss_mask = []
        placeholder.remove_sample = True
        placeholder.response_content = None
        placeholder.reason_content = None
        placeholder.tokens = prompt_tokens
        # train_data collection checks only the first sample; [] is the
        # aligned value for a zero-length response.
        placeholder.rollout_log_probs = []
        placeholder.rollout_routed_experts = None
        placeholder.teacher_log_probs = None
        samples.append(placeholder)


async def run_agent_system(args, sample):
    """Run num_parallel solvers, then num_parallel summarizers. Returns a flat
    list tagged with policy_name in {"solver", "summarizer"} for split-buffer
    routing."""
    args = deepcopy(args)
    args.sample = sample
    args.results_dict = {"solver": [], "summarizer": []}

    # sample.prompt is already chat-templated: pass it straight to the solver's
    # sglang engine, but keep a stripped raw copy for the summarizer template
    # (embedding the chat-formatted string there would nest chat structures).
    solver_prompt = sample.prompt
    raw_problem = _strip_chat_tokens(sample.prompt)

    n = args.num_parallel

    # Phase 1 — solvers (in parallel)
    tasks = [solver_worker(args, solver_prompt, wid) for wid in range(n)]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Score each solver directly (RLVR on its own response).
    rewards = await batched_async_rm(args, args.results_dict["solver"])
    for s, r in zip(args.results_dict["solver"], rewards, strict=False):
        s.reward = r
        # Keep unscaled RM verdict: reward_adjustment scales s.reward by
        # 0.8/1.2, but eval needs the raw 0/1 signal for pass@k.
        s.metadata["raw_reward"] = r

    candidate_solutions = [s.response_content for s in args.results_dict["solver"] if s.response_content is not None]

    def reward_adjustment(samples, weight):
        for s in samples:
            s.reward = s.reward * weight
        return samples

    if len(candidate_solutions) == 0:
        # No usable solver output; pad both roles to num_parallel (split-buffer
        # routing needs each rollout to contribute the same count per role).
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        _pad_role_buffer(args, "solver", n)
        _pad_role_buffer(args, "summarizer", n, donor_role="solver")
        return args.results_dict["solver"] + args.results_dict["summarizer"]

    # Phase 2 — summarizers (in parallel; each synthesizes from all solver candidates)
    tasks = [summarize_worker(args, raw_problem, candidate_solutions, wid) for wid in range(n)]
    await asyncio.gather(*tasks, return_exceptions=True)

    # No summarizer output: don't penalize correct solvers for a fully failed
    # summarizer phase.
    if not args.results_dict["summarizer"]:
        _pad_role_buffer(args, "solver", n)
        _pad_role_buffer(args, "summarizer", n, donor_role="solver")
        return args.results_dict["solver"] + args.results_dict["summarizer"]

    # Score each summarizer on sample.response (RM grade is independent of
    # whether `</think>` appears, so no response_content filter).
    summarizer_rewards = await batched_async_rm(args, args.results_dict["summarizer"])
    for s, r in zip(args.results_dict["summarizer"], summarizer_rewards, strict=False):
        s.reward = r
        s.metadata["raw_reward"] = r

    # Group reward shaping: bonus both roles if the summarizer phase was mostly
    # correct, else penalize. Mean over all summarizer samples (each is valid).
    mean_summarizer_reward = sum(s.reward for s in args.results_dict["summarizer"]) / len(
        args.results_dict["summarizer"]
    )
    weight = args.correct_reward_weight if mean_summarizer_reward > 0.5 else args.incorrect_reward_weight
    reward_adjustment(args.results_dict["solver"], weight)
    reward_adjustment(args.results_dict["summarizer"], weight)

    # Pad each role to num_parallel so the per-role count is invariant across calls.
    _pad_role_buffer(args, "solver", n)
    _pad_role_buffer(args, "summarizer", n, donor_role="solver")

    return args.results_dict["solver"] + args.results_dict["summarizer"]
