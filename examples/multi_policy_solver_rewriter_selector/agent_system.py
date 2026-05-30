import asyncio
import itertools
import logging
import re
import traceback
from copy import deepcopy

from slime.rollout.rm_hub import batched_async_rm
from slime.rollout.sglang_rollout import get_model_url
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .prompts import SOLVER_PROMPT_TEMPLATE, generate_rewriter_template, generate_select_template

logger = logging.getLogger(__name__)

# Unique index for inner samples; siblings would otherwise share the outer
# prompt's index and trip get_data_iterator's uniqueness assertion.
_INNER_SAMPLE_ID = itertools.count(start=1_000_000_000)


# Strip chat-control tokens so chat text can be embedded in another prompt
# without nesting chat structures.
_CHAT_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>(?:user|assistant|system)?\s*")


def _strip_chat_tokens(text: str) -> str:
    return _CHAT_TOKEN_RE.sub("", text).strip()


def _wrap_user_turn(tokenizer, user_content: str) -> str:
    """Render `user_content` as a single-turn user message via chat template."""
    if getattr(tokenizer, "chat_template", None) is None:
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


async def generate_response(args, prompt, key):
    try:
        sampling_params = args.sampling_params
        tokenizer = args.tokenizer
        max_context_length = args.rollout_max_context_len
        sample = deepcopy(args.sample)

        # Route to the sglang engine paired with this role (key matches --sglang-config).
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

        if "output_token_logprobs" not in output["meta_info"]:
            logger.warning("Missing output_token_logprobs for role=%s", key)
            return None

        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        if not new_response_tokens:
            return None

        # Append tokens directly, avoiding re-tokenization.
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response = output["text"]
        # Keep per-token logprobs for train-side train_rollout_logprob_abs_diff / tis_*.
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            # case "abort":
            #     sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED

        # Tag for buffer routing and give a unique index (the deepcopy inherits
        # the outer prompt's index, which would collide across siblings).
        sample.policy_name = key
        sample.index = next(_INNER_SAMPLE_ID)
        args.results_dict[key].append(sample)

        final = output["text"].replace("<|user|>", "")
        if "</think>" in final:
            contents = final.split("</think>", 1)
            if len(contents) == 2 and contents[1].strip():
                reason_content = contents[0].strip()
                response_content = contents[1].strip()
                sample.reason_content = reason_content
                sample.response_content = response_content
                return response_content
        sample.reason_content = None
        sample.response_content = _strip_chat_tokens(final).strip()
        return sample.response_content if sample.response_content else None
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


class Agent:
    """A base class for our AI agents."""

    def __init__(self):
        pass

    async def run(self, args, prompt, max_retries: int = 1, key: str = None) -> str:
        """Runs the agent by sending a prompt to the LLM."""
        for attempt in range(max_retries):
            try:
                response = await generate_response(args, prompt, key=key)
                if response is not None:
                    return response
            except Exception as e:
                print(f"Error querying LLM: {e}")
            if attempt + 1 < max_retries:
                await asyncio.sleep(1)
        print(f"Failed to query LLM after {max_retries} retries")
        return None


class SolverAgent(Agent):
    """The agent responsible for generating and improving solutions."""

    def __init__(self):
        super().__init__()

    async def generate_initial_solution(self, args, problem_statement) -> str:
        """Generates the first solution attempt."""
        prompt = SOLVER_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
        return await self.run(args, prompt, max_retries=3, key="solver")


class RewriterAgent(Agent):
    """The agent responsible for rewriting solutions."""

    def __init__(self):
        super().__init__()

    async def rewrite(self, args, problem_statement, previous_solutions: list[str]) -> str:
        """Generate the rewritten solution."""
        template = generate_rewriter_template(len(previous_solutions))

        # problem_statement is raw; strip stray chat tokens from model-output solutions.
        format_params = {"problem_statement": problem_statement}
        for i, solution in enumerate(previous_solutions):
            format_params[f"solution{i+1}"] = _strip_chat_tokens(solution)

        body = template.format(**format_params)
        prompt = _wrap_user_turn(args.tokenizer, body)  # wrap as a user turn
        return await self.run(args, prompt, max_retries=1, key="rewriter")


class SelectorAgent(Agent):
    """The agent responsible for selecting solutions."""

    def __init__(self):
        super().__init__()

    async def select(self, args, problem_statement, candidate_solutions: list[str]) -> str:
        """Generate the selector judgment."""
        template = generate_select_template(len(candidate_solutions))

        # problem_statement is raw; solutions are stripped of chat tokens too.
        format_params = {"problem_statement": problem_statement}
        for i, solution in enumerate(candidate_solutions):
            format_params[f"solution{i+1}"] = _strip_chat_tokens(solution)

        body = template.format(**format_params)
        prompt = _wrap_user_turn(args.tokenizer, body)
        return await self.run(args, prompt, max_retries=10, key="selector")

    def extract_selected_solution_idx(self, response: str, candidate_solutions: list[str]) -> int:
        """Extract the selected solution index. Accepts "Judgment: 1 / IDX 1 / Solution 1 / #1"."""
        PATTERN = re.compile(r"Judgment:\s*(?:IDX|Solution)?\s*#?(\d+)", re.IGNORECASE)
        matched = PATTERN.findall(response)
        if not matched:
            return None
        try:
            selected_id = int(matched[0]) - 1
            if selected_id < len(candidate_solutions) and selected_id >= 0:
                return selected_id
            else:
                return None
        except Exception as e:
            print(f"extract_selected_solution_idx error: {e}")
            return None


async def rewrite_worker(args, previous_solutions, problem_statement, worker_id):
    rewriter = RewriterAgent()
    new_solution = await rewriter.rewrite(args, problem_statement, previous_solutions)
    return new_solution


async def solver_worker(args, problem_statement, worker_id):
    """Run a single solver pipeline."""
    try:
        solver = SolverAgent()
        current_solution = await solver.generate_initial_solution(args, problem_statement)
        return current_solution

    except Exception as e:
        print(f"[Worker-{worker_id}] exception: {e}")
        print(f"[Worker-{worker_id}] traceback: {traceback.format_exc()}")
        return None


async def select_worker(args, problem_statement, candidate_solutions, worker_id):
    """Run one selector; num_parallel run in parallel to give GRPO its group per problem."""
    try:
        selector = SelectorAgent()
        response = await selector.select(args, problem_statement, candidate_solutions)
        return response
    except Exception as e:
        print(f"[Selector-{worker_id}] exception: {e}")
        print(f"[Selector-{worker_id}] traceback: {traceback.format_exc()}")
        return None


def _pad_role_buffer(args, role: str, target_count: int, donor_role: str | None = None):
    """Top up `args.results_dict[role]` to exactly `target_count` samples.

    Split-buffer routing needs each role's per-rollout buffer to match
    global_batch_size; a short buffer trips data.py's num_local_samples assert.
    Pad with zero-response placeholders so failed phases don't leak donor
    tokens/logprobs into another policy's buffer.
    """
    samples = args.results_dict[role]
    if len(samples) >= target_count:
        del samples[target_count:]
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

    role_has_logprobs = any(sample.rollout_log_probs is not None for sample in samples)

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
        # train_data checks the first sample only; match the buffer's logprob presence.
        placeholder.rollout_log_probs = [] if role_has_logprobs else None
        placeholder.rollout_routed_experts = None
        placeholder.teacher_log_probs = None
        samples.append(placeholder)


async def run_agent_system(args, sample):
    """Run `num_parallel` pipelines concurrently."""
    args = deepcopy(args)  # rollout_with_multi_agents mutates args
    args.sample = sample
    args.results_dict = {"solver": [], "rewriter": [], "selector": []}

    # Solver uses the chat-formatted prompt directly; rewriter/selector embed the
    # RAW problem in their own template to avoid nested chat structures.
    solver_prompt = sample.prompt
    raw_problem = _strip_chat_tokens(sample.prompt)
    tasks = [solver_worker(args, solver_prompt, worker_id) for worker_id in range(args.num_parallel)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    rewards = await batched_async_rm(args, args.results_dict["solver"])
    for sample, reward in zip(args.results_dict["solver"], rewards, strict=False):
        sample.reward = reward

    previous_solutions = [item for item in results if isinstance(item, str)]

    def reward_adjustment(samples, reward_weight):
        for sample in samples:
            sample.reward = sample.reward * reward_weight
        return samples

    n = args.num_parallel

    if len(previous_solutions) == 0:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        # Pad all roles to equal per-rollout counts (get_data_iterator asserts this).
        _pad_role_buffer(args, "solver", n)
        _pad_role_buffer(args, "rewriter", n, donor_role="solver")
        _pad_role_buffer(args, "selector", n, donor_role="solver")
        return args.results_dict["solver"] + args.results_dict["rewriter"] + args.results_dict["selector"]

    # Rewriting — feed the raw problem so solver chat tokens don't leak through.
    tasks = [
        rewrite_worker(args, previous_solutions, raw_problem, worker_id) for worker_id in range(args.num_parallel)
    ]
    rewrited_solutions_raw = await asyncio.gather(*tasks, return_exceptions=True)

    # Keep only valid rewritten solutions.
    rewrited_solutions = []
    for _i, result in enumerate(rewrited_solutions_raw):
        if isinstance(result, str):
            rewrited_solutions.append(result)

    rewards = await batched_async_rm(args, args.results_dict["rewriter"])
    for sample, reward in zip(args.results_dict["rewriter"], rewards, strict=False):
        sample.reward = reward

    if len(rewrited_solutions) == 0:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        reward_adjustment(args.results_dict["rewriter"], args.incorrect_reward_weight)
        _pad_role_buffer(args, "solver", n)
        _pad_role_buffer(args, "rewriter", n, donor_role="solver")
        _pad_role_buffer(args, "selector", n, donor_role="solver")
        return args.results_dict["solver"] + args.results_dict["rewriter"] + args.results_dict["selector"]

    # Selection — num_parallel selectors for GRPO group-norm; selector gets the
    # raw problem and chat-templates it itself.
    selector = SelectorAgent()  # for extract_selected_solution_idx
    tasks = [select_worker(args, raw_problem, rewrited_solutions, worker_id) for worker_id in range(args.num_parallel)]
    await asyncio.gather(*tasks, return_exceptions=True)

    if len(args.results_dict["selector"]) == 0:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        reward_adjustment(args.results_dict["rewriter"], args.incorrect_reward_weight)
        _pad_role_buffer(args, "solver", n)
        _pad_role_buffer(args, "rewriter", n, donor_role="solver")
        _pad_role_buffer(args, "selector", n, donor_role="solver")
        return args.results_dict["solver"] + args.results_dict["rewriter"] + args.results_dict["selector"]

    # Reward each selector by the rewriter solution it picked (matched via
    # response_content, so order is irrelevant). Unparseable judgments are
    # excluded from mean_selector_reward to avoid anti-training correct upstream.
    parsed_selector_rewards: list[float] = []
    for sel_sample in args.results_dict["selector"]:
        response = sel_sample.response_content
        if response is None:
            sel_sample.reward = 0
            continue
        selected_solution_idx = selector.extract_selected_solution_idx(response, rewrited_solutions)
        if selected_solution_idx is None:
            sel_sample.reward = 0
            continue
        selected_solution = rewrited_solutions[selected_solution_idx]
        sel_sample.reward = 0  # default if no match
        for r_sample in args.results_dict["rewriter"]:
            if r_sample.response_content is not None and selected_solution in r_sample.response_content:
                sel_sample.reward = r_sample.reward
                break
        parsed_selector_rewards.append(sel_sample.reward)

    # Group shaping: bonus all roles if mean selector reward is high, else penalize.
    # If no selector parsed, leave raw rewards (no judgment signal — don't anti-train).
    if not parsed_selector_rewards:
        _pad_role_buffer(args, "solver", n)
        _pad_role_buffer(args, "rewriter", n, donor_role="solver")
        _pad_role_buffer(args, "selector", n, donor_role="solver")
        return args.results_dict["solver"] + args.results_dict["rewriter"] + args.results_dict["selector"]

    mean_selector_reward = sum(parsed_selector_rewards) / len(parsed_selector_rewards)
    weight = args.correct_reward_weight if mean_selector_reward > 0.5 else args.incorrect_reward_weight
    reward_adjustment(args.results_dict["solver"], weight)
    reward_adjustment(args.results_dict["rewriter"], weight)
    reward_adjustment(args.results_dict["selector"], weight)

    # Final guard: each role buffer = num_parallel.
    _pad_role_buffer(args, "solver", n)
    _pad_role_buffer(args, "rewriter", n, donor_role="solver")
    _pad_role_buffer(args, "selector", n, donor_role="solver")

    return args.results_dict["solver"] + args.results_dict["rewriter"] + args.results_dict["selector"]
