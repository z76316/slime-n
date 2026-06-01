## Prompt templates for the multi-agent debate example (from the paper's
## `original_gen.py`):
##   - GENERATOR_INITIAL_TEMPLATE: round-0 independent proposal.
##   - generate_summarize_template(N): summarize N other agents' responses.
##   - GENERATOR_UPDATE_TEMPLATE: updated answer given a summary.
##
## Brace escape: a literal `{ANSWER}` in an f-string needs `{{{{ANSWER}}}}`
## to survive one f-string eval + one `.format()` (solver_summarizer KeyError).


# Round 0 — proposal. problem_statement is already chat-templated by slime.
GENERATOR_INITIAL_TEMPLATE = """{problem_statement}"""


def generate_summarize_template(num_other_agents: int) -> str:
    """Summarize-subroutine prompt over `num_other_agents` other-agent
    responses; output feeds the critic (A^C) next round. A^S is a subroutine
    (Algorithm 1): its samples are not added to results_dict or trained on."""
    sections = []
    for i in range(num_other_agents):
        sections.append(f"#### Agent {i+1} response\n{{solution{i+1}}}\n\n---")
    sections_text = "\n".join(sections)

    return f"""Here are responses from {num_other_agents} other agents on the same math problem:

{sections_text}

Write a concise summary of these solutions that calls out:
- Where the agents AGREE on the final answer or method.
- Where they DISAGREE (different answers, different approaches).
- Which approach is most defensible, and why.

Do NOT solve the problem yourself. Only summarize the other agents' reasoning.
"""


# Round k>=1 — agent updates its answer using its own prior response (so it
# iterates on its own reasoning, per the paper's chat-history debate) plus a
# summary of OTHER agents' prior responses.
# Plain string through ONE .format() → `{ANSWER}` escapes as `{{ANSWER}}`
# (two braces); the f-string template above uses four — different rules.
GENERATOR_UPDATE_TEMPLATE = """Original problem:

{problem_statement}

---

Your previous response was:

{prior_response}

---

Here is a summary of solutions from other agents on this problem:

{summary}

---

Use the other agents' reasoning above as additional advice. Re-examine your
previous response and provide an updated bullet-point answer, ending with
EXACTLY this format on its own line:

Answer: \\boxed{{ANSWER}}

where ANSWER is the final numeric or symbolic answer.
"""
