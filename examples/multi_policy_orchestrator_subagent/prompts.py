ORCHESTRATOR_PLAN_TEMPLATE = """You are a math problem orchestrator.
Read the problem and produce exactly 3 different solution approaches.
Output each approach in <approach_1>...</approach_1>,
<approach_2>...</approach_2>, <approach_3>...</approach_3> tags.
Each approach should be a brief instruction (1-3 sentences) for a
solver to follow. Do NOT solve the problem yourself.

### Problem

{problem_statement}
"""


SUBAGENT_TEMPLATE = """Solve the following math problem using the suggested approach.

End your response with EXACTLY this format on its own line:
Answer: \\boxed{{ANSWER}}
where ANSWER is the final numeric or symbolic answer.

### Problem

{problem_statement}

### Suggested approach

<approach>
{dispatch_instruction}
</approach>
"""


ORCHESTRATOR_SYNTHESIZE_TEMPLATE = """You are a math problem orchestrator. You previously planned 3 solution
approaches and received the results. Synthesize them into one correct
final answer.

End your response with EXACTLY this format on its own line:
Answer: \\boxed{{ANSWER}}
where ANSWER is the final numeric or symbolic answer.

### Problem

{problem_statement}

### Your plan

{plan}

### Result 1

{result_1}

### Result 2

{result_2}

### Result 3

{result_3}
"""


SUBAGENT_FALLBACK_DISPATCH = "Solve this problem using any approach you think is best."
