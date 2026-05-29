PEER_ROUND1_TEMPLATE = """Solve the following math problem.

End your response with EXACTLY this format on its own line:
Answer: \\boxed{{ANSWER}}
where ANSWER is the final numeric or symbolic answer.

### Problem

{problem_statement}
"""


PEER_ROUND2_TEMPLATE = """You previously attempted this problem. Another solver also attempted it
independently. Review both attempts and produce a refined answer.

End your response with EXACTLY this format on its own line:
Answer: \\boxed{{ANSWER}}
where ANSWER is the final numeric or symbolic answer.

### Problem

{problem_statement}

### Your previous attempt

<your_attempt>
{own_round1_solution}
</your_attempt>

### Other solver's attempt

<other_attempt>
{other_round1_solution}
</other_attempt>
"""


PEER_ROUND3_TEMPLATE = """This is the final round. You and another solver have each attempted this
problem twice. Review the full history and produce your best final answer.

End your response with EXACTLY this format on its own line:
Answer: \\boxed{{ANSWER}}
where ANSWER is the final numeric or symbolic answer.

### Problem

{problem_statement}

### Round 1 — your attempt

<your_round1>
{own_round1_solution}
</your_round1>

### Round 1 — other solver's attempt

<other_round1>
{other_round1_solution}
</other_round1>

### Round 2 — your refined attempt

<your_round2>
{own_round2_solution}
</your_round2>

### Round 2 — other solver's refined attempt

<other_round2>
{other_round2_solution}
</other_round2>
"""
