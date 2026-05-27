GENERATOR_ROUND1_TEMPLATE = """Solve the following math problem.

End your response with EXACTLY this format on its own line:
Answer: \\boxed{{ANSWER}}
where ANSWER is the final numeric or symbolic answer.

### Problem

{problem_statement}
"""


VERIFIER_TEMPLATE = """You will be given a math problem and one candidate solution.

First output exactly one verdict tag:
<verdict>approve</verdict> if the candidate solution is correct, or
<verdict>reject</verdict> if it is incorrect or incomplete.

Then write a brief critique. If the answer is wrong, identify the first important
mistake and what should be fixed. If the answer is correct, state the key reason
it is valid.

### Problem

{problem_statement}

### Candidate solution

{candidate_solution}
"""


GENERATOR_ROUND2_TEMPLATE = """Solve the following math problem again, using the
candidate attempt and verifier critique as feedback.

End your response with EXACTLY this format on its own line:
Answer: \\boxed{{ANSWER}}
where ANSWER is the final numeric or symbolic answer.

### Problem

{problem_statement}

### Candidate attempt

<candidate_attempt>
{candidate_solution}
</candidate_attempt>

### Verifier critique

<critique>
{critique}
</critique>
"""
