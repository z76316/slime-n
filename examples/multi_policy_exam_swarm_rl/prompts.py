## Prompt template for the Exam Swarm RL example.
##
## All 8 agents see the same problem statement, untouched. The dataset
## (DAPO-math-17k) already provides a chat-formatted question; the
## template just passes it through. Pre-existing solver-style scaffolding
## is kept minimal so per-agent answers are directly comparable.


EXAM_PROMPT_TEMPLATE = """{problem_statement}"""
