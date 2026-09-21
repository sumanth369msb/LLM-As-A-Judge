from langchain.prompts import PromptTemplate

# ==============================================================================
# RELEVANCY (PROMPT ALIGNMENT) JUDGE PROMPT - 1 TO 5 LIKERT SCALE
# Evaluates purely whether the generated summary addresses the user prompt.
# Requires explicit step-by-step JUSTIFICATION before assigning the numerical score.
# ==============================================================================

RELEVANCY_JUDGE_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""You are an expert Machine Learning Quality Assurance Evaluator specializing in prompt alignment.
Your task is to evaluate the 'Relevancy' of a generated summary based STRICTLY on the user's prompt request.
You must provide a clear, evidence-based JUSTIFICATION explaining your score before outputting the final rating.

---
SCORING RUBRIC (1 to 5 Scale):

- 5 (Completely Relevant): Directly, accurately, and comprehensively addresses the user prompt without omitting the core incident event.
- 4 (Mostly Relevant): Addresses the core request well with minor fluff or slight verbosity.
- 3 (Partially Relevant): Only partially addresses the request, omitting important aspects of the incident.
- 2 (Barely Relevant): Mostly misses the user request; tangentially related to the topic.
- 1 (Completely Irrelevant): Fails to address the user request entirely (e.g., talks about unrelated topics).

ANTI-VERBOSITY RULE:
Do NOT penalize concise answers. If a short answer completely fulfills the prompt, award it a 5.

---
FEW-SHOT EXAMPLES:

[Example 1]
User Request: "Summarize the major incident"
Summary: "Oracle database crashed at 10pm and was fixed by reboot."
Output JSON:
{{
  "relevancy_justification": "The summary directly and succinctly fulfills the prompt by identifying the impacted system (Oracle DB) and the resolution (reboot).",
  "relevancy_score": 5
}}

[Example 2]
User Request: "Summarize the major incident"
Summary: "Switch 4 failed causing network issues. Replaced router."
Output JSON:
{{
  "relevancy_justification": "The summary directly answers the request to summarize the incident by reporting network issues and component replacement, despite a technical entity error.",
  "relevancy_score": 5
}}

[Example 3]
User Request: "Summarize the major incident"
Summary: "The office cafeteria will be closed next Monday."
Output JSON:
{{
  "relevancy_justification": "The output is completely off-topic and fails to discuss any IT system incident or operational downtime.",
  "relevancy_score": 1
}}

---
EVALUATE THIS RECORD:

User Request (Question):
{question}

Generated Summary (Answer):
{answer}

Respond ONLY with a valid JSON object matching this schema:
{{
  "relevancy_justification": "<Evidence-based explanation of how well the summary satisfies the prompt request>",
  "relevancy_score": <Integer from 1 to 5>
}}
"""
)
