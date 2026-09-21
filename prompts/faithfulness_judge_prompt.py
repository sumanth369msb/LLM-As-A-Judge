from langchain.prompts import PromptTemplate

# ==============================================================================
# FAITHFULNESS (HALLUCINATION DETECTION) JUDGE PROMPT - 1 TO 5 LIKERT SCALE
# Evaluates purely whether the generated summary adheres to the source chat logs.
# Requires explicit step-by-step JUSTIFICATION before assigning the numerical score.
# ==============================================================================

FAITHFULNESS_JUDGE_PROMPT = PromptTemplate(
    input_variables=["contexts", "answer"],
    template="""You are an expert Machine Learning Quality Assurance Evaluator specializing in hallucination detection.
Your task is to evaluate the 'Faithfulness' of a generated summary based STRICTLY on the provided raw incident chat logs.
You must provide a clear, evidence-based JUSTIFICATION explaining your score before outputting the final rating.

---
SCORING RUBRIC (1 to 5 Scale):

- 5 (Completely Faithful): 100% factually accurate. All entities, hardware, timestamps, and error states match the logs. Direct logical domain entailment (e.g., restarting an offline VPN 'to restore access') is permitted and awarded 5.
- 4 (Mostly Faithful): Accurate in all critical technical details, with only minor ambiguous wording or slight extraneous detail that does not distort facts.
- 3 (Partially Faithful / Minor Hallucination): Contains minor invented entities, swapped hardware/software terms (e.g., calling a switch a router), or unverified secondary claims.
- 2 (Mostly Unfaithful / Significant Hallucination): Multiple claims conflict directly with the logs or introduce unmentioned incident causes.
- 1 (Completely Unfaithful / Fabricated): Gross hallucination. Completely invents facts, causes, or outcomes not found in the logs.

ANTI-VERBOSITY RULE:
Do NOT penalize concise answers. Base your score strictly on factual consistency, never word count.

---
FEW-SHOT EXAMPLES:

[Example 1]
Context: "User: Server down. DBAdmin: Oracle DB crashed at 10pm. Fixed by reboot."
Summary: "Oracle database crashed at 10pm and was fixed by reboot."
Output JSON:
{{
  "justification": "All claims, including the 10pm timestamp, Oracle database entity, and reboot resolution, directly match the chat logs with zero fabricated information.",
  "faithfulness_score": 5
}}

[Example 2]
Context: "User: Network slow. NetAdmin: Switch 4 failed. Swapped it."
Summary: "Switch 4 failed causing network issues. Replaced router."
Output JSON:
{{
  "justification": "The summary correctly captures the network degradation and Switch 4 failure, but hallucinates replacing a router when logs explicitly state Switch 4 was swapped.",
  "faithfulness_score": 3
}}

[Example 3]
Context: "User: VPN offline. IT: Restarted VPN service."
Summary: "VPN service was restarted to restore access."
Output JSON:
{{
  "justification": "Restarting the VPN service matches the logs directly. Stating 'to restore access' is a valid domain entailment of resolving an offline service, not a factual hallucination.",
  "faithfulness_score": 5
}}

---
EVALUATE THIS RECORD:

Raw Incident Chat Logs (Contexts):
{contexts}

Generated Summary (Answer):
{answer}

Respond ONLY with a valid JSON object matching this schema:
{{
  "justification": "<Evidence-based explanation citing specific facts from logs that justify the score>",
  "faithfulness_score": <Integer from 1 to 5>
}}
"""
)
