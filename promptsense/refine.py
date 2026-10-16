"""Intent-preserving prompt refinement.

The key design rule is: never insert generic placeholders that the user did not
provide. PromptSense should produce a usable prompt immediately.
"""
from __future__ import annotations
import re
from .llm import LLMClient
from .models import Analysis

def _clean(text):
    text=re.sub(r"\s+", " ", text).strip()
    return text

def _already_has(text, patterns):
    return any(re.search(p,text,re.I) for p in patterns)

def refine_with_rules(analysis: Analysis) -> str:
    original=_clean(analysis.normalized)
    task=analysis.task
    weak=set(analysis.weak_dimensions)

    # Simple prompts are refined minimally. This is critical for fair experiments:
    # refinement should improve prompt quality without changing task semantics.
    if task in {"explanation","question_answering"}:
        if not weak or weak <= {"context","constraints"}:
            return original
        additions=[]
        if "clarity" in weak and not original.endswith((".", "?", "!")):
            additions.append("Answer directly and clearly.")
        if "specificity" in weak:
            additions.append("Use one concrete example where helpful.")
        if "task_definition" in weak:
            additions.append("Give a concise explanation before additional detail.")
        return original + (" " + " ".join(additions) if additions else "")

    # For coding prompts, add only missing requirements.
    if task=="code_generation":
        parts=[original]
        if "specificity" in weak and not _already_has(original,[r"\bpython\b",r"\bjava\b",r"\bjavascript\b",r"\bsql\b"]):
            parts.append("Use the programming language implied by the request.")
        if "clarity" in weak:
            parts.append("Provide working, readable code and a brief explanation.")
        if "constraints" in weak:
            parts.append("Keep the solution focused on the requested task.")
        return " ".join(parts)

    if task=="summarization":
        parts=[original]
        if "constraints" in weak: parts.append("Preserve the key facts and do not add unsupported information.")
        if "specificity" in weak: parts.append("Focus on the main ideas and omit unnecessary detail.")
        return " ".join(parts)

    if task=="translation":
        parts=[original]
        if "constraints" in weak: parts.append("Preserve the original meaning and tone.")
        return " ".join(parts)

    # General fallback: concise, intent-preserving improvement.
    additions=[]
    if "clarity" in weak: additions.append("Answer directly and clearly.")
    if "specificity" in weak: additions.append("Include concrete details or an example where useful.")
    if "constraints" in weak: additions.append("Keep the response focused on the requested task.")
    return original + (" " + " ".join(additions) if additions else "")

MASTER_SYSTEM = "You are an expert prompt engineer. You rewrite prompts so they are clear, specific and complete, without changing what the user wants."

MASTER_PROMPT = """Rewrite the prompt below into ONE improved prompt that scores high on all six criteria:

1. CLARITY: one direct instruction in short, plain sentences (8-25 words each). No jargon.
2. SPECIFICITY: name the exact topic and deliverable. Include concrete numbers (length, number of examples, number of items).
3. CONTEXT: add one sentence of purpose and goal (for example "I am learning X and my goal is ... so that ..."). Keep it general. Never invent names, facts, personal details or a narrow audience the request does not imply.
4. TASK DEFINITION: start with the SAME action verb the original uses (e.g. "Explain", "Write", "Summarize"), and name the deliverable.
5. CONSTRAINTS: add only soft, quality-oriented constraints: a generous length range with numbers (e.g. "about 250-350 words"), a reading-friendly format (clear headings or short paragraphs), and 1-2 rules using "must" or "should" about accuracy and completeness (e.g. "must be accurate", "should include one short example"). NEVER restrict the answer: do not forbid code, technical terms or examples, do not fix exact paragraph counts, do not require plain text, and do not simplify the topic.
6. LOW AMBIGUITY: remove vague words (something, stuff, thing, good, nice, some, etc.), filler (um, like, pls, basically) and hedging (maybe, I guess, or something). No placeholders or brackets.

Hard rules:
- Keep the exact intent, task type and scope. A short request must not become a narrower or simpler task. Do not add new tasks or change the topic.
- If the prompt refers to missing material (e.g. "summarize this"), keep that reference and do not invent the material.
- Do not ask questions. Do not explain your changes.
- Plain text, maximum 120 words.
- Return ONLY the improved prompt.

Detected task: {task}
Weakest dimensions to fix first:
{recs}

Original prompt:
{original}
"""

def refine_with_llm(llm: LLMClient, analysis: Analysis, recommendations: list) -> str:
    recs = "\n".join(f"- {r.dimension}: {r.advice}" for r in recommendations) or "- none (still improve all six criteria)"
    prompt = MASTER_PROMPT.format(task=analysis.task, recs=recs, original=analysis.normalized)
    return llm.complete(prompt, system=MASTER_SYSTEM).strip().strip('"')

def refine(analysis, llm=None, recommendations=None):
    recommendations = recommendations or []
    if llm:
        try:
            result=refine_with_llm(llm,analysis,recommendations)
            if result and len(result)<=max(4000,len(analysis.normalized)*8):
                return result
        except Exception:
            pass
    return refine_with_rules(analysis)