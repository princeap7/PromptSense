"""Prompt quality analysis.

The scorer is deliberately conservative: a short prompt is not treated as
bad merely because it lacks optional context or constraints. Missing information
is marked weak only when it is genuinely useful for the detected task.
"""
from __future__ import annotations
import re
from typing import Optional
from .llm import LLMClient, extract_json
from .models import DIMENSIONS, WEIGHTS, Analysis, DimensionScore
from .preprocess import normalize, split_sentences, tokenize
from .task import detect_task

VAGUE = {"something","stuff","things","thing","good","nice","some","etc","whatever",
         "anything","somehow","better","great","cool","interesting","various","several"}
FILLER = {"um","uh","like","basically","actually","literally","pls","plz","kinda","sorta","lol","gonna","wanna"}
HEDGES = ("maybe","i guess","i think","sort of","kind of","or something","not sure")
ACTION_VERBS = {"write","explain","summarize","summarise","translate","classify","extract",
                "generate","create","list","compare","analyze","analyse","describe",
                "implement","build","design","draft","plan","review","rewrite","fix",
                "debug","outline","give","make","tell","show","convert","find","calculate"}
DELIVERABLES = {"summary","list","table","function","email","essay","report","outline","plan",
                "script","code","explanation","paragraph","post","story","poem","comparison",
                "analysis","answer","examples","steps","guide"}

def _clamp(x): return round(max(0.0, min(10.0, x)), 1)
def find_vague(text): return sorted({w for w in tokenize(text.lower()) if w in VAGUE})
def find_filler(text): return sorted({w for w in tokenize(text.lower()) if w in FILLER})

def _clarity(text, words, sents):
    if not words: return 0, "Empty prompt."
    avg = len(words) / max(1, len(sents))
    score = 6.0
    notes = []
    if 5 <= avg <= 35: score += 2
    elif avg > 50: score -= 2; notes.append("very long sentence structure")
    if len(words) < 4: score -= 2; notes.append("very short")
    if find_vague(text): score -= 1; notes.append("contains vague wording")
    return _clamp(score), "; ".join(notes) or "direct wording and manageable sentence structure"

def _specificity(text, words, sents):
    score = 5.0
    if re.search(r"\b\d+(?:\.\d+)?\b", text): score += 1
    if re.search(r"['\"“].+?['\"”]", text): score += 1
    if any(w.lower() in DELIVERABLES for w in words): score += 1
    if len(words) >= 12: score += 1
    vague = find_vague(text)
    score -= min(3, 0.8 * len(vague))
    return _clamp(score), ("concrete task/detail signals present" if score >= 6 else
                            "few concrete requirements; task may benefit from added detail")

def _context(text, words, sents, task):
    markers = re.findall(r"\b(i am|i'm|my|we|our|because|background|context|audience|goal|purpose|so that|in order to|beginner|expert|student|customer|team|company|project)\b", text, re.I)
    # Context is optional for simple factual questions. Do not punish these prompts heavily.
    if task in {"explanation","question_answering","general"} and len(words) <= 15 and not markers:
        return 7.0, "explicit background is not necessary for this simple task"
    score = 4.0 + min(4, len(set(m.lower() for m in markers)))
    if task in {"code_generation","summarization","translation","analysis","creative_writing","planning"} and not markers:
        score -= 1
    return _clamp(score), ("context markers found" if markers else "no explicit audience/background/goal")

def _task_definition(text, words, task):
    lowered = [w.lower() for w in words]
    verbs = [w for w in lowered if w in ACTION_VERBS]
    deliverable = [w for w in lowered if w in DELIVERABLES]
    if task in {"question_answering","explanation"} and (verbs or re.search(r"\bwhat is\b|\bwhat are\b|\bwhy\b|\bhow\b", text.lower())):
        return 9.0, f"clear question/explanation task; action verb: {verbs[0] if verbs else 'implicit'}"
    score = 4 + (2 if verbs else 0) + (2 if deliverable else 0) + (1 if task != "general" else 0)
    return _clamp(score), f"task={task}; action verb: {verbs[0] if verbs else 'none'}; deliverable: {deliverable[0] if deliverable else 'none'}"

def _constraints(text, words, task):
    patterns = [
        r"\b\d+\s*(words?|sentences?|paragraphs?|pages?|bullets?|items?|steps?|lines?)\b",
        r"\b(json|markdown|table|bullet|csv|yaml|xml|numbered|headings?|code block)\b",
        r"\b(formal|casual|friendly|professional|technical|simple|plain language|persuasive|neutral)\b",
        r"\b(must|should|do not|don't|avoid|only|exactly|never|always|include|exclude)\b",
        r"\b(in english|in hindi|in gujarati|python|javascript|java|sql|focus on)\b",
    ]
    hits = sum(bool(re.search(p, text, re.I)) for p in patterns)
    # For simple questions, constraints are optional rather than inherently missing.
    if task in {"explanation","question_answering","general"} and len(words) <= 15 and hits == 0:
        return 7.0, "no explicit constraints needed for the simple request"
    return _clamp(5 + 1.2*hits), f"{hits} explicit constraint signal(s)"

def _ambiguity(raw, text, words):
    vague, filler = find_vague(text), find_filler(text)
    hedges = [h for h in HEDGES if h in raw.lower()]
    penalty = 1.2*len(vague) + 0.8*len(filler) + 1.2*len(hedges)
    issues = []
    if vague: issues.append("vague wording")
    if filler: issues.append("filler")
    if hedges: issues.append("hedging")
    if len(words) < 3: penalty += 1
    return _clamp(9.0 - penalty), "; ".join(issues) or "no significant ambiguity/noise detected"

def heuristic_scores(raw, text, task):
    words, sents = tokenize(text), split_sentences(text)
    vals = {
        "clarity": _clarity(text, words, sents),
        "specificity": _specificity(text, words, sents),
        "context": _context(text, words, sents, task),
        "task_definition": _task_definition(text, words, task),
        "constraints": _constraints(text, words, task),
        "ambiguity_noise": _ambiguity(raw, text, words),
    }
    return {k: DimensionScore(k, s, r) for k,(s,r) in vals.items()}

def llm_scores(llm: LLMClient, text: str) -> Optional[dict]:
    prompt = """Evaluate this user prompt, not an answer to it. Score six dimensions 0-10.
Do not penalize a prompt for missing optional details when the task can be answered correctly
without them. In particular, simple factual questions can score well without context or explicit
format constraints. Return JSON only.
{"clarity":{"score":0,"rationale":""},"specificity":{"score":0,"rationale":""},
"context":{"score":0,"rationale":""},"task_definition":{"score":0,"rationale":""},
"constraints":{"score":0,"rationale":""},"ambiguity_noise":{"score":0,"rationale":""}}

PROMPT:
""" + text
    data = extract_json(llm.complete(prompt, system="You are a strict but fair prompt-quality evaluator."))
    try:
        return {d: DimensionScore(d, _clamp(float(data[d]["score"])), str(data[d].get("rationale","")))
                for d in DIMENSIONS}
    except (TypeError, KeyError, ValueError):
        return None

def quality_score(scores):
    return round(10 * sum(WEIGHTS[d] * scores[d].score for d in DIMENSIONS), 1)

def analyze(prompt, llm=None, threshold=6.0):
    normalized = normalize(prompt)
    task = detect_task(normalized)
    scores = (llm_scores(llm, normalized) if llm else None) or heuristic_scores(prompt, normalized, task)
    weak = sorted((d for d in DIMENSIONS if scores[d].score < threshold), key=lambda d:scores[d].score)
    weaknesses = [f"{d} ({scores[d].score}/10): {scores[d].rationale}" for d in weak]
    return Analysis(prompt, normalized, task, scores, quality_score(scores), weak, weaknesses)
