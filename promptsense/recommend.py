"""Targeted, task-aware recommendations."""
from .models import Analysis, Recommendation
from .scoring import find_vague, find_filler

ADVICE = {
    "clarity": "Make the request direct and unambiguous.",
    "specificity": "Add concrete details only where they materially affect the desired answer.",
    "context": "Add audience, background, goal, or source context only when it changes the expected answer.",
    "task_definition": "State the requested action or deliverable explicitly.",
    "constraints": "Add useful length, format, tone, language, or must/must-not requirements when relevant.",
    "ambiguity_noise": "Remove vague filler, hedging, or unresolved references.",
}

TASK_TIPS = {
    "code_generation": "If relevant, specify language/version, inputs, expected output, and important edge cases.",
    "summarization": "Specify the source, target audience, and desired length if they matter.",
    "translation": "Specify source/target language and register if not already clear.",
    "creative_writing": "Specify genre, tone, length, audience, and required elements when important.",
    "analysis": "Specify evaluation criteria and the desired decision/output.",
    "extraction": "Specify the fields and output schema.",
    "classification": "Specify allowed labels and handling of borderline cases.",
}

def recommend(analysis: Analysis):
    recs=[]
    for dim in analysis.weak_dimensions:
        advice=ADVICE[dim]
        if dim=="ambiguity_noise":
            bad=find_vague(analysis.normalized)+find_filler(analysis.normalized)
            if bad: advice += f" Detected: {', '.join(sorted(set(bad)))}."
        recs.append(Recommendation(dim, advice))
    tip=TASK_TIPS.get(analysis.task)
    if tip and analysis.weak_dimensions:
        recs.append(Recommendation("task_specific", tip))
    return recs
