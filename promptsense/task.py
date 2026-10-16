"""Conservative task detection for PromptSense."""
import re

TASK_PATTERNS = [
    ("creative_writing", r"\b(write|create|draft)\b.*\b(story|poem|essay|blog|article|slogan|lyrics|post)\b"),
    ("code_generation", r"\b(write|create|make|generate|implement|build|code|program|function|script|class|api|debug|fix|refactor)\b.*\b(python|java|javascript|c\+\+|c#|sql|code|program|function|script|algorithm)"),
    ("summarization", r"\b(summarize|summarise|summary|tl;?dr|condense|shorten|recap)\b"),
    ("translation", r"\b(translate|translation)\b"),
    ("classification", r"\b(classify|categorize|categorise|label|sentiment|tag)\b"),
    ("extraction", r"\b(extract|parse|pull out|identify all|list all)\b"),
    ("analysis", r"\b(analyze|analyse|compare|evaluate|review|critique|assess|pros and cons)\b"),
    ("planning", r"\b(plan|schedule|roadmap|strategy|itinerary)\b"),
    ("explanation", r"\b(explain|what is|what are|why|how does|how do|describe|teach|eli5|tell me about)\b"),
    ("question_answering", r"\?\s*$"),
]

def detect_task(text: str) -> str:
    lowered = text.lower().strip()
    for task, pattern in TASK_PATTERNS:
        if re.search(pattern, lowered, re.S):
            return task
    return "general"