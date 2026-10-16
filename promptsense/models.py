"""Core PromptSense data structures."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional

DIMENSIONS = (
    "clarity",
    "specificity",
    "context",
    "task_definition",
    "constraints",
    "ambiguity_noise",
)

# Weights sum to 1.0. PQS is reported on a 0-100 scale.
WEIGHTS = {
    "clarity": 0.20,
    "specificity": 0.20,
    "context": 0.15,
    "task_definition": 0.20,
    "constraints": 0.15,
    "ambiguity_noise": 0.10,
}

@dataclass
class DimensionScore:
    name: str
    score: float
    rationale: str

@dataclass
class Analysis:
    original: str
    normalized: str
    task: str
    scores: dict
    quality_score: float
    weak_dimensions: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)

@dataclass
class Recommendation:
    dimension: str
    advice: str

@dataclass
class OutputEvaluation:
    label: str
    output: str
    scores: dict
    overall: float

@dataclass
class Comparison:
    original: OutputEvaluation
    refined: OutputEvaluation
    delta: float
    winner: str

@dataclass
class Report:
    analysis: Analysis
    recommendations: list
    refined_prompt: str
    refined_quality_score: float
    comparison: Optional[Comparison] = None

    def to_dict(self) -> dict:
        return asdict(self)
