"""PromptSense orchestration."""
from __future__ import annotations
from typing import Optional
from .evaluator import LLMJudge, compare
from .llm import LLMClient
from .models import Report
from .recommend import recommend
from .refine import refine
from .scoring import analyze

class PromptSense:
    def __init__(self,llm:Optional[LLMClient]=None,use_llm_scoring=False,
                 use_llm_refiner=False,use_llm_judge=False,threshold=6.0):
        self.llm=llm
        self.threshold=threshold
        self.use_llm_scoring=use_llm_scoring and llm is not None
        self.use_llm_refiner=use_llm_refiner and llm is not None
        self.use_llm_judge=use_llm_judge and llm is not None

    def run(self,prompt:str,evaluate=False)->Report:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be non-empty.")
        scorer=self.llm if self.use_llm_scoring else None
        analysis=analyze(prompt,scorer,self.threshold)
        recs=recommend(analysis)
        refined=refine(analysis,self.llm if self.use_llm_refiner else None,recs)
        refined_score=analyze(refined,scorer,self.threshold).quality_score
        comparison=None
        if evaluate:
            if self.llm is None: raise ValueError("evaluate=True requires an LLM client.")
            judge=LLMJudge(self.llm) if self.use_llm_judge else None
            comparison=compare(self.llm,analysis.normalized,refined,judge)
        return Report(analysis,recs,refined,refined_score,comparison)
