"""Optional same-model output comparison."""
from __future__ import annotations
import re
from .llm import LLMClient, extract_json
from .models import Comparison, OutputEvaluation
from .preprocess import tokenize

CRITERIA=("relevance","completeness","structure","constraint_adherence")

class HeuristicJudge:
    def score(self,intent,prompt_used,output):
        kw={w.lower() for w in tokenize(intent)}-{"the","a","an","of","to","and","in","for","on","with","is","are","it","this","that"}
        out={w.lower() for w in tokenize(output)}
        overlap=len(kw&out)/max(1,min(len(kw),8))
        n=len(tokenize(output))
        structure=min(10,4+2*bool(re.search(r"^\s*[-*\d]+[.)]?\s",output,re.M))+2*bool(re.search(r"^#+\s",output,re.M))+1*("\n\n" in output))
        limit=re.search(r"(\d+)\s*words",prompt_used,re.I)
        adherence=10 if limit and n<=int(limit.group(1)) else 7
        return {"relevance":min(10,10*overlap),"completeness":min(10,n/12),"structure":structure,"constraint_adherence":adherence}

class LLMJudge:
    def __init__(self,llm): self.llm,self.fallback=llm,HeuristicJudge()
    def score(self,intent,prompt_used,output):
        p=f"""Compare this answer to the user's ORIGINAL request. Score 0-10:
relevance, completeness, structure, constraint_adherence.
Do not reward verbosity by itself. Judge against the original intent.
Return JSON only with those four numeric fields.

ORIGINAL REQUEST:
{intent}

PROMPT SENT TO MODEL:
{prompt_used}

ANSWER:
{output}
"""
        try:
            data=extract_json(self.llm.complete(p,system="You are an impartial evaluator."))
            return {c:max(0,min(10,float(data[c]))) for c in CRITERIA}
        except Exception:
            return self.fallback.score(intent,prompt_used,output)

def _evaluate(label,llm,judge,intent,prompt):
    output=llm.complete(prompt)
    scores={k:round(v,1) for k,v in judge.score(intent,prompt,output).items()}
    return OutputEvaluation(label,output,scores,round(sum(scores.values())/len(scores),2))

def compare(llm,original,refined,judge=None,margin=.25):
    judge=judge or HeuristicJudge()
    a=_evaluate("original",llm,judge,original,original)
    b=_evaluate("refined",llm,judge,original,refined)
    delta=round(b.overall-a.overall,2)
    winner="refined" if delta>margin else "original" if delta<-margin else "tie"
    return Comparison(a,b,delta,winner)
