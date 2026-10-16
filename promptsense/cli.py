from __future__ import annotations
import argparse,json,sys
from .llm import MockLLM
from .models import DIMENSIONS,Report
from .pipeline import PromptSense

def format_report(r:Report):
    a=r.analysis
    lines=[f"Task detected : {a.task}",f"Quality score : {a.quality_score}/100 -> refined {r.refined_quality_score}/100","",
           "Dimension scores"]
    lines += [f"  {d:<18} {a.scores[d].score:>4}  {a.scores[d].rationale}" for d in DIMENSIONS]
    lines += ["","Weak dimensions"] + ([f"  - {w}" for w in a.weaknesses] or ["  none"])
    lines += ["","Refined prompt","-"*60,r.refined_prompt,"-"*60]
    return "\n".join(lines)

def main(argv=None):
    p=argparse.ArgumentParser(prog="promptsense")
    p.add_argument("prompt",nargs="?")
    p.add_argument("-f","--file")
    p.add_argument("--llm",choices=["none","mock"],default="none")
    p.add_argument("--model")
    p.add_argument("--evaluate",action="store_true")
    p.add_argument("--llm-scoring",action="store_true")
    p.add_argument("--llm-refiner",action="store_true")
    p.add_argument("--llm-judge",action="store_true")
    p.add_argument("--threshold",type=float,default=6.0)
    p.add_argument("--json",action="store_true")
    args=p.parse_args(argv)
    text=open(args.file,encoding="utf-8").read() if args.file else args.prompt or sys.stdin.read()
    llm=MockLLM() if args.llm=="mock" else None
    report=PromptSense(llm,args.llm_scoring,args.llm_refiner,args.llm_judge,args.threshold).run(text,args.evaluate)
    print(json.dumps(report.to_dict(),indent=2) if args.json else format_report(report))
    return 0
