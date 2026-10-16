"""LLM client abstraction."""
from __future__ import annotations
import json, os, re
from typing import Optional, Protocol

class LLMClient(Protocol):
    def complete(self,prompt:str,system:Optional[str]=None,max_tokens:int=1024)->str: ...

class MockLLM:
    def complete(self,prompt,system=None,max_tokens=1024):
        return "Example answer based on the supplied prompt."

def extract_json(text:str)->Optional[dict]:
    match=re.search(r"\{.*\}",text,re.S)
    if not match: return None
    try: return json.loads(match.group(0))
    except json.JSONDecodeError: return None
