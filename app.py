"""PromptSense Studio: original prompt vs. PromptSense-refined prompt, side by side."""
import os
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, request, send_from_directory

from judge import judge_pair
from promptsense import PromptSense
from promptsense.scoring import analyze


def _load_dotenv(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    """Tiny .env reader: KEY=value lines; .env values take precedence."""
    if os.path.exists(path):
        for line in open(path, encoding="utf-8-sig"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


_load_dotenv()
app = Flask(__name__, static_folder="static")
ANSWER_TOKENS = 10000  # same output cap as the research notebook
SHOW_OUTPUTS = True  # False = prompt-quality analysis only (no answer/judge calls). Set True for outputs + scores.


class GeminiClient:
    """Adapter so Gemini satisfies PromptSense's LLMClient protocol."""
    def __init__(self, model=None, api_key=None):
        from google import genai
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key or key.startswith("AQ."):
            raise RuntimeError(
                "Gemini API key is missing or invalid. "
                "Please create a valid API key at https://aistudio.google.com/app/apikey "
                "and set GEMINI_API_KEY=your_key in your .env file (or switch to Qwen / Nemotron)."
            )
        self._client = genai.Client(api_key=key)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    def complete(self, prompt, system=None, max_tokens=1024):
        from google.genai import types
        cfg = types.GenerateContentConfig(temperature=0, max_output_tokens=max_tokens,
                                          system_instruction=system)
        r = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        return (r.text or "").strip()


class DemoLLM:
    """No-key mode for trying the UI."""
    model = "demo"
    def complete(self, prompt, system=None, max_tokens=1024):
        return f"[Demo mode: no real model was called]\n\nThis is where the model's answer to the following prompt would appear:\n\n{prompt[:400]}"


class OpenRouterClient:
    """Base client for models accessed via OpenRouter's Chat Completions API."""
    def __init__(self, model, api_key=None, name="OpenRouter"):
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("Install requests with: pip install requests") from e

        self._requests = requests
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                f"No API key found for {name}. Put {name.upper()}_API_KEY=your_key or "
                "OPENROUTER_API_KEY=your_key in a .env file next to app.py, then restart."
            )
        self.model = model
        self.name = name

    def complete(self, prompt, system=None, max_tokens=1024):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        token_budget = max(max_tokens, 2048)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": token_budget,
            "reasoning": {"effort": "none"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = self._requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )

        # Retry without reasoning parameter if the provider/model rejects it
        if response.status_code == 400 and "reasoning" in response.text.lower():
            payload.pop("reasoning", None)
            response = self._requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )

        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise RuntimeError("OpenRouter returned an invalid JSON response.")

        if not response.ok:
            error = data.get("error", {})
            raise RuntimeError(
                f"OpenRouter {response.status_code}: "
                f"{error.get('message', response.text)}"
            )

        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            raise RuntimeError(f"{self.name} returned no final answer.")
        return content.strip()


class NemotronClient(OpenRouterClient):
    """NVIDIA Nemotron 3 Ultra adapter using OpenRouter's Chat Completions API."""
    def __init__(self, model=None, api_key=None):
        default_model = model or os.getenv(
            "NEMOTRON_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"
        )
        key = (
            api_key
            or os.getenv("NEMOTRON_API_KEY")
            or os.getenv("OPENROUTER_NEMOTRON_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
        )
        super().__init__(model=default_model, api_key=key, name="Nemotron")


class QwenClient(OpenRouterClient):
    """Qwen (Qwen3.8 27B) adapter using OpenRouter's Chat Completions API."""
    def __init__(self, model=None, api_key=None):
        default_model = model or os.getenv(
            "QWEN_MODEL", "qwen/qwen3.8-27b"
        )
        key = (
            api_key
            or os.getenv("QWEN_API_KEY")
            or os.getenv("OPENROUTER_QWEN_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
        )
        super().__init__(model=default_model, api_key=key, name="Qwen")


def make_llm(provider, model, key):
    p = (provider or "").lower().strip()
    if p == "gemini":
        return GeminiClient(model or None, key or None)
    if p in ("nemotron", "nemotron-ultra", "nemotron3", "nvidia/nemotron-3-ultra-550b-a55b"):
        return NemotronClient(model or None, key or None)
    if p in ("qwen", "qwen3.8", "qwen/qwen3.8-27b"):
        return QwenClient(model or None, key or None)
    return DemoLLM()


def pack(a):
    return {"task": a.task, "pqs": a.quality_score, "weak": a.weak_dimensions,
            "scores": {k: {"score": v.score, "rationale": v.rationale} for k, v in a.scores.items()}}


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/run")
def run():
    d = request.get_json(force=True)
    prompt = (d.get("prompt") or "").strip()
    provider = (d.get("provider") or os.getenv("PROMPTSENSE_PROVIDER", "gemini")).lower()  # gemini | nemotron | qwen | demo
    if not prompt:
        return jsonify(error="Please enter a prompt."), 400
    try:
        llm = make_llm(provider, d.get("model") or "", d.get("key") or "")
        real = provider != "demo"
        # 1) PromptSense: heuristic PQS + recommendations + refined prompt (LLM refinement always on)
        report = PromptSense(llm, use_llm_refiner=real).run(prompt)
        orig, refined = report.analysis.normalized, report.refined_prompt
        same = refined.strip() == orig.strip()
        ref_analysis = analyze(refined)

        out_o = out_r = ev_o = ev_r = None
        reason, delta, winner = "", 0, "tie"
        if SHOW_OUTPUTS:
            # 2) Answer both prompts in parallel (reuse the answer if the prompt is unchanged)
            with ThreadPoolExecutor(2) as ex:
                fo = ex.submit(llm.complete, orig, None, ANSWER_TOKENS)
                fr = fo if same else ex.submit(llm.complete, refined, None, ANSWER_TOKENS)
                out_o, out_r = fo.result(), fr.result()
            # 3) Blind A/B judge (notebook rubric, 1-5), against the ORIGINAL intent
            ev_o, ev_r, reason = judge_pair(llm if real else None, orig, report.analysis.task, out_o, out_r)
            if same:
                ev_r = ev_o
            delta = round(ev_r["overall"] - ev_o["overall"], 2)
            winner = "tie" if same else "refined" if delta > 0 else "original" if delta < 0 else "tie"

        return jsonify(
            model=getattr(llm, "model", provider), outputs=SHOW_OUTPUTS, unchanged=same, winner=winner, delta=delta,
            judge="llm" if real else "demo", reason=reason,
            recommendations=[{"dimension": r.dimension, "advice": r.advice} for r in report.recommendations],
            original={"prompt": orig, "output": out_o, "analysis": pack(report.analysis), "eval": ev_o},
            refined={"prompt": refined, "output": out_r, "analysis": pack(ref_analysis), "eval": ev_r})
    except Exception as e:  # surface API/key errors in the UI
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)