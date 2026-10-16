# PromptSense Studio

Type a simple prompt. The app sends it to an LLM **and** to PromptSense:

| Left panel (your prompt) | Right panel (PromptSense refined) |
|---|---|
| LLM output | Suggested refined prompt |
| PQS (0-100) + 6 dimension scores, weak spots, recommendations | LLM output for the refined prompt |
| Output score (relevance, completeness, structure, constraint adherence) | Same PQS + output score, so you can compare |

## Run
pip install -r requirements.txt
set GEMINI_API_KEY=...        # Gemini (keep your existing Gemini setup)
python app.py                    # open http://127.0.0.1:5000

## Notes
- `promptsense/` is your package, unchanged. `app.py` only adds a Gemini adapter and the web layer.
- PQS uses PromptSense's heuristic scorer (fast, deterministic, as in your experiments).
- Both answers are scored against the **original** request (the notebook's ground-truth principle).
  This app scores each answer on its own; it is not the blind A/B judge used in the notebook,
  so use the notebook for research numbers.
- If the prompt is already good, PromptSense leaves it unchanged and the app says so.
- Default Gemini model is the one in your notebook; change it in the form or via `GEMINI_MODEL`.

## NVIDIA Nemotron

PromptSense also supports NVIDIA Nemotron 3.5 Lightning through OpenRouter.

Windows PowerShell:
```powershell
$env:OPENROUTER_API_KEY="your_openrouter_key"
$env:PROMPTSENSE_PROVIDER="nemotron"
```

Or put this in `.env` next to `app.py`:
```text
OPENROUTER_API_KEY=your_openrouter_key
PROMPTSENSE_PROVIDER=nemotron
NEMOTRON_MODEL=nvidia/nemotron-3.5-lightning
```

Gemini remains available and is the default provider.
