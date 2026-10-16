# PromptSense Studio

**PromptSense Studio** is a full-stack, research-grade prompt intelligence, refinement, and benchmarking suite. It analyzes user prompts, computes a deterministic **Prompt Quality Score (PQS, 0–100)**, recommends targeted improvements, refines the prompt, and evaluates model outputs side-by-side using an impartial blind A/B judge.

---

## Features

- **Prompt Quality Scoring (PQS, 0–100)**: Instant heuristic evaluation across 6 core prompt engineering dimensions without requiring slow or expensive LLM calls:
  - **Clarity**: Directness, readability, and sentence flow.
  - **Specificity**: Concrete details, parameters, and entities.
  - **Context**: Domain background, persona, and situational framing.
  - **Task Definition**: Actionable verbs and clear deliverable definition.
  - **Constraints**: Format, length, styling, and behavioral rules.
  - **Low Ambiguity**: Absence of vague qualifiers or conflicting instructions.
- **Intent-Preserving Refinement**: Automatically strengthens weak dimensions while strictly preserving your ground-truth intent (no hallucinated requirements or altered objectives).
- **Side-by-Side Model Execution**: Executes both the original prompt and the refined prompt in parallel using high-performance LLMs.
- **Blind A/B LLM Judge**: Evaluates both generated answers against your original request using randomized positions and explicit bias controls (no length or formatting bias) across five criteria:
  - *Relevance*, *Clarity*, *Completeness*, *Instruction Adherence*, and *Overall Quality* (1–5 scale).
- **Multi-Model & Multi-Provider Support**:
  - **Google Gemini** (via `google-genai` SDK, defaults to `gemini-3.5-flash-lite`).
  - **Qwen 3.8 27B** (via OpenRouter).
  - **NVIDIA Nemotron 3 Ultra** (via OpenRouter).

---

## Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/princeap7/PromptSense.git
cd PromptSense
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)

Create a `.env` file in the project root directory (next to `app.py`):

```env
# Google Gemini (from https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=paste key here
GEMINI_MODEL=gemini-3.5-flash-lite

# OpenRouter (for Qwen and NVIDIA Nemotron)
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key-here
PROMPTSENSE_PROVIDER=gemini
QWEN_MODEL=qwen/qwen3.8-27b
NEMOTRON_MODEL=nvidia/nemotron-3-ultra-550b-a55b
```

> **Note**: `.env` is included in `.gitignore`. Never commit API keys to version control.

### 3. Launch PromptSense Studio

```bash
python app.py
```

Open your browser at **`http://127.0.0.1:5000`**.

---

## CLI Usage

You can also run PromptSense directly from your terminal:

```bash
# Analyze a prompt with the heuristic scorer
python -m promptsense "write a python function for sorting"

# Output structured JSON
python -m promptsense "summarize this quarterly report" --json

# Run refinement with an LLM
python -m promptsense "explain quantum computing" --llm-refiner
```

---

## Architecture & Pipeline

```text
User Input
    │
    ▼
[1. Text Preprocessing] ──► NFKC normalization, whitespace & character cleaning
    │
    ▼
[2. Task Classification] ──► Detects task intent (coding, explanation, summarization, etc.)
    │
    ▼
[3. Heuristic PQS Engine] ──► Deterministic 0–100 score + 6 dimension diagnostics
    │
    ▼
[4. Recommendations] ──► Identifies weak dimensions & generates targeted advice
    │
    ▼
[5. Intent Refinement] ──► LLM expands weak spots while keeping core task intact
    │
    ▼
[6. Parallel Execution] ──► Generates outputs for Original & Refined prompts in parallel
    │
    ▼
[7. Blind A/B Judge] ──► Impartial scoring (1–5) strictly against original intent
```

---

## Project Structure

```text
├── app.py                      # Flask web application & LLM adapters (Gemini, OpenRouter)
├── judge.py                    # Blind A/B randomized output evaluator & rubric
├── requirements.txt            # Python package dependencies
├── .env                        # Local environment variables & API keys (gitignored)
├── .gitignore                  # Git ignore rules for keys, bytecode, and system files
├── PROJECT_DOCUMENTATION.txt   # Detailed research and architecture specification
│
├── promptsense/                # Core PromptSense engine package
│   ├── cli.py                  # Command-line interface
│   ├── evaluator.py            # Heuristic output evaluator
│   ├── llm.py                  # LLM abstractions & JSON parser
│   ├── models.py               # Pydantic / dataclass analysis models
│   ├── pipeline.py             # PromptSense execution orchestrator
│   ├── preprocess.py           # Text cleaning & normalization
│   ├── recommend.py            # Diagnostic advice generator
│   ├── refine.py               # Rule-based & LLM-driven prompt refiners
│   ├── scoring.py              # 6-dimension heuristic scoring algorithms
│   └── task.py                 # Task category classification
│
└── static/
    ├── index.html              # Responsive side-by-side dashboard UI
    └── other.html              # Alternative comparison layout
```

---

## Security Best Practices

- Store API keys exclusively in `.env` or pass them via runtime environment variables.
- Never hardcode API keys into `.py` or commit them to Git.
- Google AI Studio automatically revokes leaked keys if pushed to public repositories.

---
