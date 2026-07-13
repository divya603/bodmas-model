"""LLM task-analog of the BODMAS misconception-detection judgment task.

Standalone package (no runtime imports from the parent base-task/ or src/ trees).
Adapted from the numberlink LLM pipeline: the OpenRouter client, session layer,
seeded per-subject sampler, and JSONL->human-schema parser carry over; the board
encoder is replaced by a text-trace prompt builder (our traces are already text),
the `encoding` factor is dropped, and the response is the human 6-point Likert
rating (1 = Strongly Disagree ... 6 = Strongly Agree) instead of yes/no+confidence.
"""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
STIMULUS_POOL_PATH = DATA_DIR / "stimulus_pool.json"

# Best-effort: load llm_exp/.env so OPENROUTER_API_KEY is available without exporting it.
_ENV = PROJECT_DIR / ".env"
if _ENV.exists():
    try:
        from dotenv import load_dotenv  # optional dependency
        load_dotenv(_ENV)
    except Exception:  # minimal fallback if python-dotenv isn't installed
        for _line in _ENV.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
