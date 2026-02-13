"""
Local Article Reader with Contextual LLM Word Explanation
Backend: FastAPI + LLM API
"""

from dotenv import load_dotenv

load_dotenv()

import csv
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# LLM setup - configurable via env
# Gemini (preferred): GEMINI_API_KEY or GOOGLE_API_KEY
# OpenAI: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not GEMINI_API_KEY and not OPENAI_API_KEY and not OPENAI_BASE_URL:
        print("Warning: No GEMINI_API_KEY, OPENAI_API_KEY, or OPENAI_BASE_URL set. LLM explanations will fail.")
    yield


app = FastAPI(title="Article Reader", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# In-memory cache for explanations (optional enhancement)
_explanation_cache: dict[str, str] = {}


VOCABULARY_CSV = Path(__file__).parent / "vocabulary.csv"
VOCAB_HEADERS = ["date", "word", "concise_meaning", "importance"]


def _read_vocab_rows() -> list[dict]:
    """Read all vocabulary rows, ensuring standard columns exist."""
    if not VOCABULARY_CSV.exists():
        return []
    rows = []
    with open(VOCABULARY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = {h: row.get(h, "") for h in VOCAB_HEADERS}
            if r.get("word") or r.get("concise_meaning"):
                rows.append(r)
    return rows


def _write_vocab_rows(rows: list[dict]) -> None:
    """Write all vocabulary rows to CSV."""
    with open(VOCABULARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VOCAB_HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class ExplainRequest(BaseModel):
    selected_text: str
    full_sentence: str


class VocabularyRequest(BaseModel):
    word: str
    concise_meaning: str
    importance: str = ""


class VocabularyUpdateRequest(BaseModel):
    word: str | None = None
    concise_meaning: str | None = None
    importance: str | None = None


def extract_concise_meaning(explanation: str) -> str:
    """Extract the concise meaning from the structured explanation."""
    match = re.search(r"1\.\s*\*\*Meaning\*\*:\s*(.+?)(?=\n\s*2\.|$)", explanation, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def get_llm_explanation(selected_text: str, full_sentence: str) -> str:
    """Call LLM API for contextual explanation."""
    cache_key = f"{selected_text}|{full_sentence}"
    if cache_key in _explanation_cache:
        return _explanation_cache[cache_key]

    prompt = """You are a language tutor. Explain the selected word or phrase in the specific sentence context.

Use this exact format (use **bold** for keywords):

1. **Meaning**: Very concise definition (a few words only)
2. **Meaning in this sentence**: 1-2 sentences explaining how it's used in the given context
3. **Examples**: 1 example sentence
4. **Expressions**: 1-2 common expressions or collocations using this word

Selected text: "{selected_text}"
Full sentence: "{full_sentence}"

Your response:""".format(
        selected_text=selected_text, full_sentence=full_sentence
    )

    try:
        if GEMINI_API_KEY:
            from google import genai

            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            explanation = response.text.strip()
        else:
            if not OPENAI_API_KEY and not OPENAI_BASE_URL:
                return "Error: No LLM configured. Set GEMINI_API_KEY (for Gemini) or OPENAI_API_KEY (for OpenAI). See README for setup."

            try:
                from openai import OpenAI
            except ImportError:
                return "Error: OpenAI backend requires 'pip install openai'. Or set GEMINI_API_KEY to use Gemini instead."

            client_kwargs = {"api_key": OPENAI_API_KEY or "not-needed"}
            if OPENAI_BASE_URL:
                client_kwargs["base_url"] = OPENAI_BASE_URL

            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            explanation = response.choices[0].message.content.strip()

        _explanation_cache[cache_key] = explanation
        return explanation
    except Exception as e:
        return f"Error: {str(e)}"


@app.post("/api/explain")
def explain(request: ExplainRequest):
    """Get contextual explanation for selected text."""
    explanation = get_llm_explanation(request.selected_text, request.full_sentence)
    concise_meaning = extract_concise_meaning(explanation)
    return {"explanation": explanation, "concise_meaning": concise_meaning}


@app.get("/api/vocabulary")
def get_vocabulary():
    """Return vocabulary CSV data as JSON."""
    entries = _read_vocab_rows()
    return {"entries": entries}


@app.get("/api/vocabulary/summary")
def get_vocabulary_summary():
    """Return word counts by date for the bar chart."""
    rows = _read_vocab_rows()
    counts = Counter()
    for row in rows:
        if row.get("date"):
            counts[row["date"]] += 1
    by_date = dict(sorted(counts.items()))
    return {"by_date": by_date}


@app.post("/api/vocabulary")
def add_to_vocabulary(request: VocabularyRequest):
    """Append a word to the vocabulary CSV file."""
    rows = _read_vocab_rows()
    rows.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "word": request.word.strip(),
        "concise_meaning": request.concise_meaning.strip(),
        "importance": (request.importance or "").strip(),
    })
    _write_vocab_rows(rows)
    return {"status": "added"}


@app.put("/api/vocabulary/{index:int}")
def update_vocabulary(index: int, request: VocabularyUpdateRequest):
    """Update a vocabulary row by index."""
    rows = _read_vocab_rows()
    if index < 0 or index >= len(rows):
        raise HTTPException(status_code=404, detail="Invalid index")
    row = rows[index]
    if request.word is not None:
        row["word"] = request.word.strip()
    if request.concise_meaning is not None:
        row["concise_meaning"] = request.concise_meaning.strip()
    if request.importance is not None:
        row["importance"] = str(request.importance).strip()
    _write_vocab_rows(rows)
    return {"status": "updated"}


@app.delete("/api/vocabulary/{index:int}")
def delete_vocabulary(index: int):
    """Delete a vocabulary row by index."""
    rows = _read_vocab_rows()
    if index < 0 or index >= len(rows):
        raise HTTPException(status_code=404, detail="Invalid index")
    rows.pop(index)
    _write_vocab_rows(rows)
    return {"status": "deleted"}


@app.get("/")
def index():
    """Serve the main page."""
    static_dir = Path(__file__).parent / "static"
    return FileResponse(static_dir / "index.html")


@app.get("/vocabulary")
def vocabulary_page():
    """Serve the vocabulary view page."""
    static_dir = Path(__file__).parent / "static"
    return FileResponse(static_dir / "vocabulary.html")


@app.get("/summary")
def summary_page():
    """Serve the vocabulary summary page."""
    static_dir = Path(__file__).parent / "static"
    return FileResponse(static_dir / "summary.html")
