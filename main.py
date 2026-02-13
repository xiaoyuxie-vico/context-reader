"""
Local Article Reader with Contextual LLM Word Explanation
Backend: FastAPI + LLM API
"""

from dotenv import load_dotenv

load_dotenv()

import csv
import hashlib
import json
import random
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from ebooklib import epub, ITEM_DOCUMENT
    from bs4 import BeautifulSoup
    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False

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
READING_POSITIONS_JSON = Path(__file__).parent / "reading_positions.json"
VOCAB_HEADERS = ["date", "word", "concise_meaning", "importance", "status"]


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
    status: str = ""


class VocabularyUpdateRequest(BaseModel):
    word: str | None = None
    concise_meaning: str | None = None
    importance: str | None = None
    status: str | None = None


class ReadingPositionRequest(BaseModel):
    book_id: str
    position: int
    filename: str = ""


def _html_to_plain_text(html: bytes) -> str:
    """Convert HTML to plain text with proper paragraph breaks."""
    soup = BeautifulSoup(html, "html.parser")

    # Replace <br> with newline
    for tag in soup.find_all("br"):
        tag.replace_with("\n")

    # Replace block elements: innermost first so nested structure is preserved
    block_tags = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"]
    for tag in reversed(soup.find_all(block_tags)):
        # Preserve newlines from nested blocks; use space only within inline content
        tag.replace_with(tag.get_text(separator=" ") + "\n\n")

    # Get remaining text with spaces between inline elements (no random line breaks)
    text = soup.get_text(separator=" ")
    # Normalize: collapse multiple spaces, collapse 3+ newlines to 2
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_text_from_epub(file_content: bytes) -> str:
    """Extract plain text from EPUB file content with proper formatting."""
    if not EPUB_AVAILABLE:
        raise RuntimeError("ebooklib and beautifulsoup4 are required for EPUB support. Run: pip install ebooklib beautifulsoup4")
    import io
    book = epub.read_epub(io.BytesIO(file_content))
    parts = []
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            text = _html_to_plain_text(item.get_content())
            if text:
                parts.append(text)
    return "\n\n".join(parts) if parts else ""


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

            from openai import OpenAI

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


def _load_reading_positions() -> dict:
    """Load saved reading positions."""
    if not READING_POSITIONS_JSON.exists():
        return {}
    try:
        with open(READING_POSITIONS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_reading_position(book_id: str, position: int, filename: str = "") -> None:
    """Save reading position for a book."""
    data = _load_reading_positions()
    data[book_id] = {"position": position, "filename": filename}
    with open(READING_POSITIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@app.post("/api/import-epub")
async def import_epub(file: UploadFile):
    """Extract text from uploaded EPUB file."""
    if not file.filename or not file.filename.lower().endswith(".epub"):
        raise HTTPException(status_code=400, detail="Please upload an .epub file")
    if not EPUB_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="EPUB support not available. Run: pip install ebooklib beautifulsoup4",
        )
    try:
        content = await file.read()
        book_id = hashlib.sha256(content).hexdigest()
        text = _extract_text_from_epub(content)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content found in the EPUB file")
        return {"text": text, "filename": file.filename, "book_id": book_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read EPUB: {str(e)}")


@app.get("/api/reading-position")
def get_reading_position(book_id: str):
    """Get saved reading position for a book."""
    data = _load_reading_positions()
    entry = data.get(book_id)
    if not entry:
        return {"position": None}
    return {"position": entry.get("position", 0)}


@app.post("/api/reading-position")
def save_reading_position(request: ReadingPositionRequest):
    """Save reading position for a book."""
    _save_reading_position(request.book_id, request.position, request.filename)
    return {"status": "saved"}


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
        "status": (request.status or "").strip(),
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
    if request.status is not None:
        row["status"] = str(request.status).strip()
    _write_vocab_rows(rows)
    return {"status": "updated"}


@app.get("/api/quiz")
def get_quiz_words(importance: str = "", days: int | None = None):
    """Get 10 random vocabulary items for quiz, filtered by importance and date range."""
    rows = _read_vocab_rows()
    now = datetime.now()

    filtered = []
    for i, row in enumerate(rows):
        if importance and (row.get("importance") or "") != importance:
            continue
        if days is not None:
            try:
                d = datetime.strptime(row.get("date", ""), "%Y-%m-%d")
                if (now - d).days > days:
                    continue
            except ValueError:
                continue
        if row.get("word") or row.get("concise_meaning"):
            filtered.append((i, row))

    sample = random.sample(filtered, min(10, len(filtered)))
    return {"items": [{"index": idx, **row} for idx, row in sample]}


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


@app.get("/quiz")
def quiz_page():
    """Serve the quiz page."""
    static_dir = Path(__file__).parent / "static"
    return FileResponse(static_dir / "quiz.html")
