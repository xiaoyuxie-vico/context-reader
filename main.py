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
import sys
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

_PDF_IMPORT_ERROR: str | None = None
try:
    import fitz  # PyMuPDF (pip package name: pymupdf)
    PDF_AVAILABLE = True
except Exception as e:
    PDF_AVAILABLE = False
    _PDF_IMPORT_ERROR = f"{type(e).__name__}: {e}"

# LLM setup - configurable via env
# Gemini (preferred): GEMINI_API_KEY or GOOGLE_API_KEY
# OpenAI: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not GEMINI_API_KEY and not OPENAI_API_KEY and not OPENAI_BASE_URL:
        print("Warning: No GEMINI_API_KEY, OPENAI_API_KEY, or OPENAI_BASE_URL set. LLM explanations will fail.")
    if not PDF_AVAILABLE:
        print(
            f"Warning: PDF import failed ({_PDF_IMPORT_ERROR}). "
            f"Install with: {sys.executable} -m pip install pymupdf"
        )
    yield


app = FastAPI(title="Article Reader", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# In-memory cache for explanations (optional enhancement)
_explanation_cache: dict[str, str] = {}


VOCABULARY_CSV = Path(__file__).parent / "vocabulary.csv"
READING_POSITIONS_JSON = Path(__file__).parent / "reading_positions.json"
READING_HISTORY_JSON = Path(__file__).parent / "reading_history.json"
CACHED_CONTENT_DIR = Path(__file__).parent / "cached_content"
USAGE_STATS_JSON = Path(__file__).parent / "usage_stats.json"
VOCAB_HEADERS = ["date", "word", "concise_meaning", "explanation", "importance", "status"]
MAX_READING_HISTORY = 20


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


class ChatMessage(BaseModel):
    role: str
    content: str


class ExplainRequest(BaseModel):
    selected_text: str
    full_sentence: str
    phrase_mode: bool = False
    messages: list[ChatMessage] = []


class VocabularyRequest(BaseModel):
    word: str
    concise_meaning: str
    explanation: str = ""
    importance: str = ""
    status: str = ""


class VocabularyUpdateRequest(BaseModel):
    word: str | None = None
    concise_meaning: str | None = None
    explanation: str | None = None
    importance: str | None = None
    status: str | None = None


class VocabularyBulkRequest(BaseModel):
    indices: list[int]
    status: str | None = None
    importance: str | None = None


class VocabularyBulkDeleteRequest(BaseModel):
    indices: list[int]


class ReadingPositionRequest(BaseModel):
    book_id: str
    position: int
    filename: str = ""


class ReadingHistoryAddRequest(BaseModel):
    title: str
    type: str  # pasted, txt, epub, pdf
    content_id: str
    book_id: str = ""
    filename: str = ""
    content: str = ""  # for pasted/txt; epub uses cached content


class ReadingHistoryUpdateRequest(BaseModel):
    title: str


class UsageRequest(BaseModel):
    seconds: int


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


def _normalize_plain_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_text_from_pdf(file_content: bytes) -> str:
    """Extract plain text from PDF (text-based PDFs; scanned pages need OCR)."""
    if not PDF_AVAILABLE:
        raise RuntimeError("PyMuPDF is required for PDF support. Run: pip install pymupdf")
    doc = fitz.open(stream=file_content, filetype="pdf")
    try:
        parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                t = page.get_text(sort=True)
            except TypeError:
                t = page.get_text()
            if t:
                parts.append(t)
        raw = "\n\n".join(parts)
        return _normalize_plain_text(raw)
    finally:
        doc.close()


def extract_concise_meaning(explanation: str) -> str:
    """Extract the concise meaning from the structured explanation."""
    match = re.search(r"1\.\s*\*\*Meaning\*\*:\s*(.+?)(?=\n\s*2\.|$)", explanation, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def get_llm_chat_response(
    selected_text: str, full_sentence: str, messages: list[dict]
) -> str:
    """Chat completion for follow-up questions."""
    conv = "\n\n".join(
        f"**{'User' if m['role'] == 'user' else 'Assistant'}:** {m['content']}"
        for m in messages
    )
    prompt = f"""You are a language tutor. The user selected "{selected_text}" in this context: "{full_sentence}".

Previous conversation:
{conv}

Answer the user's follow-up question. Be direct and concise. Maximum 100 words."""

    try:
        if GEMINI_API_KEY:
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text.strip()
        else:
            from openai import OpenAI
            client_kwargs = {"api_key": OPENAI_API_KEY or "not-needed"}
            if OPENAI_BASE_URL:
                client_kwargs["base_url"] = OPENAI_BASE_URL
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def get_llm_explanation(selected_text: str, full_sentence: str, phrase_mode: bool = False) -> str:
    """Call LLM API for contextual explanation."""
    cache_key = f"{selected_text}|{full_sentence}|{phrase_mode}"
    if cache_key in _explanation_cache:
        return _explanation_cache[cache_key]

    if phrase_mode:
        prompt = """You are a language tutor. Explain this sentence in a clear, easy-to-understand way.

Selected text: "{selected_text}"
Context: "{full_sentence}"

Explain the sentence clearly. Be concise.""".format(
            selected_text=selected_text, full_sentence=full_sentence
        )
    else:
        prompt = """You are a language tutor. Explain the selected word or phrase in the specific sentence context.

Use this exact format (use **bold** for keywords):

1. **Meaning**: Very concise definition (a few words only)
2. **Meaning in this sentence**: 1-2 sentences explaining how it's used in the given context
3. **Examples**: 1 example sentence
4. **Expressions**: 1-2 common expressions or collocations using this word
5. **Why this word here?**: 1-2 sentences on why the author chose this word in this context (tone, emphasis, alternatives they could have used)
6. **Register & formality**: A few keywords only (e.g. formal, informal, academic, casual, literary)

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


def _load_reading_history() -> list[dict]:
    """Load reading history list."""
    if not READING_HISTORY_JSON.exists():
        return []
    try:
        with open(READING_HISTORY_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_reading_history(entries: list[dict]) -> None:
    """Save reading history list."""
    with open(READING_HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


CACHED_MANIFEST = CACHED_CONTENT_DIR / "_manifest.json"


def _sanitize_filename(title: str) -> str:
    """Sanitize title for use as filename."""
    s = re.sub(r'[<>:"/\\|?*\n\r]', "_", (title or "").strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:60] or "untitled").strip()


def _load_cached_manifest() -> dict:
    """Load content_id -> filename mapping."""
    if not CACHED_MANIFEST.exists():
        return {}
    try:
        with open(CACHED_MANIFEST, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cached_manifest(manifest: dict) -> None:
    """Save manifest."""
    CACHED_CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHED_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _save_cached_content(content_id: str, text: str, title: str | None = None) -> None:
    """Save text content to cache. Filename uses title when provided."""
    CACHED_CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_cached_manifest()
    suffix = content_id[:12] if len(content_id) >= 8 else content_id
    if title:
        base = _sanitize_filename(title) + "_" + suffix
    else:
        base = content_id
    filename = base + ".txt"
    path = CACHED_CONTENT_DIR / filename
    old_filename = manifest.get(content_id)
    if old_filename:
        old_path = CACHED_CONTENT_DIR / old_filename
        if old_path.exists() and old_path != path:
            try:
                old_path.unlink()
            except OSError:
                pass
    path.write_text(text, encoding="utf-8")
    manifest[content_id] = filename
    _save_cached_manifest(manifest)
    legacy = CACHED_CONTENT_DIR / f"{content_id}.txt"
    if legacy.exists() and legacy != path:
        try:
            legacy.unlink()
        except OSError:
            pass


def _rename_cached_content(content_id: str, title: str) -> None:
    """Rename existing cached file to use new title (e.g. after EPUB import)."""
    legacy = CACHED_CONTENT_DIR / f"{content_id}.txt"
    if not legacy.exists():
        return
    manifest = _load_cached_manifest()
    suffix = content_id[:12] if len(content_id) >= 8 else content_id
    base = _sanitize_filename(title) + "_" + suffix
    filename = base + ".txt"
    new_path = CACHED_CONTENT_DIR / filename
    try:
        legacy.rename(new_path)
        manifest[content_id] = filename
        _save_cached_manifest(manifest)
    except OSError:
        pass


def _load_cached_content(content_id: str) -> str | None:
    """Load text content from cache."""
    manifest = _load_cached_manifest()
    filename = manifest.get(content_id)
    if filename:
        path = CACHED_CONTENT_DIR / filename
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                pass
    legacy = CACHED_CONTENT_DIR / f"{content_id}.txt"
    if legacy.exists():
        try:
            return legacy.read_text(encoding="utf-8")
        except OSError:
            pass
    return None


def _pdf_cache_path(book_id: str) -> Path:
    return CACHED_CONTENT_DIR / f"{book_id}.pdf"


def _save_cached_pdf(book_id: str, raw: bytes) -> None:
    """Store original PDF bytes for client-side viewing."""
    CACHED_CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    _pdf_cache_path(book_id).write_bytes(raw)


def _delete_cached_content(content_id: str) -> None:
    """Remove cached content file and manifest entry."""
    manifest = _load_cached_manifest()
    filename = manifest.pop(content_id, None)
    if filename:
        path = CACHED_CONTENT_DIR / filename
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        _save_cached_manifest(manifest)
    legacy = CACHED_CONTENT_DIR / f"{content_id}.txt"
    if legacy.exists():
        try:
            legacy.unlink()
        except OSError:
            pass
    pdf_path = _pdf_cache_path(content_id)
    if pdf_path.exists():
        try:
            pdf_path.unlink()
        except OSError:
            pass


@app.post("/api/import-txt")
async def import_txt(file: UploadFile):
    """Read text from uploaded TXT file."""
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Please upload a .txt file")
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="replace")
        if not text.strip():
            raise HTTPException(status_code=400, detail="File is empty")
        return {"text": text, "filename": file.filename}
    except HTTPException:
        raise
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Could not decode file: {e}")


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
        _save_cached_content(book_id, text)
        return {"text": text, "filename": file.filename, "book_id": book_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read EPUB: {str(e)}")


@app.post("/api/import-pdf")
async def import_pdf(file: UploadFile):
    """Extract text from uploaded PDF and cache like EPUB."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")
    if not PDF_AVAILABLE:
        detail = (
            f"PDF support not available. Install PyMuPDF using the same Python that runs this app:\n"
            f"  {sys.executable} -m pip install pymupdf"
        )
        if _PDF_IMPORT_ERROR:
            detail += f"\n\nImport error: {_PDF_IMPORT_ERROR}"
        raise HTTPException(status_code=500, detail=detail)
    try:
        content = await file.read()
        book_id = hashlib.sha256(content).hexdigest()
        text = _extract_text_from_pdf(content)
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text found in this PDF. It may be scanned (image-only); OCR is not supported yet.",
            )
        _save_cached_pdf(book_id, content)
        _save_cached_content(book_id, text)
        return {"text": text, "filename": file.filename, "book_id": book_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")


@app.get("/api/pdf/{book_id}")
def serve_cached_pdf(book_id: str):
    """Serve cached PDF file for the in-browser viewer."""
    if not book_id or len(book_id) < 32:
        raise HTTPException(status_code=400, detail="Invalid id")
    path = _pdf_cache_path(book_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(path, media_type="application/pdf")


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


@app.get("/api/reading-history")
def get_reading_history():
    """Return list of recent articles/books."""
    entries = _load_reading_history()
    return {"entries": entries}


@app.post("/api/reading-history")
def add_reading_history(request: ReadingHistoryAddRequest):
    """Add an item to reading history. Caches content for pasted/txt."""
    title = (request.title or "")[:80]
    if request.content:
        _save_cached_content(request.content_id, request.content, title)
    else:
        _rename_cached_content(request.content_id, title)
    entries = _load_reading_history()
    entry = {
        "id": hashlib.sha256(f"{request.content_id}{datetime.now().isoformat()}".encode()).hexdigest()[:16],
        "title": request.title[:80],
        "type": request.type,
        "content_id": request.content_id,
        "book_id": request.book_id,
        "filename": request.filename,
        "timestamp": datetime.now().isoformat(),
    }
    entries = [e for e in entries if e.get("content_id") != request.content_id]
    entries.insert(0, entry)
    entries = entries[:MAX_READING_HISTORY]
    _save_reading_history(entries)
    return {"status": "added", "id": entry["id"]}


@app.get("/api/reading-history/{item_id}/content")
def get_reading_history_content(item_id: str):
    """Get content for a history item (for reopen)."""
    entries = _load_reading_history()
    entry = next((e for e in entries if e.get("id") == item_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Not found")
    content_id = entry.get("content_id")
    if not content_id:
        raise HTTPException(status_code=404, detail="No content")
    text = _load_cached_content(content_id)
    if text is None:
        raise HTTPException(status_code=404, detail="Content no longer available")
    return {
        "text": text,
        "book_id": entry.get("book_id", ""),
        "content_id": entry.get("content_id", ""),
        "filename": entry.get("filename", ""),
        "type": entry.get("type", ""),
    }


@app.delete("/api/reading-history/{item_id}")
def delete_reading_history(item_id: str):
    """Remove an item from reading history and its cached content."""
    entries = _load_reading_history()
    entry = next((e for e in entries if e.get("id") == item_id), None)
    if entry:
        content_id = entry.get("content_id")
        if content_id:
            _delete_cached_content(content_id)
    entries = [e for e in entries if e.get("id") != item_id]
    _save_reading_history(entries)
    return {"status": "deleted"}


@app.put("/api/reading-history/{item_id}")
def update_reading_history(item_id: str, request: ReadingHistoryUpdateRequest):
    """Rename a reading history item and its cached file."""
    title = (request.title or "").strip()[:80]
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")
    entries = _load_reading_history()
    for e in entries:
        if e.get("id") == item_id:
            e["title"] = title
            content_id = e.get("content_id")
            if content_id and _load_cached_content(content_id):
                text = _load_cached_content(content_id)
                if text is not None:
                    _save_cached_content(content_id, text, title)
            _save_reading_history(entries)
            return {"status": "updated"}
    raise HTTPException(status_code=404, detail="Not found")


@app.post("/api/explain")
def explain(request: ExplainRequest):
    """Get contextual explanation or chat follow-up."""
    if request.messages:
        explanation = get_llm_chat_response(
            request.selected_text,
            request.full_sentence,
            [{"role": m.role, "content": m.content} for m in request.messages],
        )
        concise_meaning = ""
    else:
        explanation = get_llm_explanation(
            request.selected_text, request.full_sentence, request.phrase_mode
        )
        concise_meaning = (
            explanation.strip()
            if request.phrase_mode
            else extract_concise_meaning(explanation)
        )
    return {"explanation": explanation, "concise_meaning": concise_meaning}


@app.get("/api/vocabulary")
def get_vocabulary():
    """Return vocabulary CSV data as JSON."""
    entries = _read_vocab_rows()
    return {"entries": entries}


def _load_usage_stats() -> dict:
    """Load usage statistics."""
    if not USAGE_STATS_JSON.exists():
        return {"total_seconds": 0, "by_date": {}}
    try:
        with open(USAGE_STATS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"total_seconds": 0, "by_date": {}}
    else:
        data.setdefault("total_seconds", 0)
        data.setdefault("by_date", {})
        return data


def _add_usage_seconds(seconds: int) -> None:
    """Add seconds to total usage and to today's date bucket."""
    data = _load_usage_stats()
    data["total_seconds"] = data.get("total_seconds", 0) + seconds
    today = datetime.now().strftime("%Y-%m-%d")
    by_date = data.setdefault("by_date", {})
    by_date[today] = by_date.get(today, 0) + seconds
    with open(USAGE_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@app.get("/api/vocabulary/summary")
def get_vocabulary_summary():
    """Return word counts by date for the bar chart, plus per-day app usage time."""
    rows = _read_vocab_rows()
    counts = Counter()
    for row in rows:
        if row.get("date"):
            counts[row["date"]] += 1
    usage = _load_usage_stats()
    usage_by_date = usage.get("by_date") or {}
    all_dates = sorted(set(counts.keys()) | set(usage_by_date.keys()))
    by_date = {d: counts.get(d, 0) for d in all_dates}
    usage_for_dates = {d: int(usage_by_date.get(d, 0)) for d in all_dates}
    return {
        "by_date": by_date,
        "usage_by_date": usage_for_dates,
        "total_usage_seconds": usage.get("total_seconds", 0),
    }


@app.post("/api/usage")
def record_usage(request: UsageRequest):
    """Record usage time (seconds)."""
    if request.seconds > 0 and request.seconds < 86400:
        _add_usage_seconds(request.seconds)
    return {"status": "ok"}


@app.delete("/api/usage")
def reset_usage():
    """Reset total usage time and per-day usage to zero."""
    data = _load_usage_stats()
    data["total_seconds"] = 0
    data["by_date"] = {}
    with open(USAGE_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return {"status": "ok"}


@app.post("/api/vocabulary")
def add_to_vocabulary(request: VocabularyRequest):
    """Append a word to the vocabulary CSV file."""
    rows = _read_vocab_rows()
    rows.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "word": request.word.strip(),
        "concise_meaning": request.concise_meaning.strip(),
        "explanation": (request.explanation or "").strip(),
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
    if request.explanation is not None:
        row["explanation"] = str(request.explanation).strip()
    if request.importance is not None:
        row["importance"] = str(request.importance).strip()
    if request.status is not None:
        row["status"] = str(request.status).strip()
    _write_vocab_rows(rows)
    return {"status": "updated"}


@app.get("/api/quiz")
def get_quiz_words(importance: str = "", days: int | None = None, status: str = "", count: int = 10):
    """Get random vocabulary items for quiz, filtered by importance, date range, and status."""
    count = min(50, max(1, count))  # clamp to 1–50
    rows = _read_vocab_rows()
    now = datetime.now()

    filtered = []
    for i, row in enumerate(rows):
        if importance and (row.get("importance") or "") != importance:
            continue
        if status and (row.get("status") or "") != status:
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

    sample = random.sample(filtered, min(count, len(filtered)))
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


@app.post("/api/vocabulary/bulk-delete")
def bulk_delete_vocabulary(request: VocabularyBulkDeleteRequest):
    """Delete multiple vocabulary rows by index. Indices are processed in descending order."""
    rows = _read_vocab_rows()
    valid = sorted(set(i for i in request.indices if 0 <= i < len(rows)), reverse=True)
    for i in valid:
        rows.pop(i)
    _write_vocab_rows(rows)
    return {"status": "deleted", "count": len(valid)}


@app.post("/api/vocabulary/bulk-update")
def bulk_update_vocabulary(request: VocabularyBulkRequest):
    """Update status and/or importance for multiple vocabulary rows."""
    rows = _read_vocab_rows()
    updated = 0
    for i in request.indices:
        if i < 0 or i >= len(rows):
            continue
        row = rows[i]
        if request.status is not None:
            row["status"] = str(request.status).strip()
        if request.importance is not None:
            row["importance"] = str(request.importance).strip()
        updated += 1
    _write_vocab_rows(rows)
    return {"status": "updated", "count": updated}


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
