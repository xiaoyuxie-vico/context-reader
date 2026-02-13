# Article Reader with Contextual LLM Word Explanation

A minimal local web app for reading articles. Select any word or phrase to get a concise, context-aware explanation from an LLM.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Set your LLM API key via environment variable. **Gemini is used if its key is set**; otherwise OpenAI is used.

**Gemini (recommended):**
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

**OpenAI:**
```bash
export OPENAI_API_KEY="your-api-key"
```

Optional:
- `GEMINI_MODEL` – Gemini model name (default: `gemini-2.5-flash`)
- `OPENAI_BASE_URL` – Use a different API endpoint (e.g., Ollama: `http://localhost:11434/v1`)
- `OPENAI_MODEL` – OpenAI model name (default: `gpt-4o-mini`)

## Run

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

## Usage

1. Paste your article into the text area
2. Click "Display Article"
3. Select a word or phrase while reading
4. The explanation appears in a panel on the right
5. Or press **Alt+E** (Option+E on Mac) after selecting text
