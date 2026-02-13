# Article Reader with Contextual LLM Word Explanation

![Article Reader screenshot](assets/screenshot.png)

![Vocabulary page screenshot](assets/vocabulary-screenshot.png)

A minimal local web app for reading articles. Select any word or phrase to get a concise, context-aware explanation from an LLM.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root (copy from `.env.example`). Set **one** of:

**Gemini** (preferred):
```
GEMINI_API_KEY=your-gemini-api-key
```
Get a key at https://aistudio.google.com/apikey

**OpenAI** (used when Gemini key is not set):
```
OPENAI_API_KEY=your-openai-api-key
```
Get a key at https://platform.openai.com/api-keys

**Security:** Never commit `.env` — it's in `.gitignore`. Your key stays local only.

Optional:
- `GEMINI_MODEL` – Gemini model (default: `gemini-2.5-flash`)
- `OPENAI_MODEL` – OpenAI model (default: `gpt-4o-mini`)
- `OPENAI_BASE_URL` – For OpenAI-compatible APIs, e.g. Ollama: `http://localhost:11434/v1`

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
