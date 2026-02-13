# Context Reader — AI-powered word explanations and vocabulary

Read articles and books with AI-powered contextual explanations. Select any word or phrase to get instant definitions, examples, and expressions. Save words to a vocabulary, quiz yourself, and track your progress.

![Context Reader](assets/home_page.png)

## Features

- **Read** — Paste text or import EPUB/TXT files. Customize display (background, font size, serif/sans). Reading position and progress bar for EPUBs. Recent items in reading history.
- **Explain** — Select any word or phrase for contextual AI explanation (Gemini or OpenAI). Shows meaning, meaning in sentence, examples, and expressions. Words in your vocabulary are highlighted in the article.
- **Chat** — Ask follow-up questions to dive deeper; the LLM keeps conversation context.
- **Vocabulary** — Save words from explanations. Edit inline, set importance (1–5), filter by date and status. Bulk update or delete. Import/export CSV.
- **Quiz** — Test recall: see a word, choose Master / Vague / Don't remember, then the meaning appears. After 3 seconds, the next word loads. Filter by importance, date range, or status.
- **Summary** — Bar chart of words added per date and total time in app.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file (copy from `.env.example`). Set **one** of:

| Provider | Variable | Get key |
|----------|----------|---------|
| Gemini (preferred) | `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

**Security:** `.env` is in `.gitignore` — your key stays local.

**Optional:** `GEMINI_MODEL`, `OPENAI_MODEL`, `OPENAI_BASE_URL` (for Ollama, etc.)

## Run

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

## Usage

### Reading

1. Paste your article or click **Import EPUB** / **Import TXT** to load a file
2. Click **Display Article** (EPUB imports show automatically)
3. Adjust **Background**, **Size**, and **Font** to your preference
4. Select a word or phrase → explanation appears in the right panel
5. Use the follow-up chat to ask more (e.g. "Can you give more examples?")
6. Click **Add to vocabulary** to save words
7. Reopen recent articles from **Reading history**

### Vocabulary

- **Load saved vocabulary** — View words from `vocabulary.csv`
- **Load from CSV** — Import from another file
- **Filters** — Importance (1–5), date range, status (Master/Vague/Don't remember)
- **Edit inline** — Change word, meaning, importance, or status
- **Bulk actions** — Select rows, then delete or update status/importance
- **Export** — Save or manage `vocabulary.csv` directly

### Quiz

1. Choose filters (importance, date range, status)
2. Start quiz — 10 random words per round
3. See the word, choose Master / Vague / Don't remember
4. Meaning appears; after 3 seconds, the next word loads
5. Status is saved to vocabulary

### Summary

View a bar chart of words added per date and total time spent in the app.

## Screenshots

**Vocabulary** — Manage saved words, filter by importance and status

![Vocabulary](assets/vocabulary-screenshot.png)

**Quiz** — Test recall, rate each word, update status

![Quiz](assets/quiz-screenshot.png)