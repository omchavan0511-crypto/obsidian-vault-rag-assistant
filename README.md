# Obsidian Vault RAG Knowledge Assistant

Ask natural-language questions over your own Obsidian vault (or any folder of Markdown notes) and get answers grounded in the notes, with source citations.

## Approach

- **Upload**: user uploads a `.zip` of their vault (or individual files) directly in the browser. Supports `.md` (with frontmatter/tags), `.txt`, `.pdf`, `.docx`, and generically any other text-based file (`.json`, `.csv`, code files, etc.). Binary files like images are automatically skipped.
- **Ingestion**: reads the `.md` files, strips YAML frontmatter, keeps tags/title as metadata.
- **Indexing**: chunks notes (512 tokens, 50 overlap) with LlamaIndex's `SentenceSplitter`, embeds locally with a free HuggingFace model (`BAAI/bge-small-en-v1.5`, runs on CPU, no API key needed), stores vectors in a persistent local ChromaDB collection.
- **Retrieval + generation**: on a query, retrieves the top-k most similar chunks and passes them to Groq's free hosted LLM (`openai/gpt-oss-20b`), instructed to answer only from context and cite source note titles.
- **UI**: single-page Streamlit chat app.

## Setup

```bash
pip install -r requirements.txt
```

Get a free Groq API key at console.groq.com/keys (no credit card required). Copy `env-example.txt` to `.env` and add your key:

```
GROQ_API_KEY=your_key_here
PERSIST_DIR=storage
```

## Run

```bash
cd app
streamlit run streamlit_app.py
```

In the sidebar, upload a `.zip` of your Obsidian vault (or individual `.md` files) and click "Build / Rebuild index", then ask questions in the chat box.

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to share.streamlit.io, connect the repo, set main file to `app/streamlit_app.py`.
3. In app settings, add `GROQ_API_KEY` as a secret.
4. Deploy — the resulting URL is your live demo link.

## Project structure

```
app/
  ingest.py          vault loading + index build/load
  query.py           query engine + citation extraction
  streamlit_app.py   chat UI with upload
requirements.txt
env-example.txt      copy this to .env and add your key
```
