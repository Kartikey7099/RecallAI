# RecallAI

Upload a PDF, then chat with it. Retrieval-augmented Q&A built with
Streamlit, ChromaDB, and Mistral.

This version is set up to run **either locally or on Streamlit
Community Cloud** — see `CHANGES_README.txt` for what changed and
why (short version: the local-only Ollama dependency was swapped
for Mistral's hosted API, since a deployed app has no way to run
Ollama for you).

## 1. Get a Mistral API key

Free, no credit card required:
1. Go to https://console.mistral.ai/ and sign up.
2. Create an API key under **API Keys**.

## 2. Run it locally

```bash
pip install -r requirements.txt

mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste in your real key

streamlit run app.py
```

## 3. Push to GitHub

```bash
git init
git add .
git commit -m "RecallAI"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

`secrets.toml` is git-ignored on purpose — your key never gets
committed. Double check with `git status` before your first
commit if you're unsure.

## 4. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io/ and sign in with GitHub.
2. Click **New app**, pick your repo/branch, and set the main file
   to `app.py`.
3. Before (or right after) it deploys, open **Settings → Secrets**
   on the app and paste:
   ```toml
   MISTRAL_API_KEY = "your-real-key-here"
   ```
4. Save. The app restarts automatically and picks up the key.

That's it — every time you push to the branch you deployed from,
Streamlit Cloud redeploys automatically.

## Notes / limitations

- **One document at a time.** Uploading a new PDF replaces the
  previously embedded one for everyone using that deployment —
  there's no per-user document library. Fine for a personal or
  demo deployment; if several people will use it simultaneously,
  they'll interrupt each other.
- **Storage is ephemeral.** `uploads/` and `chroma_db/` are wiped
  whenever the app reboots or redeploys, and rebuilt automatically
  the next time someone uploads a PDF.
- **Broad questions are slower.** Things like "summarize this" or
  "give me 10 key points" walk through the whole document
  (map-reduce), which takes noticeably longer than a specific
  question. This is expected — the loading spinner says so.
