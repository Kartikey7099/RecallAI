import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# LLM
# ============================================================
#
# DEPLOYMENT CHANGE: the original app called a local Ollama
# server ("mistral" model) running on your own machine. That
# works fine when you run `streamlit run app.py` locally, but a
# hosted service like Streamlit Community Cloud has no way to
# run Ollama for you -- there's no local GPU/daemon to talk to,
# so the app would just hang or crash on every question.
#
# Fix: talk to Mistral's own hosted API (La Plateforme) instead.
# Same model family ("Mistral"), so nothing in the UI text needed
# to change -- only the connection changed from "local" to
# "cloud". This needs an API key, set as the MISTRAL_API_KEY
# environment variable (locally) or as a Streamlit secret (once
# deployed). Get a free key at https://console.mistral.ai/

def _get_mistral_api_key():
    """
    Look for the API key in Streamlit secrets first (how it's
    provided once deployed on Streamlit Community Cloud), then
    fall back to a plain environment variable (how it's provided
    for local development, e.g. via a .env file + python-dotenv,
    or `export MISTRAL_API_KEY=...`).
    """
    try:
        import streamlit as st

        if "MISTRAL_API_KEY" in st.secrets:
            return st.secrets["MISTRAL_API_KEY"]
    except Exception:
        # No secrets.toml locally, or Streamlit secrets not
        # configured yet -- fall through to the env var.
        pass

    return os.environ.get("MISTRAL_API_KEY")


MISTRAL_API_KEY = _get_mistral_api_key()

local_llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.2,
    api_key=MISTRAL_API_KEY,
)


def _text(response) -> str:
    """
    Pull plain text out of an LLM response.

    ChatMistralAI (a chat model) returns an AIMessage object, not
    a plain string like the old OllamaLLM did. str(AIMessage(...))
    prints its full repr (content=... additional_kwargs=... etc),
    not just the text -- so callers that used to do
    str(local_llm.invoke(...)) need to go through this instead.
    """
    return getattr(response, "content", str(response))

# ============================================================
# STANDARD RAG PROMPT (used for normal, specific questions)
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not found in the context, say:

"I couldn't find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""
)


def get_retriever(vector_db, k: int = 4):
    """
    Build a retriever from the CURRENT document's vector store.

    Used for normal, specific questions ("what does the paper
    say about X"), where searching for the few most relevant
    chunks is enough.
    """
    return vector_db.as_retriever(search_kwargs={"k": k})


# ============================================================
# WHOLE-DOCUMENT READING (used for broad / "read it all" questions)
# ============================================================
#
# A similarity-search retriever only ever looks at the top-k
# chunks that best match the QUESTION's wording. That's fine for
# "what is the batch size used in section 3?" but it silently
# fails for broad requests like "summarize this" or "give me 10
# key points" or "read the whole article in detail" -- the query
# text doesn't semantically match most of the document, so most
# of the document never gets retrieved, and the model ends up
# inventing generic-sounding points instead of reporting what's
# actually in the file.
#
# The fix: detect these broad requests and, instead of retrieving
# top-k chunks, walk through EVERY chunk of the document (up to a
# safety cap), extract notes relevant to the question from each
# one (the "map" step), then combine all of those notes into one
# final answer (the "reduce" step). This is slower (one LLM call
# per chunk) but it actually reads the whole document.

BROAD_QUERY_KEYWORDS = [
    "summar",
    "overview",
    "main point",
    "key point",
    "important point",
    "key finding",
    "in detail",
    "detailed",
    "entire document",
    "whole document",
    "full document",
    "whole article",
    "entire article",
    "read the article",
    "read this article",
    "list all",
    "main topic",
    "explain the document",
    "explain this document",
    "what is this document about",
    "what is this about",
]

# A cap on how many chunks we'll walk through for a whole-document
# read. Each chunk costs one extra LLM call, so this keeps very
# large PDFs from taking forever on a local model. 60 chunks at
# ~800 chars each covers roughly a 40-60 page paper.
MAX_MAP_CHUNKS = 60


def is_broad_query(question: str) -> bool:
    """
    Heuristic: does this question ask about the document as a
    whole (summary, overview, key points, "read in detail"),
    rather than a specific, narrow fact?
    """
    q = question.lower()
    return any(keyword in q for keyword in BROAD_QUERY_KEYWORDS)


map_prompt = ChatPromptTemplate.from_template(
    """
You are reading one excerpt out of a longer document, in order to
help answer a question about the WHOLE document later.

From the excerpt below, pull out only the information relevant to
the question. Use short bullet points. If this excerpt has
nothing relevant, respond with exactly: NOTHING_RELEVANT

Question:
{question}

Excerpt:
{text}

Relevant notes from this excerpt:
"""
)

combine_prompt = ChatPromptTemplate.from_template(
    """
You were given notes extracted from every part of a document, in
reading order. Using ONLY these notes, write one clear, complete,
well-organized answer to the question. Remove duplicate points and
merge related ones. Do not mention "excerpts" or "notes" -- answer
as if you had read the whole document yourself.

Question:
{question}

Notes gathered from the document:
{text}

Final answer:
"""
)


def summarize_document(chunks, question: str, max_chunks: int = MAX_MAP_CHUNKS) -> str:
    """
    Map-reduce read of the WHOLE document (up to max_chunks),
    used for broad questions instead of top-k similarity search.

    chunks: the full ordered list of LangChain Document chunks
            for the currently loaded PDF (from
            chunks2.build_vector_db).
    """
    use_chunks = chunks[:max_chunks]

    notes = []

    for chunk in use_chunks:
        map_input = map_prompt.format(
            question=question,
            text=chunk.page_content,
        )

        chunk_notes = _text(local_llm.invoke(map_input)).strip()

        if chunk_notes and "NOTHING_RELEVANT" not in chunk_notes:
            notes.append(chunk_notes)

    if not notes:
        return "I couldn't find that information in the document."

    combined_notes = "\n\n".join(notes)

    combine_input = combine_prompt.format(
        question=question,
        text=combined_notes,
    )

    final_answer = local_llm.invoke(combine_input)

    return _text(final_answer)
