"""
Vector database utilities for RecallAI.

IMPORTANT CHANGE:
This module no longer builds the vector database automatically
at import time from a hardcoded "Report.pdf". Instead it exposes
build_vector_db(pdf_path), which app.py calls every time a NEW
PDF is uploaded.

The old chroma_db folder is deleted and rebuilt on every call,
so chunks from a previously uploaded document can never leak
into retrieval results for the new one. This is what fixes the
"still talking about the old document" problem.
"""

import os
import shutil

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = "./chroma_db"

# Embedding model (loaded once, reused for every document)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_vector_db(pdf_path: str):
    """
    (Re)build the Chroma vector store from a single PDF.

    Deletes any existing chroma_db directory first, so the
    vector store always reflects ONLY the most recently
    uploaded document.

    Returns (vector_db, chunks):
      - vector_db: used for normal Q&A (similarity search over
        the top few relevant chunks).
      - chunks: the FULL ordered list of chunks for this
        document, in reading order. This is needed for
        "read the whole document" style questions (summaries,
        "list all key points", etc.), where similarity search
        against only 3-4 chunks isn't enough -- see
        llm.summarize_document().
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Wipe the old vector store so old-document chunks can't
    # be retrieved anymore.
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Smaller, more overlapping chunks = better recall for
    # detail-heavy questions ("what does it say about X on some
    # specific page"), at the cost of a few more chunks overall.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vector_db, chunks
