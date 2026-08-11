# ✦ RecallAI

### AI-Powered PDF Document Assistant

RecallAI is a **Retrieval-Augmented Generation (RAG)** application that lets you upload a PDF and interact with it through natural-language conversations.

Instead of manually searching through long documents, RecallAI extracts the document content, creates semantic embeddings, stores them in a vector database, and uses **Mistral AI** to generate answers grounded in the uploaded document.

> **Upload a document → Ask questions → Get answers grounded in your document.**
> 
🚀 **[Live Demo](https://recall-ai.streamlit.app/)**
---

## 🚀 Features

* 📄 **PDF Upload** — Upload and analyze PDF documents directly in the app.
* 💬 **Conversational Q&A** — Ask natural-language questions about your document.
* 🔎 **Semantic Search** — Retrieves the most relevant sections instead of relying on keyword matching.
* 🧠 **RAG Pipeline** — Combines vector retrieval with an LLM to generate context-aware answers.
* 📚 **Whole-Document Analysis** — Broad questions such as *"Summarize this document"* are handled by reading across the document rather than retrieving only a few chunks.
* 📝 **Document Summarization** — Generate concise summaries and extract important findings.
* 🔄 **Fresh Knowledge Base** — Uploading a new PDF rebuilds the vector store so information from previous documents does not leak into the current session.
* 🎛️ **Configurable Retrieval** — Adjust the number of retrieved chunks using the UI.
* 🌡️ **Temperature Control** — Control the creativity of generated responses.
* 🌓 **Light & Dark Mode** — Built-in theme switching.
* ⚡ **Streamlit Interface** — Simple, responsive interface with a chat-based workflow.

---

## 🏗️ Architecture

RecallAI follows a RAG-based architecture:

```text
                    ┌─────────────────┐
                    │    PDF Upload   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   PyPDFLoader   │
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │ Recursive Text Splitter  │
              │ chunk_size = 800         │
              │ overlap = 150            │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │ HuggingFace Embeddings   │
              │ all-MiniLM-L6-v2         │
              └────────────┬─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  ChromaDB   │
                    │ Vector Store│
                    └──────┬──────┘
                           │
                    Similarity Search
                           │
                           ▼
              ┌──────────────────────────┐
              │ Relevant Document Chunks │
              └────────────┬─────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Mistral AI    │
                  │ mistral-small   │
                  └────────┬────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ AI Response  │
                    └─────────────┘
```

---

## 🧠 How It Works

### 1. Upload

The user uploads a PDF through the Streamlit interface.

RecallAI saves the document and begins processing it.

### 2. Document Loading

The PDF is processed using `PyPDFLoader`, which extracts the text and preserves document/page information.

### 3. Chunking

The extracted text is divided into smaller overlapping chunks using `RecursiveCharacterTextSplitter`.

Current configuration:

```text
Chunk size:     800 characters
Chunk overlap:  150 characters
```

The overlap helps preserve context between neighboring chunks.

### 4. Embedding Generation

Each chunk is converted into a vector representation using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These embeddings allow RecallAI to compare the semantic meaning of a user's question with the document content.

### 5. Vector Storage

The embeddings are stored in **ChromaDB**.

When a user asks a question, RecallAI performs similarity search and retrieves the most relevant chunks.

### 6. Retrieval-Augmented Generation

The retrieved chunks are passed as context to Mistral.

The model is instructed to answer **only using the provided document context**.

If the required information cannot be found, the application responds:

```text
I couldn't find that information in the document.
```

This helps reduce unsupported or hallucinated answers.

---

## 📖 Whole-Document Questions

A normal vector search is excellent for specific questions such as:

> "What optimizer was used in the experiment?"

However, questions such as:

> "Summarize the entire document."

or

> "What are the main findings?"

require information from many parts of a document.

RecallAI detects these broad queries and uses a **map-reduce style document analysis approach**.

```text
                 Full Document
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     Chunk 1        Chunk 2        Chunk 3
        │              │              │
        ▼              ▼              ▼
   Extract notes  Extract notes  Extract notes
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                Combine notes
                       │
                       ▼
                  Mistral AI
                       │
                       ▼
                Final Answer
```

This allows RecallAI to answer document-level questions more comprehensively than a simple top-k retrieval system.

For performance reasons, whole-document processing is capped at **60 chunks**.

---

## 🛠️ Tech Stack

| Technology                            | Purpose                   |
| ------------------------------------- | ------------------------- |
| **Python**                            | Core application language |
| **Streamlit**                         | Web interface             |
| **LangChain**                         | RAG orchestration         |
| **Mistral AI**                        | Large Language Model      |
| **ChromaDB**                          | Vector database           |
| **HuggingFace Sentence Transformers** | Text embeddings           |
| **PyPDF**                             | PDF parsing               |
| **RecursiveCharacterTextSplitter**    | Document chunking         |

### Core Libraries

```text
streamlit
langchain-core
langchain-community
langchain-text-splitters
langchain-chroma
langchain-huggingface
langchain-mistralai
sentence-transformers
pypdf
```

---

## 📂 Project Structure

```text
RecallAI/
│
├── app.py
├── chunks2.py
├── llm.py
├── requirements.txt
│
├── .streamlit/
│   └── secrets.toml.example
│
├── .gitignore
│
└── README.md
```

### `app.py`

Main Streamlit application.

Responsible for:

* UI
* PDF upload
* Chat interface
* Session state
* RAG interaction
* Settings
* Theme management

### `chunks2.py`

Handles document ingestion and vector database creation.

Responsible for:

* PDF loading
* Text splitting
* Embedding generation
* ChromaDB creation
* Rebuilding the vector database when a new document is uploaded

### `llm.py`

Contains the LLM and retrieval logic.

Responsible for:

* Mistral API configuration
* Retrieval
* RAG prompts
* Broad-query detection
* Whole-document map-reduce processing

### `requirements.txt`

Contains the Python dependencies required to run the application.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/RecallAI.git
cd RecallAI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Key Setup

RecallAI uses the hosted **Mistral API**.

Create a Mistral API key and configure it as an environment variable.

### Option 1 — Environment Variable

**Windows PowerShell**

```powershell
$env:MISTRAL_API_KEY="your-api-key"
```

**macOS/Linux**

```bash
export MISTRAL_API_KEY="your-api-key"
```

### Option 2 — Streamlit Secrets

Create:

```text
.streamlit/secrets.toml
```

and add:

```toml
MISTRAL_API_KEY = "your-api-key"
```

> ⚠️ Never commit your actual API key to GitHub.

The repository includes:

```text
.streamlit/secrets.toml.example
```

as a template.

---

## ▶️ Running the Application

Start RecallAI with:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

Typically:

```text
http://localhost:8501
```

---

## 💡 Example Questions

After uploading a PDF, you can ask questions such as:

```text
What is the main topic of this document?

Summarize the document.

What are the key findings?

Explain the methodology used.

What datasets were used?

What are the limitations mentioned?

What conclusions does the author reach?

Explain this concept in simple terms.

What does the document say about [specific topic]?
```

---

## 🔐 Document Isolation

One important design decision in RecallAI is that the vector database is rebuilt whenever a new PDF is uploaded.

```text
Old PDF
   ↓
Old ChromaDB
   ↓
DELETED

New PDF
   ↓
New chunks
   ↓
New embeddings
   ↓
New ChromaDB
```

This prevents chunks from a previously uploaded document from being accidentally retrieved when asking questions about a new document.

---

## 🎛️ Configuration

RecallAI provides several configurable options through the interface.

### Retrieved Chunks

Controls how many relevant chunks are retrieved for a normal question.

Higher values can provide more context but may increase the amount of information sent to the LLM.

### Temperature

Controls the randomness of Mistral's responses.

Lower values generally produce more deterministic answers, while higher values allow more variation.

### Theme

Switch between:

* Light
* Dark

---

## 🚀 Deployment

RecallAI can be deployed on platforms capable of running Streamlit applications.

For deployment, configure the secret:

```toml
MISTRAL_API_KEY = "your-api-key"
```

in the platform's secrets/environment configuration.

The application does **not** require a locally running Ollama server because it uses Mistral's hosted API.

---

## ⚠️ Current Limitations

* Currently supports **PDF documents**.
* Whole-document analysis is limited to **60 chunks** for performance.
* Processing large documents can take longer because broad questions may require multiple LLM calls.
* The current vector store is rebuilt when a new document is uploaded.
* Responses depend on the quality and structure of the extracted PDF text.
* Scanned/image-only PDFs may require OCR before their text can be effectively retrieved.

---

## 🔮 Future Improvements

Potential improvements include:

* [ ] Support for DOCX and TXT files
* [ ] OCR for scanned PDFs
* [ ] Multi-document conversations
* [ ] Persistent vector databases
* [ ] Source/page citations in responses
* [ ] Streaming LLM responses
* [ ] Better conversation memory
* [ ] Reranking retrieved chunks
* [ ] Hybrid keyword + semantic search
* [ ] User authentication
* [ ] Document history
* [ ] Background document processing
* [ ] Improved handling of very large documents

---

## 🎯 Why RecallAI?

Traditional document search requires users to manually scan pages and identify relevant information.

RecallAI changes that workflow into a conversational experience:

```text
Traditional Workflow

Open PDF → Search → Read pages → Find context → Understand


RecallAI

Upload PDF → Ask → Retrieve → Generate → Understand
```

The project demonstrates how **Retrieval-Augmented Generation, embeddings, vector databases, and LLMs** can be combined to build a practical document intelligence application.

---

## 👩‍💻 Author

**Kartikey Sharma**

B.Tech Computer Science & Engineering — Data Science

Interested in:

* Artificial Intelligence
* Machine Learning
* Generative AI
* RAG Systems
* NLP
* Data Science

---

## ⭐ If you found RecallAI useful

Consider giving the repository a ⭐ on GitHub.

```text
Built with Python • Streamlit • LangChain • ChromaDB • HuggingFace • Mistral AI
```
