import streamlit as st
import time
from pathlib import Path
from html import escape

from llm import get_retriever, prompt, local_llm, is_broad_query, summarize_document, MISTRAL_API_KEY
from chunks2 import build_vector_db
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RecallAI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_size" not in st.session_state:
    st.session_state.document_size = 0

# The vector store for the CURRENTLY loaded document.
# Rebuilt every time a new file is uploaded.
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# The full ordered list of chunks for the CURRENTLY loaded
# document. Used for whole-document ("summarize", "key points",
# "read in detail") questions -- see summarize_document() in
# llm.py. Similarity search alone only ever sees a handful of
# chunks, which isn't enough to answer these broad questions well.
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = None

# Tracks which file's path was last embedded, so we don't
# re-embed on every Streamlit rerun (only on an actual new upload).
if "ingested_file_path" not in st.session_state:
    st.session_state.ingested_file_path = None

if "response_times" not in st.session_state:
    st.session_state.response_times = []

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# Used by the suggested prompt buttons
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# THEME
# ============================================================

light_colors = {
    "bg": "#F7F8FC",
    "surface": "#FFFFFF",
    "surface2": "#F2F4F8",
    "surface3": "#ECEFF4",
    "border": "#DDE2EA",
    "border_hover": "#C5CBD6",
    "text": "#151821",
    "secondary": "#555D6D",
    "muted": "#788191",
    "accent": "#5B5CE2",
    "accent_hover": "#4B4CC7",
    "accent_soft": "#EEEEFF",
}

dark_colors = {
    "bg": "#0B0D12",
    "surface": "#11141B",
    "surface2": "#181C25",
    "surface3": "#1D222D",
    "border": "#282E3A",
    "border_hover": "#3B4453",
    "text": "#F5F7FB",
    "secondary": "#ADB5C4",
    "muted": "#7A8394",
    "accent": "#818CF8",
    "accent_hover": "#6366F1",
    "accent_soft": "#1B1D3B",
}


theme = st.session_state.theme

if theme == "Dark":
    colors = dark_colors
else:
    colors = light_colors


# ============================================================
# GLOBAL CSS
# ============================================================

css = f"""
<style>

/* ============================================================
   VARIABLES
============================================================ */

:root {{
    --bg: {colors["bg"]};
    --surface: {colors["surface"]};
    --surface2: {colors["surface2"]};
    --surface3: {colors["surface3"]};

    --border: {colors["border"]};
    --border-hover: {colors["border_hover"]};

    --text: {colors["text"]};
    --secondary: {colors["secondary"]};
    --muted: {colors["muted"]};

    --accent: {colors["accent"]};
    --accent-hover: {colors["accent_hover"]};
    --accent-soft: {colors["accent_soft"]};
}}


/* ============================================================
   APP
============================================================ */

.stApp {{
    background:
        radial-gradient(
            circle at 5% 0%,
            rgba(99, 102, 241, 0.055),
            transparent 25%
        ),
        radial-gradient(
            circle at 95% 0%,
            rgba(14, 165, 233, 0.045),
            transparent 22%
        ),
        var(--bg);

    color: var(--text);
}}


.block-container {{
    max-width: 1280px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}}


#MainMenu {{
    visibility: hidden;
}}


footer {{
    visibility: hidden;
}}


header {{
    background: transparent !important;
}}


/* ============================================================
   GLOBAL TEXT
============================================================ */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {{
    color: var(--text) !important;
}}


.stApp p {{
    color: var(--text);
}}


.stMarkdown {{
    color: var(--text) !important;
}}


/* ============================================================
   SIDEBAR
============================================================ */

section[data-testid="stSidebar"] {{
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    color: var(--text) !important;
}}


section[data-testid="stSidebar"] > div {{
    padding: 1.05rem 0.9rem 1.5rem 0.9rem;
}}


section[data-testid="stSidebar"] * {{
    color: var(--text);
}}


/* ============================================================
   BRAND
============================================================ */

.brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 25px;
}}


.logo {{
    width: 40px;
    height: 40px;

    border-radius: 12px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: linear-gradient(
        135deg,
        #6366F1,
        #8B5CF6
    );

    color: #FFFFFF !important;

    font-size: 20px;
    font-weight: 800;

    box-shadow:
        0 7px 20px rgba(99, 102, 241, 0.22);
}}


.brand-name {{
    color: var(--text) !important;

    font-size: 17px;

    font-weight: 800;

    letter-spacing: -0.3px;
}}


.brand-subtitle {{
    color: var(--muted) !important;

    font-size: 8px;

    margin-top: 2px;
}}


/* ============================================================
   SIDEBAR LABELS
============================================================ */

.sidebar-label {{
    color: var(--muted) !important;

    font-size: 9px;

    font-weight: 800;

    text-transform: uppercase;

    letter-spacing: 1.1px;

    margin-top: 21px;

    margin-bottom: 8px;
}}


/* ============================================================
   DOCUMENT CARD
============================================================ */

.document-card {{
    background: var(--surface2);

    border: 1px solid var(--border);

    border-radius: 12px;

    padding: 12px;

    margin-top: 8px;
}}


.document-icon {{
    width: 32px;
    height: 32px;

    border-radius: 9px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: var(--accent-soft);

    margin-bottom: 8px;

    font-size: 15px;
}}


.document-name {{
    color: var(--text) !important;

    /* Bumped up from 10px so the filename is actually readable,
       not just technically present. */
    font-size: 13px;

    font-weight: 700;

    line-height: 1.5;

    /* Always show the FULL filename -- wrap onto as many lines
       as needed instead of clipping or truncating it. */
    overflow-wrap: anywhere;
    word-break: break-word;
    white-space: normal;
}}


.document-size {{
    color: var(--muted) !important;

    font-size: 10px;

    margin-top: 4px;
}}


.ready {{
    display: inline-block;

    margin-top: 8px;

    color: #15803D !important;

    background: rgba(22, 163, 74, 0.08);

    border: 1px solid rgba(22, 163, 74, 0.16);

    border-radius: 20px;

    padding: 3px 7px;

    font-size: 8px;

    font-weight: 750;
}}


/* ============================================================
   SYSTEM CARDS
============================================================ */

.system-card {{
    background: var(--surface2);

    border: 1px solid var(--border);

    border-radius: 10px;

    padding: 9px 10px;

    margin-bottom: 6px;
}}


.system-label {{
    color: var(--muted) !important;

    font-size: 7px;

    font-weight: 800;

    text-transform: uppercase;

    letter-spacing: 0.8px;
}}


.system-value {{
    color: var(--text) !important;

    font-size: 10px;

    font-weight: 700;

    margin-top: 3px;
}}


/* ============================================================
   FILE UPLOADER
============================================================ */

[data-testid="stFileUploader"] {{
    background: var(--surface2) !important;

    border: 1px dashed var(--border-hover) !important;

    border-radius: 12px !important;

    padding: 4px !important;
}}


[data-testid="stFileUploader"] section {{
    background: var(--surface2) !important;

    border: none !important;

    border-radius: 10px !important;
}}


/* Upload button */
[data-testid="stFileUploader"] button {{
    background: var(--surface) !important;

    color: var(--text) !important;

    border: 1px solid var(--border-hover) !important;

    border-radius: 9px !important;

    opacity: 1 !important;

    font-weight: 700 !important;

    box-shadow: none !important;
}}


[data-testid="stFileUploader"] button:hover {{
    background: var(--surface3) !important;

    color: var(--accent) !important;

    border-color: var(--accent) !important;
}}


[data-testid="stFileUploader"] button p {{
    color: inherit !important;
}}


[data-testid="stFileUploader"] button span {{
    color: inherit !important;
}}


[data-testid="stFileUploader"] small {{
    color: var(--muted) !important;
}}


[data-testid="stFileUploader"] svg {{
    color: var(--secondary) !important;
}}


/* ============================================================
   SIDEBAR INPUT LABELS
============================================================ */

section[data-testid="stSidebar"] label {{
    color: var(--secondary) !important;

    font-size: 10px !important;
}}


/* ============================================================
   SELECTBOX
============================================================ */

div[data-baseweb="select"] > div {{
    background: var(--surface) !important;

    border-color: var(--border) !important;

    color: var(--text) !important;

    border-radius: 9px !important;
}}


div[data-baseweb="select"] span {{
    color: var(--text) !important;
}}


/* ============================================================
   TOP BAR
============================================================ */

.topbar {{
    display: flex;

    align-items: center;

    justify-content: space-between;

    padding-bottom: 15px;

    border-bottom: 1px solid var(--border);

    margin-bottom: 20px;
}}


.title {{
    color: var(--text) !important;

    font-size: 25px;

    font-weight: 800;

    letter-spacing: -0.7px;
}}


.subtitle {{
    color: var(--muted) !important;

    font-size: 10px;

    margin-top: 3px;
}}


.online {{
    display: flex;

    align-items: center;

    gap: 6px;

    color: #15803D !important;

    background: rgba(22, 163, 74, 0.07);

    border: 1px solid rgba(22, 163, 74, 0.15);

    border-radius: 20px;

    padding: 5px 9px;

    font-size: 8px;

    font-weight: 750;
}}


.online-dot {{
    width: 6px;
    height: 6px;

    background: #22C55E;

    border-radius: 50%;
}}


/* ============================================================
   HERO
============================================================ */

.hero {{
    position: relative;

    overflow: hidden;

    background: var(--surface);

    border: 1px solid var(--border);

    border-radius: 18px;

    padding: 28px;

    margin-bottom: 18px;

    box-shadow:
        0 8px 28px rgba(15, 23, 42, 0.04);
}}


.hero::after {{
    content: "";

    position: absolute;

    right: -70px;
    top: -90px;

    width: 190px;
    height: 190px;

    border-radius: 50%;

    background:
        rgba(99, 102, 241, 0.07);
}}


.hero-badge {{
    position: relative;

    z-index: 1;

    display: inline-flex;

    background: var(--accent-soft);

    color: var(--accent) !important;

    border: 1px solid rgba(99, 102, 241, 0.15);

    border-radius: 20px;

    padding: 5px 9px;

    font-size: 8px;

    font-weight: 800;
}}


.hero-title {{
    position: relative;

    z-index: 1;

    color: var(--text) !important;

    font-size: 26px;

    font-weight: 800;

    letter-spacing: -0.8px;

    margin-top: 12px;
}}


.hero-description {{
    position: relative;

    z-index: 1;

    color: var(--secondary) !important;

    font-size: 11px;

    line-height: 1.7;

    max-width: 650px;

    margin-top: 6px;
}}


/* ============================================================
   PROMPT CARDS
============================================================ */

.prompt-card {{
    background: var(--surface);

    border: 1px solid var(--border);

    border-radius: 14px;

    padding: 15px;

    min-height: 100px;

    transition:
        transform 0.18s ease,
        border-color 0.18s ease;
}}


.prompt-card:hover {{
    transform: translateY(-2px);

    border-color:
        rgba(99, 102, 241, 0.4);
}}


.prompt-icon {{
    color: var(--accent) !important;

    font-size: 17px;

    margin-bottom: 7px;
}}


.prompt-title {{
    color: var(--text) !important;

    font-size: 11px;

    font-weight: 750;
}}


.prompt-description {{
    color: var(--muted) !important;

    font-size: 9px;

    line-height: 1.5;

    margin-top: 4px;
}}


/* ============================================================
   BUTTONS
============================================================ */

.stButton > button {{
    background: var(--surface) !important;

    color: var(--text) !important;

    border: 1px solid var(--border) !important;

    border-radius: 10px !important;

    font-size: 10px !important;

    font-weight: 650 !important;

    min-height: 35px;

    opacity: 1 !important;

    box-shadow: none !important;
}}


.stButton > button:hover {{
    background: var(--surface2) !important;

    color: var(--accent) !important;

    border-color:
        rgba(99, 102, 241, 0.45) !important;
}}


.stButton > button p {{
    color: inherit !important;
}}


/* ============================================================
   TABS
============================================================ */

div[data-baseweb="tab-list"] {{
    gap: 8px;

    border-bottom:
        1px solid var(--border) !important;
}}


button[data-baseweb="tab"] {{
    background: transparent !important;

    color: var(--secondary) !important;

    opacity: 1 !important;

    font-size: 11px !important;

    font-weight: 700 !important;
}}


button[data-baseweb="tab"]:hover {{
    color: var(--accent) !important;

    opacity: 1 !important;
}}


button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--accent) !important;

    opacity: 1 !important;
}}


button[data-baseweb="tab"] p {{
    color: inherit !important;

    opacity: 1 !important;
}}


button[data-baseweb="tab"] span {{
    color: inherit !important;

    opacity: 1 !important;
}}


div[data-baseweb="tab-highlight"] {{
    background-color: var(--accent) !important;

    height: 2px !important;
}}


/* ============================================================
   CHAT
============================================================ */

[data-testid="stChatMessage"] {{
    background: transparent !important;

    border: none !important;

    color: var(--text) !important;
}}


[data-testid="stChatMessage"] * {{
    color: var(--text) !important;
}}


[data-testid="stChatMessage"] p {{
    color: var(--text) !important;

    font-size: 12px !important;

    line-height: 1.7 !important;
}}


[data-testid="stChatMessage"] li {{
    color: var(--text) !important;
}}


[data-testid="stChatMessage"] strong {{
    color: var(--text) !important;
}}


/* ============================================================
   CHAT INPUT
============================================================ */

[data-testid="stChatInput"] {{
    padding-top: 12px;
}}


[data-testid="stChatInput"] > div {{
    background: var(--surface) !important;

    border: 1px solid var(--border) !important;

    border-radius: 14px !important;

    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.05);
}}


[data-testid="stChatInput"] textarea {{
    background: transparent !important;

    color: var(--text) !important;

    font-size: 11px !important;
}}


[data-testid="stChatInput"] textarea::placeholder {{
    color: var(--muted) !important;
}}


/* Chat send button */
[data-testid="stChatInput"] button {{
    color: var(--text) !important;

    background: transparent !important;
}}


/* ============================================================
   ALERTS
============================================================ */

[data-testid="stAlert"] {{
    border-radius: 12px !important;
}}


/* ============================================================
   METRICS
============================================================ */

[data-testid="stMetric"] {{
    background: var(--surface);

    border: 1px solid var(--border);

    border-radius: 12px;

    padding: 12px;
}}


[data-testid="stMetricLabel"] {{
    color: var(--muted) !important;
}}


[data-testid="stMetricValue"] {{
    color: var(--text) !important;
}}


/* ============================================================
   PIPELINE
============================================================ */

.pipeline {{
    display: flex;

    align-items: center;

    justify-content: center;

    flex-wrap: wrap;

    gap: 7px;

    padding: 20px 5px;
}}


.pipeline-item {{
    background: var(--surface);

    border: 1px solid var(--border);

    border-radius: 10px;

    padding: 8px 10px;

    color: var(--text) !important;

    font-size: 8px;

    font-weight: 700;
}}


.arrow {{
    color: var(--muted) !important;

    font-size: 10px;
}}


/* ============================================================
   SLIDERS
============================================================ */

.stSlider label {{
    color: var(--secondary) !important;
}}


/* ============================================================
   FOOTER
============================================================ */

.footer {{
    text-align: center;

    color: var(--muted) !important;

    font-size: 9px;

    padding-top: 25px;
}}


/* ============================================================
   MOBILE
============================================================ */

@media (max-width: 700px) {{

    .hero {{
        padding: 22px;
    }}

    .hero-title {{
        font-size: 22px;
    }}

    .title {{
        font-size: 22px;
    }}

}}

</style>
"""


# Use st.html so the browser interprets this as HTML/CSS.
st.html(css)


# ============================================================
# RAG FUNCTIONS
# ============================================================

def format_docs(docs):
    """
    Convert retrieved LangChain documents into a single string.
    """
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def build_chain(vector_db, k, temperature):
    """
    Build the RAG chain for the CURRENT document.

    IMPORTANT CHANGE: this now takes the active vector_db, the
    top_k setting, and the temperature as arguments, so the chain
    always retrieves from whichever document was most recently
    uploaded (instead of a single retriever fixed at import time)
    and always uses the sidebar's current Temperature slider
    value (instead of a fixed default that ignored the slider).

        Retriever -> Context
        Question
        Prompt
        Mistral (hosted)
        String output
    """
    retriever = get_retriever(vector_db, k=k)

    # `.bind(temperature=...)` sends the sidebar's Temperature
    # slider value with every call, instead of always using the
    # fixed default set on local_llm in llm.py.
    llm_with_temp = local_llm.bind(temperature=temperature)

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm_with_temp
        | StrOutputParser()
    )


def clear_chat():
    """
    Reset the current conversation.
    """
    st.session_state.messages = []
    st.session_state.response_times = []
    st.session_state.pending_question = None


def save_uploaded_file(uploaded_file):
    """
    Save the uploaded PDF into the local uploads folder.
    """
    upload_dir = Path("uploads")

    upload_dir.mkdir(
        exist_ok=True
    )

    file_path = upload_dir / uploaded_file.name

    with open(file_path, "wb") as file:
        file.write(
            uploaded_file.getbuffer()
        )

    return file_path


def add_question(question):
    """
    Add a question to the pending queue.

    This is used by the suggested prompt cards.
    """
    st.session_state.pending_question = question


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    st.html(
        """
        <div class="brand">

            <div class="logo">
                ✦
            </div>

            <div>

                <div class="brand-name">
                    RecallAI
                </div>

                <div class="brand-subtitle">
                    AI-powered document intelligence
                </div>

            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    st.html(
        """
        <div class="sidebar-label">
            DOCUMENT
        </div>
        """
    )


    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload a PDF document.",
    )


    if uploaded_file:

        file_path = save_uploaded_file(
            uploaded_file
        )

        # Only re-embed if this is actually a NEW file, not just
        # a Streamlit rerun with the same upload still in the
        # widget's state.
        if str(file_path) != st.session_state.ingested_file_path:

            with st.spinner(
                f"Reading {uploaded_file.name} and building its knowledge base..."
            ):

                st.session_state.vector_db, st.session_state.doc_chunks = build_vector_db(
                    str(file_path)
                )

            st.session_state.ingested_file_path = str(file_path)

            # New document -> old chat no longer applies to it.
            clear_chat()

        st.session_state.document_name = (
            uploaded_file.name
        )

        st.session_state.document_size = (
            uploaded_file.size / 1024 / 1024
        )

        safe_name = escape(
            uploaded_file.name
        )

        st.html(
            f"""
            <div class="document-card">

                <div class="document-icon">
                    📄
                </div>

                <div class="document-name" title="{safe_name}">
                    {safe_name}
                </div>

                <div class="document-size">
                    {st.session_state.document_size:.2f} MB
                </div>

                <div class="ready">
                    ● READY
                </div>

            </div>
            """
        )


    elif st.session_state.document_name:

        safe_name = escape(
            st.session_state.document_name
        )

        st.html(
            f"""
            <div class="document-card">

                <div class="document-icon">
                    📄
                </div>

                <div class="document-name" title="{safe_name}">
                    {safe_name}
                </div>

                <div class="document-size">
                    {st.session_state.document_size:.2f} MB
                </div>

                <div class="ready">
                    ● READY
                </div>

            </div>
            """
        )


    else:

        st.caption(
            "Upload a PDF to start asking questions."
        )


    # --------------------------------------------------------
    # AI SYSTEM
    # --------------------------------------------------------

    st.html(
        """
        <div class="sidebar-label">
            AI SYSTEM
        </div>

        <div class="system-card">

            <div class="system-label">
                LANGUAGE MODEL
            </div>

            <div class="system-value">
                🧠 Mistral
            </div>

        </div>

        <div class="system-card">

            <div class="system-label">
                VECTOR DATABASE
            </div>

            <div class="system-value">
                🗃 ChromaDB
            </div>

        </div>

        <div class="system-card">

            <div class="system-label">
                ARCHITECTURE
            </div>

            <div class="system-value">
                🔎 RAG
            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    st.html(
        """
        <div class="sidebar-label">
            RETRIEVAL
        </div>
        """
    )


    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
    )


    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
    )


    # --------------------------------------------------------
    # APPEARANCE
    # --------------------------------------------------------

    st.html(
        """
        <div class="sidebar-label">
            APPEARANCE
        </div>
        """
    )


    selected_theme = st.selectbox(
        "Theme",
        ["Light", "Dark"],
        index=[
            "Light",
            "Dark",
        ].index(
            st.session_state.theme
        ),
        label_visibility="collapsed",
    )


    if selected_theme != st.session_state.theme:

        st.session_state.theme = selected_theme

        st.rerun()


    # --------------------------------------------------------
    # ACTIONS
    # --------------------------------------------------------

    st.html(
        """
        <div class="sidebar-label">
            ACTIONS
        </div>
        """
    )


    button1, button2 = st.columns(2)


    with button1:

        if st.button(
            "＋ New chat",
            use_container_width=True,
        ):

            clear_chat()

            st.rerun()


    with button2:

        if st.button(
            "Clear",
            use_container_width=True,
        ):

            clear_chat()

            st.rerun()


# ============================================================
# BUILD CHAIN FOR THE CURRENT DOCUMENT
# ============================================================

# Rebuilt every rerun using whichever document is currently
# ingested and the current "Retrieved chunks" slider value, so
# it always reflects the latest upload.
if st.session_state.vector_db is not None:
    chain = build_chain(st.session_state.vector_db, top_k, temperature)
else:
    chain = None


def answer_question(question: str) -> str:
    """
    Answer a question about the current document.

    Broad questions ("summarize this", "give me 10 key points",
    "read the article in detail") are routed to
    summarize_document(), which walks through the WHOLE document
    instead of only the handful of chunks a similarity search
    would retrieve. Everything else uses the normal fast RAG
    chain.
    """
    if chain is None:
        return (
            "Please upload a PDF from the sidebar first — "
            "I don't have a document to search yet."
        )

    from llm import MISTRAL_API_KEY

    if not MISTRAL_API_KEY:
        return (
            "**No Mistral API key found.** RecallAI now answers "
            "using Mistral's hosted API instead of a local Ollama "
            "server, so it needs an API key to work.\n\n"
            "- Locally: set the `MISTRAL_API_KEY` environment variable.\n"
            "- On Streamlit Community Cloud: add it under "
            "**App settings → Secrets** as `MISTRAL_API_KEY = \"...\"`.\n\n"
            "Get a free key at https://console.mistral.ai/"
        )

    try:

        if is_broad_query(question) and st.session_state.doc_chunks:

            with st.spinner(
                "Reading the full document in detail — this covers "
                "every section, so it can take a bit longer..."
            ):

                return summarize_document(
                    st.session_state.doc_chunks,
                    question,
                )

        else:

            with st.spinner(
                "Searching your knowledge base..."
            ):

                return str(chain.invoke(question))

    except Exception as error:

        return (
            "I couldn't generate an answer right now.\n\n"
            f"**Error:** `{error}`"
        )


# ============================================================
# TOP BAR
# ============================================================

st.html(
    """
    <div class="topbar">

        <div>

            <div class="title">
                RecallAI
            </div>

            <div class="subtitle">
                Ask questions. Retrieve context.
                Understand your documents.
            </div>

        </div>

        <div class="online">

            <span class="online-dot"></span>

            LOCAL AI ONLINE

        </div>

    </div>
    """
)


# ============================================================
# TABS
# ============================================================

chat_tab, document_tab, settings_tab = st.tabs(
    [
        "✦ Chat",
        "📄 Document",
        "⚙ Settings",
    ]
)


# ============================================================
# CHAT TAB
# ============================================================

with chat_tab:

    # --------------------------------------------------------
    # PROCESS PENDING SUGGESTED QUESTION
    # --------------------------------------------------------

    if st.session_state.pending_question:

        pending_question = (
            st.session_state.pending_question
        )

        st.session_state.pending_question = None

        st.session_state.messages.append(
            {
                "role": "user",
                "content": pending_question,
            }
        )

        start_time = time.time()

        answer = answer_question(pending_question)

        elapsed = (
            time.time()
            - start_time
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.session_state.response_times.append(
            elapsed
        )

        st.rerun()


    # --------------------------------------------------------
    # EMPTY CHAT
    # --------------------------------------------------------

    if not st.session_state.messages:

        st.html(
            """
            <div class="hero">

                <div class="hero-badge">
                    ✦ AI DOCUMENT ASSISTANT
                </div>

                <div class="hero-title">
                    What would you like to recall?
                </div>

                <div class="hero-description">
                    Ask questions about your documents and
                    get answers grounded in information
                    retrieved from your knowledge base.
                </div>

            </div>
            """
        )


        if not st.session_state.document_name:

            st.info(
                "Upload a PDF from the sidebar to start chatting with your documents."
            )


        st.markdown(
            "### Try asking"
        )


        card1, card2, card3 = st.columns(3)


        # ----------------------------------------------------
        # SUMMARY CARD
        # ----------------------------------------------------

        with card1:

            st.html(
                """
                <div class="prompt-card">

                    <div class="prompt-icon">
                        ✦
                    </div>

                    <div class="prompt-title">
                        Summarize the document
                    </div>

                    <div class="prompt-description">
                        Get a concise overview of the
                        most important points.
                    </div>

                </div>
                """
            )


            if st.button(
                "Use prompt",
                key="summary_prompt",
                use_container_width=True,
            ):

                add_question(
                    "Summarize the document."
                )

                st.rerun()


        # ----------------------------------------------------
        # KEY INFORMATION CARD
        # ----------------------------------------------------

        with card2:

            st.html(
                """
                <div class="prompt-card">

                    <div class="prompt-icon">
                        ⌕
                    </div>

                    <div class="prompt-title">
                        Find key information
                    </div>

                    <div class="prompt-description">
                        Identify the most important facts
                        and findings.
                    </div>

                </div>
                """
            )


            if st.button(
                "Use prompt",
                key="key_information_prompt",
                use_container_width=True,
            ):

                add_question(
                    "What are the key findings in this document?"
                )

                st.rerun()


        # ----------------------------------------------------
        # MAIN TOPIC CARD
        # ----------------------------------------------------

        with card3:

            st.html(
                """
                <div class="prompt-card">

                    <div class="prompt-icon">
                        ◈
                    </div>

                    <div class="prompt-title">
                        Explain the main topic
                    </div>

                    <div class="prompt-description">
                        Understand what the document
                        is mainly about.
                    </div>

                </div>
                """
            )


            if st.button(
                "Use prompt",
                key="main_topic_prompt",
                use_container_width=True,
            ):

                add_question(
                    "What is the main topic of this document?"
                )

                st.rerun()


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.messages:

        if message["role"] == "user":

            # IMPORTANT:
            # No emoji avatar here.
            # Some Streamlit versions interpret emoji
            # strings as image paths.
            with st.chat_message("user"):

                st.markdown(
                    message["content"]
                )

        else:

            # IMPORTANT:
            # No custom emoji avatar here either.
            with st.chat_message("assistant"):

                st.markdown(
                    message["content"]
                )


    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask anything about your documents..."
    )


    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        start_time = time.time()


        # No avatar parameter.
        with st.chat_message("assistant"):

            answer = answer_question(question)

            elapsed = (
                time.time()
                - start_time
            )


            st.markdown(
                answer
            )


            st.caption(
                f"Generated in {elapsed:.2f}s"
            )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


        st.session_state.response_times.append(
            elapsed
        )


        st.rerun()


# ============================================================
# DOCUMENT TAB
# ============================================================

with document_tab:

    st.markdown(
        "### 📄 Document workspace"
    )


    st.caption(
        "View the document associated with your RecallAI session."
    )


    if st.session_state.document_name:

        left, right = st.columns(
            [2, 1]
        )


        with left:

            safe_name = escape(
                st.session_state.document_name
            )

            st.html(
                f"""
                <div class="hero">

                    <div class="hero-badge">
                        ✓ DOCUMENT READY
                    </div>

                    <div class="hero-title">
                        {safe_name}
                    </div>

                    <div class="hero-description">
                        This document is available in the
                        current RecallAI workspace.
                    </div>

                </div>
                """
            )


        with right:

            st.metric(
                "File size",
                f"{st.session_state.document_size:.2f} MB",
            )


            st.metric(
                "Messages",
                len(st.session_state.messages),
            )


        st.markdown(
            "### RAG pipeline"
        )


        st.html(
            """
            <div class="pipeline">

                <div class="pipeline-item">
                    📄 PDF
                </div>

                <div class="arrow">
                    →
                </div>

                <div class="pipeline-item">
                    📝 Text
                </div>

                <div class="arrow">
                    →
                </div>

                <div class="pipeline-item">
                    ✂ Chunks
                </div>

                <div class="arrow">
                    →
                </div>

                <div class="pipeline-item">
                    🧠 Embeddings
                </div>

                <div class="arrow">
                    →
                </div>

                <div class="pipeline-item">
                    🗃 ChromaDB
                </div>

                <div class="arrow">
                    →
                </div>

                <div class="pipeline-item">
                    🔎 Retrieval
                </div>

                <div class="arrow">
                    →
                </div>

                <div class="pipeline-item">
                    ✦ Mistral
                </div>

            </div>
            """
        )


    else:

        st.html(
            """
            <div class="hero">

                <div class="hero-badge">
                    📄 DOCUMENT WORKSPACE
                </div>

                <div class="hero-title">
                    No document loaded
                </div>

                <div class="hero-description">
                    Upload a PDF from the sidebar to begin
                    working with your documents.
                </div>

            </div>
            """
        )


# ============================================================
# SETTINGS TAB
# ============================================================

with settings_tab:

    st.markdown(
        "### ⚙ RecallAI settings"
    )


    st.caption(
        "View your current AI configuration and session information."
    )


    left, right = st.columns(2)


    with left:

        st.markdown(
            "#### 🧠 Language model"
        )


        st.info(
            "Mistral, via the hosted Mistral API."
        )


        st.write(
            f"Temperature: **{temperature:.1f}**"
        )


    with right:

        st.markdown(
            "#### 🔎 Retrieval"
        )


        st.info(
            "ChromaDB vector retrieval."
        )


        st.write(
            f"Retrieved chunks: **{top_k}**"
        )


    st.divider()


    st.markdown(
        "#### 🏗 RAG architecture"
    )


    st.html(
        """
        <div class="pipeline">

            <div class="pipeline-item">
                📄 Document
            </div>

            <div class="arrow">
                →
            </div>

            <div class="pipeline-item">
                ✂ Chunking
            </div>

            <div class="arrow">
                →
            </div>

            <div class="pipeline-item">
                🧠 Embeddings
            </div>

            <div class="arrow">
                →
            </div>

            <div class="pipeline-item">
                🗃 ChromaDB
            </div>

            <div class="arrow">
                →
            </div>

            <div class="pipeline-item">
                🔎 Retriever
            </div>

            <div class="arrow">
                →
            </div>

            <div class="pipeline-item">
                ✦ Mistral
            </div>

        </div>
        """
    )


    st.divider()


    st.markdown(
        "#### 📊 Session statistics"
    )


    stat1, stat2, stat3 = st.columns(3)


    with stat1:

        st.metric(
            "Messages",
            len(st.session_state.messages),
        )


    with stat2:

        st.metric(
            "Responses",
            len(st.session_state.response_times),
        )


    with stat3:

        if st.session_state.response_times:

            average_time = (
                sum(
                    st.session_state.response_times
                )
                /
                len(
                    st.session_state.response_times
                )
            )

        else:

            average_time = 0


        st.metric(
            "Average response",
            f"{average_time:.2f}s",
        )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">
        ✦ RecallAI
        &nbsp;•&nbsp;
        Retrieval-Augmented Generation
        &nbsp;•&nbsp;
        Mistral + ChromaDB
    </div>
    """
)