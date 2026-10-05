"""Optional Streamlit UI for development and quick testing.

The primary production interface is the FastAPI SPA (run: python app.py).
To use this Streamlit UI instead, run:  streamlit run frontend.py
"""
from __future__ import annotations

import streamlit as st

from backend import (
    BACKEND_GROQ,
    BACKEND_OPTIONS,
    DIFFICULTY_OPTIONS,
    MODEL_LABELS,
    OUTPUT_OPTIONS,
    PipelineError,
    answer_question,
    compute_file_hash,
    create_pdf,
    generate_content,
    prepare_document,
    web_visual_search_enabled,
    RETRIEVER_MODES,
    DEFAULT_RETRIEVER_MODE,
    get_telemetry_summary,
)


STATE_DEFAULTS = {
    "output": "",
    "citations": [],
    "chat_history": [],
    "backend": BACKEND_GROQ,
    "document": None,
    "document_error": "",
    "generation_error": "",
    "last_uploaded_hash": "",
    "chat_input": "",
    "retriever_mode": DEFAULT_RETRIEVER_MODE,
    "framework": "native",
}


def init_state() -> None:
    for key, default in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_document_state() -> None:
    st.session_state.document = None
    st.session_state.document_error = ""
    st.session_state.generation_error = ""
    st.session_state.output = ""
    st.session_state.citations = []
    st.session_state.chat_history = []
    st.session_state.last_uploaded_hash = ""


def process_uploaded_file(uploaded_file) -> None:
    if uploaded_file is None:
        return

    file_bytes = uploaded_file.getvalue()
    file_hash = compute_file_hash(file_bytes)

    if st.session_state.last_uploaded_hash == file_hash and st.session_state.document is not None:
        return

    with st.spinner("Reading PDF and building the search index..."):
        try:
            st.session_state.document = prepare_document(file_bytes, uploaded_file.name)
            st.session_state.document_error = ""
            st.session_state.generation_error = ""
            st.session_state.output = ""
            st.session_state.citations = []
            st.session_state.chat_history = []
            st.session_state.last_uploaded_hash = file_hash
        except Exception as exc:
            reset_document_state()
            message = str(exc) if isinstance(exc, PipelineError) else f"Unexpected processing error: {exc}"
            st.session_state.document_error = message


def render_styles() -> None:
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', sans-serif !important;
            background: linear-gradient(135deg, #020617, #0f172a, #020617) !important;
            color: #e2e8f0 !important;
        }
        .block-container {
            padding: 2rem 2.5rem !important;
            max-width: 1400px !important;
        }
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(#2563eb, #38bdf8);
            border-radius: 10px;
        }
        .hero-title {
            font-size: clamp(1.8rem, 4vw, 3rem);
            font-weight: 800;
            letter-spacing: -1.5px;
            line-height: 1.1;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #ffffff, #e2e8f0, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-sub {
            font-size: 0.95rem;
            color: #94a3b8;
            max-width: 560px;
            line-height: 1.7;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 22px;
            padding: 1.8rem;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            margin-bottom: 1.5rem;
        }
        .card-label {
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #38bdf8;
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .card-label::before {
            content: '';
            width: 3px;
            height: 14px;
            background: linear-gradient(#2563eb, #38bdf8);
            border-radius: 99px;
        }
        .backend-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.68rem;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 999px;
            margin-top: 6px;
            background: rgba(167, 139, 250, 0.12);
            color: #c4b5fd;
            border: 1px solid rgba(167, 139, 250, 0.25);
        }
        .doc-status {
            background: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.25);
            border-radius: 14px;
            padding: 14px 16px;
            margin-top: 12px;
        }
        .doc-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #e2e8f0;
        }
        .doc-meta {
            font-size: 0.75rem;
            color: #6ee7b7;
            margin-top: 4px;
        }
        .rag-tag {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            margin-top: 10px;
            font-size: 0.65rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 999px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.25);
            color: #34d399;
        }
        .output-shell {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(56, 189, 248, 0.15);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            max-height: 520px;
            overflow-y: auto;
        }
        .chat-shell {
            max-height: 330px;
            overflow-y: auto;
            padding-right: 4px;
        }
        .chat-empty {
            text-align: center;
            color: #64748b;
            font-size: 0.8rem;
            padding: 1.5rem 0;
        }
        .chat-helper {
            font-size: 0.75rem;
            color: #94a3b8;
            margin-bottom: 1rem;
            line-height: 1.6;
        }
        .visual-note {
            font-size: 0.75rem;
            color: #94a3b8;
            margin-top: 0.35rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, #2563eb, #1d4ed8, #38bdf8) !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 12px 24px !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            width: 100% !important;
            box-shadow: 0 4px 20px rgba(37, 99, 235, 0.4) !important;
        }
        .stDownloadButton > button {
            background: rgba(255, 255, 255, 0.04) !important;
            color: #38bdf8 !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            border-radius: 14px !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            width: 100% !important;
        }
        div[data-baseweb="select"] > div,
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            color: #e2e8f0 !important;
        }
        .stFileUploader > div {
            background: rgba(37, 99, 235, 0.04) !important;
            border: 2px dashed rgba(56, 189, 248, 0.25) !important;
            border-radius: 18px !important;
        }
        label {
            color: #94a3b8 !important;
            font-size: 0.78rem !important;
            font-weight: 500 !important;
        }
        .citation-pill {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 12px;
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            font-size: 0.7rem;
            font-weight: 700;
        }
        .citation-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            backdrop-filter: blur(8px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_citations(citations: list) -> None:
    if not citations:
        return
    with st.expander('Source Citations'):
        for c in citations:
            score = c.get("score", 0)
            page = c.get("page", "N/A")
            chunk = c.get("chunk_id", "N/A")
            text = c.get("text", "")
            st.markdown(f'''
                <div class="citation-card">
                    <div style="margin-bottom: 8px;">
                        <span class="citation-pill">Page {page}</span>
                        <span style="font-size: 0.75rem; color: #94a3b8; margin-left: 8px;">Chunk: {chunk} | Score: {score:.2f}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1;">{text}</div>
                </div>
            ''', unsafe_allow_html=True)


def render_chat(document_ready: bool) -> None:
    st.markdown('<div class="glass-card"><div class="card-label">Ask About This Content</div>', unsafe_allow_html=True)
    if web_visual_search_enabled():
        st.markdown(
            '<div class="chat-helper">Visual assist is on: common diagrams load from the local library, and other visual hints fall back to Unsplash.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="chat-helper">Visual assist is on for local diagrams. Add <code>UNSPLASH_ACCESS_KEY</code> to <code>.env</code> to enable web image fallback.</div>',
            unsafe_allow_html=True,
        )

    if not st.session_state.chat_history:
        st.markdown('<div class="chat-empty">Ask follow-up questions about the PDF or generated content.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                for visual in message.get("visuals", []):
                    if visual["type"] == "local":
                        st.image(visual["image_path"], caption=visual["title"], width="stretch")
                        st.markdown(f'<div class="visual-note">{visual["caption"]}</div>', unsafe_allow_html=True)
                    elif visual["type"] == "remote":
                        st.image(visual["image_url"], caption=visual["title"], width="stretch")
                        photographer = visual.get("photographer", "Unsplash photographer")
                        photographer_url = visual.get("photographer_url", "")
                        source_url = visual.get("source_url", "")
                        if photographer_url and source_url:
                            st.markdown(
                                f'[Photo by {photographer}]({photographer_url}) on [Unsplash]({source_url})',
                                unsafe_allow_html=False,
                            )
                render_citations(message.get("citations", []))
        st.markdown("</div>", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        input_col, button_col = st.columns([5, 1])
        with input_col:
            user_input = st.text_input(
                "Ask a follow-up question",
                placeholder="Ask a follow-up question...",
                label_visibility="collapsed",
                key="chat_input",
                disabled=not document_ready,
            )
        with button_col:
            send = st.form_submit_button("Send", disabled=not document_ready)

    if send and user_input.strip():
        try:
            with st.spinner("Generating answer..."):
                reply = answer_question(
                    question=user_input.strip(),
                    backend=st.session_state.backend,
                    document=st.session_state.document or {},
                    generated_output=st.session_state.output,
                    retriever_mode=st.session_state.retriever_mode,
                )
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": reply["content"],
                    "visuals": reply.get("visuals", []),
                    "citations": reply.get("citations", []),
                }
            )
            st.rerun()
        except Exception as exc:
            message = str(exc) if isinstance(exc, PipelineError) else f"Unexpected chat error: {exc}"
            st.error(message)

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="DocMind AI", layout="wide")
    init_state()
    render_styles()

    st.markdown(
        """
        <div style="padding: 1rem 0 2.25rem;">
          <h1 class="hero-title">Learn Smarter,<br>Not Harder</h1>
          <p class="hero-sub">
            Upload your PDF and generate summaries, exam questions, viva prep, and MCQs with a RAG-powered study pipeline.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, top_right = st.columns([2, 1])
    with top_right:
        st.markdown('<div class="card-label" style="margin-bottom: 8px;">Backend</div>', unsafe_allow_html=True)
        st.session_state.backend = st.selectbox(
            "Backend",
            BACKEND_OPTIONS,
            index=BACKEND_OPTIONS.index(st.session_state.backend),
            label_visibility="collapsed",
        )
        st.markdown(
            f'<div class="backend-pill">{MODEL_LABELS[st.session_state.backend]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="card-label" style="margin-top: 16px; margin-bottom: 8px;">Retrieval Strategy</div>', unsafe_allow_html=True)
        retriever_map = {'Hybrid BM25+FAISS': 'hybrid', 'Dense FAISS': 'dense', 'Sparse BM25': 'sparse'}
        inv_retriever_map = {v: k for k, v in retriever_map.items()}
        current_retriever_label = inv_retriever_map.get(st.session_state.retriever_mode, 'Hybrid BM25+FAISS')
        retriever_choice = st.selectbox(
            "Retrieval Strategy",
            list(retriever_map.keys()),
            index=list(retriever_map.keys()).index(current_retriever_label),
            label_visibility="collapsed",
        )
        st.session_state.retriever_mode = retriever_map[retriever_choice]

        st.markdown('<div class="card-label" style="margin-top: 16px; margin-bottom: 8px;">Engine Framework</div>', unsafe_allow_html=True)
        framework_map = {'Native Python': 'native', 'LangChain': 'langchain'}
        inv_framework_map = {v: k for k, v in framework_map.items()}
        current_framework_label = inv_framework_map.get(st.session_state.framework, 'Native Python')
        framework_choice = st.selectbox(
            "Engine Framework",
            list(framework_map.keys()),
            index=list(framework_map.keys()).index(current_framework_label),
            label_visibility="collapsed",
        )
        st.session_state.framework = framework_map[framework_choice]

        if st.session_state.framework == 'langchain':
            st.info("LangChain engine requires additional packages. Ensure langchain and related dependencies are installed.")

    st.markdown("<hr>", unsafe_allow_html=True)

    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        st.markdown('<div class="glass-card"><div class="card-label">Upload Document</div>', unsafe_allow_html=True)
        st.caption("Drag and drop your PDF here, or click to browse.")

        uploaded_file = st.file_uploader(
            "Upload PDF",
            type="pdf",
            label_visibility="collapsed",
        )
        process_uploaded_file(uploaded_file)

        if st.session_state.document_error:
            st.error(st.session_state.document_error)

        document = st.session_state.document
        document_ready = bool(document and document.get("chunks") and document.get("faiss_index") is not None)

        if document_ready:
            st.markdown(
                f"""
                <div class="doc-status">
                    <div class="doc-title">{document["name"]}</div>
                    <div class="doc-meta">Indexed successfully | {document["chunk_count"]} chunks | FAISS ready</div>
                </div>
                <div class="rag-tag">RAG active | {document["chunk_count"]} vectors</div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card"><div class="card-label">Configure Output</div>', unsafe_allow_html=True)
        choice = st.selectbox("Output Type", OUTPUT_OPTIONS, label_visibility="collapsed")
        difficulty = st.selectbox(
            "Difficulty",
            DIFFICULTY_OPTIONS,
            index=1,
            label_visibility="collapsed",
        )
        st.markdown("<br>", unsafe_allow_html=True)

        generate_disabled = not document_ready
        if st.button("Generate Content", disabled=generate_disabled):
            try:
                with st.spinner("Retrieving context and generating content..."):
                    result = generate_content(
                        choice=choice,
                        difficulty=difficulty,
                        backend=st.session_state.backend,
                        document=st.session_state.document or {},
                        retriever_mode=st.session_state.retriever_mode,
                    )
                    st.session_state.output = result.get("output", "")
                    st.session_state.citations = result.get("citations", [])
                    st.session_state.generation_error = ""
                    st.session_state.chat_history = []
            except Exception as exc:
                st.session_state.output = ""
                st.session_state.citations = []
                message = str(exc) if isinstance(exc, PipelineError) else f"Unexpected generation error: {exc}"
                st.session_state.generation_error = message

        if generate_disabled:
            st.caption("Upload a PDF to enable generation.")

        if st.session_state.generation_error:
            st.error(st.session_state.generation_error)

        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.output:
        st.markdown("<hr>", unsafe_allow_html=True)
        output_col, chat_col = st.columns([1.2, 0.8], gap="large")

        with output_col:
            st.markdown('<div class="glass-card"><div class="card-label">Generated Result</div>', unsafe_allow_html=True)
            st.markdown('<div class="output-shell">', unsafe_allow_html=True)
            st.markdown(st.session_state.output)
            st.markdown("</div>", unsafe_allow_html=True)
            
            render_citations(st.session_state.citations)
            
            st.markdown("<br>", unsafe_allow_html=True)

            pdf_buffer = create_pdf(st.session_state.output, choice)
            st.download_button(
                "Download as PDF",
                data=pdf_buffer,
                file_name="study_material.pdf",
                mime="application/pdf",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with chat_col:
            render_chat(document_ready=True)

    if document_ready:
        st.markdown("<hr>", unsafe_allow_html=True)
        with st.expander('📊 Pipeline Analytics'):
            summary = get_telemetry_summary(st.session_state.document)
            if summary:
                c1, c2, c3 = st.columns(3)
                c1.metric("Ingestion Time (ms)", f"{summary.get('ingestion_ms', 0):.0f}")
                c2.metric("Total Queries", summary.get("total_queries", 0))
                c3.metric("Mean Latency (ms)", f"{summary.get('mean_latency', 0):.2f}")
                
                st.markdown("#### Mode Distribution")
                dist = summary.get("mode_distribution", {})
                if dist:
                    st.write(dist)
                else:
                    st.write("No queries yet.")

                recent = summary.get("recent_telemetry", [])
                if recent:
                    st.markdown("#### Recent Queries")
                    st.dataframe(recent)
            else:
                st.write("No analytics available.")


if __name__ == "__main__":
    main()
