"""LangChain-based RAG engine — drop-in alternative to the native backend pipeline.

Provides a `LangChainRAGEngine` class that mirrors the native API so that
server.py and frontend.py can switch between engines seamlessly.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from collections import deque
from io import BytesIO
from typing import Any

import numpy as np

from backend import (
    BACKEND_GROQ,
    BACKEND_OLLAMA,
    DEFAULT_RETRIEVER_MODE,
    PipelineError,
    _citation_instruction,
    _extract_visual_hint,
    _format_citations,
    _record_telemetry,
    build_bm25_index,
    collect_visuals,
    compute_file_hash,
    extract_text_from_pdf,
    get_telemetry_summary,
    _TELEMETRY_MAX_ENTRIES,
)

logger = logging.getLogger("docmind.langchain")

# ── Graceful LangChain imports ────────────────────────────
LANGCHAIN_AVAILABLE = False
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS as LangChainFAISS
    from langchain_core.documents import Document
    from langchain_groq import ChatGroq
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass


class LangChainRAGEngine:
    """LangChain-based RAG engine with identical interface to the native pipeline."""

    def __init__(self) -> None:
        self._embeddings = None

    def _ensure_available(self) -> None:
        if not LANGCHAIN_AVAILABLE:
            raise PipelineError(
                "LangChain packages are not installed. "
                "Run: pip install langchain langchain-community langchain-groq "
                "langchain-ollama langchain-huggingface langchain-text-splitters"
            )

    def _get_embeddings(self) -> Any:
        if self._embeddings is None:
            self._ensure_available()
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def _get_llm(self, backend: str) -> Any:
        """Return a LangChain LLM instance for the given backend."""
        self._ensure_available()
        if backend == BACKEND_OLLAMA:
            return ChatOllama(model="mistral", temperature=0.5, timeout=120)
        from backend import DEFAULT_GROQ_MODEL
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise PipelineError("GROQ_API_KEY is missing in .env.")
        return ChatGroq(
            model=DEFAULT_GROQ_MODEL,
            api_key=api_key,
            temperature=0.5,
        )

    # ── Document Preparation ──────────────────────────────

    def prepare_document(self, file_bytes: bytes, file_name: str) -> dict[str, Any]:
        """Process PDF using LangChain splitter and FAISS vectorstore."""
        self._ensure_available()
        t_start = time.perf_counter()

        # Extract pages using shared native function
        pages = extract_text_from_pdf(file_bytes)
        full_text = "\n\n".join(p["text"] for p in pages)

        # Build LangChain Documents with page metadata
        lc_documents: list[Document] = []
        for page_info in pages:
            lc_documents.append(Document(
                page_content=page_info["text"],
                metadata={"page": page_info["page"]},
            ))

        # Split with RecursiveCharacterTextSplitter
        # ~1100 chars ≈ 220 words to match native chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1100,
            chunk_overlap=200,
            length_function=len,
        )
        split_docs = splitter.split_documents(lc_documents)

        if not split_docs:
            raise PipelineError("No chunks produced from the PDF.")

        # Build structured chunk dicts (matching native format)
        chunks: list[dict[str, Any]] = []
        for idx, doc in enumerate(split_docs):
            page_num = doc.metadata.get("page", 1)
            chunks.append({
                "chunk_id": idx,
                "pages": [page_num],
                "text": doc.page_content,
                "word_count": len(doc.page_content.split()),
            })

        # Build LangChain FAISS vectorstore
        embeddings_model = self._get_embeddings()
        t_embed_start = time.perf_counter()
        vectorstore = LangChainFAISS.from_documents(split_docs, embeddings_model)
        embedding_ms = (time.perf_counter() - t_embed_start) * 1000

        # Also build BM25 for hybrid search
        bm25_index = build_bm25_index(chunks)

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "LangChain document prepared: %s | %d pages, %d chunks | embed=%.0fms total=%.0fms",
            file_name, len(pages), len(chunks), embedding_ms, total_ms,
        )

        return {
            "name": file_name,
            "hash": compute_file_hash(file_bytes),
            "text": full_text,
            "pages": pages,
            "chunks": chunks,
            "faiss_index": None,  # Not used — vectorstore handles search
            "bm25_index": bm25_index,
            "embeddings": None,
            "lc_vectorstore": vectorstore,
            "chunk_count": len(chunks),
            "page_count": len(pages),
            "telemetry_log": deque(maxlen=_TELEMETRY_MAX_ENTRIES),
            "ingestion_ms": total_ms,
            "embedding_ms": embedding_ms,
        }

    # ── Retrieval ─────────────────────────────────────────

    def retrieve_chunks(
        self,
        query: str,
        document: dict[str, Any],
        top_k: int = 3,
        retriever_mode: str = "dense",
    ) -> list[dict[str, Any]]:
        """Retrieve chunks via LangChain FAISS vectorstore."""
        self._ensure_available()
        chunks = document.get("chunks") or []
        vectorstore = document.get("lc_vectorstore")

        if not chunks:
            return []

        t_start = time.perf_counter()
        results: list[dict[str, Any]] = []

        if vectorstore is not None and retriever_mode in ("dense", "hybrid"):
            # LangChain similarity search with scores
            docs_and_scores = vectorstore.similarity_search_with_score(query, k=min(top_k, len(chunks)))
            for rank, (doc, score) in enumerate(docs_and_scores, start=1):
                # Find matching chunk by text
                chunk_match = None
                for c in chunks:
                    if c["text"][:100] == doc.page_content[:100]:
                        chunk_match = c
                        break
                if chunk_match is None:
                    chunk_match = {
                        "chunk_id": rank - 1,
                        "pages": [doc.metadata.get("page", 1)],
                        "text": doc.page_content,
                        "word_count": len(doc.page_content.split()),
                    }
                results.append({
                    **chunk_match,
                    "score": round(float(1.0 / (1.0 + score)), 4),  # Convert distance to similarity
                    "rank": rank,
                })

        if not results:
            # Fallback to first chunks
            for i, c in enumerate(chunks[:top_k]):
                results.append({**c, "score": 0.0, "rank": i + 1})

        logger.info("LangChain retrieved %d chunks in %.0fms", len(results), (time.perf_counter() - t_start) * 1000)
        return results

    # ── LLM Call ──────────────────────────────────────────

    def _call_llm(self, prompt: str, backend: str) -> dict[str, Any]:
        """Invoke LangChain LLM and return content + token counts."""
        llm = self._get_llm(backend)
        try:
            response = llm.invoke(prompt)
            content = response.content.strip() if response.content else ""
            if not content:
                raise PipelineError("LLM returned an empty response.")

            # Extract token usage if available
            usage = getattr(response, "usage_metadata", {}) or {}
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)

            return {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"LangChain LLM call failed: {exc}") from exc

    # ── Content Generation ────────────────────────────────

    def generate_content(
        self,
        choice: str,
        difficulty: str,
        backend: str,
        document: dict[str, Any],
        retriever_mode: str = DEFAULT_RETRIEVER_MODE,
    ) -> dict[str, Any]:
        """Generate study content with citations — matches native API."""
        if not document or not document.get("chunks"):
            raise PipelineError("Upload and process a PDF before generating content.")

        query_id = str(uuid.uuid4())[:8]
        t_start = time.perf_counter()

        retrieval_queries = {
            "Summary": f"main concepts and important points for a {difficulty.lower()} study summary",
            "Important Questions": f"{difficulty.lower()} level exam questions and important theory topics",
            "Viva Questions": f"{difficulty.lower()} level viva questions and oral exam topics",
            "MCQs": f"{difficulty.lower()} level multiple choice questions and objective concepts",
        }

        t_ret_start = time.perf_counter()
        retrieved = self.retrieve_chunks(retrieval_queries[choice], document, top_k=3, retriever_mode=retriever_mode)
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

        # Build context with page markers
        context_parts = []
        for chunk in retrieved:
            page_label = ", ".join(str(p) for p in chunk["pages"])
            context_parts.append(f"[Page {page_label}]:\n{chunk['text']}")
        context = "\n\n---\n\n".join(context_parts) if context_parts else document["text"][:4000]

        task_map = {
            "Summary": f"Create a {difficulty.lower()} level study summary with clear headings and concise bullet points.",
            "Important Questions": f"Create 7 {difficulty.lower()} level exam questions. Include a short answer below each question.",
            "Viva Questions": f"Create 7 {difficulty.lower()} level viva questions. Include a compact answer below each question.",
            "MCQs": f"Create 7 {difficulty.lower()} level MCQs. Each question must have exactly four options labeled A to D and one correct answer.",
        }

        prompt = (
            "You are a helpful academic study assistant.\n"
            "Use only the provided PDF context. Do not invent topics that are not supported by the text.\n"
            + _citation_instruction()
            + f"Task:\n{task_map[choice]}\n\n"
            f"PDF context:\n{context}"
        )

        t_llm_start = time.perf_counter()
        llm_result = self._call_llm(prompt, backend)
        llm_ms = (time.perf_counter() - t_llm_start) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000
        citations = _format_citations(retrieved)
        top_scores = [c["score"] for c in retrieved if c.get("score")]

        _record_telemetry(
            document, query_id=query_id, operation="generate",
            retrieval_ms=retrieval_ms, llm_ms=llm_ms, total_ms=total_ms,
            retriever_mode=retriever_mode, top_scores=top_scores,
            prompt_tokens=llm_result.get("prompt_tokens", 0),
            completion_tokens=llm_result.get("completion_tokens", 0),
        )

        return {
            "output": llm_result["content"],
            "citations": citations,
            "retriever_mode": retriever_mode,
        }

    # ── Q&A ───────────────────────────────────────────────

    def answer_question(
        self,
        question: str,
        backend: str,
        document: dict[str, Any],
        generated_output: str = "",
        retriever_mode: str = DEFAULT_RETRIEVER_MODE,
    ) -> dict[str, Any]:
        """Answer a question with RAG context, citations, and visuals."""
        if not question.strip():
            raise PipelineError("Ask a non-empty question.")

        query_id = str(uuid.uuid4())[:8]
        t_start = time.perf_counter()

        t_ret_start = time.perf_counter()
        retrieved = self.retrieve_chunks(question.strip(), document, top_k=3, retriever_mode=retriever_mode)
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

        context_parts = []
        if retrieved:
            labelled = []
            for chunk in retrieved:
                page_label = ", ".join(str(p) for p in chunk["pages"])
                labelled.append(f"[Page {page_label}]:\n{chunk['text']}")
            context_parts.append("Relevant PDF context:\n" + "\n\n".join(labelled))
        if generated_output.strip():
            context_parts.append("Generated content so far:\n" + generated_output.strip())
        if not context_parts and document.get("text"):
            context_parts.append("PDF context:\n" + document["text"][:4000])

        prompt = (
            "You are a helpful study assistant. Answer the question using the provided context.\n"
            "If the topic is not in the context, answer from your general knowledge and say so.\n"
            + _citation_instruction()
            + "After your answer, on a NEW LINE, add exactly one of:\n"
            "- `VISUAL_HINT: <exact topic name>` — the specific topic a diagram would illustrate\n"
            "- `VISUAL_HINT: none` — only if the question is purely factual with no useful diagram\n\n"
            f"{chr(10).join(context_parts)}\n\n"
            f"Question: {question.strip()}\n"
            "Answer:"
        )

        t_llm_start = time.perf_counter()
        llm_result = self._call_llm(prompt, backend)
        llm_ms = (time.perf_counter() - t_llm_start) * 1000

        answer_text, visual_hint = _extract_visual_hint(llm_result["content"])
        visuals = collect_visuals(question, answer_text, visual_hint)

        total_ms = (time.perf_counter() - t_start) * 1000
        citations = _format_citations(retrieved)
        top_scores = [c["score"] for c in retrieved if c.get("score")]

        _record_telemetry(
            document, query_id=query_id, operation="chat",
            retrieval_ms=retrieval_ms, llm_ms=llm_ms, total_ms=total_ms,
            retriever_mode=retriever_mode, top_scores=top_scores,
            prompt_tokens=llm_result.get("prompt_tokens", 0),
            completion_tokens=llm_result.get("completion_tokens", 0),
        )

        return {
            "content": answer_text,
            "visuals": visuals,
            "visual_hint": visual_hint or "",
            "citations": citations,
        }

    # ── Flashcards ────────────────────────────────────────

    def generate_flashcards(
        self,
        count: int,
        difficulty: str,
        backend: str,
        document: dict[str, Any],
        retriever_mode: str = DEFAULT_RETRIEVER_MODE,
    ) -> list[dict[str, str]]:
        """Generate flashcards — matches native API."""
        import re
        if not document or not document.get("chunks"):
            raise PipelineError("Upload and process a PDF before generating flashcards.")

        count = max(3, min(count, 20))
        query = f"{difficulty.lower()} level key concepts definitions and important facts"
        chunks = self.retrieve_chunks(query, document, top_k=5, retriever_mode=retriever_mode)
        context = "\n\n---\n\n".join(c["text"] for c in chunks) if chunks else document["text"][:5000]

        prompt = (
            f"You are a study assistant. Generate exactly {count} flashcards from the content below.\n"
            f"Difficulty: {difficulty}\n\n"
            "Rules:\n"
            "- Each flashcard has a QUESTION on the front and a concise ANSWER on the back\n"
            "- Questions should test understanding, not just recall\n"
            "- Answers should be 1-3 sentences maximum\n"
            "- Cover different topics from the content\n"
            "- Output ONLY in this exact format, one per line:\n"
            "Q: <question text>\n"
            "A: <answer text>\n\n"
            f"Content:\n{context}\n\n"
            f"Generate {count} flashcards now:"
        )

        llm_result = self._call_llm(prompt, backend)
        raw = llm_result["content"]

        cards: list[dict[str, str]] = []
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        i = 0
        while i < len(lines) and len(cards) < count:
            line = lines[i]
            if line.startswith("Q:") or line.startswith("Q.") or line.lower().startswith("question"):
                q = re.sub(r"^(Q[:.]\s*|Question\s*\d*[:.]\s*)", "", line, flags=re.IGNORECASE).strip()
                a = ""
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.startswith("A:") or next_line.startswith("A.") or next_line.lower().startswith("answer"):
                        a = re.sub(r"^(A[:.]\s*|Answer\s*\d*[:.]\s*)", "", next_line, flags=re.IGNORECASE).strip()
                        i += 1
                if q and a:
                    cards.append({"question": q, "answer": a})
            i += 1

        if not cards:
            raise PipelineError("Could not parse flashcards from the AI response. Try again.")
        return cards


# ── Module-level singleton ────────────────────────────────
langchain_engine = LangChainRAGEngine()
