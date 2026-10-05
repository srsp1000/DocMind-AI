from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import threading
import uuid
from collections import deque
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from xml.sax.saxutils import escape

import faiss
import numpy as np
import pandas as pd
import PyPDF2
import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer
from sentence_transformers import SentenceTransformer


# ── Logging ──────────────────────────────────────────────
logger = logging.getLogger("docmind")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(_handler)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

BACKEND_GROQ = "Groq (Cloud)"
BACKEND_OLLAMA = "Ollama (Local)"
BACKEND_OPTIONS = (BACKEND_GROQ, BACKEND_OLLAMA)

OUTPUT_OPTIONS = ("Summary", "Important Questions", "Viva Questions", "MCQs")
DIFFICULTY_OPTIONS = ("Easy", "Medium", "Hard")

RETRIEVER_MODES = ("hybrid", "dense", "sparse")
DEFAULT_RETRIEVER_MODE = "hybrid"

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

MODEL_LABELS = {
    BACKEND_GROQ: f"{DEFAULT_GROQ_MODEL} | Cloud",
    BACKEND_OLLAMA: "mistral | Local",
}

DIAGRAMS_DIR = BASE_DIR / "assets" / "diagrams"
UNSPLASH_APP_NAME = "docmind"
VISUAL_HINT_RE = re.compile(r"(?im)^\s*VISUAL_HINT:\s*(.+?)\s*$")

LOCAL_DIAGRAM_LIBRARY = (
    {
        "id": "rnn",
        "title": "Recurrent Neural Network (RNN)",
        "file": "rnn.svg",
        "keywords": ("rnn", "recurrent neural network", "hidden state", "sequence model", "lstm", "gru"),
        "caption": "RNN architecture — processes sequences with recurrent connections.",
    },
    {
        "id": "cnn",
        "title": "Convolutional Neural Network (CNN)",
        "file": "cnn.svg",
        "keywords": ("cnn", "convolutional neural network", "convolution layer", "feature map", "pooling layer", "kernel", "filter"),
        "caption": "CNN architecture — extracts spatial features via convolution and pooling.",
    },
    {
        "id": "transformer",
        "title": "Transformer / Self-Attention",
        "file": "transformer.svg",
        "keywords": ("transformer", "self attention", "self-attention", "multi head attention", "encoder decoder", "bert", "gpt", "attention mechanism"),
        "caption": "Transformer architecture — uses self-attention for sequence modelling.",
    },
)

# Wikipedia image search — free, no API key needed
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# Topic → Wikipedia article title mapping for reliable image lookup
WIKI_TOPIC_MAP = {
    "neural network": "Artificial_neural_network",
    "deep learning": "Deep_learning",
    "machine learning": "Machine_learning",
    "backpropagation": "Backpropagation",
    "gradient descent": "Gradient_descent",
    "activation function": "Activation_function",
    "relu": "Rectifier_(neural_networks)",
    "sigmoid": "Sigmoid_function",
    "softmax": "Softmax_function",
    "overfitting": "Overfitting",
    "regularization": "Regularization_(mathematics)",
    "dropout": "Dilution_(neural_networks)",
    "batch normalization": "Batch_normalization",
    "convolutional neural network": "Convolutional_neural_network",
    "cnn": "Convolutional_neural_network",
    "recurrent neural network": "Recurrent_neural_network",
    "rnn": "Recurrent_neural_network",
    "lstm": "Long_short-term_memory",
    "long short-term memory": "Long_short-term_memory",
    "gru": "Gated_recurrent_unit",
    "transformer": "Transformer_(deep_learning_architecture)",
    "attention mechanism": "Attention_(machine_learning)",
    "self attention": "Attention_(machine_learning)",
    "bert": "BERT_(language_model)",
    "gpt": "Generative_pre-trained_transformer",
    "gan": "Generative_adversarial_network",
    "generative adversarial network": "Generative_adversarial_network",
    "autoencoder": "Autoencoder",
    "variational autoencoder": "Variational_autoencoder",
    "vae": "Variational_autoencoder",
    "reinforcement learning": "Reinforcement_learning",
    "q-learning": "Q-learning",
    "support vector machine": "Support_vector_machine",
    "svm": "Support_vector_machine",
    "decision tree": "Decision_tree_learning",
    "random forest": "Random_forest",
    "k-means": "K-means_clustering",
    "principal component analysis": "Principal_component_analysis",
    "pca": "Principal_component_analysis",
    "object detection": "Object_detection",
    "image segmentation": "Image_segmentation",
    "fine tuning": "Fine-tuning_(deep_learning)",
    "fine-tuning": "Fine-tuning_(deep_learning)",
    "transfer learning": "Transfer_learning",
    "natural language processing": "Natural_language_processing",
    "nlp": "Natural_language_processing",
    "word embedding": "Word_embedding",
    "word2vec": "Word2vec",
    "operating system": "Operating_system",
    "process scheduling": "Scheduling_(computing)",
    "memory management": "Memory_management",
    "virtual memory": "Virtual_memory",
    "tcp ip": "Internet_protocol_suite",
    "osi model": "OSI_model",
    "database": "Database",
    "sql": "SQL",
    "sorting algorithm": "Sorting_algorithm",
    "binary tree": "Binary_tree",
    "graph theory": "Graph_theory",
    "linked list": "Linked_list",
    "stack": "Stack_(abstract_data_type)",
    "queue": "Queue_(abstract_data_type)",
    "hash table": "Hash_table",
    "dynamic programming": "Dynamic_programming",
    "big o notation": "Big_O_notation",
}

# ── Telemetry rolling window ─────────────────────────────
_TELEMETRY_MAX_ENTRIES = 200
_telemetry_lock = threading.Lock()


class PipelineError(RuntimeError):
    """Raised when the PDF or generation pipeline cannot complete."""


# ══════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ══════════════════════════════════════════════════════════

def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _normalize_lookup_text(*parts: str) -> str:
    text = " ".join(part for part in parts if part)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _append_utm(url: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["utm_source"] = UNSPLASH_APP_NAME
    query["utm_medium"] = "referral"
    return urlunparse(parsed._replace(query=urlencode(query)))


def _unsplash_access_key() -> str:
    return os.getenv("UNSPLASH_ACCESS_KEY", "").strip()


def web_visual_search_enabled() -> bool:
    return bool(_unsplash_access_key())


def _extract_visual_hint(reply: str) -> tuple[str, str | None]:
    match = VISUAL_HINT_RE.search(reply)
    if not match:
        return reply.strip(), None

    visual_hint = match.group(1).strip()
    cleaned_reply = VISUAL_HINT_RE.sub("", reply).strip()
    if visual_hint.lower() == "none":
        return cleaned_reply, None
    return cleaned_reply, visual_hint


# ══════════════════════════════════════════════════════════
#  VISUAL SEARCH PIPELINE
# ══════════════════════════════════════════════════════════

def get_local_diagram_matches(*texts: str) -> list[dict[str, Any]]:
    lookup_text = _normalize_lookup_text(*texts)
    matches: list[dict[str, Any]] = []

    for entry in LOCAL_DIAGRAM_LIBRARY:
        if any(keyword in lookup_text for keyword in entry["keywords"]):
            image_path = DIAGRAMS_DIR / entry["file"]
            if image_path.exists():
                matches.append(
                    {
                        "type": "local",
                        "id": entry["id"],
                        "title": entry["title"],
                        "image_path": str(image_path.resolve()),
                        "caption": entry["caption"],
                    }
                )

    return matches


def search_unsplash_image(query: str) -> dict[str, Any] | None:
    access_key = _unsplash_access_key()
    if not access_key:
        return None
    try:
        response = requests.get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "page": 1, "per_page": 1, "orientation": "landscape", "content_filter": "high"},
            headers={"Accept-Version": "v1", "Authorization": f"Client-ID {access_key}"},
            timeout=15,
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            return None
        photo = results[0]
        image_url = photo.get("urls", {}).get("regular") or photo.get("urls", {}).get("small")
        source_url = photo.get("links", {}).get("html")
        photographer = photo.get("user", {}).get("name", "Unsplash photographer")
        photographer_url = photo.get("user", {}).get("links", {}).get("html")
        if not image_url or not source_url:
            return None
        return {
            "type": "remote", "provider": "Unsplash",
            "title": f"Image: {query}",
            "image_url": image_url,
            "caption": f"Photo by {photographer} on Unsplash.",
            "source_url": _append_utm(source_url),
            "photographer": photographer,
            "photographer_url": _append_utm(photographer_url) if photographer_url else "",
            "query": query,
        }
    except requests.RequestException:
        return None


# ── Wikipedia REST summary API — returns the main article image (most relevant) ──
WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/"

def _wiki_article_for_hint(hint: str) -> str | None:
    """Return the best Wikipedia article title for a given hint."""
    lookup = _normalize_lookup_text(hint)
    # 1. Exact match in our curated map
    for keyword, title in WIKI_TOPIC_MAP.items():
        if keyword in lookup:
            return title
    # 2. Wikipedia search API
    try:
        sr = requests.get(
            WIKIPEDIA_API,
            params={"action": "query", "list": "search", "srsearch": hint + " diagram",
                    "srlimit": 3, "format": "json"},
            timeout=8,
        )
        sr.raise_for_status()
        hits = sr.json().get("query", {}).get("search", [])
        if hits:
            return hits[0]["title"].replace(" ", "_")
    except requests.RequestException:
        pass
    return None


def search_wikipedia_summary_image(hint: str) -> dict[str, Any] | None:
    """Use Wikipedia REST summary API — returns the main thumbnail of the article.
    This is the most accurate image for a topic (e.g. fine-tuning → fine-tuning diagram)."""
    article = _wiki_article_for_hint(hint)
    if not article:
        return None
    try:
        resp = requests.get(
            WIKI_REST + article,
            headers={"User-Agent": "DocMind/1.0 (educational tool)"},
            timeout=10,
        )
        if resp.status_code == 404:
            # Try without underscores
            resp = requests.get(
                WIKI_REST + hint.replace(" ", "_"),
                headers={"User-Agent": "DocMind/1.0"},
                timeout=10,
            )
        resp.raise_for_status()
        data = resp.json()

        # Prefer original image over thumbnail for better quality
        original = data.get("originalimage", {})
        thumbnail = data.get("thumbnail", {})
        image_url = original.get("source") or thumbnail.get("source")

        if not image_url:
            return None

        title = data.get("title", article.replace("_", " "))
        description = data.get("description", "")
        extract = data.get("extract", "")[:120]
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{article}")

        return {
            "type": "wiki",
            "provider": "Wikipedia",
            "title": title,
            "image_url": image_url,
            "caption": description or extract or f"From Wikipedia: {title}",
            "source_url": page_url,
            "query": hint,
        }
    except requests.RequestException:
        return None


def search_wikimedia_commons(hint: str) -> dict[str, Any] | None:
    """Search Wikimedia Commons specifically for diagrams/illustrations.
    Better than Wikipedia article images for technical diagrams."""
    try:
        # Search Commons for diagram images
        search_query = f"{hint} diagram"
        resp = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": f"File:{search_query}",
                "gsrnamespace": 6,  # File namespace
                "gsrlimit": 5,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|mime",
                "format": "json",
            },
            headers={"User-Agent": "DocMind/1.0 (educational tool)"},
            timeout=10,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        if not pages:
            return None

        skip = ("icon", "logo", "flag", "button", "arrow", "stub", "portal", "symbol", "map")
        prefer = ("diagram", "architecture", "network", "model", "flow", "chart", "graph", "structure", "overview")

        candidates = []
        for page in pages.values():
            title = page.get("title", "").lower()
            if any(w in title for w in skip):
                continue
            imageinfo = (page.get("imageinfo") or [{}])[0]
            mime = imageinfo.get("mime", "")
            if not mime.startswith("image/"):
                continue
            url = imageinfo.get("url", "")
            if not url:
                continue
            # Score: prefer SVG and diagram-named files
            score = sum(1 for w in prefer if w in title)
            if mime == "image/svg+xml":
                score += 3
            candidates.append((score, page, imageinfo))

        if not candidates:
            return None

        # Pick highest scoring
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, best_page, best_info = candidates[0]

        image_url = best_info.get("url", "")
        meta = best_info.get("extmetadata", {})
        desc = re.sub(r"<[^>]+>", "", meta.get("ImageDescription", {}).get("value", "")).strip()[:140]
        file_title = best_page.get("title", "").replace("File:", "")
        commons_url = f"https://commons.wikimedia.org/wiki/{best_page.get('title','').replace(' ','_')}"

        return {
            "type": "wiki",
            "provider": "Wikimedia Commons",
            "title": file_title,
            "image_url": image_url,
            "caption": desc or f"Diagram from Wikimedia Commons",
            "source_url": commons_url,
            "query": hint,
        }
    except requests.RequestException:
        return None


def collect_visuals(question: str, answer_text: str, visual_hint: str | None) -> list[dict[str, Any]]:
    """
    Run Wikipedia + Wikimedia Commons in parallel, return best result.
    Priority: Local SVG → best of (Wikipedia + Commons) → Unsplash
    """
    if not visual_hint or visual_hint.lower() == "none":
        return []

    # 1. Local diagrams — instant, curated
    local = get_local_diagram_matches(visual_hint)
    if local:
        return local[:1]

    # 2. Run Wikipedia REST + Wikimedia Commons in parallel
    import concurrent.futures
    results: list[dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        wiki_future    = pool.submit(search_wikipedia_summary_image, visual_hint)
        commons_future = pool.submit(search_wikimedia_commons, visual_hint)

        wiki    = wiki_future.result()
        commons = commons_future.result()

    # Pick best: prefer Commons if it has "diagram" in title (more specific),
    # otherwise prefer Wikipedia (more authoritative)
    if wiki and commons:
        commons_title = (commons.get("title") or "").lower()
        if any(w in commons_title for w in ("diagram", "architecture", "flowchart", "chart", "flow")):
            results.append(commons)
        else:
            results.append(wiki)
    elif wiki:
        results.append(wiki)
    elif commons:
        results.append(commons)

    if results:
        return results

    # 3. Unsplash fallback
    unsplash = search_unsplash_image(visual_hint)
    if unsplash:
        return [unsplash]

    return []


# ══════════════════════════════════════════════════════════
#  EMBEDDING MODEL
# ══════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ══════════════════════════════════════════════════════════
#  PDF EXTRACTION — PAGE-LEVEL METADATA
# ══════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes) -> list[dict[str, Any]]:
    """Extract text from PDF preserving per-page metadata.

    Returns:
        A list of dicts:  [{"page": 1, "text": "..."}, ...]
        Page numbers are 1-indexed.
    """
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    pages: list[dict[str, Any]] = []
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            # Normalise whitespace within the page
            page_text = re.sub(r"[ \t]+", " ", page_text)
            page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()
            pages.append({"page": page_num, "text": page_text})

    if not pages:
        raise PipelineError(
            "No readable text was found in this PDF. If it is a scanned image PDF, add OCR first."
        )

    logger.info("Extracted %d pages from PDF", len(pages))
    return pages


def chunk_text(
    pages: list[dict[str, Any]],
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[dict[str, Any]]:
    """Split page-level text into overlapping word-based chunks while tracking page origin.

    Each chunk dict has:
        chunk_id (int): 0-indexed sequential id.
        pages (list[int]): List of page numbers this chunk spans.
        text (str): The chunk text.
        word_count (int): Number of words in the chunk.
    """
    # Build (word, page_number) pairs so we can track page boundaries
    word_page_pairs: list[tuple[str, int]] = []
    for page_info in pages:
        page_num = page_info["page"]
        words = page_info["text"].split()
        for w in words:
            word_page_pairs.append((w, page_num))

    if not word_page_pairs:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[dict[str, Any]] = []
    chunk_id = 0

    for start in range(0, len(word_page_pairs), step):
        window = word_page_pairs[start : start + chunk_size]
        if not window:
            continue

        text = " ".join(w for w, _ in window).strip()
        # Collect unique page numbers this chunk spans (preserving order)
        seen_pages: list[int] = []
        for _, p in window:
            if p not in seen_pages:
                seen_pages.append(p)

        chunks.append({
            "chunk_id": chunk_id,
            "pages": seen_pages,
            "text": text,
            "word_count": len(window),
        })
        chunk_id += 1

        if start + chunk_size >= len(word_page_pairs):
            break

    logger.info("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks


# ══════════════════════════════════════════════════════════
#  EMBEDDINGS & INDEXING
# ══════════════════════════════════════════════════════════

def create_embeddings(chunks: list[dict[str, Any]], model: SentenceTransformer | None = None) -> np.ndarray:
    """Generate L2-normalised embeddings for chunk texts."""
    texts = [c["text"] for c in chunks]
    if not texts:
        raise PipelineError("The uploaded PDF did not produce any searchable text chunks.")

    model = model or load_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise PipelineError("Embeddings could not be created for this PDF.")

    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise PipelineError("The FAISS index could not be created because no embeddings were available.")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def build_bm25_index(chunks: list[dict[str, Any]]) -> BM25Okapi:
    """Build a BM25 sparse lexical index from chunk texts."""
    tokenised_corpus = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenised_corpus)


# ══════════════════════════════════════════════════════════
#  HYBRID RETRIEVAL — BM25 + FAISS + RRF
# ══════════════════════════════════════════════════════════

def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
) -> list[int]:
    """Merge multiple ranked ID lists using Reciprocal Rank Fusion.

    RRF(d) = Σ  1 / (k + rank_i(d))  for each ranking i

    Args:
        ranked_lists: List of rankings, each being a list of chunk IDs (best-first).
        k: RRF smoothing constant (default 60, per the original paper).

    Returns:
        A single merged ranking of chunk IDs, best-first.
    """
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # Sort by descending RRF score
    return sorted(scores, key=lambda d: scores[d], reverse=True)


# ══════════════════════════════════════════════════════════
#  DOCUMENT PREPARATION
# ══════════════════════════════════════════════════════════

def prepare_document(file_bytes: bytes, file_name: str) -> dict[str, Any]:
    """Full ingestion pipeline: extract → chunk → embed → index (FAISS + BM25)."""
    t_start = time.perf_counter()

    pages = extract_text_from_pdf(file_bytes)
    full_text = "\n\n".join(p["text"] for p in pages)
    chunks = chunk_text(pages)

    t_embed_start = time.perf_counter()
    embeddings = create_embeddings(chunks)
    embedding_ms = (time.perf_counter() - t_embed_start) * 1000

    faiss_index = build_faiss_index(embeddings)
    bm25_index = build_bm25_index(chunks)

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "Document prepared: %s | %d pages, %d chunks | embed=%.0fms total=%.0fms",
        file_name, len(pages), len(chunks), embedding_ms, total_ms,
    )

    return {
        "name": file_name,
        "hash": compute_file_hash(file_bytes),
        "text": full_text,
        "pages": pages,
        "chunks": chunks,
        "embeddings": embeddings,
        "faiss_index": faiss_index,
        "bm25_index": bm25_index,
        "chunk_count": len(chunks),
        "page_count": len(pages),
        "telemetry_log": deque(maxlen=_TELEMETRY_MAX_ENTRIES),
        "ingestion_ms": total_ms,
        "embedding_ms": embedding_ms,
    }


# ══════════════════════════════════════════════════════════
#  RETRIEVAL — supports dense / sparse / hybrid modes
# ══════════════════════════════════════════════════════════

def retrieve_chunks(
    query: str,
    document: dict[str, Any],
    top_k: int = 3,
    retriever_mode: str = DEFAULT_RETRIEVER_MODE,
    model: SentenceTransformer | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k relevant chunks with scores and metadata.

    Args:
        retriever_mode: "dense" (FAISS only), "sparse" (BM25 only), or "hybrid" (RRF fusion).

    Returns:
        A list of dicts, each containing:
            chunk_id, pages, text, word_count, score, rank
    """
    chunks = document.get("chunks") or []
    if not chunks:
        return []

    t_start = time.perf_counter()
    retriever_mode = retriever_mode if retriever_mode in RETRIEVER_MODES else DEFAULT_RETRIEVER_MODE

    # Number of candidates to fetch from each retriever before fusion
    candidate_k = min(top_k * 3, len(chunks))

    # ── Dense retrieval (FAISS) ──
    dense_ids: list[int] = []
    dense_scores: dict[int, float] = {}
    faiss_index = document.get("faiss_index")
    if retriever_mode in ("dense", "hybrid") and faiss_index is not None:
        model = model or load_embedding_model()
        query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = np.asarray(query_embedding, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        actual_k = max(1, min(candidate_k, len(chunks)))
        scores, indices = faiss_index.search(query_embedding, actual_k)

        for rank_pos, raw_idx in enumerate(indices[0]):
            idx = int(raw_idx)
            if 0 <= idx < len(chunks):
                dense_ids.append(idx)
                dense_scores[idx] = float(scores[0][rank_pos])

    # ── Sparse retrieval (BM25) ──
    sparse_ids: list[int] = []
    sparse_scores: dict[int, float] = {}
    bm25_index = document.get("bm25_index")
    if retriever_mode in ("sparse", "hybrid") and bm25_index is not None:
        tokenised_query = query.lower().split()
        bm25_raw_scores = bm25_index.get_scores(tokenised_query)
        # Rank by descending BM25 score
        ranked = np.argsort(bm25_raw_scores)[::-1][:candidate_k]
        for idx in ranked:
            idx = int(idx)
            if bm25_raw_scores[idx] > 0:
                sparse_ids.append(idx)
                sparse_scores[idx] = float(bm25_raw_scores[idx])

    # ── Fusion ──
    if retriever_mode == "hybrid":
        ranked_lists = []
        if dense_ids:
            ranked_lists.append(dense_ids)
        if sparse_ids:
            ranked_lists.append(sparse_ids)
        if ranked_lists:
            final_ids = reciprocal_rank_fusion(ranked_lists, k=60)[:top_k]
        else:
            final_ids = list(range(min(top_k, len(chunks))))
    elif retriever_mode == "dense":
        final_ids = dense_ids[:top_k]
    else:  # sparse
        final_ids = sparse_ids[:top_k]

    # Fallback
    if not final_ids:
        final_ids = list(range(min(top_k, len(chunks))))

    retrieval_ms = (time.perf_counter() - t_start) * 1000

    # Build result with metadata
    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(final_ids, start=1):
        chunk = chunks[idx]
        # Pick the best available similarity score
        score = dense_scores.get(idx, sparse_scores.get(idx, 0.0))
        results.append({
            **chunk,
            "score": round(score, 4),
            "rank": rank,
        })

    logger.info(
        "Retrieved %d chunks (mode=%s) in %.0fms",
        len(results), retriever_mode, retrieval_ms,
    )

    return results


# ══════════════════════════════════════════════════════════
#  TELEMETRY — Pandas + NumPy analytics
# ══════════════════════════════════════════════════════════

def _record_telemetry(
    document: dict[str, Any],
    *,
    query_id: str,
    operation: str,
    embedding_ms: float = 0.0,
    retrieval_ms: float = 0.0,
    llm_ms: float = 0.0,
    total_ms: float = 0.0,
    retriever_mode: str = "",
    top_scores: list[float] | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Append a telemetry entry to the document's rolling log (thread-safe)."""
    entry = {
        "query_id": query_id,
        "timestamp": time.time(),
        "operation": operation,
        "embedding_ms": round(embedding_ms, 2),
        "retrieval_ms": round(retrieval_ms, 2),
        "llm_ms": round(llm_ms, 2),
        "total_ms": round(total_ms, 2),
        "retriever_mode": retriever_mode,
        "mean_similarity": round(float(np.mean(top_scores)), 4) if top_scores else 0.0,
        "max_similarity": round(float(np.max(top_scores)), 4) if top_scores else 0.0,
        "min_similarity": round(float(np.min(top_scores)), 4) if top_scores else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    log: deque = document.get("telemetry_log", deque(maxlen=_TELEMETRY_MAX_ENTRIES))
    with _telemetry_lock:
        log.append(entry)


def get_telemetry_summary(document: dict[str, Any]) -> dict[str, Any]:
    """Compute aggregate telemetry stats using Pandas and return a JSON-safe summary."""
    log: deque = document.get("telemetry_log", deque())
    if not log:
        return {
            "total_queries": 0,
            "summary": {},
            "recent": [],
            "ingestion_ms": document.get("ingestion_ms", 0),
            "chunk_count": document.get("chunk_count", 0),
            "page_count": document.get("page_count", 0),
        }

    with _telemetry_lock:
        entries = list(log)

    df = pd.DataFrame(entries)

    # Aggregate statistics
    summary: dict[str, Any] = {}
    for col in ("embedding_ms", "retrieval_ms", "llm_ms", "total_ms", "mean_similarity"):
        if col in df.columns:
            summary[col] = {
                "mean": round(float(df[col].mean()), 2),
                "median": round(float(df[col].median()), 2),
                "p95": round(float(df[col].quantile(0.95)), 2),
                "min": round(float(df[col].min()), 2),
                "max": round(float(df[col].max()), 2),
            }

    # Operation breakdown
    if "operation" in df.columns:
        summary["operations"] = df["operation"].value_counts().to_dict()

    # Retriever mode breakdown
    if "retriever_mode" in df.columns:
        summary["retriever_modes"] = df["retriever_mode"].value_counts().to_dict()

    # Recent entries (last 20)
    recent = entries[-20:]

    return {
        "total_queries": len(entries),
        "summary": summary,
        "recent": recent,
        "ingestion_ms": round(document.get("ingestion_ms", 0), 2),
        "embedding_ms": round(document.get("embedding_ms", 0), 2),
        "chunk_count": document.get("chunk_count", 0),
        "page_count": document.get("page_count", 0),
    }


# ══════════════════════════════════════════════════════════
#  CITATION HELPERS
# ══════════════════════════════════════════════════════════

def _format_citations(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a clean list of citation objects from retrieved chunks."""
    citations: list[dict[str, Any]] = []
    for chunk in retrieved:
        citations.append({
            "chunk_id": chunk["chunk_id"],
            "pages": chunk["pages"],
            "score": chunk.get("score", 0.0),
            "text_preview": chunk["text"][:120] + ("…" if len(chunk["text"]) > 120 else ""),
        })
    return citations


def _citation_instruction() -> str:
    """Return the citation instruction to inject into LLM prompts."""
    return (
        "IMPORTANT: When referencing information, cite the source using this format: "
        "[Source: Page X]. If a chunk spans multiple pages, cite the first page. "
        "Include at least one citation in your response.\n\n"
    )


# ══════════════════════════════════════════════════════════
#  LLM GENERATION
# ══════════════════════════════════════════════════════════

def _groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise PipelineError("GROQ_API_KEY is missing in .env, so the Groq backend cannot generate content.")
    return api_key


def generate_ollama(prompt: str) -> dict[str, Any]:
    """Call Ollama and return content + token usage."""
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "mistral",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "").strip()
        if not content:
            raise PipelineError("Ollama returned an empty response.")
        # Extract token counts if available
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        return {"content": content, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    except requests.exceptions.ConnectionError as exc:
        raise PipelineError(
            "Ollama is not reachable at http://localhost:11434. Run `ollama serve` first."
        ) from exc
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = response.json().get("error", "")
        except Exception:
            detail = response.text.strip()
        message = detail or "Ollama rejected the request."
        raise PipelineError(message) from exc
    except requests.RequestException as exc:
        raise PipelineError(f"Ollama request failed: {exc}") from exc


def generate_groq(prompt: str) -> dict[str, Any]:
    """Call Groq and return content + token usage."""
    try:
        from groq import Groq

        client = Groq(api_key=_groq_api_key())
        response = client.chat.completions.create(
            model=DEFAULT_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise PipelineError("Groq returned an empty response.")
        # Capture token usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return {"content": content, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"Groq request failed: {exc}") from exc


def generate_ai(prompt: str, backend: str) -> dict[str, Any]:
    """Dispatch to Ollama or Groq and return {content, prompt_tokens, completion_tokens}."""
    if backend == BACKEND_OLLAMA:
        return generate_ollama(prompt)
    return generate_groq(prompt)


# ══════════════════════════════════════════════════════════
#  CONTENT GENERATION — with citations & telemetry
# ══════════════════════════════════════════════════════════

def build_generation_prompt(choice: str, difficulty: str, context: str) -> str:
    task_map = {
        "Summary": (
            f"Create a {difficulty.lower()} level study summary with clear headings and concise bullet points."
        ),
        "Important Questions": (
            f"Create 7 {difficulty.lower()} level exam questions. Include a short answer below each question."
        ),
        "Viva Questions": (
            f"Create 7 {difficulty.lower()} level viva questions. Include a compact answer below each question."
        ),
        "MCQs": (
            f"Create 7 {difficulty.lower()} level MCQs. Each question must have exactly four options labeled A to D and one correct answer."
        ),
    }

    instruction = task_map[choice]
    return (
        "You are a helpful academic study assistant.\n"
        "Use only the provided PDF context. Do not invent topics that are not supported by the text.\n"
        + _citation_instruction()
        + f"Task:\n{instruction}\n\n"
        "PDF context:\n"
        f"{context}"
    )


def generate_content(
    choice: str,
    difficulty: str,
    backend: str,
    document: dict[str, Any],
    retriever_mode: str = DEFAULT_RETRIEVER_MODE,
) -> dict[str, Any]:
    """Generate study content with citation metadata and telemetry.

    Returns:
        {"output": str, "citations": list[dict], "telemetry": dict}
    """
    if not document or not document.get("chunks") or document.get("faiss_index") is None:
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
    retrieved = retrieve_chunks(retrieval_queries[choice], document, top_k=3, retriever_mode=retriever_mode)
    retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

    # Build context with page markers
    context_parts = []
    for chunk in retrieved:
        page_label = ", ".join(str(p) for p in chunk["pages"])
        context_parts.append(f"[Page {page_label}]:\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts) if context_parts else document["text"][:4000]

    prompt = build_generation_prompt(choice, difficulty, context)

    t_llm_start = time.perf_counter()
    llm_result = generate_ai(prompt, backend)
    llm_ms = (time.perf_counter() - t_llm_start) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000

    citations = _format_citations(retrieved)
    top_scores = [c["score"] for c in retrieved if c.get("score")]

    _record_telemetry(
        document,
        query_id=query_id,
        operation="generate",
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        total_ms=total_ms,
        retriever_mode=retriever_mode,
        top_scores=top_scores,
        prompt_tokens=llm_result.get("prompt_tokens", 0),
        completion_tokens=llm_result.get("completion_tokens", 0),
    )

    return {
        "output": llm_result["content"],
        "citations": citations,
        "retriever_mode": retriever_mode,
    }


def generate_flashcards(
    count: int,
    difficulty: str,
    backend: str,
    document: dict[str, Any],
    retriever_mode: str = DEFAULT_RETRIEVER_MODE,
) -> list[dict[str, str]]:
    """Generate Q&A flashcards from the PDF using RAG retrieval."""
    if not document or not document.get("chunks") or document.get("faiss_index") is None:
        raise PipelineError("Upload and process a PDF before generating flashcards.")

    count = max(3, min(count, 20))

    # Retrieve broad context covering different parts of the document
    query = f"{difficulty.lower()} level key concepts definitions and important facts"
    chunks = retrieve_chunks(query, document, top_k=5, retriever_mode=retriever_mode)
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

    llm_result = generate_ai(prompt, backend)
    raw = llm_result["content"]

    # Parse Q/A pairs
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


def answer_question(
    question: str,
    backend: str,
    document: dict[str, Any],
    generated_output: str = "",
    retriever_mode: str = DEFAULT_RETRIEVER_MODE,
) -> dict[str, Any]:
    """Answer a question with RAG context, citations, visuals, and telemetry.

    Returns:
        {"content": str, "visuals": list, "visual_hint": str, "citations": list}
    """
    if not question.strip():
        raise PipelineError("Ask a non-empty question.")

    query_id = str(uuid.uuid4())[:8]
    t_start = time.perf_counter()

    t_ret_start = time.perf_counter()
    retrieved = retrieve_chunks(question.strip(), document, top_k=3, retriever_mode=retriever_mode)
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
        "  Examples: 'fine tuning', 'backpropagation', 'decision tree', 'TCP/IP model', 'sorting algorithm'\n"
        "- `VISUAL_HINT: none` — only if the question is purely factual with no useful diagram\n\n"
        "Rules for VISUAL_HINT:\n"
        "- Use the EXACT topic asked about, not a related concept\n"
        "- If asked about fine-tuning → 'fine tuning', NOT 'transformer'\n"
        "- If asked about LSTM → 'LSTM', NOT 'RNN'\n"
        "- If asked about sorting → 'sorting algorithm', NOT 'algorithm'\n"
        "- Always provide a hint when a diagram would genuinely help understanding\n\n"
        f"{chr(10).join(context_parts)}\n\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )

    t_llm_start = time.perf_counter()
    llm_result = generate_ai(prompt, backend)
    llm_ms = (time.perf_counter() - t_llm_start) * 1000

    answer_text, visual_hint = _extract_visual_hint(llm_result["content"])
    visuals = collect_visuals(question, answer_text, visual_hint)

    total_ms = (time.perf_counter() - t_start) * 1000

    citations = _format_citations(retrieved)
    top_scores = [c["score"] for c in retrieved if c.get("score")]

    _record_telemetry(
        document,
        query_id=query_id,
        operation="chat",
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        total_ms=total_ms,
        retriever_mode=retriever_mode,
        top_scores=top_scores,
        prompt_tokens=llm_result.get("prompt_tokens", 0),
        completion_tokens=llm_result.get("completion_tokens", 0),
    )

    return {
        "content": answer_text,
        "visuals": visuals,
        "visual_hint": visual_hint or "",
        "citations": citations,
    }


# ══════════════════════════════════════════════════════════
#  PDF REPORT GENERATION
# ══════════════════════════════════════════════════════════

def create_pdf(text: str, output_type: str = "Study Material") -> BytesIO:
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, rightMargin=50, leftMargin=50, topMargin=60, bottomMargin=50)
    styles = getSampleStyleSheet()

    blue = colors.HexColor("#2563eb")
    cyan = colors.HexColor("#0284c7")
    purple = colors.HexColor("#7c3aed")
    muted = colors.HexColor("#64748b")
    dark = colors.HexColor("#1e293b")

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=22,
        textColor=blue,
        alignment=TA_CENTER,
        spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=muted,
        alignment=TA_CENTER,
        spaceAfter=18,
    )
    h1_style = ParagraphStyle(
        "H1Style",
        parent=styles["Heading1"],
        fontSize=15,
        textColor=blue,
        spaceBefore=16,
        spaceAfter=8,
        fontName="Helvetica-Bold",
    )
    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=cyan,
        spaceBefore=14,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    h3_style = ParagraphStyle(
        "H3Style",
        parent=styles["Heading3"],
        fontSize=11,
        textColor=purple,
        spaceBefore=10,
        spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    question_style = ParagraphStyle(
        "QuestionStyle",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=cyan,
        spaceBefore=12,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    answer_style = ParagraphStyle(
        "AnswerStyle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=dark,
        spaceAfter=8,
        leftIndent=14,
        leading=15,
    )
    option_style = ParagraphStyle(
        "OptionStyle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=muted,
        spaceAfter=4,
        leftIndent=24,
    )
    source_style = ParagraphStyle(
        "SourceStyle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=muted,
        fontName="Helvetica-Oblique",
        spaceBefore=4,
        spaceAfter=12,
        leftIndent=8,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=dark,
        spaceAfter=6,
        leading=15,
    )

    content = [
        Paragraph("DocMind AI", title_style),
        Paragraph(escape(output_type), subtitle_style),
        HRFlowable(width="100%", thickness=1.5, color=blue, spaceAfter=16),
    ]

    def _clean_latex_and_markdown(s: str) -> str:
        s = re.sub(r"\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$", r" <b>[\1\2]</b> ", s)
        s = re.sub(r"\\\(([\s\S]*?)\\\)|\$([^\$\n]+)\$", r" <i>\1\2</i> ", s)
        s = (
            s.replace(r"\upsilon", "υ")
            .replace(r"\phi", "φ")
            .replace(r"\theta", "θ")
            .replace(r"\alpha", "α")
            .replace(r"\beta", "β")
            .replace(r"\sum", "∑")
            .replace(r"\in", "∈")
            .replace(r"\neq", "≠")
            .replace(r"\leq", "≤")
            .replace(r"\geq", "≥")
            .replace(r"\times", "×")
            .replace(r"\cases", "")
            .replace(r"\begin{cases}", "")
            .replace(r"\end{cases}", "")
        )
        s = re.sub(r"_\{?(.*?)\}?", r"<sub>\1</sub>", s)
        s = re.sub(r"\^\{?(.*?)\}?", r"<sup>\1</sup>", s)
        s = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"\*(.*?)\*", r"<i>\1</i>", s)
        return s

    question_number = 1
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            content.append(Spacer(1, 4))
            continue

        if line in ("---", "***", "___"):
            content.append(HRFlowable(width="100%", thickness=0.75, color=cyan, spaceBefore=8, spaceAfter=10))
            continue

        if re.match(r"^\*?Source:\s*", line, re.IGNORECASE):
            clean_src = escape(re.sub(r"^\*?Source:\s*", "", line, flags=re.IGNORECASE).rstrip("*"))
            clean_src = _clean_latex_and_markdown(clean_src)
            content.append(Paragraph(f"<b>Source:</b> {clean_src}", source_style))
            continue

        if line.startswith("# "):
            clean_h = escape(line[2:].strip())
            content.append(Paragraph(_clean_latex_and_markdown(clean_h), h1_style))
            continue
        elif line.startswith("## "):
            clean_h = escape(line[3:].strip())
            content.append(Paragraph(_clean_latex_and_markdown(clean_h), h2_style))
            continue
        elif line.startswith("### "):
            clean_h = escape(line[4:].strip())
            content.append(Paragraph(_clean_latex_and_markdown(clean_h), h3_style))
            continue
        elif line.startswith("#### "):
            clean_h = escape(line[5:].strip())
            content.append(Paragraph(_clean_latex_and_markdown(clean_h), h3_style))
            continue

        clean_line = escape(line)
        clean_line = _clean_latex_and_markdown(clean_line)

        if re.match(r"^(Question\s*\d+|Q\d+)", clean_line, re.IGNORECASE):
            body = re.sub(r"^(Question\s*\d+|Q\d+[:.]\s*)", "", clean_line, flags=re.IGNORECASE).strip()
            content.append(Paragraph(f"Q{question_number}. {body}", question_style))
            question_number += 1
        elif re.match(r"^(Answer|Ans)[:.]\s*", clean_line, re.IGNORECASE):
            body = re.sub(r"^(Answer|Ans)[:.]\s*", "", clean_line, flags=re.IGNORECASE).strip()
            content.append(Paragraph(f"<b>Answer:</b> {body}", answer_style))
        elif re.match(r"^[A-Da-d][).]", clean_line):
            content.append(Paragraph(clean_line, option_style))
        elif clean_line.startswith(("- ", "* ")):
            content.append(Paragraph(f"&bull; {clean_line[2:]}", body_style))
        else:
            content.append(Paragraph(clean_line, body_style))

    document.build(content)
    buffer.seek(0)
    return buffer
