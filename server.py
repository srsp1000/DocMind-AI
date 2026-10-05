from __future__ import annotations

import base64
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import backend as be

try:
    import langchain_pipeline
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

app = FastAPI(title="DocMind AI")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_sessions: dict[str, dict[str, Any]] = {}

def _get_doc(session_id: str) -> dict[str, Any]:
    doc = _sessions.get(session_id)
    if not doc:
        raise HTTPException(400, "No document found. Upload a PDF first.")
    return doc

@app.get("/", response_class=HTMLResponse)
async def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    file_bytes = await file.read()
    try:
        doc = be.prepare_document(file_bytes, file.filename)
    except be.PipelineError as exc:
        raise HTTPException(422, str(exc))
    session_id = str(uuid.uuid4())
    _sessions[session_id] = doc
    return {
        "session_id": session_id, 
        "file_name": doc["name"], 
        "chunk_count": doc["chunk_count"],
        "page_count": doc.get("page_count", 0)
    }

@app.post("/generate")
async def generate(
    session_id: str = Form(...), 
    choice: str = Form(...), 
    difficulty: str = Form(...), 
    backend: str = Form(...),
    retriever_mode: str = Form(default="hybrid"),
    framework: str = Form(default="native")
):
    doc = _get_doc(session_id)
    if choice not in be.OUTPUT_OPTIONS: raise HTTPException(400, f"Invalid choice.")
    if difficulty not in be.DIFFICULTY_OPTIONS: raise HTTPException(400, f"Invalid difficulty.")
    if backend not in be.BACKEND_OPTIONS: raise HTTPException(400, f"Invalid backend.")
    if retriever_mode not in be.RETRIEVER_MODES: raise HTTPException(400, f"Invalid retriever_mode.")
    if framework not in ("native", "langchain"): raise HTTPException(400, f"Invalid framework.")
    
    try:
        if framework == "langchain":
            if not HAS_LANGCHAIN:
                raise HTTPException(400, "Langchain framework is not available.")
            result = langchain_pipeline.langchain_engine.generate_content(choice, difficulty, backend, doc, retriever_mode)
        else:
            result = be.generate_content(choice, difficulty, backend, doc, retriever_mode)
    except be.PipelineError as exc:
        raise HTTPException(422, str(exc))
    return result

@app.post("/chat")
async def chat(
    session_id: str = Form(...), 
    question: str = Form(...), 
    backend: str = Form(...), 
    generated_output: str = Form(default=""),
    retriever_mode: str = Form(default="hybrid"),
    framework: str = Form(default="native")
):
    doc = _get_doc(session_id)
    if backend not in be.BACKEND_OPTIONS: raise HTTPException(400, f"Invalid backend.")
    if retriever_mode not in be.RETRIEVER_MODES: raise HTTPException(400, f"Invalid retriever_mode.")
    if framework not in ("native", "langchain"): raise HTTPException(400, f"Invalid framework.")
    
    try:
        if framework == "langchain":
            if not HAS_LANGCHAIN:
                raise HTTPException(400, "Langchain framework is not available.")
            result = langchain_pipeline.langchain_engine.answer_question(question, backend, doc, generated_output, retriever_mode)
        else:
            result = be.answer_question(question, backend, doc, generated_output, retriever_mode)
    except be.PipelineError as exc:
        raise HTTPException(422, str(exc))
        
    for v in result.get("visuals", []):
        if v.get("type") == "local":
            path = Path(v.get("image_path", ""))
            if path.exists():
                svg_bytes = path.read_bytes()
                b64 = base64.b64encode(svg_bytes).decode()
                v["data_uri"] = f"data:image/svg+xml;base64,{b64}"
    return result

@app.post("/flashcards")
async def flashcards(
    session_id: str = Form(...), 
    count: int = Form(default=10), 
    difficulty: str = Form(default="Medium"), 
    backend: str = Form(...),
    retriever_mode: str = Form(default="hybrid"),
    framework: str = Form(default="native")
):
    doc = _get_doc(session_id)
    if difficulty not in be.DIFFICULTY_OPTIONS: raise HTTPException(400, "Invalid difficulty.")
    if backend not in be.BACKEND_OPTIONS: raise HTTPException(400, "Invalid backend.")
    if retriever_mode not in be.RETRIEVER_MODES: raise HTTPException(400, f"Invalid retriever_mode.")
    if framework not in ("native", "langchain"): raise HTTPException(400, f"Invalid framework.")
    
    try:
        if framework == "langchain":
            if not HAS_LANGCHAIN:
                raise HTTPException(400, "Langchain framework is not available.")
            cards = langchain_pipeline.langchain_engine.generate_flashcards(count, difficulty, backend, doc, retriever_mode)
        else:
            cards = be.generate_flashcards(count, difficulty, backend, doc, retriever_mode)
    except be.PipelineError as exc:
        raise HTTPException(422, str(exc))
    return {"cards": cards}

@app.get("/metrics/{session_id}")
async def metrics(session_id: str):
    doc = _get_doc(session_id)
    return be.get_telemetry_summary(doc)

@app.post("/download")
async def download(text: str = Form(...), output_type: str = Form(default="Study Material")):
    try:
        buf = be.create_pdf(text, output_type)
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="study_material.pdf"'})
