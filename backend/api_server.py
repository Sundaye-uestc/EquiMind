import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# Fix GBK encoding issue on Windows — force UTF-8 for stdout/stderr
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.react_agent import ReactAgent
from rag.rag_service import RagSummarizeService
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf

app = FastAPI(title="多智能体诊断平台 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000", "http://localhost:5000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tasks: dict = {}
executor = ThreadPoolExecutor(max_workers=4)
project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_root, chroma_conf["data_path"])


class DiagnosisRequest(BaseModel):
    title: str = "诊断请求"
    description: str
    symptoms: list[str] = []
    force_optimize: bool = False


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


def _get_chroma_collection():
    client = chromadb.PersistentClient(path=chroma_conf["persist_directory"])
    return client.get_collection(chroma_conf["collection_name"])


def _run_agent(task_id: str, query: str):
    try:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["stages"][0]["status"] = "running"

        agent = ReactAgent()
        all_chunks = []

        query_preview = query[:60].replace('\n', ' ') + ('...' if len(query) > 60 else '')
        tasks[task_id]["thinking_chunks"].append(f"嗯……用户问到：{query_preview}\n\n")

        for i, chunk in enumerate(agent.execute_stream(query)):
            if i == 0:
                q = query.strip()
                c = chunk.strip()
                if c and q.startswith(c[:min(len(c), 30)]):
                    chunk = ""
                elif c.startswith(q):
                    chunk = c[len(q):].strip() + "\n" if c[len(q):].strip() else ""
            if chunk:
                all_chunks.append(chunk)
                tasks[task_id]["thinking_chunks"].append(chunk)
            if i == 0:
                tasks[task_id]["stages"][0]["status"] = "done"
                tasks[task_id]["stages"][1]["status"] = "running"

        tasks[task_id]["stages"][1]["status"] = "done"
        tasks[task_id]["stages"][2]["status"] = "running"

        if len(all_chunks) > 1:
            thinking = "".join(all_chunks[:-1])
            answer = all_chunks[-1]
            tasks[task_id]["thinking_chunks"] = all_chunks[:-1]
        elif len(all_chunks) == 1:
            thinking = ""
            answer = all_chunks[0]
            tasks[task_id]["thinking_chunks"] = []
        else:
            thinking = ""
            answer = ""
            tasks[task_id]["thinking_chunks"] = []

        tasks[task_id]["stages"][2]["status"] = "done"
        tasks[task_id]["report"] = answer
        tasks[task_id]["thinking"] = thinking
        tasks[task_id]["status"] = "completed"

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)


# ── Health ──

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ── Diagnosis Tasks ──

@app.post("/api/diagnosis/tasks")
def create_task(req: DiagnosisRequest):
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "id": task_id,
        "title": req.title,
        "description": req.description,
        "status": "pending",
        "stages": [
            {"name": "任务规划", "status": "pending"},
            {"name": "执行诊断", "status": "pending"},
            {"name": "生成报告", "status": "pending"},
        ],
        "report": None,
        "thinking": None,
        "thinking_chunks": [],
        "error": None,
        "created_at": datetime.now().isoformat(),
    }

    executor.submit(_run_agent, task_id, req.description)
    return {"task_id": task_id, "status": "pending"}


@app.get("/api/diagnosis/tasks/{task_id}")
def get_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.get("/api/diagnosis/tasks/{task_id}/report/stream")
def stream_report(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task["status"] == "failed":
        raise HTTPException(status_code=500, detail=task.get("error", "未知错误"))

    report = task.get("report", "")

    def generate():
        chunk_size = 50
        for i in range(0, len(report), chunk_size):
            yield report[i : i + chunk_size]
            time.sleep(0.02)

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/api/diagnosis/tasks/{task_id}/thinking/stream")
async def stream_thinking(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    async def generate():
        last_index = 0
        while True:
            chunks = task.get("thinking_chunks", [])
            while last_index < len(chunks):
                chunk = chunks[last_index]
                yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
                last_index += 1
            if task["status"] in ("completed", "failed"):
                yield f"data: {json.dumps({'done': True, 'status': task['status']}, ensure_ascii=False)}\n\n"
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Knowledge Base ──

@app.get("/api/kb/documents")
def list_documents():
    try:
        collection = _get_chroma_collection()
        result = collection.get(include=["metadatas"])

        docs_map: dict = {}
        metadatas = result.get("metadatas") or []
        for meta in metadatas:
            source = meta.get("source", "unknown")
            if source not in docs_map:
                docs_map[source] = {
                    "doc_id": hashlib.md5(source.encode()).hexdigest()[:8],
                    "title": os.path.basename(source),
                    "source": source,
                    "chunk_count": 0,
                    "created_at": "",
                }
            docs_map[source]["chunk_count"] += 1

        items = list(docs_map.values())
        return {"items": items, "total": len(items)}
    except Exception:
        return {"items": [], "total": 0}


@app.post("/api/kb/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        vs = VectorStoreService()
        vs.load_document()

        return {"status": "ok", "filename": file.filename, "size": len(content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/kb/documents/{doc_id}")
def delete_document(doc_id: str):
    try:
        collection = _get_chroma_collection()
        result = collection.get(include=["metadatas"])

        source_to_delete = None
        for meta in (result.get("metadatas") or []):
            source = meta.get("source", "")
            if hashlib.md5(source.encode()).hexdigest()[:8] == doc_id:
                source_to_delete = source
                break

        if not source_to_delete:
            raise HTTPException(status_code=404, detail="文档不存在")

        collection.delete(where={"source": source_to_delete})

        if os.path.exists(source_to_delete):
            os.remove(source_to_delete)

        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/kb/search")
def search_knowledge(req: SearchRequest):
    try:
        rag = RagSummarizeService()
        docs = rag.retriever_docs(req.query)

        items = []
        for doc in docs[: req.top_k]:
            items.append({
                "title": os.path.basename(doc.metadata.get("source", "未知")),
                "source": doc.metadata.get("source", ""),
                "chunk": doc.page_content,
                "score": doc.metadata.get("score", 0.0),
            })

        return {"items": items, "total": len(items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
