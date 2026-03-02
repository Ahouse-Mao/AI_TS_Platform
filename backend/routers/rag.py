# ===================== RAG 路由 =====================
# 负责向量索引的构建、状态轮询和清除。
# 构建任务在后台异步执行，前端通过 /api/rag/status 轮询进度。

import os
import traceback
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

router = APIRouter(prefix="/api/rag", tags=["rag"])


class RAGBuildRequest(BaseModel):
    embedding_model: str = 'BAAI/bge-base-zh-v1.5'


# ── 模块级可变状态 ──
rag_build_status: dict = {
    "status":     "idle",   # idle | running | completed | failed
    "message":    "No rag build in progress.",
    "start_time": None,
    "doc_count":  None,
}
rag_logs: list[str] = []


# ── 后台任务 ──
def run_rag_build_task(embedding_model: str) -> None:
    global rag_build_status, rag_logs

    # 强制离线模式，防止 HuggingFace 尝试联网导致挂起
    os.environ["HF_HUB_OFFLINE"] = "1"

    def _log(msg: str) -> None:
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        rag_logs.append(entry)
        print(entry, flush=True)

    try:
        from ..RAG.rag import build_index

        _log("开始构建 RAG 向量索引...")
        _log(f"使用嵌入模型: {embedding_model}")

        vectorstore = build_index(model_name=embedding_model, log_fn=_log)
        doc_count   = vectorstore._collection.count()

        _log(f"✓ 向量索引构建完成，共 {doc_count} 条文档")
        rag_build_status.update({
            "status":    "completed",
            "message":   f"构建完成，共 {doc_count} 条文档",
            "doc_count": doc_count,
        })

    except Exception as e:
        rag_build_status.update({
            "status":  "failed",
            "message": f"构建失败: {str(e)}",
        })
        _log(f"✗ 构建失败: {str(e)}")
        for line in traceback.format_exc().splitlines():
            _log(line)


# ── 路由 ──

@router.post('/build')
async def rag_build(background_tasks: BackgroundTasks, req: RAGBuildRequest = RAGBuildRequest()):
    """启动 RAG 向量索引构建任务（后台异步执行，立即返回）。"""
    global rag_build_status, rag_logs

    if rag_build_status["status"] == "running":
        return {"success": False, "message": "RAG 构建正在进行中，请稍候"}

    rag_logs.clear()
    rag_build_status = {
        "status":     "running",
        "message":    "RAG 构建已启动...",
        "start_time": datetime.now().isoformat(),
        "doc_count":  None,
    }

    background_tasks.add_task(run_rag_build_task, req.embedding_model)
    return {"success": True, "message": "RAG 构建任务已启动"}


@router.get('/status')
def get_rag_status():
    """返回 RAG 构建状态和日志（前端轮询专用）。"""
    return {**rag_build_status, "logs": rag_logs}


@router.delete('/index')
def delete_rag_index():
    """清除持久化向量库目录（构建进行中时拒绝操作）。"""
    global rag_build_status, rag_logs

    if rag_build_status["status"] == "running":
        return {"success": False, "message": "RAG 构建正在进行中，无法清除"}

    from ..RAG.rag import clear_index
    result = clear_index()

    rag_logs.clear()
    rag_build_status = {
        "status":     "idle",
        "message":    "向量库已清除。",
        "start_time": None,
        "doc_count":  None,
    }
    return result
