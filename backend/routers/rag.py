# ===================== RAG 路由 =====================
# 负责向量索引的构建、状态轮询和清除。
# 构建任务在后台异步执行，前端通过 /api/rag/status 轮询进度。

import os
import traceback
from datetime import datetime
import math
import threading
from types import SimpleNamespace

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/rag", tags=["rag"])


class RAGBuildRequest(BaseModel):
    embedding_model: str = 'BAAI/bge-base-zh-v1.5'
    use_struct_rag: bool = False


class RAGEvalRequest(BaseModel):
    mode: str = 'struct'  # classic | struct | both
    k_override: int | None = None
    embedding_model: str = 'BAAI/bge-base-zh-v1.5'
    samples_file: str = 'RAG/eval/rag_eval_samples_25.json'
    out_file: str = 'RAG/eval/rag_eval_report_latest.json'


class RAGRagasEvalRequest(BaseModel):
    assistant_url: str = 'http://127.0.0.1:8000/api/assistant/chat'
    api_key: str = ''
    base_url: str = 'https://api.openai.com/v1'
    model: str = 'qwen3.5-flash'
    judge_model: str = 'qwen3.5-plus'
    judge_embedding_model: str = 'text-embedding-v4'
    use_struct_rag: bool = True
    top_k: int = 3
    embedding_model: str = 'BAAI/bge-base-zh-v1.5'
    samples_file: str = 'RAG/eval/ragas_eval_samples.json'
    out_file: str = 'RAG/eval/ragas_report.json'


# ── 模块级可变状态 ──
rag_build_status: dict = {
    "status":     "idle",   # idle | running | completed | failed
    "message":    "No rag build in progress.",
    "start_time": None,
    "doc_count":  None,
}
rag_logs: list[str] = []
_ragas_eval_lock = threading.Lock()
ragas_eval_status: dict = {
    "status": "idle",   # idle | running | completed | failed
    "message": "No ragas evaluation in progress.",
    "start_time": None,
    "end_time": None,
    "report": None,
}


def _json_safe(value):
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]

    return value


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
        use_struct_rag = bool(rag_build_status.get("use_struct_rag", False))
        if use_struct_rag:
            from RAG.rag_struct import build_index
            _log("当前模式: 结构化 RAG")
        else:
            from RAG.rag import build_index
            _log("当前模式: 普通 RAG")

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


def run_ragas_eval_task(req_dict: dict) -> None:
    """后台执行 Ragas 评测，完成后更新状态。"""
    global ragas_eval_status
    try:
        from RAG.eval.evaluate_with_ragas import run_eval

        args = SimpleNamespace(**req_dict)
        report = run_eval(args)
        safe_report = _json_safe(report)

        with _ragas_eval_lock:
            ragas_eval_status.update({
                "status": "completed",
                "message": "Ragas 评测完成",
                "end_time": datetime.now().isoformat(),
                "report": safe_report,
            })
    except Exception as e:
        with _ragas_eval_lock:
            ragas_eval_status.update({
                "status": "failed",
                "message": f"Ragas 评测失败: {str(e)}",
                "end_time": datetime.now().isoformat(),
                "report": None,
            })


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
        "use_struct_rag": req.use_struct_rag,
    }

    background_tasks.add_task(run_rag_build_task, req.embedding_model)
    return {"success": True, "message": "RAG 构建任务已启动"}


@router.get('/status')
def get_rag_status():
    """返回 RAG 构建状态和日志（前端轮询专用）。"""
    return {**rag_build_status, "logs": rag_logs}


@router.delete('/index')
def delete_rag_index(use_struct_rag: bool = False):
    """清除持久化向量库目录（构建进行中时拒绝操作）。"""
    global rag_build_status, rag_logs

    if rag_build_status["status"] == "running":
        return {"success": False, "message": "RAG 构建正在进行中，无法清除"}

    if use_struct_rag:
        from RAG.rag_struct import clear_index
    else:
        from RAG.rag import clear_index
    result = clear_index()

    rag_logs.clear()
    rag_build_status = {
        "status":     "idle",
        "message":    "向量库已清除。",
        "start_time": None,
        "doc_count":  None,
    }
    return result


@router.post('/eval/run')
def run_rag_evaluation(req: RAGEvalRequest):
    """运行 RAG 检索评测并返回汇总与明细。"""
    mode = req.mode.strip().lower()
    if mode not in {'classic', 'struct', 'both'}:
        return {'success': False, 'message': 'mode 必须是 classic / struct / both'}

    try:
        from RAG.eval.evaluate_rag import run_evaluation

        report = run_evaluation(
            samples_path=req.samples_file,
            mode=mode,
            k_override=req.k_override,
            embedding_model=req.embedding_model,
            out_path=req.out_file,
        )
        return {'success': True, 'message': '评测完成', 'report': report}
    except Exception as e:
        return {'success': False, 'message': f'评测失败: {str(e)}'}


@router.post('/eval/ragas/run')
def run_ragas_evaluation(background_tasks: BackgroundTasks, req: RAGRagasEvalRequest):
    """启动 Ragas 评测后台任务（立即返回，前端轮询状态）。"""
    global ragas_eval_status

    if not req.api_key.strip():
        return {'success': False, 'message': 'api_key 不能为空'}

    if req.top_k < 1:
        return {'success': False, 'message': 'top_k 必须 >= 1'}

    with _ragas_eval_lock:
        if ragas_eval_status.get("status") == "running":
            raise HTTPException(status_code=503, detail='Ragas 评测任务正在运行，后端繁忙，请稍后重试')

        req_dict = dict(
            assistant_url=req.assistant_url,
            api_key=req.api_key,
            base_url=req.base_url,
            model=req.model,
            judge_model=req.judge_model,
            judge_embedding_model=req.judge_embedding_model,
            use_struct_rag=req.use_struct_rag,
            top_k=req.top_k,
            embedding_model=req.embedding_model,
            samples_file=req.samples_file,
            out_file=req.out_file,
        )

        ragas_eval_status = {
            "status": "running",
            "message": "Ragas 评测任务已启动...",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "report": None,
        }

    background_tasks.add_task(run_ragas_eval_task, req_dict)
    return {'success': True, 'message': 'Ragas 评测任务已启动'}


@router.get('/eval/ragas/status')
def get_ragas_eval_status():
    """查询 Ragas 评测任务状态与结果。"""
    with _ragas_eval_lock:
        return _json_safe(dict(ragas_eval_status))
