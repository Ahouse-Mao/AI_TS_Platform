# ===================== AI 助手路由 =====================
# 提供流式聊天代理和模型列表查询。
# 通过 System Prompt 注入训练/推理触发指令，让 AI 能动态调用平台能力。

import json
from typing import List

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


class _AssistantMsg(BaseModel):
    role:    str
    content: str

class AssistantChatRequest(BaseModel):
    messages: List[_AssistantMsg]
    model:    str  = 'gpt-4o'
    api_key:  str  = ''
    base_url: str  = 'https://api.openai.com/v1'
    use_rag:  bool = True
    use_struct_rag: bool = False
    top_k: int = Field(default=3, ge=1, le=10)


# ── 注入到 System Prompt 的训练 & 推理触发指令 ──
_TRIGGER_INSTRUCTION = """
---
【训练触发能力】
当用户明确表达"开始训练"、"帮我训练"、"训练一下"等指令时，在回复末尾（所有解释文字之后）额外输出一个配置块，格式严格如下（<<<TRAIN_CONFIG 和 >>> 各占单独一行，中间是单行 JSON）：
<<<TRAIN_CONFIG
{"model": "模型名", "data": "数据集名", "data_path": "数据集文件.csv", "features": "M", "seq_len": 96, "label_len": 48, "pred_len": 96, "train_epochs": 50, "patience": 10, "batch_size": 64, "learning_rate": 0.005, "use_gpu": true}
>>>
规则：
- JSON 只输出一行，不加注释；配置块只在用户明确要求训练时才输出；
- 若模型或数据集不明确，先询问用户，不输出配置块；
- 可用模型：DLinear, NLinear, Linear, PatchTST, Autoformer, Informer, Transformer；
- 可用数据集：ETTh1(ETTh1.csv), ETTh2(ETTh2.csv), ETTm1(ETTm1.csv), ETTm2(ETTm2.csv)；
- features: M(多→多), MS(多→单), S(单→单)。

---
【推理与可视化触发能力】
当用户明确表达"进行推理"、"做预测"、"可视化结果"、"看看效果"、"预测一下"等指令时，在回复末尾额外输出一个推理配置块，格式严格如下（<<<INFER_CONFIG 和 >>> 各占单独一行，中间是单行 JSON）：
<<<INFER_CONFIG
{"folder_name": "latest", "n_samples": 200, "use_onnx": false}
>>>
规则：
- folder_name 使用 "latest" 代表最新模型（系统自动解析），也可指定精确的 Checkpoint 文件夹名；
- n_samples 为推理样本数，默认 200；
- use_onnx: false 使用 PTH，true 使用 ONNX；
- 配置块只在用户明确要求推理/可视化/预测时才输出，不能与 TRAIN_CONFIG 同时输出。
"""


@router.get('/models')
def get_assistant_models(api_key: str = '', base_url: str = 'https://api.openai.com/v1'):
    """拉取指定 API 端点可用的模型列表。"""
    if not api_key.strip():
        return {'models': [], 'error': '请先配置 API Key'}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        ids    = sorted(m.id for m in client.models.list().data)
        return {'models': ids}
    except Exception as e:
        return {'models': [], 'error': str(e)}


@router.post('/chat')
def assistant_chat(req: AssistantChatRequest):
    """流式代理到 OpenAI 兼容 API，以 SSE 形式返回给前端。"""

    if not req.api_key.strip():
        async def _no_key():
            yield f"data: {json.dumps({'error': '请先在设置中配置 API Key'})}\n\n"
        return StreamingResponse(_no_key(), media_type='text/event-stream')

    try:
        from openai import OpenAI
    except ImportError:
        async def _no_lib():
            yield f"data: {json.dumps({'error': 'openai 库未安装，请运行 pip install openai'})}\n\n"
        return StreamingResponse(_no_lib(), media_type='text/event-stream')

    messages = [{'role': m.role, 'content': m.content} for m in req.messages]

    # ── RAG 上下文注入（失败时静默降级）──
    if req.use_rag:
        try:
            import os as _os
            _os.environ['HF_HUB_OFFLINE'] = '1'
            if req.use_struct_rag:
                from RAG.rag_struct import retrieve, PERSIST_DIR
            else:
                from RAG.rag import load_index, PERSIST_DIR
            if _os.path.exists(PERSIST_DIR):
                user_msgs = [m for m in req.messages if m.role == 'user']
                if user_msgs:
                    query = user_msgs[-1].content
                    if req.use_struct_rag:
                        docs = retrieve(query=query, k=req.top_k)
                    else:
                        docs = load_index().similarity_search(query, k=req.top_k) # 加载持久化ChromaDB, 进行相似度搜索
                    if docs:
                        context = '\n\n'.join(
                            f"[参考{i+1}] {d.page_content}" for i, d in enumerate(docs)
                        )
                        rag_sys = (
                            "以下是从本平台训练脚本知识库中检索到的相关参数推荐，"
                            "请结合这些信息回答用户问题：\n\n"
                            f"{context}\n\n"
                            "若以上参考与问题无关，请基于通用知识回答。"
                        )
                        sys_idx = next(
                            (i for i, m in enumerate(messages) if m['role'] == 'system'), None
                        )
                        if sys_idx is not None:
                            messages[sys_idx]['content'] += '\n\n' + rag_sys
                        else:
                            messages.insert(0, {'role': 'system', 'content': rag_sys})
        except Exception:
            pass

    # ── 训练 & 推理触发指令注入 ──
    sys_idx = next((i for i, m in enumerate(messages) if m['role'] == 'system'), None)
    if sys_idx is not None:
        messages[sys_idx]['content'] += _TRIGGER_INSTRUCTION
    else:
        messages.insert(0, {'role': 'system', 'content': _TRIGGER_INSTRUCTION.strip()})

    client = OpenAI(api_key=req.api_key, base_url=req.base_url)

    def _stream():
        try:
            resp = client.chat.completions.create(
                model=req.model, messages=messages, stream=True
            )
            for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
            yield 'data: [DONE]\n\n'
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(_stream(), media_type='text/event-stream')
