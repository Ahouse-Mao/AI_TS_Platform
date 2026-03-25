#!/usr/bin/env python
"""
Evaluate the current RAG pipeline with Ragas.

What this script does:
1) Read evaluation samples from JSON.
2) Call /api/assistant/chat (SSE) to get generated answers.
3) Retrieve contexts from local vector store (classic or structured RAG).
4) Run Ragas metrics and export a report.

Sample JSON format:
[
  {
    "question": "PatchTST on ETTh1 with pred_len=96, what config is recommended?",
    "ground_truth": "Use PatchTST, ETTh1, pred_len=96 ..."
  }
]

Accepted field aliases:
- question: question | query
- ground_truth: ground_truth | reference | expected_answer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from datasets import Dataset


ROOT_DIR = Path(__file__).resolve().parents[2]  # backend/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_metrics() -> list[Any]:
    """Best-effort compatibility for different ragas versions."""
    try:
        from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall

        return [answer_relevancy, faithfulness, context_precision, context_recall]
    except Exception:
        pass

    try:
        from ragas.metrics import AnswerRelevancy, Faithfulness, ContextPrecision, ContextRecall

        return [AnswerRelevancy(), Faithfulness(), ContextPrecision(), ContextRecall()]
    except Exception as e:
        raise RuntimeError(
            "Failed to import Ragas metrics. Please check ragas version and metric names."
        ) from e


def _build_judge_models(base_url: str, api_key: str, model_name: str, embedding_model: str):
    """Create judge LLM/embeddings for ragas when langchain_openai is available."""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except Exception as e:
        raise RuntimeError(
            "langchain-openai is required for Ragas judging. Install: pip install langchain-openai"
        ) from e

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        base_url=base_url,
        # 对 OpenAI 兼容但非 OpenAI 原生服务（如百炼）时，避免把文本转为 token id 列表提交。
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )
    return llm, embeddings


def _read_samples(samples_file: Path) -> list[dict[str, str]]:
    if not samples_file.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_file}")

    data = json.loads(samples_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Samples file must be a JSON array")

    rows: list[dict[str, str]] = []
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Sample #{i} is not an object")

        question = (item.get("question") or item.get("query") or "").strip()
        ground_truth = (
            item.get("ground_truth")
            or item.get("reference")
            or item.get("expected_answer")
            or ""
        ).strip()

        if not question:
            raise ValueError(f"Sample #{i} missing question/query")
        if not ground_truth:
            raise ValueError(f"Sample #{i} missing ground_truth/reference/expected_answer")

        rows.append({"question": question, "ground_truth": ground_truth})

    return rows


def _chat_sse_answer(
    assistant_url: str,
    api_key: str,
    base_url: str,
    model: str,
    question: str,
    use_struct_rag: bool,
    top_k: int,
    timeout_s: int = 120,
) -> str:
    payload = {
        "messages": [{"role": "user", "content": question}],
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "use_rag": True,
        "use_struct_rag": use_struct_rag,
        "top_k": top_k,
    }

    with requests.post(assistant_url, json=payload, stream=True, timeout=timeout_s) as resp:
        resp.raise_for_status()
        chunks: list[str] = []
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data: "):
                continue

            data = raw[6:]
            if data == "[DONE]":
                break

            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            if "error" in obj:
                raise RuntimeError(f"Assistant API error: {obj['error']}")
            if "content" in obj and obj["content"]:
                chunks.append(obj["content"])

    return "".join(chunks).strip()


def _retrieve_contexts(question: str, use_struct_rag: bool, top_k: int, embedding_model: str) -> list[str]:
    if use_struct_rag:
        from RAG.rag_struct import retrieve

        docs = retrieve(query=question, k=top_k, model_name=embedding_model)
    else:
        from RAG.rag import load_index

        docs = load_index(model_name=embedding_model).similarity_search(question, k=top_k)

    return [getattr(d, "page_content", "") for d in docs if getattr(d, "page_content", "")]


def _to_text(value: Any) -> str:
    """Force any value to a clean string for LLM/Embedding inputs."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    except Exception:
        return ""


def _sanitize_contexts(contexts: list[Any]) -> list[str]:
    cleaned: list[str] = []
    for c in contexts:
        text = _to_text(c)
        if text:
            cleaned.append(text)
    return cleaned


def _to_ragas_dataset(rows: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list(rows)


def _json_safe(value: Any) -> Any:
    """Convert NaN/Inf and numpy-like scalars to JSON-safe values."""
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


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(data), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")


def _answer_cache_key(
    question: str,
    model: str,
    base_url: str,
    use_struct_rag: bool,
    top_k: int,
    embedding_model: str,
) -> str:
    raw = "||".join(
        [
            question.strip(),
            model.strip(),
            base_url.strip(),
            "struct" if use_struct_rag else "classic",
            str(top_k),
            embedding_model.strip(),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    samples = _read_samples(Path(args.samples_file))

    assistant_url = args.assistant_url.rstrip("/")
    metric_rows: list[dict[str, Any]] = []
    cache_path = Path(__file__).with_name("ragas_answer_cache.json")
    history_path = Path(__file__).with_name("ragas_answer_history.jsonl")
    answer_cache: dict[str, dict[str, Any]] = _load_json(cache_path, {})

    print(f"[ragas] loaded samples: {len(samples)}")
    for idx, s in enumerate(samples, start=1):
        q = _to_text(s["question"])
        gt = _to_text(s["ground_truth"])

        cache_key = _answer_cache_key(
            question=q,
            model=args.model,
            base_url=args.base_url,
            use_struct_rag=args.use_struct_rag,
            top_k=args.top_k,
            embedding_model=args.embedding_model,
        )

        cached = answer_cache.get(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("answer"), str) and cached.get("answer").strip():
            answer = _to_text(cached["answer"])
            answer_source = "cache"
            print(f"[ragas] ({idx}/{len(samples)}) answer from cache")
        else:
            print(f"[ragas] ({idx}/{len(samples)}) generating answer...")
            answer = _chat_sse_answer(
                assistant_url=assistant_url,
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                question=q,
                use_struct_rag=args.use_struct_rag,
                top_k=args.top_k,
            )
            answer_source = "llm"
            answer = _to_text(answer)
            answer_cache[cache_key] = {
                "question": q,
                "answer": answer,
                "model": args.model,
                "base_url": args.base_url,
                "mode": "struct" if args.use_struct_rag else "classic",
                "top_k": args.top_k,
                "embedding_model": args.embedding_model,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }

        contexts = _sanitize_contexts(_retrieve_contexts(
            question=q,
            use_struct_rag=args.use_struct_rag,
            top_k=args.top_k,
            embedding_model=args.embedding_model,
        ))

        if not contexts:
            # 保证 contexts 始终是字符串列表，避免部分评测器对空结构处理不一致。
            contexts = [""]

        metric_rows.append(
            {
                "question": q,
                "answer": _to_text(answer),
                "contexts": contexts,
                "ground_truth": gt,
            }
        )

        _append_jsonl(
            history_path,
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "question": q,
                "ground_truth": gt,
                "answer": answer,
                "answer_source": answer_source,
                "generator_model": args.model,
                "mode": "struct" if args.use_struct_rag else "classic",
                "top_k": args.top_k,
                "embedding_model": args.embedding_model,
            },
        )

    _save_json(cache_path, answer_cache)

    ds = _to_ragas_dataset(metric_rows)

    from ragas import evaluate

    metrics = _load_metrics()
    llm, embeddings = _build_judge_models(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.judge_model,
        embedding_model=args.judge_embedding_model,
    )

    print("[ragas] running metrics...")
    result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=embeddings)

    result_df = result.to_pandas()
    score_summary = result_df.mean(numeric_only=True).to_dict()

    report = {
        "mode": "struct" if args.use_struct_rag else "classic",
        "samples": len(samples),
        "top_k": args.top_k,
        "assistant_url": assistant_url,
        "generator_model": args.model,
        "judge_model": args.judge_model,
        "judge_embedding_model": args.judge_embedding_model,
        "summary": _json_safe(score_summary),
        "details": _json_safe(result_df.to_dict(orient="records")),
        "answer_cache_file": str(cache_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "answer_history_file": str(history_path.relative_to(ROOT_DIR)).replace("\\", "/"),
    }

    out_path = Path(args.out_file)
    _save_json(out_path, report)
    print(f"[ragas] report written: {out_path}")

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate your RAG pipeline with Ragas")
    parser.add_argument("--samples-file", default=str(Path(__file__).with_name("ragas_eval_samples.json")))
    parser.add_argument("--out-file", default=str(Path(__file__).with_name("ragas_report.json")))

    parser.add_argument("--assistant-url", default="http://127.0.0.1:8000/api/assistant/chat")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default="qwen3.5-flash")

    parser.add_argument("--use-struct-rag", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embedding-model", default="BAAI/bge-base-zh-v1.5")

    parser.add_argument("--judge-model", default="qwen3.5-plus")
    parser.add_argument("--judge-embedding-model", default="text-embedding-v4")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Set --api-key or OPENAI_API_KEY.")

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    report = run_eval(args)
    print("[ragas] summary:")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
