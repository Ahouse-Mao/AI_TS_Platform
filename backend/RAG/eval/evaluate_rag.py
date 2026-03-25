import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

# Ensure imports like `from RAG import rag` work when running this file directly.
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)


@dataclass
class SampleResult:
    sample_id: str
    mode: str
    k: int
    retrieved: int
    relevant_total: int
    relevant_hits_docs: int
    relevant_hits_gold: int
    precision_at_k: float
    recall_at_k: float
    precision_at_1: float
    hit_at_1: float
    reciprocal_rank: float


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def _extract_attrs_from_text(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}

    md = re.search(r"使用\s+([^\s，。]+)\s+模型对\s+([^\s，。]+)\s+数据集", text)
    if md:
        attrs["model"] = md.group(1)
        attrs["dataset"] = md.group(2)

    for key in ["pred_len", "seq_len", "label_len", "features"]:
        m = re.search(rf"\({key}\)=([^，。\s]+)", text)
        if m:
            attrs[key] = m.group(1)

    return attrs


def _doc_attrs(doc: Any) -> dict[str, str]:
    metadata = getattr(doc, "metadata", None) or {}
    attrs = {k: str(v) for k, v in metadata.items() if v is not None and str(v) != ""}

    parsed = _extract_attrs_from_text(getattr(doc, "page_content", "") or "")
    for k, v in parsed.items():
        attrs.setdefault(k, v)

    return attrs


def _is_gold_match(doc_attrs: dict[str, str], gold_cond: dict[str, str]) -> bool:
    for key, expected in gold_cond.items():
        if _normalize(doc_attrs.get(key)) != _normalize(str(expected)):
            return False
    return True


def evaluate_mode(store: Any, samples: list[dict[str, Any]], mode: str, k_override: int | None) -> tuple[list[SampleResult], list[dict[str, Any]]]:
    results: list[SampleResult] = []
    details: list[dict[str, Any]] = []

    for s in samples:
        sid = s["id"]
        query = s["query"]
        k = int(k_override or s.get("k", 3))
        gold_any: list[dict[str, str]] = s.get("gold_any", [])

        if mode == "struct":
            from RAG.rag_struct import retrieve

            docs = retrieve(query=query, k=k)
        else:
            docs = store.similarity_search(query, k=k)

        hits_docs = 0
        matched_gold_indexes: set[int] = set()
        first_relevant_rank: int | None = None
        ranked_info: list[dict[str, Any]] = []

        for i, d in enumerate(docs, start=1):
            attrs = _doc_attrs(d)
            matched_this_doc = False

            for gi, cond in enumerate(gold_any):
                if _is_gold_match(attrs, cond):
                    matched_this_doc = True
                    matched_gold_indexes.add(gi)

            if matched_this_doc:
                hits_docs += 1
                if first_relevant_rank is None:
                    first_relevant_rank = i

            ranked_info.append(
                {
                    "rank": i,
                    "matched": matched_this_doc,
                    "attrs": attrs,
                    "text_preview": (getattr(d, "page_content", "") or "")[:220],
                }
            )

        relevant_total = max(len(gold_any), 1)
        precision = hits_docs / float(max(k, 1))
        recall = len(matched_gold_indexes) / float(relevant_total)
        hit_at_1 = 1.0 if first_relevant_rank == 1 else 0.0
        precision_at_1 = hit_at_1
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

        results.append(
            SampleResult(
                sample_id=sid,
                mode=mode,
                k=k,
                retrieved=len(docs),
                relevant_total=len(gold_any),
                relevant_hits_docs=hits_docs,
                relevant_hits_gold=len(matched_gold_indexes),
                precision_at_k=precision,
                recall_at_k=recall,
                precision_at_1=precision_at_1,
                hit_at_1=hit_at_1,
                reciprocal_rank=reciprocal_rank,
            )
        )

        details.append(
            {
                "id": sid,
                "mode": mode,
                "query": query,
                "k": k,
                "gold_any": gold_any,
                "precision_at_k": round(precision, 4),
                "recall_at_k": round(recall, 4),
                "precision_at_1": round(precision_at_1, 4),
                "hit_at_1": round(hit_at_1, 4),
                "reciprocal_rank": round(reciprocal_rank, 4),
                "ranked": ranked_info,
            }
        )

    return results, details


def summarize(results: list[SampleResult]) -> dict[str, Any]:
    if not results:
        return {
            "samples": 0,
            "macro_precision_at_k": 0.0,
            "macro_recall_at_k": 0.0,
            "macro_precision_at_1": 0.0,
            "macro_hit_at_1": 0.0,
            "macro_mrr": 0.0,
        }

    n = len(results)
    macro_p = sum(r.precision_at_k for r in results) / n
    macro_r = sum(r.recall_at_k for r in results) / n
    macro_p1 = sum(r.precision_at_1 for r in results) / n
    macro_h1 = sum(r.hit_at_1 for r in results) / n
    macro_mrr = sum(r.reciprocal_rank for r in results) / n

    return {
        "samples": n,
        "macro_precision_at_k": round(macro_p, 4),
        "macro_recall_at_k": round(macro_r, 4),
        "macro_precision_at_1": round(macro_p1, 4),
        "macro_hit_at_1": round(macro_h1, 4),
        "macro_mrr": round(macro_mrr, 4),
    }


def load_store(mode: str, embedding_model: str):
    if mode == "classic":
        from RAG.rag import load_index

        return load_index(model_name=embedding_model)

    from RAG.rag_struct import load_index

    return load_index(model_name=embedding_model)


def ensure_index_exists(mode: str) -> tuple[bool, str]:
    if mode == "classic":
        from RAG.rag import PERSIST_DIR

        path = PERSIST_DIR
    else:
        from RAG.rag_struct import PERSIST_DIR

        path = PERSIST_DIR

    return os.path.exists(path), path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval precision/recall on labeled samples.")
    parser.add_argument("--samples", default=os.path.join("RAG", "eval", "rag_eval_samples_25.json"), help="Path to labeled samples json file.")
    parser.add_argument("--mode", choices=["classic", "struct", "both"], default="both", help="Evaluate classic index, structured index, or both.")
    parser.add_argument("--k", type=int, default=None, help="Override top-k for all samples.")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-zh-v1.5", help="Embedding model name used to load index.")
    parser.add_argument("--out", default=os.path.join("RAG", "eval", "rag_eval_report.json"), help="Output report json path.")

    args = parser.parse_args()

    report = run_evaluation(
        samples_path=args.samples,
        mode=args.mode,
        k_override=args.k,
        embedding_model=args.embedding_model,
        out_path=args.out,
    )

    print("=== RAG Retrieval Evaluation Summary ===")
    modes = ["classic", "struct"] if args.mode == "both" else [args.mode]
    for mode in modes:
        s = report["summary"][mode]
        print(
            f"[{mode}] samples={s['samples']} "
            f"macro_precision@k={s['macro_precision_at_k']:.4f} "
            f"macro_recall@k={s['macro_recall_at_k']:.4f} "
            f"macro_hit@1={s['macro_hit_at_1']:.4f} "
            f"macro_mrr={s['macro_mrr']:.4f}"
        )
    print(f"report saved to: {args.out}")


def run_evaluation(
    samples_path: str,
    mode: str = "both",
    k_override: int | None = None,
    embedding_model: str = "BAAI/bge-base-zh-v1.5",
    out_path: str | None = None,
) -> dict[str, Any]:
    with open(samples_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    modes = ["classic", "struct"] if mode == "both" else [mode]

    report: dict[str, Any] = {
        "samples_file": samples_path,
        "embedding_model": embedding_model,
        "mode": mode,
        "k_override": k_override,
        "summary": {},
        "details": {},
    }

    for current_mode in modes:
        exists, index_path = ensure_index_exists(current_mode)
        if not exists:
            raise FileNotFoundError(
                f"Index path does not exist for mode={current_mode}: {index_path}. Please build index first."
            )

        store = load_store(mode=current_mode, embedding_model=embedding_model)
        results, details = evaluate_mode(store=store, samples=samples, mode=current_mode, k_override=k_override)

        report["summary"][current_mode] = summarize(results)
        report["details"][current_mode] = details

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    main()
