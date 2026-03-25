import itertools
import importlib
import os
import re
import shutil
import warnings
from collections import Counter
from typing import Any

try:
    HuggingFaceEmbeddings = importlib.import_module("langchain_huggingface").HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = importlib.import_module("langchain_community.embeddings").HuggingFaceEmbeddings

try:
    Chroma = importlib.import_module("langchain_chroma").Chroma
except ImportError:
    Chroma = importlib.import_module("langchain_community.vectorstores").Chroma

# 优先离线加载本地缓存模型，避免在线 HEAD 请求引发 SSL EOF。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 在仍使用旧包回退时，避免控制台被弃用警告刷屏。
warnings.filterwarnings("ignore", message=r".*HuggingFaceEmbeddings.*deprecated.*")
warnings.filterwarnings("ignore", message=r".*Chroma.*deprecated.*")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "..", "model_src", "scripts")
PERSIST_DIR = os.path.join(BASE_DIR, "rag_db_struct")

KNOWN_MODELS = [
    "PatchTST",
    "DLinear",
    "NLinear",
    "Linear",
    "Autoformer",
    "Informer",
    "Transformer",
    "ModernTCN",
]

KNOWN_DATASETS = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "WTH",
    "weather",
    "traffic",
    "electricity",
    "exchange_rate",
    "ili",
    "illness",
    "ECL",
    "Exchange",
    "ILI",
]

_EMBEDDINGS_CACHE: dict[str, Any] = {}
_VECTORSTORE_CACHE: dict[str, Any] = {}


def _normalize_dataset_name(data_path: str | None, data_name: str | None) -> str:
    if data_path:
        return data_path.split("/")[-1].split("\\")[-1].split(".")[0]
    return (data_name or "").strip()


def _extract_var_map(content: str) -> dict[str, str]:
    var_map: dict[str, str] = {}
    for m in re.finditer(r"^([A-Za-z_]\w*)=([^\n#]+)", content, re.MULTILINE):
        var_map[m.group(1)] = m.group(2).strip().strip("\"'")
    return var_map


def _extract_for_loops(content: str) -> dict[str, list[str]]:
    for_loops: dict[str, list[str]] = {}
    for m in re.finditer(r"for\s+(\w+)\s+in\s+([^\n;]+)", content):
        name = m.group(1)
        values = m.group(2).strip().split()
        if values:
            for_loops[name] = values
    return for_loops


def _extract_python_blocks(content: str) -> list[tuple[str, str]]:
    """Return [(entry_py, args_block), ...] for each python invocation in shell script."""
    pattern = re.compile(
        r"python\s+(?:-u\s+)?([^\s\\]+\.py)(.*?)(?=\n\s*python\s+(?:-u\s+)?[^\s\\]+\.py|\Z)",
        re.DOTALL,
    )

    blocks: list[tuple[str, str]] = []
    for m in pattern.finditer(content):
        entry_py = m.group(1).strip()
        args_block = m.group(2)
        blocks.append((entry_py, args_block))
    return blocks


def _extract_params(args_block: str) -> dict[str, str]:
    normalized = args_block.replace("\\\n", " ").replace("\n", " ")

    # Cut at shell redirection if present.
    end_match = re.search(r"\s+>", normalized)
    if end_match:
        normalized = normalized[: end_match.start()]

    return dict(re.findall(r"--([a-zA-Z0-9_]+)\s+([^\s\\]+)", normalized))


def _expand_params(
    raw_params: dict[str, str],
    var_map: dict[str, str],
    for_loops: dict[str, list[str]],
) -> list[dict[str, str]]:
    params: dict[str, str | None] = {}
    expand_keys: dict[str, list[str]] = {}

    for k, v in raw_params.items():
        if v.startswith("$"):
            var_name = v[1:].strip("'\"")
            if var_name in var_map:
                params[k] = var_map[var_name]
            elif var_name in for_loops:
                params[k] = None
                expand_keys[k] = for_loops[var_name]
            else:
                params[k] = v
        else:
            params[k] = v

    if not expand_keys:
        return [{k: (v or "") for k, v in params.items()}]

    keys = list(expand_keys.keys())
    value_lists = [expand_keys[k] for k in keys]
    expanded_list: list[dict[str, str]] = []
    for combo in itertools.product(*value_lists):
        expanded = dict(params)
        for i, key in enumerate(keys):
            expanded[key] = combo[i]
        expanded_list.append({k: (v or "") for k, v in expanded.items()})

    return expanded_list


def parse_sh_scripts_to_records(scripts_dir: str) -> list[dict[str, Any]]:
    """Parse scripts and return records with both text and structured metadata."""
    records: list[dict[str, Any]] = []

    for root, _, files in os.walk(scripts_dir):
        for file in sorted(files):
            if not file.endswith(".sh"):
                continue

            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, scripts_dir).replace("\\", "/")
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()

            var_map = _extract_var_map(content)
            for_loops = _extract_for_loops(content)
            blocks = _extract_python_blocks(content)

            for entry_py, args_block in blocks:
                raw_params = _extract_params(args_block)
                for params in _expand_params(raw_params, var_map, for_loops):
                    model_name = params.get("model") or var_map.get("model_name") or ""
                    dataset_name = _normalize_dataset_name(
                        params.get("data_path"),
                        params.get("data"),
                    )

                    if not model_name or not dataset_name:
                        continue

                    parts = [f"使用 {model_name} 模型对 {dataset_name} 数据集训练时推荐配置"]
                    for key, label in [
                        ("features", "特征模式(features)"),
                        ("seq_len", "输入序列长度(seq_len)"),
                        ("label_len", "标签长度(label_len)"),
                        ("pred_len", "预测长度(pred_len)"),
                        ("learning_rate", "学习率(learning_rate)"),
                        ("batch_size", "批大小(batch_size)"),
                        ("d_model", "模型维度(d_model)"),
                        ("n_heads", "注意力头数(n_heads)"),
                        ("e_layers", "编码器层数(e_layers)"),
                        ("d_layers", "解码器层数(d_layers)"),
                        ("patch_len", "Patch长度(patch_len)"),
                        ("stride", "步幅(stride)"),
                        ("patch_size", "Patch大小(patch_size)"),
                        ("patch_stride", "Patch步幅(patch_stride)"),
                    ]:
                        value = params.get(key)
                        if value:
                            parts.append(f"{label}={value}")

                    metadata = {
                        "source_type": "script",
                        "script_path": rel_path,
                        "script_name": file,
                        "entry_py": entry_py,
                        "model": model_name,
                        "dataset": dataset_name,
                        "features": params.get("features", ""),
                        "seq_len": params.get("seq_len", ""),
                        "label_len": params.get("label_len", ""),
                        "pred_len": params.get("pred_len", ""),
                    }

                    records.append(
                        {
                            "text": "，".join(parts) + "。",
                            "metadata": metadata,
                        }
                    )

    return records


def _base_manual_records() -> list[dict[str, Any]]:
    items = [
        (
            "PatchTST",
            "PatchTST模型：适用于长序列预测，利用了Transformer架构和Patching技术，计算效率高且能捕获局部语义，速度较快。",
        ),
        (
            "DLinear",
            "DLinear模型：一个极其简单的线性模型，在某些明显带有周期性和趋势性的单变量或多变量数据集上表现极好，且训练极快。",
        ),
        (
            "NLinear",
            "NLinear模型：在DLinear基础上加入了减去最后一个时间步的归一化技巧，在分布偏移场景下效果更好，速度快。",
        ),
        (
            "Autoformer",
            "Autoformer模型：采用分解架构，内置了序列分解模块，并在自注意力机制上做了创新，适合复杂时序，但是速度较慢。",
        ),
        (
            "Informer",
            "Informer模型：使用ProbSparse稀疏注意力机制，适合超长序列预测，降低了Transformer的平方复杂度，但是速度较慢。",
        ),
        (
            "Transformer",
            "Transformer模型：标准Transformer用于时序预测，适合中等长度序列，配置灵活但计算量较大，速度很慢。",
        ),
    ]

    return [
        {
            "text": text,
            "metadata": {
                "source_type": "manual",
                "script_path": "",
                "script_name": "",
                "entry_py": "",
                "model": model,
                "dataset": "",
                "features": "",
                "seq_len": "",
                "label_len": "",
                "pred_len": "",
            },
        }
        for model, text in items
    ]


def _summarize_model_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for rec in records:
        model = rec["metadata"].get("model", "")
        if model:
            counter[model] += 1
    return dict(counter)


def _get_embeddings(model_name: str):
    cached = _EMBEDDINGS_CACHE.get(model_name)
    if cached is not None:
        return cached

    emb = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cuda" if os.getenv("USE_CUDA", "0") == "1" else "cpu",
            "local_files_only": True,
        },
    )
    _EMBEDDINGS_CACHE[model_name] = emb
    return emb


def _get_vectorstore(model_name: str):
    cache_key = f"{model_name}::{PERSIST_DIR}"
    cached = _VECTORSTORE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    store = Chroma(persist_directory=PERSIST_DIR, embedding_function=_get_embeddings(model_name))
    _VECTORSTORE_CACHE[cache_key] = store
    return store


def build_index(model_name: str = "BAAI/bge-base-zh-v1.5", log_fn=print):
    """Build and persist a structured Chroma index."""
    log_fn(f"[rag_struct] loading embedding model: {model_name}")
    embeddings = _get_embeddings(model_name)
    log_fn("[rag_struct] embedding model loaded")

    records = _base_manual_records()
    if os.path.exists(SCRIPTS_DIR):
        extracted = parse_sh_scripts_to_records(SCRIPTS_DIR)
        records.extend(extracted)
        log_fn(f"[rag_struct] extracted script records: {len(extracted)}")
    else:
        log_fn(f"[rag_struct] scripts dir not found: {SCRIPTS_DIR}")

    texts = [r["text"] for r in records]
    metadatas = [r["metadata"] for r in records]

    log_fn(f"[rag_struct] writing vector db with total records: {len(records)}")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=PERSIST_DIR,
    )

    per_model = _summarize_model_counts(records)
    log_fn(f"[rag_struct] build done. persist dir: {PERSIST_DIR}")
    log_fn(f"[rag_struct] model distribution: {per_model}")

    # 重建后刷新缓存，确保后续检索读取最新索引。
    _VECTORSTORE_CACHE.pop(f"{model_name}::{PERSIST_DIR}", None)
    return vectorstore


def load_index(model_name: str = "BAAI/bge-base-zh-v1.5"):
    """Load persisted structured Chroma index."""
    return _get_vectorstore(model_name)


def search(
    query: str,
    k: int = 3,
    model: str | None = None,
    dataset: str | None = None,
    model_name: str = "BAAI/bge-base-zh-v1.5",
):
    """Similarity search with optional metadata filter."""
    store = load_index(model_name=model_name)

    where: dict[str, str] = {}
    if model:
        where["model"] = model
    if dataset:
        where["dataset"] = dataset

    if where:
        return store.similarity_search(query, k=k, filter=where)
    return store.similarity_search(query, k=k)


def clear_index() -> dict:
    """Delete persisted structured index directory."""
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        _VECTORSTORE_CACHE.clear()
        return {"success": True, "message": f"removed structured index: {PERSIST_DIR}"}
    return {"success": False, "message": "structured index directory not found"}


def _extract_query_filters(query: str) -> dict[str, str]:
    q = query.strip()
    q_lower = q.lower()

    extracted: dict[str, str] = {}

    # model
    for m in KNOWN_MODELS:
        if m.lower() in q_lower:
            extracted["model"] = m
            break

    # dataset
    for d in KNOWN_DATASETS:
        if d.lower() in q_lower:
            extracted["dataset"] = d
            break

    # pred_len patterns: pred_len=96 / 96步预测 / 96 预测
    pred = re.search(r"pred_len\s*[:=]\s*(\d+)", q_lower)
    if not pred:
        pred = re.search(r"(\d+)\s*步\s*预测", q)
    if not pred:
        pred = re.search(r"(\d+)\s*预测", q)
    if pred:
        extracted["pred_len"] = pred.group(1)

    # features: MS > M > S
    if re.search(r"\bms\b|多\s*变量\s*(?:到|->|→)\s*单\s*变量", q_lower):
        extracted["features"] = "MS"
    elif re.search(r"\bm\b|多\s*变量", q_lower):
        extracted["features"] = "M"
    elif re.search(r"\bs\b|单\s*变量", q_lower):
        extracted["features"] = "S"

    return extracted


def _dedupe_docs(docs: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for d in docs:
        md = getattr(d, "metadata", None) or {}
        key = "|".join(
            [
                str(md.get("script_path", "")),
                str(md.get("model", "")),
                str(md.get("dataset", "")),
                str(md.get("pred_len", "")),
                str(md.get("features", "")),
                str(getattr(d, "page_content", "")[:60]),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(d)
    return result


def retrieve(query: str, k: int = 3, model_name: str = "BAAI/bge-base-zh-v1.5") -> list[Any]:
    """
    Structured retrieval optimized for precision:
    1) parse query to metadata filters
    2) strict filtered retrieval first
    3) gradually relax filters
    4) fallback to global similarity search
    """
    store = load_index(model_name=model_name)
    filters = _extract_query_filters(query)

    candidates: list[Any] = []

    # If query is specific (dataset/pred_len), prefer script records first.
    specific = ("dataset" in filters) or ("pred_len" in filters)
    filter_steps: list[dict[str, str]] = []

    if filters:
        step_full = dict(filters)
        if specific:
            step_full["source_type"] = "script"
        filter_steps.append(step_full)

        for drop_key in ["features", "pred_len", "dataset"]:
            if drop_key in filters:
                relaxed = {k1: v1 for k1, v1 in filters.items() if k1 != drop_key}
                if relaxed:
                    if specific:
                        relaxed = dict(relaxed)
                        relaxed["source_type"] = "script"
                    filter_steps.append(relaxed)

        if "model" in filters:
            model_only = {"model": filters["model"]}
            if specific:
                model_only["source_type"] = "script"
            filter_steps.append(model_only)

        if "dataset" in filters:
            dataset_only = {"dataset": filters["dataset"]}
            if specific:
                dataset_only["source_type"] = "script"
            filter_steps.append(dataset_only)

    for f in filter_steps:
        try:
            docs = store.similarity_search(query, k=k, filter=f)
        except Exception:
            docs = []
        if docs:
            candidates.extend(docs)
            candidates = _dedupe_docs(candidates)
        if len(candidates) >= k:
            return candidates[:k]

    # Fallback: no/insufficient filtered result
    candidates.extend(store.similarity_search(query, k=k))
    candidates = _dedupe_docs(candidates)
    return candidates[:k]
