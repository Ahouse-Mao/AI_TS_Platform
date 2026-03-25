import os
import re
import shutil
import itertools
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 优先离线加载本地缓存模型，避免在线 HEAD 请求引发 SSL EOF。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── 修复1: 所有路径基于本文件的绝对位置，不受 CWD 影响 ──
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "..", "model_src", "scripts")
PERSIST_DIR = os.path.join(BASE_DIR, "rag_db")


# ── 修复2 & 3: 解析 shell 变量 + 按每个 python 调用块分别生成描述 ──
def parse_sh_scripts_to_texts(scripts_dir: str) -> list[str]:
    """
    遍历 scripts/ 下所有 .sh 文件：
      - 先解析脚本顶部的 shell 变量赋值（如 model_name=DLinear）
            - 再按每个 `python ...xxx.py` 调用块单独提取参数（兼容 run.py / main_informer.py / run_longExp.py）
      - 将 $xxx 替换为实际变量值
      - 每个调用块生成一条独立的自然语言描述
    """
    script_texts = []

    for root, _, files in os.walk(scripts_dir):
        for file in sorted(files):
            if not file.endswith('.sh'):
                continue
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                content = f.read()

            # step1 解析脚本顶部 shell 变量赋值，如 model_name=DLinear / seq_len=336
            var_map: dict[str, str] = {}
            for m in re.finditer(r'^([A-Za-z_]\w*)=([^\n#]+)', content, re.MULTILINE):
                var_map[m.group(1)] = m.group(2).strip().strip('"\'')

            # step2 解析 for 循环变量，如 for pred_len in 96 192 336 720
            for_loops: dict[str, list[str]] = {}
            for m in re.finditer(r'for\s+(\w+)\s+in\s+(.+)', content):
                var_name = m.group(1)
                values = m.group(2).strip().split()
                for_loops[var_name] = values

            # step3 按任意 python 脚本调用分块，兼容不同训练入口文件
            # 例如：python -u run_longExp.py / python -u run.py / python -u main_informer.py
            blocks = re.split(r'python\s+(?:-u\s+)?[^\s\\]+\.py', content)[1:]

            for block in blocks:
                # 截到输出重定向符（该调用结束处）
                end_match = re.search(r'\s+>', block)
                if end_match:
                    block = block[:end_match.start()]

                # 合并续行（反斜杠换行）
                block = block.replace('\\\n', ' ')

                # 提取 --key value 对
                raw_params = dict(re.findall(r'--([a-zA-Z0-9_]+)\s+([^\s\\]+)', block))

                # 将 $variable 替换为实际变量值；收集需要展开的 for 循环变量
                params: dict[str, str] = {}
                expand_keys: dict[str, list[str]] = {}
                for k, v in raw_params.items():
                    if v.startswith('$'):
                        var_name = v[1:].strip("'\"")
                        if var_name in var_map:
                            params[k] = var_map[var_name]
                        elif var_name in for_loops:
                            expand_keys[k] = for_loops[var_name]
                            params[k] = None  # 占位，后续展开
                        else:
                            params[k] = v
                    else:
                        params[k] = v

                # 必须有 model；数据集优先 data_path，其次使用 data
                if 'model' not in params:
                    continue

                dataset_name = ''
                if params.get('data_path'):
                    dataset_name = params['data_path'].split('/')[-1].split('\\\\')[-1].split('.')[0]
                elif params.get('data'):
                    dataset_name = params['data']

                if not dataset_name:
                    continue

                # 构建需要展开的参数组合列表
                if expand_keys:
                    # 按第一个 for 循环变量展开（最常见的情况）
                    keys = list(expand_keys.keys())
                    value_lists = [expand_keys[k] for k in keys]
                    combos = list(itertools.product(*value_lists))
                else:
                    keys = []
                    combos = [()]  # 无需展开时仍遍历一次

                # step4 展开for循环的每种组合, 生成自然语言描述
                for combo in combos:
                    expanded = dict(params)
                    for i, k in enumerate(keys):
                        expanded[k] = combo[i]

                    model_name = expanded['model']
                    current_dataset = dataset_name
                    if expanded.get('data_path'):
                        current_dataset = expanded['data_path'].split('/')[-1].split('\\\\')[-1].split('.')[0]
                    elif expanded.get('data'):
                        current_dataset = expanded['data']

                    parts = [f"使用 {model_name} 模型对 {current_dataset} 数据集训练时推荐配置"]
                    if 'seq_len'       in expanded and expanded['seq_len']:       parts.append(f"输入序列长度(seq_len)={expanded['seq_len']}")
                    if 'pred_len'      in expanded and expanded['pred_len']:      parts.append(f"预测长度(pred_len)={expanded['pred_len']}")
                    if 'label_len'     in expanded and expanded['label_len']:     parts.append(f"标签长度(label_len)={expanded['label_len']}")
                    if 'learning_rate' in expanded and expanded['learning_rate']: parts.append(f"学习率(learning_rate)={expanded['learning_rate']}")
                    if 'batch_size'    in expanded and expanded['batch_size']:    parts.append(f"批大小(batch_size)={expanded['batch_size']}")
                    if 'd_model'       in expanded and expanded['d_model']:       parts.append(f"模型维度(d_model)={expanded['d_model']}")
                    if 'n_heads'       in expanded and expanded['n_heads']:       parts.append(f"注意力头数(n_heads)={expanded['n_heads']}")
                    if 'e_layers'      in expanded and expanded['e_layers']:      parts.append(f"编码器层数(e_layers)={expanded['e_layers']}")
                    if 'patch_len'     in expanded and expanded['patch_len']:     parts.append(f"Patch长度(patch_len)={expanded['patch_len']}")
                    if 'stride'        in expanded and expanded['stride']:        parts.append(f"步幅(stride)={expanded['stride']}")
                    if 'patch_size'    in expanded and expanded['patch_size']:    parts.append(f"Patch大小(patch_size)={expanded['patch_size']}")
                    if 'patch_stride'  in expanded and expanded['patch_stride']:  parts.append(f"Patch步幅(patch_stride)={expanded['patch_stride']}")

                    script_texts.append('，'.join(parts) + '。')

    return script_texts


def build_index(model_name: str = "BAAI/bge-base-zh-v1.5", log_fn=print) -> Chroma:
    """构建并持久化向量索引，仅需运行一次。
    
    log_fn: 日志回调，默认 print；传入 list.append 可将日志写入缓冲区。
    """

    # 1. 初始化 BGE Embedding 模型
    log_fn(f"[rag] 正在加载嵌入模型: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            'device': 'cuda' if os.getenv("USE_CUDA", "0") == "1" else "cpu",
            'local_files_only': True,
        }
    )
    log_fn(f"[rag] 嵌入模型加载完成")

    # 2. 手写的模型元信息（通用背景知识）
    texts = [
        "PatchTST模型：适用于长序列预测，利用了Transformer架构和Patching技术，计算效率高且能捕获局部语义，速度较快。",
        "DLinear模型：一个极其简单的线性模型，在某些明显带有周期性和趋势性的单变量或多变量数据集上表现极好，且训练极快。",
        "NLinear模型：在DLinear基础上加入了减去最后一个时间步的归一化技巧，在分布偏移场景下效果更好，速度快。",
        "Autoformer模型：采用分解架构，内置了序列分解模块，并在自注意力机制上做了创新，适合复杂时序，但是速度较慢。",
        "Informer模型：使用ProbSparse稀疏注意力机制，适合超长序列预测，降低了Transformer的平方复杂度，但是速度较慢。",
        "Transformer模型：标准Transformer用于时序预测，适合中等长度序列，配置灵活但计算量较大，速度很慢。",
    ]

    # 3. 从 .sh 脚本动态提取参数推荐
    if os.path.exists(SCRIPTS_DIR):
        extracted = parse_sh_scripts_to_texts(SCRIPTS_DIR)
        log_fn(f"[rag] ✓ 从脚本提取了 {len(extracted)} 条参数推荐规则")
        texts.extend(extracted)
    else:
        log_fn(f"[rag] ⚠ 未找到 scripts 目录: {SCRIPTS_DIR}")

    # 4. 构建向量库并持久化
    log_fn(f"[rag] 正在写入向量数据库（共 {len(texts)} 条文档）...")
    vectorstore = Chroma.from_texts(texts, embeddings, persist_directory=PERSIST_DIR)
    log_fn(f"[rag] ✓ 知识库构建完成，已持久化到: {PERSIST_DIR}")
    return vectorstore


def load_index(model_name: str = "BAAI/bge-base-zh-v1.5") -> Chroma:
    """加载已有向量库（供 FastAPI 在线检索调用，不重新构建）。"""
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            'device': 'cuda' if os.getenv("USE_CUDA", "0") == "1" else "cpu",
            'local_files_only': True,
        }
    )
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def clear_index() -> dict:
    """删除持久化向量库目录，返回操作结果。"""
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        return {"success": True, "message": f"已清除向量库: {PERSIST_DIR}"}
    return {"success": False, "message": "向量库目录不存在，无需清除"}
