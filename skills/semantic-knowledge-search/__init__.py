#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Semantic Knowledge Search — Agent API
用法:
  from semantic_knowledge_search import search, ensure_installed, reindex, status
"""

import os
import sys
import gc
import json
import sqlite3
import subprocess
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ---- 自动定位路径 ----
SKILL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SKILL_DIR.parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "lab5-local-knowledge-assistant" / "knowledge"
INDEX_STORE_DIR = PROJECT_ROOT / "lab5-local-knowledge-assistant" / "index_store"
BGE_MODEL_DIR = PROJECT_ROOT / "lab5-local-knowledge-assistant" / "models" / "bge-small-zh-v1.5"

__all__ = ["search", "ensure_installed", "reindex", "status"]


# ============================================================
# 安装
# ============================================================

def ensure_installed(knowledge_dir: str = None, venv_name: str = None):
    """
    检查并完成安装 + 自动索引。
    如果项目、模型或索引不存在，自动调用 install.py。

    Args:
        knowledge_dir: 知识库路径。不指定则使用项目默认的 knowledge/
    """
    import platform
    if venv_name is None:
        venv_name = "ov" if platform.system() == "Windows" else "ov_workshop"
    _idx = Path(knowledge_dir) if knowledge_dir else INDEX_STORE_DIR

    needs_setup = False
    if not _idx.exists():
        needs_setup = True
    elif not (_idx / "knowledge.db").exists():
        needs_setup = True
    elif not (_idx / "embeddings.npy").exists():
        needs_setup = True

    if needs_setup:
        print("⚠️ 知识库未安装，开始自动安装...")
        install_script = SKILL_DIR / "install.py"
        if not install_script.exists():
            raise RuntimeError(f"安装脚本不存在: {install_script}")
        cmd = [sys.executable, str(install_script)]
        if knowledge_dir:
            cmd.extend(["--knowledge-dir", knowledge_dir])
        subprocess.check_call(cmd)
        print("✅ 安装完成")


# ============================================================
# 搜索
# ============================================================

_search_model = None


def _resolve_paths(knowledge_dir: str = None):
    """Resolve index/model paths based on optional knowledge_dir override."""
    if knowledge_dir:
        kb = Path(knowledge_dir)
        idx = kb.parent / "index_store"
        bge = kb.parent / "models" / "bge-small-zh-v1.5"
    else:
        kb = KNOWLEDGE_DIR
        idx = INDEX_STORE_DIR
        bge = BGE_MODEL_DIR
    return kb, idx, bge


def search(query: str, top_k: int = 3, knowledge_dir: str = None):
    """
    语义搜索知识库。

    Args:
        query: 搜索内容，如 "小学里的英文单词"
        top_k: 返回结果数量，默认 3
        knowledge_dir: 知识库路径，不指定则使用默认

    Returns:
        list[dict]: 每个 dict 包含 file, type, score, content
    """
    kb, idx, bge = _resolve_paths(knowledge_dir)
    db_path = idx / "knowledge.db"
    emb_path = idx / "embeddings.npy"

    if not db_path.exists() or not emb_path.exists():
        raise RuntimeError(
            "索引不存在，请先调用 ensure_installed() 或运行 reindex()"
        )

    import numpy as np

    # Load model with resolved path
    if not (bge.exists() and (bge / "config.json").exists()):
        print("📥 BGE 模型未找到，从 ModelScope 下载...")
        from modelscope import snapshot_download
        snapshot_download(
            "AI-ModelScope/bge-small-zh-v1.5",
            local_dir=str(bge),
        )
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(bge), device="cpu")

    query_vec = model.encode([query], normalize_embeddings=True)[0]
    embeddings = np.load(emb_path)
    scores = embeddings @ query_vec
    top_idx = np.argsort(scores)[::-1][:top_k]

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT file_name, file_type, chunk_text FROM chunks")
        chunks = cursor.fetchall()

    results = []
    for i in top_idx:
        if scores[i] > 0.15:
            file_name, file_type, text = chunks[i]
            results.append({
                "file": file_name,
                "type": file_type,
                "score": round(float(scores[i]), 4),
                "content": text,
            })

    del model
    gc.collect()
    return results


# ============================================================
# 状态
# ============================================================

def status(knowledge_dir: str = None) -> dict:
    """
    查看索引状态。

    Args:
        knowledge_dir: 知识库路径

    Returns:
        dict: {"chunks": int, "files": int, "types": dict, "vector_shape": list or None}
    """
    _, idx, _ = _resolve_paths(knowledge_dir)
    db_path = idx / "knowledge.db"
    emb_path = idx / "embeddings.npy"

    if not db_path.exists():
        return {"error": "索引数据库不存在"}

    import numpy as np

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        files = conn.execute("SELECT DISTINCT file_name FROM chunks").fetchall()
        types = conn.execute(
            "SELECT file_type, COUNT(*) FROM chunks GROUP BY file_type"
        ).fetchall()

    vector_shape = None
    if emb_path.exists():
        data = np.load(emb_path)
        vector_shape = list(data.shape)

    return {
        "chunks": count,
        "files": len(files),
        "types": dict(types),
        "vector_shape": vector_shape,
    }


# ============================================================
# 重新索引
# ============================================================

def reindex(knowledge_dir: str = None, with_ocr: bool = False):
    """
    重新扫描知识库目录并构建向量索引。

    Args:
        knowledge_dir: 知识库路径，默认使用项目内的 knowledge/
        with_ocr: 是否包含图片 OCR（需要 Qwen3-VL 模型）
    """
    index_script = SKILL_DIR / "index_kb.py"
    if not index_script.exists():
        raise RuntimeError(f"索引脚本不存在: {index_script}")

    cmd = [sys.executable, str(index_script)]
    if knowledge_dir:
        cmd.extend(["--knowledge-dir", knowledge_dir])
    if with_ocr:
        cmd.append("--with-ocr")

    subprocess.check_call(cmd)
