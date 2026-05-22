#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenClaw Skill: Semantic Knowledge Search - 命令行检索入口
用法:
  python main.py "查询内容"                  语义搜索
  python main.py --reindex                   重新索引知识库
  python main.py --status                    查看索引状态
  python main.py --install                   运行一键安装
"""

import os
import sys
import json
import gc
import sqlite3
import numpy as np
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ---- 自动定位路径 ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "lab5-local-knowledge-assistant" / "knowledge"
INDEX_STORE_DIR = PROJECT_ROOT / "lab5-local-knowledge-assistant" / "index_store"
BGE_MODEL_DIR = PROJECT_ROOT / "lab5-local-knowledge-assistant" / "models" / "bge-small-zh-v1.5"


def ensure_installed(knowledge_dir=None):
    """Check if index exists; if not, auto-run indexing."""
    kb = Path(knowledge_dir) if knowledge_dir else KNOWLEDGE_DIR
    idx = kb.parent / "index_store"

    if not idx.exists() or not (idx / "knowledge.db").exists():
        print("⚠️ 未检测到已索引的知识库，开始自动索引...")
        cmd_reindex(knowledge_dir=knowledge_dir)
        print("✅ 索引完成")


# ============================================================
# 检索引擎
# ============================================================

class SemanticSearchEngine:
    def __init__(self, knowledge_dir=None, index_store_dir=None, bge_model_dir=None):
        self.knowledge_dir = knowledge_dir or KNOWLEDGE_DIR
        self.index_store_dir = index_store_dir or INDEX_STORE_DIR
        self.bge_model_dir = bge_model_dir or BGE_MODEL_DIR

        self.db_path = self.index_store_dir / "knowledge.db"
        self.emb_path = self.index_store_dir / "embeddings.npy"

        self.model = None  # lazy load

    def _load_model(self):
        if self.model is not None:
            return
        print(f"🧠 加载语义模型 (BGE)...")
        if not (self.bge_model_dir.exists() and (self.bge_model_dir / "config.json").exists()):
            print("📥 语义模型未找到，从 ModelScope 下载...")
            try:
                from modelscope import snapshot_download
                snapshot_download(
                    "AI-ModelScope/bge-small-zh-v1.5",
                    local_dir=str(self.bge_model_dir),
                )
                print("✅ BGE 模型下载完成")
            except Exception as e:
                print(f"❌ 模型下载失败: {e}")
                sys.exit(1)

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(self.bge_model_dir), device="cpu")

    def search(self, query, top_k=3):
        self._load_model()

        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        embeddings = np.load(self.emb_path)
        scores = embeddings @ query_vec
        top_idx = np.argsort(scores)[::-1][:top_k]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT file_name, file_type, chunk_text FROM chunks")
            chunks = cursor.fetchall()

        results = []
        for idx in top_idx:
            if scores[idx] > 0.15:
                file_name, file_type, text = chunks[idx]
                results.append({
                    "file": file_name,
                    "type": file_type,
                    "score": round(float(scores[idx]), 4),
                    "content": text,
                })
        return results

    def status(self):
        if not self.db_path.exists():
            print("❌ 索引数据库不存在")
            return
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            types = conn.execute(
                "SELECT file_type, COUNT(*) FROM chunks GROUP BY file_type"
            ).fetchall()
            files = conn.execute(
                "SELECT DISTINCT file_name FROM chunks"
            ).fetchall()

        emb_size = "未知"
        if self.emb_path.exists():
            data = np.load(self.emb_path)
            emb_size = f"{data.shape[0]} x {data.shape[1]}"

        print(f"📊 索引状态:")
        print(f"  文本块总数: {count}")
        print(f"  文件总数: {len(files)}")
        print(f"  向量维度: {emb_size}")
        print(f"  类型分布:")
        for ftype, cnt in types:
            print(f"    {ftype}: {cnt}")


def format_results(results):
    if not results:
        return "📭 知识库中未找到相关内容。"
    lines = [f"✅ 找到 {len(results)} 条高相关结果:"]
    for i, res in enumerate(results, 1):
        lines.append(f"\n--- 结果 {i} (匹配度: {res['score']:.3f}) ---")
        lines.append(f"📄 来源: {res['file']} ({res['type']})")
        # Show more content for shorter results
        preview_len = 300 if len(res["content"]) < 500 else 200
        lines.append(f"📝 内容: {res['content'][:preview_len]}...")
    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def cmd_search(query, top_k=3, json_output=False, knowledge_dir=None):
    ensure_installed(knowledge_dir=knowledge_dir)

    kb = Path(knowledge_dir) if knowledge_dir else KNOWLEDGE_DIR
    idx = kb.parent / "index_store"
    bge = kb.parent / "models" / "bge-small-zh-v1.5"

    engine = SemanticSearchEngine(knowledge_dir=knowledge_dir, index_store_dir=idx, bge_model_dir=bge)
    results = engine.search(query, top_k=top_k)

    if json_output:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(format_results(results))

    del engine.model
    gc.collect()


def cmd_reindex(knowledge_dir=None, with_ocr=False):
    # Delegate to index_kb.py
    index_script = SCRIPT_DIR / "index_kb.py"
    if not index_script.exists():
        print("❌ 索引脚本不存在: index_kb.py")
        sys.exit(1)

    cmd = [sys.executable, str(index_script)]
    if knowledge_dir:
        cmd.extend(["--knowledge-dir", str(knowledge_dir)])
    if with_ocr:
        cmd.append("--with-ocr")

    subprocess = __import__("subprocess")
    subprocess.check_call(cmd)


def cmd_status(knowledge_dir=None):
    ensure_installed(knowledge_dir=knowledge_dir)

    kb = Path(knowledge_dir) if knowledge_dir else KNOWLEDGE_DIR
    idx = kb.parent / "index_store"
    bge = kb.parent / "models" / "bge-small-zh-v1.5"

    engine = SemanticSearchEngine(knowledge_dir=knowledge_dir, index_store_dir=idx, bge_model_dir=bge)
    engine.status()
    del engine.model
    gc.collect()


def cmd_install(knowledge_dir=None):
    install_script = SCRIPT_DIR / "install.py"
    if not install_script.exists():
        print("❌ 安装脚本不存在: install.py")
        sys.exit(1)
    subprocess = __import__("subprocess")
    cmd = [sys.executable, str(install_script)]
    if knowledge_dir:
        cmd.extend(["--knowledge-dir", knowledge_dir])
    subprocess.check_call(cmd)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="🧠 本地语义知识库检索引擎 (OpenClaw Skill)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py "帮我找一下销售额相关的数据"
  python main.py "小学里的英文单词" --top-k 5
  python main.py --reindex
  python main.py --status
  python main.py --install
        """,
    )
    parser.add_argument("query", nargs="?", default=None, help="搜索内容")
    parser.add_argument("--top-k", type=int, default=3, help="返回结果数量 (默认 3)")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")
    parser.add_argument("--reindex", action="store_true", help="重新索引知识库")
    parser.add_argument("--status", action="store_true", help="查看索引状态")
    parser.add_argument("--install", action="store_true", help="运行一键安装")
    parser.add_argument("--with-ocr", action="store_true", help="索引时包含图片 OCR")
    parser.add_argument("--knowledge-dir", type=str, default=None, help="指定知识库路径")

    args = parser.parse_args()

    if args.install:
        cmd_install(knowledge_dir=args.knowledge_dir)
        return

    if args.reindex:
        cmd_reindex(
            knowledge_dir=args.knowledge_dir,
            with_ocr=args.with_ocr,
        )
        return

    if args.status:
        cmd_status(knowledge_dir=args.knowledge_dir)
        return

    if args.query is None:
        parser.print_help()
        print("\n🚫 请提供搜索内容。例: python main.py 'Q1 销售额是多少'")
        return

    cmd_search(args.query, top_k=args.top_k, json_output=args.json, knowledge_dir=args.knowledge_dir)


if __name__ == "__main__":
    main()
