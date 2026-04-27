#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenClaw Skill: Semantic Knowledge Search
功能：自动加载离线语义库，回答用户问题。
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# === 1. 自动环境检查与修复 ===
REQUIRED_LIBS = ['numpy', 'pandas', 'sentence-transformers']

def ensure_environment():
    missing = []
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib.replace("-", "_"))
        except ImportError:
            missing.append(lib)
    
    if missing:
        print(f"🔧 检测到缺失依赖: {missing}，正在自动安装...")
        # 尝试使用 pip 安装
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + missing)
        print("✅ 依赖安装完成！")

# === 2. 核心检索引擎 ===
class SemanticSearchEngine:
    def __init__(self):
        # 自动定位数据目录
        script_dir = Path(__file__).parent
        data_dir = script_dir / "data"
        
        self.model_path = data_dir / "models" / "bge-small-zh-v1.5"
        self.index_dir = data_dir / "index_store"
        self.db_path = self.index_dir / "knowledge.db"
        self.emb_path = self.index_dir / "embeddings.npy"

        self._check_data()

    def _check_data(self):
        if not self.db_path.exists() or not self.emb_path.exists():
            print("❌ 错误：未找到离线语义库 (knowledge.db / embeddings.npy)")
            print("💡 请将数据放置在 skills/semantic-knowledge-search/data/index_store/ 下")
            sys.exit(1)
        
        # 模型检查
        if not (self.model_path.exists() and (self.model_path / "config.json").exists()):
            print("⚠️ 未找到本地 BGE 模型，正在尝试从 HuggingFace/ModelScope 下载...")
            print("(注意：首次下载需要联网，之后可离线运行)")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download("BAAI/bge-small-zh-v1.5", local_dir=str(self.model_path))
                print("✅ 模型下载完成！")
            except Exception as e:
                print(f"❌ 模型下载失败，请手动下载并放入 data/models/bge-small-zh-v1.5: {e}")
                sys.exit(1)

    def search(self, query, top_k=3):
        import numpy as np
        import sqlite3
        from sentence_transformers import SentenceTransformer

        print(f"🧠 正在加载语义模型 (BGE)...")
        model = SentenceTransformer(str(self.model_path), device='cpu')
        
        print(f"🔍 正在检索库中数据...")
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        
        embeddings = np.load(self.emb_path)
        scores = embeddings @ query_vec
        top_idx = np.argsort(scores)[::-1][:top_k]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT file_name, chunk_text FROM chunks")
        chunks = cursor.fetchall()
        conn.close()

        results = []
        for idx in top_idx:
            if scores[idx] > 0.15: # 过滤低相关度
                file_name, text = chunks[idx]
                results.append({
                    "file": file_name,
                    "score": float(scores[idx]),
                    "content": text
                })
        return results

# === 3. 入口点 ===
def main():
    ensure_environment() # 运行前检查环境
    
    if len(sys.argv) < 2:
        print("🚫 请提供搜索内容。例: python main.py 'Q1 销售额是多少'")
        return

    query = " ".join(sys.argv[1:])
    engine = SemanticSearchEngine()
    
    try:
        results = engine.search(query)
        if not results:
            print(f"📭 知识库中未找到关于 '{query}' 的内容。")
        else:
            print(f"✅ 找到 {len(results)} 条高相关结果:")
            for i, res in enumerate(results, 1):
                print(f"\n--- 结果 {i} (匹配度: {res['score']:.3f}) ---")
                print(f"📄 来源: {res['file']}")
                print(f"📝 内容: {res['content'][:200]}...")
    except Exception as e:
        print(f"❌ 运行出错: {e}")

if __name__ == "__main__":
    main()
