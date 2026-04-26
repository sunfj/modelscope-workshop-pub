#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenClaw Skill Entry Point: Knowledge Search
功能：直接加载本地索引和 BGE 模型，进行真实的语义搜索。
用法: python main.py "Q1 的用户增长率"
"""

import sys
import os
import json
import numpy as np
import sqlite3
from pathlib import Path

def run_real_search(query):
    print(f"🔍 正在检索知识库：'{query}' ...")
    
    # 1. 路径探测 (找到 project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # skills/knowledge-search -> skills -> project_root
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    index_dir = Path(project_root) / "lab5-local-knowledge-assistant" / "index_store"
    db_path = index_dir / "knowledge.db"
    emb_path = index_dir / "embeddings.npy"
    
    # 2. 检查数据是否存在
    if not db_path.exists() or not emb_path.exists():
        print("❌ 错误：未找到索引数据！")
        print(f"💡 路径检查: {index_dir}")
        print("💡 请先在 JupyterLab 中运行 Cell 2 (索引引擎) 以生成数据库和向量文件。")
        return

    # 3. 加载语义模型
    print("🧠 加载 BGE 语义模型...")
    try:
        from sentence_transformers import SentenceTransformer
        # 使用与 Notebook 一致的路径或默认下载
        model_name = "BAAI/bge-small-zh-v1.5"
        
        # 尝试查找本地下载的模型 (适配各种环境)
        possible_paths = [
            "/mnt/workspace/modelscope-workshop-pub/models/bge-small-zh-v1.5", # 云端路径
            os.path.join(project_root, "models/bge-small-zh-v1.5")            # 本地路径
        ]
        
        local_model_path = None
        for p in possible_paths:
            if os.path.exists(p) and os.path.exists(os.path.join(p, "config.json")):
                local_model_path = p
                break
        
        if local_model_path:
            print(f"✅ 找到本地模型: {local_model_path}")
            model = SentenceTransformer(local_model_path, device='cpu')
        else:
            print(f"📥 本地未找到模型，正在下载: {model_name}")
            model = SentenceTransformer(model_name, device='cpu')
            
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        print("💡 请确保已安装 sentence-transformers。")
        return

    # 4. 执行搜索逻辑
    try:
        # 加载向量
        embeddings = np.load(emb_path)
        
        # 加载数据库内容
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_path, file_type, chunk_text, file_name FROM chunks")
        chunks = cursor.fetchall()
        conn.close()

        # 计算 Query 向量
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        
        # 计算余弦相似度 (矩阵乘法)
        scores = embeddings @ query_vec
        
        # 排序
        top_k = 5
        top_idx = np.argsort(scores)[::-1][:top_k]

        print(f"\n📊 找到 {len(top_idx)} 条高匹配结果：")
        print("=" * 60)
        
        found_any = False
        for idx in top_idx:
            if scores[idx] > 0.15: # 阈值过滤
                chunk = chunks[idx]
                chunk_id, f_path, f_type, text, f_name = chunk
                score = float(scores[idx])
                
                print(f"📄 文件：{f_name}")
                print(f"📊 匹配度：{score:.4f}")
                print(f"📝 内容摘要：{text[:100]}...")
                print("-" * 60)
                found_any = True
            else:
                break
        
        if not found_any:
            print("😕 未找到匹配度高于阈值的结果。")
                
    except Exception as e:
        print(f"❌ 搜索出错：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_real_search(query)
    else:
        print("🚫 请提供搜索关键词。")
        print("示例: python main.py 'Q1 总结'")
