#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenClaw Skill: Knowledge Search (API Client)
Usage: python main.py "查一查销售额相关的数据"
"""
import sys
import requests

def search_via_api(query):
    url = "http://127.0.0.1:7860/api/search"
    print(f"📡 正在请求本地服务: {url}")
    print(f"🗣️ 您的查询: {query}")
    
    try:
        # 发送 POST 请求
        res = requests.post(url, json={"query": query}, timeout=5)
        res.raise_for_status()
        data = res.json()
        
        print("\n📊 检索结果：")
        print("=" * 60)
        for item in data.get("results", []):
            print(f"🎯 匹配度: {item['score']:.4f}")
            print(f"📄 内容: {item['text'][:100]}...")
            print("-" * 60)
            
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败！请确保 Gradio 服务已在 7860 端口运行。")
    except Exception as e:
        print(f"❌ 请求出错: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        search_via_api(" ".join(sys.argv[1:]))
    else:
        print("🚫 请提供搜索关键词。")
