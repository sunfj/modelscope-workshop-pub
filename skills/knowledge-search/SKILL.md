---
name: knowledge-search
version: 1.0.0
description: "基于 OpenVINO 和 Qwen3 的本地语义知识库检索助手。支持自然语言提问、OCR 图片识别和文件定位。"
triggers:
  - "search"
  - "find"
  - "lookup"
  - "search knowledge base"
  - "检索知识库"
  - "查找文件"
author: sunfj
---

# 🧠 Knowledge Search Skill

This skill provides semantic search capabilities over a local knowledge base. It uses Qwen3-ASR for voice input, Qwen3-VL for image OCR, and BGE for semantic vector matching.

## 🚀 Usage

Run the following command to search:

```bash
python main.py "你的搜索词"
```

## 📁 Requirements
- Must be run inside the `ov_workshop` environment.
- Requires `lab5-local-knowledge-assistant` module.
