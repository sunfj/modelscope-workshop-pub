---
name: semantic-knowledge-search
version: 5.0.0
description: "离线版语义知识库检索。Agent 可直接 import 使用，自动安装、索引、搜索。"
triggers:
  - "搜索知识库"
  - "查找离线文档"
  - "检索本地数据"
  - "semantic search"
  - "知识库助手"
  - "lab5"
author: sunfj
---

# 🧠 Semantic Knowledge Search

完全本地的语义知识库，支持文档/PPT/Excel/PDF/图片 OCR 入库，BGE 向量语义搜索。

## Agent 使用方式

### 1. 首次使用 — 自动安装 + 索引

```python
from semantic_knowledge_search import search, ensure_installed, reindex, status

# 使用默认知识库路径
ensure_installed()

# 或指定自定义知识库目录（包含 txt/docx/xlsx/pptx/pdf/jpg 等文件的目录）
ensure_installed(knowledge_dir="/path/to/my/documents")
```

`ensure_installed()` 会自动：
- 检查项目环境，不存在则调用 `install.py` 下载项目+依赖+模型
- **首次自动索引**知识库目录，无需手动 `reindex()`

### 2. 搜索

```python
results = search("小学里的英文单词有哪些")
for r in results:
    print(f"[{r['score']:.3f}] {r['file']}: {r['content'][:200]}")
```

返回格式：
```python
[
  {
    "file": "小学核心词汇x1.jpeg",
    "type": ".jpeg",
    "score": 0.7234,
    "content": "小学英语3-6年级576个核心英文词汇..."
  },
  ...
]
```

也可指定自定义知识库：
```python
results = search("预算审批", knowledge_dir="/path/to/my/documents")
```

### 3. 管理索引

```python
# 查看当前索引状态
info = status()
# {"chunks": 6, "files": 6, "types": {".docx": 1, ...}, "vector_shape": [6, 512]}

# 添加新文件后重新索引
reindex()

# 指定知识库路径 + 开启图片 OCR
reindex(knowledge_dir="/path/to/my/documents", with_ocr=True)
```

## 注意事项

- **首次使用较慢**：需要下载 BGE 模型 (~100MB) 和项目代码
- **OCR 可选**：图片 OCR 需要额外下载 Qwen3-VL 模型 (~2GB)，默认跳过
- **完全离线**：安装和索引完成后，搜索无需联网
- **内存友好**：搜索时 BGE 模型约占用 500MB RAM

## 文件结构

```
skills/semantic-knowledge-search/
├── SKILL.md          # 本文件 — agent 读取用
├── __init__.py       # Agent import 入口
├── install.py        # 一键安装
├── index_kb.py       # 知识库索引
└── main.py           # CLI 入口 (可选)

lab5-local-knowledge-assistant/
├── knowledge/        # 知识库原始文件 (agent 可添加新文件到这里)
├── index_store/      # 向量索引 (自动生成)
└── models/           # BGE 模型 (自动下载)
```
