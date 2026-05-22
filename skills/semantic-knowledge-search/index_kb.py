#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识库索引脚本 - 供 install.py 调用，也可独立运行。
扫描 knowledge/ 目录，加载 BGE 模型生成向量并保存到 index_store/。
"""

import os
import sys
import gc
import sqlite3
import numpy as np
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ---- 路径 ----
# 默认：script 所在 repo 的 lab5 目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "lab5-local-knowledge-assistant" / "knowledge"
INDEX_STORE_DIR = PROJECT_ROOT / "lab5-local-knowledge-assistant" / "index_store"
BGE_MODEL_DIR = PROJECT_ROOT / "lab5-local-knowledge-assistant" / "models" / "bge-small-zh-v1.5"


def download_model_if_missing():
    if BGE_MODEL_DIR.exists() and (BGE_MODEL_DIR / "config.json").exists():
        return
    print("📥 语义模型未找到，从 ModelScope 下载...")
    try:
        from modelscope import snapshot_download
        snapshot_download(
            "AI-ModelScope/bge-small-zh-v1.5",
            local_dir=str(BGE_MODEL_DIR),
        )
        print("✅ BGE 模型下载完成")
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        sys.exit(1)


def parse_file(path: Path) -> str:
    """Extract text from a file based on its extension."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".txt", ".md", ".log"):
            return path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            import pypdf
            return "\n".join(
                page.extract_text() or "" for page in pypdf.PdfReader(path).pages
            )
        elif suffix == ".docx":
            import docx
            return "\n".join(p.text for p in docx.Document(path).paragraphs)
        elif suffix == ".xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(path, data_only=True)
            return "\n".join(
                str(c)
                for ws in wb.worksheets
                for r in ws.iter_rows(values_only=True)
                for c in r if c
            )
        elif suffix == ".pptx":
            import pptx
            return "\n".join(
                p.text
                for s in pptx.Presentation(path).slides
                for sh in s.shapes
                if sh.has_text_frame
                for p in sh.text_frame.paragraphs
            )
        elif suffix in (".png", ".jpg", ".jpeg"):
            # OCR mode - load Qwen3-VL if available
            return ocr_image(path)
    except Exception as e:
        print(f"⚠️ 解析失败 {path.name}: {e}")
        return ""
    return ""


def ocr_image(img_path: Path) -> str:
    """Use Qwen3-VL to OCR an image. Skip if model not available."""
    # Try to find the model in several locations
    candidates = [
        PROJECT_ROOT / "lab1-multimodal-vlm" / "Qwen3-VL-4B-Instruct-int4-ov",
        PROJECT_ROOT / "Qwen3-VL-4B-Instruct-int4-ov",
    ]
    model_dir = None
    for c in candidates:
        if c.exists() and (c / "openvino_model.xml").exists():
            model_dir = c
            break

    if model_dir is None:
        print(f"  ⏭️ 跳过 OCR (Qwen3-VL 模型不存在): {img_path.name}")
        return ""

    try:
        from optimum.intel.openvino import OVModelForVisualCausalLM
        from transformers import AutoProcessor
        from PIL import Image

        # Thumbnail for speed
        thumb = img_path.with_suffix(".thumb.jpg")
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img.thumbnail((256, 256), Image.Resampling.LANCZOS)
            img.save(thumb, quality=85)

        print(f"  👁️ 正在 OCR: {img_path.name}...")
        model = OVModelForVisualCausalLM.from_pretrained(str(model_dir), device="CPU")
        processor = AutoProcessor.from_pretrained(str(model_dir))

        msgs = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(thumb)},
                {"type": "text", "text": "请提取图片中的所有文字和主要内容。"},
            ],
        }]
        inputs = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        out = model.generate(**inputs, max_new_tokens=256)
        text = processor.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()

        del model, processor
        gc.collect()
        if thumb.exists():
            thumb.unlink()

        print(f"  ✅ OCR 完成: {img_path.name} -> {text[:50]}...")
        return text
    except Exception as e:
        print(f"  ⚠️ OCR 失败 {img_path.name}: {e}")
        return ""


def build_index(knowledge_dir: Path = None, include_ocr: bool = False):
    """Scan knowledge dir, extract text, embed, save."""
    if knowledge_dir is None:
        knowledge_dir = KNOWLEDGE_DIR

    if not knowledge_dir.exists():
        print(f"❌ 知识库目录不存在: {knowledge_dir}")
        print("💡 请先将文件放入 lab5-local-knowledge-assistant/knowledge/")
        sys.exit(1)

    INDEX_STORE_DIR.mkdir(parents=True, exist_ok=True)
    db_path = INDEX_STORE_DIR / "knowledge.db"
    emb_path = INDEX_STORE_DIR / "embeddings.npy"

    # Init DB
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks "
            "(id INTEGER PRIMARY KEY, file_path TEXT, file_type TEXT, chunk_text TEXT, file_name TEXT)"
        )

    # Download model
    download_model_if_missing()

    # Load BGE
    print("🧠 加载 BGE 语义模型...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(BGE_MODEL_DIR), device="cpu")

    # Scan files
    chunks = []
    files = list(knowledge_dir.iterdir())
    for f in files:
        if not f.is_file():
            continue
        if f.suffix.lower() == ".thumb.jpg":
            continue
        if f.suffix.lower() in (".png", ".jpg", ".jpeg") and not include_ocr:
            print(f"  ⏭️ 跳过图片 (无 OCR): {f.name}")
            continue

        print(f"  📄 解析: {f.name}")
        text = parse_file(f)
        if not text.strip():
            continue

        # Chunk
        pieces = [text[i:i + 500] for i in range(0, len(text), 500)]
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (str(f),))
            conn.executemany(
                "INSERT INTO chunks (file_path, file_type, chunk_text, file_name) VALUES (?, ?, ?, ?)",
                [(str(f), f.suffix.lower(), c, f.name) for c in pieces],
            )
            conn.commit()
        print(f"  ✅ 入库: {f.name} ({len(pieces)} 块)")

    # Load all chunks and embed
    with sqlite3.connect(db_path) as conn:
        all_chunks = conn.execute(
            "SELECT id, file_path, file_type, chunk_text, file_name FROM chunks"
        ).fetchall()

    if not all_chunks:
        print("⚠️ 知识库为空，无文件可索引")
        return

    print(f"🔄 生成语义向量 ({len(all_chunks)} 块)...")
    texts = [c[3] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=8)
    np.save(emb_path, embeddings)

    print(f"✅ 索引完成: {len(all_chunks)} 个文本块 -> {emb_path}")

    # Cleanup
    del model
    gc.collect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="索引知识库")
    parser.add_argument("--knowledge-dir", type=Path, default=None, help="知识库目录路径")
    parser.add_argument("--with-ocr", action="store_true", help="包含图片 OCR")
    args = parser.parse_args()
    build_index(knowledge_dir=args.knowledge_dir, include_ocr=args.with_ocr)
