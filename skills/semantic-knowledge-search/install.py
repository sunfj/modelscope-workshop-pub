#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenClaw Skill: Semantic Knowledge Search - 一键安装脚本
功能：自动下载完整项目、创建虚拟环境、安装依赖、索引知识库
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path

# ============================================================
# 配置
# ============================================================
REPO_URL = "https://www.modelscope.cn/models/OpenClaw-AI/modelscope-workshop-pub/git"
REPO_NAME = "modelscope-workshop-pub"
REQUIRED_PYTHON = (3, 9)
VENV_NAME = "ov_workshop"

# 设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # repo root


def print_banner(text):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}\n")


def step(num, text):
    print(f"\n[{num}/5] {text}")


def run(cmd, **kwargs):
    """Run a shell command, inheriting stdout/stderr for visibility."""
    print(f"  $ {cmd}")
    subprocess.check_call(cmd, shell=True, **kwargs)


def check_python():
    ver = sys.version_info
    if ver < REQUIRED_PYTHON:
        print(f"❌ 需要 Python {'.'.join(map(str, REQUIRED_PYTHON))}+，当前 {ver.major}.{ver.minor}")
        sys.exit(1)
    print(f"✅ Python {ver.major}.{ver.minor}.{ver.micro}")


def ensure_repo():
    """Download or update the full project repo."""
    step(1, "检查项目代码")

    if (PROJECT_ROOT / ".git").exists():
        print("  ✅ 项目已存在，跳过下载")
        return

    # Try multiple sources
    sources = [
        REPO_URL,
        f"https://github.com/modelscope/modelscope-workshop-pub.git",
    ]
    for src in sources:
        try:
            print(f"  📥 正在从 {src} 下载...")
            # Clone to a temp location then move contents to project root
            tmp_dir = SCRIPT_DIR.parent / "_tmp_clone"
            run(f"git clone --depth 1 {src} {tmp_dir}")
            # Move everything except skills dir into project root
            for item in tmp_dir.iterdir():
                dest = PROJECT_ROOT / item.name
                if dest.exists():
                    shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                shutil.move(str(item), str(dest))
            shutil.rmtree(tmp_dir)
            print("  ✅ 项目下载完成")
            return
        except Exception as e:
            print(f"  ⚠️ 从 {src} 下载失败: {e}")

    print("❌ 所有代码源均不可达，请手动克隆项目到当前目录")
    sys.exit(1)


def create_venv():
    """Create Python virtual environment."""
    step(2, "创建 Python 虚拟环境")
    venv_dir = PROJECT_ROOT / VENV_NAME

    if venv_dir.exists() and (venv_dir / "bin" / "python").exists():
        print("  ✅ 虚拟环境已存在")
        return venv_dir

    print("  📦 创建虚拟环境...")
    run(f"{sys.executable} -m venv {venv_dir}")
    print("  ✅ 虚拟环境创建成功")
    return venv_dir


def get_pip(venv_dir):
    if platform.system() == "Windows":
        return str(venv_dir / "Scripts" / "pip")
    return str(venv_dir / "bin" / "pip")


def get_python(venv_dir):
    if platform.system() == "Windows":
        return str(venv_dir / "Scripts" / "python")
    return str(venv_dir / "bin" / "python")


def install_deps(venv_dir):
    """Install all dependencies."""
    step(3, "安装项目依赖")
    pip = get_pip(venv_dir)

    run(f"{pip} install --upgrade pip -q")
    print("  📦 安装 baseline 依赖 (首次可能较慢)...")
    run(f"{pip} install -r {PROJECT_ROOT / 'requirements.txt'}")

    # Install Qwen3-ASR if lab2 exists
    asr_dir = PROJECT_ROOT / "lab2-speech-recognition" / "Qwen3-ASR"
    if not asr_dir.exists() and (PROJECT_ROOT / "lab2-speech-recognition").exists():
        print("  📥 下载 Qwen3-ASR 组件...")
        run(
            f"cd {PROJECT_ROOT / 'lab2-speech-recognition'} && "
            f"git clone https://github.com/QwenLM/Qwen3-ASR.git && "
            f"cd Qwen3-ASR && "
            f"git checkout c17a131fe028b2e428b6e80a33d30bb4fa57b8df && "
            f"cd .. && "
            f"{pip} install -q -e Qwen3-ASR"
        )

    print("  ✅ 所有依赖安装完成")


def download_models(venv_dir):
    """Download BGE + Qwen3-VL models."""
    python = get_python(venv_dir)

    # BGE model
    step(4, "下载语义模型 (BGE)")
    bge_path = PROJECT_ROOT / "lab5-local-knowledge-assistant" / "models" / "bge-small-zh-v1.5"
    if bge_path.exists() and (bge_path / "config.json").exists():
        print("  ✅ BGE 模型已存在")
    else:
        print("  📥 从 ModelScope 下载 bge-small-zh-v1.5...")
        try:
            subprocess.check_call([
                python, "-c",
                "from modelscope import snapshot_download; "
                "snapshot_download('AI-ModelScope/bge-small-zh-v1.5', "
                "local_dir='lab5-local-knowledge-assistant/models/bge-small-zh-v1.5')"
            ], cwd=PROJECT_ROOT)
            print("  ✅ BGE 模型下载完成")
        except Exception as e:
            print(f"  ⚠️ BGE 模型下载失败 (可稍后重试): {e}")

    # Qwen3-VL OCR model
    step(5, "下载 OCR 视觉模型 (Qwen3-VL)")
    vl_path = PROJECT_ROOT / "lab1-multimodal-vlm" / "Qwen3-VL-4B-Instruct-int4-ov"
    if vl_path.exists() and (vl_path / "openvino_model.xml").exists():
        print("  ✅ Qwen3-VL 模型已存在")
    else:
        print("  📥 从 ModelScope 下载 Qwen3-VL-4B-Instruct-int4-ov (约 2GB)...")
        try:
            subprocess.check_call([
                python, "-c",
                "from modelscope import snapshot_download; "
                "snapshot_download('snake7gun/Qwen3-VL-4B-Instruct-int4-ov', "
                "local_dir='lab1-multimodal-vlm/Qwen3-VL-4B-Instruct-int4-ov')"
            ], cwd=PROJECT_ROOT)
            print("  ✅ Qwen3-VL 模型下载完成")
        except Exception as e:
            print(f"  ⚠️ Qwen3-VL 模型下载失败 (图片 OCR 将不可用): {e}")


def index_knowledge(venv_dir, knowledge_dir=None):
    """Index the knowledge base files."""
    step(5, "索引知识库")
    python = get_python(venv_dir)
    index_script = SCRIPT_DIR / "index_kb.py"

    if index_script.exists():
        cmd = [python, str(index_script)]
        if knowledge_dir:
            cmd.extend(["--knowledge-dir", knowledge_dir])
        try:
            subprocess.check_call(cmd)
            print("  ✅ 知识库索引完成")
            return
        except Exception as e:
            print(f"  ⚠️ 索引过程出错 (可稍后指定 --knowledge-dir 重试): {e}")
    else:
        print("  ℹ️ 索引脚本不存在，跳过索引")


def create_shortcuts(venv_dir):
    """Create convenience run scripts."""
    python = get_python(venv_dir)
    is_win = platform.system() == "Windows"

    # run.sh / run.bat
    if is_win:
        run_script = SCRIPT_DIR.parent.parent / "run_search.bat"
        run_script.write_text(
            f"@echo off\n"
            f"call {venv_dir}\\Scripts\\activate.bat\n"
            f'python "{SCRIPT_DIR}\\main.py" %*\n'
            f"call deactivate\n"
        )
    else:
        run_script = SCRIPT_DIR.parent.parent / "run_search.sh"
        run_script.write_text(
            f"#!/bin/bash\n"
            f"source {venv_dir}/bin/activate\n"
            f'python "{SCRIPT_DIR}/main.py" "$@"\n'
            f"deactivate\n"
        )
        run_script.chmod(0o755)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="一键安装语义知识库")
    parser.add_argument("--knowledge-dir", type=str, default=None,
                        help="指定知识库目录路径 (默认 lab5-local-knowledge-assistant/knowledge/)")
    args = parser.parse_args()

    print_banner("🧠 语义知识库 Skill - 一键安装")
    check_python()
    ensure_repo()
    venv_dir = create_venv()
    install_deps(venv_dir)
    download_models(venv_dir)
    index_knowledge(venv_dir, knowledge_dir=args.knowledge_dir)
    create_shortcuts(venv_dir)

    print_banner("✅ 安装完成！\n  使用方式: python main.py '你的问题'")
