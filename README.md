# 🏆 Lab 5: 本地语义知识库助手 (Local Semantic Knowledge Assistant)
> **2026 Intel AI PC 创新应用征文 & OpenClaw Skill 挑战赛 参赛作品**

## 🌟 项目简介
这是一个完全运行在本地 Intel AI PC 上的智能知识库检索系统。结合 **Qwen3-VL 真实 OCR** 与 **BGE 语义向量模型**，实现真正的**意图搜索**。
支持自然语言提问（如 `“帮我找找小学里的英文单词”`），精准匹配图片、文档内容。

**🔥 核心创新点 (针对 i3 + 8GB 内存深度优化):**
1. **真实 OCR 识别**：OpenVINO 加速 Qwen3-VL，提取图片真实内容。
2. **语义意图搜索**：轻量级 BGE 向量模型，告别死板关键词。
3. **Lazy Loading 内存保护**：模型用完即释放，8GB 内存稳定运行。

## 🚀 一键运行指南

### 第一步：一键配置环境 (Windows)
**双击运行根目录下的 `setup_lab.bat`**
*   该脚本会自动创建虚拟环境，并**一次性安装所有依赖**（包含 OpenVINO 基座、Lab 5 语义检索、OCR 组件等）。
*   *注意：首次运行可能需要几分钟下载基础组件，请保持网络畅通。*

### 第二步：启动工作台
**双击运行根目录下的 `run_lab.bat`**
*   这将启动 JupyterLab 界面，您的浏览器会自动打开。

### 第三步：运行 Notebook
1.  在左侧文件树中找到并打开 **`lab5-local-knowledge-assistant/lab5-local-knowledge-assistant.ipynb`**。
2.  点击顶部菜单 **Run -> Run All Cells** (或依次按 `Shift+Enter`)。
3.  等待所有单元格运行完毕，Gradio 界面弹出即可开始语音搜索！

## ⚠️ 注意事项
- **模型自动下载**：代码会自动下载 `Qwen3-VL`、`Qwen3-ASR` 和 `bge-small-zh` 模型，请勿打断首次运行的下载过程。
- **内存限制**：代码已针对 8GB 内存优化，运行期间建议关闭其他大型软件。
- **目录结构**：请勿移动 `lab5` 文件夹，代码依赖 `lab2-speech-recognition` 进行语音识别。
