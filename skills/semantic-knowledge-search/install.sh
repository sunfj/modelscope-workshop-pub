---
### 🛠️ **3. 一键安装脚本：`install.sh`**
*(给不懂代码的人用的，一键配好环境)*
```bash
#!/bin/bash
echo "🚀 OpenClaw Skill: Semantic Knowledge Search 安装中..."
# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误：未检测到 Python3"
    exit 1
fi
# 安装依赖
echo "📦 安装运行依赖 (sentence-transformers, numpy, pandas)..."
pip3 install -q sentence-transformers numpy pandas
# 检查数据目录
if [ ! -d "data/index_store" ]; then
    echo "⚠️ 警告：未检测到 data/index_store 目录。请上传离线语义库。"
else
    echo "✅ 检测到离线语义库。"
fi
if [ ! -d "data/models" ]; then
    echo "⚠️ 提示：未检测到本地模型，首次运行将自动从 HuggingFace 下载。"
else
    echo "✅ 检测到本地 BGE 模型，即将离线运行。"
fi
echo "🎉 安装完成！请运行 python main.py '你的问题' 进行测试。"
