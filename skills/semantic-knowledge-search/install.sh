#!/bin/bash
echo "🚀 初始化 Semantic Knowledge Skill..."

# 1. 安装 Python 依赖
pip3 install -q -r requirements.txt

# 2. 检查数据是否存在
DATA_DIR="data/index_store"
MODEL_DIR="data/models/bge-small-zh-v1.5"

if [ -d "$DATA_DIR" ] && [ -d "$MODEL_DIR" ]; then
    echo "✅ 检测到完整的离线数据与模型，跳过下载。"
else
    echo "⚠️ 检测到数据缺失，正在从 ModelScope 下载模型..."
    # 使用魔搭下载
    python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
import os
if not os.path.exists('$MODEL_DIR'):
    snapshot_download('AI-ModelScope/bge-small-zh-v1.5', local_dir='$MODEL_DIR')
    print('✅ 模型下载完成！')
"
fi

echo "🎉 环境准备就绪！"
