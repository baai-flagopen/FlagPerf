#!/bin/bash
current_dir=$(pwd)
echo "=====>$current_dir"
script_dir=$(dirname "$(realpath "$0")")
echo "script dir :$script_dir"

# 检查 FlagGems 目录是否存在，如果不存在则克隆
if [ ! -d "/workspace/docker_image/FlagGems" ]; then
    echo "FlagGems directory not found, cloning..."
    cd /workspace/docker_image
    git clone https://mirror.ghproxy.com/https://github.com/FlagOpen/FlagGems.git
    cd FlagGems
else
    echo "FlagGems directory found, using existing..."
    cd /workspace/docker_image/FlagGems
fi

# 安装 FlagGems
pip3 install .