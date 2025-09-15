FROM flagperf-operation-metax-ngctorch2403:t_v0.1

# 检查并安装 python3-pip（如果需要）
RUN which pip3 || (apt-get update && apt-get install -y python3-pip)

# 配置 pip 源
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 系统信息
RUN uname -a

# 设置环境变量
ENV MACA_VISIBLE_DEVICES=7

# 更新包管理器
RUN apt-get update

# 安装 Python 包
RUN pip3 install loguru
#RUN pip3 install pycuda
RUN pip3 install schedule
RUN pip3 install munch

# 安装 PyTorch (MetaX 版本)
# 注意：这里可能需要使用 MetaX 特定的 PyTorch 版本
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || \
    pip3 install torch torchvision torchaudio

# 安装其他必要的依赖
RUN pip3 install numpy scipy
