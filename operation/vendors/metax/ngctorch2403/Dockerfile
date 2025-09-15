FROM cr.metax-tech.com/public-library/maca-c500-pytorch:2.33.0.6-torch2.4-py310-ubuntu22.04-amd64

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

# MetaX 官方镜像已包含 PyTorch，只需安装额外依赖
RUN pip3 install numpy scipy
