FROM nvcr.io/nvidia/pytorch:24.08-py3
RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
RUN apt-get update
RUN pip3 install loguru
RUN pip3 install pycuda
RUN pip3 install schedule
RUN pip3 install munch

# 添加FlagGems支持
ENV FLAGGEMS_WORK_DIR /workspace
RUN mkdir -p $FLAGGEMS_WORK_DIR

# 更新Triton到最新版本以支持exp2函数
RUN pip3 install --upgrade triton

# 升级pytest到最新版本
RUN pip3 install --upgrade pytest

# 克隆并安装FlagGems
WORKDIR $FLAGGEMS_WORK_DIR
RUN git clone https://github.com/FlagOpen/FlagGems.git
WORKDIR $FLAGGEMS_WORK_DIR/FlagGems

RUN pip3 install -e .

# 设置工作目录回到根目录
WORKDIR /