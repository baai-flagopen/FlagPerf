FROM registry.mthreads.com/public/mtai/sets_llama_0903:v1.0

ENV DEBIAN_FRONTEND=noninteractive

# 安装 Eigen3
RUN apt update && apt install -y libeigen3-dev && apt clean

# 检查并创建所有必要的兼容性链接
RUN cd /usr/local/musa-4.1.0/lib && \
    echo "Current MUSA libraries:" && \
    ls -la libmu* && \
    echo "Creating compatibility links..." && \
    ln -sf libmudnn.so.3 libmudnn.so.2 && \
    find . -name "libmusart.so.*" -exec ln -sf {} libmusart.so.1.5 \; && \
    ldconfig && \
    echo "Links created:" && \
    ls -la libmu*

# 设置环境变量
ENV LD_LIBRARY_PATH="/usr/local/musa-4.1.0/lib:/usr/local/musa-4.1.0/lib64:${LD_LIBRARY_PATH}"
ENV MUSA_HOME="/usr/local/musa-4.1.0"
ENV PATH="/usr/local/musa-4.1.0/bin:${PATH}"
