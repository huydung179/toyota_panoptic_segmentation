# ----------------------------------------------- Base Docker Image -----------------------------------------------
FROM nvcr.io/nvidia/pytorch:21.11-py3

# Workaround for deadlock issue. To be removed with next major Pytorch release
ENV NCCL_LL_THRESHOLD=0

ARG python=3.8.12
ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-cu"]

RUN apt-get update
RUN apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends build-essential \
    cmake \
    git \
    curl \
    docker.io \
    vim \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    python3-tk \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    libgtk2.0-dev \
    unzip \
    bzip2 \
    htop \
    protobuf-compiler \
    libprotoc-dev \
    ffmpeg \
    libsm6 \
    libxext6

# Configure environment variables - default working directory is "/workspace"
WORKDIR /workspace
ENV PYTHONPATH="/workspace"

# Install dependencies
RUN pip install ninja yacs cython matplotlib opencv-python==4.5.5.62 tqdm onnx==1.4.1 onnxruntime coloredlogs scipy pycuda
RUN pip uninstall -y pillow
RUN pip install pillow-simd==6.2.2.post1 pycocotools


# Copy repo
ENV PYTHONPATH="/workspace/panoptic:$PYTHONPATH"
COPY . /workspace/panoptic
WORKDIR /workspace/panoptic
