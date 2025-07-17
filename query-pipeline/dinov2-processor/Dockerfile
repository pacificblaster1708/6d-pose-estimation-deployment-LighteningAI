FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for DINOv2 and GPU optimization
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    libtbb-dev \
    libpthread-stubs0-dev \
    libeigen3-dev \
    libboost-all-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Set aggressive multithreading and GPU optimization
ENV OMP_NUM_THREADS=32
ENV OPENBLAS_NUM_THREADS=32
ENV MKL_NUM_THREADS=32
ENV NUMBA_NUM_THREADS=32
ENV TBB_NUM_THREADS=32
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"

# Create symbolic links for python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install high-performance libraries
RUN pip install --no-cache-dir \
    wheel \
    setuptools \
    cython \
    numba \
    ninja

RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Create directories and set ownership
RUN mkdir -p /app/output/dino_features /app/data /app/output/logs /app/output/performance
RUN chown -R user:user /app

COPY --chown=user . /app
USER user

CMD ["python", "dinov2_processor_service.py"]
