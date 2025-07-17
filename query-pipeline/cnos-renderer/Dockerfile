FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies optimized for high-end GPUs
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
    libfftw3-dev \
    libglew-dev \
    libglfw3-dev \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    libxrender1 \
    libxext6 \
    libx11-6 \
    xvfb \
    htop \
    nvtop \
    libtbb-dev \
    libpthread-stubs0-dev \
    && rm -rf /var/lib/apt/lists/*

# Aggressive multithreading and GPU optimization for RunPod/Lightning AI
ENV OMP_NUM_THREADS=32
ENV OPENBLAS_NUM_THREADS=32
ENV MKL_NUM_THREADS=32
ENV NUMBA_NUM_THREADS=32
ENV TBB_NUM_THREADS=32
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYOPENGL_PLATFORM=egl
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_CACHE_MAXSIZE=2147483648
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"

# Create symbolic links for python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install high-performance libraries
RUN pip install --no-cache-dir \
    wheel \
    setuptools \
    cython \
    numba \
    ninja \
    tensorrt \
    cupy-cuda11x

RUN useradd -m -u 1000 user
WORKDIR /app

# Clone CNOS repository
RUN git clone https://github.com/pitcher69/cnos.git /app/cnos

# Install Python dependencies with performance optimizations
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Install PyTorch with CUDA support optimized for high-end GPUs
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional performance libraries for RunPod/Lightning AI
RUN pip install \
    nvidia-ml-py \
    gpustat \
    memory-profiler \
    line-profiler \
    py3nvml

# Set up virtual display for headless rendering
ENV DISPLAY=:99

COPY --chown=user . /app

# Ensure user owns the /app directory and switch to user
RUN chown -R user:user /app
USER user

# Create comprehensive output directories as user
RUN mkdir -p /app/output/renders /app/data/rgb /app/data/mask /app/data/depth \
    /app/output/point_cloud /app/output/dino_feature_pca /app/output/intermediate \
    /app/output/logs /app/output/performance

CMD ["python", "cnos_renderer_service.py"]
