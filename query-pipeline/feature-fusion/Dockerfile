FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgomp1 \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set aggressive multithreading environment variables
ENV OMP_NUM_THREADS=32
ENV OPENBLAS_NUM_THREADS=32
ENV MKL_NUM_THREADS=32
ENV NUMBA_NUM_THREADS=32
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Create directories and set ownership
RUN mkdir -p /app/output/fused_features /app/data /app/logs /app/performance && \
    chown -R user:user /app

COPY --chown=user . /app
USER user

CMD ["python", "feature_fusion_service.py"]
