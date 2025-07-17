FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python 3.8
RUN ln -sf /usr/bin/python3.8 /usr/bin/python3
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

# Install pip for Python 3.8 using the correct URL
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8

# Set environment variables
ENV OMP_NUM_THREADS=16
ENV OPENBLAS_NUM_THREADS=16
ENV MKL_NUM_THREADS=16
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 user
WORKDIR /app

# Download and install PyTorch 1.8.1 wheel for Python 3.8 and CUDA 11.1
RUN wget https://download.pytorch.org/whl/cu111/torch-1.8.1%2Bcu111-cp38-cp38-linux_x86_64.whl && \
    pip install torch-1.8.1+cu111-cp38-cp38-linux_x86_64.whl && \
    rm torch-1.8.1+cu111-cp38-cp38-linux_x86_64.whl

# Install compatible torchvision and torchaudio
RUN pip install torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Clone and install PointNet2 PyTorch
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git /app/pointnet2
WORKDIR /app/pointnet2/pointnet2_ops_lib
RUN pip install .

# Clone your modified GeDi repository
WORKDIR /app
RUN git clone https://github.com/pacificblaster1708/GeDi-Modified-1.git /app/gedi-repo

# Install Python dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Download GeDi pretrained models and data
WORKDIR /app/gedi-repo
RUN python download_data.py

# Create directories and set ownership
WORKDIR /app
RUN mkdir -p /app/output/gedi_features /app/data /app/logs /app/performance && \
    chown -R user:user /app

COPY --chown=user . /app
USER user

# Set Python path to include GeDi and PointNet2 (fixed syntax)
ENV PYTHONPATH="/app/gedi-repo:/app/pointnet2"

CMD ["python", "gedi_processor_service.py"]
