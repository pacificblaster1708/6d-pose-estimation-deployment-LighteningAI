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
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python 3.8
RUN ln -sf /usr/bin/python3.8 /usr/bin/python3
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

# Install pip for Python 3.8
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8

# Set environment variables
ENV OMP_NUM_THREADS=16
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 user
WORKDIR /app

# Install PyTorch 1.8.1 for Python 3.8 compatibility
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Clone and setup PointNet2 for GeDi
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git /app/pointnet2
WORKDIR /app/pointnet2/pointnet2_ops_lib
RUN pip install .

# Clone GeDi repository
WORKDIR /app
RUN git clone https://github.com/Gorilla-Lab-SCUT/gorilla-3d.git /app/gedi-repo

# Create directories
RUN mkdir -p /app/output/gedi_features /app/data /app/logs /app/temp && \
    chown -R user:user /app

COPY --chown=user . /app
USER user

ENV PYTHONPATH=/app/gedi-repo:/app/pointnet2:$PYTHONPATH

CMD ["python", "target_gedi_service.py"]
