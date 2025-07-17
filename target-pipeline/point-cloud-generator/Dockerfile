FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    libopencv-dev \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set multithreading environment variables
ENV OMP_NUM_THREADS=16
ENV OPENBLAS_NUM_THREADS=16
ENV MKL_NUM_THREADS=16
ENV NUMBA_NUM_THREADS=16
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Create directories
RUN mkdir -p /app/output/point_clouds /app/data /app/logs /app/temp && \
    chown -R user:user /app

COPY --chown=user . /app
USER user

CMD ["python", "point_cloud_generator_service.py"]
