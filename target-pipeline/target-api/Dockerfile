FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libopencv-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set multithreading environment variables
ENV OMP_NUM_THREADS=32
ENV OPENBLAS_NUM_THREADS=32
ENV MKL_NUM_THREADS=32
ENV NUMBA_NUM_THREADS=32
ENV PYTHONUNBUFFERED=1

# Performance optimizations
ENV UVICORN_WORKERS=4
ENV UVICORN_WORKER_CLASS=uvicorn.workers.UvicornWorker
ENV UVICORN_LOOP=uvloop
ENV UVICORN_HTTP=httptools

RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Create directories
RUN mkdir -p /app/uploads /app/data /app/output /app/logs /app/temp && \
    chown -R user:user /app

COPY --chown=user . /app
USER user

EXPOSE 8001
CMD ["python", "target_api_service.py"]
