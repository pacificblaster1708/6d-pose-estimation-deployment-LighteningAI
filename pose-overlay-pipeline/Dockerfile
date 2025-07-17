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
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV OMP_NUM_THREADS=16
ENV OPENBLAS_NUM_THREADS=16
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/output/pose_results /app/data /app/logs && \
    chown -R user:user /app

COPY --chown=user . /app
USER user

CMD ["python", "pose_overlay_service.py"]
