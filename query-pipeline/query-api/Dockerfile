FROM python:3.11-slim

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV and high-performance processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    libxt6 \
    libxrandr2 \
    libxss1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set multithreading environment variables for high performance
ENV OMP_NUM_THREADS=16
ENV OPENBLAS_NUM_THREADS=16
ENV MKL_NUM_THREADS=16
ENV NUMBA_NUM_THREADS=16
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Create necessary directories with proper ownership
RUN mkdir -p /app/uploads /app/data /app/output /app/logs && \
    chown -R user:user /app

# Install Python dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application files
COPY --chown=user . /app

# Switch to non-root user
USER user

# Expose port
EXPOSE 7861

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7861/health || exit 1

# Start the service
CMD ["python", "query_api_service.py"]
