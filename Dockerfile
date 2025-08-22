# Use NVIDIA's PyTorch image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements-serverless.txt /app/requirements-serverless.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-serverless.txt

# Copy the entire project
COPY . /app/

# Install the project in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Optimize for inference
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Create cache directories
RUN mkdir -p /root/.cache/huggingface
RUN mkdir -p /root/.cache/torch

# Expose port (not needed for RunPod but useful for local testing)
EXPOSE 8000

# Set the default command to run the handler
CMD ["python", "handler.py"]
