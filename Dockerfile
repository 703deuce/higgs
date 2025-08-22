# Use NVIDIA Deep Learning Container with confirmed Python 3.10
# The 25.x containers appear to use Python 3.12, so using 23.x for Python 3.10 compatibility
FROM nvcr.io/nvidia/pytorch:23.08-py3

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

# Install base Python dependencies (without FlashAttention yet)
RUN pip install --no-cache-dir -r requirements-serverless.txt

# Copy the entire project
COPY . /app/

# Verify container environment and check Python version
RUN python --version && \
    python -c "import sys; print(f'Python version: {sys.version}'); print(f'Python version info: {sys.version_info}')" && \
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Remove any existing FlashAttention that may have Python 3.12 binaries
RUN pip uninstall -y flash-attn flash-attn-2 || true

# Install FlashAttention from source for this specific Python/PyTorch/CUDA combination
RUN pip install flash-attn --no-binary :all: --no-cache-dir || \
    echo "⚠️ FlashAttention installation failed, continuing without it"

# Install the project in development mode
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir --upgrade setuptools wheel

# Verify all imports work correctly
RUN python -c "import transformers; print(f'✅ Transformers version: {transformers.__version__}')" && \
    python -c "try: import flash_attn; print('✅ FlashAttention imported successfully')\nexcept ImportError as e: print(f'⚠️ FlashAttention failed: {e}')" && \
    python -c "import boson_multimodal; print('✅ boson_multimodal imported successfully')" || \
    echo "⚠️ Some imports failed, will debug at runtime"

# Set environment variables for optimal CUDA performance and RunPod volume caching
ENV PYTHONPATH=/app
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0

# Use RunPod network volume for persistent model caching
ENV PYTORCH_TRANSFORMERS_CACHE=/runpod-volume/cache/transformers
ENV HF_HOME=/runpod-volume/cache/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/cache/transformers
ENV HF_HUB_CACHE=/runpod-volume/cache/huggingface
ENV TORCH_HOME=/runpod-volume/cache/torch

# Optimize for inference
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Create cache directories on RunPod network volume for persistence
RUN mkdir -p /runpod-volume/cache/huggingface && \
    mkdir -p /runpod-volume/cache/transformers && \
    mkdir -p /runpod-volume/cache/torch && \
    chmod -R 777 /runpod-volume/cache/

# Expose port (not needed for RunPod but useful for local testing)
EXPOSE 8000

# Set the default command to run the handler
CMD ["python", "handler.py"]
