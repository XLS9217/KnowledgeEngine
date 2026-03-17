# Use PyTorch official image with CUDA 12.6 pre-installed
FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install Python dependencies using pip (already in base image, no apt-get needed)
COPY pyproject.toml ./
RUN pip install --no-cache-dir --break-system-packages \
    fastapi uvicorn einops peft "transformers[torch]" \
    llmlingua openai scipy scikit-learn umap-learn hdbscan qdrant-client \
    pydantic huggingface-hub addict matplotlib easydict python-multipart sentence-transformers \
    colorama

# Set Hugging Face cache directory to a volume mount point
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV HF_DATASETS_CACHE=/app/models/datasets

# Create model cache directory
RUN mkdir -p /app/models

# Note: Models will be downloaded on first run and cached in the volume
# Pre-downloading during build is skipped due to potential network issues

# Copy application code
COPY . .

# Expose the application port
EXPOSE 7009

# Set environment variables
ENV PORT=7009
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "main.py"]