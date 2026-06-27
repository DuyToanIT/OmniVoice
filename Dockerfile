# ==============================================================================
# OmniVoice Dockerfile
# Supports both GPU (NVIDIA CUDA) and CPU-only modes via build args.
#
# Build args:
#   BASE_IMAGE  — override the base Docker image
#   DEVICE      — "cuda" (default) or "cpu"
#
# Usage:
#   GPU:  docker build -t omnivoice:gpu .
#   CPU:  docker build -t omnivoice:cpu --build-arg DEVICE=cpu .
# ==============================================================================

# --- Base stage: pick the right base image ---
ARG DEVICE=cuda
ARG BASE_IMAGE_GPU=nvidia/cuda:12.8.0-runtime-ubuntu24.04
ARG BASE_IMAGE_CPU=python:3.12-slim-bookworm

FROM ${BASE_IMAGE_GPU} AS base-cuda
FROM ${BASE_IMAGE_CPU} AS base-cpu

# Select the correct base based on DEVICE arg
FROM base-${DEVICE} AS base

# --- System dependencies ---
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system packages (including Python for the CUDA image which doesn't have it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Install uv for fast package management ---
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# --- Create non-root user for security ---
# Use -f flag to handle cases where GID/UID 1000 already exists in the base image
RUN groupadd --gid 1000 omnivoice 2>/dev/null || true \
    && useradd --uid 1000 --gid 1000 --create-home omnivoice 2>/dev/null \
    || useradd --create-home omnivoice

# --- Application setup ---
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Copy source code
COPY omnivoice/ ./omnivoice/
COPY README.md LICENSE ./

# --- Install dependencies using uv sync (respects uv.lock) ---
ARG DEVICE=cuda
RUN if [ "$DEVICE" = "cuda" ]; then \
        # GPU: uses the pytorch-cuda index configured in pyproject.toml
        uv sync --frozen ; \
    else \
        # CPU: --no-sources skips the CUDA index, installs from PyPI
        # PyPI wheels are CPU-only and support both x86_64 and ARM64
        uv sync --no-sources ; \
    fi

# Ensure the venv is on PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# --- HuggingFace cache directory ---
# Models will be cached here; mount a volume to persist across restarts
RUN mkdir -p /app/models_cache && chown -R omnivoice:omnivoice /app/models_cache
ENV HF_HOME=/app/models_cache
ENV TRANSFORMERS_CACHE=/app/models_cache

# --- Gradio config ---
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# --- Switch to non-root user ---
RUN chown -R omnivoice:omnivoice /app
USER omnivoice

# --- Expose ports ---
# 7860: Gradio demo UI
# 8000: REST API with Swagger
EXPOSE 7860 8000

# --- Health check (works with either service) ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || curl -f http://localhost:8000/health || exit 1

# --- Default command: launch the REST API with Swagger ---
CMD ["omnivoice-api", "--host", "0.0.0.0", "--port", "8000"]
