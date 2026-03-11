# =============================================================================
# ACE-Step 1.5 FastAPI Server - Configurable model source Dockerfile
# =============================================================================
# This version removes hardcoded Hugging Face downloads and lets you choose
# where model files come from.
#
# Supported patterns:
# 1) Mount models at runtime (recommended for large model files)
# 2) Download model archives from your own hosted URLs at build time
#
# Example build without downloading models:
#   docker build -t acestep-api:latest .
#
# Example build with externally hosted model archives:
#   docker build \
#     --build-arg MODEL_SOURCE=url \
#     --build-arg MODEL_DOWNLOAD_DIR=/opt/acestep-models \
#     --build-arg MAIN_MODEL_URL=https://your-host/models/Ace-Step1.5.tar.gz \
#     --build-arg BASE_MODEL_URL=https://your-host/models/acestep-v15-base.tar.gz \
#     -t acestep-api:latest .
#
# Example runtime mount:
#   docker run --gpus all -p 8000:8000 \
#     -v /host/models:/opt/acestep-models \
#     acestep-api:latest
# =============================================================================

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 as runtime

# -----------------------------------------------------------------------------
# Build arguments for configurable model locations / sources
# -----------------------------------------------------------------------------
ARG MODEL_SOURCE=none
ARG MODEL_DOWNLOAD_DIR=/opt/acestep-models
ARG MAIN_MODEL_URL=none
ARG BASE_MODEL_URL=http://98.82.28.3/tmp/acestep-v15-base.tar.gz
ARG LM_MODEL_URL=http://98.82.28.3/tmp/acestep-5Hz-lm-4B.tar.gz
ARG TURBO_MODEL_URL=http://98.82.28.3/tmp/acestep-v15-turbo.tar.gz

# -----------------------------------------------------------------------------
# Environment variables
# -----------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # ACE-Step configuration
    ACESTEP_PROJECT_ROOT=/app \
    ACESTEP_OUTPUT_DIR=/app/outputs \
    ACESTEP_TMPDIR=/app/outputs \
    ACESTEP_DEVICE=cuda \
    # Centralized model directory (override at runtime with -e if desired)
    ACESTEP_CHECKPOINTS_DIR=${MODEL_DOWNLOAD_DIR} \
    # ACE-Step API model paths
    ACESTEP_CONFIG_PATH=${MODEL_DOWNLOAD_DIR}/acestep-v15-base \
    ACESTEP_LM_MODEL_PATH=${MODEL_DOWNLOAD_DIR}/acestep-5Hz-lm-1.7B \
    ACESTEP_LM_BACKEND=pt \
    # Server configuration
    ACESTEP_API_HOST=0.0.0.0 \
    ACESTEP_API_PORT=8000

WORKDIR /app

# Install system dependencies including Python, pip, git, build tools, and archive tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    libsndfile1 \
    ffmpeg \
    unzip \
    xz-utils \
    tar \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install uv for faster dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone ACE-Step directly into /app and install
RUN git clone https://github.com/ace-step/ACE-Step-1.5.git /app && \
    rm -rf /app/.git && \
    uv pip install --system --no-cache .

# Prepare model directory
RUN mkdir -p "${MODEL_DOWNLOAD_DIR}"

# Optional: download model archives from your own hosted URLs
# Each archive should extract into the proper ACE-Step checkpoint folder names.
# Expected examples:
# - MAIN_MODEL_URL -> archive containing shared checkpoints from Ace-Step1.5
# - BASE_MODEL_URL -> archive that extracts to /opt/acestep-models/acestep-v15-base
# - LM_MODEL_URL   -> optional archive for /opt/acestep-models/acestep-5Hz-lm-1.7B
RUN set -eux; \
    fetch_and_extract() { \
      url="$1"; \
      name="$2"; \
      [ -n "$url" ] || return 0; \
      tmp="/tmp/${name}"; \
      echo "Downloading ${name} from ${url}"; \
      curl -fL "$url" -o "$tmp"; \
      case "$url" in \
        *.tar.gz|*.tgz) tar -xzf "$tmp" -C "${MODEL_DOWNLOAD_DIR}" ;; \
        *.tar.xz) tar -xJf "$tmp" -C "${MODEL_DOWNLOAD_DIR}" ;; \
        *.tar) tar -xf "$tmp" -C "${MODEL_DOWNLOAD_DIR}" ;; \
        *.zip) unzip -q "$tmp" -d "${MODEL_DOWNLOAD_DIR}" ;; \
        *) echo "Unsupported archive format for ${url}"; exit 1 ;; \
      esac; \
      rm -f "$tmp"; \
    }; \
    if [ "$MODEL_SOURCE" = "url" ]; then \
      fetch_and_extract "$MAIN_MODEL_URL" main_model_archive; \
      fetch_and_extract "$BASE_MODEL_URL" base_model_archive; \
      fetch_and_extract "$LM_MODEL_URL" lm_model_archive; \
      fetch_and_extract "$TURBO_MODEL_URL" turbo_model_archive; \
    else \
      echo "Skipping model download. MODEL_SOURCE=${MODEL_SOURCE}"; \
    fi

# Create symlink so ACE-Step's model discovery finds the chosen checkpoint dir
RUN ln -sfn "${MODEL_DOWNLOAD_DIR}" /usr/local/lib/python3.11/dist-packages/checkpoints

# Create placeholder for acestep-v15-turbo to satisfy check_main_model_exists()
# We use acestep-v15-base instead, but the check looks for all MAIN_MODEL_COMPONENTS
RUN mkdir -p "${MODEL_DOWNLOAD_DIR}/acestep-v15-turbo"

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Create output directory
RUN mkdir -p /app/outputs

# Expose ports (8000 for API, 7860 for Gradio UI)
EXPOSE 8000 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run both API server and Gradio UI
CMD ["/app/start.sh"]
