# ══════════════════════════════════════════════════════════════════════════════
# Dockerfile — Embedding Service (Multi-stage, CPU-only PyTorch)
# ══════════════════════════════════════════════════════════════════════════════
# Target: Azure VM Standard_B4as_v2 (4 vCPU, 16GB RAM, NO GPU)
# Image size: ~3.5GB (with models cached on host volume)
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System build deps (including sentencepiece)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc cmake libprotobuf-dev protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (saves ~1.5GB vs CUDA wheels)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install sentencepiece + remaining Python deps
RUN pip install --no-cache-dir sentencepiece protobuf
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps (curl + sentencepiece shared lib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libsentencepiece-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app.py .

# HuggingFace cache will be mounted as volume (see docker-compose)
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/hub
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface/hub

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "120"]
