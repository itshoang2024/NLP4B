# ══════════════════════════════════════════════════════════════════════════════
# Dockerfile — Embedding Service (Single-stage, CPU-only PyTorch)
# ══════════════════════════════════════════════════════════════════════════════
# Target: Azure VM Standard_B4as_v2 (4 vCPU, 16GB RAM, NO GPU)
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

WORKDIR /app

# System deps (sentencepiece + healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl cmake \
    libsentencepiece-dev && \
    rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only (saves ~1.5GB vs CUDA wheels)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Sentencepiece + protobuf (required by SigLIP tokenizer)
RUN pip install --no-cache-dir sentencepiece protobuf

# Remaining Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app.py .

# HuggingFace cache (mounted as volume)
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/hub
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface/hub

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "120"]
