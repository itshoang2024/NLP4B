# ══════════════════════════════════════════════════════════════════════════════
# Dockerfile — Embedding Service (CPU-only PyTorch)
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ cmake curl \
    libprotobuf-dev protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

# 1) sentencepiece FIRST (before transformers touches it)
RUN pip install --no-cache-dir sentencepiece==0.2.0 protobuf
RUN python -c "import sentencepiece; print('sentencepiece OK:', sentencepiece.__version__)"

# 2) PyTorch CPU-only
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 3) All other deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) spaCy model + NLTK data (baked into image)
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# 5) Verify all imports at build time
RUN python -c "\
import sentencepiece; print('OK sentencepiece'); \
import torch; print('OK torch'); \
from transformers import AutoTokenizer; print('OK transformers'); \
from sentence_transformers import SentenceTransformer; print('OK sentence-transformers'); \
from fastembed import SparseTextEmbedding; print('OK fastembed'); \
import spacy; spacy.load('en_core_web_sm'); print('OK spacy'); \
from nltk.corpus import wordnet; print('OK wordnet'); \
"

COPY app.py .
COPY query_processor.py .

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
