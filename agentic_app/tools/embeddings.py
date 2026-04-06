import torch
from transformers import AutoTokenizer, AutoModel
from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector
from loguru import logger

# Lazy-loaded singletons
_siglip_tokenizer = None
_siglip_model = None
_bge_m3_model = None
_bm25_model = None

SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
BGE_M3_MODEL_ID = "BAAI/bge-m3"
BM25_MODEL_ID = "Qdrant/bm25"
DEVICE = "cpu"  # Force CPU for local dev as requested

def get_siglip():
    global _siglip_tokenizer, _siglip_model
    if _siglip_tokenizer is None or _siglip_model is None:
        logger.info(f"Loading SigLIP {SIGLIP_MODEL_ID} on [{DEVICE}]...")
        _siglip_tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_ID)
        _siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_ID).to(DEVICE).eval()
    return _siglip_tokenizer, _siglip_model

def get_bge_m3():
    global _bge_m3_model
    if _bge_m3_model is None:
        logger.info(f"Loading BGE-M3 {BGE_M3_MODEL_ID} on [{DEVICE}]...")
        from sentence_transformers import SentenceTransformer
        _bge_m3_model = SentenceTransformer(BGE_M3_MODEL_ID, device=DEVICE)
    return _bge_m3_model

def get_bm25():
    global _bm25_model
    if _bm25_model is None:
        logger.info(f"Loading BM25 {BM25_MODEL_ID} on [{DEVICE}]...")
        _bm25_model = SparseTextEmbedding(model_name=BM25_MODEL_ID)
    return _bm25_model

def encode_siglip_text(query: str) -> list[float]:
    if not query.strip(): return []
    tokenizer, model = get_siglip()
    inputs = tokenizer([query], padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    
    # Extract tensor
    if isinstance(outputs, torch.Tensor):
        features = outputs
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        features = outputs.pooler_output
    elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        features = outputs.text_embeds
    elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        features = outputs.last_hidden_state.mean(dim=1)
    else:
        features = outputs[0]
        if features.ndim == 3:
            features = features.mean(dim=1)
            
    embedding = features[0].cpu().numpy().flatten().astype("float32")
    return embedding.tolist()

def encode_bge_m3(query: str) -> list[float]:
    if not query.strip(): return []
    model = get_bge_m3()
    vec = model.encode(query, normalize_embeddings=True)
    if hasattr(vec, "tolist"): vec = vec.tolist()
    return vec

def encode_bm25(query: str) -> SparseVector | None:
    if not query.strip(): return None
    model = get_bm25()
    results = list(model.embed([query]))
    if not results or len(results[0].indices) == 0: return None
    s = results[0]
    return SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
