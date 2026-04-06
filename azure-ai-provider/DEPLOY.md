# 🚀 DEPLOY.md — Embedding Service Deployment Guide

> **Target:** Azure VM `Standard_B4as_v2` (4 vCPU, 16GB RAM, NO GPU)
> **OS:** Ubuntu 22.04 LTS

---

## 1. Cài đặt Docker (chỉ chạy 1 lần)

```bash
# Update system
sudo apt-get update -y && sudo apt-get upgrade -y

# Install Docker Engine
curl -fsSL https://get.docker.com | sudo sh

# Allow user to run docker without sudo
sudo usermod -aG docker $USER

# Log out and back in for group changes
exit
# SSH lại vào VM
```

Verify:
```bash
docker --version            # Docker version 27.x
docker compose version      # Docker Compose version v2.x
```

---

## 2. Upload code lên VM

### Option A: Clone từ GitHub
```bash
git clone https://github.com/CallmeAndree/NLP4B.git
cd NLP4B
git checkout feat/azure-ai-provider
cd azure-ai-provider
```

### Option B: SCP từ local
```bash
# Từ máy local:
scp -r azure-ai-provider/ user@<VM_IP>:~/azure-ai-provider/
```

---

## 3. Pre-download models (Khuyến nghị)

Tải models trước vào cache để tránh timeout lúc container khởi động:

```bash
# Tạo thư mục cache
mkdir -p ~/.cache/huggingface

# Pre-download (chạy 1 lần, ~5GB total)
pip install huggingface-hub
huggingface-cli download BAAI/bge-m3
huggingface-cli download google/siglip-so400m-patch14-384
# BM25 sẽ tự tải qua fastembed (nhẹ ~50MB)
```

> **Lưu ý:** Nếu bỏ qua bước này, container sẽ tự tải lần đầu khởi động (~5-10 phút tùy mạng). Models được cache trong volume nên chỉ tải 1 lần.

---

## 4. Khởi động service

```bash
cd azure-ai-provider

# Build & start (background mode)
docker compose up -d --build

# Xem logs (theo dõi quá trình load model)
docker compose logs -f embedding_service
```

**Chờ ~2-3 phút** cho đến khi thấy:
```
✅ All models loaded in XXs — Server ready!
```

---

## 5. Kiểm tra hoạt động

### Health check
```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "device": "cpu",
  "models": {
    "semantic": "BAAI/bge-m3",
    "sparse": "Qdrant/bm25",
    "visual": "google/siglip-so400m-patch14-384"
  }
}
```

### Test embedding endpoints
```bash
# Semantic (BGE-M3 → 1024d)
curl -X POST http://localhost:8000/embed/semantic \
  -H "Content-Type: application/json" \
  -d '{"text": "người mặc áo đỏ đang nấu ăn"}' | python3 -m json.tool

# Sparse (BM25)
curl -X POST http://localhost:8000/embed/sparse \
  -H "Content-Type: application/json" \
  -d '{"text": "person cooking kitchen"}' | python3 -m json.tool

# Visual (SigLIP → 1152d)
curl -X POST http://localhost:8000/embed/visual \
  -H "Content-Type: application/json" \
  -d '{"text": "a person in red shirt cooking"}' | python3 -m json.tool
```

---

## 6. Mở port từ bên ngoài (Azure NSG)

Để gọi API từ Colab/laptop, mở port 8000 trên Azure Network Security Group:

```bash
# Azure CLI (hoặc làm trên Azure Portal → Networking → Add inbound rule)
az network nsg rule create \
  --resource-group <RG_NAME> \
  --nsg-name <NSG_NAME> \
  --name AllowEmbeddingAPI \
  --priority 1010 \
  --destination-port-ranges 8000 \
  --protocol Tcp \
  --access Allow
```

Sau đó gọi từ bên ngoài:
```bash
curl http://<VM_PUBLIC_IP>:8000/health
```

---

## 7. Gọi từ Google Colab

```python
import httpx

VM_URL = "http://<VM_PUBLIC_IP>:8000"

# Semantic embedding
resp = httpx.post(f"{VM_URL}/embed/semantic", json={"text": "nấu phở bò"})
vec = resp.json()["embedding"]
print(f"BGE-M3: dim={len(vec)}, latency={resp.json()['latency_ms']}ms")

# Visual embedding
resp = httpx.post(f"{VM_URL}/embed/visual", json={"text": "red shirt cooking"})
vec = resp.json()["embedding"]
print(f"SigLIP: dim={len(vec)}, latency={resp.json()['latency_ms']}ms")
```

---

## 8. Quản lý

```bash
# Xem status
docker compose ps

# Xem logs
docker compose logs -f --tail=50

# Restart
docker compose restart

# Stop
docker compose down

# Rebuild (sau khi sửa code)
docker compose up -d --build --force-recreate
```

---

## Resource Budget

| Component | RAM | CPU |
|-----------|-----|-----|
| BGE-M3 (~2.2GB) | ~3GB loaded | shared |
| SigLIP (~1.6GB) | ~2.5GB loaded | shared |
| BM25 (~50MB) | ~0.1GB loaded | shared |
| FastAPI + overhead | ~0.5GB | shared |
| **Total** | **~6-8GB** | **3.5 cores** |
| OS reserved | ~2GB | 0.5 core |
| **VM Total** | **16GB** | **4 cores** |
