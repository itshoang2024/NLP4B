#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# setup_host.sh — Azure VM Host OS Initialization (Ubuntu 22.04 LTS)
# Target: Standard_NC4as_T4_v3 (1x T4 GPU, 4 vCPUs, 28GB RAM)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

echo "═══════════════════════════════════════════════════════════"
echo "  🚀 Phase 1: Host OS Setup — Azure AI Provider"
echo "  Target: Standard_NC4as_T4_v3 (T4 16GB, 4 vCPU, 28GB RAM)"
echo "═══════════════════════════════════════════════════════════"

# ── 1. System Update ─────────────────────────────────────────────────────────
echo ""
echo "[1/5] 📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y \
    curl wget git htop nvtop tmux jq \
    ca-certificates gnupg lsb-release \
    build-essential software-properties-common

# ── 2. NVIDIA Driver (535-server for T4) ─────────────────────────────────────
echo ""
echo "[2/5] 🎮 Installing NVIDIA Driver 535 (server)..."
# Check if driver is already installed
if nvidia-smi &>/dev/null; then
    echo "  ✅ NVIDIA driver already installed:"
    nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader
else
    sudo apt-get install -y linux-headers-$(uname -r)
    sudo apt-get install -y nvidia-driver-535-server nvidia-utils-535-server
    echo "  ⚠️  NVIDIA driver installed. REBOOT REQUIRED before proceeding."
    echo "     Run: sudo reboot"
    echo "     Then re-run this script to continue from step 3."
    # Uncomment to auto reboot:
    # sudo reboot
fi

# ── 3. Docker Engine ─────────────────────────────────────────────────────────
echo ""
echo "[3/5] 🐳 Installing Docker Engine..."
if docker --version &>/dev/null; then
    echo "  ✅ Docker already installed: $(docker --version)"
else
    # Add Docker official GPG key
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    # Add Docker repo
    echo \
      "deb [arch=\"$(dpkg --print-architecture)\" signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Allow current user to run docker without sudo
    sudo usermod -aG docker "$USER"
    echo "  ✅ Docker installed. Log out and back in for group changes."
fi

# ── 4. NVIDIA Container Toolkit ──────────────────────────────────────────────
echo ""
echo "[4/5] 🔧 Installing NVIDIA Container Toolkit..."
if dpkg -l | grep -q nvidia-container-toolkit; then
    echo "  ✅ NVIDIA Container Toolkit already installed."
else
    # Add NVIDIA container toolkit repo
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update -y
    sudo apt-get install -y nvidia-container-toolkit

    # Configure Docker to use NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    echo "  ✅ NVIDIA Container Toolkit installed."
fi

# ── 5. Restart Docker & Verify ───────────────────────────────────────────────
echo ""
echo "[5/5] 🔄 Restarting Docker and verifying GPU access..."
sudo systemctl restart docker
sudo systemctl enable docker

# Verify GPU in Docker
echo ""
echo "── Verification ──────────────────────────────────────────"
echo "Docker version:"
docker --version
echo ""
echo "Docker Compose version:"
docker compose version
echo ""

if nvidia-smi &>/dev/null; then
    echo "Host GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
    echo "Docker GPU test:"
    docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi \
        --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
        && echo "  ✅ Docker can see GPU!" \
        || echo "  ⚠️  Docker cannot see GPU. Check nvidia-container-toolkit config."
else
    echo "  ⚠️  nvidia-smi not found. Reboot may be needed after driver install."
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Host setup complete!"
echo "  Next: cd azure-ai-provider && docker compose up -d"
echo "═══════════════════════════════════════════════════════════"
