#!/bin/bash

# ==============================================================================
# DECODE Energy Router - Edge Gateway Provisioning Script (v1.0.0)
# ==============================================================================
# Description: Bootstraps a bare-metal Linux device for K3s + GPU Inference.
# Usage: sudo ./provision_gateway.sh <NODE_TOKEN> <SERVER_URL>
# ==============================================================================

set -e  # Fail Fast: Stop on first error
set -o pipefail

# --- Configuration Constants ---
K3S_VERSION="v1.28.5+k3s1"
USER_NAME="energyuser"
DOCKER_VERSION="24.0.0"
LOG_FILE="/var/log/decode_provisioning.log"

# --- 1. Validation & Pre-flight Checks ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <K3S_TOKEN> <K3S_SERVER_URL>"
    exit 1
fi

K3S_TOKEN=$1
K3S_URL=$2

echo "⚡ Starting Edge Gateway Provisioning..." | tee -a $LOG_FILE

# Check for Root
if [ "$EUID" -ne 0 ]; then
  echo "Error: Please run as root."
  exit 1
fi

# --- 2. System Hardening (NERC CIP Compliance) ---
echo "🔒 Hardening System Security..." | tee -a $LOG_FILE

# Disable Password Auth for Root (SSH Key only)
sed -i 's/PermitRootLogin yes/PermitRootLogin prohibit-password/g' /etc/ssh/sshd_config
systemctl reload sshd

# Setup Firewall (UFW)
# Allow SSH (22), K3s API (6443), and our NodePort (30080)
apt-get update && apt-get install -y ufw
ufw default deny incoming
ufw allow 22/tcp
ufw allow 6443/tcp
ufw allow 30080/tcp
ufw --force enable

# --- 3. Install Container Runtime (NVIDIA Container Toolkit) ---
echo "🐳 Installing Container Runtime (Docker + NVIDIA Support)..." | tee -a $LOG_FILE

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker $USER_NAME

# Setup NVIDIA Container Toolkit (for Jetson/GPU support)
# Note: Checks if NVIDIA hardware exists first
if lspci | grep -i nvidia > /dev/null; then
    echo "   > NVIDIA GPU detected. Configuring runtime..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    apt-get update && apt-get install -y nvidia-container-toolkit
    systemctl restart docker
    
    # Set NVIDIA as default runtime in daemon.json
    cat > /etc/docker/daemon.json <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
    systemctl restart docker
else
    echo "   > No NVIDIA GPU detected. Using standard CPU runtime."
fi

# --- 4. Install K3s (Lightweight Kubernetes) ---
echo "☸️ Installing K3s Agent..." | tee -a $LOG_FILE

# Install K3s Agent and join the cluster
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=$K3S_VERSION K3S_URL=$K3S_URL K3S_TOKEN=$K3S_TOKEN sh -s - agent \
    --docker # Use docker runtime instead of containerd for GPU simplicity in MVP

# --- 5. Inject Configuration ---
echo "⚙️ Injecting Site Configuration..." | tee -a $LOG_FILE

# Create directory structure
mkdir -p /opt/decode/config
mkdir -p /opt/decode/models

# Set permissions
chown -R $USER_NAME:$USER_NAME /opt/decode

# --- 6. Verification ---
echo "✅ Provisioning Complete!" | tee -a $LOG_FILE
echo "   > Node Status: $(systemctl is-active k3s-agent)"
echo "   > Docker Status: $(systemctl is-active docker)"
echo "   > Firewall: Active"

exit 0
