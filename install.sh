#!/usr/bin/env bash
# ==============================================================================
# OLMoE++ Professional Environment Setup Script (Dockerized)
# Target OS: Arch Linux
# Description: Cleans the workspace, structures the project, and builds an
#              isolated, editable container for core ML development.
# ==============================================================================

set -euo pipefail

# Terminal Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}      Initializing OLMoE++ Dockerized Workspace     ${NC}"
echo -e "${BLUE}====================================================${NC}"

# 1. Prerequisite & Permission Checks
echo -e "\n${YELLOW}[1/6] Checking system prerequisites and permissions...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    exit 1
fi

if ! docker ps &> /dev/null; then
    echo -e "${RED}Error: Permission denied to Docker daemon.${NC}"
    echo -e "Your user is not in the docker group, or your terminal hasn't refreshed."
    echo -e "Please run: ${YELLOW}newgrp docker${NC} and try running this script again."
    exit 1
fi
echo -e "${GREEN}Docker found and permissions verified. Proceeding...${NC}"

# 2. Deep Clean & Project Restructuring
echo -e "\n${YELLOW}[2/6] Deep cleaning old artifacts and restructuring...${NC}"
cd "${PROJECT_ROOT}"

# Nuke the old broken virtual environments and source folders
rm -rf venv/ .conda/ miniconda3/ 2>/dev/null || true

# Create the professional workspace structure
mkdir -p src configs data logs notebooks scripts
echo -e "${GREEN}Workspace directories created: src/ configs/ data/ logs/ notebooks/ scripts/${NC}"

# 3. Clone the Repositories
echo -e "\n${YELLOW}[3/6] Cloning core repositories (Host Side)...${NC}"
cd src

echo "Cloning OLMoE..."
git clone https://github.com/allenai/OLMoE.git

echo "Cloning OLMo (Stable version)..."
git clone https://github.com/allenai/OLMo.git
cd OLMo
git checkout $(git rev-list -n 1 --before="2024-09-01" main)
cd ..

cd "${PROJECT_ROOT}"

# 4. Generate the Dockerfile
echo -e "\n${YELLOW}[4/6] Generating Dockerfile...${NC}"
cat << 'EOF' > Dockerfile
# Base image: Official PyTorch Devel (Includes full CUDA 12.1 toolkit, headers, and GCC 11)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Install essential system build tools
RUN apt-get update && apt-get install -y \
    git ninja-build build-essential vim tmux wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install core dependencies with strict version locking
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy<2" "vllm==0.5.0" "transformers" "wandb" "evaluate" "accelerate" "ninja" "packaging"

# Copy the host's source code into the container for the initial compilation
COPY ./src /workspace/src

# Compile and install MegaBlocks (Editable mode, no isolation)
WORKDIR /workspace/src/megablocks
RUN MAX_JOBS=4 pip install -e . --no-build-isolation --no-cache-dir

# Install OLMo (Editable mode)
WORKDIR /workspace/src/OLMo
RUN pip install -e .[all] --no-build-isolation --no-cache-dir

# Install OLMoE requirements
WORKDIR /workspace/src/OLMoE
RUN pip install -r requirements.txt --no-cache-dir || true

WORKDIR /workspace
CMD ["/bin/bash"]
EOF

# 5. Generate Modern docker-compose.yml
echo -e "\n${YELLOW}[5/6] Generating docker-compose.yml...${NC}"
cat << 'EOF' > docker-compose.yml
services:
  olmoe_dev:
    build:
      context: .
      dockerfile: Dockerfile
    image: olmoe-dev-env:latest
    container_name: olmoe_workspace
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    volumes:
      - ./src:/workspace/src
      - ./configs:/workspace/configs
      - ./data:/workspace/data
      - ./logs:/workspace/logs
      - ./scripts:/workspace/scripts
    working_dir: /workspace
    stdin_open: true
    tty: true
    ipc: host
    command: /bin/bash
EOF

# Generate a quick-start helper script
cat << 'EOF' > start_env.sh
#!/usr/bin/env bash
echo "Starting OLMoE++ Docker Environment..."
docker compose run --rm olmoe_dev
EOF
chmod +x start_env.sh

# 6. Build the Container
echo -e "\n${YELLOW}[6/6] Building the Docker Image...${NC}"
# Use the modern compose plugin command
docker compose build

echo -e "\n${GREEN}====================================================${NC}"
echo -e "${GREEN}        OLMoE++ Architecture Successfully Built!    ${NC}"
echo -e "${GREEN}====================================================${NC}"
echo -e "To enter your professional ML environment, run:"
echo -e "    ${BLUE}./start_env.sh${NC}\n"
echo -e "===================================================="