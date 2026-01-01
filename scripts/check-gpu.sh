#!/bin/bash
# GPU compatibility check script for NSN local development environment
# Verifies NVIDIA GPU, drivers, and Docker GPU support

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "NSN GPU Compatibility Check"
echo "=========================================="
echo ""

# Check 1: NVIDIA GPU present
echo "1. Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
else
    echo -e "${RED}✗ nvidia-smi not found. Is NVIDIA GPU installed?${NC}"
    exit 1
fi
echo ""

# Check 2: NVIDIA driver version
echo "2. Checking NVIDIA driver version..."
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
MAJOR_VERSION=${DRIVER_VERSION%%.*}

echo "   Detected driver version: $DRIVER_VERSION"

if [ "$MAJOR_VERSION" -lt 535 ]; then
    echo -e "${RED}✗ Driver version $DRIVER_VERSION is too old. Minimum required: 535.x${NC}"
    echo "   Please update your NVIDIA drivers."
    exit 1
else
    echo -e "${GREEN}✓ Driver version is compatible (>= 535)${NC}"
fi
echo ""

# Check 3: GPU VRAM
echo "3. Checking GPU VRAM..."
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$((VRAM_MB / 1024))

echo "   Total VRAM: ${VRAM_GB} GB"

if [ "$VRAM_GB" -lt 12 ]; then
    echo -e "${YELLOW}⚠ Warning: VRAM is ${VRAM_GB} GB. NSN requires minimum 12 GB for Vortex.${NC}"
    echo "   You may need to reduce model precision or disable Vortex service."
else
    echo -e "${GREEN}✓ VRAM meets requirements (>= 12 GB)${NC}"
fi
echo ""

# Check 4: Docker installed
echo "4. Checking Docker installation..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo "   $DOCKER_VERSION"
    echo -e "${GREEN}✓ Docker is installed${NC}"
else
    echo -e "${RED}✗ Docker not found. Please install Docker Desktop or Docker Engine.${NC}"
    exit 1
fi
echo ""

# Check 5: Docker Compose installed
echo "5. Checking Docker Compose..."
if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version)
    echo "   $COMPOSE_VERSION"
    echo -e "${GREEN}✓ Docker Compose is available${NC}"
else
    echo -e "${RED}✗ Docker Compose not found. Please install Docker Compose plugin.${NC}"
    exit 1
fi
echo ""

# Check 6: NVIDIA Container Toolkit
echo "6. Checking NVIDIA Container Toolkit..."
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA Container Toolkit is properly configured${NC}"
    echo "   GPU passthrough to containers is working."
else
    echo -e "${RED}✗ NVIDIA Container Toolkit not working${NC}"
    echo ""
    echo "To install NVIDIA Container Toolkit:"
    echo ""
    echo "1. Add the NVIDIA Container Toolkit repository:"
    echo "   distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
    echo "   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
    echo "   curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
    echo ""
    echo "2. Install nvidia-docker2:"
    echo "   sudo apt-get update"
    echo "   sudo apt-get install -y nvidia-docker2"
    echo ""
    echo "3. Restart Docker:"
    echo "   sudo systemctl restart docker"
    echo ""
    exit 1
fi
echo ""

# Check 7: Available disk space
echo "7. Checking available disk space..."
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')

echo "   Available space: ${AVAILABLE_GB} GB"

if [ "$AVAILABLE_GB" -lt 50 ]; then
    echo -e "${YELLOW}⚠ Warning: Only ${AVAILABLE_GB} GB available. NSN requires ~50 GB for models and containers.${NC}"
    echo "   You may encounter disk space issues."
else
    echo -e "${GREEN}✓ Sufficient disk space available${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "${GREEN}All checks passed!${NC}"
echo ""
echo "Your system is ready for NSN local development."
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env"
echo "2. Run: docker compose up"
echo "3. Wait for all services to start (~120 seconds)"
echo ""
echo "Ports:"
echo "  - Substrate RPC:  ws://localhost:9944"
echo "  - Prometheus:     http://localhost:9090"
echo "  - Grafana:        http://localhost:3000 (admin/admin)"
echo "  - Jaeger:         http://localhost:16686"
echo ""
