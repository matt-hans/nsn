#!/bin/bash
# Quick start script for NSN local development environment
# Automates the setup and verification process

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "NSN Local Development Environment"
echo "Quick Start Script"
echo -e "==========================================${NC}"
echo ""

# Step 1: Check GPU compatibility
echo -e "${BLUE}Step 1/5: Checking GPU compatibility...${NC}"
if [ -f "./scripts/check-gpu.sh" ]; then
    if ./scripts/check-gpu.sh; then
        echo -e "${GREEN}✓ GPU compatibility verified${NC}"
    else
        echo -e "${RED}✗ GPU compatibility check failed${NC}"
        echo "Please fix GPU issues before continuing."
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ GPU check script not found, skipping...${NC}"
fi
echo ""

# Step 2: Setup environment file
echo -e "${BLUE}Step 2/5: Setting up environment configuration...${NC}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env from template${NC}"
    else
        echo -e "${RED}✗ .env.example not found${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ .env already exists, skipping...${NC}"
fi
echo ""

# Step 3: Pull/build images
echo -e "${BLUE}Step 3/5: Building Docker images...${NC}"
echo "This may take 5-10 minutes on first run..."
if docker compose build; then
    echo -e "${GREEN}✓ Docker images built successfully${NC}"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi
echo ""

# Step 4: Start services
echo -e "${BLUE}Step 4/5: Starting services...${NC}"
echo "This will take ~60-120 seconds for all services to become healthy..."
docker compose up -d

# Wait for services to be healthy
echo ""
echo "Waiting for services to start..."
TIMEOUT=120
ELAPSED=0
INTERVAL=5

while [ $ELAPSED -lt $TIMEOUT ]; do
    HEALTHY=$(docker compose ps --format json | jq -r 'select(.Health == "healthy") | .Service' 2>/dev/null | wc -l)
    TOTAL=$(docker compose ps --format json | jq -r '.Service' 2>/dev/null | wc -l)

    echo -ne "\rHealthy services: $HEALTHY/$TOTAL (${ELAPSED}s elapsed)"

    if [ "$HEALTHY" -eq "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
        echo ""
        echo -e "${GREEN}✓ All services are healthy!${NC}"
        break
    fi

    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo ""
    echo -e "${YELLOW}⚠ Timeout waiting for services. Some may still be starting...${NC}"
    echo "Check status with: docker compose ps"
fi
echo ""

# Step 5: Verify connectivity
echo -e "${BLUE}Step 5/5: Verifying service connectivity...${NC}"

# Check Substrate RPC
if curl -s http://localhost:9933/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Substrate RPC accessible${NC}"
else
    echo -e "${YELLOW}⚠ Substrate RPC not yet ready${NC}"
fi

# Check Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Prometheus accessible${NC}"
else
    echo -e "${YELLOW}⚠ Prometheus not yet ready${NC}"
fi

# Check Grafana
if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Grafana accessible${NC}"
else
    echo -e "${YELLOW}⚠ Grafana not yet ready${NC}"
fi

echo ""
echo -e "${BLUE}=========================================="
echo "Setup Complete!"
echo -e "==========================================${NC}"
echo ""
echo "Access your services:"
echo ""
echo -e "${GREEN}Blockchain:${NC}"
echo "  Substrate RPC:  ws://localhost:9944"
echo "  Polkadot.js:    https://polkadot.js.org/apps/?rpc=ws://localhost:9944"
echo ""
echo -e "${GREEN}Observability:${NC}"
echo "  Prometheus:     http://localhost:9090"
echo "  Grafana:        http://localhost:3000 (admin/admin)"
echo "  Jaeger:         http://localhost:16686"
echo ""
echo -e "${GREEN}AI Engine:${NC}"
echo "  Vortex gRPC:    localhost:50051"
echo "  GPU Status:     docker compose exec vortex nvidia-smi"
echo ""
echo "View logs:         docker compose logs -f"
echo "Stop services:     docker compose down"
echo "Full reset:        docker compose down -v && docker compose up"
echo ""
echo "For more information, see docs/local-development.md"
echo ""
