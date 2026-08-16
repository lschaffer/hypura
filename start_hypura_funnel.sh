#!/usr/bin/env bash
set -e

# ==============================================================================
# Hypura Startup Script: Dynamic Multi-Model Serve & Tailscale Funnel
# ==============================================================================

PORT=6000
CONTEXT=8192  # Default 8k context window (client can override per request via num_ctx)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/../target/release/hypura" ]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

HYPURA_BIN="$ROOT_DIR/target/release/hypura"

# Optional pre-warmed model
MODEL="${1:-}"

# Build binary if not present
if [ ! -f "$HYPURA_BIN" ]; then
    echo "Building Hypura release binary..."
    (cd "$ROOT_DIR" && cargo build --release)
fi

# Stop any existing server on this port
echo "Stopping any existing Hypura instance..."
pkill -f "hypura serve" || true
sleep 1

# Setup Tailscale Funnel in background
if command -v tailscale &> /dev/null; then
    echo "Configuring Tailscale Funnel on port $PORT..."
    tailscale funnel --bg --yes $PORT || true
    echo "Tailscale Funnel active."
else
    echo "Warning: tailscale CLI not found in PATH."
fi

# Display Network Endpoints
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "127.0.0.1")
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "")

echo ""
echo "=========================================================="
echo " Hypura Ollama-Compatible Multi-Model Server"
if [ -n "$MODEL" ]; then
    echo " Mode:    Pre-warmed ($MODEL)"
else
    echo " Mode:    Dynamic On-Demand (All Local + Ollama Models)"
fi
echo " Context: $CONTEXT default tokens (client adjustable)"
echo " Port:    $PORT"
echo "----------------------------------------------------------"
echo " Local URL:     http://localhost:$PORT"
echo " LAN URL:       http://$LOCAL_IP:$PORT"
if [ -n "$TAILSCALE_IP" ]; then
    echo " Tailscale URL: http://$TAILSCALE_IP:$PORT"
fi
echo "=========================================================="
echo ""

# Start Hypura Server (with or without pre-warmed model)
if [ -n "$MODEL" ]; then
    exec "$HYPURA_BIN" serve "$MODEL" --host 0.0.0.0 --port "$PORT" --context "$CONTEXT"
else
    exec "$HYPURA_BIN" serve --host 0.0.0.0 --port "$PORT" --context "$CONTEXT"
fi
