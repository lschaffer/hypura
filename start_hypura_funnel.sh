#!/usr/bin/env bash
set -e

# ==============================================================================
# Hypura Startup Script with 12k Context & Tailscale Funnel
# ==============================================================================

PORT=6000
CONTEXT=12288  # 12k context window
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/../target/release/hypura" ]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

HYPURA_BIN="$ROOT_DIR/target/release/hypura"

# Auto-detect default model
MODEL="${1:-}"
if [ -z "$MODEL" ]; then
    if [ -f "$ROOT_DIR/mistral-small3.2-24b.gguf" ]; then
        MODEL="$ROOT_DIR/mistral-small3.2-24b.gguf"
    elif [ -f "$ROOT_DIR/gemma4-98e-v5-coder.gguf" ]; then
        MODEL="$ROOT_DIR/gemma4-98e-v5-coder.gguf"
    else
        echo "Error: No model specified and no default .gguf found in $ROOT_DIR"
        echo "Usage: $0 [path_to_model.gguf]"
        exit 1
    fi
fi

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
echo " Hypura Ollama-Compatible Server"
echo " Model:   $MODEL"
echo " Context: $CONTEXT tokens (12k)"
echo " Port:    $PORT"
echo "----------------------------------------------------------"
echo " Local URL:     http://localhost:$PORT"
echo " LAN URL:       http://$LOCAL_IP:$PORT"
if [ -n "$TAILSCALE_IP" ]; then
    echo " Tailscale URL: http://$TAILSCALE_IP:$PORT"
fi
echo "=========================================================="
echo ""

# Start Hypura Server
exec "$HYPURA_BIN" serve "$MODEL" --host 0.0.0.0 --port "$PORT" --context "$CONTEXT"
