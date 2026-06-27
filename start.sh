#!/usr/bin/env bash
# ==============================================================================
# OmniVoice — Start Script
#
# Automatically detects GPU availability and starts the appropriate container.
# Can also be used with explicit mode flags.
#
# Usage:
#   ./start.sh              # Auto-detect GPU/CPU, start API (Swagger)
#   ./start.sh api          # Start REST API with Swagger UI
#   ./start.sh demo         # Start Gradio demo UI
#   ./start.sh all          # Start both API + Demo
#   ./start.sh gpu          # Force GPU mode (API)
#   ./start.sh cpu          # Force CPU mode (API)
#   ./start.sh stop         # Stop running containers
#   ./start.sh logs         # View logs
#   ./start.sh build        # Build only (no start)
#
# Environment Variables:
#   OMNIVOICE_DEVICE=gpu|cpu    Override GPU auto-detection
#   OMNIVOICE_SERVICE=api|demo|all  Override service selection
# ==============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

print_banner() {
    echo -e "${CYAN}"
    echo "  ╔═══════════════════════════════════════════════════╗"
    echo "  ║            🌍 OmniVoice TTS Server               ║"
    echo "  ║     600+ Languages · Voice Clone · Voice Design   ║"
    echo "  ╚═══════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log_info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# Check if NVIDIA GPU + Docker runtime is available
check_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        return 1
    fi
    if ! nvidia-smi &>/dev/null; then
        return 1
    fi
    if ! docker info 2>/dev/null | grep -qi "nvidia"; then
        if ! docker info 2>/dev/null | grep -qi "runtimes.*nvidia"; then
            log_warn "NVIDIA GPU detected but NVIDIA Container Toolkit not found."
            log_warn "Install it: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            return 1
        fi
    fi
    return 0
}

# Detect the compose command
get_compose_cmd() {
    if docker compose version &>/dev/null 2>&1; then
        echo "docker compose"
    elif command -v docker-compose &>/dev/null; then
        echo "docker-compose"
    else
        log_error "Docker Compose not found. Please install Docker with Compose plugin."
        exit 1
    fi
}

# Auto-detect device (GPU or CPU)
detect_device() {
    local device="${OMNIVOICE_DEVICE:-}"
    if [ -n "$device" ]; then
        echo "$device"
        return
    fi
    if check_gpu; then
        echo "gpu"
    else
        echo "cpu"
    fi
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

do_start() {
    local device="$1"
    local service="$2"  # api, demo, or all
    local compose_cmd
    compose_cmd=$(get_compose_cmd)
    local compose_file="docker-compose.${device}.yml"

    if [ ! -f "$compose_file" ]; then
        log_error "Compose file not found: $compose_file"
        exit 1
    fi

    print_banner

    if [ "$device" = "gpu" ]; then
        log_info "Device: ${GREEN}GPU${NC} (NVIDIA CUDA) 🚀"
    else
        log_info "Device: ${YELLOW}CPU${NC} 🐢"
        log_warn "Inference will be slower without GPU"
    fi

    log_info "Service: ${MAGENTA}${service}${NC}"
    echo ""
    log_info "Building and starting containers..."
    $compose_cmd -f "$compose_file" --profile "$service" up --build -d

    echo ""
    log_info "Container(s) started successfully! ✅"
    echo ""
    echo -e "  ${CYAN}┌──────────────────────────────────────────────────────┐${NC}"
    if [ "$service" = "api" ] || [ "$service" = "all" ]; then
    echo -e "  ${CYAN}│${NC}  📡 REST API:     ${GREEN}http://localhost:8000${NC}              ${CYAN}│${NC}"
    echo -e "  ${CYAN}│${NC}  📖 Swagger UI:   ${GREEN}http://localhost:8000/docs${NC}         ${CYAN}│${NC}"
    echo -e "  ${CYAN}│${NC}  📘 ReDoc:        ${GREEN}http://localhost:8000/redoc${NC}        ${CYAN}│${NC}"
    fi
    if [ "$service" = "demo" ] || [ "$service" = "all" ]; then
    echo -e "  ${CYAN}│${NC}  🌐 Gradio Demo:  ${GREEN}http://localhost:7860${NC}              ${CYAN}│${NC}"
    fi
    echo -e "  ${CYAN}│${NC}                                                      ${CYAN}│${NC}"
    echo -e "  ${CYAN}│${NC}  📋 Logs:         ${BLUE}./start.sh logs${NC}                   ${CYAN}│${NC}"
    echo -e "  ${CYAN}│${NC}  🛑 Stop:         ${RED}./start.sh stop${NC}                   ${CYAN}│${NC}"
    echo -e "  ${CYAN}└──────────────────────────────────────────────────────┘${NC}"
    echo ""
    log_info "First startup will download models (~5GB). This may take a while..."
    log_info "Watch progress with: ./start.sh logs"
}

do_stop() {
    local compose_cmd
    compose_cmd=$(get_compose_cmd)

    log_info "Stopping OmniVoice containers..."

    for mode in gpu cpu; do
        local compose_file="docker-compose.${mode}.yml"
        if [ -f "$compose_file" ]; then
            # Stop all profiles
            $compose_cmd -f "$compose_file" --profile all down 2>/dev/null || true
            $compose_cmd -f "$compose_file" --profile api down 2>/dev/null || true
            $compose_cmd -f "$compose_file" --profile demo down 2>/dev/null || true
        fi
    done

    log_info "All containers stopped ✅"
}

do_logs() {
    local compose_cmd
    compose_cmd=$(get_compose_cmd)

    # Find which containers are running
    for mode in gpu cpu; do
        local compose_file="docker-compose.${mode}.yml"
        if [ -f "$compose_file" ]; then
            for svc in api demo; do
                local container_name="omnivoice-${mode}-${svc}"
                if docker ps --format '{{.Names}}' | grep -q "$container_name"; then
                    log_info "Showing logs (Ctrl+C to exit)..."
                    # Show logs from all running containers for this mode
                    $compose_cmd -f "$compose_file" --profile all logs -f 2>/dev/null || \
                    $compose_cmd -f "$compose_file" --profile "$svc" logs -f
                    return
                fi
            done
        fi
    done

    log_warn "No running OmniVoice container found."
}

do_build() {
    local device="$1"
    local compose_cmd
    compose_cmd=$(get_compose_cmd)
    local compose_file="docker-compose.${device}.yml"

    if [ ! -f "$compose_file" ]; then
        log_error "Compose file not found: $compose_file"
        exit 1
    fi

    log_info "Building OmniVoice image (${device} mode)..."
    $compose_cmd -f "$compose_file" --profile all build --pull
    log_info "Build completed ✅"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    local action="${1:-auto}"
    local arg2="${2:-}"

    case "$action" in
        api)
            local device
            device=$(detect_device)
            log_info "Auto-detected device: $device"
            do_start "$device" "api"
            ;;
        demo)
            local device
            device=$(detect_device)
            log_info "Auto-detected device: $device"
            do_start "$device" "demo"
            ;;
        all)
            local device
            device=$(detect_device)
            log_info "Auto-detected device: $device"
            do_start "$device" "all"
            ;;
        gpu)
            local service="${arg2:-api}"
            do_start "gpu" "$service"
            ;;
        cpu)
            local service="${arg2:-api}"
            do_start "cpu" "$service"
            ;;
        start)
            # Alias: ./start.sh start [gpu|cpu] [api|demo|all]
            local device="${arg2:-}"
            local service="${3:-api}"
            if [ -z "$device" ]; then
                device=$(detect_device)
                log_info "Auto-detected device: $device"
            fi
            do_start "$device" "$service"
            ;;
        stop)
            do_stop
            ;;
        logs)
            do_logs
            ;;
        build)
            if [ -z "$arg2" ]; then
                log_info "Building both GPU and CPU images..."
                do_build "gpu"
                do_build "cpu"
            else
                do_build "$arg2"
            fi
            ;;
        auto|"")
            local device
            device=$(detect_device)
            log_info "Auto-detected device: $device"
            do_start "$device" "api"
            ;;
        -h|--help|help)
            echo ""
            echo "Usage: $0 [command] [options]"
            echo ""
            echo "Commands:"
            echo "  (none)             Auto-detect GPU, start API server"
            echo "  api                Start REST API server (Swagger at :8000/docs)"
            echo "  demo               Start Gradio demo UI (:7860)"
            echo "  all                Start both API + Demo"
            echo "  gpu [api|demo|all] Start with GPU (default: api)"
            echo "  cpu [api|demo|all] Start with CPU (default: api)"
            echo "  stop               Stop all containers"
            echo "  logs               View container logs"
            echo "  build [gpu|cpu]    Build images"
            echo "  help               Show this help"
            echo ""
            echo "Environment Variables:"
            echo "  OMNIVOICE_DEVICE=gpu|cpu     Override GPU auto-detection"
            echo ""
            echo "Examples:"
            echo "  $0                    # Auto-detect GPU, start API"
            echo "  $0 api               # Start API (Swagger at localhost:8000/docs)"
            echo "  $0 demo              # Start Gradio demo"
            echo "  $0 all               # Start API + Demo"
            echo "  $0 gpu all           # GPU mode, both services"
            echo "  $0 cpu demo          # CPU mode, demo only"
            echo ""
            ;;
        *)
            log_error "Unknown command: $action"
            echo "Run '$0 help' for usage."
            exit 1
            ;;
    esac
}

main "$@"
