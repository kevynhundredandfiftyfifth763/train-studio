#!/usr/bin/env bash
# 🚀 Train Studio — one-click installer
#
#   curl -sSL https://raw.githubusercontent.com/nanofatdog/train-studio/master/install.sh | bash
#
# Options (pass after the pipe is NOT possible; use env vars):
#   APP_DIR=/path  RUN_AFTER=1  curl ... | bash
#   หรือ clone แล้ว: bash install.sh --run --dir=/path
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[-]${NC} $*" >&2; }

APP_DIR="${APP_DIR:-$HOME/train-studio}"
RUN_AFTER="${RUN_AFTER:-0}"
REPO_URL="https://github.com/nanofatdog/train-studio.git"

for arg in "$@"; do
  case "$arg" in
    --run) RUN_AFTER=1 ;;
    --dir=*) APP_DIR="${arg#*=}" ;;
  esac
done

info "🚀 Train Studio Installer"
info "Target dir : $APP_DIR"
info "Run after  : $([ "$RUN_AFTER" = 1 ] && echo yes || echo no)"

# ---------- 0. System prerequisites (auto apt install) ----------
SYS_NEEDED=0
command -v git  >/dev/null 2>&1 || SYS_NEEDED=1
command -v curl >/dev/null 2>&1 || SYS_NEEDED=1
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
    SYS_NEEDED=1
fi
if [ "$SYS_NEEDED" = "1" ]; then
    info "ติดตั้ง system packages ที่จำเป็น (python3-venv, git, curl)..."
    if command -v apt-get >/dev/null 2>&1; then
        if [ "$(id -u)" = "0" ]; then
            apt-get update -qq && apt-get install -y -qq python3-venv python3-pip git curl || { err "apt install failed"; exit 1; }
        else
            sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-pip git curl || { err "apt install failed (ลอง sudo)"; exit 1; }
        fi
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache python3 py3-pip git curl || { err "apk install failed"; exit 1; }
    elif command -v dnf >/dev/null 2>&1; then
        dnf install -y python3-pip git curl || { err "dnf install failed"; exit 1; }
    else
        warn "ไม่รู้จัก package manager — ติดตั้ง python3-venv / git / curl ด้วยตัวเองก่อน"
    fi
    # fallback: บาง distro ต้องระบุ version (python3.12-venv)
    if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
        PYV=$(python3 -c 'import sys; print(f"python3.{sys.version_info.minor}-venv")' 2>/dev/null || echo "python3-venv")
        if [ "$(id -u)" = "0" ]; then
            apt-get install -y -qq "$PYV" 2>/dev/null || true
        else
            sudo apt-get install -y -qq "$PYV" 2>/dev/null || true
        fi
    fi
fi

# ---------- 1. Python check ----------
if ! command -v python3 >/dev/null 2>&1; then
  err "python3 not found — install Python 3.10+ first"
  exit 1
fi
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Python: $PYVER"
if [[ $(echo "$PYVER" | cut -d. -f1) -lt 3 ]] || { [[ $(echo "$PYVER" | cut -d. -f1) -eq 3 ]] && [[ $(echo "$PYVER" | cut -d. -f2) -lt 10 ]]; }; then
  err "Python 3.10+ required (got $PYVER)"
  exit 1
fi

# ---------- 2. GPU / CUDA check ----------
CUDA_FULL=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  CUDA_FULL=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: //' | awk '{print $1}')
  info "GPU detected — CUDA Version: ${CUDA_FULL:-unknown}"
else
  warn "No NVIDIA GPU detected — จะติดตั้ง torch CPU version (เปิด UI ได้ แต่เทรนไม่ได้)"
fi

# ---------- 3. Disk check ----------
mkdir -p "$APP_DIR" 2>/dev/null || true
FREE_GB=$(df -BG "$APP_DIR" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
info "Free disk : ${FREE_GB:-?} GB (torch+model ต้องการ ~20GB)"

# ---------- 4. Clone / update ----------
if [ -d "$APP_DIR/.git" ]; then
  info "Updating existing install..."
  git -C "$APP_DIR" pull --ff-only 2>/dev/null || warn "git pull failed — continue with existing files"
elif [ -f "$APP_DIR/app.py" ]; then
  info "Existing install found (no .git) — reusing files in $APP_DIR"
  warn "เพื่ออัปเดตโค้ด: cd $APP_DIR && git init && git remote add origin $REPO_URL && git fetch && git checkout -f origin/master"
else
  if ! command -v git >/dev/null 2>&1; then
    err "git not found — install git first"
    exit 1
  fi
  mkdir -p "$APP_DIR"
  info "Cloning repo..."
  git clone --depth 1 "$REPO_URL" "$APP_DIR" || { err "clone failed"; exit 1; }
fi
cd "$APP_DIR"

# ---------- 5. venv ----------
if [ ! -d venv ]; then
  info "Creating venv..."
  python3 -m venv venv || { rm -rf venv; err "venv creation failed — ลอง: apt install python3-venv แล้วรันใหม่"; exit 1; }
fi
./venv/bin/pip install -q --upgrade pip 2>/dev/null \
  || { ./venv/bin/python -m ensurepip --upgrade 2>/dev/null; ./venv/bin/pip install -q --upgrade pip; }

# ---------- 6. torch (ตาม CUDA) ----------
if ! ./venv/bin/python -c "import torch" >/dev/null 2>&1; then
  TORCH_INDEX=""
  if [ -n "$CUDA_FULL" ]; then
    case "$CUDA_FULL" in
      13.*) TORCH_INDEX="https://download.pytorch.org/whl/cu130" ;;
      12.8*) TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
      12.6*) TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
      12.4*) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
      12.1*) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    esac
  fi
  if [ -n "$TORCH_INDEX" ]; then
    info "Installing torch (CUDA $CUDA_FULL -> $TORCH_INDEX)..."
    ./venv/bin/pip install torch --index-url "$TORCH_INDEX" || { err "torch install failed"; exit 1; }
  else
    info "Installing torch (default)..."
    ./venv/bin/pip install torch || { err "torch install failed"; exit 1; }
  fi
else
  info "torch already installed"
fi

# ---------- 7. requirements + unsloth ----------
info "Installing requirements..."
./venv/bin/pip install -r requirements.txt || { err "requirements install failed"; exit 1; }
info "Installing unsloth (training backend)..."
./venv/bin/pip install unsloth >/dev/null 2>&1 || warn "unsloth install failed — เทรนต้องใช้ unsloth (ดู README)"

# ---------- 8. Verify ----------
./venv/bin/python -c "import gradio, torch, transformers; print('gradio', gradio.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)" \
  || { err "import verify failed"; exit 1; }

info "✅ Install complete!"
echo
info "Run:  $APP_DIR/venv/bin/python $APP_DIR/app.py"
info "Open: http://localhost:7860"
echo

if [ "$RUN_AFTER" = "1" ]; then
  info "Starting app..."
  cd "$APP_DIR"
  exec ./venv/bin/python app.py
fi
