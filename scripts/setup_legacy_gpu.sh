#!/usr/bin/env bash
# setup_legacy_gpu.sh — Enable CUDA on GTX 960 / sm_52 (Maxwell) and other
# legacy GPUs where torch>=2.7 dropped sm_50 support.
#
# Evidence (this box): GTX 960 sm_52 2GB driver 580 CUDA 13, nvcc 12.9
#   torch 2.13+cu130 arch=['sm_75',...] → "no kernel image is available"
#   torch 2.6+cu124 arch=['sm_50',...] → sm_50 covers 5.2 → tensor([0.], device='cuda:0') ok, PPO 95 fps (n_envs 1) 297 fps (n_envs 4)
#   torch 2.7/2.8+cu128 arch sm_75+ → fails on sm_52
# Last sm_52 wheel: 2.6.0 (cu124/cu121). 2.6 supports numpy 2.x; 2.2 needs numpy<2.
#
# Usage:
#   ./scripts/setup_legacy_gpu.sh           # (now default) installs/ensures torch==2.6.0+cu124 for sm_52
#   ./scripts/setup_legacy_gpu.sh --check   # only check compatibility (torch arch vs GPU cap)
#   ./scripts/setup_legacy_gpu.sh --revert  # modern GPU: upgrade to latest torch (cu130, sm_75+)
#
# After setup (uv.lock now 2.6), just run:
#   uv run train.py --device cuda --window null --sessions 1 --steps 256 --n-envs 4
#   uv run play.py --device cuda --window SDL2 --model models/tetris_ppo_model
# (auto still picks cpu on <4GB; use --device cuda to force)

set -euo pipefail

CHECK_ONLY=0
REVERT=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1;;
    --revert) REVERT=1;;
    -h|--help) echo "Usage: $0 [--check|--revert]"; echo "  --check  only report GPU + torch arch"; echo "  --revert install latest torch from lock"; exit 0;;
    *) echo "unknown arg $arg" >&2; exit 1;;
  esac
done

echo "=== GPU check ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv 2>&1 | head -5 || echo "nvidia-smi not found"
echo "---"
uv run python -c "import torch; print(f'torch {torch.__version__} cuda {torch.version.cuda}'); print(f'arch {torch.cuda.get_arch_list() if hasattr(torch.cuda,\"get_arch_list\") else \"n/a\"}'); print(f'cuda_available {torch.cuda.is_available()}'); print(f'cap {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"n/a\"}')" 2>&1 | grep -v UserWarning | grep -v "warnings.warn" | head -20

if [[ $CHECK_ONLY -eq 1 ]]; then
  echo "---"; echo "Check via tensor alloc:"
  uv run python -c "import torch; x=torch.zeros(1, device='cuda'); print('cuda alloc ok', x.device)" 2>&1 | tail -5
  exit 0
fi

if [[ $REVERT -eq 1 ]]; then
  echo "=== Modern GPU: upgrade to latest torch (cu130, sm_75+) ==="
  echo "Note: uv.lock is pinned to 2.6 for GTX 960; upgrading needs --no-sync until lock bumped."
  uv pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -20
  uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list()[:3])" 2>&1 | tail -5
  echo "Done. Use 'uv run --no-sync train.py --device cuda' to keep latest, or 'uv sync --extra dev' to revert to 2.6."
  exit 0
fi

echo "=== Ensuring torch 2.6.0+cu124 for sm_52 (GTX 960) — uv.lock now defaults to 2.6 ==="
echo "If you previously upgraded to cu130, this re-installs 2.6. ~400MiB VRAM, sb3 2.8.0 + numpy 2.x compatible."
echo "Requires: driver >=550 (580 ok), CUDA 12.x runtime. Disk ~2GB."

uv pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -20

echo "--- verifying ---"
uv run python -c "
import torch
print(f'torch {torch.__version__} cuda {torch.version.cuda}')
print(f'arch {torch.cuda.get_arch_list()}')
print(f'cap {torch.cuda.get_device_capability(0)}')
x=torch.zeros(1, device='cuda')
print('cuda alloc ok', x.device, 'mem', torch.cuda.mem_get_info(0))
" 2>&1 | grep -v UserWarning | tail -20

echo ""
echo "Done. Test GPU training (uv.lock 2.6 → no --no-sync needed):"
echo "  uv run train.py --device cuda --window null --sessions 1 --steps 256 --n-envs 4 2>&1 | tail -20"
echo "  uv run train.py --device cuda --window null --sessions 2 --runs 4 --steps 2048 --n-envs 4 --checkpoint-freq 1 --model-name /tmp/gpu_smoke"
echo ""
echo "Note: --device auto still picks cpu on <4GB; use --device cuda to force. n_envs 4 gives ~3× fps (95→297) within 400MiB."
echo "Modern GPU upgrade: $0 --revert (then use --no-sync until lock bump)"
