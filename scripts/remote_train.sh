#!/usr/bin/env bash
# Run the pinned overfit protocol ON the remote GPU machine.
#
# The bundle is extracted at the repository root, producing:
#   outputs/remote_bundle_payload/{MANIFEST.json,requirements.txt,configs/...}
#   data/lerobot/train_v1/...
# Both paths are git-ignored, so the checkout can stay clean.
#
# Every step fails loudly. The only network actions are the pip install printed
# below and the Hugging Face base-model downloads (unless already cached); no
# uploads, accounts, W&B, or MLflow are used.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUNDLE_DIR="outputs/remote_bundle_payload"
CONFIG="$BUNDLE_DIR/configs/training/smolvla_lora_overfit_remote.yaml"
OUTPUT_DIR="outputs/remote/overfit_remote"
VENV=".venv-remote"
PYTHON="python3"
MAX_STEPS=""
MIN_MEMORY_MIB=8192
MIN_MEDIAN_SM_MHZ=1000

usage() {
  cat <<'EOF'
Usage: scripts/remote_train.sh [options]
  --bundle-dir DIR   extracted payload (default outputs/remote_bundle_payload)
  --config FILE      training config (default from the bundle)
  --output-dir DIR   checkpoints, logs and RUN_REPORT.json (default outputs/remote/overfit_remote)
  --venv DIR         virtual environment (default .venv-remote)
  --python PATH      interpreter used to create the venv (default python3)
  --max-steps N      override config training.max_steps
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

log() {
  echo "==> $*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir) BUNDLE_DIR="${2:?}"; shift 2 ;;
    --config) CONFIG="${2:?}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    --venv) VENV="${2:?}"; shift 2 ;;
    --python) PYTHON="${2:?}"; shift 2 ;;
    --max-steps) MAX_STEPS="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument $1" ;;
  esac
done

MANIFEST="$BUNDLE_DIR/MANIFEST.json"
[[ -f "$MANIFEST" ]] || die "manifest not found: $MANIFEST (extract the bundle at the repo root)"
[[ -f "$CONFIG" ]] || die "config not found: $CONFIG"
command -v git >/dev/null || die "git is required"
command -v nvidia-smi >/dev/null || die "nvidia-smi is not available"

log "verifying MANIFEST hashes and config bytes"
python3 - "$MANIFEST" "$ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
root = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("git_dirty") is not False:
    raise SystemExit("manifest git_dirty is not false")
files = manifest.get("files")
if not isinstance(files, dict) or not files:
    raise SystemExit("manifest has no files")
for relative, entry in sorted(files.items()):
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"missing bundled file: {relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry.get("sha256"):
        raise SystemExit(f"sha256 mismatch: {relative}")
    if path.stat().st_size != int(entry.get("size", -1)):
        raise SystemExit(f"size mismatch: {relative}")
print(f"MANIFEST verified: {len(files)} files")
PY

COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_commit"])' "$MANIFEST")"
HEAD="$(git rev-parse HEAD)"
[[ "$HEAD" == "$COMMIT" ]] || die "HEAD $HEAD != manifest commit $COMMIT"
[[ -z "$(git status --porcelain)" ]] || die "working tree is dirty; clone the public repo at $COMMIT and re-extract the bundle"

log "code commit $COMMIT is checked out clean"
cat <<EOF
Network actions (printed before they happen):
  1. pip install -r $BUNDLE_DIR/requirements.txt
     indexes: PyPI plus https://download.pytorch.org/whl/cu126
  2. Hugging Face downloads unless cached:
     lerobot/smolvla_base @ 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6
     HuggingFaceTB/SmolVLM2-500M-Video-Instruct @ 7b375e1b73b11138ff12fe22c8f2822d8fe03467
No upload, Hub push, W&B, or MLflow traffic is used.
EOF

if [[ ! -x "$VENV/bin/python" ]]; then
  log "creating $VENV from pinned requirements"
  "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --disable-pip-version-check -r "$BUNDLE_DIR/requirements.txt"

log "GPU matmul test and clock/power sample"
"$VENV/bin/python" - "$MIN_MEMORY_MIB" "$MIN_MEDIAN_SM_MHZ" <<'PY'
import json
import statistics
import subprocess
import sys
import time

import torch

min_memory_mib = int(sys.argv[1])
min_clock = float(sys.argv[2])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available: the remote GPU check failed")
name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
total_mib = props.total_memory / (1024**2)
torch.manual_seed(0)
a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
for _ in range(3):
    _ = a @ b
torch.cuda.synchronize()
started = time.perf_counter()
clocks = []
powers = []
temps = []
reasons = set()
deadline = time.time() + 10.0
while time.time() < deadline:
    for _ in range(4):
        c = a @ b
    torch.cuda.synchronize()
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=clocks.sm,power.draw,temperature.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    parts = [part.strip() for part in query.split(",")]
    clocks.append(float(parts[0]))
    powers.append(float(parts[1]))
    temps.append(float(parts[2]))
    perf = subprocess.check_output(["nvidia-smi", "-q", "-d", "PERFORMANCE"], text=True)
    for line in perf.splitlines():
        if "SW Power Cap" in line and "Active" in line:
            reasons.add("SW Power Cap")
        if "HW Slowdown" in line and "Active" in line:
            reasons.add("HW Slowdown")
        if "SW Thermal Slowdown" in line and "Active" in line:
            reasons.add("SW Thermal Slowdown")
elapsed = time.perf_counter() - started
checksum = float(c.sum().item())
report = {
    "gpu": name,
    "total_memory_mib": round(total_mib, 1),
    "checksum": checksum,
    "matmul_seconds": round(elapsed, 3),
    "median_sm_clock_mhz": statistics.median(clocks),
    "min_sm_clock_mhz": min(clocks),
    "max_sm_clock_mhz": max(clocks),
    "median_power_w": statistics.median(powers),
    "max_power_w": max(powers),
    "max_temperature_c": max(temps),
    "throttle_reasons": sorted(reasons),
}
print(json.dumps(report, indent=2))
if total_mib < min_memory_mib:
    raise SystemExit(f"GPU has {total_mib:.0f} MiB; decision rule requires >= {min_memory_mib} MiB")
if statistics.median(clocks) < min_clock:
    raise SystemExit(
        f"median SM clock {statistics.median(clocks):.0f} MHz is below the "
        f"{min_clock:.0f} MHz floor; this GPU is capped/degraded"
    )
PY

mkdir -p "$OUTPUT_DIR"
GPU_LOG="$OUTPUT_DIR/gpu_monitor.csv"
TRAIN_LOG="$OUTPUT_DIR/train.log"
nvidia-smi \
  --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > "$GPU_LOG" 2>&1 &
GPU_PID=$!
cleanup() {
  kill "$GPU_PID" 2>/dev/null || true
}
trap cleanup EXIT

log "training: $CONFIG -> $OUTPUT_DIR"
SECONDS=0
set +e
PYTHONPATH=src "$VENV/bin/python" scripts/train_vla.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  ${MAX_STEPS:+--max-steps "$MAX_STEPS"} 2>&1 | tee "$TRAIN_LOG"
TRAIN_RC=${PIPESTATUS[0]}
set -e
WALL_SECONDS=$SECONDS
kill "$GPU_PID" 2>/dev/null || true
trap - EXIT
[[ "$TRAIN_RC" -eq 0 ]] || die "train_vla.py exited with $TRAIN_RC"
[[ -f "$OUTPUT_DIR/losses.jsonl" ]] || die "training wrote no losses.jsonl"

log "writing RUN_REPORT.json"
PYTHONPATH=src "$VENV/bin/python" - "$OUTPUT_DIR" "$MANIFEST" "$GPU_LOG" "$WALL_SECONDS" <<'PY'
import json
import statistics
import sys
from pathlib import Path

from sim2real.training.checkpoint import adapter_tree_sha256, file_sha256

out = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
gpu_log = Path(sys.argv[3])
wall_seconds = float(sys.argv[4])
losses = [json.loads(line) for line in (out / "losses.jsonl").read_text().splitlines() if line.strip()]
if not losses:
    raise SystemExit("losses.jsonl is empty")
step_times = [float(row["step_seconds"]) for row in losses if "step_seconds" in row]
last20 = statistics.mean(float(row["loss"]) for row in losses[-20:])
summary = {}
text = (out / "train.log").read_text(encoding="utf-8")
index = text.rfind('{\n  "run_kind"')
if index >= 0:
    summary = json.loads(text[index:])

rows = []
for line in gpu_log.read_text(encoding="utf-8").splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 6:
        continue
    try:
        rows.append(
            {
                "clock_mhz": float(parts[1]),
                "power_w": float(parts[2]),
                "temperature_c": float(parts[3]),
                "utilization_pct": float(parts[4]),
                "memory_used_mib": float(parts[5]),
            }
        )
    except ValueError:
        continue

checkpoints = []
for path in sorted(out.glob("checkpoint-*")):
    metadata_path = path / "checkpoint_metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"{path}: checkpoint_metadata.json is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoints.append(
        {
            "step": int(metadata.get("step", path.name.split("-")[-1])),
            "path": str(path),
            "name": path.name,
            "metadata_sha256": file_sha256(metadata_path),
            "adapter_sha256": adapter_tree_sha256(path / "adapter"),
            "metadata_adapter_hash": metadata.get("adapter_hash"),
            "git_commit": metadata.get("git_commit"),
            "git_dirty": metadata.get("git_dirty"),
            "config_hash": metadata.get("config_hash"),
            "stats_file_hash": metadata.get("stats_file_hash"),
        }
    )
if not checkpoints:
    raise SystemExit("no checkpoints were written")

report = {
    "schema_version": 1,
    "run_name": out.name,
    "git_commit": manifest["git_commit"],
    "config": str(summary.get("config", "")),
    "config_hash": summary.get("config_hash"),
    "max_steps_requested": summary.get("final_step"),
    "steps": int(summary.get("final_step", len(losses))),
    "wall_time_s": wall_seconds,
    "final_loss": float(losses[-1]["loss"]),
    "last20_mean_loss": float(last20),
    "convergence_rule": "mean training loss of the last 20 steps < 0.1",
    "converged": bool(last20 < 0.1),
    "step_time_s": {
        "median": statistics.median(step_times) if step_times else None,
        "mean": statistics.mean(step_times) if step_times else None,
    },
    "peak_vram_mib": summary.get("peak_vram_mib"),
    "peak_rss_mib": summary.get("peak_rss_mib"),
    "gpu": {
        "samples": len(rows),
        "median_sm_clock_mhz": statistics.median([row["clock_mhz"] for row in rows]) if rows else None,
        "max_power_w": max((row["power_w"] for row in rows), default=None),
        "max_temperature_c": max((row["temperature_c"] for row in rows), default=None),
        "max_memory_used_mib": max((row["memory_used_mib"] for row in rows), default=None),
    },
    "checkpoints": checkpoints,
}
(out / "RUN_REPORT.json").write_text(
    json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, indent=2))
PY

log "done: $OUTPUT_DIR/RUN_REPORT.json"
