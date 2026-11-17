#!/usr/bin/env bash
# Run the pinned overfit protocol ON the remote GPU machine.
#
# The archive extracts at the repository root as:
#   data/lerobot/train_v1/...
#   outputs/remote_bundle_payload/{MANIFEST.json,requirements.txt,repo.bundle,remote_train.sh,configs/...}
# Both data/ and outputs/ are git-ignored, so a clean checkout stays clean.
#
# --check-only verifies the interpreter, manifest hashes, commit/tree and GPU
# identity without creating a venv, installing, downloading or training.
# --from-bundle bootstraps a clone from repo.bundle when GitHub is unavailable.
set -euo pipefail

ORIG_PWD="$(pwd)"

BUNDLE_DIR="outputs/remote_bundle_payload"
CONFIG=""
OUTPUT_DIR="outputs/remote/overfit_remote"
VENV=".venv-remote"
PYTHON="python3"
MAX_STEPS=""
CHECK_ONLY=0
FROM_BUNDLE=""
WORKDIR=""
MIN_MEMORY_MIB=8192
MIN_MEDIAN_SM_MHZ=1000
ALLOW_LOW_CLOCK=0

usage() {
  cat <<'EOF'
Usage: scripts/remote_train.sh [options]
  --bundle-dir DIR    extracted payload (default outputs/remote_bundle_payload)
  --config FILE       training config (default from the bundle)
  --output-dir DIR    checkpoints, logs and RUN_REPORT.json
  --venv DIR          virtual environment (default .venv-remote)
  --python PATH       interpreter used to create the venv (default python3)
  --max-steps N       override config training.max_steps
  --check-only        verify and print preflight, then exit before install/train
  --allow-low-clock   continue when the median SM clock is below the floor,
                      print a loud WARNING and record overrides: ["low_clock"]
                      in RUN_REPORT.json (the memory floor is never overridable)
  --from-bundle FILE  clone/verify repository from repo.bundle
  --workdir DIR       clone target for --from-bundle (default a unique /tmp path)
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

log() {
  echo "==> $*"
}

absolute_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$ORIG_PWD/$1" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir) BUNDLE_DIR="${2:?}"; shift 2 ;;
    --config) CONFIG="${2:?}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    --venv) VENV="${2:?}"; shift 2 ;;
    --python) PYTHON="${2:?}"; shift 2 ;;
    --max-steps) MAX_STEPS="${2:?}"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --allow-low-clock) ALLOW_LOW_CLOCK=1; shift ;;
    --from-bundle) FROM_BUNDLE="${2:?}"; shift 2 ;;
    --workdir) WORKDIR="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument $1" ;;
  esac
done

BUNDLE_DIR="$(absolute_path "$BUNDLE_DIR")"
OUTPUT_DIR="$(absolute_path "$OUTPUT_DIR")"
[[ -n "$CONFIG" ]] || CONFIG="$BUNDLE_DIR/configs/training/smolvla_lora_overfit_remote.yaml"
CONFIG="$(absolute_path "$CONFIG")"
MANIFEST="$BUNDLE_DIR/MANIFEST.json"
[[ -f "$MANIFEST" ]] || die "manifest not found: $MANIFEST (extract the bundle at the repo root)"
[[ -f "$CONFIG" ]] || die "config not found: $CONFIG"
command -v git >/dev/null || die "git is required"

MANIFEST_COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_commit"])' "$MANIFEST")"

# ---------------------------------------------------------------------------
# Offline bootstrap: if there is no usable checkout at the manifest commit and
# a repo.bundle exists, clone it locally. This is a local operation only.
# ---------------------------------------------------------------------------
CURRENT_ROOT="$(git -C "$ORIG_PWD" rev-parse --show-toplevel 2>/dev/null || true)"
CURRENT_HEAD=""
if [[ -n "$CURRENT_ROOT" ]]; then
  CURRENT_HEAD="$(git -C "$CURRENT_ROOT" rev-parse HEAD)"
fi
if [[ -z "$FROM_BUNDLE" && -z "$CURRENT_ROOT" && -f "$BUNDLE_DIR/repo.bundle" ]]; then
  FROM_BUNDLE="$BUNDLE_DIR/repo.bundle"
  log "no checkout found; using the bundle's repo.bundle automatically"
fi

if [[ -n "$FROM_BUNDLE" && ( -z "$CURRENT_ROOT" || "$CURRENT_HEAD" != "$MANIFEST_COMMIT" ) ]]; then
  FROM_BUNDLE="$(absolute_path "$FROM_BUNDLE")"
  [[ -f "$FROM_BUNDLE" ]] || die "repo.bundle not found: $FROM_BUNDLE"
  [[ -n "$WORKDIR" ]] || WORKDIR="${TMPDIR:-/tmp}/sim2real_remote_worktree_$$"
  WORKDIR="$(absolute_path "$WORKDIR")"
  [[ ! -e "$WORKDIR" ]] || die "workdir already exists: $WORKDIR (remove it or pass --workdir)"
  log "cloning repository from local bundle $FROM_BUNDLE -> $WORKDIR"
  git clone "$FROM_BUNDLE" "$WORKDIR" >/dev/null 2>&1 || die "git clone from repo.bundle failed"
  git -C "$WORKDIR" bundle verify "$FROM_BUNDLE" >/dev/null 2>&1 \
    || die "repo.bundle failed git bundle verify"
  if [[ -d "$ORIG_PWD/data/lerobot" ]]; then
    mkdir -p "$WORKDIR/data"
    ln -s "$ORIG_PWD/data/lerobot" "$WORKDIR/data/lerobot"
  fi
  if [[ -d "$ORIG_PWD/outputs/remote_bundle_payload" ]]; then
    mkdir -p "$WORKDIR/outputs"
    ln -s "$ORIG_PWD/outputs/remote_bundle_payload" "$WORKDIR/outputs/remote_bundle_payload"
  fi
  REARGS=(
    --bundle-dir "$BUNDLE_DIR"
    --config "$CONFIG"
    --output-dir "$OUTPUT_DIR"
    --venv "$VENV"
    --python "$PYTHON"
  )
  [[ "$CHECK_ONLY" -eq 1 ]] && REARGS+=(--check-only)
  [[ "$ALLOW_LOW_CLOCK" -eq 1 ]] && REARGS+=(--allow-low-clock)
  [[ -n "$MAX_STEPS" ]] && REARGS+=(--max-steps "$MAX_STEPS")
  cd "$WORKDIR"
  exec env SIM2REAL_BOOTSTRAPPED=1 "$WORKDIR/scripts/remote_train.sh" "${REARGS[@]}"
fi

if [[ -z "$CURRENT_ROOT" ]]; then
  die "not inside a git checkout; pass --from-bundle <repo.bundle> or clone the public repo"
fi
ROOT="$CURRENT_ROOT"
cd "$ROOT"

# ---------------------------------------------------------------------------
# Manifest, commit and interpreter preflight (no network, no install).
# ---------------------------------------------------------------------------
log "verifying MANIFEST hashes"
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

HEAD="$(git rev-parse HEAD)"
[[ "$HEAD" == "$MANIFEST_COMMIT" ]] || die "HEAD $HEAD != manifest commit $MANIFEST_COMMIT"
[[ -z "$(git status --porcelain)" ]] || die "working tree is dirty; use a clean checkout at $MANIFEST_COMMIT"
log "code commit $MANIFEST_COMMIT is checked out clean"

PY_VERSION="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" \
  || die "cannot run $PYTHON"
EXPECTED_PY_VERSION="$(python3 - "$MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
python = manifest.get("python") or {}
expected = python.get("major_minor")
if not expected:
    raise SystemExit(
        "manifest has no python.major_minor (old bundle); rebuild it with the "
        "current exporter so the interpreter pin can be checked"
    )
print(expected)
PY
)"
[[ "$PY_VERSION" == "$EXPECTED_PY_VERSION" ]] || die \
  "Python $PY_VERSION does not match manifest $EXPECTED_PY_VERSION. Use a matching interpreter (installing one is a download that must be approved), or rebuild the bundle with that interpreter."
log "Python $PY_VERSION matches the manifest"

LEROBOT_REQUIRES="$(python3 - "$MANIFEST" <<'PY'
import json
import sys

python = (json.load(open(sys.argv[1], encoding="utf-8")).get("python") or {})
print(python.get("lerobot_requires_python") or "")
PY
)"
if [[ -n "$LEROBOT_REQUIRES" ]]; then
  "$PYTHON" - "$LEROBOT_REQUIRES" <<'PY' || die "Python does not satisfy lerobot Requires-Python"
import re
import sys

spec = sys.argv[1]
version = (sys.version_info.major, sys.version_info.minor)
for clause in spec.split(","):
    clause = clause.strip()
    if not clause:
        continue
    match = re.fullmatch(r"(>=|<=|==|>|<)\s*(\d+)\.(\d+)(?:\.\d+)?", clause)
    if not match:
        print(f"UNVERIFIED: cannot parse Requires-Python clause {clause!r}")
        continue
    op, major, minor = match.group(1), int(match.group(2)), int(match.group(3))
    target = (major, minor)
    ok = {
        ">=": version >= target,
        "<=": version <= target,
        "==": version == target,
        ">": version > target,
        "<": version < target,
    }[op]
    if not ok:
        raise SystemExit(f"Python {version[0]}.{version[1]} violates {spec}")
print(f"lerobot Requires-Python {spec}: satisfied by Python {version[0]}.{version[1]}")
PY
else
  echo "UNVERIFIED: manifest has no lerobot Requires-Python"
fi

GPU_NAME=""
DRIVER_VERSION=""
CUDA_VERSION=""
GPU_MEMORY_MIB=""
if command -v nvidia-smi >/dev/null; then
  GPU_LINE="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -n 1)"
  IFS=',' read -r GPU_NAME DRIVER_VERSION GPU_MEMORY_MIB <<< "$GPU_LINE"
  GPU_NAME="$(echo "$GPU_NAME" | sed 's/^ *//;s/ *$//')"
  DRIVER_VERSION="$(echo "$DRIVER_VERSION" | sed 's/^ *//;s/ *$//')"
  GPU_MEMORY_MIB="$(echo "$GPU_MEMORY_MIB" | sed 's/^ *//;s/ *$//')"
  CUDA_VERSION="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -n 1)"
  echo "GPU: $GPU_NAME; driver $DRIVER_VERSION; nvidia-smi CUDA $CUDA_VERSION; memory ${GPU_MEMORY_MIB} MiB"
  echo "NOTE: the driver/CUDA pair above is measured, not a compatibility claim."
else
  echo "nvidia-smi not found"
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  log "CHECK ONLY: manifest, commit and interpreter verified; no install, download or training"
  exit 0
fi

[[ -n "$GPU_NAME" ]] || die "nvidia-smi is required for a real run"

# ---------------------------------------------------------------------------
# Full-run preflight: network, install, CUDA matmul.
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
PREFLIGHT="$OUTPUT_DIR/preflight.json"
NETWORK_JSON="$OUTPUT_DIR/network.json"
MATMUL_JSON="$OUTPUT_DIR/matmul.json"

log "network check (HEAD requests, 8 s timeout)"
if ! "$PYTHON" - "$NETWORK_JSON" <<'PY'
import json
import sys
import urllib.request

urls = [
    "https://pypi.org/simple/",
    "https://download.pytorch.org/whl/cu126/",
    "https://huggingface.co/lerobot/smolvla_base",
]
results = {}
for url in urls:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=8) as response:
            results[url] = {"ok": True, "status": int(response.status)}
    except Exception as exc:  # noqa: BLE001 - report every failure clearly
        results[url] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(results, indent=2))
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(results, indent=2) + "\n")
if not all(entry["ok"] for entry in results.values()):
    print(
        "Network check failed; notebooks and some VMs block outbound traffic. "
        "Allow PyPI, download.pytorch.org and huggingface.co before retrying.",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY
then
  die "required network endpoints are blocked; see the printed URLs"
fi

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

log "CUDA matmul test and clock/power sample"
if ! "$VENV/bin/python" - "$MIN_MEMORY_MIB" "$MIN_MEDIAN_SM_MHZ" "$DRIVER_VERSION" "$ALLOW_LOW_CLOCK" > "$MATMUL_JSON" <<'PY'
import json
import statistics
import subprocess
import sys
import time

import torch

min_memory_mib = int(sys.argv[1])
min_clock = float(sys.argv[2])
driver = sys.argv[3]
allow_low_clock = sys.argv[4] == "1"
build = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "driver": driver,
    "cuda_available": bool(torch.cuda.is_available()),
}
if not torch.cuda.is_available():
    print(json.dumps(build, indent=2))
    raise SystemExit(
        f"CUDA is not available for this torch build (torch {torch.__version__}, "
        f"torch CUDA {torch.version.cuda}, driver {driver})"
    )
name = torch.cuda.get_device_name(0)
total_mib = torch.cuda.get_device_properties(0).total_memory / (1024**2)
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
median_clock = statistics.median(clocks)
overrides = []
if allow_low_clock and median_clock < min_clock:
    overrides.append("low_clock")
report = {
    **build,
    "gpu": name,
    "total_memory_mib": round(total_mib, 1),
    "checksum": float(c.sum().item()),
    "matmul_seconds": round(elapsed, 3),
    "median_sm_clock_mhz": median_clock,
    "min_sm_clock_mhz": min(clocks),
    "max_sm_clock_mhz": max(clocks),
    "median_power_w": statistics.median(powers),
    "max_power_w": max(powers),
    "max_temperature_c": max(temps),
    "throttle_reasons": sorted(reasons),
    "overrides": overrides,
}
print(json.dumps(report, indent=2))
if total_mib < min_memory_mib:
    raise SystemExit(f"GPU has {total_mib:.0f} MiB; decision rule requires >= {min_memory_mib} MiB")
if median_clock < min_clock:
    if not allow_low_clock:
        raise SystemExit(
            f"median SM clock {median_clock:.0f} MHz is below the "
            f"{min_clock:.0f} MHz floor; this GPU is capped/degraded "
            f"(pass --allow-low-clock to record an explicit override)"
        )
    print(
        "\nWARNING: LOW CLOCK OVERRIDE\n"
        f"  median SM clock {median_clock:.0f} MHz is below the {min_clock:.0f} MHz floor\n"
        "  overrides=['low_clock'] will be recorded in RUN_REPORT.json\n"
        "  this run is NOT a healthy-GPU result\n",
        file=sys.stderr,
    )
PY
then
  cat "$MATMUL_JSON" >&2
  die "CUDA matmul test failed (driver $DRIVER_VERSION, nvidia-smi CUDA $CUDA_VERSION)"
fi
cat "$MATMUL_JSON"

"$PYTHON" - "$PREFLIGHT" "$MANIFEST" "$NETWORK_JSON" "$MATMUL_JSON" "$PY_VERSION" "$DRIVER_VERSION" "$CUDA_VERSION" "$GPU_NAME" "$GPU_MEMORY_MIB" "$BUNDLE_DIR/requirements.txt" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

preflight_path = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
manifest_path = Path(sys.argv[2])
requirements_path = Path(sys.argv[10])


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle_entry(path):
    if not path.is_file():
        return {"path": str(path), "sha256": None, "present": False}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "present": True,
    }

payload = {
    "check_only": False,
    "python": {
        "version": sys.argv[5],
        "manifest_major_minor": (manifest.get("python") or {}).get("major_minor"),
        "lerobot_requires_python": (manifest.get("python") or {}).get("lerobot_requires_python"),
    },
    "driver": {
        "version": sys.argv[6],
        "nvidia_smi_cuda": sys.argv[7],
        "gpu": sys.argv[8],
        "memory_total_mib": sys.argv[9],
    },
    "network": json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")),
    "matmul": json.loads(Path(sys.argv[4]).read_text(encoding="utf-8")),
    "bundle": {
        "manifest": bundle_entry(manifest_path),
        "requirements": bundle_entry(requirements_path),
    },
}
preflight_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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

preflight = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
overrides = list((preflight.get("matmul") or {}).get("overrides") or [])
report = {
    "schema_version": 1,
    "run_name": out.name,
    "git_commit": manifest["git_commit"],
    "config": str(summary.get("config", "")),
    "config_hash": summary.get("config_hash"),
    "final_step": summary.get("final_step"),
    "steps": int(summary.get("final_step", len(losses))),
    "wall_time_s": wall_seconds,
    "final_loss": float(losses[-1]["loss"]),
    "last20_mean_loss": float(last20),
    "convergence_rule": "mean training loss of the last 20 steps < 0.1 (advisory only)",
    "converged": bool(last20 < 0.1),
    "step_time_s": {
        "median": statistics.median(step_times) if step_times else None,
        "mean": statistics.mean(step_times) if step_times else None,
    },
    "peak_vram_mib": summary.get("peak_vram_mib"),
    "peak_rss_mib": summary.get("peak_rss_mib"),
    "overrides": overrides,
    "preflight": preflight,
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
if overrides:
    print(
        f"WARNING: this run used overrides {overrides}; it is NOT a healthy-GPU result",
        file=sys.stderr,
    )
PY

log "done: $OUTPUT_DIR/RUN_REPORT.json"
