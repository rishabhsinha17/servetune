#!/bin/bash
# One-command sample audit on a rented H100. You run this (it spends your RunPod credit, about $3-6).
#   export RUNPOD_API_KEY=...        # key of your choice
#   bash run_sample.sh               # creates pod, runs the sweep, copies results back, terminates pod
# A local watchdog terminates the pod after WATCHDOG_MIN minutes even if this script dies.
# create_first.py tries H100 SXM community, then secure, then NVL and PCIe, because community H100s sell out.
# Lessons from the 2026-10-08 run: the runpod/pytorch CUDA 12.8 image lacks NVCC 12.9, so FP8 checkpoints need
# VLLM_USE_DEEP_GEMM=0 (set per candidate in audit_sample.toml); expect about 80 minutes and $5-6 on secure cloud.
set -euo pipefail
: "${RUNPOD_API_KEY:?export RUNPOD_API_KEY first}"
HERE="$(cd "$(dirname "$0")" && pwd)"; B="$(dirname "$HERE")"
GPU="${GPU:-NVIDIA H100 80GB HBM3}"; CLOUD="${CLOUD:-SECURE}"; WATCHDOG_MIN="${WATCHDOG_MIN:-95}"
IMAGE="${IMAGE:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04}"
RUN="$B/runs/sample_$(date +%Y%m%d_%H%M)"; mkdir -p "$RUN" "$HERE/logs"
KEY="$HERE/logs/pod_key"; [ -f "$KEY" ] || ssh-keygen -t ed25519 -N "" -f "$KEY" -q
SSHO=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30)

POD=$(cd "$HERE" && python3 create_first.py "$KEY.pub" | python3 -c "import json,sys;t=sys.stdin.read().strip();print(json.loads(t)['id'] if t else '')")
[ -n "$POD" ] || { echo "pod create failed; see $HERE/logs/rp_calls.jsonl"; exit 1; }
echo "pod $POD"
(nohup bash "$HERE/watchdog.sh" "$POD" "$WATCHDOG_MIN" > "$HERE/logs/watchdog_$POD.log" 2>&1 &)
sleep 2; pgrep -f "watchdog.sh $POD" >/dev/null && echo "watchdog armed ($WATCHDOG_MIN min)"
cleanup() { python3 "$HERE/rp.py" terminate "$POD" >/dev/null || true; pkill -f "watchdog.sh $POD" || true; echo "pod $POD terminated"; }
trap cleanup EXIT

for i in $(seq 1 60); do
  read -r IP PORT < <(python3 "$HERE/rp.py" status "$POD" | python3 -c "
import json,sys
d=json.load(sys.stdin); p=((d.get('data') or {}).get('pod') or {}).get('runtime') or {}
for x in p.get('ports') or []:
    if x.get('privatePort')==22 and x.get('isIpPublic'): print(x['ip'], x['publicPort']); break
else: print('', '')") || true
  [ -n "${IP:-}" ] && break; sleep 10
done
[ -n "${IP:-}" ] || { echo "no ssh port after 10 min"; exit 1; }
echo "ssh root@$IP -p $PORT"
for i in $(seq 1 30); do ssh "${SSHO[@]}" -p "$PORT" root@"$IP" true 2>/dev/null && break; sleep 10; done

scp "${SSHO[@]}" -P "$PORT" -r "$B/servetune" "$B/audit_sample.toml" root@"$IP":/root/
ssh "${SSHO[@]}" -p "$PORT" root@"$IP" bash -lc "'
set -e
pip install -q --upgrade pip >/dev/null
pip install -q vllm 2>&1 | tail -2
vllm --version
cd /root && python3 servetune/audit.py audit_sample.toml --out runs/sample 2>&1 | tee runs_sample.log
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > runs/sample/gpu.csv
pip show vllm | head -2 > runs/sample/vllm_version.txt
'"
scp "${SSHO[@]}" -P "$PORT" -r root@"$IP":/root/runs/sample/. "$RUN/"
python3 "$B/servetune/analyze.py" "$RUN"
echo "report: $RUN/report.html"
