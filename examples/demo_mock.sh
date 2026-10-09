#!/bin/bash
# End-to-end demo with no GPU: two fake engines, a synthetic trace, replay, report.
# Needs: pip install aiohttp
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; OUT="${OUT:-$HERE/runs/demo_mock}"; mkdir -p "$OUT"
python3 "$HERE/servetune/mock_engine.py" --port 8011 --max-seqs 64 --step-tokens 4096 & A=$!
python3 "$HERE/servetune/mock_engine.py" --port 8012 --max-seqs 24 --step-tokens 2048 & B=$!
trap 'kill $A $B 2>/dev/null' EXIT
sleep 2
python3 "$HERE/servetune/replay.py" --make-trace "$OUT/trace.csv" --rate 3 --duration 40 --input-mean 1500 --output-mean 120 --prefix-groups 2
cat > "$OUT/audit.toml" <<TOML
baseline = "baseline"
[engine]
name = "external"
model = "mock"
[hardware]
gpu = "H100 80GB"
usd_per_gpu_hour = 3.28
[workload]
mode = "replay"
trace = "$OUT/trace.csv"
speeds = [4, 16, 32, 48]
shared_prefix_len = 800
[slo]
ttft_p95_ms = 1000
tpot_p95_ms = 60
[client]
peak_requests_per_s = 40
[[candidates]]
name = "baseline"
base_url = "http://127.0.0.1:8012"
why = "Fake engine with a small batch and prefill budget."
[[candidates]]
name = "bigger_batch"
base_url = "http://127.0.0.1:8011"
why = "Fake engine with a larger batch and prefill budget."
TOML
python3 "$HERE/servetune/audit.py" "$OUT/audit.toml" --out "$OUT/run"
python3 "$HERE/servetune/analyze.py" "$OUT/run"
echo "report: $OUT/run/report.html"
