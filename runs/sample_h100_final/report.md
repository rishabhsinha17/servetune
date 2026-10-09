# Sample inference cost audit: Qwen3-8B, agent-style traffic, 1x H100

**Headline.** Of the tested configurations, `fp8_model_kv_16k` serves 8.40 requests/s within the SLO against 5.36 for the current setup, which cuts cost per million output tokens from $0.45 to $0.29 (-36%). At the stated peak of 40 requests/s that is 5 replicas instead of 8, about $5,891 a month.

## Setup

- Engine: vllm 0.31.0, model `Qwen/Qwen3-8B`, baseline flags `--max-model-len 16384 --gpu-memory-utilization 0.90`
- Hardware: 1 x H100 80GB per replica at $2.69/GPU-hour
- Workload: ~3000 input / ~300 output tokens (range ratio 0.5), 2000 shared prefix tokens, 200 requests per rate, rates [3, 4.5, 5.5, 6.5, 8, 10, 0]
- SLO: p95 TTFT <= 1500 ms and p95 TPOT <= 50 ms

## Results, cheapest first

| Config | Capacity at SLO (req/s) | vs current | Output tok/s at capacity | $ per 1M output tok | vs current | p95 TTFT / TPOT at capacity (ms) |
|---|---|---|---|---|---|---|
| `fp8_model_kv_16k` | 8.40 | +57% | 2,583 | $0.289 | -36% | 140 / 13.1 |
| `fp8_kv` | 8.24 | +54% | 2,533 | $0.295 | -35% | 125 / 13.8 |
| `baseline` | 5.36 | +0% | 1,649 | $0.453 | +0% | 564 / 43.3 |
| `batched_16k` | 5.35 | -0% | 1,644 | $0.455 | +0% | 579 / 44.0 |
| `no_prefix_cache` | 2.84 | -47% | 873 | $0.856 | +89% | 436 / 23.0 |
| `fp8_model` | none passed | n/a | n/a | n/a | n/a | see notes |

## What each config changes

- `baseline`: `(baseline)`. vLLM defaults (prefix caching and chunked prefill are already on in the V1 engine).
- `no_prefix_cache`: `--no-enable-prefix-caching`. Shows the cost of running with prefix caching off, a setting that still appears in older configs and some gateways.
- `batched_16k`: `--max-num-batched-tokens 16384`. Larger prefill budget per step. It trades decode smoothness for prefill throughput.
- `fp8_kv`: `--kv-cache-dtype fp8`. Halves KV cache memory, so more concurrent sequences fit.
- `fp8_model`: `model Qwen/Qwen3-8B-FP8`. Official FP8 checkpoint, with lower weight memory and faster GEMMs on Hopper.
- `fp8_model_kv_16k`: `model Qwen/Qwen3-8B-FP8, --kv-cache-dtype fp8 --max-num-batched-tokens 16384`. All three changes above together.

## Method and limits

servetune reports capacity as the highest tested request rate that met both SLOs with every request completed. True capacity lies between that rate and the next one tested. Synthetic prompts come from the engine's own random dataset with the stated length ranges and shared prefix. In a client audit, servetune replays the client's own traffic instead. Quality effects of quantization are outside this run's scope, so check any FP8 change on your own eval set before rollout.

- Most of the gain comes from one flag, `--kv-cache-dtype fp8`. Adding the FP8 checkpoint and the larger prefill budget on top of it gains about 2 percent more.
- Treat the fp8_model row as an anomaly to rerun. With DeepGEMM disabled, the CUTLASS block-scaled fallback decoded at about 100 ms per token at every rate, while the same FP8 weights with an FP8 KV cache decoded at 11 to 13 ms. That points to the kernel path in this image, so rerun it on a CUDA 12.9 image before drawing any conclusion about FP8 weights.
- At 5.5 requests/s, the first rate measured for fp8_model_kv_16k, the run included warm-up compilation and shows a 2.1 s p95 TTFT. Every higher rate passed, and the capacity figure comes from those.
- Capacity counts delivered requests per second over a finite 200-request run, including the drain at the end. That understates sustained capacity by a similar margin for every config, so compare the configs with each other rather than reading the absolute figure.
- This run used one RunPod H100 80GB SXM (driver 580, vLLM 0.31.0, CUDA 12.8 image) on 2026-10-08. Both FP8-checkpoint runs set VLLM_USE_DEEP_GEMM=0 because DeepGEMM needs CUDA 12.9, and covered rates 5.5 to 10 only.
- Random synthetic tokens defeat n-gram and draft models, so this run leaves speculative decoding out. A client audit tests it on replayed traffic.
