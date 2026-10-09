# servetune

Measure how many GPUs a vLLM or SGLang deployment needs for a given traffic profile and latency target, and which engine flags change that number.

servetune starts the engine once per candidate configuration, drives it with the engine's own load generator at a ladder of request rates, and reports for each configuration the highest tested rate that met the SLO, the cost per million output tokens at that rate, and the replica count at your peak. A second tool reads each engine release and lists the changes likely to move those numbers, so you know when to rerun.

## Sample result

`runs/sample_h100_final/` contains a full run on one H100 80GB with vLLM 0.31.0 serving Qwen3-8B, using agent-style prompts of about 3,000 input tokens (2,000 of them shared) and 300 output tokens, against a p95 time to first token of 1.5 s and a p95 time per output token of 50 ms.

| Config | Requests/s within the SLO | Cost per 1M output tokens |
|---|---|---|
| vLLM defaults | 5.36 | $0.45 |
| `--kv-cache-dtype fp8` | 8.24 | $0.29 |
| FP8 checkpoint + FP8 KV cache + 16k prefill budget | 8.40 | $0.29 |
| `--max-num-batched-tokens 16384` | 5.35 | $0.45 |
| `--no-enable-prefix-caching` | 2.84 | $0.86 |

Costs use $2.69 per GPU-hour. One flag accounts for most of the gain, and turning prefix caching off nearly halves capacity for prompts that share a long prefix. The report in that folder lists the caveats, including an FP8-weights-only run that needs a rerun on a CUDA 12.9 image.

## Quick start

```bash
pip install vllm            # on the GPU host
python3 servetune/audit.py audit_sample.toml --out runs/sample
python3 servetune/analyze.py runs/sample      # writes report.md and report.html
```

`--dry-run` prints every server and benchmark command without running anything. `--only a,b` reruns a subset. Results checkpoint after each candidate.

## The audit file

```toml
baseline = "baseline"
engine    = { name = "vllm", model = "Qwen/Qwen3-8B", baseline_flags = "--max-model-len 16384" }
hardware  = { gpu = "H100 80GB", usd_per_gpu_hour = 3.28, gpus_per_replica = 1 }
workload  = { input_len = 3000, output_len = 300, shared_prefix_len = 2000, request_rates = [2, 4, 8, 16, 0] }
slo       = { ttft_p95_ms = 1500, tpot_p95_ms = 50 }
client    = { peak_requests_per_s = 40 }
candidates = [
  { name = "baseline", flags = "" },
  { name = "fp8_kv", flags = "--kv-cache-dtype fp8" },
  { name = "fp8_model", model = "Qwen/Qwen3-8B-FP8" },
]
```

See `audit_sample.toml` for a complete example. A rate of 0 sends every request at once to find saturation throughput.

## Replay your own traffic

Most audits should replay the client's real traffic shape rather than synthetic prompts. `servetune/replay.py` takes request metadata only (`arrival_s,input_tokens,output_tokens,prefix_id`, no prompt content), synthesizes prompts of those lengths with the model's tokenizer, and drives any OpenAI-compatible endpoint. One run works on vLLM, SGLang or a staging gateway alike. Playing the trace faster (`speeds = [1, 2, 4, 8]`) finds the highest rate that still meets the SLO.

```toml
[engine]
name = "external"            # or "vllm" / "sglang" to launch the engine per candidate
model = "Qwen/Qwen3-8B"
[workload]
mode = "replay"
trace = "trace.csv"
speeds = [1, 2, 4, 8]
shared_prefix_len = 2000
[[candidates]]
name = "baseline"
base_url = "http://staging-a:8000"
```

`examples/demo_mock.sh` runs the whole pipeline against two fake engines on a laptop, with no GPU.

## Cache probe for agent traffic

`servetune/cache_probe.py` checks whether multi-turn agent conversations keep their prefix cache on a vLLM endpoint. It renders each turn of a scripted conversation through the render endpoints, which return prompt token IDs without generating, and reports how many tokens of the previous prompt the next turn repeats. On vLLM 0.29 with Qwen3 it found full reuse for plain chat, tool calls and Anthropic-style requests, 4 percent reuse with a timestamp at the top of the system prompt, and re-prefilled replies for reasoning output. Results from that run are in `findings/`.

## Release diff

```bash
python3 servetune/release_diff.py --repo vllm-project/vllm --since 2026-07-01 --out notes/
```

Add `--hardware hopper --models qwen,llama` to drop lines about other hardware and boost lines naming your models. Writes one note per release with the lines that touch defaults, scheduling, prefix caching, KV offload, speculative decoding, quantization, attention kernels, CUDA graphs and parallelism, and flags the ones that change a default on a hot path or claim a gain of 10 percent or more. Needs the `gh` CLI.

## Definitions and limits

servetune reports capacity as the highest measured passing rate. The true value lies between that rate and the next one tested. Synthetic prompts come from the engine's random dataset, which defeats speculative decoding, so test that on a replayed trace. Quantization can change model output. Check any FP8 or FP4 change on your own evaluation set before rollout.

## Paid audits

I run fixed-fee audits with this tool for teams that would rather not do it themselves: rishabhsinha.dev/audit.

## License

Apache-2.0
