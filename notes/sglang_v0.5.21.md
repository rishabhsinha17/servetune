# sgl-project/sglang v0.5.21 (2026-10-02): what to re-tune

250 performance or default-related lines; 19 likely to move a common config by more than 10%.

## Re-benchmark triggers

- [defaults, attention_kernels, moe_parallel] [MoE] Disable FlashInfer fused finalize by default for numerical accuracy: [#40105](https://github.com/sgl-project/sglang/pull/40105)
- [defaults, disagg] [PD] Enable deferred decode-side KV release by default: [#41023](https://github.com/sgl-project/sglang/pull/41023)
- [spec_decode, quantization] feat(kv-cache): support SM100 NVFP4 GenMHA and speculative decoding (1.355x faster decode attention kernel at 1M context): [#36340](https://github.com/sgl-project/sglang/pull/36340)
- [disagg, hardware] [PD] Bound cached-prefix DCP transfers by pack capacity (cached-prefix throughput about 10.9x higher at concurrency 8 on Kimi Linear, 8x B200): [#40376](https://github.com/sgl-project/sglang/pull/40376)
- [prefix_cache, quantization] Add Agentic-Aware Tail-Optimized LRU eviction to the unified radix cache (opt-in; p99 ITL down 43.9% at concurrency 32 on DeepSeek-V4-Pro FP4, 4x B300, agentic trace): [#34012](https://github.com/sgl-project/sglang/pull/34012)
- [kv_offload, quantization] GLM-5.2: MI355X MXFP4 recipes move to HIP Top-K and newer images, and the high-throughput cell adds HiCache: [#40148](https://github.com/sgl-project/sglang/pull/40148), [#40570](https://github.com/sgl-project/sglang/pull/40570), [#41109](https://github.com/sgl-project/sglang/pull/41109), [#41597](https://github.com/sgl-project/sglang/pull/41597)
- [disagg] Kimi K3 gets 20.6% higher prefill throughput in PD serving ([#40045](https://github.com/sgl-project/sglang/pull/40045)).
- [spec_decode] [LFM2-VL] Add DSpark speculative decoding (1.66x to 2.56x speedup at batch 1): [#40651](https://github.com/sgl-project/sglang/pull/40651)
- [spec_decode] Reduce decode bootstrap latency with request-owned speculative KV (mean TTFT 19.4 to 15.1 s on DeepSeek-V4-Pro 1P1D, MI355X, concurrency 256): [#38978](https://github.com/sgl-project/sglang/pull/38978)
- [kv_offload] [Unified Cache][9/N] add opt-in MLA load deduplication for Mooncake Linker (TTFT 3.85 to 2.23 s with a 256K cached prefix, GLM-5.2 on 8x H20): [#39565](https://github.com/sgl-project/sglang/pull/39565)
- [hardware] [Perf] Mamba2 selective_state_update up to 2x faster on B200 via 8x1 launch config for dstate 128 (+7.5% Nemotron-3-Super serving): [#41223](https://github.com/sgl-project/sglang/pull/41223)
- [quantization] [AMD] Enable GLM DSA prefill top-k to the v2 kernel (+4.9% token throughput at 70K input on GLM-5.2-MXFP4, 4x MI355X): [#37889](https://github.com/sgl-project/sglang/pull/37889)
- [hardware] [AMD] Prefer HIP Top-K for GLM-5.x on ROCm (P90 interactivity 96.97 to 106.69 tok/s/user on MI355X at concurrency 8): [#39631](https://github.com/sgl-project/sglang/pull/39631)
- [perf] DeepSeek-V4.1 gets 22% faster first token on long prompts ([#40352](https://github.com/sgl-project/sglang/pull/40352)).
- [perf] [DSV4][BCG] Optimize the heavy memory use of C4 Indexer when BCG is enabled (capture memory down 58% to 18 GB): [#36534](https://github.com/sgl-project/sglang/pull/36534)

## Touch count by knob

- quantization: 59
- kv_offload: 47
- hardware: 44
- moe_parallel: 34
- disagg: 33
- spec_decode: 28
- other_perf: 19
- prefix_cache: 18
- attention_kernels: 15
- cuda_graphs: 15
- scheduler: 9
- defaults: 2

## Other notable lines

- [moe_parallel, hardware] [Perf] Optimize w4a8 MoE for glm5.2 on H200: [#38220](https://github.com/sgl-project/sglang/pull/38220)
- [kv_offload, disagg] [PD] Enable optimistic prefill with buffer-only L3 write-through HiCache: [#40043](https://github.com/sgl-project/sglang/pull/40043)
- [hardware] [Diffusion] Fuse lossless SenseNova RoPE for 5% faster H200 inference: [#40374](https://github.com/sgl-project/sglang/pull/40374)
- [quantization, hardware] [ROCm][Perf] aiter: page-level KV view for gfx950 fp8 page-64 asm prefill: [#36505](https://github.com/sgl-project/sglang/pull/36505)
- [attention_kernels, moe_parallel] FlashInfer MoE fused finalize is off by default** for numerical accuracy. This costs some W4A4 MoE latency; set `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=1` to restore the fused atomic finalize: [#40105](https://github.com/sgl-project/sglang/pull/40105)
- [spec_decode, quantization, moe_parallel] GLM-5.3-Flash now runs on AMD MI355X, with FP8 / MXFP4 MoE and MTP speculative decoding ([#38546](https://github.com/sgl-project/sglang/pull/38546), [#39341](https://github.com/sgl-project/sglang/pull/39341), [#39778](https://github.com/sgl-project/sglang/pull/39778)).
- [quantization, moe_parallel] Fix int4 MoE tuner config filename: [#35260](https://github.com/sgl-project/sglang/pull/35260)
- [quantization, attention_kernels, moe_parallel] [MegaMoE] Wire Qwen MoE blocks to DeepGEMM MegaMoE (MXFP4 and NVFP4 experts): [#38080](https://github.com/sgl-project/sglang/pull/38080)
- [attention_kernels, moe_parallel] [Fix] Guard FlashInfer CUTLASS MoE against 0-token inputs: [#38780](https://github.com/sgl-project/sglang/pull/38780)
- [quantization, moe_parallel] [Kernel] Add H20 block-FP8 MoE configs for GLM-5.3-Flash EP4/EP8: [#38913](https://github.com/sgl-project/sglang/pull/38913)
- [quantization, attention_kernels, moe_parallel] Accept MXFP8 dispatch in FlashInfer A2A TRT-LLM MoE: [#39613](https://github.com/sgl-project/sglang/pull/39613)
- [quantization, moe_parallel] Support MXFP8 and deferred route weighting in DeepEP v2: [#40030](https://github.com/sgl-project/sglang/pull/40030)
