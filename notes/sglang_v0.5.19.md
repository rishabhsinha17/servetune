# sgl-project/sglang v0.5.19 (2026-09-05): what to re-tune

169 performance or default-related lines; 32 likely to move a common config by more than 10%.

## Re-benchmark triggers

- [defaults, prefix_cache, kv_offload, disagg] Unified radix tree by default.** The unified tree is now the cache for every model, not just hybrid ones ([#35081](https://github.com/sgl-project/sglang/pull/35081), see Breaking Changes). It also picked up three things this cycle: PD decode workers can reuse cached prefixes for SWA hybrid models like gpt-oss ([#27770](https://github.com/sgl-project/sglang/pull/27770)), you can attach or detach L3
- [quantization, attention_kernels, moe_parallel, hardware] W4A8 MoE on Hopper.** If you serve MXFP4 experts on Hopper, you can now quantize the activations to FP8 as well with `--flashinfer-mxfp4-moe-precision fp8`. DeepSeek-V4-Flash gains about 12% output throughput with no change in GSM8K accuracy. Needs FlashInfer 0.6.18 ([#34967](https://github.com/sgl-project/sglang/pull/34967)).
- [spec_decode, hardware] Faster speculative kernels.** DSA prefill top-k moves to the v2 kernel, 1.3 to 1.8 times faster on B200 ([#35175](https://github.com/sgl-project/sglang/pull/35175)). KDA models get an opt-in fused accept path, `SGLANG_OPT_KDA_FUSED_ACCEPT_STATE=1`, that cuts MTP verify-and-commit time by 45% to 63% on Kimi-Linear shapes with bit-identical output ([#33722](https://github.com/sgl-project/sglang/pull
- [disagg, hardware] DSA models on ROCm.** Disaggregated GLM-5.2 serving now uses the fused top-k seed remap, which brings decode TPOT from 23 ms down to 8 ms on eight MI355Xs ([#36714](https://github.com/sgl-project/sglang/pull/36714)). DeepSeek-V4 gets the v2 top-k kernel, up to three times faster ([#36684](https://github.com/sgl-project/sglang/pull/36684)), and a shared-experts gate fix gives GLM-5.2 up to 16% bett
- [spec_decode, moe_parallel] [Spec] Add LFM2 and LFM2-MoE DSpark speculative decoding support (1.05 to 2.42x faster decoding on LFM2.5 targets, 1xH100): [#31041](https://github.com/sgl-project/sglang/pull/31041)
- [quantization, attention_kernels, moe_parallel] [MoE] Add FlashInfer SM90 MXFP4 W4A8 CUTLASS MoE (W4A8 kernel 1.63 to 2.08x faster than W4A16, +11.7% output throughput on DeepSeek-V4-Flash): [#34967](https://github.com/sgl-project/sglang/pull/34967) ⭐
- [quantization, moe_parallel] feat(humming): FP8 DeepEP dispatch for humming MoE backend (TPOT -35% and TTFT -23% vs BF16 dispatch on H20): [#31429](https://github.com/sgl-project/sglang/pull/31429)
- [cuda_graphs, hardware] [Perf] PaddleOCR-VL: overlap page preprocessing, pack the ViT, enable prefill CUDA graph (1080p page TTFT 235 ms to 120 ms, 1.84x req/s on H200): [#35318](https://github.com/sgl-project/sglang/pull/35318)
- [quantization, moe_parallel] [Kernel] Add GB300 Triton MoE configs for GLM-4.5 FP8 (decode throughput +21.9% on GB300): [#37159](https://github.com/sgl-project/sglang/pull/37159)
- [disagg, hardware] [AMD][Spec][PD] Enable the PD DSA fused-TopK seed remap on ROCm (GLM-5.2 PD decode TPOT 23.16 ms to 7.94 ms on 8xMI355X): [#36714](https://github.com/sgl-project/sglang/pull/36714)
- [cuda_graphs] [Performance] Reduce idle DP work in breakable prefill CUDA graphs (median TTFT -18.9%, QPS +12.9%): [#33871](https://github.com/sgl-project/sglang/pull/33871)
- [attention_kernels] Add fmha_v2 attention backend for SM90/120 (about 15% faster than FA3 at the kernel level): [#23112](https://github.com/sgl-project/sglang/pull/23112)
- [moe_parallel] [DeepSeek-V4] Enable shared-experts fusion on the flashinfer_mxfp4 (trtllm-gen) MoE path (TTFT -13% to -21%, warmed GSM8K e2e +5-7%): [#35505](https://github.com/sgl-project/sglang/pull/35505)
- [moe_parallel] [Perf] Tune the W4AFP8 DeepEP low-latency requant launch geometry (requant kernel up to 10.6x faster at 256 experts): [#35760](https://github.com/sgl-project/sglang/pull/35760)
- [quantization] [Diffusion] Optimizing MiniMax-H3 for consumer-level GPUs: INT8 Linear + pluggable DiT attention backends (MiniMax-H3 on one 24 GB GPU, denoise 1.34x faster with exact attention and up to 2.48x with approximate attention): [#34581](https://github.com/sgl-project/sglang/pull/34581)

## Touch count by knob

- quantization: 43
- moe_parallel: 32
- hardware: 32
- attention_kernels: 22
- other_perf: 20
- kv_offload: 17
- disagg: 15
- spec_decode: 15
- cuda_graphs: 15
- prefix_cache: 8
- scheduler: 6
- defaults: 2
- memory: 1

## Other notable lines

- [defaults, hardware] DCP on the default Blackwell MLA backend.** Decode context parallelism now runs on `trtllm_mla`, not just CuTe DSL and Tokenspeed. It pays off at long context: at 128K input, plain TP stops scaling around 680 tokens per second on eight B200s, while DCP keeps going as concurrency grows ([#33926](https://github.com/sgl-project/sglang/pull/33926)).
- [disagg, moe_parallel] perf: overlap Qwen shared expert with DeepEP routed experts (+2.1% output throughput on 24xGB300 disaggregated Qwen3.5-397B): [#34938](https://github.com/sgl-project/sglang/pull/34938)
- [quantization, attention_kernels] [FlashInfer 0.6.18] add FlashInfer CuTe DSL NVFP4 W4A16 mode (Qwen3-30B-A3B W4A16 GSM8K 0.965 to 0.980, 1,662 to 1,858 tok/s): [#35120](https://github.com/sgl-project/sglang/pull/35120)
- [quantization, moe_parallel, hardware] [Kernel] Add H200 MoE configs for Qwen3.5 and Qwen3.6 (+3.8% throughput on Qwen3.5 FP8 and +7.1% on Qwen3.6 FP8, H200): [#35374](https://github.com/sgl-project/sglang/pull/35374)
- [quantization, cuda_graphs, moe_parallel] DeepEP v2.** DeepEP's new ElasticBuffer engine is available as `--moe-a2a-backend deepep_v2` for DeepSeek-V3/V4 and Qwen3-MoE in FP8. Its buffers have a fixed size, so decode can run under CUDA graphs even across nodes. Performance is on par with the classic backend ([#35634](https://github.com/sgl-project/sglang/pull/35634), [#34923](https://github.com/sgl-project/sglang/pull/34923)).
- [attention_kernels] [Performance] Tune FlashInfer EXTEND for DP prefill (DP prefill throughput +4.9% to +6.8%): [#36219](https://github.com/sgl-project/sglang/pull/36219)
- [quantization] [Kernel] Fix SM90 FP8 decode regression with benchmarked M/K/N routing (restores about 215 tok/s decode from 190 after the SM90 FP8 routing regression): [#37018](https://github.com/sgl-project/sglang/pull/37018)
- [spec_decode] [AMD] Optimize Qwen3.5 MTP unified attention on gfx950 (+9.6% throughput at concurrency 8 with 70K input): [#36330](https://github.com/sgl-project/sglang/pull/36330)
- [kv_offload, quantization] Qwen3.5 MXFP4 on MI355X with an FP8 KV cache or a HiCache host-memory tier: [#35445](https://github.com/sgl-project/sglang/pull/35445), [#36245](https://github.com/sgl-project/sglang/pull/36245)
- [kv_offload, disagg, spec_decode] Beam search.** SGLang can now do beam search. Pass `beam_width` in your request and you get back the `n` best sequences instead of a single sample. It works out of the box next to regular requests, though it does not yet mix with speculative decoding, disaggregation, DP attention, or HiCache ([#31626](https://github.com/sgl-project/sglang/pull/31626)).
- [moe_parallel, hardware] LayerNorm sequence parallelism.** With `--enable-layernorm-sp`, each tensor-parallel rank normalizes only its own share of the prefill tokens instead of all of them. That takes 3.5% off Qwen3-8B prefill on H100 and 5.6% on B200, and the saving grows with the TP degree. Dense Qwen3 models only for now ([#30915](https://github.com/sgl-project/sglang/pull/30915)).
- [kv_offload, attention_kernels, hardware] Dependencies.** FlashInfer moves to 0.6.18 ([#36954](https://github.com/sgl-project/sglang/pull/36954)), sgl-deep-ep to 0.1.2 ([#35450](https://github.com/sgl-project/sglang/pull/35450)), sgl-deep-gemm to 0.1.7 ([#37279](https://github.com/sgl-project/sglang/pull/37279)), and mooncake to 0.3.13 ([#36493](https://github.com/sgl-project/sglang/pull/36493)). There is a new CUDA 13.4 preview image for
