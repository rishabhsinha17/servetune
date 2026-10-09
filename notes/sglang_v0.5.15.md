# sgl-project/sglang v0.5.15 (2026-07-10): what to re-tune

103 performance or default-related lines; 7 likely to move a common config by more than 10%.

## Re-benchmark triggers

- [defaults, cuda_graphs] Breakable CUDA Graph on by default**: Breakable CUDA Graph is now the default capture path, reducing per-step kernel-launch overhead ([#29458](https://github.com/sgl-project/sglang/pull/29458)); full CUDA Graph support for the prefill phase lands as experimental ([#27988](https://github.com/sgl-project/sglang/pull/27988)).
- [defaults, attention_kernels] FlashMLA sparse prefill is now enabled by default for DeepSeek-V4, reaching >10% throghput gain on long context. ([#29775](https://github.com/sgl-project/sglang/pull/29775)).
- [defaults, attention_kernels] [DeepSeek V4] Enable FlashMLA sparse prefill by default: [#29775](https://github.com/sgl-project/sglang/pull/29775) ⭐
- [defaults, cuda_graphs] Disable dsr1 prefill cudagraphs by default: [#28053](https://github.com/sgl-project/sglang/pull/28053)
- [defaults, cuda_graphs] [diffusion] feat: performance_mode=speed enables torch.compile by default: [#30016](https://github.com/sgl-project/sglang/pull/30016)
- [quantization, hardware] GLM-5.2 NVFP4, tuned for production**: We took time this cycle to tune GLM-5.2 NVFP4 on Blackwell for optimized production serving. It now runs at **500+ tok/s/user on 8x B300, 450 on 4x GB300** (bs=1). Run GLM-5.2 with our [cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2).
- [scheduler, cuda_graphs] Spec V2 by default**: zero-overhead scheduling via CUDA-graphable DSA draft-extend, dropped D2H/H2D syncs, fused metadata ops. +11% end-to-end TPS ([#29413](https://github.com/sgl-project/sglang/pull/29413), [#29343](https://github.com/sgl-project/sglang/pull/29343), [#29166](https://github.com/sgl-project/sglang/pull/29166), [#29077](https://github.com/sgl-project/sglang/pull/29077)).

## Touch count by knob

- quantization: 22
- other_perf: 22
- attention_kernels: 15
- hardware: 14
- cuda_graphs: 13
- spec_decode: 12
- moe_parallel: 12
- kv_offload: 7
- defaults: 6
- scheduler: 4
- disagg: 4
- prefix_cache: 1

## Other notable lines

- [defaults] [Cherry-pick to release/v0.5.15] [DeepSeek-V4] Enable non-paged indexer by default for large prefill chunks (#30140): [#30436](https://github.com/sgl-project/sglang/pull/30436)
- [attention_kernels, moe_parallel, hardware] [MoE] Consolidate ungrouped + grouped gate/topk onto one Triton router (#26771) — faster than AOT on B200/H100/H200, at parity with flashinfer: [#29771](https://github.com/sgl-project/sglang/pull/29771)
- [quantization, attention_kernels, moe_parallel] [Perf][Kernel] Fuse SiLU+Mul into NVFP4 Expert Quantization for CUTLASS MoE: [#18612](https://github.com/sgl-project/sglang/pull/18612)
- [perf] Indexer prologue fusion**: 12 kernels to 4, ~8% faster decode at bs=1 ([#27705](https://github.com/sgl-project/sglang/pull/27705)).
- [attention_kernels, moe_parallel] FlashInfer A2A for routed MoE**: Adds FlashInfer all-to-all with the `flashinfer_trtllm_routed` MoE runner ([#22394](https://github.com/sgl-project/sglang/pull/22394)).
- [perf] Non paged indexer support for long context prefill, with >5% e2e throughput gain.([#29619](https://github.com/sgl-project/sglang/pull/29619)).
- [spec_decode, quantization, hardware] [CI] Add GLM52 NVFP4 MTP B200 tests: [#30021](https://github.com/sgl-project/sglang/pull/30021)
- [quantization, hardware] docs: add B200 NVFP4 recipes + benchmarks to GLM-5.2 cookbook: [#29674](https://github.com/sgl-project/sglang/pull/29674)
- [spec_decode, quantization] [AMD] [GLM5] GLM-5.1 MXFP4 (MI355X) + enable EAGLE for gfx950 in cookbook: [#29194](https://github.com/sgl-project/sglang/pull/29194)
- [quantization, attention_kernels] [AMD] [GLM5] Add opt-in Triton fp8 sparse-MLA prefill kernel for gfx950: [#28975](https://github.com/sgl-project/sglang/pull/28975)
- [quantization] [Cherry-pick to release/v0.5.15] [DSV4] perf: Make FP8 quant output tensor contiguous (#27926): [#30449](https://github.com/sgl-project/sglang/pull/30449)
- [attention_kernels, hardware] [AMD][DeepSeek V4] Fix default FlashMLA sparse prefill off on ROCm/HIP: [#29982](https://github.com/sgl-project/sglang/pull/29982)
