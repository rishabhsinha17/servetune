# sgl-project/sglang v0.5.16 (2026-07-25): what to re-tune

91 performance or default-related lines; 4 likely to move a common config by more than 10%.

## Re-benchmark triggers

- [defaults, quantization] UnifiedRadixTree is now the default** for SWA, Mamba and DSA models. Replay SSM and Mamba int8 checkpoints are synced onto it, and a cache hit now resets only the state it used ([#30468](https://github.com/sgl-project/sglang/pull/30468), [#30636](https://github.com/sgl-project/sglang/pull/30636), [#30626](https://github.com/sgl-project/sglang/pull/30626), [#31643](https://github.com/sgl-project/sg
- [spec_decode, quantization, moe_parallel, hardware] Inkling support**: A 975B-parameter multimodal MoE with a 1M-token context. It mixes sliding-window, full and Mamba2 linear attention, and adds an NVFP4 MoE, optional vision/audio towers and native MTP. On Blackwell it reaches up to **71.7k tok/s input** and **171.0 tok/s per-user decode**. Verified on Blackwell TP4/TP8, H200 and AMD MI350X / MI355X ([#31681](https://github.com/sgl-project/sglang/
- [spec_decode, attention_kernels, hardware] Linear attention on Blackwell (SM100)**: The first correct KDA MTP path. Its `recurrent_kda` decode kernel runs at **29.6 us vs 36.8 us** for Triton (ncu, B=64). The full decode path reaches parity by B=128 and **1.35x at B=256**, and is slower below that ([#30113](https://github.com/sgl-project/sglang/pull/30113)). Separately, GDN/KDA CuteDSL prefill fuses state I/O into the chunk-h kernel ([#301
- [spec_decode] ReplaySSM Ring Spec-Verify (GDN)**: Drops the per-draft SSM snapshot. Speculative scratch goes from **11.5 GB to 1.8 GB per GPU (6.4x smaller)** on Qwen3.5-35B-A3B at TP1, at accuracy and throughput parity. Opt in with `--enable-gdn-replayssm-spec` (default off; GDN with a linear draft chain only, `--speculative-eagle-topk` in {None, 1}), and tune the ring via `--linear-replayssm-cache-len` ([#286

## Touch count by knob

- quantization: 23
- spec_decode: 16
- attention_kernels: 14
- moe_parallel: 11
- cuda_graphs: 10
- hardware: 9
- kv_offload: 6
- other_perf: 5
- scheduler: 5
- disagg: 5
- prefix_cache: 5
- defaults: 4

## Other notable lines

- [defaults] `UnifiedRadixTree` is now the default** for SWA, Mamba and DSA models. A behavior change on those architectures: [#30468](https://github.com/sgl-project/sglang/pull/30468)
- [defaults] [DeepSeek-V4] Enable non-paged indexer by default for large prefill chunks: [#30140](https://github.com/sgl-project/sglang/pull/30140)
- [defaults] [Fix] Enable chunked input-logprob processing by default to cap peak memory: [#31498](https://github.com/sgl-project/sglang/pull/31498)
- [spec_decode] DSpark: confidence-driven speculative decoding**: A new speculative algorithm. It drafts semi-autoregressively in blocks, then sizes each verify window from the draft's own confidence instead of a fixed draft length. Reaches **383.7 tok/s at accept length ~5** on DeepSeek-V4-Pro, TP8 on B300 (bs=1). Enable with `--speculative-algorithm DSPARK` and `SGLANG_RAGGED_VERIFY_MODE=compact`; tune the bloc
- [attention_kernels, moe_parallel] [GLM5][MoE] perf: Write FlashInfer TRT-LLM MoE output directly: [#28416](https://github.com/sgl-project/sglang/pull/28416)
- [quantization, attention_kernels] QServe and FBGEMM FP8 quantization are removed**: the experimental QServe (QoQ) W4A8 and FBGEMM FP8 paths are gone. `--fp4-gemm-backend cutlass` goes too, along with the in-tree NVFP4 JIT kernels, so NVFP4 GEMM now requires FlashInfer ([#31109](https://github.com/sgl-project/sglang/pull/31109), [#30448](https://github.com/sgl-project/sglang/pull/30448)).
- [quantization, attention_kernels] CUTLASS FP8 blockwise deleted for SM90 / SM100**, SM120 moved to JIT: [#30438](https://github.com/sgl-project/sglang/pull/30438)
- [quantization, attention_kernels] `--fp4-gemm-backend cutlass` is removed** along with the in-tree NVFP4 JIT kernels, so NVFP4 GEMM now requires FlashInfer. Use `auto`, which picks `flashinfer_cutedsl` on SM100 and `flashinfer_cutlass` on SM120: [#30448](https://github.com/sgl-project/sglang/pull/30448)
- [quantization, cuda_graphs] Temperature-0 nondeterminism under DP attention with breakable prefill CUDA graph.** On the DSV4-Flash FP4 recipe, the idle-rank dummy extend introduced by [#30898](https://github.com/sgl-project/sglang/pull/30898) perturbs real requests' logits, so identical temperature-0 requests can diverge. The guarding determinism test is disabled as a stopgap rather than fixed ([#31125](https://github.com/sg
- [spec_decode, cuda_graphs] [Spec] fix inkling multi layer mtp draft extend cuda graph: [#32254](https://github.com/sgl-project/sglang/pull/32254) (cherry-picked as [#32260](https://github.com/sgl-project/sglang/pull/32260))
- [disagg, spec_decode, cuda_graphs] [Fix] Stabilize GLM-5.2 MTP IndexShare across PD and CUDA graph replay: [#30839](https://github.com/sgl-project/sglang/pull/30839)
- [scheduler, spec_decode] [DSV4] Remove per-step seqlen D2H from speculative to make overlap scheduler work: [#30365](https://github.com/sgl-project/sglang/pull/30365)
