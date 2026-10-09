# sgl-project/sglang v0.5.18 (2026-08-22): what to re-tune

152 performance or default-related lines; 3 likely to move a common config by more than 10%.

## Re-benchmark triggers

- [defaults, moe_parallel] Enable MoE deferred finalize by default and drop its expert_weights dtype workaround: [#33618](https://github.com/sgl-project/sglang/pull/33618)
- [cuda_graphs, hardware] Overlapped checkpoint staging at startup**: Checkpoint pages now stage from storage while CUDA graphs capture. Qwen3-32B on H100 starts **8.6-11.7% faster** than serial with prefetch, and **2.38x faster (35.6s vs 84.8s)** than the plain default. Opt in with `--startup-weight-load-mode overlap` ([#32017](https://github.com/sgl-project/sglang/pull/32017)).
- [perf] Kimi K3 tuned for MI355X**: a grouped-head MLA verify kernel replaces the MHA-shaped split-KV path that re-read the shared latent once per head, for **1.37-1.77x throughput and 1.45-2.42x ITL** at concurrency 2-32; the AITER MLA prefill kernel now accepts K3's 12-head shape (TTFT up to -14.9%); a gfx950-tuned decode stage-1 geometry adds **46-73% ITL** at 68k input, opt-in via `SGLANG_MLA_DECODE_T

## Touch count by knob

- moe_parallel: 27
- quantization: 26
- attention_kernels: 20
- hardware: 17
- kv_offload: 17
- cuda_graphs: 16
- spec_decode: 16
- other_perf: 12
- scheduler: 11
- disagg: 10
- prefix_cache: 7
- defaults: 3

## Other notable lines

- [defaults] [DSV4] Turn on mhc post pre fusion by default: [#35214](https://github.com/sgl-project/sglang/pull/35214)
- [defaults] Enable unified cache out-of-window slot freeing by default: [#34653](https://github.com/sgl-project/sglang/pull/34653)
- [hardware] TP LMHead with All-to-All**: The TP LMHead's allgather + scatter becomes a single all-to-all for pure-DP dp-attention. On DeepSeek-V4-Pro B200 decode, LMHead time drops **320us to 169us** and TPOT improves 36.97ms to 35.67ms ([#32313](https://github.com/sgl-project/sglang/pull/32313)).
- [attention_kernels, moe_parallel, hardware] Dependencies**: torch 2.13.0 with triton 3.7.1 ([#28836](https://github.com/sgl-project/sglang/pull/28836)), flashinfer 0.6.17 ([#33997](https://github.com/sgl-project/sglang/pull/33997)), CuTeDSL 4.6.2, fixing an FA4 startup regression on Blackwell ([#34372](https://github.com/sgl-project/sglang/pull/34372)), DeepEP now installed from released `sgl-deep-ep` wheels ([#33932](https://github.com/sgl
- [moe_parallel, hardware] perf: add H200 Triton MoE configs for E256 N512: [#34150](https://github.com/sgl-project/sglang/pull/34150)
- [attention_kernels, moe_parallel] [AMD][Perf] Fuse GatedDeltaNet QKVZBA split/reshape/cat into a single Triton kernel for Qwen3.5-architecture MoE on HIP: [#34421](https://github.com/sgl-project/sglang/pull/34421)
- [attention_kernels, hardware] FlashInfer MNNVL for pure allreduce**: Non-fused allreduce sites now reuse the FlashInfer MNNVL workspace instead of falling back to NCCL. DeepSeek-V4-Flash TP4 decode on Blackwell gains **up to +6.9% at small batches**. Auto-enabled for DeepSeek-V3/V3.2/V4; elsewhere `--enable-flashinfer-pure-allreduce` ([#30700](https://github.com/sgl-project/sglang/pull/30700)).
- [spec_decode] [Spec] Reduce host-side overhead in ngram draft prep: [#35207](https://github.com/sgl-project/sglang/pull/35207)
- [cuda_graphs] [trtllm_mha] perf: Stop allocating per-layer scratch inside the decode CUDA graph: [#33063](https://github.com/sgl-project/sglang/pull/33063)
- [quantization, attention_kernels] add flashinfer cute-dsl backend for mxfp8 gemm: [#34042](https://github.com/sgl-project/sglang/pull/34042)
- [quantization, attention_kernels, moe_parallel] [FlashInfer v0.6.16] Support FlashInfer CuTe DSL NVFP4 MoE quantization: [#28354](https://github.com/sgl-project/sglang/pull/28354)
- [moe_parallel] [Fix] Support 128-aligned hidden sizes in the W4AFP8 DeepEP low-latency requant kernel: [#35593](https://github.com/sgl-project/sglang/pull/35593)
