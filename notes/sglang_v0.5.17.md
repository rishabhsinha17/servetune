# sgl-project/sglang v0.5.17 (2026-08-08): what to re-tune

188 performance or default-related lines; 3 likely to move a common config by more than 10%.

## Re-benchmark triggers

- [defaults, cuda_graphs] Turn on breakable prefill cuda graph for dp attention by default: [#31682](https://github.com/sgl-project/sglang/pull/31682)
- [moe_parallel, hardware] DWDP for MoE prefill**: A new prefill parallelism strategy that prefetches peer expert weights over NVLink P2P and computes all experts locally, removing EP all-to-all token dispatch. On 4x B200 with gpt-oss-120b, prefill-only, DWDP4 reaches **1.92x over DEP4** at MNT 32K / ISL 32K, and **506K vs 329K tok/s (1.54x)** at saturation (CONC=128, ISL=8K). Enable with `--dwdp-size`; the authors mark it 
- [moe_parallel] Faster large-MoE model loading**: Oversized or non-contiguous CPU weight views were driving pathological H2D transfers, with DeepSeek-V4-Pro TP8 spending 27 to 32 minutes in H2D on some ranks. Copying those views into contiguous storage before H2D cuts full model loading from about **35 minutes to 6m20s (5.6x)**, and GPT-OSS-20B BF16 from **545s to 70s (7.8x)**, with Qwen3.5-397B measured at 1.93x

## Touch count by knob

- quantization: 42
- moe_parallel: 32
- attention_kernels: 30
- kv_offload: 22
- spec_decode: 21
- other_perf: 21
- cuda_graphs: 16
- hardware: 14
- disagg: 13
- scheduler: 10
- prefix_cache: 8
- defaults: 1
- memory: 1

## Other notable lines

- [quantization, cuda_graphs] Faster engine recovery**: Large-model restarts cost 3 to 6+ minutes today, about 6.5 minutes for Qwen3-235B FP8 on 4 GPUs, because weights reload from storage and CUDA graphs recapture. A weight-cache daemon holds weights per GPU so a restarting engine can recover from cache instead ([#27139](https://github.com/sgl-project/sglang/pull/27139)).
- [scheduler, spec_decode, cuda_graphs] Lower host overhead in hybrid-linear MTP decode**: Under spec-v2 overlap scheduling each decode step runs draft, verify and extend CUDA graphs, and the eager seams between them become GPU idle time at low concurrency. This trims that host work so the host stays off the critical path ([#32219](https://github.com/sgl-project/sglang/pull/32219)).
- [moe_parallel, memory] [NPU] Fix LLaDA2 MoE OOM after the FRACTAL_NZ cast, re-enabling the NZ speedup: [#31772](https://github.com/sgl-project/sglang/pull/31772)
- [scheduler, prefix_cache, kv_offload, spec_decode, quantization] Kimi K3 day-0 support**: A 2.8T-parameter multimodal LatentMoE (896 experts, top-16, routed in a 3584-dim latent space) with a 1M-token context, 69 KDA linear-attention layers interleaved with 24 MLA layers, and a MoonViT3d vision tower, shipping as a native MXFP4 checkpoint. SGLang serves it from day 0 with DCP, DSpark speculative decoding, chunked-prefill PP with TP decode, KDA-aware prefix cach
- [quantization, attention_kernels, hardware] DCP communication backends and q-replicate (Helix)**: The DeepSeek-MLA decode context-parallel path gains pluggable comm backends. `a2a` exchanges packed attention output plus fp32 LSE in a single NCCL collective per layer, with fp8 KV carried as uint8 byte transport; `fi_a2a` delegates the cross-rank exchange to the FlashInfer MNNVL kernel on GB200. `--dcp-replicate-q-proj` projects full-head Q l
- [quantization, attention_kernels] SM90 FP8 MegaMoE for DeepSeek-V4**: Adds the DeepGEMM MegaMoE A2A path on SM90 for DeepSeek-V4-Flash/Pro FP8, including the pre-dispatch JIT kernel and FP8 expert weight preparation. Guarded behind `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1` ([#29016](https://github.com/sgl-project/sglang/pull/29016)).
- [quantization, moe_parallel] Lower DeepSeek-V4 memory on AMD**: Removing unnecessary expert padding drops MI355X FP4 MoE model weights from **159.07 GB to 112.36 GB**, and bringing the HIP compress-state pool into the memory_saver KV_CACHE region lets colocated RL reclaim it, cutting the measured training-phase footprint from about **143 GiB to 87 GiB per GPU** ([#31450](https://github.com/sgl-project/sglang/pull/31450), [#31
- [kv_offload, attention_kernels] Dependencies**: flashinfer 0.6.15.post1 ([#31927](https://github.com/sgl-project/sglang/pull/31927)), sgl-deep-gemm 0.1.5.post1 ([#32345](https://github.com/sgl-project/sglang/pull/32345), [#33143](https://github.com/sgl-project/sglang/pull/33143)), helion 1.4 ([#32562](https://github.com/sgl-project/sglang/pull/32562)), mooncake 0.3.12.post1 ([#32302](https://github.com/sgl-project/sglang/pull/32
- [spec_decode] [MTP] Cut spec-v2 host-seam overhead in hybrid-linear MTP decode: [#32219](https://github.com/sgl-project/sglang/pull/32219) ⭐
- [scheduler, spec_decode] [Spec] Enable grammar overlap scheduling for STANDALONE speculative decoding: [#32110](https://github.com/sgl-project/sglang/pull/32110)
- [spec_decode, attention_kernels] [Spec] Add `trtllm_mha` support for Gemma 4 MTP draft attention backend: [#25545](https://github.com/sgl-project/sglang/pull/25545)
- [spec_decode] [Perf] Fast-path chain-style draft token organization in multi-layer EAGLE: [#32887](https://github.com/sgl-project/sglang/pull/32887)
