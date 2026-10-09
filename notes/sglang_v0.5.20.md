# sgl-project/sglang v0.5.20 (2026-09-18): what to re-tune

238 performance or default-related lines; 37 likely to move a common config by more than 10%.

## Re-benchmark triggers

- [defaults, disagg] `/v1/responses` storage is off by default.** Start with `--enable-response-store` to keep retrieval, `previous_response_id` chaining, and background requests; PD deployments cannot enable it and reject those features at admission. `background=true` with `store=false` is rejected: [#39122](https://github.com/sgl-project/sglang/pull/39122)
- [defaults, attention_kernels] Diffusion defaults change**: video DiTs stay resident on GPUs with enough free memory unless `--dit-cpu-offload` is passed explicitly ([#38001](https://github.com/sgl-project/sglang/pull/38001)); the SDPA attention backend is chosen by measurement, retiring the no-op `allow_cudnn_sdp` ([#38689](https://github.com/sgl-project/sglang/pull/38689)); component loader capability switches are replaced by
- [prefix_cache, kv_offload] Unified radix tree.** Branching-point caching for the SWA component keeps the sliding-window state at the point where requests fork from a shared prefix, so branches reuse it instead of recomputing. On DeepSeek-V4-Flash with a shared system prompt, token hit rate rises from 43.8% to 60.8% and mean TTFT falls from 1.57 s to 1.07 s ([#34565](https://github.com/sgl-project/sglang/pull/34565)). An opt
- [scheduler, prefix_cache] SGLang Simulator.** A CPU-only simulator runs the real scheduler, radix cache, and hierarchical cache with a latency predictor in place of the model forward. Against measured serving traces it predicts TTFT within about 6% on most traces (up to 10% on the longest 32K to 128K ones) and prefix reuse within 0.05 percentage points, for cache and scheduling studies without GPUs ([#33824](https://github
- [quantization, attention_kernels, moe_parallel, hardware] DeepSeek-V4 on Blackwell.** TRT-LLM attention kernels now cover DeepSeek-V4's CSA and HCA layers on SM100 and SM103: about 1.2x faster prefill and 1.45x faster decode than FlashMLA at the kernel level on B200 ([#30805](https://github.com/sgl-project/sglang/pull/30805)). FlashInfer MegaMoE is available as `--moe-runner-backend flashinfer_megamoe`; on DeepSeek-V4-Flash NVFP4 at TP4/DP4 it adds up to
- [quantization, attention_kernels, moe_parallel] DeepSeek-V4 on RTX PRO 6000.** On SM120 the sparse-MLA indexer now runs on DeepGEMM's paged-MQA kernel and the DeepGEMM FP4 MoE backend is enabled, replacing the torch fallback that was the only working path. On 4x RTX PRO 6000, DeepSeek-V4-Flash decode TPOT drops from 36.1 to 10.5 ms at batch 1 and TTFT falls 20% from 8K to 128K input. Opt-in through the environment flags in the PR ([#29927](http
- [attention_kernels, hardware] [DSv4] Integrate TRT-LLM DSv4 Attention for SM100/103 (kernel-level prefill about 1.2x and decode about 1.45x faster than FlashMLA on B200): [#30805](https://github.com/sgl-project/sglang/pull/30805)
- [quantization, attention_kernels, moe_parallel] [NVIDIA] Support flashinfer Mega Moe (prefill input throughput up to +11.9% vs the trtllm runner on DeepSeek-V4-Flash NVFP4 at 8192 tokens per rank): [#31470](https://github.com/sgl-project/sglang/pull/31470)
- [disagg, hardware] [Intra-node PD][DSV4] Pack all layers into one batch for INTRA_NODE_NVLINK path (KV transfer latency 178 to 119 ms, E2E latency -20% on a 2P1D B200 node): [#38984](https://github.com/sgl-project/sglang/pull/38984)
- [cuda_graphs, moe_parallel] [LoRA] Support MoE in full and breakable prefill CUDA graphs (prefill throughput +39.5 to 40.9% with Triton LoRA on Qwen breakable graphs, GB300): [#38578](https://github.com/sgl-project/sglang/pull/38578)
- [quantization, attention_kernels, moe_parallel] [SM120] DeepSeek-V4: DeepGEMM paged-MQA indexer +FP4 MoE+ page-split (decode TPOT up to 3.4x faster and TTFT -20% on 4x RTX PRO 6000): [#29927](https://github.com/sgl-project/sglang/pull/29927)
- [scheduler, quantization] [AMD][gfx95] Fill the chunked-prefill compute budget exactly (mean TTFT -5.4 to -9.5% on GLM-5.2-MXFP4 MI355X TP4): [#32888](https://github.com/sgl-project/sglang/pull/32888)
- [spec_decode, quantization] [AMD] gfx950 assembly attention for EAGLE verify, draft extend and decode (median TPOT -25.6% on Qwen3.5-397B MXFP4 TP4): [#37465](https://github.com/sgl-project/sglang/pull/37465)
- [spec_decode, quantization] [AMD] Quantize the bf16 MTP draft experts online to MXFP4 for Qwen3.5 (+13.6% throughput per GPU, -10.3% median TPOT): [#38748](https://github.com/sgl-project/sglang/pull/38748)
- [scheduler] Sampling masks for RL rollouts.** With `return_sampling_mask`, each decode step returns the exact token support the sampler drew from and the log-probability of the sampled token under it, so a trainer can replay the rollout without reconstructing top-k or top-p ([#36630](https://github.com/sgl-project/sglang/pull/36630)). Masks now run under overlap scheduling: on Qwen3-8B, decode throughput is 1

## Touch count by knob

- quantization: 58
- kv_offload: 37
- hardware: 37
- moe_parallel: 25
- attention_kernels: 24
- spec_decode: 24
- disagg: 23
- cuda_graphs: 20
- other_perf: 19
- scheduler: 15
- prefix_cache: 8
- defaults: 2

## Other notable lines

- [quantization, attention_kernels] [FP8] SM120: route FP8 linear to per-tensor (cudnn/nvjet) instead of channelwise cutlass (Ministral-3-14B decode 46.9 to 86.6 tokens/s at batch 1): [#38006](https://github.com/sgl-project/sglang/pull/38006)
- [spec_decode] [EAGLE] Prune draft-extend logits to selected rows (+2.5% output tokens/s/GPU at 102K context per rank): [#35546](https://github.com/sgl-project/sglang/pull/35546)
- [kv_offload] HiCache: Reduce the number of `all_reduce` in `check_hicache_events` for PP (TTFT -7% on DeepSeek-V4-Flash with HiCache L3, PP4 TP2): [#37562](https://github.com/sgl-project/sglang/pull/37562)
- [kv_offload, hardware] [ROCm] Raise HiCache JIT block quota for mapped-host throughput: [#39036](https://github.com/sgl-project/sglang/pull/39036)
- [kv_offload, disagg] DSpark under PD with decode context parallelism.** A DCP1 prefill can now transfer its DSpark draft KV to a DCP-N decode, so hybrid models such as Kimi-Linear run DSpark in disaggregated, context-parallel serving. Verified on 8x B300 over NIXL and Mooncake up to 256K input ([#37709](https://github.com/sgl-project/sglang/pull/37709)).
- [spec_decode, cuda_graphs] Improve CUDA graph and speculative execution output handling: [#37329](https://github.com/sgl-project/sglang/pull/37329)
- [quantization] [Perf] trtllm_mla: reuse the fused fp8 KV/Q prepare on target verify: [#39232](https://github.com/sgl-project/sglang/pull/39232)
- [quantization, moe_parallel] [MoE Refactor] Migrate SM100 trtllm-gen mxfp4 MoE onto MoeRunner: [#32405](https://github.com/sgl-project/sglang/pull/32405)
- [quantization, moe_parallel] [Deepseek V4] Keep fp32 routing weights in the mxfp4 trtllm MoE: [#33608](https://github.com/sgl-project/sglang/pull/33608)
- [quantization, moe_parallel] Keep fp32 routing weights in the fp8 block-scale and bf16 trtllm MoE: [#33631](https://github.com/sgl-project/sglang/pull/33631)
- [attention_kernels, moe_parallel] Drop the routing bias casts in flashinfer trtllm MoE: [#33591](https://github.com/sgl-project/sglang/pull/33591)
- [quantization, moe_parallel] [Moe] Fix flashinfer_trtllm silently dropping swiglu_limit clamped SwiGLU activation (GLM-5.3-Flash FP8 GSM8K 0.928 to 0.936, matching the Triton runner): [#39920](https://github.com/sgl-project/sglang/pull/39920)
