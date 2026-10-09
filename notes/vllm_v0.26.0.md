# vllm-project/vllm v0.26.0 (2026-07-27): what to re-tune

36 performance or default-related lines; 3 likely to move a common config by more than 10%.

## Re-benchmark triggers

- [spec_decode, hardware] DeepSeek-V4 performance push** across vendors: a specialized routing kernel (2.94% E2E TPOT, #48660), `fused_topk_bias` (1.5–2x kernel, #47463), and redundant repeat/copy removal (1.8% E2E TPOT, #48137), plus ROCm two-stage compressor for HCA prefill (#47718), sparse decode/prefill optimizations (#48519, #48788, #46275), and DSpark speculative decoding on AMD (#47419) and XPU (#47677).
- [spec_decode] CPU: DFlash speculative decoding for GDN models on CPU (#46090), s390x NUMA topology (#40714), native macOS arm64 CPU wheel builds (#48289); POWER VSX math function optimization (#47321) and IBM Power docker builds using prebuilt wheels (#46017).
- [perf] DeepSeek-V4: specialized routing kernel (2.94% E2E TPOT, #48660), `fused_topk_bias` 1.5–2x (#47463), redundant repeat/copy removal (1.8% TPOT, #48137).

## Touch count by knob

- spec_decode: 9
- attention_kernels: 8
- cuda_graphs: 7
- hardware: 7
- moe_parallel: 7
- quantization: 5
- kv_offload: 4
- other_perf: 2
- scheduler: 1
- prefix_cache: 1
- memory: 1
- disagg: 1

## Other notable lines

- [attention_kernels, moe_parallel] LoRA: FlashInfer MoE LoRA for BF16 models (#48632), LoRA for tower/connector in LlavaNextVideo (#48594), fp32 `lm_head` on the LoRA path (#48525), optimized `TrtLlmLoRAExperts` (#48759).
- [spec_decode, quantization, cuda_graphs, hardware] ROCm: fp32 `head_dtype` `torch.mm` fast path (#48688), DSv4 two-stage compressor kernel (#47718), sparse decode/prefill optimizations (#48519, #48788, #46275), DSv3.2 sparse MLA KV-split heuristic (#46832) and MTP CUDA-graph mode (#45149), MXFP8 GEMM for MiniMax-M3 (#46117), AITER sparse paged attention + spec decode for MiniMax-M3 (#47287, #47984), MiniMax-M2 fused QK-norm + all-reduce via AITER 
- [spec_decode, quantization, cuda_graphs, hardware] New Inkling model family** with a full support stack: base modeling (#48799), piecewise CUDA graph support (#48822), Hopper FA4 relative attention (#48858), MTP=1 speculative decoding (#48869), LoRA (#48884), and standard ModelOpt NVFP4 quantization (#48990).
- [scheduler, cuda_graphs] Fixes: correct pooling scores for chunked prefill under `torch.compile` (#48901).
- [spec_decode] Reasoning: optimize TPOT for thinking budget when used with speculative decoding (#46662).
- [spec_decode, memory] Fixes: host memory leak from undrained `new_block_ids` (#44490), DSv3.2 + MTP + sequence-parallel accuracy (#48036).
- [attention_kernels, moe_parallel] MoE router GEMMs: BF16x3 router GEMM (#47973), FP32 router GEMV (#48335), generic CuteDSL LL BF16 router GEMM (#42562); TRTLLM BF16 MoE modular kernel (#45182); write FlashInfer combine into final output (#47156).
- [moe_parallel] Qwen: fuse more RMSNorm + all-reduce in Qwen3.5 (#46998), replace MoE all-reduce with reduce-scatter (#47006), Qwen3.5 H20 optimization (#48350), expand Triton warmup coverage (#47546).
- [kv_offload, disagg] PD disaggregation: NIXL pipeline-parallel prefill in push mode (#45880).
- [quantization, moe_parallel] Humming w[2-7]a[4,8] weight-only inference with compressed-tensors (#46390); int4 quantization for the emulation MoE backend (#48451); INT2 XPU weight-only quant linear (#47521).
- [quantization, attention_kernels, moe_parallel] NVFP4/MXFP4: `nvfp4_per_token` online MoE quantization (#48538), CuTe-DSL FlashInfer MXFP4 quantization (#48417); bounded peak memory when repacking FP4 MoE weights for Marlin (#47851) and for NVFP4 MoE weight loading (#46276).
- [kv_offload, attention_kernels, hardware] Transformers 5.13.0 (#47867), FlashInfer 0.6.14 (#47669), NIXL 1.3.1 (#47559), tpu-inference v0.24.0 (#47835), nvidia-cutlass-dsl 4.6.0 (#47442), vllm_xpu_kernels v0.1.11.1 (#48942).
