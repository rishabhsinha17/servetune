# vllm-project/vllm v0.25.0 (2026-07-11): what to re-tune

37 performance or default-related lines; 3 likely to move a common config by more than 10%.

## Re-benchmark triggers

- [defaults, prefix_cache, spec_decode, quantization, cuda_graphs] Model Runner V2 is now the default for all dense models** (#44443). Building on quantized-model support from the previous release, MRv2 is now the standard execution path, with new support for EVS (#46535), realtime embeddings (#46762), prefix caching for Mamba hybrid models (#42406), multimodal-prefix bidirectional attention (#46942), and dynamic speculative decoding compatible with full CUDA gra
- [quantization, attention_kernels, moe_parallel] GLM-5.2 / DeepSeek: `fused_indexer_q_rope_quant` Triton kernel (1.9–3.3% E2E throughput) (#46862), reduce-scatter MoE all-reduce (3.1–3.2% E2E) (#46635), op fusion for GLM5/DSV3.2 (#46876), `token_to_req_indices` cache for DSv4 (5–6x kernel speedup) (#47474), better DSv4 MXFP8 kernel (#47229), redundant-op removal (#47198, #46651).
- [quantization, attention_kernels, hardware] NVIDIA/Blackwell: FlashInfer fused all-reduce tuned for world_size=16 on GB300 (#46392), restored NVFP4 swizzled-scale zero-init to recover Blackwell decode throughput (#45739), CuTeDSL/FA4-MLA warmup infrastructure (#46182), skip cooperative top-K on SM120 (#47164), B12x backend for non-gated MoEs (#43328).

## Touch count by knob

- quantization: 19
- moe_parallel: 15
- attention_kernels: 11
- spec_decode: 7
- hardware: 6
- cuda_graphs: 4
- prefix_cache: 3
- kv_offload: 3
- other_perf: 3
- scheduler: 2
- defaults: 1
- disagg: 1
- memory: 1

## Other notable lines

- [quantization, attention_kernels, moe_parallel, hardware] AMD/ROCm: moved to torch 2.11 stable ABI (#47128); AITER FlashAttention MLA prefill backend `ROCM_AITER_FA` (#45033); fused shared-expert for GLM-4.5/6/7 (#44313) and MiniMax-M3 (#46474, #46545); AITER MoE optimization for DeepSeek-V4 (#46122); AITER custom all-reduce in CudaCommunicator (#46065); INT3 quantization for quickreduce (#45666).
- [scheduler, prefix_cache, quantization, moe_parallel] CPU: accelerated unquantized MoE for AArch64 (#46353), macOS/Apple Silicon hang fix via OpenMP (#46769) and broken-install fix (#47457), compressed-tensor w8a8 int8 MoE (#42920), Mamba ShortConv (#35059), chunked prefill + prefix caching for Qwen3.5 (#46202), faster gelu via tanh AOR (#44639).
- [quantization, moe_parallel, hardware] ROCm: stabilized high-throughput DBO for DP+EP (#46990), EPLB for Quark OCP MXFP4 MoE (#47220).
- [spec_decode, quantization, moe_parallel] Fixes: W8A8 int-quant scheme-selection regression (#46860), tied quantized embeddings for ModelOpt Gemma4 (#45544), NVFP4+MTP crash on Qwen3Next (#46316), ModelOpt mixed-precision for sparse configs (#47318), CPU w4a8_int8 MoE path (#46739), actionable error on group-size/TP mismatch (#46230).
- [quantization, cuda_graphs, moe_parallel] The Transformers modeling backend is now as fast as native vLLM** (#47187), and gained FP8 MoE support (#46820), CUDA graph + embed scaling fixes (#48010), and migration of GPTBigCode/Starcoder2 (#30966) and RoBERTa (#47452).
- [quantization, cuda_graphs, moe_parallel] Transformers backend: now as fast as native vLLM (#47187), FP8 MoE fix (#46820), embed scaling + CUDA graph fix (#48010), GPTBigCode/Starcoder2 (#30966) and RoBERTa (#47452) migration, M-RoPE `mm_token_type_ids` fix (#46552), tied-embedding `lm_head.bias` fix (#46835).
- [spec_decode, moe_parallel] Gemma family: Gemma4 sliding-window/FA4 attention fixes (#47217, #47332), Gemma4 MTP quant_config fix (#47091); DiffusionGemma tensor parallelism (#45719) and HF stability-window semantics (#45965).
- [scheduler, prefix_cache] Model Runner V2: default for all dense models (#44443); EVS (#46535), realtime embeddings (#46762), Mamba hybrid prefix caching (#42406), multimodal-prefix bidirectional attention (#46942), cross-attention warmup/block-table fixes (#46753, #47308), Mamba2 crash fix (#47428), scheduling slot accounting (#46974), model-ref cleanup on shutdown (#47483), bounded memory for large-logprobs requests (#46
- [quantization, attention_kernels, hardware] Attention: FlashAttention block-size restriction removed for hybrid models (#36701), `FLASH_ATTN_MLA_SPARSE` Hopper sparse-MLA backend (#46189), DCP + FP8 KV cache in MLA decode (#44044), XQA decode kernels (#43232).
- [moe_parallel] Kernels: Helion `fused_qk_norm_rope` (#44010) and `silu_and_mul_per_block_quant` (#43994), Triton MLA logits workspace (#46819), swap-AB optimization for fused MoE (#36559), vectorized fp32 `moe_sum` supporting any top-k (#46643), blocking CUDA events to avoid busy-polling the driver lock (#47081).
- [quantization, cuda_graphs] Intel XPU: W8A8 FP8 linear kernel with multi-granularity quant (#43645), pipeline-parallel accuracy fix (#47253), uniform-batch CUDA graph for FA2 (#46555), route mm_prefix models to Triton attention (#47688), C++ `get_memory_info` (#47134).
- [perf] Sequence parallelism without requiring DP, 1.9–5.0% E2E throughput improvement (#47070).
