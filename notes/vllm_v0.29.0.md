# vllm-project/vllm v0.29.0 (2026-09-09): what to re-tune

46 performance or default-related lines; 7 likely to move a common config by more than 10%.

## Re-benchmark triggers

- [defaults, prefix_cache, attention_kernels] New defaults**: FlashInfer all-reduce enabled by default for TP CUDA groups, opt out with `VLLM_ALLREDUCE_USE_FLASHINFER=0` (#52998); prefix-cache `NONE_HASH` is deterministic by default so distributed KV cache users no longer need to pin `PYTHONHASHSEED` (#51875); new `--max-num-queued-reqs` / `--max-num-queued-tokens` admission-control flags (#49445).
- [defaults, prefix_cache, kv_offload, disagg, spec_decode, attention_kernels, cuda_graphs, hardware] Context parallelism**: Kimi-K3 DCP with DSpark (#52188) and partial prefix cache hits (#50493), FlashInfer native CP for MLA decode (#54012), FlashMLA sparse DCP on Hopper with MTP (#46514), NIXL P/D DCP for MLA models (#50611), PCP for DSv3.2 (#52046) and NIXL PCP producers (#52779), `--dcp-q-replicate` with query replication default-on for GLM sparse attention (#50382), DCP fused attention fix f
- [defaults, attention_kernels] FlashInfer all-reduce enabled by default (#52998).
- [prefix_cache, quantization, attention_kernels, moe_parallel, hardware] Kimi-K3 and DeepSeek V4 performance**: fused MXFP4 top-k finalization in the K3 latent tail (about 5% E2E latency, #53152), K3 Mamba metadata preparation in one Triton launch (6.6-7.6x kernel speedup, #52388), tuned Hopper low-latency GEMM (#54088) now also dispatched on SM100 (#53534) and used for `eh_proj` (12.9-25.2% kernel speedup, #53942), GEMM-RS extended to GEMM-AR (#53053), MLA gate merged
- [prefix_cache, spec_decode] Mamba prefix caching**: internal prefill checkpoints deliver a 9%-25% TTFT improvement (#52789); `prefix_cache_retention_interval` is now a CLI argument defaulting to 0 (#52216), with dense retention automatically restored for hybrid models using EAGLE/MTP (#55760, #55861).
- [moe_parallel, hardware] Determinism**: `trace_decode_token_ids` for deterministic decode replay (#46701), per-arch tuned batch-invariant matmul configs (about 3x decode kernels on RTX 4090D/H20, #53247), Blackwell autotuning with 33.6% E2E latency reduction (#53649), deterministic MoE combine under DP+EP (#45683), and `fuse_allreduce_rms` disabled under batch invariance (#51292).
- [prefix_cache, kv_offload, spec_decode, quantization, cuda_graphs, hardware] AMD ROCm**: dual-stream decode with hipgraphs (#52033), W4A4 preshuffled asm GEMM by default (+15% throughput on Llama-3.3-70B MXFP4, #53141), ROCr/CLR update fixing graph replay segfaults with up to about 20% TPOT improvement (#53712), fused KDA decode on MI325X (#52293), FULL cudagraphs for AITER MLA spec decode (#51171), FP8 asm MLA prefill for non-divisor head counts (#51040), DCP causal multi

## Touch count by knob

- quantization: 13
- spec_decode: 12
- moe_parallel: 12
- attention_kernels: 12
- prefix_cache: 11
- hardware: 10
- cuda_graphs: 9
- kv_offload: 8
- defaults: 5
- other_perf: 5
- scheduler: 2
- disagg: 1

## Other notable lines

- [quantization, attention_kernels, moe_parallel, hardware] Kernels**: fused embedding kernel (#53677), FA4 re-enabled for head_size=256 on Blackwell (#52980), vectorized sparse MLA mask loads (#52217), masked MHA prefill for GLM-5 head dimensions (#53785), GLM-5.2 sparse MLA Q concatenation fused with head padding (#53878), fused QK-norm + partial MRoPE + gate for Qwen3.6 (#52676), FlashInfer CuTeDSL BF16 low-latency GEMM as opt-in `--linear-backend flash
- [defaults] InstantTensor added to CUDA dependencies; the loader is not enabled by default (#52801). CuPy constraint relaxed to exclude only 14.1.0 (#44284).
- [defaults] `prefix_cache_retention_interval` default changed from dense to 0 for SWA/SSM models; the env var is deprecated in favor of the argument (#52216).
- [prefix_cache, cuda_graphs] Multimodal performance**: ViT full CUDA graph for Idefics3 and SmolVLM (#47625), packed encoder attention for Pixtral (#52185), fused Kimi vision Q/K RoPE kernel (#50400), Dots3 NOTE runtime (#53517) and Omni encoder optimizations (#53460), MM tensors no longer broadcast to workers for prefix-cache-covered items (#52041), redundant placeholder scans skipped (#52925), common token sequences cached 
- [quantization, moe_parallel] CPU**: AMX high-performance MLA backend for DeepSeek V2/V3/R1 (#52616) with MLA now running end-to-end (#51471), FP16/BF16 persisted GDN state on AMX (#52191), Int8 MoE through zentorch on AMD Zen (#44834), Voxtral support (#53921), C++ causal_conv1d GDN on non-AMX AVX-512BF16 (#49688), and assorted fixes (#54042).
- [quantization, attention_kernels, cuda_graphs, moe_parallel] MoE communication**: DeepEP v2 receiver CPU overhead (#51114) and MXFP8 activation scale dispatch (#51398), FlashInfer one-sided All2All refinements (#51924), DeepEP v2 fixes for `--enforce-eager` startup (#51824) and the decode/cudagraph path (#52632), NCCL>=2.31 heap overflow fix (#53008), cross-node MNNVL all-reduce gated by capability (#53253), and MNNVL all-reduce buffers sized for DSpark (#5
- [spec_decode, cuda_graphs, hardware] Model Runner V2 is now the default for all models** (#53183), completing the rollout that began with pooling models (#48290). MRV2 also gained CUDA graph memory profiling for KV cache auto-sizing (#53306), batch-sharded sampling that cuts per-step logits memory by 1/TP (#50465), prompt embeds (#42963), `extract_hidden_states` speculation (#49811), padded FULL cudagraph dispatch for uniform decode 
- [spec_decode, quantization, moe_parallel] New models**: Hy4-preview, Tencent's 770B/49B-active MoE with Gated DeepSeek Sparse Attention and native MTP (#54160); Qwen3.8-Flash-Next with BF16/FP8/NVFP4 and MTP (#53896); GraniteSWA and GraniteMoeSWA (#52706); NemotronH_Omni_Reasoning_V3 with MTP (#52929, #53121); Kimi K3 NVFP4 checkpoints (#53132).
- [spec_decode, moe_parallel] Some [features](https://github.com/vllm-project/vllm/issues/47172) are not yet supported in MRV2 but we are planning for these gaps to be closed within the next 2-3 weeks. These include sequence parallelism, dual-batch overlap, elastic expert parallellism, custom logits processors and certain speculative decoding methods. For now, vLLM will still fall back to use MRV1 if any of these features are 
- [spec_decode, quantization] New models**: Hy4-preview (#54160), Qwen3.8-Flash-Next (#53896), GraniteSWA and GraniteMoeSWA via the existing Granite implementation (#52706), NemotronH_Omni_Reasoning_V3 (#52929) with MTP for Nemotron VL models (#53121), bidirectional attention for DeepSeek-backbone embedding models (#52948), FP8 ModernBERT (#53101), and Kimi K3 NVFP4 checkpoints (#53132).
- [prefix_cache, quantization, cuda_graphs, moe_parallel] Correctness**: PaliGemma stale image scaling removed (#52692), Mistral3 placeholder grid with processor size overrides (#52874), MiniCPM-o on Transformers v5 (#54501), JinaVL cache order (#53553), mixed CLIP/SigLIP pooling batches (#53165), XD-RoPE models on prefix-cache hits (#53456), Qwen3-Omni audio encoder with non-divisible TP (#50858), GLM-5.2 no longer uses dense MHA (#52512), Moondream3 Mo
- [spec_decode, attention_kernels, cuda_graphs] Speculative decoding**: per-request acceptance stats (#48915), adaptive verification with logprobs (#52242), on SM100 sparse MLA (#52783) and DSv4 + SM90 (#52795), varlen trtllm-gen decode (#52157), FlashInfer MLA for DSpark drafting under DCP (#54277), fused GDN MTP for all Qwen head ratios (#52539), widest uniform decode batch captured by default (#50488) with memory-safe graph sizes (#54418), a
