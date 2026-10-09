# vllm-project/vllm v0.30.0 (2026-09-22): what to re-tune

53 performance or default-related lines; 5 likely to move a common config by more than 10%.

## Re-benchmark triggers

- [defaults, quantization, attention_kernels, moe_parallel] New defaults: FlashInfer CuTeDSL NVFP4 W4A16 over Marlin on SM100/103 (#53014), W4A4 NVFP4 over weight-only kernels on SM120/121 (#55170), BF16x3 router GEMM on SM100 (#55899), DeepEP v2 combine overlap on with `VLLM_DEEPEP_V2_COMBINE_OVERLAP=0` to disable (#52781), AITER custom AG/RS on for DP attention + TP experts with `VLLM_ROCM_USE_AITER_CUSTOM_AR=0` to disable (#48247), native CUDA AttnRes f
- [spec_decode, cuda_graphs, hardware] Model Runner V2**: dual-batch overlap in eager mode (#50945) and with FULL CUDA graphs for microbatched steps (#51700); MTP (#46994) and EAGLE3/DFlash/DSpark (#50514) speculative decoding under pipeline parallelism; adaptive verification for every draft-model speculator through an online acceptance estimator (#52228); gc frozen during graph capture, cutting capture from 12s to 2s and engine init f
- [prefix_cache, disagg, spec_decode, quantization, attention_kernels] Kimi K3 performance**: native CUDA AttnRes default on SM100 (#54261), KDA mixed-batch gather/scatter removed (5.2-7.7% E2E throughput, #56159), grouped FP8 MLA cache insertion (4-6x kernel speedup at small batch, #55356), DSV3 low-latency GEMM on strided tensors (12-81% kernel speedup, #54565), overlapped TP8 KDA projections (#54697), FlashInfer KDA kernels (#55364), internal prefix checkpoints wi
- [kv_offload, spec_decode, quantization, attention_kernels, cuda_graphs, moe_parallel, hardware] AMD ROCm**: TheRock base bumped to ROCm 10.0 (#55246) with nightly TheRock images (#55014), AITER 0.1.21.post2 (#52826, #55968), AITER custom AG/RS on the DP group (about 3% TPOT, #48247), multi-stream performance and rocprofiler fixes (#55099, #56098), bpreshuffled blockscaled FP8 GEMM (+4-8% QPS on DSv3, #51692), low-M FP32 router GEMM on gfx950 (#54845), DeepSeek V4 FHMoE shared-expert fusion (
- [scheduler] GLM-5.3-Flash**: EPLB support (#55119), FlashKDA for KDA chunked prefill (1.7-3.8x faster than the Triton chunk path, #55737), dense and masked-MHA sparse prefill for the NoPE (256, 0, 256) layout (#55738), decode hot-path cleanups removing per-layer copies and a duplicate router GEMM (#55736), and the `glm5next` NVIDIA subtree now packaged in wheels (#55214).

## Touch count by knob

- quantization: 22
- attention_kernels: 19
- spec_decode: 17
- hardware: 16
- moe_parallel: 16
- cuda_graphs: 13
- kv_offload: 9
- prefix_cache: 4
- disagg: 4
- scheduler: 3
- other_perf: 2
- defaults: 2
- memory: 1

## Other notable lines

- [quantization, moe_parallel] CPU**: DeepSeek-V4 backend (#55355), native AMX-FP8 attention for Diamond Rapids with the Docker base moved to Ubuntu 25.04 and GCC 15 (#49410), faster LM head on Arm (about 6% on Llama-3.1-8B W8A8, #55352), DeepSeek-R1 FP8 MLA + MoE correctness (#56773), profiler tables grouped by input shape (#56016), a dedicated Zen5 image (#50314), and nightly CPU images on Docker Hub (#55163).
- [defaults] Default audio resampler switched from PyAV to torchaudio (#52598); audio decoding is backend-selectable with soundfile first in `auto` (#51826, #55642).
- [quantization, cuda_graphs, moe_parallel] Qwen3.8-Flash-Next performance**: separate prefill and decode QSA indexer kernels (#54513), fused PLE kernels (#54517), FP8 indexer cache (#54890), padded-index skipping in sparse GQA (#54873), fused PLE residual and QSA output gate (#55309), UVA PLE offload and Engram tensor parallelism via `--engram-config` (#54371), and torch.compile removed from the NVIDIA implementation so FP8 fits on a singl
- [prefix_cache, spec_decode, quantization, attention_kernels, moe_parallel, hardware, memory] Kimi K3**: native CUDA AttnRes default on SM100 (#54261), ll_bf16 router weight prefetch (#53524), overlapped TP8 KDA projections (#54697), strided DSV3 GEMM (#54565), MLA decode epilogue latency (#54896), NVFP4 in_proj alignment routing to DeepGEMM (#55242), mixed-batch KDA without gather/scatter (#56159), grouped FP8 MLA cache insertion (#55356), FlashInfer KDA kernels (#55364), FlashKDA bf16 ch
- [quantization, attention_kernels, cuda_graphs, moe_parallel] Robustness**: attention implementations must explicitly declare DCP support (#55780), unavailable piecewise CUDA graphs raise instead of producing garbled output (#54782), non-compiled models fall back to `FULL_DECODE_ONLY` graphs (#55095), MRV1 breakable graphs scoped to the piecewise path (#56312), sync checks cover paged async copies (#53491) and torch.compile (#56904), more H2D copies pinned (
- [quantization, attention_kernels, hardware] New models**: DeepSeek-V4.1-Flash (#56214, #56228, #56208) with the whole KV stored in MXFP8 through the FlashMLA V4.1 record on SM100 (#56893), DeepGEMM Mega-mHC (#56962), and async Engram prefetch with Engram DP sharding (#56512); DeepSeek-V4-Flash-Vision-Exp (#54566), also on ROCm (#55107) and with LoRA (#55897); GLM-5.3-Flash (#53906) with EPLB (#55119); K2-Horizon (#55063); Cohere Compass (#5
- [kv_offload, spec_decode, attention_kernels, cuda_graphs, moe_parallel] Large scale serving**: PCP+DCP on sparse-MLA models (#56157), PCP with single-module MTP and replicated DSpark (#56107) and decode-only FULL CUDA graphs (#53867), Elastic EP reusing CUDA graphs across reconfiguration (#54985), an opt-in FlashInfer PCIe IPC all-reduce for NVLink-less boxes (#53576), DeepEP v2 async finalize overlapping shared experts with combine (#52781), Mooncake Store heterogene
- [quantization, attention_kernels] Quantization**: targeted online quantization through `quantization_config.targets` (#51285) and on partially pre-quantized checkpoints from any quant method (#51392), W4A16 DSA with the `nvfp4_fp8_ds_mla` KV cache (#51724), FlashInfer CuTeDSL NVFP4 W4A16 default over Marlin on SM100/103 (#53014), NVFP4 in the torch linear backend (#53319), per-quantization linear backend overrides (#51204), AutoRo
- [spec_decode, hardware] New models**: DeepSeek-V4.1-Flash (#56214, #56228, #56208), DeepSeek-V4-Flash-Vision-Exp (#54566) with ROCm support (#55107) and LoRA (#55897), GLM-5.3-Flash (#53906), K2-Horizon with its reasoning and tool parsers (#55063), Cohere Compass (#54774, #55949), Bailing V3 VL with MTP (#55921), Nanbeige4.2 via the Transformers backend (#56071), and a DeepSeek-V4 CPU backend (#55355).
- [quantization, attention_kernels] DeepSeek-V4.1-Flash**: whole KV in MXFP8 through the FlashMLA V4.1 record on SM100 (#56893), Mega-mHC from the DeepGEMM fork (#56962), mHC post block folded into the delayed pre projection (#56633), Triton-fused input metadata preparation (#56562), async prefetch for CPU-offloaded Engram lookups and Engram DP sharding (#56512), DSpark draft states collapsed before the SP all-gather (#56903), DSpar
- [quantization, attention_kernels] DeepSeek V4**: plain RoPE without YaRN on sparse SWA layers (#54815), `--kv-cache-dtype auto` resolving to `fp8_ds_mla` with FlashMLA (#45091), prefill sparse index workspace sentinel seeded (#55299), dequant gather grid sized by rows (#55061), optional Q-norm in the fused MLA epilogue and group_size=32 packed FP8 quant (#56215), sparse settings read from the text config for composite models (#561
- [spec_decode, quantization, cuda_graphs, moe_parallel, hardware] Qwen3.8-Flash-Next**: separate prefill and decode QSA indexer kernels with `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (#54513), fused PLE kernels and merged K/V projections (#54517), sparse GQA skipping padded indices (#54873), FP8 indexer cache (#54890), fused PLE residual and QSA output gate (#55309), UVA PLE offload and Engram tensor parallelism via `--engram-config` (#54371), torch.compile removed fr
