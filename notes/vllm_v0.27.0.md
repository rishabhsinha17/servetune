# vllm-project/vllm v0.27.0 (2026-08-10): what to re-tune

40 performance or default-related lines; 5 likely to move a common config by more than 10%.

## Re-benchmark triggers

- [quantization, attention_kernels] DeepSeek-V4 performance push**: sequence parallelism (#46789), ~2x kernel improvement by skipping empty c128 launches (#48957), 3.4% E2E TTFT from skipping unneeded topk/router (#49486), 3.9% E2E TTFT from workspace reuse (#49236), 1.88x kernel from removing a redundant full kernel (#50298), adaptive topk width (1.0% E2E, #50004), 448 MiB GPU memory saved in the PP buffer (#50312), a compact MXFP4
- [quantization, attention_kernels] DeepSeek-V4: sequence parallelism (#46789), ~2x kernel skipping empty c128 launches (#48957), 3.4% E2E TTFT skipping topk/router in decode (#49486), 3.9% E2E TTFT workspace reuse (#49236), 1.88x kernel removing a redundant full kernel (#50298), adaptive topk width 1.0% E2E (#50004), 448 MiB GPU memory saved (#50312), compact MXFP4 indexer KV cache (#48993), sparse-MLA q-head padding removed for Fl
- [quantization, attention_kernels, cuda_graphs, moe_parallel] Kernels: RMSNorm uncontiguous support with 1.2–3.1x kernel improvement (#49750), MoE `reduce_scatter` regression fix restoring 5% E2E throughput (#48763), non-grouped bias-less topk routing dispatched to the fused path (#49618), tuned LL BF16 router GEMM (#48774) with warmup skipped for non-MoE models (#49659), Triton tensor-descriptor path for fused MoE via `VLLM_TRITON_USE_TD` (#42436), cudagrap
- [quantization, cuda_graphs, hardware] ROCm: gfx1250 architecture enabled (#46516), AITER FP8 ViT encoder attention (#49937), fused shared expert for Quark DeepSeek-V4 checkpoints (#48044), Quark GLM-5.2 checkpoint inference fixes (#48886), DSv3.2 per-decode FillFunctor launches eliminated in the sparse-MLA hot loop (#44527), B-preshuffled attention FP8 projections for DSv4 (#46720), TML Inkling enabled (#48841), tuned selective_state_
- [spec_decode, quantization, moe_parallel] CPU: INT8 fused MoE kernel for Arm CPUs (#48637), s390x inference optimization with oneDNN INT8 GEMM (#50219), GDN conv path optimized for speculative decoding (#48577), granite-4 enabled (#47641), FAST_EXP for Power (#49571), CPU kernels bumped to the latest version (#50387), macOS build fixes (#49021, #50915).

## Touch count by knob

- quantization: 14
- attention_kernels: 13
- moe_parallel: 9
- cuda_graphs: 7
- spec_decode: 6
- kv_offload: 6
- disagg: 3
- hardware: 3
- memory: 3
- other_perf: 2
- prefix_cache: 1

## Other notable lines

- [spec_decode, quantization, cuda_graphs] Multimodal: VidCom2 video token pruning (#47750), EVS for Qwen3.5 (#48912), ViT CUDA graph for Gemma-4 (#46837), Cosmos3 FP8 ModelOpt/Diffusers remapping (#48952), MiniMax-M3 MSA speculative decode verification (#50032) and default video processor (#50305), DeepSeek-OCR-2 TTFT optimization (#49531), longer max audio duration for MOSS-TD (#49403).
- [quantization, attention_kernels] Kimi K3 support** with a full stack landing in one release: core model files and kernels (#50089, #50000), Python (#50093) and Rust (#50104) frontends, AttnRes kernels (#50090), DeepGEMM support (#50458), compressed-tensors quantized checkpoints (#50500), DSpark AR fusion (#50242), and an option to shard the shared expert instead of replicating it (#50656).
- [quantization, attention_kernels, cuda_graphs] FlashAttention 4 integration deepens on SM100**: FP8 KV cache support (#42569) and headdim-256 support (#42669), backed by a new JIT warmup infrastructure (#47451) and runner-owned Triton kernel warmup (#49903) that remove first-request compilation stalls.
- [kv_offload, disagg] Disaggregation for hybrid models**: NIXL P/D for hybrid MLA+SSM models (#49762), heterogeneous P/D block sizes for hybrid models (#49612), and MoRIIO heterogeneous TP<->DP prefill/decode read routing (#46116).
- [quantization, attention_kernels, cuda_graphs] Attention: FlashAttention 4 SM100 FP8 KV cache (#42569) and headdim-256 (#42669); query replication for MLA decode under DCP for DeepSeek-V2/R1 and Kimi-K2.5 (#45964); masked MHA for sparse MLA prefills (#48770); skip sparse indexer scoring for short dense prefills (#48407); FlexAttention epilogue hook (#45841) and encoder block-mask compile explosion avoided (#50339); attention backends stay elig
- [spec_decode, memory] Model Runner V2: encoder-only attention (#49331), sequence pooling for embedding and classification (#48791), encoder token classification (#50293), encoder token embedding (#50574), BGE-M3 pooling (#50661), multimodal on CPU (#50073), multi-layer MTP speculator (#48892), PCP selects MRV2 (#50034), attention metadata always built at capture time (#49364), encoder cache profiling (#47985), skipped 
- [attention_kernels] Mamba/hybrid: ReplaySSM caching for faster Mamba2 standard decode (#48018), fused align-mode DS-conv state migration with num_accepted_tokens > 1 (#49291), FlashInfer Mamba SSU algorithm selection (#50157), fixed `/wake_up` crash on hybrid models (#41602).
- [attention_kernels, moe_parallel] RL: weight version tagging for RL rollouts (#49040), stateful trainer-send IPC (#48981), vLLM config set during weight reload (#45989), router replay output from the FlashInfer monolithic MoE kernel (#44214).
- [prefix_cache, cuda_graphs] Memory & robustness: CuMem slept-L1 fragmentation accounting (#49208), cgroup memory limits respected on all platforms (#49966), fail fast when /dev/shm is too small (#48879), zero-copy tensor pickling in shm_broadcast (#48442), LRU hash-split skipped in free_blocks when prefix caching is off (#48017), location-derived path vars excluded from torch.compile cache factors (#47573), `CustomOp.forward
- [quantization, moe_parallel] XPU: QK Norm + RoPE fusion pass (#49394), FP8 o_proj with fp8_bmm and load-time scale transpose (#48334), DeepSeek-V4 fuse_index_q SYCL kernel path (#45991), TD operand loads for batched MoE GEMM (#46340), RMSNorm kernels unified with vllm_c (#46981).
- [kv_offload, disagg] P/D disaggregation: NIXL P/D for hybrid MLA+SSM models (#49762), NIXL heterogeneous P/D block sizes for hybrid models (#49612), MoRIIO heterogeneous TP<->DP prefill/decode read routing (#46116), optional lookup disable on PD decode (#50498), prefill token ids reused on the decode chat path (#48145), detokenization streaming derender (#47301), NixlPush skips an extra handshake step in D->P (#49345)
- [kv_offload, disagg] Fixes: P/D preemption race (#50297), KV lease deadlines rebased onto the worker clock (#50326), NIXL hybrid MLA+mamba heterogeneous TP (#49297), internal LB load-balancing (#49204).
