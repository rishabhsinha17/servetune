# vllm-project/vllm v0.28.0 (2026-08-26): what to re-tune

40 performance or default-related lines; 3 likely to move a common config by more than 10%.

## Re-benchmark triggers

- [defaults, scheduler, prefix_cache, cuda_graphs, hardware] New defaults**: `max_num_batched_tokens` raised from 8192 to 16384 (#51726), prefix caching enabled by default for Mamba models (#50991), and the Blackwell CUDA graph capture default raised to 1024 (#49390).
- [spec_decode, hardware] Kimi-K3 performance push**: a major optimization effort for Kimi-K3 across the stack — Decode Context Parallel (DCP) support (#50484), fused FlashKDA decode and prefill kernels (#50654, #51311, #52458), SiTU activation support for MegaMoE (#50510), GEMM-RS for sequence parallelism (#52079), combined all-gathers with 1.5~3x kernel-level speedup (#51070), an adaptive speculative token budget deliver
- [quantization, attention_kernels, hardware] AMD ROCm**: torch 2.12 / triton 3.7 stack bump (#50607), AITER and FP8 inference enabled on GFX120x (#43615), DeepSeek-V4 on gfx11 (#47017), optimized Triton sparse-MLA decode on gfx950 (#52212), FlyDSL decode-attention kernel for 4-bit TurboQuant KV cache (#47896) and an fp8 MQA logits kernel on gfx942 (#49544), a fused Kimi-K3 KDA decode kernel (#50654), fused bf16→fp32 router GEMM (#50268), pin

## Touch count by knob

- quantization: 12
- moe_parallel: 12
- spec_decode: 9
- hardware: 8
- attention_kernels: 7
- scheduler: 6
- kv_offload: 5
- cuda_graphs: 4
- prefix_cache: 3
- other_perf: 3
- disagg: 2
- defaults: 1
- memory: 1

## Other notable lines

- [spec_decode, quantization, cuda_graphs, hardware] DeepSeek V4**: sparse MLA now works end-to-end for plain decode, MTP, and DSpark speculative decoding (#51538), joined by AMD Quark NVFP4 support (#47972), reasoning-effort prompts and mappings (#50580), sparse top-k metadata kernel optimizations (#52084, #51967), narrowed eager CUDA graph regions (#51430, #52401), and ROCm enablement on gfx11 and gfx950 (#47017, #52212).
- [scheduler, spec_decode] Speculative decoding advances**: DFlash2 with local convolution and a candidate selector (#52816), DSpark confidence-scheduled verification (#47808), and async scheduling auto-enabled for draft models (#48341).
- [disagg, spec_decode, cuda_graphs] Model Runner V2 maturation**: E/P/D disaggregation (#38390), weight offloading (#51413), multi-layer MTP KV cache support (#50062), encoder CUDA graphs (#49852), decoder token-wise pooling (#50931) plus Transformers pooling models (#52425), attention-free models (#52374), and `thinking_token_budget` support (#46727).
- [spec_decode, quantization] New models**: Muse Glimmer (#51655), Ling 3.0 Flash with BF16, MTP, and parser support (#51045) plus an FP8 variant (#51265) and hybrid MXFP4 routed experts (#52114), Dots3 NOTE native multimodal support (#51255), and Interns2mobius (#51149).
- [spec_decode, hardware] Qwen**: Qwen3.8 enabled on AMD ROCm (#50068), fused CUDA post-conv MTP decode kernel for Qwen3.5 GDN (#51674), GDN gates aligned with speculative tokens (#51812), and Qwen3.5 fixes for text-only checkpoints (#50734, #50355).
- [quantization, moe_parallel] Correctness**: MiniMax-M3 NVFP4 inference (#48929) and compressed-tensors FP8 MoE SwiGLU params (#46845), Gemma3n/Gemma4 variable-length audio batch padding (#50958), Gemma 4 compatibility with the upcoming Transformers version (#49797), and a Qwen3-Omni crash on video without an audio track (#48420).
- [scheduler, spec_decode] Speculative decoding**: DSpark confidence-scheduled verification (#47808), top-k DSpark Markov projection (#49969), DFlash2 with local convolution and a candidate selector (#52816), async scheduling auto-enabled for draft models (#48341), fused MTP trailing all-reduce with local-argmax draft tokens (#49793), and an adaptive budget for speculative scheduled input tokens (#51725).
- [scheduler, prefix_cache] KV cache & scheduling**: per-request scheduling for MLA chunked context (#50613), partial-tail prefix reuse with fine-grained prefix matching (#50507), backend-published KV packing in the KV-cache layout refactor (#51612, #51704), LIFO free-block reuse order restored when prefix caching is off (#51482), and silent request skipping in priority scheduling fixed (#49206).
- [prefix_cache, attention_kernels] Hybrid/Mamba**: prefix caching on by default (#50991), the final part of the Mamba attention module refactor (#44857), 3D-grid tiling of the state-copy Triton kernels (#49436), and Mamba alignment applied before encoder caps (#51603).
- [spec_decode, quantization, attention_kernels, moe_parallel] NVIDIA**: FlashInfer XQA decode support on SM12x (#49718), a CuTeDSL fused query kernel on SM100 (#49792), programmatic dependent launch for the DSA decode kernels (#50230), the native DSA decode path for MTP=3 on SM90 (#52164), GB10 fused-MoE FP8 tuning configs (#52502), and B12X dense linear backends (#52016).
- [scheduler, quantization] Intel XPU**: a torch linear backend including blockwise GEMM (#49664, #50826), MXFP8 linear weights for the INC DeepSeek V4 model (#48476), async-scheduling PP sampled-token broadcast overlapped with compute (#51650), an XPU wheel added to the release pipeline (#52108), tuned Mamba SSU configs for Arc Pro B70 (#50534), and UVA weight offloading fixes (#51770).
- [quantization, attention_kernels, moe_parallel] CPU**: an MLA backend so DeepSeek-V2/V3 can run on CPU (#49453), a triton-cpu wheel (#52092), GPTQ and AWQ enabled on s390x (#51148) along with tcmalloc (#50841), BF16 MoE routed through zentorch on AMD (#44201), an unquantized MoE backend for Power (VSX) (#51624), unquantized MoE migrated to the modular-kernel experts structure (#50133), and the MXFP4 block scale folded in 2 instructions instead 
