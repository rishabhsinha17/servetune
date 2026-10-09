# sgl-project/sglang v0.5.15.post1 (2026-07-14): what to re-tune

3 performance or default-related lines; 0 likely to move a common config by more than 10%.

## Re-benchmark triggers

- None flagged. Spot-check only if the client runs a knob listed below.

## Touch count by knob

- attention_kernels: 2
- quantization: 1
- moe_parallel: 1
- disagg: 1

## Other notable lines

- [quantization, attention_kernels, moe_parallel] #31001: Fix NaN outputs caused by flashinfer trtllm FP4 MoE kernels on long input
- [attention_kernels] #30858: Fix flashinfer dependency on Cuda 12 images
- [disagg] #30839: Fix GLM 5.2 IndexShare on PD disaggregation setting
