# vllm-project/vllm v0.25.1 (2026-07-14): what to re-tune

1 performance or default-related lines; 0 likely to move a common config by more than 10%.

## Re-benchmark triggers

- None flagged. Spot-check only if the client runs a knob listed below.

## Touch count by knob

- quantization: 1
- attention_kernels: 1

## Other notable lines

- [quantization, attention_kernels] Guard mixed-dtype allreduce RMSNorm quant fusions** (#48330). The fused FlashInfer allreduce + RMSNorm + static-quantization patterns could match graphs where the activation and RMSNorm weight dtypes differ (e.g. a BF16 residual stream with an FP32 Gemma/Qwen-style RMSNorm weight in NVFP4 models), corrupting the hidden state and producing garbage output such as repeated `!!!!!` tokens. A dtype-mat
