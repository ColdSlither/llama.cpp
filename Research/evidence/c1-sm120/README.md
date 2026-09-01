# C1 Evidence — qwen3moe SOFT_MAX fix on Blackwell sm_120

Commit: `0bc5f67887` (feature/kv-compression)
Date: 2026-08-31
GPU: RTX 5060 Ti, sm_120, driver 580, CUDA 13

## Root cause

On CUDA 13 + Blackwell sm_120, `cudaGetDeviceProperties` fills
`sharedMemPerBlockOptin` with a corrupted value (4294967297 = 0x100000001,
4 GiB + 1). The device attribute query
(`cudaDevAttrMaxSharedMemoryPerBlockOptin`) correctly reports 101376 (99 KB).

ggml stored the corrupted value in `devices[].smpbo`, so every
`CUDA_SET_SHARED_MEMORY_LIMIT` call requested a 4 GB dynamic-shared-memory
limit. `cudaFuncSetAttribute` failed with `cudaErrorInvalidValue`, and the
sticky error detonated at the next post-op check as "SOFT_MAX failed /
CUDA error: invalid argument" on any qwen3moe graph (the MoE router softmax,
ncols = n_expert, is the first caller at warmup/decode). Dense graphs escaped
only by graph luck.

## Fix

In `ggml_cuda_info()` device init (ggml/src/ggml-cuda/ggml-cuda.cu): when
`sharedMemPerBlockOptin` is not a sane per-block value (> 232448 = 227 KB,
the largest real opt-in on any current arch), override it with
`cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)` and log a
warning. This repairs every `CUDA_SET_SHARED_MEMORY_LIMIT` consumer, not just
softmax.

## Verification

The guard fires on startup in every run:
`sharedMemPerBlockOptin reported 4294967297 (corrupted), using device attribute 101376 instead`

### V1 — Qwen3-Coder-30B-A3B via llama-cli (c1-v1-llama-cli.txt)
- Generates a correct Fibonacci implementation.
- Prompt 126.3 t/s, generation 113.6 t/s.
- Zero SOFT_MAX / CUDA errors (previously aborted at warmup/first decode).
- Note: this fork's llama-cli enters its conversation loop after the -p turn
  and ignores EOF, so a literal "exit 0" is unreachable; generation and
  error-free criteria are fully met.

### V2 — llama-server (-a coder, :8092) (c1-v2-server.log)
- The handoff's exact curl returned a valid JSON completion (memoized
  Fibonacci, 190 chars).
- Server alive throughout; 83 tokens, 126.5 t/s generation, zero errors.

### V3 — dense Qwen3-8B-uncensored-cma (c1-v3-dense.txt)
- Generates normally at 68.6 t/s, no CUDA errors — no regression.

## Files
- c1-v1-llama-cli.txt — V1 signal (guard + fib + timing)
- c1-v2-server.log — V2 full server log
- c1-v3-dense.txt — V3 signal (guard + timing)
