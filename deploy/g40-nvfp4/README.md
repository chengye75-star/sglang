# G40 production stack: NVFP4 KV + NEXTN MTP on 2x RTX 5060 Ti (SM120)

Lineage: `v0.5.21` + upstream PR#36038 (NVFP4 spec-decode: trtllm_mha native
FP4 multi-token decode + XQA mask path) + three local fixes below. This branch
is what `~/sg-env-0521-nvfp4mtp` runs in production.

## Commits on this branch
1. `PR36038 merged onto v0.5.21 + test adaptations` — manual merge (9 conflict
   sites; v0.5.21 already carries #36340 GenMHA/out-buffer infra;
   `is_cp_v2_active`→`is_cp_active` rename; fake backend missing
   `prefill_uses_native_fp4` handled via getattr; duplicate `mask` kwarg in
   `run_group` removed). Without this, nvfp4 KV + NEXTN crashes in the
   FlashInfer extend path.
2. `fix(spec): bind target embed/lm_head into draft before KV pool profiling`
   — the draft shell (embed + lm_head copies, ~1.9 GB/GPU) must be freed
   BEFORE the pool is sized. KV pool 51,456 → 195,968 tokens. Upstream
   equivalent: PR #42472 (against main, class EagleDraftWorker).
3. `feat(vit): item-exact encoder chunking via SGLANG_VIT_CHUNK_TOKENS` —
   caps vision-tower activation spikes on 16 GB cards; exact by construction
   (block-diagonal cu_seqlens attention). Production value: 8192.

## deploy/g40-nvfp4 — companion artifacts
- `flashinfer-0.6.18-xqa-qfold.patch` — the cold-start JIT killer fix, on the
  flashinfer side (applied to `flashinfer/xqa.py` inside the venv, NOT in this
  repo). SGLang's NEXTN draft-extend routes the *runtime* max-q through the
  XQA module cache key; for q*head_group_ratio>32 the q never enters compile
  flags, so every new q compiles a byte-identical module under a new name.
  A fresh long-context request (q=1061) triggered nvcc on the request path and
  nvcc segfaulted → scheduler died → engine restart (2026-10-04 19:57). The
  patch folds every ineligible q onto `32//hgr+1` (6 at hgr=6): bounded set
  {1,2,3,4,5,6} modules, zero runtime JIT afterwards. Verified on-box:
  nvdisasm of the folded cubin is instruction-identical to the per-q builds;
  3x ~30K-token spec-decode stress with zero compiler activity in journal.
- `warm_xqa.py` — CPU-only pre-compiler for exactly those 6 modules
  (page_size 64 / head_dim 256 / hgr 6 / bf16-in / u8 nvfp4 KV). Mirrors
  sglang's `set_cuda_arch()` FLASHINFER_CUDA_ARCH_LIST ("<cap>a") AFTER the
  flashinfer env import so artifacts land in the same workspace dir with the
  same ninja lines — the engine's boot-time build() then no-ops. Wired into
  `start_sglang.sh` before `exec` (non-fatal).

## Recipe (start_sglang.sh)
`--kv-cache-dtype nvfp4 --prefill-attention-backend flashinfer
--decode-attention-backend trtllm_mha --page-size 64
--disable-prefill-cuda-graph --disable-custom-all-reduce
--max-total-tokens 160000 --mem-fraction-static 0.93
--speculative-algorithm NEXTN --speculative-num-steps 2
--speculative-eagle-topk 1 --speculative-num-draft-tokens 3`
env: `MAX_JOBS=2`, `NCCL_P2P_LEVEL=PHB` (patched P2P driver),
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`SGLANG_VIT_CHUNK_TOKENS=8192`, HiCache 20G write_through.
