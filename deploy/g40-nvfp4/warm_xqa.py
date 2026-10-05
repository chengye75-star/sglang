#!/usr/bin/env python3
"""Pre-compile the fixed set of XQA kernels used by the G40 NVFP4+NEXTN recipe,
so the request path never triggers nvcc JIT (which randomly segfaulted and took
down the scheduler on 2026-10-04).

Runs CPU-only: CompilationContext reads device capability without allocating GPU
memory, and flashinfer's JIT build is just nvcc+ninja producing a .so. Safe to
run while production is serving. Idempotent: ninja no-ops when artifacts exist.

Must be invoked with the SAME env the server uses (FLASHINFER_WORKSPACE_BASE ->
~/.cache/sglang, CUDA_HOME, MAX_JOBS=2). Call gen_xqa_module + build directly to
bypass the q-fold patch and force every q variant we rely on.
"""
import os, sys, torch

# Match the server's cache location BEFORE importing flashinfer (it resolves
# its workspace at import time).
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", os.path.expanduser("~/.cache/sglang"))

import flashinfer.jit.env as jit_env  # noqa: E402  (workspace dir fixed HERE, from auto-detected cap -> "120f")

# Now (AFTER env import, exactly like sglang's set_cuda_arch() which runs
# before any CompilationContext use) pin the target arch the same way the
# engine does: "<cap>a" for major>=9. CompilationContext reads this at call
# time, so we build compute_120a into the 120f workspace dir — byte-identical
# ninja lines mean the engine's boot build() no-ops instead of rebuilding.
# Capability comes from nvidia-smi so this script never creates a CUDA context.
if not os.environ.get("FLASHINFER_CUDA_ARCH_LIST"):
    import subprocess
    cap = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()[0].strip()
    os.environ["FLASHINFER_CUDA_ARCH_LIST"] = (
        cap + ("a" if int(cap.split(".")[0]) >= 9 else "")
    )
print(f"workspace = {jit_env.FLASHINFER_JIT_DIR}", flush=True)
print(f"FLASHINFER_CUDA_ARCH_LIST={os.environ['FLASHINFER_CUDA_ARCH_LIST']}", flush=True)

from flashinfer.jit.xqa import gen_xqa_module  # noqa: E402

# (page_size, head_dim, head_group_ratio) for this model/deploy:
#   - Qwen3.8-27B attn head_dim=256, GQA group ratio 6, page_size 64 (B5/M2SP recipe).
# kv cache dtype for nvfp4 = uint8; input/output bf16.
BF16 = torch.bfloat16
KV_U8 = torch.uint8  # nvfp4 packed KV

CONFIGS = [dict(page_size=64, head_dim=256, head_group_ratio=6)]

# q variants we must have on disk:
#   q=1               -> non-spec decode (use_spec_dec_False)
#   q in {2,3,4,5}    -> SWAP_AB specialized (SPEC_Q_SEQ_LEN defined)
#   folded q          -> covers ALL q>=6 in one module (the q-fold patch routes
#                        every ineligible q here; 32//6+1 == 6)
QS = [1, 2, 3, 4, 5, 6]

def one(cfg, q):
    spec = gen_xqa_module(
        input_dtype=BF16,
        kv_cache_dtype=KV_U8,
        output_dtype=BF16,
        use_sliding_window=False,
        q_seq_len=q,
        use_ragged_q=False,
        **cfg,
    )
    so = spec.jit_library_path
    existed = so.exists()
    spec.build_and_load()
    return spec.name, existed, so.exists()

def main():
    print(f"workspace = {jit_env.FLASHINFER_JIT_DIR}", flush=True)
    ok = True
    for cfg in CONFIGS:
        for q in QS:
            try:
                name, existed, built = one(cfg, q)
                tag = "cached" if existed else "built  "
                print(f"[{tag}] q={q:<3} {name}.so -> {built}", flush=True)
                ok = ok and built
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"[FAIL  ] q={q}: {type(e).__name__}: {e}", flush=True)
    print("WARM", "OK" if ok else "INCOMPLETE", flush=True)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
