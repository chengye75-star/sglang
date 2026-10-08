"""synth20: position-encoded verdict on the nvfp4 draft H2D geometry.

Engine-measured geometry (geom.log):
  host draft layer view: (H, 2, 256) uint8  -> 512 B/token
  device draft k_buffer: (S, 2, 128) uint8  -> 256 B/token
  jit one-layer called with element_dim=512 (host element_dim)

Host rows are filled with POSITION ENCODING: byte value = (token*7 + off*3) % 250 + 1.
Device starts poison=0x5A everywhere. After jit1:
  expected-if-correct: device row di has bytes [enc(tok, 0..255)]
  check where each byte actually landed: map every non-poison byte of dst to
  (row, col) and test against enc of its host source token.
Two index sets: SMALL (di=0..63) and LARGE (di=80000..80063, engine-like).
"""
import sys

import torch

sys.path.insert(0, "/home/jerry/sg-env-0521-nvfp4mtp/lib/python3.12/site-packages")
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_one_layer as jit1,
    use_hicache_tma_kernel,
)

DEV = torch.device("cuda:0")
PAGE = 64
H = 300000  # host rows
S = 150016  # device rows (engine size)
POISON = 0x5A


def enc(tok, off):
    return (tok * 7 + off * 3) % 250 + 1


def main():
    hp = torch.zeros(H, 17, 2, 256, dtype=torch.uint8, device=DEV)
    kref = hp.transpose(0, 1)[16]  # (H,2,256): the engine's host draft view
    dp_k = torch.full((S, 2, 128), POISON, dtype=torch.uint8, device=DEV)
    dp_v = torch.full((S, 2, 128), POISON, dtype=torch.uint8, device=DEV)
    vref = hp.transpose(0, 1)[16]  # v host view (reuse same; v_cache_src=kref in engine)
    print(
        "tma?",
        use_hicache_tma_kernel(
            element_size=512, block_quota=8, page_size=PAGE
        ),
    )

    for base, tag in ((100, "SMALL"), (200000, "LARGE")):
        n = 64
        hi = torch.arange(base, base + n, dtype=torch.int64, device=DEV)
        kref[base : base + n] = 0  # reset
        # position-encode host rows
        tokv = torch.arange(base, base + n, dtype=torch.int64, device=DEV)
        offs = torch.arange(512, dtype=torch.int64, device=DEV)
        vals = ((tokv[:, None] * 7 + offs[None, :] * 3) % 250 + 1).to(torch.uint8)
        kref[base : base + n] = vals.reshape(n, 2, 256)
        torch.cuda.synchronize()

        di = torch.arange(0, n, dtype=torch.int64, device=DEV)
        if tag == "LARGE":
            # engine-like: big device indices near the tail
            di = torch.arange(S - 2 * n, S - n, dtype=torch.int64, device=DEV)
            kref[base : base + n] = vals.reshape(n, 2, 256)
            # fresh poison in that region
            dp_k[di] = POISON
            dp_k[di + n] = POISON
            torch.cuda.synchronize()

        jit1(
            k_cache_dst=dp_k,
            v_cache_dst=dp_v,
            k_cache_src=kref,
            v_cache_src=vref,
            indices_dst=di,
            indices_src=hi,
            element_dim=512,
            page_size=PAGE,
        )
        torch.cuda.synchronize()

        # analysis: for each written device row r (any byte != poison), find which
        # host token's encoding best matches its 256 bytes
        region_lo = int(min(di[0], di[-1])) - 4
        region_hi = int(max(di[0], di[-1])) + 2 * n + 4
        seg = dp_k[region_lo:region_hi].reshape(-1, 256)  # per-256B chunks
        written_rows = []
        for r in range(seg.shape[0]):
            row = seg[r].cpu().numpy()
            nz = (row != POISON).sum()
            if nz == 0:
                continue
            # try to match against each host token encoding
            match = None
            for tok in range(base - 8, base + n + 8):
                exp = ((tok * 7 + torch.arange(256) * 3) % 250 + 1).numpy()
                same = (row == exp).sum()
                if same > 200:
                    match = (tok, int(same))
                    break
                exp2 = ((tok * 7 + (torch.arange(256) + 256) * 3) % 250 + 1).numpy()
                same2 = (row == exp2).sum()
                if same2 > 200:
                    match = (tok, int(same2), "UPPER-half")
                    break
            written_rows.append((region_lo + r, int(nz), match))
        correct = sum(
            1
            for (r, nz, m) in written_rows
            if m and len(m) == 2 and int(di[0]) <= r <= int(di[-1]) and nz == 256
        )
        print(f"== {tag}: di=[{int(di[0])}..{int(di[-1])}] written chunks={len(written_rows)} fully-correct-chunks={correct}")
        for item in written_rows[:10]:
            print("   row", item[0], "nz", item[1], "match", item[2])


if __name__ == "__main__":
    main()
