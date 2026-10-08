"""synth23: byte-exact acceptance battery for MHATokenToKVPoolFP4Host.

Engine geometry (from v21 GEOM logs): device k/v rows (S,2,128) uint8,
scale rows (S,2,16) uint8, page_size=64, 16 target layers + 1 packed
draft layer. Never uniform fill — every byte position-encoded so any
indexing bug is detectable.

Battery:
 1. position-encode device payload + scales (target & draft),
 2. page-aligned D2H backup via the fixed host pool,
 3. wipe device,
 4. engine-style H2D loop (layers 0..15 non-draft + tail draft),
 5. assert EVERY byte of payload AND scale equals the encoding,
 6. footprint guard: untouched device rows stay zero.
"""
import importlib.util
import sys

import torch

sys.path.insert(0, "/home/jerry/sg-env-0521-nvfp4mtp/lib/python3.12/site-packages")

spec = importlib.util.spec_from_file_location(
    "mha_fp4",
    "/home/jerry/sg-env-0521-nvfp4mtp/lib/python3.12/site-packages/sglang/srt/mem_cache/pool_host/mha_fp4.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda:0")
PAGE = 64
DSIZE = 4096
TL = 16
N = 128  # two pages


def expected_payload(layer, kind, size):
    tok = torch.arange(size, dtype=torch.int64)[:, None, None]
    head = torch.arange(2, dtype=torch.int64)[None, :, None]
    off = torch.arange(128, dtype=torch.int64)[None, None, :]
    v = (tok * 5 + off * 3 + head * 7 + layer * 13 + kind * 37) % 250 + 1
    return v.to(torch.uint8).to(DEV)


def expected_scale(layer, kind, size):
    tok = torch.arange(size, dtype=torch.int64)[:, None, None]
    off = torch.arange(32, dtype=torch.int64)[None, None, :]
    v = (tok * 11 + off * 13 + layer * 17 + kind * 41) % 250 + 1
    return v.reshape(size, 2, 16).to(torch.uint8).to(DEV)


class FakePool:
    def __init__(self, layer_num, size=DSIZE + PAGE):
        self.layer_num = layer_num
        self.start_layer = 0
        self.end_layer = layer_num
        self.store_dtype = torch.uint8
        self.head_num = 2
        self.head_dim = 256
        self.v_head_dim = 256
        self.row_dim = 512
        self.v_row_dim = 512
        self.hicache_write_back_staging = None
        self.layer_shard_enabled = False
        self.device = DEV
        self.size = size
        self.page_size = PAGE
        self.k_buffer = [
            torch.zeros(size, 2, 128, dtype=torch.uint8, device=DEV)
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.zeros(size, 2, 128, dtype=torch.uint8, device=DEV)
            for _ in range(layer_num)
        ]
        self.k_scale_buffer = [
            torch.zeros(size, 2, 16, dtype=torch.uint8, device=DEV)
            for _ in range(layer_num)
        ]
        self.v_scale_buffer = [
            torch.zeros(size, 2, 16, dtype=torch.uint8, device=DEV)
            for _ in range(layer_num)
        ]
        self.k_data_ptrs = torch.tensor(
            [t.data_ptr() for t in self.k_buffer], dtype=torch.uint64, device=DEV
        )
        self.v_data_ptrs = torch.tensor(
            [t.data_ptr() for t in self.v_buffer], dtype=torch.uint64, device=DEV
        )


def check(name, got, exp, fails):
    if bool((got == exp).all().item()):
        return fails
    d = (got != exp).nonzero()[0]
    ti, hi_, oi = int(d[0]), int(d[1]), int(d[2])
    print(
        f"FAIL {name}: tok={ti} h={hi_} o={oi} "
        f"got={int(got[ti, hi_, oi])} want={int(exp[ti, hi_, oi])}"
    )
    return fails + 1


def main():
    tgt = FakePool(TL)
    drf = FakePool(1)
    hp = mod.MHATokenToKVPoolFP4Host(
        tgt,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=PAGE,
        layout="page_first",
        mtp_draft_device_pools=(drf,),
        pool_label="kv",
    )
    print(
        f"host: spt={hp.size_per_token} stride={hp.token_stride_size} "
        f"element_dim={hp.element_dim} dtype={hp.dtype} layers={hp.layer_num} "
        f"jit={hp.can_use_jit} wbjit={hp.can_use_write_back_jit}"
    )

    t = torch.arange(N, dtype=torch.int64)
    tdev = t.to(DEV)

    for j in range(TL):
        tgt.k_buffer[j][:N] = expected_payload(j, 0, N)
        tgt.v_buffer[j][:N] = expected_payload(j, 1, N)
        tgt.k_scale_buffer[j][:N] = expected_scale(j, 0, N)
        tgt.v_scale_buffer[j][:N] = expected_scale(j, 1, N)
    drf.k_buffer[0][:N] = expected_payload(TL, 0, N)
    drf.v_buffer[0][:N] = expected_payload(TL, 1, N)
    drf.k_scale_buffer[0][:N] = expected_scale(TL, 0, N)
    drf.v_scale_buffer[0][:N] = expected_scale(TL, 1, N)
    torch.cuda.synchronize()

    hp.backup_from_device_all_layer(tgt, t, tdev, "kernel")
    torch.cuda.synchronize()

    for j in range(TL):
        for b in (
            tgt.k_buffer[j],
            tgt.v_buffer[j],
            tgt.k_scale_buffer[j],
            tgt.v_scale_buffer[j],
        ):
            b[:N] = 0
    for b in (
        drf.k_buffer[0],
        drf.v_buffer[0],
        drf.k_scale_buffer[0],
        drf.v_scale_buffer[0],
    ):
        b[:N] = 0
    torch.cuda.synchronize()

    for j in range(TL):
        hp.load_to_device_per_layer(tgt, tdev, tdev, j, "kernel", is_draft=False)
    hp.load_to_device_per_layer(
        drf, tdev, tdev, hp.layer_num - 1, "kernel", is_draft=True
    )
    torch.cuda.synchronize()

    fails = 0
    for j in range(TL):
        fails = check(f"L{j}.k", tgt.k_buffer[j][:N], expected_payload(j, 0, N), fails)
        fails = check(f"L{j}.v", tgt.v_buffer[j][:N], expected_payload(j, 1, N), fails)
        fails = check(f"L{j}.ks", tgt.k_scale_buffer[j][:N], expected_scale(j, 0, N), fails)
        fails = check(f"L{j}.vs", tgt.v_scale_buffer[j][:N], expected_scale(j, 1, N), fails)
    fails = check("draft.k", drf.k_buffer[0][:N], expected_payload(TL, 0, N), fails)
    fails = check("draft.v", drf.v_buffer[0][:N], expected_payload(TL, 1, N), fails)
    fails = check("draft.ks", drf.k_scale_buffer[0][:N], expected_scale(TL, 0, N), fails)
    fails = check("draft.vs", drf.v_scale_buffer[0][:N], expected_scale(TL, 1, N), fails)

    stray = sum(
        int(b[N : N + PAGE].float().sum())
        for b in (
            tgt.k_buffer[0],
            tgt.k_scale_buffer[0],
            drf.k_buffer[0],
            drf.k_scale_buffer[0],
        )
    )
    print(f"fails={fails} stray={stray}")
    print("RESULT:", "FIX-PASS" if fails == 0 and stray == 0 else "FIX-FAIL")


if __name__ == "__main__":
    main()
