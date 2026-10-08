"""Synth #3: byte-exact replication of production pool geometry (from geom.log).

D2H real config:
  host  MHATokenToKVPoolHost dtype=uint8 head_num=2 head_dim=256 layout=page_first
        k_buffer=(H,17,2,256)  -> per-token per-layer host row = 512 bytes
        token_stride=512, layout_dim=8704, can_jit=True, can_wb_jit=True (staged!)
  device 16-layer pool: store_dtype=uint8 row_dim=512 head_dim=256 v_head_dim=256
        k_buffer[j]=(S,2,128) -> 256 bytes/token
  draft 1-layer pool, same geometry; packed as host layer 16.

Sentinels are uint8 patterns so correctness does not depend on numeric dtypes.
Compares host row content after backup, then draft-only loadback.
"""
import sys

import torch

sys.path.insert(0, "/home/jerry/sg-env-0521-nvfp4mtp/lib/python3.12/site-packages")

from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

DEV = torch.device("cuda:0")
PAGE = 64
DSIZE = 2048  # device tokens (pools)
HEADS, ROW_BYTES = 2, 128  # device k row: (2,128) uint8
TL = 16


class FakePool:
    def __init__(self, layer_num, size=DSIZE + PAGE):
        self.layer_num = layer_num
        self.start_layer = 0
        self.end_layer = layer_num
        self.store_dtype = torch.uint8
        self.head_num = HEADS
        self.head_dim = 256
        self.v_head_dim = 256
        self.row_dim = 512
        self.v_row_dim = 512
        self.hicache_write_back_staging = None
        self.device = DEV
        self.size = size
        self.page_size = PAGE
        self.k_buffer = [
            torch.zeros(size, HEADS, ROW_BYTES, dtype=torch.uint8, device=DEV)
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.zeros(size, HEADS, ROW_BYTES, dtype=torch.uint8, device=DEV)
            for _ in range(layer_num)
        ]
        self.k_data_ptrs = torch.tensor(
            [t.data_ptr() for t in self.k_buffer], dtype=torch.uint64, device=DEV
        )
        self.v_data_ptrs = torch.tensor(
            [t.data_ptr() for t in self.v_buffer], dtype=torch.uint64, device=DEV
        )


def fill(buf, t, value):
    # buf rows: (size, heads, bytes)
    buf[t] = value


def main():
    tgt = FakePool(TL)
    drf = FakePool(1)
    hp = MHATokenToKVPoolHost(
        tgt,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=PAGE,
        layout="page_first",
        mtp_draft_device_pools=(drf,),
        pool_label="kv",
    )
    print("host layer_num", hp.layer_num, "size", hp.size, "element_dim", hp.element_dim,
          "token_stride", hp.token_stride_size, "layout_dim", hp.layout_dim)
    print("can_jit", hp.can_use_jit, "can_wb_jit", hp.can_use_write_back_jit)

    n = 8 * PAGE
    t = torch.arange(n, dtype=torch.int64)
    tdev = t.to(DEV)

    for j in range(TL):
        fill(tgt.k_buffer[j], t, 0xA0 + j)
        fill(tgt.v_buffer[j], t, 0xB0 + j)
    fill(drf.k_buffer[0], t, 0x11)
    fill(drf.v_buffer[0], t, 0x22)

    hp.backup_from_device_all_layer(tgt, t, tdev, "kernel")
    torch.cuda.synchronize()

    hK16 = hp.k_data_refs[hp.layer_num - 1][t]
    hV16 = hp.v_data_refs[hp.layer_num - 1][t]
    print("host16 K bytes uniq:", hK16.unique().tolist(), "V:", hV16.unique().tolist())
    hK0 = hp.k_data_refs[0][t]
    print("host0  K bytes uniq:", hK0.unique().tolist())

    # wipe draft device, loadback draft only (exactly like submit_host_to_device)
    drf.k_buffer[0].zero_()
    drf.v_buffer[0].zero_()
    hp.load_to_device_per_layer(
        drf, tdev, tdev, hp.layer_num - 1, "kernel", is_draft=True
    )
    torch.cuda.synchronize()
    got_k = drf.k_buffer[0][t]
    got_v = drf.v_buffer[0][t]
    print("draft device K uniq after load:", got_k.unique().tolist(), "(want [17])")
    print("draft device V uniq after load:", got_v.unique().tolist(), "(want [34])")
    if got_k.unique().tolist() == [0x11] and got_v.unique().tolist() == [0x22]:
        print("RESULT: production-geometry round-trip OK")
    else:
        print("RESULT: OFFLINE REPRODUCED with production geometry + jit/staged paths")
        # byte-level view: first 4 tokens, first 8 bytes
        print("device draft K rows head:", got_k[:2, 0, :8].tolist())
        for j in range(hp.layer_num):
            print(f"  host row {j} head:", hp.k_data_refs[j][t[:2], 0, :8].tolist())


if __name__ == "__main__":
    main()
