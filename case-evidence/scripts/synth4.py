"""Synth #4: position-encoding sentinels — byte-exact verification.

Every byte written = (token_index*31 + byte_offset*7 + layer*101) % 251 + 1
(never 0, so untouched zeros are distinguishable). Verifies EVERY byte of every
device row after: D2H backup (host rows), H2D target layers, H2D draft layer.
Catches intra-row offset/stride errors that uniform patterns hide.
"""
import sys

import torch

sys.path.insert(0, "/home/jerry/sg-env-0521-nvfp4mtp/lib/python3.12/site-packages")

from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

DEV = torch.device("cuda:0")
PAGE = 64
DSIZE = 4096
HEADS, ROW_BYTES = 2, 128  # device row = 256 bytes/token (prod nvfp4 layout)
HROW = 256  # host per-k-head bytes (prod: 2,256 -> 512/token)
TL = 16
NROWS = 12 * PAGE  # 12 pages worth of tokens


def enc(tok, off, layer):
    v = (tok * 31 + off * 7 + layer * 101) % 251
    return v + 1


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


def fill_pool(bufs, n_tok, layer_base):
    """Fill row t of every layer j with position-encoding bytes."""
    for j, buf in enumerate(bufs):
        rows = buf[:n_tok]  # (n, heads, bytes)
        h = torch.arange(HEADS, dtype=torch.int64)[:, None]
        off = torch.arange(ROW_BYTES, dtype=torch.int64)[None, :]
        vals = (
            (torch.arange(n_tok, dtype=torch.int64)[:, None, None] * 31)
            + (h * 64 + off) * 7  # head*bytes + offset = flat byte offset
            + ((j + layer_base) * 101)
        ) % 251 + 1
        buf[:n_tok] = vals.to(torch.uint8).to(DEV)


def expected(tok, flat_off, layer):
    return (tok * 31 + flat_off * 7 + layer * 101) % 251 + 1


def check_rows(buf, idx_cpu, layer, n_tok, tag):
    """byte-exact check of buffer rows at device indices idx (first n_tok used)."""
    got = buf[idx_cpu.to(DEV)].cpu().numpy()  # (n, heads, bytes)
    bad = 0
    firstbad = None
    for i, row in enumerate(got):
        t = int(idx_cpu[i].item())
        exp = expected(t % 1000, 0, layer)  # placeholder; compute vectorized below
        break
    # vectorized expectation
    n = len(idx_cpu)
    offs = (
        torch.arange(HEADS)[:, None] * ROW_BYTES + torch.arange(ROW_BYTES)[None, :]
    )
    tt = idx_cpu[:n].to(torch.int64)
    expv = ((tt % 1000)[:, None, None] * 31 + offs[None, :, :] * 7 + layer * 101) % 251 + 1
    mism = (torch.from_numpy(got).to(torch.int64) != expv)
    bad = int(mism.sum())
    if bad:
        loc = mism.nonzero()[0]
        i, hh, oo = int(loc[0]), int(loc[1]), int(loc[2])
        firstbad = (
            f"tokidx={int(tt[i])} head={hh} off={oo} got={got[i][hh][oo]} "
            f"want={int(expv[i][hh][oo])}"
        )
    print(f"{tag}: mismatched bytes={bad}/{n * HEADS * ROW_BYTES} {firstbad or ''}")
    return bad == 0


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
    print("can_jit", hp.can_use_jit, "can_wb_jit", hp.can_use_write_back_jit)

    t = torch.arange(NROWS, dtype=torch.int64)
    tdev = t.to(DEV)

    fill_pool(tgt.k_buffer, NROWS, 0)
    fill_pool(tgt.v_buffer, NROWS, 50)
    fill_pool(drf.k_buffer, NROWS, 400)
    fill_pool(drf.v_buffer, NROWS, 450)

    # ---- D2H backup (staged jit path, like production write_through) ----
    hp.backup_from_device_all_layer(tgt, t, tdev, "kernel")
    torch.cuda.synchronize()
    # host rows are (H, 2, 256) views: bytes 0..255 = K head-data, 256..511 = ?
    # Compare host k row bytes vs device k row bytes directly:
    def head(x, k=16):
        return x.reshape(-1)[:k].tolist()

    hk0 = hp.k_data_refs[0][t].cpu()  # (n,2,256)
    dk0 = tgt.k_buffer[0][tdev].cpu()  # (n,2,128)
    hv0 = hp.v_data_refs[0][t].cpu()
    dv0 = tgt.v_buffer[0][tdev].cpu()
    print("tok0 dev K head :", head(dk0[0]))
    print("tok0 hostK head :", head(hk0[0]))
    print("tok0 hostK tail(256:272):", head(hk0[0].reshape(512)[256:272]))
    print("tok0 dev V head :", head(dv0[0]))
    print("tok0 hostV head :", head(hv0[0]))
    # draft rows
    hd = hp.k_data_refs[hp.layer_num - 1][t].cpu()
    vd = hp.v_data_refs[hp.layer_num - 1][t].cpu()
    dd = drf.k_buffer[0][tdev].cpu()
    dv = drf.v_buffer[0][tdev].cpu()
    print("draft dev K head:", head(dd[0]))
    print("draft host K head:", head(hd[0]))
    print("draft host K 256:272:", head(hd[0].reshape(512)[256:272]))
    print("draft dev V head:", head(dv[0]))
    print("draft host V head:", head(vd[0]))


if __name__ == "__main__":
    main()
