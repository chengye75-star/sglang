"""v19: WRITE-FOOTPRINT mapper — final conviction.

v17: device draft rows == poison after load (no writes at checked rows).
v18: engine calls jit one_layer(layer_id=16, is_draft=True) and NOTHING is
written at device_indices rows.

v19 poisons the ENTIRE draft device buffer (K and V) before the draft H2D
transfer, then maps where ANY byte landed afterwards:
  - footprint == device_indices rows        -> load fine
  - footprint rows == {2*i for i in di}     -> element_dim/view 2x collapse
  - empty footprint (all poison)            -> kernel skipped (OOB guard)
  - writes beyond buffer end never visible  -> footprint capped, report max
Also record: buffer bytes, element_dim, di range. One-shot x4.
"""
import ast

SP = "/home/jerry/sg-env-0521-nvfp4mtp/lib/python3.12/site-packages/sglang/srt/mem_cache"
P = SP + "/pool_host/mha.py"
try:
    src = open(P + ".bak-forensic", encoding="utf-8").read()
except FileNotFoundError:
    src = open(P, encoding="utf-8").read()
assert "G40 FORENSIC" not in src
open(P + ".bak-forensic", "w", encoding="utf-8").write(src)

helper = '''

# G40 FORENSIC 10/08 v19 -- full-buffer write-footprint mapper
import os as _f_os

_F_LOG = _f_os.path.expanduser("~/g40-slow-case/instr/forensic.log")
_F_V19 = [0]


def _f_log(msg):
    try:
        with open(_F_LOG, "a") as fh:
            fh.write(msg + "\\n")
    except Exception:
        pass


def _f_wrap_load_v19(cls):
    orig = cls.load_to_device_per_layer

    def traced(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if (
            is_draft
            and getattr(device_pool, "k_buffer", None)
            and _F_V19[0] < 6
        ):
            _F_V19[0] += 1
            n = _F_V19[0]
            try:
                dev = device_pool.k_buffer[0].device
                kb = device_pool.k_buffer[0]
                vb = device_pool.v_buffer[0]
                with torch.no_grad():
                    kb.view(-1).fill_(0x5A)
                    vb.view(-1).fill_(0x5A)
                torch.cuda.synchronize(dev)
                di = device_indices
                ret = orig(
                    self,
                    device_pool,
                    host_indices,
                    device_indices,
                    layer_id,
                    io_backend,
                    is_draft=is_draft,
                )
                torch.cuda.synchronize(dev)
                flat = kb.view(-1).ne(0x5A)  # bytes written
                w = flat.nonzero().flatten()
                nbytes = kb.numel()
                if w.numel() == 0:
                    _f_log(
                        f"V19 n{n} FOOTPRINT-EMPTY buffer_bytes={nbytes} "
                        f"di=[{int(di.min())}..{int(di.max())}] n={di.numel()} "
                        f"layer_id={layer_id} element_dim={self.element_dim}"
                    )
                else:
                    bmin = int(w[0])
                    bmax = int(w[-1])
                    rows = (w // 256).unique()
                    rows_head = rows[:8].tolist()
                    ratio = float((rows % 2 == 0).float().mean()) if rows.numel() else -1
                    _f_log(
                        f"V19 n{n} FOOTPRINT bytes={w.numel()}/{nbytes} "
                        f"byte_range=[{bmin}..{bmax}] rows=[{rows[0]}..{rows[-1]}] "
                        f"rows_head={rows_head} even_ratio={ratio:.2f} "
                        f"di=[{int(di.min())}..{int(di.max())}] n={di.numel()} "
                        f"element_dim={self.element_dim} layer_id={layer_id}"
                    )
                return ret
            except Exception as e:
                import traceback

                _f_log(f"V19 ERR {type(e).__name__}: {e}")
                try:
                    _f_log(traceback.format_exc(limit=4))
                except Exception:
                    pass
        return orig(
            self,
            device_pool,
            host_indices,
            device_indices,
            layer_id,
            io_backend,
            is_draft=is_draft,
        )

    cls.load_to_device_per_layer = traced
    return cls


try:
    _f_wrap_load_v19(MHATokenToKVPoolHost)
    _f_log("V19 wrapper installed")
except Exception as e:
    _f_log(f"V19 INSTALL ERR {type(e).__name__}: {e}")
'''
src = src + "\n" + helper
ast.parse(src)
open(P, "w", encoding="utf-8").write(src)
print("v19 installed on", P)
