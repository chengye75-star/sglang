"""v20: TARGET-layer write-location probe (read-only, no poison).

Question: does the non-draft jit one-layer call also land writes at 2*di
rows (512B-element view of a 256B-row pool), or correctly at di?

Method (first 3 target calls only): hash device rows at {di} and at
{2*di, 2*di+1} interleaved set, before and after the real call (the call is
live traffic; we only READ around it, sync current stream). Classify:
  WROTE-AT-DI / WROTE-AT-2DI / NO-OP.
"""
import ast

SP = "/home/jerry/sg-env-0521-nvfp4mtp/lib/python3.12/site-packages/sglang/srt/mem_cache"
P = SP + "/pool_host/mha.py"
try:
    src = open(P + ".bak-forensic", encoding="utf-8").read()
except FileNotFoundError:
    src = open(P, encoding="utf-8").read()
assert "G40 FORENSIC" not in src

helper = '''

# G40 FORENSIC 10/08 v20 -- target write-location probe (read-only)
import os as _f_os


def _f_log20(msg):
    try:
        with open(
            _f_os.path.expanduser("~/g40-slow-case/instr/forensic.log"), "a"
        ) as fh:
            fh.write(msg + "\\n")
    except Exception:
        pass


_F_V20 = [0]


def _f_wrap_load_v20(cls):
    orig = cls.load_to_device_per_layer

    def _rows_hash(buf, idx):
        import hashlib

        if idx.numel() == 0:
            return "empty"
        x = buf[idx].contiguous().view(torch.uint8)
        return hashlib.md5(x.cpu().numpy().tobytes()).hexdigest()[:10]

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
        kb = getattr(device_pool, "k_buffer", None)
        if (
            not is_draft
            and kb is not None
            and layer_id == 0
            and getattr(device_pool, "layer_num", 99) > 1
            and _F_V20[0] < 3
        ):
            _F_V20[0] += 1
            n = _F_V20[0]
            try:
                m = min(device_indices.numel(), 512)
                di = device_indices[:m]
                d2 = (di * 2).clamp(max=kb[0].shape[0] - 2)
                d2v = torch.unique(torch.stack([d2, d2 + 1]).flatten())
                torch.cuda.current_stream().synchronize()
                a1 = _rows_hash(kb[0], di)
                a2 = _rows_hash(kb[0], d2v)
                ret = orig(
                    self,
                    device_pool,
                    host_indices,
                    device_indices,
                    layer_id,
                    io_backend,
                    is_draft=is_draft,
                )
                torch.cuda.current_stream().synchronize()
                b1 = _rows_hash(kb[0], di)
                b2 = _rows_hash(kb[0], d2v)
                v = "WROTE-AT-DI" if a1 != b1 else ("WROTE-AT-2DI" if a2 != b2 else "NO-OP")
                _f_log20(
                    f"V20 n{n} layer0 target di0={int(di[0])} m={m} {v} "
                    f"rowbytes={kb[0].stride(0)} host_elem_dim={self.element_dim}"
                )
                return ret
            except Exception as e:
                _f_log20(f"V20 ERR {type(e).__name__}: {e}")
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
    _f_wrap_load_v20(MHATokenToKVPoolHost)
    _f_log20("V20 wrapper installed")
except Exception as e:
    _f_log20(f"V20 INSTALL ERR {type(e).__name__}: {e}")
'''
src = src + "\n" + helper
ast.parse(src)
open(P, "w", encoding="utf-8").write(src)
print("v20 installed")
