"""v18: hook load_to_device_per_layer on MHATokenToKVPoolHost (draft calls only).

v17 proved: submit loop DOES issue the draft transfer (instrument said 0D>16
earlier) yet the device draft rows are untouched => the function returned
early or took a no-op branch. Capture the ACTUAL args the engine passes and
which branch is taken, for is_draft=True calls only (few per recall).

Log: layer_id arg, io_backend, layout, can_use_jit, is_draft kwarg,
host_pool.device_pool is None?, _is_device_layer_owned(draft, layer_id)?,
then before/after hash of ONE device row to see if it wrote.
"""
import ast

SP = "/home/jerry/sg-env-0521-nvfp4mtp/lib/python3.12/site-packages/sglang/srt/mem_cache"
P = SP + "/pool_host/mha.py"
src = open(P, encoding="utf-8").read()
assert "G40 FORENSIC" not in src
open(P + ".bak-forensic", "w", encoding="utf-8").write(src)

helper = '''

# G40 FORENSIC 10/08 v18 -- draft-branch tracer on MHATokenToKVPoolHost
import os as _f_os


def _f_log(msg):
    try:
        with open(
            _f_os.path.expanduser("~/g40-slow-case/instr/forensic.log"), "a"
        ) as fh:
            fh.write(msg + "\\n")
    except Exception:
        pass


_F_COUNT = [0]


def _f_wrap_load(cls):
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
        if is_draft and _F_COUNT[0] < 24:
            _F_COUNT[0] += 1
            try:
                import hashlib

                def h(buf, idx):
                    x = buf[idx[:4]].contiguous().view(torch.uint8)
                    return hashlib.md5(
                        x.reshape(x.shape[0], -1)[:, :256]
                        .cpu()
                        .numpy()
                        .tobytes()
                    ).hexdigest()[:10]

                before = h(device_pool.k_buffer[0], device_indices)
                owned = (
                    self._is_device_layer_owned(device_pool, layer_id)
                    if hasattr(self, "_is_device_layer_owned")
                    else "n/a"
                )
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
                after = h(device_pool.k_buffer[0], device_indices)
                _f_log(
                    f"V18 n{_F_COUNT[0]} layer_id={layer_id} io={io_backend} "
                    f"layout={self.layout} jit={self.can_use_jit} is_draft={is_draft} "
                    f"owned={owned} n_idx={device_indices.numel()} "
                    f"host_idx_dev={host_indices.device} "
                    f"row_before={before} row_after={after} "
                    f"{'WROTE' if before != after else 'NO-OP'}"
                )
                return ret
            except Exception as e:
                _f_log(f"V18 ERR {type(e).__name__}: {e}")
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


def _f_install():
    try:
        _f_wrap_load(MHATokenToKVPoolHost)
        _f_log("V18 wrapper installed (this process)")
    except Exception as e:
        _f_log(f"V18 INSTALL ERR {type(e).__name__}: {e}")


_f_install()
'''
# append at end of module (class defined above)
src = src + "\n" + helper
ast.parse(src)
open(P, "w", encoding="utf-8").write(src)
print("v18 installed on", P)
