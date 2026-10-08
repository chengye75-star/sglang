"""Instrument main's submit_host_to_device loop: log every executed copy."""
import ast
import os

SP = "/home/jerry/sg-env-0521-nvfp4mtp/lib/python3.12/site-packages/sglang/srt/mem_cache"
p = os.path.join(SP, "l2_transfer.py")
src = open(p, encoding="utf-8").read()

if "G40 INSTR" in src:
    print("already instrumented")
    raise SystemExit(0)

open(p + ".bak-instr", "w", encoding="utf-8").write(src)

marker = "        # G40 INSTR\n"

old = '''        with self._submission(
            transfers, self.host_to_device_stream, "host_to_device", start_event
        ) as (transfers, completion):
            primary = transfers[0] if transfers else None
            for layer_id in range(transfer_layer_id_max):'''
assert old in src, "anchor1 missing"
new = marker + '''        import os as _os
        import time as _time

        def _instr(msg):
            try:
                with open(
                    _os.path.expanduser("~/g40-slow-case/instr/instr.log"), "a"
                ) as _fh:
                    _fh.write(f"[{_time.time():.1f}] pid={_os.getpid()} {msg}\\n")
            except Exception:
                pass

        _instr(
            f"H2D_SUBMIT tlm={transfer_layer_id_max} n_tr={len(transfers)} "
            f"draft_tr={sum(1 for t in transfers if t.is_draft)} "
            f"host_layers={transfers[0].host_pool.layer_num if transfers else 0}"
        )
        _exec = []
        with self._submission(
            transfers, self.host_to_device_stream, "host_to_device", start_event
        ) as (transfers, completion):
            primary = transfers[0] if transfers else None
            for layer_id in range(transfer_layer_id_max):'''
src = src.replace(old, new, 1)

old2 = """                    transfer.host_pool.load_to_device_per_layer_physical(
                        transfer.device_pool,
                        transfer.host_indices,
                        transfer.device_indices,
                        local_layer_id,
                        self.io_backend,
                        is_draft=transfer.is_draft,
                    )"""
assert old2 in src, "anchor2 missing"
new2 = """                    _exec.append((layer_id, transfer.is_draft, local_layer_id))
                    transfer.host_pool.load_to_device_per_layer_physical(
                        transfer.device_pool,
                        transfer.host_indices,
                        transfer.device_indices,
                        local_layer_id,
                        self.io_backend,
                        is_draft=transfer.is_draft,
                    )"""
src = src.replace(old2, new2, 1)

old3 = """                if on_layer_done is not None:
                    on_layer_done(layer_id)
        return completion"""
assert src.count(old3) == 1, f"anchor3 count={src.count(old3)}"
new3 = """                if on_layer_done is not None:
                    on_layer_done(layer_id)
        _instr(
            f"H2D_EXEC n={len(_exec)} "
            + " ".join(f"{a}{'D' if b else ''}>{c}" for a, b, c in _exec)
        )
        return completion"""
src = src.replace(old3, new3, 1)

ast.parse(src)
open(p, "w", encoding="utf-8").write(src)
print("l2_transfer instrumented OK")
