"""v17: ZERO-PERTURBATION poison verdict.

v15c lesson: heavy pre-submit device reads desynchronised the eviction race
and suppressed the trigger (that arm never corrupted). v17 does the bare
minimum:

 - D2H submit: record nothing except (host_pool, draft_layer, hi_cpu clone,
   di clone). Hashing happens on the NEXT submit after d2h-stream sync.
 - H2D submit (draft entry): device_to_host_stream.synchronize() only
   (cheap, no reads), then poison draft device rows (fast fills), snapshot
   host hash, submit proceeds.
 - Next submit boundary: h2d-stream sync + event sync, then byte-compare
   host snapshot vs device rows:
     LOAD-OK  host==dev            (load copies faithfully; save must be dirty)
     LOAD-NOOP dev==poison         (load kernel never wrote draft rows)
     LOAD-BAD  dev != host         (load kernel wrote wrong bytes)
   SAVE check (from recorded D2H pairs):
     SAVE-OK      host == device-source-now (careful: device rows may have
                  been rewritten by forward since! -> we ALSO hash the device
                  draft rows at H2D-poison time as 'dev_at_load', and compare
                  host_saved with dev_at_load: if equal, save+nothing-else
                  touched; the definitive signal is LOAD verdict + host_saved
                  vs host_at_load stability.)
"""
import ast

P = "/home/jerry/sg-env-0521-nvfp4mtp/lib/python3.12/site-packages/sglang/srt/mem_cache/l2_transfer.py"
src = open(P, encoding="utf-8").read()
assert "G40 FORENSIC" not in src and "G40 RACE" not in src
open(P + ".bak-forensic", "w", encoding="utf-8").write(src)

helper = '''

# G40 FORENSIC 10/08 v17 -- minimal-perturbation poison verdict
import os as _f_os
import time as _f_time

_F_LOG = _f_os.path.expanduser("~/g40-slow-case/instr/forensic.log")
_F_POISON = 0x5A
_F_D2H = []  # (hp, hl, hi_cpu, tag)
_F_H2D = []  # (dp, di, hh_before, host_after_poison, ph, slot, n, t_submit)


def _f_log(msg):
    try:
        with open(_F_LOG, "a") as fh:
            fh.write(msg + "\\n")
    except Exception:
        pass


def _f_md5(x):
    import hashlib

    return hashlib.md5(x.contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def _f_hrow(hp, hl, idx_cpu, n=256):
    x = hp.k_data_refs[hl][idx_cpu].contiguous().view(torch.uint8)
    return _f_md5(x.reshape(x.shape[0], -1)[:, :n])


def _f_drow(dp, di, n=256):
    x = dp.k_buffer[0][di].contiguous().view(torch.uint8)
    return _f_md5(x.reshape(x.shape[0], -1)[:, :n])


def _f_drain(eng):
    while _F_D2H:
        hp, hl, hi_cpu, di_gpu = _F_D2H.pop(0)
        try:
            eng.device_to_host_stream.synchronize()
            hh = _f_hrow(hp, hl, hi_cpu)
            _f_log(
                f"V17 SAVED slot={int(hi_cpu[0])} n={len(hi_cpu)} host_after={hh}"
            )
        except Exception as e:
            _f_log(f"V17 SAVE ERR {type(e).__name__}: {e}")
    while _F_H2D:
        dp, di, hb, ha, ph, slot, n, ts = _F_H2D.pop(0)
        try:
            eng.host_to_device_stream.synchronize()
            after = _f_drow(dp, di)
            v = (
                "LOAD-OK"
                if after == ha
                else ("LOAD-NOOP" if after == ph else "LOAD-BAD")
            )
            _f_log(
                f"V17 LOAD slot={slot} n={n} host_before={hb} host_afterpoison={ha} "
                f"poison={ph} dev_after={after} {v} age={round(_f_time.time()-ts,1)}"
            )
        except Exception as e:
            _f_log(f"V17 LOAD ERR {type(e).__name__}: {e}")


def _f_d2h_note(eng, transfers):
    _f_drain(eng)
    for t in transfers:
        try:
            if getattr(t, "is_draft", False):
                continue
            hp = t.host_pool
            if getattr(hp, "layer_num", 0) < 2:
                continue
            hi = t.host_indices
            hi_cpu = (hi.cpu() if hi.is_cuda else hi).clone()
            _F_D2H.append((hp, hp.layer_num - 1, hi_cpu, t.device_indices.clone()))
        except Exception:
            pass


def _f_h2d_poison(eng, transfers):
    _f_drain(eng)
    for t in transfers:
        try:
            if not getattr(t, "is_draft", False):
                continue
            hp = t.host_pool
            dp = t.device_pool
            hi = t.host_indices
            hi_cpu = hi.cpu() if hi.is_cuda else hi
            di = t.device_indices
            hb = _f_hrow(hp, hp.layer_num - 1, hi_cpu)  # host now (pre-poison)
            eng.device_to_host_stream.synchronize()
            with torch.no_grad():
                dp.k_buffer[0].index_fill_(0, di, _F_POISON)
                dp.v_buffer[0].index_fill_(0, di, _F_POISON)
            ha = _f_hrow(hp, hp.layer_num - 1, hi_cpu)  # host unchanged?
            ph = _f_drow(dp, di)
            _F_H2D.append(
                (dp, di.clone(), hb, ha, ph, int(hi_cpu[0]), len(hi_cpu), _f_time.time())
            )
        except Exception as e:
            _f_log(f"V17 PRE ERR {type(e).__name__}: {e}")
'''
anchor = "class L2TransferEngine:"
assert src.count(anchor) == 1
src = src.replace(anchor, helper + "\n" + anchor, 1)

old_d2h = """        with self._submission(
            transfers, self.device_to_host_stream, "device_to_host"
        ) as (transfers, completion):"""
assert src.count(old_d2h) == 1
src = src.replace(
    old_d2h,
    "        try:\n            _f_d2h_note(self, transfers)\n        except Exception:\n            pass\n"
    + old_d2h,
    1,
)

old_h2d = """        with self._submission(
            transfers, self.host_to_device_stream, "host_to_device", start_event
        ) as (transfers, completion):"""
assert src.count(old_h2d) == 1
src = src.replace(
    old_h2d,
    "        try:\n            _f_h2d_poison(self, transfers)\n        except Exception as _e:\n            _f_log(f'V17 PRE ERR2 {type(_e).__name__}: {_e}')\n"
    + old_h2d,
    1,
)

ast.parse(src)
open(P, "w", encoding="utf-8").write(src)
print("v17 installed (minimal perturbation)")
