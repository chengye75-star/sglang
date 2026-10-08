#!/usr/bin/env python3
"""Soak hammer: replicate the incident's per-cycle state (usage ~85%, mamba 8 slots)
with strictly ONE request in flight (mamba-safe), 10 cycles, probe each cycle.

Cycle i: prefill fresh slice full[i*70K : i*70K+540K chars] (~129K tok, chunked
prefill stashes -> mamba 8/10 high), max_new_tokens=120 (decode growth pushes
pool past eviction watermark). Then count-probe. doc0 kept warm (single resident)
so each new doc evicts its pages.

Usage: soak.py START_CYCLE END_CYCLE
"""
import json, os, re, subprocess, sys, time, urllib.request

PORT = int(os.environ.get("PORT", "8000"))
URL = f"http://localhost:{PORT}/generate"
LOG = os.path.expanduser("~/g40-slow-case/boot-packed.log")
full = open(os.path.expanduser("~/doc197k.txt"), errors="ignore").read()

def post(payload, timeout=2400):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def log_tail_off():
    return os.path.getsize(LOG)

def accept_since(off):
    with open(LOG, "rb") as f:
        f.seek(off)
        tail = f.read().decode(errors="ignore")
    vals = [float(m.group(1)) for m in re.finditer(r"accept len: ([0-9.]+), accept rate", tail)]
    vals = [v for v in vals if v >= 0.5]
    return (sum(vals) / len(vals)) if vals else -1, len(vals)

lo, hi = int(sys.argv[1]), int(sys.argv[2])
# warm doc0 resident first, single request at a time
anchor = full[: int(len(full) * 0.66)]
t0 = time.time()
post({"text": anchor, "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
print(f"WARM0 chars={len(anchor)} {time.time()-t0:.0f}s", flush=True)

for i in range(lo, hi):
    s = i * 70000
    sl = full[s : s + 540000]
    off = log_tail_off()
    t0 = time.time()
    try:
        post({"text": f"CYCLE{i:02d}. " + sl, "sampling_params": {"max_new_tokens": 120, "temperature": 0}})
    except Exception as e:
        print(f"CYCLE {i:02d} ENGINE_ERR {type(e).__name__} {e} after {time.time()-t0:.0f}s", flush=True)
        break
    am, rows = accept_since(off)
    # mamba watermark from decode lines this cycle
    with open(LOG, "rb") as f:
        f.seek(off)
        mm = re.findall(r"mamba usage: ([0-9.]+)", f.read().decode(errors="ignore"))
    mmax = max((float(x) for x in mm), default=-1)
    print(f"CYCLE {i:02d} {time.time()-t0:.0f}s accept_mean={am:.2f} rows={rows} mamba_max={mmax:.2f}", flush=True)

# final: recall anchor + probe health
off = log_tail_off()
t0 = time.time()
try:
    post({"text": anchor + " THEEND", "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
    print(f"RECALL0 {time.time()-t0:.0f}s", flush=True)
except Exception as e:
    print(f"RECALL0 ERR {type(e).__name__} {e}", flush=True)
off2 = log_tail_off()
try:
    post({"text": "Soak-final-count 1..200, one per line:\n1\n2\n3",
          "sampling_params": {"max_new_tokens": 300, "temperature": 0}})
    am, rows = accept_since(off2)
    print(f"FINAL accept_mean={am:.2f} rows={rows}", flush=True)
except Exception as e:
    print(f"FINAL ERR {type(e).__name__} {e}", flush=True)
print("SOAK_DONE", flush=True)
