#!/usr/bin/env python3
"""Dual-session eviction/loadback reproduction (incident 10/07 21:38 shape).

Per cycle:
  warm   docA (~137K tok) max_new=1        -> A resident, usage ~0.86
  press  docB (rest, ~60K tok) max_new=8000 -> prefill+decode overflow the
         160K pool => A's pages evicted to host under write_through
  recall docA+" END" max_new=1             -> ~137K host->device loadback,
         evicting B's finished-but-tree-held pages in the same breath
  probe  greedy count task                 -> accept_mean from engine log

Corruption signal: accept_mean < 1.3 with rows > 3 (target answers stay fine).
Usage: dual.py [NCYCLES]
"""
import json, os, re, sys, time, urllib.request

PORT = os.environ.get("PORT", "8000")
URL = f"http://localhost:{PORT}/generate"
LOG = os.path.expanduser("~/g40-slow-case/boot-packed.log")
full = open(os.path.expanduser("~/doc197k.txt"), errors="ignore").read()
L = len(full)
docA = full[: int(L * 0.69)]
docB = full[int(L * 0.69):]

def post(text, mx, timeout=3600):
    body = json.dumps({"text": text,
                       "sampling_params": {"max_new_tokens": mx, "temperature": 0}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def accept_since(off):
    with open(LOG, "rb") as f:
        f.seek(off)
        tail = f.read().decode(errors="ignore")
    vals = [float(m.group(1)) for m in re.finditer(r"accept len: ([0-9.]+),", tail)]
    vals = [v for v in vals if v >= 0.5]
    if not vals:
        return -1.0, 0
    return sum(vals) / len(vals), len(vals)

def usage():
    with urllib.request.urlopen(f"http://localhost:{PORT}/metrics", timeout=15) as f:
        for lb in f:
            l = lb.decode()
            if l.startswith("sglang:token_usage{"):
                return float(l.split()[-1])
    return -1.0

NCYC = int(sys.argv[1]) if len(sys.argv) > 1 else 5
print(f"docs: A={len(docA)}ch B={len(docB)}ch", flush=True)
for i in range(NCYC):
    print(f"=== cycle {i}", flush=True)
    t0 = time.time(); post(docA, 1)
    print(f"  warm {time.time()-t0:.0f}s usage={usage():.2f}", flush=True)
    off0 = os.path.getsize(LOG)
    t0 = time.time(); post(docB, 8000)
    am0, rows0 = accept_since(off0)
    print(f"  press {time.time()-t0:.0f}s accept_during={am0:.2f}({rows0}) usage_after={usage():.2f}", flush=True)
    off2 = os.path.getsize(LOG)
    t0 = time.time(); post(docA + " END", 1)
    print(f"  recall {time.time()-t0:.0f}s usage={usage():.2f}", flush=True)
    post("Dual-repro-%02d count 1..250 one per line:\n1\n2\n3" % i, 350)
    am, rows = accept_since(off2)
    print(f"  probe accept_mean={am:.2f} rows={rows}", flush=True)
    if 0 <= am < 1.3 and rows > 3:
        print(f"CORRUPTION_REPRODUCED cycle={i}", flush=True)
        break
print("DUAL_DONE", flush=True)
