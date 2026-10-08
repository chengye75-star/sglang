#!/usr/bin/env python3
"""Count-probe accept-length measurement against local SGLang.

Sends a greedy counting prompt (highly predictable -> healthy NEXTN accept ~3.0),
then averages 'accept len' from the engine log lines emitted during the probe.
Usage: probe_accept.py <log_path> [label]
Prints: PROBE <label> tps=.. accept_mean=.. rows=..
"""
import json, re, sys, time, urllib.request, secrets

LOG = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else "probe"
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8000

ACC = re.compile(r"accept len: ([0-9.]+), accept rate: ([0-9.]+)")

def log_size():
    try:
        return open(LOG, "rb").seek(0, 2)
    except FileNotFoundError:
        return 0

off = log_size()
url = f"http://localhost:{PORT}/generate"
body = json.dumps({
    "text": "Probe-" + secrets.token_hex(6) + ". Count from 1 to 250, one number per line:\n1\n2\n3",
    "sampling_params": {"max_new_tokens": 350, "temperature": 0},
}).encode()
t0 = time.time()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=300))
el = time.time() - t0
ct = r.get("meta_info", {}).get("completion_tokens", 0)
time.sleep(1.5)

with open(LOG, "rb") as f:
    f.seek(off)
    tail = f.read().decode(errors="ignore")
acc = [float(m.group(1)) for m in ACC.finditer(tail)]
acc = [a for a in acc if a >= 0.5]
mean = sum(acc) / len(acc) if acc else -1
print(f"PROBE {LABEL} elapsed={el:.1f}s tokens={ct} tps={ct/el:.1f} accept_mean={mean:.2f} rows={len(acc)}")
