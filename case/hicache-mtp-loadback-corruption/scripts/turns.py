#!/usr/bin/env python3
"""Incident-shape test: multi-turn conversation on a 137K resident prefix,
each turn a LONG generation (thousands of decode tokens) whose tail gets
write-through backed up at cache_finished_req, then immediately recalled by
the next turn's prefill. Probe accept from engine log after every turn.

Mechanical match of 21:37:49 death: prefill new~500 cached~137.5K usage 0.86,
first decode rows 1.95 -> 1.00 forever.

Usage: turns.py  (10 turns)
"""
import json, os, re, sys, time, urllib.request

PORT = int(os.environ.get("PORT", "8000"))
CHAT = f"http://localhost:{PORT}/v1/chat/completions"
LOG = os.path.expanduser("~/g40-slow-case/boot-packed.log")
full = open(os.path.expanduser("~/doc197k.txt"), errors="ignore").read()
DOC = full[: int(len(full) * 0.69)]   # ~132K tok resident

QS = [
    "Summarize the argument in the first third of this document in detail.",
    "List every numbered claim you can find in the middle section, with commentary.",
    "Now analyze the style: enumerate ten stylistic features with examples.",
    "Compare the first and last parts at length, enumerating differences.",
    "Extract all causal statements and evaluate each one thoroughly.",
    "Rewrite the core thesis as a numbered list of 40 propositions with notes.",
    "Find all numerical data points and discuss each in a paragraph.",
    "Produce a 30-point critique of the weakest section, point by point.",
    "Trace every definition given, then test each for circularity at length.",
    "Synthesize: write the definitive 25-point summary with cross-references.",
]

def post_chat(messages, timeout=2400):
    body = json.dumps({
        "model": "unsloth/Qwen3.8-27B-NVFP4",
        "messages": messages,
        "max_tokens": 6000,
        "temperature": 0.6,
    }).encode()
    req = urllib.request.Request(CHAT, data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def usage():
    with urllib.request.urlopen(f"http://localhost:{PORT}/metrics", timeout=15) as f:
        for lb in f:
            l = lb.decode()
            if l.startswith("sglang:token_usage{"):
                return float(l.split()[-1])
    return -1

def accept_since(off):
    with open(LOG, "rb") as f:
        f.seek(off)
        tail = f.read().decode(errors="ignore")
    vals = [float(m.group(1)) for m in re.finditer(r"accept len: ([0-9.]+),", tail)]
    vals = [v for v in vals if v >= 0.5]
    return (sum(vals) / len(vals)) if vals else -1, len(vals)

msgs = [{"role": "system", "content": "You are a meticulous analyst. Think carefully.\n\nDOCUMENT:\n" + DOC}]
for i, q in enumerate(QS):
    off = os.path.getsize(LOG)
    t0 = time.time()
    msgs.append({"role": "user", "content": q})
    try:
        r = post_chat(msgs)
    except Exception as e:
        print(f"TURN {i:02d} ERR {type(e).__name__} {e} after {time.time()-t0:.0f}s", flush=True)
        break
    ch = r["choices"][0]["message"]
    ans = (ch.get("content") or "")[:400]
    msgs.append({"role": "assistant", "content": ans})
    am, rows = accept_since(off)
    u = usage()
    ct = r.get("usage", {}).get("completion_tokens", 0)
    pt = r.get("usage", {}).get("prompt_tokens", 0)
    print(f"TURN {i:02d} {time.time()-t0:.0f}s prompt={pt} completion={ct} usage={u:.2f} accept_mean={am:.2f} rows={rows}", flush=True)
    if am >= 0 and am < 1.3 and rows > 3:
        print("CORRUPTION_DETECTED", flush=True)
        break
print("TURNS_DONE", flush=True)
