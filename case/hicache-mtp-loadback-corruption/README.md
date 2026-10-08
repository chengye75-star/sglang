# HiCache × packed-MTP loadback silently corrupts NEXTN draft KV (G40 case file)

Status: **root cause NOT fixed; one candidate patch disproven.** This branch is an
evidence archive + reproduction, not a fix. Watchdog-based mitigation in production.

## Symptom

Qwen3.8-27B (GDN-hybrid) on SGLang v0.5.21, TP2 (2x RTX 5060 Ti 16G),
`--enable-hierarchical-cache` (write_through, kernel io, page_first) +
`--speculative-algorithm NEXTN` (packed draft path, #30393; boot log
`packed MTP KV layers: target_layers=16, draft_layers=1, total_layers=17`).

After a large resident prefix (~137K tokens, pool usage >0.85) is evicted to the
host pool and later loaded back, `spec_accept_length` collapses to a constant
1.00 (healthy: 2.0–3.0) and stays there **permanently** — even greedy counting
probes, which normally accept ~3.0/token, accept 0 extra tokens. Target model
outputs remain fully correct; throughput drops to ~29 t/s, i.e. ~25% *below* the
no-MTP baseline (draft tax paid, no draft benefit). No crash, no warning, no
recovery except process restart. `POST /flush_cache` does not help.

## Deterministic reproduction (scripts/)

`dual.py N` (needs a ~197K-token text at `~/doc197k.txt`):

1. boot engine with production flags (`~/start_sglang.sh`), page_size=64,
   max_total_tokens 150–160K
2. per cycle:
   - warm: docA (69% of corpus ≈ 137K tok), `max_new_tokens=1` → usage ≈ 0.85–0.91
   - press: docB (31%) with `max_new_tokens=8000` → evicts docA pages to host
   - recall: docA again → host→device loadback (~1 s)
   - probe: greedy count; mean accept computed from decode-log window
3. verdict: probe `accept_mean < 1.3` → `CORRUPTION_REPRODUCED`

Observed (all runs, CUDA_LAUNCH_BLOCKING=1 + 150K pool to remove an unrelated
triton-lazy-load IMA crash):

| arm | result |
|---|---|
| stock v0.5.21 | cycle 0 corrupted, accept_mean 1.11 / 1.05 (3 runs) |
| "TLM+draft" patch (see below) | cycle 0 corrupted, accept_mean 1.16 |
| HiCache OFF | healthy, accept 3.00 (4 cycles) |
| MTP OFF (HiCache on) | healthy, true loadback 137K < 1 s |
| forced sidecar draft | CUDA illegal memory access in press |

10/07 production incident matches the same signature (18:37 boot healthy at
70–88 t/s; 21:38, right after a 137K request pushed usage to 0.86, accept pins
to 1.00 for the remaining 3307 decode lines; restart heals).

## Disproven: transfer_layer_id_max off-by-draft

Hypothesis was: `build_hybrid_mamba_stack()` passes the un-widened
`transfer_layer_id_max` (64) to `HybridCacheController`, so the H2D loop
(`l2_transfer.submit_host_to_device`, `for layer_id in range(tlm)`) can never
reach the packed draft's transfer id (64) and silently skips restoring it.

Instrumentation (hook recording every executed copy, joined per submission)
shows this is wrong:

- stock, tlm=64: the draft restore **does execute** — `_l2_load_transfers`
  registers the draft `L2Transfer` whose mapper fires at `expected_layer_id =
  depth = 0`, i.e. inside the loop's first round (`... 0>0 0>0 0D>16 ...`).
- with the "fix" (tlm=65): `_l2_load_transfers` looks up
  `layer_mapper(tlm + depth)` = `layer_mapper(65)` — outside the entry's
  mapping domain → returns None → `continue` → the draft transfer is
  **dropped** (`n_tr=3 draft_tr=0`). The patch, if deployed, would make the
  skip real.

And behaviorally, stock and patched both corrupt at cycle 0. So tlm is not the
root cause; the patch is not a fix (and is actively harmful).

## Open questions (next forensic steps)

- Round-trip checksums (pre-eviction draft device row vs post-loadback draft
  device row, joined on host-page identity; same-view/two-time comparisons
  only — cross-device host-vs-device row equality is unreliable on nvfp4
  packed rows and produced false alarms in early hooks)
- whether the host backup is complete at D2H time (draft slot rewritten
  concurrently during write_through?) vs whether restore reads wrong rows
- relation between the silent corruption and the IMA crash in
  `eagle_worker_v2._forward_prefill_batch` (same window, both arms, async-only)

## Mitigation in production

`watchdog_spec.sh` via cron: if `sglang:spec_accept_length < 1.3` for 3
consecutive samples while traffic is present, SIGTERM the engine in an idle
window (graceful; never SIGKILL — see dead-machine case). Self-heals within
~6–10 minutes of any corruption event.

## Files

- `scripts/dual.py` — reproduction battery (evict→loadback×N, greedy-count probe)
- `scripts/turns.py`, `scripts/soak.py` — negative-control batteries
  (single-session loadback: never reproduces)
- `scripts/probe_accept.py` — accept-mean probe from decode-log windows
- `scripts/instrument_l2.py` — per-copy H2D instrumentation (disproof tool)

Original case dir on G40: `~/g40-slow-case/` (logs: incident-original.log,
final-decisive-patched-LB.log, dual-dec2.out, instr/instr.log …).
