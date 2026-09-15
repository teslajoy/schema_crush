#!/usr/bin/env python3
"""baseline eval: column name -> FHIR R5 path.

measures two pipelines against evals/gold_columns.json:

  cascade   knowledge base -> fuzzy -> calibrated embeddings. no api key.
  agent     ClaudeAgent over the same tools. needs ANTHROPIC_API_KEY.

results are reported per stratum, never as one blended number: a single figure
averages GDC-native input (which the knowledge base already contains) together
with lab-native input (which it has never seen), and hides the only case a new
user actually hits.

usage:
    python evals/run_eval.py                    # cascade only
    python evals/run_eval.py --agent            # cascade + agent (needs key)
    python evals/run_eval.py --agent --only lab_native
    python evals/run_eval.py --out evals/baseline.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "space"))

GOLD = Path(__file__).parent / "gold_columns.json"
STRATA = ["gdc_native", "public_cohort", "lab_native"]


def score_one(predicted: str | None, accept: list[str]) -> str:
    """classify a prediction.

    EXACT    predicted is one of the accepted targets
    FAMILY   same resource and one path prefixes the other, e.g.
             Condition.code vs Condition.code.coding. counted separately
             because it is usable but not what was asked for.
    WRONG    a different resource, or an incompatible path. the worst outcome:
             a confident wrong answer costs more than no answer.
    UNMAPPED nothing returned. honest, and recoverable by a human.
    """
    if not predicted or predicted in ("—", "None"):
        return "UNMAPPED"
    if predicted in accept:
        return "EXACT"
    pred_res = predicted.split(".")[0]
    for a in accept:
        if a.split(".")[0] != pred_res:
            continue
        if predicted.startswith(a + ".") or a.startswith(predicted + "."):
            return "FAMILY"
        if predicted.split(".")[:2] == a.split(".")[:2]:
            return "FAMILY"
    return "WRONG"


def run_cascade(entries):
    import app  # space/app.py
    out = []
    for e in entries:
        rows = app._map_columns([e["column"]], {e["column"]: e.get("samples", [])})
        r = rows[0]
        out.append({**e, "predicted": r["target"], "confidence": r["confidence"],
                    "stage": r["stage"], "verdict": score_one(r["target"], e["accept"])})
    return out


class _UsageProxy:
    """forwards everything to the wrapped runnable, recording token usage.

    agent.llm is a pydantic RunnableBinding, which refuses attribute assignment,
    so its .invoke cannot be monkey-patched. ClaudeAgent itself is a plain
    object, so swapping the whole attribute for this proxy works instead.
    """

    def __init__(self, inner, totals):
        self._inner = inner
        self._totals = totals

    def invoke(self, *args, **kwargs):
        resp = self._inner.invoke(*args, **kwargs)
        try:
            um = getattr(resp, "usage_metadata", None) or {}
            det = um.get("input_token_details") or {}
            t = self._totals
            t["calls"] += 1
            t["input"] += um.get("input_tokens", 0) or 0
            t["output"] += um.get("output_tokens", 0) or 0
            t["cache_read"] += det.get("cache_read", 0) or 0
            t["cache_creation"] += det.get("cache_creation", 0) or 0
        except Exception:
            # telemetry must never break the measurement it is observing
            pass
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


def run_agent(entries, model=None):
    from schema_crush.orchestrator.agents import ClaudeAgent
    kwargs = {"task": "field"}
    if model:
        kwargs["model"] = model
    agent = ClaudeAgent(**kwargs)

    totals = {"cache_read": 0, "cache_creation": 0, "input": 0, "output": 0, "calls": 0}
    try:
        agent.llm = _UsageProxy(agent.llm, totals)
    except Exception as exc:
        print(f"  (usage tracking unavailable: {type(exc).__name__}; eval continues)")

    out = []
    for e in entries:
        try:
            props = agent.propose_mappings(
                e["column"], candidate_targets=None,
                context={"sample_values": e.get("samples", [])})
            pred = props[0].target_field if props else None
            conf = float(props[0].confidence) if props else 0.0
            reason = (props[0].reasoning or "")[:160] if props else ""
        except Exception as exc:
            pred, conf, reason = None, 0.0, f"{type(exc).__name__}: {exc}"
        out.append({**e, "predicted": pred, "confidence": conf, "stage": "agent",
                    "reasoning": reason, "verdict": score_one(pred, e["accept"])})

    u = totals
    if u["calls"]:
        billed = u["input"] + u["cache_creation"] + u["cache_read"]
        hit = 100 * u["cache_read"] / billed if billed else 0
        print(f"\ntokens over {u['calls']} llm calls: "
              f"{u['input']:,} uncached in, {u['cache_creation']:,} cache write, "
              f"{u['cache_read']:,} cache read, {u['output']:,} out")
        print(f"prompt cache hit rate: {hit:.0f}% of input tokens")
        if u["cache_read"] == 0 and u["calls"] > 1:
            print("  WARNING: zero cache reads across multiple calls - the prefix "
                  "is being invalidated and this run paid full price.")
        # opus 4.6 rates: $5/MTok in, $25/MTok out; writes 1.25x, reads 0.1x
        cost = (u["input"] * 5 + u["cache_creation"] * 6.25
                + u["cache_read"] * 0.5 + u["output"] * 25) / 1e6
        uncached = ((u["input"] + u["cache_creation"] + u["cache_read"]) * 5
                    + u["output"] * 25) / 1e6
        print(f"approx cost ${cost:.2f} (without caching: ${uncached:.2f})")
    return out


def report(name, results):
    print(f"\n{'='*74}\n{name}\n{'='*74}")
    print(f"{'stratum':16s} {'n':>3}  {'EXACT':>6} {'FAMILY':>7} {'WRONG':>6} {'UNMAPPED':>9}   usable%")
    totals = {}
    for st in STRATA:
        rows = [r for r in results if r["stratum"] == st]
        if not rows:
            continue
        c = {v: sum(1 for r in rows if r["verdict"] == v) for v in
             ("EXACT", "FAMILY", "WRONG", "UNMAPPED")}
        totals[st] = c
        usable = 100 * (c["EXACT"] + c["FAMILY"]) / len(rows)
        print(f"{st:16s} {len(rows):>3}  {c['EXACT']:>6} {c['FAMILY']:>7} "
              f"{c['WRONG']:>6} {c['UNMAPPED']:>9}   {usable:5.0f}%")
    allc = {v: sum(1 for r in results if r["verdict"] == v) for v in
            ("EXACT", "FAMILY", "WRONG", "UNMAPPED")}
    n = len(results)
    print(f"{'-'*74}")
    print(f"{'ALL':16s} {n:>3}  {allc['EXACT']:>6} {allc['FAMILY']:>7} "
          f"{allc['WRONG']:>6} {allc['UNMAPPED']:>9}   "
          f"{100*(allc['EXACT']+allc['FAMILY'])/n:5.0f}%")

    wrong = [r for r in results if r["verdict"] == "WRONG"]
    if wrong:
        print(f"\nWRONG ({len(wrong)}) - these are worse than unmapped:")
        for r in wrong:
            print(f"   {r['column']:24s} -> {str(r['predicted'])[:40]:42s} "
                  f"(want {r['accept'][0]})")
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", action="store_true", help="also run ClaudeAgent (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--model", help="override agent model")
    ap.add_argument("--only", choices=STRATA, help="restrict to one stratum")
    ap.add_argument("--out", help="write full results as json")
    args = ap.parse_args()

    gold = json.loads(GOLD.read_text())
    entries = gold["columns"]
    if args.only:
        entries = [e for e in entries if e["stratum"] == args.only]
    print(f"gold set: {len(entries)} columns "
          f"({', '.join(f'{st}={sum(1 for e in entries if e['stratum']==st)}' for st in STRATA)})")

    payload = {"gold_n": len(entries)}
    cas = run_cascade(entries)
    payload["cascade"] = {"results": cas, "totals": report("CASCADE (no api key)", cas)}

    if args.agent:
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("\nANTHROPIC_API_KEY is not set; skipping the agent run.")
        else:
            ag = run_agent(entries, args.model)
            payload["agent"] = {"results": ag, "totals": report("AGENT", ag)}

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()