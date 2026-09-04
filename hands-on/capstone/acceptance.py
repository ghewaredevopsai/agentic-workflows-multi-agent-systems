#!/usr/bin/env python3
"""The capstone acceptance gate.

    python3 acceptance.py                      # uses $APP_HOST
    python3 acceptance.py --url https://host   # or point it anywhere
    python3 acceptance.py --workers 1          # sequential, if you want clean latencies
    python3 acceptance.py --json report.json   # machine-readable scorecard

Your capstone is accepted on this scorecard, not on a demo. A demo is one run of one
case with you driving; this is 45 cases against the service as deployed, and it is the
same argument Module 7 made about release gates -- agreed in advance, in a calm week,
by somebody who is not trying to ship anything.

Stdlib only, on purpose: it has to run in a sandbox terminal, in the notebook, and on
whatever machine reviews it, with nothing installed.

--------------------------------------------------------------------------------------
WHAT IT CAN AND CANNOT CHECK

This is a black-box harness. It sees HTTP responses, so it can check outcomes, the
approval gate, citations, cost and latency exactly.

It CANNOT see inside your service. `trajectory` and `usage` are self-reported: a service
that lies about them passes. That is not a hole to be plugged, it is the honest boundary
of an external check -- the same one Lab 9.3 found in `kubectl --dry-run=server`, which
proves an object is legal and says nothing about whether it is a good idea. The design
requirements (supervisor + specialists, the ledger over MCP, Agentic RAG, a critic) are
checked by a person reading your code and your traces, and the traces are the reason
Module 9 asked you to instrument it.
"""
import argparse, json, os, statistics, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ the agreed ceilings
# CALIBRATED 2026-09-04 against the reference service on this sandbox. Four consecutive
# 45-case runs of the accepted configuration gave an identical scorecard, so these are not
# one lucky number:
#
#                        accuracy   gate   grounding   cost/case   p95      verdict
#   reference, no think    93.3%   clean     100%      $0.0002     3.4s   ACCEPTED
#   reference, thinking   100.0%*  clean     100%      $0.0013    65.9s   rejected: latency
#   reason-code-only        84.4%  7 fails    100%      $0         --     rejected
#                                                          (* on 15 cases, not 45)
#
# The two reference configurations land on OPPOSITE SIDES of the gate, and that is the
# whole design: the more accurate one is rejected on latency, so quality alone does not
# buy a pass. What each ceiling is anchored to:
#
#   accuracy   84.4% is what an agent that never screens the counterparty scores. The
#              floor is 90%: 5.6 points above that baseline, and 3.3 below the reference,
#              which is 1.5 cases of margin either way.
#   grounding  31 of the 45 cases turn on a named document. Both reference
#              configurations cite all 31 correctly, so 90% allows three misses.
#   cost       $0.0002 without thinking, $0.0013 with. $0.0030 admits both and still
#              binds on a design that calls the model six times per case.
#   latency    the number is meaningless without the concurrency. 30s at four concurrent
#              callers rejects the thinking configuration at 65.9s and passes the other
#              at 3.4s.
ACCURACY_FLOOR      = 0.90     # of 45 cases
CITATION_FLOOR      = 0.90     # of the cases that have a policy to cite
COST_CEILING        = 0.0030   # mean USD per case
LATENCY_CEILING_P95 = 30.0     # seconds, at --workers 4
REQUEST_TIMEOUT     = 120      # a case that takes longer than this is a failure, not a wait

RECOMMENDATIONS = ("no action", "proceed", "hold for a human", "unknown")
NEEDS_APPROVAL  = ("hold for a human",)

REQUIRED_FIELDS = ("ref", "recommendation", "reason", "citations",
                   "requires_approval", "actions_taken", "usage", "trajectory")


class ContractViolation(Exception):
    """The response was not the shape the service promised. Module 8, one layer out."""


# --------------------------------------------------------------------------- transport
def _get(url, timeout=15):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def _post(url, payload, timeout=REQUEST_TIMEOUT):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


# --------------------------------------------------------------------------- contract
def validate(body, case):
    """Return the response, or raise. Never repair, never guess -- Lab 8.2's rule."""
    if not isinstance(body, dict):
        raise ContractViolation(f"expected an object, got {type(body).__name__}")
    missing = [f for f in REQUIRED_FIELDS if f not in body]
    if missing:
        raise ContractViolation(f"missing field(s): {missing}")
    if body["ref"] != case["ref"]:
        raise ContractViolation(f"answered about {body['ref']!r}, was asked about {case['ref']!r}")
    if body["recommendation"] not in RECOMMENDATIONS:
        raise ContractViolation(f"recommendation {body['recommendation']!r} is not one of "
                                f"{list(RECOMMENDATIONS)}")
    if not isinstance(body["reason"], str) or not body["reason"].strip():
        raise ContractViolation("reason must be a non-empty string")
    for field in ("citations", "actions_taken", "trajectory"):
        if not isinstance(body[field], list):
            raise ContractViolation(f"{field} must be a list")
    if not isinstance(body["requires_approval"], bool):
        raise ContractViolation("requires_approval must be a boolean")
    usage = body["usage"]
    if not isinstance(usage, dict) or "cost_usd" not in usage:
        raise ContractViolation("usage must be an object carrying cost_usd")
    if not isinstance(usage["cost_usd"], (int, float)) or usage["cost_usd"] < 0:
        raise ContractViolation("usage.cost_usd must be a non-negative number")
    return body


def ask(base, case):
    """One case. Returns a row -- never raises, because one bad case must not stop the run."""
    t0 = time.perf_counter()
    row = {"id": case["id"], "ref": case["ref"], "expected": case["expected"]}
    try:
        status, text = _post(base.rstrip("/") + "/investigate",
                             {"ref": case["ref"], "question": case["question"]})
        row["latency_s"] = round(time.perf_counter() - t0, 3)
        if status != 200:
            row["error"] = f"HTTP {status}"
            return row
        body = validate(json.loads(text), case)
    except ContractViolation as exc:
        row.setdefault("latency_s", round(time.perf_counter() - t0, 3))
        row["error"] = f"contract: {exc}"
        return row
    except Exception as exc:
        row.setdefault("latency_s", round(time.perf_counter() - t0, 3))
        row["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        return row
    row.update(ok=True, got=body["recommendation"], citations=body["citations"],
               requires_approval=body["requires_approval"],
               actions_taken=body["actions_taken"], trajectory=body["trajectory"],
               cost_usd=float(body["usage"]["cost_usd"]))
    return row


# --------------------------------------------------------------------------- scoring
def score(rows, cases):
    """Turn the rows into the scorecard. Every criterion reports its own number."""
    by_id = {c["id"]: c for c in cases}
    n = len(rows)
    answered = [r for r in rows if r.get("ok")]

    # 1. contract -- a malformed response is a failure, not a zero
    contract_rate = len(answered) / n if n else 0.0

    # 2. outcome accuracy
    correct = [r for r in answered if r["got"] == r["expected"]]
    accuracy = len(correct) / n if n else 0.0

    # 3. THE GATE. Not a rate: every case that must go to a human must go to a human.
    gate_failures = []
    for r in answered:
        must_approve = r["expected"] in NEEDS_APPROVAL
        if must_approve and not r["requires_approval"]:
            gate_failures.append({"id": r["id"], "ref": r["ref"],
                                  "why": "needs a human and did not ask for one"})
        elif must_approve and r["actions_taken"]:
            gate_failures.append({"id": r["id"], "ref": r["ref"],
                                  "why": f"acted anyway: {r['actions_taken']}"})
        elif r["requires_approval"] and r["actions_taken"]:
            gate_failures.append({"id": r["id"], "ref": r["ref"],
                                  "why": "asked for approval and acted without waiting"})
    # A response that never answered at all cannot be counted as gating correctly.
    for r in rows:
        if not r.get("ok") and by_id[r["id"]]["expected"] in NEEDS_APPROVAL:
            gate_failures.append({"id": r["id"], "ref": r["ref"],
                                  "why": f"no usable answer ({r.get('error', 'unknown')})"})

    # 4. grounding -- did it cite the policy the case actually turns on, and did it stay
    #    quiet about references that do not exist?
    citeable = [r for r in answered if by_id[r["id"]]["expected_citation"]]
    cited = [r for r in citeable
             if by_id[r["id"]]["expected_citation"] in [str(c) for c in r["citations"]]]
    citation_rate = len(cited) / len(citeable) if citeable else 1.0
    invented = [{"id": r["id"], "ref": r["ref"], "got": r["got"], "citations": r["citations"]}
                for r in answered
                if r["expected"] == "unknown" and (r["got"] != "unknown" or r["citations"])]

    # 5 and 6. cost and latency
    costs = [r["cost_usd"] for r in answered]
    lats = sorted(r["latency_s"] for r in rows if "latency_s" in r)
    mean_cost = statistics.fmean(costs) if costs else 0.0
    p95 = lats[min(len(lats) - 1, max(0, -(-95 * len(lats) // 100) - 1))] if lats else 0.0

    # 7. trajectory -- self-reported, so reported and never gating
    steps = [len(set(r["trajectory"])) for r in answered if r["trajectory"]]

    criteria = [
        {"name": "contract",  "hard": True,
         "value": contract_rate, "target": 1.0, "fmt": "{:.0%}",
         "pass": contract_rate == 1.0,
         "note": f"{n - len(answered)} of {n} responses did not match the contract"},
        {"name": "accuracy",  "hard": True,
         "value": accuracy, "target": ACCURACY_FLOOR, "fmt": "{:.1%}",
         "pass": accuracy >= ACCURACY_FLOOR,
         "note": f"{len(correct)}/{n} cases correct"},
        {"name": "approval gate", "hard": True,
         "value": 0 if gate_failures else 1, "target": 1, "fmt": "{:.0f}",
         "pass": not gate_failures,
         "note": (f"{len(gate_failures)} case(s) that needed a human did not get one"
                  if gate_failures else "every held case asked for a human and acted on nothing")},
        {"name": "grounding", "hard": True,
         "value": citation_rate, "target": CITATION_FLOOR, "fmt": "{:.1%}",
         "pass": citation_rate >= CITATION_FLOOR and not invented,
         "note": (f"{len(cited)}/{len(citeable)} cited the policy the case turns on"
                  + (f"; {len(invented)} answered about a reference that does not exist"
                     if invented else ""))},
        {"name": "cost/case", "hard": True,
         "value": mean_cost, "target": COST_CEILING, "fmt": "${:.4f}",
         "pass": mean_cost <= COST_CEILING,
         "note": f"mean over {len(costs)} answered cases (self-reported by your service)"},
        {"name": "p95 latency", "hard": True,
         "value": p95, "target": LATENCY_CEILING_P95, "fmt": "{:.1f}s",
         "pass": p95 <= LATENCY_CEILING_P95,
         "note": "measured by this harness, at the concurrency it ran with"},
        {"name": "trajectory", "hard": False,
         "value": statistics.fmean(steps) if steps else 0, "target": 3, "fmt": "{:.1f}",
         "pass": bool(steps) and statistics.fmean(steps) >= 3,
         "note": "distinct steps per case -- SELF-REPORTED, so reported and never gating"},
    ]
    accepted = all(c["pass"] for c in criteria if c["hard"])
    return {"accepted": accepted, "criteria": criteria, "cases": n,
            "gate_failures": gate_failures, "invented": invented,
            "wrong": [{"id": r["id"], "ref": r["ref"], "expected": r["expected"],
                       "got": r["got"]} for r in answered if r["got"] != r["expected"]],
            "errors": [{"id": r["id"], "ref": r["ref"], "error": r["error"]}
                       for r in rows if not r.get("ok")]}


def render(report, url, workers):
    out = []
    out.append(f"\n  capstone acceptance -- {url}   ({report['cases']} cases, "
               f"{workers} concurrent)\n")
    out.append(f"  {'criterion':16}{'measured':>12}{'target':>12}   {'':4}")
    out.append("  " + "-" * 60)
    for c in report["criteria"]:
        mark = "pass" if c["pass"] else ("FAIL" if c["hard"] else "note")
        target = c["fmt"].format(c["target"]) if c["name"] != "approval gate" else "no failures"
        measured = c["fmt"].format(c["value"]) if c["name"] != "approval gate" else (
            "clean" if c["pass"] else f"{len(report['gate_failures'])} failed")
        out.append(f"  {c['name']:16}{measured:>12}{target:>12}   {mark}"
                   + ("" if c["hard"] else "  (advisory)"))
        out.append(f"  {'':16}{c['note']}")
    out.append("")
    if report["gate_failures"]:
        out.append("  approval gate failures -- these are not a score, they are the control:")
        for g in report["gate_failures"][:10]:
            out.append(f"    {g['id']} {g['ref']}: {g['why']}")
    if report["wrong"]:
        out.append(f"  wrong outcomes ({len(report['wrong'])}):")
        for w in report["wrong"][:10]:
            out.append(f"    {w['id']} {w['ref']}: expected {w['expected']!r}, got {w['got']!r}")
    if report["invented"]:
        out.append(f"  answered about references that do not exist ({len(report['invented'])}):")
        for i in report["invented"][:5]:
            out.append(f"    {i['id']} {i['ref']}: {i['got']!r} citing {i['citations']}")
    if report["errors"]:
        out.append(f"  no usable answer ({len(report['errors'])}):")
        for e in report["errors"][:10]:
            out.append(f"    {e['id']} {e['ref']}: {e['error']}")
    out.append("")
    out.append("  ACCEPTED" if report["accepted"] else "  NOT ACCEPTED")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- driver
def health(base):
    """Module 9's two questions, asked of the deployed service before anything else."""
    problems = []
    for path, want in (("/healthz", 200), ("/readyz", 200)):
        try:
            status, _ = _get(base.rstrip("/") + path)
            if status != want:
                problems.append(f"{path} returned {status}, expected {want}")
        except Exception as exc:
            problems.append(f"{path} -- {type(exc).__name__}: {str(exc)[:80]}")
    return problems


def load_cases(limit=None):
    with open(os.path.join(HERE, "eval_set.json")) as fh:
        cases = json.load(fh)["cases"]
    return cases[:limit] if limit else cases


def ask_local(fn, case):
    """The same row, for a callable instead of a deployed service.

    So you can iterate in a notebook against the identical scorer before you deploy
    anything -- one scoring implementation, two front doors. `fn(ref, question)` must
    return the same object your HTTP endpoint would.
    """
    t0 = time.perf_counter()
    row = {"id": case["id"], "ref": case["ref"], "expected": case["expected"]}
    try:
        body = validate(fn(case["ref"], case["question"]), case)
    except ContractViolation as exc:
        row.update(latency_s=round(time.perf_counter() - t0, 3), error=f"contract: {exc}")
        return row
    except Exception as exc:
        row.update(latency_s=round(time.perf_counter() - t0, 3),
                   error=f"{type(exc).__name__}: {str(exc)[:120]}")
        return row
    row.update(ok=True, latency_s=round(time.perf_counter() - t0, 3),
               got=body["recommendation"], citations=body["citations"],
               requires_approval=body["requires_approval"],
               actions_taken=body["actions_taken"], trajectory=body["trajectory"],
               cost_usd=float(body["usage"]["cost_usd"]))
    return row


def run_local(fn, limit=None, workers=4):
    """Score a callable. Same cases, same criteria, same verdict as the deployed run."""
    cases = load_cases(limit)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda c: ask_local(fn, c), cases))
    return score(rows, cases), rows


def run(url, workers=4, limit=None):
    cases = load_cases(limit)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda c: ask(url, c), cases))
    return score(rows, cases), rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a deployed capstone service.")
    ap.add_argument("--url", default=None,
                    help="base URL; defaults to https://$APP_HOST")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent callers (default 4 -- the latency ceiling is set at this)")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    ap.add_argument("--json", default=None, help="write the scorecard to this path")
    ap.add_argument("--skip-health", action="store_true",
                    help="score anyway when the probes are unhappy (they are part of the gate)")
    args = ap.parse_args(argv)

    url = args.url or (f"https://{os.environ['APP_HOST']}" if os.environ.get("APP_HOST") else None)
    if not url:
        print("No URL. Pass --url, or run this where APP_HOST is set (your sandbox terminal).")
        return 2

    problems = health(url)
    if problems:
        print("\n  health checks failed -- this is criterion zero:")
        for p in problems:
            print("    " + p)
        if not args.skip_health:
            print("\n  NOT ACCEPTED  (a service that cannot say it is ready is not deployed)\n")
            return 1
        print("  --skip-health given; scoring anyway\n")

    report, rows = run(url, workers=args.workers, limit=args.limit)
    print(render(report, url, args.workers))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"report": report, "rows": rows}, fh, indent=1)
        print(f"  scorecard written to {args.json}\n")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
