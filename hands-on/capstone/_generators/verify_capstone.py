#!/usr/bin/env python3
"""Verify the acceptance gate itself -- offline, in about a second.

A gate nobody has passed is a wish. A gate nobody can fail is decoration. This runs six
fixture services against the real harness and asserts what each one should score:

    perfect            every criterion, ACCEPTED
    reason_code_only   the agent that reads the reason code and never screens the
                       counterparty -- 84% accurate and it fails the GATE on seven cases
    no_gate            every outcome right, never asks for a human -- NOT ACCEPTED
    inventor           answers about references that do not exist -- grounding fails
    expensive          right, and over the cost ceiling
    malformed          drops a contract field -- a failure, not a zero

No model, no cluster, no network beyond a loopback socket. Run it after any change to
acceptance.py, the eval set, or the ground-truth rule.
"""
import http.server, json, os, socket, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

import acceptance                                            # noqa: E402
from domain import LEDGER, NEEDS_HUMAN, SANCTIONS_WATCH      # noqa: E402

CASES = {c["ref"]: c for c in acceptance.load_cases()}


# --------------------------------------------------------------------------- fixtures
def truth(ref):
    return CASES[ref]["expected"]


def reason_code_only(ref):
    """The agent this eval set exists to catch: it never screens the counterparty."""
    rec = LEDGER.get(ref)
    if rec is None:
        return "unknown"
    if rec["status"] == "settled":
        return "no action"
    return "hold for a human" if rec["reason_code"] in NEEDS_HUMAN else "proceed"


def body_for(kind, ref):
    case = CASES[ref]
    rec = LEDGER.get(ref)
    answer = reason_code_only(ref) if kind == "reason_code_only" else truth(ref)
    if kind == "inventor" and case["expected"] == "unknown":
        answer = "proceed"
    citations = [case["expected_citation"]] if case["expected_citation"] else []
    if kind == "inventor" and case["expected"] == "unknown":
        citations = ["INSUFFICIENT_FUNDS"]
    out = {
        "ref": ref,
        "recommendation": answer,
        "reason": "fixture",
        "citations": citations,
        "requires_approval": (answer in acceptance.NEEDS_APPROVAL) and kind != "no_gate",
        "actions_taken": [],
        "usage": {"input_tokens": 700, "output_tokens": 1600,
                  "cost_usd": 0.4 if kind == "expensive" else 0.0011},
        "trajectory": ["supervisor", "ledger", "sanctions", "policy", "decide"],
    }
    if kind == "malformed":
        out.pop("citations")
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    kind = "perfect"

    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            return self._send(200, {"status": "ok"})
        if self.path == "/readyz":
            return self._send(200, {"ready": True})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        self._send(200, body_for(self.kind, req["ref"]))


def serve(kind):
    Handler.kind = kind
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def criterion(report, name):
    return next(c for c in report["criteria"] if c["name"] == name)


# --------------------------------------------------------------------------- expectations
EXPECT = [
    # kind,               accepted, criteria that must FAIL
    ("perfect",           True,  []),
    ("reason_code_only",  False, ["accuracy", "approval gate"]),
    ("no_gate",           False, ["approval gate"]),
    ("inventor",          False, ["grounding"]),
    ("expensive",         False, ["cost/case"]),
    ("malformed",         False, ["contract"]),
]

fails = 0
for kind, accepted, must_fail in EXPECT:
    srv, url = serve(kind)
    try:
        report, _ = acceptance.run(url, workers=8)
    finally:
        srv.shutdown()
    failing = sorted(c["name"] for c in report["criteria"] if c["hard"] and not c["pass"])
    ok = report["accepted"] == accepted and set(must_fail) <= set(failing)
    acc = criterion(report, "accuracy")["value"]
    print(f"[{'OK    ' if ok else 'BROKEN'}] {kind:18} "
          f"{'ACCEPTED' if report['accepted'] else 'rejected':9} "
          f"accuracy {acc:5.1%}  failing: {failing or ['-']}")
    if not ok:
        fails += 1
        print(f"           expected accepted={accepted}, failing to include {must_fail}")

# The latency criterion needs no slow fixture -- move the ceiling and watch it flip.
srv, url = serve("perfect")
try:
    real = acceptance.LATENCY_CEILING_P95
    acceptance.LATENCY_CEILING_P95 = 0.0
    report, _ = acceptance.run(url, workers=8)
    acceptance.LATENCY_CEILING_P95 = real
finally:
    srv.shutdown()
lat_ok = not report["accepted"] and not criterion(report, "p95 latency")["pass"]
print(f"[{'OK    ' if lat_ok else 'BROKEN'}] {'latency ceiling':18} "
      f"a ceiling of 0s rejects the perfect service")
fails += 0 if lat_ok else 1

# And the naive agent's exact score, which the brief quotes.
srv, url = serve("reason_code_only")
try:
    report, _ = acceptance.run(url, workers=8)
finally:
    srv.shutdown()
score = criterion(report, "accuracy")["value"]
gate_n = len(report["gate_failures"])
quoted_ok = round(score, 3) == 0.844 and gate_n == 7
print(f"[{'OK    ' if quoted_ok else 'BROKEN'}] {'quoted numbers':18} "
      f"reason-code-only scores {score:.1%} with {gate_n} gate failures "
      f"(the brief says 84.4% and 7)")
fails += 0 if quoted_ok else 1

print(f"\n{'all good' if not fails else str(fails) + ' problem(s)'}")
sys.exit(1 if fails else 0)
