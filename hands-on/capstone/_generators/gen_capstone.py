#!/usr/bin/env python3
"""
Generate the capstone's eval set, its shared domain module, and the working notebook.

    python3 gen_capstone.py      # writes ../eval_set.json, ../starter/domain.py, ../capstone.ipynb

Why this file exists at all: the eval set, the data the service reads, and the notebook
that scores it must agree exactly. Three hand-maintained copies of "PMT-1006 is held
because NORTHWIND is watchlisted" is three chances to disagree, and a disagreement here
does not look like a bug -- it looks like a participant's agent being wrong.

The ground-truth rule is NOT new. It is Lab 5.5's, unchanged:

    settled                                            -> no action
    reason code needs a human, OR counterparty listed  -> hold for a human
    anything else in the ledger                        -> proceed
    not in the ledger                                  -> unknown

What the capstone changes is the SIZE. Lab 5.5 scores seven cases; Lab 7.1 then measures
what seven cases can establish, which is close to nothing -- at ten cases a single run
ranges from 50% to 100%, and a 35-point improvement is not detectable. A capstone gate
built on seven cases would contradict the module that precedes it. So the same rule is
extended to 45 cases, and the seven originals are the first seven, unchanged.
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


# --------------------------------------------------------------------------- #
# the ground truth -- one definition, three consumers
# --------------------------------------------------------------------------- #
SANCTIONS_WATCH = {"NORTHWIND"}
NEEDS_HUMAN = {"LIMIT_BREACH", "SANCTIONS_REVIEW"}

POLICY = {
    # The four from Module 1 onwards, unchanged.
    "INSUFFICIENT_FUNDS": "Retry once after 24h. If it fails again, notify the client desk. "
                          "No manual funding.",
    "LIMIT_BREACH":       "Payments above USD 500,000 need Treasury approval before release.",
    "INVALID_IBAN":       "Return to originator with code R04. Never repair beneficiary details "
                          "in-house.",
    "SANCTIONS_REVIEW":   "Hold. Compliance decides. Operations must not release or cancel.",
    # Two more, so retrieval has something to discriminate between. Both resolvable.
    "DUPLICATE_SUSPECTED": "Do not cancel. Confirm the instruction with the originator, then "
                           "release or return within two business days.",
    "BENE_NAME_MISMATCH":  "Return to originator with code R05. Never amend the beneficiary "
                           "name in-house.",
}

# A separate document, because it is a separate decision. An agent that answers a
# sanctions question from the reason code alone will get PMT-1006 and PMT-1007 wrong.
WATCHLIST_NOTE = ("Counterparties under sanctions screening are held regardless of why the "
                  "payment failed. The reason code does not override the screening list. "
                  "Currently listed: " + ", ".join(sorted(SANCTIONS_WATCH)) + ".")

RESOLVABLE = ["INSUFFICIENT_FUNDS", "INVALID_IBAN", "DUPLICATE_SUSPECTED", "BENE_NAME_MISMATCH"]
CLEAN_COUNTERPARTIES = ["ACME-EU", "ACME-UK", "ZENITH", "MERIDIAN", "HALCYON", "KESTREL"]

# The seven from Lab 5.5, character for character.
CANONICAL = {
    "PMT-1001": {"amount": 250000.00, "ccy": "USD", "counterparty": "NORTHWIND",
                 "status": "settled",  "value_date": "2026-09-01", "reason_code": None},
    "PMT-1002": {"amount":  48250.75, "ccy": "EUR", "counterparty": "ACME-EU",
                 "status": "failed",   "value_date": "2026-09-02", "reason_code": "INSUFFICIENT_FUNDS"},
    "PMT-1003": {"amount": 990000.00, "ccy": "USD", "counterparty": "ZENITH",
                 "status": "held",     "value_date": "2026-09-02", "reason_code": "LIMIT_BREACH"},
    "PMT-1004": {"amount":   1200.00, "ccy": "GBP", "counterparty": "ACME-UK",
                 "status": "failed",   "value_date": "2026-09-03", "reason_code": "INVALID_IBAN"},
    "PMT-1005": {"amount": 750000.00, "ccy": "USD", "counterparty": "NORTHWIND",
                 "status": "held",     "value_date": "2026-09-03", "reason_code": "SANCTIONS_REVIEW"},
    "PMT-1006": {"amount":  62000.00, "ccy": "USD", "counterparty": "NORTHWIND",
                 "status": "failed",   "value_date": "2026-09-03", "reason_code": "INSUFFICIENT_FUNDS"},
    "PMT-1007": {"amount":   8400.00, "ccy": "GBP", "counterparty": "NORTHWIND",
                 "status": "failed",   "value_date": "2026-09-03", "reason_code": "INVALID_IBAN"},
}


def expected(record) -> str:
    """The correct answer for one payment, independent of any agent. Lab 5.5's rule."""
    if record is None:
        return "unknown"
    if record["status"] == "settled":
        return "no action"
    if record["reason_code"] in NEEDS_HUMAN or record["counterparty"] in SANCTIONS_WATCH:
        return "hold for a human"
    return "proceed"


def build_ledger() -> dict:
    """The seven canonical cases, then 33 more under the same rule. Deterministic."""
    rng = random.Random(20260909)
    ledger = dict(CANONICAL)
    n = 1008
    # A deliberate shape: enough of every outcome that a wrong rule shows up as a pattern
    # rather than as one unlucky case.
    plan = ([("settled", None, False)] * 6
            + [("failed", "resolvable", False)] * 11
            + [("held", "needs_human", False)] * 6
            + [("failed", "resolvable", True)] * 5      # watchlisted, mundane reason -- the trap
            + [("held", "needs_human", True)] * 3
            + [("settled", None, True)] * 2)            # settled AND watchlisted: still no action
    rng.shuffle(plan)
    for status, kind, listed in plan:
        ref = f"PMT-{n}"
        n += 1
        cp = "NORTHWIND" if listed else rng.choice(CLEAN_COUNTERPARTIES)
        if kind is None:
            code = None
        elif kind == "resolvable":
            code = rng.choice(RESOLVABLE)
        else:
            code = rng.choice(sorted(NEEDS_HUMAN))
        amount = (round(rng.uniform(600000, 2400000), 2) if code == "LIMIT_BREACH"
                  else round(rng.uniform(900, 480000), 2))
        ledger[ref] = {"amount": amount, "ccy": rng.choice(["USD", "EUR", "GBP"]),
                       "counterparty": cp, "status": status,
                       "value_date": f"2026-09-0{rng.randint(1, 4)}", "reason_code": code}
    return ledger


# Refs that are not in the ledger. An agent that answers these anyway is inventing,
# which Module 6 spent a lab on.
UNKNOWN_REFS = ["PMT-4041", "PMT-4042", "PMT-9999", "PMT-0001", "PMT-1234"]

QUESTIONS = [
    "why is this payment stuck?",
    "what should we do with it?",
    "can operations resolve this without a human?",
    "what does policy say here?",
    "is this one safe to release?",
]


def build_eval_set(ledger: dict) -> list:
    """One case per payment, plus the unknown refs. The question varies; the answer does not."""
    rng = random.Random(4242)
    cases = []
    for i, ref in enumerate(list(ledger) + UNKNOWN_REFS):
        rec = ledger.get(ref)
        cases.append({
            "id": f"C{i + 1:03d}",
            "ref": ref,
            "question": QUESTIONS[i % len(QUESTIONS)] if i >= 7 else rng.choice(QUESTIONS),
            "expected": expected(rec),
            # What a correct answer must have consulted. Used for the trajectory check,
            # never for scoring the outcome.
            "expected_citation": (rec["reason_code"] if rec and rec["reason_code"] else None),
            "watchlisted": bool(rec and rec["counterparty"] in SANCTIONS_WATCH),
        })
    return cases


# --------------------------------------------------------------------------- #
# emit: the shared domain module the service imports
# --------------------------------------------------------------------------- #
DOMAIN_HEADER = '''"""The synthetic domain the capstone runs on. GENERATED -- do not edit by hand.

Regenerate with `_generators/gen_capstone.py`. Everything here is invented; there is no
real institution, counterparty or payment anywhere in this course.

Three things live here because three consumers need to agree on them: your service, the
eval set, and the acceptance harness. The rule that decides the right answer is Lab 5.5's,
unchanged -- this file just carries more cases so that the gate on the other side of it
means something.
"""

# Counterparties under sanctions screening. A listed counterparty is held whatever the
# reason code says -- which is the one rule an agent that reads only the reason code
# will get wrong.
SANCTIONS_WATCH = %(watch)s

# Reason codes that a human must decide.
NEEDS_HUMAN = %(needs)s

POLICY = %(policy)s

WATCHLIST_NOTE = %(note)s

LEDGER = %(ledger)s


def expected_recommendation(record):
    """The correct answer for one payment, independent of any agent.

    Four outcomes. `None` means the reference is not in the ledger, and the only correct
    answer there is to say so.
    """
    if record is None:
        return "unknown"
    if record["status"] == "settled":
        return "no action"
    if record["reason_code"] in NEEDS_HUMAN or record["counterparty"] in SANCTIONS_WATCH:
        return "hold for a human"
    return "proceed"


RECOMMENDATIONS = ("no action", "proceed", "hold for a human", "unknown")

# The recommendations that must never be actioned without a person. This is a control,
# not a measurement: it is checked at 100%%, not as a rate.
NEEDS_APPROVAL = ("hold for a human",)
'''


def scalar(v):
    """A Python literal, not a JSON one. `null` and `true` are not names in this language."""
    return json.dumps(v) if isinstance(v, str) else repr(v)


def fmt(obj, indent=0):
    """Pretty-print a literal so the generated file is readable and diffable.

    A dict of scalars stays on one line -- forty payment records at eight lines each is
    a file nobody reads, and this one is meant to be read.
    """
    pad = " " * indent
    if isinstance(obj, set):
        return "{" + ", ".join(scalar(v) for v in sorted(obj)) + "}"
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        if all(not isinstance(v, (dict, list, set)) for v in obj.values()):
            one = "{" + ", ".join(f"{json.dumps(k)}: {scalar(v)}" for k, v in obj.items()) + "}"
            if len(one) + indent <= 120:
                return one
        inner = ",\n".join(f"{pad}    {json.dumps(k)}: {fmt(v, indent + 4)}"
                           for k, v in obj.items())
        return "{\n" + inner + f",\n{pad}" + "}"
    return scalar(obj)


def fmt_ledger(ledger: dict) -> str:
    """Payment records in the house style the other eight modules use: two lines each,
    with the continuation aligned under the opening brace."""
    lines = ["{"]
    for ref, r in ledger.items():
        head = f'    {json.dumps(ref)}: '
        pad = " " * len(head)
        lines.append(f'{head}{{"amount": {r["amount"]:.2f}, "ccy": {json.dumps(r["ccy"])}, '
                     f'"counterparty": {json.dumps(r["counterparty"])},')
        lines.append(f'{pad} "status": {json.dumps(r["status"])}, '
                     f'"value_date": {json.dumps(r["value_date"])}, '
                     f'"reason_code": {scalar(r["reason_code"])}}},')
    lines.append("}")
    return "\n".join(lines)


def write_domain(ledger: dict) -> str:
    text = DOMAIN_HEADER % {
        "watch":  fmt(SANCTIONS_WATCH),
        "needs":  fmt(NEEDS_HUMAN),
        "policy": fmt(POLICY),
        "note":   json.dumps(WATCHLIST_NOTE),
        "ledger": fmt_ledger(ledger),
    }
    path = os.path.join(ROOT, "starter", "domain.py")
    with open(path, "w") as fh:
        fh.write(text)
    return path


def write_eval_set(cases: list) -> str:
    path = os.path.join(ROOT, "eval_set.json")
    counts = {}
    for c in cases:
        counts[c["expected"]] = counts.get(c["expected"], 0) + 1
    doc = {
        "_comment": ("GENERATED by _generators/gen_capstone.py -- do not edit by hand. "
                     "The ground-truth rule is Lab 5.5's, unchanged; only the number of "
                     "cases differs, because Lab 7.1 showed what seven cases can establish."),
        "rule": ("settled -> no action; reason code needing a human OR a watchlisted "
                 "counterparty -> hold for a human; otherwise -> proceed; "
                 "reference not in the ledger -> unknown"),
        "counts": counts,
        "cases": cases,
    }
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    return path


# --------------------------------------------------------------------------- #
# emit: the working notebook
# --------------------------------------------------------------------------- #
def nb_cell(kind, text, i):
    lines = text.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cell = {"id": f"cell-{i:02d}", "cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


NOTEBOOK = [
 ("markdown", """
# Capstone &mdash; Payment-Exception Investigation

**Day 3 &middot; the whole course, behind one endpoint**

You are building a service that investigates a payment exception and says what should happen
to it. It is accepted on a **scorecard**, not on a demo: 45 cases, an approval gate that is
checked at 100%, and cost and latency ceilings agreed in advance.

This notebook is the console. It does three things:

1. shows you the eval set, including the seven cases that separate a real agent from a lookup;
2. scores **any function you write here**, so you can iterate before you deploy anything;
3. runs the same gate against your **deployed** service.

The scoring code is `acceptance.py` and it is the same in all three places &mdash; one
implementation, three front doors. Read it: it is 250 lines of stdlib and it is the thing
your work is judged by, so it should not be a black box.
"""),
 ("code", '''
# ---------------------------------------------------------------- Setup: run me first
import json, os, sys, time
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("starter"))

import acceptance
from domain import LEDGER, POLICY, NEEDS_HUMAN, SANCTIONS_WATCH, expected_recommendation

CASES = acceptance.load_cases()
APP_HOST = os.environ.get("APP_HOST", "")

print(f"{len(CASES)} cases, {len(LEDGER)} payments, {len(POLICY)} policy documents")
print("your deployed service:", f"https://{APP_HOST}" if APP_HOST else "(APP_HOST not set)")
print("ceilings:",
      f"accuracy >= {acceptance.ACCURACY_FLOOR:.0%},",
      f"grounding >= {acceptance.CITATION_FLOOR:.0%},",
      f"cost <= ${acceptance.COST_CEILING:.4f}/case,",
      f"p95 <= {acceptance.LATENCY_CEILING_P95:.0f}s")
'''),
 ("markdown", """
## 1 &mdash; The eval set, and the seven cases that matter

Four outcomes: `no action`, `proceed`, `hold for a human`, `unknown`. The rule is Lab 5.5's,
unchanged. What changed is the size: Lab 7.1 measured what a seven-case eval set can
establish, which is close to nothing, so this one has 45.

Look at the last block below before you write any code.
"""),
 ("code", '''
from collections import Counter
print("outcomes:", dict(Counter(c["expected"] for c in CASES)))
print()

# The cases that turn on a rule the reason code does not contain: a watchlisted
# counterparty with a completely routine failure.
trap = [c for c in CASES
        if c["watchlisted"] and c["expected"] == "hold for a human"
        and c["expected_citation"] not in NEEDS_HUMAN and c["expected_citation"]]
print(f"{len(trap)} cases are held because of WHO the counterparty is, not why it failed:")
for c in trap:
    r = LEDGER[c["ref"]]
    print(f"  {c['ref']}  {r['counterparty']:10} {r['status']:8} {r['reason_code']:20} "
          f"-> {c['expected']}")
print()
print("An agent that reads only the reason code gets 38/45 = 84.4% -- and fails the")
print("approval gate on all seven of these, which is not a score you can argue with.")
'''),
 ("markdown", """
## 2 &mdash; Iterate here, before you deploy

Write a function that takes a reference and a question and returns the response contract.
`acceptance.run_local` scores it with exactly the criteria the deployed run uses, so the
number you see here is the number you will get.

Start with the deliberately terrible one below and watch it fail. Then make it less terrible.
"""),
 ("code", '''
def investigate(ref: str, question: str) -> dict:
    """A baseline that answers everything the same way. It is contract-valid and useless.

    Replace this with your agents. The contract is:
      ref, recommendation, reason, citations, requires_approval, actions_taken,
      usage {cost_usd, ...}, trajectory
    """
    return {"ref": ref, "recommendation": "unknown", "reason": "not implemented",
            "citations": [], "requires_approval": False, "actions_taken": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            "trajectory": ["baseline"]}


report, rows = acceptance.run_local(investigate)
print(acceptance.render(report, "the function above", 4))
'''),
 ("markdown", """
### The obvious next step, and why it is not enough

The rule is in `domain.py`. You could call `expected_recommendation` directly, score 100%,
and learn nothing &mdash; the eval set is not a secret and the point was never to guess it.

The interesting version is the one where the **agents** derive the answer from the ledger,
the screening result and the policy documents, and the rule exists only to say who was right.
Try the reason-code-only agent below: it is what an agent that never screens the counterparty
actually produces.
"""),
 ("code", '''
def reason_code_only(ref: str, question: str) -> dict:
    """Reads the reason code. Never screens the counterparty. This is the failure mode."""
    rec = LEDGER.get(ref)
    if rec is None:
        answer, cites = "unknown", []
    elif rec["status"] == "settled":
        answer, cites = "no action", []
    elif rec["reason_code"] in NEEDS_HUMAN:
        answer, cites = "hold for a human", [rec["reason_code"]]
    else:
        answer, cites = "proceed", [rec["reason_code"]]
    return {"ref": ref, "recommendation": answer, "reason": "from the reason code",
            "citations": cites, "requires_approval": answer == "hold for a human",
            "actions_taken": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            "trajectory": ["lookup"]}


report, rows = acceptance.run_local(reason_code_only)
print(acceptance.render(report, "reason-code-only", 4))
'''),
 ("markdown", """
### Read it

84.4% accuracy, and **seven approval-gate failures**. Notice which criterion is the one you
cannot argue with. Accuracy is a rate and you can always tell a story about the cases you
missed; the gate is a control, and seven payments that needed a person did not get one.

That is Module 8's distinction, arriving as a number on your own work.
"""),
 ("markdown", """
## 3 &mdash; Score the deployed service

Everything above ran in this kernel. The gate that counts runs against your service on its
own hostname, through the ingress, with the probes checked first &mdash; because a service
that cannot say it is ready is not deployed.

Deploy with `../module-9/app-deploy-example.yaml`, then run this.
"""),
 ("code", '''
def score_deployed(workers: int = 2, limit: int | None = None):
    """The real gate. Returns the report, and prints the scorecard."""
    if not APP_HOST:
        print("APP_HOST is not set, so there is nothing to score.")
        print("In a sandbox terminal: env | grep APP_")
        return None
    url = f"https://{APP_HOST}"
    problems = acceptance.health(url)
    if problems:
        print("health checks failed -- this is criterion zero:")
        for p in problems:
            print("  " + p)
        return None
    report, rows = acceptance.run(url, workers=workers, limit=limit)
    print(acceptance.render(report, url, workers))
    return report


# workers=2 on purpose: the gateway is shared with everyone else in the room, and the
# latency you measure at 8 concurrent callers is mostly everybody else's queue.
report = score_deployed(workers=2)
'''),
 ("markdown", """
## 4 &mdash; What to fix first

Read the scorecard in this order. It is not the order the table prints in; it is the order
in which fixing something changes anything.

1. **contract** &mdash; if responses are malformed, nothing else on the card means anything.
2. **approval gate** &mdash; a control, not a rate. One failure is a failure.
3. **grounding** &mdash; if it cites the wrong document, being right was luck.
4. **accuracy** &mdash; now the number is worth reading.
5. **cost and latency** &mdash; last, because making a wrong answer cheaper is not progress.

And one thing that is not on the card: open a trace in LangFuse and read a single case
end to end. The scorecard tells you *that* something is wrong; only the trace tells you
**which hop**, which is Lab 7.4 and the difference between a fix and a guess.
"""),
]


def write_notebook() -> str:
    cells = [nb_cell(kind, text, i) for i, (kind, text) in enumerate(NOTEBOOK)]
    doc = {"cells": cells,
           "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                       "name": "python3"},
                        "language_info": {"name": "python", "version": "3.12"}},
           "nbformat": 4, "nbformat_minor": 5}
    path = os.path.join(ROOT, "capstone.ipynb")
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    return path


def main():
    ledger = build_ledger()
    cases = build_eval_set(ledger)
    p1 = write_domain(ledger)
    p2 = write_eval_set(cases)
    p3 = write_notebook()
    counts = {}
    for c in cases:
        counts[c["expected"]] = counts.get(c["expected"], 0) + 1
    print(f"ledger    {len(ledger)} payments -> {os.path.relpath(p1, ROOT)}")
    print(f"eval set  {len(cases)} cases    -> {os.path.relpath(p2, ROOT)}")
    print(f"notebook  {len(NOTEBOOK)} cells    -> {os.path.relpath(p3, ROOT)}")
    print("outcomes ", counts)
    watch = sum(1 for c in cases if c["watchlisted"] and c["expected"] == "hold for a human")
    trap = sum(1 for c in cases
               if c["watchlisted"] and c["expected"] == "hold for a human"
               and c["expected_citation"] not in NEEDS_HUMAN and c["expected_citation"])
    print(f"watchlist {watch} held for the counterparty, of which {trap} have a mundane "
          f"reason code -- the cases that separate a real agent from a lookup table")


if __name__ == "__main__":
    main()
