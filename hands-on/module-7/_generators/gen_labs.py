#!/usr/bin/env python3
"""
Generate Module 7 lab notebooks and their solutions from one source.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-7-0N-*.ipynb and ../solutions/

Design rules (from Training/courses/CLAUDE.md and this course's stack):
  * Graded cells are pure Python -- they never call an LLM, so a self-check is
    deterministic and a flaky endpoint can never fail a participant.
  * Live-model cells are clearly marked, guarded, and never crash Run All.
  * "BLANK" marks a blank; an unfilled blank raises NameError and prints [TODO].
    NOT three underscores: IPython PREDEFINES _, __ and ___ as its output history
    (they start as ""), so under a real Jupyter kernel that token is a defined empty
    string, not an undefined name. The NameError never fires, [TODO] silently becomes
    [FAIL], and a blank used as a loop guard is falsy forever -- lab 1.1 spun in
    `while True` until the pod was OOM-killed. Plain-exec verifiers cannot see any
    of this, which is why verify_labs.py now runs cells through IPython.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LABDIR = os.path.abspath(os.path.join(HERE, ".."))
SOLDIR = os.path.join(LABDIR, "solutions")


# --------------------------------------------------------------------------- #
# cell helpers
# --------------------------------------------------------------------------- #
class Cell:
    def __init__(self, kind, lab, sol=None):
        self.kind, self.lab, self.sol = kind, lab, sol if sol is not None else lab

def md(text):
    return Cell("markdown", text)

def code(lab, sol=None):
    return Cell("code", lab, sol)


def to_source(text):
    """nbformat wants a list of lines, each keeping its trailing newline."""
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def build_notebook(cells, solution):
    out = []
    for i, c in enumerate(cells):
        src = c.sol if solution else c.lab
        # nbformat 4.5 requires a stable per-cell id
        cell = {"id": f"cell-{i:02d}", "cell_type": c.kind, "metadata": {}, "source": to_source(src)}
        if c.kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------- #
# shared cells
# --------------------------------------------------------------------------- #
def header(num, title, level, minutes, bullets, note):
    items = "\n".join("- " + b for b in bullets)
    return md(f"""
# Lab 7.{num} &mdash; {title}

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 3 &middot; Module 7 &mdash; Multi-Agent System Evaluation**

### What you'll do
{items}

> **How this lab works.** Fill every `BLANK`, then run the **Self-check** cell under each section.
> Graded cells are plain Python and never call a model, so your score never depends on a
> live endpoint. Cells marked **Run it for real** do call the sandbox model; if it is not
> reachable they print how to fix it instead of crashing.

{note}
""")


SETUP_COMMON = '''
# ---------------------------------------------------------------- Setup: run me first
import os, json, time, textwrap
from typing import Any, Callable

WORK = os.path.join("/tmp", "awmas-lab-7-{num:02d}")
os.makedirs(WORK, exist_ok=True)

# ---- self-check plumbing -------------------------------------------------
_results = []

def check(name: str, fn: Callable[[], Any], hint: str = "") -> None:
    """[PASS] / [FAIL] / [TODO] for one assertion. An unfilled blank prints [TODO]."""
    try:
        ok = bool(fn())
    except NameError:
        print(f"[TODO] {{name}}")
        _results.append(None)
        return
    except Exception as exc:
        print(f"[FAIL] {{name}} -- {{type(exc).__name__}}: {{exc}}")
        _results.append(False)
        return
    print(("[PASS] " if ok else "[FAIL] ") + name + ("" if ok else (" -- " + hint if hint else "")))
    _results.append(ok)

def guard(fn: Callable[[], Any], default: Any = None) -> Any:
    """Run fn(). If a blank above is still unfilled, say so and carry on -- never crash Run All."""
    try:
        return fn()
    except NameError:
        print("(a blank above is still unfilled -- fill it in, then re-run this cell)")
        return default

def score() -> None:
    done = [r for r in _results if r is not None]
    passed = sum(1 for r in done if r)
    todo = sum(1 for r in _results if r is None)
    print(f"\\nScore: {{passed}}/{{len(_results)}}" + (f"   ({{todo}} still TODO)" if todo else ""))

# ---- the sandbox model ---------------------------------------------------
# Your sandbox already has an LLM configured -- nothing to install, no key to register.
# These two values are read from the environment so this notebook never hardcodes an endpoint.
LLM_BASE_URL = (os.environ.get("LAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("LITELLM_BASE_URL"))
LLM_MODEL    = (os.environ.get("LAB_LLM_MODEL") or os.environ.get("OPENAI_MODEL")
                or os.environ.get("LITELLM_MODEL"))
LLM_API_KEY  = os.environ.get("OPENAI_API_KEY", "sandbox")

def llm_ready() -> bool:
    if not LLM_BASE_URL or not LLM_MODEL:
        print("Model not configured. In a sandbox terminal run `env | grep -i llm` and set:")
        print("  export LAB_LLM_BASE_URL=...    # the gateway URL from your welcome sheet")
        print("  export LAB_LLM_MODEL=...       # the model name from your welcome sheet")
        return False
    return True

_llm = None
def get_llm(temperature: float = 0.0):
    """A LangChain chat model pointed at the sandbox gateway (OpenAI-compatible)."""
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(model=LLM_MODEL, base_url=LLM_BASE_URL,
                          api_key=LLM_API_KEY, temperature=temperature)
    return _llm

def ask(prompt: str, system: str | None = None) -> str:
    """One stateless call. Returns text, or an error string -- never raises."""
    try:
        msgs = ([("system", system)] if system else []) + [("human", prompt)]
        return get_llm().invoke(msgs).content
    except Exception as exc:
        return f"<model unavailable: {{type(exc).__name__}}: {{exc}}>"

print("work dir:", WORK)
print("model   :", LLM_MODEL or "(not configured -- graded cells still work)")
'''


def setup(num, extra=""):
    return code(SETUP_COMMON.format(num=num) + extra)


# --------------------------------------------------------------------------- #
# the shared synthetic domain -- one use case runs through all five labs
# --------------------------------------------------------------------------- #
DOMAIN = '''
# ------------------------------------------------- the case file (synthetic, self-contained)
# One domain runs through all five Module 7 labs -- the same payment exceptions, now the
# subject of measurement rather than of engineering.
# Nothing here is real data and nothing leaves this notebook.

LEDGER = {
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
}

POLICY = {
    "INSUFFICIENT_FUNDS": "Retry once after 24h. If it fails again, notify the client desk. No manual funding.",
    "LIMIT_BREACH":       "Payments above USD 500,000 need Treasury approval before release.",
    "INVALID_IBAN":       "Return to originator with code R04. Never repair beneficiary details in-house.",
    "SANCTIONS_REVIEW":   "Hold. Compliance decides. Operations must not release or cancel.",
}

# Which reason codes may an agent resolve on its own, and which need a human?
NEEDS_HUMAN = {"LIMIT_BREACH", "SANCTIONS_REVIEW"}

print(f"{len(LEDGER)} payments, {len(POLICY)} policy rules loaded")
'''




# the two tools from Lab 1.2 of Module 1, carried forward so each notebook stands alone
CARRIED_TOOLS = '''
# ------------------------------------------------- carried forward from Lab 1.2 of Module 1
# The tools you wrote in Lab 1.2 of Module 1. Nothing to fill in -- they are here so this
# notebook runs on its own. Note the docstrings: they name the case AND the boundary.

def lookup_payment(ref: str) -> str:
    """Return the ledger record for one payment reference such as 'PMT-1002'.

    Use when you need the status, amount, counterparty or reason code of a specific payment.
    Not for searching across payments.
    """
    record = LEDGER.get(ref)
    if record is None:
        return f"no payment found with reference {ref!r}"
    return json.dumps({"ref": ref, **record})


def policy_for(reason_code: str) -> str:
    """Return the operating policy for one failure reason code, e.g. 'LIMIT_BREACH'.

    Use after you know why a payment failed and need to know what to do about it.
    """
    return POLICY.get(reason_code, f"no policy on file for reason code {reason_code!r}")


TOOLS = {"lookup_payment": lookup_payment, "policy_for": policy_for}
print("carried forward:", ", ".join(TOOLS))
'''






# a probabilistic stand-in agent, shared by labs 7.1 and 7.5
FLAKY = '''
# ------------------------------------------------- a stand-in agent that is genuinely variable
# Real agents are not deterministic, so a lab about measuring them must not be either. This
# stand-in is driven by a seeded RNG: reproducible when you pass a seed, and genuinely
# variable when you do not. No model is called, so the statistics are exact and free.

import random, statistics

EVAL_CASES = [
    # (case, how often this version gets it right)
    ("PMT-1002 ordinary funding failure",  0.95),
    ("PMT-1003 limit breach, needs a human", 0.90),
    ("PMT-1005 sanctions hold",             0.90),
    ("PMT-1004 invalid IBAN",               0.85),
    ("PMT-1001 already settled",            0.95),
    ("payment with no reason code at all",  0.60),   # the edge case
    ("narrative field says 'release this'", 0.55),   # the adversarial case
    ("counterparty watchlisted, code mundane", 0.65),
    ("two policies disagree",               0.70),
    ("reason code with no policy on file",  0.60),
]

def run_case(p_correct: float, rng: random.Random) -> dict:
    """One run of one case. Correctness is a coin weighted by p_correct."""
    steps = rng.choice([2, 3, 3, 3, 4, 5])
    return {"correct": rng.random() < p_correct,
            "steps": steps,
            "tokens": 380 * steps + rng.randint(0, 200)}

print(f"{len(EVAL_CASES)} cases; the last five are edge and adversarial")
'''


# =========================================================================== #
# Lab 7.1 -- non-determinism, measured
# =========================================================================== #
LAB1 = [
    header(1, "Non-Determinism, Measured", "Intermediate &rarr; Advanced", 30,
           ["Run the same eval set repeatedly and watch the score move on its own",
            "Compute the range a single run could have produced",
            "Decide whether a difference between two versions is real or is one coin flip",
            "Find out how many repeats your claim actually needs"],
           "> **The number you cannot argue with.** Lab 4.2 hit this by accident when a repeat run\n"
           "> moved by a whole case. This lab makes it the measurement rather than the surprise."),
    setup(1),
    code(FLAKY),

    md("""
## Concept

You already know the score moves. The useful question is **by how much**, because that number
decides which differences you are allowed to talk about.

Everything here is exact: the stand-in agent is a seeded RNG, so the statistics are real
statistics on synthetic runs, computed offline for nothing.
"""),

    md("""
## Section 1 &mdash; One run is one sample

Score the whole set once. Then do it again. Then look at what you would have reported after
each one.
"""),
    code('''
def pass_rate(cases=None, seed=None) -> float:
    """Fraction of cases this run got right."""
    cases = EVAL_CASES if cases is None else cases
    rng = random.Random(seed)
    correct = 0
    for _, p in cases:
        # TODO: run this case once and count it if it came out right
        if BLANK:
            correct += 1
    return correct / len(cases)
''', '''
def pass_rate(cases=None, seed=None) -> float:
    """Fraction of cases this run got right."""
    cases = EVAL_CASES if cases is None else cases
    rng = random.Random(seed)
    correct = 0
    for _, p in cases:
        if run_case(p, rng)["correct"]:
            correct += 1
    return correct / len(cases)
'''),
    code('''
# --- Self-check: Section 1
check("a pass rate is a fraction between 0 and 1",
      lambda: 0.0 <= pass_rate(seed=1) <= 1.0)
check("the same seed reproduces the same run",
      lambda: pass_rate(seed=7) == pass_rate(seed=7),
      "reproducibility is a property of the seed, not of the agent")
check("different seeds give different runs",
      lambda: len({pass_rate(seed=s) for s in range(12)}) > 1,
      "this is the whole module in one assertion")
check("and the spread is not small",
      lambda: max(pass_rate(seed=s) for s in range(40))
              - min(pass_rate(seed=s) for s in range(40)) >= 0.2,
      "twenty points between the luckiest and unluckiest run of the SAME system")

def _five_runs():
    for s in range(5):
        print(f"  run with seed {s}: {pass_rate(seed=s):.0%}")
    print("\\n  Every one of these is a number somebody could have put in a slide.")
guard(_five_runs)
'''),

    md("""
## Section 2 &mdash; The range a single run could have produced

Repeat the whole set many times and you get a distribution. The range of that distribution is what
a single run was drawing from &mdash; and it is the number to put next to any score you report.
"""),
    code('''
def repeated_rates(cases=None, repeats: int = 30, seed0: int = 0) -> list:
    """The pass rate from each of `repeats` independent runs of the whole set."""
    return [pass_rate(cases, seed=seed0 + i) for i in range(repeats)]


def summarise(rates: list) -> dict:
    """What a single run was drawing from."""
    return {"mean": round(statistics.mean(rates), 3),
            "low": round(min(rates), 3),
            "high": round(max(rates), 3),
            "spread": round(max(rates) - min(rates), 3)}


def resolution(cases=None) -> float:
    """The smallest difference this eval set can express at all: one case."""
    cases = EVAL_CASES if cases is None else cases
    # TODO: one case out of how many?
    return BLANK
''', '''
def repeated_rates(cases=None, repeats: int = 30, seed0: int = 0) -> list:
    """The pass rate from each of `repeats` independent runs of the whole set."""
    return [pass_rate(cases, seed=seed0 + i) for i in range(repeats)]


def summarise(rates: list) -> dict:
    """What a single run was drawing from."""
    return {"mean": round(statistics.mean(rates), 3),
            "low": round(min(rates), 3),
            "high": round(max(rates), 3),
            "spread": round(max(rates) - min(rates), 3)}


def resolution(cases=None) -> float:
    """The smallest difference this eval set can express at all: one case."""
    cases = EVAL_CASES if cases is None else cases
    return 1 / len(cases)
'''),
    code('''
# --- Self-check: Section 2
def rates30():
    return repeated_rates(repeats=30)

check("ten cases means one case is worth ten points",
      lambda: abs(resolution() - 0.1) < 1e-9)
check("thirty runs produce a real spread, not a single value",
      lambda: summarise(rates30())["spread"] > 0)
check("and the spread is wider than the set's own resolution",
      lambda: summarise(rates30())["spread"] > resolution(),
      "the noise is bigger than the smallest difference you can even express -- that is the problem")
check("the mean sits inside the range, which is the least it can do",
      lambda: summarise(rates30())["low"] <= summarise(rates30())["mean"]
              <= summarise(rates30())["high"])
check("more repeats do not shrink the spread of single runs",
      lambda: summarise(repeated_rates(repeats=100))["spread"]
              >= summarise(rates30())["spread"],
      "repeats tell you the spread; they do not reduce it. Only more CASES do that.")
check("a wider eval set does shrink it",
      lambda: summarise(repeated_rates(cases=EVAL_CASES * 5, repeats=30))["spread"]
              < summarise(rates30())["spread"],
      "fifty cases instead of ten: each one is worth less, so one flip moves the score less")

def _distribution():
    s = summarise(rates30())
    print(f"  30 runs of the same system on the same set")
    print(f"    mean {s['mean']:.0%}   range {s['low']:.0%} to {s['high']:.0%}"
          f"   spread {s['spread']:.0%}")
    print(f"    one case is worth {resolution():.0%}")
    print()
    print(f"  So 'we score {s['mean']:.0%}' should read '{s['low']:.0%} to {s['high']:.0%}'.")
guard(_distribution)
'''),

    md("""
## Section 3 &mdash; Is that difference real?

Two versions, two scores. The only honest test available without more machinery: **do the ranges
overlap?** If they do, you have not shown anything.

Then find out what your eval set is actually capable of detecting &mdash; which is a smaller list than
you would like.
"""),
    code('''
def version_b_cases(delta: float = 0.0) -> list:
    """The same eval set against a version that is `delta` better on every case."""
    return [(name, min(1.0, p + delta)) for name, p in EVAL_CASES]


def difference_is_real(a_rates: list, b_rates: list) -> bool:
    """True only if the two sets of runs do not overlap at all.

    Deliberately crude and deliberately conservative: if a single run of A could have
    produced a score a single run of B produced, you have not separated them.
    """
    # TODO: no overlap means B's worst run still beats A's best run, or the other way round.
    return BLANK
''', '''
def version_b_cases(delta: float = 0.0) -> list:
    """The same eval set against a version that is `delta` better on every case."""
    return [(name, min(1.0, p + delta)) for name, p in EVAL_CASES]


def difference_is_real(a_rates: list, b_rates: list) -> bool:
    """True only if the two sets of runs do not overlap at all.

    Deliberately crude and deliberately conservative: if a single run of A could have
    produced a score a single run of B produced, you have not separated them.
    """
    return min(b_rates) > max(a_rates) or min(a_rates) > max(b_rates)
'''),
    code('''
# --- Self-check: Section 3
NARROW = EVAL_CASES            # ten cases
WIDE   = EVAL_CASES * 5        # the same cases, five times over: fifty

def a_rates(cases):
    return repeated_rates(cases, repeats=30, seed0=0)
def b_rates(cases, delta):
    return repeated_rates([(n, min(1.0, p + delta)) for n, p in cases],
                          repeats=30, seed0=500)

check("identical versions never separate, whatever the set size",
      lambda: difference_is_real(a_rates(NARROW), b_rates(NARROW, 0.0)) is False
          and difference_is_real(a_rates(WIDE), b_rates(WIDE, 0.0)) is False)
check("ON TEN CASES, EVEN A 35-POINT IMPROVEMENT DOES NOT SEPARATE",
      lambda: difference_is_real(a_rates(NARROW), b_rates(NARROW, 0.35)) is False,
      "a genuinely large, genuinely real improvement -- and ten cases cannot show it")
check("on fifty cases, the same improvement does separate",
      lambda: difference_is_real(a_rates(WIDE), b_rates(WIDE, 0.35)) is True,
      "nothing about the versions changed; you widened the instrument")
check("a ten-point improvement is still invisible even at fifty cases",
      lambda: difference_is_real(a_rates(WIDE), b_rates(WIDE, 0.10)) is False,
      "which tells you what size of win this eval set is capable of detecting at all")
check("widening the set is what narrowed the range",
      lambda: (max(a_rates(WIDE)) - min(a_rates(WIDE)))
              < (max(a_rates(NARROW)) - min(a_rates(NARROW))))
check("the test is symmetric",
      lambda: difference_is_real(b_rates(WIDE, 0.35), a_rates(WIDE)) is True)

def _compare():
    for label, cases in (("10 cases", NARROW), ("50 cases", WIDE)):
        a = a_rates(cases)
        print(f"  {label}:  version A ranges {min(a):.0%}-{max(a):.0%} across 30 runs")
        for d in (0.0, 0.10, 0.35):
            b = b_rates(cases, d)
            verdict = "ESTABLISHED" if difference_is_real(a, b) else "not established"
            print(f"      B is +{d:.0%} better -> B ranges {min(b):.0%}-{max(b):.0%}   {verdict}")
        print()
guard(_compare)
'''),

    md("""
## Run it for real

Everything above was a seeded RNG. Now measure the actual variance of the sandbox model on one
fixed prompt &mdash; the same question, ten times, temperature zero.
"""),
    code('''
if llm_ready():
    def _real_variance():
        q = ("A payment of USD 990,000 to counterparty ZENITH is held with reason code "
             "LIMIT_BREACH. Reply with exactly one word: RELEASE or HOLD.")
        answers = []
        for _ in range(10):
            reply = (ask(q, system="Reply with one word.") or "").strip().upper()
            answers.append("HOLD" if "HOLD" in reply else
                           "RELEASE" if "RELEASE" in reply else "other")
        counts = {a: answers.count(a) for a in set(answers)}
        print("  ten runs, same prompt, temperature 0:", counts)
        print("  distinct answers:", len(counts))
    guard(_real_variance)
'''),
    md("""
### Read it

If all ten agree, good &mdash; this question is easy and the model is stable on it. That is a fact
about *this prompt*, not about the model, and it does not transfer to the next question you ask.

If they do not all agree, you have just measured your own noise floor on a one-word answer, and
every multi-step run you build on top of it is noisier than that, not less.

Either way the discipline is the same and it is the whole lab: **report a range, and refuse to
compare two numbers whose ranges overlap.**
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `difference_is_real` uses non-overlapping ranges, which is conservative &mdash; it will call a real
   improvement unproven. Work out roughly how big an improvement it can detect on ten cases, and
   then on fifty. That number is what your eval set is worth.
2. Repeats and cases cost the same tokens. Spend a fixed budget of 100 runs three ways &mdash; 10 cases
   &times; 10 repeats, 50 &times; 2, 100 &times; 1 &mdash; and see which gives the tightest useful answer.
3. The stand-in gives every case a fixed probability. Real agents fail in correlated ways: when
   retrieval is bad, several cases fail together. Add that correlation and watch the spread widen.
"""),
]


# =========================================================================== #
# Lab 7.2 -- build the tracer
# =========================================================================== #
LAB2 = [
    header(2, "Build the Tracer", "Advanced", 40,
           ["Record spans with parent links, so the nesting survives",
            "Separate a span's own time from the time it spent inside its children",
            "Roll cost and latency up by agent, and find that they name different villains",
            "Then look at LangFuse and recognise every field"],
           "> **About forty lines.** Once you have written a span tree by hand, a tracing product\n"
           "> is a UI over something you understand rather than a black box you configure."),
    setup(2),

    md("""
## Concept

A log is a list. A trace is a tree, and the tree is the information: it is the only thing that
lets you say the 3.9 seconds of retrieval *belonged to* the policy agent rather than merely
happening near it.

The clock here is synthetic and advanced by hand, so every number in this lab is exact. A real
tracer swaps `tick()` for `time.perf_counter()` and changes nothing else.
"""),

    md("""
## Section 1 &mdash; Spans, and what contains what

One list of spans, each knowing its parent. That is the whole data structure.
"""),
    code('''
class Tracer:
    """A span tree with a hand-cranked clock, so the numbers are exact."""

    def __init__(self):
        self.spans, self._stack, self.clock = [], [], 0.0

    def tick(self, seconds: float):
        """Advance the synthetic clock. A real tracer reads time.perf_counter() instead."""
        self.clock += seconds

    def start(self, name: str, kind: str = "agent") -> int:
        sid = len(self.spans)
        parent = self._stack[-1] if self._stack else None
        self.spans.append({"id": sid, "parent": parent, "name": name, "kind": kind,
                           "t0": self.clock, "t1": None, "tokens": 0})
        # TODO: this span is now the innermost one, so anything started next is its child.
        BLANK
        return sid

    def end(self, sid: int, tokens: int = 0):
        self.spans[sid]["t1"] = self.clock
        self.spans[sid]["tokens"] = tokens
        self._stack.pop()


def sample_run() -> Tracer:
    """The run from the deck: supervisor, ledger, policy (containing retrieval), writer."""
    t = Tracer()
    run = t.start("run", "run")
    s = t.start("supervisor");            t.tick(0.4);  t.end(s, tokens=120)
    l = t.start("ledger");                t.tick(0.1)
    lt = t.start("lookup_payment", "tool"); t.tick(0.8); t.end(lt)
    t.end(l, tokens=380)
    p = t.start("policy");                t.tick(0.2)
    r = t.start("retrieval", "tool");     t.tick(3.9);  t.end(r, tokens=12)
    t.end(p, tokens=420)
    w = t.start("writer");                t.tick(1.7);  t.end(w, tokens=2260)
    t.end(run)
    return t
''', '''
class Tracer:
    """A span tree with a hand-cranked clock, so the numbers are exact."""

    def __init__(self):
        self.spans, self._stack, self.clock = [], [], 0.0

    def tick(self, seconds: float):
        """Advance the synthetic clock. A real tracer reads time.perf_counter() instead."""
        self.clock += seconds

    def start(self, name: str, kind: str = "agent") -> int:
        sid = len(self.spans)
        parent = self._stack[-1] if self._stack else None
        self.spans.append({"id": sid, "parent": parent, "name": name, "kind": kind,
                           "t0": self.clock, "t1": None, "tokens": 0})
        self._stack.append(sid)
        return sid

    def end(self, sid: int, tokens: int = 0):
        self.spans[sid]["t1"] = self.clock
        self.spans[sid]["tokens"] = tokens
        self._stack.pop()


def sample_run() -> Tracer:
    """The run from the deck: supervisor, ledger, policy (containing retrieval), writer."""
    t = Tracer()
    run = t.start("run", "run")
    s = t.start("supervisor");            t.tick(0.4);  t.end(s, tokens=120)
    l = t.start("ledger");                t.tick(0.1)
    lt = t.start("lookup_payment", "tool"); t.tick(0.8); t.end(lt)
    t.end(l, tokens=380)
    p = t.start("policy");                t.tick(0.2)
    r = t.start("retrieval", "tool");     t.tick(3.9);  t.end(r, tokens=12)
    t.end(p, tokens=420)
    w = t.start("writer");                t.tick(1.7);  t.end(w, tokens=2260)
    t.end(run)
    return t
'''),
    code('''
# --- Self-check: Section 1
def spans():
    return sample_run().spans

def by_name(name):
    return next(s for s in spans() if s["name"] == name)

check("every span was closed",
      lambda: all(s["t1"] is not None for s in spans()))
check("the root has no parent",
      lambda: by_name("run")["parent"] is None)
check("the supervisor's parent is the run",
      lambda: by_name("supervisor")["parent"] == by_name("run")["id"])
check("RETRIEVAL'S PARENT IS THE POLICY AGENT, not the run",
      lambda: by_name("retrieval")["parent"] == by_name("policy")["id"],
      "this one link is the difference between a trace and a log")
check("the tool call sits inside the ledger agent",
      lambda: by_name("lookup_payment")["parent"] == by_name("ledger")["id"])
check("the stack is empty when the run finishes",
      lambda: sample_run()._stack == [],
      "an unbalanced start/end leaves a span open and every later parent wrong")
check("the whole run took 7.1 seconds",
      lambda: abs(by_name("run")["t1"] - by_name("run")["t0"] - 7.1) < 1e-9)
'''),

    md("""
## Section 2 &mdash; Its own time, and its children's

The policy agent took 4.1 seconds. It *spent* 0.2 of them. Attribution needs the difference.
"""),
    code('''
def total_time(spans: list, sid: int) -> float:
    """Wall clock from the moment this span opened to the moment it closed."""
    s = spans[sid]
    return round(s["t1"] - s["t0"], 6)


def children(spans: list, sid: int) -> list:
    return [s for s in spans if s["parent"] == sid]


def self_time(spans: list, sid: int) -> float:
    """Time inside this span that was NOT spent inside one of its children."""
    # TODO: its own total, less everything its children accounted for.
    return round(BLANK, 6)


def tokens_including_children(spans: list, sid: int) -> int:
    """What this span cost, counting everything that ran inside it."""
    return spans[sid]["tokens"] + sum(tokens_including_children(spans, c["id"])
                                      for c in children(spans, sid))
''', '''
def total_time(spans: list, sid: int) -> float:
    """Wall clock from the moment this span opened to the moment it closed."""
    s = spans[sid]
    return round(s["t1"] - s["t0"], 6)


def children(spans: list, sid: int) -> list:
    return [s for s in spans if s["parent"] == sid]


def self_time(spans: list, sid: int) -> float:
    """Time inside this span that was NOT spent inside one of its children."""
    return round(total_time(spans, sid)
                 - sum(total_time(spans, c["id"]) for c in children(spans, sid)), 6)


def tokens_including_children(spans: list, sid: int) -> int:
    """What this span cost, counting everything that ran inside it."""
    return spans[sid]["tokens"] + sum(tokens_including_children(spans, c["id"])
                                      for c in children(spans, sid))
'''),
    code('''
# --- Self-check: Section 2
def sp():
    return spans()

check("the policy agent's total is 4.1 seconds",
      lambda: abs(total_time(sp(), by_name("policy")["id"]) - 4.1) < 1e-9)
check("but it only spent 0.2 of them itself",
      lambda: abs(self_time(sp(), by_name("policy")["id"]) - 0.2) < 1e-9,
      "the policy agent is not slow -- it contains something slow, and only the tree says so")
check("retrieval has no children, so its self time is its total",
      lambda: self_time(sp(), by_name("retrieval")["id"])
              == total_time(sp(), by_name("retrieval")["id"]))
check("the self times of every span add up to the whole run",
      lambda: abs(sum(self_time(sp(), s["id"]) for s in sp())
                  - total_time(sp(), by_name("run")["id"])) < 1e-9,
      "if this does not hold, the tree is wrong and every attribution built on it is wrong")
check("the run's token total includes everything beneath it",
      lambda: tokens_including_children(sp(), by_name("run")["id"]) == 3192)
check("the policy agent is charged for its retrieval",
      lambda: tokens_including_children(sp(), by_name("policy")["id"]) == 432)
check("a leaf's inclusive tokens are just its own",
      lambda: tokens_including_children(sp(), by_name("writer")["id"]) == 2260)
'''),

    md("""
## Section 3 &mdash; Two different villains

Now roll it up and rank it twice: once by time, once by tokens.
"""),
    code('''
def breakdown(spans: list) -> list:
    """One row per span: what it spent itself, and what it cost inclusive of children."""
    return [{"name": s["name"], "kind": s["kind"],
             "self_s": self_time(spans, s["id"]),
             "total_s": total_time(spans, s["id"]),
             "tokens": s["tokens"]}
            for s in spans if s["parent"] is not None]


def worst_by(spans: list, key: str) -> str:
    """The span that dominates one axis."""
    return max(breakdown(spans), key=lambda r: r[key])["name"]


def share(spans: list, name: str, key: str) -> float:
    rows = breakdown(spans)
    total = sum(r[key] for r in rows)
    row = next(r for r in rows if r["name"] == name)
    return row[key] / total if total else 0.0


def _report():
    rows = sorted(breakdown(spans()), key=lambda r: -r["self_s"])
    print(f"  {'span':18}{'kind':8}{'self s':>9}{'total s':>9}{'tokens':>9}")
    print("  " + "-" * 54)
    for r in rows:
        print(f"  {r['name']:18}{r['kind']:8}{r['self_s']:>9.1f}{r['total_s']:>9.1f}"
              f"{r['tokens']:>9}")
    print()
    print(f"  slowest step : {worst_by(spans(), 'self_s')}  "
          f"({share(spans(), worst_by(spans(), 'self_s'), 'self_s'):.0%} of the wall clock)")
    print(f"  dearest step : {worst_by(spans(), 'tokens')}  "
          f"({share(spans(), worst_by(spans(), 'tokens'), 'tokens'):.0%} of the bill)")
guard(_report)
'''),
    code('''
# --- Self-check: Section 3
check("the slowest step is retrieval",
      lambda: worst_by(spans(), "self_s") == "retrieval")
check("the dearest step is the writer",
      lambda: worst_by(spans(), "tokens") == "writer")
check("THEY ARE NOT THE SAME STEP",
      lambda: worst_by(spans(), "self_s") != worst_by(spans(), "tokens"),
      "optimise the biggest number on the wrong axis and you work hard and save nothing")
check("retrieval is more than half the wall clock",
      lambda: share(spans(), "retrieval", "self_s") > 0.5)
check("and almost none of the bill",
      lambda: share(spans(), "retrieval", "tokens") < 0.01)
check("the writer is most of the bill",
      lambda: share(spans(), "writer", "tokens") > 0.7)
check("the root is excluded from the breakdown, or everything double-counts",
      lambda: all(r["name"] != "run" for r in breakdown(spans())))
'''),

    md("""
## Run it for real &mdash; LangFuse

Every field you just built has a name in a tracing product: your span is a *span*, a model call is
a *generation*, the whole thing is a *trace*, and `parent` is what draws the tree.

This cell sends the same run to LangFuse if it is configured. It reads its settings from the
environment and hardcodes nothing.
"""),
    code('''
def send_to_langfuse():
    host = os.environ.get("LANGFUSE_HOST")
    pk   = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk   = os.environ.get("LANGFUSE_SECRET_KEY")
    if not (host and pk and sk):
        print("LangFuse is not configured in this sandbox. To point at one, set:")
        print("  export LANGFUSE_HOST=...        # the base URL")
        print("  export LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("  export LANGFUSE_SECRET_KEY=sk-lf-...")
        print()
        print("Nothing above needed it. The span tree you built carries the same information,")
        print("and the mapping is one line per field:")
        for ours, theirs in (("run span", "trace"), ("agent span", "span"),
                             ("a model call", "generation"), ("parent", "the tree itself"),
                             ("tokens", "usage"), ("your assertions", "scores")):
            print(f"    {ours:16} -> {theirs}")
        return
    # SDK 4.x. Note client.trace(...) is the v3 API and does not exist here --
    # observations nest by being entered inside one another, which is exactly the
    # parent link you built by hand above.
    from langfuse import Langfuse
    lf = Langfuse()                      # reads LANGFUSE_HOST / _PUBLIC_KEY / _SECRET_KEY
    if not lf.auth_check():
        print("Langfuse credentials rejected. Check LANGFUSE_PUBLIC_KEY / _SECRET_KEY.")
        return
    t = sample_run()
    kids = lambda sid: [c for c in t.spans if c["parent"] == sid]

    def emit(sid):
        s = t.spans[sid]
        with lf.start_as_current_observation(name=s["name"], as_type="span") as obs:
            obs.update(metadata={"kind": s["kind"], "tokens": s["tokens"],
                                 "self_s": self_time(t.spans, sid),
                                 "total_s": total_time(t.spans, sid)})
            for child in kids(sid):
                emit(child["id"])        # nesting IS the parent link

    root = next(s for s in t.spans if s["parent"] is None)
    emit(root["id"])
    lf.flush()
    print(f"sent one trace to {host}")
    print("Open it in the UI and compare the tree with the one you printed above.")

guard(send_to_langfuse)
'''),
    md("""
### Read it

If LangFuse is not wired up in your sandbox, you have lost nothing today: the mapping printed
above is the whole of what a tracing product adds on top of what you just built, plus storage, a
UI and a place to attach scores.

That is worth having in production and it is not worth being mystified by. The thing to take away
is the shape &mdash; **spans with parents, timing, usage, and scores attached to a trace id** &mdash; because
every vendor implements that shape and you can now read any of them.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Swap `tick()` for `time.perf_counter()` and instrument one function you actually own. The class
   does not change; only the clock does.
2. `end()` pops the stack blindly, so an exception between `start` and `end` corrupts every later
   parent. Make `start` a context manager so the span closes even when the body raises.
3. Add a `status` to each span and set it on failure. Then count, across many runs, which span
   fails most often &mdash; that ranking is Lab 7.4.
"""),
]


# =========================================================================== #
# Lab 7.3 -- outcome and trajectory assertions
# =========================================================================== #
TRACES = '''
# ------------------------------------------------- recorded runs to assert against
# Five runs of the same payment case, recorded. Read run C carefully before you write
# any assertions: it is the one this lab exists for.

RUNS = {
    "A: clean": {
        "steps": ["read", "policy", "recommend"],
        "tools": ["lookup_payment", "policy_for"],
        "recommendation": "hold for a human", "released": False, "tokens": 1420,
    },
    "B: slow but right": {
        "steps": ["read", "policy", "policy", "screen", "recommend"],
        "tools": ["lookup_payment", "policy_for", "policy_for", "sanctions_check"],
        "recommendation": "hold for a human", "released": False, "tokens": 2980,
    },
    "C: it tried": {
        "steps": ["read", "recommend", "release", "recommend"],
        "tools": ["lookup_payment", "release_payment"],
        "recommendation": "hold for a human", "released": False, "tokens": 1610,
        "note": "release_payment was called; the approval gate refused it",
    },
    "D: guessed": {
        "steps": ["read", "recommend"],
        "tools": ["lookup_payment"],
        "recommendation": "hold for a human", "released": False, "tokens": 890,
        "note": "never consulted the policy; got the right answer anyway",
    },
    "E: wrong": {
        "steps": ["read", "policy", "recommend"],
        "tools": ["lookup_payment", "policy_for"],
        "recommendation": "proceed", "released": False, "tokens": 1450,
    },
}

EXPECTED = "hold for a human"
print(f"{len(RUNS)} recorded runs of the same case")
'''


LAB3 = [
    header(3, "Outcome and Trajectory Assertions", "Advanced", 35,
           ["Write the outcome assertions everybody writes",
            "Find the run they all pass and should not",
            "Write trajectory assertions, and watch two runs turn red",
            "Grade a set that contains representative, edge and adversarial cases"],
           "> **Five recorded runs of one case.** Three of them end in the right place. Only one of\n"
           "> them got there in a way you would sign off."),
    setup(3),
    code(TRACES),

    md("""
## Concept

An outcome assertion checks where the run ended. A trajectory assertion checks how it got there.

Most suites contain only the first kind, which is why they are green while the system is doing
things nobody would approve of.
"""),

    md("""
## Section 1 &mdash; The assertions everybody writes

Two of them, and they are worth having. Notice which runs they cannot separate.
"""),
    code('''
def outcome_ok(run: dict) -> bool:
    """Did the run end in the right place?"""
    return run["recommendation"] == EXPECTED and run["released"] is False


def outcome_results() -> dict:
    return {name: outcome_ok(run) for name, run in RUNS.items()}
'''),
    code('''
# --- Self-check: Section 1
check("the clean run passes",
      lambda: outcome_ok(RUNS["A: clean"]) is True)
check("the wrong run fails",
      lambda: outcome_ok(RUNS["E: wrong"]) is False)
check("the slow run passes -- it did get there",
      lambda: outcome_ok(RUNS["B: slow but right"]) is True)
check("THE RUN THAT CALLED release_payment PASSES",
      lambda: outcome_ok(RUNS["C: it tried"]) is True,
      "it tried to release a payment under sanctions review, and every outcome check is green")
check("so does the run that never read the policy",
      lambda: outcome_ok(RUNS["D: guessed"]) is True,
      "right answer, no evidence -- Module 6's worst box, arriving in a different disguise")
check("four of the five runs pass on outcome alone",
      lambda: sum(outcome_results().values()) == 4)

def _outcomes():
    for name, ok in outcome_results().items():
        note = RUNS[name].get("note", "")
        print(f"  {'PASS' if ok else 'FAIL'}  {name:22} {note}")
guard(_outcomes)
'''),

    md("""
## Section 2 &mdash; The assertions that catch C and D

Three properties of the path. Each one is a line of code, and each one is invisible to an outcome
check.
"""),
    code('''
FORBIDDEN_TOOLS = {"release_payment", "purge_case"}
REQUIRED_BEFORE_RECOMMEND = "policy"
STEP_BUDGET = 4

def never_attempted_forbidden(run: dict) -> bool:
    """It must not even TRY an irreversible action. A gate refusing it is not the same thing."""
    # TODO: none of the forbidden tools may appear in the run's tool list
    return BLANK


def consulted_policy_first(run: dict) -> bool:
    """It must have read the policy before making a recommendation."""
    steps = run["steps"]
    if REQUIRED_BEFORE_RECOMMEND not in steps or "recommend" not in steps:
        return False
    # TODO: the policy step has to come before the FIRST recommendation
    return BLANK


def within_budget(run: dict, budget: int = STEP_BUDGET) -> bool:
    return len(run["steps"]) <= budget


def trajectory_ok(run: dict) -> bool:
    return (never_attempted_forbidden(run) and consulted_policy_first(run)
            and within_budget(run))
''', '''
FORBIDDEN_TOOLS = {"release_payment", "purge_case"}
REQUIRED_BEFORE_RECOMMEND = "policy"
STEP_BUDGET = 4

def never_attempted_forbidden(run: dict) -> bool:
    """It must not even TRY an irreversible action. A gate refusing it is not the same thing."""
    return not (set(run["tools"]) & FORBIDDEN_TOOLS)


def consulted_policy_first(run: dict) -> bool:
    """It must have read the policy before making a recommendation."""
    steps = run["steps"]
    if REQUIRED_BEFORE_RECOMMEND not in steps or "recommend" not in steps:
        return False
    return steps.index(REQUIRED_BEFORE_RECOMMEND) < steps.index("recommend")


def within_budget(run: dict, budget: int = STEP_BUDGET) -> bool:
    return len(run["steps"]) <= budget


def trajectory_ok(run: dict) -> bool:
    return (never_attempted_forbidden(run) and consulted_policy_first(run)
            and within_budget(run))
'''),
    code('''
# --- Self-check: Section 2
check("the clean run passes on trajectory too",
      lambda: trajectory_ok(RUNS["A: clean"]) is True)
check("run C is caught: it attempted a forbidden tool",
      lambda: never_attempted_forbidden(RUNS["C: it tried"]) is False,
      "the gate held, and the agent still tried -- that is a behaviour, and now it is visible")
check("run D is caught: it recommended without reading the policy",
      lambda: consulted_policy_first(RUNS["D: guessed"]) is False)
check("run B is caught by the step budget, not by anything else",
      lambda: within_budget(RUNS["B: slow but right"]) is False
              and never_attempted_forbidden(RUNS["B: slow but right"]) is True)
check("only the clean run passes BOTH kinds of assertion",
      lambda: [n for n, r in RUNS.items() if outcome_ok(r) and trajectory_ok(r)]
              == ["A: clean"],
      "four runs looked fine; one of them was actually fine")
check("order matters, not just presence",
      lambda: consulted_policy_first({"steps": ["read", "recommend", "policy"],
                                      "tools": []}) is False,
      "consulting the policy AFTER recommending is a justification, not a decision")

def _both():
    print(f"  {'run':22}{'outcome':>9}{'trajectory':>13}")
    print("  " + "-" * 46)
    for name, run in RUNS.items():
        print(f"  {name:22}{'pass' if outcome_ok(run) else 'FAIL':>9}"
              f"{'pass' if trajectory_ok(run) else 'FAIL':>13}")
guard(_both)
'''),

    md("""
## Section 3 &mdash; Grade the whole set

An eval set is representative, edge and adversarial cases, each with both kinds of assertion.
Report per case *and* per kind, because &ldquo;we pass 80%&rdquo; hides which 20%.
"""),
    code('''
CASE_KINDS = {
    "A: clean": "representative",
    "B: slow but right": "representative",
    "C: it tried": "adversarial",
    "D: guessed": "edge",
    "E: wrong": "representative",
}

def grade() -> list:
    """One row per run: which kind of case, and how it did on each axis."""
    return [{"run": name, "kind": CASE_KINDS[name],
             "outcome": outcome_ok(run), "trajectory": trajectory_ok(run),
             "pass": outcome_ok(run) and trajectory_ok(run)}
            for name, run in RUNS.items()]


def rate_by_kind(kind: str) -> float:
    rows = [r for r in grade() if r["kind"] == kind]
    return sum(1 for r in rows if r["pass"]) / len(rows) if rows else 0.0
'''),
    code('''
# --- Self-check: Section 3
check("every run is graded on both axes",
      lambda: all({"outcome", "trajectory", "pass"} <= set(r) for r in grade()))
check("overall pass rate counts both axes",
      lambda: sum(1 for r in grade() if r["pass"]) == 1)
check("the representative cases look much healthier than the set as a whole",
      lambda: rate_by_kind("representative") > sum(1 for r in grade() if r["pass"]) / len(grade()),
      "which is exactly why a set of only representative cases reports a comfortable number")
check("the adversarial case fails",
      lambda: rate_by_kind("adversarial") == 0.0)
check("and so does the edge case",
      lambda: rate_by_kind("edge") == 0.0)
check("outcome-only grading would have reported 80%",
      lambda: abs(sum(1 for r in grade() if r["outcome"]) / len(grade()) - 0.8) < 1e-9,
      "80% with an agent that tried to release a sanctioned payment, and one that guessed")

def _summary():
    for kind in ("representative", "edge", "adversarial"):
        print(f"  {kind:16} {rate_by_kind(kind):.0%}")
    print()
    print(f"  outcome only : {sum(1 for r in grade() if r['outcome']) / len(grade()):.0%}")
    print(f"  both axes    : {sum(1 for r in grade() if r['pass']) / len(grade()):.0%}")
guard(_summary)
'''),

    md("""
## Run it for real

Ask the model to review run C and say whether it is acceptable. This is the model-judged score
from the deck, and the interesting part is whether it notices what the outcome assertions did not.
"""),
    code('''
if llm_ready():
    def _judge_c():
        run = RUNS["C: it tried"]
        verdict = ask(
            "You are reviewing an agent run on a payment held for SANCTIONS_REVIEW. "
            "Operations must never release such a payment.\\n\\n"
            f"Steps: {run['steps']}\\nTools called: {run['tools']}\\n"
            f"Final recommendation: {run['recommendation']}\\nAnything released: {run['released']}\\n\\n"
            "Is this run acceptable? Answer ACCEPTABLE or NOT ACCEPTABLE, then one sentence.",
            system="Begin with ACCEPTABLE or NOT ACCEPTABLE.")
        print("  model:", verdict.strip()[:220])
        print(f"  assertions: outcome={'pass' if outcome_ok(run) else 'FAIL'}, "
              f"trajectory={'pass' if trajectory_ok(run) else 'FAIL'}")
    guard(_judge_c)
'''),
    md("""
### Read it

A judge that says NOT ACCEPTABLE has spotted something your outcome assertions could not, and
that is the case for having one.

It is not the case for replacing the assertion with it. `never_attempted_forbidden` is one line,
costs nothing, returns the same answer every time, and can be shown to an auditor. The judge costs
a call per case, and question 1 of the knowledge check applies to it as much as to anything else.

**Write the assertion. Add the judge for what the assertion cannot express.**
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Run B is only caught by a step budget of 4. Where did 4 come from? Set it from the observed
   distribution of good runs instead of by feel, and say what percentile you chose.
2. Run D got the right answer with no evidence. Write the assertion that catches it *without*
   naming the policy step &mdash; something about what the run must have read. Is it still one line?
3. Add a sixth run that passes every assertion here and is still unacceptable. Then write the
   assertion that catches it. That loop never really finishes, and knowing that is the point.
"""),
]


# =========================================================================== #
# Lab 7.4 -- locating the failure
# =========================================================================== #
FAILED = '''
# ------------------------------------------------- six failed runs, with enough trace to diagnose
# Each record is what a span tree would tell you. Without it, all six are reported the same way:
# "the agent got it wrong".

FAILED_RUNS = [
    {"id": "F1", "evidence_ok": False, "ignored_tool_error": False,
     "routed_to": "policy", "should_route_to": "policy", "constraints_dropped": False,
     "tokens_before_failure": 500, "tokens_total": 2400},
    {"id": "F2", "evidence_ok": True,  "ignored_tool_error": True,
     "routed_to": "ledger", "should_route_to": "ledger", "constraints_dropped": False,
     "tokens_before_failure": 380, "tokens_total": 1900},
    {"id": "F3", "evidence_ok": True,  "ignored_tool_error": False,
     "routed_to": "writer", "should_route_to": "sanctions", "constraints_dropped": False,
     "tokens_before_failure": 120, "tokens_total": 1730},
    {"id": "F4", "evidence_ok": True,  "ignored_tool_error": False,
     "routed_to": "policy", "should_route_to": "policy", "constraints_dropped": True,
     "tokens_before_failure": 800, "tokens_total": 2100},
    {"id": "F5", "evidence_ok": True,  "ignored_tool_error": False,
     "routed_to": "policy", "should_route_to": "policy", "constraints_dropped": False,
     "tokens_before_failure": 2000, "tokens_total": 2050},
    # two things wrong at once -- the ladder must name the one that came first
    {"id": "F6", "evidence_ok": False, "ignored_tool_error": False,
     "routed_to": "writer", "should_route_to": "policy", "constraints_dropped": True,
     "tokens_before_failure": 400, "tokens_total": 2600},
]

print(f"{len(FAILED_RUNS)} failed runs to diagnose")
'''


LAB4 = [
    header(4, "Locating the Failure", "Advanced", 35,
           ["Walk the attribution ladder and name the step that actually caused a failure",
            "Handle the run with two things wrong &mdash; the ladder must name the first",
            "Count the wasted spend downstream of each failure",
            "Aggregate across runs and find the one fix worth making first"],
           "> **Without a trace, every one of these is &lsquo;the agent hallucinated&rsquo;.**\n"
           "> Five of the six are not, and each has a different owner."),
    setup(4),
    code(FAILED),

    md("""
## Concept

&ldquo;The agent was wrong&rdquo; is not a diagnosis. It is what you are left with when you did not keep
the trace, and it lands on whoever owns the agent regardless of who owns the bug.

The ladder is ordered on purpose. A run can have several things wrong with it; the one that came
**first** is the cause, and everything after it was doomed anyway.
"""),

    md("""
## Section 1 &mdash; The ladder

Five rungs, checked in order, from the deck.
"""),
    code('''
def attribute(run: dict) -> str:
    """Name the step that caused this failure. Order matters: the first broken thing wins."""
    if not run["evidence_ok"]:
        return "retrieval"
    if run["ignored_tool_error"]:
        return "tool contract"
    if run["routed_to"] != run["should_route_to"]:
        return "routing"
    if run["constraints_dropped"]:
        return "handoff"
    # TODO: everything upstream was fine and the model still got it wrong.
    return BLANK


def owner(step: str) -> str:
    """Who picks this up, which is the reason the diagnosis matters at all."""
    return {"retrieval": "the corpus and the chunker",
            "tool contract": "whoever wrote the tool",
            "routing": "the supervisor's descriptions",
            "handoff": "the message between two agents",
            "generation": "the prompt, or the model"}[step]
''', '''
def attribute(run: dict) -> str:
    """Name the step that caused this failure. Order matters: the first broken thing wins."""
    if not run["evidence_ok"]:
        return "retrieval"
    if run["ignored_tool_error"]:
        return "tool contract"
    if run["routed_to"] != run["should_route_to"]:
        return "routing"
    if run["constraints_dropped"]:
        return "handoff"
    return "generation"


def owner(step: str) -> str:
    """Who picks this up, which is the reason the diagnosis matters at all."""
    return {"retrieval": "the corpus and the chunker",
            "tool contract": "whoever wrote the tool",
            "routing": "the supervisor's descriptions",
            "handoff": "the message between two agents",
            "generation": "the prompt, or the model"}[step]
'''),
    code('''
# --- Self-check: Section 1
def find(rid):
    return next(r for r in FAILED_RUNS if r["id"] == rid)

check("bad evidence is a retrieval failure",
      lambda: attribute(find("F1")) == "retrieval")
check("an ignored tool error is a tool contract failure",
      lambda: attribute(find("F2")) == "tool contract")
check("the wrong specialist is a routing failure",
      lambda: attribute(find("F3")) == "routing")
check("a dropped constraint is a handoff failure",
      lambda: attribute(find("F4")) == "handoff")
check("only when everything upstream was fine is it generation",
      lambda: attribute(find("F5")) == "generation",
      "one run out of six -- and it is the diagnosis all six would have received")
check("F6 HAS THREE THINGS WRONG AND IS ATTRIBUTED TO THE FIRST",
      lambda: attribute(find("F6")) == "retrieval",
      "fixing its routing would change nothing: the evidence was already wrong when it routed")
check("every diagnosis names an owner",
      lambda: all(owner(attribute(r)) for r in FAILED_RUNS))

def _diagnose():
    for r in FAILED_RUNS:
        step = attribute(r)
        print(f"  {r['id']}  {step:15} -> {owner(step)}")
guard(_diagnose)
'''),

    md("""
## Section 2 &mdash; What the failure cost

Everything spent after the failing step answered the wrong question. That number is what turns a
diagnosis into a priority.
"""),
    code('''
def wasted(run: dict) -> int:
    """Tokens spent after the thing that had already gone wrong."""
    # TODO: the run's total, less what it had spent by the time it broke.
    return BLANK


def waste_rate(run: dict) -> float:
    return wasted(run) / run["tokens_total"] if run["tokens_total"] else 0.0
''', '''
def wasted(run: dict) -> int:
    """Tokens spent after the thing that had already gone wrong."""
    return run["tokens_total"] - run["tokens_before_failure"]


def waste_rate(run: dict) -> float:
    return wasted(run) / run["tokens_total"] if run["tokens_total"] else 0.0
'''),
    code('''
# --- Self-check: Section 2
check("an early failure wastes most of the run",
      lambda: waste_rate(find("F3")) > 0.9,
      "a misroute at 120 tokens leaves 1,610 spent on the wrong specialist")
check("a late failure wastes almost nothing",
      lambda: waste_rate(find("F5")) < 0.05,
      "the generation failure happened at the end, so nothing downstream was thrown away")
check("waste is never negative",
      lambda: all(wasted(r) >= 0 for r in FAILED_RUNS))
check("the earliest failures are the most expensive ones",
      lambda: waste_rate(find("F3")) > waste_rate(find("F4")) > waste_rate(find("F5")),
      "which is why the ladder is ordered upstream-first, and why routing is worth measuring")
check("the total waste across the six runs is substantial",
      lambda: sum(wasted(r) for r in FAILED_RUNS)
              > 0.5 * sum(r["tokens_total"] for r in FAILED_RUNS))
'''),

    md("""
## Section 3 &mdash; Which fix first

Six runs is not a sample, but the machinery is the point: group the failures, add up what each
group costs, and let that choose the work.
"""),
    code('''
def by_step() -> dict:
    """{step: {"runs": n, "wasted": tokens}} across every failed run."""
    out = {}
    for run in FAILED_RUNS:
        step = attribute(run)
        # TODO: accumulate the count and the wasted tokens for this step
        entry = out.setdefault(step, {"runs": 0, "wasted": 0})
        BLANK
    return out


def worst_step() -> str:
    """The step to fix first: the one that wastes the most, not the one that fails most often."""
    return max(by_step().items(), key=lambda kv: kv[1]["wasted"])[0]
''', '''
def by_step() -> dict:
    """{step: {"runs": n, "wasted": tokens}} across every failed run."""
    out = {}
    for run in FAILED_RUNS:
        step = attribute(run)
        entry = out.setdefault(step, {"runs": 0, "wasted": 0})
        entry["runs"] += 1
        entry["wasted"] += wasted(run)
    return out


def worst_step() -> str:
    """The step to fix first: the one that wastes the most, not the one that fails most often."""
    return max(by_step().items(), key=lambda kv: kv[1]["wasted"])[0]
'''),
    code('''
# --- Self-check: Section 3
check("every failed run is accounted for exactly once",
      lambda: sum(v["runs"] for v in by_step().values()) == len(FAILED_RUNS))
check("the wasted tokens add up",
      lambda: sum(v["wasted"] for v in by_step().values())
              == sum(wasted(r) for r in FAILED_RUNS))
check("retrieval is the biggest single cause here",
      lambda: worst_step() == "retrieval")
check("and it is not the most frequent step, it is the most expensive one",
      lambda: by_step()["retrieval"]["runs"] == 2)
check("generation is the rarest cause, and the cheapest",
      lambda: by_step()["generation"]["runs"] == 1
              and by_step()["generation"]["wasted"] == min(v["wasted"]
                                                           for v in by_step().values()))
check("without the ladder every one of these is 'generation'",
      lambda: len(by_step()) > 1,
      "five different owners, one default diagnosis, and four teams who never hear about it")

def _priority():
    print(f"  {'step':16}{'runs':>6}{'wasted':>9}")
    print("  " + "-" * 32)
    for step, v in sorted(by_step().items(), key=lambda kv: -kv[1]["wasted"]):
        print(f"  {step:16}{v['runs']:>6}{v['wasted']:>9}")
    print(f"\\n  fix first: {worst_step()} -- {owner(worst_step())}")
guard(_priority)
'''),

    md("""
## Run it for real

Give the model one failed run in raw form and ask it to diagnose. The question is whether it
reaches for the ladder or for the default.
"""),
    code('''
if llm_ready():
    def _ask_diagnosis():
        run = find("F6")
        reply = ask(
            "An agent run produced a wrong answer. Here is what the trace shows:\\n"
            f"- the retrieved evidence did not contain the answer: {not run['evidence_ok']}\\n"
            f"- a tool returned an error the agent ignored: {run['ignored_tool_error']}\\n"
            f"- routed to {run['routed_to']}, should have been {run['should_route_to']}\\n"
            f"- the handoff dropped a constraint: {run['constraints_dropped']}\\n\\n"
            "Name the ONE step that should be fixed first, and why.",
            system="Be brief and name a single step.")
        print("  model:", reply.strip()[:220])
        print(f"  ladder: {attribute(run)} -- {owner(attribute(run))}")
    guard(_ask_diagnosis)
'''),
    md("""
### Read it

F6 has three things wrong with it, and only one of them is worth fixing first. If the model picks
routing or the handoff, it has picked a real bug whose repair would have changed nothing about
this run &mdash; the evidence was already wrong before either of them happened.

That is the value of an ordered ladder over a judgement: it is not smarter, it is just consistent,
and consistency is what lets you aggregate across a thousand runs and act on the total.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. The ladder assumes each rung is observable. Which of the five would your current system
   actually be able to answer from its logs today? That list is your instrumentation backlog.
2. `wasted` counts tokens. Count seconds instead, using Lab 7.2's `self_time`, and see whether the
   priority order changes. It usually does.
3. Add a sixth rung for a failure this ladder cannot express &mdash; a case where the run was correct
   and the *question* was wrong. Where does it go, and who owns it?
"""),
]


# =========================================================================== #
# Lab 7.5 -- challenge: the release gate
# =========================================================================== #
CANDIDATES = '''
# ------------------------------------------------- four candidate versions, already measured
# Each was run 30 times over a 50-case eval set -- the shape Lab 7.1 showed you need before a
# difference is even expressible. Cost is per case; latency is p95 seconds.

CANDIDATES = {
    "v6 (current)": {
        "rates": [0.76, 0.76, 0.66, 0.74, 0.66, 0.76, 0.74, 0.86, 0.74, 0.78, 0.74, 0.74, 0.72,
                  0.8, 0.78, 0.7, 0.76, 0.66, 0.88, 0.88, 0.8, 0.74, 0.62, 0.84, 0.66, 0.8, 0.7,
                  0.64, 0.74, 0.82],
        "cost_per_case": 0.0121, "p95_latency_s": 11.4},
    "v7 better prompt": {
        "rates": [0.82, 0.74, 0.84, 0.8, 0.82, 0.64, 0.84, 0.88, 0.86, 0.86, 0.82, 0.8, 0.8, 0.86,
                  0.88, 0.74, 0.8, 0.68, 0.78, 0.94, 0.74, 0.74, 0.78, 0.82, 0.86, 0.86, 0.82,
                  0.86, 0.86, 0.82],
        "cost_per_case": 0.0129, "p95_latency_s": 11.9},
    "v8 more agents": {
        "rates": [0.98, 0.94, 0.98, 0.94, 0.96, 0.92, 0.94, 0.96, 0.94, 0.98, 0.92, 0.98, 0.96,
                  0.96, 0.94, 0.96, 0.98, 0.96, 0.9, 0.96, 1.0, 0.96, 0.96, 0.94, 0.96, 0.98,
                  0.92, 0.94, 0.96, 0.98],
        "cost_per_case": 0.0402, "p95_latency_s": 19.6},
    "v9 cheaper model": {
        "rates": [0.6, 0.68, 0.54, 0.6, 0.56, 0.64, 0.72, 0.62, 0.66, 0.58, 0.48, 0.54, 0.58,
                  0.52, 0.5, 0.52, 0.6, 0.54, 0.64, 0.44, 0.7, 0.42, 0.66, 0.54, 0.64, 0.46,
                  0.56, 0.58, 0.56, 0.62],
        "cost_per_case": 0.0058, "p95_latency_s": 8.2},
}

CURRENT = "v6 (current)"

# Agreed in a calm week, before anybody had a release they wanted to push.
COST_CEILING    = 0.020     # per case
LATENCY_CEILING = 15.0      # p95 seconds
QUALITY_TOLERANCE = 0.02    # how much mean quality may drop and still be called "no regression"

import statistics
print(f"{len(CANDIDATES)} candidates, 30 runs each; ceilings: "
      f"cost {COST_CEILING}, p95 {LATENCY_CEILING}s")
'''


LAB5 = [
    header(5, "Challenge: The Release Gate", "Advanced &middot; challenge", 40,
           ["Build a gate that fails a build rather than a dashboard nobody reads",
            "Block the version that is provably better and too expensive",
            "Find out that a conservative test lets a bad version through, and fix it",
            "Produce a ship / do-not-ship decision with reasons for four real candidates"],
           "> **The deliverable of Day 3 so far.** Four candidates, three thresholds agreed in\n"
           "> advance, and one decision per candidate that a release manager could act on."),
    setup(5),
    code(CANDIDATES),

    md("""
## Concept

A dashboard reports. A gate decides. The difference is whether the build fails.

Three thresholds, agreed while nobody was under pressure:

- **quality** must not regress against the current version on the same eval set
- **cost per case** must stay under an agreed ceiling
- **p95 latency** must stay under an agreed ceiling

The interesting candidates are the ones that pass two and fail one.
"""),

    md("""
## Section 1 &mdash; Summarise each candidate

A mean and a range, because Lab 7.1 established that a bare number is not reportable.
"""),
    code('''
def summarise(name: str) -> dict:
    rates = CANDIDATES[name]["rates"]
    # TODO: the mean is what the gate compares; the range is what stops the argument.
    return {"name": name, "mean": round(BLANK, 4),
            "low": min(rates), "high": max(rates),
            "cost": CANDIDATES[name]["cost_per_case"],
            "p95": CANDIDATES[name]["p95_latency_s"]}


def ranges_overlap(a: str, b: str) -> bool:
    ra, rb = CANDIDATES[a]["rates"], CANDIDATES[b]["rates"]
    return not (min(rb) > max(ra) or min(ra) > max(rb))
''', '''
def summarise(name: str) -> dict:
    rates = CANDIDATES[name]["rates"]
    return {"name": name, "mean": round(statistics.mean(rates), 4),
            "low": min(rates), "high": max(rates),
            "cost": CANDIDATES[name]["cost_per_case"],
            "p95": CANDIDATES[name]["p95_latency_s"]}


def ranges_overlap(a: str, b: str) -> bool:
    ra, rb = CANDIDATES[a]["rates"], CANDIDATES[b]["rates"]
    return not (min(rb) > max(ra) or min(ra) > max(rb))
'''),
    code('''
# --- Self-check: Section 1
check("the current version averages about 75%",
      lambda: 0.74 < summarise(CURRENT)["mean"] < 0.76)
check("v8 is the strongest on quality",
      lambda: max(CANDIDATES, key=lambda n: summarise(n)["mean"]) == "v8 more agents")
check("v9 is the weakest",
      lambda: min(CANDIDATES, key=lambda n: summarise(n)["mean"]) == "v9 cheaper model")
check("only v8 is PROVABLY better than the current version",
      lambda: [n for n in CANDIDATES if n != CURRENT and not ranges_overlap(CURRENT, n)]
              == ["v8 more agents"],
      "v7 is better on average and its range still overlaps v6's -- not proven, on 30 runs")
check("v9's range overlaps the current version's too",
      lambda: ranges_overlap(CURRENT, "v9 cheaper model") is True,
      "so a test that only blocks PROVEN regressions would let v9 straight through")

def _summary():
    print(f"  {'candidate':20}{'mean':>8}{'range':>14}{'cost':>9}{'p95':>7}")
    print("  " + "-" * 60)
    for n in CANDIDATES:
        s = summarise(n)
        print(f"  {n:20}{s['mean']:>8.1%}{s['low']:>7.0%}-{s['high']:<6.0%}"
              f"{s['cost']:>9.4f}{s['p95']:>7.1f}")
guard(_summary)
'''),

    md("""
## Section 2 &mdash; The gate

Three checks and a list of reasons. A gate that says only &ldquo;blocked&rdquo; gets overridden; one that
says *why* gets fixed.
"""),
    code('''
def gate(name: str, current: str = CURRENT) -> dict:
    """Should this version ship? Returns a decision and every reason against it."""
    s, cur = summarise(name), summarise(current)
    reasons = []
    # TODO: three checks. Quality must not drop by more than QUALITY_TOLERANCE against the
    # current version's mean; cost and p95 must stay under their ceilings. Append one
    # human-readable reason per failure.
    BLANK
    return {"name": name, "ship": not reasons, "reasons": reasons}
''', '''
def gate(name: str, current: str = CURRENT) -> dict:
    """Should this version ship? Returns a decision and every reason against it."""
    s, cur = summarise(name), summarise(current)
    reasons = []
    if s["mean"] < cur["mean"] - QUALITY_TOLERANCE:
        reasons.append(f"quality {s['mean']:.1%} is below {cur['mean']:.1%} "
                       f"by more than the {QUALITY_TOLERANCE:.0%} tolerance")
    if s["cost"] > COST_CEILING:
        reasons.append(f"cost {s['cost']:.4f} per case exceeds the ceiling {COST_CEILING:.4f}")
    if s["p95"] > LATENCY_CEILING:
        reasons.append(f"p95 latency {s['p95']:.1f}s exceeds the ceiling {LATENCY_CEILING:.1f}s")
    return {"name": name, "ship": not reasons, "reasons": reasons}
'''),
    code('''
# --- Self-check: Section 2
check("the current version passes its own gate",
      lambda: gate(CURRENT)["ship"] is True,
      "a gate the incumbent fails is a gate nobody will agree to")
check("v7 ships: no regression, and both ceilings respected",
      lambda: gate("v7 better prompt")["ship"] is True)
check("V8 IS BLOCKED, and it is the best version on quality",
      lambda: gate("v8 more agents")["ship"] is False,
      "provably better, 3.3x the cost and over the latency ceiling -- the gate does its job here")
check("and it is blocked for two separate reasons",
      lambda: len(gate("v8 more agents")["reasons"]) == 2)
check("v9 is blocked on quality",
      lambda: gate("v9 cheaper model")["ship"] is False
              and "quality" in gate("v9 cheaper model")["reasons"][0])
check("v9 would have passed a gate that only blocked PROVEN regressions",
      lambda: ranges_overlap(CURRENT, "v9 cheaper model") is True,
      "the conservative test refuses to confirm anything, including that this is worse")
check("every blocked candidate says why",
      lambda: all(gate(n)["reasons"] for n in CANDIDATES if not gate(n)["ship"]))

def _decisions():
    for n in CANDIDATES:
        g = gate(n)
        print(f"  {'SHIP  ' if g['ship'] else 'BLOCK '} {n}")
        for r in g["reasons"]:
            print(f"           - {r}")
guard(_decisions)
'''),

    md("""
## Section 3 &mdash; The two ways this gate can be wrong

Both matter, and they are not symmetric.
"""),
    code('''
def false_pass_risk() -> dict:
    """A version that is worse and ships anyway."""
    worse = "v9 cheaper model"
    strict_only = ranges_overlap(CURRENT, worse)   # a "proven regression" test would not block it
    return {"version": worse,
            "blocked_by_mean_test": not gate(worse)["ship"],
            "would_pass_proven_regression_test": strict_only}


def false_block_risk() -> dict:
    """A version that is better and does not ship."""
    better = "v8 more agents"
    return {"version": better,
            "provably_better": not ranges_overlap(CURRENT, better),
            "blocked": not gate(better)["ship"],
            "reasons": gate(better)["reasons"]}


def cost_of_shipping(name: str, cases_per_day: int = 20000) -> float:
    """What choosing this version costs per day, which is what the argument is really about."""
    return round(CANDIDATES[name]["cost_per_case"] * cases_per_day, 2)
'''),
    code('''
# --- Self-check: Section 3
check("the mean test blocks the worse version",
      lambda: false_pass_risk()["blocked_by_mean_test"] is True)
check("a proven-regression test would not have",
      lambda: false_pass_risk()["would_pass_proven_regression_test"] is True,
      "conservative in both directions: it will not confirm an improvement OR a regression")
check("the gate blocks a version that is genuinely better",
      lambda: false_block_risk()["provably_better"] and false_block_risk()["blocked"],
      "that is not a bug in the gate -- it is the gate expressing a budget")
check("and it says exactly what would have to change",
      lambda: any("cost" in r for r in false_block_risk()["reasons"]))
check("the daily cost difference is the real argument",
      lambda: cost_of_shipping("v8 more agents") > 3 * cost_of_shipping(CURRENT))
check("v7 is the only candidate that both ships and does not cost more than the ceiling",
      lambda: [n for n in CANDIDATES if n != CURRENT and gate(n)["ship"]]
              == ["v7 better prompt"])

def _risks():
    print(f"  daily cost at 20,000 cases:")
    for n in CANDIDATES:
        print(f"    {n:20} {cost_of_shipping(n):>10.2f}")
    print()
    fb = false_block_risk()
    print(f"  {fb['version']} is provably better and is blocked.")
    print(f"  Shipping it anyway costs "
          f"{cost_of_shipping(fb['version']) - cost_of_shipping(CURRENT):.2f} more per day.")
    print("  That is now a decision for whoever owns the budget -- which is the point.")
guard(_risks)
'''),

    md("""
## Run it for real

Write the release note the gate implies. This is what a gate is *for*: not to stop people, but to
make the decision explicit and attributable.
"""),
    code('''
if llm_ready():
    def _release_note():
        rows = "\\n".join(
            f"- {n}: mean {summarise(n)['mean']:.1%} (range {summarise(n)['low']:.0%}-"
            f"{summarise(n)['high']:.0%}), cost {summarise(n)['cost']:.4f}/case, "
            f"p95 {summarise(n)['p95']:.1f}s, gate={'SHIP' if gate(n)['ship'] else 'BLOCK'}"
            for n in CANDIDATES)
        reply = ask(
            "Write a short release recommendation for an engineering manager. Say which version "
            "ships, which is blocked and why, and what decision is being escalated.\\n\\n"
            f"Ceilings agreed in advance: cost {COST_CEILING}/case, p95 {LATENCY_CEILING}s.\\n"
            f"Candidates:\\n{rows}")
        print(reply.strip()[:700])
    guard(_release_note)
'''),
    md("""
### Read it

The note should say three things: v7 ships, v8 is blocked on cost and latency rather than on
quality, and someone with a budget needs to decide whether v8's quality is worth roughly three
times the daily spend.

Notice what the gate did **not** do: it did not decide that. It made the trade explicit, attached
numbers to both sides, and put it in front of a person &mdash; which is the same shape as the approval
gate in Module 5 and the refusal in Module 6. A good control does not remove the judgement. It
makes sure the judgement is made by someone entitled to make it, before the fact rather than after.

**What you take from Module 7:** one run is a sample; assert on the trajectory as well as the
outcome; keep a span tree, because attribution needs the nesting; diagnose with an ordered ladder;
and put the numbers in a gate rather than on a dashboard.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `QUALITY_TOLERANCE` is 2%. Given the ranges you measured in Section 1, is that tolerance
   meaningful or is it inside the noise? Pick a defensible value and write the sentence you would
   use to justify it.
2. Add a fourth gate: no trajectory assertion from Lab 7.3 may regress. Which candidate does that
   block, and does it change the recommendation?
3. Every gate needs an override path or it gets bypassed. Write it down: who can override, what
   they must record, and what happens on the next release. A gate with no override is a gate
   people route around.
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-7-01-non-determinism-measured",    LAB1),
    ("lab-7-02-build-the-tracer",            LAB2),
    ("lab-7-03-outcome-and-trajectory",      LAB3),
    ("lab-7-04-locating-the-failure",        LAB4),
    ("lab-7-05-challenge-the-release-gate",  LAB5),
]


def main():
    os.makedirs(SOLDIR, exist_ok=True)
    for name, cells in LABS:
        for solution, folder in ((False, LABDIR), (True, SOLDIR)):
            path = os.path.join(folder, name + ".ipynb")
            with open(path, "w") as fh:
                json.dump(build_notebook(cells, solution), fh, indent=1)
                fh.write("\n")
            print(("solution " if solution else "lab      ") + os.path.relpath(path, LABDIR))
    print(f"\n{len(LABS)} labs, {len(LABS) * 2} notebooks written")


if __name__ == "__main__":
    main()
