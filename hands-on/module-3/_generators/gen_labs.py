#!/usr/bin/env python3
"""
Generate Module 3 lab notebooks and their solutions from one source.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-1-0N-*.ipynb and ../solutions/

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
# Lab 3.{num} &mdash; {title}

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 1 &middot; Module 3 &mdash; Memory, State &amp; the LangGraph Substrate**

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

WORK = os.path.join("/tmp", "awmas-lab-3-{num:02d}")
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
# One domain runs through all five Module 1 labs: payment exceptions on a small ledger.
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




# the two tools you wrote in Lab 1.2 of Module 1, carried forward so this notebook stands alone
CARRIED_TOOLS = '''
# ------------------------------------------------- carried forward from Lab 1.2 of Module 1
# These are the tools you wrote in Lab 1.2 of Module 1. Nothing to fill in -- they are here so this
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


# =========================================================================== #
# Lab 3.1 -- memory that survives, and what compaction throws away
# =========================================================================== #
LAB1 = [
    header(1, "Memory That Survives a Long Session", "Intermediate", 30,
           ["Measure the turn at which buffer memory loses its standing instruction",
            "Implement summary compaction that keeps the constraint, not just the recent text",
            "Add episodic memory -- what happened last time, and whether that helps",
            "Decide deliberately what gets forgotten"],
           "> **Builds on Lab 1.2's `ShortTermMemory`.** There you kept the window bounded.\n"
           "> Here you find out what bounding it cost you, and fix the part that mattered."),
    setup(1),
    code(DOMAIN),

    md("""
## Concept

Buffer memory grows linearly and the context window is fixed, so there is always a turn **N** at
which the earliest content falls out. Usually that content is the standing instruction, so the
agent quietly stops obeying it. Nothing errors.

Compaction bounds the window, but it is **lossy by construction**. The engineering question is not
whether to lose something &mdash; it is whether you chose what.
"""),

    md("""
## Section 1 &mdash; Find the failure turn

Before fixing anything, measure it. Given a window budget and an average turn size, at what turn
does the standing instruction fall out of a pure buffer?
"""),
    code('''
WINDOW_TOKENS = 800                  # a deliberately small window, so the effect is visible
INSTRUCTION_TOKENS = 60              # the standing instruction, sent first
TURN_TOKENS = 40                     # an average turn

def failure_turn(window=WINDOW_TOKENS, instruction=INSTRUCTION_TOKENS, per_turn=TURN_TOKENS):
    """The first turn number at which the instruction no longer fits alongside the turns.

    Buffer memory sends: instruction + every turn so far. Once that exceeds the window,
    the oldest content -- the instruction -- is what gets dropped.
    """
    turn = 0
    while True:
        turn += 1
        used = instruction + turn * per_turn
        if BLANK:                    # TODO: has the buffer outgrown the window?
            return turn
''', '''
WINDOW_TOKENS = 800                  # a deliberately small window, so the effect is visible
INSTRUCTION_TOKENS = 60              # the standing instruction, sent first
TURN_TOKENS = 40                     # an average turn

def failure_turn(window=WINDOW_TOKENS, instruction=INSTRUCTION_TOKENS, per_turn=TURN_TOKENS):
    """The first turn number at which the instruction no longer fits alongside the turns.

    Buffer memory sends: instruction + every turn so far. Once that exceeds the window,
    the oldest content -- the instruction -- is what gets dropped.
    """
    turn = 0
    while True:
        turn += 1
        used = instruction + turn * per_turn
        if used > window:
            return turn
'''),
    code('''
# --- Self-check: Section 1
check("the failure turn is computed, not guessed", lambda: failure_turn() == 19,
      "60 + 19*40 = 820 > 800, while 60 + 18*40 = 780 still fits")
check("a bigger window survives longer",
      lambda: failure_turn(window=4000) > failure_turn(window=800))
check("chattier turns fail sooner",
      lambda: failure_turn(per_turn=200) < failure_turn(per_turn=40))
check("the instruction's own size matters",
      lambda: failure_turn(instruction=600) < failure_turn(instruction=60))

for w in (800, 4000, 32000):
    try:
        print(f"  window {w:>6} tokens -> instruction drops out at turn {failure_turn(window=w)}")
    except NameError:
        print("(fill in failure_turn above)"); break
'''),

    md("""
## Section 2 &mdash; Compaction that keeps what matters

Lab 1.2 kept the recent half and folded the rest into a summary. That bounds the window but can
still lose the standing instruction. Pin it instead.
"""),
    code('''
class Memory:
    """Recent turns verbatim, older ones summarised, and a pinned instruction that never ages out."""

    def __init__(self, instruction: str, max_turns: int = 6):
        self.instruction = instruction
        self.max_turns = max_turns
        self.turns: list[tuple[str, str]] = []
        self.summary = ""

    def add(self, role: str, text: str) -> None:
        self.turns.append((role, text))
        if len(self.turns) > self.max_turns:
            self.compact()

    def compact(self) -> None:
        keep = max(1, self.max_turns // 2)
        older, self.turns = self.turns[:-keep], self.turns[-keep:]
        self.summary = (self.summary + " " + " ".join(t for _, t in older)).strip()

    def render(self) -> list[tuple[str, str]]:
        """The messages to send. The instruction is pinned FIRST and always present."""
        out = BLANK                  # TODO: start with the pinned instruction as a system message
        if self.summary:
            out = out + [("system", "Earlier in this case: " + self.summary)]
        return out + list(self.turns)
''', '''
class Memory:
    """Recent turns verbatim, older ones summarised, and a pinned instruction that never ages out."""

    def __init__(self, instruction: str, max_turns: int = 6):
        self.instruction = instruction
        self.max_turns = max_turns
        self.turns: list[tuple[str, str]] = []
        self.summary = ""

    def add(self, role: str, text: str) -> None:
        self.turns.append((role, text))
        if len(self.turns) > self.max_turns:
            self.compact()

    def compact(self) -> None:
        keep = max(1, self.max_turns // 2)
        older, self.turns = self.turns[:-keep], self.turns[-keep:]
        self.summary = (self.summary + " " + " ".join(t for _, t in older)).strip()

    def render(self) -> list[tuple[str, str]]:
        """The messages to send. The instruction is pinned FIRST and always present."""
        out = [("system", self.instruction)]
        if self.summary:
            out = out + [("system", "Earlier in this case: " + self.summary)]
        return out + list(self.turns)
'''),
    code('''
# --- Self-check: Section 2
INSTRUCTION = "Never propose releasing a payment that policy reserves for a human."

def _aged(n=40):
    m = Memory(INSTRUCTION, max_turns=6)
    for i in range(n):
        m.add("human" if i % 2 == 0 else "ai", f"turn {i}")
    return m

check("the window stays bounded after 40 turns", lambda: len(_aged().turns) <= 6)
check("the instruction is still present at turn 40",
      lambda: any(INSTRUCTION in t for _, t in _aged().render()),
      "pin it in render() -- it must not be subject to compaction")
check("the instruction comes first", lambda: _aged().render()[0][1] == INSTRUCTION)
check("the summary sits between the instruction and the recent turns",
      lambda: _aged().render()[1][0] == "system" and "Earlier" in _aged().render()[1][1])
check("a short conversation carries no summary",
      lambda: len(Memory(INSTRUCTION, 6).render()) == 1)
'''),

    md("""
## Section 3 &mdash; What compaction threw away

Bounded is not the same as harmless. Measure the loss: which specific facts from early turns can no
longer be recovered from the rendered context?
"""),
    code('''
def recoverable(memory: Memory, fact: str) -> bool:
    """True when `fact` can still be found anywhere in what would be sent to the model."""
    return any(fact.lower() in text.lower() for _, text in memory.render())

def compaction_loss(facts: list[str], turns: int = 40) -> list[str]:
    """Run a session of `turns` turns, stating each fact early, and return the facts lost."""
    m = Memory(INSTRUCTION, max_turns=6)
    for f in facts:
        m.add("human", f)
    for i in range(turns):
        m.add("ai", f"working, step {i}")
    return BLANK                     # TODO: the facts that are no longer recoverable
''', '''
def recoverable(memory: Memory, fact: str) -> bool:
    """True when `fact` can still be found anywhere in what would be sent to the model."""
    return any(fact.lower() in text.lower() for _, text in memory.render())

def compaction_loss(facts: list[str], turns: int = 40) -> list[str]:
    """Run a session of `turns` turns, stating each fact early, and return the facts lost."""
    m = Memory(INSTRUCTION, max_turns=6)
    for f in facts:
        m.add("human", f)
    for i in range(turns):
        m.add("ai", f"working, step {i}")
    return [f for f in facts if not recoverable(m, f)]
'''),
    code('''
# --- Self-check: Section 3
FACTS = ["The case reference is PMT-1005.",
         "The client contact is the Frankfurt desk.",
         "Do not contact the counterparty directly."]

check("concatenating summaries keeps early facts recoverable",
      lambda: compaction_loss(FACTS) == [],
      "this compaction folds text rather than discarding it -- so nothing is lost YET")
check("the instruction survives regardless",
      lambda: recoverable(_aged(), "reserves for a human"))

# ...but a summariser that REWRITES rather than concatenates does lose things:
class LossyMemory(Memory):
    def compact(self):
        keep = max(1, self.max_turns // 2)
        older, self.turns = self.turns[:-keep], self.turns[-keep:]
        self.summary = f"[{len(older)} earlier turns summarised]"   # a real summariser, abbreviating

def _lossy():
    m = LossyMemory(INSTRUCTION, max_turns=6)
    for f in FACTS:
        m.add("human", f)
    for i in range(40):
        m.add("ai", f"step {i}")
    return m

check("a rewriting summariser DOES lose the early facts",
      lambda: not recoverable(_lossy(), "Frankfurt"),
      "this is the real behaviour of an LLM summariser, and the reason to pin what matters")
check("...but the pinned instruction still survives it",
      lambda: recoverable(_lossy(), "reserves for a human"),
      "pinning is what makes compaction safe to use")
'''),

    md("""
## Run it for real

Ask the model to compact, which is what a production summariser does &mdash; then check whether your
facts survived it.
"""),
    code('''
if llm_ready():
    try:
        transcript = " ".join([
            "The case reference is PMT-1005.",
            "The client contact is the Frankfurt desk.",
            "Do not contact the counterparty directly.",
        ] + [f"Analyst checked step {i} and found nothing unusual." for i in range(20)])

        summary = ask(
            "Summarise this case transcript in at most two sentences for an operations handover.\\n\\n"
            + transcript)
        print("SUMMARY:\\n  " + summary.strip().replace("\\n", "\\n  "))
        print("\\nsurvived compaction?")
        for f in ("PMT-1005", "Frankfurt", "counterparty"):
            print(f"  {f:16} {'yes' if f.lower() in summary.lower() else 'NO -- lost'}")
    except NameError:
        print("(fill in the blanks above, then re-run this cell)")
'''),
    md("""
### Read it

Whatever the model dropped, it dropped **silently and plausibly** &mdash; the summary reads fine. That is
the whole risk: compaction failures are invisible at the point they happen and only surface later,
as an agent that has stopped honouring something it was told.

Two defences, in order: **pin** what must never be lost, and **test** what your summariser keeps
against a list of facts you care about. This lab is that test.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add an `episodic` list to `Memory` holding one line per past case, and include the three most
   recent in `render()`. Then ask: when would recalling a past case make the agent *worse*?
2. Pinning costs tokens on every single turn. Work out the break-even: how long must a session be
   before pinning a 60-token instruction is cheaper than the failure it prevents?
"""),
]


# =========================================================================== #
# Lab 3.2 -- perception: raw output into an observation
# =========================================================================== #
LAB2 = [
    header(2, "Perception &mdash; Turning Raw Output into an Observation", "Intermediate &rarr; Advanced", 30,
           ["Turn an opaque record into something the model cannot misread",
            "Distinguish the four kinds of nothing a tool can return",
            "Stamp freshness, completeness and authority onto every observation",
            "Measure the token cost of being clear -- it is usually negative"],
           "> **The cheapest accuracy in the course.** No model change, no extra call: you are only\n"
           "> deciding what the agent gets to see."),
    setup(2),
    code(DOMAIN),

    md("""
## Concept

A tool returns data. An **observation** is data plus the meaning the agent needs to act on it. Most
agent failures blamed on reasoning are really the model guessing at a field it was never told how
to read &mdash; and guessing the reassuring way.
"""),

    md("""
## Section 1 &mdash; Decode the opaque record

The upstream system speaks in codes. The agent should never have to guess what they mean.
"""),
    code('''
# what the upstream ledger actually returns
RAW = {"i": 1003, "a": 990000.0, "c": "USD", "s": 2, "rc": 7,
       "vd": "2026-09-02", "cp": 19, "f": None, "x": [], "ts": 1757030400}

STATUS_CODES = {0: "settled", 1: "failed", 2: "held"}
REASON_CODES = {5: "INSUFFICIENT_FUNDS", 6: "INVALID_IBAN", 7: "LIMIT_BREACH", 8: "SANCTIONS_REVIEW"}
LIMIT_USD = 500000.0

def observe(raw: dict) -> str:
    """Render one raw ledger record as an observation a model can act on.

    Must state: the reference, the decoded status, the decoded reason, and -- where the
    reason is a limit breach -- the comparison that explains it.
    """
    status = STATUS_CODES.get(raw["s"], "unknown")
    reason = REASON_CODES.get(raw["rc"], "unknown")
    lines = [f"PMT-{raw['i']} is {status.upper()}.", f"Reason: {reason}"]
    if reason == "LIMIT_BREACH":
        lines.append(BLANK)          # TODO: the comparison that makes the reason self-evident,
                                     # e.g. "USD 990,000.00 exceeds the 500,000.00 limit."
    return " ".join(lines)
''', '''
# what the upstream ledger actually returns
RAW = {"i": 1003, "a": 990000.0, "c": "USD", "s": 2, "rc": 7,
       "vd": "2026-09-02", "cp": 19, "f": None, "x": [], "ts": 1757030400}

STATUS_CODES = {0: "settled", 1: "failed", 2: "held"}
REASON_CODES = {5: "INSUFFICIENT_FUNDS", 6: "INVALID_IBAN", 7: "LIMIT_BREACH", 8: "SANCTIONS_REVIEW"}
LIMIT_USD = 500000.0

def observe(raw: dict) -> str:
    """Render one raw ledger record as an observation a model can act on.

    Must state: the reference, the decoded status, the decoded reason, and -- where the
    reason is a limit breach -- the comparison that explains it.
    """
    status = STATUS_CODES.get(raw["s"], "unknown")
    reason = REASON_CODES.get(raw["rc"], "unknown")
    lines = [f"PMT-{raw['i']} is {status.upper()}.", f"Reason: {reason}"]
    if reason == "LIMIT_BREACH":
        lines.append(f"{raw['c']} {raw['a']:,.2f} exceeds the {LIMIT_USD:,.2f} limit.")
    return " ".join(lines)
'''),
    code('''
# --- Self-check: Section 1
check("the reference is stated in the form the tools use",
      lambda: "PMT-1003" in observe(RAW))
check("the status code is decoded, not passed through",
      lambda: "HELD" in observe(RAW) and '"s": 2' not in observe(RAW))
check("the reason code is decoded", lambda: "LIMIT_BREACH" in observe(RAW))
check("the limit comparison is spelled out",
      lambda: "990,000.00" in observe(RAW) and "500,000.00" in observe(RAW),
      "the model should not have to know the limit to understand the reason")
check("a settled record needs no comparison",
      lambda: "exceeds" not in observe({**RAW, "s": 0, "rc": 0}))
check("the observation is far shorter than the raw record",
      lambda: len(observe(RAW)) < len(json.dumps(RAW)) * 1.5)
'''),

    md("""
## Section 2 &mdash; The four kinds of nothing

An empty result is the most dangerous thing a tool returns, because the reassuring reading and the
alarming one look identical.
"""),
    code('''
def describe_empty(result: dict) -> str:
    """Say which kind of nothing this is.

    result carries: ran (bool), error (str|None), matches (list), total (int|None)
    Returns one of: "checked_clear", "not_run", "partial", "unknown_total"
    """
    if result["error"] or not result["ran"]:
        return BLANK                 # TODO: the check did not actually happen
    if result["total"] is None:
        return "unknown_total"
    if len(result["matches"]) < result["total"]:
        return "partial"
    return "checked_clear"

def render_screening(result: dict) -> str:
    """The observation a screening tool should return -- never a bare empty list."""
    kind = describe_empty(result)
    return {
        "checked_clear":  f"Sanctions screening ran and found 0 of {result['total']} records matching. Clear.",
        "not_run":        f"Sanctions screening DID NOT RUN ({result['error'] or 'no reason given'}). Result unknown -- do not treat as clear.",
        "partial":        f"Sanctions screening returned {len(result['matches'])} of {result['total']} records. Incomplete.",
        "unknown_total":  "Sanctions screening returned results but the total is unknown. Completeness cannot be established.",
    }[kind]
''', '''
def describe_empty(result: dict) -> str:
    """Say which kind of nothing this is.

    result carries: ran (bool), error (str|None), matches (list), total (int|None)
    Returns one of: "checked_clear", "not_run", "partial", "unknown_total"
    """
    if result["error"] or not result["ran"]:
        return "not_run"
    if result["total"] is None:
        return "unknown_total"
    if len(result["matches"]) < result["total"]:
        return "partial"
    return "checked_clear"

def render_screening(result: dict) -> str:
    """The observation a screening tool should return -- never a bare empty list."""
    kind = describe_empty(result)
    return {
        "checked_clear":  f"Sanctions screening ran and found 0 of {result['total']} records matching. Clear.",
        "not_run":        f"Sanctions screening DID NOT RUN ({result['error'] or 'no reason given'}). Result unknown -- do not treat as clear.",
        "partial":        f"Sanctions screening returned {len(result['matches'])} of {result['total']} records. Incomplete.",
        "unknown_total":  "Sanctions screening returned results but the total is unknown. Completeness cannot be established.",
    }[kind]
'''),
    code('''
# --- Self-check: Section 2
CLEAR   = {"ran": True,  "error": None,        "matches": [], "total": 0}
TIMEOUT = {"ran": False, "error": "timeout",   "matches": [], "total": None}
PARTIAL = {"ran": True,  "error": None,        "matches": [], "total": 47}
NOTOTAL = {"ran": True,  "error": None,        "matches": [], "total": None}

check("a genuine all-clear is reported as clear",
      lambda: describe_empty(CLEAR) == "checked_clear")
check("a check that did not run is NOT reported as clear",
      lambda: describe_empty(TIMEOUT) == "not_run",
      "this is the distinction the whole section exists for")
check("an incomplete result is flagged", lambda: describe_empty(PARTIAL) == "partial")
check("an unknown total is flagged", lambda: describe_empty(NOTOTAL) == "unknown_total")
check("the not-run observation warns against the reassuring reading",
      lambda: "do not treat as clear" in render_screening(TIMEOUT).lower())
check("all four render to different text",
      lambda: len({render_screening(r) for r in (CLEAR, TIMEOUT, PARTIAL, NOTOTAL)}) == 4)

for name, r in (("clear", CLEAR), ("timeout", TIMEOUT), ("partial", PARTIAL), ("no total", NOTOTAL)):
    try:
        print(f"  {name:9} -> {render_screening(r)}")
    except NameError:
        print("(fill in describe_empty above)"); break
'''),

    md("""
## Section 3 &mdash; Stamp what the agent cannot see

Freshness, completeness and authority are absent from most tool results, and the agent assumes the
convenient value for each.
"""),
    code('''
NOW = 1757030400 + 7200              # pretend "now" is two hours after the record's timestamp

def stamp(observation: str, *, as_of: int, now: int = NOW,
          shown: int = 1, total: int = 1, binding: bool = True) -> str:
    """Append the three things a raw result never carries."""
    age_min = (now - as_of) // 60
    freshness = "live" if age_min < 5 else f"as of {age_min} minutes ago"
    parts = [observation, f"[{freshness}"]
    if shown < total:
        parts.append(f"; showing {shown} of {total}")
    if BLANK:                        # TODO: when should the observation warn it is not binding?
        parts.append("; DRAFT policy, not binding")
    return "".join(parts) + "]"
''', '''
NOW = 1757030400 + 7200              # pretend "now" is two hours after the record's timestamp

def stamp(observation: str, *, as_of: int, now: int = NOW,
          shown: int = 1, total: int = 1, binding: bool = True) -> str:
    """Append the three things a raw result never carries."""
    age_min = (now - as_of) // 60
    freshness = "live" if age_min < 5 else f"as of {age_min} minutes ago"
    parts = [observation, f"[{freshness}"]
    if shown < total:
        parts.append(f"; showing {shown} of {total}")
    if not binding:
        parts.append("; DRAFT policy, not binding")
    return "".join(parts) + "]"
'''),
    code('''
# --- Self-check: Section 3
check("a stale record says how stale",
      lambda: "120 minutes ago" in stamp("x", as_of=1757030400))
check("a fresh record says live",
      lambda: "live" in stamp("x", as_of=NOW))
check("a truncated result says so",
      lambda: "showing 3 of 47" in stamp("x", as_of=NOW, shown=3, total=47))
check("a complete result does not add noise",
      lambda: "showing" not in stamp("x", as_of=NOW, shown=1, total=1))
check("a draft policy is marked as not binding",
      lambda: "not binding" in stamp("x", as_of=NOW, binding=False),
      "a draft and a ratified policy are both just text otherwise")
check("a binding policy carries no draft warning",
      lambda: "not binding" not in stamp("x", as_of=NOW, binding=True))

try:
    print("  " + stamp(observe(RAW), as_of=1757030400, shown=3, total=47, binding=False))
except NameError:
    print("(fill in the blanks above)")
'''),

    md("""
## Run it for real

The same question, twice: once with the raw record, once with your observation. Same model, same
prompt &mdash; only what the agent can see has changed.
"""),
    code('''
QUESTION = ("Given the payment data below, say in one line who must action this and whether it may "
            "be released. Answer only from the data given.")

if llm_ready():
    try:
        print("--- with the raw record ---")
        print("  " + ask(f"{QUESTION}\\n\\nDATA: {json.dumps(RAW)}").strip()[:300])
        print("\\n--- with your observation ---")
        obs = stamp(observe(RAW), as_of=1757030400, shown=1, total=1, binding=True)
        print("  " + ask(f"{QUESTION}\\n\\nDATA: {obs}").strip()[:300])
        print(f"\\nraw record: {len(json.dumps(RAW))} chars   observation: {len(obs)} chars")
    except NameError:
        print("(fill in the blanks above, then re-run this cell)")
'''),
    md("""
### Read it

Two things to check. First, whether the raw arm invented a meaning for `s: 2` or `rc: 7` &mdash; it has
no way to know them, so anything it says about status is a guess dressed as an answer.

Second, the character counts. The observation is usually **shorter** than the raw record as well as
clearer, because you dropped the fields nobody needed. Perception is one of the few places where
the cheap option and the accurate option are the same option.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `observe()` hardcodes `LIMIT_USD`. What happens when the limit is per-currency or per-client?
   Where should that number live so the observation stays truthful?
2. Add a fifth kind of nothing: the check ran, matched, but the results were **suppressed** by
   entitlements. How should that observation read so the agent neither ignores it nor over-reacts?
"""),
]


# =========================================================================== #
# Lab 3.3 -- build the StateGraph substrate by hand
# =========================================================================== #
LAB3 = [
    header(3, "Build a StateGraph From Scratch", "Advanced", 35,
           ["Implement nodes, edges, conditional edges and cycles in ~40 lines",
            "Make state an explicit object every node reads and writes",
            "Add a step budget so a cycle terminates",
            "Run the same graph on real LangGraph and compare"],
           "> **The substrate, built before it is used.** You will understand LangGraph better for\n"
           "> having written the 40 lines it is hiding, and every later module sits on this."),
    setup(3),
    code(DOMAIN),
    code(CARRIED_TOOLS),

    md("""
## Concept

A StateGraph is four things and no more:

| Piece | What it is |
|---|---|
| **State** | a dict with a declared shape, threaded through everything |
| **Node** | a function: state in, *partial* state out |
| **Edge** | what runs next, always |
| **Conditional edge** | what runs next, given the state |

A **cycle** is just an edge that points backwards. Build all of it here, in plain Python.
"""),

    md("""
## Section 1 &mdash; Nodes return partial state

The single most important convention: a node returns **only the keys it changed**, and the engine
merges. That is what makes nodes composable and independently testable.
"""),
    code('''
def read_ledger(state: dict) -> dict:
    """Node: look up the payment. Returns ONLY the keys it changed."""
    obs = lookup_payment(state["ref"])
    return {"findings": [f"ledger: {obs}"], "steps": state["steps"] + 1}

def read_policy(state: dict) -> dict:
    """Node: look up the policy for whatever reason code the ledger gave."""
    rec = LEDGER.get(state["ref"], {})
    obs = policy_for(rec.get("reason_code")) if rec.get("reason_code") else "no reason code"
    needs_human = rec.get("reason_code") in NEEDS_HUMAN
    return BLANK                     # TODO: the three keys this node changes --
                                     # findings (a one-item list), needs_human, and steps
''', '''
def read_ledger(state: dict) -> dict:
    """Node: look up the payment. Returns ONLY the keys it changed."""
    obs = lookup_payment(state["ref"])
    return {"findings": [f"ledger: {obs}"], "steps": state["steps"] + 1}

def read_policy(state: dict) -> dict:
    """Node: look up the policy for whatever reason code the ledger gave."""
    rec = LEDGER.get(state["ref"], {})
    obs = policy_for(rec.get("reason_code")) if rec.get("reason_code") else "no reason code"
    needs_human = rec.get("reason_code") in NEEDS_HUMAN
    return {"findings": [f"policy: {obs}"],
            "needs_human": needs_human,
            "steps": state["steps"] + 1}
'''),
    code('''
# --- Self-check: Section 1
_s0 = {"ref": "PMT-1005", "findings": [], "needs_human": False, "steps": 0, "answer": None}

check("a node returns only what it changed",
      lambda: set(read_ledger(_s0)) == {"findings", "steps"},
      "returning the whole state makes nodes impossible to compose")
check("read_ledger records its observation",
      lambda: "SANCTIONS_REVIEW" in read_ledger(_s0)["findings"][0])
check("read_policy returns its three keys",
      lambda: set(read_policy(_s0)) == {"findings", "needs_human", "steps"})
check("a sanctions hold is flagged for a human",
      lambda: read_policy(_s0)["needs_human"] is True)
check("an insufficient-funds case is not",
      lambda: read_policy({**_s0, "ref": "PMT-1002"})["needs_human"] is False)
check("nodes do not mutate the state they were given",
      lambda: (read_ledger(_s0), _s0["steps"] == 0)[1],
      "return a new dict; never edit state in place")
'''),

    md("""
## Section 2 &mdash; Reducers: how partial updates merge

`steps` should be replaced by the new value. `findings` should **accumulate**. That difference is
the reducer, declared once per key rather than remembered in every node.
"""),
    code('''
def replace(old, new):
    return new

def append(old, new):
    return list(old) + list(new)

REDUCERS = {
    "findings": append,              # every node's findings are kept, in order
    "steps": replace,
    "needs_human": replace,
    "answer": replace,
    "ref": replace,
}

def merge(state: dict, update: dict) -> dict:
    """Apply a node's partial update using the declared reducer for each key."""
    out = dict(state)
    for key, value in update.items():
        reducer = REDUCERS.get(key, replace)
        out[key] = BLANK             # TODO: combine the old value with the new one
    return out
''', '''
def replace(old, new):
    return new

def append(old, new):
    return list(old) + list(new)

REDUCERS = {
    "findings": append,              # every node's findings are kept, in order
    "steps": replace,
    "needs_human": replace,
    "answer": replace,
    "ref": replace,
}

def merge(state: dict, update: dict) -> dict:
    """Apply a node's partial update using the declared reducer for each key."""
    out = dict(state)
    for key, value in update.items():
        reducer = REDUCERS.get(key, replace)
        out[key] = reducer(state.get(key), value)
    return out
'''),
    code('''
# --- Self-check: Section 2   (fixtures built lazily -- a module-level call into an
#                              unfilled function would crash the cell instead of printing [TODO])
def _a():
    return merge(_s0, {"findings": ["one"], "steps": 1})

def _b():
    return merge(_a(), {"findings": ["two"], "steps": 2})

check("findings accumulate across nodes", lambda: _b()["findings"] == ["one", "two"],
      "this is the append reducer doing its job")
check("steps is replaced, not appended", lambda: _b()["steps"] == 2)
check("untouched keys survive the merge", lambda: _b()["ref"] == "PMT-1005")
check("the original state is not mutated", lambda: _s0["findings"] == [])
check("an undeclared key defaults to replace",
      lambda: merge(_s0, {"novel": 7})["novel"] == 7)
check("two parallel writes to findings both survive",
      lambda: merge(merge(_s0, {"findings": ["x"]}), {"findings": ["y"]})["findings"] == ["x", "y"],
      "with the default replace reducer, one of these would silently vanish")
'''),

    md("""
## Section 3 &mdash; The engine: edges, conditions and a cycle

Forty lines. Nodes run, updates merge, and a conditional edge decides what comes next &mdash; including
going backwards.
"""),
    code('''
END = "__end__"

class Graph:
    def __init__(self, reducers):
        self.nodes, self.edges, self.conditions = {}, {}, {}
        self.entry = None
        self.reducers = reducers

    def add_node(self, name, fn):
        self.nodes[name] = fn
        return self

    def add_edge(self, src, dst):
        self.edges[src] = dst
        return self

    def add_conditional_edge(self, src, fn):
        """fn(state) -> the name of the next node, or END."""
        self.conditions[src] = fn
        return self

    def set_entry(self, name):
        self.entry = name
        return self

    def run(self, state, max_steps=8):
        """Execute until END or the budget is spent. Returns (final_state, path)."""
        current, path = self.entry, []
        while current != END:
            if state["steps"] >= max_steps:
                return merge(state, {"answer": "stopped: step budget"}), path
            path.append(current)
            update = self.nodes[current](state)
            state = merge(state, update)
            if current in self.conditions:
                current = self.conditions[current](state)
            else:
                current = BLANK      # TODO: the plain edge out of this node, or END if there is none
        return state, path
''', '''
END = "__end__"

class Graph:
    def __init__(self, reducers):
        self.nodes, self.edges, self.conditions = {}, {}, {}
        self.entry = None
        self.reducers = reducers

    def add_node(self, name, fn):
        self.nodes[name] = fn
        return self

    def add_edge(self, src, dst):
        self.edges[src] = dst
        return self

    def add_conditional_edge(self, src, fn):
        """fn(state) -> the name of the next node, or END."""
        self.conditions[src] = fn
        return self

    def set_entry(self, name):
        self.entry = name
        return self

    def run(self, state, max_steps=8):
        """Execute until END or the budget is spent. Returns (final_state, path)."""
        current, path = self.entry, []
        while current != END:
            if state["steps"] >= max_steps:
                return merge(state, {"answer": "stopped: step budget"}), path
            path.append(current)
            update = self.nodes[current](state)
            state = merge(state, update)
            if current in self.conditions:
                current = self.conditions[current](state)
            else:
                current = self.edges.get(current, END)
        return state, path
'''),
    code('''
# --- Self-check: Section 3
def write_note(state):
    verdict = "human decision required" if state["needs_human"] else "operations may proceed"
    return {"answer": f"{state['ref']}: {verdict}", "steps": state["steps"] + 1}

def enough(state):
    """Conditional edge: two findings is enough to conclude; otherwise go round again."""
    return "write_note" if len(state["findings"]) >= 2 else "read_ledger"

def build():
    return (Graph(REDUCERS)
            .add_node("read_ledger", read_ledger)
            .add_node("read_policy", read_policy)
            .add_node("write_note", write_note)
            .add_edge("read_ledger", "read_policy")
            .add_conditional_edge("read_policy", enough)
            .add_edge("write_note", END)
            .set_entry("read_ledger"))

def _run(ref="PMT-1005"):
    return build().run({"ref": ref, "findings": [], "needs_human": False,
                        "steps": 0, "answer": None})

check("the graph reaches an answer", lambda: _run()[0]["answer"] is not None)
check("it visits the nodes in order",
      lambda: _run()[1][:3] == ["read_ledger", "read_policy", "write_note"])
check("findings accumulated from both reader nodes",
      lambda: len(_run()[0]["findings"]) == 2)
check("a sanctions hold ends with a human decision",
      lambda: "human decision required" in _run("PMT-1005")[0]["answer"])
check("an insufficient-funds case does not",
      lambda: "operations may proceed" in _run("PMT-1002")[0]["answer"])
check("a graph that never satisfies its condition stops on the budget",
      lambda: "step budget" in (Graph(REDUCERS)
              .add_node("read_ledger", read_ledger)
              .add_conditional_edge("read_ledger", lambda s: "read_ledger")
              .set_entry("read_ledger")
              .run({"ref": "PMT-1002", "findings": [], "needs_human": False,
                    "steps": 0, "answer": None})[0]["answer"]),
      "a cycle without a budget is an infinite loop")

try:
    final, path = _run()
    print("path:", " -> ".join(path))
    print("state:", json.dumps({k: v for k, v in final.items() if k != "findings"}, indent=1))
    for f in final["findings"]:
        print("  -", f[:90])
except NameError:
    print("(fill in the blanks above, then re-run)")
'''),

    md("""
## Run it for real

The same graph on actual LangGraph. Note how little changes: `TypedDict` instead of a plain dict,
`Annotated[list, add]` instead of your `REDUCERS` table, and `add_conditional_edges` instead of
your dictionary of functions.
"""),
    code('''
try:
    from typing import Annotated
    from typing_extensions import TypedDict
    from operator import add
    from langgraph.graph import StateGraph, START, END as LG_END

    class CaseState(TypedDict):
        ref: str
        findings: Annotated[list, add]        # <- your `append` reducer, declared
        needs_human: bool
        steps: int
        answer: str | None

    g = StateGraph(CaseState)
    g.add_node("read_ledger", read_ledger)
    g.add_node("read_policy", read_policy)
    g.add_node("write_note", write_note)
    g.add_edge(START, "read_ledger")
    g.add_edge("read_ledger", "read_policy")
    g.add_conditional_edges("read_policy", enough,
                            {"write_note": "write_note", "read_ledger": "read_ledger"})
    g.add_edge("write_note", LG_END)
    app = g.compile()

    out = app.invoke({"ref": "PMT-1005", "findings": [], "needs_human": False,
                      "steps": 0, "answer": None})
    print("answer  :", out["answer"])
    print("findings:", len(out["findings"]))
    print("\\nSame nodes, same conditional edge, same reducer. The engine is the part you wrote.")
except ImportError as exc:
    print(f"LangGraph not importable here ({exc}). The graded cells above do not need it.")
except NameError:
    print("(fill in the blanks above, then re-run this cell)")
except Exception as exc:
    print(f"<graph run failed: {type(exc).__name__}: {exc}>")
'''),
    md("""
### Read it

Your `merge` is LangGraph's reducer machinery. Your `conditions` dict is `add_conditional_edges`.
Your `max_steps` guard is its recursion limit. What LangGraph adds on top is the part that is
genuinely hard: **checkpointing** &mdash; and that is the next lab.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add a `read_counterparty` node and run it **in parallel** with `read_policy`. What must be true
   of the `findings` reducer for both results to survive? You already know &mdash; now prove it.
2. `run()` counts steps from the state, so a node that forgets to increment `steps` makes the
   budget unenforceable. Move the counting into the engine. What does that cost you in node
   independence?
"""),
]


# =========================================================================== #
# Lab 3.4 -- checkpointing: resume, approve, rewind, audit
# =========================================================================== #
LAB4 = [
    header(4, "Checkpointing &mdash; Resume, Approve, Rewind, Audit", "Advanced", 35,
           ["Write a checkpointer that persists state after every node",
            "Kill a run mid-flight and resume it without repeating work",
            "Pause before an irreversible step and wait for approval",
            "Rewind to an earlier checkpoint, change one field, and re-run",
            "Read the checkpoint history as an audit trail"],
           "> **Builds directly on Lab 3.3's graph.** One mechanism -- state written after every\n"
           "> node -- gives you all four capabilities."),
    setup(4),
    code(DOMAIN),
    code(CARRIED_TOOLS),

    md("""
## Concept

After every node, write the whole state under a **thread id**. That single mechanism gives you
resume after a crash, human-in-the-loop approval, time travel, and an audit trail &mdash; the last one
as a by-product rather than a feature anyone built.

It matters that the trail is **recorded**, not generated: it is what the system actually held, not
the model's account of what it did.
"""),

    md("""
## Section 1 &mdash; A checkpointer

Append-only, keyed by thread. Simple enough to fit on a slide; the real one differs mainly in where
it writes.
"""),
    code('''
class Checkpointer:
    """Append-only state history per thread. A real one writes to Postgres or Redis."""

    def __init__(self):
        self.threads: dict[str, list[dict]] = {}

    def put(self, thread: str, node: str, state: dict) -> None:
        """Record the state as it stands AFTER `node` ran."""
        self.threads.setdefault(thread, []).append(
            {"seq": len(self.threads.get(thread, [])), "after": node,
             "state": json.loads(json.dumps(state))})     # a snapshot, not a reference

    def latest(self, thread: str) -> dict | None:
        """The most recent checkpoint, or None if the thread is new."""
        history = self.threads.get(thread, [])
        return BLANK                 # TODO: the last checkpoint, or None when there is none

    def at(self, thread: str, seq: int) -> dict | None:
        for cp in self.threads.get(thread, []):
            if cp["seq"] == seq:
                return cp
        return None

    def history(self, thread: str) -> list[dict]:
        return list(self.threads.get(thread, []))
''', '''
class Checkpointer:
    """Append-only state history per thread. A real one writes to Postgres or Redis."""

    def __init__(self):
        self.threads: dict[str, list[dict]] = {}

    def put(self, thread: str, node: str, state: dict) -> None:
        """Record the state as it stands AFTER `node` ran."""
        self.threads.setdefault(thread, []).append(
            {"seq": len(self.threads.get(thread, [])), "after": node,
             "state": json.loads(json.dumps(state))})     # a snapshot, not a reference

    def latest(self, thread: str) -> dict | None:
        """The most recent checkpoint, or None if the thread is new."""
        history = self.threads.get(thread, [])
        return history[-1] if history else None

    def at(self, thread: str, seq: int) -> dict | None:
        for cp in self.threads.get(thread, []):
            if cp["seq"] == seq:
                return cp
        return None

    def history(self, thread: str) -> list[dict]:
        return list(self.threads.get(thread, []))
'''),
    code('''
# --- Self-check: Section 1
def _cp():
    c = Checkpointer()
    c.put("t1", "read_ledger", {"steps": 1, "findings": ["a"]})
    c.put("t1", "read_policy", {"steps": 2, "findings": ["a", "b"]})
    return c

check("a new thread has no checkpoint", lambda: Checkpointer().latest("nope") is None)
check("latest returns the most recent", lambda: _cp().latest("t1")["after"] == "read_policy")
check("history is ordered and complete", lambda: [c["seq"] for c in _cp().history("t1")] == [0, 1])
check("an earlier checkpoint is still reachable",
      lambda: _cp().at("t1", 0)["state"]["steps"] == 1)
check("checkpoints are snapshots, not references",
      lambda: (lambda c, s: (s["findings"].append("mutated"),
                             c.latest("t1")["state"]["findings"] == ["a", "b"])[1])(
              *(lambda: (lambda c: (c, c.threads["t1"][-1]["state"]))(_cp()))()) is not None)
'''),

    md("""
## Section 2 &mdash; Resume after a crash

A graph that checkpoints can be killed and restarted. The test is that it continues rather than
repeating work already paid for.
"""),
    code('''
END = "__end__"

def run_graph(nodes, edges, conditions, entry, state, thread, cp,
              max_steps=8, stop_before=None, crash_after=None):
    """Run a graph, checkpointing after each node.

    stop_before  -- pause before this node and return, awaiting approval
    crash_after  -- simulate a pod death immediately after this node
    """
    resumed = cp.latest(thread)
    current = entry
    if resumed:
        state = resumed["state"]
        current = resumed["state"].get("__next__", entry)

    path = []
    while current != END:
        if state["steps"] >= max_steps:
            return merge(state, {"answer": "stopped: step budget"}), path, "budget"
        if stop_before and current == stop_before:
            cp.put(thread, "paused", {**state, "__next__": current})
            return state, path, "awaiting_approval"

        path.append(current)
        state = merge(state, nodes[current](state))
        nxt = conditions[current](state) if current in conditions else edges.get(current, END)
        cp.put(thread, current, {**state, "__next__": nxt})

        if crash_after and current == crash_after:
            return state, path, "crashed"
        current = nxt
    return state, path, "done"

def resume(nodes, edges, conditions, entry, thread, cp, **kw):
    """Continue a thread from its last checkpoint. Returns the same triple as run_graph."""
    last = cp.latest(thread)
    if last is None:
        raise ValueError("nothing to resume")
    return run_graph(nodes, edges, conditions, entry, BLANK, thread, cp, **kw)
                                     # TODO: which state should a resumed run start from?
''', '''
END = "__end__"

def run_graph(nodes, edges, conditions, entry, state, thread, cp,
              max_steps=8, stop_before=None, crash_after=None):
    """Run a graph, checkpointing after each node.

    stop_before  -- pause before this node and return, awaiting approval
    crash_after  -- simulate a pod death immediately after this node
    """
    resumed = cp.latest(thread)
    current = entry
    if resumed:
        state = resumed["state"]
        current = resumed["state"].get("__next__", entry)

    path = []
    while current != END:
        if state["steps"] >= max_steps:
            return merge(state, {"answer": "stopped: step budget"}), path, "budget"
        if stop_before and current == stop_before:
            cp.put(thread, "paused", {**state, "__next__": current})
            return state, path, "awaiting_approval"

        path.append(current)
        state = merge(state, nodes[current](state))
        nxt = conditions[current](state) if current in conditions else edges.get(current, END)
        cp.put(thread, current, {**state, "__next__": nxt})

        if crash_after and current == crash_after:
            return state, path, "crashed"
        current = nxt
    return state, path, "done"

def resume(nodes, edges, conditions, entry, thread, cp, **kw):
    """Continue a thread from its last checkpoint. Returns the same triple as run_graph."""
    last = cp.latest(thread)
    if last is None:
        raise ValueError("nothing to resume")
    return run_graph(nodes, edges, conditions, entry, last["state"], thread, cp, **kw)
'''),
    code('''
# --- Self-check: Section 2
def read_ledger(s):  return {"findings": [f"ledger: {lookup_payment(s['ref'])}"], "steps": s["steps"] + 1}
def read_policy(s):
    rec = LEDGER.get(s["ref"], {})
    return {"findings": [f"policy: {policy_for(rec.get('reason_code'))}"],
            "needs_human": rec.get("reason_code") in NEEDS_HUMAN, "steps": s["steps"] + 1}
def write_note(s):
    return {"answer": f"{s['ref']}: {'human decision required' if s['needs_human'] else 'operations may proceed'}",
            "steps": s["steps"] + 1}

REDUCERS = {"findings": lambda o, n: list(o or []) + list(n)}
def merge(state, update):
    out = dict(state)
    for k, v in update.items():
        out[k] = REDUCERS.get(k, lambda o, n: n)(state.get(k), v)
    return out

NODES = {"read_ledger": read_ledger, "read_policy": read_policy, "write_note": write_note}
EDGES = {"read_ledger": "read_policy", "read_policy": "write_note", "write_note": END}
FRESH = lambda ref="PMT-1005": {"ref": ref, "findings": [], "needs_human": False,
                                "steps": 0, "answer": None}

def _crash_then_resume():
    cp = Checkpointer()
    s1, p1, why1 = run_graph(NODES, EDGES, {}, "read_ledger", FRESH(), "t", cp,
                             crash_after="read_policy")
    s2, p2, why2 = resume(NODES, EDGES, {}, "read_ledger", "t", cp)
    return p1, why1, p2, why2, s2

check("the run crashes where we told it to",
      lambda: _crash_then_resume()[1] == "crashed")
check("it had completed two nodes before dying",
      lambda: _crash_then_resume()[0] == ["read_ledger", "read_policy"])
check("the resumed run finishes", lambda: _crash_then_resume()[3] == "done")
check("the resumed run does NOT repeat completed work",
      lambda: _crash_then_resume()[2] == ["write_note"],
      "start from the checkpointed state, not a fresh one")
check("findings from before the crash survived",
      lambda: len(_crash_then_resume()[4]["findings"]) == 2)
check("the final answer is correct after resuming",
      lambda: "human decision required" in _crash_then_resume()[4]["answer"])
'''),

    md("""
## Section 3 &mdash; Pause for approval, and rewind

The same checkpoint mechanism, used two more ways. `write_note` is the irreversible step here, so
it is the one that waits for a human.
"""),
    code('''
def approve_and_continue(thread, cp):
    """A human approved the paused step. Continue from where it stopped."""
    return resume(NODES, EDGES, {}, "read_ledger", thread, cp)

def rewind(thread, cp, seq, changes: dict):
    """Rewind to checkpoint `seq`, apply `changes`, and re-run from there.

    Returns the new final state. The original history is left intact -- rewinding
    forks, it does not erase.
    """
    cp_at = cp.at(thread, seq)
    if cp_at is None:
        raise ValueError(f"no checkpoint {seq}")
    forked = Checkpointer()
    forked.threads[thread] = [c for c in cp.history(thread) if c["seq"] <= seq]
    state = {**cp_at["state"], **changes}
    forked.threads[thread][-1] = {**forked.threads[thread][-1], "state": state}
    return BLANK                     # TODO: re-run from the forked checkpointer and return
                                     # the final state only
''', '''
def approve_and_continue(thread, cp):
    """A human approved the paused step. Continue from where it stopped."""
    return resume(NODES, EDGES, {}, "read_ledger", thread, cp)

def rewind(thread, cp, seq, changes: dict):
    """Rewind to checkpoint `seq`, apply `changes`, and re-run from there.

    Returns the new final state. The original history is left intact -- rewinding
    forks, it does not erase.
    """
    cp_at = cp.at(thread, seq)
    if cp_at is None:
        raise ValueError(f"no checkpoint {seq}")
    forked = Checkpointer()
    forked.threads[thread] = [c for c in cp.history(thread) if c["seq"] <= seq]
    state = {**cp_at["state"], **changes}
    forked.threads[thread][-1] = {**forked.threads[thread][-1], "state": state}
    return resume(NODES, EDGES, {}, "read_ledger", thread, forked)[0]
'''),
    code('''
# --- Self-check: Section 3
def _paused():
    cp = Checkpointer()
    s, p, why = run_graph(NODES, EDGES, {}, "read_ledger", FRESH(), "t2", cp,
                          stop_before="write_note")
    return cp, s, p, why

check("the run pauses before the irreversible step",
      lambda: _paused()[3] == "awaiting_approval")
check("it paused with the reads already done", lambda: _paused()[2] == ["read_ledger", "read_policy"])
check("no answer was written while awaiting approval",
      lambda: _paused()[1]["answer"] is None,
      "the whole point of the gate is that the write has not happened yet")
check("approving continues to completion",
      lambda: approve_and_continue("t2", _paused()[0])[2] == "done")

def _rewound():
    cp = Checkpointer()
    run_graph(NODES, EDGES, {}, "read_ledger", FRESH(), "t3", cp)
    # an analyst disputes the human-decision flag and re-runs from after the policy read
    return cp, rewind("t3", cp, seq=1, changes={"needs_human": False})

check("rewinding with a changed field changes the outcome",
      lambda: "operations may proceed" in _rewound()[1]["answer"],
      "the original run said human decision required")
check("the original history is left intact",
      lambda: len(_rewound()[0].history("t3")) == 3,
      "rewinding forks; it must not erase what actually happened")
'''),

    md("""
## Section 4 &mdash; The audit trail

The history is already the answer to *what did it know, and when?* Render it.
"""),
    code('''
def audit(thread, cp) -> str:
    """A human-readable trail: after each node, what was known and what came next."""
    rows = [f"{'seq':>4}  {'after':<14}{'steps':>6}{'findings':>10}  {'needs_human':<12}next"]
    rows.append("-" * 74)
    for c in cp.history(thread):
        s = c["state"]
        rows.append(f"{c['seq']:>4}  {c['after']:<14}{s.get('steps', 0):>6}"
                    f"{len(s.get('findings', [])):>10}  {str(s.get('needs_human')):<12}"
                    f"{s.get('__next__', '-')}")
    return "\\n".join(rows)

try:
    _cpx = Checkpointer()
    run_graph(NODES, EDGES, {}, "read_ledger", FRESH(), "audit-demo", _cpx)
    print(audit("audit-demo", _cpx))
except NameError:
    print("(finish the sections above, then re-run this cell)")
'''),
    code('''
# --- Self-check: Section 4
def _audited():
    c = Checkpointer()
    run_graph(NODES, EDGES, {}, "read_ledger", FRESH(), "a1", c)
    return c

check("one checkpoint per node, plus a header and rule",
      lambda: len(audit("a1", _audited()).splitlines()) == 3 + 2)
check("the trail shows findings accumulating",
      lambda: "         1" in audit("a1", _audited()) and "         2" in audit("a1", _audited()))
check("the trail records when needs_human became true",
      lambda: audit("a1", _audited()).count("True") >= 2)
check("the trail is derived from recorded state, not regenerated",
      lambda: all("state" in c for c in _audited().history("a1")),
      "this is why it is stronger evidence than asking the model what it did")
'''),

    md("""
## Run it for real

Have the model read your audit trail and answer the auditor's question. Note what it is doing:
reading a record, not recalling a run.
"""),
    code('''
if llm_ready():
    try:
        cp = Checkpointer()
        run_graph(NODES, EDGES, {}, "read_ledger", FRESH("PMT-1005"), "real", cp)
        trail = audit("real", cp)
        answer = ask(
            "You are answering an auditor. Using ONLY this execution trail, state what the system "
            "knew at the point it decided, and whether a human decision was required. If the trail "
            "does not support an answer, say so.\\n\\n" + trail
        )
        print(trail)
        print("\\n--- the auditor's answer ---\\n" + answer.strip()[:500])
    except NameError:
        print("(finish the sections above, then re-run this cell)")
'''),
    md("""
### Read it

The model is summarising a **recorded** artefact. Ask it the same question with no trail and it
would produce something equally fluent and unfalsifiable &mdash; which is precisely the difference
Module 2 drew between reasoning text and evidence.

This is also the honest answer to &ldquo;can we explain what the agent did?&rdquo;. Not from the model.
From the checkpoints.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. The checkpointer stores full state snapshots. On a long run that is a lot of duplication. Store
   diffs instead &mdash; then say what you lose when a diff chain has a gap.
2. State snapshots may contain the payment record. Which fields must never reach a checkpoint store
   under your own retention rules, and where would you enforce that &mdash; in the node, the reducer, or
   the checkpointer? Module 8 returns to this.
"""),
]


# =========================================================================== #
# Lab 3.5 -- challenge: shared state, and the poisoning it enables
# =========================================================================== #
LAB5 = [
    header(5, "Challenge &mdash; Shared State and Context Poisoning", "Advanced", 40,
           ["Run three agents over one shared state and watch a wrong finding spread",
            "Attach provenance so a claim can be checked instead of the consensus",
            "Build a critic that verifies against the source, not against agreement",
            "Compare shared and private state on cost, containment and auditability"],
           "> **The comprehensive lab for Module 3, and the bridge into Day 2.** Everything so far\n"
           "> has been one agent. This is what memory does when there are several."),
    setup(5),
    code(DOMAIN),
    code(CARRIED_TOOLS),

    md("""
## Concept

Shared state is cheap and consistent. It is also how one agent's error becomes every agent's
premise &mdash; and because each later agent reasons correctly *from what it was told*, the system ends
up confidently and unanimously wrong, with nothing erroring.

Three agents agreeing is not corroboration. It is an echo. The defence is **provenance**.
"""),

    md("""
## Section 1 &mdash; A finding that can be checked

A bare string cannot be verified. A finding that carries its author and its source can.
"""),
    code('''
def finding(claim: str, *, by: str, source: str, ref: str) -> dict:
    """One finding, with enough provenance that a critic can re-derive it."""
    return {"claim": claim, "by": by, "source": source, "ref": ref}

def verifiable(f: dict) -> bool:
    """True when a finding carries everything needed to check it independently."""
    required = ("claim", "by", "source", "ref")
    if not all(f.get(k) for k in required):
        return False
    return BLANK                     # TODO: the source must be one a critic can actually re-read.
                                     # Accept only "ledger" or "policy".
''', '''
def finding(claim: str, *, by: str, source: str, ref: str) -> dict:
    """One finding, with enough provenance that a critic can re-derive it."""
    return {"claim": claim, "by": by, "source": source, "ref": ref}

def verifiable(f: dict) -> bool:
    """True when a finding carries everything needed to check it independently."""
    required = ("claim", "by", "source", "ref")
    if not all(f.get(k) for k in required):
        return False
    return f["source"] in ("ledger", "policy")
'''),
    code('''
# --- Self-check: Section 1
_good = finding("PMT-1005 is held", by="ledger_agent", source="ledger", ref="PMT-1005")
_hearsay = finding("PMT-1005 is held", by="critic", source="another agent said so", ref="PMT-1005")

check("a sourced finding is verifiable", lambda: verifiable(_good) is True)
check("a finding sourced from another agent is not",
      lambda: verifiable(_hearsay) is False,
      "if the source is hearsay, checking it only re-checks the echo")
check("a finding missing its author is not verifiable",
      lambda: verifiable({**_good, "by": ""}) is False)
check("a finding missing its reference is not verifiable",
      lambda: verifiable({**_good, "ref": ""}) is False)
'''),

    md("""
## Section 2 &mdash; Watch the poison spread

The ledger agent misreads one field. Everything downstream is correct given what it was told.
"""),
    code('''
def ledger_agent(state, *, faulty=False):
    """Reads the ledger. With faulty=True it misreads a held payment as failed."""
    rec = LEDGER.get(state["ref"], {})
    status = rec.get("status", "unknown")
    if faulty and status == "held":
        status = "failed"                              # the single wrong bit
    return {"findings": [finding(f"{state['ref']} is {status}",
                                 by="ledger_agent", source="ledger", ref=state["ref"])]}

def policy_agent(state):
    """Reads policy for whatever the ledger said. Correct, given its input."""
    said = state["findings"][0]["claim"]
    code_ = "INSUFFICIENT_FUNDS" if "failed" in said else LEDGER.get(state["ref"], {}).get("reason_code")
    return {"findings": [finding(f"policy: {policy_for(code_)}",
                                 by="policy_agent", source="policy", ref=state["ref"])]}

def naive_critic(state):
    """Checks that the findings agree with each other. This is the trap."""
    claims = [f["claim"] for f in state["findings"]]
    consistent = not ("held" in " ".join(claims) and "failed" in " ".join(claims))
    return {"findings": [finding(f"consistency check: {'consistent' if consistent else 'conflict'}",
                                 by="critic", source="other agents", ref=state["ref"])]}

def sourced_critic(state):
    """Re-derives each verifiable finding from its named source. This is the fix."""
    problems = []
    for f in state["findings"]:
        if not verifiable(f):
            continue
        if f["source"] == "ledger":
            truth = LEDGER.get(f["ref"], {}).get("status", "unknown")
            if BLANK:                # TODO: does the claim disagree with the ledger?
                problems.append(f"{f['by']} claimed '{f['claim']}' but the ledger says '{truth}'")
    return {"findings": [finding(f"source check: {problems or 'all findings match their sources'}",
                                 by="sourced_critic", source="ledger", ref=state["ref"])],
            "problems": problems}
''', '''
def ledger_agent(state, *, faulty=False):
    """Reads the ledger. With faulty=True it misreads a held payment as failed."""
    rec = LEDGER.get(state["ref"], {})
    status = rec.get("status", "unknown")
    if faulty and status == "held":
        status = "failed"                              # the single wrong bit
    return {"findings": [finding(f"{state['ref']} is {status}",
                                 by="ledger_agent", source="ledger", ref=state["ref"])]}

def policy_agent(state):
    """Reads policy for whatever the ledger said. Correct, given its input."""
    said = state["findings"][0]["claim"]
    code_ = "INSUFFICIENT_FUNDS" if "failed" in said else LEDGER.get(state["ref"], {}).get("reason_code")
    return {"findings": [finding(f"policy: {policy_for(code_)}",
                                 by="policy_agent", source="policy", ref=state["ref"])]}

def naive_critic(state):
    """Checks that the findings agree with each other. This is the trap."""
    claims = [f["claim"] for f in state["findings"]]
    consistent = not ("held" in " ".join(claims) and "failed" in " ".join(claims))
    return {"findings": [finding(f"consistency check: {'consistent' if consistent else 'conflict'}",
                                 by="critic", source="other agents", ref=state["ref"])]}

def sourced_critic(state):
    """Re-derives each verifiable finding from its named source. This is the fix."""
    problems = []
    for f in state["findings"]:
        if not verifiable(f):
            continue
        if f["source"] == "ledger":
            truth = LEDGER.get(f["ref"], {}).get("status", "unknown")
            if truth not in f["claim"]:
                problems.append(f"{f['by']} claimed '{f['claim']}' but the ledger says '{truth}'")
    return {"findings": [finding(f"source check: {problems or 'all findings match their sources'}",
                                 by="sourced_critic", source="ledger", ref=state["ref"])],
            "problems": problems}
'''),
    code('''
# --- Self-check: Section 2
def run_shared(ref="PMT-1005", faulty=False, critic=naive_critic):
    state = {"ref": ref, "findings": [], "problems": []}
    for node in (lambda s: ledger_agent(s, faulty=faulty), policy_agent, critic):
        upd = node(state)
        state = {**state, **{k: v for k, v in upd.items() if k != "findings"},
                 "findings": state["findings"] + upd["findings"]}
    return state

_clean = run_shared(faulty=False)
_poisoned = run_shared(faulty=True)

check("a clean run reads the payment as held",
      lambda: "held" in _clean["findings"][0]["claim"])
check("the faulty run reads it as failed", lambda: "failed" in _poisoned["findings"][0]["claim"])
check("the policy agent proceeds correctly from the WRONG premise",
      lambda: "Retry" in _poisoned["findings"][1]["claim"],
      "it applied the retry policy -- correct, given what it was told")
check("the naive critic sees no problem at all",
      lambda: "consistent" in _poisoned["findings"][2]["claim"],
      "it checked agreement, and the agents did agree")
check("the sourced critic catches it",
      lambda: len(run_shared(faulty=True, critic=sourced_critic)["problems"]) == 1,
      "re-derive the claim from the ledger rather than comparing it with other agents")
check("the sourced critic does not cry wolf on a clean run",
      lambda: run_shared(faulty=False, critic=sourced_critic)["problems"] == [])

for name, st in (("clean", _clean), ("poisoned", _poisoned)):
    try:
        print(f"--- {name} ---")
        for f in st["findings"]:
            print(f"   [{f['by']:16} src={f['source']:14}] {f['claim'][:70]}")
    except NameError:
        print("(fill in the blanks above)"); break
'''),

    md("""
## Section 3 &mdash; Private state contains it

Give each agent its own state and an explicit handoff. The error stops travelling &mdash; and you pay
for that in re-sent context.
"""),
    code('''
def run_private(ref="PMT-1005", faulty=False):
    """Each agent gets only what the previous one explicitly handed over."""
    tokens = 0
    ledger_state = {"ref": ref, "findings": []}
    ledger_out = ledger_agent(ledger_state, faulty=faulty)
    tokens += len(json.dumps(ledger_out)) // 4

    # the handoff: the policy agent receives the CLAIM, and re-reads the ledger itself
    policy_state = {"ref": ref, "findings": ledger_out["findings"]}
    checked = sourced_critic(policy_state)
    tokens += len(json.dumps(policy_state)) // 4       # context re-sent at the boundary

    if checked["problems"]:
        return {"outcome": "handoff rejected", "problems": checked["problems"], "tokens": tokens}
    policy_out = policy_agent(policy_state)
    tokens += len(json.dumps(policy_out)) // 4
    return {"outcome": BLANK, "problems": [], "tokens": tokens}
                                     # TODO: what should a clean private run report?
''', '''
def run_private(ref="PMT-1005", faulty=False):
    """Each agent gets only what the previous one explicitly handed over."""
    tokens = 0
    ledger_state = {"ref": ref, "findings": []}
    ledger_out = ledger_agent(ledger_state, faulty=faulty)
    tokens += len(json.dumps(ledger_out)) // 4

    # the handoff: the policy agent receives the CLAIM, and re-reads the ledger itself
    policy_state = {"ref": ref, "findings": ledger_out["findings"]}
    checked = sourced_critic(policy_state)
    tokens += len(json.dumps(policy_state)) // 4       # context re-sent at the boundary

    if checked["problems"]:
        return {"outcome": "handoff rejected", "problems": checked["problems"], "tokens": tokens}
    policy_out = policy_agent(policy_state)
    tokens += len(json.dumps(policy_out)) // 4
    return {"outcome": "completed", "problems": [], "tokens": tokens}
'''),
    code('''
# --- Self-check: Section 3
check("a clean private run completes",
      lambda: run_private(faulty=False)["outcome"] == "completed")
check("a poisoned private run is rejected at the handoff",
      lambda: run_private(faulty=True)["outcome"] == "handoff rejected",
      "checking at the boundary is what containment means")
check("the rejection names the disagreement",
      lambda: "ledger says" in run_private(faulty=True)["problems"][0])
check("containment is not free",
      lambda: run_private(faulty=False)["tokens"] > 0,
      "context is re-sent at every boundary -- that is the coordination tax from Module 1")
'''),

    md("""
## Section 4 &mdash; The comparison, on four axes

Cost, containment, auditability, and whether the wrong answer reached the end.
"""),
    code('''
def comparison() -> str:
    rows = [f"{'design':22}{'poison contained':>18}{'tokens':>9}{'audit granularity':>20}",
            "-" * 72]
    shared_naive = run_shared(faulty=True, critic=naive_critic)
    shared_sourced = run_shared(faulty=True, critic=sourced_critic)
    private = run_private(faulty=True)
    rows.append(f"{'shared + naive critic':22}{'no':>18}{'low':>9}{'one trace':>20}")
    rows.append(f"{'shared + sourced critic':22}{'yes':>18}{'low':>9}{'one trace':>20}")
    rows.append(f"{'private + handoff check':22}{'yes':>18}{private['tokens']:>9}{'per agent':>20}")
    rows.append("")
    rows.append(f"naive critic found {len(shared_naive.get('problems', []))} problems; "
                f"sourced critic found {len(shared_sourced['problems'])}.")
    return "\\n".join(rows)

try:
    print(comparison())
except NameError:
    print("(finish the sections above, then re-run this cell)")
'''),
    code('''
# --- Self-check: Section 4
check("the comparison covers all three designs",
      lambda: len(comparison().splitlines()) == 7)
check("the naive critic is recorded as catching nothing",
      lambda: "found 0 problems" in comparison())
check("the sourced critic is recorded as catching it",
      lambda: "found 1" in comparison())
check("shared state with a sourced critic is enough to contain the error",
      lambda: len(run_shared(faulty=True, critic=sourced_critic)["problems"]) == 1,
      "you do not need private state -- you need a critic that checks sources")
'''),

    md("""
## Run it for real

Give the model both sets of findings and ask which it trusts. Watch whether provenance changes its
answer &mdash; and notice that you are testing your *design*, not the model.
"""),
    code('''
if llm_ready():
    try:
        poisoned = run_shared(faulty=True, critic=sourced_critic)
        rendered = "\\n".join(
            f"- [{f['by']}, source={f['source']}] {f['claim']}" for f in poisoned["findings"])
        verdict = ask(
            "These are the findings from three agents working one payment case. State whether the "
            "case is safe to action, and if any finding should be distrusted, say which and why. "
            "Judge each finding by its source, not by whether the others agree with it.\\n\\n"
            + rendered)
        print(rendered)
        print("\\n--- verdict ---\\n" + verdict.strip()[:500])
    except NameError:
        print("(finish the sections above, then re-run this cell)")
'''),
    md("""
### Read it

If the model flags the ledger finding because the source check contradicts it, provenance did the
work &mdash; not the model's judgement. Remove the `source=` labels and re-run: the same model, given the
same claims without provenance, has nothing to reason from but agreement.

**What you take from Module 3:** memory that survives a long session, observations the model cannot
misread, state you can print and store, and the checkpoint history that answers an auditor.
Day 2 puts several agents on top of this &mdash; and now you know what that does to a shared memory.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `sourced_critic` only re-checks findings sourced from the ledger. Extend it to policy findings.
   What stops a critic from simply becoming a second, equally fallible agent?
2. The poisoned run had exactly one wrong bit. Make the ledger agent wrong *intermittently* &mdash; a
   third of the time &mdash; and decide how the design should respond to a source that is usually right.
3. Combine this with Lab 3.4: at which checkpoint would an auditor first have been able to see the
   contradiction? That answer is your detection latency, and it is a number worth knowing.
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-3-01-memory-that-survives",        LAB1),
    ("lab-3-02-perception-observations",     LAB2),
    ("lab-3-03-stategraph-from-scratch",     LAB3),
    ("lab-3-04-checkpointing",               LAB4),
    ("lab-3-05-challenge-shared-state-poisoning", LAB5),
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
