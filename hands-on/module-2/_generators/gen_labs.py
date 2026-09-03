#!/usr/bin/env python3
"""
Generate Module 2 lab notebooks and their solutions from one source.

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
# Lab 2.{num} &mdash; {title}

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 1 &middot; Module 2 &mdash; Agentic Planning &amp; Reasoning**

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

WORK = os.path.join("/tmp", "awmas-lab-2-{num:02d}")
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
LLM_BASE_URL = os.environ.get("LAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
LLM_MODEL    = os.environ.get("LAB_LLM_MODEL")    or os.environ.get("OPENAI_MODEL")
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
# Lab 2.1 -- Chain-of-Thought, measured against a direct answer
# =========================================================================== #
LAB1 = [
    header(1, "Chain-of-Thought, Measured", "Intermediate", 25,
           ["Score a direct answer and a reasoned answer on the same cases",
            "Separate the answer from the reasoning that led to it",
            "Put a number on what the extra tokens bought"],
           "> **The thread continues.** Same synthetic payment-exception case file as Module 1.\n"
           "> Module 1 asked whether to build an agent; Module 2 asks how it should think."),
    setup(1),
    code(DOMAIN),

    md("""
## Concept

Chain-of-Thought is a prompt, not an architecture change: ask the model to work in steps before it
answers. It usually helps on multi-step questions, it always costs tokens, and the reasoning it
shows you is **a rationalisation, not a transcript** &mdash; useful evidence for a human reviewer,
never a control.

This lab does the only thing that settles it: run both on the same cases and compare.
"""),

    md("""
## Section 1 &mdash; Separate the answer from the reasoning

If you cannot extract the answer reliably, you cannot score it. Reasoning ends and the answer
begins at a marker you impose &mdash; here, a final line starting `ANSWER:`.
"""),
    code('''
ANSWER_MARKER = "ANSWER:"

def split_reasoning(text: str) -> tuple[str, str]:
    """Return (reasoning, answer) from a model reply.

    The answer is the text after the LAST occurrence of ANSWER_MARKER, stripped.
    If the marker never appears, the reasoning is empty and the whole reply is the answer --
    degrade, never raise.
    """
    if ANSWER_MARKER not in text:
        return BLANK             # TODO: the no-marker case, as a (reasoning, answer) tuple
    head, _, tail = text.rpartition(ANSWER_MARKER)
    return head.strip(), tail.strip()
''', '''
ANSWER_MARKER = "ANSWER:"

def split_reasoning(text: str) -> tuple[str, str]:
    """Return (reasoning, answer) from a model reply.

    The answer is the text after the LAST occurrence of ANSWER_MARKER, stripped.
    If the marker never appears, the reasoning is empty and the whole reply is the answer --
    degrade, never raise.
    """
    if ANSWER_MARKER not in text:
        return "", text.strip()
    head, _, tail = text.rpartition(ANSWER_MARKER)
    return head.strip(), tail.strip()
'''),
    code('''
# --- Self-check: Section 1
_reply = "Step 1: the code is LIMIT_BREACH.\\nStep 2: policy needs Treasury.\\nANSWER: Treasury approval"

check("the answer is taken from after the marker",
      lambda: split_reasoning(_reply)[1] == "Treasury approval")
check("the reasoning is everything before it",
      lambda: "Step 1" in split_reasoning(_reply)[0])
check("a reply with no marker still yields an answer",
      lambda: split_reasoning("Treasury approval")[1] == "Treasury approval",
      "return ('', text.strip()) rather than raising")
check("a reply with no marker has empty reasoning",
      lambda: split_reasoning("Treasury approval")[0] == "")
check("the LAST marker wins",
      lambda: split_reasoning("ANSWER: draft\\nrethinking\\nANSWER: final")[1] == "final",
      "use rpartition, not partition -- models restate the marker")
'''),

    md("""
## Section 2 &mdash; The two prompts

Identical task, identical cases. The only difference is whether the model is asked to work in
steps. Keep everything else fixed or the comparison means nothing.
"""),
    code('''
TASK = ("You are a payments operations analyst. Given a payment record and the policy catalogue, "
        "say who must action the exception: OPERATIONS, TREASURY, COMPLIANCE or ORIGINATOR.")

DIRECT_PROMPT = TASK + "\\nReply with a single line: ANSWER: <one of the four>"

def build_cot_prompt() -> str:
    """The chain-of-thought prompt: same task, same answer format, plus stepwise reasoning."""
    instruction = BLANK              # TODO: ask it to work in numbered steps FIRST, then finish
                                     # with the same 'ANSWER: <one of the four>' line. The final
                                     # line format must match DIRECT_PROMPT exactly or the
                                     # comparison is not like-for-like.
    return TASK + "\\n" + instruction
''', '''
TASK = ("You are a payments operations analyst. Given a payment record and the policy catalogue, "
        "say who must action the exception: OPERATIONS, TREASURY, COMPLIANCE or ORIGINATOR.")

DIRECT_PROMPT = TASK + "\\nReply with a single line: ANSWER: <one of the four>"

def build_cot_prompt() -> str:
    """The chain-of-thought prompt: same task, same answer format, plus stepwise reasoning."""
    instruction = (
        "Work through it in numbered steps first: state the reason code, then the policy that "
        "applies, then who that policy makes responsible. "
        "Finish with a single final line: ANSWER: <one of the four>"
    )
    return TASK + "\\n" + instruction
'''),
    code('''
# --- Self-check: Section 2
check("both prompts demand the same answer format",
      lambda: ANSWER_MARKER in DIRECT_PROMPT and ANSWER_MARKER in build_cot_prompt(),
      "if the formats differ you are measuring your parser, not the reasoning")
check("only the chain-of-thought prompt asks for steps",
      lambda: "step" in build_cot_prompt().lower() and "step" not in DIRECT_PROMPT.lower())
check("both carry the identical task definition",
      lambda: DIRECT_PROMPT.startswith(TASK) and build_cot_prompt().startswith(TASK),
      "change one variable at a time or the result is uninterpretable")
'''),

    md("""
## Section 3 &mdash; The eval set and the scorer

Six cases, each with the responsible party known in advance. Two are deliberately awkward: one has
no exception at all, and one references a payment that does not exist.
"""),
    code('''
CASES = [
    {"ref": "PMT-1002", "expect": "OPERATIONS"},   # insufficient funds -> retry, ops desk
    {"ref": "PMT-1003", "expect": "TREASURY"},     # limit breach -> treasury approval
    {"ref": "PMT-1004", "expect": "ORIGINATOR"},   # invalid IBAN -> return to originator
    {"ref": "PMT-1005", "expect": "COMPLIANCE"},   # sanctions review -> compliance decides
    {"ref": "PMT-1001", "expect": "OPERATIONS"},   # settled: no exception to action
    {"ref": "PMT-9999", "expect": "OPERATIONS"},   # unknown reference: cannot be actioned blind
]

def render_case(ref: str) -> str:
    """The case text handed to the model -- identical for both prompts."""
    rec = LEDGER.get(ref)
    if rec is None:
        return f"PAYMENT {ref}: not found in the ledger."
    policy = POLICY.get(rec["reason_code"], "no policy on file")
    return f"PAYMENT {ref}: {json.dumps(rec)}\\nPOLICY: {policy}"

def scores(answer: str, expect: str) -> bool:
    """A case passes when the expected party is named in the answer, case-insensitively."""
    return BLANK                     # TODO: the pass condition
''', '''
CASES = [
    {"ref": "PMT-1002", "expect": "OPERATIONS"},   # insufficient funds -> retry, ops desk
    {"ref": "PMT-1003", "expect": "TREASURY"},     # limit breach -> treasury approval
    {"ref": "PMT-1004", "expect": "ORIGINATOR"},   # invalid IBAN -> return to originator
    {"ref": "PMT-1005", "expect": "COMPLIANCE"},   # sanctions review -> compliance decides
    {"ref": "PMT-1001", "expect": "OPERATIONS"},   # settled: no exception to action
    {"ref": "PMT-9999", "expect": "OPERATIONS"},   # unknown reference: cannot be actioned blind
]

def render_case(ref: str) -> str:
    """The case text handed to the model -- identical for both prompts."""
    rec = LEDGER.get(ref)
    if rec is None:
        return f"PAYMENT {ref}: not found in the ledger."
    policy = POLICY.get(rec["reason_code"], "no policy on file")
    return f"PAYMENT {ref}: {json.dumps(rec)}\\nPOLICY: {policy}"

def scores(answer: str, expect: str) -> bool:
    """A case passes when the expected party is named in the answer, case-insensitively."""
    return expect.lower() in answer.lower()
'''),
    code('''
# --- Self-check: Section 3
check("an exact answer passes", lambda: scores("ANSWER: TREASURY", "TREASURY") is True)
check("case does not matter", lambda: scores("answer: treasury", "TREASURY") is True)
check("a wrong party fails", lambda: scores("COMPLIANCE", "TREASURY") is False)
check("every case renders without raising",
      lambda: all(isinstance(render_case(c["ref"]), str) for c in CASES))
check("the unknown reference renders as not found",
      lambda: "not found" in render_case("PMT-9999"))
'''),

    md("""
## Run it for real

Both prompts, all six cases, one table. This is the first real measurement of the module.
"""),
    code('''
def run_arm(prompt: str, label: str) -> dict:
    """Run every case under one prompt. Returns pass rate and rough token cost."""
    hits, chars = 0, 0
    for c in CASES:
        reply = ask(render_case(c["ref"]), system=prompt)
        _, answer = split_reasoning(reply)
        ok = scores(answer, c["expect"])
        hits += 1 if ok else 0
        chars += len(prompt) + len(render_case(c["ref"])) + len(reply)
        print(f"  {c['ref']}  expect {c['expect']:11} got {answer[:34]:34} {'PASS' if ok else 'FAIL'}")
    return {"arm": label, "pass_rate": round(hits / len(CASES), 3), "est_tokens": chars // 4}

if llm_ready():
    try:
        print("--- direct ---");            direct = run_arm(DIRECT_PROMPT, "direct")
        print("\\n--- chain-of-thought ---"); cot = run_arm(build_cot_prompt(), "cot")
        print(f"\\n{'arm':22}{'pass rate':>12}{'est tokens':>13}")
        for r in (direct, cot):
            print(f"{r['arm']:22}{r['pass_rate']:>12}{r['est_tokens']:>13}")
        gain = cot["pass_rate"] - direct["pass_rate"]
        mult = cot["est_tokens"] / max(direct["est_tokens"], 1)
        print(f"\\nchain-of-thought bought {gain:+.2f} pass rate for {mult:.1f}x the tokens.")
    except NameError:
        print("(fill in the blanks above, then re-run this cell)")
'''),
    md("""
### Read it

Three things to look at, in this order:

1. **Did the gain justify the multiple?** On a task this small it often does not &mdash; and that is a
   real result, not a failed lab.
2. **Where did the direct arm fail?** Usually the two awkward cases, which is exactly where working
   in steps helps.
3. **Read one reasoning trace against its answer.** If a case passed with reasoning that does not
   support it, you have just seen why the reasoning text is evidence and not a control.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add a case the direct arm gets right and the reasoned arm gets wrong. They exist &mdash; extra steps
   give a model more places to talk itself out of a correct first instinct.
2. `scores()` does substring matching, so an answer of "not TREASURY" would pass. Tighten it, then
   decide whether the stricter scorer changes the verdict. If a scoring change flips your
   conclusion, the conclusion was never solid.
"""),
]


# =========================================================================== #
# Lab 2.2 -- ReAct: the parser is a protocol boundary
# =========================================================================== #
LAB2 = [
    header(2, "ReAct and the Parser Contract", "Intermediate &rarr; Advanced", 30,
           ["Parse Thought / Action / Observation out of free-form model text",
            "Break your own parser on eight real drift cases",
            "Choose between strict and lenient, and defend the choice",
            "Drive the Module 1 loop with the parser you wrote"],
           "> **Builds on Lab 1.1.** You wrote the loop; here you write the thing that turns the\n"
           "> model's prose into an actual tool call, every single time, or fails safely."),
    setup(2),
    code(DOMAIN),
    code(CARRIED_TOOLS),

    md("""
## Concept

ReAct interleaves reasoning with tool use: **Thought** &rarr; **Action** &rarr; **Observation**, and
round again. The loop is Module 1's. What is new is that the model's *text* must become a *call*.

That parser is a **protocol boundary between a system that guarantees its output format and one
that does not.** It will meet malformed input in production. The only question is what it does then.
"""),

    md("""
## Section 1 &mdash; The happy path

Start with well-formed output. `parse_step` returns a dict describing what the model wants.
"""),
    code('''
import re

def parse_step(text: str) -> dict:
    """Parse one ReAct step.

    Returns one of:
      {"kind": "action", "tool": str, "arg": str, "thought": str}
      {"kind": "final",  "answer": str, "thought": str}
      {"kind": "unparseable", "raw": str}
    """
    thought = ""
    m = re.search(r"Thought:\\**\\s*(.+)", text)
    if m:
        thought = m.group(1).strip()

    m = re.search(r"Final Answer:\\**\\s*(.+)", text, re.S)
    if m:
        return {"kind": "final", "answer": m.group(1).strip(), "thought": thought}

    m = re.search(r'Action:\\**\\s*(\\w+)\\s*\\(\\s*"([^"]*)"\\s*\\)', text)
    if m:
        return BLANK                 # TODO: the action result -- tool name and its argument
    return {"kind": "unparseable", "raw": text}
''', '''
import re

def parse_step(text: str) -> dict:
    """Parse one ReAct step.

    Returns one of:
      {"kind": "action", "tool": str, "arg": str, "thought": str}
      {"kind": "final",  "answer": str, "thought": str}
      {"kind": "unparseable", "raw": str}
    """
    thought = ""
    m = re.search(r"Thought:\\**\\s*(.+)", text)
    if m:
        thought = m.group(1).strip()

    m = re.search(r"Final Answer:\\**\\s*(.+)", text, re.S)
    if m:
        return {"kind": "final", "answer": m.group(1).strip(), "thought": thought}

    m = re.search(r'Action:\\**\\s*(\\w+)\\s*\\(\\s*"([^"]*)"\\s*\\)', text)
    if m:
        return {"kind": "action", "tool": m.group(1), "arg": m.group(2), "thought": thought}
    return {"kind": "unparseable", "raw": text}
'''),
    code('''
# --- Self-check: Section 1
_good = 'Thought: I need the record.\\nAction: lookup_payment("PMT-1002")'
_final = 'Thought: I have enough.\\nFinal Answer: retry once after 24h'

check("a well-formed action parses", lambda: parse_step(_good)["kind"] == "action")
check("the tool name is extracted", lambda: parse_step(_good)["tool"] == "lookup_payment")
check("the argument is extracted", lambda: parse_step(_good)["arg"] == "PMT-1002")
check("the thought is kept", lambda: "record" in parse_step(_good)["thought"])
check("a final answer parses as final", lambda: parse_step(_final)["kind"] == "final")
check("noise is reported, not raised", lambda: parse_step("hello")["kind"] == "unparseable")
'''),

    md("""
## Section 2 &mdash; Eight ways it drifts

Every one of these is something a real model emits. Run the cell and see how many your parser
survives &mdash; the point is not to score well, it is to see the shape of the problem.
"""),
    code('''
DRIFT = [
    ("bold markers",      '**Thought:** need it\\n**Action:** lookup_payment("PMT-1002")'),
    ("unquoted argument", 'Action: lookup_payment(PMT-1002)'),
    ("single quotes",     "Action: lookup_payment('PMT-1002')"),
    ("prose argument",    'Action: lookup_payment for payment PMT-1002'),
    ("two actions",       'Action: lookup_payment("PMT-1002")\\nAction: policy_for("X")'),
    ("thought only",      'Thought: I should probably check the ledger first.'),
    ("fenced in code",    '```\\nAction: lookup_payment("PMT-1002")\\n```'),
    ("trailing chatter",  'Action: lookup_payment("PMT-1002")\\nLet me know if you need more!'),
]

print(f"{'drift case':22}{'parsed as':16}{'tool':18}arg")
for name, text in DRIFT:
    try:
        r = parse_step(text)
        print(f"{name:22}{r['kind']:16}{r.get('tool',''):18}{r.get('arg','')}")
    except NameError:
        print("(fill in parse_step above, then re-run)"); break
'''),
    code('''
# --- Self-check: Section 2   (what a STRICT parser must and must not do)
def kind_of(text):
    return parse_step(text)["kind"]

check("bold markers still parse -- the regex is not anchored to line start",
      lambda: kind_of(DRIFT[0][1]) == "action")
check("an unquoted argument is refused rather than guessed",
      lambda: kind_of(DRIFT[1][1]) == "unparseable",
      "a strict parser must not invent the argument it did not see")
check("a thought with no action is refused",
      lambda: kind_of(DRIFT[5][1]) == "unparseable")
check("a fenced action still parses", lambda: kind_of(DRIFT[6][1]) == "action")
check("trailing chatter does not break the action",
      lambda: parse_step(DRIFT[7][1])["arg"] == "PMT-1002")
check("two actions in one turn takes only the first",
      lambda: parse_step(DRIFT[4][1])["arg"] == "PMT-1002",
      "executing both is how one turn becomes two side effects")
'''),

    md("""
## Section 3 &mdash; Strict or lenient?

Now the judgement. A **lenient** parser accepts the unquoted argument and keeps the run going. A
**strict** one refuses and asks the model to restate. Both are defensible; they fail differently.
"""),
    code('''
def parse_lenient(text: str) -> dict:
    """Strict first; if that fails, accept a bare unquoted argument."""
    r = parse_step(text)
    if r["kind"] != "unparseable":
        return r
    m = re.search(r"Action:\\**\\s*(\\w+)\\s*\\(\\s*([^)\\"']+?)\\s*\\)", text)
    if m:
        return {"kind": "action", "tool": m.group(1), "arg": m.group(2).strip(), "thought": ""}
    return r

def parser_for(tool_kind: str) -> str:
    """Which parser should drive this kind of tool: "strict" or "lenient"?

    strict  -- refuse anything ambiguous and ask the model to restate
    lenient -- keep the run going, accepting a best guess at the argument
    """
    if tool_kind == "write":
        return BLANK                 # TODO: a misparsed write is irreversible
    return BLANK                     # TODO: a misparsed read costs one wasted step
''', '''
def parse_lenient(text: str) -> dict:
    """Strict first; if that fails, accept a bare unquoted argument."""
    r = parse_step(text)
    if r["kind"] != "unparseable":
        return r
    m = re.search(r"Action:\\**\\s*(\\w+)\\s*\\(\\s*([^)\\"']+?)\\s*\\)", text)
    if m:
        return {"kind": "action", "tool": m.group(1), "arg": m.group(2).strip(), "thought": ""}
    return r

def parser_for(tool_kind: str) -> str:
    """Which parser should drive this kind of tool: "strict" or "lenient"?

    strict  -- refuse anything ambiguous and ask the model to restate
    lenient -- keep the run going, accepting a best guess at the argument
    """
    if tool_kind == "write":
        return "strict"              # a wrong guess here is an irreversible action
    return "lenient"                 # a wrong read costs one step, and the agent recovers
'''),
    code('''
# --- Self-check: Section 3
check("the lenient parser recovers the unquoted argument",
      lambda: parse_lenient(DRIFT[1][1])["arg"] == "PMT-1002")
check("the lenient parser still refuses a thought with no action",
      lambda: parse_lenient(DRIFT[5][1])["kind"] == "unparseable",
      "lenient means tolerant of format, not of missing intent")
check("writes are driven by the strict parser",
      lambda: parser_for("write") == "strict",
      "a misparsed write is irreversible; a refusal costs one retry")
check("reads can afford the lenient parser",
      lambda: parser_for("read") == "lenient",
      "a wrong read wastes a step and the agent recovers from the observation")
'''),

    md("""
## Section 4 &mdash; Drive the loop

Wire the parser into a ReAct loop with Module 1's budget and stop conditions. Deterministic here:
a scripted model, so the loop is exercised without a live call.
"""),
    code('''
def react_loop(steps, tools, max_steps=6):
    """steps: a list of model replies, replayed in order. Returns the run state."""
    state = {"steps": 0, "answer": None, "trace": [], "stopped": None}
    for reply in steps:
        if state["steps"] >= max_steps:
            state["stopped"] = "budget"
            return state
        r = parse_step(reply)
        if r["kind"] == "final":
            state["answer"] = r["answer"]
            state["stopped"] = "goal"
            return state
        if r["kind"] == "unparseable":
            state["trace"].append(("unparseable", reply[:30], "asked to restate"))
            state["steps"] += 1
            continue
        fn = tools.get(r["tool"])
        obs = fn(r["arg"]) if fn else f"no such tool {r['tool']!r}"
        state["trace"].append((r["tool"], r["arg"], obs))
        state["steps"] = BLANK       # TODO: spend one unit of budget
    state["stopped"] = state["stopped"] or "ran out of replies"
    return state
''', '''
def react_loop(steps, tools, max_steps=6):
    """steps: a list of model replies, replayed in order. Returns the run state."""
    state = {"steps": 0, "answer": None, "trace": [], "stopped": None}
    for reply in steps:
        if state["steps"] >= max_steps:
            state["stopped"] = "budget"
            return state
        r = parse_step(reply)
        if r["kind"] == "final":
            state["answer"] = r["answer"]
            state["stopped"] = "goal"
            return state
        if r["kind"] == "unparseable":
            state["trace"].append(("unparseable", reply[:30], "asked to restate"))
            state["steps"] += 1
            continue
        fn = tools.get(r["tool"])
        obs = fn(r["arg"]) if fn else f"no such tool {r['tool']!r}"
        state["trace"].append((r["tool"], r["arg"], obs))
        state["steps"] = state["steps"] + 1
    state["stopped"] = state["stopped"] or "ran out of replies"
    return state
'''),
    code('''
# --- Self-check: Section 4
_script = [
    'Thought: get the record.\\nAction: lookup_payment("PMT-1003")',
    'Thought: now the policy.\\nAction: policy_for("LIMIT_BREACH")',
    'Thought: done.\\nFinal Answer: Treasury must approve before release.',
]
def _run():
    return react_loop(_script, TOOLS)

check("the run reaches its goal", lambda: _run()["stopped"] == "goal")
check("both tool calls are traced", lambda: len(_run()["trace"]) == 2)
check("the observation carries the real ledger data",
      lambda: "LIMIT_BREACH" in _run()["trace"][0][2])
check("the final answer is captured", lambda: "Treasury" in _run()["answer"])
check("an unparseable step costs budget but does not stop the run",
      lambda: react_loop(["garbage"] + _script, TOOLS)["stopped"] == "goal",
      "state['steps'] must advance on the unparseable branch too")
check("a script that never finishes stops on the budget",
      lambda: react_loop(['Action: lookup_payment("PMT-1002")'] * 10, TOOLS)["stopped"] == "budget")
'''),

    md("""
## Run it for real

A live model, your parser, real tools. Watch the format the model actually chooses &mdash; you did not
specify it, so it picked one.
"""),
    code('''
REACT_SYSTEM = """You investigate payment exceptions. Work in this exact format:

Thought: <one line of reasoning>
Action: <tool>("<argument>")

Available tools:
  lookup_payment("PMT-1002")  -- returns the ledger record
  policy_for("LIMIT_BREACH")  -- returns the policy for a reason code

After an Observation, continue with another Thought/Action, or finish with:
Final Answer: <what should happen>

Emit exactly one Thought and one Action per turn."""

if llm_ready():
    try:
        transcript, state = [], {"steps": 0}
        convo = "Investigate PMT-1005."
        for turn in range(5):
            reply = ask(convo, system=REACT_SYSTEM)
            r = parse_step(reply)
            print(f"--- turn {turn + 1} [{r['kind']}]")
            print("   " + reply.strip().replace("\\n", "\\n   ")[:240])
            if r["kind"] == "final":
                break
            if r["kind"] == "unparseable":
                convo += "\\nObservation: could not parse that. Reply with exactly one Thought and one Action."
                continue
            fn = TOOLS.get(r["tool"])
            obs = fn(r["arg"]) if fn else f"no such tool {r['tool']!r}"
            print(f"   Observation: {obs[:160]}")
            convo += f"\\n{reply}\\nObservation: {obs}"
    except NameError:
        print("(fill in the blanks above, then re-run this cell)")
'''),
    md("""
### Read the trace

Count the `[unparseable]` turns. Zero means this model happens to follow your format today &mdash; not
that your parser is safe. The drift table in Section 2 is what it looks like when that changes,
which it does with every model swap and most prompt edits.

Notice too that the recovery path matters: an unparseable turn feeds a corrective observation back
rather than crashing. That is the same "tools report failure, they do not raise" discipline from
Module 1, applied to the model's own output.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add a ninth drift case from your own experience and decide, with a reason, whether strict or
   lenient should handle it.
2. `react_loop` gives an unparseable turn the same budget cost as a real step. Should it? Argue
   both sides &mdash; then consider what a separate, smaller "reformat" budget would buy you.
"""),
]


# =========================================================================== #
# Lab 2.3 -- decomposition that finishes, and re-planning that knows why
# =========================================================================== #
LAB3 = [
    header(3, "Sub-goals That Finish, and Knowing When to Re-plan", "Advanced", 35,
           ["Write the test that separates a finishable sub-goal from a wish",
            "Classify a failure as transient or a wrong plan -- from the error, not a guess",
            "Implement the four-rung escalation ladder with a bound on every rung",
            "Watch a retry-only agent and a diagnosing agent meet the same failures"],
           "> **Builds on Lab 1.2's `order_steps`.** There you ordered a plan someone gave you.\n"
           "> Here you judge whether the steps were worth ordering, and what to do when one fails."),
    setup(3),
    code(DOMAIN),
    code(CARRIED_TOOLS),

    md("""
## Concept

Two failures look identical from inside the loop and need opposite responses:

- **Transient** &mdash; the plan was fine, the world was briefly busy. **Retry**, bounded.
- **Wrong plan** &mdash; the step cannot succeed as written. **Re-plan**, feeding the error back.

Retry a wrong plan and you buy the same failure repeatedly. Re-plan a blip and you throw away
correct work. The diagnosis is the whole job, and the error your tools return is the evidence.
"""),

    md("""
## Section 1 &mdash; Is this sub-goal finishable?

A sub-goal an agent cannot finish becomes a loop that gets misdiagnosed as a reasoning bug. The
test is mechanical: is there an output that would settle it?
"""),
    code('''
VAGUE = ("understand", "thoroughly", "make sure", "as needed", "properly",
         "fully", "appropriate", "confident", "comprehensive")

def is_finishable(subgoal: str) -> bool:
    """True when a sub-goal has a detectable definition of done.

    Rejects a sub-goal that contains a vague qualifier from VAGUE, or that does not
    start with a concrete verb naming what will be produced.
    """
    text = subgoal.lower().strip()
    if any(w in text for w in VAGUE):
        return False
    return BLANK                     # TODO: does it start with a verb that names an output?
                                     # The concrete ones here are: return, fetch, compute,
                                     # classify, state, list.
''', '''
VAGUE = ("understand", "thoroughly", "make sure", "as needed", "properly",
         "fully", "appropriate", "confident", "comprehensive")

CONCRETE = ("return", "fetch", "compute", "classify", "state", "list")

def is_finishable(subgoal: str) -> bool:
    """True when a sub-goal has a detectable definition of done.

    Rejects a sub-goal that contains a vague qualifier from VAGUE, or that does not
    start with a concrete verb naming what will be produced.
    """
    text = subgoal.lower().strip()
    if any(w in text for w in VAGUE):
        return False
    return text.startswith(CONCRETE)
'''),
    code('''
# --- Self-check: Section 1
check("a concrete sub-goal passes",
      lambda: is_finishable("Return the reason code for PMT-1002") is True)
check("a vague qualifier fails it",
      lambda: is_finishable("Understand the payment problem") is False)
check("'make sure nothing is missed' fails",
      lambda: is_finishable("Make sure nothing has been missed") is False)
check("a concrete verb is required",
      lambda: is_finishable("Look into the ledger a bit") is False,
      "if the step does not name an output, its completion is not detectable")
check("'state whether ...' passes",
      lambda: is_finishable("State whether policy reserves this for a human") is True)
check("'investigate until confident' fails",
      lambda: is_finishable("Investigate until confident") is False)
'''),

    md("""
## Section 2 &mdash; Diagnose the failure

The error a tool returns is data. Classify on it rather than on intuition, and the right response
follows automatically.
"""),
    code('''
TRANSIENT = ("timeout", "timed out", "503", "502", "429", "connection reset",
             "temporarily unavailable", "deadlock", "try again")
PERMANENT = ("404", "not found", "no payment found", "403", "forbidden",
             "invalid", "no such", "malformed", "unauthorised")

def diagnose(error: str) -> str:
    """Return 'transient', 'wrong_plan' or 'unknown' for one error string."""
    e = error.lower()
    if any(t in e for t in TRANSIENT):
        return "transient"
    if BLANK:                        # TODO: does it look permanent?
        return "wrong_plan"
    return "unknown"

def respond_to(diagnosis: str, attempts: int, max_attempts: int = 3) -> str:
    """The rung of the ladder to take. One of: retry, replan, escalate."""
    if diagnosis == "transient":
        return "retry" if attempts < max_attempts else BLANK   # TODO: a bounded rung must end
    if diagnosis == "wrong_plan":
        return "replan"
    return "escalate"                # unknown errors go to a human, never to a guess
''', '''
TRANSIENT = ("timeout", "timed out", "503", "502", "429", "connection reset",
             "temporarily unavailable", "deadlock", "try again")
PERMANENT = ("404", "not found", "no payment found", "403", "forbidden",
             "invalid", "no such", "malformed", "unauthorised")

def diagnose(error: str) -> str:
    """Return 'transient', 'wrong_plan' or 'unknown' for one error string."""
    e = error.lower()
    if any(t in e for t in TRANSIENT):
        return "transient"
    if any(p in e for p in PERMANENT):
        return "wrong_plan"
    return "unknown"

def respond_to(diagnosis: str, attempts: int, max_attempts: int = 3) -> str:
    """The rung of the ladder to take. One of: retry, replan, escalate."""
    if diagnosis == "transient":
        return "retry" if attempts < max_attempts else "escalate"
    if diagnosis == "wrong_plan":
        return "replan"
    return "escalate"                # unknown errors go to a human, never to a guess
'''),
    code('''
# --- Self-check: Section 2
check("a 503 is transient", lambda: diagnose("HTTP 503 service unavailable") == "transient")
check("a timeout is transient", lambda: diagnose("read timed out after 30s") == "transient")
check("a 404 is a wrong plan", lambda: diagnose("HTTP 404 not found") == "wrong_plan")
check("the ledger's own miss is a wrong plan",
      lambda: diagnose("no payment found with reference 'PMT-9999'") == "wrong_plan")
check("an unrecognised error is not guessed at",
      lambda: diagnose("kernel panic in the mainframe") == "unknown",
      "returning 'transient' by default is how a permanent failure gets retried forever")
check("a transient failure retries while attempts remain",
      lambda: respond_to("transient", attempts=1) == "retry")
check("retries are bounded -- the rung ends in escalation",
      lambda: respond_to("transient", attempts=3) == "escalate",
      "unbounded retry turns a blip into an outage on your side")
check("a wrong plan re-plans rather than retrying",
      lambda: respond_to("wrong_plan", attempts=0) == "replan")
check("an unknown error escalates", lambda: respond_to("unknown", attempts=0) == "escalate")
'''),

    md("""
## Section 3 &mdash; Two agents meet the same failures

A retry-only agent and a diagnosing agent, run against a scripted sequence of failures. The
difference is not subtle.
"""),
    code('''
FAILURES = [
    "HTTP 503 service unavailable",                  # transient -- retry works
    "HTTP 503 service unavailable",
    "ok: {'ref': 'PMT-1003', 'status': 'held'}",     # succeeds on the third attempt
    "no payment found with reference 'PMT-0000'",    # permanent -- retry can never work
]

def run_retry_only(events, max_attempts=3):
    """Retries everything. The naive agent most teams ship first."""
    calls, attempts = 0, 0
    for e in events:
        calls += 1
        if e.startswith("ok:"):
            return {"calls": calls, "outcome": "success"}
        attempts += 1
        if attempts >= max_attempts:
            return {"calls": calls, "outcome": "gave up after retrying"}
    return {"calls": calls, "outcome": "exhausted events"}

def run_diagnosing(events, max_attempts=3):
    """Classifies each failure and takes the matching rung."""
    calls, attempts = 0, 0
    for e in events:
        calls += 1
        if e.startswith("ok:"):
            return {"calls": calls, "outcome": "success"}
        action = respond_to(diagnose(e), attempts)
        if action == "retry":
            attempts += 1
            continue
        return {"calls": calls, "outcome": BLANK}    # TODO: report the rung taken, e.g. "replan"
    return {"calls": calls, "outcome": "exhausted events"}
''', '''
FAILURES = [
    "HTTP 503 service unavailable",                  # transient -- retry works
    "HTTP 503 service unavailable",
    "ok: {'ref': 'PMT-1003', 'status': 'held'}",     # succeeds on the third attempt
    "no payment found with reference 'PMT-0000'",    # permanent -- retry can never work
]

def run_retry_only(events, max_attempts=3):
    """Retries everything. The naive agent most teams ship first."""
    calls, attempts = 0, 0
    for e in events:
        calls += 1
        if e.startswith("ok:"):
            return {"calls": calls, "outcome": "success"}
        attempts += 1
        if attempts >= max_attempts:
            return {"calls": calls, "outcome": "gave up after retrying"}
    return {"calls": calls, "outcome": "exhausted events"}

def run_diagnosing(events, max_attempts=3):
    """Classifies each failure and takes the matching rung."""
    calls, attempts = 0, 0
    for e in events:
        calls += 1
        if e.startswith("ok:"):
            return {"calls": calls, "outcome": "success"}
        action = respond_to(diagnose(e), attempts)
        if action == "retry":
            attempts += 1
            continue
        return {"calls": calls, "outcome": action}
    return {"calls": calls, "outcome": "exhausted events"}
'''),
    code('''
# --- Self-check: Section 3
_transient_run = FAILURES[:3]
_permanent_first = ["no payment found with reference 'PMT-0000'"] + FAILURES[:3]

check("both agents ride out a transient blip",
      lambda: run_retry_only(_transient_run)["outcome"] == "success"
              and run_diagnosing(_transient_run)["outcome"] == "success")
check("the retry-only agent burns three calls on a permanent failure",
      lambda: run_retry_only(_permanent_first)["calls"] == 3)
check("the diagnosing agent re-plans on the first permanent failure",
      lambda: run_diagnosing(_permanent_first)["outcome"] == "replan")
check("...and it spends exactly one call to find that out",
      lambda: run_diagnosing(_permanent_first)["calls"] == 1,
      "the error said 'not found' on attempt one; nothing is learned by asking again")

for name, fn in (("retry-only", run_retry_only), ("diagnosing", run_diagnosing)):
    try:
        r = fn(_permanent_first)
        print(f"{name:12} calls={r['calls']}  outcome={r['outcome']}")
    except NameError:
        print("(fill in the blanks above, then re-run)"); break
'''),

    md("""
## Section 4 &mdash; Plan, then validate it

Put the two halves together: decompose a goal, reject the sub-goals that cannot finish, and order
what survives with Lab 1.2's dependency rule.
"""),
    code('''
def validate_plan(subgoals: dict[str, list[str]]) -> dict:
    """Split a plan into the sub-goals worth running and the ones that cannot finish.

    Returns {"runnable": [...ordered...], "rejected": [...]}.
    """
    rejected = [s for s in subgoals if not is_finishable(s)]
    keep = {s: [d for d in deps if d not in rejected]
            for s, deps in subgoals.items() if s not in rejected}

    ordered, done = [], set()
    while len(ordered) < len(keep):
        progressed = False
        for name, deps in keep.items():
            if name in done:
                continue
            if all(d in done for d in deps):
                ordered.append(name); done.add(name); progressed = True
        if not progressed:
            raise ValueError("cycle in plan")
    return {"runnable": ordered, "rejected": rejected}
'''),
    code('''
# --- Self-check: Section 4
PLAN = {
    "Return the ledger record for the payment": [],
    "Understand the payment problem": [],                     # cannot finish
    "Return the policy text for its reason code": ["Return the ledger record for the payment"],
    "State whether policy reserves this for a human": ["Return the policy text for its reason code"],
    "Make sure nothing has been missed": [],                  # cannot finish
}

def _v():
    return validate_plan(PLAN)

check("both unfinishable sub-goals are rejected", lambda: len(_v()["rejected"]) == 2)
check("the three runnable sub-goals survive", lambda: len(_v()["runnable"]) == 3)
check("the ledger read comes first",
      lambda: _v()["runnable"][0].startswith("Return the ledger"))
check("dependencies are respected",
      lambda: (lambda o: o.index("Return the policy text for its reason code")
                       < o.index("State whether policy reserves this for a human"))(_v()["runnable"]))

try:
    v = _v()
    print("runnable:"); [print("   ", s) for s in v["runnable"]]
    print("rejected:"); [print("   ", s) for s in v["rejected"]]
except NameError:
    print("(fill in is_finishable above, then re-run)")
'''),

    md("""
## Run it for real

Have the model decompose a goal, then put its plan through your validator. Models produce vague
sub-goals readily &mdash; this is the check that catches them before they become loops.
"""),
    code('''
if llm_ready():
    try:
        raw = ask(
            "Break this goal into 4 to 6 sub-goals, one per line, no numbering or bullets: "
            "'Determine who must action the exception on payment PMT-1005 and why.' "
            "Each sub-goal must start with one of: Return, Fetch, Compute, Classify, State, List."
        )
        proposed = [l.strip("-* \\t") for l in raw.splitlines() if l.strip()]
        print("the model proposed:")
        for s in proposed:
            mark = "keep  " if is_finishable(s) else "REJECT"
            print(f"  [{mark}] {s}")
        bad = [s for s in proposed if not is_finishable(s)]
        print(f"\\n{len(proposed) - len(bad)} of {len(proposed)} sub-goals are finishable.")
        if bad:
            print("Rejected because completion is not detectable from any output:")
            for s in bad:
                print("   " + s)
    except NameError:
        print("(fill in the blanks above, then re-run this cell)")
'''),
    md("""
### Read it

Even told exactly which verbs to use, models slip in a "thoroughly" or an "as needed". Each one is
a step whose completion nothing can detect &mdash; and a step the agent will keep working on.

This validator is cheap and runs before any tokens are spent on execution. It is the highest
return-per-line check in the whole module.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `diagnose()` returns `unknown` for anything unrecognised, and `unknown` escalates. Argue for the
   opposite default, then say what would have to be true about your tools for it to be safe.
2. Rung 2 of the ladder &mdash; *try a different tool for the same sub-goal* &mdash; is not implemented.
   Add it, and decide where it sits between retry and re-plan.
"""),
]


# =========================================================================== #
# Lab 2.4 -- Tree-of-Thought and Reflection: breadth and second opinions
# =========================================================================== #
LAB4 = [
    header(4, "Branch, Score, Prune &mdash; and Know When to Stop Reflecting", "Advanced", 35,
           ["Count what a tree costs before you build one",
            "Write the scorer, which is the part that decides whether the tree helps at all",
            "Prune under a budget and see which branch you lost",
            "Find the reflection knee: where another round stops paying"],
           "> **The two expensive architectures.** Both buy quality with tokens. This lab is about\n"
           "> knowing, in advance and then in evidence, whether the purchase was worth it."),
    setup(4),
    code(DOMAIN),
    code(CARRIED_TOOLS),

    md("""
## Concept

**Tree-of-Thought** explores several routes, scores them, and keeps the promising ones. Its cost is
`branches x depth` model calls, and its quality is entirely hostage to the **scorer**: a weak scorer
prunes the right branch and keeps the wrong one, which is worse than not branching at all and far
dearer.

**Reflection** drafts, criticises and revises. Gains flatten fast &mdash; typically a big win in round
one, a small one in round two, noise after that, at 2&ndash;3&times; cost per round.
"""),

    md("""
## Section 1 &mdash; Count the cost first

Before writing any tree, work out what it will cost. This is the number that stops most trees from
being built, which is the correct outcome.
"""),
    code('''
def tree_calls(branches: int, depth: int, prune_to: int | None = None) -> int:
    """Model calls to build a tree, counting every node generated.

    Without pruning, level d holds branches**d nodes and every one is generated.
    With prune_to=k, only k nodes survive each level and are expanded.
    """
    if prune_to is None:
        return BLANK                 # TODO: total nodes at levels 1..depth
    total, frontier = 0, 1
    for _ in range(depth):
        generated = frontier * branches
        total += generated
        frontier = min(prune_to, generated)
    return total
''', '''
def tree_calls(branches: int, depth: int, prune_to: int | None = None) -> int:
    """Model calls to build a tree, counting every node generated.

    Without pruning, level d holds branches**d nodes and every one is generated.
    With prune_to=k, only k nodes survive each level and are expanded.
    """
    if prune_to is None:
        return sum(branches ** d for d in range(1, depth + 1))
    total, frontier = 0, 1
    for _ in range(depth):
        generated = frontier * branches
        total += generated
        frontier = min(prune_to, generated)
    return total
'''),
    code('''
# --- Self-check: Section 1
check("3 branches, 3 deep, no pruning is 39 calls",
      lambda: tree_calls(3, 3) == 39,
      "3 + 9 + 27 -- every interior node was generated too, not just the 27 leaves")
check("one level is just the branching factor", lambda: tree_calls(3, 1) == 3)
check("pruning to 2 cuts it sharply", lambda: tree_calls(3, 3, prune_to=2) == 15,
      "3 + 6 + 6 -- after the first level the frontier is capped at 2")
check("pruning to 1 is a single line of reasoning, widened",
      lambda: tree_calls(3, 3, prune_to=1) == 9)
check("wider trees grow fast", lambda: tree_calls(5, 3) == 155)

for b, d in ((3, 2), (3, 3), (5, 3), (5, 4)):
    try:
        print(f"  {b} branches x {d} deep: {tree_calls(b, d):>5} calls unpruned, "
              f"{tree_calls(b, d, prune_to=2):>4} pruned to 2")
    except NameError:
        print("(fill in tree_calls above)"); break
'''),

    md("""
## Section 2 &mdash; The scorer is the design

Four candidate explanations for one exception. The scorer decides which survive. Write it to reward
what actually matters: consistency with the ledger record, and support from policy.
"""),
    code('''
CANDIDATES = [
    {"claim": "Held for sanctions review; Compliance must decide.",
     "cites_policy": True,  "matches_record": True,  "proposes_action": False},
    {"claim": "Held because the amount exceeds the limit; Treasury can release it.",
     "cites_policy": True,  "matches_record": False, "proposes_action": True},
    {"claim": "Probably a technical glitch; retry the payment.",
     "cites_policy": False, "matches_record": False, "proposes_action": True},
    {"claim": "Held; Operations should cancel and re-issue.",
     "cites_policy": False, "matches_record": True,  "proposes_action": True},
]

def score_candidate(c: dict) -> float:
    """Score a candidate explanation in [0, 1].

    Weighting, deliberately: agreeing with the record is worth most, citing policy next.
    Proposing an action on a payment reserved for a human is a PENALTY, not a bonus.
    """
    s = 0.0
    if c["matches_record"]:
        s += 0.5
    if c["cites_policy"]:
        s += 0.3
    if c["proposes_action"] and not c["cites_policy"]:
        s += BLANK                   # TODO: the penalty for acting without policy support
                                     # (a negative number)
    return max(0.0, min(1.0, s))
''', '''
CANDIDATES = [
    {"claim": "Held for sanctions review; Compliance must decide.",
     "cites_policy": True,  "matches_record": True,  "proposes_action": False},
    {"claim": "Held because the amount exceeds the limit; Treasury can release it.",
     "cites_policy": True,  "matches_record": False, "proposes_action": True},
    {"claim": "Probably a technical glitch; retry the payment.",
     "cites_policy": False, "matches_record": False, "proposes_action": True},
    {"claim": "Held; Operations should cancel and re-issue.",
     "cites_policy": False, "matches_record": True,  "proposes_action": True},
]

def score_candidate(c: dict) -> float:
    """Score a candidate explanation in [0, 1].

    Weighting, deliberately: agreeing with the record is worth most, citing policy next.
    Proposing an action on a payment reserved for a human is a PENALTY, not a bonus.
    """
    s = 0.0
    if c["matches_record"]:
        s += 0.5
    if c["cites_policy"]:
        s += 0.3
    if c["proposes_action"] and not c["cites_policy"]:
        s += -0.3
    return max(0.0, min(1.0, s))
'''),
    code('''
# --- Self-check: Section 2
def _ranked():
    return sorted(CANDIDATES, key=score_candidate, reverse=True)

check("the correct explanation ranks first",
      lambda: _ranked()[0]["claim"].startswith("Held for sanctions"))
check("acting without policy support is penalised, not rewarded",
      lambda: score_candidate(CANDIDATES[2]) < score_candidate(CANDIDATES[1]),
      "an unsupported action is worse than a wrong-but-grounded reading")
check("scores stay inside [0, 1]",
      lambda: all(0.0 <= score_candidate(c) <= 1.0 for c in CANDIDATES))
check("the record outweighs policy on its own",
      lambda: score_candidate({"cites_policy": False, "matches_record": True,
                               "proposes_action": False})
              > score_candidate({"cites_policy": True, "matches_record": False,
                                 "proposes_action": False}))

try:
    for c in _ranked():
        print(f"  {score_candidate(c):.2f}  {c['claim']}")
except NameError:
    print("(fill in score_candidate above)")
'''),

    md("""
## Section 3 &mdash; Prune, and look at what you lost

Pruning is where the saving comes from and where the risk lives. Keep the top `k` and record what
was dropped &mdash; because the branch you pruned is the one you will never hear about again.
"""),
    code('''
def prune(candidates, keep=2):
    """Return (kept, dropped), each sorted best-first."""
    ranked = sorted(candidates, key=score_candidate, reverse=True)
    return ranked[:keep], ranked[keep:]

def prune_regret(candidates, keep=2) -> float:
    """How much score was thrown away: the best dropped candidate's score.

    A high regret means the scorer is discarding something that looked good -- either the
    scorer is wrong, or keep is too small.
    """
    _, dropped = prune(candidates, keep)
    if not dropped:
        return 0.0
    return BLANK                     # TODO: the score of the best candidate you discarded
''', '''
def prune(candidates, keep=2):
    """Return (kept, dropped), each sorted best-first."""
    ranked = sorted(candidates, key=score_candidate, reverse=True)
    return ranked[:keep], ranked[keep:]

def prune_regret(candidates, keep=2) -> float:
    """How much score was thrown away: the best dropped candidate's score.

    A high regret means the scorer is discarding something that looked good -- either the
    scorer is wrong, or keep is too small.
    """
    _, dropped = prune(candidates, keep)
    if not dropped:
        return 0.0
    return max(score_candidate(c) for c in dropped)
'''),
    code('''
# --- Self-check: Section 3
check("pruning to 2 keeps two", lambda: len(prune(CANDIDATES, 2)[0]) == 2)
check("the correct explanation survives pruning",
      lambda: any(c["claim"].startswith("Held for sanctions") for c in prune(CANDIDATES, 2)[0]))
check("regret is the best score among the dropped",
      lambda: abs(prune_regret(CANDIDATES, 2)
                  - max(score_candidate(c) for c in prune(CANDIDATES, 2)[1])) < 1e-9)
check("keeping everything has zero regret",
      lambda: prune_regret(CANDIDATES, keep=len(CANDIDATES)) == 0.0)
check("pruning to 1 costs more regret than pruning to 2",
      lambda: prune_regret(CANDIDATES, 1) >= prune_regret(CANDIDATES, 2),
      "the tighter the prune, the more you risk discarding")

for k in (1, 2, 3):
    try:
        print(f"  keep={k}: regret {prune_regret(CANDIDATES, k):.2f}, "
              f"calls {tree_calls(3, 3, prune_to=k)}")
    except NameError:
        print("(fill in the blanks above)"); break
'''),

    md("""
## Section 4 &mdash; The reflection knee

Reflection pays until it does not. Given a measured series of quality gains and a cost multiple per
round, decide how many rounds ship.
"""),
    code('''
def rounds_worth_running(gains: list[float], cost_per_round: float, min_gain: float) -> int:
    """How many reflection rounds to ship.

    gains[i] is the pass-rate gain from round i+1. Stop at the first round whose gain
    falls below min_gain -- later rounds do not rescue it, and each one costs cost_per_round.
    Returns the number of rounds to run (0 means do not reflect at all).
    """
    n = 0
    for g in gains:
        if BLANK:                    # TODO: is this round worth running?
            break
        n += 1
    return n
''', '''
def rounds_worth_running(gains: list[float], cost_per_round: float, min_gain: float) -> int:
    """How many reflection rounds to ship.

    gains[i] is the pass-rate gain from round i+1. Stop at the first round whose gain
    falls below min_gain -- later rounds do not rescue it, and each one costs cost_per_round.
    Returns the number of rounds to run (0 means do not reflect at all).
    """
    n = 0
    for g in gains:
        if g < min_gain:
            break
        n += 1
    return n
'''),
    code('''
# --- Self-check: Section 4
MEASURED = [0.12, 0.02, 0.00]        # the shape from the slides: big, small, nothing

check("with a 10-point bar, only round 1 ships",
      lambda: rounds_worth_running(MEASURED, cost_per_round=3.0, min_gain=0.10) == 1)
check("with a 1-point bar, two rounds ship",
      lambda: rounds_worth_running(MEASURED, cost_per_round=3.0, min_gain=0.01) == 2)
check("a first round below the bar means no reflection at all",
      lambda: rounds_worth_running([0.01, 0.20], cost_per_round=3.0, min_gain=0.10) == 0,
      "stop at the FIRST round below the bar -- do not pay two rounds hoping for the second")
check("no gains means no rounds",
      lambda: rounds_worth_running([], cost_per_round=3.0, min_gain=0.10) == 0)
'''),

    md("""
## Run it for real

Reflection with a critic that has a checklist &mdash; the version that works &mdash; against a critic told
only to "improve this". Same drafter, same case, one variable changed.
"""),
    code('''
CHECKLIST_CRITIC = (
    "You are reviewing an operations note. Check exactly three things and list any that fail: "
    "(1) does it name the reason code from the record? (2) does it quote the applicable policy? "
    "(3) if policy reserves the decision for a human, does it say so and refrain from proposing "
    "an action? Reply with the failures, or 'OK' if none."
)
VAGUE_CRITIC = "Review this note and suggest improvements."

if llm_ready():
    try:
        rec = lookup_payment("PMT-1005")
        pol = policy_for("SANCTIONS_REVIEW")
        draft = ask(f"Write a two-sentence operations note.\\nRECORD: {rec}\\nPOLICY: {pol}")
        print("DRAFT:\\n  " + draft.strip().replace("\\n", "\\n  ")[:400])

        for name, critic in (("checklist critic", CHECKLIST_CRITIC), ("vague critic", VAGUE_CRITIC)):
            crit = ask(f"NOTE: {draft}", system=critic)
            revised = ask(f"Revise the note using this critique.\\nNOTE: {draft}\\nCRITIQUE: {crit}")
            print(f"\\n--- {name} ---")
            print("  critique: " + crit.strip().replace("\\n", " ")[:220])
            print("  revised : " + revised.strip().replace("\\n", " ")[:220])
    except NameError:
        print("(fill in the blanks above, then re-run this cell)")
'''),
    md("""
### Read it

Compare the two critiques, not the two revisions. The checklist critic can only find the three
things it was told to look for &mdash; and it finds them. The vague critic produces fluent suggestions
that mostly restate the draft, because the same model with the same knowledge has nothing new to
add.

That is the rule: **reflection works when the critic knows something the drafter did not use.** A
checklist, a schema, a policy. Without one you are paying 2&ndash;3&times; for a rephrase.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add a fifth candidate that is *plausible and wrong* &mdash; cites policy, matches the record, but
   draws the opposite conclusion. Does your scorer catch it? If not, what feature would?
2. `prune_regret` reports the best score you discarded, but the scorer produced that score too. If
   the scorer is wrong, regret is wrong in the same direction. Suggest a check that does not share
   the scorer's blind spot.
"""),
]


# =========================================================================== #
# Lab 2.5 -- challenge: the architecture bake-off
# =========================================================================== #
LAB5 = [
    header(5, "Challenge &mdash; The Architecture Bake-Off", "Advanced", 40,
           ["Run four architectures over one eval set with one meter",
            "Name the acceptance threshold before you look at any number",
            "Produce a scorecard that is allowed to reject the interesting answer",
            "Write the recommendation you would defend in a design review"],
           "> **The comprehensive lab for Module 2.** It uses everything: the eval set from 2.1, the\n"
           "> parser from 2.2, the diagnosis ladder from 2.3, and the cost counting from 2.4."),
    setup(5),
    code(DOMAIN),
    code(CARRIED_TOOLS),

    md("""
## Concept

Four architectures, one eval set, one meter, one threshold written down in advance. The scorecard
is allowed to say *the cheap one wins* &mdash; and on a task this size it usually does.

That outcome is the lesson, not a failed lab.
"""),

    md("""
## Section 1 &mdash; One meter for all four arms

Reuse Module 1's discipline: measure every arm the same way or the comparison is decoration.
"""),
    code('''
class Meter:
    """Counts calls and estimated tokens for one architecture arm."""

    def __init__(self, name: str):
        self.name = name
        self.calls = 0
        self.tokens = 0

    def record(self, prompt: str, reply: str) -> None:
        self.calls += 1
        self.tokens += (len(prompt) + len(reply)) // 4

    def report(self, pass_rate: float) -> dict:
        return {"arm": self.name, "pass_rate": round(pass_rate, 3),
                "calls": self.calls, "tokens": self.tokens}
'''),
    code('''
# --- Self-check: Section 1
_m = Meter("t")
_m.record("a" * 400, "b" * 400)
_m.record("a" * 400, "b" * 400)
check("two turns are counted", lambda: _m.report(1.0)["calls"] == 2)
check("tokens accumulate across turns", lambda: _m.report(1.0)["tokens"] == 400)
check("the pass rate is carried into the report", lambda: _m.report(0.5)["pass_rate"] == 0.5)
'''),

    md("""
## Section 2 &mdash; Four arms, deterministic

Real model calls make the bake-off non-reproducible, so the arms here are deterministic stand-ins
whose **cost shapes** match the real thing. The architecture comparison is what is being taught;
the live version is at the end.
"""),
    code('''
CASES = [
    {"ref": "PMT-1002", "expect": "OPERATIONS"},
    {"ref": "PMT-1003", "expect": "TREASURY"},
    {"ref": "PMT-1004", "expect": "ORIGINATOR"},
    {"ref": "PMT-1005", "expect": "COMPLIANCE"},
    {"ref": "PMT-1001", "expect": "OPERATIONS"},
    {"ref": "PMT-9999", "expect": "OPERATIONS"},
]

OWNER = {"INSUFFICIENT_FUNDS": "OPERATIONS", "LIMIT_BREACH": "TREASURY",
         "INVALID_IBAN": "ORIGINATOR", "SANCTIONS_REVIEW": "COMPLIANCE", None: "OPERATIONS"}

def _facts(ref):
    rec = LEDGER.get(ref)
    return "" if rec is None else json.dumps(rec)

def arm_direct(case, meter):
    """One call, no reasoning, no tools: it only sees the reference."""
    reply = OWNER.get(LEDGER.get(case["ref"], {}).get("reason_code")) if case["ref"] in LEDGER else "OPERATIONS"
    meter.record(case["ref"], str(reply))
    return str(reply)

def arm_cot(case, meter):
    """One call with reasoning: more tokens, and it reads the record it was given."""
    facts = _facts(case["ref"])
    reply = OWNER.get(LEDGER.get(case["ref"], {}).get("reason_code"), "OPERATIONS")
    meter.record(case["ref"] + facts + "reason step by step" * 12, str(reply))
    return str(reply)

def arm_react(case, meter):
    """Two tool round trips -- record, then policy -- each re-sending the context."""
    ctx = case["ref"]
    for tool_out in (lookup_payment(case["ref"]), policy_for(LEDGER.get(case["ref"], {}).get("reason_code"))):
        meter.record(ctx, tool_out)
        ctx += tool_out
    return OWNER.get(LEDGER.get(case["ref"], {}).get("reason_code"), "OPERATIONS")

def arm_tot(case, meter, branches=3, depth=2):
    """Branch and score: every generated node costs a call."""
    ctx = case["ref"] + _facts(case["ref"])
    for _ in range(branches * depth + branches):
        meter.record(ctx, "candidate explanation with a score")
    return OWNER.get(LEDGER.get(case["ref"], {}).get("reason_code"), "OPERATIONS")

ARMS = {"direct": arm_direct, "cot": arm_cot, "react": arm_react, "tot": arm_tot}

def run_arm(name, fn) -> dict:
    """Run one architecture over every case. Returns its scorecard row."""
    meter = Meter(name)
    hits = 0
    for c in CASES:
        answer = fn(c, meter)
        if BLANK:                    # TODO: did this case pass? (the expected owner, case-insensitive)
            hits += 1
    return meter.report(hits / len(CASES))
''', '''
CASES = [
    {"ref": "PMT-1002", "expect": "OPERATIONS"},
    {"ref": "PMT-1003", "expect": "TREASURY"},
    {"ref": "PMT-1004", "expect": "ORIGINATOR"},
    {"ref": "PMT-1005", "expect": "COMPLIANCE"},
    {"ref": "PMT-1001", "expect": "OPERATIONS"},
    {"ref": "PMT-9999", "expect": "OPERATIONS"},
]

OWNER = {"INSUFFICIENT_FUNDS": "OPERATIONS", "LIMIT_BREACH": "TREASURY",
         "INVALID_IBAN": "ORIGINATOR", "SANCTIONS_REVIEW": "COMPLIANCE", None: "OPERATIONS"}

def _facts(ref):
    rec = LEDGER.get(ref)
    return "" if rec is None else json.dumps(rec)

def arm_direct(case, meter):
    """One call, no reasoning, no tools: it only sees the reference."""
    reply = OWNER.get(LEDGER.get(case["ref"], {}).get("reason_code")) if case["ref"] in LEDGER else "OPERATIONS"
    meter.record(case["ref"], str(reply))
    return str(reply)

def arm_cot(case, meter):
    """One call with reasoning: more tokens, and it reads the record it was given."""
    facts = _facts(case["ref"])
    reply = OWNER.get(LEDGER.get(case["ref"], {}).get("reason_code"), "OPERATIONS")
    meter.record(case["ref"] + facts + "reason step by step" * 12, str(reply))
    return str(reply)

def arm_react(case, meter):
    """Two tool round trips -- record, then policy -- each re-sending the context."""
    ctx = case["ref"]
    for tool_out in (lookup_payment(case["ref"]), policy_for(LEDGER.get(case["ref"], {}).get("reason_code"))):
        meter.record(ctx, tool_out)
        ctx += tool_out
    return OWNER.get(LEDGER.get(case["ref"], {}).get("reason_code"), "OPERATIONS")

def arm_tot(case, meter, branches=3, depth=2):
    """Branch and score: every generated node costs a call."""
    ctx = case["ref"] + _facts(case["ref"])
    for _ in range(branches * depth + branches):
        meter.record(ctx, "candidate explanation with a score")
    return OWNER.get(LEDGER.get(case["ref"], {}).get("reason_code"), "OPERATIONS")

ARMS = {"direct": arm_direct, "cot": arm_cot, "react": arm_react, "tot": arm_tot}

def run_arm(name, fn) -> dict:
    """Run one architecture over every case. Returns its scorecard row."""
    meter = Meter(name)
    hits = 0
    for c in CASES:
        answer = fn(c, meter)
        if c["expect"].lower() in str(answer).lower():
            hits += 1
    return meter.report(hits / len(CASES))
'''),
    code('''
# --- Self-check: Section 2
def _board():
    return [run_arm(n, f) for n, f in ARMS.items()]

check("every arm answers every case",
      lambda: all(r["calls"] >= len(CASES) for r in _board()))
check("react costs two calls per case", lambda: run_arm("react", arm_react)["calls"] == 12)
check("the tree costs far more than react",
      lambda: run_arm("tot", arm_tot)["calls"] > run_arm("react", arm_react)["calls"] * 2)
check("chain-of-thought costs more tokens than direct for the same call count",
      lambda: run_arm("cot", arm_cot)["tokens"] > run_arm("direct", arm_direct)["tokens"]
              and run_arm("cot", arm_cot)["calls"] == run_arm("direct", arm_direct)["calls"])
check("at least one arm scores on the eval set",
      lambda: max(r["pass_rate"] for r in _board()) >= 0.8,
      "check the pass condition in run_arm")
'''),

    md("""
## Section 3 &mdash; The threshold, written first

Name what an architecture must deliver before you have seen a single number. This is the clause
that lets the scorecard reject the interesting answer.
"""),
    code('''
THRESHOLD = {
    "min_gain": 0.10,        # pass-rate points over the cheapest arm
    "max_token_ratio": 5.0,  # ...within 5x its tokens
}

def verdict(row: dict, baseline: dict, threshold: dict = THRESHOLD) -> tuple[bool, str]:
    """Should this arm displace the baseline? Returns (yes, reason)."""
    gain = row["pass_rate"] - baseline["pass_rate"]
    ratio = row["tokens"] / max(baseline["tokens"], 1)
    if row["arm"] == baseline["arm"]:
        return True, "the baseline"
    if gain < threshold["min_gain"]:
        return False, f"gain {gain:+.2f} below the {threshold['min_gain']:.2f} bar"
    if BLANK:                        # TODO: is it over the token ceiling?
        return False, f"{ratio:.1f}x tokens exceeds {threshold['max_token_ratio']}x"
    return True, f"gain {gain:+.2f} at {ratio:.1f}x tokens"
''', '''
THRESHOLD = {
    "min_gain": 0.10,        # pass-rate points over the cheapest arm
    "max_token_ratio": 5.0,  # ...within 5x its tokens
}

def verdict(row: dict, baseline: dict, threshold: dict = THRESHOLD) -> tuple[bool, str]:
    """Should this arm displace the baseline? Returns (yes, reason)."""
    gain = row["pass_rate"] - baseline["pass_rate"]
    ratio = row["tokens"] / max(baseline["tokens"], 1)
    if row["arm"] == baseline["arm"]:
        return True, "the baseline"
    if gain < threshold["min_gain"]:
        return False, f"gain {gain:+.2f} below the {threshold['min_gain']:.2f} bar"
    if ratio > threshold["max_token_ratio"]:
        return False, f"{ratio:.1f}x tokens exceeds {threshold['max_token_ratio']}x"
    return True, f"gain {gain:+.2f} at {ratio:.1f}x tokens"
'''),
    code('''
# --- Self-check: Section 3
_base = {"arm": "direct", "pass_rate": 0.60, "calls": 6, "tokens": 100}
_cheap_win = {"arm": "cot", "pass_rate": 0.85, "calls": 6, "tokens": 300}
_dear_win  = {"arm": "tot", "pass_rate": 0.90, "calls": 60, "tokens": 4000}
_no_gain   = {"arm": "react", "pass_rate": 0.62, "calls": 12, "tokens": 400}

check("a real gain inside the ceiling is accepted",
      lambda: verdict(_cheap_win, _base)[0] is True)
check("a big gain outside the ceiling is rejected",
      lambda: verdict(_dear_win, _base)[0] is False,
      "a 30-point gain does not license 40x the tokens -- that is what the ceiling is for")
check("a marginal gain is rejected", lambda: verdict(_no_gain, _base)[0] is False)
check("the baseline always passes against itself",
      lambda: verdict(_base, _base)[0] is True)
check("every rejection states a reason", lambda: len(verdict(_dear_win, _base)[1]) > 10)
'''),

    md("""
## Section 4 &mdash; The scorecard

One table. The cheapest arm is the baseline; everything else has to earn its place against it.
"""),
    code('''
def scorecard() -> str:
    rows = sorted((run_arm(n, f) for n, f in ARMS.items()), key=lambda r: r["tokens"])
    baseline = rows[0]
    out = [f"{'arm':10}{'pass':>8}{'calls':>8}{'tokens':>9}{'x base':>9}  verdict",
           "-" * 78]
    for r in rows:
        ok, why = verdict(r, baseline)
        ratio = r["tokens"] / max(baseline["tokens"], 1)
        out.append(f"{r['arm']:10}{r['pass_rate']:>8}{r['calls']:>8}{r['tokens']:>9}"
                   f"{ratio:>8.1f}x  {'SHIP' if ok else 'no'} -- {why}")
    winners = [r["arm"] for r in rows if verdict(r, baseline)[0]]
    out.append("")
    out.append(f"Ships: {winners[-1]} (cheapest arm clearing the bar)")
    return "\\n".join(out)

try:
    print(scorecard())
except NameError:
    print("(finish the sections above, then re-run this cell)")
'''),
    code('''
# --- Self-check: Section 4
def _rows():
    return sorted((run_arm(n, f) for n, f in ARMS.items()), key=lambda r: r["tokens"])

check("the table has a row per arm plus a header, a rule and a footer",
      lambda: len(scorecard().splitlines()) == len(ARMS) + 4)
check("the cheapest arm is the baseline",
      lambda: _rows()[0]["tokens"] == min(r["tokens"] for r in _rows()))
check("the tree does not ship on this eval set",
      lambda: verdict([r for r in _rows() if r["arm"] == "tot"][0], _rows()[0])[0] is False,
      "it costs an order of magnitude more for no measured gain")
check("a tie goes to the cheaper arm",
      lambda: verdict({"arm": "x", "pass_rate": _rows()[0]["pass_rate"],
                       "calls": 1, "tokens": _rows()[0]["tokens"] * 2}, _rows()[0])[0] is False)
'''),

    md("""
## Run it for real

Have the model write the design-review paragraph &mdash; explaining a decision your scorecard already
made, not making one.
"""),
    code('''
if llm_ready():
    try:
        board = scorecard()
        summary = ask(
            "Write one short paragraph for an engineering design review. State which reasoning "
            "architecture was chosen, the evidence, and the condition under which the team should "
            "revisit it. Be plain and specific; add no claims beyond the table.\\n\\n"
            f"THRESHOLD AGREED IN ADVANCE: {THRESHOLD}\\n\\nSCORECARD:\\n{board}"
        )
        print(summary)
    except NameError:
        print("(finish the sections above, then re-run this cell)")
'''),
    md("""
### Read it

If the paragraph argues for the cheapest arm that cleared the bar, the process worked. If it
reaches for the tree because trees sound thorough, look at which clause let it through.

**What you take from Module 2:** a way to choose a reasoning architecture on evidence, a parser you
know the failure modes of, a failure diagnosis that picks retry or re-plan correctly, and a
scorecard that is allowed to tell you the boring answer. Module 3 gives all of it somewhere to keep
its state.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. The arms here are deterministic, so pass rate is fixed and only cost varies. Swap `arm_cot` and
   `arm_react` for real `ask()` calls, run three times, and report the spread. How much of your
   verdict survives the variance?
2. Add a case where ReAct genuinely beats Chain-of-Thought &mdash; one whose answer is not in the prompt
   at all. That is the case that justifies tools, and your eval set currently lacks it.
3. Raise `max_token_ratio` until the tree ships. Would you defend that number to a reviewer? If not,
   you have found your real ceiling.
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-2-01-chain-of-thought-measured",   LAB1),
    ("lab-2-02-react-parser-contract",       LAB2),
    ("lab-2-03-subgoals-and-replanning",     LAB3),
    ("lab-2-04-tree-of-thought-reflection",  LAB4),
    ("lab-2-05-challenge-architecture-bakeoff", LAB5),
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
