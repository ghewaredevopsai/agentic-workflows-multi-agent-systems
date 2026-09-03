#!/usr/bin/env python3
"""
Generate Module 1 lab notebooks and their solutions from one source.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-1-0N-*.ipynb and ../solutions/

Design rules (from Training/courses/CLAUDE.md and this course's stack):
  * Graded cells are pure Python -- they never call an LLM, so a self-check is
    deterministic and a flaky endpoint can never fail a participant.
  * Live-model cells are clearly marked, guarded, and never crash Run All.
  * "___" marks a blank; an unfilled blank raises NameError and prints [TODO].
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
# Lab 1.{num} &mdash; {title}

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 1 &middot; Module 1 &mdash; Agents vs. Multi-Agent Systems**

### What you'll do
{items}

> **How this lab works.** Fill every `___`, then run the **Self-check** cell under each section.
> Graded cells are plain Python and never call a model, so your score never depends on a
> live endpoint. Cells marked **Run it for real** do call the sandbox model; if it is not
> reachable they print how to fix it instead of crashing.

{note}
""")


SETUP_COMMON = '''
# ---------------------------------------------------------------- Setup: run me first
import os, json, time, textwrap
from typing import Any, Callable

WORK = os.path.join("/tmp", "awmas-lab-1-{num:02d}")
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


# =========================================================================== #
# Lab 1.1 -- LLM vs agent: statelessness, the loop, and knowing when to stop
# =========================================================================== #
LAB1 = [
    header(1, "From a Stateless Call to an Agent Loop", "Intermediate", 25,
           ["Prove to yourself that a model call carries nothing from the call before it",
            "Build the loop by hand -- decide, act, observe, and above all *stop*",
            "Add loop detection, the failure that quietly burns a budget in production"],
           "> **The thread.** All five Module 1 labs work one case: payment exceptions on a small\n"
           "> synthetic ledger. What you build here is extended in every later lab."),
    setup(1),
    code(DOMAIN),

    md("""
## Concept

A model call is a **function**: text in, text out, nothing retained. An **agent** is that call
placed inside a **loop**, where the output chooses the next action and the result is fed back.

Three things make the loop safe rather than merely clever:

| Piece | Question it answers |
|---|---|
| **State** | what has happened so far? |
| **Stop condition** | are we done, or out of budget? |
| **Loop detection** | are we going round without learning anything? |

The last two are what separate a demo from something you would run unattended.
"""),

    md("""
## Section 1 &mdash; State is something you resend

The model has no memory, so *you* carry the conversation. `carry()` builds the full message
list for the next call: every earlier turn, then the new message.
"""),
    code('''
def carry(history, user_msg):
    """Build the message list for the next call.

    history: [(role, text), ...] of earlier turns, oldest first.
    Returns: [(role, text), ...] ending with the new human message.
    """
    msgs = []
    for role, text in ___:          # TODO: which sequence replays the earlier turns?
        msgs.append((role, text))
    msgs.append(("human", user_msg))
    return msgs
''', '''
def carry(history, user_msg):
    """Build the message list for the next call.

    history: [(role, text), ...] of earlier turns, oldest first.
    Returns: [(role, text), ...] ending with the new human message.
    """
    msgs = []
    for role, text in history:      # the model gets the whole history back, every time
        msgs.append((role, text))
    msgs.append(("human", user_msg))
    return msgs
'''),
    code('''
# --- Self-check: Section 1
h = [("human", "The reference is PMT-1002."), ("ai", "Noted.")]
check("carry() replays every earlier turn", lambda: len(carry(h, "which reference?")) == 3)
check("carry() preserves the fact from turn 1",
      lambda: any("PMT-1002" in t for _, t in carry(h, "which reference?")),
      "the first turn must survive into the new call")
check("carry() puts the new message last",
      lambda: carry(h, "which reference?")[-1] == ("human", "which reference?"))
'''),

    md("""
## Section 2 &mdash; The loop, and the budget that bounds it

`run_agent` is the whole agent: ask `decide` what to do, do it, record what happened, repeat.
It is `should_stop` that keeps it from running forever, so that is the part you write.
"""),
    code('''
MAX_STEPS = 6

def should_stop(state):
    """Return (stop, reason). Two reasons matter here: the goal, and the budget."""
    if state["answer"] is not None:
        return True, "goal"
    if ___:                          # TODO: has the step budget been spent?
        return True, "budget"
    return False, None


def run_agent(goal, decide, tools, max_steps=MAX_STEPS):
    """decide(state) -> {"tool": name, "args": {...}}; the tool name "final" ends the run."""
    state = {"goal": goal, "steps": 0, "answer": None, "trace": [], "max_steps": max_steps}
    while True:
        stop, why = should_stop(state)
        if stop:
            state["stopped"] = why
            return state
        action = decide(state)
        if action["tool"] == "final":
            state["answer"] = action["args"]["text"]
            continue
        observation = tools[action["tool"]](**action["args"])
        state["trace"].append((action["tool"], action["args"], observation))
        state["steps"] = ___         # TODO: spend one unit of budget
''', '''
MAX_STEPS = 6

def should_stop(state):
    """Return (stop, reason). Two reasons matter here: the goal, and the budget."""
    if state["answer"] is not None:
        return True, "goal"
    if state["steps"] >= state["max_steps"]:     # a hard number, not a hope
        return True, "budget"
    return False, None


def run_agent(goal, decide, tools, max_steps=MAX_STEPS):
    """decide(state) -> {"tool": name, "args": {...}}; the tool name "final" ends the run."""
    state = {"goal": goal, "steps": 0, "answer": None, "trace": [], "max_steps": max_steps}
    while True:
        stop, why = should_stop(state)
        if stop:
            state["stopped"] = why
            return state
        action = decide(state)
        if action["tool"] == "final":
            state["answer"] = action["args"]["text"]
            continue
        observation = tools[action["tool"]](**action["args"])
        state["trace"].append((action["tool"], action["args"], observation))
        state["steps"] = state["steps"] + 1
'''),
    code('''
# --- Self-check: Section 2   (a scripted `decide` -- no model involved, so this is deterministic)
_tools = {"peek": lambda ref: LEDGER.get(ref, {}).get("status", "unknown")}

def _finisher(state):
    if state["steps"] >= 2:
        return {"tool": "final", "args": {"text": "PMT-1002 failed: INSUFFICIENT_FUNDS"}}
    return {"tool": "peek", "args": {"ref": "PMT-1002"}}

def _never_finishes(state):
    return {"tool": "peek", "args": {"ref": "PMT-1002"}}

check("a run that reaches its goal stops with reason 'goal'",
      lambda: run_agent("g", _finisher, _tools)["stopped"] == "goal")
check("a run that never finishes stops on the budget",
      lambda: run_agent("g", _never_finishes, _tools)["stopped"] == "budget",
      "should_stop() must compare steps against max_steps")
check("the budget is actually respected",
      lambda: run_agent("g", _never_finishes, _tools)["steps"] == MAX_STEPS,
      "state['steps'] has to advance on every tool call")
check("the trace records every observation",
      lambda: len(run_agent("g", _finisher, _tools)["trace"]) == 2)
'''),

    md("""
## Section 3 &mdash; Loop detection

A budget stops a runaway agent *eventually*. Loop detection stops it **as soon as it stops
learning** &mdash; the same tool, the same arguments, no new information. In production this is
usually the difference between a cheap failure and an expensive one.
"""),
    code('''
def is_looping(trace, window=3):
    """True when the last `window` tool calls are identical in both tool and arguments."""
    calls = [(tool, json.dumps(args, sort_keys=True)) for tool, args, _ in trace]
    if len(calls) < window:
        return False
    return ___                      # TODO: are the last `window` calls all the same call?
''', '''
def is_looping(trace, window=3):
    """True when the last `window` tool calls are identical in both tool and arguments."""
    calls = [(tool, json.dumps(args, sort_keys=True)) for tool, args, _ in trace]
    if len(calls) < window:
        return False
    return len(set(calls[-window:])) == 1        # one distinct call across the window
'''),
    code('''
# --- Self-check: Section 3
_same = [("peek", {"ref": "PMT-1002"}, "failed")] * 3
_mixed = [("peek", {"ref": "PMT-1002"}, "failed"),
          ("peek", {"ref": "PMT-1003"}, "held"),
          ("peek", {"ref": "PMT-1002"}, "failed")]

check("three identical calls count as a loop", lambda: is_looping(_same) is True)
check("varied calls are not a loop", lambda: is_looping(_mixed) is False,
      "different arguments mean the agent is still learning something")
check("too short a trace is not yet a loop", lambda: is_looping(_same[:2]) is False)
check("the window is honoured", lambda: is_looping(_same, window=2) is True)
'''),

    md("""
## Run it for real

Two calls, where the second depends on the first. Watch the model fail to recall &mdash; then watch
`carry()` fix it, by resending what it already told you.
"""),
    code('''
if llm_ready():
    print("--- without carry() -------------------------------------------")
    print("call 1:", ask("Remember this reference: PMT-1002. Reply with just OK."))
    print("call 2:", ask("Which payment reference did I just give you?"))

    print("\\n--- with carry() ----------------------------------------------")
    history = [("human", "Remember this reference: PMT-1002."), ("ai", "OK")]
    try:
        msgs = carry(history, "Which payment reference did I just give you?")
        print("call 2:", get_llm().invoke(msgs).content)
    except NameError:
        print("(fill in carry() above, then re-run this cell)")
    except Exception as exc:
        print(f"<model unavailable: {type(exc).__name__}: {exc}>")
'''),
    md("""
### Read it

The first pair shows the gap: call 2 has no access to call 1. The second pair shows the patch,
and the bill that comes with it &mdash; you resend the entire history on **every** turn. That is why
Module 3 spends its time on compaction rather than on bigger context windows.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add a third stop reason to `should_stop`: **no path forward** &mdash; the last observation was an
   error and the agent has no untried tool. Which of the four stop conditions from the slides
   does that leave unimplemented?
2. `is_looping` compares arguments exactly. An agent that asks for `PMT-1002` then ` PMT-1002 `
   would slip past it. Normalise the arguments and decide, in a sentence, whether being strict
   or being lenient is the safer default here.
"""),
]


# =========================================================================== #
# Lab 1.2 -- the four building blocks, assembled by hand
# =========================================================================== #
LAB2 = [
    header(2, "The Four Building Blocks", "Intermediate", 30,
           ["Write tools that report their failures instead of raising them",
            "Build short-term memory that compacts instead of growing without bound",
            "Turn a goal into dependency-ordered steps",
            "Assemble all four blocks into one small agent over the case file"],
           "> **Builds on Lab 1.1.** The loop you wrote there is the fourth block; here you build\n"
           "> the other three and wire them together."),
    setup(2),
    code(DOMAIN),

    md("""
## Concept

Each block patches one thing a model cannot do on its own:

| Block | The gap it closes |
|---|---|
| **LLM** | judgement under ambiguity |
| **Memory** | the call is stateless |
| **Tools** | the model cannot read or change anything |
| **Planning** | a goal is not a sequence of steps |

Miss one and you have a pipeline with a model in it &mdash; often the right build, but not an agent.
"""),

    md("""
## Section 1 &mdash; Tools that fail safely

A tool that raises aborts the run. A tool that **returns a description of the failure** hands the
model something it can reason about. Note the docstring: it names the case the tool is for *and*
the case it is not for &mdash; that text is the only thing the model reads when choosing.
"""),
    code('''
def lookup_payment(ref: str) -> str:
    """Return the ledger record for one payment reference such as 'PMT-1002'.

    Use when you need the status, amount, counterparty or reason code of a specific payment.
    Not for searching across payments.
    """
    record = LEDGER.get(ref)
    if record is None:
        return ___                  # TODO: a tool REPORTS failure, it does not raise it
    return json.dumps({"ref": ref, **record})


def policy_for(reason_code: str) -> str:
    """Return the operating policy for one failure reason code, e.g. 'LIMIT_BREACH'.

    Use after you know why a payment failed and need to know what to do about it.
    """
    return POLICY.get(reason_code, f"no policy on file for reason code {reason_code!r}")


TOOLS = {"lookup_payment": lookup_payment, "policy_for": policy_for}
''', '''
def lookup_payment(ref: str) -> str:
    """Return the ledger record for one payment reference such as 'PMT-1002'.

    Use when you need the status, amount, counterparty or reason code of a specific payment.
    Not for searching across payments.
    """
    record = LEDGER.get(ref)
    if record is None:
        return f"no payment found with reference {ref!r}"    # the agent can recover from this
    return json.dumps({"ref": ref, **record})


def policy_for(reason_code: str) -> str:
    """Return the operating policy for one failure reason code, e.g. 'LIMIT_BREACH'.

    Use after you know why a payment failed and need to know what to do about it.
    """
    return POLICY.get(reason_code, f"no policy on file for reason code {reason_code!r}")


TOOLS = {"lookup_payment": lookup_payment, "policy_for": policy_for}
'''),
    code('''
# --- Self-check: Section 1
def _no_raise(fn, *a):
    try:
        return fn(*a), None
    except Exception as exc:
        return None, exc

check("a known reference returns its status",
      lambda: "INSUFFICIENT_FUNDS" in lookup_payment("PMT-1002"))
check("an unknown reference does NOT raise",
      lambda: _no_raise(lookup_payment, "PMT-9999")[1] is None,
      "a raising tool aborts the whole agent run")
check("an unknown reference returns a readable string",
      lambda: isinstance(lookup_payment("PMT-9999"), str) and "PMT-9999" in lookup_payment("PMT-9999"))
check("every tool carries a docstring the model can choose from",
      lambda: all((f.__doc__ or "").strip() for f in TOOLS.values()))
'''),

    md("""
## Section 2 &mdash; Memory that compacts

Unbounded buffer memory is the classic failure: fine in the demo, degraded by week two. Keep the
recent turns verbatim, fold the rest into a summary, and the window stops growing.
"""),
    code('''
class ShortTermMemory:
    """Recent turns kept verbatim; older ones folded into a running summary."""

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.turns: list[tuple[str, str]] = []
        self.summary = ""

    def add(self, role: str, text: str) -> None:
        self.turns.append((role, text))
        if len(self.turns) > self.max_turns:
            self.compact()

    def compact(self) -> None:
        """Fold all but the most recent turns into `summary`."""
        keep = ___                   # TODO: how many recent turns stay verbatim? (half the window)
        older, self.turns = self.turns[:-keep], self.turns[-keep:]
        folded = " ".join(text for _, text in older)
        self.summary = (self.summary + " " + folded).strip()

    def render(self) -> list[tuple[str, str]]:
        """The message list to send: the summary first, then the verbatim turns."""
        head = [("system", "Earlier in this case: " + self.summary)] if self.summary else []
        return head + list(self.turns)
''', '''
class ShortTermMemory:
    """Recent turns kept verbatim; older ones folded into a running summary."""

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.turns: list[tuple[str, str]] = []
        self.summary = ""

    def add(self, role: str, text: str) -> None:
        self.turns.append((role, text))
        if len(self.turns) > self.max_turns:
            self.compact()

    def compact(self) -> None:
        """Fold all but the most recent turns into `summary`."""
        keep = max(1, self.max_turns // 2)      # keep the recent half, summarise the rest
        older, self.turns = self.turns[:-keep], self.turns[-keep:]
        folded = " ".join(text for _, text in older)
        self.summary = (self.summary + " " + folded).strip()

    def render(self) -> list[tuple[str, str]]:
        """The message list to send: the summary first, then the verbatim turns."""
        head = [("system", "Earlier in this case: " + self.summary)] if self.summary else []
        return head + list(self.turns)
'''),
    code('''
# --- Self-check: Section 2   (fixtures are built lazily so an unfilled blank cannot crash the cell)
def _short():
    m = ShortTermMemory(max_turns=6)
    for t in ("a", "b", "c"):
        m.add("human", t)
    return m

def _long():
    m = ShortTermMemory(max_turns=6)
    m.add("human", "Investigate PMT-1002.")
    for i in range(20):
        m.add("ai", f"step {i}")
    return m

check("the window stays bounded after 21 turns",
      lambda: len(_long().turns) <= 6,
      "compact() must actually drop turns from self.turns")
check("compaction keeps the earliest instruction somewhere",
      lambda: "PMT-1002" in _long().summary,
      "the oldest turns should be folded into the summary, not discarded")
check("render() puts the summary first",
      lambda: _long().render()[0][0] == "system")
check("a short conversation is left untouched",
      lambda: len(_short().turns) == 3 and _short().summary == "",
      "compact() should only fire once the window is exceeded")
'''),

    md("""
## Section 3 &mdash; Planning: a goal is not a sequence

Decomposition is only half of it. The steps have **dependencies**, and running them out of order
is one of the quieter ways an agent wastes a budget.
"""),
    code('''
def order_steps(steps: dict[str, list[str]]) -> list[str]:
    """steps maps a step name to the steps it depends on. Return a runnable order.

    Raises ValueError if the dependencies cannot be satisfied (a cycle, or a missing step).
    """
    ordered: list[str] = []
    done: set[str] = set()
    while len(ordered) < len(steps):
        progressed = False
        for name, deps in steps.items():
            if name in done:
                continue
            if ___:                  # TODO: may this step run yet?
                ordered.append(name)
                done.add(name)
                progressed = True
        if not progressed:
            raise ValueError("cycle or missing dependency in plan")
    return ordered
''', '''
def order_steps(steps: dict[str, list[str]]) -> list[str]:
    """steps maps a step name to the steps it depends on. Return a runnable order.

    Raises ValueError if the dependencies cannot be satisfied (a cycle, or a missing step).
    """
    ordered: list[str] = []
    done: set[str] = set()
    while len(ordered) < len(steps):
        progressed = False
        for name, deps in steps.items():
            if name in done:
                continue
            if all(d in done for d in deps):     # every dependency already ordered
                ordered.append(name)
                done.add(name)
                progressed = True
        if not progressed:
            raise ValueError("cycle or missing dependency in plan")
    return ordered
'''),
    code('''
# --- Self-check: Section 3
PLAN = {
    "read_payment":  [],
    "read_policy":   ["read_payment"],       # you cannot look up a policy before you know the code
    "decide_action": ["read_payment", "read_policy"],
    "write_note":    ["decide_action"],
}

def _cycles():
    try:
        order_steps({"a": ["b"], "b": ["a"]})
        return False
    except ValueError:
        return True

check("every step follows its dependencies",
      lambda: (lambda o: all(o.index(d) < o.index(n) for n, ds in PLAN.items() for d in ds))(order_steps(PLAN)))
check("the plan starts with the only step that has no dependency",
      lambda: order_steps(PLAN)[0] == "read_payment")
check("all four steps are present exactly once",
      lambda: sorted(order_steps(PLAN)) == sorted(PLAN))
check("a cyclic plan is rejected", lambda: _cycles(),
      "order_steps must raise ValueError rather than loop forever")
'''),

    md("""
## Section 4 &mdash; Assemble the four blocks

Now put them together. Nothing clever &mdash; the point is that you can name which block every line
belongs to.
"""),
    code('''
class MiniAgent:
    """model + memory + tools + planning, and the loop that binds them."""

    def __init__(self, tools, memory, plan):
        self.tools = tools                # block 3: the hands
        self.memory = memory              # block 2: the state
        self.plan = ___                   # TODO: block 4 -- store the dependency-ordered plan
        self.max_steps = 6

    def blocks(self) -> list[str]:
        return ["llm", "memory", "tools", "planning"]

    def run(self, ref: str) -> dict:
        """Walk the plan over one payment. Deterministic -- the model comes in below."""
        self.memory.add("human", f"Investigate {ref}.")
        facts = {}
        for step in self.plan:
            if step == "read_payment":
                facts["payment"] = self.tools["lookup_payment"](ref)
            elif step == "read_policy":
                code_ = json.loads(facts["payment"]).get("reason_code") if facts["payment"].startswith("{") else None
                facts["policy"] = self.tools["policy_for"](code_) if code_ else "no reason code"
            elif step == "decide_action":
                facts["needs_human"] = (json.loads(facts["payment"]).get("reason_code") in NEEDS_HUMAN
                                        if facts["payment"].startswith("{") else False)
            elif step == "write_note":
                self.memory.add("ai", f"{ref}: {facts.get('policy', '')}")
        return facts
''', '''
class MiniAgent:
    """model + memory + tools + planning, and the loop that binds them."""

    def __init__(self, tools, memory, plan):
        self.tools = tools                # block 3: the hands
        self.memory = memory              # block 2: the state
        self.plan = order_steps(plan)     # block 4: the strategy, in a runnable order
        self.max_steps = 6

    def blocks(self) -> list[str]:
        return ["llm", "memory", "tools", "planning"]

    def run(self, ref: str) -> dict:
        """Walk the plan over one payment. Deterministic -- the model comes in below."""
        self.memory.add("human", f"Investigate {ref}.")
        facts = {}
        for step in self.plan:
            if step == "read_payment":
                facts["payment"] = self.tools["lookup_payment"](ref)
            elif step == "read_policy":
                code_ = json.loads(facts["payment"]).get("reason_code") if facts["payment"].startswith("{") else None
                facts["policy"] = self.tools["policy_for"](code_) if code_ else "no reason code"
            elif step == "decide_action":
                facts["needs_human"] = (json.loads(facts["payment"]).get("reason_code") in NEEDS_HUMAN
                                        if facts["payment"].startswith("{") else False)
            elif step == "write_note":
                self.memory.add("ai", f"{ref}: {facts.get('policy', '')}")
        return facts
'''),
    code('''
# --- Self-check: Section 4
def _agent():
    return MiniAgent(TOOLS, ShortTermMemory(6), PLAN)

def _out():
    return _agent().run("PMT-1003")

check("the plan was stored in dependency order",
      lambda: _agent().plan[0] == "read_payment",
      "pass the plan through order_steps() in __init__")
check("the agent found the payment", lambda: "LIMIT_BREACH" in _out()["payment"])
check("the agent found the matching policy", lambda: "Treasury" in _out()["policy"])
check("a limit breach is flagged for a human", lambda: _out()["needs_human"] is True)
check("an unknown payment degrades without raising",
      lambda: "no payment found" in MiniAgent(TOOLS, ShortTermMemory(6), PLAN).run("PMT-0000")["payment"])
'''),

    md("""
## Run it for real

Everything above is deterministic. Now let the model do the one part it is actually for &mdash;
turning the facts your tools gathered into a judgement an operator can read.
"""),
    code('''
if llm_ready():
    try:
        facts = MiniAgent(TOOLS, ShortTermMemory(6), PLAN).run("PMT-1003")
        verdict = ask(
            "You are a payments operations analyst. Using ONLY the facts below, state in two "
            "sentences what happened and what should be done next. If the policy requires a human, "
            "say so explicitly and do not propose acting yourself.\\n\\n"
            f"PAYMENT: {facts['payment']}\\nPOLICY: {facts['policy']}"
        )
        print(verdict)
    except NameError:
        print("(fill in the blanks above, then re-run this cell)")
'''),
    md("""
### Read it

The model never touched the ledger and never chose a policy &mdash; your tools did, deterministically.
It only phrased the judgement. That division is the whole point of the four blocks: put the
verifiable work in code, and leave the model the part that genuinely needs it.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `ShortTermMemory.compact()` concatenates old turns. Replace it with a model-written summary
   (use `ask()`). What did you gain, and what did it cost you in tokens and determinism?
2. `MiniAgent.run` hardcodes the branch per step. Rewrite it as a dict of step handlers. Does
   that make the plan easier to extend, or just harder to read? Argue either way.
"""),
]


# the two tools you wrote in Lab 1.2, carried forward so this notebook stands alone
CARRIED_TOOLS = '''
# ------------------------------------------------- carried forward from Lab 1.2
# These are the tools you wrote in Lab 1.2. Nothing to fill in -- they are here so this
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
# Lab 1.3 -- the LangChain 1.x on-ramp: create_agent
# =========================================================================== #
LAB3 = [
    header(3, "The create_agent On-Ramp", "Intermediate &rarr; Advanced", 25,
           ["Judge a tool description the way the model does -- and write one that passes",
            "Assemble a create_agent configuration and run it against the case file",
            "Compare the framework agent with your hand-rolled loop on the same briefs"],
           "> **Builds on Labs 1.1 and 1.2.** Same tools, same case file. What changes is who owns\n"
           "> the loop: you, or `create_agent`."),
    setup(3),
    code(DOMAIN),
    code(CARRIED_TOOLS),

    md("""
## Concept

`create_agent(model=..., tools=..., prompt=...)` **is** the loop you wrote in Lab 1.1, prebuilt.
Nothing is hidden: the model decides, a tool runs, the result comes back, repeat until it stops.

> **Naming.** In LangChain 1.x it is `create_agent`. The older `create_react_agent` name is
> everywhere online and is **not** what this course uses.
"""),

    md("""
## Section 1 &mdash; A tool description is an instruction

The model picks a tool using its description and nothing else. Rather than take that on faith,
write the checker: what must a description contain to be choosable?
"""),
    code('''
def describes_well(doc: str) -> bool:
    """True when a tool docstring gives the model enough to choose correctly.

    A usable description does three things:
      1. says what the tool returns,
      2. names the situation it is FOR ("use when ..."),
      3. names at least one situation it is NOT for -- the boundary.
    """
    if not doc:
        return False
    text = doc.lower()
    says_return = "return" in text
    says_when = ___                  # TODO: does it name the situation it is for?
    says_boundary = ___              # TODO: does it name a situation it is NOT for?
    return says_return and says_when and says_boundary
''', '''
def describes_well(doc: str) -> bool:
    """True when a tool docstring gives the model enough to choose correctly.

    A usable description does three things:
      1. says what the tool returns,
      2. names the situation it is FOR ("use when ..."),
      3. names at least one situation it is NOT for -- the boundary.
    """
    if not doc:
        return False
    text = doc.lower()
    says_return = "return" in text
    says_when = "use when" in text or "use after" in text
    says_boundary = "not for" in text or "never" in text or "do not" in text
    return says_return and says_when and says_boundary
'''),
    code('''
# --- Self-check: Section 1
_weak = "Gets data."
_strong = ("Return the ledger record for one payment reference. Use when you need the status of a "
           "specific payment. Not for searching across payments.")

check("a vague description is rejected", lambda: describes_well(_weak) is False)
check("a specific description is accepted", lambda: describes_well(_strong) is True,
      "look for a 'use when' phrase and a 'not for' boundary")
check("an empty description is rejected", lambda: describes_well("") is False)
check("a description with no boundary is rejected",
      lambda: describes_well("Return the record. Use when you need a payment.") is False,
      "the boundary is what stops the model reaching for the wrong tool")
'''),

    md("""
## Section 2 &mdash; Rewrite the weak tool

`get_data` below is the kind of tool that quietly ruins an agent. Rewrite its docstring so it
passes your own checker &mdash; and note that you are not changing a single line of logic.
"""),
    code('''
def get_data(ref: str) -> str:
    """___"""                        # TODO: rewrite so describes_well() passes. Logic stays as-is.
    record = LEDGER.get(ref)
    if record is None:
        return f"no payment found with reference {ref!r}"
    return json.dumps({"ref": ref, **record})
''', '''
def get_data(ref: str) -> str:
    """Return the ledger record for one payment reference such as 'PMT-1002'.

    Use when you need the status, amount, counterparty or reason code of a specific payment.
    Not for searching across payments, and not for policy questions.
    """
    record = LEDGER.get(ref)
    if record is None:
        return f"no payment found with reference {ref!r}"
    return json.dumps({"ref": ref, **record})
'''),
    code('''
# --- Self-check: Section 2
check("the rewritten docstring passes your checker",
      lambda: describes_well(get_data.__doc__) is True,
      "it needs a return clause, a 'use when', and a boundary")
check("the docstring names the concrete input format",
      lambda: "PMT-" in (get_data.__doc__ or ""),
      "an example reference removes a whole class of malformed calls")
check("the behaviour is unchanged", lambda: "INSUFFICIENT_FUNDS" in get_data("PMT-1002"))
'''),

    md("""
## Section 3 &mdash; The agent configuration

`create_agent` takes four things. Three of them map straight onto the blocks from Lab 1.2.
Assemble the configuration as plain data first, so the shape is checkable before anything runs.
"""),
    code('''
def build_config() -> dict:
    """Assemble the create_agent configuration as plain data."""
    system_prompt = ___              # TODO: a standing instruction. It must (a) give the agent its
                                     # role, and (b) tell it never to act on a payment that policy
                                     # reserves for a human. Mention "human" explicitly.
    return {
        "model": LLM_MODEL,          # block 1: the brain
        "tools": ___,                # TODO: a list of the two tool FUNCTIONS carried forward above
        "prompt": system_prompt,     # the standing instruction
        "max_steps": 6,              # the budget from Lab 1.1
    }
''', '''
def build_config() -> dict:
    """Assemble the create_agent configuration as plain data."""
    system_prompt = (
        "You are a payments operations analyst. Investigate one payment exception at a time using "
        "the tools provided. Never propose releasing, cancelling or repairing a payment whose "
        "policy reserves the decision for a human -- say that a human must decide instead."
    )
    return {
        "model": LLM_MODEL,          # block 1: the brain
        "tools": [lookup_payment, policy_for],
        "prompt": system_prompt,     # the standing instruction
        "max_steps": 6,              # the budget from Lab 1.1
    }
'''),
    code('''
# --- Self-check: Section 3   (structure only -- no model call)
check("the config carries exactly the two tools",
      lambda: len(build_config()["tools"]) == 2)
check("both tools are callables with docstrings",
      lambda: all(callable(t) and (t.__doc__ or "").strip() for t in build_config()["tools"]))
check("the prompt gives the agent a role",
      lambda: len(build_config()["prompt"]) > 40)
check("the prompt defers irreversible decisions to a human",
      lambda: "human" in build_config()["prompt"].lower(),
      "an approval boundary belongs in the standing instruction, not in each request")
check("the budget survived from Lab 1.1", lambda: build_config()["max_steps"] == 6)
'''),

    md("""
## Run it for real

Now hand your configuration to `create_agent`. The `@tool` decorator turns a plain function into
something the model can call &mdash; it reads the signature and the docstring you just wrote.
"""),
    code('''
if llm_ready():
    try:
        from langchain.tools import tool
        from langchain.agents import create_agent

        cfg = build_config()
        tools = [tool(f) for f in cfg["tools"]]
        agent = create_agent(model=get_llm(), tools=tools, prompt=cfg["prompt"])

        result = agent.invoke({"messages": [("human", "Why is PMT-1005 held, and what do we do?")]})
        for m in result["messages"]:
            kind = getattr(m, "type", "?")
            body = str(getattr(m, "content", ""))[:300]
            calls = getattr(m, "tool_calls", None)
            print(f"[{kind}] {body}" + (f"  -> calls: {[c['name'] for c in calls]}" if calls else ""))
    except ImportError as exc:
        print(f"LangChain not importable here ({exc}). The graded cells above do not need it.")
    except NameError:
        print("(fill in the blanks above, then re-run this cell)")
    except Exception as exc:
        print(f"<agent run failed: {type(exc).__name__}: {exc}>")
'''),
    md("""
### Read the trace

Count the messages. Every `[ai]` with `tool_calls` is one turn of the loop you wrote by hand in
Lab 1.1; every `[tool]` is the observation coming back. `create_agent` saved you the plumbing and
nothing else &mdash; which is exactly why it stops being enough the moment you need to *see* or
*steer* that state. That is Module 3.

Check the last message: with `SANCTIONS_REVIEW` the agent should defer to a human. If it proposed
an action instead, your standing instruction was not firm enough &mdash; and no amount of model
capability fixes an instruction that never said it.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Degrade `get_data`'s docstring back to `"Gets data."`, re-run the live cell, and count the
   tool calls. That difference is the entire content of the Day 2 tool-description lab.
2. Add a third tool that *overlaps* with `lookup_payment` (say `get_payment_status`). Which
   description does the model prefer, and what does that tell you about writing boundaries?
"""),
]


# =========================================================================== #
# Lab 1.4 -- measuring the coordination tax
# =========================================================================== #
LAB4 = [
    header(4, "One Agent or Three: Measuring the Coordination Tax", "Advanced", 35,
           ["Build a supervisor that routes work to specialist workers",
            "Instrument both designs -- steps, estimated tokens, wall time",
            "Score single-agent against multi-agent on one eval set",
            "Count the failure surface, and see why it does not grow linearly"],
           "> **Builds on Lab 1.3.** The claim 'we need multiple agents' is a hypothesis.\n"
           "> This lab is how you test it. Workers here are deterministic stubs, so the numbers\n"
           "> are reproducible and the comparison is about *architecture*, not model variance."),
    setup(4),
    code(DOMAIN),
    code(CARRIED_TOOLS),

    md("""
## Concept

Every agent you add costs 3&ndash;10&times; the tokens, adds latency, and opens a new class of failure.
Two things repay that tax, and only two: **specialisation** (the sub-tasks genuinely need
different tools) and **parallelism** (they genuinely run at the same time).

Anything else is tax paid for nothing. This lab produces the numbers that settle the argument.
"""),

    md("""
## Section 1 &mdash; The supervisor's routing rule

The supervisor does one job: decide which worker gets the brief. Rule-based here, so the routing
is inspectable; Module 5 replaces the rule with a model.
"""),
    code('''
WORKERS = ("ledger", "policy", "critic")

def route(brief: dict) -> str:
    """Pick the worker for one brief. Returns a name from WORKERS.

    brief looks like {"ref": "PMT-1002", "needs": "status" | "policy" | "review"}
    """
    need = brief.get("needs")
    if need == "status":
        return ___                   # TODO: which worker reads the ledger?
    if need == "policy":
        return "policy"
    if need == "review":
        return "critic"
    return ___                       # TODO: an unknown need must still route somewhere sensible
''', '''
WORKERS = ("ledger", "policy", "critic")

def route(brief: dict) -> str:
    """Pick the worker for one brief. Returns a name from WORKERS.

    brief looks like {"ref": "PMT-1002", "needs": "status" | "policy" | "review"}
    """
    need = brief.get("needs")
    if need == "status":
        return "ledger"
    if need == "policy":
        return "policy"
    if need == "review":
        return "critic"
    return "ledger"                  # unknown need: start from the facts, never guess
'''),
    code('''
# --- Self-check: Section 1
check("a status brief goes to the ledger worker",
      lambda: route({"ref": "PMT-1002", "needs": "status"}) == "ledger")
check("a policy brief goes to the policy worker",
      lambda: route({"ref": "PMT-1002", "needs": "policy"}) == "policy")
check("a review brief goes to the critic",
      lambda: route({"ref": "PMT-1002", "needs": "review"}) == "critic")
check("an unknown need still routes to a real worker",
      lambda: route({"ref": "PMT-1002", "needs": "???"}) in WORKERS,
      "returning None would strand the brief")
'''),

    md("""
## Section 2 &mdash; The meter

You cannot argue about the coordination tax without measuring it. A rough token estimate is fine
&mdash; what matters is that both designs are measured **the same way**.
"""),
    code('''
class Meter:
    """Counts what an architecture costs: steps, estimated tokens, wall time."""

    def __init__(self):
        self.steps = 0
        self.tokens = 0
        self.t0 = time.perf_counter()

    def record(self, prompt: str, response: str) -> None:
        """One model turn. Estimate ~4 characters per token -- crude, but applied equally."""
        self.steps += 1
        self.tokens += ___           # TODO: estimated tokens for this turn (prompt AND response)

    @property
    def seconds(self) -> float:
        return time.perf_counter() - self.t0

    def report(self) -> dict:
        return {"steps": self.steps, "tokens": self.tokens, "seconds": round(self.seconds, 4)}
''', '''
class Meter:
    """Counts what an architecture costs: steps, estimated tokens, wall time."""

    def __init__(self):
        self.steps = 0
        self.tokens = 0
        self.t0 = time.perf_counter()

    def record(self, prompt: str, response: str) -> None:
        """One model turn. Estimate ~4 characters per token -- crude, but applied equally."""
        self.steps += 1
        self.tokens += (len(prompt) + len(response)) // 4

    @property
    def seconds(self) -> float:
        return time.perf_counter() - self.t0

    def report(self) -> dict:
        return {"steps": self.steps, "tokens": self.tokens, "seconds": round(self.seconds, 4)}
'''),
    code('''
# --- Self-check: Section 2
def _metered():
    m = Meter()
    m.record("a" * 400, "b" * 400)
    return m

check("a turn advances the step count", lambda: _metered().report()["steps"] == 1)
check("both prompt and response are counted", lambda: _metered().report()["tokens"] == 200,
      "800 characters at ~4 chars/token is 200 -- count the prompt as well as the response")
check("two turns accumulate",
      lambda: (lambda m: (m.record("x" * 40, "y" * 40), m.report()["steps"])[1])(_metered()) == 2)
check("wall time is reported", lambda: _metered().report()["seconds"] >= 0)
'''),

    md("""
## Section 3 &mdash; Two architectures, one eval set

The single agent handles a brief in one pass. The supervisor decomposes it, dispatches to
workers, and aggregates. Both answer the same six briefs, and both are metered identically.
"""),
    code('''
EVAL_SET = [
    {"ref": "PMT-1002", "question": "why did it fail?",        "expect": "INSUFFICIENT_FUNDS"},
    {"ref": "PMT-1003", "question": "can we release it?",      "expect": "Treasury"},
    {"ref": "PMT-1004", "question": "what do we do?",          "expect": "R04"},
    {"ref": "PMT-1005", "question": "who decides?",            "expect": "Compliance"},
    {"ref": "PMT-1001", "question": "is there an exception?",  "expect": "settled"},
    {"ref": "PMT-9999", "question": "why did it fail?",        "expect": "no payment found"},
]

def _worker(name, ref):
    """A deterministic stand-in for a specialist agent."""
    if name == "ledger":
        return lookup_payment(ref)
    if name == "policy":
        rec = LEDGER.get(ref)
        return policy_for(rec["reason_code"]) if rec and rec["reason_code"] else "no reason code"
    return "reviewed: findings are consistent with the ledger record"

def run_single(case, meter):
    """One agent, one pass: it holds every tool itself."""
    facts = lookup_payment(case["ref"])
    rec = LEDGER.get(case["ref"])
    policy = policy_for(rec["reason_code"]) if rec and rec["reason_code"] else ""
    answer = f"{facts} {policy}"
    meter.record(case["question"] + facts, answer)
    return answer

def run_supervised(case, meter):
    """Supervisor + workers: each hop re-reads the context it was handed."""
    answer = ""
    for need in ("status", "policy", "review"):
        worker = route({"ref": case["ref"], "needs": need})
        out = _worker(worker, case["ref"])
        meter.record(case["question"] + answer, out)      # the handoff re-sends what came before
        answer = (answer + " " + out).strip()
    return answer

def pass_rate(runner):
    """Fraction of EVAL_SET whose expected string appears in the answer, plus the meter report."""
    meter = Meter()
    hits = 0
    for case in EVAL_SET:
        answer = runner(case, meter)
        if ___:                      # TODO: did this case pass?
            hits += 1
    return {"pass_rate": round(hits / len(EVAL_SET), 3), **meter.report()}
''', '''
EVAL_SET = [
    {"ref": "PMT-1002", "question": "why did it fail?",        "expect": "INSUFFICIENT_FUNDS"},
    {"ref": "PMT-1003", "question": "can we release it?",      "expect": "Treasury"},
    {"ref": "PMT-1004", "question": "what do we do?",          "expect": "R04"},
    {"ref": "PMT-1005", "question": "who decides?",            "expect": "Compliance"},
    {"ref": "PMT-1001", "question": "is there an exception?",  "expect": "settled"},
    {"ref": "PMT-9999", "question": "why did it fail?",        "expect": "no payment found"},
]

def _worker(name, ref):
    """A deterministic stand-in for a specialist agent."""
    if name == "ledger":
        return lookup_payment(ref)
    if name == "policy":
        rec = LEDGER.get(ref)
        return policy_for(rec["reason_code"]) if rec and rec["reason_code"] else "no reason code"
    return "reviewed: findings are consistent with the ledger record"

def run_single(case, meter):
    """One agent, one pass: it holds every tool itself."""
    facts = lookup_payment(case["ref"])
    rec = LEDGER.get(case["ref"])
    policy = policy_for(rec["reason_code"]) if rec and rec["reason_code"] else ""
    answer = f"{facts} {policy}"
    meter.record(case["question"] + facts, answer)
    return answer

def run_supervised(case, meter):
    """Supervisor + workers: each hop re-reads the context it was handed."""
    answer = ""
    for need in ("status", "policy", "review"):
        worker = route({"ref": case["ref"], "needs": need})
        out = _worker(worker, case["ref"])
        meter.record(case["question"] + answer, out)      # the handoff re-sends what came before
        answer = (answer + " " + out).strip()
    return answer

def pass_rate(runner):
    """Fraction of EVAL_SET whose expected string appears in the answer, plus the meter report."""
    meter = Meter()
    hits = 0
    for case in EVAL_SET:
        answer = runner(case, meter)
        if case["expect"].lower() in answer.lower():
            hits += 1
    return {"pass_rate": round(hits / len(EVAL_SET), 3), **meter.report()}
'''),
    code('''
# --- Self-check: Section 3
_ZERO = {"pass_rate": 0.0, "steps": 0, "tokens": 0, "seconds": 0.0}
_single = guard(lambda: pass_rate(run_single), _ZERO)
_multi = guard(lambda: pass_rate(run_supervised), _ZERO)
print("single    :", _single)
print("supervised:", _multi)
print(f"\\ntoken ratio: {_multi['tokens'] / max(_single['tokens'], 1):.1f}x   "
      f"step ratio: {_multi['steps'] / max(_single['steps'], 1):.1f}x")

check("both architectures answer every case",
      lambda: _single["steps"] == len(EVAL_SET) and _multi["steps"] == len(EVAL_SET) * 3)
check("the single agent scores on the eval set", lambda: _single["pass_rate"] >= 0.8,
      "check the pass condition -- the expected string should appear in the answer")
check("the supervised design costs strictly more tokens",
      lambda: _multi["tokens"] > _single["tokens"],
      "three hops re-read the context; that is the tax")
check("the supervised design takes three times the steps",
      lambda: _multi["steps"] == _single["steps"] * 3)
'''),

    md("""
## Section 4 &mdash; The failure surface

Tokens and latency grow roughly **linearly** with the agent count. The number of handoff paths
does not &mdash; and that is what you debug at 02:00.
"""),
    code('''
def handoff_paths(n_agents: int) -> int:
    """Ordered handoff paths between n agents: every agent may hand to every other."""
    return ___                       # TODO: the formula (each edge runs in both directions)
''', '''
def handoff_paths(n_agents: int) -> int:
    """Ordered handoff paths between n agents: every agent may hand to every other."""
    return n_agents * (n_agents - 1)
'''),
    code('''
# --- Self-check: Section 4
check("two agents give two paths", lambda: handoff_paths(2) == 2)
check("three agents give six", lambda: handoff_paths(3) == 6)
check("five agents give twenty", lambda: handoff_paths(5) == 20,
      "n(n-1) -- not n(n-1)/2, because a handoff has a direction")
check("one agent has nothing to hand off to", lambda: handoff_paths(1) == 0)

for n in (1, 2, 3, 4, 5):
    try:
        print(f"{n} agents -> {handoff_paths(n):>2} handoff paths, ~{n}x tokens")
    except NameError:
        print("(fill in handoff_paths above)")
        break
'''),

    md("""
### Read the numbers

You now have the argument in a form nobody can wave away: the supervised design costs measurably
more for this eval set and scores no better, because the work needs neither different tools nor
parallelism &mdash; the stubs read the same ledger. That is the coordination tax paid for nothing.

Note what would change the verdict: give the workers genuinely different tools, or run them
concurrently, and the ratios move. Module 5 does exactly that.
"""),
    code('''
score()
'''),
    md("""
## Your turn

1. Make the policy worker slow (`time.sleep(0.05)`) and run the three workers concurrently with
   `concurrent.futures.ThreadPoolExecutor`. At what worker latency does parallelism start to pay
   for the extra tokens?
2. Add a seventh eval case the single agent gets **wrong** and the supervisor gets right. What
   property does that case need? If you cannot construct one, that is itself the finding.
"""),
]


# =========================================================================== #
# Lab 1.5 -- challenge: the decision rubric, made executable
# =========================================================================== #
LAB5 = [
    header(5, "Challenge &mdash; The Decision Rubric, Made Executable", "Advanced", 40,
           ["Encode the four-question rubric as code that returns a defensible verdict",
            "Run it over six real-shaped briefs, including two designed to mislead",
            "Gate the verdict on evidence: a budget named BEFORE you measured",
            "Produce a recommendation you could defend in a design review"],
           "> **The comprehensive lab for Module 1.** It uses the rubric from the slides, the\n"
           "> scorecard from Lab 1.4, and the honesty that the two together are supposed to enforce."),
    setup(5),
    code(DOMAIN),

    md("""
## Concept

The rubric picks a **candidate**, not a winner. A candidate becomes a decision only when the
scorecard says it earned its cost &mdash; against a threshold you wrote down **before** you measured,
because a number chosen afterwards can always be argued into looking acceptable.

This lab makes both halves executable.
"""),

    md("""
## Section 1 &mdash; The rubric, as code

Four questions, in order. The order matters: most real briefs terminate at the first or second.
"""),
    code('''
VERDICTS = ("workflow", "single_agent", "supervisor_worker", "peer_to_peer")

def rubric(brief: dict) -> str:
    """Return one of VERDICTS for a brief.

    brief keys:
      steps_vary       -- do the steps change with the input?
      fits_one_agent   -- can one agent hold every tool it needs? (roughly a dozen)
      different_tools  -- do the sub-tasks need genuinely different tools/permissions?
      parallel         -- can the sub-tasks genuinely run at the same time?
      route_varies     -- does the path through the work change per case?
    """
    if ___:                          # TODO: question 1 -- are the steps the same every time?
        return "workflow"
    if brief["fits_one_agent"]:
        return "single_agent"
    if ___:                          # TODO: question 3 -- neither specialisation nor parallelism?
        return "single_agent"        #        then splitting buys nothing measurable
    return ___                       # TODO: question 4 -- varying route or fixed?
''', '''
VERDICTS = ("workflow", "single_agent", "supervisor_worker", "peer_to_peer")

def rubric(brief: dict) -> str:
    """Return one of VERDICTS for a brief.

    brief keys:
      steps_vary       -- do the steps change with the input?
      fits_one_agent   -- can one agent hold every tool it needs? (roughly a dozen)
      different_tools  -- do the sub-tasks need genuinely different tools/permissions?
      parallel         -- can the sub-tasks genuinely run at the same time?
      route_varies     -- does the path through the work change per case?
    """
    if not brief["steps_vary"]:
        return "workflow"
    if brief["fits_one_agent"]:
        return "single_agent"
    if not (brief["different_tools"] or brief["parallel"]):
        return "single_agent"        # splitting buys nothing measurable
    return "peer_to_peer" if brief["route_varies"] else "supervisor_worker"
'''),
    code('''
# --- Self-check: Section 1
BRIEFS = [
    {"name": "nightly ledger reconciliation", "steps_vary": False, "fits_one_agent": True,
     "different_tools": False, "parallel": False, "route_varies": False},
    {"name": "policy Q&A with citations", "steps_vary": True, "fits_one_agent": True,
     "different_tools": False, "parallel": False, "route_varies": False},
    # designed to mislead: sounds big, but splitting buys nothing
    {"name": "three-stage summary pipeline", "steps_vary": True, "fits_one_agent": False,
     "different_tools": False, "parallel": False, "route_varies": False},
    {"name": "payment exception investigation", "steps_vary": True, "fits_one_agent": False,
     "different_tools": True, "parallel": True, "route_varies": False},
    {"name": "open-ended client complaint triage", "steps_vary": True, "fits_one_agent": False,
     "different_tools": True, "parallel": False, "route_varies": True},
    # designed to mislead: varying route, but it still fits in one agent
    {"name": "ad-hoc data question over one warehouse", "steps_vary": True, "fits_one_agent": True,
     "different_tools": False, "parallel": False, "route_varies": True},
]

for b in BRIEFS:
    try:
        print(f"{b['name']:38} -> {rubric(b)}")
    except NameError:
        print("(fill in rubric() above)")
        break

check("fixed steps give a workflow", lambda: rubric(BRIEFS[0]) == "workflow")
check("one skill, one source gives a single agent", lambda: rubric(BRIEFS[1]) == "single_agent")
check("splitting without specialisation or parallelism falls back to one agent",
      lambda: rubric(BRIEFS[2]) == "single_agent",
      "question 3 is the one that catches this brief")
check("different tools + parallel + fixed route gives supervisor/worker",
      lambda: rubric(BRIEFS[3]) == "supervisor_worker")
check("a varying route gives peer to peer", lambda: rubric(BRIEFS[4]) == "peer_to_peer")
check("a varying route that still fits one agent stays a single agent",
      lambda: rubric(BRIEFS[5]) == "single_agent",
      "question 2 comes before question 4 for a reason")
'''),

    md("""
## Section 2 &mdash; The budget, written down first

Name the thresholds now, while you have no result to defend. A multi-agent design must beat the
single agent by at least `min_gain` **and** stay inside the cost and latency ceilings.
"""),
    code('''
BUDGET = {
    "min_gain": 0.10,        # pass-rate points the new design must add, as a fraction
    "max_token_ratio": 3.0,  # at most 3x the single agent's tokens
    "max_latency_ratio": 2.0
}

def clears_budget(single: dict, multi: dict, budget: dict = BUDGET) -> tuple[bool, str]:
    """Return (cleared, reason). Every clause must hold for the multi-agent design to win."""
    gain = multi["pass_rate"] - single["pass_rate"]
    token_ratio = multi["tokens"] / max(single["tokens"], 1)
    latency_ratio = multi["seconds"] / max(single["seconds"], 1e-9)

    if gain < budget["min_gain"]:
        return False, f"gain {gain:+.2f} is below the {budget['min_gain']:.2f} threshold"
    if ___:                          # TODO: is it over the token ceiling?
        return False, f"tokens {token_ratio:.1f}x exceed {budget['max_token_ratio']}x"
    if latency_ratio > budget["max_latency_ratio"]:
        return False, f"latency {latency_ratio:.1f}x exceeds {budget['max_latency_ratio']}x"
    return True, f"gain {gain:+.2f} within {token_ratio:.1f}x tokens"
''', '''
BUDGET = {
    "min_gain": 0.10,        # pass-rate points the new design must add, as a fraction
    "max_token_ratio": 3.0,  # at most 3x the single agent's tokens
    "max_latency_ratio": 2.0
}

def clears_budget(single: dict, multi: dict, budget: dict = BUDGET) -> tuple[bool, str]:
    """Return (cleared, reason). Every clause must hold for the multi-agent design to win."""
    gain = multi["pass_rate"] - single["pass_rate"]
    token_ratio = multi["tokens"] / max(single["tokens"], 1)
    latency_ratio = multi["seconds"] / max(single["seconds"], 1e-9)

    if gain < budget["min_gain"]:
        return False, f"gain {gain:+.2f} is below the {budget['min_gain']:.2f} threshold"
    if token_ratio > budget["max_token_ratio"]:
        return False, f"tokens {token_ratio:.1f}x exceed {budget['max_token_ratio']}x"
    if latency_ratio > budget["max_latency_ratio"]:
        return False, f"latency {latency_ratio:.1f}x exceeds {budget['max_latency_ratio']}x"
    return True, f"gain {gain:+.2f} within {token_ratio:.1f}x tokens"
'''),
    code('''
# --- Self-check: Section 2
_base = {"pass_rate": 0.80, "tokens": 1000, "seconds": 1.0}
_worse_cost = {"pass_rate": 0.95, "tokens": 9000, "seconds": 1.5}   # big gain, absurd cost
_no_gain    = {"pass_rate": 0.82, "tokens": 1500, "seconds": 1.2}   # cheap, but no real gain
_good       = {"pass_rate": 0.95, "tokens": 2500, "seconds": 1.5}

check("a design that gains little is rejected", lambda: clears_budget(_base, _no_gain)[0] is False)
check("a design that costs too many tokens is rejected",
      lambda: clears_budget(_base, _worse_cost)[0] is False,
      "a 15-point gain does not license a 9x bill -- that is what the ceiling is for")
check("a design that clears every clause wins", lambda: clears_budget(_base, _good)[0] is True)
check("the rejection always carries a reason",
      lambda: len(clears_budget(_base, _no_gain)[1]) > 10)
'''),

    md("""
## Section 3 &mdash; Candidate, then evidence

Put the two halves together. The rubric proposes; the scorecard disposes. Note the asymmetry:
when there is no evidence yet, the honest answer is the **cheaper** design, not the interesting
one.
"""),
    code('''
def recommend(brief: dict, single: dict | None = None, multi: dict | None = None) -> dict:
    """Return {"candidate", "decision", "why"} for one brief.

    With no measurements, the decision falls back to the cheaper design and says so.
    """
    candidate = rubric(brief)

    if candidate in ("workflow", "single_agent"):
        return {"candidate": candidate, "decision": candidate,
                "why": "the rubric terminates before multi-agent is on the table"}

    if single is None or multi is None:
        return {"candidate": candidate, "decision": ___,     # TODO: no evidence yet -- what ships?
                "why": "no scorecard yet; the cheaper design holds until the numbers exist"}

    cleared, reason = clears_budget(single, multi)
    return {"candidate": candidate,
            "decision": candidate if cleared else "single_agent",
            "why": reason}
''', '''
def recommend(brief: dict, single: dict | None = None, multi: dict | None = None) -> dict:
    """Return {"candidate", "decision", "why"} for one brief.

    With no measurements, the decision falls back to the cheaper design and says so.
    """
    candidate = rubric(brief)

    if candidate in ("workflow", "single_agent"):
        return {"candidate": candidate, "decision": candidate,
                "why": "the rubric terminates before multi-agent is on the table"}

    if single is None or multi is None:
        return {"candidate": candidate, "decision": "single_agent",
                "why": "no scorecard yet; the cheaper design holds until the numbers exist"}

    cleared, reason = clears_budget(single, multi)
    return {"candidate": candidate,
            "decision": candidate if cleared else "single_agent",
            "why": reason}
'''),
    code('''
# --- Self-check: Section 3
_inv = BRIEFS[3]                       # payment exception investigation
_measured_bad = ({"pass_rate": 0.83, "tokens": 1000, "seconds": 1.0},
                 {"pass_rate": 0.83, "tokens": 8000, "seconds": 3.0})
_measured_good = ({"pass_rate": 0.70, "tokens": 1000, "seconds": 1.0},
                  {"pass_rate": 0.92, "tokens": 2400, "seconds": 1.6})

check("with no evidence, the cheaper design ships",
      lambda: recommend(_inv)["decision"] == "single_agent",
      "an unmeasured multi-agent design is a hypothesis, not a decision")
check("with no evidence, the candidate is still reported",
      lambda: recommend(_inv)["candidate"] == "supervisor_worker",
      "record what the rubric proposed even when you do not ship it")
check("evidence that fails the budget sends you back to one agent",
      lambda: recommend(_inv, *_measured_bad)["decision"] == "single_agent")
check("evidence that clears the budget promotes the candidate",
      lambda: recommend(_inv, *_measured_good)["decision"] == "supervisor_worker")
check("a workflow brief never reaches the scorecard",
      lambda: recommend(BRIEFS[0], *_measured_good)["decision"] == "workflow")
'''),

    md("""
## Section 4 &mdash; The design-review table

One table, six briefs. This is the artefact you take back to work.
"""),
    code('''
def review_table(briefs, single=None, multi=None) -> str:
    """A fixed-width table of candidate vs decision for every brief."""
    rows = [f"{'brief':38} {'candidate':20} {'decision':20} why"]
    rows.append("-" * 110)
    for b in briefs:
        r = recommend(b, single, multi)
        rows.append(f"{b['name']:38} {r['candidate']:20} {r['decision']:20} {r['why']}")
    return "\\n".join(rows)

try:
    print(review_table(BRIEFS))
    print()
    print(review_table([BRIEFS[3]], *_measured_good))
except NameError:
    print("(finish the sections above, then re-run this cell)")
'''),
    code('''
# --- Self-check: Section 4
check("the table has a row per brief plus a header and rule",
      lambda: len(review_table(BRIEFS).splitlines()) == len(BRIEFS) + 2)
check("every brief resolves to a real verdict",
      lambda: all(recommend(b)["decision"] in VERDICTS for b in BRIEFS))
check("only briefs whose steps vary escape 'workflow'",
      lambda: all((recommend(b)["decision"] == "workflow") == (not b["steps_vary"]) for b in BRIEFS))
'''),

    md("""
## Run it for real

Have the model write the paragraph you would put in front of a design review. Notice what you are
asking it to do: not to *decide*, but to explain a decision your code already made and can defend.
"""),
    code('''
if llm_ready():
    try:
        verdict = recommend(BRIEFS[3], *_measured_good)
        summary = ask(
            "Write one short paragraph for an engineering design review. State the chosen "
            "architecture, the evidence that justified it, and the condition under which the team "
            "should revisit the decision. Be plain and specific; do not add claims beyond the "
            "facts given.\\n\\n"
            f"BRIEF: {BRIEFS[3]['name']}\\n"
            f"CANDIDATE FROM RUBRIC: {verdict['candidate']}\\n"
            f"DECISION: {verdict['decision']}\\n"
            f"EVIDENCE: {verdict['why']}\\n"
            f"BUDGET: {BUDGET}"
        )
        print(summary)
    except NameError:
        print("(finish the sections above, then re-run this cell)")
'''),
    md("""
### Read it

If the paragraph reads as a justification you would actually sign, the rubric did its job. If it
reads as advocacy for the interesting architecture, look again at which clause let it through.

**What you take from Module 1:** a rubric that terminates early, a budget written before the
measurement, and a scorecard that can overrule your own design preference. Modules 2 and 3 make
the agent better. This lab is what stops you building one you did not need.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `rubric()` takes booleans, which assumes someone already made the hard calls. Replace
   `fits_one_agent` with a function of the tool count and argue for the threshold you pick.
2. Add a fifth question &mdash; **can this fail unattended?** &mdash; that can force a supervisor even when
   the rubric would otherwise say single agent. Where in the order does it belong, and why?
3. Take a real brief from your own team, fill in the five booleans honestly, and run it. If the
   verdict surprises you, which boolean were you tempted to fill in dishonestly?
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-1-01-llm-vs-agent-loop",        LAB1),
    ("lab-1-02-four-building-blocks",     LAB2),
    ("lab-1-03-create-agent-on-ramp",     LAB3),
    ("lab-1-04-coordination-tax",         LAB4),
    ("lab-1-05-challenge-decision-rubric", LAB5),
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
