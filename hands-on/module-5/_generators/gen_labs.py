#!/usr/bin/env python3
"""
Generate Module 5 lab notebooks and their solutions from one source.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-5-0N-*.ipynb and ../solutions/

Rebuilt 2026-09-10 as THREE REAL SYSTEMS, one per lab, matching the three slides the
Module 5 deck now carries. The old five labs taught the orchestration primitives one at
a time -- topologies, decomposition, reducers, HITL, a scorecard -- and Modules 1 and 3
had already taught every one of them. So the module was a recap with more code in it.
Now each lab is a system a participant recognises, built end to end:

    5.1  a support desk that triages itself      supervisor -> one of N workers
    5.2  a research brief written in parallel    plan -> N researchers -> reducer -> writer
    5.3  an incident responder that stops        cycle with a budget -> escalate to a human

Derived from the flagship (../../../agentic-ai/hands-on): session-9 lab01 supervisor/worker
for 5.1, session-8 lab02/lab03 parallel execution and custom reducers plus session-9 lab06
aggregation and conflict for 5.2, and session-8 lab04/lab05 error handling and retry with
backoff for 5.3.

Design rules (unchanged from the Day 1 rebuild):
  * The participant writes REAL LangGraph code in every lab.
  * Self-checks assert on framework OBJECTS -- a compiled StateGraph, a declared reducer,
    a routing key, a real invoke of a real graph. All of that is deterministic and needs
    no endpoint. Only model INVOCATION needs the gateway, and that lives in "Run it for
    real" cells, which are observed, not scored.
  * Nodes in a graded graph are plain functions over state, so a graph's structure AND its
    behaviour are exact offline.
  * Blank the DECISION, give the mechanics. If the answer is a comprehension, a slice or a
    set operation, it is written out; the blank is which node, which key, which reducer,
    which way out.
  * "BLANK" marks a blank; an unfilled blank raises NameError and prints [TODO].
    NOT three underscores: IPython PREDEFINES _, __ and ___ as its output history
    (they start as ""), so under a real Jupyter kernel that token is a defined empty
    string, not an undefined name. The NameError never fires, [TODO] silently becomes
    [FAIL], and a blank used as a loop guard is falsy forever.
  * Blanks live INSIDE function bodies. A module-level `x = BLANK` crashes the cell
    instead of reporting [TODO], and so does a blank in a TypedDict class body --
    `Annotated[int, BLANK]` is evaluated when the class is created. Compiling a graph
    whose NODE BODY holds a blank is fine; invoking it at module level is not.
  * No node may be named after a setup helper (ask, check, guard, score, llm_ready,
    get_llm). A node called `ask` shadowed the model helper in Module 3 and only the
    live verifier caught it.
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
# Lab 5.{num} &mdash; {title}

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 2 &middot; Module 5 &mdash; Multi-Agent Collaboration &amp; Orchestration**

### What you'll do
{items}

> **How this lab works.** You write real LangGraph code. Fill every `BLANK`, then run the
> **Self-check** cell under each section &mdash; those assert on the *objects you built*
> (a compiled `StateGraph`, a declared reducer, a routing key), so they are deterministic
> and never depend on the model. Cells marked **Run it for real** put your work in front of
> the sandbox model; that is the part worth watching. The score line is feedback, not a grade.

{note}
""")


SETUP_COMMON = '''
# ---------------------------------------------------------------- Setup: run me first
import os, json, time, textwrap
from typing import Any, Callable

WORK = os.path.join("/tmp", "awmas-lab-5-{num:02d}")
os.makedirs(WORK, exist_ok=True)

# ---- self-check plumbing -------------------------------------------------
_results = []

def _is_todo(exc: BaseException) -> bool:
    """Is this exception really an unfilled blank?

    LangGraph runs your nodes inside tasks, so the NameError from an unfilled BLANK can
    arrive wrapped. Walk the cause chain before calling anything a failure -- telling you
    your answer is wrong when you have not written one yet is the worst thing a lab does.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, NameError):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False

def check(name: str, fn: Callable[[], Any], hint: str = "") -> None:
    """[PASS] / [FAIL] / [TODO] for one assertion. An unfilled blank prints [TODO]."""
    try:
        ok = bool(fn())
    except Exception as exc:
        if _is_todo(exc):
            print(f"[TODO] {{name}}")
            _results.append(None)
            return
        print(f"[FAIL] {{name}} -- {{type(exc).__name__}}: {{exc}}")
        _results.append(False)
        return
    print(("[PASS] " if ok else "[FAIL] ") + name + ("" if ok else (" -- " + hint if hint else "")))
    _results.append(ok)

def guard(fn: Callable[[], Any], default: Any = None) -> Any:
    """Run fn(). If a blank above is still unfilled, say so and carry on -- never crash Run All."""
    try:
        return fn()
    except Exception as exc:
        if not _is_todo(exc):
            raise
        print(f"(a blank above is still unfilled: {{exc}} -- fill it in, then re-run this cell)")
        return default

def score() -> None:
    done = [r for r in _results if r is not None]
    passed = sum(1 for r in done if r)
    todo = sum(1 for r in _results if r is None)
    print(f"\\nScore: {{passed}}/{{len(_results)}}" + (f"   ({{todo}} still TODO)" if todo else ""))

# ---- the sandbox model ---------------------------------------------------
# Your sandbox already has an LLM configured -- nothing to install, no key to register.
# These values are read from the environment so this notebook never hardcodes an endpoint.
LLM_BASE_URL = (os.environ.get("LAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("LITELLM_BASE_URL"))
LLM_MODEL    = (os.environ.get("LAB_LLM_MODEL") or os.environ.get("OPENAI_MODEL")
                or os.environ.get("LITELLM_MODEL"))
LLM_API_KEY  = os.environ.get("OPENAI_API_KEY", "sandbox")

# The served model reasons before it answers, and the reasoning is billed as completion
# tokens. Off is the default here because the "Run it for real" cells make many small calls.
NO_THINK = {{"chat_template_kwargs": {{"enable_thinking": False}}}}

def llm_ready() -> bool:
    if not LLM_BASE_URL or not LLM_MODEL:
        print("Model not configured. In a sandbox terminal run `env | grep -i llm` and set:")
        print("  export LAB_LLM_BASE_URL=...    # the gateway URL from your welcome sheet")
        print("  export LAB_LLM_MODEL=...       # the model name from your welcome sheet")
        return False
    return True

_llm_cache = {{}}
def get_llm(temperature: float = 0.0, think: bool = False):
    """A LangChain chat model pointed at the sandbox gateway (OpenAI-compatible)."""
    from langchain_openai import ChatOpenAI
    key = (temperature, think)
    if key not in _llm_cache:
        kwargs = {{}} if think else {{"extra_body": NO_THINK}}
        _llm_cache[key] = ChatOpenAI(model=LLM_MODEL, base_url=LLM_BASE_URL,
                                     api_key=LLM_API_KEY, temperature=temperature, **kwargs)
    return _llm_cache[key]

def ask(prompt: str, system: str | None = None, think: bool = False) -> str:
    """One stateless call. Returns text, or an error string -- never raises."""
    try:
        msgs = ([("system", system)] if system else []) + [("human", prompt)]
        return get_llm(think=think).invoke(msgs).content
    except Exception as exc:
        return f"<model unavailable: {{type(exc).__name__}}: {{exc}}>"

print("work dir:", WORK)
print("model   :", LLM_MODEL or "(not configured -- the object-level self-checks still work)")
'''


def setup(num, extra=""):
    return code(SETUP_COMMON.format(num=num) + extra)


SCORE = code('''
score()
''')


# =========================================================================== #
# Lab 5.1 -- a support desk that triages itself
# =========================================================================== #
DESK_CASE = '''
# ------------------------------------------------- the case file (synthetic, self-contained)
# A customer support desk for a SaaS product. One queue in, three specialists behind it.
# This case file is Module 5 lab 5.1 only -- 5.2 and 5.3 are different systems.

SPECIALISTS = ("billing", "tech", "account")

DESK = {
    "billing": "Charges, refunds, invoices, plan and price changes.",
    "tech":    "Errors, outages, failing API calls, anything broken.",
    "account": "Seats, owners, permissions, sign-in and access.",
}

# Twelve tickets with a known correct specialist. The last four name no keyword at all --
# their intent is only implied, which is exactly where a rule table runs out and the reason
# anyone reaches for a model.
TICKETS = [
    ("I was charged twice for March.",                              "billing"),
    ("Can I get an invoice with our VAT number on it?",             "billing"),
    ("We want to downgrade to the starter plan.",                   "billing"),
    ("Your API returns 500 on every /sync call since 09:00.",       "tech"),
    ("The export button throws an error and nothing downloads.",    "tech"),
    ("Webhooks stopped firing after your deploy.",                  "tech"),
    ("Please add two more seats for the new joiners.",              "account"),
    ("Move the workspace owner to priya@example.com.",              "account"),
    # intent implied, no keyword names it
    ("Nobody on my team can get in this morning.",                  "account"),
    ("We were told this would be free until June.",                 "billing"),
    ("Everything was fine yesterday and now nothing loads.",        "tech"),
    ("Someone who left in May can still see our data.",             "account"),
]

print(f"{len(TICKETS)} tickets, {len(DESK)} specialists")
'''

LAB1 = [
    header(1, "A Support Desk That Triages Itself", "Intermediate &rarr; Advanced", 35,
           ["Build the supervisor/worker graph: one queue in, three specialists, one reply out",
            "Wire the supervisor as a real <code>add_conditional_edges</code>, not an <code>if</code> statement",
            "Score the router against twelve labelled tickets &mdash; a supervisor is a classifier",
            "Swap in a model-routed supervisor and score that on the same twelve"],
           "> **The system on slide 2.** Tickets arrive on one queue; a supervisor reads each one and\n"
           "> picks a specialist; one specialist runs; a `resolve` node writes the reply."),
    setup(1),
    code(DESK_CASE),

    md("""
## Concept

A supervisor decides which specialist handles a request. In LangGraph that is one thing: a
**conditional edge** out of a supervisor node.

`add_conditional_edges(source, fn, path_map)` needs two different things, and people mix them up:

| | |
|---|---|
| `fn` | a function of **state** that returns a **key** |
| `path_map` | `{key: node name}` &mdash; which node each key means |

Which makes the supervisor a classifier with a known correct answer. So it has an accuracy, and
almost nobody measures it &mdash; even though a misroute wastes every token spent downstream of it.
"""),

    md("""
## Section 1 &mdash; The graph

Three worker nodes, a supervisor node that writes its decision **into state**, and a `resolve`
node that all three feed. Two decisions are yours: the **fallback** (every ticket the keyword
table does not recognise ends up there, which makes that one line the router's whole failure
mode) and which function is the **routing adapter**.
"""),
    code('''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END


class DeskState(TypedDict):
    ticket: str                       # what the customer wrote
    route: str | None                 # the supervisor's decision, readable afterwards
    answer: str | None                # the specialist's reply
    trail: Annotated[list, add]       # append: every node leaves a mark


# Order matters: the most specific group goes first.
KEYWORDS = [
    ("billing", ("charge", "charged", "invoice", "refund", "plan", "price", "vat")),
    ("tech",    ("error", "500", "api", "webhook", "broken", "throws", "down")),
    ("account", ("seat", "owner", "permission", "access", "sign-in")),
]

def route_by_keyword(ticket: str) -> str:
    """The first keyword group that matches wins."""
    low = (ticket or "").lower()
    for specialist, words in KEYWORDS:
        if any(w in low for w in words):
            return specialist
    # Nothing matched. Every unrecognised ticket in the system lands here, so choose on
    # purpose: which desk would you rather a stranger's ticket landed on by accident?
    return BLANK                      # TODO: the fallback, named from SPECIALISTS


def supervisor(state: DeskState) -> dict:
    """A node like any other. It decides, and it writes the decision down."""
    choice = route_by_keyword(state["ticket"])
    return {"route": choice, "trail": [f"supervisor -> {choice}"]}


def make_worker(name: str):
    """Three specialists that differ only in who they are. Deterministic, so the graph
    can be asserted exactly offline; the model shows up in the live cell below."""
    def worker(state: DeskState) -> dict:
        return {"answer": f"[{name}] {DESK[name]} Re: {state['ticket'][:40]}",
                "trail": [f"{name} handled it"]}
    return worker


def resolve(state: DeskState) -> dict:
    return {"trail": ["resolved"]}


def pick_specialist(state: DeskState) -> str:
    """The adapter: takes STATE, returns a KEY of the path map."""
    return state["route"]


def build_desk():
    g = StateGraph(DeskState)
    g.add_node("supervisor", supervisor)
    for name in SPECIALISTS:
        g.add_node(name, make_worker(name))
    g.add_node("resolve", resolve)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", BLANK, {n: n for n in SPECIALISTS})
    #                                     ^ TODO: which function above routes here?
    #   route_by_keyword takes a ticket STRING, so it is not the one.
    for name in SPECIALISTS:
        g.add_edge(name, "resolve")
    g.add_edge("resolve", END)
    return g.compile()


def fresh(ticket: str) -> dict:
    return {"ticket": ticket, "route": None, "answer": None, "trail": []}
''', '''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END


class DeskState(TypedDict):
    ticket: str                       # what the customer wrote
    route: str | None                 # the supervisor's decision, readable afterwards
    answer: str | None                # the specialist's reply
    trail: Annotated[list, add]       # append: every node leaves a mark


# Order matters: the most specific group goes first.
KEYWORDS = [
    ("billing", ("charge", "charged", "invoice", "refund", "plan", "price", "vat")),
    ("tech",    ("error", "500", "api", "webhook", "broken", "throws", "down")),
    ("account", ("seat", "owner", "permission", "access", "sign-in")),
]

def route_by_keyword(ticket: str) -> str:
    """The first keyword group that matches wins."""
    low = (ticket or "").lower()
    for specialist, words in KEYWORDS:
        if any(w in low for w in words):
            return specialist
    # Nothing matched. Tech is the least damaging place to land a stranger: it reads the
    # ticket and can hand it on, where billing would be answering about money it has not read.
    return "tech"


def supervisor(state: DeskState) -> dict:
    """A node like any other. It decides, and it writes the decision down."""
    choice = route_by_keyword(state["ticket"])
    return {"route": choice, "trail": [f"supervisor -> {choice}"]}


def make_worker(name: str):
    """Three specialists that differ only in who they are. Deterministic, so the graph
    can be asserted exactly offline; the model shows up in the live cell below."""
    def worker(state: DeskState) -> dict:
        return {"answer": f"[{name}] {DESK[name]} Re: {state['ticket'][:40]}",
                "trail": [f"{name} handled it"]}
    return worker


def resolve(state: DeskState) -> dict:
    return {"trail": ["resolved"]}


def pick_specialist(state: DeskState) -> str:
    """The adapter: takes STATE, returns a KEY of the path map."""
    return state["route"]


def build_desk():
    g = StateGraph(DeskState)
    g.add_node("supervisor", supervisor)
    for name in SPECIALISTS:
        g.add_node(name, make_worker(name))
    g.add_node("resolve", resolve)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", pick_specialist, {n: n for n in SPECIALISTS})
    for name in SPECIALISTS:
        g.add_edge(name, "resolve")
    g.add_edge("resolve", END)
    return g.compile()


def fresh(ticket: str) -> dict:
    return {"ticket": ticket, "route": None, "answer": None, "trail": []}
'''),
    code('''
# --- Self-check: Section 1   (a REAL compiled graph, really running -- still no model)
def _run(ticket: str) -> dict:
    return build_desk().invoke(fresh(ticket))

check("the desk compiles",
      lambda: build_desk() is not None)
check("the fallback is a specialist that actually exists",
      lambda: route_by_keyword("zzzz nothing here zzzz") in SPECIALISTS,
      "a conditional edge returning a key the path map does not have is a runtime error")
check("a billing ticket reaches the billing agent",
      lambda: _run("I was charged twice for March.")["route"] == "billing")
check("an outage reaches the tech agent",
      lambda: _run("Your API returns 500 on every /sync call.")["route"] == "tech")
check("the decision is written into state, not hidden in control flow",
      lambda: _run("Please add two more seats.")["route"] == "account",
      "a routing decision you cannot read back afterwards is one you cannot audit")
check("exactly ONE specialist runs per ticket",
      lambda: sum(1 for m in _run("Please add two more seats.")["trail"]
                  if "handled it" in m) == 1,
      "a conditional edge picks one path; fanning out to all three is Lab 5.2's shape")
check("and every ticket still reaches resolve",
      lambda: _run("Everything was fine yesterday.")["trail"][-1] == "resolved")

def _trace():
    for chunk in build_desk().stream(fresh("Move the workspace owner to priya@example.com.")):
        for node, update in chunk.items():
            print(f"  {node:12} -> {list(update)}")
guard(_trace)
'''),

    md("""
## Section 2 &mdash; Score the supervisor

Twelve tickets with a known correct specialist. The harness is given &mdash; nothing in it is a
design decision. What *is* a decision is the last line: the bar a router has to clear before you
would put it in front of customers. Pick a number you would defend in a review, not one that
makes your router pass.
"""),
    code('''
def selections(router) -> dict:
    """{ticket: chosen specialist} over the whole eval set."""
    return {ticket: router(ticket) for ticket, _ in TICKETS}


def accuracy(sel: dict) -> float:
    """Fraction routed to the expected specialist. No selection counts as wrong."""
    return sum(1 for t, want in TICKETS if sel.get(t) == want) / len(TICKETS)


def confusion(sel: dict) -> dict:
    """{(expected, chosen): count} over the misses only."""
    out = {}
    for ticket, want in TICKETS:
        got = sel.get(ticket)
        if got != want:
            out[(want, got)] = out.get((want, got), 0) + 1
    return out


def clears_the_bar(acc: float) -> bool:
    """Would you ship a supervisor that routes this well? Decide the bar and defend it.

    Anything you can argue for above 0.6 and up to 0.95 passes the self-check -- the check
    is that you HAVE a bar, not that you picked the number this notebook would have picked.
    """
    return acc >= BLANK               # TODO: your acceptance bar, as a fraction


def _report():
    sel = selections(route_by_keyword)
    print(f"rule-based supervisor: {accuracy(sel):.0%} on {len(TICKETS)} tickets\\n")
    for (want, got), n in sorted(confusion(sel).items(), key=lambda kv: -kv[1]):
        print(f"  {n}x  should have been {want:8} -> went to {got}")
guard(_report)
''', '''
def selections(router) -> dict:
    """{ticket: chosen specialist} over the whole eval set."""
    return {ticket: router(ticket) for ticket, _ in TICKETS}


def accuracy(sel: dict) -> float:
    """Fraction routed to the expected specialist. No selection counts as wrong."""
    return sum(1 for t, want in TICKETS if sel.get(t) == want) / len(TICKETS)


def confusion(sel: dict) -> dict:
    """{(expected, chosen): count} over the misses only."""
    out = {}
    for ticket, want in TICKETS:
        got = sel.get(ticket)
        if got != want:
            out[(want, got)] = out.get((want, got), 0) + 1
    return out


def clears_the_bar(acc: float) -> bool:
    """Would you ship a supervisor that routes this well? Decide the bar and defend it.

    0.85 here: at 12 tickets that is one miss allowed, and a support desk can absorb one
    handoff in eight. Below that the specialists spend their day forwarding.
    """
    return acc >= 0.85


def _report():
    sel = selections(route_by_keyword)
    print(f"rule-based supervisor: {accuracy(sel):.0%} on {len(TICKETS)} tickets\\n")
    for (want, got), n in sorted(confusion(sel).items(), key=lambda kv: -kv[1]):
        print(f"  {n}x  should have been {want:8} -> went to {got}")
guard(_report)
'''),
    code('''
# --- Self-check: Section 2
_rule = None
def rule_selections():
    global _rule
    if _rule is None:
        _rule = selections(route_by_keyword)
    return _rule

check("the eval set covers every specialist",
      lambda: {w for _, w in TICKETS} == set(SPECIALISTS))
check("it contains tickets whose intent is only implied",
      lambda: sum(1 for t, _ in TICKETS
                  if not any(w in t.lower() for _, ws in KEYWORDS for w in ws)) >= 3,
      "an eval set of keyword-shaped tickets measures the keywords, not the routing")
check("the rule supervisor gets most of it right",
      lambda: accuracy(rule_selections()) > 0.6)
check("but not all of it -- there is headroom to argue about",
      lambda: accuracy(rule_selections()) < 1.0)
check("every miss lands on the FALLBACK, not on a random specialist",
      lambda: {got for _, got in confusion(rule_selections())} == {route_by_keyword("zzzz")},
      "a rule router's failure mode IS its fallback -- unrecognised intent all piles up there")
check("your acceptance bar is a real bar",
      lambda: clears_the_bar(1.0) and not clears_the_bar(0.5),
      "a bar nothing can clear is not a bar, and neither is one a coin flip clears")
check("and it is a number a support desk could live with",
      lambda: clears_the_bar(0.95) and not clears_the_bar(0.6),
      "at 0.6 two tickets in five reach the wrong desk and get forwarded by hand")

def _verdict():
    acc = accuracy(rule_selections())
    print(f"rule-based router: {acc:.0%} -> "
          f"{'clears your bar' if clears_the_bar(acc) else 'does NOT clear your bar'}")
guard(_verdict)
'''),

    md("""
## Run it for real &mdash; a model-routed supervisor

Same graph, same eval set, same metric. The only thing that changes is the function inside
`route_with_model`. What the model gets to read is `DESK` &mdash; three one-line descriptions,
which is the entire difference between the two routers.
"""),
    code('''
ROUTE_SYSTEM = ("You route one customer support ticket to exactly one specialist desk. "
                "Reply with the desk's name alone -- no punctuation, no explanation.")

def route_with_model(ticket: str) -> str:
    """Ask the model to pick a desk. Anything unrecognised falls back to the keywords."""
    listing = "\\n".join(f"- {n}: {d}" for n, d in DESK.items())
    reply = ask(f"Desks:\\n{listing}\\n\\nTicket: {ticket}\\n\\nDesk:", system=ROUTE_SYSTEM)
    word = (reply or "").strip().strip("`.\\"' ").lower().split()
    return word[0] if word and word[0] in DESK else route_by_keyword(ticket)


def _bake_off():
    rule = rule_selections()
    model = selections(route_with_model)
    print(f"{'supervisor':16}{'accuracy':>10}   clears your bar?")
    print("-" * 46)
    for label, sel in (("rule-based", rule), ("model", model)):
        acc = accuracy(sel)
        print(f"{label:16}{acc:>9.0%}   {'yes' if clears_the_bar(acc) else 'no'}")
    print()
    for (want, got), n in sorted(confusion(model).items(), key=lambda kv: -kv[1]):
        print(f"  model missed {n}x: {want} -> {got}")
    print("\\nAnd the model-routed supervisor inside the real graph:")
    # The graph is unchanged. Only the function the supervisor node calls is different.
    global route_by_keyword
    keep, route_by_keyword = route_by_keyword, route_with_model
    try:
        out = build_desk().invoke(fresh("Nobody on my team can get in this morning."))
        print(f"  route={out['route']}  trail={out['trail']}")
    finally:
        route_by_keyword = keep

if llm_ready():
    guard(_bake_off)
'''),

    SCORE,
    md("""
## Your turn

1. The four implied-intent tickets are where the rule table loses. Add keywords until it wins
   all twelve &mdash; then write down how many keywords you added, and ask whether a desk with
   real tickets could keep that table current.
2. Route the ambiguous tickets to a model and the obvious ones to the keyword table. Score the
   hybrid. It is usually the cheapest thing that clears the bar, and nobody builds it.
3. Add a fourth key to the path map &mdash; `escalate` &mdash; for tickets the supervisor is not
   confident about, and decide what "not confident" means when the router is a keyword table.
"""),
]


# =========================================================================== #
# Lab 5.2 -- a research brief written by several agents at once
# =========================================================================== #
BRIEF_CASE = '''
# ------------------------------------------------- the case file (synthetic, self-contained)
# A vendor due-diligence brief. Three questions, three sources each, one report.
# This case file is Module 5 lab 5.2 only -- 5.1 and 5.3 are different systems.

BRIEF = "Should we adopt Vendor X for document storage?"

# Each source carries WHERE it came from and how much weight that origin deserves.
# authority: 3 = a signed contract, 2 = a filed report, 1 = the vendor's own marketing.
SOURCES = {
    "pricing": [
        {"claim": "list price is 18 USD per seat per month", "source": "order form",  "authority": 3},
        {"claim": "20% discount above 500 seats",            "source": "order form",  "authority": 3},
        {"claim": "cheapest in its class",                   "source": "vendor site", "authority": 1},
    ],
    "security": [
        {"claim": "data is stored in the EU only",           "source": "vendor site", "authority": 1},
        {"claim": "data may be replicated to us-east-1",     "source": "signed DPA",  "authority": 3},
        {"claim": "SOC 2 Type II, audited 2026-03",          "source": "audit report", "authority": 2},
    ],
    "support": [
        {"claim": "99.9% uptime commitment",                 "source": "order form",  "authority": 3},
        {"claim": "4-hour response on P1",                   "source": "order form",  "authority": 3},
        {"claim": "24/7 human support",                      "source": "vendor site", "authority": 1},
    ],
}

# The two that cannot both be true. Data residency is the decision the whole brief turns on.
CONFLICT = ("data is stored in the EU only", "data may be replicated to us-east-1")

print(f"brief: {BRIEF}\\n{sum(len(v) for v in SOURCES.values())} claims across "
      f"{len(SOURCES)} questions")
'''

LAB2 = [
    header(2, "A Research Brief Written by Several Agents at Once", "Advanced", 35,
           ["Fan out from one planner to three researchers that run in a single superstep",
            "Meet <code>InvalidUpdateError</code> for real, then declare the reducer that fixes it",
            "Resolve a disagreement between two sources &mdash; by authority, not by vote",
            "Carry provenance into the report, so every line can be traced back"],
           "> **The system on slide 3.** A planner splits the brief, three researchers work at the\n"
           "> same time, a declared reducer merges what they return, and a writer produces one report."),
    setup(2),
    code(BRIEF_CASE),

    md("""
## Concept

Fanning out is easy: give three nodes the same predecessor and LangGraph runs them in one
**superstep**. Merging is the part that is a design decision.

Every one of those three nodes returns a partial state. If they all write the same key, LangGraph
needs to be told what "both" means &mdash; and if you never told it, it does **not** quietly keep
the last one. It raises `InvalidUpdateError`. Silent loss is what a hand-rolled `dict.update` does,
and a hand-rolled merge is what most people write first.

So: the framework protects the keys you declared, and only those.
"""),

    md("""
## Section 1 &mdash; Fan out, and declare the merge

Two graphs, built from the same nodes. The first leaves `notes` un-annotated and is here to fail
in front of you. The second is yours to finish: pick the **reducer** that makes three parallel
writes into one list.
"""),
    code('''
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

QUESTIONS = ("pricing", "security", "support")


# ---- the graph that does NOT work, so you can see what it does instead of being told
class NaiveState(TypedDict):
    brief: str
    notes: list                       # no reducer -- three nodes are about to write it


def naive_researcher(topic: str):
    def node(state: NaiveState) -> dict:
        return {"notes": [c["claim"] for c in SOURCES[topic]]}
    return node


def collision() -> str:
    """Run the un-annotated fan-out and report what actually happens."""
    g = StateGraph(NaiveState)
    for q in QUESTIONS:
        g.add_node(q, naive_researcher(q))
        g.add_edge(START, q)
        g.add_edge(q, END)
    try:
        out = g.compile().invoke({"brief": BRIEF, "notes": []})
        return f"no error -- {len(out['notes'])} of 9 claims survived"
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"


# ---- the graph that does
def merge_findings(existing: list, incoming: list) -> list:
    """The merge rule for `findings`. A reducer is just this: (existing, incoming) -> combined.

    LangGraph calls it once per writer, so three researchers in one superstep call it three
    times. Decide what "both" means -- keep everything, keep the newest, keep the first?
    Only one of those lets the writer see all nine claims.
    """
    return BLANK                       # TODO: combine the two lists


class BriefState(TypedDict):
    brief: str
    questions: list
    # Three researchers finish in the same superstep and all write this key. The merge rule
    # is declared here, once, and it holds for every future writer of the key.
    findings: Annotated[list, merge_findings]
    report: list


def plan(state: BriefState) -> dict:
    """Split the brief. Real planners ask a model; this one is fixed so the graph is exact."""
    return {"questions": list(QUESTIONS)}


RESEARCH_SECONDS = 0.15          # stands in for the reading a real researcher would do

def make_researcher(topic: str):
    def researcher(state: BriefState) -> dict:
        time.sleep(RESEARCH_SECONDS)
        return {"findings": [{"question": topic, **c} for c in SOURCES[topic]]}
    return researcher


def build_brief_graph(writer):
    g = StateGraph(BriefState)
    g.add_node("plan", plan)
    for q in QUESTIONS:
        g.add_node(q, make_researcher(q))
    g.add_node("writer", writer)

    g.add_edge(START, "plan")
    for q in QUESTIONS:
        g.add_edge("plan", q)          # fan out: one superstep, three nodes
        g.add_edge(q, "writer")        # fan in
    g.add_edge("writer", END)
    return g.compile()


def fresh_brief() -> dict:
    return {"brief": BRIEF, "questions": [], "findings": [], "report": []}
''', '''
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

QUESTIONS = ("pricing", "security", "support")


# ---- the graph that does NOT work, so you can see what it does instead of being told
class NaiveState(TypedDict):
    brief: str
    notes: list                       # no reducer -- three nodes are about to write it


def naive_researcher(topic: str):
    def node(state: NaiveState) -> dict:
        return {"notes": [c["claim"] for c in SOURCES[topic]]}
    return node


def collision() -> str:
    """Run the un-annotated fan-out and report what actually happens."""
    g = StateGraph(NaiveState)
    for q in QUESTIONS:
        g.add_node(q, naive_researcher(q))
        g.add_edge(START, q)
        g.add_edge(q, END)
    try:
        out = g.compile().invoke({"brief": BRIEF, "notes": []})
        return f"no error -- {len(out['notes'])} of 9 claims survived"
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"


# ---- the graph that does
def merge_findings(existing: list, incoming: list) -> list:
    """The merge rule for `findings`. A reducer is just this: (existing, incoming) -> combined.

    LangGraph calls it once per writer, so three researchers in one superstep call it three
    times. Concatenation keeps every writer's findings, in arrival order -- which is exactly
    what `operator.add` does on a list, and why you usually see `Annotated[list, add]`.
    """
    return existing + incoming


class BriefState(TypedDict):
    brief: str
    questions: list
    # Three researchers finish in the same superstep and all write this key. The merge rule
    # is declared here, once, and it holds for every future writer of the key.
    findings: Annotated[list, merge_findings]
    report: list


def plan(state: BriefState) -> dict:
    """Split the brief. Real planners ask a model; this one is fixed so the graph is exact."""
    return {"questions": list(QUESTIONS)}


RESEARCH_SECONDS = 0.15          # stands in for the reading a real researcher would do

def make_researcher(topic: str):
    def researcher(state: BriefState) -> dict:
        time.sleep(RESEARCH_SECONDS)
        return {"findings": [{"question": topic, **c} for c in SOURCES[topic]]}
    return researcher


def build_brief_graph(writer):
    g = StateGraph(BriefState)
    g.add_node("plan", plan)
    for q in QUESTIONS:
        g.add_node(q, make_researcher(q))
    g.add_node("writer", writer)

    g.add_edge(START, "plan")
    for q in QUESTIONS:
        g.add_edge("plan", q)          # fan out: one superstep, three nodes
        g.add_edge(q, "writer")        # fan in
    g.add_edge("writer", END)
    return g.compile()


def fresh_brief() -> dict:
    return {"brief": BRIEF, "questions": [], "findings": [], "report": []}
'''),
    code('''
# --- Self-check: Section 1   (real graphs, really running -- still no model)
def _noop_writer(state: BriefState) -> dict:
    return {}

def _collect() -> dict:
    return build_brief_graph(_noop_writer).invoke(fresh_brief())

check("the un-annotated fan-out does NOT silently drop two of the three",
      lambda: "InvalidUpdate" in collision(),
      "LangGraph refuses the ambiguous write; silent loss is what a hand-rolled merge does")
check("the graph with a declared merge rule compiles",
      lambda: build_brief_graph(_noop_writer) is not None)
check("all nine claims survive the merge",
      lambda: len(_collect()["findings"]) == 9,
      "the reducer decides this: without one there is no answer, with the wrong one there are three")
check("every question is represented",
      lambda: {f["question"] for f in _collect()["findings"]} == set(QUESTIONS))
check("provenance survived the merge too",
      lambda: all("source" in f and "authority" in f for f in _collect()["findings"]),
      "a merged claim you cannot trace is a claim you cannot defend")

def _supersteps():
    print(collision(), "\\n")
    # `values` mode emits the state once per SUPERSTEP, not once per node -- which is how
    # you can see the three researchers land together rather than one after another.
    for i, snap in enumerate(build_brief_graph(_noop_writer).stream(fresh_brief(),
                                                                    stream_mode="values")):
        print(f"  after step {i}: {len(snap['findings'])} findings")
    t0 = time.time()
    build_brief_graph(_noop_writer).invoke(fresh_brief())
    print(f"\\n  three researchers x {RESEARCH_SECONDS}s of work each, "
          f"whole graph: {time.time() - t0:.2f}s")
    print("  ^ nine findings arrive in ONE step, and the clock reads one researcher, not three.")
    print("    Note what did NOT change: the same nine claims were paid for either way.")
guard(_supersteps)
'''),

    md("""
## Section 2 &mdash; The writer, and the disagreement

Two of the nine claims cannot both be true: the vendor's site says data stays in the EU, the
signed DPA says it may be replicated to `us-east-1`. Two sources say EU-ish things and one says
otherwise, so **a vote gets this wrong**.

Agreement is not truth &mdash; three researchers given the same marketing page agree confidently,
because they inherited the error rather than each finding it. Authority and provenance are
checkable; a vote is not.
"""),
    code('''
def resolve_conflict(claims: list[dict]) -> dict:
    """Two claims contradict each other. Return the one the report should carry.

    Given: `claims` are the contradicting findings, each already carrying `source`,
    `authority` (higher is stronger) and `agreeing` (how many researchers said it).
    """
    return max(claims, key=lambda c: c[BLANK])    # TODO: what decides it


def writer(state: BriefState) -> dict:
    """Turn the merged findings into report lines, one per claim, conflict resolved."""
    findings = state["findings"]
    contested = [f for f in findings if f["claim"] in CONFLICT]
    for f in contested:
        f["agreeing"] = sum(1 for g in findings if g["claim"] == f["claim"])
    winner = resolve_conflict(contested)["claim"] if contested else None

    lines = []
    for f in findings:
        if f["claim"] in CONFLICT and f["claim"] != winner:
            continue                                   # the losing side of the conflict
        # A line nobody can trace is a line nobody can defend, so carry the origin
        # through from the finding into the report.
        lines.append(f"{f['question']}: {f['claim']}  [{f[BLANK]}]")
        #                                               ^ TODO: which field makes it traceable
    return {"report": lines}
''', '''
def resolve_conflict(claims: list[dict]) -> dict:
    """Two claims contradict each other. Return the one the report should carry.

    Given: `claims` are the contradicting findings, each already carrying `source`,
    `authority` (higher is stronger) and `agreeing` (how many researchers said it).

    Authority, not agreement. The signed DPA outranks the marketing page even when the
    marketing page is repeated more often -- repetition is not evidence.
    """
    return max(claims, key=lambda c: c["authority"])


def writer(state: BriefState) -> dict:
    """Turn the merged findings into report lines, one per claim, conflict resolved."""
    findings = state["findings"]
    contested = [f for f in findings if f["claim"] in CONFLICT]
    for f in contested:
        f["agreeing"] = sum(1 for g in findings if g["claim"] == f["claim"])
    winner = resolve_conflict(contested)["claim"] if contested else None

    lines = []
    for f in findings:
        if f["claim"] in CONFLICT and f["claim"] != winner:
            continue                                   # the losing side of the conflict
        # A line nobody can trace is a line nobody can defend, so carry the origin
        # through from the finding into the report.
        lines.append(f"{f['question']}: {f['claim']}  [{f['source']}]")
    return {"report": lines}
'''),
    code('''
# --- Self-check: Section 2
_EU, _US = CONFLICT

def _report():
    return build_brief_graph(writer).invoke(fresh_brief())["report"]

check("the signed DPA beats the vendor's own site",
      lambda: resolve_conflict([
          {"claim": _EU, "source": "vendor site", "authority": 1, "agreeing": 2},
          {"claim": _US, "source": "signed DPA",  "authority": 3, "agreeing": 1},
      ])["claim"] == _US,
      "authority is checkable; a show of hands is not")
check("and still beats it when the weaker claim has MORE voices",
      lambda: resolve_conflict([
          {"claim": _EU, "source": "vendor site", "authority": 1, "agreeing": 9},
          {"claim": _US, "source": "signed DPA",  "authority": 3, "agreeing": 1},
      ])["claim"] == _US,
      "if agreement decided it, three agents fed one bad page would carry the report")
check("the report keeps one side of the conflict, not both",
      lambda: sum(1 for line in _report() if _EU in line or _US in line) == 1)
check("and it is the side the contract says",
      lambda: any(_US in line for line in _report()))
check("every line names where it came from",
      lambda: all(any(f"[{s}]" in line for s in
                      ("order form", "vendor site", "signed DPA", "audit report"))
                  for line in _report()),
      "provenance is the whole reason the merge kept those fields")
check("the report is shorter than the findings by exactly the losing claim",
      lambda: len(_report()) == 8)

def _show():
    for line in _report():
        print("  " + line)
guard(_show)
'''),

    md("""
## Run it for real &mdash; the writer becomes an agent

The graph does not change at all. `writer` is swapped for a node that hands the merged findings
to the model and asks for two paragraphs &mdash; and the same conflict is still in the input, so
watch whether the model resolves it the way your rule did, or splits the difference.
"""),
    code('''
WRITER_SYSTEM = ("You write a short vendor due-diligence note for a procurement committee. "
                 "Two paragraphs, no bullet points. Every factual sentence must name its "
                 "source in square brackets. If two sources contradict each other, say so "
                 "explicitly and follow the contractual one.")

def llm_writer(state: BriefState) -> dict:
    findings = state["findings"]
    listing = "\\n".join(f"- ({f['question']}) {f['claim']}  [source: {f['source']}]"
                        for f in findings)
    note = ask(f"Question: {state['brief']}\\n\\nFindings:\\n{listing}", system=WRITER_SYSTEM)
    return {"report": [note]}


def _write_for_real():
    out = build_brief_graph(llm_writer).invoke(fresh_brief())
    print(textwrap.fill(out["report"][0], 96))
    print("\\n--- did it notice the contradiction? ---")
    body = out["report"][0].lower()
    print("  mentions replication to us-east-1:", "us-east-1" in body)
    print("  mentions the DPA               :", "dpa" in body)

if llm_ready():
    guard(_write_for_real)
'''),

    SCORE,
    md("""
## Your turn

1. Add a fourth researcher &mdash; `legal` &mdash; with claims that contradict `pricing`. You should
   not have to touch the reducer, the writer or the state to do it. If you did, the merge rule was
   in the wrong place.
2. Replace `max(..., key=authority)` with a real tie-break: what happens when two claims have the
   same authority? Decide, and write the check that would have caught the old behaviour.
3. Time the fan-out against the same three researchers in a row. Tokens will not move; wall clock
   will. Say out loud which budget you just spent, because they are not the same budget.
"""),
]


# =========================================================================== #
# Lab 5.3 -- an incident responder that knows when to stop
# =========================================================================== #
INCIDENT_CASE = '''
# ------------------------------------------------- the case file (synthetic, self-contained)
# Three production incidents and the runbook for each. `heals_on_attempt` makes the fake
# remediation deterministic, so retry behaviour is exact offline -- no sleeping, no luck.
# This case file is Module 5 lab 5.3 only -- 5.1 and 5.2 are different systems.

INCIDENTS = {
    "INC-901": {"symptom": "checkout p99 above 2s",     "service": "checkout",
                "heals_on_attempt": 2},        # a flap: the second try holds
    "INC-902": {"symptom": "payment webhooks 500ing",   "service": "webhook",
                "heals_on_attempt": 1},        # the runbook works first time
    "INC-903": {"symptom": "disk 96% on db-primary",    "service": "db",
                "heals_on_attempt": None},     # no safe fix exists -- a human must decide
}

RUNBOOK = {
    "checkout": "recycle the slowest pod in the checkout deployment",
    "webhook":  "replay the dead-letter queue for the last 15 minutes",
    "db":       "extend the volume -- capacity changes need a human",
}

def apply_fix(ref: str, attempt: int) -> bool:
    """The remediation, standing in for kubectl. Deterministic on purpose."""
    heals = INCIDENTS[ref]["heals_on_attempt"]
    return heals is not None and attempt >= heals

print(f"{len(INCIDENTS)} incidents, {len(RUNBOOK)} runbook entries")
'''

LAB3 = [
    header(3, "An Incident Responder That Knows When to Stop", "Advanced", 35,
           ["Build the retry cycle: an edge that points backwards, and the budget that ends it",
            "Route the three ways out of <code>verify</code> &mdash; and see which one is the infinite loop",
            "Prove a never-healing incident terminates, rather than hoping it does",
            "Write the escalation handoff: what an on-call human needs at 3am"],
           "> **The system on slide 4.** An alert is triaged against a runbook, a fix is attempted and\n"
           "> verified, failures cycle back with backoff, and a spent budget hands off to a human."),
    setup(3),
    code(INCIDENT_CASE),

    md("""
## Concept

A cycle is just an edge pointing at a node that has already run. What makes it safe is not the
edge &mdash; it is the **budget**, and where the budget lives.

A per-node retry count multiplies: three nodes with three attempts each is twenty-seven calls, and
people write per-node counts because that is where the code is. A budget that belongs to the
**run**, decremented by every attempt, is the only thing standing between a bug and a bill.

Then the part worth arguing about: running out of attempts is **not an error**. It is a routing
decision, and the way out is a node.
"""),

    md("""
## Section 1 &mdash; The cycle, and the way out of it

`verify` has three ways out and they all have to exist in the path map. Two of them are yours:
which key the failure-but-attempts-remain case returns, and which node the retry edge points
back at.
"""),
    code('''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END


class IncidentState(TypedDict):
    ref: str
    symptom: str
    runbook_step: str | None
    attempts: int                     # NOT annotated: each attempt sets it, nothing merges
    budget: int                       # a property of the RUN, set once at the top
    healthy: bool
    outcome: str | None
    handoff: dict | None
    log: Annotated[list, add]         # append: what was tried, and what happened each time


def triage(state: IncidentState) -> dict:
    """Match the symptom to the runbook. One lookup; a real one would ask a model."""
    step = RUNBOOK[INCIDENTS[state["ref"]]["service"]]
    return {"runbook_step": step, "log": [f"triaged: {step}"]}


def backoff(attempt: int) -> float:
    """1s, 2s, 4s in production. Scaled down here so the lab does not sleep for a minute."""
    return 0.01 * (2 ** (attempt - 1))


def remediate(state: IncidentState) -> dict:
    """Attempt the runbook step. Charges one attempt against the run's budget."""
    attempt = state["attempts"] + 1
    time.sleep(backoff(attempt))
    worked = apply_fix(state["ref"], attempt)
    return {"attempts": attempt, "healthy": worked,
            "log": [f"attempt {attempt}/{state['budget']}: "
                    f"{'held' if worked else 'did not hold'}"]}


def verify(state: IncidentState) -> dict:
    """Did it hold? Separate from remediate because checking is not fixing."""
    return {"log": [f"verified: {'healthy' if state['healthy'] else 'still failing'}"]}


def next_step(state: IncidentState) -> str:
    """Which way out of verify. Every key returned here must exist in the path map."""
    if state["healthy"]:
        return "resolved"
    if state["attempts"] >= state["budget"]:
        # Out of attempts, and still broken. One of the remaining keys is the infinite loop.
        return BLANK                  # TODO: where does a run with no attempts left go?
    return "retry"


def resolved(state: IncidentState) -> dict:
    return {"outcome": "resolved", "log": ["closed"]}
''', '''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END


class IncidentState(TypedDict):
    ref: str
    symptom: str
    runbook_step: str | None
    attempts: int                     # NOT annotated: each attempt sets it, nothing merges
    budget: int                       # a property of the RUN, set once at the top
    healthy: bool
    outcome: str | None
    handoff: dict | None
    log: Annotated[list, add]         # append: what was tried, and what happened each time


def triage(state: IncidentState) -> dict:
    """Match the symptom to the runbook. One lookup; a real one would ask a model."""
    step = RUNBOOK[INCIDENTS[state["ref"]]["service"]]
    return {"runbook_step": step, "log": [f"triaged: {step}"]}


def backoff(attempt: int) -> float:
    """1s, 2s, 4s in production. Scaled down here so the lab does not sleep for a minute."""
    return 0.01 * (2 ** (attempt - 1))


def remediate(state: IncidentState) -> dict:
    """Attempt the runbook step. Charges one attempt against the run's budget."""
    attempt = state["attempts"] + 1
    time.sleep(backoff(attempt))
    worked = apply_fix(state["ref"], attempt)
    return {"attempts": attempt, "healthy": worked,
            "log": [f"attempt {attempt}/{state['budget']}: "
                    f"{'held' if worked else 'did not hold'}"]}


def verify(state: IncidentState) -> dict:
    """Did it hold? Separate from remediate because checking is not fixing."""
    return {"log": [f"verified: {'healthy' if state['healthy'] else 'still failing'}"]}


def next_step(state: IncidentState) -> str:
    """Which way out of verify. Every key returned here must exist in the path map."""
    if state["healthy"]:
        return "resolved"
    if state["attempts"] >= state["budget"]:
        # Out of attempts, and still broken. Returning "retry" here is the infinite loop:
        # the budget test would never be reached again with a different answer.
        return "escalate"
    return "retry"


def resolved(state: IncidentState) -> dict:
    return {"outcome": "resolved", "log": ["closed"]}
'''),

    md("""
## Section 2 &mdash; The handoff

`escalate` is the node the graph reaches when it has run out of safe things to try. A page that
says *"automation failed"* is a page. A page that says what was tried, and what happened each
time, is a **case** &mdash; and the difference is one key of state.
"""),
    code('''
def escalate(state: IncidentState) -> dict:
    """Assemble the case a human can act on. Every piece of it is already in state."""
    return {"outcome": "escalated",
            "handoff": {
                "incident":       state["ref"],
                "symptom":        state["symptom"],
                "runbook_step":   state["runbook_step"],
                "attempts_spent": state["attempts"],
                # A page that says "it failed" wastes the first ten minutes of the call.
                # Which key holds what was actually tried, and what happened each time?
                "what_happened":  state[BLANK],       # TODO
            },
            "log": ["escalated to the on-call"]}


def build_responder():
    g = StateGraph(IncidentState)
    g.add_node("triage", triage)
    g.add_node("remediate", remediate)
    g.add_node("verify", verify)
    g.add_node("resolved", resolved)
    g.add_node("escalate", escalate)

    g.add_edge(START, "triage")
    g.add_edge("triage", "remediate")
    g.add_edge("remediate", "verify")
    g.add_conditional_edges("verify", next_step, {
        "resolved": "resolved",
        "escalate": "escalate",
        # The backwards edge. Pointing it at the wrong node re-triages every attempt --
        # correct, but it pays for the diagnosis again on every retry.
        "retry":    BLANK,            # TODO: which node does a retry go back to?
    })
    g.add_edge("resolved", END)
    g.add_edge("escalate", END)
    return g.compile()


def fresh_incident(ref: str, budget: int = 3) -> dict:
    inc = INCIDENTS[ref]
    return {"ref": ref, "symptom": inc["symptom"], "runbook_step": None,
            "attempts": 0, "budget": budget, "healthy": False,
            "outcome": None, "handoff": None, "log": []}
''', '''
def escalate(state: IncidentState) -> dict:
    """Assemble the case a human can act on. Every piece of it is already in state."""
    return {"outcome": "escalated",
            "handoff": {
                "incident":       state["ref"],
                "symptom":        state["symptom"],
                "runbook_step":   state["runbook_step"],
                "attempts_spent": state["attempts"],
                # The log is the whole reason it was annotated with a reducer: every node
                # appended to it, so it is the narrative of the run.
                "what_happened":  state["log"],
            },
            "log": ["escalated to the on-call"]}


def build_responder():
    g = StateGraph(IncidentState)
    g.add_node("triage", triage)
    g.add_node("remediate", remediate)
    g.add_node("verify", verify)
    g.add_node("resolved", resolved)
    g.add_node("escalate", escalate)

    g.add_edge(START, "triage")
    g.add_edge("triage", "remediate")
    g.add_edge("remediate", "verify")
    g.add_conditional_edges("verify", next_step, {
        "resolved": "resolved",
        "escalate": "escalate",
        # Back to remediate, not to triage: the runbook step has not changed, so
        # re-triaging would pay for the diagnosis again on every single attempt.
        "retry":    "remediate",
    })
    g.add_edge("resolved", END)
    g.add_edge("escalate", END)
    return g.compile()


def fresh_incident(ref: str, budget: int = 3) -> dict:
    inc = INCIDENTS[ref]
    return {"ref": ref, "symptom": inc["symptom"], "runbook_step": None,
            "attempts": 0, "budget": budget, "healthy": False,
            "outcome": None, "handoff": None, "log": []}
'''),
    code('''
# --- Self-check: both sections   (a real graph with a real cycle -- still no model)
def _respond(ref: str, budget: int = 3) -> dict:
    return build_responder().invoke(fresh_incident(ref, budget))

check("the responder compiles, cycle and all",
      lambda: build_responder() is not None)
check("the runbook fix that works first time resolves in one attempt",
      lambda: (_respond("INC-902")["outcome"], _respond("INC-902")["attempts"]) == ("resolved", 1))
check("the flap resolves on the SECOND attempt -- the cycle really ran",
      lambda: (_respond("INC-901")["outcome"], _respond("INC-901")["attempts"]) == ("resolved", 2))
check("the incident with no safe fix terminates instead of looping",
      lambda: _respond("INC-903")["outcome"] == "escalated",
      "returning 'retry' when the budget is spent is an infinite loop, not a retry")
check("and it stops at exactly the budget, not one past it",
      lambda: _respond("INC-903")["attempts"] == 3)
check("the budget belongs to the run, so lowering it lowers the bill",
      lambda: _respond("INC-903", budget=1)["attempts"] == 1)
check("the handoff carries what was tried, not just that it failed",
      lambda: len(_respond("INC-903")["handoff"]["what_happened"]) >= 5,
      "the on-call needs the narrative -- that is what the annotated log was for")
check("the handoff names every attempt and its outcome",
      lambda: sum(1 for line in _respond("INC-903")["handoff"]["what_happened"]
                  if line.startswith("attempt ")) == 3)
check("a resolved incident writes no handoff at all",
      lambda: _respond("INC-902")["handoff"] is None)

def _trace():
    for ref in INCIDENTS:
        out = _respond(ref)
        print(f"  {ref}  {out['outcome']:10} after {out['attempts']} attempt(s)")
    print()
    for line in _respond("INC-903")["log"]:
        print("   ", line)
guard(_trace)
'''),

    md("""
## Run it for real &mdash; the page a human receives

The handoff is a dict. A person at 3am wants three sentences. This is the one node in the graph
that genuinely needs a model, which is worth noticing: the routing, the retrying and the budget
were all better off without one.
"""),
    code('''
PAGE_SYSTEM = ("You are writing an on-call page. Three sentences, no greeting, no bullet "
               "points: what is broken, what automation already tried and what happened, "
               "and the single decision you need the human to make. Never invent a fact "
               "that is not in the handoff.")

def _page_the_human():
    out = build_responder().invoke(fresh_incident("INC-903"))
    handoff = out["handoff"]
    print(json.dumps(handoff, indent=2)[:500], "\\n")
    page = ask(f"Handoff:\\n{json.dumps(handoff, indent=2)}", system=PAGE_SYSTEM)
    print("--- the page ---")
    print(textwrap.fill(page, 96))

if llm_ready():
    guard(_page_the_human)
'''),

    SCORE,
    md("""
## Your turn

1. Give `triage` a second runbook step to try when the first one fails twice. You will find the
   retry edge now points at the wrong node &mdash; which is the honest reason that blank was a
   decision and not a formality.
2. Make the budget a *token* budget rather than an attempt count, decremented by what each node
   actually spent. Then ask which of the two a finance team would rather you enforced.
3. `escalate` is a node, so it can do more than write a dict: have it decide *who* to page from
   the service in the incident. Then say what happens when that lookup is wrong at 3am.
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-5-01-support-desk-triage",     LAB1),
    ("lab-5-02-parallel-research-brief", LAB2),
    ("lab-5-03-incident-responder",      LAB3),
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
