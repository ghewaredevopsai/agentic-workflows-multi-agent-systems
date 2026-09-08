#!/usr/bin/env python3
"""
Generate Module 5 lab notebooks and their solutions from one source.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-5-0N-*.ipynb and ../solutions/

Design rules (revised 2026-09-09 -- framework-forward, matching the Day 1 rebuild):
  * The participant writes REAL LangGraph code in every lab. A module called
    "Multi-Agent Orchestration" that builds no StateGraph teaches people to hand-roll
    a graph engine, which is what this module used to do.
  * Self-checks assert on framework OBJECTS -- a compiled StateGraph, a declared
    reducer, a checkpointed interrupt, a replayed checkpoint. All of that is
    deterministic and needs no endpoint. Only model INVOCATION needs the gateway, and
    that lives in "Run it for real" cells, which are observed, not scored.
  * Nodes in a graded graph are plain functions over state. The specialists in
    AGENTKIT are exactly that, so a graph's structure AND its cost are exact offline.
  * Blank the DECISION, give the mechanics. If the answer is a comprehension, a slice
    or a set operation, it is written out; the blank is which object, which constant,
    which ordering, which reducer, which verdict.
  * "BLANK" marks a blank; an unfilled blank raises NameError and prints [TODO].
    NOT three underscores: IPython PREDEFINES _, __ and ___ as its output history
    (they start as ""), so under a real Jupyter kernel that token is a defined empty
    string, not an undefined name. The NameError never fires, [TODO] silently becomes
    [FAIL], and a blank used as a loop guard is falsy forever -- lab 1.1 spun in
    `while True` until the pod was OOM-killed.
  * Blanks live INSIDE function bodies. A module-level `x = BLANK` crashes the cell
    instead of reporting [TODO]. Compiling a graph whose NODE BODY holds a blank is
    fine -- compilation does not call the node -- but invoking it at module level is not.
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
> (a compiled `StateGraph`, a declared reducer, a checkpointed interrupt), so they are
> deterministic and never depend on the model. Cells marked **Run it for real** put your work
> in front of the sandbox model; that is the part worth watching. The score line is feedback,
> not a grade.

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


# --------------------------------------------------------------------------- #
# the shared synthetic domain -- one use case runs through all five labs
# --------------------------------------------------------------------------- #
DOMAIN = '''
# ------------------------------------------------- the case file (synthetic, self-contained)
# One domain runs through all five Module 5 labs -- the same payment exceptions, now worked
# by several agents at once, and finally priced against the single agent from Day 1.
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


# the specialists, used as LangGraph NODES in every lab from here on
AGENTKIT = '''
# ------------------------------------------------- the specialists, as LangGraph nodes
# Each one takes the graph state and returns a PARTIAL state -- exactly the node shape from
# Module 3 -- and reports what it spent. They are deterministic, so a graph's structure AND
# its cost can be asserted offline and exactly. The "Run it for real" cells put the sandbox
# model behind the same interface.

SANCTIONS_WATCH = {"NORTHWIND"}

COST = {"supervisor": 120, "ledger": 380, "policy": 420, "sanctions": 90, "writer": 610}


def agent_ledger(state: dict) -> dict:
    """Read the payment named in the state."""
    ref = state.get("ref")
    record = LEDGER.get(ref)
    if record is None:
        return {"problems": [f"no payment on file with reference {ref!r}"],
                "tokens": COST["ledger"]}
    return {"facts": {"ref": ref, **record},
            "findings": [{"by": "ledger", "source": "ledger",
                          "claim": f"{ref} is {record['status']} "
                                   f"for {record['amount']:,.2f} {record['ccy']}"}],
            "tokens": COST["ledger"]}


def agent_policy(state: dict) -> dict:
    """Say what the operating policy is for whatever went wrong."""
    code = (state.get("facts") or {}).get("reason_code")
    if code is None:
        return {"problems": ["policy ran before the reason code existed"],
                "tokens": COST["policy"]}
    return {"findings": [{"by": "policy", "source": "policy",
                          "claim": POLICY.get(code, f"no policy on file for {code}")}],
            "needs_human": code in NEEDS_HUMAN,
            "tokens": COST["policy"]}


def agent_sanctions(state: dict) -> dict:
    """A set-membership test. No model needed, and none used -- note the cost column."""
    counterparty = (state.get("facts") or {}).get("counterparty")
    listed = counterparty in SANCTIONS_WATCH
    return {"findings": [{"by": "sanctions", "source": "watchlist",
                          "claim": f"{counterparty} is "
                                   f"{'ON the watchlist' if listed else 'not on the watchlist'}"}],
            "blocked": listed,
            "tokens": COST["sanctions"]}


def agent_writer(state: dict) -> dict:
    """Turn whatever findings arrived into one recommendation."""
    findings = state.get("findings") or []
    if (state.get("facts") or {}).get("status") == "settled":
        action = "no action"                      # nothing to release; it already went
    elif state.get("blocked") or state.get("needs_human"):
        action = "hold for a human"
    else:
        action = "release"
    return {"recommendation": action,
            "rationale": [f["claim"] for f in findings],
            "tokens": COST["writer"]}


AGENTS = {"ledger": agent_ledger, "policy": agent_policy,
          "sanctions": agent_sanctions, "writer": agent_writer}
print("specialists:", ", ".join(AGENTS))
'''


# =========================================================================== #
# Lab 5.1 -- the supervisor is a router, so measure it like one
# =========================================================================== #
LAB1 = [
    header(1, "The Supervisor Is a Router", "Intermediate &rarr; Advanced", 35,
           ["Build a rule-based supervisor, and measure its routing accuracy honestly",
            "Read the confusion table &mdash; and notice where every unrecognised request piles up",
            "Price a misroute: the wasted tokens are everything downstream of the mistake",
            "Wire the supervisor into a real <code>StateGraph</code> with <code>add_conditional_edges</code>"],
           "> **Module 3's graph, with several workers in it.** A supervisor picks a specialist the\n"
           "> way an agent picks a tool &mdash; so it is a conditional edge, and it has an accuracy."),
    setup(1),
    code(DOMAIN),
    code(AGENTKIT),

    md("""
## Concept

A supervisor decides which specialist handles a request. In LangGraph that is one thing: a
**conditional edge** out of a supervisor node. `add_conditional_edges(node, fn, path_map)` calls
`fn(state)`, which returns a key of `path_map`, and the graph goes there.

Which makes the supervisor a classifier with a known correct answer &mdash; so it has an accuracy,
and almost nobody measures it.

It is worth measuring because a misroute is the most expensive mistake in the graph: everything
spent downstream of it answered the wrong question. The supervisor's own call is the cheapest one
in the system, so &ldquo;save money on the router&rdquo; is usually a bad trade.
"""),

    md("""
## Section 1 &mdash; A rule-based supervisor

Keywords, in order, with a fallback. Unglamorous, free, instant, and identical every time &mdash;
and right far more often than people expect.

The mechanics are written out. The decision left to you is the **fallback**, because every
request the table does not recognise ends up there, which makes that one line the router's
entire failure mode.
"""),
    code('''
SPECIALISTS = ("ledger", "policy", "sanctions", "writer")

# Order matters: the most specific group goes first.
ROUTING_KEYWORDS = [
    ("sanctions", ("sanction", "embargo", "screening")),
    ("policy",    ("policy", "runbook", "rule", "limit", "breach", "allowed", "permitted")),
    ("writer",    ("draft", "write", "note", "letter", "summar")),
    ("ledger",    ("status", "amount", "record", "look up", "reference", "pull")),
]

def route_by_rule(request: str) -> str:
    """The first keyword group that matches wins."""
    low = (request or "").lower()
    for specialist, keywords in ROUTING_KEYWORDS:
        if any(k in low for k in keywords):
            return specialist
    # Nothing matched. Every unrecognised request in the system lands here, so choose on
    # purpose: which specialist can still do useful work knowing nothing but the reference?
    return BLANK          # TODO: the fallback specialist, named from SPECIALISTS
''', '''
SPECIALISTS = ("ledger", "policy", "sanctions", "writer")

# Order matters: the most specific group goes first.
ROUTING_KEYWORDS = [
    ("sanctions", ("sanction", "embargo", "screening")),
    ("policy",    ("policy", "runbook", "rule", "limit", "breach", "allowed", "permitted")),
    ("writer",    ("draft", "write", "note", "letter", "summar")),
    ("ledger",    ("status", "amount", "record", "look up", "reference", "pull")),
]

def route_by_rule(request: str) -> str:
    """The first keyword group that matches wins."""
    low = (request or "").lower()
    for specialist, keywords in ROUTING_KEYWORDS:
        if any(k in low for k in keywords):
            return specialist
    # Nothing matched. The ledger read is the only step that needs no other findings first,
    # so an unrecognised request is at least started rather than answered wrongly.
    return "ledger"
'''),
    code('''
# --- Self-check: Section 1
check("an explicit sanctions request routes to sanctions",
      lambda: route_by_rule("Run the embargo check on ZENITH.") == "sanctions")
check("a policy question routes to policy",
      lambda: route_by_rule("What is the runbook for an INVALID_IBAN return?") == "policy")
check("a lookup routes to the ledger",
      lambda: route_by_rule("What is the status of PMT-1001?") == "ledger")
check("a drafting request routes to the writer",
      lambda: route_by_rule("Draft the customer note for PMT-1002.") == "writer")
check("the fallback is a specialist that actually exists",
      lambda: route_by_rule("zzzz nothing matches here zzzz") in SPECIALISTS,
      "a conditional edge that returns a key the path map does not have is a runtime error")
check("the fallback is the one step that needs no prior findings",
      lambda: route_by_rule("zzzz nothing matches here zzzz") == "ledger",
      "policy needs a reason code, sanctions needs a counterparty, the writer needs findings")
check("every rule points at a specialist that exists",
      lambda: all(s in SPECIALISTS for s, _ in ROUTING_KEYWORDS))
'''),

    md("""
## Section 2 &mdash; Measure it

Fifteen requests with a known correct specialist. Four of them state their intent only by
implication &mdash; no keyword names it &mdash; because those are the cases a rule table cannot reach and
the reason anyone reaches for a model. This whole section is given: nothing here is a design
decision, it is the harness.
"""),
    code('''
ROUTE_EVAL = [
    ("Is PMT-1005 clear of sanctions screening?",                  "sanctions"),
    ("Run the embargo check on ZENITH.",                           "sanctions"),
    ("PMT-1003 breached the limit -- what does policy say?",       "policy"),
    ("What is the runbook for an INVALID_IBAN return?",            "policy"),
    ("Are we allowed to retry this one automatically?",            "policy"),
    ("What is the status of PMT-1001?",                            "ledger"),
    ("Look up the amount on reference PMT-1004.",                  "ledger"),
    ("Pull the record for PMT-1002.",                              "ledger"),
    ("Draft the customer note for PMT-1002.",                      "writer"),
    ("Write up the case summary for the file.",                    "writer"),
    ("Summarise why this payment is held and what happens next.",  "writer"),
    # the four whose intent is implied, not stated
    ("Who is the counterparty on PMT-1003, and is that name a problem?",       "sanctions"),
    ("Is there anything about ZENITH we should worry about before releasing?", "sanctions"),
    ("This one has been sitting for three days. What are we supposed to do?",  "policy"),
    ("Tell the client what happened and why.",                                 "writer"),
]

def selections(router) -> dict:
    """{request: chosen specialist} for the whole eval set."""
    return {request: router(request) for request, _ in ROUTE_EVAL}


def accuracy(sel: dict) -> float:
    """Fraction routed to the expected specialist. No selection counts as wrong."""
    return sum(1 for r, expected in ROUTE_EVAL if sel.get(r) == expected) / len(ROUTE_EVAL)


def confusion(sel: dict) -> dict:
    """{(expected, chosen): count} over the misses -- the pairs whose boundaries overlap."""
    out = {}
    for request, expected in ROUTE_EVAL:
        chosen = sel.get(request)
        if chosen != expected:
            out[(expected, chosen)] = out.get((expected, chosen), 0) + 1
    return out


def _report():
    sel = selections(route_by_rule)
    print(f"rule-based supervisor: {accuracy(sel):.0%} on {len(ROUTE_EVAL)} requests\\n")
    for (expected, chosen), n in sorted(confusion(sel).items(), key=lambda kv: -kv[1]):
        print(f"  {n}x  should have been {expected:10} -> went to {chosen}")
guard(_report)
'''),
    code('''
# --- Self-check: Section 2
_rule = None
def rule_selections():
    global _rule
    if _rule is None:
        _rule = selections(route_by_rule)
    return _rule

check("the eval set covers every specialist",
      lambda: {e for _, e in ROUTE_EVAL} == set(SPECIALISTS))
check("it contains requests whose intent is only implied",
      lambda: sum(1 for r, _ in ROUTE_EVAL
                  if not any(k in r.lower() for _, ks in ROUTING_KEYWORDS for k in ks)) >= 4,
      "an eval set of keyword-shaped requests measures the keywords, not the routing")
check("the rule supervisor gets most of it right",
      lambda: accuracy(rule_selections()) > 0.6)
check("but not all of it -- there is headroom to argue about",
      lambda: accuracy(rule_selections()) < 1.0)
check("every miss lands on the FALLBACK, not on a random specialist",
      lambda: {chosen for _, chosen in confusion(rule_selections())} == {"ledger"},
      "a rule router's failure mode is its fallback -- that is where unrecognised intent piles up")
check("so the confusion table names one problem, not four",
      lambda: len({chosen for _, chosen in confusion(rule_selections())}) == 1)
'''),

    md("""
## Section 3 &mdash; What a misroute costs

The supervisor's own call is the cheapest thing in the graph. The specialist it wakes up is not.
Price the mistake and the argument about which model to route with settles itself.
"""),
    code('''
# COST came with the specialists. The supervisor is charged too -- routing is not free.

def cost_of(chosen: str) -> int:
    """Tokens for one routing decision plus the specialist it woke up."""
    return COST["supervisor"] + COST.get(chosen, 0)


def wasted_tokens(sel: dict) -> int:
    """Tokens spent answering the wrong question."""
    total = 0
    for request, expected in ROUTE_EVAL:
        chosen = sel.get(request)
        if chosen and chosen != expected:
            # A misroute wastes the specialist's work. Does it waste the routing call that
            # caused it as well, or is that a sunk cost you would have paid anyway?
            total += BLANK          # TODO: what one misroute cost you
    return total


def spent_tokens(sel: dict) -> int:
    """Everything the run spent, right or wrong."""
    return sum(cost_of(sel[r]) for r, _ in ROUTE_EVAL if sel.get(r))
''', '''
# COST came with the specialists. The supervisor is charged too -- routing is not free.

def cost_of(chosen: str) -> int:
    """Tokens for one routing decision plus the specialist it woke up."""
    return COST["supervisor"] + COST.get(chosen, 0)


def wasted_tokens(sel: dict) -> int:
    """Tokens spent answering the wrong question."""
    total = 0
    for request, expected in ROUTE_EVAL:
        chosen = sel.get(request)
        if chosen and chosen != expected:
            # The whole hop is wasted: you paid to choose wrongly and then paid the wrong
            # specialist. Charging only the specialist flatters the router that caused it.
            total += cost_of(chosen)
    return total


def spent_tokens(sel: dict) -> int:
    """Everything the run spent, right or wrong."""
    return sum(cost_of(sel[r]) for r, _ in ROUTE_EVAL if sel.get(r))
'''),
    code('''
# --- Self-check: Section 3
_perfect = {r: e for r, e in ROUTE_EVAL}

check("a perfect router wastes nothing",
      lambda: wasted_tokens(_perfect) == 0)
check("the waste is the supervisor call plus the specialist it woke up",
      lambda: wasted_tokens({**_perfect,
                             "Tell the client what happened and why.": "ledger"})
              == COST["supervisor"] + COST["ledger"],
      "the routing call is part of the mistake, not a sunk cost -- you would not have made it")
check("misrouting to the writer costs more than misrouting to sanctions",
      lambda: cost_of("writer") > cost_of("sanctions"),
      "the cost of a mistake depends on which specialist you woke up, not on the mistake")
check("the rule router wastes a real fraction of what it spends",
      lambda: 0 < wasted_tokens(rule_selections()) < spent_tokens(rule_selections()))
check("cheapening the supervisor cannot recover that waste",
      lambda: wasted_tokens(rule_selections()) > COST["supervisor"] * len(ROUTE_EVAL),
      "even a FREE supervisor would not save what the misroutes already cost -- that is the whole point")

def _price():
    sel = rule_selections()
    spent, wasted = spent_tokens(sel), wasted_tokens(sel)
    print(f"  spent   {spent:>6} tokens")
    print(f"  wasted  {wasted:>6} tokens  ({wasted / spent:.0%} of the bill)")
    print(f"  the supervisor's own calls were only {COST['supervisor'] * len(ROUTE_EVAL)} of that")
guard(_price)
'''),

    md("""
## Section 4 &mdash; The supervisor as a real conditional edge

Everything above was a function returning a string. Now make it a graph.

`add_conditional_edges(source, fn, path_map)` needs two different things and people mix them up:

| | |
|---|---|
| `fn` | a function of **state** that returns a **key** |
| `path_map` | `{key: node name}` &mdash; which node each key means |

`route_by_rule` is *not* `fn`: it takes a request string, not a state. Write the one-line adapter
that is.
"""),
    code('''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

class RouterState(TypedDict):
    request: str                              # what the user asked for
    ref: str                                  # the payment the request is about
    route: str | None                         # the supervisor's decision, written into state
    facts: dict | None
    findings: Annotated[list, add]            # append: every node's findings survive
    problems: Annotated[list, add]
    tokens: Annotated[int, add]               # add: the bill is the sum, not the last write
    blocked: bool
    needs_human: bool
    recommendation: str | None
    rationale: list


def supervisor_node(state: RouterState) -> dict:
    """The supervisor is a node like any other. It decides, and it charges for deciding."""
    return {"route": route_by_rule(state["request"]), "tokens": COST["supervisor"]}


def choose_specialist(state: RouterState) -> str:
    """The adapter: takes STATE, returns a KEY of the path map."""
    return state["route"]


def build_router_graph():
    g = StateGraph(RouterState)
    g.add_node("supervisor", supervisor_node)
    for name in SPECIALISTS:
        g.add_node(name, AGENTS[name])        # the specialists, unchanged, as nodes
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", BLANK, {n: n for n in SPECIALISTS})
    #                                     ^ TODO: which function above does the routing here?
    for name in SPECIALISTS:
        g.add_edge(name, END)
    return g.compile()


def fresh_router_state(request: str, ref: str = "PMT-1005") -> dict:
    return {"request": request, "ref": ref, "route": None, "facts": None,
            "findings": [], "problems": [], "tokens": 0, "blocked": False,
            "needs_human": False, "recommendation": None, "rationale": []}
''', '''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

class RouterState(TypedDict):
    request: str                              # what the user asked for
    ref: str                                  # the payment the request is about
    route: str | None                         # the supervisor's decision, written into state
    facts: dict | None
    findings: Annotated[list, add]            # append: every node's findings survive
    problems: Annotated[list, add]
    tokens: Annotated[int, add]               # add: the bill is the sum, not the last write
    blocked: bool
    needs_human: bool
    recommendation: str | None
    rationale: list


def supervisor_node(state: RouterState) -> dict:
    """The supervisor is a node like any other. It decides, and it charges for deciding."""
    return {"route": route_by_rule(state["request"]), "tokens": COST["supervisor"]}


def choose_specialist(state: RouterState) -> str:
    """The adapter: takes STATE, returns a KEY of the path map."""
    return state["route"]


def build_router_graph():
    g = StateGraph(RouterState)
    g.add_node("supervisor", supervisor_node)
    for name in SPECIALISTS:
        g.add_node(name, AGENTS[name])        # the specialists, unchanged, as nodes
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", choose_specialist, {n: n for n in SPECIALISTS})
    for name in SPECIALISTS:
        g.add_edge(name, END)
    return g.compile()


def fresh_router_state(request: str, ref: str = "PMT-1005") -> dict:
    return {"request": request, "ref": ref, "route": None, "facts": None,
            "findings": [], "problems": [], "tokens": 0, "blocked": False,
            "needs_human": False, "recommendation": None, "rationale": []}
'''),
    code('''
# --- Self-check: Section 4   (a REAL compiled graph, running -- still no model)
def _via_graph(request: str) -> dict:
    return build_router_graph().invoke(fresh_router_state(request))

check("the supervisor graph compiles",
      lambda: build_router_graph() is not None)
check("the supervisor's decision is written into state, not hidden in control flow",
      lambda: _via_graph("What is the status of PMT-1005?")["route"] == "ledger",
      "a routing decision you cannot read back is one you cannot audit")
check("and the ledger node really ran",
      lambda: _via_graph("What is the status of PMT-1005?")["facts"]["ref"] == "PMT-1005")
check("a sanctions request reaches the sanctions specialist",
      lambda: _via_graph("Run the embargo check on ZENITH.")["route"] == "sanctions")
check("exactly one specialist runs per request",
      lambda: len(_via_graph("Run the embargo check on ZENITH.")["findings"]) == 1,
      "a conditional edge picks ONE path; fanning out to all four is a different design "
      "and Lab 5.2 costs it")
check("but that specialist had no case facts to work with",
      lambda: _via_graph("Run the embargo check on ZENITH.")["blocked"] is False,
      "nothing read the payment first, so the screen had no counterparty -- ordering is Lab 5.2")
check("the run is charged for the routing decision as well as the specialist",
      lambda: _via_graph("What is the status of PMT-1005?")["tokens"]
              == COST["supervisor"] + COST["ledger"],
      "that total is the Annotated[int, add] reducer on `tokens` doing the adding")
check("the graph agrees with the function it was built from",
      lambda: all(_via_graph(r)["route"] == route_by_rule(r) for r, _ in ROUTE_EVAL[:4]))

def _trace():
    for chunk in build_router_graph().stream(
            fresh_router_state("Summarise why PMT-1005 is held and what happens next.")):
        for node, update in chunk.items():
            print(f"  {node:12} -> {list(update)}")
guard(_trace)
'''),

    md("""
## Run it for real &mdash; the model behind the same interface

Same eval set, same metric, same confusion table, same conditional edge. The only thing that
changes is the function inside `route_with_model`.
"""),
    code('''
ROUTE_SYSTEM = ("You route one operations request to exactly one specialist. "
                "Reply with the specialist's name alone -- no punctuation, no explanation.")

SPECIALIST_DESCRIPTIONS = {
    "ledger":    "Reads one payment record: status, amount, counterparty, reason code.",
    "policy":    "Says what the operating policy or runbook requires for a failure reason.",
    "sanctions": "Screens a counterparty name against the watchlist.",
    "writer":    "Turns findings into a summary or a customer-facing note.",
}

def route_with_model(request: str) -> str:
    """Ask the model to pick a specialist. Anything unrecognised falls back to the rules."""
    listing = "\\n".join(f"- {n}: {d}" for n, d in SPECIALIST_DESCRIPTIONS.items())
    reply = ask(f"Specialists:\\n{listing}\\n\\nRequest: {request}\\n\\nSpecialist:",
                system=ROUTE_SYSTEM)
    word = (reply or "").strip().strip("`.\\"' ").split()
    return word[0] if word and word[0] in SPECIALIST_DESCRIPTIONS else route_by_rule(request)


if llm_ready():
    def _compare():
        rule = rule_selections()
        model = selections(route_with_model)
        print(f"{'router':16}{'accuracy':>10}{'wasted tokens':>16}")
        print("-" * 44)
        print(f"{'rule-based':16}{accuracy(rule):>9.0%}{wasted_tokens(rule):>16}")
        print(f"{'model':16}{accuracy(model):>9.0%}{wasted_tokens(model):>16}")
        print()
        for (expected, chosen), n in sorted(confusion(model).items(), key=lambda kv: -kv[1]):
            print(f"  model: {n}x  {expected} -> {chosen}")
    guard(_compare)
'''),
    md("""
### Read it

Three things to look at, and the second is the one that decides your design:

1. **Did the model beat the rule table?** If not, the rules are free and deterministic, and you
   have your answer.
2. **Where did the model's misses land?** The rule router's misses all pile up on the fallback,
   which is one problem you can name. If the model's misses are scattered across four specialists,
   that is four overlapping descriptions &mdash; and Module 4 told you how to fix each one.
3. **Run it twice.** If the same request routes differently on the second run, you have met
   Module 7's opening problem a day early.

The usual production answer is neither: rules for the requests you can name, a model only for the
ones that fall through &mdash; which is exactly what `route_with_model`'s fallback line already does.
And note that swapping routers changed **no graph code at all**: the conditional edge does not
care where the key came from.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Make the hybrid explicit: try the rules, and call the model *only* when nothing matched.
   Measure its accuracy and its cost, and decide whether the saving is worth the second code path.
2. `route_by_rule` returns the first match, so a request mentioning both a policy and a draft goes
   to policy. Have `choose_specialist` return a **list** of keys instead &mdash; `add_conditional_edges`
   accepts that and dispatches to all of them. What have you just committed to paying?
3. Add a fifth key to the path map: `clarify` &mdash; requests where the honest answer is a question
   back to the user. What does that do to your accuracy, and is the drop real?
"""),
]


# =========================================================================== #
# Lab 5.2 -- decomposition, distribution, and what a handoff drops
# =========================================================================== #
LAB2 = [
    header(2, "Decomposition, Distribution and Handoffs", "Advanced", 40,
           ["Level a dependency graph into execution waves &mdash; what may run at once, and what may not",
            "Compile those dependencies into a real <code>StateGraph</code>, edge by edge",
            "Build the handoff payload, and decide what crosses it",
            "Reproduce the bug where both agents are right and the answer is wrong"],
           "> **Builds on Lab 5.1's supervisor.** Routing picked <em>who</em>. This lab is about\n"
           "> <em>in what order</em>, and what each one is told when its turn comes."),
    setup(2),
    code(DOMAIN),
    code(AGENTKIT),

    md("""
## Concept

Two ideas that get run together and should not be:

- **Decomposition** is a dependency graph. It tells you what *may* run at the same time. Nothing
  about wanting four agents makes four agents able to start.
- **Distribution** is a handoff. It carries exactly what you put in the message &mdash; and by default
  it drops the reasoning, the constraints and the failure history.

In LangGraph the first one is edges and the second one is state. The second produces the bug in
Section 4, where both agents behave correctly and the recommendation is still wrong.
"""),

    md("""
## Section 1 &mdash; What may run at once

Level the dependency graph into waves. Everything in one wave is independent; the number of waves
is the critical path, and no amount of parallelism shortens it. This is given whole &mdash; it is
arithmetic on the spec, and the spec is the interesting part.
"""),
    code('''
TASKS = {
    "read":      {"agent": "ledger",    "needs": []},
    "policy":    {"agent": "policy",    "needs": ["read"]},
    "screen":    {"agent": "sanctions", "needs": ["read"]},
    "recommend": {"agent": "writer",    "needs": ["policy", "screen"]},
}

def waves(tasks: dict = None) -> list:
    """Group tasks into execution waves. Everything within a wave may run at the same time."""
    tasks = TASKS if tasks is None else tasks
    done, out, remaining = set(), [], dict(tasks)
    while remaining:
        ready = [name for name, t in remaining.items()
                 if all(need in done for need in t["needs"])]
        if not ready:
            raise ValueError(f"circular dependency among {sorted(remaining)}")
        out.append(sorted(ready))
        done |= set(ready)
        for name in ready:
            remaining.pop(name)
    return out
'''),
    code('''
# --- Self-check: Section 1
_cycle = {"a": {"agent": "ledger", "needs": ["b"]}, "b": {"agent": "policy", "needs": ["a"]}}

def raises(fn, exc) -> bool:
    """True if fn() raises exc, False if it raises anything else or nothing at all.

    NameError is deliberately re-raised: a helper that swallows it turns an unfilled
    blank into a [FAIL] instead of a [TODO], which is a lie about what went wrong.
    """
    try:
        fn()
    except NameError:
        raise
    except exc:
        return True
    except Exception:
        return False
    return False

check("four tasks resolve into three waves",
      lambda: len(waves()) == 3)
check("nothing can start before the payment is read",
      lambda: waves()[0] == ["read"])
check("policy and screening are independent, so they share a wave",
      lambda: waves()[1] == ["policy", "screen"])
check("the recommendation waits for both",
      lambda: waves()[2] == ["recommend"])
check("every task appears exactly once",
      lambda: sorted(t for w in waves() for t in w) == sorted(TASKS))
check("the widest wave is two, so four agents never run four-abreast",
      lambda: max(len(w) for w in waves()) == 2,
      "the critical path is three hops whatever you spend on parallelism")
check("a circular dependency is refused rather than looping forever",
      lambda: raises(lambda: waves(_cycle), ValueError))
'''),

    md("""
## Section 2 &mdash; Compile the dependencies into a graph

`waves()` told you the shape. LangGraph wants it as edges, and it works out the waves for itself:
nodes with no unmet predecessor run **in the same superstep**.

One edge per dependency. The only thing to get right is which way it points.
"""),
    code('''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

class CaseState(TypedDict):
    ref: str
    facts: dict | None
    findings: Annotated[list, add]        # two nodes write this in one superstep -- see Lab 5.3
    problems: Annotated[list, add]
    tokens: Annotated[int, add]
    blocked: bool
    needs_human: bool
    recommendation: str | None
    rationale: list


def build_from_tasks(tasks: dict = None):
    """Turn the dependency spec into a compiled graph. The `needs` ARE the edges."""
    tasks = TASKS if tasks is None else tasks
    g = StateGraph(CaseState)
    for name, t in tasks.items():
        g.add_node(name, AGENTS[t["agent"]])
    for name, t in tasks.items():
        if not t["needs"]:
            g.add_edge(START, name)               # nothing to wait for
        for need in t["needs"]:
            # An edge runs FROM the task that must finish TO the task that was waiting.
            g.add_edge(need, BLANK)               # TODO: which end is the waiting task?
    for name in tasks:
        if not any(name in t["needs"] for t in tasks.values()):
            g.add_edge(name, END)                 # nothing waits on it, so it is a leaf
    return g.compile()


def fresh_case(ref: str) -> dict:
    return {"ref": ref, "facts": None, "findings": [], "problems": [], "tokens": 0,
            "blocked": False, "needs_human": False, "recommendation": None, "rationale": []}
''', '''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

class CaseState(TypedDict):
    ref: str
    facts: dict | None
    findings: Annotated[list, add]        # two nodes write this in one superstep -- see Lab 5.3
    problems: Annotated[list, add]
    tokens: Annotated[int, add]
    blocked: bool
    needs_human: bool
    recommendation: str | None
    rationale: list


def build_from_tasks(tasks: dict = None):
    """Turn the dependency spec into a compiled graph. The `needs` ARE the edges."""
    tasks = TASKS if tasks is None else tasks
    g = StateGraph(CaseState)
    for name, t in tasks.items():
        g.add_node(name, AGENTS[t["agent"]])
    for name, t in tasks.items():
        if not t["needs"]:
            g.add_edge(START, name)               # nothing to wait for
        for need in t["needs"]:
            # An edge runs FROM the task that must finish TO the task that was waiting.
            g.add_edge(need, name)
    for name in tasks:
        if not any(name in t["needs"] for t in tasks.values()):
            g.add_edge(name, END)                 # nothing waits on it, so it is a leaf
    return g.compile()


def fresh_case(ref: str) -> dict:
    return {"ref": ref, "facts": None, "findings": [], "problems": [], "tokens": 0,
            "blocked": False, "needs_human": False, "recommendation": None, "rationale": []}
'''),
    code('''
# --- Self-check: Section 2   (a REAL compiled graph, running -- no model)
def ran(ref: str = "PMT-1005") -> dict:
    return build_from_tasks().invoke(fresh_case(ref))

check("the graph compiles from the dependency spec alone",
      lambda: build_from_tasks() is not None)
check("all three fact-finding specialists contributed",
      lambda: {f["by"] for f in ran()["findings"]} == {"ledger", "policy", "sanctions"})
check("policy saw the reason code, so it ran AFTER the ledger read",
      lambda: ran()["problems"] == [],
      "'policy ran before the reason code existed' is what an edge pointing the wrong way "
      "looks like from inside a node")
check("the writer ran last, and had every finding in hand",
      lambda: len(ran()["rationale"]) == 3)
check("the sanctions case is held",
      lambda: ran()["recommendation"] == "hold for a human")
check("the bill is the sum of all four specialists",
      lambda: ran()["tokens"] == COST["ledger"] + COST["policy"]
                               + COST["sanctions"] + COST["writer"],
      "Annotated[int, add] again -- without it the last node to write would set the total")
check("a settled payment needs no action",
      lambda: ran("PMT-1001")["recommendation"] == "no action")
check("a payment that does not exist does not crash the graph",
      lambda: ran("PMT-0000")["problems"] != [],
      "the whole graph must survive one specialist finding nothing")

def _trace():
    for chunk in build_from_tasks().stream(fresh_case("PMT-1005")):
        print("  superstep ->", sorted(chunk))
    print("\\n  waves() said:", waves())
guard(_trace)
'''),

    md("""
## Section 3 &mdash; The handoff payload

An edge decides *when* an agent runs. It does not decide what the agent knows: that is state, and
in a real system it is a message you build by hand. What crosses is whatever you put in the dict,
and nothing else.
"""),
    code('''
def handoff(state: dict, to_agent: str, task: str) -> dict:
    """The message one agent sends another. ONLY what is in this dict crosses."""
    message = {"to": to_agent, "task": task, "facts": state.get("facts")}
    # A handoff silently drops three things: the reasoning, the constraints, and the failure
    # history. Two of those three are in this graph's state already. Add them, defaulting each
    # to an empty list so the receiving agent always finds the key.
    message.update(BLANK)     # TODO: a dict of the two the receiving agent cannot do without
    return message
''', '''
def handoff(state: dict, to_agent: str, task: str) -> dict:
    """The message one agent sends another. ONLY what is in this dict crosses."""
    message = {"to": to_agent, "task": task, "facts": state.get("facts")}
    message.update({"constraints":   state.get("constraints") or [],
                    "already_tried": state.get("already_tried") or []})
    return message
'''),
    code('''
# --- Self-check: Section 3
_rich = {"ref": "PMT-1005",
         "facts": {"ref": "PMT-1005", "reason_code": "SANCTIONS_REVIEW"},
         "constraints": ["Do not release PMT-1005 without a human decision"],
         "already_tried": ["auto-retry failed at 09:14"]}

check("the task and the facts cross",
      lambda: handoff(_rich, "policy", "decide")["task"] == "decide"
              and handoff(_rich, "policy", "decide")["facts"]["reason_code"] == "SANCTIONS_REVIEW")
check("the constraints cross",
      lambda: handoff(_rich, "policy", "decide")["constraints"] ==
              ["Do not release PMT-1005 without a human decision"])
check("the failure history crosses, so the next agent does not retry it",
      lambda: handoff(_rich, "policy", "decide")["already_tried"] == ["auto-retry failed at 09:14"])
check("a state with no constraints still hands over the key, empty",
      lambda: handoff({"facts": {}}, "policy", "x")["constraints"] == [],
      "a missing key and an empty list read very differently to the code on the other side")
check("the receiving agent is named",
      lambda: handoff(_rich, "sanctions", "screen")["to"] == "sanctions")
'''),

    md("""
## Section 4 &mdash; Both agents right, answer wrong

Triage establishes that a payment must not be released. It hands off. The policy agent recommends
releasing it. Neither agent malfunctioned.

Triage here is the first wave of your graph, run on its own &mdash; `build_from_tasks` takes any
sub-spec, so a one-task graph is a legitimate graph.
"""),
    code('''
def triage(ref: str) -> dict:
    """Wave 1 by itself: read the payment, then say what must not happen to it."""
    state = build_from_tasks({"read": TASKS["read"]}).invoke(fresh_case(ref))
    if (state.get("facts") or {}).get("reason_code") in NEEDS_HUMAN:
        state["constraints"] = [f"Do not release {ref} without a human decision"]
    return state


def policy_from_handoff(message: dict) -> dict:
    """A policy agent that knows only what the handoff told it -- the realistic case."""
    constraints = message.get("constraints") or []
    if any("do not release" in c.lower() for c in constraints):
        # A constraint from upstream outranks the policy text. Say so, and say what instead.
        return {"recommendation": BLANK,          # TODO: what does a forbidden release become?
                "why": "a constraint forbids release"}
    code = (message.get("facts") or {}).get("reason_code")
    return {"recommendation": "release", "why": POLICY.get(code, "no policy on file")}


def investigate(ref: str, carry_constraints: bool = True) -> dict:
    """Triage, hand off, decide. Flip carry_constraints to drop one key from the message."""
    message = handoff(triage(ref), "policy", f"decide whether {ref} can be released")
    if not carry_constraints:
        message.pop("constraints", None)          # the bug, made explicit
    return policy_from_handoff(message)
''', '''
def triage(ref: str) -> dict:
    """Wave 1 by itself: read the payment, then say what must not happen to it."""
    state = build_from_tasks({"read": TASKS["read"]}).invoke(fresh_case(ref))
    if (state.get("facts") or {}).get("reason_code") in NEEDS_HUMAN:
        state["constraints"] = [f"Do not release {ref} without a human decision"]
    return state


def policy_from_handoff(message: dict) -> dict:
    """A policy agent that knows only what the handoff told it -- the realistic case."""
    constraints = message.get("constraints") or []
    if any("do not release" in c.lower() for c in constraints):
        # A constraint from upstream outranks the policy text. Say so, and say what instead.
        return {"recommendation": "hold for a human",
                "why": "a constraint forbids release"}
    code = (message.get("facts") or {}).get("reason_code")
    return {"recommendation": "release", "why": POLICY.get(code, "no policy on file")}


def investigate(ref: str, carry_constraints: bool = True) -> dict:
    """Triage, hand off, decide. Flip carry_constraints to drop one key from the message."""
    message = handoff(triage(ref), "policy", f"decide whether {ref} can be released")
    if not carry_constraints:
        message.pop("constraints", None)          # the bug, made explicit
    return policy_from_handoff(message)
'''),
    code('''
# --- Self-check: Section 4
check("triage picks up the constraint from the reason code",
      lambda: triage("PMT-1005")["constraints"] != [])
check("carrying the constraint, the sanctions case is held",
      lambda: investigate("PMT-1005")["recommendation"] == "hold for a human")
check("DROPPING IT, the very same case is released",
      lambda: investigate("PMT-1005", carry_constraints=False)["recommendation"] == "release",
      "this is the bug: nothing errored, and both agents did exactly what they were asked")
check("the two runs disagree on the same payment",
      lambda: investigate("PMT-1005")["recommendation"]
              != investigate("PMT-1005", carry_constraints=False)["recommendation"])
check("the limit-breach case is protected the same way",
      lambda: investigate("PMT-1003")["recommendation"] == "hold for a human")
check("a case with no constraint is unaffected either way",
      lambda: investigate("PMT-1002")["recommendation"]
              == investigate("PMT-1002", carry_constraints=False)["recommendation"],
      "the dropped constraint only changes the cases where a constraint existed -- which is why it hides")
check("the held recommendation says which constraint stopped it",
      lambda: "constraint" in investigate("PMT-1005")["why"])

def _both_ways():
    for ref in ("PMT-1005", "PMT-1003", "PMT-1002"):
        with_c = investigate(ref)["recommendation"]
        without = investigate(ref, carry_constraints=False)["recommendation"]
        flag = "  <-- DIFFERENT" if with_c != without else ""
        print(f"  {ref}   carried: {with_c:18} dropped: {without:18}{flag}")
guard(_both_ways)
'''),

    md("""
## Run it for real

Give the model the two handoff messages &mdash; one with the constraint, one without &mdash; and ask it
for a recommendation. It is not being tested. Your message is.
"""),
    code('''
if llm_ready():
    def _ask_both():
        full = handoff(triage("PMT-1005"), "policy",
                       "decide whether PMT-1005 can be released")
        thin = {k: v for k, v in full.items() if k != "constraints"}
        for label, message in (("with constraint", full), ("without      ", thin)):
            reply = ask("You are the policy agent. Given this handoff, reply in one sentence with "
                        "your recommendation.\\n\\n" + json.dumps(message, default=str))
            print(f"  [{label}] {reply.strip()[:180]}")
            print()
    guard(_ask_both)
'''),
    md("""
### Read it

If the two replies differ, you have watched a correct agent reach a wrong conclusion because of
what it was not told. No prompt engineering fixes that, and no stronger model does either &mdash; the
information was not in the room.

**The rule:** a handoff carries the task, the findings, the constraints and what has already been
tried. Three of those four are the ones people forget, and each has its own signature bug &mdash;
paying twice, breaking a rule it never saw, and retrying what already failed.

And note the division of labour in the graph you built. **Edges decided the order; state decided
the knowledge.** Getting the edges right does nothing at all for a message that leaves the
constraint out.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add the third dropped thing &mdash; the reasoning &mdash; and measure what carrying it costs in
   tokens against what re-deriving it costs. One of those is a bill and one is a risk.
2. Point `build_from_tasks` at a spec where `recommend` needs only `policy`. Run it and watch the
   writer produce a recommendation with the sanctions screen still in flight. Which finding is
   missing from `rationale`, and would you have noticed in a log?
3. Turn `TASKS` into two specs &mdash; one for high-value payments and one for low &mdash; and let
   Lab 5.1's supervisor choose between them. That is routing by value, and Lab 5.5 prices it.
"""),
]


# =========================================================================== #
# Lab 5.3 -- parallel execution, reducers, and disagreement
# =========================================================================== #
LAB3 = [
    header(3, "Parallel Execution, Reducers and Disagreement", "Advanced", 40,
           ["Lose a whole specialist's work to a hand-rolled merge, silently",
            "Watch LangGraph refuse to do the same thing, and read the error it gives you",
            "Declare a reducer per key and prove every branch's findings survive",
            "Settle a disagreement by authority and provenance rather than by headcount"],
           "> **Module 3's reducers, with consequences.** There a lost key was a puzzle.\n"
           "> Here it is a compliance finding that never reached the recommendation."),
    setup(3),
    code(DOMAIN),
    code(AGENTKIT),

    md("""
## Concept

Two nodes in the same superstep both return `{"findings": [...]}`. What happens next depends
entirely on what you declared:

| You wrote | What happens |
|---|---|
| your own `dict.update` merge | the second overwrites the first, **silently** |
| `findings: list` in a LangGraph state | LangGraph **refuses** &mdash; one value per key per step |
| `findings: Annotated[list, add]` | both survive, in whatever order they finished |

The first row is the dangerous one, and it is what people write when they orchestrate agents by
hand. The summary still reads perfectly &mdash; a summary of one finding reads exactly as well as a
summary of two &mdash; and **nothing in your logs will show it**. Only a test that counts.
"""),

    md("""
## Section 1 &mdash; Watch it disappear, twice

First by hand, then through the framework. Both given: the point of this section is to see the
difference, not to write it.
"""),
    code('''
def fan_out(state: dict, agent_names) -> list:
    """Run several agents on the SAME input state. None of them sees the others' output."""
    return [(name, AGENTS[name](dict(state))) for name in agent_names]


def merge_naive(state: dict, partials: list) -> dict:
    """Last write wins. This is what a hand-rolled orchestrator does by default."""
    out = dict(state)
    for _, partial in partials:
        out.update(partial)
    return out


def base_state(ref: str = "PMT-1005") -> dict:
    """Everything wave 1 established, ready for the parallel wave."""
    seed = {"ref": ref, "tokens": 0}
    return {**seed, **agent_ledger(seed)}


def _by_hand():
    partials = fan_out(base_state(), ["policy", "sanctions"])
    returned = sum(len(p.get("findings") or []) for _, p in partials)
    merged = merge_naive(base_state(), partials)
    print(f"  the two specialists returned  {returned} findings")
    print(f"  the merged state contains     {len(merged.get('findings') or [])}")
    print(f"  errors raised                 0")
    for f in merged.get("findings") or []:
        print(f"    survivor: [{f['by']}] {f['claim'][:56]}")
guard(_by_hand)
'''),
    code('''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

class LooseState(TypedDict):
    """The same state with NO reducer anywhere. Note what LangGraph does about that."""
    ref: str
    facts: dict | None
    findings: list
    problems: list
    tokens: int
    blocked: bool
    needs_human: bool


def fan_out_graph(state_cls):
    """ledger, then policy and sanctions IN PARALLEL, then stop.

    Two edges out of one node is the whole of parallelism in LangGraph: both targets have
    their predecessor satisfied, so both run in the same superstep.
    """
    g = StateGraph(state_cls)
    g.add_node("ledger", agent_ledger)
    g.add_node("policy", agent_policy)
    g.add_node("sanctions", agent_sanctions)
    g.add_edge(START, "ledger")
    g.add_edge("ledger", "policy")
    g.add_edge("ledger", "sanctions")
    g.add_edge("policy", END)
    g.add_edge("sanctions", END)
    return g.compile()


def blank_case(ref: str = "PMT-1005") -> dict:
    return {"ref": ref, "facts": None, "findings": [], "problems": [], "tokens": 0,
            "blocked": False, "needs_human": False}


def parallel_outcome(state_cls) -> str:
    """Run the fan-out and report what happened: an exception name, or what survived."""
    try:
        out = fan_out_graph(state_cls).invoke(blank_case())
        return f"kept {len(out['findings'])} of 3 findings"
    except Exception as exc:
        return type(exc).__name__


guard(lambda: print("  with no reducer declared, LangGraph says:", parallel_outcome(LooseState)))
'''),
    code('''
# --- Self-check: Section 1
def _partials():
    return fan_out(base_state(), ["policy", "sanctions"])

check("both specialists really did return a finding",
      lambda: sum(len(p.get("findings") or []) for _, p in _partials()) == 2)
check("neither of them raised",
      lambda: all(isinstance(p, dict) for _, p in _partials()))
check("but the hand-rolled merge keeps only one",
      lambda: len(merge_naive(base_state(), _partials()).get("findings") or []) == 1,
      "one specialist's entire contribution is gone, and nothing said so")
check("the survivor is whichever one ran last -- an ordering accident",
      lambda: merge_naive(base_state(), _partials())["findings"][0]["by"] == "sanctions")
check("the token count is wrong too, and in the cheaper direction",
      lambda: merge_naive(base_state(), _partials())["tokens"] == COST["sanctions"],
      "you will under-report your own spend, which is the one bug nobody reports")
check("LangGraph does NOT quietly do the same thing",
      lambda: parallel_outcome(LooseState) != "kept 3 of 3 findings",
      "an un-annotated key written twice in one superstep is an error, not a coin toss -- "
      "print parallel_outcome(LooseState) to see exactly which one")
'''),

    md("""
## Section 2 &mdash; Declare a reducer per key

A reducer says how two writes to the same key combine. `operator.add` appends lists and sums
counters, so those two are written for you.

The interesting one is `blocked`. Two screens ran in parallel and reached different answers; the
reducer is where you decide, **in advance**, which answer a system like this must take.
"""),
    code('''
def any_blocker(old: bool, new: bool) -> bool:
    """The reducer for `blocked` and `needs_human`.

    Two branches ran at the same time. One came back saying stop; the other saw nothing wrong.
    Reducers must also be safe in either order -- you do not control which branch finishes first.
    """
    return BLANK          # TODO: combine the two so the right one wins, either way round


class MergedState(TypedDict):
    ref: str
    facts: dict | None
    findings: Annotated[list, add]              # append: every branch's findings survive
    problems: Annotated[list, add]
    tokens: Annotated[int, add]                 # sum: the bill is not the last write
    blocked: Annotated[bool, any_blocker]       # your rule, applied by the framework
    needs_human: Annotated[bool, any_blocker]
''', '''
def any_blocker(old: bool, new: bool) -> bool:
    """The reducer for `blocked` and `needs_human`.

    Two branches ran at the same time. One came back saying stop; the other saw nothing wrong.
    Reducers must also be safe in either order -- you do not control which branch finishes first.
    """
    return bool(old) or bool(new)      # one blocker is enough to block, whichever arrives first


class MergedState(TypedDict):
    ref: str
    facts: dict | None
    findings: Annotated[list, add]              # append: every branch's findings survive
    problems: Annotated[list, add]
    tokens: Annotated[int, add]                 # sum: the bill is not the last write
    blocked: Annotated[bool, any_blocker]       # your rule, applied by the framework
    needs_human: Annotated[bool, any_blocker]
'''),
    code('''
# --- Self-check: Section 2   (the same graph, the same nodes, a different state class)
def reduced():
    return fan_out_graph(MergedState).invoke(blank_case())

check("the reducer keeps a blocker whichever branch it arrives from",
      lambda: any_blocker(True, False) is True and any_blocker(False, True) is True,
      "a reducer you can apply in either order is one that survives a scheduler you "
      "do not control")
check("and an all-clear stays clear",
      lambda: any_blocker(False, False) is False)
check("with reducers declared, the same graph runs at all",
      lambda: reduced() is not None)
check("all three specialists' findings survive",
      lambda: len(reduced()["findings"]) == 3)
check("...so all three are represented",
      lambda: {f["by"] for f in reduced()["findings"]} == {"ledger", "policy", "sanctions"})
check("the spend is the sum of all three, not whichever wrote last",
      lambda: reduced()["tokens"] == COST["ledger"] + COST["policy"] + COST["sanctions"])
check("one blocker is enough to block",
      lambda: reduced()["blocked"] is True)
check("EVERY key the parallel specialists write is declared on the state",
      lambda: {k for _, p in _partials() for k in p} <= set(MergedState.__annotations__),
      "this is the check to run in CI -- a new specialist writing a new key is the next "
      "silent loss, or the next InvalidUpdateError in production")
'''),

    md("""
## Section 3 &mdash; The test that catches it

No log line shows a lost finding. What shows it is asserting that everyone you dispatched came
back. Given whole, because the value is in running it against both merges.
"""),
    code('''
def contributed(state: dict) -> set:
    """Which specialists actually appear in the merged findings."""
    return {f["by"] for f in (state.get("findings") or [])}


def everyone_came_back(state: dict, dispatched) -> bool:
    """The assertion that catches a silent parallel loss: count what came back."""
    return set(dispatched) <= contributed(state)


def missing(state: dict, dispatched) -> set:
    """Who was dispatched and is not in the findings."""
    return set(dispatched) - contributed(state)
'''),
    code('''
# --- Self-check: Section 3
_dispatched = ["ledger", "policy", "sanctions"]

check("the reducer-backed graph passes the test",
      lambda: everyone_came_back(reduced(), _dispatched) is True)
check("the hand-rolled merge FAILS it",
      lambda: everyone_came_back(merge_naive(base_state(), _partials()),
                                 ["policy", "sanctions"]) is False,
      "this single assertion is the whole defence against the bug in Section 1")
check("and the failure names who went missing",
      lambda: missing(merge_naive(base_state(), _partials()),
                      ["policy", "sanctions"]) == {"policy"})
check("nothing is missing from a correct merge",
      lambda: missing(reduced(), _dispatched) == set())
check("a specialist that returned no finding at all is also caught",
      lambda: everyone_came_back(reduced(), _dispatched + ["writer"]) is False,
      "'it ran and found nothing' and 'its result was dropped' both need to surface")
'''),

    md("""
## Section 4 &mdash; When they disagree

Three specialists say release. One, quoting the watchlist it read, says hold. Counting opinions
gets you the wrong answer confidently &mdash; which is Module 3's poisoning lab with a quorum.
"""),
    code('''
CONFLICT = [
    {"by": "writer",    "source": "inference", "verdict": "release",
     "claim": "nothing in the case looks unusual"},
    {"by": "policy",    "source": "policy",    "verdict": "release",
     "claim": "policy permits release once funded"},
    {"by": "ledger",    "source": "ledger",    "verdict": "release",
     "claim": "no block flag recorded against the payment"},
    {"by": "sanctions", "source": "watchlist", "verdict": "hold",
     "claim": "NORTHWIND is ON the watchlist"},
]

# Declared in advance, per question. On a sanctions question, compliance wins by definition.
AUTHORITY = {"sanctions": 3, "policy": 2, "ledger": 1, "writer": 0}

# Sources something other than a model can re-read.
CHECKABLE_SOURCES = {"ledger", "policy", "watchlist"}


def by_majority(findings: list) -> str:
    """The tempting rule. It counts opinions, and opinions are not evidence."""
    votes = {}
    for f in findings:
        votes[f["verdict"]] = votes.get(f["verdict"], 0) + 1
    return max(votes, key=votes.get)


def by_authority(findings: list, authority: dict = None) -> str:
    """The declared expert on this question wins, whatever the others think."""
    authority = AUTHORITY if authority is None else authority
    # max() needs a key. Ranking by how many agree is by_majority, which you already have.
    top = max(findings, key=lambda f: BLANK)   # TODO: rank one finding, by what?
    return top["verdict"]


def checkable(finding: dict) -> bool:
    """Can this claim be settled by re-reading a source, rather than by asking again?"""
    return finding["source"] in CHECKABLE_SOURCES


def settle(findings: list) -> str:
    """The rule THIS system uses when its specialists disagree about a release."""
    # One of the two rules above releases a watchlisted payment. Choose the other, and be
    # able to say why -- "three of the four said so" is not a reason a regulator accepts.
    return BLANK(findings)                     # TODO: by_majority or by_authority?
''', '''
CONFLICT = [
    {"by": "writer",    "source": "inference", "verdict": "release",
     "claim": "nothing in the case looks unusual"},
    {"by": "policy",    "source": "policy",    "verdict": "release",
     "claim": "policy permits release once funded"},
    {"by": "ledger",    "source": "ledger",    "verdict": "release",
     "claim": "no block flag recorded against the payment"},
    {"by": "sanctions", "source": "watchlist", "verdict": "hold",
     "claim": "NORTHWIND is ON the watchlist"},
]

# Declared in advance, per question. On a sanctions question, compliance wins by definition.
AUTHORITY = {"sanctions": 3, "policy": 2, "ledger": 1, "writer": 0}

# Sources something other than a model can re-read.
CHECKABLE_SOURCES = {"ledger", "policy", "watchlist"}


def by_majority(findings: list) -> str:
    """The tempting rule. It counts opinions, and opinions are not evidence."""
    votes = {}
    for f in findings:
        votes[f["verdict"]] = votes.get(f["verdict"], 0) + 1
    return max(votes, key=votes.get)


def by_authority(findings: list, authority: dict = None) -> str:
    """The declared expert on this question wins, whatever the others think."""
    authority = AUTHORITY if authority is None else authority
    top = max(findings, key=lambda f: authority.get(f["by"], 0))
    return top["verdict"]


def checkable(finding: dict) -> bool:
    """Can this claim be settled by re-reading a source, rather than by asking again?"""
    return finding["source"] in CHECKABLE_SOURCES


def settle(findings: list) -> str:
    """The rule THIS system uses when its specialists disagree about a release."""
    # Compliance is the declared expert on a release question, so authority decides it.
    return by_authority(findings)
'''),
    code('''
# --- Self-check: Section 4
check("three of the four say release",
      lambda: sum(1 for f in CONFLICT if f["verdict"] == "release") == 3)
check("so the majority rule releases a watchlisted payment",
      lambda: by_majority(CONFLICT) == "release",
      "confidently, unanimously among the three, and wrong")
check("authority holds it",
      lambda: by_authority(CONFLICT) == "hold")
check("an agent with no declared authority ranks below every one that has it",
      lambda: by_authority(CONFLICT + [{"by": "stranger", "source": "inference",
                                        "verdict": "release", "claim": "looks fine"}]) == "hold")
check("authority is declared in advance, not derived from the case",
      lambda: set(AUTHORITY) >= {f["by"] for f in CONFLICT},
      "a rule chosen while looking at one disagreement is a rule fitted to that disagreement")
check("the rule you chose holds the watchlisted payment",
      lambda: settle(CONFLICT) == "hold")
check("...and it is not the headcount rule",
      lambda: settle(CONFLICT) != by_majority(CONFLICT))
check("exactly one finding rests on nothing re-readable",
      lambda: [f["by"] for f in CONFLICT if not checkable(f)] == ["writer"])
check("and the dissenting finding is one of the checkable ones",
      lambda: checkable(next(f for f in CONFLICT if f["verdict"] == "hold")) is True,
      "which is why you can settle this by reading the watchlist rather than by taking a vote")

def _settle():
    print(f"  {'rule':14}{'verdict':10}")
    print("  " + "-" * 26)
    print(f"  {'majority':14}{by_majority(CONFLICT):10}")
    print(f"  {'authority':14}{by_authority(CONFLICT):10}")
    print()
    for f in sorted(CONFLICT, key=lambda f: -AUTHORITY.get(f["by"], 0)):
        mark = "checkable" if checkable(f) else "not checkable"
        print(f"  {f['by']:10} {f['verdict']:8} {mark:14} {f['claim'][:44]}")
guard(_settle)
'''),

    md("""
## Run it for real

Hand the model the four findings and ask it to settle them. Then hand it the same four with the
sources removed. The question is whether provenance changes its answer &mdash; and whether you would
be willing to depend on that.
"""),
    code('''
if llm_ready():
    def _judge():
        def render(findings, with_sources):
            return "\\n".join(
                (f"- [{f['by']}, source={f['source']}] {f['claim']} -> {f['verdict']}"
                 if with_sources else f"- {f['claim']} -> {f['verdict']}")
                for f in findings)
        for label, sourced in (("with sources   ", True), ("without sources", False)):
            reply = ask("Four agents disagree about whether one payment may be released. "
                        "Give the verdict and one sentence of reasoning.\\n\\n"
                        + render(CONFLICT, sourced))
            print(f"  [{label}] {reply.strip()[:200]}")
            print()
    guard(_judge)
'''),
    md("""
### Read it

If removing the sources flips the answer to *release*, provenance did the work &mdash; and that is
good news, because provenance is something you control. If the model holds either way, do not
turn that into a control: `by_authority` is four lines and cannot be argued out of its answer.

**What you take from this lab:** declare a reducer for every key two branches can write &mdash; the
framework will tell you when you have not, but only in the branches you actually exercise; assert
that everyone you dispatched came back; and settle disagreements on authority and sources rather
than on a headcount.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add a fourth node to `fan_out_graph` that writes a key `MergedState` does not declare. Does
   LangGraph reject the update, ignore it, or accept it? Whatever it does, decide how you would
   have found out in production.
2. `by_authority` breaks ties arbitrarily. Two equal-authority agents disagreeing is a real case:
   decide whether it escalates or falls back to provenance, and write it.
3. `any_blocker` ORs, so one blocker blocks. Build the opposite case &mdash; a key where OR is wrong
   &mdash; and say what that tells you about choosing a reducer from the data type alone.
"""),
]


# =========================================================================== #
# Lab 5.4 -- human-in-the-loop: interrupt, approve, time out, escalate
# =========================================================================== #
LAB4 = [
    header(4, "Human-in-the-Loop as an Orchestration Mechanism", "Advanced", 40,
           ["Stop a compiled graph before the irreversible node with <code>interrupt_before</code>",
            "Resume it, recording an identity rather than a boolean",
            "Time out, and climb an escalation ladder that actually terminates",
            "Rewind into the gate and find out who the interrupt really belongs to"],
           "> **Module 3's checkpointing, applied.** You can only pause a run whose state you can\n"
           "> write down and pick up again &mdash; an approval gate is that mechanism with a person in it."),
    setup(4),
    code(DOMAIN),
    code(AGENTKIT),

    md("""
## Concept

Human-in-the-loop appears twice in this course. Here it is an **orchestration mechanism**: a way
to pause a graph, ask, and carry on. In Module 8 the same machinery is a **safety control**.

Four parts, and the one people leave out is the third:

| | |
|---|---|
| **interrupt** | `compile(interrupt_before=[...])` &mdash; stop before a named node, state saved |
| **approve** | `update_state` who said yes, then `invoke(None, cfg)` to carry on |
| **timeout** | a gate with no deadline is a run that waits until Monday |
| **escalate** | expiry is not refusal and not approval &mdash; it is a different queue |

One thing that is **not** true: &ldquo;an approval gate needs a checkpointer&rdquo;. A gate that
simply refuses to act without a named approver needs nothing at all &mdash; you will build one in
Section 1's `node_release`. What needs a checkpointer is **pause and resume**: stopping now and
finishing later, from another process, after the person replies.
"""),

    md("""
## Section 1 &mdash; Interrupt before the irreversible node

The graph from Lab 5.2, with one more node on the end that actually changes something. The
checkpointer writes the state after every node; `interrupt_before` says where not to walk past.
"""),
    code('''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

RELEASED = set()          # the irreversible side effect, so we can prove whether it happened

GATED_NODES = ["release"]  # nodes that may not run unattended


class GateState(TypedDict):
    ref: str
    facts: dict | None
    findings: Annotated[list, add]
    problems: Annotated[list, add]
    tokens: Annotated[int, add]
    blocked: bool
    needs_human: bool
    recommendation: str | None
    rationale: list
    approved_by: str | None
    released: bool


def node_release(state: GateState) -> dict:
    """The one node that changes the world. It refuses unless a PERSON is named.

    This much needs no checkpointer: it is a gate because the tool will not fire without
    an approver in state. The checkpointer is what lets the person answer tomorrow.
    """
    who = state.get("approved_by")
    if not isinstance(who, str) or not who.strip():
        return {"released": False, "tokens": 10,
                "problems": ["release refused: no named human approver"]}
    RELEASED.add(state["ref"])
    return {"released": True, "tokens": 40}


def node_release_unattended(state: GateState) -> dict:
    """The same action with the review moved to the END of the run. It just goes."""
    RELEASED.add(state["ref"])
    return {"released": True, "tokens": 40}


def build(checkpointer=None, gate: bool = True):
    g = StateGraph(GateState)
    g.add_node("read", agent_ledger)
    g.add_node("policy", agent_policy)
    g.add_node("screen", agent_sanctions)
    g.add_node("recommend", agent_writer)
    g.add_node("release", node_release if gate else node_release_unattended)
    g.add_edge(START, "read")
    g.add_edge("read", "policy")
    g.add_edge("read", "screen")
    g.add_edge("policy", "recommend")
    g.add_edge("screen", "recommend")
    g.add_edge("recommend", "release")
    g.add_edge("release", END)
    # An approval gate is a place the graph is not allowed to walk past on its own.
    return g.compile(checkpointer=checkpointer,
                     interrupt_before=BLANK if gate else [])
    #                                 ^ TODO: the nodes a human must see BEFORE they run


def cfg(thread_id: str) -> dict:
    """A thread is one case. Two cases must never share one."""
    return {"configurable": {"thread_id": thread_id}}


def fresh_gate(ref: str) -> dict:
    return {"ref": ref, "facts": None, "findings": [], "problems": [], "tokens": 0,
            "blocked": False, "needs_human": False, "recommendation": None,
            "rationale": [], "approved_by": None, "released": False}


def pending(app, thread: str):
    """What is this thread waiting to do?"""
    return app.get_state(cfg(thread)).next
''', '''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

RELEASED = set()          # the irreversible side effect, so we can prove whether it happened

GATED_NODES = ["release"]  # nodes that may not run unattended


class GateState(TypedDict):
    ref: str
    facts: dict | None
    findings: Annotated[list, add]
    problems: Annotated[list, add]
    tokens: Annotated[int, add]
    blocked: bool
    needs_human: bool
    recommendation: str | None
    rationale: list
    approved_by: str | None
    released: bool


def node_release(state: GateState) -> dict:
    """The one node that changes the world. It refuses unless a PERSON is named.

    This much needs no checkpointer: it is a gate because the tool will not fire without
    an approver in state. The checkpointer is what lets the person answer tomorrow.
    """
    who = state.get("approved_by")
    if not isinstance(who, str) or not who.strip():
        return {"released": False, "tokens": 10,
                "problems": ["release refused: no named human approver"]}
    RELEASED.add(state["ref"])
    return {"released": True, "tokens": 40}


def node_release_unattended(state: GateState) -> dict:
    """The same action with the review moved to the END of the run. It just goes."""
    RELEASED.add(state["ref"])
    return {"released": True, "tokens": 40}


def build(checkpointer=None, gate: bool = True):
    g = StateGraph(GateState)
    g.add_node("read", agent_ledger)
    g.add_node("policy", agent_policy)
    g.add_node("screen", agent_sanctions)
    g.add_node("recommend", agent_writer)
    g.add_node("release", node_release if gate else node_release_unattended)
    g.add_edge(START, "read")
    g.add_edge("read", "policy")
    g.add_edge("read", "screen")
    g.add_edge("policy", "recommend")
    g.add_edge("screen", "recommend")
    g.add_edge("recommend", "release")
    g.add_edge("release", END)
    # An approval gate is a place the graph is not allowed to walk past on its own.
    return g.compile(checkpointer=checkpointer,
                     interrupt_before=GATED_NODES if gate else [])


def cfg(thread_id: str) -> dict:
    """A thread is one case. Two cases must never share one."""
    return {"configurable": {"thread_id": thread_id}}


def fresh_gate(ref: str) -> dict:
    return {"ref": ref, "facts": None, "findings": [], "problems": [], "tokens": 0,
            "blocked": False, "needs_human": False, "recommendation": None,
            "rationale": [], "approved_by": None, "released": False}


def pending(app, thread: str):
    """What is this thread waiting to do?"""
    return app.get_state(cfg(thread)).next
'''),
    code('''
# --- Self-check: Section 1   (a real checkpointed graph hitting a real interrupt -- no model)
def started(thread: str = "g1", ref: str = "PMT-1005"):
    """A fresh saver, a fresh thread, run until it stops."""
    RELEASED.clear()
    app = build(checkpointer=InMemorySaver())
    app.invoke(fresh_gate(ref), cfg(thread))
    return app

check("the run stops instead of finishing",
      lambda: pending(started(), "g1") == ("release",))
check("NOTHING WAS RELEASED",
      lambda: (started(), "PMT-1005" not in RELEASED)[1] is True,
      "the point of interrupting BEFORE the node rather than after it")
check("everything before the gate did run",
      lambda: len(started().get_state(cfg("g1")).values["findings"]) == 3)
check("so the human is looking at evidence, not at a blank form",
      lambda: started().get_state(cfg("g1")).values["recommendation"] == "hold for a human")
check("the checkpointer wrote a checkpoint per step, not one at the end",
      lambda: len(list(started().get_state_history(cfg("g1")))) >= 4,
      "that is what makes resuming, rewinding and auditing possible at all")
def _ungated():
    """The same graph compiled with no interrupt at all."""
    RELEASED.clear()
    app = build(checkpointer=InMemorySaver(), gate=False)
    app.invoke(fresh_gate("PMT-1002"), cfg("g0"))
    return app

check("an ungated build runs straight through to the end",
      lambda: pending(_ungated(), "g0") == (),
      "nothing pending means nothing stopped it -- and PMT-1002 has now been released")
check("two cases on two threads do not see each other",
      lambda: started("gA").get_state(cfg("gB")).values in ({}, None),
      "one thread per case is the whole of the isolation you get")
'''),

    md("""
## Section 2 &mdash; Approval is an identity, not a boolean

Resuming is two calls. First write what the human decided into the checkpoint with
`update_state`; then `invoke(None, cfg)` &mdash; **`None` means carry on**, not start again.

`node_release` above already refuses anything that is not a name. So the only question left is
what an approval has to put into state for it to be satisfied.
"""),
    code('''
def resume_unapproved(app, thread: str) -> dict:
    """Carry on with nothing written. The release node decides what to do about that."""
    return app.invoke(None, cfg(thread))


def approve(app, thread: str, approved_by: str) -> dict:
    """Record WHO approved, then carry on.

    'approved: true' answers none of the questions an auditor asks -- who, and on what
    evidence. So the caller must hand over a name, and the name goes into the state that
    the release node reads.
    """
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError("an approval needs a named human")
    app.update_state(cfg(thread), BLANK)   # TODO: the state update node_release is waiting for
    return app.invoke(None, cfg(thread))
''', '''
def resume_unapproved(app, thread: str) -> dict:
    """Carry on with nothing written. The release node decides what to do about that."""
    return app.invoke(None, cfg(thread))


def approve(app, thread: str, approved_by: str) -> dict:
    """Record WHO approved, then carry on.

    'approved: true' answers none of the questions an auditor asks -- who, and on what
    evidence. So the caller must hand over a name, and the name goes into the state that
    the release node reads.
    """
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError("an approval needs a named human")
    app.update_state(cfg(thread), {"approved_by": approved_by})
    return app.invoke(None, cfg(thread))
'''),
    code('''
# --- Self-check: Section 2   (real update_state, real resume -- no model)
def _unapproved():
    return resume_unapproved(started("s2a"), "s2a")

def _approved(who="ops-duty-manager"):
    app = started("s2b")
    return app, approve(app, "s2b", who)

def _refused_with(who):
    """Approving with something that is not a name must leave the world unchanged."""
    app = started("s2c")
    try:
        approve(app, "s2c", who)
    except NameError:
        raise
    except ValueError:
        pass
    return "PMT-1005" not in RELEASED

check("resuming with nothing written does not release",
      lambda: _unapproved()["released"] is False)
check("...and it says why, in state a person can read",
      lambda: any("no named human approver" in p for p in _unapproved()["problems"]))
check("a bare True is not an approver",
      lambda: _refused_with(True) is True,
      "'approved: true' cannot answer 'who approved this, and on what evidence?'")
check("nor is an empty string",
      lambda: _refused_with("   ") is True)
check("a named human opens the gate",
      lambda: _approved()[1]["released"] is True)
check("and the payment actually went",
      lambda: (_approved(), "PMT-1005" in RELEASED)[1] is True)
check("the approver's name is on the final state, not just a flag",
      lambda: _approved()[1]["approved_by"] == "ops-duty-manager")
check("after resuming there is nothing pending",
      lambda: pending(_approved()[0], "s2b") == ())
check("the work done before the pause was kept, not redone",
      lambda: _approved()[1]["tokens"]
              == COST["ledger"] + COST["policy"] + COST["sanctions"] + COST["writer"] + 40,
      "a pause is not a rollback -- the checkpoint carried the findings and the bill across")
'''),

    md("""
## Section 3 &mdash; Timeout, and a ladder that ends

A gate with no deadline is a run that waits for someone who has gone home. The ladder is given;
the decision is what the deadline passing actually *means*.
"""),
    code('''
ESCALATION = ["ops-duty-manager", "treasury-lead", "head-of-operations"]

def escalate(current, ladder=None):
    """Who to ask next. None means the ladder is exhausted and a person must own it manually."""
    ladder = ESCALATION if ladder is None else ladder
    if current is None:
        return ladder[0]
    if current not in ladder:
        return None
    i = ladder.index(current)
    return ladder[i + 1] if i + 1 < len(ladder) else None


def gate_status(waited_s: int, deadline_s: int, approver=None) -> str:
    """What to do with a gate that has been waiting."""
    if isinstance(approver, str) and approver.strip():
        return "approved"
    if waited_s < deadline_s:
        return "waiting"
    # The deadline passed and nobody answered. That is not a yes, and it is not a no.
    return BLANK          # TODO: one word, and it must not be "approved" or "refused"
''', '''
ESCALATION = ["ops-duty-manager", "treasury-lead", "head-of-operations"]

def escalate(current, ladder=None):
    """Who to ask next. None means the ladder is exhausted and a person must own it manually."""
    ladder = ESCALATION if ladder is None else ladder
    if current is None:
        return ladder[0]
    if current not in ladder:
        return None
    i = ladder.index(current)
    return ladder[i + 1] if i + 1 < len(ladder) else None


def gate_status(waited_s: int, deadline_s: int, approver=None) -> str:
    """What to do with a gate that has been waiting."""
    if isinstance(approver, str) and approver.strip():
        return "approved"
    if waited_s < deadline_s:
        return "waiting"
    # The deadline passed and nobody answered. That is not a yes, and it is not a no --
    # it is a different queue, and someone further up owns it now.
    return "escalate"
'''),
    code('''
# --- Self-check: Section 3
check("an unopened gate starts at the bottom of the ladder",
      lambda: escalate(None) == "ops-duty-manager")
check("and climbs one rung at a time",
      lambda: escalate("ops-duty-manager") == "treasury-lead")
check("the ladder TERMINATES",
      lambda: escalate("head-of-operations") is None,
      "an escalation path that loops is a gate that never resolves")
check("someone outside the ladder cannot be escalated from",
      lambda: escalate("a-passing-colleague") is None)
check("inside the deadline the gate simply waits",
      lambda: gate_status(waited_s=30, deadline_s=900) == "waiting")
check("an approval short-circuits the deadline entirely",
      lambda: gate_status(waited_s=99999, deadline_s=900, approver="treasury-lead") == "approved")
check("expiry is neither approval nor refusal",
      lambda: gate_status(waited_s=901, deadline_s=900) not in ("approved", "refused"),
      "a timeout that auto-approves is not a gate; one that auto-refuses loses real work")
check("and what it is instead is something the ladder can act on",
      lambda: gate_status(waited_s=901, deadline_s=900) == "escalate")

def _ladder():
    who, waited = None, 0
    while True:
        who = escalate(who)
        if who is None:
            print("  ladder exhausted -- this case now belongs to a person, not to the graph")
            break
        waited += 900
        print(f"  after {waited // 60:>3} min -> ask {who}  ({gate_status(waited, 900)})")
guard(_ladder)
'''),

    md("""
## Section 4 &mdash; Placement, and who the interrupt belongs to

Two questions decide whether you built a gate or a notification.

**Where is it?** At the moment the human says no, has anything irreversible already happened?

**Whose is it?** Rewind to an earlier checkpoint and replay. A run-scoped interrupt would sail
through; a graph-scoped one stops again. Find out which you have &mdash; this catches people out.
"""),
    code('''
def no_is_free(gate: bool, ref: str = "PMT-1005") -> bool:
    """Run it, have nobody approve, and ask whether anything happened anyway."""
    RELEASED.clear()
    app = build(checkpointer=InMemorySaver(), gate=gate)
    app.invoke(fresh_gate(ref), cfg("placement"))
    return ref not in RELEASED


def checkpoint_before(app, thread: str, node: str):
    """The config of the checkpoint at which `node` was the next thing to run."""
    for snap in app.get_state_history(cfg(thread)):
        if snap.next == (node,):
            return snap.config      # a config carrying that checkpoint_id, not just the thread
    return None


def rewind_and_replay(thread: str = "rw"):
    """Approve once, then go back to before the recommendation and run it again."""
    app = started(thread)
    approve(app, thread, "ops-duty-manager")            # it completed, once
    back = checkpoint_before(app, thread, "recommend")
    app.invoke(None, back)                              # replay from the older checkpoint
    return app
''', '''
def no_is_free(gate: bool, ref: str = "PMT-1005") -> bool:
    """Run it, have nobody approve, and ask whether anything happened anyway."""
    RELEASED.clear()
    app = build(checkpointer=InMemorySaver(), gate=gate)
    app.invoke(fresh_gate(ref), cfg("placement"))
    return ref not in RELEASED


def checkpoint_before(app, thread: str, node: str):
    """The config of the checkpoint at which `node` was the next thing to run."""
    for snap in app.get_state_history(cfg(thread)):
        if snap.next == (node,):
            return snap.config      # a config carrying that checkpoint_id, not just the thread
    return None


def rewind_and_replay(thread: str = "rw"):
    """Approve once, then go back to before the recommendation and run it again."""
    app = started(thread)
    approve(app, thread, "ops-duty-manager")            # it completed, once
    back = checkpoint_before(app, thread, "recommend")
    app.invoke(None, back)                              # replay from the older checkpoint
    return app
'''),
    code('''
# --- Self-check: Section 4
check("with the gate before the write, saying nothing costs nothing",
      lambda: no_is_free(gate=True) is True)
check("with the review after the write, the payment already went",
      lambda: no_is_free(gate=False) is False,
      "the reviewer sees a complete, sourced summary of something they can no longer stop")
check("both runs showed the reviewer exactly the same evidence",
      lambda: len(started("cmp").get_state(cfg("cmp")).values["findings"]) == 3,
      "quality of evidence was never the difference -- placement was")
check("checkpoint_before finds a real point in the past",
      lambda: checkpoint_before(started("cb"), "cb", "recommend") is not None)
check("what it returns addresses a checkpoint, not just the thread",
      lambda: "checkpoint_id" in checkpoint_before(started("cb2"), "cb2",
                                                   "recommend")["configurable"],
      "a config with only a thread_id points at NOW, which is not a rewind")
check("rewinding into a gated graph PAUSES AT THE GATE AGAIN",
      lambda: pending(rewind_and_replay("rw1"), "rw1") == ("release",),
      "the interrupt belongs to the compiled graph, not to a run -- every path through it "
      "pauses, including a replay of one that was already approved")
check("and the earlier approval did not come back with the rewind",
      lambda: rewind_and_replay("rw2").get_state(cfg("rw2")).values["approved_by"] is None,
      "you rewound to a checkpoint that predates the approval, so the person decides again")

def _placement():
    for label, gate in (("before the write", True), ("after the write ", False)):
        free = no_is_free(gate=gate)
        print(f"  gate {label}:  'no' still free? {'yes -- a gate' if free else 'NO -- a notification'}")
    app = rewind_and_replay("rw3")
    print(f"\\n  after the rewind the thread is pending: {pending(app, 'rw3')}")
guard(_placement)
'''),

    md("""
## Run it for real

Render the checkpoint the way a human reviewer would see it and ask the model to write the
approval request. What you are judging is whether the state you checkpointed contains enough for
a person to say no.
"""),
    code('''
if llm_ready():
    def _brief():
        app = started("brief")
        values = app.get_state(cfg("brief")).values
        evidence = "\\n".join(f"- [{f['by']}, source={f['source']}] {f['claim']}"
                             for f in values["findings"])
        reply = ask("Write a short approval request for a duty manager. State what is being asked, "
                    "the evidence for and against, and what happens if they do nothing.\\n\\n"
                    f"Action awaiting approval: {pending(app, 'brief')} {values['ref']}\\n"
                    f"Agent recommendation: {values.get('recommendation')}\\n"
                    f"Findings:\\n{evidence}")
        print(reply.strip()[:600])
    guard(_brief)
'''),
    md("""
### Read it

If the model has to hedge or invent, your checkpoint is missing something a reviewer needs &mdash;
and that is a state design problem, not a prompt problem. A good approval request is mostly a
rendering of state you already had.

**Section 4 is the one to look at twice.** The replay ran forward and stopped at the gate again,
even though that thread had already been approved once. The interrupt is a property of the
**compiled graph**, not of a run: every path through it pauses. People who assume otherwise build
a &ldquo;replay for audit&rdquo; feature and are surprised to find it asking for approvals.

**What you take from this lab:** interrupt before the node, not after it; record an identity
rather than a boolean; give every gate a deadline and a ladder that ends; and remember that the
refusal in `node_release` needed no checkpointer at all &mdash; the checkpointer bought you *later*,
not *safer*.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Swap `InMemorySaver` for `SqliteSaver` pointed at a file under `WORK`. Run to the gate,
   restart the kernel, rebuild against the same file and resume. That is recovery after a crash,
   and it is why in-memory is a development convenience only.
2. Gate on a **condition** rather than always: pause only when `needs_human` is true.
   `interrupt_before` is static, so this belongs in a conditional edge to a node that interrupts.
   Confirm PMT-1002 runs straight through.
3. `approve` trusts its caller for the name. Where does that name actually have to come from for
   the audit trail to mean anything, and what stops an agent from supplying it?
"""),
]


# =========================================================================== #
# Lab 5.5 -- challenge: the scorecard, and when the answer is "don't"
# =========================================================================== #
LAB5 = [
    header(5, "Challenge: The Scorecard", "Advanced &middot; challenge", 45,
           ["State the ground truth &mdash; what the right recommendation actually is, and why",
            "Run a single agent and a four-node <code>StateGraph</code> over the same cases",
            "Price both: quality, tokens, and the critical path &mdash; and check the multiple",
            "Turn it into a decision that names what a wrong answer costs"],
           "> **The module's deliverable.** Not a graph &mdash; an argument about whether to build one,\n"
           "> with numbers in it that a risk owner can agree or disagree with."),
    setup(5),
    code(DOMAIN),
    code(AGENTKIT),

    md("""
## Concept

You have a single agent from Day 1 and a graph from this module. The question is not which is
more sophisticated. It is whether the errors the graph prevents are worth the tokens it burns.

**What is graded here** is the scorecard machinery: the ground truth, the metric, the cost model
and the decision rule. The graph is a real compiled `StateGraph` and the single agent is a plain
function, both deterministic, so the comparison is exact and repeatable offline. Point the same
machinery at your own system and only the numbers move.
"""),

    md("""
## Section 1 &mdash; Ground truth

Before measuring anything, say what the right answer is. Two cases are added here: a watchlisted
counterparty whose reason code says nothing whatever about sanctions. Those are the cases that
separate the two designs, and real case files are full of them.
"""),
    code('''
EXTRA = {
    "PMT-1006": {"amount": 62000.00, "ccy": "USD", "counterparty": "NORTHWIND",
                 "status": "failed", "value_date": "2026-09-03",
                 "reason_code": "INSUFFICIENT_FUNDS"},
    "PMT-1007": {"amount":  8400.00, "ccy": "GBP", "counterparty": "NORTHWIND",
                 "status": "failed", "value_date": "2026-09-03",
                 "reason_code": "INVALID_IBAN"},
}
LEDGER.update(EXTRA)          # the specialists read LEDGER, so the new cases must be in it
CASES = dict(LEDGER)


def expected_recommendation(record: dict) -> str:
    """The correct answer for one payment, independent of any agent."""
    if record["status"] == "settled":
        return "no action"
    # Two independent conditions each force a hold. One is about why the payment failed;
    # the other is about who is being paid, and no reason code ever hints at it.
    if BLANK:                          # TODO: the two conditions, either of which holds it
        return "hold for a human"
    return "release"


def eval_cases() -> list:
    """The cases paired with their ground-truth answers. Built on demand, not at import:
    a module-level call into a function with a blank in it crashes the whole cell."""
    return [(ref, expected_recommendation(rec)) for ref, rec in sorted(CASES.items())]
''', '''
EXTRA = {
    "PMT-1006": {"amount": 62000.00, "ccy": "USD", "counterparty": "NORTHWIND",
                 "status": "failed", "value_date": "2026-09-03",
                 "reason_code": "INSUFFICIENT_FUNDS"},
    "PMT-1007": {"amount":  8400.00, "ccy": "GBP", "counterparty": "NORTHWIND",
                 "status": "failed", "value_date": "2026-09-03",
                 "reason_code": "INVALID_IBAN"},
}
LEDGER.update(EXTRA)          # the specialists read LEDGER, so the new cases must be in it
CASES = dict(LEDGER)


def expected_recommendation(record: dict) -> str:
    """The correct answer for one payment, independent of any agent."""
    if record["status"] == "settled":
        return "no action"
    # Two independent conditions each force a hold. One is about why the payment failed;
    # the other is about who is being paid, and no reason code ever hints at it.
    if record["reason_code"] in NEEDS_HUMAN or record["counterparty"] in SANCTIONS_WATCH:
        return "hold for a human"
    return "release"


def eval_cases() -> list:
    """The cases paired with their ground-truth answers. Built on demand, not at import:
    a module-level call into a function with a blank in it crashes the whole cell."""
    return [(ref, expected_recommendation(rec)) for ref, rec in sorted(CASES.items())]
'''),
    code('''
# --- Self-check: Section 1
check("a settled payment needs no action",
      lambda: expected_recommendation(CASES["PMT-1001"]) == "no action")
check("a sanctions review is held",
      lambda: expected_recommendation(CASES["PMT-1005"]) == "hold for a human")
check("so is a limit breach",
      lambda: expected_recommendation(CASES["PMT-1003"]) == "hold for a human")
check("an ordinary funding failure to an unlisted party is released",
      lambda: expected_recommendation(CASES["PMT-1002"]) == "release")
check("a watchlisted counterparty is held even when its reason code is mundane",
      lambda: expected_recommendation(CASES["PMT-1006"]) == "hold for a human",
      "nothing in INSUFFICIENT_FUNDS hints at sanctions -- the counterparty is the whole reason")
check("and again for the second one",
      lambda: expected_recommendation(CASES["PMT-1007"]) == "hold for a human")
check("the eval set covers all three outcomes",
      lambda: {e for _, e in eval_cases()} == {"no action", "hold for a human", "release"})
check("seven cases in total",
      lambda: len(eval_cases()) == 7)
'''),

    md("""
## Section 2 &mdash; The two designs

The single agent reads the payment, consults policy when there is a reason code, and screens the
counterparty **only when something in the case points at sanctions**. That is not a strawman: it
is what one prompt with a step budget does &mdash; it follows the happy path the case suggests.

The graph does not get to choose. Every specialist is on an unconditional edge, so every
specialist runs. That is the whole of its advantage, and the whole of its cost.
"""),
    code('''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

class ScoreState(TypedDict):
    ref: str
    facts: dict | None
    findings: Annotated[list, add]
    problems: Annotated[list, add]
    tokens: Annotated[int, add]
    blocked: bool
    needs_human: bool
    recommendation: str | None
    rationale: list


def build_scorecard_graph():
    """Every specialist, every time."""
    g = StateGraph(ScoreState)
    g.add_node("read", agent_ledger)
    g.add_node("policy", agent_policy)
    g.add_node("screen", agent_sanctions)
    g.add_node("recommend", agent_writer)
    g.add_edge(START, "read")
    g.add_edge("read", "policy")
    # The single agent screens only when the case points at sanctions. An unconditional edge
    # is the entire difference between the two designs -- so it has to be to the right node.
    g.add_edge("read", BLANK)          # TODO: which specialist must run whether the case
    #                                          looks like it needs one or not?
    g.add_edge("policy", "recommend")
    g.add_edge("screen", "recommend")
    g.add_edge("recommend", END)
    return g.compile()


def fresh_score(ref: str) -> dict:
    return {"ref": ref, "facts": None, "findings": [], "problems": [], "tokens": 0,
            "blocked": False, "needs_human": False, "recommendation": None, "rationale": []}


def single_agent(ref: str) -> dict:
    """One agent, one context. Reads, consults policy, screens only if prompted to."""
    record = CASES.get(ref)
    used = ["ledger"]
    if record is None:
        return {"recommendation": "no action", "used": used, "dispatches": 0}
    needs_human = False
    if record["reason_code"]:
        used.append("policy")
        needs_human = record["reason_code"] in NEEDS_HUMAN
    if record["reason_code"] == "SANCTIONS_REVIEW":     # the only prompt it ever gets
        used.append("sanctions")
        needs_human = needs_human or record["counterparty"] in SANCTIONS_WATCH
    used.append("writer")
    if record["status"] == "settled":
        action = "no action"
    elif needs_human:
        action = "hold for a human"
    else:
        action = "release"
    return {"recommendation": action, "used": used, "dispatches": 0}


def graph_agent(ref: str) -> dict:
    """Supervisor plus four specialists, as a compiled graph. Four dispatches, every time."""
    out = build_scorecard_graph().invoke(fresh_score(ref))
    return {"recommendation": out["recommendation"],
            "used": ["ledger", "policy", "sanctions", "writer"], "dispatches": 4}
''', '''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

class ScoreState(TypedDict):
    ref: str
    facts: dict | None
    findings: Annotated[list, add]
    problems: Annotated[list, add]
    tokens: Annotated[int, add]
    blocked: bool
    needs_human: bool
    recommendation: str | None
    rationale: list


def build_scorecard_graph():
    """Every specialist, every time."""
    g = StateGraph(ScoreState)
    g.add_node("read", agent_ledger)
    g.add_node("policy", agent_policy)
    g.add_node("screen", agent_sanctions)
    g.add_node("recommend", agent_writer)
    g.add_edge(START, "read")
    g.add_edge("read", "policy")
    # The single agent screens only when the case points at sanctions. An unconditional edge
    # is the entire difference between the two designs -- so it has to be to the right node.
    g.add_edge("read", "screen")
    g.add_edge("policy", "recommend")
    g.add_edge("screen", "recommend")
    g.add_edge("recommend", END)
    return g.compile()


def fresh_score(ref: str) -> dict:
    return {"ref": ref, "facts": None, "findings": [], "problems": [], "tokens": 0,
            "blocked": False, "needs_human": False, "recommendation": None, "rationale": []}


def single_agent(ref: str) -> dict:
    """One agent, one context. Reads, consults policy, screens only if prompted to."""
    record = CASES.get(ref)
    used = ["ledger"]
    if record is None:
        return {"recommendation": "no action", "used": used, "dispatches": 0}
    needs_human = False
    if record["reason_code"]:
        used.append("policy")
        needs_human = record["reason_code"] in NEEDS_HUMAN
    if record["reason_code"] == "SANCTIONS_REVIEW":     # the only prompt it ever gets
        used.append("sanctions")
        needs_human = needs_human or record["counterparty"] in SANCTIONS_WATCH
    used.append("writer")
    if record["status"] == "settled":
        action = "no action"
    elif needs_human:
        action = "hold for a human"
    else:
        action = "release"
    return {"recommendation": action, "used": used, "dispatches": 0}


def graph_agent(ref: str) -> dict:
    """Supervisor plus four specialists, as a compiled graph. Four dispatches, every time."""
    out = build_scorecard_graph().invoke(fresh_score(ref))
    return {"recommendation": out["recommendation"],
            "used": ["ledger", "policy", "sanctions", "writer"], "dispatches": 4}
'''),
    code('''
# --- Self-check: Section 2   (a real compiled graph over every case -- no model)
check("the scorecard graph compiles",
      lambda: build_scorecard_graph() is not None)
check("the screen runs even when the reason code says nothing about sanctions",
      lambda: any(f["by"] == "sanctions"
                  for f in build_scorecard_graph().invoke(fresh_score("PMT-1006"))["findings"]),
      "that unconditional edge IS the design difference this lab is about to price")
check("so the graph catches the watchlisted counterparty",
      lambda: graph_agent("PMT-1006")["recommendation"] == "hold for a human")
check("and the single agent does not, because nothing told it to look",
      lambda: single_agent("PMT-1006")["recommendation"] == "release")
check("on the case that DOES point at sanctions, both agree",
      lambda: single_agent("PMT-1005")["recommendation"]
              == graph_agent("PMT-1005")["recommendation"] == "hold for a human")

def _side_by_side():
    print(f"  {'case':10}{'expected':18}{'single':18}{'graph':18}")
    print("  " + "-" * 62)
    for ref, expected in eval_cases():
        s, g = single_agent(ref)["recommendation"], graph_agent(ref)["recommendation"]
        flag = "" if s == expected else "   <-- single is wrong"
        print(f"  {ref:10}{expected:18}{s:18}{g:18}{flag}")
guard(_side_by_side)
'''),

    md("""
## Section 3 &mdash; Score and price them

Quality is agreement with the ground truth. Cost is what each design woke up, plus &mdash; and this
is the assumption that does all the work &mdash; the case context re-sent to every specialist you
dispatch. Make that assumption visible, then vary it.
"""),
    code('''
CONTEXT_TOKENS = 600      # the case file re-sent at each hop: payment, policy text, findings

def run_cost(result: dict, context_tokens: int = CONTEXT_TOKENS) -> int:
    """Tokens for one case.

    A single agent holds ONE context and reuses it across its own turns. A graph re-sends the
    case context to every specialist it dispatches, and pays a supervisor to route each time.
    The re-sending is where a cost multiple comes from -- not from the agents themselves.
    """
    specialists = sum(COST[a] for a in result["used"])
    dispatches = result.get("dispatches", 0)
    contexts = dispatches if dispatches else 1
    return specialists + context_tokens * contexts + COST["supervisor"] * dispatches


def evaluate(design, context_tokens: int = CONTEXT_TOKENS) -> dict:
    """Run one design over every case. Returns correct count, accuracy and total tokens."""
    correct, tokens, misses = 0, 0, []
    for ref, expected in eval_cases():
        result = design(ref)
        tokens += run_cost(result, context_tokens)
        if result["recommendation"] == expected:
            correct += 1
        else:
            misses.append((ref, expected, result["recommendation"]))
    return {"correct": correct, "accuracy": correct / len(eval_cases()),
            "tokens": tokens, "misses": misses}


def critical_path(design_name: str) -> int:
    """Supersteps on the critical path -- what parallelism can and cannot shorten."""
    return 2 if design_name == "single" else 3
'''),
    code('''
# --- Self-check: Section 3
_cache = {}
def S():
    if "s" not in _cache:
        _cache["s"] = evaluate(single_agent)
    return _cache["s"]
def G():
    if "g" not in _cache:
        _cache["g"] = evaluate(graph_agent)
    return _cache["g"]

check("the graph gets every case right",
      lambda: G()["correct"] == len(eval_cases()))
check("the single agent does not",
      lambda: S()["correct"] < len(eval_cases()))
check("and it misses exactly the two watchlisted-but-mundane cases",
      lambda: sorted(r for r, _, _ in S()["misses"]) == ["PMT-1006", "PMT-1007"],
      "the cases where nothing in the reason code told it to look")
check("both of its misses are the dangerous direction -- releasing when it should hold",
      lambda: all(got == "release" and want == "hold for a human"
                  for _, want, got in S()["misses"]))
check("with the case context re-sent at every hop, the graph costs about twice as much",
      lambda: 1.5 < G()["tokens"] / S()["tokens"] < 3.0)
check("with nothing re-sent, the two bills are close -- the multiple IS the re-sending",
      lambda: evaluate(graph_agent, 0)["tokens"] / evaluate(single_agent, 0)["tokens"] < 1.6,
      "measured on this sandbox the token totals for one agent and several came out level "
      "(9,794 vs 9,720); what doubled was the number of CALLS. Do not quote a 3-10x token "
      "multiple you have not measured on your own prompts")
check("the graph's critical path is longer, and no parallelism shortens it",
      lambda: critical_path("graph") > critical_path("single"))

def _score():
    s, g = S(), G()
    print(f"  {'':18}{'single':>12}{'graph':>12}")
    print("  " + "-" * 42)
    print(f"  {'accuracy':18}{s['accuracy']:>11.0%}{g['accuracy']:>12.0%}")
    print(f"  {'tokens':18}{s['tokens']:>12}{g['tokens']:>12}")
    print(f"  {'cost multiple':18}{'1.0x':>12}{g['tokens'] / s['tokens']:>11.1f}x")
    print(f"  {'...with no re-send':18}"
          f"{'1.0x':>12}"
          f"{evaluate(graph_agent, 0)['tokens'] / evaluate(single_agent, 0)['tokens']:>11.1f}x")
    print(f"  {'critical path':18}{critical_path('single'):>12}{critical_path('graph'):>12}")
guard(_score)
'''),

    md("""
## Section 4 &mdash; The decision

Two numbers finish the argument, and neither is technical: what one wrong recommendation costs to
put right, and what a token costs. Put your own figures in.
"""),
    code('''
TOKEN_PRICE = 0.0000006      # currency per token -- substitute your own
ERROR_COST  = 2500.0         # what putting one wrong recommendation right costs you

def verdict(single: dict, graph: dict,
            error_cost: float = ERROR_COST, token_price: float = TOKEN_PRICE) -> dict:
    """Ship the graph only if the errors it prevents are worth more than the tokens it burns."""
    errors_prevented = graph["correct"] - single["correct"]
    value_saved = errors_prevented * error_cost
    extra_spend = (graph["tokens"] - single["tokens"]) * token_price
    return {"errors_prevented": errors_prevented,
            "value_saved": round(value_saved, 4),
            "extra_spend": round(extra_spend, 4),
            # The whole module comes down to one comparison between those two numbers.
            # "don't" has to be a real possible answer, or this is not a decision rule.
            "decision": BLANK}      # TODO: "ship the graph" or "don't", from the numbers above


def breakeven_error_cost(single: dict, graph: dict,
                         token_price: float = TOKEN_PRICE) -> float:
    """How expensive one error has to be before the graph pays for itself."""
    prevented = graph["correct"] - single["correct"]
    if prevented <= 0:
        return float("inf")
    return (graph["tokens"] - single["tokens"]) * token_price / prevented
''', '''
TOKEN_PRICE = 0.0000006      # currency per token -- substitute your own
ERROR_COST  = 2500.0         # what putting one wrong recommendation right costs you

def verdict(single: dict, graph: dict,
            error_cost: float = ERROR_COST, token_price: float = TOKEN_PRICE) -> dict:
    """Ship the graph only if the errors it prevents are worth more than the tokens it burns."""
    errors_prevented = graph["correct"] - single["correct"]
    value_saved = errors_prevented * error_cost
    extra_spend = (graph["tokens"] - single["tokens"]) * token_price
    return {"errors_prevented": errors_prevented,
            "value_saved": round(value_saved, 4),
            "extra_spend": round(extra_spend, 4),
            "decision": "ship the graph" if value_saved > extra_spend else "don't"}


def breakeven_error_cost(single: dict, graph: dict,
                         token_price: float = TOKEN_PRICE) -> float:
    """How expensive one error has to be before the graph pays for itself."""
    prevented = graph["correct"] - single["correct"]
    if prevented <= 0:
        return float("inf")
    return (graph["tokens"] - single["tokens"]) * token_price / prevented
'''),
    code('''
# --- Self-check: Section 4
check("the graph prevents two errors on this eval set",
      lambda: verdict(S(), G())["errors_prevented"] == 2)
check("at a realistic error cost, ship it",
      lambda: verdict(S(), G())["decision"] == "ship the graph")
check("if an error costs almost nothing, do not",
      lambda: verdict(S(), G(), error_cost=0.0001)["decision"] == "don't",
      "the same graph, the same quality gain, the opposite answer -- the economics decide")
check("the breakeven is a number you can quote",
      lambda: 0 < breakeven_error_cost(S(), G()) < ERROR_COST)
check("and the decision flips either side of it",
      lambda: verdict(S(), G(), error_cost=breakeven_error_cost(S(), G()) * 2)["decision"]
              == "ship the graph"
          and verdict(S(), G(), error_cost=breakeven_error_cost(S(), G()) / 2)["decision"]
              == "don't")
check("a graph that prevents nothing never pays, at any error cost",
      lambda: breakeven_error_cost(G(), G()) == float("inf"),
      "two designs of equal quality are decided on cost alone, and the cheaper one wins")

def _verdict():
    v = verdict(S(), G())
    print(f"  errors prevented per {len(eval_cases())} cases : {v['errors_prevented']}")
    print(f"  value saved                     : {v['value_saved']}")
    print(f"  extra spend                     : {v['extra_spend']}")
    print(f"  breakeven cost of one error     : {breakeven_error_cost(S(), G()):.4f}")
    print()
    print(f"  DECISION: {v['decision']}")
    print()
    print("  Read it as: the graph pays for itself as long as one wrong recommendation")
    print(f"  costs more than {breakeven_error_cost(S(), G()):.4f} to put right.")
guard(_verdict)
'''),

    md("""
## Run it for real

Everything above is deterministic, which is what makes it repeatable. Now run the same seven cases
through the model twice and see how stable the answer is &mdash; because a quality number from a
single run of a non-deterministic system is an anecdote.
"""),
    code('''
if llm_ready():
    def _stability():
        def model_recommendation(ref):
            rec = CASES[ref]
            reply = ask("You are a payments operations agent. Reply with exactly one of: "
                        "no action / hold for a human / release.\\n\\n"
                        f"Payment: {json.dumps(rec)}\\n"
                        f"Reference: {ref}\\n"
                        f"Watchlisted counterparties: {sorted(SANCTIONS_WATCH)}\\n"
                        f"Reason codes that require a human: {sorted(NEEDS_HUMAN)}",
                        system="Reply with the phrase alone.")
            return (reply or "").strip().lower().rstrip(".")
        for run in (1, 2):
            correct = sum(1 for ref, expected in eval_cases()
                          if model_recommendation(ref) == expected)
            print(f"  run {run}: {correct}/{len(eval_cases())} correct")
    guard(_stability)
'''),
    md("""
### Read it

If the two runs disagree, you have just met Module 7's opening problem: the same input, twice, two
different answers. That does not invalidate the scorecard &mdash; it tells you the scorecard needs
repeats and a confidence interval, which is Day 3's work.

**And look hard at the two cost rows.** With the case context re-sent at every hop the graph costs
roughly double; with nothing re-sent the two bills are within half a multiple of each other. The
&ldquo;multi-agent costs 3&ndash;10x&rdquo; figure you will read online is a claim about context
management, not about agents &mdash; measured on this sandbox the token totals came out level and it
was the **call count** that doubled. Measure your own before you quote anyone's.

**What you take from Module 5:** a supervisor is a conditional edge you can measure; a handoff
carries only what you put in it; parallel branches lose findings unless you declare a reducer;
disagreement is settled by authority and provenance, not by counting; the human is a node with a
deadline and an escalation ladder; and the graph earns its place with a number or it does not earn
it at all.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Route by value: run the graph only above a threshold and the single agent below it. Find the
   threshold that maximises value, and check whether it is one you would defend to a regulator.
2. `single_agent` misses the two cases nothing prompted it to look at. Fix it with one sentence of
   prompt &mdash; &ldquo;always screen the counterparty&rdquo; &mdash; and re-score. If that closes the gap, the
   honest answer for this workload is that you never needed the graph.
3. The scorecard has no row for operating cost: five nodes are five things to trace, alert on and
   page someone about. Add that row in whatever unit you can defend, and see whether the decision
   survives it.
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-5-01-supervisor-as-router",           LAB1),
    ("lab-5-02-decomposition-and-handoffs",     LAB2),
    ("lab-5-03-parallel-reducers-disagreement", LAB3),
    ("lab-5-04-human-in-the-loop",              LAB4),
    ("lab-5-05-challenge-the-scorecard",        LAB5),
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
