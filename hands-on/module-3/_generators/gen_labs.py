#!/usr/bin/env python3
"""
Generate Module 3 lab notebooks and their solutions from one source.

Module 3 is EIGHT small labs, one concept each, in the order the flagship's LangGraph
session established: first graph -> multi-step -> conditional routing -> reducers ->
cycles and retry -> checkpointing -> human-in-the-loop -> challenge. It replaced five
much larger labs on 2026-09-09; the two that carried the LangGraph substrate were
~300 LOC apiece and put reducers in front of a participant who had never watched a
one-node graph run.

The domain is a leave-request workflow in a small HR system -- chosen because every
participant already knows the rules, so the only new thing in the notebook is LangGraph.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-3-0N-*.ipynb and ../solutions/

Design rules (revised 2026-09-08 -- framework-forward Day 1):
  * The participant writes REAL LangChain / LangGraph code in every lab. The frameworks
    are the learning, not an optional appendix.
  * Self-checks assert on framework OBJECTS -- a compiled graph, a bound tool, an emitted
    tool_call -- which is deterministic and needs no endpoint. Only model INVOCATION needs
    the gateway, and that lives in "Run it for real" cells, which are observed, not scored.
  * The score line is feedback, not a gate. Do not let it shape what a lab teaches.
  * "BLANK" marks a blank; an unfilled blank raises NameError and prints [TODO].
    NOT three underscores: IPython PREDEFINES _, __ and ___ as its output history
    (they start as ""), so under a real Jupyter kernel that token is a defined empty
    string, not an undefined name. The NameError never fires, [TODO] silently becomes
    [FAIL], and a blank used as a loop guard never stops its loop.
  * Blanks live INSIDE function bodies, and anything that builds a framework object at
    module level is wrapped in guard(), so an untouched lab survives Run All.
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

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 1 &middot; Module 3 &mdash; LangGraph: Stateful Agent Workflows**

### What you'll do
{items}

> **How this lab works.** You write real LangChain and LangGraph code. Fill every `BLANK`,
> then run the **Self-check** cell under each section &mdash; those check the *objects you built*
> (a bound tool, a compiled graph, an emitted tool call), so they are deterministic and do not
> depend on the model. Cells marked **Run it for real** put your code in front of the sandbox
> model; that is the part worth watching. The score line is feedback, not a grade.

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
    except NameError as exc:
        print(f"(a blank above is still unfilled: {{exc}} -- fill it in, then re-run this cell)")
        return default

def score() -> None:
    done = [r for r in _results if r is not None]
    passed = sum(1 for r in done if r)
    todo = sum(1 for r in _results if r is None)
    print(f"\\nSelf-check: {{passed}}/{{len(_results)}}" + (f"   ({{todo}} still TODO)" if todo else ""))

# ---- the sandbox model ---------------------------------------------------
# Your sandbox already has an LLM configured -- nothing to install, no key to register.
# These values are read from the environment so this notebook never hardcodes an endpoint.
LLM_BASE_URL = (os.environ.get("LAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("LITELLM_BASE_URL"))
LLM_MODEL    = (os.environ.get("LAB_LLM_MODEL") or os.environ.get("OPENAI_MODEL")
                or os.environ.get("LITELLM_MODEL"))
LLM_API_KEY  = os.environ.get("OPENAI_API_KEY", "sandbox")

# The served model reasons before it answers, and the reasoning is billed as completion
# tokens: 24.1s / 980 tokens with it on, 0.7s / 29 with it off, for the same answer. Off is
# the default here because you will make a lot of calls today. Pass think=True to see the
# difference for yourself -- and note that prompts written as an explicit ordered procedure
# survive thinking being off, while vague ones do not.
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

def show_messages(messages, width: int = 88) -> None:
    """Print a message list the way a trace reads: type, content, and any tool calls."""
    for m in messages:
        kind = getattr(m, "type", "?")
        body = str(getattr(m, "content", "")).replace("\\n", " ")[:width]
        calls = getattr(m, "tool_calls", None)
        line = f"  [{{kind:9}}] {{body}}"
        if calls:
            line += "  -> calls: " + ", ".join(f"{{c['name']}}({{c['args']}})" for c in calls)
        print(line)

print("work dir:", WORK)
print("model   :", LLM_MODEL or "(not configured -- the object-level self-checks still work)")
'''


def setup(num, extra=""):
    return code(SETUP_COMMON.format(num=num) + extra)


# --------------------------------------------------------------------------- #
# the shared domain -- deliberately one flat dict, so no lab spends time on
# Python plumbing. Every lookup in every lab is REQUESTS[rid].
# --------------------------------------------------------------------------- #
DOMAIN = '''
# ------------------------------------------------ the case file (synthetic, self-contained)
# Leave requests in a small HR system. Ordinary rules on purpose: the only new thing in these
# eight labs is LangGraph. One flat dict -- no joins, no helpers, nothing to learn here.

REQUESTS = {
    "LV-5001": {"who": "Priya Nair",   "days":  3, "kind": "annual", "reason": "family wedding",
                "balance": 12, "manager": "Devi R."},
    "LV-5002": {"who": "Rahul Menon",  "days":  5, "kind": "annual", "reason": "",
                "balance":  3, "manager": "Devi R."},
    "LV-5003": {"who": "Anita Sharma", "days":  2, "kind": "annual", "reason": "moving house",
                "balance":  0, "manager": "Sam O."},
    "LV-5004": {"who": "Vikram Rao",   "days": 15, "kind": "annual", "reason": "sabbatical",
                "balance": 20, "manager": "Sam O."},
    "LV-5005": {"who": "Priya Nair",   "days":  1, "kind": "sick",   "reason": "flu",
                "balance": 12, "manager": "Devi R."},
}

# The handbook, as three numbers. Every routing decision in this module comes from these.
POLICY = {"manager_over_days": 2, "hr_over_days": 10, "max_clarifications": 2}

print(len(REQUESTS), "leave requests loaded")
'''

THREAD_NOTE = (
    "> **The thread.** All eight Module 3 labs work one case: leave requests in a small HR\n"
    "> system. The rules are ordinary on purpose &mdash; the only new thing here is LangGraph."
)


# =========================================================================== #
# Lab 3.1 -- your first graph
# =========================================================================== #
LAB1 = [
    header(1, "Your First Graph", "Intermediate", 20,
           ["Declare workflow state as a <code>TypedDict</code>",
            "Write a node &mdash; a plain function, state in, <i>partial</i> state out",
            "Wire <code>START</code> and <code>END</code>, then <code>compile()</code> and <code>invoke()</code>"],
           THREAD_NOTE),
    setup(1),
    code(DOMAIN),

    md("""
## Concept

`create_agent` from Module 1 is one fixed loop: model, tools, repeat. When the control flow is
yours &mdash; branch here, go round again there, stop for a person &mdash; you want the thing
underneath it.

A `StateGraph` has three parts and no more:

| Part | What it is |
|---|---|
| **state** | a `TypedDict`. One shared object every node reads and writes |
| **nodes** | plain functions: `state -> partial state` |
| **edges** | what runs next. `START` is the way in, `END` the way out |

**There is no model in this lab.** That is the point: the graph is control flow, and control flow
is ordinary software you can test. The model goes *inside one node*, from Lab 3.5 on.
"""),

    md("""
## Section 1 &mdash; State, and a node that returns part of it

A node gets the **whole** state and returns **only the keys it changed**. LangGraph merges the rest.

Returning the whole state is the classic first mistake: it works until two nodes run and one
writes back a stale copy of what the other just changed. Both candidates are written out below.
"""),

    code('''
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class LeaveState(TypedDict):
    request_id: str
    summary: str
    notes: list


def summarise(state: LeaveState) -> dict:
    r = REQUESTS[state["request_id"]]
    line = f'{r["who"]}: {r["days"]} day(s) of {r["kind"]} leave'

    whole_state  = {**state, "summary": line, "notes": state["notes"] + ["summarise"]}
    only_changed = {"summary": line, "notes": state["notes"] + ["summarise"]}

    return BLANK          # TODO: which one does a LangGraph node return?
''', '''
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class LeaveState(TypedDict):
    request_id: str
    summary: str
    notes: list


def summarise(state: LeaveState) -> dict:
    r = REQUESTS[state["request_id"]]
    line = f'{r["who"]}: {r["days"]} day(s) of {r["kind"]} leave'

    whole_state  = {**state, "summary": line, "notes": state["notes"] + ["summarise"]}
    only_changed = {"summary": line, "notes": state["notes"] + ["summarise"]}

    return only_changed   # a node returns ONLY the keys it changed
'''),

    code('''
# --- Self-check: Section 1
check("summarise() fills in a readable summary",
      lambda: "Priya Nair" in summarise({"request_id": "LV-5001", "notes": []})["summary"])
check("summarise() returns ONLY the keys it changed",
      lambda: set(summarise({"request_id": "LV-5001", "notes": []})) == {"summary", "notes"},
      "request_id was not changed by this node, so it should not be in the return value")
score()
'''),

    md("""
## Section 2 &mdash; Edges, compile, invoke

Four lines build it and one runs it. `START` and `END` are ordinary edge endpoints, not settings.
"""),

    code('''
def build_graph():
    builder = StateGraph(LeaveState)
    builder.add_node("summarise", summarise)
    builder.add_edge(BLANK, "summarise")   # TODO: what runs before the first node?
    builder.add_edge("summarise", BLANK)   # TODO: and what marks the run finished?
    return builder.compile()
''', '''
def build_graph():
    builder = StateGraph(LeaveState)
    builder.add_node("summarise", summarise)
    builder.add_edge(START, "summarise")   # START is the way in
    builder.add_edge("summarise", END)     # END is the way out
    return builder.compile()
'''),

    code('''
# --- Self-check: Section 2   (a real compiled graph, really invoked -- no model in it)
def run(rid):
    return build_graph().invoke({"request_id": rid, "summary": "", "notes": []})

check("the graph compiles and runs end to end",
      lambda: run("LV-5001")["summary"].startswith("Priya Nair"),
      "both edges are needed: START -> summarise -> END")
check("state you did not touch survives the run",
      lambda: run("LV-5004")["request_id"] == "LV-5004",
      "LangGraph merged request_id through for you -- that is what partial state buys")
score()
'''),

    md("""
## Watch it run
"""),

    code('''
app = guard(build_graph)

if app is not None:
    for rid in ["LV-5001", "LV-5004"]:
        out = app.invoke({"request_id": rid, "summary": "", "notes": []})
        print(f"{rid} -> {out['summary']}   notes={out['notes']}")
    g = app.get_graph()
    print("\\nthe graph you just built:")
    for e in g.edges:
        print(f"   {e.source} -> {e.target}")
'''),

    md("""
### Read it

The state came back whole &mdash; your node returned two keys and `request_id` is still there, and
you wrote no merge code. And the graph is deterministic: same input, same output, every time.
Put it under a unit test and it will never flake.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Add a second node, `stamp_policy`, that appends `" -- needs manager"` to `summary` when `days`
   exceeds `POLICY["manager_over_days"]`. Wire `START -> summarise -> stamp_policy -> END`.
2. Delete the `add_edge(START, ...)` line and re-compile. Read the error once, deliberately,
   rather than at 4pm on Day 3.
"""),
]


# =========================================================================== #
# Lab 3.2 -- a multi-step workflow
# =========================================================================== #
LAB2 = [
    header(2, "A Multi-Step Workflow", "Intermediate", 20,
           ["Chain three nodes so each reads what the last one wrote",
            "Discover that the <i>order</i> of your edges is a real design decision",
            "Watch a run with <code>stream()</code>, one node at a time"],
           THREAD_NOTE),
    setup(2),
    code(DOMAIN),

    md("""
## Concept

A node can read anything an earlier node wrote &mdash; the state is the channel between them.

That gives you something a message list never did: a **dependency order**. `draft` cannot run
before `check`, because it reads a field `check` creates. In a conversation that constraint lives
in your head. In a graph it lives in the edges, and breaking it is a `KeyError` on the first run
rather than a confident answer built on a field nobody set.
"""),

    md("""
## Section 1 &mdash; Three nodes

`check` never looks at `REQUESTS`. It reads `days` and `balance` out of the **state**, because
`summarise` put them there.
"""),

    code('''
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class LeaveState(TypedDict):
    request_id: str
    days: int
    balance: int
    covered: bool        # created by check
    needs_manager: bool  # created by check
    decision: str        # created by draft
    notes: list


def summarise(state: LeaveState) -> dict:
    r = REQUESTS[state["request_id"]]
    return {"days": r["days"], "balance": r["balance"],
            "notes": state["notes"] + ["summarise"]}


def check_request(state: LeaveState) -> dict:
    return {"covered": state["balance"] >= state["days"],
            "needs_manager": state["days"] > POLICY[BLANK],   # TODO: which handbook rule?
            "notes": state["notes"] + ["check"]}


def draft(state: LeaveState) -> dict:
    if not state["covered"]:
        text = "declined: not enough balance"
    elif state["needs_manager"]:
        text = f'pending: {state["days"]} days needs manager approval'
    else:
        text = "approved automatically"
    return {"decision": text, "notes": state["notes"] + ["draft"]}
''', '''
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class LeaveState(TypedDict):
    request_id: str
    days: int
    balance: int
    covered: bool        # created by check
    needs_manager: bool  # created by check
    decision: str        # created by draft
    notes: list


def summarise(state: LeaveState) -> dict:
    r = REQUESTS[state["request_id"]]
    return {"days": r["days"], "balance": r["balance"],
            "notes": state["notes"] + ["summarise"]}


def check_request(state: LeaveState) -> dict:
    return {"covered": state["balance"] >= state["days"],
            "needs_manager": state["days"] > POLICY["manager_over_days"],
            "notes": state["notes"] + ["check"]}


def draft(state: LeaveState) -> dict:
    if not state["covered"]:
        text = "declined: not enough balance"
    elif state["needs_manager"]:
        text = f'pending: {state["days"]} days needs manager approval'
    else:
        text = "approved automatically"
    return {"decision": text, "notes": state["notes"] + ["draft"]}
'''),

    code('''
# --- Self-check: Section 1   (the middle node on its own)
check("2 days does not need a manager, 3 days does",
      lambda: check_request({"days": 2, "balance": 9, "notes": []})["needs_manager"] is False
          and check_request({"days": 3, "balance": 9, "notes": []})["needs_manager"] is True,
      "POLICY['manager_over_days'] is 2, and the comparison is strictly greater than")
check("a 5-day request against a 3-day balance is not covered",
      lambda: check_request({"days": 5, "balance": 3, "notes": []})["covered"] is False)
score()
'''),

    md("""
## Section 2 &mdash; The edges that impose the order

`draft` reads `covered` and `needs_manager` &mdash; fields that **do not exist** until `check` has
run. The edges are what guarantee they are there.
"""),

    code('''
def build_chain():
    builder = StateGraph(LeaveState)
    builder.add_node("summarise", summarise)
    builder.add_node("check", check_request)
    builder.add_node("draft", draft)

    builder.add_edge(START, "summarise")
    builder.add_edge("summarise", BLANK)   # TODO: which node must run second, and why?
    builder.add_edge(BLANK, "draft")       # TODO: what must be finished before this one?
    builder.add_edge("draft", END)
    return builder.compile()
''', '''
def build_chain():
    builder = StateGraph(LeaveState)
    builder.add_node("summarise", summarise)
    builder.add_node("check", check_request)
    builder.add_node("draft", draft)

    builder.add_edge(START, "summarise")
    builder.add_edge("summarise", "check")   # covered / needs_manager are created here
    builder.add_edge("check", "draft")       # ...and read here
    builder.add_edge("draft", END)
    return builder.compile()
'''),

    code('''
# --- Self-check: Section 2   (three real nodes in a real compiled graph)
def run(rid):
    return build_chain().invoke({"request_id": rid, "notes": []})

check("all three nodes ran, in dependency order",
      lambda: run("LV-5001")["notes"] == ["summarise", "check", "draft"],
      "notes is the run order -- a missing node means a missing edge")
check("LV-5003 declined, LV-5005 auto-approved, LV-5004 pending",
      lambda: (run("LV-5003")["decision"].startswith("declined")
               and run("LV-5005")["decision"] == "approved automatically"
               and run("LV-5004")["decision"].startswith("pending")))
score()
'''),

    md("""
## Watch it run

`stream()` gives you the run a node at a time instead of only the final state &mdash; the view you
want when a workflow does something you did not expect.
"""),

    code('''
app = guard(build_chain)

if app is not None:
    for rid in ["LV-5001", "LV-5003", "LV-5004"]:
        print(f"\\n=== {rid} ===")
        for step in app.stream({"request_id": rid, "notes": []}):
            for node, update in step.items():
                print(f"  {node:10} wrote: {', '.join(k for k in update if k != 'notes')}")
'''),

    md("""
### Read it

`summarise` produced `days` and `balance`; `check` consumed them and produced `covered` and
`needs_manager`; `draft` consumed those. The state is a pipeline and every arrow in it is one
`add_edge` line &mdash; which is why swapping two edges gives you `KeyError: 'covered'` on the
first run rather than a plausible answer later.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Wire `summarise -> draft -> check` deliberately and run it. That crash is the design working.
2. `LV-5002` has an empty `reason`. Add a `validate` node at the front writing `complete: bool`,
   and have `draft` refuse when it is false. You have just invented Lab 3.5's loop.
"""),
]


# =========================================================================== #
# Lab 3.3 -- conditional routing
# =========================================================================== #
LAB3 = [
    header(3, "Conditional Routing", "Intermediate", 25,
           ["Write a routing function &mdash; state in, the name of the next node out",
            "Attach it with <code>add_conditional_edges</code> and a branch map",
            "Send three requests down three paths through one graph"],
           THREAD_NOTE),
    setup(3),
    code(DOMAIN),

    md("""
## Concept

Every edge so far was unconditional: after A, always B. A **conditional edge** asks first.

```python
builder.add_conditional_edges("check", route, {"auto": "approve", "no": "decline"})
```

Three arguments: the node the decision happens **after**, a routing function, and a **branch map**
from the strings the router returns to the nodes they mean.

Two things to get right from the start:

- The router **reads state and returns a string**. It does no work, calls no model, changes
  nothing. Keep it that way and your routing stays unit-testable.
- It hangs off **the node that produced the facts it reads**. Attach it earlier and it decides on
  fields that do not exist yet.

Module 5's supervisor is this function with a model choosing the string. You are writing one now.
"""),

    md("""
## Section 1 &mdash; The router

The `if`/`elif` is written for you. The decision is which branch each situation belongs to &mdash;
and the order, because someone with no balance left should be declined *whether or not* the
request is long enough to need a manager.
"""),

    code('''
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class LeaveState(TypedDict):
    request_id: str
    days: int
    balance: int
    covered: bool
    needs_manager: bool
    decision: str
    notes: list


def route(state: LeaveState) -> str:
    """Which node runs next? Returns a branch name and nothing else."""
    if not state["covered"]:
        return BLANK       # TODO: no balance to cover it -- which branch?
    if state["needs_manager"]:
        return BLANK       # TODO: covered, but too long to decide alone -- which branch?
    return "auto"
''', '''
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class LeaveState(TypedDict):
    request_id: str
    days: int
    balance: int
    covered: bool
    needs_manager: bool
    decision: str
    notes: list


def route(state: LeaveState) -> str:
    """Which node runs next? Returns a branch name and nothing else."""
    if not state["covered"]:
        return "decline"   # checked first: no balance beats every other consideration
    if state["needs_manager"]:
        return "manager"
    return "auto"
'''),

    code('''
# --- Self-check: Section 1   (a pure function, so this is four dicts)
check("no balance -> decline, whatever else is true",
      lambda: route({"covered": False, "needs_manager": False}) == "decline"
          and route({"covered": False, "needs_manager": True}) == "decline",
      "there is nothing for a manager to approve if the balance is not there")
check("covered and long -> manager; covered and short -> auto",
      lambda: route({"covered": True, "needs_manager": True}) == "manager"
          and route({"covered": True, "needs_manager": False}) == "auto")
score()
'''),

    md("""
## Section 2 &mdash; Attaching it

The branch map is why the router can return a short name like `"auto"` without knowing what the
node is called &mdash; routing logic and graph layout stay separable.
"""),

    code('''
def summarise(state):
    r = REQUESTS[state["request_id"]]
    return {"days": r["days"], "balance": r["balance"], "notes": state["notes"] + ["summarise"]}

def check_request(state):
    return {"covered": state["balance"] >= state["days"],
            "needs_manager": state["days"] > POLICY["manager_over_days"],
            "notes": state["notes"] + ["check"]}

def approve(state):  return {"decision": "approved automatically",  "notes": state["notes"] + ["approve"]}
def to_manager(s):   return {"decision": f'queued for {REQUESTS[s["request_id"]]["manager"]}',
                             "notes": s["notes"] + ["to_manager"]}
def decline(state):  return {"decision": "declined: not enough balance", "notes": state["notes"] + ["decline"]}


def build_router_graph():
    builder = StateGraph(LeaveState)
    for name, fn in [("summarise", summarise), ("check", check_request), ("approve", approve),
                     ("to_manager", to_manager), ("decline", decline)]:
        builder.add_node(name, fn)

    builder.add_edge(START, "summarise")
    builder.add_edge("summarise", "check")

    builder.add_conditional_edges(
        BLANK,                                   # TODO: after WHICH node is the decision made?
        route,
        {"auto": "approve", "manager": "to_manager", "decline": "decline"},
    )

    for name in ["approve", "to_manager", "decline"]:
        builder.add_edge(name, END)
    return builder.compile()
''', '''
def summarise(state):
    r = REQUESTS[state["request_id"]]
    return {"days": r["days"], "balance": r["balance"], "notes": state["notes"] + ["summarise"]}

def check_request(state):
    return {"covered": state["balance"] >= state["days"],
            "needs_manager": state["days"] > POLICY["manager_over_days"],
            "notes": state["notes"] + ["check"]}

def approve(state):  return {"decision": "approved automatically",  "notes": state["notes"] + ["approve"]}
def to_manager(s):   return {"decision": f'queued for {REQUESTS[s["request_id"]]["manager"]}',
                             "notes": s["notes"] + ["to_manager"]}
def decline(state):  return {"decision": "declined: not enough balance", "notes": state["notes"] + ["decline"]}


def build_router_graph():
    builder = StateGraph(LeaveState)
    for name, fn in [("summarise", summarise), ("check", check_request), ("approve", approve),
                     ("to_manager", to_manager), ("decline", decline)]:
        builder.add_node(name, fn)

    builder.add_edge(START, "summarise")
    builder.add_edge("summarise", "check")

    builder.add_conditional_edges(
        "check",                                 # covered / needs_manager are written here
        route,
        {"auto": "approve", "manager": "to_manager", "decline": "decline"},
    )

    for name in ["approve", "to_manager", "decline"]:
        builder.add_edge(name, END)
    return builder.compile()
'''),

    code('''
# --- Self-check: Section 2   (one real graph, three real paths, no model)
def run(rid):
    return build_router_graph().invoke({"request_id": rid, "notes": []})

check("three requests take three different branches",
      lambda: (run("LV-5005")["notes"][-1], run("LV-5004")["notes"][-1],
               run("LV-5003")["notes"][-1]) == ("approve", "to_manager", "decline"))
check("only ONE branch runs per request",
      lambda: len({"approve", "to_manager", "decline"} & set(run("LV-5004")["notes"])) == 1,
      "a conditional edge chooses one target; it does not run them all")
score()
'''),

    md("""
## Watch it run
"""),

    code('''
app = guard(build_router_graph)

if app is not None:
    print(f"{'request':10}{'days':>5}{'bal':>5}   {'path':38} decision")
    print("-" * 100)
    for rid in sorted(REQUESTS):
        out = app.invoke({"request_id": rid, "notes": []})
        print(f"{rid:10}{out['days']:5}{out['balance']:5}   "
              f"{' -> '.join(out['notes']):38} {out['decision']}")
'''),

    md("""
### Read it

Every row went through `summarise` and `check`, then diverged. That divergence is the first thing
in this module you could not have written as a chain.

The router stayed a pure function, which is why the hardest logic in the workflow is also the
cheapest thing to test &mdash; four dicts, no graph.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. `POLICY["hr_over_days"]` is 10 and nothing uses it. Add an `hr_queue` node and a fourth branch.
   Notice you change the router and the map, and touch no existing node.
2. Make `route` return a name that is not in the branch map and read the error. That is the
   failure mode of an LLM-routed supervisor in Module 5, met under controlled conditions.
"""),
]


# =========================================================================== #
# Lab 3.4 -- state reducers
# =========================================================================== #
LAB4 = [
    header(4, "State Reducers", "Intermediate &rarr; Advanced", 25,
           ["Declare <i>how</i> a field combines, once, instead of hand-merging in every node",
            "Decide which fields need a reducer and which should simply be replaced",
            "Run two nodes in the same step and see what happens without one"],
           THREAD_NOTE),
    setup(4),
    code(DOMAIN),

    md("""
## Concept

Look at what every node has been doing:

```python
return {"notes": state["notes"] + ["check"]}
```

Read, append, write the whole thing back. It works, and it puts the merge rule in every node.
A **reducer** moves it into the schema, once:

```python
notes: Annotated[list, add]     # nodes return only their own line; LangGraph appends
```

The default, if you say nothing, is **replace**: last write wins. Right for `decision`, a bug for
`notes` &mdash; and not a matter of taste at all once two nodes run in the same step.
"""),

    md("""
## Section 1 &mdash; Which fields need one

A reducer on every field is as wrong as a reducer on none.
"""),

    code('''
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


def merge_approvers(old: list, new: list) -> list:
    """Two nodes may independently name the same approver."""
    keep_all  = old + new
    keep_once = old + [a for a in new if a not in old]
    return BLANK        # TODO: which is right for a list of people who must sign off?


def needs_a_reducer() -> set:
    """Which of these combine across nodes, rather than being replaced?"""
    # "notes"     -- every node adds its own line
    # "approvers" -- different nodes may each require a sign-off
    # "decision"  -- one sentence; whoever writes last is the answer
    # "days"      -- written once and never again
    return BLANK        # TODO: a set of the field names that need one
''', '''
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


def merge_approvers(old: list, new: list) -> list:
    """Two nodes may independently name the same approver."""
    keep_all  = old + new
    keep_once = old + [a for a in new if a not in old]
    return keep_once    # each approver once, in the order first required


def needs_a_reducer() -> set:
    """Which of these combine across nodes, rather than being replaced?"""
    # "notes"     -- every node adds its own line
    # "approvers" -- different nodes may each require a sign-off
    # "decision"  -- one sentence; whoever writes last is the answer
    # "days"      -- written once and never again
    return {"notes", "approvers"}
'''),

    code('''
# --- Self-check: Section 1
check("merge_approvers does not record the same person twice",
      lambda: merge_approvers(["Devi R."], ["HR", "Devi R."]) == ["Devi R.", "HR"])
check("notes and approvers accumulate; decision and days do not",
      lambda: needs_a_reducer() == {"notes", "approvers"},
      "appending decisions would leave you holding every answer the graph considered")
score()
'''),

    md("""
## Section 2 &mdash; Two nodes at the same time

Both `check_balance` and `check_calendar` have an edge from `summarise`, so LangGraph runs them
**in one step**. Both write `notes` and `approvers`.

Notice how much simpler the nodes got: they return their own line, and nothing reads
`state["notes"]` any more.
"""),

    code('''
from operator import add

class RunState(TypedDict):
    request_id: str
    days: int
    notes: Annotated[list, add]                  # declared once, obeyed by every node
    approvers: Annotated[list, merge_approvers]
    decision: str                                # no reducer: replaced


def summarise(state):
    return {"days": REQUESTS[state["request_id"]]["days"], "notes": ["summarise"]}

def check_balance(state):
    out = {"notes": ["check_balance"]}
    if state["days"] > POLICY["manager_over_days"]:
        out["approvers"] = [REQUESTS[state["request_id"]]["manager"]]
    return out

def check_calendar(state):
    out = {"notes": ["check_calendar"]}
    if state["days"] > POLICY["manager_over_days"]:          # concludes the same thing
        out["approvers"] = [REQUESTS[state["request_id"]]["manager"]]
    if state["days"] > POLICY["hr_over_days"]:
        out["approvers"] = out.get("approvers", []) + ["HR"]
    return out

def finalise(state):
    return {"decision": f'sign-off from: {", ".join(state["approvers"]) or "nobody"}',
            "notes": ["finalise"]}


def build_parallel_graph():
    builder = StateGraph(RunState)
    for name, fn in [("summarise", summarise), ("check_balance", check_balance),
                     ("check_calendar", check_calendar), ("finalise", finalise)]:
        builder.add_node(name, fn)
    builder.add_edge(START, "summarise")
    builder.add_edge("summarise", "check_balance")     # two edges out of one node,
    builder.add_edge("summarise", "check_calendar")    # so both run in the same step
    builder.add_edge("check_balance", "finalise")
    builder.add_edge("check_calendar", "finalise")
    builder.add_edge("finalise", END)
    return builder.compile()
'''),

    code('''
# --- Self-check: Section 2   (a real fan-out, really executed)
def run(rid):
    return build_parallel_graph().invoke({"request_id": rid, "notes": [], "approvers": []})

check("both parallel nodes' notes survive -- four nodes, four notes",
      lambda: len(run("LV-5004")["notes"]) == 4
          and {"check_balance", "check_calendar"} <= set(run("LV-5004")["notes"]),
      "without a reducer on notes, one would have overwritten the other")
check("LV-5004 needs the manager once, not twice, plus HR",
      lambda: run("LV-5004")["approvers"] == ["Sam O.", "HR"],
      "both nodes named the manager; merge_approvers is what stops the duplicate")
score()
'''),

    md("""
## Watch it run &mdash; and watch it fail without one

The same fan-out, with `notes` declared as a plain `list`.
"""),

    code('''
def with_reducers():
    out = build_parallel_graph().invoke({"request_id": "LV-5004", "notes": [], "approvers": []})
    print("with reducers:", out["notes"], "|", out["approvers"], "|", out["decision"])

guard(with_reducers)          # a reducer raises at INVOKE time, not at build time

class NoReducer(TypedDict):
    request_id: str
    days: int
    notes: list            # plain list: nothing says how two writes combine
    approvers: list
    decision: str

b = StateGraph(NoReducer)
b.add_node("summarise", summarise)
b.add_node("a", lambda s: {"notes": ["a"]})
b.add_node("b", lambda s: {"notes": ["b"]})
b.add_edge(START, "summarise")
b.add_edge("summarise", "a"); b.add_edge("summarise", "b")
b.add_edge("a", END); b.add_edge("b", END)

try:
    print("without   :", b.compile().invoke({"request_id": "LV-5004", "notes": [], "approvers": []})["notes"])
except Exception as exc:
    print(f"without   : {type(exc).__name__}: {str(exc)[:150]}")
'''),

    md("""
### Read it

LangGraph **refuses** rather than picking a winner: two nodes wrote the same key in one step and
nothing said how to combine them.

Remember that, because the sequential version of this bug is not loud. In a chain, a node
returning `{"notes": ["mine"]}` on a field with no reducer silently throws away everything before
it, and you find out when the audit trail has one line in it.

**The habit:** decide the merge rule where you declare the field, not in each node that writes it.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Replace `merge_approvers` with `add` and re-run. Most LangGraph code you read uses
   `Annotated[list, add]` &mdash; now you know what it means and when it is not enough.
2. Add a third parallel node that also writes `days`. Predict what happens, then run it.
"""),
]


# =========================================================================== #
# Lab 3.5 -- cycles and retry
# =========================================================================== #
LAB5 = [
    header(5, "Cycles and Retry", "Advanced", 30,
           ["Build a loop &mdash; an edge that points backwards, and nothing more exotic",
            "Give it a step budget, because the runaway loop is Module 2's failure mode",
            "Put the model inside <i>one</i> node and leave the control flow deterministic"],
           THREAD_NOTE),
    setup(5),
    code(DOMAIN),

    md("""
## Concept

A cycle is an edge that points backwards. There is no loop construct in LangGraph &mdash;
`add_edge("ask", "validate")` and you have one.

So the interesting part is not building the loop, it is **stopping** it. In a graph you write that
down as a field and a comparison.

`LV-5002` has an empty `reason`. Go back to the employee, ask, try again &mdash; but not forever.

```
              +-----------------+
              v                 |
START -> validate -> (route) -> ask
                       |
                       +--> decide     (it is complete)
                       +--> withdraw   (we asked enough times)
```
"""),

    md("""
## Section 1 &mdash; The router, and the budget

Three ways out of `validate`. `POLICY` has three numbers in it; one is how many times we are
willing to go back.
"""),

    code('''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END


class ReviewState(TypedDict):
    request_id: str
    reason: str
    complete: bool
    attempts: int                   # replaced, not accumulated -- it is a counter
    inbox: dict                     # what the employee replies, keyed by attempt number
    decision: str
    notes: Annotated[list, add]


def route_after_validate(state: ReviewState) -> str:
    """complete -> decide.  incomplete with budget left -> ask.  budget spent -> withdraw."""
    if state["complete"]:
        return BLANK                            # TODO: nothing more to ask for -- which branch?

    if state["attempts"] >= POLICY[BLANK]:      # TODO: which handbook number is the budget?
        return BLANK                            # TODO: we have asked enough -- which branch?

    return "ask"
''', '''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END


class ReviewState(TypedDict):
    request_id: str
    reason: str
    complete: bool
    attempts: int                   # replaced, not accumulated -- it is a counter
    inbox: dict                     # what the employee replies, keyed by attempt number
    decision: str
    notes: Annotated[list, add]


def route_after_validate(state: ReviewState) -> str:
    """complete -> decide.  incomplete with budget left -> ask.  budget spent -> withdraw."""
    if state["complete"]:
        return "decide"

    if state["attempts"] >= POLICY["max_clarifications"]:
        return "withdraw"                       # the budget is what makes the cycle terminate

    return "ask"
'''),

    code('''
# --- Self-check: Section 1
def r(complete, attempts):
    return route_after_validate({"complete": complete, "attempts": attempts})

check("the budget is 2, so a third pass withdraws instead of asking again",
      lambda: r(False, 0) == "ask" and r(False, 1) == "ask" and r(False, 2) == "withdraw",
      ">= is what makes it a budget rather than a suggestion")
check("a complete request is never withdrawn, however long it took",
      lambda: r(True, 0) == "decide" and r(True, 9) == "decide",
      "check completeness first -- getting the answer late is still getting the answer")
score()
'''),

    md("""
## Section 2 &mdash; The backward edge

`ask` bumps the counter and reads whatever the employee sent back. Then one edge returns the run
to `validate` &mdash; and that edge is the cycle.
"""),

    code('''
def validate(state: ReviewState) -> dict:
    reason = state["reason"] or REQUESTS[state["request_id"]]["reason"]
    return {"reason": reason, "complete": bool(reason.strip()),
            "notes": [f'validate(attempt={state["attempts"]}, complete={bool(reason.strip())})']}

def ask_employee(state: ReviewState) -> dict:
    n = state["attempts"] + 1
    return {"attempts": n, "reason": state["inbox"].get(n, ""),
            "notes": [f'ask(attempt={n})']}

def decide(state: ReviewState) -> dict:
    return {"decision": f'accepted: {state["reason"]}', "notes": ["decide"]}

def withdraw(state: ReviewState) -> dict:
    return {"decision": f'withdrawn after {state["attempts"]} request(s)', "notes": ["withdraw"]}


def build_loop(ask_node=ask_employee):
    builder = StateGraph(ReviewState)
    builder.add_node("validate", validate)
    builder.add_node("ask", ask_node)
    builder.add_node("decide", decide)
    builder.add_node("withdraw", withdraw)

    builder.add_edge(START, "validate")
    builder.add_conditional_edges("validate", route_after_validate,
                                  {"ask": "ask", "decide": "decide", "withdraw": "withdraw"})

    builder.add_edge("ask", BLANK)     # TODO: the cycle -- where does the run go after asking?

    builder.add_edge("decide", END)
    builder.add_edge("withdraw", END)
    return builder.compile()
''', '''
def validate(state: ReviewState) -> dict:
    reason = state["reason"] or REQUESTS[state["request_id"]]["reason"]
    return {"reason": reason, "complete": bool(reason.strip()),
            "notes": [f'validate(attempt={state["attempts"]}, complete={bool(reason.strip())})']}

def ask_employee(state: ReviewState) -> dict:
    n = state["attempts"] + 1
    return {"attempts": n, "reason": state["inbox"].get(n, ""),
            "notes": [f'ask(attempt={n})']}

def decide(state: ReviewState) -> dict:
    return {"decision": f'accepted: {state["reason"]}', "notes": ["decide"]}

def withdraw(state: ReviewState) -> dict:
    return {"decision": f'withdrawn after {state["attempts"]} request(s)', "notes": ["withdraw"]}


def build_loop(ask_node=ask_employee):
    builder = StateGraph(ReviewState)
    builder.add_node("validate", validate)
    builder.add_node("ask", ask_node)
    builder.add_node("decide", decide)
    builder.add_node("withdraw", withdraw)

    builder.add_edge(START, "validate")
    builder.add_conditional_edges("validate", route_after_validate,
                                  {"ask": "ask", "decide": "decide", "withdraw": "withdraw"})

    builder.add_edge("ask", "validate")   # the cycle: go round and check again

    builder.add_edge("decide", END)
    builder.add_edge("withdraw", END)
    return builder.compile()
'''),

    code('''
# --- Self-check: Section 2   (a real cycle that really terminates -- no model)
BASE = {"reason": "", "complete": False, "attempts": 0, "inbox": {}, "decision": "", "notes": []}

def run(rid, inbox=None, ask_node=ask_employee):
    return build_loop(ask_node).invoke({**BASE, "request_id": rid, "inbox": inbox or {}})

check("LV-5001 already has a reason, so the loop never runs",
      lambda: run("LV-5001")["decision"].startswith("accepted")
          and not any("ask(" in n for n in run("LV-5001")["notes"]))
check("LV-5002 gets asked, replies on attempt 2, and the loop ends",
      lambda: run("LV-5002", {2: "conference in Berlin"})["decision"].startswith("accepted"),
      "validate has to run again to notice the reply -- that is the backward edge")
check("with no reply ever, the budget stops it rather than looping forever",
      lambda: run("LV-5002")["decision"].startswith("withdrawn")
          and run("LV-5002")["attempts"] == POLICY["max_clarifications"],
      "this is the check that would hang if the budget were missing")
score()
'''),

    md("""
## Watch it run

Both exits. The repeated `validate` lines are the cycle.
"""),

    code('''
if guard(build_loop) is not None:
    for label, inbox in [("replies on attempt 2", {2: "conference in Berlin"}),
                         ("never replies", {})]:
        out = run("LV-5002", inbox)
        print(f"=== LV-5002, {label} ===")
        for n in out["notes"]:
            print("   ", n)
        print("    ->", out["decision"], "\\n")
'''),

    md("""
## Run it for real &mdash; the model inside one node

Same graph. `ask_llm` has the model write the message to the employee, and changes nothing else.
That is the pattern to take away: **the model is a node, not the architecture.** The routing, the
budget and the cycle stay ordinary code you can test offline.
"""),

    code('''
def ask_llm(state: ReviewState) -> dict:
    n = state["attempts"] + 1
    r = REQUESTS[state["request_id"]]
    msg = ask(f'Ask {r["who"]} to give a reason for their {r["days"]}-day {r["kind"]} '
                    f'leave request. One short polite sentence, no greeting, no sign-off.',
                    system="You write brief internal HR messages. Plain text only.")
    return {"attempts": n, "reason": state["inbox"].get(n, ""),
            "notes": [f'ask({n}): {msg.strip()[:100]}']}


def live_run():
    out = run("LV-5002", {2: "conference in Berlin"}, ask_node=ask_llm)
    for n in out["notes"]:
        print("   ", n)
    print("    ->", out["decision"])

if llm_ready():
    guard(live_run)
'''),

    md("""
### Read it

The run went round the cycle exactly as many times as the deterministic one, and stopped for the
same reason. The model wrote a sentence; it decided nothing.

Nothing about the loop got less testable: the Section 2 checks still run offline and still cover
the budget &mdash; the one behaviour you cannot afford to have flake. Lab 1.1's agent was a
`while True` too, and the stop rule you wrote there is the only thing that ended it. Here the same
job is a missing `>=` in a four-line function.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Set `POLICY["max_clarifications"]` to 0 and re-run. Does it ask once, or not at all? Read the
   router and decide before you run it.
2. Remove your budget check and pass `{"recursion_limit": 4}` to `invoke`. Which error would you
   rather explain &mdash; "withdrawn after 2 requests" or `GraphRecursionError`?
"""),
]


# =========================================================================== #
# Lab 3.6 -- checkpointing
# =========================================================================== #
LAB6 = [
    header(6, "Checkpointing", "Advanced", 30,
           ["Get persistence from one keyword argument",
            "Decide what a <code>thread_id</code> should be &mdash; it is an identity, not a token",
            "Read one snapshot with <code>get_state</code>, the whole trail with <code>get_state_history</code>"],
           THREAD_NOTE),
    setup(6),
    code(DOMAIN),

    md("""
## Concept

Everything so far vanished the moment `invoke()` returned. A **checkpointer** writes the state
after every node, so a run stops being an event and becomes a record.

```python
app = builder.compile(checkpointer=InMemorySaver())
app.invoke(state, {"configurable": {"thread_id": "LV-5002"}})
```

One argument, and four things become possible: **resume** a run later or in another process,
**inspect** it without re-running, **rewind** it, and **audit** it.

The fourth is the one nobody builds deliberately &mdash; it falls out of the other three, and it is
a stronger artefact than the model's own account of what it did, because it was recorded as it
happened rather than reconstructed afterwards.

`InMemorySaver` is the development one. `SqliteSaver` and `PostgresSaver` take the same argument;
only the constructor changes.
"""),

    md("""
## Section 1 &mdash; Attach one, and choose the thread

`thread_id` names the thing being worked on. Runs that share one see each other's state; runs that
do not are strangers. Too broad and unrelated requests contaminate each other; too narrow and
"resume tomorrow" quietly starts from scratch.
"""),

    code('''
import uuid
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class CaseState(TypedDict):
    request_id: str
    days: int
    decision: str
    notes: Annotated[list, add]


def summarise(state): return {"days": REQUESTS[state["request_id"]]["days"], "notes": ["summarise"]}
def decide(state):    return {"decision": "auto" if state["days"] <= POLICY["manager_over_days"]
                                          else "manager", "notes": ["decide"]}


def build_persistent():
    builder = StateGraph(CaseState)
    builder.add_node("summarise", summarise)
    builder.add_node("decide", decide)
    builder.add_edge(START, "summarise")
    builder.add_edge("summarise", "decide")
    builder.add_edge("decide", END)
    return builder.compile(checkpointer=BLANK)   # TODO: what makes this graph remember?


def thread_for(request_id: str) -> dict:
    """Two runs should share a thread when they are working the same ... what?"""
    per_request  = {"configurable": {"thread_id": request_id}}
    per_employee = {"configurable": {"thread_id": REQUESTS[request_id]["who"]}}
    per_run      = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return BLANK        # TODO: which one lets you pick this request up again tomorrow?
''', '''
import uuid
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class CaseState(TypedDict):
    request_id: str
    days: int
    decision: str
    notes: Annotated[list, add]


def summarise(state): return {"days": REQUESTS[state["request_id"]]["days"], "notes": ["summarise"]}
def decide(state):    return {"decision": "auto" if state["days"] <= POLICY["manager_over_days"]
                                          else "manager", "notes": ["decide"]}


def build_persistent():
    builder = StateGraph(CaseState)
    builder.add_node("summarise", summarise)
    builder.add_node("decide", decide)
    builder.add_edge(START, "summarise")
    builder.add_edge("summarise", "decide")
    builder.add_edge("decide", END)
    return builder.compile(checkpointer=InMemorySaver())


def thread_for(request_id: str) -> dict:
    """Two runs should share a thread when they are working the same ... what?"""
    per_request  = {"configurable": {"thread_id": request_id}}
    per_employee = {"configurable": {"thread_id": REQUESTS[request_id]["who"]}}
    per_run      = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return per_request  # the request is the thing being worked on
'''),

    code('''
# --- Self-check: Section 1   (a real checkpointer, really invoked -- no model)
check("the same request twice lands on the same thread",
      lambda: thread_for("LV-5001") == thread_for("LV-5001"),
      "a fresh id per run would make 'resume tomorrow' impossible")
check("LV-5001 and LV-5005 are the same person and still do not share a thread",
      lambda: thread_for("LV-5001") != thread_for("LV-5005"),
      "keying on the employee would let one request overwrite the other")

def ran(rid):
    app = build_persistent()
    app.invoke({"request_id": rid, "notes": []}, thread_for(rid))
    return app

check("state is still there after invoke() returned",
      lambda: ran("LV-5005").get_state(thread_for("LV-5005")).values["decision"] == "auto")
score()
'''),

    md("""
## Section 2 &mdash; Reading it back

Two calls, and the difference between them is the difference between a status page and an audit
trail. `get_state(config)` is one snapshot &mdash; where is this now? `get_state_history(config)`
is every checkpoint, newest first &mdash; what did it know, and when?
"""),

    code('''
def audit_trail(app, config) -> list:
    """What you hand someone who asks how this request reached its decision."""
    latest = [app.get_state(config)]
    every   = list(app.get_state_history(config))
    return BLANK        # TODO: which one answers "what was known at each step?"
''', '''
def audit_trail(app, config) -> list:
    """What you hand someone who asks how this request reached its decision."""
    latest = [app.get_state(config)]
    every   = list(app.get_state_history(config))
    return every        # a snapshot says where it ended; the trail says how it got there
'''),

    code('''
# --- Self-check: Section 2   (real checkpoint history off a real run)
def trail(rid="LV-5004"):
    return audit_trail(ran(rid), thread_for(rid))

check("the trail has a checkpoint per step, not just the final one",
      lambda: len(trail()) > 1,
      "get_state returns one snapshot; get_state_history returns all of them")
check("an earlier checkpoint exists in which no decision had been made yet",
      lambda: any(not s.values.get("decision") for s in trail()),
      "that is the point of a trail: it shows what was NOT yet known")
score()
'''),

    md("""
## Watch it run &mdash; come back to it later
"""),

    code('''
app = guard(build_persistent)
cfg = guard(lambda: thread_for("LV-5004"))

if app is not None and cfg is not None:
    app.invoke({"request_id": "LV-5004", "notes": []}, cfg)

    snap = app.get_state(cfg)
    print("where is it now?  decision =", snap.values["decision"],
          " next =", snap.next or "(finished)")

    print("\\nhow did it get there?")
    for s in reversed(list(app.get_state_history(cfg))):
        print(f"   next={str(s.next or ('END',)):22} notes={s.values.get('notes', [])}")
'''),

    md("""
## Run it for real &mdash; the model's output lands in the record

`explain` asks the model for the sentence the employee reads. Run it, then look at the trail: the
output is *in* a checkpoint, at a known step, next to the state that produced it.
"""),

    code('''
if llm_ready():
    def explain(state):
        r = REQUESTS[state["request_id"]]
        text = ask(f'Tell {r["who"]} in one short sentence that their {r["days"]}-day leave '
                   f'request outcome is "{state["decision"]}". No greeting.',
                   system="You write brief internal HR messages.")
        return {"notes": [f"explain: {text.strip()[:100]}"]}

    b = StateGraph(CaseState)
    b.add_node("summarise", summarise); b.add_node("decide", decide); b.add_node("explain", explain)
    b.add_edge(START, "summarise"); b.add_edge("summarise", "decide")
    b.add_edge("decide", "explain"); b.add_edge("explain", END)
    live = b.compile(checkpointer=InMemorySaver())

    live_cfg = {"configurable": {"thread_id": "LV-5002"}}
    live.invoke({"request_id": "LV-5002", "notes": []}, live_cfg)

    for s in reversed(list(live.get_state_history(live_cfg))):
        print(f"   next={str(s.next or ('END',)):22} notes={s.values.get('notes', [])}")
'''),

    md("""
### Read it

**Nobody wrote a log.** The history exists because the state is explicit and the checkpointer
saved it after every node &mdash; the return on the work done in Lab 3.1.

**It beats asking the model.** An agent asked to explain itself produces a plausible
reconstruction. The trail is what was actually true at each step, recorded before anyone knew the
outcome would matter.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Invoke the same thread again with a different `request_id` and read the state back. Did the
   notes reset? Should they have? That is the contamination a bad `thread_id` produces.
2. Fetch one old checkpoint with `app.get_state({"configurable": {"thread_id": ...,
   "checkpoint_id": ...}})` using an id from the history. That lookup is where an incident review
   starts &mdash; and where Lab 3.7's rewind starts.
"""),
]


# =========================================================================== #
# Lab 3.7 -- human-in-the-loop
# =========================================================================== #
LAB7 = [
    header(7, "Human-in-the-Loop", "Advanced", 30,
           ["Stop a graph before the step that cannot be taken back",
            "Resume it with <code>None</code> &mdash; and know why it is not the original input",
            "Let a person edit the state with <code>update_state</code>, then meet the rewind surprise"],
           THREAD_NOTE),
    setup(7),
    code(DOMAIN),

    md("""
## Concept

A checkpointer lets a run stop and start again. `interrupt_before` makes it stop on purpose.

```python
app = builder.compile(checkpointer=InMemorySaver(), interrupt_before=["notify"])
...
app.invoke(None, config)      # None means "carry on", not "start again"
```

`invoke()` runs up to `notify`, saves, and returns with `state.next == ("notify",)`. A person
looks. Then you resume.

**Where the gate goes is the design decision.** Not before the thinking &mdash; there is nothing to
look at yet, and you will have annoyed the approver by the third request. Before the step with a
consequence outside your process.

> An approval gate does not always need a checkpointer: a node that refuses to act without a named
> approver is already a gate, and the capstone uses exactly that. What a checkpointer buys is
> *pause and resume* &mdash; stopping now and finishing tomorrow, from another process.
"""),

    md("""
## Section 1 &mdash; Where the gate belongs, and how to resume

Three nodes. One of them does something you cannot quietly undo.
"""),

    code('''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

SENT = []          # stands in for the outside world: emails sent, balances debited


class ApprovalState(TypedDict):
    request_id: str
    recommendation: str
    approved_by: str
    notes: Annotated[list, add]


def recommend(state):
    """Works out what SHOULD happen. Nothing leaves the process."""
    r = REQUESTS[state["request_id"]]
    return {"recommendation": "granted" if r["balance"] >= r["days"] else "refused",
            "notes": ["recommend"]}

def notify(state):
    """Emails the employee and debits the balance. There is no unsend button."""
    SENT.append(f'{REQUESTS[state["request_id"]]["who"]}: leave {state["recommendation"]} '
                f'by {state["approved_by"] or "NOBODY"}')
    return {"notes": [f"notify -> {SENT[-1]}"]}


def gate_nodes() -> list:
    """Which node must not run until a person has said yes?"""
    # "recommend" -- computes a recommendation. Nothing leaves the process.
    # "notify"    -- emails the employee and debits their balance.
    return BLANK        # TODO: a list of the node name(s) to interrupt before


def resume_input():
    """What do you hand invoke() to continue a paused run?"""
    start_again = {"request_id": "LV-5004", "notes": []}
    carry_on    = None
    return BLANK        # TODO: which continues from the checkpoint instead of re-running?


def build_gated():
    builder = StateGraph(ApprovalState)
    builder.add_node("recommend", recommend)
    builder.add_node("notify", notify)
    builder.add_edge(START, "recommend")
    builder.add_edge("recommend", "notify")
    builder.add_edge("notify", END)
    return builder.compile(checkpointer=InMemorySaver(), interrupt_before=gate_nodes())
''', '''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

SENT = []          # stands in for the outside world: emails sent, balances debited


class ApprovalState(TypedDict):
    request_id: str
    recommendation: str
    approved_by: str
    notes: Annotated[list, add]


def recommend(state):
    """Works out what SHOULD happen. Nothing leaves the process."""
    r = REQUESTS[state["request_id"]]
    return {"recommendation": "granted" if r["balance"] >= r["days"] else "refused",
            "notes": ["recommend"]}

def notify(state):
    """Emails the employee and debits the balance. There is no unsend button."""
    SENT.append(f'{REQUESTS[state["request_id"]]["who"]}: leave {state["recommendation"]} '
                f'by {state["approved_by"] or "NOBODY"}')
    return {"notes": [f"notify -> {SENT[-1]}"]}


def gate_nodes() -> list:
    """Which node must not run until a person has said yes?"""
    # "recommend" -- computes a recommendation. Nothing leaves the process.
    # "notify"    -- emails the employee and debits their balance.
    return ["notify"]   # gate the irreversible step, not the thinking


def resume_input():
    """What do you hand invoke() to continue a paused run?"""
    start_again = {"request_id": "LV-5004", "notes": []}
    carry_on    = None
    return carry_on     # no new input -- resume from where you stopped


def build_gated():
    builder = StateGraph(ApprovalState)
    builder.add_node("recommend", recommend)
    builder.add_node("notify", notify)
    builder.add_edge(START, "recommend")
    builder.add_edge("recommend", "notify")
    builder.add_edge("notify", END)
    return builder.compile(checkpointer=InMemorySaver(), interrupt_before=gate_nodes())
'''),

    code('''
# --- Self-check: Section 1   (a real interrupt on a real graph -- no model)
BASE = {"recommendation": "", "approved_by": "", "notes": []}

def paused(rid="LV-5004"):
    SENT.clear()
    app, cfg = build_gated(), {"configurable": {"thread_id": rid}}
    app.invoke({**BASE, "request_id": rid}, cfg)
    return app, cfg

def paused_snapshot():
    app, cfg = paused()
    return app.get_state(cfg)

check("the run stops before notify, with the thinking already done",
      lambda: paused_snapshot().next == ("notify",)
          and paused_snapshot().values["recommendation"] == "granted",
      "state.next is what the graph would do if you let it carry on")
check("nothing reached the outside world while it was paused",
      lambda: (paused(), SENT == [])[1],
      "gating recommend instead would stop the run before there was anything to approve")

def resumed_values():
    app, cfg = paused()
    app.invoke(resume_input(), cfg)
    return app.get_state(cfg).values

check("resuming runs the gated node once, and does not re-run recommend",
      lambda: resumed_values()["notes"].count("recommend") == 1 and len(SENT) == 1,
      "passing the original input instead of None would run recommend a second time")
score()
'''),

    md("""
## Section 2 &mdash; The person changes something

Approval is rarely just yes. `update_state` writes into the paused checkpoint before you resume,
which is how the approver's decision gets **into the record** rather than living in an email.
"""),

    code('''
def approver_writes(manager: str) -> dict:
    """A manager overrules the recommendation and refuses. What must land in the state?"""
    verdict_only    = {"recommendation": "refused"}
    verdict_and_who = {"recommendation": "refused", "approved_by": manager}
    return BLANK        # TODO: which one can you still defend six months from now?
''', '''
def approver_writes(manager: str) -> dict:
    """A manager overrules the recommendation and refuses. What must land in the state?"""
    verdict_only    = {"recommendation": "refused"}
    verdict_and_who = {"recommendation": "refused", "approved_by": manager}
    return verdict_and_who   # a decision with no decider is not an audit record
'''),

    code('''
# --- Self-check: Section 2   (a human edit, then a real resume)
def overridden_values():
    app, cfg = paused()
    app.update_state(cfg, approver_writes("Sam O."))
    app.invoke(None, cfg)
    return app.get_state(cfg).values

check("the approver's verdict overrode the recommendation, and who decided is recorded",
      lambda: overridden_values()["recommendation"] == "refused"
          and overridden_values()["approved_by"] == "Sam O.",
      "the graph recommended granted; a person said refused, and the person won")
check("...and that is what reached the outside world",
      lambda: (overridden_values(), "refused" in SENT[-1])[1])
score()
'''),

    md("""
## Watch it run

Run, pause, look, decide, resume.
"""),

    code('''
app = guard(build_gated)

if app is not None:
    SENT.clear()
    cfg = {"configurable": {"thread_id": "LV-5004"}}

    app.invoke({**BASE, "request_id": "LV-5004"}, cfg)
    s = app.get_state(cfg)
    print("1. paused before:", s.next, "  recommends:", s.values["recommendation"],
          "  sent so far:", SENT)

    app.update_state(cfg, {"approved_by": "Sam O.", "notes": ["Sam O. checked the team calendar"]})
    app.invoke(None, cfg)
    print("2. resumed, sent:", SENT)
    print("3. the record  :", app.get_state(cfg).values["notes"])
'''),

    md("""
## Run it for real &mdash; and the surprise

The model drafts what the approver reads: a good use for it, because a person checks it before
anything happens. Then rewind into the gate.
"""),

    code('''
if llm_ready() and app is not None:
    SENT.clear()
    cfg2 = {"configurable": {"thread_id": "LV-5002-live"}}
    app.invoke({**BASE, "request_id": "LV-5002"}, cfg2)

    r = REQUESTS["LV-5002"]
    print("for the approver:", ask(
        f'Summarise for a manager deciding whether to approve: {r["who"]} asks for {r["days"]} '
        f'days of {r["kind"]} leave, reason given: "{r["reason"] or "none"}", balance '
        f'{r["balance"]} days. The system recommends '
        f'{app.get_state(cfg2).values["recommendation"]}.',
        system="You brief a busy manager. One sentence, no greeting.").strip()[:250])

    earlier = [h for h in app.get_state_history(cfg2) if h.next == ("recommend",)]
    if earlier:
        print("\\nrewinding to before 'recommend' and running forward again...")
        app.invoke(None, earlier[0].config)
        print("   where did it stop?", app.get_state(cfg2).next, "  sent:", SENT)
'''),

    md("""
### Read it

**The rewind stopped at the gate again** rather than running through to a new answer. That
surprises everyone once.

`interrupt_before` is a property of the **compiled graph**, not of a run. Replaying from an older
checkpoint runs forward through the same graph, so it meets the same gate &mdash; and rewinding
past the approval un-does it, because approval was a state edit at one point in history.

That is right for an approval workflow and wrong to discover in production. If you want a rewind
that keeps an approval, the approval has to be a **fact in the state** the gate reads, not the
fact that someone once resumed.

And `SENT` stayed empty until you resumed. The gate did its job.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Move the gate to `["recommend"]` and run it. What is the approver looking at? That is the
   argument for gating late.
2. An approver never comes back. Write the check that finds threads whose `next` has been
   non-empty for too long &mdash; timeout and escalation are policy on top of exactly this state.
"""),
]


# =========================================================================== #
# Lab 3.8 -- challenge
# =========================================================================== #
LAB8 = [
    header(8, "Challenge &mdash; The Leave Request Workflow", "Advanced", 40,
           ["Assemble everything: state, routing, a cycle, a budget, checkpoints and a gate",
            "Gate <i>one branch</i> rather than the whole graph &mdash; routing, not a bigger interrupt",
            "Run four requests down four different paths through one compiled graph"],
           THREAD_NOTE),
    setup(8),
    code(DOMAIN),

    md("""
## The brief

One graph that handles every request in the case file.

```
START -> validate --(incomplete, budget left)--> ask ---+
            |                                            |   (cycle)
            |  <-----------------------------------------+
            |--(incomplete, budget spent)--> withdraw -> END
            |
            +--(complete)--> assess --(short, covered)------> notify -> END
                                |
                                +--(long or uncovered)--> manager_review -> notify -> END
                                                          ^ paused here
```

Four requirements:

1. **A cycle with a budget.** `LV-5002` has no reason. Ask, and give up after
   `POLICY["max_clarifications"]`.
2. **A checkpointer**, so a paused request can be picked up later.
3. **A gate on one branch only.** `manager_review` pauses; the auto-approval branch must not.
   You do this with *routing*, not with a bigger `interrupt_before`.
4. **`notify` is irreversible** and must never run before its branch's approval.
"""),

    md("""
## Section 1 &mdash; The two routers

`validate` decides whether we can proceed at all. `assess` decides who has to say yes.
"""),

    code('''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

SENT = []


class LeaveState(TypedDict):
    request_id: str
    reason: str
    complete: bool
    attempts: int
    inbox: dict
    verdict: str
    approved_by: str
    notes: Annotated[list, add]


def after_validate(state: LeaveState) -> str:
    if state["complete"]:
        return "assess"
    if state["attempts"] >= POLICY["max_clarifications"]:
        return BLANK        # TODO: we have asked enough times -- which branch?
    return BLANK            # TODO: still incomplete, budget left -- which branch?


def after_assess(state: LeaveState) -> str:
    """Short and covered goes straight out. Anything else needs a person."""
    r = REQUESTS[state["request_id"]]
    if r["days"] <= POLICY["manager_over_days"] and r["balance"] >= r["days"]:
        return "auto"
    return BLANK            # TODO: which branch ends up in front of the manager?
''', '''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

SENT = []


class LeaveState(TypedDict):
    request_id: str
    reason: str
    complete: bool
    attempts: int
    inbox: dict
    verdict: str
    approved_by: str
    notes: Annotated[list, add]


def after_validate(state: LeaveState) -> str:
    if state["complete"]:
        return "assess"
    if state["attempts"] >= POLICY["max_clarifications"]:
        return "withdraw"
    return "ask"


def after_assess(state: LeaveState) -> str:
    """Short and covered goes straight out. Anything else needs a person."""
    r = REQUESTS[state["request_id"]]
    if r["days"] <= POLICY["manager_over_days"] and r["balance"] >= r["days"]:
        return "auto"
    return "review"
'''),

    code('''
# --- Self-check: Section 1   (two pure functions)
check("validate routes: complete -> assess, budget left -> ask, budget spent -> withdraw",
      lambda: (after_validate({"complete": True, "attempts": 0}),
               after_validate({"complete": False, "attempts": 0}),
               after_validate({"complete": False, "attempts": 2})) == ("assess", "ask", "withdraw"))
check("assess routes LV-5005 to auto, LV-5004 and LV-5003 to review",
      lambda: (after_assess({"request_id": "LV-5005"}),
               after_assess({"request_id": "LV-5004"}),
               after_assess({"request_id": "LV-5003"})) == ("auto", "review", "review"),
      "LV-5003 is short but has no balance, so a person decides")
score()
'''),

    md("""
## Section 2 &mdash; The graph

Nodes are given &mdash; you have written all of them before. Wire the cycle, and put the gate on the
branch that needs it.
"""),

    code('''
def validate(state):
    reason = state["reason"] or REQUESTS[state["request_id"]]["reason"]
    return {"reason": reason, "complete": bool(reason.strip()),
            "notes": [f'validate(attempt={state["attempts"]})']}

def ask_employee(state):
    n = state["attempts"] + 1
    return {"attempts": n, "reason": state["inbox"].get(n, ""), "notes": [f"ask({n})"]}

def withdraw(state):
    return {"verdict": "withdrawn", "notes": ["withdraw"]}

def assess(state):
    return {"notes": ["assess"]}

def auto(state):
    return {"verdict": "granted", "approved_by": "policy", "notes": ["auto"]}

def manager_review(state):
    """Paused before this node. The approver's edit lands in the checkpoint."""
    return {"verdict": state["verdict"] or "granted",
            "approved_by": state["approved_by"] or REQUESTS[state["request_id"]]["manager"],
            "notes": ["manager_review"]}

def notify(state):
    """Irreversible."""
    SENT.append(f'{REQUESTS[state["request_id"]]["who"]}: {state["verdict"]} '
                f'by {state["approved_by"] or "NOBODY"}')
    return {"notes": [f"notify -> {SENT[-1]}"]}


def build_workflow():
    b = StateGraph(LeaveState)
    for name, fn in [("validate", validate), ("ask", ask_employee), ("withdraw", withdraw),
                     ("assess", assess), ("auto", auto),
                     ("manager_review", manager_review), ("notify", notify)]:
        b.add_node(name, fn)

    b.add_edge(START, "validate")
    b.add_conditional_edges("validate", after_validate,
                            {"ask": "ask", "withdraw": "withdraw", "assess": "assess"})
    b.add_edge("ask", BLANK)              # TODO: the cycle

    b.add_conditional_edges("assess", after_assess,
                            {"auto": "auto", "review": "manager_review"})
    b.add_edge("auto", "notify")
    b.add_edge("manager_review", "notify")
    b.add_edge("notify", END)
    b.add_edge("withdraw", END)

    return b.compile(checkpointer=InMemorySaver(),
                     interrupt_before=BLANK)   # TODO: gate ONE branch, not the whole graph
''', '''
def validate(state):
    reason = state["reason"] or REQUESTS[state["request_id"]]["reason"]
    return {"reason": reason, "complete": bool(reason.strip()),
            "notes": [f'validate(attempt={state["attempts"]})']}

def ask_employee(state):
    n = state["attempts"] + 1
    return {"attempts": n, "reason": state["inbox"].get(n, ""), "notes": [f"ask({n})"]}

def withdraw(state):
    return {"verdict": "withdrawn", "notes": ["withdraw"]}

def assess(state):
    return {"notes": ["assess"]}

def auto(state):
    return {"verdict": "granted", "approved_by": "policy", "notes": ["auto"]}

def manager_review(state):
    """Paused before this node. The approver's edit lands in the checkpoint."""
    return {"verdict": state["verdict"] or "granted",
            "approved_by": state["approved_by"] or REQUESTS[state["request_id"]]["manager"],
            "notes": ["manager_review"]}

def notify(state):
    """Irreversible."""
    SENT.append(f'{REQUESTS[state["request_id"]]["who"]}: {state["verdict"]} '
                f'by {state["approved_by"] or "NOBODY"}')
    return {"notes": [f"notify -> {SENT[-1]}"]}


def build_workflow():
    b = StateGraph(LeaveState)
    for name, fn in [("validate", validate), ("ask", ask_employee), ("withdraw", withdraw),
                     ("assess", assess), ("auto", auto),
                     ("manager_review", manager_review), ("notify", notify)]:
        b.add_node(name, fn)

    b.add_edge(START, "validate")
    b.add_conditional_edges("validate", after_validate,
                            {"ask": "ask", "withdraw": "withdraw", "assess": "assess"})
    b.add_edge("ask", "validate")         # the cycle

    b.add_conditional_edges("assess", after_assess,
                            {"auto": "auto", "review": "manager_review"})
    b.add_edge("auto", "notify")
    b.add_edge("manager_review", "notify")
    b.add_edge("notify", END)
    b.add_edge("withdraw", END)

    return b.compile(checkpointer=InMemorySaver(),
                     interrupt_before=["manager_review"])   # only the branch that needs a person
'''),

    code('''
# --- Self-check: Section 2   (four requests, four paths, one graph -- no model)
BASE = {"reason": "", "complete": False, "attempts": 0, "inbox": {},
        "verdict": "", "approved_by": "", "notes": []}

def start(rid, inbox=None):
    SENT.clear()
    app, cfg = build_workflow(), {"configurable": {"thread_id": rid}}
    app.invoke({**BASE, "request_id": rid, "inbox": inbox or {}}, cfg)
    return app, cfg

check("LV-5005 is short and covered: it runs straight through, no pause",
      lambda: (start("LV-5005")[0].get_state(start("LV-5005")[1]).next == ()
               and len(SENT) == 1))
check("LV-5004 is long: it pauses before manager_review and sends nothing",
      lambda: (start("LV-5004")[0].get_state(start("LV-5004")[1]).next == ("manager_review",)
               and SENT == []),
      "gate the branch, not the graph -- LV-5005 must not have paused")
check("LV-5002 has no reason and never replies: it withdraws inside the budget",
      lambda: start("LV-5002")[0].get_state(start("LV-5002")[1]).values["verdict"] == "withdrawn")
check("LV-5002 replying on attempt 2 gets through validation and reaches the gate",
      lambda: start("LV-5002", {2: "conference"})[0]
              .get_state(start("LV-5002", {2: "conference"})[1]).next == ("manager_review",))
score()
'''),

    md("""
## Section 3 &mdash; Approve one, and resume
"""),

    code('''
def approve(app, cfg, who: str, verdict: str):
    app.update_state(cfg, {"verdict": verdict, "approved_by": who,
                           "notes": [f"{who} decided {verdict}"]})
    return app.invoke(BLANK, cfg)     # TODO: continue the paused run
''', '''
def approve(app, cfg, who: str, verdict: str):
    app.update_state(cfg, {"verdict": verdict, "approved_by": who,
                           "notes": [f"{who} decided {verdict}"]})
    return app.invoke(None, cfg)      # None continues from the checkpoint
'''),

    code('''
# --- Self-check: Section 3
def approved(verdict="granted"):
    app, cfg = start("LV-5004")
    approve(app, cfg, "Sam O.", verdict)
    return app.get_state(cfg).values

check("approving resumes the run and notify finally fires",
      lambda: (approved(), len(SENT) == 1)[1])
check("a refusal is what goes out, and the approver is named",
      lambda: (approved("refused"), "refused" in SENT[-1] and "Sam O." in SENT[-1])[1],
      "the person overrides the graph, and the record says who")
check("the trail contains the approver's own line",
      lambda: any("Sam O. decided" in n for n in approved()["notes"]))
score()
'''),

    md("""
## Watch it run

Every request in the case file through one compiled graph.
"""),

    code('''
def walk_every_request():
    for rid in sorted(REQUESTS):
        app, cfg = start(rid)
        s = app.get_state(cfg)
        if s.next == ("manager_review",):
            approve(app, cfg, REQUESTS[rid]["manager"], "granted")
            s = app.get_state(cfg)
            tag = "paused, then approved"
        else:
            tag = "no pause needed"
        print(f"{rid}  {tag:22} {' -> '.join(s.values['notes'][-6:])}")
        print(f"{'':10}verdict={s.values['verdict']!r} sent={SENT}")

guard(walk_every_request)
'''),

    md("""
## Run it for real &mdash; the model writes what the employee reads
"""),

    code('''
def live_workflow():
    app, cfg = start("LV-5004")
    r = REQUESTS["LV-5004"]

    print("brief for the manager:", ask(
        f'One sentence for a manager deciding on leave: {r["who"]} ({r["days"]} days '
        f'{r["kind"]}, reason "{r["reason"]}", balance {r["balance"]}).',
        system="You brief a busy manager. One sentence, no greeting.").strip()[:220])

    approve(app, cfg, r["manager"], "granted")
    print("\\nmessage to the employee:", ask(
        f'Tell {r["who"]} their {r["days"]}-day leave was granted by {r["manager"]}. '
        f'One short sentence, no greeting.',
        system="You write brief internal HR messages.").strip()[:220])

    print("\\nthe record:")
    for n in app.get_state(cfg).values["notes"]:
        print("   ", n)

if llm_ready():
    guard(live_workflow)
'''),

    md("""
### Read it

**The gate is on a branch, not on the graph.** `interrupt_before=["manager_review"]` pauses only
requests routed there. `LV-5005` never stopped. If you had gated `notify` instead, every
auto-approval would sit waiting for a manager who has nothing to decide &mdash; the fastest way to
make people ignore an approval queue.

**Four behaviours, one state object.** The cycle, the budget, the conditional gate and the audit
trail are all fields in `LeaveState` plus edges. Nothing here needed a framework feature you have
not already used in this module.

**What you take from Module 3:** state you can print and assert on; nodes and edges as ordinary
testable code; conditional routing; reducers for the fields that accumulate; a cycle with a budget;
checkpoints that make resume, approval, rewind and audit possible. Module 5 puts several agents
into exactly this shape, and its supervisor is `after_assess` with a model choosing the string.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. `POLICY["hr_over_days"]` is 10 and nothing uses it. Add an `hr_review` node so `LV-5004` needs
   two approvals, in order. You will need a second gate and a second resume.
2. Rewind an approved request to before `manager_review` and run it forward. It pauses again &mdash;
   see Lab 3.7. Make the approval survive the rewind by having the gate read a fact in the state.
3. Swap `InMemorySaver` for `SqliteSaver`, run `LV-5004`, restart the kernel, and resume the paused
   request from a fresh process. That is the whole production migration.
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-3-01-your-first-graph",        LAB1),
    ("lab-3-02-multi-step-workflow",     LAB2),
    ("lab-3-03-conditional-routing",     LAB3),
    ("lab-3-04-state-reducers",          LAB4),
    ("lab-3-05-cycles-and-retry",        LAB5),
    ("lab-3-06-checkpointing",           LAB6),
    ("lab-3-07-human-in-the-loop",       LAB7),
    ("lab-3-08-challenge-leave-workflow", LAB8),
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
