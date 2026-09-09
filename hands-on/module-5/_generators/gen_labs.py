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
# Lab 5.1 -- a support desk that triages itself (INTERACTIVE WALKTHROUGH)
# =========================================================================== #
# Rebuilt 2026-09-10 on request: every code cell is a "Run it for real" cell, there is
# nothing to fill in and nothing to score, and the centre of the lab is a small app the
# participant drives. Registered in verify.py's WALKTHROUGH set, exactly like Module 4's
# labs 4.1-4.3 -- zero blanks on both sides, the two files byte-identical, no score line.
#
# Why no graded cells here: a score computed from model output is a score a flaky endpoint
# can move, which is the whole reason the graded-cell rule exists. The lesson of this lab
# is what the router DOES with awkward tickets, and that has to be watched, not asserted.
# Labs 5.2 and 5.3 keep their blanks and their self-checks.

DESK_CASE = '''
# ------------------------------------------------- the case file (synthetic, self-contained)
# A customer support desk for a SaaS product. One queue in, three specialists behind it.
# This case file is Module 5 lab 5.1 only -- 5.2 and 5.3 are different systems.

SPECIALISTS = ("billing", "tech", "account")

# What the supervisor gets to read. These three lines ARE the router's program -- there is
# no other instruction anywhere. You will edit them at the end of the lab and re-measure.
DESK = {
    "billing": "Charges, refunds, invoices, plan and price changes.",
    "tech":    "Errors, outages, failing API calls, anything broken.",
    "account": "Seats, owners, permissions, sign-in and access.",
}

# How each specialist answers once it has the ticket.
PERSONA = {
    "billing": "You are the billing desk of a SaaS company. Answer in one sentence, plainly, "
               "and name the next concrete step. Never promise a refund amount.",
    "tech":    "You are the technical support desk of a SaaS company. Answer in one sentence, "
               "ask for the one diagnostic detail you most need, and never guess a root cause.",
    "account": "You are the account desk of a SaaS company. Answer in one sentence and say "
               "who has to authorise the change. Never change access on the asker's word alone.",
}

# 29 tickets to play with. `expected` is the desk a human would pick; five of them are
# deliberately unroutable and carry None, so they are excluded from any accuracy.
SCENARIOS = [
    # (ticket, expected desk or None, why this one is in the set)
    ("I was charged twice for March.",                                    "billing", "plain"),
    ("Can I get an invoice with our VAT number on it?",                   "billing", "plain"),
    ("We want to downgrade to the starter plan.",                         "billing", "plain"),
    ("Your API returns 500 on every /sync call since 09:00.",             "tech",    "plain"),
    ("The export button throws an error and nothing downloads.",          "tech",    "plain"),
    ("Webhooks stopped firing after your deploy.",                        "tech",    "plain"),
    ("Please add two more seats for the new joiners.",                    "account", "plain"),
    ("Move the workspace owner to priya@example.com.",                    "account", "plain"),

    # the loudest word points at one desk and the problem belongs to another
    ("The invoice page throws a 500.",                                    "tech",    "keyword trap"),
    ("I cannot open billing settings -- it says permission denied.",      "account", "keyword trap"),
    ("Our refund never arrived and the support chat is down too.",        "billing", "keyword trap"),

    # the intent is only implied; no word in the ticket names the desk
    ("Nobody on my team can get in this morning.",                        "account", "implied"),
    ("We were told this would be free until June.",                       "billing", "implied"),
    ("Everything was fine yesterday and now nothing loads.",              "tech",    "implied"),
    ("Someone who left in May can still see our data.",                   "account", "implied"),

    # written the way people actually write to a support desk
    ("cant login sicne mornign, urgnt",                                   "account", "typos"),
    ("Hi! Hope you are well. Quick one -- can we switch to annual billing?",
                                                                          "billing", "buried in politeness"),
    ("This is the third time. Cancel everything and refund us.",          "billing", "angry"),

    # the desk turns on which fact is the PROBLEM, not on which words are present
    ("Can you confirm the seat count on our last invoice?",               "billing", "contested primary"),
    ("We are being billed for a user we deleted in April.",               "billing", "contested primary"),
    ("Why was my card declined when I tried to add a seat?",              "billing", "contested primary"),
    ("The 500 error only happens for users on the free plan.",            "tech",    "contested primary"),
    ("Our SSO broke right after you changed the pricing page.",           "tech",    "contested primary"),
    ("The new admin cannot approve invoices -- she has no such button.",  "account", "contested primary"),

    # no single defensible answer -- these are the interesting ones
    ("It is not working.",                                                None, "too vague to route"),
    ("I was charged for seats we never got and now I cannot log in either.",
                                                                          None, "two desks at once"),
    ("Do you sponsor conferences?",                                       None, "not a support ticket"),
    ("Renewal is next week and our admin's SSO login is broken.",         None, "two desks at once"),
    ("Please remove the card on file and delete the workspace.",          None, "two desks at once"),
]

SCORED = [(t, e) for t, e, _ in SCENARIOS if e is not None]
print(f"{len(SCENARIOS)} scenarios ({len(SCORED)} with a defensible answer), "
      f"{len(DESK)} desks")
'''

LAB1_GRAPH = '''
# ---------------------------------------------- Run it for real: the graph, model and all
# Every node here calls the model. The supervisor decides which desk; the desk answers.
# That is the whole system on slide 2, and there is nothing stubbed in it.

from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END


class DeskState(TypedDict):
    ticket: str                       # what the customer wrote
    route: str | None                 # the supervisor's decision, readable afterwards
    why: str | None                   # and its reason, in its own words
    answer: str | None                # the specialist's reply
    trail: Annotated[list, add]       # append: every node leaves a mark


def desk_listing(desks: dict) -> str:
    return "\\n".join(f"- {name}: {text}" for name, text in desks.items())


def ask_the_router(ticket: str, desks: dict | None = None) -> tuple[str, str]:
    """One model call. Returns (desk, why).

    Two plain lines rather than with_structured_output, deliberately: on this gateway the
    schema route is several times slower, and an app you are clicking through has to feel
    like an app. The last cell times both on your own run -- do not take the ratio on trust,
    it moves with load.
    """
    desks = desks or DESK
    system = ("You route one customer support ticket to exactly one desk.\\n"
              f"{desk_listing(desks)}\\n"
              "Answer with exactly two lines and nothing else:\\n"
              "desk: <one of " + ", ".join(desks) + ">\\n"
              "why: <at most 10 words>")
    reply = ask(ticket, system=system) or ""
    desk, why = None, reply.strip().replace("\\n", " ")[:90]
    for line in reply.splitlines():
        low = line.strip().lower()
        if low.startswith("desk:"):
            word = low[5:].strip().strip("`*.\\"' ")
            desk = word if word in desks else None
        elif low.startswith("why:"):
            why = line.strip()[4:].strip()
    if desk is None:                                  # the model ignored the format
        desk = next((d for d in desks if d in reply.lower()), list(desks)[0])
        why = "(unparsed reply -- fell back to the first desk named)"
    return desk, why


def supervisor(state: DeskState) -> dict:
    """A node like any other. It decides, and it writes the decision down."""
    desk, why = ask_the_router(state["ticket"])
    return {"route": desk, "why": why, "trail": [f"supervisor -> {desk}"]}


def make_specialist(name: str):
    """Three desks that differ only in the system prompt they answer with."""
    def specialist(state: DeskState) -> dict:
        reply = ask(f"Ticket: {state['ticket']}", system=PERSONA[name])
        return {"answer": reply, "trail": [f"{name} answered"]}
    return specialist


def resolve(state: DeskState) -> dict:
    return {"trail": ["resolved"]}


def pick_specialist(state: DeskState) -> str:
    """The adapter a conditional edge needs: takes STATE, returns a KEY of the path map."""
    return state["route"]


def build_desk():
    g = StateGraph(DeskState)
    g.add_node("supervisor", supervisor)
    for name in SPECIALISTS:
        g.add_node(name, make_specialist(name))
    g.add_node("resolve", resolve)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", pick_specialist, {n: n for n in SPECIALISTS})
    for name in SPECIALISTS:
        g.add_edge(name, "resolve")
    g.add_edge("resolve", END)
    return g.compile()


def fresh(ticket: str) -> dict:
    return {"ticket": ticket, "route": None, "why": None, "answer": None, "trail": []}


print("graph: START -> supervisor -> [ billing | tech | account ] -> resolve -> END")
'''

LAB1_ONE = '''
# ---------------------------------------------- Run it for real: one ticket, end to end
def route_ticket(ticket: str) -> dict:
    """Run one ticket through the whole graph and print what each node did."""
    if not llm_ready():
        return {}
    result = build_desk().invoke(fresh(ticket))
    print(f'  ticket      "{ticket}"')
    print(f"  supervisor  -> {result['route']}   ({result['why']})")
    print(f"  {result['route']} desk replies:")
    print(textwrap.fill(result["answer"] or "", 92,
                        initial_indent="    ", subsequent_indent="    "))
    print(f"  trail       {result['trail']}")
    return result

route_ticket("The invoice page throws a 500.")
'''

LAB1_SWEEP = '''
# ---------------------------------------------- Run it for real: all 21 scenarios
# Only the SUPERVISOR runs here -- routing is what is being measured, and waking a desk
# for every scenario would triple the wall clock without changing a single decision.

def sweep(desks: dict | None = None, quiet: bool = False) -> dict:
    """Route every scenario. Returns {ticket: (desk, why)}."""
    out = {}
    for ticket, expected, tag in SCENARIOS:
        desk, why = ask_the_router(ticket, desks)
        out[ticket] = (desk, why)
        if not quiet:
            mark = " " if expected is None else ("ok" if desk == expected else "XX")
            want = expected or "--"
            print(f"  {mark}  {desk:8} (wanted {want:8}) [{tag:20}] {ticket[:44]}")
    return out


def accuracy(routed: dict) -> float:
    """Over the scenarios that HAVE a defensible answer. The other three are not failures."""
    return sum(1 for t, e in SCORED if routed.get(t, (None,))[0] == e) / len(SCORED)


def confusion(routed: dict) -> dict:
    out = {}
    for ticket, expected in SCORED:
        got = routed.get(ticket, (None,))[0]
        if got != expected:
            out[(expected, got)] = out.get((expected, got), 0) + 1
    return out


if llm_ready():
    ROUTED = sweep()
    print(f"\\n  accuracy on the {len(SCORED)} scorable tickets: {accuracy(ROUTED):.0%}")
    for (want, got), n in sorted(confusion(ROUTED).items(), key=lambda kv: -kv[1]):
        print(f"    {n}x  should have been {want:8} -> went to {got}")
    print("\\n  and the ones with no defensible answer:")
    for ticket, expected, tag in SCENARIOS:
        if expected is None:
            print(f"    {ROUTED[ticket][0]:8} <- {ticket[:52]}   ({tag})")
    print("\\n  It answered every one of them, confidently. Nothing in the prompt gave it")
    print("  permission to say 'I do not know' -- which is a design choice you made by omission.")
    if accuracy(ROUTED) > 0.95:
        print("\\n  Note the score. A model supervisor with three good descriptions is very")
        print("  hard to beat on a queue like this, and an eval set your system already passes")
        print("  is not an eval set any more -- it is a regression test. The console below is")
        print("  how you go and find the tickets it does get wrong. Those are the keepers.")
'''

LAB1_APP = '''
# ---------------------------------------------- Run it for real: the triage console
# A small app. Pick a scenario or type your own, watch the graph route it and the desk
# answer, and when it goes to the wrong desk say so -- that is how the eval set gets built.

MY_EVAL = []          # (ticket, the desk you say it should be, the desk it chose)


def triage_console():
    try:
        import ipywidgets as W
        from IPython.display import display, clear_output
    except ImportError:
        print("ipywidgets is not available here. Use route_ticket('your ticket') instead.")
        return

    picker = W.Dropdown(
        options=[("-- type your own below --", "")] +
                [(f"[{tag}]  {t[:56]}", t) for t, _, tag in SCENARIOS],
        layout=W.Layout(width="780px"))
    text = W.Textarea(value=SCENARIOS[0][0], placeholder="type a support ticket",
                      layout=W.Layout(width="780px", height="62px"))
    go = W.Button(description="Route it", button_style="primary")
    seen = W.Button(description="My eval set", layout=W.Layout(width="150px"))
    should = W.Dropdown(options=[("should have been...", None)] + [(d, d) for d in SPECIALISTS],
                        layout=W.Layout(width="220px"))
    log = W.Button(description="Log that", layout=W.Layout(width="130px"))
    out = W.Output()
    last = {"ticket": None, "chose": None}

    picker.observe(lambda c: c["new"] and setattr(text, "value", c["new"]), names="value")

    def on_go(_):
        with out:
            clear_output()
            if not llm_ready():
                return
            try:
                r = build_desk().invoke(fresh(text.value))
            except Exception as exc:
                print(f"  the graph raised {type(exc).__name__}: {exc}")
                return
            last.update(ticket=text.value, chose=r["route"])
            print(f"  supervisor -> {r['route']}    ({r['why']})")
            print(f"  the {r['route']} desk replies:\\n")
            print(textwrap.fill(r["answer"] or "", 90,
                                initial_indent="    ", subsequent_indent="    "))

    def on_log(_):
        with out:
            if not last["ticket"]:
                print("\\n  route a ticket first.")
            elif should.value is None:
                print("\\n  pick the desk it should have gone to, then press Log that.")
            else:
                MY_EVAL.append((last["ticket"], should.value, last["chose"]))
                verdict = "agreed" if should.value == last["chose"] else "MISROUTE"
                print(f"\\n  logged ({verdict}): {last['chose']} -> should be {should.value}"
                      f"    [{len(MY_EVAL)} in your eval set]")

    def on_seen(_):
        with out:
            clear_output()
            if not MY_EVAL:
                print("  nothing logged yet -- route a few and correct the ones it gets wrong.")
                return
            for ticket, want, got in MY_EVAL:
                mark = "ok" if want == got else "XX"
                print(f"  {mark}  wanted {want:8} got {got:8}  {ticket[:56]}")
            miss = sum(1 for _, w, g in MY_EVAL if w != g)
            print(f"\\n  {miss} misroute(s) in {len(MY_EVAL)} labelled tickets"
                  f"  ({1 - miss / len(MY_EVAL):.0%} accurate on YOUR eval set)")

    go.on_click(on_go)
    log.on_click(on_log)
    seen.on_click(on_seen)
    display(W.VBox([picker, text, W.HBox([go, seen]), W.HBox([should, log]), out]))


triage_console()
'''

LAB1_DESCRIPTIONS = '''
# ---------------------------------------------- Run it for real: the descriptions ARE the router
# Same model, same graph, same 21 scenarios. The only thing that changes is the three lines
# the supervisor reads. This is the A/B from Module 1's tool descriptions, one layer up.

VAGUE = {
    "billing": "Money things.",
    "tech":    "Technical things.",
    "account": "Account things.",
}

SHARPER = {
    "billing": ("Anything about money that has already moved or is about to: a charge, a "
                "refund, an invoice, a price, a plan change, a contract term someone was "
                "promised. Route here when the customer's loss is financial."),
    "tech":    ("Anything the product is doing wrong: an error, a 500, a page that will not "
                "load, an integration that stopped, a deploy that broke something. Route "
                "here when something that used to work does not. A billing PAGE that errors "
                "is a tech ticket."),
    "account": ("Anything about who may do what: seats, owners, roles, permissions, sign-in, "
                "offboarding, access that should or should not exist. Route here when the "
                "answer is about a person rather than about money or a bug."),
}

if llm_ready():
    print("three one-line descriptions (the original):")
    base = ROUTED if "ROUTED" in dir() else sweep(quiet=True)
    print(f"  accuracy {accuracy(base):.0%}\\n")
    for label, desks in (("vague", VAGUE), ("sharper", SHARPER)):
        r = sweep(desks, quiet=True)
        print(f"{label} descriptions:")
        print(f"  accuracy {accuracy(r):.0%}")
        for (want, got), n in sorted(confusion(r).items(), key=lambda kv: -kv[1]):
            print(f"    {n}x  {want} -> {got}")
        print()
    print("  Nothing about the model or the graph changed between those three runs.")
'''

LAB1_STRUCTURED = '''
# ---------------------------------------------- Run it for real: why not with_structured_output?
# The framework way to get a field out of a model is a schema. Here is what it costs on this
# gateway, on the same three tickets, so the choice in ask_the_router is one you can check.

def route_with_schema(tickets: list[str]) -> tuple[list, float]:
    """The framework way: a Pydantic schema and with_structured_output."""
    from pydantic import BaseModel, Field
    class Routing(BaseModel):
        desk: str = Field(description="one of: billing, tech, account")
        why:  str = Field(description="at most 10 words")
    router = get_llm().with_structured_output(Routing)
    system = "Route one customer support ticket to exactly one desk.\\n" + desk_listing(DESK)
    t0 = time.time()
    picks = []
    for t in tickets:
        r = router.invoke([("system", system), ("human", t)])
        picks.append(None if r is None else r.desk)
    return picks, time.time() - t0


def route_with_two_lines(tickets: list[str]) -> tuple[list, float]:
    t0 = time.time()
    picks = [ask_the_router(t)[0] for t in tickets]
    return picks, time.time() - t0


if llm_ready():
    hard = [t for t, _, tag in SCENARIOS
            if tag in ("keyword trap", "contested primary", "too vague to route",
                       "two desks at once")]
    sample = hard[:8]
    schema_picks, schema_s = route_with_schema(sample)
    plain_picks, plain_s = route_with_two_lines(sample)

    agreed = sum(1 for a, b in zip(schema_picks, plain_picks) if a == b)
    print(f"{'ticket':46}{'schema':>10}{'two lines':>12}")
    print("-" * 68)
    for t, a, b in zip(sample, schema_picks, plain_picks):
        print(f"{t[:44]:46}{str(a):>10}{str(b):>12}" + ("" if a == b else "   <- differ"))
    print(f"\\n  with_structured_output  {schema_s:5.1f}s")
    print(f"  two plain lines         {plain_s:5.1f}s   ({schema_s / plain_s:.1f}x faster)")
    print(f"  they agreed on {agreed}/{len(sample)} of these tickets")
    print("\\n  A schema is guided decoding, and guided decoding is not free. Use it when the")
    print("  SHAPE matters more than the latency -- Lab 5.2's findings, not an app you click.")
    print("  Note the disagreements, if you got any: the schema call and the two-line call")
    print("  are different prompts, so this is not a pure latency comparison and you should")
    print("  not read it as one.")
'''

LAB1 = [
    header(1, "A Support Desk That Triages Itself", "Intermediate &rarr; Advanced", 40,
           ["Build the supervisor/worker graph where <em>every</em> node calls the model",
            "Route 29 tickets &mdash; keyword traps, contested primaries, typos, and five that cannot be routed at all",
            "Drive the triage console, and go hunting for a ticket the router actually gets wrong",
            "Rewrite the three desk descriptions and watch the accuracy move without touching the graph"],
           "> **This one is a walkthrough.** There is nothing to fill in and nothing to score &mdash;\n"
           "> every cell runs against the sandbox model, and the point is what the router *does* with\n"
           "> awkward tickets. Labs 5.2 and 5.3 go back to blanks and self-checks.\n"
           ">\n"
           "> **The system on slide 2.** Tickets arrive on one queue; a supervisor reads each one and\n"
           "> picks a desk; one desk answers; a `resolve` node closes it."),
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

The thing to watch for in this lab: the supervisor's entire program is the three sentences in
`DESK`. There is no other instruction anywhere in the system.
"""),

    md("""
## Section 1 &mdash; The graph, with the model in every node

The supervisor is a model call. So is each desk. Nothing here is stubbed, which is why every
cell in this notebook is a *Run it for real* cell.
"""),
    code(LAB1_GRAPH),
    code(LAB1_ONE),

    md("""
## Section 2 &mdash; Twenty-one tickets, and three of them have no right answer

Eight plain ones; three where the loudest word points at the wrong desk; four whose intent is
only implied; three written the way people actually write; six where two desks are both
mentioned and the answer turns on which fact is the *problem* &mdash; and five that genuinely
cannot be routed: too vague, two desks at once, and one that is not a support ticket at all.

Those last five are excluded from the accuracy. Watch what the router does with them anyway:
it will answer confidently, because nothing in the prompt gives it permission not to.
"""),
    code(LAB1_SWEEP),

    md("""
## Section 3 &mdash; The triage console

Pick a scenario or type your own. Route it, read what the desk says back, and label it &mdash;
that is how `MY_EVAL` fills up, and a ticket you disagreed with is worth ten you did not.

**The exercise: find a ticket this router gets wrong.** It is harder than it looks, and that is
the point &mdash; the twenty-nine above did not manage it. Things that tend to work: a ticket
whose problem belongs to one desk and whose *urgency* belongs to another; a ticket quoting an
error message from some other product; a ticket in a language the descriptions are not written
in; a ticket where the customer has already diagnosed it themselves, wrongly.

When you find one, log it. That ticket is now worth more than the whole starter set, because it
is the only one that can tell you whether tomorrow's change made things better or worse.
"""),
    code(LAB1_APP),

    md("""
## Section 4 &mdash; The descriptions are the router

Same model, same graph, same tickets. Three different sets of desk descriptions.

This is the Module 1 tool-description A/B one layer up: there, wording changed which *tool* an
agent picked; here it changes which *agent* the work goes to, and the blast radius is a whole
downstream conversation rather than one call.
"""),
    code(LAB1_DESCRIPTIONS),

    md("""
## Section 5 &mdash; And why the router does not use a schema

Worth knowing before you reach for `with_structured_output` in something a person is waiting on.
"""),
    code(LAB1_STRUCTURED),

    md("""
## Your turn

1. Add a fourth desk &mdash; `none` &mdash; and give it a description that says what does *not*
   belong on this queue. Re-run the sweep. The three unroutable tickets should move; check
   whether anything else moved with them, because a new option changes every decision, not just
   the ones you meant it to.
2. Make the supervisor say how sure it is, and route anything below your threshold to a human.
   You now have to pick the threshold, and `MY_EVAL` is the only evidence you have for it.
3. The desks answer without ever reading the ticket's history, the account, or the ledger. Give
   one of them a tool and watch the reply change &mdash; that is Module 4's work arriving inside
   Module 5's graph.
4. Time the console end to end. Two model calls per ticket is not free: decide out loud whether
   a first-line desk should route with a small model and answer with a large one.
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
    ("lab-5-01-support-desk-triage",     LAB1),   # WALKTHROUGH: no blanks, no score
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
