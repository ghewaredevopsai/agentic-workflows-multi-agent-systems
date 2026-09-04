#!/usr/bin/env python3
"""
Generate Module 5 lab notebooks and their solutions from one source.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-5-0N-*.ipynb and ../solutions/

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
# Lab 5.{num} &mdash; {title}

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 2 &middot; Module 5 &mdash; Multi-Agent Collaboration &amp; Orchestration**

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

WORK = os.path.join("/tmp", "awmas-lab-5-{num:02d}")
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




# the specialists, shared by labs 5.2 onward
AGENTKIT = '''
# ------------------------------------------------- the specialists
# Deterministic stand-ins. Each takes the run state and returns a PARTIAL state -- exactly
# the LangGraph node shape from Module 3 -- and reports what it spent. No model is called,
# so a graph's structure AND its cost can be graded offline and exactly. The "Run it for
# real" cells put the sandbox model behind the same interface.

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
    header(1, "The Supervisor Is a Router", "Intermediate &rarr; Advanced", 30,
           ["Build a rule-based supervisor, and measure its routing accuracy honestly",
            "Read the confusion table &mdash; and notice where every unrecognised request piles up",
            "Price a misroute: the wasted tokens are everything downstream of the mistake",
            "Put the model behind the same interface and compare on the same eval set"],
           "> **Lab 4.2's harness, unchanged.** A supervisor picks a worker the way an agent picks\n"
           "> a tool, so the measurement is the same one &mdash; which is why this lab comes first."),
    setup(1),
    code(DOMAIN),

    md("""
## Concept

A supervisor decides which specialist handles a request. That is a classification problem with a
known correct answer, so it has an accuracy &mdash; and almost nobody measures it.

It is worth measuring because a misroute is the most expensive mistake in the graph: everything
spent downstream of it answered the wrong question. The supervisor's own call is the cheapest one
in the system, so &ldquo;save money on the router&rdquo; is usually a bad trade.
"""),

    md("""
## Section 1 &mdash; A rule-based supervisor

Keywords, in order, with a default. Unglamorous, free, instant, and identical every time &mdash; and
right far more often than people expect.
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

def route_by_rule(request: str, default: str = "ledger") -> str:
    """The first keyword group that matches wins. An unrecognised request goes to the default."""
    low = (request or "").lower()
    for specialist, keywords in ROUTING_KEYWORDS:
        if BLANK:                # TODO: does the request mention any keyword in this group?
            return specialist
    return default
''', '''
SPECIALISTS = ("ledger", "policy", "sanctions", "writer")

# Order matters: the most specific group goes first.
ROUTING_KEYWORDS = [
    ("sanctions", ("sanction", "embargo", "screening")),
    ("policy",    ("policy", "runbook", "rule", "limit", "breach", "allowed", "permitted")),
    ("writer",    ("draft", "write", "note", "letter", "summar")),
    ("ledger",    ("status", "amount", "record", "look up", "reference", "pull")),
]

def route_by_rule(request: str, default: str = "ledger") -> str:
    """The first keyword group that matches wins. An unrecognised request goes to the default."""
    low = (request or "").lower()
    for specialist, keywords in ROUTING_KEYWORDS:
        if any(k in low for k in keywords):
            return specialist
    return default
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
check("an unrecognised request goes to the default rather than to a guess",
      lambda: route_by_rule("Tell the client what happened and why.") == "ledger")
check("the default is configurable, because the right default is domain-specific",
      lambda: route_by_rule("nothing matches here", default="writer") == "writer")
check("every rule points at a specialist that exists",
      lambda: all(s in SPECIALISTS for s, _ in ROUTING_KEYWORDS))
'''),

    md("""
## Section 2 &mdash; Measure it

Fifteen requests with a known correct specialist. Four of them state their intent only by
implication &mdash; no keyword names it &mdash; because those are the cases a rule table cannot reach and
the reason anyone reaches for a model.
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
check("every miss lands on the DEFAULT, not on a random specialist",
      lambda: {chosen for _, chosen in confusion(rule_selections())} == {"ledger"},
      "a rule router's failure mode is its default -- that is where unrecognised intent piles up")
check("so the confusion table names one problem, not four",
      lambda: len({chosen for _, chosen in confusion(rule_selections())}) == 1)
'''),

    md("""
## Section 3 &mdash; What a misroute costs

The supervisor's own call is the cheapest thing in the graph. The specialist it wakes up is not.
Price the mistake and the argument about which model to route with settles itself.
"""),
    code('''
COST = {"supervisor": 120, "ledger": 380, "policy": 420, "sanctions": 90, "writer": 610}

def cost_of(chosen: str) -> int:
    """Tokens spent on one routing decision plus the specialist it woke up."""
    return COST["supervisor"] + COST.get(chosen, 0)


def wasted_tokens(sel: dict) -> int:
    """Tokens spent answering the wrong question."""
    total = 0
    for request, expected in ROUTE_EVAL:
        chosen = sel.get(request)
        if chosen and chosen != expected:
            total += BLANK          # TODO: everything that misroute cost
    return total


def spent_tokens(sel: dict) -> int:
    """Everything the run spent, right or wrong."""
    return sum(cost_of(sel[r]) for r, _ in ROUTE_EVAL if sel.get(r))
''', '''
COST = {"supervisor": 120, "ledger": 380, "policy": 420, "sanctions": 90, "writer": 610}

def cost_of(chosen: str) -> int:
    """Tokens spent on one routing decision plus the specialist it woke up."""
    return COST["supervisor"] + COST.get(chosen, 0)


def wasted_tokens(sel: dict) -> int:
    """Tokens spent answering the wrong question."""
    total = 0
    for request, expected in ROUTE_EVAL:
        chosen = sel.get(request)
        if chosen and chosen != expected:
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
              == COST["supervisor"] + COST["ledger"])
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
## Run it for real &mdash; the model behind the same interface

Same eval set, same metric, same confusion table. The only thing that changes is the router.
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

def route_with_model(request: str, default: str = "ledger") -> str:
    """Ask the model to pick a specialist. Anything unrecognised falls back to the default."""
    listing = "\\n".join(f"- {n}: {d}" for n, d in SPECIALIST_DESCRIPTIONS.items())
    reply = ask(f"Specialists:\\n{listing}\\n\\nRequest: {request}\\n\\nSpecialist:",
                system=ROUTE_SYSTEM)
    word = (reply or "").strip().strip("`.\\"' ").split()
    return word[0] if word and word[0] in SPECIALIST_DESCRIPTIONS else default


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

1. **Did the model beat 73%?** If not, the rule table is free and deterministic, and you have your
   answer.
2. **Where did the model's misses land?** The rule router's misses all pile up on the default,
   which is one problem you can name. If the model's misses are scattered across four specialists,
   that is four overlapping descriptions &mdash; and Module 4 told you how to fix each one.
3. **Run it twice.** If the same request routes differently on the second run, you have met
   Module 7's opening problem a day early.

The usual production answer is neither: rules for the requests you can name, and a model only for
the ones that fall through &mdash; so you pay for the model on the four cases, not on all fifteen.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Build that hybrid: rules first, model only on a fall-through. Measure its accuracy and its
   cost, and decide whether the saving is worth the second code path.
2. `route_by_rule` returns the first match, so a request mentioning both a policy and a draft goes
   to policy. Make it return every match and let the supervisor dispatch two specialists. What have
   you just committed to paying?
3. Add a fifth outcome to the eval set: `clarify` &mdash; requests where the honest answer is a
   question back to the user. What does that do to your accuracy, and is the drop real?
"""),
]


# =========================================================================== #
# Lab 5.2 -- decomposition, distribution, and what a handoff drops
# =========================================================================== #
LAB2 = [
    header(2, "Decomposition, Distribution and Handoffs", "Advanced", 35,
           ["Turn a dependency graph into execution waves &mdash; what may run at once, and what may not",
            "Thread state through a run and watch the critical path refuse to get shorter",
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

The second one produces the bug in Section 4, where both agents behave correctly and the
recommendation is still wrong.
"""),

    md("""
## Section 1 &mdash; What may run at once

Level the dependency graph into waves. Everything in one wave is independent; the number of waves
is the critical path, and no amount of parallelism shortens it.
"""),
    code('''
TASKS = {
    "read":      {"agent": "ledger",    "needs": []},
    "policy":    {"agent": "policy",    "needs": ["read"]},
    "screen":    {"agent": "sanctions", "needs": ["read"]},
    "recommend": {"agent": "writer",    "needs": ["policy", "screen"]},
}

def waves(tasks: dict) -> list:
    """Group tasks into execution waves. Everything within a wave may run at the same time."""
    done, out, remaining = set(), [], dict(tasks)
    while remaining:
        ready = [name for name, t in remaining.items()
                 if BLANK]              # TODO: is every one of this task's needs already done?
        if not ready:
            raise ValueError(f"circular dependency among {sorted(remaining)}")
        out.append(sorted(ready))
        done |= set(ready)
        for name in ready:
            remaining.pop(name)
    return out
''', '''
TASKS = {
    "read":      {"agent": "ledger",    "needs": []},
    "policy":    {"agent": "policy",    "needs": ["read"]},
    "screen":    {"agent": "sanctions", "needs": ["read"]},
    "recommend": {"agent": "writer",    "needs": ["policy", "screen"]},
}

def waves(tasks: dict) -> list:
    """Group tasks into execution waves. Everything within a wave may run at the same time."""
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

check("four tasks resolve into three waves",
      lambda: len(waves(TASKS)) == 3)
check("nothing can start before the payment is read",
      lambda: waves(TASKS)[0] == ["read"])
check("policy and screening are independent, so they share a wave",
      lambda: waves(TASKS)[1] == ["policy", "screen"])
check("the recommendation waits for both",
      lambda: waves(TASKS)[2] == ["recommend"])
check("every task appears exactly once",
      lambda: sorted(t for w in waves(TASKS) for t in w) == sorted(TASKS))
check("the widest wave is two, so four agents never run four-abreast",
      lambda: max(len(w) for w in waves(TASKS)) == 2,
      "the critical path is three hops whatever you spend on parallelism")
def guard_raises(fn, exc) -> bool:
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

check("a circular dependency is refused rather than looping forever",
      lambda: guard_raises(lambda: waves(_cycle), ValueError))
'''),

    md("""
## Section 2 &mdash; Run it

Fold each agent's partial state into the run state, wave by wave. This merge is deliberately
simple: within a wave the agents here write different keys, so nothing collides. Lab 5.3 removes
that comfort.
"""),
    code('''
def merge(state: dict, partial: dict) -> dict:
    """Fold one agent's partial state into the run state.

    Lists accumulate, counters add, everything else is replaced. That last rule is the one
    that goes wrong once two agents run at the same time -- which is Lab 5.3.
    """
    out = dict(state)
    for key, value in partial.items():
        if key == "tokens":
            out["tokens"] = out.get("tokens", 0) + value
        elif key in ("findings", "problems"):
            out[key] = (out.get(key) or []) + value
        else:
            out[key] = value
    return out


def run_waves(ref: str, tasks: dict = None) -> dict:
    """Run every wave in order, threading the state through."""
    tasks = TASKS if tasks is None else tasks
    state, order = {"ref": ref, "tokens": 0}, []
    for wave in waves(tasks):
        for name in wave:
            state = merge(state, AGENTS[tasks[name]["agent"]](state))
            order.append(name)
    return {"state": state, "order": order, "waves": waves(tasks)}


def _show():
    out = run_waves("PMT-1005")
    print("  waves:", out["waves"])
    print("  spend:", out["state"]["tokens"], "tokens")
    print("  says :", out["state"].get("recommendation"))
    for f in out["state"]["findings"]:
        print(f"    [{f['by']:9} src={f['source']:9}] {f['claim'][:60]}")
guard(_show)
'''),
    code('''
# --- Self-check: Section 2
def run1005():
    return run_waves("PMT-1005")["state"]

check("all four tasks ran",
      lambda: len(run_waves("PMT-1005")["order"]) == 4)
check("and in dependency order -- read first, recommend last",
      lambda: run_waves("PMT-1005")["order"][0] == "read"
              and run_waves("PMT-1005")["order"][-1] == "recommend")
check("three specialists each contributed a finding",
      lambda: len(run1005()["findings"]) == 3)
check("the watchlisted counterparty was caught",
      lambda: run1005()["blocked"] is True)
check("and the reason code independently demands a human",
      lambda: run1005()["needs_human"] is True)
check("so the recommendation is to hold",
      lambda: run1005()["recommendation"] == "hold for a human")
check("the bill is the sum of what each specialist spent",
      lambda: run1005()["tokens"] == COST["ledger"] + COST["policy"]
                                   + COST["sanctions"] + COST["writer"])
check("a settled payment needs no action",
      lambda: run_waves("PMT-1001")["state"]["recommendation"] == "no action")
check("a payment that does not exist does not crash the run",
      lambda: "problems" in run_waves("PMT-0000")["state"],
      "the whole graph must survive one specialist finding nothing")
'''),

    md("""
## Section 3 &mdash; The handoff payload

A handoff feels like passing a task. What actually crosses is whatever you put in the message,
and nothing else. Build it.
"""),
    code('''
def handoff(state: dict, to_agent: str, task: str) -> dict:
    """The message one agent sends another. ONLY what is in this dict crosses."""
    message = {"to": to_agent, "task": task, "facts": state.get("facts")}
    # TODO: a handoff silently drops three things -- the reasoning, the constraints and the
    # failure history. Add the two this graph keeps in its state, defaulting each to an
    # empty list so the receiving agent always finds the key.
    message.update(BLANK)
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

Run it both ways and watch the constraint decide the outcome.
"""),
    code('''
def policy_from_handoff(message: dict) -> dict:
    """A policy agent that knows only what the handoff told it -- which is the realistic case."""
    constraints = message.get("constraints") or []
    if any(BLANK for c in constraints):     # TODO: does any constraint forbid a release?
        return {"recommendation": "hold for a human", "why": "a constraint forbids release"}
    code = (message.get("facts") or {}).get("reason_code")
    return {"recommendation": "release", "why": POLICY.get(code, "no policy on file")}


def investigate(ref: str, carry_constraints: bool = True) -> dict:
    """Triage reads the payment, sets a constraint if one applies, then hands off."""
    state = merge({"ref": ref, "tokens": 0}, agent_ledger({"ref": ref}))
    if (state.get("facts") or {}).get("reason_code") in NEEDS_HUMAN:
        state["constraints"] = [f"Do not release {ref} without a human decision"]
    message = handoff(state, "policy", f"decide whether {ref} can be released")
    if not carry_constraints:
        message.pop("constraints", None)    # the bug, made explicit
    return policy_from_handoff(message)
''', '''
def policy_from_handoff(message: dict) -> dict:
    """A policy agent that knows only what the handoff told it -- which is the realistic case."""
    constraints = message.get("constraints") or []
    if any("do not release" in c.lower() for c in constraints):
        return {"recommendation": "hold for a human", "why": "a constraint forbids release"}
    code = (message.get("facts") or {}).get("reason_code")
    return {"recommendation": "release", "why": POLICY.get(code, "no policy on file")}


def investigate(ref: str, carry_constraints: bool = True) -> dict:
    """Triage reads the payment, sets a constraint if one applies, then hands off."""
    state = merge({"ref": ref, "tokens": 0}, agent_ledger({"ref": ref}))
    if (state.get("facts") or {}).get("reason_code") in NEEDS_HUMAN:
        state["constraints"] = [f"Do not release {ref} without a human decision"]
    message = handoff(state, "policy", f"decide whether {ref} can be released")
    if not carry_constraints:
        message.pop("constraints", None)    # the bug, made explicit
    return policy_from_handoff(message)
'''),
    code('''
# --- Self-check: Section 4
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
        state = merge({"ref": "PMT-1005", "tokens": 0}, agent_ledger({"ref": "PMT-1005"}))
        state["constraints"] = ["Do not release PMT-1005 without a human decision"]
        full = handoff(state, "policy", "decide whether PMT-1005 can be released")
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
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add the third dropped thing &mdash; the reasoning &mdash; and measure what carrying it costs in
   tokens against what re-deriving it costs. One of those is a bill and one is a risk.
2. `waves` assumes every task in a wave succeeds. Make `screen` fail and decide what wave 3 should
   receive: a partial state, or nothing at all.
3. Turn `TASKS` into two graphs &mdash; one for high-value payments and one for low &mdash; and let
   Lab 5.1's supervisor choose between them. That is routing by value, and Lab 5.5 prices it.
"""),
]


# =========================================================================== #
# Lab 5.3 -- parallel execution, reducers, and disagreement
# =========================================================================== #
LAB3 = [
    header(3, "Parallel Execution, Reducers and Disagreement", "Advanced", 40,
           ["Reproduce the parallel bug that never raises &mdash; a whole specialist's work, gone",
            "Declare a reducer per key, and prove the findings survive",
            "Write the test that catches it, because no log line will",
            "Settle a disagreement by authority and provenance rather than by headcount"],
           "> **Module 3's reducers, with consequences.** There a lost key was a puzzle.\n"
           "> Here it is a compliance finding that never reached the recommendation."),
    setup(3),
    code(DOMAIN),
    code(AGENTKIT),

    md("""
## Concept

Two agents in the same wave both return `{"findings": [...]}`. Fold them in with a plain
`dict.update` and the second overwrites the first. Both calls succeeded, nothing raised, and one
specialist's entire contribution is gone.

The summary still reads perfectly &mdash; a summary of one finding reads exactly as well as a summary
of two. **Nothing in your logs will show this.** Only a test that counts.
"""),

    md("""
## Section 1 &mdash; Watch it disappear

Run the two independent specialists on the same state, then merge them the obvious way.
"""),
    code('''
def fan_out(state: dict, agent_names) -> list:
    """Run several agents on the SAME input state. None of them sees the others' output."""
    return [(name, AGENTS[name](dict(state))) for name in agent_names]


def merge_naive(state: dict, partials: list) -> dict:
    """Last write wins. This is what you get when you do not declare a reducer."""
    out = dict(state)
    for _, partial in partials:
        out.update(partial)
    return out


def base_state(ref: str = "PMT-1005") -> dict:
    """Everything wave 1 established, ready for the parallel wave."""
    seed = {"ref": ref, "tokens": 0}
    first = agent_ledger(seed)
    return {**seed, **first}


def _demo():
    partials = fan_out(base_state(), ["policy", "sanctions"])
    returned = sum(len(p.get("findings") or []) for _, p in partials)
    merged = merge_naive(base_state(), partials)
    print(f"  the two specialists returned  {returned} findings")
    print(f"  the merged state contains     {len(merged.get('findings') or [])}")
    print(f"  errors raised                 0")
    for f in merged.get("findings") or []:
        print(f"    survivor: [{f['by']}] {f['claim'][:56]}")
guard(_demo)
'''),
    code('''
# --- Self-check: Section 1
def _partials():
    return fan_out(base_state(), ["policy", "sanctions"])

check("both specialists really did return a finding",
      lambda: sum(len(p.get("findings") or []) for _, p in _partials()) == 2)
check("neither of them raised",
      lambda: all(isinstance(p, dict) for _, p in _partials()))
check("but the naive merge keeps only one",
      lambda: len(merge_naive(base_state(), _partials()).get("findings") or []) == 1,
      "one specialist's entire contribution is gone, and nothing said so")
check("the survivor is whichever one ran last -- an ordering accident",
      lambda: merge_naive(base_state(), _partials())["findings"][0]["by"] == "sanctions")
check("the token count is wrong too, and in the cheaper direction",
      lambda: merge_naive(base_state(), _partials())["tokens"] == COST["sanctions"],
      "you will under-report your own spend, which is the one bug nobody reports")
'''),

    md("""
## Section 2 &mdash; Declare a reducer per key

A reducer says how two writes to the same key combine. Lists append. Counters add. Booleans that
mean &ldquo;something is wrong&rdquo; OR together, so one blocker is enough to block.
"""),
    code('''
REDUCERS = {
    "findings":    lambda old, new: (old or []) + new,
    "problems":    lambda old, new: (old or []) + new,
    "tokens":      lambda old, new: (old or 0) + new,
    "blocked":     lambda old, new: bool(old) or bool(new),
    "needs_human": lambda old, new: bool(old) or bool(new),
}

def merge_reduced(state: dict, partials: list) -> dict:
    """Fold parallel results through the declared reducer for each key."""
    out = dict(state)
    for _, partial in partials:
        for key, value in partial.items():
            if key in REDUCERS:
                out[key] = BLANK       # TODO: combine what is already there with what arrived
            else:
                out[key] = value       # no reducer declared: last write still wins
    return out
''', '''
REDUCERS = {
    "findings":    lambda old, new: (old or []) + new,
    "problems":    lambda old, new: (old or []) + new,
    "tokens":      lambda old, new: (old or 0) + new,
    "blocked":     lambda old, new: bool(old) or bool(new),
    "needs_human": lambda old, new: bool(old) or bool(new),
}

def merge_reduced(state: dict, partials: list) -> dict:
    """Fold parallel results through the declared reducer for each key."""
    out = dict(state)
    for _, partial in partials:
        for key, value in partial.items():
            if key in REDUCERS:
                out[key] = REDUCERS[key](out.get(key), value)
            else:
                out[key] = value       # no reducer declared: last write still wins
    return out
'''),
    code('''
# --- Self-check: Section 2
def _reduced():
    return merge_reduced(base_state(), _partials())

check("both parallel findings survive, on top of what wave 1 established",
      lambda: len(_reduced()["findings"]) == 3)
check("so all three specialists are represented",
      lambda: {f["by"] for f in _reduced()["findings"]} == {"ledger", "policy", "sanctions"})
check("the spend is now the sum of all three, not whichever wrote last",
      lambda: _reduced()["tokens"] == COST["ledger"] + COST["policy"] + COST["sanctions"])
check("one blocker is enough to block",
      lambda: _reduced()["blocked"] is True)
check("merge order does not change the outcome",
      lambda: sorted(f["by"] for f in
                     merge_reduced(base_state(), list(reversed(_partials())))["findings"])
              == sorted(f["by"] for f in _reduced()["findings"]),
      "a reducer you can apply in either order is one that survives a scheduler you do not control")
check("a key with no declared reducer is still replaced",
      lambda: merge_reduced({"facts": {"a": 1}}, [("x", {"facts": {"b": 2}})])["facts"] == {"b": 2})
check("EVERY key the parallel specialists write has a reducer declared",
      lambda: {k for _, p in _partials() for k in p} <= set(REDUCERS),
      "this is the check to run in CI -- a new specialist writing a new key is the next silent loss")
'''),

    md("""
## Section 3 &mdash; The test that catches it

No log line shows this. What shows it is asserting that everyone you dispatched came back.
"""),
    code('''
def contributed(state: dict) -> set:
    """Which specialists actually appear in the merged findings."""
    return {f["by"] for f in (state.get("findings") or [])}


def everyone_came_back(state: dict, dispatched) -> bool:
    """The assertion that catches a silent parallel loss: count what came back."""
    return BLANK                # TODO: is every dispatched specialist represented?


def missing(state: dict, dispatched) -> set:
    """Who was dispatched and is not in the findings."""
    return set(dispatched) - contributed(state)
''', '''
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
_dispatched = ["policy", "sanctions"]

check("the reduced merge passes the test",
      lambda: everyone_came_back(_reduced(), _dispatched) is True)
check("the naive merge FAILS it",
      lambda: everyone_came_back(merge_naive(base_state(), _partials()), _dispatched) is False,
      "this single assertion is the whole defence against the bug in Section 1")
check("and the failure names who went missing",
      lambda: missing(merge_naive(base_state(), _partials()), _dispatched) == {"policy"})
check("nothing is missing from a correct merge",
      lambda: missing(_reduced(), _dispatched) == set())
check("a specialist that returned no finding at all is also caught",
      lambda: everyone_came_back(_reduced(), ["policy", "sanctions", "writer"]) is False,
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
    top = max(findings, key=lambda f: BLANK)   # TODO: rank a finding by declared authority
    return top["verdict"]


def checkable(finding: dict) -> bool:
    """Can this claim be settled by re-reading a source, rather than by asking again?"""
    return finding["source"] in CHECKABLE_SOURCES
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
check("the two rules disagree on this case",
      lambda: by_majority(CONFLICT) != by_authority(CONFLICT))
check("an agent with no declared authority ranks below every one that has it",
      lambda: by_authority(CONFLICT + [{"by": "stranger", "source": "inference",
                                        "verdict": "release", "claim": "looks fine"}]) == "hold")
check("authority is declared in advance, not derived from the case",
      lambda: set(AUTHORITY) >= {f["by"] for f in CONFLICT},
      "a rule chosen while looking at one disagreement is a rule fitted to that disagreement")
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

**What you take from this lab:** declare a reducer for every key two branches can write, assert
that everyone you dispatched came back, and settle disagreements on authority and sources rather
than on a headcount.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add a fifth specialist that writes a new key with no reducer. The CI check in Section 2 should
   fail. Make sure it does &mdash; that check is the only thing standing between you and the next
   silent loss.
2. `by_authority` breaks ties arbitrarily. Two equal-authority agents disagreeing is a real case:
   decide whether it escalates or falls back to provenance, and write it.
3. The `blocked` reducer ORs, so one blocker blocks. Build the opposite case &mdash; a key where OR is
   wrong &mdash; and say what that tells you about choosing reducers by data type.
"""),
]


# =========================================================================== #
# Lab 5.4 -- human-in-the-loop: interrupt, approve, time out, escalate
# =========================================================================== #
LAB4 = [
    header(4, "Human-in-the-Loop as an Orchestration Mechanism", "Advanced", 35,
           ["Interrupt a run before the irreversible node and checkpoint what it knows",
            "Open the gate for a named human &mdash; an identity, not a boolean",
            "Time out, and climb an escalation ladder that actually terminates",
            "Prove the one property that makes a gate a gate: at the gate, &ldquo;no&rdquo; is still free"],
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
| **interrupt** | stop before a named node and write the state down |
| **approve** | resume, recording *who* said yes and on what evidence |
| **timeout** | a gate with no deadline is a run that waits until Monday |
| **escalate** | expiry is not refusal and not approval &mdash; it is a different queue |
"""),

    md("""
## Section 1 &mdash; Interrupt before the irreversible node

The whole graph, including a `release` node that actually changes something. Stop before it.
"""),
    code('''
RELEASED = set()          # the irreversible side effect, so we can prove whether it happened

def node_release(state: dict) -> dict:
    """The one node that changes the world."""
    RELEASED.add(state["ref"])
    return {"released": True, "tokens": 40}


NODES = {"read": agent_ledger, "policy": agent_policy, "screen": agent_sanctions,
         "recommend": agent_writer, "release": node_release}
PLAN = ["read", "policy", "screen", "recommend", "release"]

INTERRUPT_BEFORE = {"release"}       # nodes that may not run unattended


def merge_one(state: dict, partial: dict) -> dict:
    """Fold one node's partial state in. Lists accumulate, counters add."""
    out = dict(state)
    for key, value in partial.items():
        if key == "tokens":
            out["tokens"] = out.get("tokens", 0) + value
        elif key in ("findings", "problems"):
            out[key] = (out.get(key) or []) + value
        else:
            out[key] = value
    return out


def run_until_interrupt(ref: str, plan=None, interrupt_before=None) -> dict:
    """Run nodes in order, stopping BEFORE any node that needs a human."""
    plan = PLAN if plan is None else plan
    interrupt_before = INTERRUPT_BEFORE if interrupt_before is None else interrupt_before
    state, trace = {"ref": ref, "tokens": 0}, []
    for node in plan:
        if BLANK:              # TODO: must this node wait for a human before it runs?
            return {"status": "interrupted", "at": node, "state": state,
                    "trace": trace, "remaining": plan[plan.index(node):]}
        state = merge_one(state, NODES[node](state))
        trace.append(node)
    return {"status": "completed", "at": None, "state": state, "trace": trace, "remaining": []}
''', '''
RELEASED = set()          # the irreversible side effect, so we can prove whether it happened

def node_release(state: dict) -> dict:
    """The one node that changes the world."""
    RELEASED.add(state["ref"])
    return {"released": True, "tokens": 40}


NODES = {"read": agent_ledger, "policy": agent_policy, "screen": agent_sanctions,
         "recommend": agent_writer, "release": node_release}
PLAN = ["read", "policy", "screen", "recommend", "release"]

INTERRUPT_BEFORE = {"release"}       # nodes that may not run unattended


def merge_one(state: dict, partial: dict) -> dict:
    """Fold one node's partial state in. Lists accumulate, counters add."""
    out = dict(state)
    for key, value in partial.items():
        if key == "tokens":
            out["tokens"] = out.get("tokens", 0) + value
        elif key in ("findings", "problems"):
            out[key] = (out.get(key) or []) + value
        else:
            out[key] = value
    return out


def run_until_interrupt(ref: str, plan=None, interrupt_before=None) -> dict:
    """Run nodes in order, stopping BEFORE any node that needs a human."""
    plan = PLAN if plan is None else plan
    interrupt_before = INTERRUPT_BEFORE if interrupt_before is None else interrupt_before
    state, trace = {"ref": ref, "tokens": 0}, []
    for node in plan:
        if node in interrupt_before:
            return {"status": "interrupted", "at": node, "state": state,
                    "trace": trace, "remaining": plan[plan.index(node):]}
        state = merge_one(state, NODES[node](state))
        trace.append(node)
    return {"status": "completed", "at": None, "state": state, "trace": trace, "remaining": []}
'''),
    code('''
# --- Self-check: Section 1
def interrupted():
    RELEASED.clear()
    return run_until_interrupt("PMT-1005")

check("the run stops rather than finishing",
      lambda: interrupted()["status"] == "interrupted")
check("and it stops at the release node",
      lambda: interrupted()["at"] == "release")
check("everything before it did run",
      lambda: interrupted()["trace"] == ["read", "policy", "screen", "recommend"])
check("NOTHING WAS RELEASED",
      lambda: (interrupted(), "PMT-1005" not in RELEASED)[1] is True,
      "the point of interrupting before the node rather than after it")
check("the checkpoint carries the evidence a human needs",
      lambda: len(interrupted()["state"]["findings"]) == 3)
check("and a recommendation to agree or disagree with",
      lambda: interrupted()["state"]["recommendation"] == "hold for a human")
check("and it knows what is left to do",
      lambda: interrupted()["remaining"] == ["release"])
check("a graph with no gated node runs straight through",
      lambda: run_until_interrupt("PMT-1002", plan=["read", "policy"],
                                  interrupt_before=set())["status"] == "completed")
'''),

    md("""
## Section 2 &mdash; Approval is an identity, not a boolean

Resume the checkpoint. The gate opens for a *named* person &mdash; because &ldquo;approved: true&rdquo;
answers none of the questions an auditor will ask.
"""),
    code('''
def resume(checkpoint: dict, approved_by=None) -> dict:
    """Resume an interrupted run. Refuse unless a named human approved it."""
    # TODO: refuse unless approved_by is a real name. A boolean is not an approver,
    # and neither is an empty string.
    if BLANK:
        return {"status": "refused", "state": checkpoint["state"],
                "reason": f"{checkpoint['at']} needs a named human approver"}
    state = dict(checkpoint["state"])
    state["approved_by"] = approved_by
    trace = list(checkpoint["trace"])
    for node in checkpoint["remaining"]:
        state = merge_one(state, NODES[node](state))
        trace.append(node)
    return {"status": "completed", "state": state, "trace": trace, "reason": ""}
''', '''
def resume(checkpoint: dict, approved_by=None) -> dict:
    """Resume an interrupted run. Refuse unless a named human approved it."""
    if not isinstance(approved_by, str) or not approved_by.strip():
        return {"status": "refused", "state": checkpoint["state"],
                "reason": f"{checkpoint['at']} needs a named human approver"}
    state = dict(checkpoint["state"])
    state["approved_by"] = approved_by
    trace = list(checkpoint["trace"])
    for node in checkpoint["remaining"]:
        state = merge_one(state, NODES[node](state))
        trace.append(node)
    return {"status": "completed", "state": state, "trace": trace, "reason": ""}
'''),
    code('''
# --- Self-check: Section 2
def _resume(approver):
    RELEASED.clear()
    return resume(run_until_interrupt("PMT-1005"), approved_by=approver)

check("resuming with no approver is refused",
      lambda: _resume(None)["status"] == "refused")
check("and still nothing was released",
      lambda: (_resume(None), "PMT-1005" not in RELEASED)[1] is True)
check("a bare True is NOT an approver",
      lambda: _resume(True)["status"] == "refused",
      "'approved: true' cannot answer 'who approved this, and on what evidence?'")
check("nor is an empty string",
      lambda: _resume("   ")["status"] == "refused")
check("a named human opens the gate",
      lambda: _resume("ops-duty-manager")["status"] == "completed")
check("and the release actually happened",
      lambda: (_resume("ops-duty-manager"), "PMT-1005" in RELEASED)[1] is True)
check("the approver's name is on the final state",
      lambda: _resume("ops-duty-manager")["state"]["approved_by"] == "ops-duty-manager")
check("the trace shows the whole run, both halves",
      lambda: _resume("ops-duty-manager")["trace"]
              == ["read", "policy", "screen", "recommend", "release"])
'''),

    md("""
## Section 3 &mdash; Timeout, and a ladder that ends

A gate with no deadline is a run that waits for someone who has gone home. Expiry is not refusal
and it is not approval &mdash; it is a different queue, and the queue eventually runs out.
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
    return BLANK               # TODO: the next rung up, or None if this is already the top


def gate_status(waited_s: int, deadline_s: int, approver=None) -> str:
    """What to do with a gate that has been waiting: approved, waiting, or time to escalate."""
    if isinstance(approver, str) and approver.strip():
        return "approved"
    return "waiting" if waited_s < deadline_s else "escalate"
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
    """What to do with a gate that has been waiting: approved, waiting, or time to escalate."""
    if isinstance(approver, str) and approver.strip():
        return "approved"
    return "waiting" if waited_s < deadline_s else "escalate"
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
check("past the deadline it escalates",
      lambda: gate_status(waited_s=901, deadline_s=900) == "escalate")
check("an approval short-circuits the deadline entirely",
      lambda: gate_status(waited_s=99999, deadline_s=900, approver="treasury-lead") == "approved")
check("expiry is neither approval nor refusal",
      lambda: gate_status(waited_s=901, deadline_s=900) not in ("approved", "refused"),
      "a timeout that auto-approves is not a gate; one that auto-refuses loses real work")

def _ladder():
    who, waited = None, 0
    while True:
        who = escalate(who)
        if who is None:
            print("  ladder exhausted -- this case now belongs to a person, not to the graph")
            break
        waited += 900
        print(f"  after {waited // 60:>3} min -> ask {who}")
guard(_ladder)
'''),

    md("""
## Section 4 &mdash; The test of a gate

One question decides whether you have built an approval gate or a notification:
**at the moment the human says no, has anything irreversible already happened?**
"""),
    code('''
def run_gate_before_write(ref: str, approved_by=None) -> dict:
    """Gate placed after the evidence and before the release."""
    checkpoint = run_until_interrupt(ref)
    return resume(checkpoint, approved_by=approved_by)


def run_gate_after_write(ref: str, approved_by=None) -> dict:
    """Gate placed at the end -- the shape that feels thorough and controls nothing."""
    state = {"ref": ref, "tokens": 0}
    for node in PLAN:                       # everything, release included
        state = merge_one(state, NODES[node](state))
    return {"status": "completed" if approved_by else "reviewer said no",
            "state": state, "trace": list(PLAN), "reason": ""}


def no_is_free(run_fn, ref: str) -> bool:
    """Run it, have the human refuse, and ask whether anything happened anyway."""
    RELEASED.clear()
    run_fn(ref, approved_by=None)
    return ref not in RELEASED
''', '''
def run_gate_before_write(ref: str, approved_by=None) -> dict:
    """Gate placed after the evidence and before the release."""
    checkpoint = run_until_interrupt(ref)
    return resume(checkpoint, approved_by=approved_by)


def run_gate_after_write(ref: str, approved_by=None) -> dict:
    """Gate placed at the end -- the shape that feels thorough and controls nothing."""
    state = {"ref": ref, "tokens": 0}
    for node in PLAN:                       # everything, release included
        state = merge_one(state, NODES[node](state))
    return {"status": "completed" if approved_by else "reviewer said no",
            "state": state, "trace": list(PLAN), "reason": ""}


def no_is_free(run_fn, ref: str) -> bool:
    """Run it, have the human refuse, and ask whether anything happened anyway."""
    RELEASED.clear()
    run_fn(ref, approved_by=None)
    return ref not in RELEASED
'''),
    code('''
# --- Self-check: Section 4
check("with the gate before the write, saying no costs nothing",
      lambda: no_is_free(run_gate_before_write, "PMT-1005") is True)
check("with the gate after the write, the payment already went",
      lambda: no_is_free(run_gate_after_write, "PMT-1005") is False,
      "the reviewer sees a complete, sourced summary of something they can no longer stop")
check("both runs show the reviewer exactly the same evidence",
      lambda: len(run_gate_after_write("PMT-1005")["state"]["findings"])
              == len(run_until_interrupt("PMT-1005")["state"]["findings"]),
      "quality of evidence was never the difference -- placement was")
check("only the first one is an approval gate",
      lambda: no_is_free(run_gate_before_write, "PMT-1005")
              and not no_is_free(run_gate_after_write, "PMT-1005"))

def _placement():
    for label, fn in (("before the write", run_gate_before_write),
                      ("after the write ", run_gate_after_write)):
        free = no_is_free(fn, "PMT-1005")
        print(f"  gate {label}:  'no' still free? {'yes -- a gate' if free else 'NO -- a notification'}")
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
        cp = run_until_interrupt("PMT-1005")
        evidence = "\\n".join(f"- [{f['by']}, source={f['source']}] {f['claim']}"
                             for f in cp["state"]["findings"])
        reply = ask("Write a short approval request for a duty manager. State what is being asked, "
                    "the evidence for and against, and what happens if they do nothing.\\n\\n"
                    f"Action awaiting approval: {cp['at']} {cp['state']['ref']}\\n"
                    f"Agent recommendation: {cp['state'].get('recommendation')}\\n"
                    f"Findings:\\n{evidence}")
        print(reply.strip()[:600])
    guard(_brief)
'''),
    md("""
### Read it

If the model has to hedge or invent, your checkpoint is missing something a reviewer needs &mdash;
and that is a state design problem, not a prompt problem. A good approval request is mostly a
rendering of state you already had.

**What you take from this lab:** interrupt before the node, not after it; record an identity
rather than a boolean; give every gate a deadline and a ladder that ends; and test placement with
one question &mdash; is &ldquo;no&rdquo; still free?
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `resume` trusts its caller for `approved_by`. Where does that name actually have to come from
   for the audit trail to mean anything, and what stops an agent from supplying it?
2. Add a fourth outcome to the gate: *approved with a change* &mdash; the human edits the
   recommendation before resuming. What does that do to the trace, and to who is responsible?
3. `run_until_interrupt` restarts from the checkpoint's remaining nodes. Make the human wait long
   enough that the ledger has changed underneath them, and decide what a resume owes a reviewer
   whose evidence is now stale.
"""),
]


# =========================================================================== #
# Lab 5.5 -- challenge: the scorecard, and when the answer is "don't"
# =========================================================================== #
LAB5 = [
    header(5, "Challenge: The Scorecard", "Advanced &middot; challenge", 40,
           ["State the ground truth &mdash; what the right recommendation actually is, and why",
            "Run a single agent and a four-specialist graph over the same cases",
            "Price both: quality, tokens, and the critical path",
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

**A note on what is graded here.** The two implementations below are readable *models* of the two
designs &mdash; deterministic, so the comparison is exact and repeatable offline. What is graded is
the **scorecard machinery**: the ground truth, the metric, the cost model and the decision rule.
Point them at your own system and the machinery is unchanged; only the numbers move.
"""),

    md("""
## Section 1 &mdash; Ground truth

Before measuring anything, say what the right answer is. Two extra cases are added here: a
watchlisted counterparty whose reason code says nothing about sanctions. Those are the cases that
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
CASES = {**LEDGER, **EXTRA}


def expected_recommendation(record: dict) -> str:
    """The correct answer for one payment, independent of any agent.

    Three outcomes: a settled payment needs nothing; anything a human must decide, or any
    watchlisted counterparty, is held; everything else proceeds.
    """
    if record["status"] == "settled":
        return "no action"
    # TODO: which two conditions each force a hold, whatever else the case says?
    if BLANK:
        return "hold for a human"
    return "proceed"


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
CASES = {**LEDGER, **EXTRA}


def expected_recommendation(record: dict) -> str:
    """The correct answer for one payment, independent of any agent.

    Three outcomes: a settled payment needs nothing; anything a human must decide, or any
    watchlisted counterparty, is held; everything else proceeds.
    """
    if record["status"] == "settled":
        return "no action"
    if record["reason_code"] in NEEDS_HUMAN or record["counterparty"] in SANCTIONS_WATCH:
        return "hold for a human"
    return "proceed"


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
check("an ordinary funding failure proceeds",
      lambda: expected_recommendation(CASES["PMT-1002"]) == "proceed")
check("a watchlisted counterparty is held even when its reason code is mundane",
      lambda: expected_recommendation(CASES["PMT-1006"]) == "hold for a human",
      "nothing in INSUFFICIENT_FUNDS hints at sanctions -- the counterparty is the whole reason")
check("and again for the second one",
      lambda: expected_recommendation(CASES["PMT-1007"]) == "hold for a human")
check("the eval set covers all three outcomes",
      lambda: {e for _, e in eval_cases()} == {"no action", "hold for a human", "proceed"})
check("seven cases in total",
      lambda: len(eval_cases()) == 7)
'''),

    md("""
## Section 2 &mdash; The two designs

The single agent reads the payment, consults policy when there is a reason code, and screens the
counterparty **only when something in the case points at sanctions**. That is not a strawman: it
is what one prompt with a step budget does &mdash; it follows the happy path the case suggests.

The graph runs every specialist every time. That is the whole of its advantage, and the whole of
its cost.
"""),
    code('''
def single_agent(ref: str) -> dict:
    """One agent, one context. Reads, consults policy, screens only if prompted to."""
    record = CASES.get(ref)
    used = ["ledger"]
    if record is None:
        return {"recommendation": "no action", "used": used}
    needs_human = False
    if record["reason_code"]:
        used.append("policy")
        needs_human = record["reason_code"] in NEEDS_HUMAN
    # it screens only when the case points that way -- nothing else prompts it to
    if record["reason_code"] == "SANCTIONS_REVIEW":
        used.append("sanctions")
        needs_human = needs_human or record["counterparty"] in SANCTIONS_WATCH
    used.append("writer")
    if record["status"] == "settled":
        action = "no action"
    elif needs_human:
        action = "hold for a human"
    else:
        action = "proceed"
    return {"recommendation": action, "used": used}


def graph_agent(ref: str) -> dict:
    """Supervisor plus four specialists. Every specialist runs, every time."""
    record = CASES.get(ref)
    used = ["ledger", "policy", "sanctions", "writer"]
    if record is None:
        return {"recommendation": "no action", "used": used, "dispatches": 4}
    blocked = record["counterparty"] in SANCTIONS_WATCH
    needs_human = record["reason_code"] in NEEDS_HUMAN
    if record["status"] == "settled":
        action = "no action"
    elif blocked or needs_human:
        action = "hold for a human"
    else:
        action = "proceed"
    return {"recommendation": action, "used": used, "dispatches": 4}


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

Quality is agreement with the ground truth. Cost is what each design woke up. The supervisor is
charged once per dispatch, because routing is not free.
"""),
    code('''
COST = {"supervisor": 120, "ledger": 380, "policy": 420, "sanctions": 90, "writer": 610}
CONTEXT_TOKENS = 600      # the case file: payment, policy text, findings so far

def run_cost(result: dict) -> int:
    """Tokens for one case.

    A single agent holds ONE context and reuses it across its own turns. A graph re-sends the
    case context to every specialist it dispatches, and pays the supervisor to route each time.
    That re-sending is where the cost multiple comes from -- not from the agents themselves.
    """
    specialists = sum(COST[a] for a in result["used"])
    dispatches = result.get("dispatches", 0)
    contexts = dispatches if dispatches else 1
    return specialists + CONTEXT_TOKENS * contexts + COST["supervisor"] * dispatches


def evaluate(design) -> dict:
    """Run one design over every case. Returns correct count, accuracy and total tokens."""
    correct, tokens, misses = 0, 0, []
    for ref, expected in eval_cases():
        result = design(ref)
        tokens += run_cost(result)
        if result["recommendation"] == expected:
            correct += 1
        else:
            misses.append((ref, expected, result["recommendation"]))
    return {"correct": correct, "accuracy": correct / len(eval_cases()),
            "tokens": tokens, "misses": misses}


def critical_path(design_name: str) -> int:
    """Waves on the critical path -- what parallelism can and cannot shorten."""
    return 2 if design_name == "single" else 3
'''),
    code('''
# --- Self-check: Section 3
def S():
    return evaluate(single_agent)
def G():
    return evaluate(graph_agent)

check("the graph gets every case right",
      lambda: G()["correct"] == len(eval_cases()))
check("the single agent does not",
      lambda: S()["correct"] < len(eval_cases()))
check("and it misses exactly the two watchlisted-but-mundane cases",
      lambda: sorted(r for r, _, _ in S()["misses"]) == ["PMT-1006", "PMT-1007"],
      "the cases where nothing in the reason code told it to look")
check("both of its misses are the dangerous direction -- proceeding when it should hold",
      lambda: all(got == "proceed" and want == "hold for a human"
                  for _, want, got in S()["misses"]))
check("the graph costs more",
      lambda: G()["tokens"] > S()["tokens"])
check("and by a multiple worth naming, not a rounding error",
      lambda: G()["tokens"] / S()["tokens"] > 1.5)
check("the graph's critical path is longer too",
      lambda: critical_path("graph") > critical_path("single"))

def _score():
    s, g = S(), G()
    print(f"  {'':16}{'single':>12}{'graph':>12}")
    print("  " + "-" * 40)
    print(f"  {'accuracy':16}{s['accuracy']:>11.0%}{g['accuracy']:>12.0%}")
    print(f"  {'tokens':16}{s['tokens']:>12}{g['tokens']:>12}")
    print(f"  {'cost multiple':16}{'1.0x':>12}{g['tokens'] / s['tokens']:>11.1f}x")
    print(f"  {'critical path':16}{critical_path('single'):>12}{critical_path('graph'):>12}")
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
            # TODO: the whole module comes down to this comparison
            "decision": BLANK}


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
                        "no action / hold for a human / proceed.\\n\\n"
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

**What you take from Module 5:** a supervisor is a router you can measure; a handoff carries only
what you put in it; parallel branches lose findings unless you declare a reducer; disagreement is
settled by authority and provenance, not by counting; the human is a node with a deadline and an
escalation ladder; and the graph earns its place with a number or it does not earn it at all.
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
3. The scorecard has no row for operating cost: five agents are five things to trace, alert on and
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
