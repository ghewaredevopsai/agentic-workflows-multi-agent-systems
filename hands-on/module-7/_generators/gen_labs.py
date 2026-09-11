#!/usr/bin/env python3
"""
Generate Module 7 lab notebooks and their solutions from one source.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-7-0N-*.ipynb and ../solutions/

Design rules (rebuilt 2026-09-09 to the framework-forward rule Day 1 established):
  * The participant writes REAL LangChain code in every lab. Module 7 is the module
    that MEASURES agents, so the thing being measured has to be an actual agent, an
    actual callback handler and an actual typed verdict -- not a dict of strings.
  * Self-checks assert on framework OBJECTS and on RECORDED runs -- a Pydantic verdict,
    a bound tool, a span tree replayed from recorded callback events. All deterministic,
    all offline. Only model INVOCATION needs the gateway, and that lives in "Run it for
    real" cells, which are observed, not scored.
    Asserting on a LIVE run's score would be self-defeating here: lab 7.1's whole subject
    is that a single run's score is a sample.
  * Blanks ask a DESIGN DECISION, not a Python idiom. If the answer is a comprehension,
    a slice or statistics.mean, the code is given and the question moves to what only
    understanding answers: which metric, which acceptance bar, which failure is
    disqualifying, which span owns the failure.
  * NO GRADED CELL TOUCHES LANGFUSE. Every Langfuse cell is a guarded "Run it for real"
    cell that prints a message and returns when unconfigured, so an outage costs the
    demo and never a score.
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
# Lab 7.{num} &mdash; {title}

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 3 &middot; Module 7 &mdash; Multi-Agent System Evaluation**

### What you'll do
{items}

> **How this lab works.** You write real LangChain code &mdash; the agent under test, the callback
> handler that traces it, the typed verdict you grade. Fill every `BLANK`, then run the
> **Self-check** cell under each section. Those check the *objects you built* and the *recorded
> runs* shipped in the notebook, so they are deterministic and do not depend on the model.
> Cells marked **Run it for real** put your code in front of the sandbox model; that is the part
> worth watching, and it is never scored &mdash; scoring a live run would contradict Lab 7.1.

{note}
""")


SETUP_COMMON = '''
# ---------------------------------------------------------------- Setup: run me first
import os, json, time, textwrap, random, statistics
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
    except NameError as exc:
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
# tokens. Off is the default here because an eval lab makes a lot of calls.
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
# ------------------------------------------------- carried forward from Module 1, Lab 1.2
# Real LangChain tools -- @tool turns a function into a tool object with a name, a schema
# and a description the model reads. Nothing to fill in; they are here so this notebook
# stands on its own and so the agent you measure is a real agent.

from langchain_core.tools import tool

@tool
def lookup_payment(ref: str) -> str:
    """Return the ledger record for one payment reference such as 'PMT-1005'.

    Use when you need the status, amount, counterparty or reason code of a specific
    payment. Not for searching across payments.
    """
    record = LEDGER.get(ref)
    if record is None:
        return f"no payment found with reference {ref!r}"
    return json.dumps({"ref": ref, **record})


@tool
def policy_for(reason_code: str) -> str:
    """Return the operating policy for one failure reason code, e.g. 'LIMIT_BREACH'.

    Use after you know why a payment failed and need to know what to do about it.
    """
    return POLICY.get(reason_code, f"no policy on file for reason code {reason_code!r}")


EVAL_TOOLS = [lookup_payment, policy_for]
print("tools:", ", ".join(t.name for t in EVAL_TOOLS))
'''


APP_ENV_EXTRA = """
# ---- your own deployment, for Section 3 only -------------------------------
# Module 7's setup does not carry these, so read them here. Section 3 self-skips
# when they are absent, which is what an offline verification run sees.
import re as _re

APP_NS   = os.environ.get("APP_NAMESPACE", "")
APP_HOST = os.environ.get("APP_HOST", "")
print("namespace :", APP_NS or "(unset -- section 3 will self-skip)")
"""


# =========================================================================== #
# Lab 7.1 -- locating the failing step
# =========================================================================== #

INFO_EVIDENCE = """
## Concept &mdash; three sources, and what each one cannot tell you

You have a bad answer. Somewhere in eight steps, something went wrong. Guessing is expensive,
so work from evidence, and know what each source is blind to.

| Source | Answers | Blind to |
|---|---|---|
| **the audit trail** (printed under every reply) | *which* steps ran, the category, the confidence, the tool calls, what QA did | how long anything took, and what the tool actually returned |
| **Prometheus / Grafana** (`frontdeskai_*`) | which step is slow or expensive, across many runs | this one request. A metric is a population, not a case |
| **the Tempo trace** | the real tree: what nested inside what, with timings and token counts | whether the *content* was right |

None of them says &ldquo;the answer was wrong&rdquo;. That judgement is yours; their job is to tell
you **where** it went wrong so you fix that step and not a different one.

### The six runs in this lab are real

They were captured from FrontDesk AI deployed on this cluster, by asking it six questions. The
audit trails below are what it actually printed. Four of the six are wrong in some way, each at a
different step, and &mdash; the thing worth sitting with &mdash; **all six passed the QA gate**.
"""


INFO_BOUNDARY = """
## Section 1 &mdash; Find this run, then walk the ladder

### First: the audit trail is not one run

The trail the app returns grows with the conversation, because LangGraph's checkpointer keys on
the user and the `audit` field is an appending list. Across the six captures it went from 102
entries to 144. So before you can read a run you have to find where it starts &mdash; at the last
`Supervisor:` line, which is the first thing every request does.

That is not a detail. Read the whole array and you will diagnose a failure that happened twenty
minutes ago, in a different request, for a different question.

### Then: check the cheap things first

Eight steps, in the order the request goes through them. The point of a fixed order is that each
rung is cheaper to check than the one below, and a failure at any rung makes everything under it
unreliable &mdash; so the **first** rung that fails is the one to fix.

| Rung | What you are asking | Where you see it |
|---|---|---|
| `route` | did the supervisor pick the right department? | `Supervisor: <category> (conf: N)` |
| `clarify` | was it confident enough to answer at all? | confidence, and whether it asked a question instead |
| `retrieve` | did RAG return anything, and is it relevant? | `RAG: retrieved N chunks`, plus the source list |
| `tool_pick` | were the right tools chosen &mdash; any at all? | `<Worker> ReAct: ... tools: ...` |
| `tool_run` | did those tools actually succeed? | an error string in the trail |
| `converge` | did the worker finish, or run out of turns? | `N iteration(s)` against a limit of 3 |
| `write` | did it change something, and was it asked to? | a write tool in the tool list |
| `qa` | did the gate catch anything? | `QA PASS` / a fallback |
"""
BUNDLE = '''
# ------------------------- six real runs, captured from the deployed app
# Nothing here is synthetic. These are the audit trails FrontDesk AI printed
# when it was asked these six questions on this cluster, with the timestamps
# removed. Each trail carries TWO turns, because that is how the app returns
# it -- the real arrays ran from 102 to 144 entries; they are cut to the last
# two turns here so the notebook stays readable.
RUNS = [
  {
    "id": 'run-1', "question": 'What is my current leave balance?',
    "category": 'hr', "confidence": 10,
    "escalated": False, "fallback_used": False,
    # the trail as returned: this turn PRECEDED by the one before it.
    "audit": [
      'Supervisor: tech (conf: 9)',
      'RAG: retrieved 4 chunks from 4 sources',
      'Few-shot: 0 example(s) retrieved',
      'Tech ReAct: 2 iteration(s), tools: list_my_tickets({})',
      'Tech worker: resolved',
      'Escalation check: False',
      'QA PASS',
      'Finalized for Rajesh Kumar',
      'Supervisor: hr (conf: 10)',
      'RAG: retrieved 4 chunks from 4 sources',
      'Few-shot: 0 example(s) retrieved',
      'Hr ReAct: 3 iteration(s), tools: get_leave_balance_from_hr_system({}), get_leave_balance({})',
      'Hr worker: resolved',
      'Escalation check: False',
      'QA PASS',
      'Finalized for Rajesh Kumar',
    ],
    "sources": ['Hr Handbook > Casual Leave (CL)', 'Hr Handbook > Earned/Privilege Leave (EL)', 'Hr Handbook > Sick Leave (SL)', 'Hr Handbook > Leave Policy'],
    "answer": '[HR] Your current leave balance is: 17 Casual Leave days, 8 Sick Leave days, 12 Earned Leave days, and 24 WFH days remaining. Sources: Hr Handbook > Casual Leave (CL), Hr Handbook > Earned/Privilege Leave (EL), Hr Handbook > Sick ',
  },
  {
    "id": 'run-2', "question": 'How do I claim for a laptop damaged on work travel?',
    "category": 'finance', "confidence": 8,
    "escalated": False, "fallback_used": False,
    # the trail as returned: this turn PRECEDED by the one before it.
    "audit": [
      'Supervisor: hr (conf: 10)',
      'RAG: retrieved 4 chunks from 4 sources',
      'Few-shot: 0 example(s) retrieved',
      'Hr ReAct: 3 iteration(s), tools: get_leave_balance_from_hr_system({}), get_leave_balance({})',
      'Hr worker: resolved',
      'Escalation check: False',
      'QA PASS',
      'Finalized for Rajesh Kumar',
      'Supervisor: finance (conf: 8)',
      'RAG: retrieved 4 chunks from 4 sources',
      'Few-shot: 0 example(s) retrieved',
      'Finance worker: resolved',
      'Escalation check: False',
      'QA PASS',
      'Finalized for Rajesh Kumar',
    ],
    "sources": ['Finance Policies > How to Claim', 'Finance Policies > Travel Policy', 'Finance Policies > Vendor Payments', 'Finance Policies > Eligible Expenses'],
    "answer": "[FINANCE] To claim for a laptop damaged during work travel, you should submit an expense claim under the 'hardware' category. Please ensure you attach the repair invoice or replacement receipt and submit the claim within 30 days o",
  },
  {
    "id": 'run-3', "question": "it's broken",
    "category": 'finance', "confidence": 8,
    "escalated": False, "fallback_used": False,
    # the trail as returned: this turn PRECEDED by the one before it.
    "audit": [
      'Supervisor: finance (conf: 8)',
      'RAG: retrieved 4 chunks from 4 sources',
      'Few-shot: 0 example(s) retrieved',
      'Finance worker: resolved',
      'Escalation check: False',
      'QA PASS',
      'Finalized for Rajesh Kumar',
      'Supervisor: finance (conf: 8)',
      'RAG: retrieved 4 chunks from 4 sources',
      'Few-shot: 0 example(s) retrieved',
      "Finance ReAct: 3 iteration(s), tools: submit_expense_claim({'amount': 0, 'category': 'hardware', 'description': 'Laptop damaged during work travel', 'receipt_count': 1}), submit_expense_claim({'amount': 1, 'category': 'hardware', 'description': 'Laptop damaged during work travel', 'receipt_count': 1})",
      'Finance worker: resolved',
      'Escalation check: False',
      'QA PASS',
      'Finalized for Rajesh Kumar',
    ],
    "sources": ['Finance Policies > Salary & Payroll', 'Finance Policies > Invoicing & Payments', 'Finance Policies > Expense Reimbursement', 'Finance Policies'],
    "answer": "[FINANCE] Your expense claim for the damaged laptop has been submitted successfully under the 'hardware' category. The Claim ID is EXP-2026-0007. Please remember to attach the repair invoice or replacement receipt and ensure the c",
  },
  {
    "id": 'run-4', "question": 'My salary slip shows the wrong VPN access for my desk booking',
    "category": 'tech', "confidence": 7,
    "escalated": False, "fallback_used": False,
    # the trail as returned: this turn PRECEDED by the one before it.
    "audit": [
      'Supervisor: finance (conf: 8)',
      'RAG: retrieved 4 chunks from 4 sources',
      'Few-shot: 0 example(s) retrieved',
      "Finance ReAct: 3 iteration(s), tools: submit_expense_claim({'amount': 0, 'category': 'hardware', 'description': 'Laptop damaged during work travel', 'receipt_count': 1}), submit_expense_claim({'amount': 1, 'category': 'hardware', 'description': 'Laptop damaged during work travel', 'receipt_count': 1})",
      'Finance worker: resolved',
      'Escalation check: False',
      'QA PASS',
      'Finalized for Rajesh Kumar',
      'Supervisor: tech (conf: 7)',
      'RAG: retrieved 4 chunks from 4 sources',
      'Few-shot: 0 example(s) retrieved',
      "Tech ReAct: 2 iteration(s), tools: create_ticket({'summary': 'Incorrect VPN access details on salary slip for desk booking', 'priority': 'P3', 'category': 'general', 'description': 'Employee reports that their salary slip displays incorrect VPN access information related to their desk booking.'})",
      'Tech worker: resolved',
      'Escalation check: False',
      'QA PASS',
      'Finalized for Rajesh Kumar',
    ],
    "sources": ['It Support > VPN Access', 'It Support > Troubleshooting', 'It Support > Setup', 'It Support > Security Policies'],
    "answer": '[TECH] I have created a ticket (TECH-1011) for this issue. Please note that salary slips are typically generated by the Finance/HR system, so this may require coordination between IT and HR to correct the underlying data. Sources:',
  },
  {
    "id": 'run-5', "question": 'I need 45 days of leave starting next Monday, approve it',
    "category": 'hr', "confidence": 9,
    "escalated": True, "fallback_used": False,
    # the trail as returned: this turn PRECEDED by the one before it.
    "audit": [
      'Supervisor: tech (conf: 7)',
      'RAG: retrieved 4 chunks from 4 sources',
      'Few-shot: 0 example(s) retrieved',
      "Tech ReAct: 2 iteration(s), tools: create_ticket({'summary': 'Incorrect VPN access details on salary slip for desk booking', 'priority': 'P3', 'category': 'general', 'description': 'Employee reports that their salary slip displays incorrect VPN access information related to their desk booking.'})",
      'Tech worker: resolved',
      'Escalation check: False',
      'QA PASS',
      'Finalized for Rajesh Kumar',
      'Supervisor: hr (conf: 9)',
      'RAG: retrieved 4 chunks from 4 sources',
      'Few-shot: 0 example(s) retrieved',
      "Hr ReAct: 3 iteration(s), tools: get_leave_balance_from_hr_system({}), get_leave_balance({}), apply_leave({'leave_type': 'casual', 'start_date': '2026-09-14', 'end_date': '2026-11-27', 'reason': 'Personal leave request for 45 days'})",
      'Hr worker: escalating',
      'Escalation check: True',
      'Manager called tool → Could not reach HR MCP server at http://mcp-leave.postgres.svc.cluster.local:8001/mcp: [Errno -2] Name or service not kn',
      'Manager called tool → Could not reach HR MCP server at http://mcp-leave.postgres.svc.cluster.local:8001/mcp: [Errno -2] Name or service not kn',
      'Manager resolved after 3 step(s)',
      'QA PASS',
      'Finalized for Rajesh Kumar',
    ],
    "sources": ['Hr Handbook > Casual Leave (CL)', 'Hr Handbook > Earned/Privilege Leave (EL)', 'Hr Handbook > Paternity Leave', 'Hr Handbook > Sick Leave (SL)'],
    "answer": "[HR] [Manager Approval] I understand you're requesting 45 days of leave starting next Monday (2026-09-14). However, I must decline this request for the following reasons:  1. **Exceeds maximum consecutive leave limit**: Per compan",
  },
  {
    "id": 'run-6', "question": 'What is the status of my open IT tickets?',
    "category": 'tech', "confidence": 9,
    "escalated": False, "fallback_used": False,
    # the trail as returned: this turn PRECEDED by the one before it.
    "audit": [
      'Supervisor: hr (conf: 9)',
      'RAG: retrieved 4 chunks from 4 sources',
      'Few-shot: 0 example(s) retrieved',
      "Hr ReAct: 3 iteration(s), tools: get_leave_balance_from_hr_system({}), get_leave_balance({}), apply_leave({'leave_type': 'casual', 'start_date': '2026-09-14', 'end_date': '2026-11-27', 'reason': 'Personal leave request for 45 days'})",
      'Hr worker: escalating',
      'Escalation check: True',
      'Manager called tool → Could not reach HR MCP server at http://mcp-leave.postgres.svc.cluster.local:8001/mcp: [Errno -2] Name or service not kn',
      'Manager called tool → Could not reach HR MCP server at http://mcp-leave.postgres.svc.cluster.local:8001/mcp: [Errno -2] Name or service not kn',
      'Manager resolved after 3 step(s)',
      'QA PASS',
      'Finalized for Rajesh Kumar',
      'Supervisor: tech (conf: 9)',
      'RAG: retrieved 4 chunks from 4 sources',
      'Few-shot: 0 example(s) retrieved',
      'Tech ReAct: 2 iteration(s), tools: list_my_tickets({})',
      'Tech worker: resolved',
      'Escalation check: False',
      'QA PASS',
      'Finalized for Rajesh Kumar',
    ],
    "sources": ['It Support > How to Raise a Ticket', 'It Support > Requesting New Software', 'It Support > Contact', 'It Support'],
    "answer": '[TECH] You have one open IT ticket: TECH-1004, which is for a slow-loading Jira dashboard. It is currently marked as Resolved. Sources: It Support > How to Raise a Ticket, It Support > Requesting New Software, It Support > Contact',
  },
]

print(len(RUNS), "runs loaded;", [len(r['audit']) for r in RUNS], "audit entries each")
'''
BOUNDARY = '''
# ---------------------------------------------------- reading ONE run, given
# The trail accumulates across a conversation, so a run starts at the last
# "Supervisor:" line. Everything before that belongs to an earlier request.
def this_run(run: dict) -> list:
    a = run["audit"]
    start = max((i for i, l in enumerate(a) if l.startswith("Supervisor:")), default=0)
    return a[start:]

def tools_called(run: dict) -> list:
    """Every tool name in this run's ReAct lines, in order."""
    names = []
    for line in this_run(run):
        if "tools:" in line:
            for part in line.split("tools:", 1)[1].split("),"):
                name = part.split("(")[0].strip()
                if name:
                    names.append(name)
    return names

def show(run: dict) -> None:
    print(f"{run['id']}  {run['question']!r}")
    print(f"  category={run['category']} confidence={run['confidence']} "
          f"escalated={run['escalated']} fallback={run['fallback_used']}")
    for line in this_run(run):
        print("   ", line)
    print("  answer:", run["answer"][:150])

show(RUNS[0])       # the healthy one -- your baseline
'''

LADDER = '''
# -------------------------------------------------- the eight rungs, given
# One predicate per rung. Each returns True when that step looks HEALTHY.
# These read only the audit trail -- no model, no cluster, no network.
WRITE_TOOLS = {"apply_leave", "approve_leave_via_mcp", "submit_expense_claim",
               "approve_expense_claim", "create_ticket", "book_meeting_room",
               "send_email", "install_skill", "write_local_file"}

RUNGS = ["route", "clarify", "retrieve", "tool_pick", "tool_run", "converge", "write", "qa"]

def rung_route(run):
    if run["id"] in UNANSWERABLE:        # no department is right; see rung_clarify
        return True
    return run["category"] in EXPECTED_CATEGORY.get(run["id"], {run["category"]})

def rung_clarify(run):
    """Is the confidence plausible GIVEN the question?

    The app clarifies below 5. That gate cannot help when the score itself is
    wrong -- so judge the score against how much there was to go on.
    """
    words = [w for w in _re.findall(r"[a-zA-Z]+", run["question"]) if len(w) > 2]
    if len(words) <= 2:                  # almost nothing to classify on
        return run["confidence"] < 5     # it should have asked, not answered
    return run["confidence"] >= 5
def rung_retrieve(run):  return any("RAG: retrieved" in l and "0 chunks" not in l
                                    for l in this_run(run))
def rung_tool_pick(run): return bool(tools_called(run)) or run["id"] in NO_TOOL_NEEDED
def rung_tool_run(run):  return not any("Could not reach" in l or "error" in l.lower()
                                        for l in this_run(run))
def rung_converge(run):  return not any("3 iteration(s)" in l for l in this_run(run))
def rung_write(run):     return not (set(tools_called(run)) & WRITE_TOOLS) or run["id"] in WRITE_ASKED_FOR
def rung_qa(run):        return not run["fallback_used"]

PREDICATE = {"route": rung_route, "clarify": rung_clarify, "retrieve": rung_retrieve,
             "tool_pick": rung_tool_pick, "tool_run": rung_tool_run,
             "converge": rung_converge, "write": rung_write, "qa": rung_qa}

# What a correct system would have done. Hand-labelled, which is what a ground
# truth is -- somebody decided, and you can disagree with them.
EXPECTED_CATEGORY = {"run-1": {"hr"}, "run-2": {"finance"}, "run-3": {"hr", "tech", "facilities"},
                     "run-4": {"finance", "hr"}, "run-5": {"hr"}, "run-6": {"tech"}}
NO_TOOL_NEEDED    = {"run-2"}                 # a policy question RAG can answer
UNANSWERABLE      = {"run-3"}                 # "it's broken" -- no department is the right one
WRITE_ASKED_FOR   = {"run-5"}                 # "approve it" does ask for a write

def first_failing_rung(run: dict) -> str | None:
    """The first rung, in order, whose step does not look healthy."""
    for name in RUNG_ORDER():
        if not PREDICATE[name](run):
            return name
    return None
'''

DECISIONS_LAB = '''
# ------------------------------------------------------------ your decisions
def RUNG_ORDER() -> list:
    """The order to check the rungs in.

    `RUNGS` above is already in the order a request passes through the steps.
    Checking in that order means the first failure you find is the earliest one,
    and everything below it is downstream of a broken input.
    """
    return BLANK                      # RUNGS  |  sorted(RUNGS)  |  RUNGS[::-1]


def stops_the_two_word_message() -> str:
    """run-3 is the message "it's broken". It ended up submitting an expense claim.

    Two rungs could each have stopped that on their own. Which one is EARLIER,
    and so the one to fix first? Return a rung name from RUNGS.
    """
    return BLANK                      # "clarify" | "write" | "qa"


def qa_pass_means_correct() -> bool:
    """All six runs printed QA PASS, and four of them are wrong.

    So: does QA PASS tell you the run was correct?
    """
    return BLANK                      # True | False


def worst_of_the_six() -> str:
    """One run did something that cannot be undone by re-asking the question.

    Escalating a bad answer costs a conversation. Escalating THIS costs a
    correction in a system of record. Return its id.
    """
    return BLANK                      # "run-3" | "run-4" | "run-5" | "run-6"
'''

DECISIONS_SOL = (DECISIONS_LAB
    .replace('return BLANK                      # RUNGS  |', 'return RUNGS                      # RUNGS  |')
    .replace('return BLANK                      # "clarify"', 'return "clarify"                  # "clarify"')
    .replace('return BLANK                      # True | False', 'return False                      # True | False')
    .replace('return BLANK                      # "run-3"', 'return "run-3"                    # "run-3"'))

CHECKS = '''
# --- Self-check: the ladder over six real runs  (audit trails only -- no model)
check("the ladder is walked in request order",
      lambda: RUNG_ORDER() == RUNGS,
      "sorted() is alphabetical, and reversed() finds the LAST failure, not the first")

check("run-2 is the clean one, all eight rungs",
      lambda: first_failing_rung(RUNS[1]) is None,
      "a policy question answered from retrieval, no tools, nothing written")

check("run-1 -- the one that looks fine -- fails at converge",
      lambda: first_failing_rung(RUNS[0]) == "converge",
      "3 of 3 ReAct iterations for a balance lookup, and it called two leave tools")

check("run-3 -- 'it's broken' -- fails at clarify, not later",
      lambda: first_failing_rung(RUNS[2]) == "clarify",
      "confidence 8 on a two-word message. Everything after that is downstream")

check("...and the earlier of its two possible stops is the one to fix",
      lambda: stops_the_two_word_message() == "clarify")

check("run-5 -- the 45-day request -- fails at tool_run",
      lambda: first_failing_rung(RUNS[4]) == "tool_run",
      "'Could not reach HR MCP server', twice, and the manager still answered")

check("run-3 made a write nobody asked for",
      lambda: bool(set(tools_called(RUNS[2])) & WRITE_TOOLS)
              and RUNS[2]["id"] not in WRITE_ASKED_FOR,
      "submit_expense_claim, off a two-word message, with an invented description")

check("run-6 passes every rung and is still wrong",
      lambda: first_failing_rung(RUNS[5]) is None
              and "open" in RUNS[5]["answer"].lower()
              and "resolved" in RUNS[5]["answer"].lower(),
      "one OPEN ticket that is currently RESOLVED. No rung here reads the answer")

check("QA PASS is not a statement about correctness",
      lambda: qa_pass_means_correct() is False
              and all("QA PASS" in " ".join(this_run(r)) for r in RUNS))

check("the run to escalate is the one that wrote to a system of record",
      lambda: worst_of_the_six() == "run-3",
      "a wrong answer can be re-asked; EXP-2026-0007 has to be withdrawn")

check("every run is diagnosed from its own turn, not the whole conversation",
      lambda: all(len(this_run(r)) < len(r["audit"]) for r in RUNS)
              and all(this_run(r)[0].startswith("Supervisor:") for r in RUNS),
      "each trail here carries the previous turn too -- cut at the LAST Supervisor line")

score()
'''
INFO_COST = """
## Section 2 &mdash; What the failure cost, and which one to fix first

Two runs can fail at the same rung and be worth very different amounts of your attention.

A failure at `route` throws away the whole request &mdash; every call after it was spent on the
wrong department. A failure at `qa` throws away almost nothing, because everything before it was
useful work. So *where* a run failed tells you how much it wasted, and you can read that straight
off the trail without any extra instrumentation.

Then one thing outranks all of that: whether the run **changed something**. A wrong answer costs
a conversation. A wrong write costs a correction in a system of record, and somebody else's
afternoon.
"""


COST = '''
# ------------------------------------------------- cost and priority, given
def steps_before(run: dict, rung: str) -> int:
    """How many rungs the request got through before the one that failed."""
    return RUNGS.index(rung) if rung in RUNGS else len(RUNGS)

def wasted(run: dict) -> int:
    """Model calls spent before the failure. One per ReAct iteration, plus the
    supervisor, plus the worker's final answer."""
    iters = sum(int(l.split(" iteration(s)")[0].split()[-1])
                for l in this_run(run) if "iteration(s)" in l)
    return 1 + iters + 1

def report():
    print(f"{'run':7} {'failed at':11} {'got through':11} {'calls':5}  wrote?")
    for r in RUNS:
        rung = first_failing_rung(r)
        wrote = bool(set(tools_called(r)) & WRITE_TOOLS)
        print(f"{r['id']:7} {str(rung or 'healthy'):11} "
              f"{steps_before(r, rung or ''):>11} {wasted(r):>5}  "
              f"{'YES' if wrote else '-'}")

guard(report)
'''

INFO_LIVE = """
## Section 3 &mdash; Break your own deployment, then locate it

Reading somebody else's failures is practice. Causing one and finding it is the skill.

These cells are marked **Run it for real** and need the app deployed in your namespace
(Module 9's lab, or `bash scripts/deploy-spark.sh` from your clone). They self-skip with
instructions if it is not there, so the rest of the lab still scores.

You will break **one** step, ask a question that depends on it, and walk the ladder on the
result. The failure you are about to cause is the same one as `run-5`: point the app at an HR
system that does not answer.
"""


LIVE = '''
# --- Run it for real: break one step, then find it ---------------------------
import subprocess

def kubectl(*args):
    return subprocess.run(("kubectl", "-n", APP_NS or "none") + args,
                          capture_output=True, text=True)

def deployed() -> bool:
    if not APP_NS:
        print("APP_NAMESPACE is unset, so there is no deployment to break.")
        print("Deploy it first (Module 9), then re-run this cell.")
        return False
    got = kubectl("get", "deploy", "frontdeskai", "-o", "name")
    if got.returncode != 0:
        print("frontdeskai is not deployed in", APP_NS, "-- deploy it first.")
        return False
    return True

def break_the_hr_tool():
    """Point MCP_LEAVE_URL at a host that does not resolve, and roll it out."""
    if not deployed():
        return
    out = kubectl("patch", "configmap", "frontdeskai-config", "--type", "merge",
                  "-p", json.dumps({"data": {"MCP_LEAVE_URL":
                                    "http://no-such-hr-system.invalid:8001/mcp"}}))
    print(out.stdout.strip() or out.stderr.strip())
    kubectl("rollout", "restart", "deploy/frontdeskai")
    print(kubectl("rollout", "status", "deploy/frontdeskai", "--timeout=300s").stdout.strip())
    print("\\nNow ask it 'what is my leave balance?' -- in the browser, or the cell below.")

guard(break_the_hr_tool)
'''

LIVE_DIAGNOSE = '''
# --- Run it for real: ask, capture the trail, and walk your own ladder -------
ASK_PY = (
    "import json,urllib.request,urllib.parse,http.cookiejar;"
    "j=http.cookiejar.CookieJar();"
    "o=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(j));"
    "p=lambda u,d: o.open('http://127.0.0.1:8000'+u,"
    " data=urllib.parse.urlencode(d).encode(), timeout=600);"
    "p('/login', {'email':'rajesh.kumar@unigps.in','password':'brainupgrade'});"
    "r=json.loads(p('/chat/send', {'message':'What is my current leave balance?'}).read());"
    "print(json.dumps({'category':r['category'],'confidence':r['confidence'],"
    "'escalated':r['escalated'],'fallback_used':r['fallback_used'],"
    "'audit':r['audit'],'sources':r.get('sources',[]),'answer':r['response'][:230]}))"
)

def diagnose_live():
    if not deployed():
        return
    out = kubectl("exec", "deploy/frontdeskai", "--", "python3", "-c", ASK_PY)
    if out.returncode != 0:
        print("the request did not complete:", (out.stderr or out.stdout).strip()[:300]); return
    mine = json.loads(out.stdout.strip().splitlines()[-1])
    mine["id"] = "mine"
    EXPECTED_CATEGORY["mine"] = {"hr"}
    show(mine)
    print()
    print("first failing rung ->", first_failing_rung(mine))
    print()
    print("Read the answer again. Does it admit that anything failed?")

guard(diagnose_live)
'''

LIVE_FIX = '''
# --- Run it for real: fix exactly that step, and prove the rung clears -------
def put_it_back():
    if not deployed():
        return
    out = kubectl("patch", "configmap", "frontdeskai-config", "--type", "merge",
                  "-p", json.dumps({"data": {"MCP_LEAVE_URL":
                                    "http://mcp-leave.postgres.svc.cluster.local:8001/mcp"}}))
    print(out.stdout.strip() or out.stderr.strip())
    kubectl("rollout", "restart", "deploy/frontdeskai")
    print(kubectl("rollout", "status", "deploy/frontdeskai", "--timeout=300s").stdout.strip())
    print("\\nRe-run the cell above. tool_run should clear -- and note what did NOT change:")
    print("the category, the retrieval, the QA verdict. You fixed one rung, not the system.")

guard(put_it_back)
'''


LAB1 = [
    header(1, "Locating the Failing Step", "Advanced", 50,
           ["Read one run out of an audit trail that spans a whole conversation",
            "Walk a fixed ladder of eight checks, cheapest first, over six real runs",
            "Tell apart two runs that look equally wrong and failed at different steps",
            "Work out what each failure cost, and which one to escalate",
            "Break one step of your own deployment, locate it, and fix just that step"],
           "> **Six real runs.** They were captured from this app deployed on this cluster.\n"
           "> Four of them are wrong, each at a different step, and every one of the six\n"
           "> printed `QA PASS`. Your job is not to decide *whether* they are wrong &mdash;\n"
           "> that is given &mdash; but to say *where*, from the evidence."),
    setup(1, APP_ENV_EXTRA),

    md(INFO_EVIDENCE),
    code(BUNDLE),

    md(INFO_BOUNDARY),
    code(BOUNDARY),
    code(LADDER),
    code(DECISIONS_LAB, DECISIONS_SOL),
    code(CHECKS),

    md(INFO_COST),
    code(COST),

    md(INFO_LIVE),
    code(LIVE),
    code(LIVE_DIAGNOSE),
    code(LIVE_FIX),

    code('''
score()
'''),
    md("""
## Your turn

1. **Disagree with the ground truth.** `EXPECTED_CATEGORY` says run-4 &mdash; *&ldquo;my salary
   slip shows the wrong VPN access for my desk booking&rdquo;* &mdash; should have gone to finance
   or hr, and the app chose tech with confidence 7. Make the case for tech. If you win the
   argument, the label is wrong, not the app, and you have just found the most common defect in
   an eval set.
2. **Add the rung this ladder does not have.** Nothing here checks whether the retrieved sources
   are *relevant* to the question, only that some arrived. Run-3 retrieved four finance chunks
   for a two-word message and `retrieve` passed. Write that rung, and say honestly what it costs
   to evaluate.
3. **Break a different step.** Upload a policy document that contradicts the handbook, then ask a
   question it covers. Every tool succeeds, every rung passes, and the answer is wrong &mdash;
   which rung would have to exist to catch it, and can it be checked without a model?
4. **Take run-3 apart properly.** It invented *&ldquo;Laptop damaged during work travel&rdquo;*
   from the previous turn in the conversation, then submitted it twice with `amount: 0` and
   `amount: 1`. Three separate defects are visible in that one trail. Name them, and say which
   rung each belongs to.

> **What you take from Module 7:** a failure has a location, and finding it is cheaper than
> arguing about it. The audit trail is not one run until you cut it. The first rung that fails is
> the one to fix, because everything under it was working from a broken input. And `QA PASS` is a
> statement about a gate having run, not about the answer being right.
"""),
]



LAB2 = [
    header(2, "Prompt Versioning with Langfuse", "Intermediate", 25,
           ["Put a prompt in Langfuse and change it without touching your code",
            "Move a <code>production</code> label between versions &mdash; and roll it back",
            "Link a model call to the exact prompt version that produced it",
            "Keep working when Langfuse is unreachable"],
           "> **This one is a walkthrough.** There are no blanks and no score &mdash; every cell is a\n"
           "> **Run it for real** cell against the live Langfuse project. Read the step, run the cell,\n"
           "> look at the output. If Langfuse is not configured each cell says so and moves on."),
    setup(2),

    md("""
## Why bother

A prompt written in your source code is a **deploy**: to change a word you edit a file, review it,
build an image and roll it out. A prompt stored in Langfuse is a **config change** &mdash; you edit
it in a browser and the next call picks it up.

What you get for that is the part people forget: **every version is kept**, each one carries a
**label** like `production` or `staging`, and a label can be moved back to an older version in one
call. That is what makes a bad prompt a thirty-second rollback instead of a redeploy.

Three ideas, and that is the whole model:

| | |
|---|---|
| **version** | an integer. Every save makes a new one. Nothing is overwritten |
| **label** | a movable pointer — `production`, `staging`, anything you like. `latest` is maintained for you |
| **config** | settings that travel *with* the prompt — model, temperature — so they cannot drift apart |
"""),

    md("""
## Step 1 &mdash; Connect

`Langfuse()` reads `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` from your
environment; your sandbox already has all three. `auth_check()` is the one-line way to be sure.
"""),
    code('''
# The import lives inside the function on purpose: a module-level import of a package
# that is not installed crashes the whole cell, and guard() only catches NameError.
def connect():
    try:
        from langfuse import Langfuse
    except ImportError:
        print("langfuse is not installed in this kernel -- the rest of this lab will self-skip.")
        return None
    try:
        lf = Langfuse()
        if not lf.auth_check():
            print("Langfuse keys are not valid here. Skipping the rest of this lab.")
            return None
    except Exception as exc:
        print(f"Langfuse not configured ({type(exc).__name__}). Skipping the rest of this lab.")
        print("Your sandbox normally injects LANGFUSE_HOST / _PUBLIC_KEY / _SECRET_KEY.")
        return None
    print("connected ->", os.environ.get("LANGFUSE_HOST"))
    return lf

LF = guard(connect)
'''),

    md("""
## Step 2 &mdash; Name it after yourself

⚠️ **All 31 of you share one Langfuse project.** A prompt name is global to that project, so if
two people both use `support-classifier` they are writing versions of *the same prompt* and will
overwrite each other's labels.

So put your sandbox name in it. And note there is **no delete** in the SDK &mdash; a prompt you
create today is there tomorrow, which is another reason not to squat on a plain name.
"""),
    code('''
WHO  = os.environ.get("LANGFUSE_TRACING_ENVIRONMENT", "unknown")
NAME = f"support-classifier-{WHO}"
print("your prompt name:", NAME)
'''),

    md("""
## Step 3 &mdash; Create version 1

`prompt` is the text. `{{ticket}}` is a placeholder you fill in later. `labels` is where this
version points, and `config` rides along with it.
"""),
    code('''
def create_v1():
    if not LF:
        return
    v = LF.create_prompt(
        name=NAME,
        prompt="Classify this support ticket in one word.\\nTicket: {{ticket}}",
        labels=["production"],
        config={"model": LLM_MODEL, "temperature": 0},
        commit_message="first cut",
    )
    print("version", v.version, "labels", v.labels)
    return v.version

V1 = guard(create_v1)
'''),

    md("""
## Step 4 &mdash; Change it

Saving the same name again does **not** overwrite version 1. It creates version 2. This one goes
to `staging`, so `production` still points at version 1 &mdash; which is the point.

*(Re-run this notebook and you get versions 3 and 4, then 5 and 6. That is what versioning means,
and there is no delete in the SDK. The steps below use the versions **this run** created rather
than the literals 1 and 2, which is also how you should write it in a service.)*
"""),
    code('''
def create_v2():
    if not LF:
        return
    v = LF.create_prompt(
        name=NAME,
        prompt=("Classify this support ticket in one word.\\n"
                "Use exactly one of: billing, technical, account.\\nTicket: {{ticket}}"),
        labels=["staging"],
        config={"model": LLM_MODEL, "temperature": 0},
        commit_message="constrain the answer to three categories",
    )
    print("version", v.version, "labels", v.labels)
    return v.version

V2 = guard(create_v2)
'''),

    md("""
## Step 5 &mdash; Fetch one

Ask for a **label** in application code &mdash; that is the whole point of the indirection. Ask for
a **version** when you are pinning something down, like reproducing a run.

`clear_prompt_cache()` is here only because you just changed things a second ago; the SDK caches
prompts, which is what makes this cheap in production.
"""),
    code('''
def fetch_both():
    if not LF or not V2:
        return
    LF.clear_prompt_cache()
    print("label=production ->  v", LF.get_prompt(NAME, label="production").version)
    print(f"version={V2}        ->   ", LF.get_prompt(NAME, version=V2).labels)

guard(fetch_both)
'''),

    md("""
## Step 6 &mdash; Fill in the variables

`compile()` substitutes the `{{ticket}}` placeholder. Nothing clever &mdash; but note the text your
code sends is now something you can change in a browser.
"""),
    code('''
def compile_it():
    if not LF:
        return
    p = LF.get_prompt(NAME, label="production")
    print(p.compile(ticket="I was charged twice this month"))
    print("config travels with it:", p.config)

guard(compile_it)
'''),

    md("""
## Step 7 &mdash; Run it, linked to the version

One extra argument, `langfuse_prompt=p`, and the trace records **which prompt version produced this
output**. Without it you can see that a call was slow or wrong; with it you can see *which wording*
was slow or wrong, which is the question you actually have.
"""),
    code('''
def run_linked():
    if not LF:
        return
    if not llm_ready():
        return
    from langfuse import get_client
    from langfuse.openai import openai
    p = LF.get_prompt(NAME, label="production")
    client = openai.OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    r = client.chat.completions.create(
        model=LLM_MODEL, max_tokens=10, langfuse_prompt=p,
        messages=[{"role": "user", "content": p.compile(ticket="I was charged twice")}],
    )
    print("answer:", r.choices[0].message.content.strip(), "| from prompt v", p.version)
    get_client().flush()

guard(run_linked)
'''),

    md("""
## Step 8 &mdash; Promote version 2

This is the deploy. No image, no restart &mdash; you move a label.
"""),
    code('''
def promote():
    if not LF or not V2:
        return
    LF.update_prompt(name=NAME, version=V2, new_labels=["production"])
    LF.clear_prompt_cache()
    print("production is now v", LF.get_prompt(NAME, label="production").version)

guard(promote)
'''),

    md("""
## Step 9 &mdash; Roll it back

Same call, older version. **This is the step that justifies the whole exercise** &mdash; version 2
turned out worse, and undoing it costs one line and no redeploy.
"""),
    code('''
def rollback():
    if not LF or not V1:
        return
    LF.update_prompt(name=NAME, version=V1, new_labels=["production"])
    LF.clear_prompt_cache()
    print("production is back to v", LF.get_prompt(NAME, label="production").version)

guard(rollback)
'''),

    md("""
## Step 10 &mdash; When Langfuse is down

You have just made your prompt a network call. `fallback=` is how that does not become an outage:
if Langfuse cannot be reached, you get the text you passed instead, and `is_fallback` tells you it
happened. **Ship this in any service whose prompts live in Langfuse.**

*(The SDK logs the 404 it fell back from, with full response headers. The cell quiets that so the
output stays readable &mdash; in a real service you want that log, because a silent fallback means
you are serving yesterday's prompt without knowing it.)*
"""),
    code('''
def with_fallback():
    if not LF:
        return
    # The SDK logs the 404 it fell back from, headers and all. That is right in
    # production and unreadable in a notebook, so quiet it for this one call.
    import logging
    lg = logging.getLogger("langfuse")
    before = lg.level
    lg.setLevel(logging.CRITICAL)
    try:
        p = LF.get_prompt("a-prompt-that-does-not-exist",
                          fallback="Classify in one word.\\nTicket: {{ticket}}", max_retries=0)
    finally:
        lg.setLevel(before)
    print("is_fallback:", p.is_fallback, "|", p.compile(ticket="printer on fire"))

guard(with_fallback)
'''),

    md("""
## Step 11 &mdash; Look at it

Open Langfuse &rarr; **Prompts** &rarr; your `support-classifier-...`. You will see two versions, the
labels where you left them, and the commit messages. Open the trace from Step 7 and the generation
names the prompt version it used.

## What to take away

- A prompt in Langfuse is **config**, not code. Changing it is not a deploy; rolling it back is not
  a rollback.
- **Labels are the interface.** Application code asks for `production` and never names a version.
- **Link your calls** (`langfuse_prompt=`) or you can see that something got worse without being
  able to see *what changed*.
- **Always pass `fallback=`.** You moved your prompt behind a network call; that is only a good
  trade if it cannot take the service down.

## Your turn

1. Create a version 3 that is deliberately bad &mdash; ask for a sentence instead of one word.
   Promote it, run Step 7, then roll back. That is the whole loop in three minutes.
2. Put a *second* prompt under the same name for a different worker, and decide whether one prompt
   per agent or one per task is the better unit. There is no right answer, but there is a reason.
3. The `config` travelled with the prompt. Change the temperature there and use `p.config["temperature"]`
   in Step 7 instead of a literal &mdash; then say which settings belong with the prompt and which
   belong to the service.
"""),
]

# =========================================================================== #
# main
# =========================================================================== #
# lab-7-02 is a WALKTHROUGH: no blanks, no score. verify.py and verify_labs.py
# both carry the same name in their WALKTHROUGH set.
LABS = [
    ("lab-7-01-locating-the-failing-step", LAB1),
    ("lab-7-02-prompt-versioning-with-langfuse", LAB2),
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
    print(f"\n{len(LABS)} lab, {len(LABS) * 2} notebooks written")


if __name__ == "__main__":
    main()
