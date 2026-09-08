#!/usr/bin/env python3
"""
Generate Module 1 lab notebooks and their solutions from one source.

Rebuilt 2026-09-09: five labs still, but each is two sections instead of three or four,
the scaffolding a participant reads before reaching a framework call is cut to a minimum,
and the domain is a tech-support ticket queue -- something everyone already understands,
so the only unfamiliar thing in the notebook is LangChain.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-1-0N-*.ipynb and ../solutions/

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
# Lab 1.{num} &mdash; {title}

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 1 &middot; Module 1 &mdash; Agents vs. Multi-Agent Systems**

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
# the shared domain -- one flat dict per table, no joins, nothing to learn here
# --------------------------------------------------------------------------- #
DOMAIN = '''
# ------------------------------------------------ the case file (synthetic, self-contained)
# A small tech-support ticket queue. Ordinary rules on purpose: the only new thing in these
# five labs is LangChain. Nothing here is real data and nothing leaves this notebook.

TICKETS = {
    "TCK-4001": {"customer": "Priya Nair",   "product": "VPN Client", "version": "4.2",
                 "severity": "high",   "error_code": "VPN-513",
                 "text": "Cannot connect since the upgrade. Error VPN-513."},
    "TCK-4002": {"customer": "Rahul Menon",  "product": "Reports",    "version": "3.9.1",
                 "severity": "low",    "error_code": "APP-002",
                 "text": "Monthly export finishes but the PDF is blank."},
    "TCK-4003": {"customer": "Anita Sharma", "product": "Reports",    "version": "3.9.1",
                 "severity": "medium", "error_code": None,
                 "text": "It is just slow today. Nothing else to add."},
    "TCK-4004": {"customer": "Vikram Rao",   "product": "VPN Client", "version": "4.2",
                 "severity": "high",   "error_code": "SEC-900",
                 "text": "Got a login alert from a country I have never visited."},
    "TCK-4005": {"customer": "Priya Nair",   "product": "Reports",    "version": "3.9.1",
                 "severity": "low",    "error_code": "APP-002",
                 "text": "Same blank PDF as my colleague reported."},
}

# The runbook: what support is allowed to do about each error code.
RUNBOOK = {
    "VPN-513": "Certificate pinning changed in 4.2. Have the user clear the local trust store "
               "and re-enrol. Five minutes, no data loss. Support may do this without approval.",
    "APP-002": "Known defect in 3.9.1, fixed in 3.9.2. Advise the upgrade. Do not issue a refund "
               "for this and do not raise a new defect -- link the existing one.",
    "SEC-900": "Possible credential compromise. Escalate to the security desk immediately. "
               "Support must not resolve, close or advise the customer directly.",
}

# Which error codes may an agent resolve on its own, and which need a human?
MUST_ESCALATE = {"SEC-900"}

print(f"{len(TICKETS)} tickets, {len(RUNBOOK)} runbook entries loaded")
'''

THREAD_NOTE = (
    "> **The thread.** All five Module 1 labs work one case: a small tech-support ticket queue.\n"
    "> What you build in each lab is picked up by the next one."
)


# =========================================================================== #
# Lab 1.1 -- from a stateless call to an agent loop
# =========================================================================== #
LAB1 = [
    header(1, "From a Stateless Call to an Agent Loop", "Intermediate", 30,
           ["Carry state the way LangChain does &mdash; a list of messages you resend",
            "Let the model choose a tool for real, with <code>bind_tools</code> and <code>tool_calls</code>",
            "Close the loop, and decide what makes it stop",
            "Then replace the whole thing with <code>create_agent</code> and compare"],
           THREAD_NOTE),
    setup(1),
    code(DOMAIN),

    md("""
## Concept

A model call is a **function**: messages in, one message out. It remembers nothing. Everything an
agent appears to know is something your code put back in front of it.

An agent is that function inside a loop:

| Piece | What it is |
|---|---|
| `llm.bind_tools([...])` | a model allowed to answer with a **tool call** instead of prose |
| `msg.tool_calls` | what it asked for &mdash; name and arguments, already parsed |
| `ToolMessage` | the result you hand back, tied to the request by `tool_call_id` |
| the loop | resend, run what it asked for, resend again &mdash; until you stop |

Three decisions in this lab, and they are the three that decide whether an agent works: what you
resend, what you append, and when you stop.
"""),

    md("""
## Section 1 &mdash; Messages are the state

Two tools and a history. The model sees only what is in that list.
"""),

    code('''
import json
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

SYSTEM = ("You are a tech support analyst. Answer only from the data the tools give you. "
          "If the runbook says escalate, say so and stop.")

@tool
def lookup_ticket(ref: str) -> str:
    """Return the support ticket for one reference such as 'TCK-4001'.

    Use when you need the customer, product, severity or error code of a specific ticket.
    """
    t = TICKETS.get(ref)
    return json.dumps({"ref": ref, **t}) if t else f"no ticket {ref!r}"

@tool
def runbook_for(error_code: str) -> str:
    """Return what support is allowed to do about one error code, e.g. 'VPN-513'.

    Use after you know why a ticket failed and need to know what to do about it.
    """
    return RUNBOOK.get(error_code, f"no runbook entry for {error_code!r}")

TOOLS = {t.name: t for t in (lookup_ticket, runbook_for)}


def new_history(question: str) -> list:
    """The model is stateless. What does it need in front of it to answer at all?"""
    with_instructions = [SystemMessage(SYSTEM), HumanMessage(question)]
    question_only     = [HumanMessage(question)]
    return BLANK        # TODO: which one, and why does it have to be resent every turn?
''', '''
import json
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

SYSTEM = ("You are a tech support analyst. Answer only from the data the tools give you. "
          "If the runbook says escalate, say so and stop.")

@tool
def lookup_ticket(ref: str) -> str:
    """Return the support ticket for one reference such as 'TCK-4001'.

    Use when you need the customer, product, severity or error code of a specific ticket.
    """
    t = TICKETS.get(ref)
    return json.dumps({"ref": ref, **t}) if t else f"no ticket {ref!r}"

@tool
def runbook_for(error_code: str) -> str:
    """Return what support is allowed to do about one error code, e.g. 'VPN-513'.

    Use after you know why a ticket failed and need to know what to do about it.
    """
    return RUNBOOK.get(error_code, f"no runbook entry for {error_code!r}")

TOOLS = {t.name: t for t in (lookup_ticket, runbook_for)}


def new_history(question: str) -> list:
    """The model is stateless. What does it need in front of it to answer at all?"""
    with_instructions = [SystemMessage(SYSTEM), HumanMessage(question)]
    question_only     = [HumanMessage(question)]
    return with_instructions   # the system prompt is not remembered -- it is resent
'''),

    code('''
def advance(history: list, ai_msg) -> list:
    """The model asked for tools. Run them, and return the history for the next turn."""
    results = [ToolMessage(content=str(TOOLS[c["name"]].invoke(c["args"])), tool_call_id=c["id"])
               for c in ai_msg.tool_calls]

    results_only         = history + results
    request_and_results  = history + [ai_msg] + results

    return BLANK        # TODO: what must the model see on its next turn?
''', '''
def advance(history: list, ai_msg) -> list:
    """The model asked for tools. Run them, and return the history for the next turn."""
    results = [ToolMessage(content=str(TOOLS[c["name"]].invoke(c["args"])), tool_call_id=c["id"])
               for c in ai_msg.tool_calls]

    results_only         = history + results
    request_and_results  = history + [ai_msg] + results

    return request_and_results   # a result with no request attached is an orphan
'''),

    code('''
# --- Self-check: Section 1   (real tools and real message objects -- no model call)
class _FakeAI:
    """Stands in for the model's reply so this check needs no endpoint."""
    tool_calls = [{"name": "lookup_ticket", "args": {"ref": "TCK-4001"}, "id": "call_1"}]

check("the history the model sees carries your instructions, not just the question",
      lambda: any(isinstance(m, SystemMessage) for m in new_history("what is wrong with TCK-4001?")),
      "nothing is remembered between calls -- the system prompt is resent every turn")
check("a tool result goes back with the request that asked for it",
      lambda: [type(m).__name__ for m in advance([], _FakeAI())] == ["_FakeAI", "ToolMessage"],
      "a ToolMessage with no AIMessage before it is an orphan the model cannot match up")
check("the result carries the real tool output and the call's id",
      lambda: (lambda m: "Priya Nair" in m.content and m.tool_call_id == "call_1")(
              advance([], _FakeAI())[-1]))
score()
'''),

    md("""
## Section 2 &mdash; The loop, and what stops it

Resend, run what it asked for, resend again. The only hard part is the exit.

Three things can end a run, and an agent that checks only the first is the single most common
thing to go wrong in production.
"""),

    code('''
MAX_STEPS = 6

def should_stop(ai_msg, steps: int, seen: list) -> bool:
    """Three reasons a run should end. Which of them count?"""
    answered      = not ai_msg.tool_calls
    budget_spent  = steps >= MAX_STEPS
    going_in_circles = len(seen) != len(set(seen))

    return BLANK        # TODO: combine the ones that should end the loop
''', '''
MAX_STEPS = 6

def should_stop(ai_msg, steps: int, seen: list) -> bool:
    """Three reasons a run should end. Which of them count?"""
    answered      = not ai_msg.tool_calls
    budget_spent  = steps >= MAX_STEPS
    going_in_circles = len(seen) != len(set(seen))

    return answered or budget_spent or going_in_circles
'''),

    code('''
def run_agent(question: str, model=None):
    """The whole agent: history, model, tools, loop, stop."""
    model = model or get_llm().bind_tools(list(TOOLS.values()))
    history, seen, steps = new_history(question), [], 0
    while True:
        ai = model.invoke(history)
        seen += [f'{c["name"]}({c["args"]})' for c in ai.tool_calls]
        steps += 1
        if should_stop(ai, steps, seen):
            return ai, seen, steps
        history = advance(history, ai)


# --- Self-check: Section 2   (the stop rule alone -- pure function, no model)
class _Ask:  tool_calls = [{"name": "lookup_ticket", "args": {}, "id": "c"}]
class _Done: tool_calls = []

check("a reply with no tool call ends the run",
      lambda: should_stop(_Done(), 1, []) is True)
check("the step budget ends a run that keeps asking",
      lambda: should_stop(_Ask(), MAX_STEPS, ["a", "b"]) is True,
      "without this an agent that never converges never stops")
check("a repeated tool call ends it too",
      lambda: should_stop(_Ask(), 2, ["same", "same"]) is True,
      "the same call twice means it is not learning anything from the results")
check("an ordinary turn in progress does not stop",
      lambda: should_stop(_Ask(), 2, ["a", "b"]) is False)
score()
'''),

    md("""
## Run it for real &mdash; part 1: the model remembers nothing
"""),

    code('''
if llm_ready():
    print("Q1:", ask("Ticket TCK-4001 is about a VPN client. Reply with just the product name.")[:80])
    print("Q2:", ask("Which ticket did I just ask you about?")[:120])
    print("\\n^ the second call has no idea. Nothing was carried over, because nothing carries itself.")
'''),

    md("""
## Run it for real &mdash; part 2: your loop, then the one-liner

`TCK-4004` is the interesting one: the model has to look the ticket up, find `SEC-900`, read the
runbook, and discover it is not allowed to act.
"""),

    code('''
def show(label, question):
    ai, seen, steps = run_agent(question)
    print(f"=== {label} ===")
    for s in seen:
        print("   called:", s)
    print(f"   steps: {steps}\\n   answer: {str(ai.content)[:260]}\\n")

if llm_ready():
    guard(lambda: show("your loop", "What should we do about ticket TCK-4004?"))
'''),

    code('''
if llm_ready():
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver

    agent = create_agent(model=get_llm(), tools=list(TOOLS.values()),
                         system_prompt=SYSTEM, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "TCK-4004"}}

    out = agent.invoke({"messages": [("user", "What should we do about ticket TCK-4004?")]}, cfg)
    print("create_agent:", out["messages"][-1].content[:260])

    # and because it has a checkpointer, the follow-up needs no context from you
    out = agent.invoke({"messages": [("user", "Which customer was that?")]}, cfg)
    print("\\nfollow-up  :", out["messages"][-1].content[:160])
'''),

    md("""
### Read it

1. **The loop is the agent.** Not the model, not the tools &mdash; the loop that resends. You wrote
   about fifteen lines and it is a working agent.
2. **`create_agent` is that loop.** Not a different thing: the same thing, with the message
   plumbing and the stop rule already written. Note the argument is `system_prompt=`, not `prompt=`
   &mdash; `prompt=` raises `TypeError` on this version.
3. **The follow-up question worked.** A `checkpointer` plus a `thread_id` is Module 3's whole
   subject, arriving four hours early because it is one keyword argument.
4. **`TCK-4004` stopped rather than helped.** The runbook said escalate and the agent said so.
   That behaviour came from the data, not from the model being careful &mdash; which is Module 8's
   entire argument.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Set `MAX_STEPS = 1` and run `TCK-4004`. The agent answers from a single lookup, confidently and
   wrongly. A budget too tight is its own failure mode.
2. Ask about `TCK-4003`, which has no error code. Watch what the agent does when a tool returns
   nothing useful &mdash; that is the failure Module 4 is about.
"""),
]


# =========================================================================== #
# Lab 1.2 -- the four building blocks
# =========================================================================== #
LAB2 = [
    header(2, "The Four Building Blocks", "Intermediate", 30,
           ["Build a <code>@tool</code> that knows what it is not allowed to do",
            "Bound the history with <code>trim_messages</code> &mdash; and meet the counter that breaks here",
            "Turn a goal into a typed <code>Plan</code> with <code>with_structured_output</code>",
            "Assemble all four blocks into one agent"],
           THREAD_NOTE),
    setup(2),
    code(DOMAIN),

    md("""
## Concept

Every agent is the same four parts. Module 1 gives you each as a real LangChain object.

| Block | The idea | The object |
|---|---|---|
| **LLM** | the reasoning | `ChatOpenAI` |
| **Tools** | the actions | `@tool` |
| **Memory** | what survives a turn | `trim_messages`, a checkpointer |
| **Planning** | a goal is not a sequence | `with_structured_output(...)` |

The one worth slowing down on is **Tools**, because a tool is not a function &mdash; it is a
function *plus what the model is told about it*, and both halves are yours to get right.
"""),

    md("""
## Section 1 &mdash; Tools and memory

A tool that changes something needs to know what it must not change. And a history that grows
without bound will eventually push your instructions out of the window.
"""),

    code('''
import json
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

SYSTEM = ("You are a tech support analyst. Answer only from the data the tools give you. "
          "If the runbook says escalate, say so and stop.")


@tool
def resolve_ticket(ref: str, resolved_by: str) -> str:
    """Close a support ticket. Only call this once the runbook says support may act.

    Requires the name of the person taking responsibility for the resolution.
    """
    code = TICKETS.get(ref, {}).get("error_code")
    if BLANK:                      # TODO: when must this tool refuse, whatever the model asked?
        return f"refused: {code} must go to the security desk, not be resolved here"
    return f"{ref} resolved by {resolved_by}"


def bounded(messages: list, max_tokens: int = 120) -> list:
    """Keep the system message and as many recent turns as fit."""
    # token_counter=<your chat model> is what everyone writes first, and it raises
    # NotImplementedError here: there is no tiktoken encoding for this model.
    with_the_model     = get_llm                   # referenced, not called
    with_a_plain_count = count_tokens_approximately

    return trim_messages(messages, max_tokens=max_tokens,
                         token_counter=BLANK,      # TODO: only one of these works on this model
                         strategy="last", include_system=True,
                         start_on="human", allow_partial=False)
''', '''
import json
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

SYSTEM = ("You are a tech support analyst. Answer only from the data the tools give you. "
          "If the runbook says escalate, say so and stop.")


@tool
def resolve_ticket(ref: str, resolved_by: str) -> str:
    """Close a support ticket. Only call this once the runbook says support may act.

    Requires the name of the person taking responsibility for the resolution.
    """
    code = TICKETS.get(ref, {}).get("error_code")
    if code in MUST_ESCALATE:      # the tool refuses; it does not rely on the model behaving
        return f"refused: {code} must go to the security desk, not be resolved here"
    return f"{ref} resolved by {resolved_by}"


def bounded(messages: list, max_tokens: int = 120) -> list:
    """Keep the system message and as many recent turns as fit."""
    # token_counter=<your chat model> is what everyone writes first, and it raises
    # NotImplementedError here: there is no tiktoken encoding for this model.
    with_the_model     = get_llm                   # referenced, not called
    with_a_plain_count = count_tokens_approximately

    return trim_messages(messages, max_tokens=max_tokens,
                         token_counter=with_a_plain_count,
                         strategy="last", include_system=True,
                         start_on="human", allow_partial=False)
'''),

    code('''
# --- Self-check: Section 1   (a real @tool and a real trim -- neither calls the model)
def long_history():
    msgs = [SystemMessage(SYSTEM), HumanMessage("Look at TCK-4001.")]
    for i in range(12):
        msgs += [AIMessage(f"step {i}: " + "checking the ticket. " * 12), HumanMessage(f"and then? ({i})")]
    return msgs

check("resolve_ticket refuses the security ticket even when told to close it",
      lambda: "refused" in resolve_ticket.invoke({"ref": "TCK-4004", "resolved_by": "Dev"}),
      "the runbook says escalate -- so the TOOL enforces it, rather than hoping the model read it")
check("it still resolves an ordinary ticket",
      lambda: "resolved by" in resolve_ticket.invoke({"ref": "TCK-4001", "resolved_by": "Dev"}))
check("the history is bounded",
      lambda: count_tokens_approximately(bounded(long_history())) <= 130,
      "token_counter=get_llm() raises NotImplementedError here -- there is no tiktoken encoding "
      "for this model, so the counter has to be a plain function over the text")
check("trimming kept the instructions and dropped old turns",
      lambda: (isinstance(bounded(long_history())[0], SystemMessage)
               and len(bounded(long_history())) < len(long_history())))
score()
'''),

    md("""
## Section 2 &mdash; Planning, and all four together

`with_structured_output(Plan)` makes the model return a **`Plan` object**, not prose you then have
to parse.

The `Field(description=...)` lines are the part that matters and the part everyone treats as
documentation. They are not documentation: LangChain sends them to the model **as the schema**.
They are the only instruction it gets about what belongs in each field.
"""),

    code('''
from typing import List
from pydantic import BaseModel, Field


class Step(BaseModel):
    """One step of a support investigation."""
    name: str = Field(description="BLANK")   # TODO: what must a step name look like?
    tool: str = Field(description="BLANK")   # TODO: name the allowed values, precisely


class Plan(BaseModel):
    """An ordered plan for handling one support ticket."""
    goal: str = Field(description="The question this plan answers, in one line")
    steps: List[Step] = Field(description="The steps, in the order they should run")
''', '''
from typing import List
from pydantic import BaseModel, Field


class Step(BaseModel):
    """One step of a support investigation."""
    name: str = Field(description="A short imperative label, e.g. 'read the ticket'")
    tool: str = Field(description="One of: lookup_ticket, runbook_for, resolve_ticket, none")


class Plan(BaseModel):
    """An ordered plan for handling one support ticket."""
    goal: str = Field(description="The question this plan answers, in one line")
    steps: List[Step] = Field(description="The steps, in the order they should run")
'''),

    code('''
# --- Self-check: Section 2   (the schema object, before any model sees it)
def described(field):
    """The description the model will be sent. Unfilled blanks are still the literal 'BLANK'."""
    d = Step.model_fields[field].description
    if d.strip() == "BLANK":
        raise NameError(f"Step.{field} description is still BLANK")   # -> [TODO], not [FAIL]
    return d

check("the step name description says what a name should look like",
      lambda: len(described("name")) > 20)
check("the tool description names the allowed values",
      lambda: sum(t in described("tool") for t in ("lookup_ticket", "runbook_for", "none")) >= 2,
      "the model cannot pick from a list it was never shown")
def _rejects():
    """Pydantic must object to a step with no tool."""
    try:
        Plan(goal="g", steps=[{"name": "read"}])
        return False
    except Exception:
        return True

check("Plan validates a well-formed plan and rejects a malformed one",
      lambda: bool(Plan(goal="g", steps=[Step(name="read the ticket", tool="lookup_ticket")]))
              and _rejects())
score()
'''),

    md("""
## Run it for real &mdash; all four blocks, one agent
"""),

    code('''
if llm_ready():
    plan = guard(lambda: get_llm().with_structured_output(Plan).invoke(
        "Plan how to handle support ticket TCK-4004. Use only the tools named in the schema."))
    if plan:
        print("goal:", plan.goal)
        for s in plan.steps:
            print(f"   {s.name:38} -> {s.tool}")
'''),

    code('''
def assemble_and_run():
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver

    @tool
    def lookup_ticket(ref: str) -> str:
        """Return the support ticket for one reference such as 'TCK-4001'."""
        t = TICKETS.get(ref)
        return json.dumps({"ref": ref, **t}) if t else f"no ticket {ref!r}"

    @tool
    def runbook_for(error_code: str) -> str:
        """Return what support is allowed to do about one error code, e.g. 'VPN-513'."""
        return RUNBOOK.get(error_code, f"no runbook entry for {error_code!r}")

    agent = create_agent(model=get_llm(),                                     # LLM
                         tools=[lookup_ticket, runbook_for, resolve_ticket],  # Tools
                         system_prompt=SYSTEM,                                # Planning, of a sort
                         checkpointer=InMemorySaver())                        # Memory

    for ref in ["TCK-4001", "TCK-4004"]:
        out = agent.invoke({"messages": [("user", f"Handle {ref} end to end.")]},
                           {"configurable": {"thread_id": ref}})
        print(f"{ref}: {out['messages'][-1].content[:200]}\\n")

if llm_ready():
    guard(assemble_and_run)      # resolve_ticket carries a blank -- the agent INVOKES it
'''),

    md("""
### Read it

**`TCK-4001` was resolved and `TCK-4004` was not** &mdash; and the difference was not the model
being careful. `resolve_ticket` refuses, in Python, whatever it is asked. That is the whole idea
behind Module 8: a guardrail the model can talk its way past is not a guardrail.

**The token counter is a real trap.** `token_counter=get_llm()` is the obvious thing to write and
it raises `NotImplementedError` here &mdash; there is no tiktoken encoding for this model.
`count_tokens_approximately` is a plain function over the message text and works everywhere.

**The `Field` descriptions are prompt engineering.** You wrote them as documentation and the model
read them as instructions. Lab 1.3 measures exactly how much that is worth.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Set `max_tokens=40` in `bounded` and print what survives. At what point do your instructions
   fall out of the window? That number is when your agent starts ignoring its system prompt.
2. Change `Step.tool`'s description to just `"the tool"` and re-run the plan. Read what the model
   puts there now.
"""),
]


# =========================================================================== #
# Lab 1.3 -- tool descriptions are instructions, and a typed answer
# =========================================================================== #
LAB3 = [
    header(3, "Tool Descriptions Are Instructions", "Intermediate &rarr; Advanced", 30,
           ["Run the same agent twice: same model, same functions, different descriptions",
            "Measure the difference on a small eval set instead of asserting it",
            "Make the agent return a typed object rather than a paragraph"],
           THREAD_NOTE),
    setup(3),
    code(DOMAIN),

    md("""
## Concept

The model never sees your function. It sees a **name, a description and a parameter schema** &mdash;
and it picks from those alone.

So a tool description is not documentation. It is the prompt that decides whether the right tool
gets called, and it is the cheapest accuracy in this course: no model change, no extra call, no
new framework.

This lab measures it. Same model, same three functions, two sets of words.
"""),

    md("""
## Section 1 &mdash; The same tools, described two ways

The functions and the scoring harness are given. Your job is the words.

Write descriptions that answer the two questions a model actually has: **what does this return**,
and **when should I reach for it rather than the other one?**
"""),

    code('''
from langchain_core.tools import StructuredTool

def _ticket(ref: str) -> str:
    t = TICKETS.get(ref)
    return json.dumps({"ref": ref, **t}) if t else f"no ticket {ref!r}"

def _runbook(error_code: str) -> str:
    return RUNBOOK.get(error_code, f"no runbook entry for {error_code!r}")

def _similar(error_code: str) -> str:
    return json.dumps([r for r, t in TICKETS.items() if t["error_code"] == error_code])

OPS = {"lookup_ticket": _ticket, "runbook_for": _runbook, "find_similar": _similar}

def build_tools(descriptions: dict) -> list:
    """Three real LangChain tools over the same three functions, with the words you choose."""
    return [StructuredTool.from_function(func=OPS[n], name=n, description=d)
            for n, d in descriptions.items()]


POOR = {
    "lookup_ticket": "gets ticket data",
    "runbook_for":   "gets runbook data",
    "find_similar":  "finds things",
}

def good_descriptions() -> dict:
    """Same three functions. Tell the model what each returns and when to prefer it."""
    return {
        "lookup_ticket": BLANK,   # TODO: takes a ticket ref like 'TCK-4001'. Returns what?
        "runbook_for":   BLANK,   # TODO: takes an error code. When is it the right call?
        "find_similar":  BLANK,   # TODO: takes an error code. How is it NOT runbook_for?
    }
''', '''
from langchain_core.tools import StructuredTool

def _ticket(ref: str) -> str:
    t = TICKETS.get(ref)
    return json.dumps({"ref": ref, **t}) if t else f"no ticket {ref!r}"

def _runbook(error_code: str) -> str:
    return RUNBOOK.get(error_code, f"no runbook entry for {error_code!r}")

def _similar(error_code: str) -> str:
    return json.dumps([r for r, t in TICKETS.items() if t["error_code"] == error_code])

OPS = {"lookup_ticket": _ticket, "runbook_for": _runbook, "find_similar": _similar}

def build_tools(descriptions: dict) -> list:
    """Three real LangChain tools over the same three functions, with the words you choose."""
    return [StructuredTool.from_function(func=OPS[n], name=n, description=d)
            for n, d in descriptions.items()]


POOR = {
    "lookup_ticket": "gets ticket data",
    "runbook_for":   "gets runbook data",
    "find_similar":  "finds things",
}

def good_descriptions() -> dict:
    """Same three functions. Tell the model what each returns and when to prefer it."""
    return {
        "lookup_ticket": ("Return one support ticket by its reference, e.g. 'TCK-4001': customer, "
                          "product, version, severity and error code. Start here when you are "
                          "given a ticket reference and do not yet know what is wrong."),
        "runbook_for":   ("Return what support is ALLOWED to do about one error code, e.g. "
                          "'VPN-513'. Call this once you know the error code and need the action, "
                          "including whether it must be escalated."),
        "find_similar":  ("Return the references of other tickets reporting the same error code. "
                          "Use this to judge how widespread an issue is -- not to find out what "
                          "to do about it, which is runbook_for."),
    }
'''),

    code('''
# --- Self-check: Section 1   (real StructuredTool objects -- built, not invoked, so no model)
def tool_schema(descriptions):
    return {t.name: t.description for t in build_tools(descriptions)}

check("three real tools are built from the words you wrote",
      lambda: set(tool_schema(good_descriptions())) == set(OPS))
check("each description says what the tool returns, not just that it exists",
      lambda: all(len(d) > 60 for d in tool_schema(good_descriptions()).values()),
      "'gets ticket data' is the version we are measuring against -- say what comes back")
check("find_similar distinguishes itself from runbook_for",
      lambda: "runbook" in tool_schema(good_descriptions())["find_similar"].lower(),
      "two tools taking the same argument is exactly where a model guesses; say which is which")
score()
'''),

    md("""
## Section 2 &mdash; A typed answer, not a paragraph

A paragraph has to be parsed by whatever comes next. A schema does not. Declare what a resolution
*is*, and the model fills it in.

The escalation rule is the interesting field: it must come from the runbook, not from the model's
judgement about how serious the ticket sounds.
"""),

    code('''
from pydantic import BaseModel, Field

class Resolution(BaseModel):
    """What support decided about one ticket."""
    ticket: str    = Field(description="The ticket reference, e.g. 'TCK-4001'")
    action: str    = Field(description="BLANK")   # TODO: what should this say, and how long?
    escalate: bool = Field(description="BLANK")   # TODO: on what basis is this true?


def expected_escalation(ref: str) -> bool:
    """Ground truth for the eval: which tickets must NOT be resolved by support."""
    return TICKETS[ref]["error_code"] in BLANK    # TODO: which set decides this?
''', '''
from pydantic import BaseModel, Field

class Resolution(BaseModel):
    """What support decided about one ticket."""
    ticket: str    = Field(description="The ticket reference, e.g. 'TCK-4001'")
    action: str    = Field(description="The single next action, in one imperative sentence")
    escalate: bool = Field(description="True only if the runbook says this must go to another "
                                       "desk, not if the ticket merely sounds serious")


def expected_escalation(ref: str) -> bool:
    """Ground truth for the eval: which tickets must NOT be resolved by support."""
    return TICKETS[ref]["error_code"] in MUST_ESCALATE
'''),

    code('''
# --- Self-check: Section 2   (the schema, and the ground truth -- no model yet)
def described(field):
    d = Resolution.model_fields[field].description
    if d.strip() == "BLANK":
        raise NameError(f"Resolution.{field} description is still BLANK")   # -> [TODO]
    return d

check("action says what to write, not just that something goes there",
      lambda: len(described("action")) > 25)
check("escalate is tied to the runbook rather than to how the ticket sounds",
      lambda: "runbook" in described("escalate").lower(),
      "'high severity' and 'must escalate' are different things -- TCK-4001 is high and resolvable")
check("only the security ticket must be escalated",
      lambda: [r for r in sorted(TICKETS) if expected_escalation(r)] == ["TCK-4004"])
score()
'''),

    md("""
## Run it for real &mdash; the measurement

Five questions, each with one obviously correct first tool. Same model, same functions, twice.
"""),

    code('''
EVAL = [
    ("What is wrong with TCK-4001?",                       "lookup_ticket"),
    ("What are we allowed to do about error VPN-513?",     "runbook_for"),
    ("How many customers are hitting APP-002?",            "find_similar"),
    ("Who raised TCK-4004?",                               "lookup_ticket"),
    ("Is SEC-900 something support can close?",            "runbook_for"),
]

def first_tool(question, tools):
    """Which tool does the model reach for first? Given -- you are measuring, not building."""
    msg = get_llm().bind_tools(tools).invoke(
        [("system", "You are a tech support analyst. Use a tool."), ("human", question)])
    return msg.tool_calls[0]["name"] if msg.tool_calls else "(none)"

def score_arm(descriptions):
    tools = build_tools(descriptions)
    hits = [(q, want, first_tool(q, tools)) for q, want in EVAL]
    return sum(w == g for _, w, g in hits), hits


if llm_ready():
    def measure():
        for label, desc in [("poor", POOR), ("good", good_descriptions())]:
            n, hits = score_arm(desc)
            print(f"--- {label} descriptions: {n}/{len(EVAL)} first tools correct")
            for q, want, got in hits:
                print(f"    {'ok ' if want == got else 'MISS'} {q[:46]:48} want={want:14} got={got}")
    guard(measure)
'''),

    md("""
### Read it

Same model. Same three Python functions. Only the words changed.

**What the poor arm gets wrong is not random.** It confuses `runbook_for` and `find_similar` &mdash;
two tools taking the same argument, described in a way that does not distinguish them. That is
where a model always guesses, and it is the thing your descriptions have to fix.

**This is the cheapest accuracy in the course.** No model change, no extra call, no new framework
&mdash; and it is the first thing to check when an agent picks the wrong tool. Module 4 does this
properly against a larger eval set; the habit starts here.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Make `find_similar`'s description say only &ldquo;searches tickets&rdquo; and re-run. Which
   questions move? Bad descriptions fail in a *predictable* place, which is why this is debuggable.
2. Add a fourth tool that genuinely overlaps &mdash; `ticket_history(ref)` &mdash; and get the
   agent to five out of five again. The fix is in the words, not the code.
"""),
]


# =========================================================================== #
# Lab 1.4 -- the coordination tax
# =========================================================================== #
LAB4 = [
    header(4, "The Coordination Tax", "Advanced", 35,
           ["Build the same capability twice: one agent, and three specialists",
            "Instrument both &mdash; calls, tokens, latency &mdash; before arguing about either",
            "Score them on one eval set and let the result be whatever it is"],
           THREAD_NOTE),
    setup(4),
    code(DOMAIN),

    md("""
## Concept

&ldquo;Split it into specialists&rdquo; sounds obviously right. It is a design decision with a
price, and the price is rarely the one people quote.

You are going to build both, measure both, and then decide. The rule for this lab: **the team is
allowed to lose.** That outcome is the lesson, not a lab failure.
"""),

    md("""
## Section 1 &mdash; Two architectures over the same tools

Same model, same three tools, same questions. The only difference is how the work is divided.

Deciding **which tools each specialist gets** is the entire design of a split &mdash; and giving
one of them too little is how a handoff starts losing information.
"""),

    code('''
import json, time
from langchain_core.tools import tool
from langchain.agents import create_agent

@tool
def lookup_ticket(ref: str) -> str:
    """Return the support ticket for one reference such as 'TCK-4001': customer, product,
    version, severity and error code."""
    t = TICKETS.get(ref)
    return json.dumps({"ref": ref, **t}) if t else f"no ticket {ref!r}"

@tool
def runbook_for(error_code: str) -> str:
    """Return what support is allowed to do about one error code, e.g. 'VPN-513', including
    whether it must be escalated."""
    return RUNBOOK.get(error_code, f"no runbook entry for {error_code!r}")

@tool
def find_similar(error_code: str) -> str:
    """Return the references of other tickets reporting the same error code."""
    return json.dumps([r for r, t in TICKETS.items() if t["error_code"] == error_code])

ALL_TOOLS = {"lookup_ticket": lookup_ticket, "runbook_for": runbook_for, "find_similar": find_similar}


def specialist_tools() -> dict:
    """Which tools does each specialist need to do its own job -- and only its own job?"""
    return {
        "triage":   BLANK,   # TODO: reads the ticket and reports what it is. Which tool(s)?
        "diagnose": BLANK,   # TODO: given an error code, works out what is allowed. Which tool(s)?
        "resolve":  BLANK,   # TODO: writes the answer. Does it need a tool at all?
    }
''', '''
import json, time
from langchain_core.tools import tool
from langchain.agents import create_agent

@tool
def lookup_ticket(ref: str) -> str:
    """Return the support ticket for one reference such as 'TCK-4001': customer, product,
    version, severity and error code."""
    t = TICKETS.get(ref)
    return json.dumps({"ref": ref, **t}) if t else f"no ticket {ref!r}"

@tool
def runbook_for(error_code: str) -> str:
    """Return what support is allowed to do about one error code, e.g. 'VPN-513', including
    whether it must be escalated."""
    return RUNBOOK.get(error_code, f"no runbook entry for {error_code!r}")

@tool
def find_similar(error_code: str) -> str:
    """Return the references of other tickets reporting the same error code."""
    return json.dumps([r for r, t in TICKETS.items() if t["error_code"] == error_code])

ALL_TOOLS = {"lookup_ticket": lookup_ticket, "runbook_for": runbook_for, "find_similar": find_similar}


def specialist_tools() -> dict:
    """Which tools does each specialist need to do its own job -- and only its own job?"""
    return {
        "triage":   ["lookup_ticket"],
        "diagnose": ["runbook_for", "find_similar"],
        "resolve":  [],              # it writes the answer from what it was handed
    }
'''),

    code('''
ROLES = {
    "triage":   "Read the ticket and state the error code and severity. Two lines maximum.",
    "diagnose": "Given an error code, state what support is allowed to do and whether it escalates.",
    "resolve":  "Write the final answer for the customer from what you were told. Three lines maximum.",
}

def make_agent(names, instructions):
    return create_agent(model=get_llm(), tools=[ALL_TOOLS[n] for n in names],
                        system_prompt=instructions)

def usage(out):
    """Calls, tokens and the final text out of an agent result. Given -- you are measuring."""
    msgs = out["messages"]
    calls = sum(len(getattr(m, "tool_calls", []) or []) for m in msgs)
    toks = sum((getattr(m, "usage_metadata", None) or {}).get("total_tokens", 0) for m in msgs)
    return calls, toks, msgs[-1].content


# --- Self-check: Section 1   (the split, and that it names real tools -- no model needed)
check("every specialist names tools that exist",
      lambda: all(n in ALL_TOOLS for names in specialist_tools().values() for n in names))
check("triage can read a ticket and diagnose can read the runbook",
      lambda: ("lookup_ticket" in specialist_tools()["triage"]
               and "runbook_for" in specialist_tools()["diagnose"]))
check("no specialist got every tool",
      lambda: all(len(v) < len(ALL_TOOLS) for v in specialist_tools().values()),
      "if one of them has all three you have built one agent with extra steps")
score()
'''),

    md("""
## Section 2 &mdash; One eval set, both arms

Five questions with an answer you can check by substring. Crude on purpose: a number you actually
compute beats a number you assert.

Then the part that matters &mdash; deciding **what the comparison is for**. Write the rule before
you see the result.
"""),

    code('''
EVAL = [
    ("What should we do about TCK-4001?", "trust store"),
    ("What should we do about TCK-4002?", "3.9.2"),
    ("What should we do about TCK-4004?", "security"),
    ("What should we do about TCK-4005?", "3.9.2"),
    ("Can support close TCK-4004?",       "security"),
]

def verdict(single: dict, team: dict) -> str:
    """single and team each look like {"passed": int, "tokens": int, "seconds": float}."""
    cheaper_wins  = "team" if team["tokens"] < single["tokens"] else "single"
    quality_first = ("team" if team["passed"] > single["passed"] else
                     "single" if single["passed"] > team["passed"] else
                     ("team" if team["tokens"] < single["tokens"] else "single"))

    return BLANK        # TODO: which rule would you defend in a design review?
''', '''
EVAL = [
    ("What should we do about TCK-4001?", "trust store"),
    ("What should we do about TCK-4002?", "3.9.2"),
    ("What should we do about TCK-4004?", "security"),
    ("What should we do about TCK-4005?", "3.9.2"),
    ("Can support close TCK-4004?",       "security"),
]

def verdict(single: dict, team: dict) -> str:
    """single and team each look like {"passed": int, "tokens": int, "seconds": float}."""
    cheaper_wins  = "team" if team["tokens"] < single["tokens"] else "single"
    quality_first = ("team" if team["passed"] > single["passed"] else
                     "single" if single["passed"] > team["passed"] else
                     ("team" if team["tokens"] < single["tokens"] else "single"))

    return quality_first   # cost only settles a tie -- a cheaper wrong answer is not a saving
'''),

    code('''
# --- Self-check: Section 2   (the decision rule, on numbers you invent -- no model)
def r(passed, tokens, seconds=1.0):
    return {"passed": passed, "tokens": tokens, "seconds": seconds}

check("a better answer wins even when it costs more",
      lambda: verdict(r(3, 5000), r(5, 9000)) == "team")
check("a cheaper wrong answer does not win",
      lambda: verdict(r(5, 9000), r(2, 3000)) == "single",
      "this is the rule that stops 'the team is cheaper' ending a design review")
check("cost settles a tie",
      lambda: verdict(r(5, 9000), r(5, 4000)) == "team")
score()
'''),

    md("""
## Run it for real &mdash; both arms, one eval set
"""),

    code('''
def run_single(q):
    a = make_agent(list(ALL_TOOLS), "You are a tech support analyst. Answer the question fully.")
    t0 = time.time()
    calls, toks, text = usage(a.invoke({"messages": [("user", q)]}))
    return calls, toks, time.time() - t0, text

def run_team(q):
    """Three agents, each handing the next one prose -- which is the point."""
    ag = {role: make_agent(specialist_tools()[role], ROLES[role]) for role in ROLES}
    t0, calls, toks = time.time(), 0, 0
    handoff = q
    for role in ["triage", "diagnose", "resolve"]:
        c, k, handoff = usage(ag[role].invoke({"messages": [("user", handoff)]}))
        calls, toks = calls + c, toks + k
    return calls, toks, time.time() - t0, handoff

def arm(runner):
    passed = calls = toks = 0; secs = 0.0
    for q, want in EVAL:
        c, k, s, text = runner(q)
        ok = want.lower() in (text or "").lower()
        passed += ok; calls += c; toks += k; secs += s
        print(f"   {'ok  ' if ok else 'MISS'} {q[:34]:36} want={want:12} {c} calls, {k} tokens")
    return {"passed": passed, "calls": calls, "tokens": toks, "seconds": secs}


if llm_ready():
    def bakeoff():
        print("--- one agent, three tools ---");    single = arm(run_single)
        print("--- three specialists ---");         team   = arm(run_team)
        print(f"\\n{'':10}{'passed':>8}{'calls':>8}{'tokens':>9}{'seconds':>9}")
        for name, m in [("single", single), ("team", team)]:
            print(f"{name:10}{m['passed']:>6}/5{m['calls']:>8}{m['tokens']:>9}{m['seconds']:>9.1f}")
        print("\\nverdict:", verdict(single, team))
    guard(bakeoff)
'''),

    md("""
### Read it

Whatever your numbers were, read them against the claim everybody repeats &mdash; that multi-agent
costs **3&ndash;10&times;** the tokens.

On this sandbox it does not. Measured over this eval set, tokens come out roughly **level**; what
actually multiplies is **calls** (about 2&times;) and **latency** (about 1.5&times;), and what
falls is the **pass rate**. The coordination tax is real and it is not mainly a token bill.

**Where the quality goes.** Each specialist hands the next one prose. The error code survives that
&mdash; check it &mdash; but the *context* does not: `resolve` never saw the ticket, only a summary
of a summary. That is fragmentation, not data loss, and it is why Module 5 hands structured state
between agents rather than sentences.

**So the honest answer here is &ldquo;one agent&rdquo;**, and being able to say that with a number
behind it is the point of Module 1. Lab 1.5 turns it into a rule you can apply before building.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Give `resolve` the `lookup_ticket` tool so it can re-read the ticket instead of trusting the
   handoff. Does the pass rate recover? What did it cost?
2. Hand a dict between the specialists instead of prose &mdash; ticket, error code, runbook text.
   You have just invented Module 5's shared state, and you can measure what it bought.
"""),
]


# =========================================================================== #
# Lab 1.5 -- challenge: the decision rubric
# =========================================================================== #
LAB5 = [
    header(5, "Challenge &mdash; The Decision Rubric", "Advanced", 35,
           ["Turn &ldquo;do we need multiple agents?&rdquo; into a rule you can apply before building",
            "Include the answer everyone forgets: sometimes the right build is no agent",
            "Apply it to four real systems, then watch a model disagree with it"],
           THREAD_NOTE),
    setup(5),
    code(DOMAIN),

    md("""
## Concept

Lab 1.4 measured one split and the honest answer was &ldquo;one agent&rdquo;. That is a result
about one system. This lab turns it into something you can carry into a design review before
anything is built.

Three answers, not two. The one that gets skipped is the first:

| Answer | When |
|---|---|
| **no agent** | nothing has to be *decided* at runtime &mdash; a script, a query or a RAG lookup is the whole job |
| **one agent** | there is a decision, and one loop with the right tools makes it |
| **multi-agent** | the split has a reason you can state, and a measured gain |

This is the take-home artifact of Module 1. Write it so a colleague could apply it without you.
"""),

    md("""
## Section 1 &mdash; The rubric, as code

A system is described by five facts. Two of them decide it.
"""),

    code('''
def recommend(s: dict) -> str:
    """s: decides_at_runtime, distinct_skills, runs_in_parallel, shared_context, measured_gain.

    Returns 'no agent', 'one agent' or 'multi-agent'.
    """
    if not s["decides_at_runtime"]:
        return BLANK        # TODO: nothing chooses anything here. What should they build?

    worth_splitting = BLANK # TODO: what must be true before a split is justified?
                            #       s["distinct_skills"] > 1 is necessary. Is it sufficient?
    return "multi-agent" if worth_splitting else "one agent"
''', '''
def recommend(s: dict) -> str:
    """s: decides_at_runtime, distinct_skills, runs_in_parallel, shared_context, measured_gain.

    Returns 'no agent', 'one agent' or 'multi-agent'.
    """
    if not s["decides_at_runtime"]:
        return "no agent"

    # More than one skill is necessary and nowhere near sufficient -- Lab 1.4 had three skills
    # and still lost. A split needs a reason it pays: either it genuinely runs in parallel, or
    # someone has measured it winning.
    worth_splitting = s["distinct_skills"] > 1 and (s["runs_in_parallel"] or s["measured_gain"])
    return "multi-agent" if worth_splitting else "one agent"
'''),

    code('''
SYSTEMS = {
    # the Lab 1.4 support desk: three skills, sequential, and it MEASURED WORSE
    "support desk (measured in 1.4)":
        dict(decides_at_runtime=True,  distinct_skills=3, runs_in_parallel=False,
             shared_context=True,  measured_gain=False),
    # a nightly report: no decision anywhere
    "nightly ticket summary email":
        dict(decides_at_runtime=False, distinct_skills=2, runs_in_parallel=True,
             shared_context=False, measured_gain=False),
    # answering from a document set: retrieval, not decision
    "answer questions from the runbook PDF":
        dict(decides_at_runtime=False, distinct_skills=1, runs_in_parallel=False,
             shared_context=False, measured_gain=False),
    # genuinely parallel: three independent scans over the same incident
    "incident triage: logs, metrics and traces at once":
        dict(decides_at_runtime=True,  distinct_skills=3, runs_in_parallel=True,
             shared_context=True,  measured_gain=False),
}

# --- Self-check: Section 1   (a pure function over five dicts -- no model)
check("the support desk from Lab 1.4 comes out as one agent",
      lambda: recommend(SYSTEMS["support desk (measured in 1.4)"]) == "one agent",
      "three distinct skills, and it still lost -- so skills alone cannot justify a split")
check("both of the no-decision systems come out as no agent",
      lambda: all(recommend(SYSTEMS[k]) == "no agent" for k in
                  ("nightly ticket summary email", "answer questions from the runbook PDF")),
      "a scheduled job and a RAG lookup are not agents, however fashionable")
check("genuine parallelism justifies the split",
      lambda: recommend(SYSTEMS["incident triage: logs, metrics and traces at once"]) == "multi-agent")
score()
'''),

    md("""
## Section 2 &mdash; The bar, written down first

A rubric answers &ldquo;should we?&rdquo;. You still need &ldquo;did it work?&rdquo; &mdash; and
that number has to be written **before** you see any result, or it becomes a description of
whatever you got.

This is the bar Day 1 hands to Day 3: the capstone is accepted against it, not against a demo.
"""),

    code('''
def bar() -> dict:
    """The acceptance bar for a multi-agent build, agreed before anyone runs anything."""
    generous = {"min_pass_rate": 0.6, "max_token_multiple": 10.0}   # anything can clear this
    defensible = {"min_pass_rate": BLANK,   # TODO: what pass rate would you sign off on?
                  "max_token_multiple": BLANK}   # TODO: how much more than one agent is acceptable?
    return defensible


def accepted(result: dict, b: dict) -> bool:
    """result: {"pass_rate": float, "token_multiple": float}. Given."""
    return result["pass_rate"] >= b["min_pass_rate"] and \\
           result["token_multiple"] <= b["max_token_multiple"]
''', '''
def bar() -> dict:
    """The acceptance bar for a multi-agent build, agreed before anyone runs anything."""
    generous = {"min_pass_rate": 0.6, "max_token_multiple": 10.0}   # anything can clear this
    defensible = {"min_pass_rate": 0.9,        # support answers are shown to customers
                  "max_token_multiple": 2.0}   # twice the cost needs a visible reason
    return defensible


def accepted(result: dict, b: dict) -> bool:
    """result: {"pass_rate": float, "token_multiple": float}. Given."""
    return result["pass_rate"] >= b["min_pass_rate"] and \\
           result["token_multiple"] <= b["max_token_multiple"]
'''),

    code('''
# --- Self-check: Section 2
check("the bar is stricter than one that everything passes",
      lambda: bar()["min_pass_rate"] > 0.6 and bar()["max_token_multiple"] < 10.0,
      "a bar nothing can fail is not a bar")
check("Lab 1.4's team result would be rejected by it",
      lambda: not accepted({"pass_rate": 0.4, "token_multiple": 1.0}, bar()),
      "level on tokens, worse on answers -- cost cannot rescue that")
check("a genuinely better team would be accepted",
      lambda: accepted({"pass_rate": 1.0, "token_multiple": 1.8}, bar()))
score()
'''),

    md("""
## Run it for real &mdash; let a model disagree with you

Describe one system in prose and ask for a recommendation. Then compare it with your rubric.
"""),

    code('''
DESCRIPTION = (
    "A support desk. A ticket arrives; the system reads it, looks up the runbook for its error "
    "code, and writes a reply to the customer. The steps are strictly sequential. We measured a "
    "three-agent version against a single agent on the same five cases: the single agent answered "
    "5 out of 5, the three-agent version 2 out of 5, for roughly the same token count.")

if llm_ready():
    def compare():
        opinion = ask("Should this be built as one agent or as multiple specialised agents? "
                      "Answer in two sentences.\\n\\n" + DESCRIPTION,
                      system="You are a pragmatic AI architect.")
        print("the model says:\\n  ", opinion.strip()[:420])
        print("\\nyour rubric says:", recommend(SYSTEMS["support desk (measured in 1.4)"]))
        print("\\nOnly one of those two is checkable, and only one of them cites the measurement.")
    guard(compare)
'''),

    md("""
### Read it

The model may well agree with you. That is not the point &mdash; ask it twice and see whether it
still does.

**A rubric is checkable and an opinion is not.** Yours is four lines, it names the facts it uses,
and anyone can run it on a system description and get the same answer. That is what makes it
usable in a design review, and it is why this is Module 1's take-home artifact rather than a slide.

**What you take from Module 1:** an agent is a loop you can write in fifteen lines; the four blocks
are real objects; tool descriptions are prompts and worth measuring; splitting into specialists has
a price you can measure; and the answer is sometimes &ldquo;no agent&rdquo;. Module 2 asks how the
agent should *think*, and Module 3 gives it a state you can pause and audit.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Add a fifth fact &mdash; `human_gate` &mdash; and decide whether it changes the recommendation
   or only the design. Getting that distinction right is most of Module 8.
2. Run your rubric over a system you actually own. If it says &ldquo;no agent&rdquo;, that is the
   most valuable answer it can give you.
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-1-01-stateless-call-to-agent-loop", LAB1),
    ("lab-1-02-four-building-blocks",         LAB2),
    ("lab-1-03-tool-descriptions",            LAB3),
    ("lab-1-04-coordination-tax",             LAB4),
    ("lab-1-05-challenge-decision-rubric",    LAB5),
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
