#!/usr/bin/env python3
"""
Generate Module 1 lab notebooks and their solutions from one source.

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
# Lab 1.1 -- LLM vs agent: messages as state, the tool-calling loop, stopping
# =========================================================================== #
LAB1 = [
    header(1, "From a Stateless Call to an Agent Loop", "Intermediate", 35,
           ["Carry state the way LangChain does it &mdash; as a list of message objects you resend",
            "Let the model choose a tool for real, with `bind_tools` and `tool_calls`",
            "Close the loop by feeding results back as `ToolMessage`, and make it stop",
            "Then replace the whole thing with `create_agent` + a checkpointer, and compare"],
           "> **The thread.** All five Module 1 labs work one case: payment exceptions on a small\n"
           "> synthetic ledger. What you build here is extended in every later lab."),
    setup(1),
    code(DOMAIN),

    md("""
## Concept

A model call is a **function**: messages in, message out, nothing retained. An **agent** is that
call placed inside a **loop**, where the model's output chooses the next action and the result is
fed back in as another message.

In LangChain that loop has a precise shape, and it is worth learning the names now because every
later module uses them:

| Object | What it is |
|---|---|
| `HumanMessage` / `AIMessage` / `SystemMessage` | the conversation, as data you own |
| `llm.bind_tools([...])` | a model that is allowed to answer with a **tool call** |
| `AIMessage.tool_calls` | the model's chosen action &mdash; structured, not parsed out of prose |
| `ToolMessage` | the result you hand back, tied to the call by `tool_call_id` |

Three things make the loop safe rather than merely clever: **state**, a **stop condition**, and
**loop detection**. The last two are what separate a demo from something you would run unattended.
"""),

    md("""
## Section 1 &mdash; State is a list of messages you resend

The model has no memory, so *you* carry the conversation. `carry()` builds the message list for
the next call: a system message, every earlier turn, then the new human message.

These are real `langchain_core` objects, not tuples &mdash; every later lab, and LangGraph itself,
passes exactly this list around.
"""),
    code('''
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

SYSTEM = ("You are a payments operations analyst. Answer only from the data you are given. "
          "If you do not have the data, say so.")

def carry(history: list, user_msg: str) -> list:
    """Build the message list for the next call.

    history: earlier message objects, oldest first.
    Returns: [SystemMessage, *history, HumanMessage(user_msg)]
    """
    msgs = [SystemMessage(SYSTEM)]
    for m in history:                 # the model gets the whole history back, every time
        msgs.append(m)
    msgs.append(HumanMessage(user_msg))
    return msgs
''', '''
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

SYSTEM = ("You are a payments operations analyst. Answer only from the data you are given. "
          "If you do not have the data, say so.")

def carry(history: list, user_msg: str) -> list:
    """Build the message list for the next call.

    history: earlier message objects, oldest first.
    Returns: [SystemMessage, *history, HumanMessage(user_msg)]
    """
    msgs = [SystemMessage(SYSTEM)]
    for m in history:                 # the model gets the whole history back, every time
        msgs.append(m)
    msgs.append(HumanMessage(user_msg))
    return msgs
'''),
    code('''
# --- Self-check: Section 1   (message objects only -- no model call)
h = [HumanMessage("The reference is PMT-1002."), AIMessage("Noted.")]

check("carry() replays every earlier turn",
      lambda: len(carry(h, "which reference?")) == 4)
check("carry() leads with the system message",
      lambda: carry(h, "x")[0].type == "system")
check("carry() preserves the fact from turn 1",
      lambda: any("PMT-1002" in str(m.content) for m in carry(h, "which reference?")),
      "the first turn must survive into the new call")
check("carry() puts the new human message last",
      lambda: carry(h, "which reference?")[-1].content == "which reference?")
check("the turns stay LangChain message objects",
      lambda: all(hasattr(m, "type") for m in carry(h, "x")),
      "append the message objects themselves, not their .content")
'''),

    md("""
## Section 2 &mdash; The loop: tool calls in, tool messages out

Now the real thing. `llm.bind_tools([...])` returns a model that may answer with an **action**
instead of prose. When it does, `response.tool_calls` is a list of
`{"name", "args", "id"}` &mdash; already structured. Your job in the loop is to run the named tool
and hand the result back as a `ToolMessage` carrying the same `id`.

Note what you are **not** doing: parsing "Action: lookup_payment" out of free text. Module 2
shows what that costs when you have to.
"""),
    code('''
from langchain_core.tools import tool

@tool
def lookup_payment(ref: str) -> str:
    """Return the ledger record for one payment reference such as 'PMT-1002'.

    Use when you need the status, amount, counterparty or reason code of a specific payment.
    Not for searching across payments.
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


TOOLS = {t.name: t for t in (lookup_payment, policy_for)}
print("tools:", list(TOOLS))
'''),
    code('''
MAX_STEPS = 6

def run_tool_calls(ai_message, tools: dict) -> list:
    """Execute every tool call on an AIMessage. Return the ToolMessages to send back.

    A tool that raises would abort the run, so failures are returned as text the model
    can reason about instead.
    """
    out = []
    for call in ai_message.tool_calls:
        try:
            result = tools[call["name"]].invoke(call["args"])
        except Exception as exc:
            result = f"tool error: {type(exc).__name__}: {exc}"
        out.append(ToolMessage(content=str(result), tool_call_id=BLANK))   # TODO: tie it to the call
    return out


def should_stop(messages: list, steps: int, max_steps: int = MAX_STEPS):
    """Return (stop, reason). Two ways a run ends: the goal is reached, or the budget is spent."""
    last = messages[-1]
    if getattr(last, "type", None) == "ai" and BLANK:   # TODO: an agent is DONE when the model does what?
        return True, "goal"
    if steps >= max_steps:             # the backstop: a hard number, not a hope
        return True, "budget"
    return False, None


def run_agent(question: str, decide, tools: dict, max_steps: int = MAX_STEPS) -> dict:
    """decide(messages) -> AIMessage, possibly carrying tool_calls. The loop is the agent."""
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]
    steps = 0
    while True:
        ai = decide(messages)
        messages.append(ai)
        stop, why = should_stop(messages, steps, max_steps)
        if stop:
            return {"messages": messages, "steps": steps, "stopped": why}
        messages.extend(run_tool_calls(ai, tools))
        steps += 1
''', '''
MAX_STEPS = 6

def run_tool_calls(ai_message, tools: dict) -> list:
    """Execute every tool call on an AIMessage. Return the ToolMessages to send back.

    A tool that raises would abort the run, so failures are returned as text the model
    can reason about instead.
    """
    out = []
    for call in ai_message.tool_calls:
        try:
            result = tools[call["name"]].invoke(call["args"])
        except Exception as exc:
            result = f"tool error: {type(exc).__name__}: {exc}"
        out.append(ToolMessage(content=str(result), tool_call_id=call["id"]))  # the id pairs them
    return out


def should_stop(messages: list, steps: int, max_steps: int = MAX_STEPS):
    """Return (stop, reason). Two ways a run ends: the goal is reached, or the budget is spent."""
    last = messages[-1]
    if getattr(last, "type", None) == "ai" and not last.tool_calls:   # answering, not asking
        return True, "goal"
    if steps >= max_steps:             # a hard number, not a hope
        return True, "budget"
    return False, None


def run_agent(question: str, decide, tools: dict, max_steps: int = MAX_STEPS) -> dict:
    """decide(messages) -> AIMessage, possibly carrying tool_calls. The loop is the agent."""
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]
    steps = 0
    while True:
        ai = decide(messages)
        messages.append(ai)
        stop, why = should_stop(messages, steps, max_steps)
        if stop:
            return {"messages": messages, "steps": steps, "stopped": why}
        messages.extend(run_tool_calls(ai, tools))
        steps += 1
'''),
    code('''
# --- Self-check: Section 2   (a scripted `decide` returns real AIMessages -- no model involved)
def _call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid, "type": "tool_call"}])

def _finisher(messages):
    if sum(1 for m in messages if m.type == "tool") >= 2:
        return AIMessage("PMT-1002 failed: INSUFFICIENT_FUNDS. Retry once after 24h.")
    if not any(m.type == "tool" for m in messages):
        return _call("lookup_payment", {"ref": "PMT-1002"}, "c1")
    return _call("policy_for", {"reason_code": "INSUFFICIENT_FUNDS"}, "c2")

def _never_finishes(messages):
    return _call("lookup_payment", {"ref": "PMT-1002"}, f"c{len(messages)}")

check("a ToolMessage is produced per tool call",
      lambda: len(run_tool_calls(_call("lookup_payment", {"ref": "PMT-1002"}, "c1"), TOOLS)) == 1)
check("the ToolMessage carries the call's id",
      lambda: run_tool_calls(_call("lookup_payment", {"ref": "PMT-1002"}, "c9"), TOOLS)[0].tool_call_id == "c9",
      "tool_call_id must be call['id'] -- the model pairs result to request by that id")
check("the ToolMessage carries the tool's real output",
      lambda: "INSUFFICIENT_FUNDS" in run_tool_calls(
          _call("lookup_payment", {"ref": "PMT-1002"}, "c1"), TOOLS)[0].content)
check("an unknown reference does not raise",
      lambda: "no payment found" in run_tool_calls(
          _call("lookup_payment", {"ref": "PMT-9999"}, "c1"), TOOLS)[0].content,
      "a raising tool aborts the whole agent run")
check("a run that answers without a tool call stops with reason 'goal'",
      lambda: run_agent("q", _finisher, TOOLS)["stopped"] == "goal",
      "an agent is finished when the model replies WITHOUT asking for a tool")
check("a run that never finishes stops on the budget",
      lambda: run_agent("q", _never_finishes, TOOLS)["stopped"] == "budget",
      "the budget is given -- this fails while the goal test above it is unfilled")
check("the budget is actually respected",
      lambda: run_agent("q", _never_finishes, TOOLS)["steps"] == MAX_STEPS)
'''),

    md("""
## Section 3 &mdash; Loop detection

A budget stops a runaway agent *eventually*. Loop detection stops it **as soon as it stops
learning** &mdash; the same tool, the same arguments, no new information. In production this is
usually the difference between a cheap failure and an expensive one.

Because `tool_calls` is structured, you can detect this exactly, without any string matching.
"""),
    code('''
def is_looping(messages: list, window: int = 3) -> bool:
    """True when the last `window` tool calls are identical in both name and arguments."""
    calls = [(c["name"], json.dumps(c["args"], sort_keys=True))
             for m in messages if getattr(m, "type", None) == "ai"
             for c in (m.tool_calls or [])]
    if len(calls) < window:
        return False
    return len(set(calls[-window:])) == 1        # one distinct call across the window
''', '''
def is_looping(messages: list, window: int = 3) -> bool:
    """True when the last `window` tool calls are identical in both name and arguments."""
    calls = [(c["name"], json.dumps(c["args"], sort_keys=True))
             for m in messages if getattr(m, "type", None) == "ai"
             for c in (m.tool_calls or [])]
    if len(calls) < window:
        return False
    return len(set(calls[-window:])) == 1        # one distinct call across the window
'''),
    code('''
# --- Self-check: Section 3
_same  = [_call("lookup_payment", {"ref": "PMT-1002"}, f"c{i}") for i in range(3)]
_mixed = [_call("lookup_payment", {"ref": "PMT-1002"}, "c1"),
          _call("lookup_payment", {"ref": "PMT-1003"}, "c2"),
          _call("lookup_payment", {"ref": "PMT-1002"}, "c3")]

check("three identical calls count as a loop", lambda: is_looping(_same) is True)
check("varied calls are not a loop", lambda: is_looping(_mixed) is False,
      "different arguments mean the agent is still learning something")
check("too short a trace is not yet a loop", lambda: is_looping(_same[:2]) is False)
check("the window is honoured", lambda: is_looping(_same, window=2) is True)
check("the id is ignored -- only name and args identify a call",
      lambda: is_looping(_same) is True,
      "each of those has a different id but the same call")
'''),

    md("""
## Run it for real &mdash; part 1: statelessness

Two calls, where the second depends on the first. Watch the model fail to recall &mdash; then watch
`carry()` fix it, by resending what it already told you.
"""),
    code('''
if llm_ready():
    llm = get_llm()
    print("--- without carry() -------------------------------------------")
    print("call 1:", llm.invoke([HumanMessage("Remember this reference: PMT-1002. Reply with just OK.")]).content)
    print("call 2:", llm.invoke([HumanMessage("Which payment reference did I just give you?")]).content[:160])

    print("\\n--- with carry() ----------------------------------------------")
    def _carried():
        history = [HumanMessage("Remember this reference: PMT-1002."), AIMessage("OK")]
        reply = llm.invoke(carry(history, "Which payment reference did I just give you?"))
        print("call 2:", reply.content[:160])
    guard(_carried)
'''),

    md("""
## Run it for real &mdash; part 2: your loop, driving a real model

`decide` is now the model itself, bound to your two tools. Everything else is the loop you wrote.
Watch the trace: the model asks for the ledger record, you answer, it asks for the policy, you
answer, and only then does it stop.
"""),
    code('''
if llm_ready():
    def _live_loop():
        model = get_llm().bind_tools(list(TOOLS.values()))
        result = run_agent("Why is PMT-1003 held, and what must we do about it?",
                           lambda msgs: model.invoke(msgs), TOOLS)
        show_messages(result["messages"])
        print(f"\\nstopped: {result['stopped']}   steps: {result['steps']}")
        print("looping:", is_looping(result["messages"]))
    guard(_live_loop)
'''),

    md("""
## Run it for real &mdash; part 3: the same agent, in one line

Everything you just wrote &mdash; the loop, the tool dispatch, the `ToolMessage` plumbing, the stop
condition &mdash; is what `create_agent` gives you. Add a **checkpointer** and a `thread_id` and the
message history is kept for you too, which is the whole of Section 1 handled.

Run it and ask a follow-up question that only makes sense if the agent remembered.
"""),
    code('''
if llm_ready():
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver

    agent = create_agent(model=get_llm(), tools=list(TOOLS.values()), system_prompt=SYSTEM,
                         checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "case-1003"}}

    first = agent.invoke({"messages": [HumanMessage("Why is PMT-1003 held?")]}, cfg)
    show_messages(first["messages"])

    follow = agent.invoke({"messages": [HumanMessage("What was the amount again?")]}, cfg)
    print("\\nfollow-up:", follow["messages"][-1].content[:200])
    print(f"messages on this thread: {len(follow['messages'])}")
'''),
    md("""
### Read it

Three things to take away, in order of how much they will cost you later.

1. **You own the state.** `carry()` resends the entire history on *every* turn &mdash; the model
   keeps nothing. Everything an agent "remembers" is something your code chose to put back in
   front of it, which is why Module 3 is about deciding what to keep.
2. **Structured beats parsed.** The model chose its tools through `tool_calls`, so there was no
   format to get wrong. Module 2 shows the same loop without that guarantee.
3. **`create_agent` is the loop you wrote.** Not a different thing &mdash; the same thing, with the
   stop condition, dispatch and history handled. You now know what it is hiding, which is the
   only safe way to use it.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `run_agent` ignores `is_looping`. Wire it in as a third stop reason and give it its own
   label in the return value. Which of the four stop conditions from the slides does that
   still leave unimplemented?
2. Give `create_agent` a **third** tool that overlaps with `lookup_payment` &mdash; say
   `get_payment_status(ref)` &mdash; and see which one it picks. Lab 1.3 turns that into a measurement.
3. Run part 3 again with a second `thread_id`, asking the follow-up question first. Confirm for
   yourself that threads do not leak into each other, then look at where that isolation actually
   lives.
"""),
]


# =========================================================================== #
# Lab 1.2 -- the four building blocks, each one a real LangChain object
# =========================================================================== #
CARRY_1_1 = '''
# ------------------------------------------------- carried forward from Lab 1.1
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

SYSTEM = ("You are a payments operations analyst. Answer only from the data you are given. "
          "If you do not have the data, say so.")

def run_tool_calls(ai_message, tools: dict) -> list:
    out = []
    for call in ai_message.tool_calls:
        try:
            result = tools[call["name"]].invoke(call["args"])
        except Exception as exc:
            result = f"tool error: {type(exc).__name__}: {exc}"
        out.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return out

print("Lab 1.1 helpers loaded")
'''

LAB2 = [
    header(2, "The Four Building Blocks", "Intermediate", 40,
           ["Write tools with a real argument schema, and docstrings the model actually reads",
            "Bound memory with `trim_messages` &mdash; including the token counter that breaks on this model",
            "Turn a goal into a dependency-ordered plan with `with_structured_output` and Pydantic",
            "Assemble all four blocks into one agent over the case file"],
           "> **Builds on Lab 1.1.** The loop you wrote there is the fourth block; here you build\n"
           "> the other three as first-class LangChain objects and wire them together."),
    setup(2),
    code(DOMAIN),
    code(CARRY_1_1),

    md("""
## Concept

Each block patches one thing a model cannot do on its own &mdash; and each has a LangChain object
that *is* that block:

| Block | The gap it closes | The object |
|---|---|---|
| **LLM** | judgement under ambiguity | `ChatOpenAI` |
| **Memory** | the call is stateless | a message list + `trim_messages` |
| **Tools** | the model cannot read or change anything | `@tool` / `StructuredTool` |
| **Planning** | a goal is not a sequence of steps | `with_structured_output(...)` |

Miss one and you have a pipeline with a model in it &mdash; often the right build, but not an agent.
"""),

    md("""
## Section 1 &mdash; Tools: the docstring is the instruction

`@tool` turns a function into something the model can be offered. Three parts of it are read by
the model and by nothing else:

- the **name** &mdash; taken from the function name,
- the **description** &mdash; taken from the docstring,
- the **argument schema** &mdash; inferred from your type hints, or given explicitly with Pydantic.

Get the docstring wrong and the model picks the wrong tool. Lab 1.3 measures exactly that. Here,
write one that says both what the tool is for *and* what it is not for.
"""),
    code('''
from pydantic import BaseModel, Field

class ReleaseArgs(BaseModel):
    """Arguments for release_payment."""
    ref: str = Field(description="The payment reference, e.g. 'PMT-1003'")
    approved_by: str = Field(description="Name of the human who approved the release")


@tool(args_schema=ReleaseArgs)
def release_payment(ref: str, approved_by: str) -> str:
    """BLANK"""                       # TODO: one line on what this releases, and when NOT to use it
    record = LEDGER.get(ref)
    if record is None:
        return f"no payment found with reference {ref!r}"
    if record["reason_code"] in NEEDS_HUMAN and not approved_by:
        return f"refused: {record['reason_code']} needs a named human approver"
    return f"released {ref} on the authority of {approved_by}"


@tool
def lookup_payment(ref: str) -> str:
    """Return the ledger record for one payment reference such as 'PMT-1002'.

    Use when you need the status, amount, counterparty or reason code of a specific payment.
    Not for searching across payments.
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


TOOLS = {t.name: t for t in (lookup_payment, policy_for, release_payment)}
''', '''
from pydantic import BaseModel, Field

class ReleaseArgs(BaseModel):
    """Arguments for release_payment."""
    ref: str = Field(description="The payment reference, e.g. 'PMT-1003'")
    approved_by: str = Field(description="Name of the human who approved the release")


@tool(args_schema=ReleaseArgs)
def release_payment(ref: str, approved_by: str) -> str:
    """Release one held payment for settlement, on a named human's authority.

    Use only after a human has approved the release. Never use it to clear a sanctions
    hold, and never invent an approver.
    """
    record = LEDGER.get(ref)
    if record is None:
        return f"no payment found with reference {ref!r}"
    if record["reason_code"] in NEEDS_HUMAN and not approved_by:
        return f"refused: {record['reason_code']} needs a named human approver"
    return f"released {ref} on the authority of {approved_by}"


@tool
def lookup_payment(ref: str) -> str:
    """Return the ledger record for one payment reference such as 'PMT-1002'.

    Use when you need the status, amount, counterparty or reason code of a specific payment.
    Not for searching across payments.
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


TOOLS = {t.name: t for t in (lookup_payment, policy_for, release_payment)}
'''),
    code('''
# --- Self-check: Section 1   (inspects the tool objects -- no model call)
def _doc(name):
    d = (TOOLS[name].description or "").strip()
    if d == "BLANK":
        raise NameError("release_payment still has the placeholder docstring")
    return d

check("@tool takes its name from the function",
      lambda: TOOLS["lookup_payment"].name == "lookup_payment")
check("the schema was inferred from the type hints",
      lambda: "ref" in TOOLS["lookup_payment"].args)
check("release_payment declares both arguments",
      lambda: set(TOOLS["release_payment"].args) == {"ref", "approved_by"})
check("the Field descriptions reached the schema",
      lambda: "approver" in json.dumps(TOOLS["release_payment"].args).lower()
              or "approved" in json.dumps(TOOLS["release_payment"].args).lower())
check("release_payment has a real description",
      lambda: len(_doc("release_payment")) > 40,
      "the docstring is the only thing the model reads when choosing this tool")
check("the description says when NOT to use it",
      lambda: any(w in _doc("release_payment").lower() for w in ("never", "not ", "only")),
      "a tool that only says what it does gets called when it should not be")
check("a tool invoked with a dict returns its result",
      lambda: "INSUFFICIENT_FUNDS" in TOOLS["lookup_payment"].invoke({"ref": "PMT-1002"}))
check("an unknown reference does NOT raise",
      lambda: "no payment found" in TOOLS["lookup_payment"].invoke({"ref": "PMT-9999"}),
      "a raising tool aborts the whole agent run")
'''),

    md("""
## Section 2 &mdash; Memory: `trim_messages`, and the counter that breaks

Unbounded history is the classic failure: fine in the demo, degraded and expensive by week two.
LangChain's `trim_messages` bounds it for you &mdash; but it needs to know how to count tokens, and
**the obvious answer does not work here**:

```python
trim_messages(msgs, max_tokens=200, token_counter=llm)   # NotImplementedError on this model
```

`token_counter=llm` asks the model class to count, and `langchain-openai` only knows how to do
that for models `tiktoken` has an encoding for. `qwen36-35b-a3b-lab` is not one. The fix is
`count_tokens_approximately`, which is a plain function over the message text.

This is not a quirk of our sandbox &mdash; it is what happens to every self-hosted or gateway-served
model, and it is the kind of thing that only shows up under load.
"""),
    code('''
from langchain_core.messages.utils import count_tokens_approximately

def bounded(messages: list, max_tokens: int = 120) -> list:
    """Keep the system message and as many recent turns as fit inside `max_tokens`."""
    from langchain_core.messages import trim_messages
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        token_counter=BLANK,          # TODO: which counter works for a model tiktoken cannot see?
        strategy="last",              # keep the END of the conversation, not the start
        include_system=True,          # never drop the instructions
        start_on="human",             # a valid history starts on a human turn
        allow_partial=False,
    )
''', '''
from langchain_core.messages.utils import count_tokens_approximately

def bounded(messages: list, max_tokens: int = 120) -> list:
    """Keep the system message and as many recent turns as fit inside `max_tokens`."""
    from langchain_core.messages import trim_messages
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        token_counter=count_tokens_approximately,   # a plain function over the text
        strategy="last",              # keep the END of the conversation, not the start
        include_system=True,          # never drop the instructions
        start_on="human",             # a valid history starts on a human turn
        allow_partial=False,
    )
'''),
    code('''
# --- Self-check: Section 2   (trim_messages is pure -- no model call)
def _long_history():
    msgs = [SystemMessage(SYSTEM), HumanMessage("Investigate PMT-1003, it is held.")]
    for i in range(12):
        msgs.append(AIMessage(f"step {i}: " + "checking the ledger. " * 12))
        msgs.append(HumanMessage(f"and then? ({i})"))
    return msgs

check("the history is bounded",
      lambda: count_tokens_approximately(bounded(_long_history())) <= 130,
      "trim_messages needs a token_counter it can actually call on this model")
check("trimming actually dropped turns",
      lambda: len(bounded(_long_history())) < len(_long_history()))
check("the system message survives",
      lambda: bounded(_long_history())[0].type == "system",
      "include_system=True -- dropping the instructions is the worst possible trim")
check("what survives is the END of the conversation",
      lambda: bounded(_long_history())[-1].content == _long_history()[-1].content,
      'strategy="last" keeps recent turns; "first" would keep the stale ones')
check("a short conversation is left untouched",
      lambda: len(bounded([SystemMessage(SYSTEM), HumanMessage("hi")])) == 2)
'''),

    md("""
## Section 3 &mdash; Planning: a goal is not a sequence

Decomposition is only half of it. The steps have **dependencies**, and running them out of order
is one of the quieter ways an agent wastes a budget.

`with_structured_output(Plan)` makes the model return a **`Plan` object**, not prose that you then
have to parse. You declare the shape; LangChain gives the model the schema and validates what
comes back. Define the schema first, then write the ordering.
"""),
    code('''
from typing import List

class Step(BaseModel):
    """One step of an investigation plan."""
    # These descriptions are not documentation. with_structured_output sends them to the
    # model AS THE SCHEMA -- they are the only instruction it gets about what goes here.
    name: str = Field(description="BLANK")          # TODO: what must a step name look like?
    depends_on: List[str] = Field(default_factory=list,
                                  description="BLANK")  # TODO: names of WHAT? say it precisely
    tool: str = Field(description="Which tool this step calls, or 'none'")


class Plan(BaseModel):
    """An ordered investigation plan for one payment exception."""
    goal: str = Field(description="The question the plan answers, in one line")
    steps: List[Step] = Field(description="The steps, which may be given in any order")


def order_steps(plan: Plan) -> list[str]:
    """Return a runnable order for plan.steps, respecting depends_on.

    Raises ValueError if the dependencies cannot be satisfied (a cycle, or a missing step).
    """
    deps = {s.name: list(s.depends_on) for s in plan.steps}
    ordered: list[str] = []
    done: set[str] = set()
    while len(ordered) < len(deps):
        progressed = False
        for name, d in deps.items():
            if name in done:
                continue
            if all(x in done for x in d):        # every dependency already ordered
                ordered.append(name)
                done.add(name)
                progressed = True
        if not progressed:
            raise ValueError("cycle or missing dependency in plan")
    return ordered
''', '''
from typing import List

class Step(BaseModel):
    """One step of an investigation plan."""
    name: str = Field(description="Short snake_case name for this step")
    depends_on: List[str] = Field(default_factory=list,
                                  description="Names of steps that must finish before this one")
    tool: str = Field(description="Which tool this step calls, or 'none'")


class Plan(BaseModel):
    """An ordered investigation plan for one payment exception."""
    goal: str = Field(description="The question the plan answers, in one line")
    steps: List[Step] = Field(description="The steps, which may be given in any order")


def order_steps(plan: Plan) -> list[str]:
    """Return a runnable order for plan.steps, respecting depends_on.

    Raises ValueError if the dependencies cannot be satisfied (a cycle, or a missing step).
    """
    deps = {s.name: list(s.depends_on) for s in plan.steps}
    ordered: list[str] = []
    done: set[str] = set()
    while len(ordered) < len(deps):
        progressed = False
        for name, d in deps.items():
            if name in done:
                continue
            if all(x in done for x in d):        # every dependency already ordered
                ordered.append(name)
                done.add(name)
                progressed = True
        if not progressed:
            raise ValueError("cycle or missing dependency in plan")
    return ordered
'''),
    code('''
# --- Self-check: Section 3   (Pydantic objects only -- no model call)
HAND_PLAN = Plan(goal="Decide what to do about PMT-1003", steps=[
    Step(name="decide_action", depends_on=["read_payment", "read_policy"], tool="none"),
    Step(name="read_policy",   depends_on=["read_payment"], tool="policy_for"),
    Step(name="read_payment",  depends_on=[], tool="lookup_payment"),
    Step(name="write_note",    depends_on=["decide_action"], tool="none"),
])

def _cycles():
    try:
        order_steps(Plan(goal="g", steps=[Step(name="a", depends_on=["b"], tool="none"),
                                          Step(name="b", depends_on=["a"], tool="none")]))
        return False
    except ValueError:
        return True

check("the schema declares a goal and steps",
      lambda: set(Plan.model_fields) == {"goal", "steps"})
def _desc(field: str) -> str:
    d = (Step.model_fields[field].description or "").strip()
    if d == "BLANK":
        raise NameError(f"{field} still has the placeholder description")
    return d

check("every step field carries a description the model can read",
      lambda: all(_desc(f) for f in Step.model_fields),
      "with_structured_output sends these descriptions to the model as the schema")
check("depends_on says what the names REFER to",
      lambda: "step" in _desc("depends_on").lower(),
      "the model has to know these are other steps' names, not tool names")
check("dependencies come before dependants",
      lambda: order_steps(HAND_PLAN).index("read_payment") < order_steps(HAND_PLAN).index("read_policy"))
check("every step is scheduled exactly once",
      lambda: sorted(order_steps(HAND_PLAN)) == sorted(s.name for s in HAND_PLAN.steps))
check("a plan given out of order is still ordered correctly",
      lambda: order_steps(HAND_PLAN)[0] == "read_payment")
check("an impossible plan raises rather than half-running", _cycles)
'''),

    md("""
## Section 4 &mdash; Assemble the four blocks

One object now holds all four: the model, the bound tools, the bounded history, and a plan.
`answer()` runs the loop until the model replies without asking for a tool.
"""),
    code('''
class MiniAgent:
    """LLM + Memory + Tools + Planning, assembled by hand. `create_agent` is this, hardened."""

    def __init__(self, tools: dict, max_tokens: int = 600, max_steps: int = 6):
        self.tools = tools
        self.max_tokens, self.max_steps = max_tokens, max_steps
        self.history = [SystemMessage(SYSTEM)]

    def tool_list(self) -> list:
        """The tool OBJECTS this agent may call -- bind_tools() needs the objects, not names."""
        return BLANK                            # TODO: which tools may this agent call?

    @property
    def model(self):
        """The model, bound to this agent's tools so it can answer with a tool call."""
        return get_llm().bind_tools(self.tool_list())

    def plan(self, goal: str) -> Plan:
        """Ask the model for a Plan object -- not prose about a plan."""
        return get_llm().with_structured_output(Plan).invoke(
            "Produce an investigation plan for this goal. Steps must name their dependencies. "
            f"Available tools: {list(self.tools)}.\\n\\nGOAL: {goal}")

    def answer(self, question: str) -> str:
        self.history.append(HumanMessage(question))
        for _ in range(self.max_steps):
            self.history = bounded(self.history, self.max_tokens)
            ai = self.model.invoke(self.history)
            self.history.append(ai)
            if not ai.tool_calls:
                return ai.content
            self.history.extend(run_tool_calls(ai, self.tools))
        return "(step budget spent)"
''', '''
class MiniAgent:
    """LLM + Memory + Tools + Planning, assembled by hand. `create_agent` is this, hardened."""

    def __init__(self, tools: dict, max_tokens: int = 600, max_steps: int = 6):
        self.tools = tools
        self.max_tokens, self.max_steps = max_tokens, max_steps
        self.history = [SystemMessage(SYSTEM)]

    def tool_list(self) -> list:
        """The tool OBJECTS this agent may call -- bind_tools() needs the objects, not names."""
        return list(self.tools.values())

    @property
    def model(self):
        """The model, bound to this agent's tools so it can answer with a tool call."""
        return get_llm().bind_tools(self.tool_list())

    def plan(self, goal: str) -> Plan:
        """Ask the model for a Plan object -- not prose about a plan."""
        return get_llm().with_structured_output(Plan).invoke(
            "Produce an investigation plan for this goal. Steps must name their dependencies. "
            f"Available tools: {list(self.tools)}.\\n\\nGOAL: {goal}")

    def answer(self, question: str) -> str:
        self.history.append(HumanMessage(question))
        for _ in range(self.max_steps):
            self.history = bounded(self.history, self.max_tokens)
            ai = self.model.invoke(self.history)
            self.history.append(ai)
            if not ai.tool_calls:
                return ai.content
            self.history.extend(run_tool_calls(ai, self.tools))
        return "(step budget spent)"
'''),
    code('''
# --- Self-check: Section 4   (structure only -- .model builds an object, it does not call out)
check("MiniAgent starts with just the system message",
      lambda: [m.type for m in MiniAgent(TOOLS).history] == ["system"])
check("all three tools were handed to the agent",
      lambda: set(MiniAgent(TOOLS).tools) == {"lookup_payment", "policy_for", "release_payment"})
check("the agent hands bind_tools the tool objects, not their names",
      lambda: all(hasattr(t, "invoke") and hasattr(t, "name") for t in MiniAgent(TOOLS).tool_list()),
      "bind_tools() needs the @tool objects -- a list of strings binds nothing")
check("all three tools are offered to the model",
      lambda: {t.name for t in MiniAgent(TOOLS).tool_list()}
              == {"lookup_payment", "policy_for", "release_payment"})
'''),

    md("""
## Run it for real

First a plan as a typed object, then the assembled agent answering a real question.
"""),
    code('''
if llm_ready():
    def _plan():
        agent = MiniAgent(TOOLS)
        p = agent.plan("Decide what to do about PMT-1003, which is held.")
        print("goal:", p.goal)
        for s in p.steps:
            print(f"  {s.name:16} tool={s.tool:16} after={s.depends_on}")
        print("\\nrunnable order:", order_steps(p))
        return agent
    agent = guard(_plan)
'''),
    code('''
if llm_ready() and agent is not None:
    def _answer():
        print(agent.answer("Why is PMT-1003 held, and what must we do about it?")[:400])
        print("\\n--- history after the run ---")
        show_messages(agent.history)
    guard(_answer)
'''),
    md("""
### Read it

The plan came back as a `Plan`, so `order_steps` could run over it directly &mdash; no parsing, no
"the model used a different heading this time". That is the whole argument for structured output,
and Module 2 shows what the alternative costs.

Watch the history: it is bounded, so a long investigation cannot grow the bill without limit. And
notice which tool the model did *not* reach for. `release_payment` says "never use it to clear a
sanctions hold" in its docstring, and that sentence is the only control that stopped it.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Delete the second paragraph of `release_payment`'s docstring, re-run, and ask the agent to
   release PMT-1005. Put the sentence back once you have seen what happens.
2. `bounded()` uses `strategy="last"`. Switch it to `"first"` and ask a follow-up question.
   Explain, in one line, why keeping the *start* of a conversation is almost always wrong for an
   agent and almost always right for a chat product.
3. `MiniAgent.plan` never uses the plan &mdash; `answer()` just loops. Make `answer()` follow the
   ordered plan instead, and note the first thing that breaks.
"""),
]


# =========================================================================== #
# Lab 1.3 -- create_agent: the description experiment, and a typed answer
# =========================================================================== #
LAB3 = [
    header(3, "create_agent, and What a Tool Description Is Worth", "Intermediate &rarr; Advanced", 40,
           ["Build the same agent twice &mdash; once with opaque tool descriptions, once with good ones",
            "<b>Measure</b> the difference in tool-selection accuracy against the live model",
            "Make the agent return a typed <code>Verdict</code> with <code>response_format</code>, not prose",
            "Read the message trace `create_agent` produces, and find where a run went wrong"],
           "> **Builds on Labs 1.1 and 1.2.** You know what the loop does; now you use the built one\n"
           "> and spend your effort on the part that actually decides whether it works."),
    setup(3),
    code(DOMAIN),

    md("""
## Concept

`create_agent(model, tools, system_prompt)` is the loop from Lab 1.1, hardened. Which means the
interesting engineering moves somewhere else &mdash; to the three things you still control:

1. **the tool descriptions**, which are the only guide the model has when choosing,
2. **the system prompt**, which sets the procedure,
3. **the output contract**, which decides whether the caller gets prose or data.

This lab measures the first and fixes the third. Everything here is a real agent making real
calls, so the numbers you get are yours, not slides.
"""),

    md("""
## Section 1 &mdash; The same five tools, described two ways

Both toolsets do **exactly the same work**. Only the names and descriptions differ. That is the
experiment: nothing changes except what the model can read.

Write the four missing descriptions in `GOOD`. A good one says what the tool returns, when to
reach for it, and when not to.
"""),
    code('''
from langchain_core.tools import tool, StructuredTool

# The five underlying operations, as plain functions. Shared by both arms.
def _payment(ref: str) -> str:
    r = LEDGER.get(ref)
    return json.dumps({"ref": ref, **r}) if r else f"no payment found with reference {ref!r}"

def _policy(reason_code: str) -> str:
    return POLICY.get(reason_code, f"no policy on file for reason code {reason_code!r}")

def _counterparty(name: str) -> str:
    hits = [k for k, v in LEDGER.items() if v["counterparty"] == name]
    return json.dumps({"counterparty": name, "payments": hits}) if hits else f"no counterparty {name!r}"

def _by_status(status: str) -> str:
    hits = [k for k, v in LEDGER.items() if v["status"] == status]
    return json.dumps({"status": status, "payments": hits})

def _needs_human(reason_code: str) -> str:
    return json.dumps({"reason_code": reason_code, "needs_human": reason_code in NEEDS_HUMAN})

OPS = {"payment": _payment, "policy": _policy, "counterparty": _counterparty,
       "by_status": _by_status, "needs_human": _needs_human}

# --- arm A: opaque. A name and a shrug -- what a rushed codebase actually looks like.
POOR = [
    StructuredTool.from_function(_payment,      name="tool_a", description="Gets data."),
    StructuredTool.from_function(_policy,       name="tool_b", description="Gets data."),
    StructuredTool.from_function(_counterparty, name="tool_c", description="Looks things up."),
    StructuredTool.from_function(_by_status,    name="tool_d", description="Looks things up."),
    StructuredTool.from_function(_needs_human,  name="tool_e", description="Checks something."),
]

# --- arm B: described. Same functions, same order.
GOOD = [
    StructuredTool.from_function(
        _payment, name="lookup_payment",
        description="Return the full ledger record (amount, currency, counterparty, status, "
                    "reason code) for ONE payment reference such as 'PMT-1003'. Use when you "
                    "have a reference. Not for searching."),
    StructuredTool.from_function(
        _policy, name="policy_for",
        description="BLANK"),         # TODO: what does this return, and when would you reach for it?
    StructuredTool.from_function(
        _counterparty, name="payments_for_counterparty",
        description="BLANK"),         # TODO: ...and how is it different from lookup_payment?
    StructuredTool.from_function(
        _by_status, name="payments_by_status",
        description="BLANK"),         # TODO: name the valid statuses -- the model cannot guess them
    StructuredTool.from_function(
        _needs_human, name="requires_human_approval",
        description="BLANK"),         # TODO: say what a True answer obliges the caller to do
]
''', '''
from langchain_core.tools import tool, StructuredTool

# The five underlying operations, as plain functions. Shared by both arms.
def _payment(ref: str) -> str:
    r = LEDGER.get(ref)
    return json.dumps({"ref": ref, **r}) if r else f"no payment found with reference {ref!r}"

def _policy(reason_code: str) -> str:
    return POLICY.get(reason_code, f"no policy on file for reason code {reason_code!r}")

def _counterparty(name: str) -> str:
    hits = [k for k, v in LEDGER.items() if v["counterparty"] == name]
    return json.dumps({"counterparty": name, "payments": hits}) if hits else f"no counterparty {name!r}"

def _by_status(status: str) -> str:
    hits = [k for k, v in LEDGER.items() if v["status"] == status]
    return json.dumps({"status": status, "payments": hits})

def _needs_human(reason_code: str) -> str:
    return json.dumps({"reason_code": reason_code, "needs_human": reason_code in NEEDS_HUMAN})

OPS = {"payment": _payment, "policy": _policy, "counterparty": _counterparty,
       "by_status": _by_status, "needs_human": _needs_human}

# --- arm A: opaque. A name and a shrug -- what a rushed codebase actually looks like.
POOR = [
    StructuredTool.from_function(_payment,      name="tool_a", description="Gets data."),
    StructuredTool.from_function(_policy,       name="tool_b", description="Gets data."),
    StructuredTool.from_function(_counterparty, name="tool_c", description="Looks things up."),
    StructuredTool.from_function(_by_status,    name="tool_d", description="Looks things up."),
    StructuredTool.from_function(_needs_human,  name="tool_e", description="Checks something."),
]

# --- arm B: described. Same functions, same order.
GOOD = [
    StructuredTool.from_function(
        _payment, name="lookup_payment",
        description="Return the full ledger record (amount, currency, counterparty, status, "
                    "reason code) for ONE payment reference such as 'PMT-1003'. Use when you "
                    "have a reference. Not for searching."),
    StructuredTool.from_function(
        _policy, name="policy_for",
        description="Return the operating policy text for ONE failure reason code such as "
                    "'LIMIT_BREACH' or 'SANCTIONS_REVIEW'. Use after you know why a payment "
                    "failed and need to know what to do about it. Not for looking up payments."),
    StructuredTool.from_function(
        _counterparty, name="payments_for_counterparty",
        description="Return every payment reference belonging to ONE counterparty name such as "
                    "'NORTHWIND'. Use when the question names a party rather than a reference. "
                    "Returns references only -- call lookup_payment for the details of each."),
    StructuredTool.from_function(
        _by_status, name="payments_by_status",
        description="Return every payment reference with a given status. Valid statuses are "
                    "exactly 'settled', 'failed' and 'held'. Use for questions about a whole "
                    "queue, such as what is currently held."),
    StructuredTool.from_function(
        _needs_human, name="requires_human_approval",
        description="Return whether a reason code obliges a human decision before any action. "
                    "Call it before proposing to release, cancel or repair a payment; if it "
                    "returns true you must stop and escalate rather than act."),
]
'''),
    code('''
# --- Self-check: Section 1   (tool objects only -- no model call)
def _descs():
    out = []
    for t in GOOD:
        d = (t.description or "").strip()
        if d == "BLANK" or not d:
            raise NameError(f"{t.name} still has no description")
        out.append(d)
    return out

check("both arms expose five tools", lambda: len(POOR) == 5 and len(GOOD) == 5)
check("the two arms wrap the same functions",
      lambda: [t.func for t in POOR] == [t.func for t in GOOD],
      "the ONLY difference between the arms must be name and description")
check("every GOOD tool has a real description",
      lambda: all(len(d) > 60 for d in _descs()),
      "a one-liner is not a description -- say what it returns and when to use it")
check("the descriptions distinguish the two lookup tools",
      lambda: "not for searching" in _descs()[0].lower()
              and any("reference" in d.lower() for d in _descs()[2:3]))
check("payments_by_status names its valid values",
      lambda: all(s in _descs()[3] for s in ("settled", "failed", "held")),
      "the model cannot guess an enum it was never shown")
check("the POOR arm really is uninformative",
      lambda: all(len(t.description) < 25 for t in POOR))
'''),

    md("""
## Section 2 &mdash; The harness that scores a run

For each question we know which tool *should* be called first. `first_tool()` digs that out of
the trace `create_agent` returns; `bake_off()` runs a whole set and reports a pass rate.

This is your first eval harness. Day 2 measures a multi-agent graph against exactly this shape.
"""),
    code('''
CASES = [
    # question,                                                    poor name,  good name
    ("What is the status of PMT-1003?",                            "tool_a",  "lookup_payment"),
    ("What should we do about a LIMIT_BREACH?",                    "tool_b",  "policy_for"),
    ("Which payments belong to NORTHWIND?",                        "tool_c",  "payments_for_counterparty"),
    ("List everything currently held.",                            "tool_d",  "payments_by_status"),
    ("Does a SANCTIONS_REVIEW need a person to sign it off?",      "tool_e",  "requires_human_approval"),
]

def first_tool(result: dict) -> str | None:
    """The name of the FIRST tool the agent chose, from the messages it returned."""
    for m in result["messages"]:
        calls = getattr(m, "tool_calls", None)
        if calls:
            return BLANK              # TODO: a tool CHOICE lives on the AI message -- which part of it?
    return None


def bake_off(agent, arm: str) -> dict:
    """Run every case through `agent`; score the first tool chosen against the expectation."""
    idx = 1 if arm == "poor" else 2
    hits, rows = 0, []
    for case in CASES:
        question, expected = case[0], case[idx]
        try:
            chosen = first_tool(agent.invoke({"messages": [("human", question)]}))
        except Exception as exc:
            chosen = f"<error: {type(exc).__name__}>"
        ok = chosen == expected
        hits += ok
        rows.append((question, expected, chosen, ok))
    return {"arm": arm, "hits": hits, "of": len(CASES),
            "rate": hits / len(CASES), "rows": rows}
''', '''
CASES = [
    # question,                                                    poor name,  good name
    ("What is the status of PMT-1003?",                            "tool_a",  "lookup_payment"),
    ("What should we do about a LIMIT_BREACH?",                    "tool_b",  "policy_for"),
    ("Which payments belong to NORTHWIND?",                        "tool_c",  "payments_for_counterparty"),
    ("List everything currently held.",                            "tool_d",  "payments_by_status"),
    ("Does a SANCTIONS_REVIEW need a person to sign it off?",      "tool_e",  "requires_human_approval"),
]

def first_tool(result: dict) -> str | None:
    """The name of the FIRST tool the agent chose, from the messages it returned."""
    for m in result["messages"]:
        calls = getattr(m, "tool_calls", None)
        if calls:
            return calls[0]["name"]
    return None


def bake_off(agent, arm: str) -> dict:
    """Run every case through `agent`; score the first tool chosen against the expectation."""
    idx = 1 if arm == "poor" else 2
    hits, rows = 0, []
    for case in CASES:
        question, expected = case[0], case[idx]
        try:
            chosen = first_tool(agent.invoke({"messages": [("human", question)]}))
        except Exception as exc:
            chosen = f"<error: {type(exc).__name__}>"
        ok = chosen == expected
        hits += ok
        rows.append((question, expected, chosen, ok))
    return {"arm": arm, "hits": hits, "of": len(CASES),
            "rate": hits / len(CASES), "rows": rows}
'''),
    code('''
# --- Self-check: Section 2   (canned traces -- no model call)
from langchain_core.messages import AIMessage, HumanMessage

_canned = {"messages": [
    HumanMessage("q"),
    AIMessage(content="", tool_calls=[{"name": "lookup_payment", "args": {"ref": "PMT-1003"},
                                       "id": "c1", "type": "tool_call"}]),
    AIMessage("done"),
]}
_no_tools = {"messages": [HumanMessage("q"), AIMessage("answered from memory")]}

check("first_tool finds the first chosen tool",
      lambda: first_tool(_canned) == "lookup_payment")
check("first_tool returns None when no tool was used",
      lambda: first_tool(_no_tools) is None,
      "an agent that answers without a tool is a result, not a crash")
check("every case names a tool that exists in both arms",
      lambda: all(c[1] in {t.name for t in POOR} and c[2] in {t.name for t in GOOD} for c in CASES))
check("the cases cover all five tools",
      lambda: len({c[2] for c in CASES}) == 5)
'''),

    md("""
## Section 3 &mdash; A typed answer, not a paragraph

An agent that returns prose forces every caller to parse it. `response_format=Verdict` makes
`create_agent` return a validated `Verdict` object alongside the messages, in
`result["structured_response"]`.

Declare the contract you would want if you had to call this service from another system.
"""),
    code('''
from pydantic import BaseModel, Field

class Verdict(BaseModel):
    """The outcome of investigating one payment exception."""
    ref: str = Field(description="The payment reference investigated, e.g. 'PMT-1003'")
    reason_code: str = Field(description="The ledger reason code, or 'NONE' if the payment is fine")
    needs_human: bool = Field(description="BLANK")  # TODO: describe it so the model fills it correctly
    action: str = Field(description="The single next action, in one short line")
    evidence: str = Field(description="The policy text or ledger field that justifies the action")
''', '''
from pydantic import BaseModel, Field

class Verdict(BaseModel):
    """The outcome of investigating one payment exception."""
    ref: str = Field(description="The payment reference investigated, e.g. 'PMT-1003'")
    reason_code: str = Field(description="The ledger reason code, or 'NONE' if the payment is fine")
    needs_human: bool = Field(
        description="True if policy requires a named human to decide before any action is taken; "
                    "false only if the agent may act on its own authority")
    action: str = Field(description="The single next action, in one short line")
    evidence: str = Field(description="The policy text or ledger field that justifies the action")
'''),
    code('''
# --- Self-check: Section 3   (schema only -- no model call)
def _needs_human_desc():
    d = Verdict.model_fields["needs_human"].description
    if not d or d == "BLANK":
        raise NameError("needs_human still has no description")
    return d

check("the contract has all five fields",
      lambda: set(Verdict.model_fields) == {"ref", "reason_code", "needs_human", "action", "evidence"})
check("every field carries a description",
      lambda: all(f.description for f in Verdict.model_fields.values()),
      "with_structured_output sends these to the model -- an undescribed field is a guess")
check("needs_human says what true MEANS",
      lambda: len(_needs_human_desc()) > 40 and "human" in _needs_human_desc().lower())
check("a Verdict validates",
      lambda: Verdict(ref="PMT-1003", reason_code="LIMIT_BREACH", needs_human=True,
                      action="Escalate to Treasury", evidence="above USD 500,000").needs_human is True)
'''),

    md("""
## Run it for real &mdash; the description experiment

Two agents. Same model, same functions, same questions. Only the descriptions differ.

Ten agent runs, so give it a moment.
"""),
    code('''
if llm_ready():
    from langchain.agents import create_agent

    def _experiment():
        sysmsg = ("You are a payments operations analyst. Use exactly one tool to answer, "
                  "then reply. Do not guess if a tool can tell you.")
        results = {}
        for arm, tools in (("poor", POOR), ("good", GOOD)):
            ag = create_agent(model=get_llm(), tools=tools, system_prompt=sysmsg)
            r = bake_off(ag, arm)
            results[arm] = r
            print(f"\\n=== {arm.upper()} descriptions: {r['hits']}/{r['of']} correct "
                  f"({r['rate']:.0%}) ===")
            for q, expected, chosen, ok in r["rows"]:
                print(f"  [{'ok ' if ok else 'MISS'}] {q[:46]:48} want={expected:26} got={chosen}")
        d = results["good"]["rate"] - results["poor"]["rate"]
        print(f"\\nDelta from description quality alone: {d:+.0%}")
        return results
    RESULTS = guard(_experiment)
'''),
    md("""
### Read it

Nothing about the model, the questions or the underlying functions changed between those two
runs. Whatever gap you just measured is the value of writing a sentence.

Two things worth noticing in the misses. First, where the opaque arm guessed, it usually guessed
the *first* tool &mdash; with nothing to choose on, order becomes the tiebreak. Second, a miss is not
always a wrong answer: the agent sometimes recovers by calling a second tool, which costs tokens
and latency rather than correctness. That distinction is the whole subject of Module 7.
"""),

    md("""
## Run it for real &mdash; the typed answer

Now the same agent with a contract. Note that the caller never touches `.content`.
"""),
    code('''
if llm_ready():
    def _typed():
        ag = create_agent(
            model=get_llm(), tools=GOOD,
            system_prompt=("You investigate payment exceptions. Procedure, in order: "
                           "1) look up the payment; 2) look up the policy for its reason code; "
                           "3) check whether it requires human approval; 4) answer."),
            response_format=Verdict)
        out = ag.invoke({"messages": [("human", "Investigate PMT-1005 and tell me what to do.")]})
        v = out.get("structured_response")
        if v is None:
            # This happens, and it is worth seeing rather than hiding. response_format asks
            # the model to finish by calling a Verdict tool; if it answers in prose instead,
            # there is no object -- and a caller expecting one gets None, not an error.
            print("NO structured_response -- the model answered without filling the contract.")
            print("Look at the last message: it replied in prose instead of calling Verdict.\\n")
        else:
            print(f"ref         : {v.ref}")
            print(f"reason_code : {v.reason_code}")
            print(f"needs_human : {v.needs_human}")
            print(f"action      : {v.action}")
            print(f"evidence    : {v.evidence}")
        print(f"\\n--- the trace behind it ({len(out['messages'])} messages) ---")
        show_messages(out["messages"])
        return v
    VERDICT = guard(_typed)
'''),
    md("""
### Read the trace

`needs_human` should be **true** for PMT-1005 &mdash; it is a sanctions hold, and the policy says
Compliance decides. If it came back false, the trace tells you which step was skipped: look for
whether `requires_human_approval` was ever called at all.

And you may have got **no object at all**. `response_format` asks the model to finish by calling
a `Verdict` tool; a model that decides to reply in prose instead leaves
`result["structured_response"]` as `None`. Nothing raises. Run the cell a few times &mdash; on this
model it does not happen every time, which is worse than if it never worked.

That is the honest lesson of this lab, in two parts. A typed contract guarantees the *shape* of
the answer **when you get one**, so a caller must still handle its absence. And it guarantees
nothing at all about the truth of it &mdash; the system prompt's ordered procedure is doing that
work, and Module 8 is where you learn not to trust either without a check.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Build a third arm, `MEDIUM` &mdash; real names, but one-line descriptions with no "not for"
   clause. Where does it land between the two? That gap is the value of the negative half.
2. Add a sixth tool that genuinely overlaps with an existing one (`get_status(ref)` beside
   `lookup_payment`) and re-run. Which description do you have to change to fix the confusion?
3. Remove the numbered procedure from the system prompt in the typed run and re-run it five
   times. Count how often `needs_human` comes back wrong. That number is your argument for
   Module 8's guardrails.
"""),
]


# =========================================================================== #
# Lab 1.4 -- one agent or three, measured with real agents and real tokens
# =========================================================================== #
LAB4 = [
    header(4, "One Agent or Three: Building the Same App Twice", "Advanced", 50,
           ["Build the same capability twice: one agent with three tools, three agents with one each",
            "Run one eval set through both and find out which one you would actually ship",
            "Find the bug the second architecture introduces &mdash; a handoff that loses information",
            "Fix it with typed handoffs (<code>response_format</code>) and watch the accuracy come back"],
           "> **Builds on Lab 1.3.** Same tools, same case file. The question is no longer whether\n"
           "> an agent works &mdash; it is what breaks when you split one into three, and how you fix it."),
    setup(4),
    code(DOMAIN),

    md("""
## Concept

Splitting one agent into three buys you specialisation: shorter prompts, fewer tools each, a
clearer place to put a control. It costs you **coordination** &mdash; every handoff is another model
call, another context to rebuild, another place to lose information.

The usual mistake is to assume the cost of that is a bit more latency. It is not. The expensive
part is that **every handoff is a lossy re-encoding**: the worker answers in prose, the supervisor
has to recover a fact from it, and whatever does not survive that step is silently gone. The run
still completes. The answer is still confident. It is just wrong.

So this lab builds the same capability three ways &mdash; one agent, three agents handing off in
prose, three agents handing off a typed object &mdash; and asks the only question that matters
first: **which one would you ship?** Calls, latency and tokens are printed too, because you should
know how to read them, but they are not the finding.
"""),

    md("""
## Section 1 &mdash; Instrument first, argue later

Before comparing two architectures you need to see what each one *did*: how many model calls, how
many tool calls, how long, and &mdash; since this is the one lab in Module 1 that looks at cost &mdash;
how many tokens. `usage_metadata` on each `AIMessage` carries what the gateway actually reported,
so you are reading the real thing rather than estimating from string length.

Do not read too much into the token column. It is here once, so you know how to get it when you
need it. Everything that follows is about whether the thing *works*.
"""),
    code('''
class Meter:
    """Tokens, wall time and model calls for one architecture over one eval set."""

    def __init__(self, label: str):
        self.label = label
        self.in_tokens = self.out_tokens = self.calls = self.tool_calls = 0
        self.seconds = 0.0

    def record(self, result: dict, seconds: float) -> None:
        """Add one agent run. `result` is what create_agent returned."""
        self.seconds += seconds
        for m in result["messages"]:
            if getattr(m, "type", None) != "ai":
                continue
            self.calls += 1
            self.tool_calls += len(m.tool_calls or [])
            usage = getattr(m, "usage_metadata", None) or {}
            self.in_tokens += usage.get("input_tokens", 0)     # you are billed for the context too
            self.out_tokens += usage.get("output_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.in_tokens + self.out_tokens

    def row(self, cases: int) -> str:
        return (f"{self.label:22} {self.total_tokens:>8} tok  {self.calls:>3} calls  "
                f"{self.tool_calls:>3} tools  {self.seconds:>6.1f}s  "
                f"{self.total_tokens / max(cases, 1):>7.0f} tok/case")
''', '''
class Meter:
    """Tokens, wall time and model calls for one architecture over one eval set."""

    def __init__(self, label: str):
        self.label = label
        self.in_tokens = self.out_tokens = self.calls = self.tool_calls = 0
        self.seconds = 0.0

    def record(self, result: dict, seconds: float) -> None:
        """Add one agent run. `result` is what create_agent returned."""
        self.seconds += seconds
        for m in result["messages"]:
            if getattr(m, "type", None) != "ai":
                continue
            self.calls += 1
            self.tool_calls += len(m.tool_calls or [])
            usage = getattr(m, "usage_metadata", None) or {}
            self.in_tokens += usage.get("input_tokens", 0)     # you are billed for the context too
            self.out_tokens += usage.get("output_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.in_tokens + self.out_tokens

    def row(self, cases: int) -> str:
        return (f"{self.label:22} {self.total_tokens:>8} tok  {self.calls:>3} calls  "
                f"{self.tool_calls:>3} tools  {self.seconds:>6.1f}s  "
                f"{self.total_tokens / max(cases, 1):>7.0f} tok/case")
'''),
    code('''
# --- Self-check: Section 1   (canned messages -- no model call)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

def _fake_run(n_in, n_out, tools=0):
    ai = AIMessage(content="x", tool_calls=[{"name": "t", "args": {}, "id": f"c{i}",
                                             "type": "tool_call"} for i in range(tools)])
    ai.usage_metadata = {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}
    return {"messages": [HumanMessage("q"), ai]}

def _metered():
    m = Meter("test")
    m.record(_fake_run(100, 20, tools=1), 1.5)
    m.record(_fake_run(300, 30), 2.5)
    return m

check("input tokens are counted", lambda: _metered().in_tokens == 400,
      "the context you resend is the larger half of the bill -- count it")
check("output tokens are counted", lambda: _metered().out_tokens == 50)
check("total is both sides", lambda: _metered().total_tokens == 450)
check("model calls are counted", lambda: _metered().calls == 2)
check("tool calls are counted separately", lambda: _metered().tool_calls == 1)
check("wall time accumulates", lambda: abs(_metered().seconds - 4.0) < 1e-6)
check("a run with no usage metadata does not crash",
      lambda: Meter("x").record({"messages": [AIMessage("no usage")]}, 0.1) is None)
'''),

    md("""
## Section 2 &mdash; Two architectures over the same tools

**Arm 1 &mdash; one agent, three tools.** One `create_agent`, one context, one loop.

**Arm 2 &mdash; three specialists and a supervisor.** Each specialist is its own `create_agent` with
exactly one tool and a narrow prompt. The supervisor decides who to call, and the answer has to
be assembled from what comes back.

Both arms must answer the same questions. Write the supervisor's routing rule.
"""),
    code('''
from langchain_core.tools import tool
from langchain.agents import create_agent
from pydantic import BaseModel, Field

@tool
def lookup_payment(ref: str) -> str:
    """Return the ledger record for one payment reference such as 'PMT-1003'."""
    r = LEDGER.get(ref)
    return json.dumps({"ref": ref, **r}) if r else f"no payment found with reference {ref!r}"

@tool
def policy_for(reason_code: str) -> str:
    """Return the operating policy for one failure reason code such as 'LIMIT_BREACH'."""
    return POLICY.get(reason_code, f"no policy on file for reason code {reason_code!r}")

@tool
def requires_human_approval(reason_code: str) -> str:
    """Return whether a reason code obliges a human decision before any action."""
    return json.dumps({"reason_code": reason_code, "needs_human": reason_code in NEEDS_HUMAN})

ALL_TOOLS = [lookup_payment, policy_for, requires_human_approval]

SPECIALISTS = {
    "ledger": ("You read the ledger. Look up the payment and report its fields verbatim. "
               "Do not interpret policy.", [lookup_payment]),
    "policy": ("You read the policy catalogue. Given a reason code, report the policy text "
               "verbatim. Do not look up payments.", [policy_for]),
    "control": ("You decide whether a human must approve. Given a reason code, report "
                "true or false and nothing else.", [requires_human_approval]),
}

WORKERS = tuple(SPECIALISTS)          # ("ledger", "policy", "control")

def route(step: str) -> str:
    """Which specialist handles this step of the investigation?

    step is one of "read_payment", "read_policy", "check_approval".
    """
    mapping = {"read_payment": "ledger", "read_policy": "policy", "check_approval": "control"}
    if step not in mapping:
        raise ValueError(f"no worker for step {step!r}")
    return mapping[step]
''', '''
from langchain_core.tools import tool
from langchain.agents import create_agent
from pydantic import BaseModel, Field

@tool
def lookup_payment(ref: str) -> str:
    """Return the ledger record for one payment reference such as 'PMT-1003'."""
    r = LEDGER.get(ref)
    return json.dumps({"ref": ref, **r}) if r else f"no payment found with reference {ref!r}"

@tool
def policy_for(reason_code: str) -> str:
    """Return the operating policy for one failure reason code such as 'LIMIT_BREACH'."""
    return POLICY.get(reason_code, f"no policy on file for reason code {reason_code!r}")

@tool
def requires_human_approval(reason_code: str) -> str:
    """Return whether a reason code obliges a human decision before any action."""
    return json.dumps({"reason_code": reason_code, "needs_human": reason_code in NEEDS_HUMAN})

ALL_TOOLS = [lookup_payment, policy_for, requires_human_approval]

SPECIALISTS = {
    "ledger": ("You read the ledger. Look up the payment and report its fields verbatim. "
               "Do not interpret policy.", [lookup_payment]),
    "policy": ("You read the policy catalogue. Given a reason code, report the policy text "
               "verbatim. Do not look up payments.", [policy_for]),
    "control": ("You decide whether a human must approve. Given a reason code, report "
                "true or false and nothing else.", [requires_human_approval]),
}

WORKERS = tuple(SPECIALISTS)          # ("ledger", "policy", "control")

def route(step: str) -> str:
    """Which specialist handles this step of the investigation?

    step is one of "read_payment", "read_policy", "check_approval".
    """
    mapping = {"read_payment": "ledger", "read_policy": "policy", "check_approval": "control"}
    if step not in mapping:
        raise ValueError(f"no worker for step {step!r}")
    return mapping[step]
'''),
    code('''
# --- Self-check: Section 2   (routing + agent construction -- no model call)
def _bad_step():
    try:
        route("send_email")
        return False
    except ValueError:
        return True

check("each step routes to its specialist",
      lambda: [route(s) for s in ("read_payment", "read_policy", "check_approval")]
              == ["ledger", "policy", "control"])
check("an unknown step is refused, not guessed", _bad_step,
      "a supervisor that invents a worker is the single commonest multi-agent bug")
check("each specialist holds exactly one tool",
      lambda: all(len(tools) == 1 for _, tools in SPECIALISTS.values()))
check("between them the specialists cover every tool",
      lambda: {t.name for _, ts in SPECIALISTS.values() for t in ts}
              == {t.name for t in ALL_TOOLS})
check("each specialist prompt says what it must NOT do",
      lambda: all("not" in p.lower() or "nothing else" in p.lower()
                  for p, _ in SPECIALISTS.values()),
      "a narrow worker needs its boundary written down or it drifts wide")
'''),

    md("""
## Section 3 &mdash; One eval set, both arms

Five cases. Each has a reference, and an assertion that is true of a correct answer &mdash; a
substring we require in the final text. Crude, deliberately: this is the baseline pass rate the
whole course measures against, and it has to be something you can defend.
"""),
    code('''
EVAL_SET = [
    {"ref": "PMT-1003", "q": "Investigate PMT-1003 and say what must happen next.",
     "must_contain": ["treasury"],  "needs_human": True},
    {"ref": "PMT-1005", "q": "Investigate PMT-1005 and say what must happen next.",
     "must_contain": ["compliance"], "needs_human": True},
    {"ref": "PMT-1002", "q": "Investigate PMT-1002 and say what must happen next.",
     "must_contain": ["retry"],     "needs_human": False},
    {"ref": "PMT-1004", "q": "Investigate PMT-1004 and say what must happen next.",
     "must_contain": ["originator", "r04"], "needs_human": False},
    {"ref": "PMT-1001", "q": "Investigate PMT-1001 and say what must happen next.",
     "must_contain": ["settled"],   "needs_human": False},
]

def passes(case: dict, answer: str) -> bool:
    """A case passes when the answer mentions any of its required terms."""
    low = (answer or "").lower()
    return any(term in low for term in case["must_contain"])


def run_single(meter: Meter) -> list[bool]:
    """Arm 1: one agent, all three tools, one context per case."""
    agent = create_agent(
        model=get_llm(), tools=ALL_TOOLS,
        system_prompt=("You investigate payment exceptions. Procedure, in order: "
                       "1) look up the payment; 2) look up the policy for its reason code; "
                       "3) check whether it requires human approval; 4) state the next action."))
    out = []
    for case in EVAL_SET:
        t0 = time.time()
        result = agent.invoke({"messages": [("human", case["q"])]})
        meter.record(result, time.time() - t0)
        out.append(passes(case, result["messages"][-1].content))
    return out


def run_supervised(meter: Meter) -> list[bool]:
    """Arm 2: a supervisor calls three single-tool specialists and assembles the answer."""
    agents = {name: create_agent(model=get_llm(), tools=tools, system_prompt=prompt)
              for name, (prompt, tools) in SPECIALISTS.items()}

    def call(worker: str, message: str) -> str:
        t0 = time.time()
        result = agents[worker].invoke({"messages": [("human", message)]})
        meter.record(result, time.time() - t0)
        return result["messages"][-1].content

    out = []
    for case in EVAL_SET:
        ledger = call(route("read_payment"), f"Look up {case['ref']}.")
        code_ = next((c for c in POLICY if c in ledger), "NONE")
        policy = call(route("read_policy"), f"What is the policy for {code_}?")
        control = call(route("check_approval"), f"Does {code_} need human approval?")
        # the supervisor's own call: assemble, and pay for the context again
        t0 = time.time()
        final = create_agent(model=get_llm(), tools=[], system_prompt=(
            "You are the supervisor. Using only the specialist reports, state the next action "
            "in one short line.")).invoke({"messages": [("human",
                f"LEDGER: {ledger}\\nPOLICY: {policy}\\nCONTROL: {control}\\n\\n{case['q']}")]})
        meter.record(final, time.time() - t0)
        out.append(passes(case, final["messages"][-1].content))
    return out


def run_supervised_typed(meter: Meter) -> list[bool]:
    """Arm 3: the same three specialists, but the ledger worker returns a LedgerReport."""
    ledger_agent = create_agent(model=get_llm(), tools=[lookup_payment],
                                system_prompt=SPECIALISTS["ledger"][0],
                                response_format=LedgerReport)
    others = {name: create_agent(model=get_llm(), tools=tools, system_prompt=prompt)
              for name, (prompt, tools) in SPECIALISTS.items() if name != "ledger"}

    def call(worker: str, message: str) -> str:
        t0 = time.time()
        result = others[worker].invoke({"messages": [("human", message)]})
        meter.record(result, time.time() - t0)
        return result["messages"][-1].content

    out = []
    for case in EVAL_SET:
        t0 = time.time()
        rep = ledger_agent.invoke({"messages": [("human", f"Look up {case['ref']}.")]})
        meter.record(rep, time.time() - t0)
        report = rep.get("structured_response")             # a LedgerReport, not a paragraph
        code_ = extract_reason_code(report)                 # one field access
        policy = call("policy", f"What is the policy for {code_}?")
        control = call("control", f"Does {code_} need human approval?")
        t0 = time.time()
        final = create_agent(model=get_llm(), tools=[], system_prompt=(
            "You are the supervisor. Using only the specialist reports, state the next action "
            "in one short line.")).invoke({"messages": [("human",
                f"LEDGER: {report}\\nPOLICY: {policy}\\nCONTROL: {control}\\n\\n{case['q']}")]})
        meter.record(final, time.time() - t0)
        out.append(passes(case, final["messages"][-1].content))
    return out
''', '''
EVAL_SET = [
    {"ref": "PMT-1003", "q": "Investigate PMT-1003 and say what must happen next.",
     "must_contain": ["treasury"],  "needs_human": True},
    {"ref": "PMT-1005", "q": "Investigate PMT-1005 and say what must happen next.",
     "must_contain": ["compliance"], "needs_human": True},
    {"ref": "PMT-1002", "q": "Investigate PMT-1002 and say what must happen next.",
     "must_contain": ["retry"],     "needs_human": False},
    {"ref": "PMT-1004", "q": "Investigate PMT-1004 and say what must happen next.",
     "must_contain": ["originator", "r04"], "needs_human": False},
    {"ref": "PMT-1001", "q": "Investigate PMT-1001 and say what must happen next.",
     "must_contain": ["settled"],   "needs_human": False},
]

def passes(case: dict, answer: str) -> bool:
    """A case passes when the answer mentions any of its required terms."""
    low = (answer or "").lower()
    return any(term in low for term in case["must_contain"])


def run_single(meter: Meter) -> list[bool]:
    """Arm 1: one agent, all three tools, one context per case."""
    agent = create_agent(
        model=get_llm(), tools=ALL_TOOLS,
        system_prompt=("You investigate payment exceptions. Procedure, in order: "
                       "1) look up the payment; 2) look up the policy for its reason code; "
                       "3) check whether it requires human approval; 4) state the next action."))
    out = []
    for case in EVAL_SET:
        t0 = time.time()
        result = agent.invoke({"messages": [("human", case["q"])]})
        meter.record(result, time.time() - t0)
        out.append(passes(case, result["messages"][-1].content))
    return out


def run_supervised(meter: Meter) -> list[bool]:
    """Arm 2: a supervisor calls three single-tool specialists and assembles the answer."""
    agents = {name: create_agent(model=get_llm(), tools=tools, system_prompt=prompt)
              for name, (prompt, tools) in SPECIALISTS.items()}

    def call(worker: str, message: str) -> str:
        t0 = time.time()
        result = agents[worker].invoke({"messages": [("human", message)]})
        meter.record(result, time.time() - t0)
        return result["messages"][-1].content

    out = []
    for case in EVAL_SET:
        ledger = call(route("read_payment"), f"Look up {case['ref']}.")
        code_ = next((c for c in POLICY if c in ledger), "NONE")
        policy = call(route("read_policy"), f"What is the policy for {code_}?")
        control = call(route("check_approval"), f"Does {code_} need human approval?")
        # the supervisor's own call: assemble, and pay for the context again
        t0 = time.time()
        final = create_agent(model=get_llm(), tools=[], system_prompt=(
            "You are the supervisor. Using only the specialist reports, state the next action "
            "in one short line.")).invoke({"messages": [("human",
                f"LEDGER: {ledger}\\nPOLICY: {policy}\\nCONTROL: {control}\\n\\n{case['q']}")]})
        meter.record(final, time.time() - t0)
        out.append(passes(case, final["messages"][-1].content))
    return out


def run_supervised_typed(meter: Meter) -> list[bool]:
    """Arm 3: the same three specialists, but the ledger worker returns a LedgerReport."""
    ledger_agent = create_agent(model=get_llm(), tools=[lookup_payment],
                                system_prompt=SPECIALISTS["ledger"][0],
                                response_format=LedgerReport)
    others = {name: create_agent(model=get_llm(), tools=tools, system_prompt=prompt)
              for name, (prompt, tools) in SPECIALISTS.items() if name != "ledger"}

    def call(worker: str, message: str) -> str:
        t0 = time.time()
        result = others[worker].invoke({"messages": [("human", message)]})
        meter.record(result, time.time() - t0)
        return result["messages"][-1].content

    out = []
    for case in EVAL_SET:
        t0 = time.time()
        rep = ledger_agent.invoke({"messages": [("human", f"Look up {case['ref']}.")]})
        meter.record(rep, time.time() - t0)
        report = rep.get("structured_response")             # a LedgerReport, not a paragraph
        code_ = extract_reason_code(report)                 # one field access
        policy = call("policy", f"What is the policy for {code_}?")
        control = call("control", f"Does {code_} need human approval?")
        t0 = time.time()
        final = create_agent(model=get_llm(), tools=[], system_prompt=(
            "You are the supervisor. Using only the specialist reports, state the next action "
            "in one short line.")).invoke({"messages": [("human",
                f"LEDGER: {report}\\nPOLICY: {policy}\\nCONTROL: {control}\\n\\n{case['q']}")]})
        meter.record(final, time.time() - t0)
        out.append(passes(case, final["messages"][-1].content))
    return out
'''),
    code('''
# --- Self-check: Section 3   (the scorer, on canned answers -- no model call)
_c = EVAL_SET[3]      # PMT-1004: passes on "originator" OR "r04"

check("an answer containing one required term passes",
      lambda: passes(_c, "Return to originator.") is True)
check("either term is enough",
      lambda: passes(_c, "Send it back with code R04.") is True,
      'must_contain is a list of alternatives -- "any", not "all"')
check("an answer containing none of them fails",
      lambda: passes(_c, "Escalate to Treasury.") is False)
check("the scorer is case-insensitive",
      lambda: passes(_c, "RETURN TO ORIGINATOR") is True)
check("an empty answer fails rather than crashing",
      lambda: passes(_c, "") is False)
check("every case names its expectation and its ref",
      lambda: all(c["must_contain"] and c["ref"] in LEDGER for c in EVAL_SET))
'''),

    md("""
## Section 4 &mdash; Find where the quality went

You now have two supervisor arms that score badly. Before theorising, instrument.

The obvious suspect is the handoff. `run_supervised` recovers the reason code from the ledger
worker's **prose** with a substring search:

```python
code_ = next((c for c in POLICY if c in ledger), "NONE")
```

That looks fragile, and it is &mdash; but "looks fragile" is not evidence. Build the typed
alternative, then **measure whether the handoff is actually losing anything**. A contract is what
makes that measurable: you cannot assert on a paragraph, but you can assert on a field.
"""),
    code('''
class LedgerReport(BaseModel):
    """What the ledger specialist returns. A contract, not a paragraph."""
    ref: str = Field(description="The payment reference that was looked up")
    status: str = Field(description="One of: settled, failed, held")
    reason_code: str = Field(description="BLANK")  # TODO: what must the supervisor be able to read?
    amount: float = Field(description="The payment amount")


def extract_reason_code(report) -> str:
    """The supervisor's read step, for both handoff styles.

    report is a LedgerReport when the worker has a contract, or prose when it does not.
    """
    if isinstance(report, LedgerReport):
        return BLANK                  # TODO: read the field -- no searching, no guessing
    return next((c for c in POLICY if c in str(report)), "NONE")   # the substring version


TRUTH = {ref: (rec["reason_code"] or "NONE") for ref, rec in LEDGER.items()}
''', '''
class LedgerReport(BaseModel):
    """What the ledger specialist returns. A contract, not a paragraph."""
    ref: str = Field(description="The payment reference that was looked up")
    status: str = Field(description="One of: settled, failed, held")
    reason_code: str = Field(
        description="The ledger reason code exactly as stored, e.g. 'LIMIT_BREACH' or "
                    "'SANCTIONS_REVIEW'. Use the literal string 'NONE' if the payment has none.")
    amount: float = Field(description="The payment amount")


def extract_reason_code(report) -> str:
    """The supervisor's read step, for both handoff styles.

    report is a LedgerReport when the worker has a contract, or prose when it does not.
    """
    if isinstance(report, LedgerReport):
        return report.reason_code     # one field access; nothing to misphrase
    return next((c for c in POLICY if c in str(report)), "NONE")   # the substring version


TRUTH = {ref: (rec["reason_code"] or "NONE") for ref, rec in LEDGER.items()}
'''),
    code('''
# --- Self-check: Section 4   (both handoff styles, on canned worker output -- no model call)
_typed = LedgerReport(ref="PMT-1003", status="held", reason_code="LIMIT_BREACH", amount=990000.0)
_prose_verbatim    = "PMT-1003 is held with reason code LIMIT_BREACH for USD 990,000."
_prose_paraphrased = "Payment PMT-1003 is on hold because it breaches the value limit."

def _desc():
    d = LedgerReport.model_fields["reason_code"].description
    if not d or d == "BLANK":
        raise NameError("reason_code still has no description")
    return d

check("the contract carries everything the supervisor needs",
      lambda: set(LedgerReport.model_fields) == {"ref", "status", "reason_code", "amount"})
check("reason_code tells the worker exactly what to put there",
      lambda: len(_desc()) > 40 and "NONE" in _desc(),
      "a worker that invents its own spelling breaks the supervisor just as badly as prose")
check("a typed handoff reads the field",
      lambda: extract_reason_code(_typed) == "LIMIT_BREACH")
check("the substring version works when the worker quotes the code",
      lambda: extract_reason_code(_prose_verbatim) == "LIMIT_BREACH")
check("...and silently returns NONE when it paraphrases instead",
      lambda: extract_reason_code(_prose_paraphrased) == "NONE",
      "same fact, different sentence -- and nothing anywhere reports an error")
check("we know the right answer for every case, so the handoff can be scored",
      lambda: TRUTH["PMT-1001"] == "NONE" and TRUTH["PMT-1005"] == "SANCTIONS_REVIEW")
'''),

    md("""
### Now measure it, instead of assuming

`_prose_paraphrased` proves the substring handoff **can** lose a fact. Whether it **does** on this
workload is a different question, and the only way to answer it is to run it.

`handoff_integrity()` puts the ledger specialist through both styles and scores the recovered
reason code against the ledger itself. Run it before you read the next section.
"""),
    code('''
def handoff_integrity() -> dict:
    """Does the reason code survive the handoff? Score both styles against the ledger."""
    typed_agent = create_agent(model=get_llm(), tools=[lookup_payment],
                               system_prompt=SPECIALISTS["ledger"][0],
                               response_format=LedgerReport)
    prose_agent = create_agent(model=get_llm(), tools=[lookup_payment],
                               system_prompt=SPECIALISTS["ledger"][0])
    rows, typed_ok, prose_ok = [], 0, 0
    for ref, want in TRUTH.items():
        t = typed_agent.invoke({"messages": [("human", f"Look up {ref}.")]})
        report = t.get("structured_response")
        got_t = extract_reason_code(report) if report is not None else "(no structured_response)"
        p = prose_agent.invoke({"messages": [("human", f"Look up {ref}.")]})
        got_p = extract_reason_code(p["messages"][-1].content)
        typed_ok += got_t == want
        prose_ok += got_p == want
        rows.append((ref, want, got_t, got_p))
    return {"rows": rows, "typed": typed_ok, "prose": prose_ok, "of": len(TRUTH)}
'''),

    md("""
## Run it for real

Three arms plus the handoff audit: about fifty agent runs, so give this cell a minute or two.
Read it top to bottom &mdash; does it work, what it took, and then whether the handoff is to blame.
"""),
    code('''
if llm_ready():
    def _bakeoff():
        single_m = Meter("single agent")
        super_m  = Meter("supervisor (prose)")
        typed_m  = Meter("supervisor (typed)")
        single_r = run_single(single_m)
        super_r  = run_supervised(super_m)
        typed_r  = run_supervised_typed(typed_m)
        n = len(EVAL_SET)

        print("does it work?")
        print("  case       single   supervisor(prose)   supervisor(typed)")
        for case, a, b, c in zip(EVAL_SET, single_r, super_r, typed_r):
            f = lambda x: "pass" if x else "FAIL"
            print(f"  {case['ref']}   {f(a):8} {f(b):19} {f(c)}")
        print(f"\\n  pass rate  {sum(single_r)}/{n} single   "
              f"{sum(super_r)}/{n} prose handoff   {sum(typed_r)}/{n} typed handoff")

        print("\\nwhat it took")
        print("  architecture            calls   tools     time     tokens")
        for m in (single_m, super_m, typed_m):
            print(f"  {m.label:22} {m.calls:>5}   {m.tool_calls:>5}   {m.seconds:>6.1f}s   "
                  f"{m.total_tokens:>7}")

        print("\\nis the handoff to blame?")
        hi = handoff_integrity()
        print("  ref        truth                typed                prose")
        for ref, want, got_t, got_p in hi["rows"]:
            print(f"  {ref}   {want:20} {got_t:20} {got_p}")
        print(f"\\n  reason code recovered:  typed {hi['typed']}/{hi['of']}   "
              f"prose {hi['prose']}/{hi['of']}")
        return {"single": (single_m, single_r), "prose": (super_m, super_r),
                "typed": (typed_m, typed_r), "handoff": hi}
    MEASURED = guard(_bakeoff)
'''),
    md("""
### Read the results

1. **The single agent won, and not narrowly.** On this eval set it answers four or five of the
   five; the supervisor arms typically manage nought to three, and the spread between runs is
   itself worth noticing &mdash; the split is not just worse, it is less predictable. That result is
   the point of Module 1's rubric, not a failure of the lab.

2. **The handoff is innocent.** This is the part worth slowing down for. The audit at the bottom
   scores the reason code recovered from the ledger worker, and on this workload *both* styles
   recover it &mdash; typically 5/5 and 5/5. The specialist was told to "report its fields verbatim",
   so it quotes the code, and even the substring search finds it. The fragile-looking line is not
   what is costing you the accuracy.

   Note what just happened: the obvious explanation was wrong, and one measurement was enough to
   retire it. Nothing else in this lab is as valuable as that habit.

3. **So where does it go?** Compare a failing case across the arms. The single agent holds the
   ledger record, the policy text and the approval flag **in one context** when it composes its
   answer. The supervisor holds three separate summaries, each written by a worker that could not
   see the question. Every fact needed to answer correctly is present somewhere in the system;
   no single agent ever holds them all at once. What is lost is not data &mdash; it is **context**.

   That is why the typed arm does not rescue the score either. Typing one edge makes that edge
   assertable, which is worth having and is exactly how you ruled it out above. It does nothing
   about the fragmentation, because the fragmentation is the architecture.

4. **The costs, briefly.** More agents means more calls and more wall time, always. Tokens often
   come out roughly level, because specialists get shorter prompts. If you expected the split to
   be dramatically more expensive and it was not, that is worth knowing too &mdash; the argument
   against splitting is rarely the bill.

5. **What would justify the split anyway?** Not elegance. Different credentials per tool,
   independent auditability, or one worker that has to be deployable on its own. None of those
   are visible in any of the numbers above, which is exactly why Lab 1.5's rubric asks about them
   *before* it looks at a result.

Module 5 is where the fragmentation gets a real fix: one typed state object every agent reads and
writes, instead of a relay of summaries. Keep this lab's numbers &mdash; you will score that graph
against them.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. **Make the handoff guilty.** The audit came back clean because the ledger worker was told to
   report its fields *verbatim*. Change its prompt to "summarise the payment in one friendly
   sentence" and re-run `handoff_integrity()`. Watch the prose column collapse while the typed
   column holds. You have just reproduced, deliberately, the bug that was not there &mdash; which
   tells you what the contract is really insuring against.
2. **Give the workers the question.** Each specialist is asked its narrow sub-question and never
   sees what the user actually wanted. Pass the original question along with each worker call and
   re-run. How much of the gap closes? This is the cheap half of what Module 5 does properly.
3. **Make the failure loud.** `extract_reason_code` returns `"NONE"` when it finds nothing, and
   everything downstream carries on regardless. Change the supervisor to refuse rather than
   continue, and decide where that check belongs &mdash; in the worker, the supervisor, or the tool.
   Module 8 argues for one of the three.
4. **Argue with the scorer.** `passes()` accepts an answer that contains *any* required term.
   PMT-1004 requires two of them (`originator`, `r04`). Switch it to `all` and re-run. Which
   arm loses more, and is your eval now measuring the architecture or the wording?
"""),
]


# =========================================================================== #
# Lab 1.5 -- challenge: the decision rubric, executable and evidence-led
# =========================================================================== #
LAB5 = [
    header(5, "Challenge &mdash; The Decision Rubric, Made Executable", "Advanced", 45,
           ["Encode the &ldquo;do I need multiple agents?&rdquo; rubric as code that terminates early",
            "Write the acceptance bar down <i>before</i> you look at any result",
            "Feed Lab 1.4's real pass rates in and let them overrule the architecture you wanted",
            "Have an agent produce the design-review record as a typed object, not a paragraph"],
           "> **The take-home artifact.** This is the one thing from Module 1 you will use next\n"
           "> week: a rubric that answers the question before anyone starts building."),
    setup(5),
    code(DOMAIN),

    md("""
## Concept

Most multi-agent systems exist because the diagram was appealing, not because a question was
asked. The rubric below asks four questions in a fixed order and **stops at the first one that
decides**. Order matters: a cheaper architecture that answers the requirement wins, and asking
about elegance before asking about need is how teams talk themselves into a supervisor.

Then the measurement gets a veto. A rubric that cannot be overruled by evidence is just a
preference with a flowchart.
"""),

    md("""
## Section 1 &mdash; The rubric, as code

Four questions, asked in order, first decisive answer wins:

1. Is the work **deterministic**? &rarr; a workflow, no agent at all.
2. Does it fit **one context** with a handful of tools? &rarr; a single agent.
3. Do the parts need **separate authority** &mdash; different credentials, independent audit,
   independent deployment? &rarr; supervisor and workers.
4. Otherwise &rarr; peers that negotiate. Rare, and expensive.
"""),
    code('''
from pydantic import BaseModel, Field

VERDICTS = ("workflow", "single_agent", "supervisor_worker", "peer_to_peer")

def rubric(deterministic: bool, fits_one_agent: bool, separate_authority: bool) -> str:
    """Return the first verdict the answers decide. Order is the design."""
    if deterministic:
        return "workflow"
    if fits_one_agent:
        return BLANK                  # TODO: which verdict, and why is it asked BEFORE authority?
    if separate_authority:
        return "supervisor_worker"
    return "peer_to_peer"
''', '''
from pydantic import BaseModel, Field

VERDICTS = ("workflow", "single_agent", "supervisor_worker", "peer_to_peer")

def rubric(deterministic: bool, fits_one_agent: bool, separate_authority: bool) -> str:
    """Return the first verdict the answers decide. Order is the design."""
    if deterministic:
        return "workflow"
    if fits_one_agent:
        return "single_agent"         # cheapest thing that works, before any question of elegance
    if separate_authority:
        return "supervisor_worker"
    return "peer_to_peer"
'''),
    code('''
# --- Self-check: Section 1
check("deterministic work needs no agent",
      lambda: rubric(True, False, True) == "workflow")
check("determinism is asked first",
      lambda: rubric(True, True, True) == "workflow",
      "if it can be a workflow, nothing later in the rubric should be able to override that")
check("work that fits one context gets one agent",
      lambda: rubric(False, True, False) == "single_agent")
check("fitting one context beats wanting separate authority",
      lambda: rubric(False, True, True) == "single_agent",
      "asking about authority first is how a team talks itself into a supervisor")
check("separate authority earns a supervisor",
      lambda: rubric(False, False, True) == "supervisor_worker")
check("peer-to-peer is the residue, never the goal",
      lambda: rubric(False, False, False) == "peer_to_peer")
check("every path returns a known verdict",
      lambda: all(rubric(a, b, c) in VERDICTS
                  for a in (0, 1) for b in (0, 1) for c in (0, 1)))
'''),

    md("""
## Section 2 &mdash; The acceptance bar, written down first

Write the acceptance bar **before** you see a result, or you will fit it to whatever you got.
This is the same discipline as pre-registering an experiment, and it is the only reason the veto
in Section 3 means anything.

Note what is on the bar and what is not. **Silent failures** are there because Lab 1.4 showed you
what they look like: a fluent answer assembled from a lookup that quietly returned nothing. Cost
appears once, as latency, because a caller who will not wait is a real constraint. Token price is
not on the bar at all &mdash; at the scale most teams run, it is the least of the things that will
go wrong.
"""),
    code('''
BAR = {
    "min_pass_rate":      0.80,       # below this the architecture is not a candidate at all
    "max_silent_failures": 0,         # a wrong answer that reports no error -- Lab 1.4's bug
    "max_seconds_case":   30.0,       # the one cost clause: an answer nobody waits for is no answer
    "min_quality_gain":   0.10,       # a split must buy at least this much pass rate to be worth it
}

def meets_bar(m: dict) -> tuple[bool, str]:
    """m: {"pass_rate", "silent_failures", "seconds_case"}. Return (ok, first failing reason)."""
    if m["pass_rate"] < BAR["min_pass_rate"]:
        return False, f"pass rate {m['pass_rate']:.0%} below {BAR['min_pass_rate']:.0%}"
    if m["silent_failures"] > BAR["max_silent_failures"]:
        return False, (f"{m['silent_failures']} answer(s) wrong with nothing in the trace saying so")
    if BLANK:                         # TODO: the third clause -- how long a caller will wait
        return False, f"{m['seconds_case']:.1f}s/case over {BAR['max_seconds_case']}"
    return True, "meets the bar"
''', '''
BAR = {
    "min_pass_rate":      0.80,       # below this the architecture is not a candidate at all
    "max_silent_failures": 0,         # a wrong answer that reports no error -- Lab 1.4's bug
    "max_seconds_case":   30.0,       # the one cost clause: an answer nobody waits for is no answer
    "min_quality_gain":   0.10,       # a split must buy at least this much pass rate to be worth it
}

def meets_bar(m: dict) -> tuple[bool, str]:
    """m: {"pass_rate", "silent_failures", "seconds_case"}. Return (ok, first failing reason)."""
    if m["pass_rate"] < BAR["min_pass_rate"]:
        return False, f"pass rate {m['pass_rate']:.0%} below {BAR['min_pass_rate']:.0%}"
    if m["silent_failures"] > BAR["max_silent_failures"]:
        return False, (f"{m['silent_failures']} answer(s) wrong with nothing in the trace saying so")
    if m["seconds_case"] > BAR["max_seconds_case"]:
        return False, f"{m['seconds_case']:.1f}s/case over {BAR['max_seconds_case']}"
    return True, "meets the bar"
'''),
    code('''
# --- Self-check: Section 2
_ok     = {"pass_rate": 0.9,  "silent_failures": 0, "seconds_case": 8.0}
_slow   = {"pass_rate": 0.9,  "silent_failures": 0, "seconds_case": 99.0}
_silent = {"pass_rate": 0.9,  "silent_failures": 2, "seconds_case": 8.0}
_bad    = {"pass_rate": 0.40, "silent_failures": 0, "seconds_case": 1.0}

check("a good result passes",                 lambda: meets_bar(_ok)[0] is True)
check("a slow one is rejected",               lambda: meets_bar(_slow)[0] is False,
      "an answer nobody waits for is not an answer")
check("silent failures are disqualifying",    lambda: meets_bar(_silent)[0] is False,
      "a 90% pass rate with two invisible failures is worse than an 80% one that shouts")
check("an inaccurate one is rejected first",  lambda: "pass rate" in meets_bar(_bad)[1],
      "correctness is checked before anything else")
check("the reason names the failing clause",  lambda: "s/case" in meets_bar(_slow)[1])
'''),

    md("""
## Section 3 &mdash; Candidate, then evidence

The rubric proposes; the measurement disposes. `recommend()` takes the brief and, optionally,
the two measurements from Lab 1.4. With no measurements it returns a candidate and says so.
With them, it may **overrule** the candidate &mdash; and that is the point of the whole lab.
"""),
    code('''
def recommend(brief: dict, single: dict | None = None, multi: dict | None = None) -> dict:
    """brief: {"name", "deterministic", "fits_one_agent", "separate_authority"}.
    single/multi: {"pass_rate", "silent_failures", "seconds_case"} from a real run.
    """
    candidate = rubric(brief["deterministic"], brief["fits_one_agent"],
                       brief["separate_authority"])
    out = {"name": brief["name"], "candidate": candidate,
           "decision": candidate, "why": "rubric only; no measurement supplied"}
    if not (single and multi):
        return out

    s_ok, s_why = meets_bar(single)
    m_ok, m_why = meets_bar(multi)
    gain = multi["pass_rate"] - single["pass_rate"]

    if candidate in ("supervisor_worker", "peer_to_peer"):
        # the split has to EARN itself against the single agent
        if not m_ok:
            out.update(decision="single_agent", why=f"multi-agent below the bar: {m_why}")
        elif BLANK:                   # TODO: did the split buy enough quality to be worth it?
            out.update(decision="single_agent",
                       why=f"split changed the pass rate by only {gain:+.0%}")
        else:
            out.update(decision=candidate, why=f"split earned it: {gain:+.0%} pass rate")
    else:
        out.update(why=f"single arm {s_why}" if s_ok else f"single arm rejected: {s_why}")
    return out
''', '''
def recommend(brief: dict, single: dict | None = None, multi: dict | None = None) -> dict:
    """brief: {"name", "deterministic", "fits_one_agent", "separate_authority"}.
    single/multi: {"pass_rate", "silent_failures", "seconds_case"} from a real run.
    """
    candidate = rubric(brief["deterministic"], brief["fits_one_agent"],
                       brief["separate_authority"])
    out = {"name": brief["name"], "candidate": candidate,
           "decision": candidate, "why": "rubric only; no measurement supplied"}
    if not (single and multi):
        return out

    s_ok, s_why = meets_bar(single)
    m_ok, m_why = meets_bar(multi)
    gain = multi["pass_rate"] - single["pass_rate"]

    if candidate in ("supervisor_worker", "peer_to_peer"):
        # the split has to EARN itself against the single agent
        if not m_ok:
            out.update(decision="single_agent", why=f"multi-agent below the bar: {m_why}")
        elif gain < BAR["min_quality_gain"]:
            out.update(decision="single_agent",
                       why=f"split changed the pass rate by only {gain:+.0%}")
        else:
            out.update(decision=candidate, why=f"split earned it: {gain:+.0%} pass rate")
    else:
        out.update(why=f"single arm {s_why}" if s_ok else f"single arm rejected: {s_why}")
    return out
'''),
    code('''
# --- Self-check: Section 3
SPLIT_BRIEF = {"name": "payment exceptions", "deterministic": False,
               "fits_one_agent": False, "separate_authority": True}

_single_good  = {"pass_rate": 0.80, "silent_failures": 0, "seconds_case": 9.0}
_multi_worth  = {"pass_rate": 0.95, "silent_failures": 0, "seconds_case": 20.0}  # +15%, clean
_multi_silent = {"pass_rate": 0.95, "silent_failures": 1, "seconds_case": 20.0}  # +15%, but hides one
_multi_flat   = {"pass_rate": 0.82, "silent_failures": 0, "seconds_case": 15.0}  # +2%

check("with no measurement it returns the rubric's candidate",
      lambda: recommend(SPLIT_BRIEF)["decision"] == "supervisor_worker")
check("a split that clearly earns it is kept",
      lambda: recommend(SPLIT_BRIEF, _single_good, _multi_worth)["decision"] == "supervisor_worker")
check("a split that hides a failure is overruled despite scoring higher",
      lambda: recommend(SPLIT_BRIEF, _single_good, _multi_silent)["decision"] == "single_agent",
      "a higher pass rate does not buy the right to fail invisibly")
check("the overrule says what was wrong with it",
      lambda: "nothing in the trace" in recommend(SPLIT_BRIEF, _single_good, _multi_silent)["why"])
check("a split that buys nothing is overruled",
      lambda: recommend(SPLIT_BRIEF, _single_good, _multi_flat)["decision"] == "single_agent")
check("the candidate is reported even when overruled",
      lambda: recommend(SPLIT_BRIEF, _single_good, _multi_silent)["candidate"] == "supervisor_worker",
      "a design review needs to see what was proposed AND what the evidence did to it")
'''),

    md("""
## Section 4 &mdash; The design-review table

Four briefs, one table. This is the artifact you take away.
"""),
    code('''
BRIEFS = [
    {"name": "nightly reconciliation",   "deterministic": True,
     "fits_one_agent": False, "separate_authority": False},
    {"name": "single-desk triage",       "deterministic": False,
     "fits_one_agent": True,  "separate_authority": False},
    {"name": "payment exceptions",       "deterministic": False,
     "fits_one_agent": False, "separate_authority": True},
    {"name": "cross-desk negotiation",   "deterministic": False,
     "fits_one_agent": False, "separate_authority": False},
]

def review_table(briefs, single=None, multi=None) -> str:
    rows = ["  brief                  candidate            decision             why",
            "  " + "-" * 100]
    for b in briefs:
        r = recommend(b, single, multi)
        flag = " " if r["decision"] == r["candidate"] else "!"
        rows.append(f"{flag} {r['name']:22} {r['candidate']:20} {r['decision']:20} {r['why']}")
    return "\\n".join(rows)

guard(lambda: print(review_table(BRIEFS)))
print("\\n(rows marked ! are where the evidence overruled the rubric)")
'''),
    code('''
# --- Self-check: Section 4   (built lazily -- a module-level call into a blanked
#     function would crash the cell instead of reporting [TODO])
def _t():
    return review_table(BRIEFS, _single_good, _multi_silent)

check("every brief appears", lambda: all(b["name"] in _t() for b in BRIEFS))
check("the overruled row is flagged", lambda: "!" in _t(),
      "a design review must be able to see where evidence beat the proposal")
check("the deterministic brief is still a workflow", lambda: "workflow" in _t())
'''),

    md("""
## Run it for real

Feed in what you actually got in Lab 1.4, then have the model produce the design-review record.
Notice what it is being asked to do: not to *decide*, but to write up a decision your code
already made and can defend &mdash; and to return it as an object, so it can be filed rather than
re-read.
"""),
    code('''
class DesignDecision(BaseModel):
    """The record a design review actually needs."""
    architecture: str = Field(description="The architecture chosen, in one or two words")
    rationale: str = Field(description="Two sentences at most, citing only the evidence given")
    revisit_when: str = Field(
        description="One concrete, checkable condition that would reopen this decision")
'''),
    code('''
if llm_ready():
    def _review():
        # Edit these two dicts with YOUR results from Lab 1.4, then re-run.
        single = {"pass_rate": 1.00, "silent_failures": 0, "seconds_case": 3.4}
        multi  = {"pass_rate": 0.40, "silent_failures": 3, "seconds_case": 4.7}

        print(review_table(BRIEFS, single, multi))
        verdict = recommend(BRIEFS[2], single, multi)

        # The record is an OBJECT, so it can go straight into a design-review system --
        # the same pattern as Lab 1.3's Verdict, applied to your own decision.
        record = get_llm().with_structured_output(DesignDecision).invoke(
            "Produce the design-review record for this decision. Use only the facts given; "
            "do not add claims. `revisit_when` must name a concrete, checkable condition.\\n\\n"
            f"BRIEF: {verdict['name']}\\n"
            f"CANDIDATE FROM RUBRIC: {verdict['candidate']}\\n"
            f"DECISION: {verdict['decision']}\\n"
            f"EVIDENCE: {verdict['why']}\\n"
            f"ACCEPTANCE BAR: {BAR}\\n"
            f"MEASURED single={single} multi={multi}")
        print("\\n--- the design-review record ---")
        print(f"architecture : {record.architecture}")
        print(f"rationale    : {record.rationale}")
        print(f"revisit when : {record.revisit_when}")
    guard(_review)
'''),
    md("""
### Read it

If the paragraph reads as a justification you would actually sign, the rubric did its job. If it
reads as advocacy for the interesting architecture, look again at which clause let it through.

**What you take from Module 1:** the agent loop, in your own code and in `create_agent`; tools
whose descriptions you can defend with a number; typed contracts between agents, and the bug that
appears the moment you drop one; a rubric that terminates early; and an acceptance bar written
before the result that can overrule your own design preference. Modules 2
and 3 make the agent better. This lab is what stops you building one you did not need.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `rubric()` takes booleans, which assumes someone already made the hard calls. Replace
   `fits_one_agent` with a function of the tool count and argue for the threshold you pick.
2. Add a fifth question &mdash; **can this fail unattended?** &mdash; that can force a supervisor even
   when the rubric would otherwise say single agent. Where in the order does it belong, and why?
3. Take a real brief from your own team, fill in the three booleans honestly, and run it with
   your Lab 1.4 results. If the verdict surprises you, which boolean were you tempted to fill in
   dishonestly?
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
