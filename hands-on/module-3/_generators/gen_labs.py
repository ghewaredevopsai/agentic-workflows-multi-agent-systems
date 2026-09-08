#!/usr/bin/env python3
"""
Generate Module 3 lab notebooks and their solutions from one source.

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

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 1 &middot; Module 3 &mdash; Memory, State &amp; the LangGraph Substrate**

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
# the shared synthetic domain -- one use case runs through all five labs
# --------------------------------------------------------------------------- #
DOMAIN = '''
# ------------------------------------------------- the case file (synthetic, self-contained)
# One domain runs through all five Module 3 labs: payment exceptions on a small ledger.
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
# Lab 3.1 -- memory that survives a long session
# =========================================================================== #
LAB1 = [
    header(1, "Memory That Survives a Long Session", "Intermediate", 35,
           ["Watch an agent forget, by growing the history until the window bites",
            "Bound it with <code>trim_messages</code> &mdash; including the counter that fails here",
            "Summarise what you drop, so compaction is not amnesia",
            "Hand the whole problem to a checkpointer and a <code>thread_id</code>"],
           "> **The thread.** All five Module 3 labs work one case: payment exceptions on a small\n"
           "> synthetic ledger. Modules 1 and 2 built the agent; Module 3 gives it a memory and a\n"
           "> state you can inspect."),
    setup(1),
    code(DOMAIN),

    md("""
## Concept

Everything an agent "remembers" is something your code put back in front of it. That leaves three
jobs, and LangChain has a piece for each:

| Job | The piece |
|---|---|
| keep the window bounded | `trim_messages` |
| keep what you dropped | a summary chain |
| keep it across turns and restarts | a checkpointer + `thread_id` |

Do the first without the second and you have built amnesia with a token budget.
"""),

    md("""
## Section 1 &mdash; Find the turn where it forgets

A long investigation, one important fact stated at the very beginning. Grow the history until
the fact falls out of the window, and note which turn it happened on.
"""),
    code('''
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.utils import count_tokens_approximately

SYSTEM = ("You are a payments operations analyst working one case. Answer only from this "
          "conversation.")

THE_FACT = "The client contact for this case is Priya Raman on the Singapore desk."

def long_session(turns: int = 14) -> list:
    """A conversation whose first human turn carries the fact that matters."""
    msgs = [SystemMessage(SYSTEM), HumanMessage(f"Open the case for PMT-1005. {THE_FACT}")]
    for i in range(turns):
        msgs.append(AIMessage(f"Noted. Checking sanction screening batch {i}: "
                              + "reviewing the counterparty records in detail. " * 6))
        msgs.append(HumanMessage(f"And what about batch {i + 1}?"))
    return msgs


def window_fits(messages: list, budget: int) -> bool:
    """Does this conversation still fit the budget?"""
    return count_tokens_approximately(messages) <= budget
''', '''
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.utils import count_tokens_approximately

SYSTEM = ("You are a payments operations analyst working one case. Answer only from this "
          "conversation.")

THE_FACT = "The client contact for this case is Priya Raman on the Singapore desk."

def long_session(turns: int = 14) -> list:
    """A conversation whose first human turn carries the fact that matters."""
    msgs = [SystemMessage(SYSTEM), HumanMessage(f"Open the case for PMT-1005. {THE_FACT}")]
    for i in range(turns):
        msgs.append(AIMessage(f"Noted. Checking sanction screening batch {i}: "
                              + "reviewing the counterparty records in detail. " * 6))
        msgs.append(HumanMessage(f"And what about batch {i + 1}?"))
    return msgs


def window_fits(messages: list, budget: int) -> bool:
    """Does this conversation still fit the budget?"""
    return count_tokens_approximately(messages) <= budget
'''),
    code('''
# --- Self-check: Section 1   (counting only -- no model call)
BUDGET = 400

check("a short session fits",
      lambda: window_fits(long_session(0), BUDGET) is True)
check("a long session does not",
      lambda: window_fits(long_session(14), BUDGET) is False)
check("the count grows with the conversation",
      lambda: count_tokens_approximately(long_session(10))
              > count_tokens_approximately(long_session(2)))
check("the fact is in the session to begin with",
      lambda: any("Priya" in str(m.content) for m in long_session(14)),
      "everything below is about whether it is still there LATER")

def _first_overflow():
    for n in range(0, 20):
        if not window_fits(long_session(n), BUDGET):
            return n
    return None
guard(lambda: print(f"\\nthe window overflows at turn {_first_overflow()} "
                    f"on a {BUDGET}-token budget"))
'''),

    md("""
## Section 2 &mdash; Bound it, and keep what you drop

`trim_messages` bounds the window. On its own that is amnesia: the fact from turn one is simply
gone. So summarise what you are about to drop and put the summary back as a system message.

**The counter matters.** The obvious `token_counter=llm` raises `NotImplementedError` here &mdash;
`langchain-openai` can only count for models `tiktoken` has an encoding for, and a
gateway-served model is not one. Use `count_tokens_approximately`, a plain function over the text.
"""),
    code('''
from langchain_core.messages import trim_messages

def bounded(messages: list, budget: int = 400) -> list:
    """Keep the system message and as many recent turns as fit."""
    return trim_messages(
        messages, max_tokens=budget,
        token_counter=BLANK,          # TODO: which counter works for a gateway-served model?
        strategy="last", include_system=True, start_on="human", allow_partial=False)


def compact(messages: list, budget: int = 400) -> list:
    """Trim, then put a summary of what was dropped back in front of what survived."""
    kept = bounded(messages, budget)
    kept_ids = {id(m) for m in kept}
    dropped = [m for m in messages if id(m) not in kept_ids and m.type != "system"]
    if not dropped:
        return kept
    summary = summarise(dropped)
    head = [m for m in kept if m.type == "system"]
    tail = [m for m in kept if m.type != "system"]
    return head + [SystemMessage("Earlier in this case: " + summary)] + tail


def summarise(dropped: list) -> str:
    """One line covering the turns that are about to be discarded."""
    if not llm_ready():
        return " ".join(str(m.content) for m in dropped)[:300]
    joined = "\\n".join(f"{m.type}: {m.content}" for m in dropped)[:4000]
    return ask("Summarise these earlier turns in two sentences. Preserve every proper noun, "
               "reference number and named person exactly.\\n\\n" + joined)
''', '''
from langchain_core.messages import trim_messages

def bounded(messages: list, budget: int = 400) -> list:
    """Keep the system message and as many recent turns as fit."""
    return trim_messages(
        messages, max_tokens=budget,
        token_counter=count_tokens_approximately,   # a plain function over the message text
        strategy="last", include_system=True, start_on="human", allow_partial=False)


def compact(messages: list, budget: int = 400) -> list:
    """Trim, then put a summary of what was dropped back in front of what survived."""
    kept = bounded(messages, budget)
    kept_ids = {id(m) for m in kept}
    dropped = [m for m in messages if id(m) not in kept_ids and m.type != "system"]
    if not dropped:
        return kept
    summary = summarise(dropped)
    head = [m for m in kept if m.type == "system"]
    tail = [m for m in kept if m.type != "system"]
    return head + [SystemMessage("Earlier in this case: " + summary)] + tail


def summarise(dropped: list) -> str:
    """One line covering the turns that are about to be discarded."""
    if not llm_ready():
        return " ".join(str(m.content) for m in dropped)[:300]
    joined = "\\n".join(f"{m.type}: {m.content}" for m in dropped)[:4000]
    return ask("Summarise these earlier turns in two sentences. Preserve every proper noun, "
               "reference number and named person exactly.\\n\\n" + joined)
'''),
    code('''
# --- Self-check: Section 2   (trim_messages is pure; summarise degrades offline)
_long = long_session(14)

check("trimming bounds the window",
      lambda: window_fits(bounded(_long), 420) is True,
      "trim_messages needs a token_counter it can actually call on this model")
check("trimming really dropped turns",
      lambda: len(bounded(_long)) < len(_long))
check("the system message survives",
      lambda: bounded(_long)[0].type == "system",
      "include_system=True -- dropping the instructions is the worst possible trim")
check("what survives is the END of the conversation",
      lambda: bounded(_long)[-1].content == _long[-1].content,
      \'strategy="last" keeps recent turns; "first" would keep the stale ones\')
check("plain trimming LOSES the fact from turn one",
      lambda: not any("Priya" in str(m.content) for m in bounded(_long)),
      "this is the point of the section -- a bounded window is amnesia unless you do more")
check("compaction puts a summary back",
      lambda: sum(1 for m in compact(_long) if m.type == "system") == 2,
      "one system message for the instructions, one for what was dropped")
check("a short session is left alone",
      lambda: len(compact(long_session(0))) == len(long_session(0)))
'''),

    md("""
## Section 3 &mdash; Or: let a checkpointer do it

Everything above is what you write when you are managing the message list yourself. Attach a
**checkpointer** to `create_agent` and the history is stored for you, keyed by `thread_id` &mdash;
across turns, across restarts, and separately per case.

The two approaches are not rivals. The checkpointer decides *where the history lives*; trimming
decides *how much of it you send*. Real systems do both.
"""),
    code('''
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

@tool
def lookup_payment(ref: str) -> str:
    """Return the ledger record for one payment reference such as 'PMT-1005'."""
    rec = LEDGER.get(ref)
    return json.dumps({"ref": ref, **rec}) if rec else f"no payment found with reference {ref!r}"

def remembering_agent():
    """An agent whose history is kept for it, per thread."""
    return create_agent(model=get_llm(), tools=[lookup_payment], system_prompt=SYSTEM,
                        checkpointer=InMemorySaver())

def thread(case_ref: str) -> dict:
    """The config that selects which conversation this call belongs to."""
    return {"configurable": {"thread_id": BLANK}}   # TODO: what separates one case from another?
''', '''
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

@tool
def lookup_payment(ref: str) -> str:
    """Return the ledger record for one payment reference such as 'PMT-1005'."""
    rec = LEDGER.get(ref)
    return json.dumps({"ref": ref, **rec}) if rec else f"no payment found with reference {ref!r}"

def remembering_agent():
    """An agent whose history is kept for it, per thread."""
    return create_agent(model=get_llm(), tools=[lookup_payment], system_prompt=SYSTEM,
                        checkpointer=InMemorySaver())

def thread(case_ref: str) -> dict:
    """The config that selects which conversation this call belongs to."""
    return {"configurable": {"thread_id": case_ref}}
'''),
    code('''
# --- Self-check: Section 3   (config shape only -- no model call)
check("the thread config has the shape LangGraph expects",
      lambda: set(thread("PMT-1005")) == {"configurable"})
check("the thread is keyed by the case",
      lambda: thread("PMT-1005")["configurable"]["thread_id"] == "PMT-1005")
check("two cases get two threads",
      lambda: thread("PMT-1005") != thread("PMT-1003"),
      "one thread_id for everything is how one client sees another client's case")
'''),

    md("""
## Run it for real &mdash; forgetting, and not forgetting
"""),
    code('''
if llm_ready():
    def _forget():
        session = long_session(14)
        question = "Who is the client contact for this case?"

        naive = bounded(session) + [HumanMessage(question)]
        kept  = compact(session)  + [HumanMessage(question)]

        print("--- trimmed only ---")
        print(get_llm().invoke(naive).content[:220])
        print("\\n--- trimmed, with a summary of what was dropped ---")
        print(get_llm().invoke(kept).content[:220])
        print("\\nsummary that was carried forward:")
        print("  " + next(m.content for m in kept if "Earlier in this case" in str(m.content))[:300])
    guard(_forget)
'''),
    md("""
## Run it for real &mdash; the checkpointer
"""),
    code('''
if llm_ready():
    def _threads():
        agent = remembering_agent()
        agent.invoke({"messages": [HumanMessage("Open PMT-1005. The contact is Priya Raman.")]},
                     thread("PMT-1005"))
        agent.invoke({"messages": [HumanMessage("Open PMT-1003. The contact is Lee Wei.")]},
                     thread("PMT-1003"))

        for ref in ("PMT-1005", "PMT-1003"):
            out = agent.invoke({"messages": [HumanMessage("Who is the contact for this case?")]},
                               thread(ref))
            print(f"{ref}: {out['messages'][-1].content[:120]}")
            print(f"          {len(out['messages'])} messages on this thread")
    guard(_threads)
'''),
    md("""
### Read it

**The first pair.** Trimming alone answers the contact question wrongly or not at all &mdash; Priya
Raman fell out of the window twelve turns ago. Trimming *with a summary* still has her, in one
line instead of twenty. That is the difference between compaction and amnesia, and it is why
`summarise` is told to preserve proper nouns exactly: a summary that paraphrases names has thrown
away the only part that mattered.

**The second pair.** Two threads, two histories, no code of yours managing either. Each `thread_id`
is a separate conversation, and the agent answered each from its own. Note what would happen if
`thread()` returned a constant: every case would append to one history, and the second client's
contact would be answered with the first client's name. That is not a hypothetical bug &mdash; it is
the single commonest way agent memory leaks between users.

Neither approach removes the need for the other. The checkpointer stores everything forever;
`trim_messages` decides how much of it you pay to send on this turn.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Put the two together: trim the messages the checkpointed agent sends without deleting them
   from the thread. (`create_agent` takes middleware for this; failing that, trim before you
   pass them in.) Confirm the thread still has the full history afterwards.
2. Change `summarise` to drop the "preserve every proper noun" instruction and re-run. Count how
   many turns it takes before the contact's name is gone. That sentence is the whole safeguard.
3. Swap `InMemorySaver` for `SqliteSaver` writing to a file under `WORK`, restart the kernel, and
   ask the follow-up question again. Lab 3.4 is about what that buys you.
"""),
]


# =========================================================================== #
# Lab 3.2 -- perception: turning raw output into an observation
# =========================================================================== #
LAB2 = [
    header(2, "Perception: Raw Output Is Not an Observation", "Intermediate &rarr; Advanced", 35,
           ["Turn an opaque upstream record into a typed observation with a schema",
            "Distinguish the four kinds of &ldquo;nothing&rdquo; a tool can return",
            "Stamp in what the agent cannot see &mdash; time, authority, provenance",
            "Watch the same model answer well and badly on the same facts"],
           "> **Builds on Lab 3.1.** Memory decides what the agent still knows. Perception decides\n"
           "> what it ever knew in the first place."),
    setup(2),
    code(DOMAIN),

    md("""
## Concept

An agent does not see the world. It sees whatever your tool returned, rendered as text. Handing
a model a raw upstream payload and hoping is the commonest reason a competent agent gives an
incompetent answer.

**Perception** is the step between: decode the record, resolve the codes, add what the model
cannot know, and say plainly what is missing. `with_structured_output` gives you somewhere to put
the result that is checkable.
"""),

    md("""
## Section 1 &mdash; Decode the opaque record

This is what the upstream ledger actually returns. Every field is a code, an epoch, or a flag.
"""),
    code('''
from pydantic import BaseModel, Field
from typing import Literal, Optional

RAW = {
    "id": "PMT-1005",
    "amt": 75000000,                  # minor units
    "cur": 840,                       # ISO 4217 numeric
    "st": 3,                          # 1 settled, 2 failed, 3 held
    "rc": "SR",                       # abbreviated reason code
    "vd": 1788393600,                 # value date, epoch seconds (2026-09-03)
    "cp": "NORTHWIND",
}

CURRENCIES = {840: "USD", 978: "EUR", 826: "GBP"}
STATUSES = {1: "settled", 2: "failed", 3: "held"}
REASONS = {"IF": "INSUFFICIENT_FUNDS", "LB": "LIMIT_BREACH",
           "II": "INVALID_IBAN", "SR": "SANCTIONS_REVIEW"}


class Observation(BaseModel):
    """What the agent is actually told about one payment."""
    ref: str
    amount: float = Field(description="In major units, not minor")
    currency: str = Field(description="Three-letter code, e.g. USD")
    status: Literal["settled", "failed", "held"]
    reason_code: Optional[str] = Field(description="Expanded reason code, or None")
    value_date: str = Field(description="ISO date, e.g. 2026-09-03")


def perceive(raw: dict) -> Observation:
    """Turn the upstream record into something a model can reason about."""
    return Observation(
        ref=raw["id"],
        amount=BLANK,                 # TODO: minor units -> major units
        currency=CURRENCIES.get(raw["cur"], f"UNKNOWN({raw['cur']})"),
        status=STATUSES[raw["st"]],
        reason_code=REASONS.get(raw["rc"]),
        value_date=time.strftime("%Y-%m-%d", time.gmtime(raw["vd"])),
    )
''', '''
from pydantic import BaseModel, Field
from typing import Literal, Optional

RAW = {
    "id": "PMT-1005",
    "amt": 75000000,                  # minor units
    "cur": 840,                       # ISO 4217 numeric
    "st": 3,                          # 1 settled, 2 failed, 3 held
    "rc": "SR",                       # abbreviated reason code
    "vd": 1788393600,                 # value date, epoch seconds (2026-09-03)
    "cp": "NORTHWIND",
}

CURRENCIES = {840: "USD", 978: "EUR", 826: "GBP"}
STATUSES = {1: "settled", 2: "failed", 3: "held"}
REASONS = {"IF": "INSUFFICIENT_FUNDS", "LB": "LIMIT_BREACH",
           "II": "INVALID_IBAN", "SR": "SANCTIONS_REVIEW"}


class Observation(BaseModel):
    """What the agent is actually told about one payment."""
    ref: str
    amount: float = Field(description="In major units, not minor")
    currency: str = Field(description="Three-letter code, e.g. USD")
    status: Literal["settled", "failed", "held"]
    reason_code: Optional[str] = Field(description="Expanded reason code, or None")
    value_date: str = Field(description="ISO date, e.g. 2026-09-03")


def perceive(raw: dict) -> Observation:
    """Turn the upstream record into something a model can reason about."""
    return Observation(
        ref=raw["id"],
        amount=raw["amt"] / 100,      # the ledger speaks in cents; the model does not
        currency=CURRENCIES.get(raw["cur"], f"UNKNOWN({raw['cur']})"),
        status=STATUSES[raw["st"]],
        reason_code=REASONS.get(raw["rc"]),
        value_date=time.strftime("%Y-%m-%d", time.gmtime(raw["vd"])),
    )
'''),
    code('''
# --- Self-check: Section 1   (pure decoding -- no model call)
check("the amount is in major units",
      lambda: perceive(RAW).amount == 750000.0,
      "75000000 minor units is USD 750,000 -- a model told 75000000 will reason about the wrong number")
check("the currency code is resolved",  lambda: perceive(RAW).currency == "USD")
check("the status is resolved",         lambda: perceive(RAW).status == "held")
check("the reason code is expanded",    lambda: perceive(RAW).reason_code == "SANCTIONS_REVIEW",
      \'"SR" means nothing to a model; SANCTIONS_REVIEW appears in the policy catalogue\')
check("the epoch is a readable date",   lambda: perceive(RAW).value_date.startswith("2026-09"))
def _rejects_bad_status():
    try:
        Observation(ref="X", amount=1.0, currency="USD", status="pending",
                    reason_code=None, value_date="2026-09-03")
        return False
    except Exception:
        return True

check("an invalid status is rejected by the schema",
      lambda: _rejects_bad_status(),
      "Literal[...] means a decoding bug fails here, not three steps later in a policy lookup")
'''),

    md("""
## Section 2 &mdash; The four kinds of nothing

An empty result is not one thing. "No such payment", "no permission to see it", "the upstream is
down" and "it exists and has no reason code" all arrive as something falsy, and they require four
different responses. Collapsing them is how an agent confidently reports that a payment does not
exist when it simply could not read it.
"""),
    code('''
def describe_empty(result: dict) -> str:
    """Say which kind of nothing this is."""
    if result.get("error") == "not_found":
        return "no such payment exists"
    if result.get("error") == "forbidden":
        return BLANK                  # TODO: the agent must NOT conclude the payment is absent
    if result.get("error") in ("timeout", "unavailable"):
        return "could not read the ledger; state unknown"
    if result.get("record") is not None and not result["record"].get("rc"):
        return "the payment exists and has no reason code"
    return "unrecognised result shape"
''', '''
def describe_empty(result: dict) -> str:
    """Say which kind of nothing this is."""
    if result.get("error") == "not_found":
        return "no such payment exists"
    if result.get("error") == "forbidden":
        return "not permitted to read this payment; existence unknown"
    if result.get("error") in ("timeout", "unavailable"):
        return "could not read the ledger; state unknown"
    if result.get("record") is not None and not result["record"].get("rc"):
        return "the payment exists and has no reason code"
    return "unrecognised result shape"
'''),
    code('''
# --- Self-check: Section 2
check("a missing record says so",
      lambda: "no such payment" in describe_empty({"error": "not_found"}))
check("a permission failure does NOT claim the payment is missing",
      lambda: "no such payment" not in describe_empty({"error": "forbidden"}),
      "this is the dangerous one: 'I cannot see it' is not 'it is not there'")
check("a permission failure says existence is unknown",
      lambda: "unknown" in describe_empty({"error": "forbidden"}).lower())
check("an outage says state unknown",
      lambda: "unknown" in describe_empty({"error": "timeout"}).lower())
check("a real record with no reason code is not an error",
      lambda: "exists" in describe_empty({"record": {"id": "PMT-1001"}}))
check("all four kinds give different answers",
      lambda: len({describe_empty({"error": e}) for e in
                   ("not_found", "forbidden", "timeout")}) == 3)
'''),

    md("""
## Section 3 &mdash; Stamp in what the agent cannot see

The model has no clock, no idea who is asking, and no memory of where a fact came from. If those
matter to the decision &mdash; and here they do &mdash; they have to be in the observation.
"""),
    code('''
NOW = 1788393600 + 7200               # pretend "now" is two hours after the value date

class Context(BaseModel):
    """Everything true of the situation rather than of the payment."""
    observed_at: str = Field(description="ISO timestamp when this was read")
    hours_since_value_date: float
    source: str = Field(description="Which system this came from, for provenance")
    caller_may_release: bool = Field(description="Whether the human asking has release authority")


def contextualise(obs: Observation, raw: dict, caller_role: str) -> Context:
    return Context(
        observed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(NOW)),
        hours_since_value_date=round((NOW - raw["vd"]) / 3600, 1),
        source="ledger-core",
        caller_may_release=BLANK,     # TODO: which roles may release? see NEEDS_HUMAN and the policy
    )
''', '''
NOW = 1788393600 + 7200               # pretend "now" is two hours after the value date

class Context(BaseModel):
    """Everything true of the situation rather than of the payment."""
    observed_at: str = Field(description="ISO timestamp when this was read")
    hours_since_value_date: float
    source: str = Field(description="Which system this came from, for provenance")
    caller_may_release: bool = Field(description="Whether the human asking has release authority")


def contextualise(obs: Observation, raw: dict, caller_role: str) -> Context:
    return Context(
        observed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(NOW)),
        hours_since_value_date=round((NOW - raw["vd"]) / 3600, 1),
        source="ledger-core",
        # Operations may release ordinary holds but never one policy reserves for a human
        # decision -- a sanctions review is Compliance's call whoever is asking.
        caller_may_release=(caller_role == "compliance"
                            or (caller_role == "operations"
                                and obs.reason_code not in NEEDS_HUMAN)),
    )
'''),
    code('''
# --- Self-check: Section 3
# built lazily: perceive() may still contain a blank, and a module-level call would
# crash this cell instead of reporting [TODO]
_obs   = lambda: perceive(RAW)                       # PMT-1005, SANCTIONS_REVIEW
_clean = lambda: perceive({**RAW, "rc": "IF"})       # same payment, an ordinary failure

check("the observation is stamped with a time",
      lambda: contextualise(_obs(), RAW, "operations").observed_at.startswith("2026-"))
check("elapsed time is computed, not left to the model",
      lambda: contextualise(_obs(), RAW, "operations").hours_since_value_date == 2.0,
      "a model asked to subtract two epochs will sometimes get it wrong; do it in Python")
check("operations may NOT release a sanctions hold",
      lambda: contextualise(_obs(), RAW, "operations").caller_may_release is False,
      "policy reserves this decision for Compliance -- authority is context, not preference")
check("operations MAY release an ordinary failure",
      lambda: contextualise(_clean(), RAW, "operations").caller_may_release is True)
check("compliance may release a sanctions hold",
      lambda: contextualise(_obs(), RAW, "compliance").caller_may_release is True)
check("an unknown role gets no authority",
      lambda: contextualise(_clean(), RAW, "intern").caller_may_release is False,
      "default deny -- an unrecognised role is not a permitted one")
'''),

    md("""
## Run it for real

The same model, the same underlying payment, asked the same question. Once from the raw record,
once from the observation and its context.
"""),
    code('''
if llm_ready():
    def _compare():
        question = ("Who must action this payment, and may the operations desk release it? "
                    "Answer in two lines.")
        obs = perceive(RAW)
        ctx = contextualise(obs, RAW, "operations")
        policy = POLICY.get(obs.reason_code, "no policy applies")

        print("=== given the RAW record ===")
        print(ask(f"RECORD: {json.dumps(RAW)}\\n\\n{question}")[:400])

        print("\\n=== given the OBSERVATION and its context ===")
        print(ask(f"OBSERVATION: {obs.model_dump_json()}\\n"
                  f"CONTEXT: {ctx.model_dump_json()}\\n"
                  f"POLICY: {policy}\\n\\n{question}")[:400])
    guard(_compare)
'''),
    md("""
### Read it

The raw answer is the interesting one. The model has to guess that `st: 3` means held, that `rc`
is a reason code at all, that `amt` is in cents, and that `cur: 840` is dollars. Most of the time
it will guess several of those correctly and state the rest with complete confidence &mdash; which is
exactly the failure you cannot detect downstream, because nothing looks wrong.

The second answer is not smarter. It is the same model, given the same facts, in a form it does
not have to decode. Note in particular that `caller_may_release` was decided by **your code**,
against the policy, before the model saw anything. Asking a model to work out who is authorised
is asking it to make a control decision; computing it in `contextualise` and telling it the answer
is not.

That is the general rule this lab is for: **anything you can determine in Python, determine in
Python.** Leave the model the part that actually needs judgement.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add `cur: 392` (JPY) to `CURRENCIES` &mdash; but the yen has no minor units, so `amt` is already
   in major units. Where does that belong: in `perceive`, in the schema, or in the tool that
   produced the record? Defend your answer.
2. Feed `describe_empty` into the run-it-for-real cell: ask the model what to do when the ledger
   returns `{"error": "forbidden"}`, once with the raw error and once with your description. Watch
   how often the raw version concludes the payment does not exist.
3. `Observation` has no field for what is **missing**. Add `unknown: list[str]` and populate it
   when a code fails to resolve. An agent that can say "I do not know the currency" is worth more
   than one that quietly says `UNKNOWN(392)`.
"""),
]


# =========================================================================== #
# Lab 3.3 -- build a real StateGraph
# =========================================================================== #
GRAPH_TOOLS = '''
# ------------------------------------------------- the case tools, carried through 3.3 - 3.5
def read_ledger_record(ref: str) -> dict:
    rec = LEDGER.get(ref)
    return {"ref": ref, **rec} if rec else {"ref": ref, "error": "not_found"}

def read_policy_text(reason_code: str | None) -> str:
    return POLICY.get(reason_code, "no policy applies")

print("case helpers loaded")
'''

LAB3 = [
    header(3, "Build a Real StateGraph", "Advanced", 45,
           ["Declare state as a <code>TypedDict</code> with <code>Annotated</code> reducers",
            "Write nodes that return <i>partial</i> state, and let the reducers merge it",
            "Wire edges, a conditional edge and a cycle, then <code>compile()</code>",
            "Watch it run with <code>stream()</code>, one node at a time"],
           "> **Builds on Lab 3.1.** A checkpointer gave you memory. A graph gives you a state you\n"
           "> can declare, inspect and reason about &mdash; which is what Modules 5 and 9 build on."),
    setup(3),
    code(DOMAIN),
    code(GRAPH_TOOLS),

    md("""
## Concept

`create_agent` is one fixed loop: model, tools, repeat. When the control flow is yours &mdash; branch
here, loop there, stop for a human &mdash; you need the graph underneath it.

A LangGraph `StateGraph` has three parts and no more:

| Part | What it is |
|---|---|
| **state** | a `TypedDict`; each field may carry a **reducer** saying how updates merge |
| **nodes** | plain functions `state -> partial state` |
| **edges** | fixed (`add_edge`) or chosen at runtime (`add_conditional_edges`) |

Then `compile()` gives you a runnable with the same `.invoke()` / `.stream()` interface as
everything else you have built today.

The part people get wrong is the reducer, so start there.
"""),

    md("""
## Section 1 &mdash; State, and how updates merge

By default a node's return value **overwrites** the field. That is right for `answer` and wrong
for `findings` &mdash; you want those to accumulate. `Annotated[list, add]` says "append, do not
replace".
"""),
    code('''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

def accumulate(old: list, new: list) -> list:
    """The reducer for `findings`: how one node's update merges with what is already there."""
    return BLANK                                # TODO: combine them so BOTH nodes' findings survive


class CaseState(TypedDict):
    ref: str                                    # set once, overwritten if a node returns it
    findings: Annotated[list, accumulate]       # merged by your reducer, not replaced
    steps: int
    needs_human: bool
    answer: str | None
''', '''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

def accumulate(old: list, new: list) -> list:
    """The reducer for `findings`: how one node's update merges with what is already there."""
    return old + new                            # `operator.add` is exactly this, and is the idiom


class CaseState(TypedDict):
    ref: str                                    # set once, overwritten if a node returns it
    findings: Annotated[list, accumulate]       # merged by your reducer, not replaced
    steps: int
    needs_human: bool
    answer: str | None
'''),
    code('''
# --- Self-check: Section 1   (the reducer, exercised through a real one-node graph -- no model)
def _reducer_of(field):
    """The reducer LangGraph will use for one field of CaseState."""
    from typing import get_type_hints
    hints = get_type_hints(CaseState, include_extras=True)
    meta = getattr(hints[field], "__metadata__", ())
    return meta[0] if meta else None

def _two_appends():
    """Two nodes, each returning one finding. Does the state end up with both?"""
    g = StateGraph(CaseState)
    g.add_node("a", lambda s: {"findings": ["from a"], "steps": 1})
    g.add_node("b", lambda s: {"findings": ["from b"], "steps": 2})
    g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", END)
    return g.compile().invoke({"ref": "PMT-1005", "findings": [], "steps": 0,
                               "needs_human": False, "answer": None})

check("findings declares a reducer",
      lambda: _reducer_of("findings") is not None,
      "without one, the second node's findings replace the first node's")
check("the reducer merges two updates instead of replacing",
      lambda: accumulate(["a"], ["b"]) == ["a", "b"],
      "returning just `new` is the bug this whole section exists to prevent")
check("two nodes' findings both survive",
      lambda: _two_appends()["findings"] == ["from a", "from b"],
      "this is what the reducer is FOR -- run it and see")
check("a field with no reducer is overwritten",
      lambda: _two_appends()["steps"] == 2,
      "steps has no reducer, so the last write wins -- that is the default, and it is fine here")
check("the untouched fields are still there",
      lambda: _two_appends()["ref"] == "PMT-1005")
'''),

    md("""
## Section 2 &mdash; Nodes return partial state

A node receives the whole state and returns **only the keys it changed**. LangGraph merges the
rest for you, using the reducers from Section 1. Returning the whole state is the other common
beginner mistake: it works, until two nodes run and one silently undoes the other.
"""),
    code('''
def read_ledger(state: CaseState) -> dict:
    """Look the payment up and record what we found."""
    rec = read_ledger_record(state["ref"])
    return {"findings": [f"ledger: status={rec.get('status')} rc={rec.get('reason_code')}"],
            "steps": state["steps"] + 1,
            "needs_human": rec.get("reason_code") in NEEDS_HUMAN}


def read_policy(state: CaseState) -> dict:
    """Look up the policy for whatever reason code the ledger gave us."""
    rec = read_ledger_record(state["ref"])
    return {"findings": [f"policy: {read_policy_text(rec.get('reason_code'))}"],
            "steps": state["steps"] + 1}


def write_note(state: CaseState) -> dict:
    """Compose the answer from what is in state -- and nothing else."""
    who = "a human must decide" if state["needs_human"] else "operations may act"
    return {"answer": BLANK}          # TODO: an answer built from the findings and `who`
''', '''
def read_ledger(state: CaseState) -> dict:
    """Look the payment up and record what we found."""
    rec = read_ledger_record(state["ref"])
    return {"findings": [f"ledger: status={rec.get('status')} rc={rec.get('reason_code')}"],
            "steps": state["steps"] + 1,
            "needs_human": rec.get("reason_code") in NEEDS_HUMAN}


def read_policy(state: CaseState) -> dict:
    """Look up the policy for whatever reason code the ledger gave us."""
    rec = read_ledger_record(state["ref"])
    return {"findings": [f"policy: {read_policy_text(rec.get('reason_code'))}"],
            "steps": state["steps"] + 1}


def write_note(state: CaseState) -> dict:
    """Compose the answer from what is in state -- and nothing else."""
    who = "a human must decide" if state["needs_human"] else "operations may act"
    return {"answer": f"{state['ref']}: {who}. " + " | ".join(state["findings"])}
'''),
    code('''
# --- Self-check: Section 2   (nodes are plain functions -- call them directly, no model)
_s = {"ref": "PMT-1005", "findings": [], "steps": 0, "needs_human": False, "answer": None}

check("a node returns only what it changed",
      lambda: set(read_ledger(_s)) == {"findings", "steps", "needs_human"},
      "returning the whole state is how one node silently undoes another")
check("the ledger node finds the reason code",
      lambda: "SANCTIONS_REVIEW" in read_ledger(_s)["findings"][0])
check("it sets needs_human for a sanctions hold",
      lambda: read_ledger(_s)["needs_human"] is True)
check("...and not for an ordinary failure",
      lambda: read_ledger({**_s, "ref": "PMT-1002"})["needs_human"] is False)
check("the policy node returns the policy text",
      lambda: "Compliance" in read_policy(_s)["findings"][0])
check("write_note uses the findings it was given",
      lambda: "ledger:" in write_note({**_s, "findings": ["ledger: x"], "needs_human": True})["answer"])
check("write_note says who decides",
      lambda: "human" in write_note({**_s, "findings": ["x"], "needs_human": True})["answer"])
'''),

    md("""
## Section 3 &mdash; Edges, a condition, and a cycle

`add_edge(a, b)` always goes to `b`. `add_conditional_edges(a, fn, mapping)` calls `fn(state)` and
goes wherever it says. A cycle is just an edge that points backwards &mdash; which is why the step
budget is not optional.
"""),
    code('''
MAX_STEPS = 6

def enough(state: CaseState) -> str:
    """Have we gathered enough to answer? Returns the KEY of the next branch."""
    if state["steps"] >= MAX_STEPS:
        return "write_note"
    return "write_note" if len(state["findings"]) >= 2 else "read_ledger"


def build_graph():
    g = StateGraph(CaseState)
    g.add_node("read_ledger", read_ledger)
    g.add_node("read_policy", read_policy)
    g.add_node("write_note", write_note)

    g.add_edge(START, "read_ledger")
    g.add_edge("read_ledger", "read_policy")
    g.add_conditional_edges("read_policy", enough,
                            {"write_note": "write_note",
                             "read_ledger": BLANK})     # TODO: where does "not enough yet" go?
    g.add_edge("write_note", END)
    return g.compile()


def fresh(ref: str) -> dict:
    return {"ref": ref, "findings": [], "steps": 0, "needs_human": False, "answer": None}
''', '''
MAX_STEPS = 6

def enough(state: CaseState) -> str:
    """Have we gathered enough to answer? Returns the KEY of the next branch."""
    if state["steps"] >= MAX_STEPS:
        return "write_note"
    return "write_note" if len(state["findings"]) >= 2 else "read_ledger"


def build_graph():
    g = StateGraph(CaseState)
    g.add_node("read_ledger", read_ledger)
    g.add_node("read_policy", read_policy)
    g.add_node("write_note", write_note)

    g.add_edge(START, "read_ledger")
    g.add_edge("read_ledger", "read_policy")
    g.add_conditional_edges("read_policy", enough,
                            {"write_note": "write_note",
                             "read_ledger": "read_ledger"})   # the backward edge -- the cycle
    g.add_edge("write_note", END)
    return g.compile()


def fresh(ref: str) -> dict:
    return {"ref": ref, "findings": [], "steps": 0, "needs_human": False, "answer": None}
'''),
    code('''
# --- Self-check: Section 3   (a REAL compiled graph, running -- still no model)
def _run(ref="PMT-1005"):
    return build_graph().invoke(fresh(ref))

check("the graph compiles",              lambda: build_graph() is not None)
check("it runs to an answer",            lambda: _run()["answer"] is not None)
check("both nodes contributed findings", lambda: len(_run()["findings"]) >= 2)
check("the reducer accumulated them",    lambda: any("ledger:" in f for f in _run()["findings"])
                                                 and any("policy:" in f for f in _run()["findings"]))
check("the sanctions case needs a human",
      lambda: _run("PMT-1005")["needs_human"] is True)
check("an ordinary failure does not",
      lambda: _run("PMT-1002")["needs_human"] is False)
check("the cycle terminates",            lambda: _run()["steps"] <= MAX_STEPS,
      "a backward edge with no budget is an infinite loop, and the graph will happily run it")
check("the conditional edge can actually loop",
      lambda: enough({"steps": 0, "findings": []}) == "read_ledger",
      "if both branches go forward you have written a straight line, not a cycle")
'''),

    md("""
## Watch it run

`stream()` yields one entry per node as it completes, which is the cheapest debugger you will
ever have for a graph.
"""),
    code('''
def _trace():
    for chunk in build_graph().stream(fresh("PMT-1005")):
        for node, update in chunk.items():
            print(f"  {node:14} -> {list(update)}")
    print("\\nfinal answer:")
    print("  " + str(build_graph().invoke(fresh("PMT-1005"))["answer"])[:300])
guard(_trace)
'''),

    md("""
## Run it for real &mdash; put the model in a node

Nothing so far needed a model, which is the point: **the graph is deterministic scaffolding, and
you can test all of it offline.** Now add one node that does need one. Note that it reads only
from state, and returns only a partial update &mdash; exactly like the others.
"""),
    code('''
if llm_ready():
    def _with_model():
        def draft_note(state: CaseState) -> dict:
            who = "a human must decide" if state["needs_human"] else "operations may act"
            text = ask("Write one line telling the operations desk what happens next. "
                       "Use only these facts; do not add any.\\n"
                       f"CASE: {state['ref']}\\nAUTHORITY: {who}\\n"
                       f"FINDINGS: {state['findings']}")
            return {"answer": text.strip()}

        g = StateGraph(CaseState)
        g.add_node("read_ledger", read_ledger)
        g.add_node("read_policy", read_policy)
        g.add_node("draft_note", draft_note)
        g.add_edge(START, "read_ledger")
        g.add_edge("read_ledger", "read_policy")
        g.add_conditional_edges("read_policy", lambda s: "draft_note" if len(s["findings"]) >= 2
                                else "read_ledger",
                                {"draft_note": "draft_note", "read_ledger": "read_ledger"})
        g.add_edge("draft_note", END)

        app = g.compile()
        for ref in ("PMT-1005", "PMT-1002"):
            out = app.invoke(fresh(ref))
            print(f"{ref}: needs_human={out['needs_human']}")
            print(f"          {out['answer'][:200]}\\n")
    guard(_with_model)
'''),
    md("""
### Read it

Three things worth taking away.

1. **The graph is testable without a model.** Every self-check in this lab ran a real compiled
   `StateGraph` and asserted on real merged state, offline and deterministically. That is not a
   trick of the lab &mdash; it is how you should test agent control flow generally. Put the model in
   one node, and everything around it stays ordinary software.
2. **The reducer is the design.** `findings` accumulates because you said so; `steps` overwrites
   because you did not. Get that wrong and the bug looks like "the second agent lost the first
   agent's work", which is Lab 3.5's subject.
3. **The cycle needs the budget.** `enough` checks `MAX_STEPS` before it checks anything else. A
   backward edge with no ceiling is an infinite loop that LangGraph will run for you, cheerfully,
   until something else stops it.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add a `needs_escalation` node that runs only when `needs_human` is true, and route to it with
   a second conditional edge. Confirm PMT-1002 never enters it.
2. Give `steps` the reducer `add` instead of leaving it to overwrite, and change the nodes to
   return `{"steps": 1}`. Which do you prefer, and what happens if two nodes ever run in
   parallel?
3. `read_policy` calls `read_ledger_record` again, because the reason code was never put in state.
   Add a `reason_code` field, set it in `read_ledger`, and read it in `read_policy`. That is the
   difference between passing state and re-fetching it &mdash; and it is exactly what Module 5's
   multi-agent graphs depend on.
"""),
]


# =========================================================================== #
# Lab 3.4 -- checkpointing: resume, approve, rewind, audit
# =========================================================================== #
CARRY_GRAPH = '''
# ------------------------------------------------- carried forward from Lab 3.3
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

class CaseState(TypedDict):
    ref: str
    findings: Annotated[list, add]
    steps: int
    needs_human: bool
    answer: str | None

def read_ledger(state: CaseState) -> dict:
    rec = read_ledger_record(state["ref"])
    return {"findings": [f"ledger: status={rec.get('status')} rc={rec.get('reason_code')}"],
            "steps": state["steps"] + 1,
            "needs_human": rec.get("reason_code") in NEEDS_HUMAN}

def read_policy(state: CaseState) -> dict:
    rec = read_ledger_record(state["ref"])
    return {"findings": [f"policy: {read_policy_text(rec.get('reason_code'))}"],
            "steps": state["steps"] + 1}

def write_note(state: CaseState) -> dict:
    who = "a human must decide" if state["needs_human"] else "operations may act"
    return {"answer": f"{state['ref']}: {who}. " + " | ".join(state["findings"])}

def fresh(ref: str) -> dict:
    return {"ref": ref, "findings": [], "steps": 0, "needs_human": False, "answer": None}

print("Lab 3.3 graph pieces loaded")
'''

LAB4 = [
    header(4, "Checkpointing: Resume, Approve, Rewind, Audit", "Advanced", 45,
           ["Attach a checkpointer and watch state survive a crash",
            "Stop the graph before an irreversible node with <code>interrupt_before</code>",
            "Resume, and rewind to an earlier checkpoint to try a different decision",
            "Read <code>get_state_history()</code> as the audit trail it is"],
           "> **Builds on Lab 3.3.** Same graph. The difference is that every step is now written\n"
           "> down, which is what makes approval, recovery and audit possible at all."),
    setup(4),
    code(DOMAIN),
    code(GRAPH_TOOLS),
    code(CARRY_GRAPH),

    md("""
## Concept

A checkpointer saves the state after **every node**, under a `thread_id`. Four capabilities fall
out of that one fact, and none of them is available without it:

| Capability | How |
|---|---|
| **resume** | re-invoke the same thread; it carries on where it stopped |
| **approve** | `interrupt_before` a node; the graph pauses, you decide, then resume |
| **rewind** | invoke from an *older* checkpoint's config and take a different branch |
| **audit** | `get_state_history()` &mdash; what was known, and when |

This is the mechanism behind human-in-the-loop, which Module 8 turns into a control.
"""),

    md("""
## Section 1 &mdash; A checkpointer and a thread

`compile(checkpointer=...)` is the whole change. Everything else is the config you pass at
call time.
"""),
    code('''
from langgraph.checkpoint.memory import InMemorySaver

def build(checkpointer=None, interrupt_before=None):
    """The Lab 3.3 graph, now compilable with persistence and an approval gate."""
    g = StateGraph(CaseState)
    g.add_node("read_ledger", read_ledger)
    g.add_node("read_policy", read_policy)
    g.add_node("write_note", write_note)
    g.add_edge(START, "read_ledger")
    g.add_edge("read_ledger", "read_policy")
    g.add_edge("read_policy", "write_note")
    g.add_edge("write_note", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}
'''),
    code('''
# --- Self-check: Section 1   (a real checkpointed graph -- no model)
def _saved():
    saver = InMemorySaver()
    app = build(checkpointer=saver)
    app.invoke(fresh("PMT-1005"), cfg("t1"))
    return app

check("a checkpointed graph still runs",
      lambda: _saved().get_state(cfg("t1")).values["answer"] is not None)
check("the state is readable after the run",
      lambda: _saved().get_state(cfg("t1")).values["ref"] == "PMT-1005")
check("there is a checkpoint per step, not just one",
      lambda: len(list(_saved().get_state_history(cfg("t1")))) >= 4,
      "START, then one after each of the three nodes")
check("a thread that was never run is empty",
      lambda: _saved().get_state(cfg("never")).values in ({}, None),
      "threads are independent -- that is what keeps two cases apart")
'''),

    md("""
## Section 2 &mdash; Pause before something irreversible

`interrupt_before=["write_note"]` stops the graph *before* that node runs and returns. The state
is saved; `get_state(...).next` tells you what it was about to do.

Resume by invoking the same thread with `None` as the input &mdash; which means "carry on", not
"start again".
"""),
    code('''
def start_with_gate(ref: str, thread: str, saver):
    """Run until the approval gate, then stop."""
    app = build(checkpointer=saver, interrupt_before=["write_note"])
    app.invoke(fresh(ref), cfg(thread))
    return app


def pending(app, thread: str) -> tuple:
    """What is this thread waiting to do?"""
    return app.get_state(cfg(thread)).next


def approve(app, thread: str):
    """Let it proceed. The input is None -- carry on, do not start again."""
    return app.invoke(BLANK, cfg(thread))   # TODO: what does "resume" pass as the input?
''', '''
def start_with_gate(ref: str, thread: str, saver):
    """Run until the approval gate, then stop."""
    app = build(checkpointer=saver, interrupt_before=["write_note"])
    app.invoke(fresh(ref), cfg(thread))
    return app


def pending(app, thread: str) -> tuple:
    """What is this thread waiting to do?"""
    return app.get_state(cfg(thread)).next


def approve(app, thread: str):
    """Let it proceed. The input is None -- carry on, do not start again."""
    return app.invoke(None, cfg(thread))
'''),
    code('''
# --- Self-check: Section 2   (a real interrupt and resume -- no model)
def _gated():
    saver = InMemorySaver()
    app = start_with_gate("PMT-1005", "gate1", saver)
    return app

check("the graph stopped before the gated node",
      lambda: pending(_gated(), "gate1") == ("write_note",))
check("it stopped BEFORE doing the thing",
      lambda: _gated().get_state(cfg("gate1")).values["answer"] is None,
      "interrupt_before means the node has not run -- that is what makes it an approval gate")
check("the work done so far was kept",
      lambda: len(_gated().get_state(cfg("gate1")).values["findings"]) == 2,
      "a pause is not a rollback")
check("resuming finishes the run",
      lambda: approve(_gated(), "gate1")["answer"] is not None)
def _finished_has_no_next():
    saver = InMemorySaver()
    app = start_with_gate("PMT-1005", "gate2", saver)
    approve(app, "gate2")
    return app.get_state(cfg("gate2")).next == ()

check("after resuming there is nothing pending",
      lambda: _finished_has_no_next())
'''),

    md("""
## Section 3 &mdash; Change your mind: update, and rewind

Two different operations, and the difference matters.

**`update_state`** writes into the *current* checkpoint &mdash; a human adding a fact before the graph
continues. It goes through the reducers, so an `Annotated[list, add]` field appends.

**Rewinding** means invoking from an *older* checkpoint's config. The graph replays from there,
and anything after it is superseded.
"""),
    code('''
def add_human_finding(app, thread: str, note: str):
    """A person adds something the tools could not know."""
    app.update_state(cfg(thread), {"findings": [f"human: {note}"]})
    return app.get_state(cfg(thread)).values


def checkpoint_before(app, thread: str, node: str):
    """The config of the checkpoint at which `node` was the next thing to run."""
    for snap in app.get_state_history(cfg(thread)):
        if snap.next == (node,):
            return BLANK              # TODO: what identifies that point in history?
    return None
''', '''
def add_human_finding(app, thread: str, note: str):
    """A person adds something the tools could not know."""
    app.update_state(cfg(thread), {"findings": [f"human: {note}"]})
    return app.get_state(cfg(thread)).values


def checkpoint_before(app, thread: str, node: str):
    """The config of the checkpoint at which `node` was the next thing to run."""
    for snap in app.get_state_history(cfg(thread)):
        if snap.next == (node,):
            return snap.config        # a config carrying that checkpoint_id, not just the thread
    return None
'''),
    code('''
# --- Self-check: Section 3   (real update_state and real history -- no model)
def _updated():
    saver = InMemorySaver()
    app = start_with_gate("PMT-1005", "upd", saver)
    values = add_human_finding(app, "upd", "Compliance confirmed the hold by phone.")
    return app, values

check("the human's note went into state",
      lambda: any("human:" in f for f in _updated()[1]["findings"]))
check("update_state APPENDS rather than replacing",
      lambda: len(_updated()[1]["findings"]) == 3,
      "it goes through the reducers -- Annotated[list, add] means the tool findings survive")
check("the graph is still paused at the gate",
      lambda: _updated()[0].get_state(cfg("upd")).next == ("write_note",),
      "adding a fact is not the same as approving")
check("the resumed answer contains the human's note",
      lambda: "human:" in approve(_updated()[0], "upd")["answer"])
check("checkpoint_before finds the right point in history",
      lambda: checkpoint_before(_updated()[0], "upd", "read_policy") is not None)
check("what it returns is a checkpoint, not just the thread",
      lambda: "checkpoint_id" in checkpoint_before(_updated()[0], "upd",
                                                   "read_policy")["configurable"],
      "a config with only a thread_id points at NOW, which is not a rewind")
'''),

    md("""
## Section 4 &mdash; The audit trail

`get_state_history()` returns every checkpoint, **newest first**. That is the record of what the
system knew and when it knew it &mdash; which is the question an auditor actually asks.
"""),
    code('''
def audit(app, thread: str) -> str:
    """The thread's history, oldest first, as something a person can read."""
    rows = []
    for snap in reversed(list(app.get_state_history(cfg(thread)))):
        rows.append(f"  next={str(snap.next):22} steps={snap.values.get('steps', 0)} "
                    f"findings={len(snap.values.get('findings', []))} "
                    f"answer={'set' if snap.values.get('answer') else '-'}")
    return "\\n".join(rows)
'''),
    code('''
# --- Self-check: Section 4
def _history():
    saver = InMemorySaver()
    app = build(checkpointer=saver)
    app.invoke(fresh("PMT-1005"), cfg("aud"))
    return app, list(app.get_state_history(cfg("aud")))

check("history has a checkpoint per node plus the start",
      lambda: len(_history()[1]) >= 4)
check("history is returned newest first",
      lambda: _history()[1][0].values.get("answer") is not None
              and _history()[1][-1].next == ("__start__",),
      "reverse it before you show it to a person")
check("the earliest checkpoint has no findings yet",
      lambda: len(_history()[1][-1].values.get("findings", [])) == 0)
check("the audit trail is readable",
      lambda: "next=" in audit(_history()[0], "aud"))
check("every checkpoint carries its own id",
      lambda: len({s.config["configurable"]["checkpoint_id"] for s in _history()[1]})
              == len(_history()[1]),
      "identical ids would mean you cannot address a point in the past")
'''),

    md("""
## Run it &mdash; crash, resume, approve, rewind
"""),
    code('''
def _demo():
    saver = InMemorySaver()

    print("--- 1. approval gate ---")
    app = start_with_gate("PMT-1005", "case", saver)
    print("   paused before:", pending(app, "case"))
    print("   answer so far:", app.get_state(cfg("case")).values["answer"])

    print("\\n--- 2. a human adds what the tools could not know ---")
    vals = add_human_finding(app, "case", "Compliance confirmed the hold by phone at 14:02.")
    for f in vals["findings"]:
        print("   " + f[:100])

    print("\\n--- 3. approve, and let it finish ---")
    out = approve(app, "case")
    print("   " + str(out["answer"])[:220])

    print("\\n--- 4. rewind to before read_policy and replay ---")
    back = checkpoint_before(app, "case", "read_policy")
    if back:
        replayed = app.invoke(None, back)
        print("   replayed from an earlier checkpoint; next =",
              app.get_state(cfg("case")).next)
        print("   answer:", str(replayed.get("answer"))[:160])

    print("\\n--- 5. the audit trail ---")
    print(audit(app, "case"))
guard(_demo)
'''),
    md("""
### Read it

**Step 4 is the one to look at twice.** Rewinding replays the graph from an older checkpoint &mdash;
and because this graph was compiled with `interrupt_before=["write_note"]`, the replay runs
forward and then **stops at the gate again**. It does not sail through to a new answer. That is
correct, and it surprises people: the interrupt is a property of the compiled graph, not of a
particular run, so every path through it pauses. If you want the replay to finish, resume it
again.

**Step 5 is what you show an auditor.** Not the answer &mdash; the sequence. Each row is a point at
which the system had a definite set of findings and had not yet done the next thing. The question
"what did it know when it decided?" has an exact answer, and the human's note appears in it as a
finding like any other, tagged `human:` so provenance survives.

And note what made all of this available: one keyword argument. Everything in this lab is a
consequence of `compile(checkpointer=...)`. Without it, a crash loses the case, an approval gate
is impossible, and the audit question has no answer at all.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Swap `InMemorySaver` for `SqliteSaver` pointed at a file under `WORK`. Run the graph to the
   gate, restart the kernel, rebuild the app against the same file, and resume. That is recovery
   after a crash, and it is why in-memory is a development convenience only.
2. Gate on a **condition** rather than always: interrupt before `write_note` only when
   `needs_human` is true. (`interrupt_before` is static, so this belongs in a conditional edge to
   a node that interrupts.) Confirm PMT-1002 runs straight through.
3. Rewind, then use `update_state` to change `needs_human` to `False` before resuming. You have
   just overridden a control decision and left a record of doing so. Decide who in your
   organisation is allowed to make that call, and how the audit trail would show it.
"""),
]


# =========================================================================== #
# Lab 3.5 -- challenge: shared state, private state, and context poisoning
# =========================================================================== #
LAB5 = [
    header(5, "Challenge &mdash; Shared State and Context Poisoning", "Advanced", 45,
           ["Build a graph where three agents write into one shared state",
            "Introduce one wrong finding and watch it spread through the others",
            "Give each agent private state, and measure how far the damage gets",
            "Make every finding carry its source, so a wrong one can be traced and dropped"],
           "> **The take-home artifact.** The state design that Module 5's multi-agent graphs are\n"
           "> built on, and the failure it is designed to contain."),
    setup(5),
    code(DOMAIN),
    code(GRAPH_TOOLS),

    md("""
## Concept

When several agents share one state object, everything one writes is context for the next. That
is the point &mdash; it is what stops the fragmentation Lab 1.4 measured. It is also the risk:
**a wrong finding is indistinguishable from a right one**, and every agent downstream builds on it.

Two defences, and you need both:

- **provenance** &mdash; every finding records who produced it and from what, so a bad one can be
  identified and removed rather than argued with;
- **scope** &mdash; not everything an agent computes belongs in the shared state. Private working
  notes stay private.
"""),

    md("""
## Section 1 &mdash; A finding you can check

An unattributed string is not evidence. Make the shape carry its own provenance.
"""),
    code('''
from typing import Annotated, Optional
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

class Finding(BaseModel):
    """One claim, with enough attached to check it."""
    claim: str = Field(description="What is asserted, in one line")
    by: str = Field(description="Which agent produced it")
    source: str = Field(description="Which system or document it came from")
    ref: str = Field(description="The identifier within that source")

    def __str__(self) -> str:
        return f"[{self.by}/{self.source}:{self.ref}] {self.claim}"


def trustworthy(f: Finding, known_sources=("ledger", "policy")) -> bool:
    """A finding is checkable when we know where it came from and can go back to it."""
    return BLANK                      # TODO: a known source, AND a ref that is not empty
''', '''
from typing import Annotated, Optional
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

class Finding(BaseModel):
    """One claim, with enough attached to check it."""
    claim: str = Field(description="What is asserted, in one line")
    by: str = Field(description="Which agent produced it")
    source: str = Field(description="Which system or document it came from")
    ref: str = Field(description="The identifier within that source")

    def __str__(self) -> str:
        return f"[{self.by}/{self.source}:{self.ref}] {self.claim}"


def trustworthy(f: Finding, known_sources=("ledger", "policy")) -> bool:
    """A finding is checkable when we know where it came from and can go back to it."""
    return f.source in known_sources and bool(f.ref.strip())
'''),
    code('''
# --- Self-check: Section 1
_good    = Finding(claim="status is held", by="ledger_agent", source="ledger", ref="PMT-1005")
_no_ref  = Finding(claim="status is held", by="ledger_agent", source="ledger", ref="")
_hearsay = Finding(claim="Compliance already cleared it", by="critic",
                   source="recollection", ref="n/a")

check("a sourced finding is trustworthy",     lambda: trustworthy(_good) is True)
check("a finding with no ref is not",         lambda: trustworthy(_no_ref) is False,
      "'the ledger says so' without saying WHERE cannot be checked")
check("an unsourced claim is not",            lambda: trustworthy(_hearsay) is False,
      "this is the shape a hallucination arrives in -- confident, fluent, unattributable")
check("a finding prints its provenance",      lambda: "ledger:PMT-1005" in str(_good))
def _rejects_partial_finding():
    try:
        Finding(claim="x", by="y")
        return False
    except Exception:
        return True

check("the schema forces all four fields",    lambda: _rejects_partial_finding())
'''),

    md("""
## Section 2 &mdash; Watch the poison spread

Three agents in one graph, all writing into one `findings` list. The ledger agent can be told to
produce a wrong finding; the others read it and build on it.
"""),
    code('''
class SharedState(TypedDict):
    ref: str
    findings: Annotated[list, add]        # every agent appends here
    verdict: Optional[str]

FAULTY = {"ledger": False}                # flip this to inject one wrong finding

def ledger_agent(state: SharedState) -> dict:
    rec = read_ledger_record(state["ref"])
    if FAULTY["ledger"]:
        # plausible, fluent, and wrong: the payment is held, not settled
        return {"findings": [Finding(claim="the payment already settled normally",
                                     by="ledger_agent", source="ledger", ref=state["ref"])]}
    return {"findings": [Finding(claim=f"status={rec.get('status')}, "
                                       f"reason_code={rec.get('reason_code')}",
                                 by="ledger_agent", source="ledger", ref=state["ref"])]}


def policy_agent(state: SharedState) -> dict:
    """Reads the ledger agent's finding and looks up the matching policy."""
    text = " ".join(str(f.claim) for f in state["findings"])
    code_ = next((c for c in POLICY if c in text), None)
    if code_ is None:
        return {"findings": [Finding(claim="no reason code in evidence, so no policy applies",
                                     by="policy_agent", source="policy", ref="none")]}
    return {"findings": [Finding(claim=read_policy_text(code_),
                                 by="policy_agent", source="policy", ref=code_)]}


def critic_agent(state: SharedState) -> dict:
    """Decides, from the findings and nothing else."""
    text = " ".join(str(f.claim) for f in state["findings"]).lower()
    if "settled" in text and "sanctions" not in text:
        return {"verdict": "no action required"}
    if "compliance decides" in text:
        return {"verdict": "hold; escalate to Compliance"}
    return {"verdict": "unclear; escalate"}


def shared_graph():
    g = StateGraph(SharedState)
    g.add_node("ledger", ledger_agent)
    g.add_node("policy", policy_agent)
    g.add_node("critic", critic_agent)
    g.add_edge(START, "ledger")
    g.add_edge("ledger", "policy")
    g.add_edge("policy", "critic")
    g.add_edge("critic", END)
    return g.compile()


def run_shared(ref="PMT-1005", faulty=False) -> dict:
    FAULTY["ledger"] = faulty
    try:
        return shared_graph().invoke({"ref": ref, "findings": [], "verdict": None})
    finally:
        FAULTY["ledger"] = False
'''),
    code('''
# --- Self-check: Section 2   (a real three-node graph -- no model)
check("a clean run reaches the right verdict",
      lambda: "Compliance" in run_shared()["verdict"])
check("all three agents contributed",
      lambda: {f.by for f in run_shared()["findings"]}
              == {"ledger_agent", "policy_agent"} and run_shared()["verdict"] is not None)
check("ONE wrong finding changes the verdict",
      lambda: run_shared(faulty=True)["verdict"] == "no action required",
      "the ledger agent lied once; the critic never touched the ledger and believed it")
check("the poison is visible in the shared findings",
      lambda: any("already settled" in f.claim for f in run_shared(faulty=True)["findings"]))
check("the policy agent was misled too",
      lambda: any("no policy applies" in f.claim for f in run_shared(faulty=True)["findings"]),
      "the damage is not one wrong answer -- it is every agent downstream")
check("the wrong finding still LOOKS trustworthy",
      lambda: all(trustworthy(f) for f in run_shared(faulty=True)["findings"]),
      "provenance tells you where a claim came from, not whether it is true. Both matter.")
'''),

    md("""
## Section 3 &mdash; Scope: not everything belongs in the shared state

Give each agent a private scratch area and share only what it is prepared to stand behind. The
poison still happens &mdash; but it happens to one agent's working notes instead of to the record
every other agent reads.
"""),
    code('''
class ScopedState(TypedDict):
    ref: str
    findings: Annotated[list, add]        # shared: published, attributable claims
    scratch: dict                         # private: each agent's own working notes
    verdict: Optional[str]

def publish(state: ScopedState, finding: Finding) -> dict:
    """Put a finding into the SHARED record -- only if it is checkable."""
    if not trustworthy(finding):
        return {"scratch": {**state["scratch"],
                            finding.by: f"withheld (unsourced): {finding.claim}"}}
    return BLANK                      # TODO: publish it to the shared findings
''', '''
class ScopedState(TypedDict):
    ref: str
    findings: Annotated[list, add]        # shared: published, attributable claims
    scratch: dict                         # private: each agent's own working notes
    verdict: Optional[str]

def publish(state: ScopedState, finding: Finding) -> dict:
    """Put a finding into the SHARED record -- only if it is checkable."""
    if not trustworthy(finding):
        return {"scratch": {**state["scratch"],
                            finding.by: f"withheld (unsourced): {finding.claim}"}}
    return {"findings": [finding]}
'''),
    code('''
# --- Self-check: Section 3
_state = {"ref": "PMT-1005", "findings": [], "scratch": {}, "verdict": None}

check("a checkable finding is published",
      lambda: publish(_state, _good).get("findings") == [_good])
check("an unsourced one is NOT published",
      lambda: "findings" not in publish(_state, _hearsay),
      "the shared record is the thing every other agent trusts -- keep hearsay out of it")
check("the withheld claim is not silently discarded",
      lambda: "critic" in publish(_state, _hearsay)["scratch"],
      "it goes to the agent's own scratch, where it can be inspected but not believed")
check("the withheld note says why",
      lambda: "unsourced" in publish(_state, _hearsay)["scratch"]["critic"])
check("scratch is not an accumulating channel",
      lambda: "scratch" not in str(ScopedState.__annotations__["findings"]),
      "findings accumulate across agents; scratch is overwritten, because it is nobody else's")
'''),

    md("""
## Section 4 &mdash; Trace it back and drop it

Provenance earns its keep at exactly one moment: when something is wrong and you have to find out
what else is wrong because of it.
"""),
    code('''
def quarantine(findings: list, bad_source: str, bad_ref: str) -> tuple[list, list]:
    """Split findings into (kept, dropped) once one source is known to be unreliable."""
    dropped = [f for f in findings if f.source == bad_source and f.ref == bad_ref]
    kept = [f for f in findings if f not in dropped]
    return kept, dropped


def recheck(kept: list) -> str:
    """Re-run the critic's rule over only the findings that survived."""
    text = " ".join(str(f.claim) for f in kept).lower()
    if "compliance decides" in text:
        return "hold; escalate to Compliance"
    if not kept:
        return "no evidence; escalate"
    return "unclear; escalate"
'''),
    code('''
# --- Self-check: Section 4
def _poisoned():
    return run_shared(faulty=True)["findings"]

check("the bad finding can be found by its source and ref",
      lambda: len(quarantine(_poisoned(), "ledger", "PMT-1005")[1]) == 1)
check("everything else is kept",
      lambda: len(quarantine(_poisoned(), "ledger", "PMT-1005")[0])
              == len(_poisoned()) - 1)
check("re-deciding on the survivors no longer says 'no action'",
      lambda: recheck(quarantine(_poisoned(), "ledger", "PMT-1005")[0]) != "no action required",
      "that is the recovery: drop the source, re-decide, do not argue with the conclusion")
check("with no evidence left it escalates rather than guessing",
      lambda: recheck([]) == "no evidence; escalate",
      "an agent with nothing to go on must say so -- silence is not agreement")
'''),

    md("""
## Run it &mdash; the comparison
"""),
    code('''
def _compare():
    print("=== clean run ===")
    clean = run_shared()
    for f in clean["findings"]:
        print("  " + str(f)[:110])
    print("  VERDICT:", clean["verdict"])

    print("\\n=== one wrong finding from the ledger agent ===")
    bad = run_shared(faulty=True)
    for f in bad["findings"]:
        print("  " + str(f)[:110])
    print("  VERDICT:", bad["verdict"], "   <-- wrong, and nothing reported an error")

    print("\\n=== after quarantining the bad source ===")
    kept, dropped = quarantine(bad["findings"], "ledger", "PMT-1005")
    for f in dropped:
        print("  DROPPED " + str(f)[:100])
    print("  VERDICT:", recheck(kept))
guard(_compare)
'''),
    code('''
if llm_ready():
    # The ONLY difference between these two is the last sentence of the first one.
    LICENSED = ("You are a payments control reviewer. Decide what must happen next, using ONLY "
                "the evidence below. If the evidence is inconsistent or insufficient, say so "
                "instead of deciding.")
    PLAIN    = ("You are a payments control reviewer. Decide what must happen next, using ONLY "
                "the evidence below.")

    def _flags_a_problem(text: str) -> bool:
        return any(w in text.lower() for w in ("inconsist", "insufficient", "cannot determine",
                                               "not enough", "unclear", "contradict"))

    def _model_critic():
        out = run_shared(faulty=True)
        evidence = "\\n".join(str(f) for f in out["findings"])
        print("the poisoned evidence a model critic is given:")
        for f in out["findings"]:
            print("  " + str(f)[:110])
        print()
        for label, sysmsg in (("licensed to refuse", LICENSED), ("not licensed", PLAIN)):
            flagged, first = 0, None
            for i in range(3):
                verdict = ask(evidence, system=sysmsg)
                flagged += _flags_a_problem(verdict)
                if i == 0:
                    first = verdict
            print(f"--- {label}: flagged a problem {flagged}/3 ---")
            print("  " + first.strip()[:300] + "\\n")
    guard(_model_critic)
'''),
    md("""
### Read it

**The rule-based critic was fooled.** It was handed a well-formed, correctly attributed finding
that happened to be false, and it had no way to know. That is what makes context poisoning
different from an ordinary bug: no exception, no anomaly in the trace, just a confident answer
built on one bad input.

**The model critic depends entirely on one sentence.** The two runs above differ only in whether
the reviewer was told *"if the evidence is inconsistent or insufficient, say so instead of
deciding."* With that licence it reliably notices that a settled payment needs no next action and
says the evidence is inconsistent. Without it, it does what it was asked &mdash; decides &mdash; and
closes the case.

Sit with that for a moment, because it is the most portable thing in this module. The model was
capable of catching the poisoning the whole time. What it lacked was **permission to refuse**. An
agent given only "decide" will decide, on whatever it has, every time. If you have not written
down what it should do when the evidence does not support an answer, you have written an agent
that cannot do anything but answer.

Three more things follow, and they are the take-home of Module 3.

1. **Provenance is not verification.** Every finding in the poisoned run passed `trustworthy()`.
   Knowing where a claim came from does not tell you it is right &mdash; it tells you *what else to
   throw away* when you discover it is wrong. That is Section 4, and it is worth a great deal, but
   it is a recovery mechanism, not a preventative one.
2. **Scope limits the blast radius.** An agent's uncertain working notes belong in `scratch`,
   where they can be inspected and not believed. The shared `findings` list is the thing every
   other agent treats as fact, so the bar for entering it should be higher than "an agent said so".
3. **Shared state is still the right answer.** Lab 1.4 measured what happens without it: agents
   that cannot see each other's work produce worse answers than one agent that can. The fix for
   poisoning is not to go back to isolation &mdash; it is provenance, scope, and a critic that is
   allowed to say the evidence is inconsistent. You have now measured all three.

**What you take from Module 3:** memory that survives a long session; observations the model does
not have to decode; a `StateGraph` you can test without a model in it; checkpoints that make
resume, approval, rewind and audit possible; and a state design that contains one agent's mistake
instead of spreading it. Module 5 puts several agents into exactly this shape.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Give the *rule-based* critic the same licence the model one has: when the findings contain a
   "settled" claim alongside a reason code, or no policy at all, return "evidence inconsistent;
   escalate". Then poison the run and confirm it fires. Which is safer for your organisation: a
   wrong answer or a refusal?
2. Add a `confidence: float` field to `Finding` and have `publish` withhold anything below a
   threshold. Then find the flaw in that idea &mdash; the poisoned finding in this lab would have
   been published with confidence 1.0.
3. Rebuild Section 2's graph with a checkpointer from Lab 3.4, poison it, then use
   `get_state_history()` to find the exact checkpoint at which the bad finding entered. That is
   the incident investigation, and it takes about four lines.
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-3-01-memory-that-survives",              LAB1),
    ("lab-3-02-perception-observations",           LAB2),
    ("lab-3-03-stategraph-from-scratch",           LAB3),
    ("lab-3-04-checkpointing",                     LAB4),
    ("lab-3-05-challenge-shared-state-poisoning",  LAB5),
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
