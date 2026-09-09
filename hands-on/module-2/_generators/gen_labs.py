#!/usr/bin/env python3
"""
Generate Module 2 lab notebooks and their solutions from one source.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-2-0N-*.ipynb and ../solutions/

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
# Lab 2.{num} &mdash; {title}

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 1 &middot; Module 2 &mdash; Agentic Planning &amp; Reasoning**

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

WORK = os.path.join("/tmp", "awmas-lab-2-{num:02d}")
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
# An internal employee help desk. Ordinary rules on purpose: the only new thing in these five
# labs is LangChain. Nothing here is real data and nothing leaves this notebook.

REQUESTS = {
    "EHD-7001": {"who": "Priya Nair",   "category": "access",   "urgency": "high",
                 "wants": "reset",
                 "text": "Locked out of the payroll portal after the password reset."},
    "EHD-7002": {"who": "Rahul Menon",  "category": "hardware", "urgency": "high",
                 "wants": "replacement",
                 "text": "Laptop battery has swollen and the case is bulging."},
    "EHD-7003": {"who": "Anita Sharma", "category": "software", "urgency": "low",
                 "wants": "licence",
                 "text": "Need a licence for the diagramming tool, about 180 USD a year."},
    "EHD-7004": {"who": "Vikram Rao",   "category": "access",   "urgency": "medium",
                 "wants": "admin-rights",
                 "text": "Please give me admin rights on the finance reporting system."},
    "EHD-7005": {"who": "Priya Nair",   "category": "hardware", "urgency": "low",
                 "wants": "replacement",
                 "text": "Second monitor flickers every few minutes."},
}

# The handbook, one entry per category. Every judgement in this module comes from these.
HANDBOOK = {
    "access":   "Verify identity, then reset. The help desk NEVER grants elevated or admin "
                "rights -- route those to Identity and Access Management.",
    "hardware": "Replace under warranty. A swollen battery is a safety issue: stop use "
                "immediately and replace the same day, whatever urgency the employee set.",
    "software": "Licences over 100 USD per year need the cost-centre owner's approval first.",
}

SLA_HOURS = {"high": 4, "medium": 24, "low": 72}
ROUTE_OUT = {"admin-rights"}     # what the help desk must hand to another team, never do itself

print(f"{len(REQUESTS)} help desk requests, {len(HANDBOOK)} handbook entries loaded")
'''

THREAD_NOTE = (
    "> **The thread.** All five Module 2 labs work one case: an internal employee help desk.\n"
    "> The rules are ordinary on purpose &mdash; the only new thing here is how the agent reasons."
)
# =========================================================================== #
# Lab 2.1 -- chain-of-thought, built as a chain
# =========================================================================== #
LAB1 = [
    header(1, "Chain-of-Thought, Built as a Chain", "Intermediate", 30,
           ["Compose a real LCEL chain: <code>prompt | model | parser</code>",
            "Build two arms that differ in <i>one</i> string and nothing else",
            "Write the verdict down first, then run the eval set with <code>.batch()</code>"],
           THREAD_NOTE),
    setup(1),
    code(DOMAIN),

    md("""
## Concept

**Chain-of-thought** asks the model to show its working before it answers. It usually helps on
tasks that combine two facts, and it always costs tokens. How much of each is a property of
*your* task, not a fact about language models &mdash; so measure it on this one.

The thing you measure it with is worth as much as the answer. **LCEL** composes a prompt, a
model and a parser into one runnable with `|`:

```python
chain = prompt | model | StrOutputParser()
chain.invoke({...})     # one case
chain.batch([{...}, {...}, ...])   # the whole eval set, concurrently
```

Everything later in this course is built this way.
"""),

    md("""
## Section 1 &mdash; The chain, and the one string that makes it chain-of-thought

Three stages, one pipe. `ChatPromptTemplate` turns variables into messages, the model answers,
`StrOutputParser` pulls `.content` out so the chain returns a plain string.

The two arms share the human message, the model, the parser and the case file. The *only*
difference is the system instruction &mdash; which is what makes this a measurement rather than
an anecdote. Both candidate instructions are written out below.
"""),

    code('''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage

HUMAN = ("REQUEST {rid}: {text}\\n"
         "CATEGORY: {category}   URGENCY SET BY THE EMPLOYEE: {urgency}\\n"
         "HANDBOOK: {rule}\\n\\n"
         "What must the help desk do next?")


def instruction(mode: str) -> str:
    """The only difference between the two arms."""
    answer_only  = ("You are an employee help desk analyst. Reply with the single next "
                    "action and nothing else.")
    show_working = ("You are an employee help desk analyst. First restate the handbook rule, "
                    "then the urgency the employee set, then say which of the two decides the "
                    "timing. Finish with one line beginning 'ACTION:'.")

    if mode == "direct":
        return BLANK        # TODO: which one asks for the answer with no working shown?
    return BLANK            # TODO: and which one is chain-of-thought?


def build_prompt(mode: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", instruction(mode)), ("human", HUMAN)])


def build_chain(mode: str, model):
    """prompt | model | parser. `model` is any Runnable -- a chat model is only one kind."""
    return build_prompt(mode) | model | StrOutputParser()


def case_vars(rid: str) -> dict:
    """One request, flattened into the five template variables."""
    r = REQUESTS[rid]
    return {"rid": rid, "text": r["text"], "category": r["category"],
            "urgency": r["urgency"], "rule": HANDBOOK[r["category"]]}
''', '''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage

HUMAN = ("REQUEST {rid}: {text}\\n"
         "CATEGORY: {category}   URGENCY SET BY THE EMPLOYEE: {urgency}\\n"
         "HANDBOOK: {rule}\\n\\n"
         "What must the help desk do next?")


def instruction(mode: str) -> str:
    """The only difference between the two arms."""
    answer_only  = ("You are an employee help desk analyst. Reply with the single next "
                    "action and nothing else.")
    show_working = ("You are an employee help desk analyst. First restate the handbook rule, "
                    "then the urgency the employee set, then say which of the two decides the "
                    "timing. Finish with one line beginning 'ACTION:'.")

    if mode == "direct":
        return answer_only          # no working, straight to the action
    return show_working             # the working is the chain of thought


def build_prompt(mode: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", instruction(mode)), ("human", HUMAN)])


def build_chain(mode: str, model):
    """prompt | model | parser. `model` is any Runnable -- a chat model is only one kind."""
    return build_prompt(mode) | model | StrOutputParser()


def case_vars(rid: str) -> dict:
    """One request, flattened into the five template variables."""
    r = REQUESTS[rid]
    return {"rid": rid, "text": r["text"], "category": r["category"],
            "urgency": r["urgency"], "rule": HANDBOOK[r["category"]]}
'''),

    code('''
# --- Self-check: Section 1   (a real chain, really invoked -- with a stub where the model goes)
STUB = RunnableLambda(lambda messages: AIMessage(content="ACTION: replace it the same day"))

def sys_text(mode):
    return build_prompt(mode).format_messages(**case_vars("EHD-7002"))[0].content

def human_text(mode):
    return build_prompt(mode).format_messages(**case_vars("EHD-7002"))[1].content

check("the template renders a system message and a human message",
      lambda: [m.type for m in build_prompt("cot").format_messages(**case_vars("EHD-7002"))]
              == ["system", "human"])
check("the request text is substituted in, not left as a brace",
      lambda: "swollen" in human_text("cot") and "{text}" not in human_text("cot"))
check("the two arms differ in the system message ONLY",
      lambda: human_text("direct") == human_text("cot") and sys_text("direct") != sys_text("cot"),
      "if anything else differs, the comparison measures that instead")
check("the direct arm asks for the action and no working",
      lambda: "nothing else" in sys_text("direct"))
check("the chain-of-thought arm asks for the working first",
      lambda: "restate" in sys_text("cot") and "ACTION:" in sys_text("cot"))
check("prompt | model | parser returns a plain string, not an AIMessage",
      lambda: isinstance(build_chain("cot", STUB).invoke(case_vars("EHD-7002")), str),
      "StrOutputParser is the third stage -- drop it and you get a message object back")
score()
'''),

    md("""
## Section 2 &mdash; The eval set, and the verdict you write down first

Five requests, and what a correct answer has to mention. Keyword lists rather than exact
wording, because you are grading the decision, not the prose.

Four of the five are settled. **EHD-7002 is the one that carries the lesson.** Rahul set the
urgency to `high`, which is a 4-hour SLA. The handbook says a swollen battery is a safety issue:
replace it *the same day, whatever urgency the employee set*. Four hours is inside the same day,
so an answer that says &ldquo;within 4 hours per the high SLA&rdquo; lands on a timing that
happens to be acceptable &mdash; by reading the row the handbook told it to ignore.

Decide what you will accept **before** you see a single model output. That is the difference
between an evaluation and a story about one run.
"""),

    code('''
def expected_for(rid: str) -> list:
    """What a correct answer must mention, lower-cased. Substrings, so wording is free."""
    settled = {
        "EHD-7001": ["reset"],
        "EHD-7003": ["approval"],
        "EHD-7004": ["identity and access"],
        "EHD-7005": ["replace"],
    }
    sla_wins  = ["within 4 hours"]      # the urgency the employee set
    rule_wins = ["same day"]            # the handbook's safety rule

    if rid == "EHD-7002":
        return BLANK        # TODO: which one is the correct answer for a swollen battery?
    return settled[rid]


def graded(rid: str, answer: str) -> bool:
    """Correct if the answer mentions everything that case requires."""
    text = (answer or "").lower()
    return all(k in text for k in expected_for(rid))


def pass_count(answers: dict) -> int:
    return sum(1 for rid, a in answers.items() if graded(rid, a))
''', '''
def expected_for(rid: str) -> list:
    """What a correct answer must mention, lower-cased. Substrings, so wording is free."""
    settled = {
        "EHD-7001": ["reset"],
        "EHD-7003": ["approval"],
        "EHD-7004": ["identity and access"],
        "EHD-7005": ["replace"],
    }
    sla_wins  = ["within 4 hours"]      # the urgency the employee set
    rule_wins = ["same day"]            # the handbook's safety rule

    if rid == "EHD-7002":
        return rule_wins    # the safety rule overrides the urgency; 4 hours is a coincidence
    return settled[rid]


def graded(rid: str, answer: str) -> bool:
    """Correct if the answer mentions everything that case requires."""
    text = (answer or "").lower()
    return all(k in text for k in expected_for(rid))


def pass_count(answers: dict) -> int:
    return sum(1 for rid, a in answers.items() if graded(rid, a))
'''),

    code('''
# --- Self-check: Section 2   (hand-written answers, graded offline -- no model)
HAND = {
    "EHD-7001": "Verify identity, then reset the payroll portal password.",
    "EHD-7002": "Stop use and replace the laptop the same day.",
    "EHD-7003": "The cost-centre owner's approval is needed first -- 180 USD is over the limit.",
    "EHD-7004": "Route to Identity and Access Management; the help desk never grants admin rights.",
    "EHD-7005": "Replace the monitor under warranty.",
}

check("a same-day answer for the swollen battery is correct",
      lambda: graded("EHD-7002", "Stop use and replace the laptop the same day."))
check("'within 4 hours per the high SLA' is NOT accepted",
      lambda: not graded("EHD-7002", "Replace within 4 hours per the high SLA."),
      "the timing is fine and the reasoning is wrong -- it read the row the handbook overrides")
check("granting admin rights is not an acceptable answer to EHD-7004",
      lambda: not graded("EHD-7004", "Grant admin rights on the finance reporting system."))
check("five hand-written correct answers score 5/5",
      lambda: pass_count(HAND) == 5,
      "if this is not 5/5 the bar is grading wording rather than decisions")
score()
'''),

    md("""
## Run it for real

Two arms, five cases, one `.batch()` each.
"""),

    code('''
CASES = [case_vars(rid) for rid in sorted(REQUESTS)]

def run_arm(mode: str) -> dict:
    outs = build_chain(mode, get_llm()).batch(CASES)
    return {c["rid"]: o for c, o in zip(CASES, outs)}

def bake_off():
    for mode in ["direct", "cot"]:
        answers = run_arm(mode)
        print(f"=== {mode}: {pass_count(answers)}/{len(CASES)} ===")
        for rid, a in answers.items():
            last = (a.strip().splitlines() or [""])[-1]
            print(f'  {"ok " if graded(rid, a) else "NO "}{rid}: {last[:88]}')
        print()

if llm_ready():
    guard(bake_off)
'''),

    md("""
### Read it

Ten model calls went out in two `.batch()` calls, and `.batch()` ran each set **concurrently**
&mdash; that is the whole reason an eval set is a thing you run every time you touch a prompt
rather than a thing you promise to do later.

Look at EHD-7002 in both arms. The direct arm tends to answer from the loudest field on the page,
which is `URGENCY: high`; the chain-of-thought arm is forced to put the handbook rule down first,
and once the rule is written out the override is hard to miss. Nothing changed but one string.

The bar decided that. &ldquo;Within 4 hours&rdquo; is a defensible answer to a human reader and
you would have accepted it, silently, if you had written the bar after seeing it.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Add a third arm whose instruction just says &ldquo;think step by step&rdquo; and nothing about
   the handbook. Does the generic phrase buy what the ordered procedure buys?
2. Change `expected_for("EHD-7002")` to `sla_wins` and re-run. Both arms will look better. Say
   out loud what you gave up to get that number.
"""),
]


# =========================================================================== #
# Lab 2.2 -- ReAct, and the contract that actually breaks
# =========================================================================== #
LAB2 = [
    header(2, "ReAct, and the Contract That Actually Breaks", "Intermediate &rarr; Advanced", 35,
           ["Parse the classic Thought / Action / Action Input text format",
            "Watch the format hold while the <i>argument</i> drifts under an aged prompt",
            "Move the contract into an <code>args_schema</code> the model is shown upfront"],
           THREAD_NOTE),
    setup(2),
    code(DOMAIN),

    md("""
## Concept

**ReAct** interleaves reasoning and acting: *Thought* (what do I know?), *Action* (what shall I
do?), *Observation* (what came back?), repeat.

The original formulation asks the model to emit that shape **as text**, so you have to parse it.
Everyone expects the parse to be the fragile part. On this sandbox it is not: measured over four
requests, the format parsed 4/4 strictly and 4/4 loosely, twice.

What broke was the **argument**. The same agent went from 4/4 usable to 0/4 &mdash; same format,
same parser, same tool &mdash; because the model started passing a description where a reference
belongs. This lab reproduces that, and then fixes it in the one place that works.
"""),

    md("""
## Section 1 &mdash; The format, and what the parser does when it is wrong

The happy path:

```
Thought: I need the stored request first.
Action: lookup_request
Action Input: EHD-7002
```

The regex is given &mdash; it is a Python idiom, not the lesson. The decision is the branch below
it: the text parsed cleanly and named a tool that **does not exist**. A parser can do three
things there. It can `raise`, which ends the run mid-turn and throws away a thought the model has
already paid for. It can return `None`, which tells the agent loop nothing at all. Or it can hand
back something the loop can act on next turn.
"""),

    code('''
import re
from langchain_core.tools import tool
from pydantic import BaseModel, Field

STEP_RE = re.compile(
    r"Thought:\\s*(?P<thought>.*?)\\s*"
    r"Action:\\s*(?P<action>[\\w_]+)\\s*"
    r"Action Input:\\s*(?P<input>.*?)\\s*$",
    re.DOTALL | re.IGNORECASE)

TOOLS = {"lookup_request", "handbook_rule"}      # every tool this agent actually has


def parse_step(text: str):
    """-> {"thought", "action", "input"} for an action step, or None for a final answer."""
    m = STEP_RE.search(text or "")
    if not m:
        return None
    step = {"thought": m.group("thought").strip(),
            "action": m.group("action").strip(),
            "input": m.group("input").strip().strip('"').strip("'")}

    if step["action"] not in TOOLS:
        # The text parsed. The model named a tool that is not on the list.
        give_up        = None
        tell_the_agent = {**step, "error": f'no tool named "{step["action"]}". '
                                           f'Available: {", ".join(sorted(TOOLS))}'}
        return BLANK    # TODO: the agent loop gets this back. Which value lets it recover?

    return step
''', '''
import re
from langchain_core.tools import tool
from pydantic import BaseModel, Field

STEP_RE = re.compile(
    r"Thought:\\s*(?P<thought>.*?)\\s*"
    r"Action:\\s*(?P<action>[\\w_]+)\\s*"
    r"Action Input:\\s*(?P<input>.*?)\\s*$",
    re.DOTALL | re.IGNORECASE)

TOOLS = {"lookup_request", "handbook_rule"}      # every tool this agent actually has


def parse_step(text: str):
    """-> {"thought", "action", "input"} for an action step, or None for a final answer."""
    m = STEP_RE.search(text or "")
    if not m:
        return None
    step = {"thought": m.group("thought").strip(),
            "action": m.group("action").strip(),
            "input": m.group("input").strip().strip('"').strip("'")}

    if step["action"] not in TOOLS:
        # The text parsed. The model named a tool that is not on the list.
        give_up        = None
        tell_the_agent = {**step, "error": f'no tool named "{step["action"]}". '
                                           f'Available: {", ".join(sorted(TOOLS))}'}
        return tell_the_agent   # an observation the model can correct from next turn

    return step
'''),

    code('''
# --- Self-check: Section 1   (pure string work -- no model)
GOOD   = "Thought: I need the record.\\nAction: lookup_request\\nAction Input: EHD-7002"
QUOTED = 'Thought: t\\nAction: lookup_request\\nAction Input: "EHD-7002"'
NOSUCH = "Thought: I will check the ticket.\\nAction: fetch_ticket\\nAction Input: EHD-7002"

check("a well-formed step parses and names the tool",
      lambda: parse_step(GOOD)["action"] == "lookup_request")
check("the argument comes out without its quotes",
      lambda: parse_step(QUOTED)["input"] == "EHD-7002")
check("a final answer is not an action step",
      lambda: parse_step("Replace the laptop the same day.") is None)
check("an unknown tool still MATCHES the format -- the format was never the problem",
      lambda: STEP_RE.search(NOSUCH) is not None)
check("...and the parser hands the loop something it can act on",
      lambda: "fetch_ticket" in parse_step(NOSUCH)["error"]
              and "lookup_request" in parse_step(NOSUCH)["error"],
      "None tells the loop nothing; raising ends the run and bins the thought")
score()
'''),

    md("""
## Section 2 &mdash; Where the contract belongs

Now the failure a parser cannot catch. Under a **fresh** prompt the model passes `EHD-7002`.
Under an **aged** prompt &mdash; a few turns of history in which requests were discussed by
description &mdash; it starts passing `the swollen battery one`, or `Rahul Menon`, or
`{"request_id": "EHD-7002"}`. Every one of those parses. None of them resolves.

You can chase that in the parser: reject anything that is not `EHD-` plus four digits, and retry
the turn. That works and it is a correction applied *after* the model has spoken, once per drift,
forever.

Or you give the tool a typed signature. `args_schema` is not validation for your benefit &mdash;
LangChain sends that schema, field descriptions and all, to the model **with the request**. The
model never sees your function; it sees a name, a description and a shape.
"""),

    code('''
def lookup_args_schema():
    """The model READS this description before it answers. That is what it is for."""
    names_the_field = "The request to look up."
    names_the_shape = ("The help desk reference exactly as it appears on the request, "
                       "e.g. EHD-7002. Not the employee's name and not the request text.")

    class LookupArgs(BaseModel):
        request_id: str = Field(description=BLANK)   # TODO: which one prevents the drift?

    return LookupArgs


def build_lookup_tool():
    @tool("lookup_request", args_schema=lookup_args_schema())
    def lookup_request(request_id: str) -> str:
        """Look up one stored help desk request by its reference."""
        r = REQUESTS.get(request_id)
        return "no such request" if r is None else f'{r["who"]} | {r["category"]} | {r["text"]}'
    return lookup_request


def where_the_contract_belongs() -> str:
    stricter_parser = "reject any Action Input that is not EHD-nnnn, and retry the turn"
    typed_signature = "give the tool an args_schema, so the model is told the shape upfront"
    return BLANK        # TODO: which one stops the drift rather than catching it afterwards?
''', '''
def lookup_args_schema():
    """The model READS this description before it answers. That is what it is for."""
    names_the_field = "The request to look up."
    names_the_shape = ("The help desk reference exactly as it appears on the request, "
                       "e.g. EHD-7002. Not the employee's name and not the request text.")

    class LookupArgs(BaseModel):
        request_id: str = Field(description=names_the_shape)

    return LookupArgs


def build_lookup_tool():
    @tool("lookup_request", args_schema=lookup_args_schema())
    def lookup_request(request_id: str) -> str:
        """Look up one stored help desk request by its reference."""
        r = REQUESTS.get(request_id)
        return "no such request" if r is None else f'{r["who"]} | {r["category"]} | {r["text"]}'
    return lookup_request


def where_the_contract_belongs() -> str:
    stricter_parser = "reject any Action Input that is not EHD-nnnn, and retry the turn"
    typed_signature = "give the tool an args_schema, so the model is told the shape upfront"
    return typed_signature   # shown before the answer beats corrected after it
'''),

    code('''
# --- Self-check: Section 2   (a real @tool and its real args_schema -- no model)
def field_description():
    return lookup_args_schema().model_fields["request_id"].description

check("the tool carries a typed args_schema",
      lambda: build_lookup_tool().args_schema is not None)
check("the description the model reads names the SHAPE of the reference",
      lambda: "EHD-7002" in field_description(),
      '"The request to look up." leaves the model to invent a format, which it will')
check("it also rules out the two things the model reaches for instead",
      lambda: "name" in field_description().lower() and "text" in field_description().lower())
check("the tool resolves a reference and refuses a description",
      lambda: "Rahul" in build_lookup_tool().invoke({"request_id": "EHD-7002"})
              and build_lookup_tool().invoke({"request_id": "the swollen battery one"})
                  == "no such request")
check("the contract belongs where the model can read it before it answers",
      lambda: where_the_contract_belongs().startswith("give the tool an args_schema"))
score()
'''),

    md("""
## Run it for real

Three runs over the same four requests: text ReAct on a fresh prompt, text ReAct on an aged one,
and the bound tool on the aged one. Watch `parsed` stay flat while `usable` collapses.
"""),

    code('''
RIDS  = ["EHD-7001", "EHD-7002", "EHD-7003", "EHD-7005"]
FRESH = ("Answer with exactly one step, in this format and nothing else:\\n"
         "Thought: <why>\\nAction: <tool name>\\nAction Input: <the argument>\\n"
         "Tools: lookup_request(request_id), handbook_rule(category).")
AGED  = FRESH + ("\\n\\nEarlier in this conversation you handled the payroll lockout, the "
                 "flickering monitor and the licence for the diagramming tool. Keep referring "
                 "to requests the way the employee described them.")

def usable(step) -> bool:
    """Usable means the argument is a reference the tool can actually resolve."""
    return bool(step) and "error" not in step and step["input"] in REQUESTS

def run_text_arm(label, system):
    print(f"=== text ReAct, {label} ===")
    parsed = ok = 0
    for rid in RIDS:
        step = parse_step(ask(f'Employee request: "{REQUESTS[rid]["text"]}"\\n'
                              "Look up the stored request.", system=system))
        parsed += step is not None
        ok += usable(step)
        print(f'  {rid}: input={(step or {}).get("input")!r}')
    print(f"  -> parsed {parsed}/{len(RIDS)}, usable {ok}/{len(RIDS)}\\n")

def compare():
    run_text_arm("fresh prompt", FRESH)
    run_text_arm("aged prompt", AGED)

    print("=== bound tool with an args_schema, aged prompt ===")
    bound = get_llm().bind_tools([build_lookup_tool()])
    ok = 0
    for rid in RIDS:
        calls = bound.invoke([("system", AGED),
                              ("human", f'Employee request: "{REQUESTS[rid]["text"]}" '
                                        "-- look up the stored request.")]).tool_calls
        arg = calls[0]["args"].get("request_id") if calls else None
        ok += arg in REQUESTS
        print(f'  {rid}: request_id={arg!r}')
    print(f"  -> usable {ok}/{len(RIDS)}")

if llm_ready():
    guard(compare)
'''),

    md("""
### Read it

The parse counts barely moved. The usable counts did &mdash; the aged prompt did not corrupt the
format, it corrupted the argument, and a regex that checks the shape of the *text* has nothing to
say about that. Every step it rejected was a step the model was entitled to believe was fine.

The bound arm got the same aged prompt and the same drifting history, and passed the reference
anyway. Nothing corrected it. The schema went out **with** the question, so there was no wrong
answer to correct.

That is the general shape, and it is worth carrying into Module 4: a contract the model is
**shown** beats a parser that corrects it afterwards. The parser was never the problem.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Set the `request_id` description back to `names_the_field` and re-run the bound arm on the
   aged prompt. The schema is still there and still typed. How much of the fix was the type?
2. Add `handbook_rule` as a second bound tool with a `category` field whose description does
   *not* list the three valid categories. Count how often the model invents a fourth.
"""),
]
# =========================================================================== #
# Lab 2.3 -- sub-goals that finish, and knowing when to re-plan
# =========================================================================== #
LAB3 = [
    header(3, "Sub-goals That Finish, and Knowing When to Re-plan", "Advanced", 30,
           ["Have the model return a <code>Plan</code> object with declared dependencies",
            "Write the field descriptions the model actually reads",
            "Tell a transient failure from a wrong plan &mdash; retry one, re-plan the other"],
           THREAD_NOTE),
    setup(3),
    code(DOMAIN),

    md("""
## Concept

EHD-7003 is a 180 USD licence, and the software handbook entry says licences over 100 USD need
the cost-centre owner's approval first. So the plan is not one step and it is not three
independent steps: **check the price &rarr; get approval &rarr; purchase**. Step three cannot start
until step two finishes, and step two only exists because of what step one found.

Two things have to be written down for that to work at all:

1. the plan has to be an **object**, with the order in it, not a paragraph you re-read every turn;
2. when a step fails you have to know **which kind of failure it was**, because a transient one and
   a wrong plan need opposite responses and the same `except` block catches both.
"""),

    md("""
## Section 1 &mdash; A plan the model returns as an object

`with_structured_output(Plan)` sends your schema to the model and gives you back a `Plan`.

The `Field(description=...)` strings are the part everyone treats as documentation. They are not
documentation. LangChain sends them to the model **as the schema** &mdash; they are the only
instruction it ever gets about what belongs in each field.
"""),

    code('''
from typing import List
from pydantic import BaseModel, Field


class Step(BaseModel):
    """One step of a help desk plan."""
    action: str = Field(description="BLANK")
    # TODO ^ what must one step say? Something an analyst could carry out without asking you.

    depends_on: List[int] = Field(default_factory=list, description="BLANK")
    # TODO ^ these integers are what turns a list into an order. Say what they point at.

    done_when: str = Field(
        description="A test somebody else could apply to say this step is finished")


class Plan(BaseModel):
    """An ordered plan for handling one help desk request."""
    goal: str = Field(description="The outcome this plan reaches, in one line")
    steps: List[Step] = Field(description="The steps, in the order they should run")
''', '''
from typing import List
from pydantic import BaseModel, Field


class Step(BaseModel):
    """One step of a help desk plan."""
    action: str = Field(
        description="One concrete thing to do, imperative and specific, "
                    "e.g. 'check the annual licence price against the 100 USD limit'")

    depends_on: List[int] = Field(
        default_factory=list,
        description="The 1-based positions of the steps that must finish before this one "
                    "may start; empty if it can start immediately")

    done_when: str = Field(
        description="A test somebody else could apply to say this step is finished")


class Plan(BaseModel):
    """An ordered plan for handling one help desk request."""
    goal: str = Field(description="The outcome this plan reaches, in one line")
    steps: List[Step] = Field(description="The steps, in the order they should run")
'''),

    code('''
# --- Self-check: Section 1   (the schema object, before any model sees it)
def described(field: str) -> str:
    """The description the model will be sent. An unfilled blank is the literal 'BLANK'."""
    d = Step.model_fields[field].description
    if d.strip() == "BLANK":
        raise NameError(f"Step.{field} description is still BLANK")   # -> [TODO], not [FAIL]
    return d


def _accepts(step: dict) -> bool:
    """Does Plan let this step through?"""
    try:
        Plan(goal="g", steps=[step])
        return True
    except NameError:
        raise                       # a blank above, not a schema verdict
    except Exception:
        return False


def ehd_7003_plan() -> Plan:
    """The three steps the software handbook forces for a 180 USD licence."""
    return Plan(goal="Anita Sharma has a licence for the diagramming tool, or a written refusal",
                steps=[Step(action="check the annual price of the licence",
                            depends_on=[], done_when="the price in USD is written down"),
                       Step(action="get the cost-centre owner's approval",
                            depends_on=[1], done_when="the owner has replied yes or no"),
                       Step(action="purchase the licence and send the key",
                            depends_on=[2], done_when="Anita has the key")])


check("the action description tells the model to write one concrete thing to do",
      lambda: len(described("action")) > 30)
check("the depends_on description says what the numbers point at",
      lambda: "step" in described("depends_on").lower() and len(described("depends_on")) > 30,
      "a bare list of integers means nothing to the model unless you say what they index")
check("Plan can express the EHD-7003 dependency: approval before purchase",
      lambda: ehd_7003_plan().steps[2].depends_on == [2]
              and ehd_7003_plan().steps[0].depends_on == [])
check("Plan rejects a step with no completion test",
      lambda: not _accepts({"action": "buy it", "depends_on": []})
              and _accepts({"action": "buy it", "depends_on": [], "done_when": "key sent"}),
      "a sub-goal no one can call finished is how an agent loops forever on step three")
score()
'''),

    md("""
## Section 2 &mdash; Transient failure, or wrong plan

The licence catalogue returns `503`. Retry it &mdash; bounded, because an unbounded retry is a hang
with a progress bar.

The catalogue returns `no such tool: grant_admin_rights`. Retrying that is not optimism, it is
arithmetic: the identical call will fail identically forever. What is wrong is the **plan**, and
the only move that changes anything is to make a new one.

The commonest version of this bug in production is a single blanket `except: retry`.
"""),

    code('''
TRANSIENT = ("503", "502", "timeout", "timed out", "rate limit", "connection reset")
WRONG_PLAN = ("no such tool", "no such request", "invalid argument", "unknown field",
              "not permitted for this desk")

RETRY_BUDGET = 2


def classify_failure(error: str) -> str:
    """'retry' -- the world misbehaved.  're-plan' -- the plan was wrong."""
    e = error.lower()
    world_misbehaved = any(s in e for s in TRANSIENT)
    plan_was_wrong = any(s in e for s in WRONG_PLAN)

    if plan_was_wrong:
        return BLANK        # TODO: this exact call, sent again, gets this exact error. So?
    if world_misbehaved:
        return "retry"
    return "re-plan"        # unrecognised: assume the plan, because that is the cheaper mistake


def next_move(error: str, attempts_so_far: int) -> str:
    """What to do after a failed step: 'retry', 're-plan' or 'give up'."""
    verdict = classify_failure(error)
    if verdict == "retry" and attempts_so_far >= RETRY_BUDGET:
        return "give up"    # a bounded retry is a retry; an unbounded one is an outage
    return verdict
''', '''
TRANSIENT = ("503", "502", "timeout", "timed out", "rate limit", "connection reset")
WRONG_PLAN = ("no such tool", "no such request", "invalid argument", "unknown field",
              "not permitted for this desk")

RETRY_BUDGET = 2


def classify_failure(error: str) -> str:
    """'retry' -- the world misbehaved.  're-plan' -- the plan was wrong."""
    e = error.lower()
    world_misbehaved = any(s in e for s in TRANSIENT)
    plan_was_wrong = any(s in e for s in WRONG_PLAN)

    if plan_was_wrong:
        return "re-plan"    # the same call cannot start working; only a different plan can
    if world_misbehaved:
        return "retry"
    return "re-plan"        # unrecognised: assume the plan, because that is the cheaper mistake


def next_move(error: str, attempts_so_far: int) -> str:
    """What to do after a failed step: 'retry', 're-plan' or 'give up'."""
    verdict = classify_failure(error)
    if verdict == "retry" and attempts_so_far >= RETRY_BUDGET:
        return "give up"    # a bounded retry is a retry; an unbounded one is an outage
    return verdict
'''),

    code('''
# --- Self-check: Section 2   (hand-written errors, no model, no network)
FAILURES = {
    "503 Service Unavailable from the licence catalogue": "retry",
    "rate limit exceeded on the approvals API, try again in 30s": "retry",
    "no such tool: grant_admin_rights": "re-plan",
    "no such request: EHD-9999": "re-plan",
    "invalid argument: amount must be a number, got '180 USD'": "re-plan",
}

check("every transient failure is retried",
      lambda: all(classify_failure(e) == "retry" for e, v in FAILURES.items() if v == "retry"))
check("a wrong-plan failure is never retried",
      lambda: all(classify_failure(e) == "re-plan" for e, v in FAILURES.items() if v == "re-plan"),
      "'no such tool' is not a blip -- the same call will fail identically every time")
check("an unrecognised failure re-plans rather than hammering",
      lambda: classify_failure("the printer is on fire") == "re-plan")
check("the retry is bounded",
      lambda: next_move("503 Service Unavailable", 0) == "retry"
              and next_move("503 Service Unavailable", RETRY_BUDGET) == "give up")
score()
'''),

    md("""
## Run it for real

Ask the model for a `Plan` for EHD-7003, then put the two kinds of failure through the classifier.
"""),

    code('''
def real_plan(rid: str) -> Plan:
    r = REQUESTS[rid]
    return get_llm().with_structured_output(Plan).invoke(
        "You are an employee help desk analyst. Plan the handling of this request.\\n"
        f"REQUEST: {json.dumps({'id': rid, **r})}\\n"
        f"HANDBOOK ({r['category']}): {HANDBOOK[r['category']]}\\n"
        "Use depends_on to say which steps cannot start until an earlier one has finished.")


if llm_ready():
    plan = guard(lambda: real_plan("EHD-7003"))
    if plan:
        print("goal:", plan.goal, "\\n")
        for i, s in enumerate(plan.steps, 1):
            print(f"{i}. {s.action}")
            print(f"   after step(s) {s.depends_on or 'none'} | done when: {s.done_when}")
'''),

    code('''
def show_recovery():
    for err, attempts in [("503 Service Unavailable from the licence catalogue", 0),
                          ("503 Service Unavailable from the licence catalogue", RETRY_BUDGET),
                          ("no such tool: grant_admin_rights", 0)]:
        print(f"{next_move(err, attempts):8}  <- attempt {attempts}: {err}")

guard(show_recovery)
'''),

    md("""
### Read it

The model returned a `Plan`, not a paragraph. Look at what that bought: `depends_on` is a list of
integers you can sort on, so "may this step start yet?" is a comparison rather than a re-read of
prose. If the model put purchase before approval, you can see that without the model's help.

The two failures went in looking almost identical &mdash; both are strings from a service that said
no &mdash; and came out with opposite moves. The 503 is worth two more attempts and then a stop.
`no such tool: grant_admin_rights` is worth zero attempts, because nothing about attempt two is
different from attempt one.

Note the default in the third branch. Unrecognised failures re-plan. Guessing "transient" costs you
a retry loop; guessing "wrong plan" costs you one extra planning call.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Delete the `done_when` field from `Step` and ask the model for the EHD-7003 plan again. Read
   step three. Could an agent tell whether it had finished?
2. Move `"invalid argument"` from `WRONG_PLAN` to `TRANSIENT` and re-run the Section 2 check. Count
   the calls a real agent would then make against `amount='180 USD'`.
"""),
]


# =========================================================================== #
# Lab 2.4 -- branch, score, prune, and the reflection knee
# =========================================================================== #
LAB4 = [
    header(4, "Branch, Score, Prune &mdash; and When to Stop Reflecting", "Advanced", 35,
           ["Generate three candidate resolutions concurrently with <code>RunnableParallel</code>",
            "Write the scorer that decides which one survives &mdash; and see it pick the trap",
            "Put a stop condition on a reflection loop before it runs out of your budget"],
           THREAD_NOTE),
    setup(4),
    code(DOMAIN),

    md("""
## Concept

Two ways of spending extra calls to get a better answer.

**Branch and prune:** generate several candidates, score them, keep one. `RunnableParallel` runs
them at the same time, so three candidates cost one round trip of latency rather than three.

**Reflection:** draft, criticise, revise. Improves the answer for a round or two, and then stops
paying while continuing to cost.

Both look like they are about generating. Neither is. Branching is about the **scorer**, and
reflection is about the **stop condition**, and those are the two things this lab makes you write.

The case is EHD-7004: Vikram Rao wants admin rights on the finance reporting system.
`wants` is `"admin-rights"`, which is in `ROUTE_OUT` &mdash; the help desk must hand it to another
team, never do it itself. So one of the three candidates is going to sound extremely helpful and
be exactly wrong.
"""),

    md("""
## Section 1 &mdash; Three branches at once, and the scorer that judges them

`RunnableParallel(fastest=..., handbook=..., safest=...)` is one runnable holding three chains.
Invoke it once with one input and all three run concurrently; you get a dict back, keyed by name.
"""),

    code('''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

ANGLES = {
    "fastest":  "Get this employee working again today, with the fewest hops.",
    "handbook": "Answer strictly from the handbook, quoting its operative words.",
    "safest":   "Say what must NOT happen here, and who owns the decision instead.",
}


def branches_runnable(model):
    """One runnable, three chains, one invoke -- the branches run concurrently."""
    def chain(angle):
        return ChatPromptTemplate.from_messages([
            ("system", "You are an employee help desk analyst. " + angle + " Two sentences."),
            ("human", "REQUEST: {request}\\nHANDBOOK: {handbook}\\n\\nWhat happens next?"),
        ]) | model | StrOutputParser()
    return RunnableParallel(**{name: chain(angle) for name, angle in ANGLES.items()})


def score_branch(text: str) -> int:
    """Score one candidate resolution of EHD-7004, out of 3."""
    t = text.lower()
    grants = any(p in t for p in ("grant admin", "grant the admin", "grant elevated",
                                  "grant him admin", "provision admin", "add him to the admin"))
    routes = "identity and access" in t or "iam" in t

    rewards_resolution = 3 if grants else (2 if routes else 1)   # "is the employee sorted?"
    rewards_handbook = 0 if grants else (3 if routes else 1)     # "did we follow the handbook?"

    return BLANK
    # TODO ^ EHD-7004 wants "admin-rights", and that is in ROUTE_OUT. One of these two scorers
    #        can reward a branch that grants them. Which one must you never ship?
''', '''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

ANGLES = {
    "fastest":  "Get this employee working again today, with the fewest hops.",
    "handbook": "Answer strictly from the handbook, quoting its operative words.",
    "safest":   "Say what must NOT happen here, and who owns the decision instead.",
}


def branches_runnable(model):
    """One runnable, three chains, one invoke -- the branches run concurrently."""
    def chain(angle):
        return ChatPromptTemplate.from_messages([
            ("system", "You are an employee help desk analyst. " + angle + " Two sentences."),
            ("human", "REQUEST: {request}\\nHANDBOOK: {handbook}\\n\\nWhat happens next?"),
        ]) | model | StrOutputParser()
    return RunnableParallel(**{name: chain(angle) for name, angle in ANGLES.items()})


def score_branch(text: str) -> int:
    """Score one candidate resolution of EHD-7004, out of 3."""
    t = text.lower()
    grants = any(p in t for p in ("grant admin", "grant the admin", "grant elevated",
                                  "grant him admin", "provision admin", "add him to the admin"))
    routes = "identity and access" in t or "iam" in t

    rewards_resolution = 3 if grants else (2 if routes else 1)   # "is the employee sorted?"
    rewards_handbook = 0 if grants else (3 if routes else 1)     # "did we follow the handbook?"

    return rewards_handbook   # a scorer that rewards "resolved it" rewards the ROUTE_OUT breach
'''),

    code('''
# --- Self-check: Section 1   (a real RunnableParallel, really invoked -- with no model in it)
CANDIDATES = {
    "fastest":  "Grant admin rights on finance reporting to Vikram Rao today so he is unblocked.",
    "handbook": "The help desk never grants elevated rights. Route this to Identity and Access "
                "Management, who own the entitlement.",
    "safest":   "Do not change any entitlement here. Identity and Access Management decides.",
}

ECHO = RunnableLambda(lambda pv: pv.to_string())      # stands in for the model, offline

def parallel_run():
    return branches_runnable(ECHO).invoke({"request": "EHD-7004", "handbook": HANDBOOK["access"]})

check("one invoke drives all three branches, each with its own framing",
      lambda: set(parallel_run()) == set(ANGLES) and len(set(parallel_run().values())) == 3,
      "three copies of one prompt is not a tree, it is one answer billed three times")
check("the branch that grants admin rights scores zero",
      lambda: score_branch(CANDIDATES["fastest"]) == 0,
      "EHD-7004 is a ROUTE_OUT case: resolving it here is a breach, however fast it is")
check("the routing branches score full marks, and pruning keeps one of them",
      lambda: score_branch(CANDIDATES["handbook"]) == 3
              and max(CANDIDATES, key=lambda k: score_branch(CANDIDATES[k])) != "fastest")
score()
'''),

    md("""
## Section 2 &mdash; Reflection, and the knee

Draft, criticise, revise. The first revision usually earns its call. The second sometimes does. By
the fourth the critic is inventing work, and you are paying two calls a round for a rewording.

That curve has a knee, and a reflection loop with no stop condition is Module 2's most expensive
failure &mdash; it does not crash, it just bills.

So the loop is three lines. What you have to decide is when it ends.
"""),

    code('''
MAX_ROUNDS = 3


def should_stop(round_no: int, critique: str, last_score: int, this_score: int) -> bool:
    """Called after each revision. True = stop reflecting."""
    critic_found_nothing = critique.strip().lower().startswith("no change")
    out_of_rounds = round_no >= MAX_ROUNDS
    no_longer_paying = this_score <= last_score

    stop_when_clean = critic_found_nothing
    stop_on_any = critic_found_nothing or out_of_rounds or no_longer_paying

    return BLANK
    # TODO ^ a critic asked to find a fault will usually find one. Which of these can never run
    #        forever AND never keeps paying past the knee?


def reflect(draft: str, critique_of, revise, hard_cap: int = MAX_ROUNDS + 4) -> list:
    """[(round, text, score), ...]. hard_cap is a seatbelt, not the stop condition."""
    history = [(0, draft, score_branch(draft))]
    text = draft
    for n in range(1, hard_cap + 1):
        critique = critique_of(text)
        text = revise(text, critique)
        previous, now = history[-1][2], score_branch(text)
        history.append((n, text, now))
        if should_stop(n, critique, previous, now):
            break
    return history
''', '''
MAX_ROUNDS = 3


def should_stop(round_no: int, critique: str, last_score: int, this_score: int) -> bool:
    """Called after each revision. True = stop reflecting."""
    critic_found_nothing = critique.strip().lower().startswith("no change")
    out_of_rounds = round_no >= MAX_ROUNDS
    no_longer_paying = this_score <= last_score

    stop_when_clean = critic_found_nothing
    stop_on_any = critic_found_nothing or out_of_rounds or no_longer_paying

    return stop_on_any      # clean OR out of rounds OR the last round bought nothing


def reflect(draft: str, critique_of, revise, hard_cap: int = MAX_ROUNDS + 4) -> list:
    """[(round, text, score), ...]. hard_cap is a seatbelt, not the stop condition."""
    history = [(0, draft, score_branch(draft))]
    text = draft
    for n in range(1, hard_cap + 1):
        critique = critique_of(text)
        text = revise(text, critique)
        previous, now = history[-1][2], score_branch(text)
        history.append((n, text, now))
        if should_stop(n, critique, previous, now):
            break
    return history
'''),

    code('''
# --- Self-check: Section 2   (a canned critic and reviser -- deterministic, no model)
DRAFTS = [CANDIDATES["fastest"],
          "Do not grant anything. Send this to Identity and Access Management.",
          "Do not change entitlements. Identity and Access Management owns admin rights here."]

def canned_critique(text: str) -> str:
    return ("no change needed" if "identity and access" in text.lower()
            else "the handbook routes elevated rights to another team")

def canned_revise(text: str, critique: str) -> str:
    i = DRAFTS.index(text) if text in DRAFTS else len(DRAFTS) - 1
    return DRAFTS[min(i + 1, len(DRAFTS) - 1)]

def canned_history():
    return reflect(DRAFTS[0], canned_critique, canned_revise)


check("the stop rule fires when the critic runs out of things to say",
      lambda: should_stop(1, "no change needed", 2, 3) is True)
check("it also fires on the round budget, however talkative the critic",
      lambda: should_stop(MAX_ROUNDS, "one more nit", 1, 2) is True
              and should_stop(1, "one more nit", 1, 2) is False,
      "a critic asked for a fault will invent one, so 'until it is clean' alone never terminates")
check("round 1 earns its call (0 -> 3), and the loop then stops at the knee",
      lambda: canned_history()[1][2] == 3 and len(canned_history()) == 3,
      "round 2 scored the same as round 1; paying for round 3 is the failure this section prevents")
score()
'''),

    md("""
## Run it for real

Three branches from one `invoke`, scored and pruned. Then three rounds of reflection with the
score printed each round, so you can see where the knee is on this case.
"""),

    code('''
def live_branches():
    r = REQUESTS["EHD-7004"]
    out = branches_runnable(get_llm()).invoke(
        {"request": json.dumps({"id": "EHD-7004", **r}), "handbook": HANDBOOK[r["category"]]})
    for name, text in out.items():
        print(f"[{score_branch(text)}] {name}: {' '.join(text.split())[:150]}")
    best = max(out, key=lambda k: score_branch(out[k]))
    print("\\nkept:", best, "| pruned:", [k for k in out if k != best])

if llm_ready():
    guard(live_branches)
'''),

    code('''
def live_reflection():
    r = REQUESTS["EHD-7004"]
    ctx = f'REQUEST: {json.dumps({"id": "EHD-7004", **r})}\\nHANDBOOK: {HANDBOOK[r["category"]]}'

    def critic(text):
        return ask(f"{ctx}\\n\\nDRAFT: {text}\\n\\nName the single worst way this draft departs "
                   "from the handbook, in one sentence. If it does not, reply exactly with: "
                   "no change needed")

    def reviser(text, critique):
        return ask(f"{ctx}\\n\\nDRAFT: {text}\\nCRITIQUE: {critique}\\n\\nRewrite the draft in two "
                   "sentences, fixing only what the critique names. Output the draft only.")

    for n, text, sc in reflect(CANDIDATES["fastest"], critic, reviser):
        print(f"round {n}: score {sc} | {' '.join(text.split())[:120]}")

if llm_ready():
    guard(live_reflection)
'''),

    md("""
### Read it

Look at the branch scores first. The `fastest` branch is fluent, confident and does the thing the
handbook says the help desk never does. Under a scorer that rewards "the employee is working
again", it wins. Nothing about the branching stopped that &mdash; the branching only produced the
candidate. **The scorer is the design decision.** Writing three prompts is the easy half.

Now the reflection column. The first round moves the score; after that the number stops moving
while the calls keep going out. That is the knee, and it is the whole reason `should_stop` exists.
A loop that runs "until the critic is happy" does not terminate, because a critic asked to find a
fault finds one &mdash; so the budget is not a fallback, it is the termination proof.

`hard_cap` in `reflect` is a seatbelt so a wrong stop rule shows up as a long run rather than a
hung notebook. Do not mistake it for a stop condition; it is what you are trying not to reach.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Change `score_branch` to return the other scorer and re-run the live branch cell. Which branch
   is kept now, and what would have happened to Vikram Rao's entitlements?
2. Set `MAX_ROUNDS` to 8 and run the live reflection again. Count the calls, then find the last
   round that changed the score.
"""),
]
# =========================================================================== #
# Lab 2.5 -- challenge: the architecture bake-off
# =========================================================================== #
LAB5 = [
    header(5, "Challenge &mdash; The Architecture Bake-Off", "Advanced", 35,
           ["Write the acceptance bar down <i>before</i> you have seen a single result",
            "Build four architectures behind one interface: direct, chain-of-thought, ReAct, reflection",
            "Let one disqualifying case beat a pass-rate average"],
           THREAD_NOTE),
    setup(5),
    code(DOMAIN),

    md("""
## The brief

Four architectures. The same five help desk requests. One interface, so they are swappable and the
only thing that varies is how the arm reasons.

The comparison is the easy part. The part teams get wrong is deciding what "better" means
*after* they have seen the table, which is how the arm that reads best on average gets shipped
with a hole in it.

So the bar comes first, in Section 1, before any arm exists. Two ideas have to be in it:

* **a minimum pass rate** &mdash; how many of the five it has to get right, and
* **a disqualifying case** &mdash; something no average can buy back. `EHD-7004` asks for
  `admin-rights`, which is in `ROUTE_OUT`. The help desk never grants that. An arm that resolves it
  itself is rejected however well it scores elsewhere.
"""),

    md("""
## Section 1 &mdash; The acceptance bar, written first

Two decisions, and you make both of them blind. That is the point: a bar you write after seeing
the results is not a bar, it is a description of the winner.
"""),

    code('''
def acceptance_bar() -> dict:
    """The rule every arm is judged by. Written before any arm has run."""
    # Five requests. A 60% bar accepts an arm that is wrong about two employees in five.
    lenient    = {"min_pass_rate": 0.60, "reject_if_resolves_route_out": True}
    defensible = {"min_pass_rate": 0.80, "reject_if_resolves_route_out": True}
    return BLANK        # TODO: which of these would you sign your name to, sight unseen?


def judge(result: dict, bar: dict) -> str:
    """result: {"arm", "pass_rate", "resolved_route_out"} -> "accepted" or "rejected"."""
    clears_bar   = result["pass_rate"] >= bar["min_pass_rate"]
    disqualified = result["resolved_route_out"] and bar["reject_if_resolves_route_out"]

    # Two readings of an arm that granted admin rights and still scored top of the table:
    average_wins = "accepted" if clears_bar else "rejected"
    rule_wins    = "rejected" if disqualified else ("accepted" if clears_bar else "rejected")
    return BLANK        # TODO: can a good average buy back granting admin rights?
''', '''
def acceptance_bar() -> dict:
    """The rule every arm is judged by. Written before any arm has run."""
    # Five requests. A 60% bar accepts an arm that is wrong about two employees in five.
    lenient    = {"min_pass_rate": 0.60, "reject_if_resolves_route_out": True}
    defensible = {"min_pass_rate": 0.80, "reject_if_resolves_route_out": True}
    return defensible   # one wrong answer in five is the most we would defend


def judge(result: dict, bar: dict) -> str:
    """result: {"arm", "pass_rate", "resolved_route_out"} -> "accepted" or "rejected"."""
    clears_bar   = result["pass_rate"] >= bar["min_pass_rate"]
    disqualified = result["resolved_route_out"] and bar["reject_if_resolves_route_out"]

    # Two readings of an arm that granted admin rights and still scored top of the table:
    average_wins = "accepted" if clears_bar else "rejected"
    rule_wins    = "rejected" if disqualified else ("accepted" if clears_bar else "rejected")
    return rule_wins    # the disqualifying case is not an input to an average
'''),

    code('''
# --- Self-check: Section 1   (the bar, over hand-written results -- no arms, no model)
def verdict(rate: float, resolved_route_out: bool) -> str:
    return judge({"arm": "x", "pass_rate": rate, "resolved_route_out": resolved_route_out}, acceptance_bar())

check("3 of 5 right is not good enough for an employee who is one of the other two",
      lambda: verdict(0.60, False) == "rejected",
      "a 60% bar is the one you write when you already know the scores")
check("4 of 5 right, nothing wrongly resolved: accepted",
      lambda: verdict(0.80, False) == "accepted")
check("5 of 5 right but it granted admin rights itself: rejected anyway",
      lambda: verdict(1.00, True) == "rejected",
      "no pass-rate average buys back a request the help desk must never handle")
score()
'''),

    md("""
## Section 2 &mdash; Four arms behind one interface

Each arm is the same LangChain shape &mdash; a `ChatPromptTemplate`, the model, a parser &mdash; and
differs only in the system line that tells it how to work. That is what makes the comparison fair:
same case template, same model, same output schema, one variable.

The output schema is a Pydantic model, so every arm has to answer in the same terms whatever it did
to get there. `Field(description=...)` is not a comment &mdash; it is what the model actually reads.
"""),

    code('''
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Handling(BaseModel):
    """What every arm must produce, whatever reasoning style it used to get there."""
    action: Literal["resolve", "route_out"] = Field(
        description="resolve if the help desk handles it; route_out if another team must")
    sla_hours: int = Field(description="hours allowed, from SLA_HOURS unless a handbook rule is stricter")
    why: str = Field(description="one sentence naming the handbook rule you applied")


CASE = ("Request {rid}: {text}\\n"
        "category={category} urgency={urgency} wants={wants}\\n"
        "Handbook rule for this category: {rule}\\n"
        "SLA hours by urgency: {sla}\\n"
        "The help desk must NEVER do these itself: {route_out}")

STYLES = {
    "direct":  "Answer immediately. Do not show any working.",
    "cot":     "Work in order in your reply: category, then the handbook rule, then whether that "
               "rule overrides the urgency the employee set, then the action.",
    "react":   "Work in Thought / Action / Observation steps, then a final answer. Actions: "
               "handbook(category), sla(urgency).",
    "reflect": "Draft an answer, critique your own draft against the handbook rule, then correct it.",
}


def arm_prompt(style: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "You triage employee help desk requests. " + STYLES[style]),
        ("human", CASE),
    ])

def build_arm(style: str, llm) -> dict:
    """One arm, two chains over the same prompt: the prose it wrote, the verdict we score."""
    prompt = arm_prompt(style)
    return {"prose":   prompt | llm | StrOutputParser(),
            "verdict": prompt | llm.with_structured_output(Handling),
            # no parser on this leg, so the AIMessage -- and its usage_metadata -- survives
            "raw":     prompt | llm}

def case_inputs() -> list:
    return [{"rid": rid, "text": r["text"], "category": r["category"], "urgency": r["urgency"],
             "wants": r["wants"], "rule": HANDBOOK[r["category"]],
             "sla": SLA_HOURS, "route_out": sorted(ROUTE_OUT)}
            for rid, r in sorted(REQUESTS.items())]
'''),

    md("""
Now the selection rule. Four arms have run, you have a pass rate for each and whether each one
resolved `EHD-7004`. The table is in front of you &mdash; and this is exactly the moment the bar you
wrote in Section 1 has to be allowed to win.
"""),

    code('''
# What a correct handling looks like: (action, the slowest clock we would accept)
EXPECTED = {
    "EHD-7001": ("resolve",    4),
    "EHD-7002": ("resolve",    4),    # the safety rule, not the urgency box, sets this one
    "EHD-7003": ("resolve",   72),    # 180 USD needs approval first -- but it is still ours
    "EHD-7004": ("route_out", 24),    # the disqualifying case
    "EHD-7005": ("resolve",   72),
}

def graded(rid: str, h) -> bool:
    action, max_hours = EXPECTED[rid]
    return h is not None and h.action == action and h.sla_hours <= max_hours


def pick_winner(results: list, bar: dict):
    """results: {"arm", "pass_rate", "resolved_route_out", "tokens"}. Returns one, or None."""
    accepted = [r for r in results if judge(r, bar) == "accepted"]

    top_scorer       = sorted(results, key=lambda r: -r["pass_rate"])[0]
    best_accepted    = sorted(accepted, key=lambda r: -r["pass_rate"])[0] if accepted else None
    cheapest_accepted = sorted(accepted, key=lambda r: r["tokens"])[0] if accepted else None

    return BLANK        # TODO: the bar already says what quality you need. So what does cost decide?
''', '''
# What a correct handling looks like: (action, the slowest clock we would accept)
EXPECTED = {
    "EHD-7001": ("resolve",    4),
    "EHD-7002": ("resolve",    4),    # the safety rule, not the urgency box, sets this one
    "EHD-7003": ("resolve",   72),    # 180 USD needs approval first -- but it is still ours
    "EHD-7004": ("route_out", 24),    # the disqualifying case
    "EHD-7005": ("resolve",   72),
}

def graded(rid: str, h) -> bool:
    action, max_hours = EXPECTED[rid]
    return h is not None and h.action == action and h.sla_hours <= max_hours


def pick_winner(results: list, bar: dict):
    """results: {"arm", "pass_rate", "resolved_route_out", "tokens"}. Returns one, or None."""
    accepted = [r for r in results if judge(r, bar) == "accepted"]

    top_scorer       = sorted(results, key=lambda r: -r["pass_rate"])[0]
    best_accepted    = sorted(accepted, key=lambda r: -r["pass_rate"])[0] if accepted else None
    cheapest_accepted = sorted(accepted, key=lambda r: r["tokens"])[0] if accepted else None

    # Above the bar, extra quality is something you are paying for and not using. Ranking only
    # ever runs over arms that already cleared it, so cost is what is left to decide.
    return cheapest_accepted
'''),

    code('''
# --- Self-check: Section 2   (prompt objects, and the rule over results we typed by hand)
FIELDS = {"rid", "text", "category", "urgency", "wants", "rule", "sla", "route_out"}
RECORDED = [
    {"arm": "direct",  "pass_rate": 0.80, "resolved_route_out": False, "tokens":  2100},
    {"arm": "cot",     "pass_rate": 1.00, "resolved_route_out": False, "tokens":  5400},
    {"arm": "react",   "pass_rate": 0.80, "resolved_route_out": True,  "tokens":  6800},
    {"arm": "reflect", "pass_rate": 0.80, "resolved_route_out": False, "tokens": 11900},
]

check("all four arms are ChatPromptTemplates over one identical case",
      lambda: all(set(arm_prompt(s).input_variables) == FIELDS for s in STYLES) and len(STYLES) == 4,
      "if the arms saw different facts you would be comparing prompts, not architectures")
check("the schema tells the model what route_out means, in words it reads",
      lambda: "route_out" in Handling.model_fields["action"].description and "route_out" in str(Handling.model_json_schema()))
check("of the three arms that clear the bar, the cheapest is what ships",
      lambda: pick_winner(RECORDED, acceptance_bar())["arm"] == "direct",
      "cot scores higher and costs 2.5x -- above the bar you are paying for quality you said "
      "you did not need")
check("if the cheapest arm is the disqualified one, it is still not shipped",
      lambda: pick_winner([{**r, "tokens": 100 if r["arm"] == "react" else r["tokens"]}
                           for r in RECORDED], acceptance_bar())["arm"] != "react")
check("react is refused even when it alone tops the table",
      lambda: pick_winner([{**r, "pass_rate": 1.00 if r["arm"] == "react" else 0.80}
                           for r in RECORDED], acceptance_bar())["arm"] != "react",
      "sort first and you ship the disqualified arm -- filter first, then rank")
score()
'''),

    md("""
## Run it for real &mdash; the bake-off

Four arms, five requests each, run with `.batch()` so the five go out together. One prose sample
first, so you can see what the arm actually wrote before you read it as a number.
"""),

    code('''
def bake_off():
    llm, inputs, bar = get_llm(), case_inputs(), acceptance_bar()
    print(f"bar: pass rate >= {bar['min_pass_rate']:.0%}; resolving {sorted(ROUTE_OUT)} disqualifies\\n")
    print("--- chain-of-thought, in its own words, on EHD-7004 ---")
    print(build_arm("cot", llm)["prose"].invoke(inputs[3]).strip()[:420], "\\n")

    results = []
    for style in STYLES:
        arm = build_arm(style, llm)
        out = arm["verdict"].batch(inputs)
        passed = sum(1 for i, h in zip(inputs, out) if graded(i["rid"], h))
        wrong  = any(h is not None and h.action == "resolve"
                     for i, h in zip(inputs, out) if i["rid"] == "EHD-7004")
        # what the arm cost: the prose leg carries the usage metadata the verdict leg hides
        msgs = arm["raw"].batch(inputs)
        toks = sum((getattr(m, "usage_metadata", None) or {}).get("total_tokens", 0) for m in msgs)
        results.append({"arm": style, "pass_rate": passed / len(inputs),
                        "resolved_route_out": wrong, "tokens": toks})

    print(f"{'arm':10} {'pass':>5} {'tokens':>8}  {'EHD-7004 resolved':>18}  verdict")
    for r in results:
        print(f"{r['arm']:10} {r['pass_rate']:>5.0%} {r['tokens']:>8}  "
              f"{str(r['resolved_route_out']):>18}  {judge(r, bar)}")

    win = pick_winner(results, bar)
    print("\\nship:", win["arm"] if win else "nothing -- no arm cleared the bar")
    print("      (of the arms that cleared the bar, the cheapest one)")

if llm_ready():
    guard(bake_off)
'''),

    md("""
### Read it

The shape we recorded on this sandbox: **direct 80%, chain-of-thought 100%, ReAct 80%, reflection
80%** &mdash; and ReAct **rejected**, because it resolved `EHD-7004` itself instead of routing the
admin-rights request out. Your run will not match it exactly. The same arm can score 80% on one
pass and 60% on the next with nothing changed; a small model is not a fixed function, and four
runs of five cases is far too little data to call one architecture better than another by two
percentage points.

**What does not move is the rule.** ReAct got rejected on a case, not on an average, and a case
verdict is stable in a way a mean of five is not. That asymmetry is the whole lab: put your
confidence in the disqualifying cases you can name, not in the third decimal place of a pass rate.
Notice also that `pick_winner` filters before it ranks. Rank first and the highest number wins,
which is precisely how a disqualified arm gets shipped.

**And look at the tokens column before you crown chain-of-thought.** It scores highest and costs
several times the direct arm. The bar already said what quality you need; above it, extra quality
is something you are paying for and not using &mdash; so `pick_winner` ships the *cheapest arm that
clears*, which on this set is usually the direct one. Chain-of-thought earns its cost only where it
changes the verdict: two of these five cases need two facts combined (`EHD-7002`'s safety rule
against the urgency box, `EHD-7003`'s 180 USD against the approval threshold). On a set of one-fact
requests it would buy you nothing and bill you for it.

That is also why the bar is written first. Decide the quality you need, then buy it as cheaply as
you can &mdash; rather than admiring the biggest number and calling it a decision.

**What you take from Module 2.** Reasoning style is a design choice with a measurable cost, not a
personality. Chain-of-thought makes the intermediate facts explicit; ReAct interleaves acting with
thinking and lives or dies on the argument contract, not the text format; reflection buys a second
look at your own draft. None of them is the default. You pick with an eval set, an acceptance bar
written before the results, and at least one case that no average is allowed to override. Module 3
turns the reasoning you chose here into a graph you can pause, inspect and resume.
"""),

    code('''
score()
'''),

    md("""
## Your turn

1. Move `min_pass_rate` to `1.0` and re-run the selection. Most likely nothing ships. Decide which
   you would actually do &mdash; lower the bar, or keep the human in the loop for the cases no arm
   gets right &mdash; and say what your answer implies about who carries the risk.
2. Add a sixth request that also `wants` something in `ROUTE_OUT`, and re-run. If an arm routes one
   out and resolves the other, is that a pass rate of 50% on those two, or still a disqualification?
   Your answer is a change to `judge`, so make it.
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-2-01-chain-of-thought",          LAB1),
    ("lab-2-02-react-argument-contract",   LAB2),
    ("lab-2-03-subgoals-and-replanning",   LAB3),
    ("lab-2-04-branch-score-reflect",      LAB4),
    ("lab-2-05-challenge-architecture-bakeoff", LAB5),
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
