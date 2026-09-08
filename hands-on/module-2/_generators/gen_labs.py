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
# the shared synthetic domain -- one use case runs through all five labs
# --------------------------------------------------------------------------- #
DOMAIN = '''
# ------------------------------------------------- the case file (synthetic, self-contained)
# One domain runs through all five Module 2 labs: payment exceptions on a small ledger.
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
# Lab 2.1 -- Chain-of-Thought, built as an LCEL chain and measured
# =========================================================================== #
LAB1 = [
    header(1, "Chain-of-Thought, Built as a Chain", "Intermediate", 35,
           ["Compose your first LCEL chain: <code>prompt | model | parser</code>",
            "Build two arms that differ only in their prompt, and nothing else",
            "Run a whole eval set in one call with <code>.batch()</code>",
            "Measure what &ldquo;think step by step&rdquo; is actually worth on your task"],
           "> **The thread.** All five Module 2 labs work one case: payment exceptions on a small\n"
           "> synthetic ledger. Module 1 built the agent loop; Module 2 is about what goes on\n"
           "> inside one turn of it."),
    setup(1),
    code(DOMAIN),

    md("""
## Concept

**Chain-of-Thought** asks the model to show its working before it answers. It usually helps on
multi-step tasks and usually costs latency, and *how much* of each is a property of your task,
not a fact about LLMs. So measure it.

The mechanism you use to measure it is worth as much as the answer. **LCEL** &mdash; LangChain
Expression Language &mdash; composes a prompt, a model and a parser into one runnable with `|`:

```python
chain = prompt | model | StrOutputParser()
chain.invoke({"case": ...})        # one
chain.batch([{...}, {...}, ...])   # many, concurrently
```

Everything downstream in this course is built this way, and `.batch()` is what makes an eval set
practical instead of a coffee break.
"""),

    md("""
## Section 1 &mdash; Your first chain

Three parts, one pipe. `ChatPromptTemplate` turns variables into messages; the model answers;
`StrOutputParser` pulls `.content` out so the chain returns a plain string.
"""),
    code('''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

DIRECT = ("You are a payments operations analyst. Answer with the single next action, "
          "and nothing else.")

def output_parser():
    """The last stage of the chain: what turns the AIMessage into a plain string?"""
    return BLANK                      # TODO: which parser, INSTANTIATED -- note the ()


def build_chain(system: str):
    """prompt | model | parser -- the three-part chain every later lab uses."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "PAYMENT: {payment}\\nPOLICY: {policy}\\n\\nWhat must happen next?"),
    ])
    return prompt | get_llm() | output_parser()
''', '''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

DIRECT = ("You are a payments operations analyst. Answer with the single next action, "
          "and nothing else.")

def output_parser():
    """The last stage of the chain: what turns the AIMessage into a plain string?"""
    return StrOutputParser()


def build_chain(system: str):
    """prompt | model | parser -- the three-part chain every later lab uses."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "PAYMENT: {payment}\\nPOLICY: {policy}\\n\\nWhat must happen next?"),
    ])
    return prompt | get_llm() | output_parser()
'''),
    code('''
# --- Self-check: Section 1   (the prompt half is pure -- no model call)
_p = ChatPromptTemplate.from_messages([
    ("system", DIRECT),
    ("human", "PAYMENT: {payment}\\nPOLICY: {policy}\\n\\nWhat must happen next?"),
])

check("the template declares both variables",
      lambda: set(_p.input_variables) == {"payment", "policy"})
check("it renders to a system message and a human message",
      lambda: [m.type for m in _p.invoke({"payment": "p", "policy": "q"}).messages]
              == ["system", "human"])
check("the variables are substituted, not left as braces",
      lambda: "PAYMENT: p" in _p.invoke({"payment": "p", "policy": "q"}).messages[1].content)

check("the chain ends in a string parser",
      lambda: isinstance(output_parser(), StrOutputParser),
      "it must be StrOutputParser() -- an INSTANCE; passing the class silently breaks the pipe")
check("the parser turns an AIMessage into a plain string",
      lambda: output_parser().invoke(AIMessage("hello")) == "hello",
      "that is all StrOutputParser does, and it is why the chain returns str not AIMessage")
'''),

    md("""
## Section 2 &mdash; Two arms, one difference

Both arms use the **same** `build_chain`. The only thing that changes is the system prompt. That
is what makes the comparison worth anything: if you also changed the model, the temperature or
the question, you would learn nothing from the result.

Write the Chain-of-Thought prompt. It has to ask for the working *and* keep the final answer
findable, or you cannot score it.
"""),
    code('''
ANSWER_MARKER = "ACTION:"

# TODO: replace this string with your Chain-of-Thought system prompt. It must ask for the
# reasoning first and finish with a line beginning ANSWER_MARKER, or you cannot score it.
COT = "BLANK"
''', '''
ANSWER_MARKER = "ACTION:"

COT = ("You are a payments operations analyst. Work through the case step by step: state the "
       "payment's status, then its reason code, then what the policy for that code requires. "
       f"Finish with a final line that begins with {ANSWER_MARKER!r} followed by the single "
       "next action and nothing else.")
'''),
    code('''
def final_action(text: str) -> str:
    """The answer, separated from the working.

    With the marker: everything after the LAST marker line.
    Without it: the whole text, which is what the direct arm produces.
    """
    if ANSWER_MARKER in text:
        return text.rsplit(ANSWER_MARKER, 1)[1].strip()
    return text.strip()
'''),
    code('''
# --- Self-check: Section 2   (string handling only -- no model call)
def _cot():
    # A blank inside a string is not a blank -- it is the literal word, and reading it can
    # never raise. So raise it by hand, or an untouched lab shows [FAIL] instead of [TODO].
    if not isinstance(COT, str) or COT.strip() == "BLANK":
        raise NameError("COT is not written yet")
    return COT

check("the CoT prompt asks for reasoning",
      lambda: any(w in _cot().lower() for w in ("step by step", "step-by-step", "work through")),
      "if it does not ask for the working, it is not Chain-of-Thought")
check("the CoT prompt names the answer marker",
      lambda: ANSWER_MARKER in _cot(),
      "you have to be able to find the answer inside the working, or you cannot score it")
check("both arms are still the same analyst",
      lambda: "payments operations analyst" in _cot(),
      "change one thing between arms -- the reasoning instruction -- and nothing else")
check("final_action strips the working away",
      lambda: final_action("thinking...\\nACTION: Escalate to Treasury.") == "Escalate to Treasury.")
check("final_action takes the LAST marker",
      lambda: final_action("ACTION: wrong\\nmore\\nACTION: right") == "right",
      "models sometimes restate the format before using it")
check("an unmarked answer survives unchanged",
      lambda: final_action("Escalate to Treasury.") == "Escalate to Treasury.")
'''),

    md("""
## Section 3 &mdash; The eval set, and `.batch()`

Five cases, each with the terms a correct action must mention. Crude on purpose &mdash; it has to be
something you can defend and re-run, not something you have to read.

`.batch()` sends the whole list at once and returns the answers in order. It is the difference
between an eval set you run every time and one you ran once.
"""),
    code('''
CASES = [
    {"ref": "PMT-1003", "must_contain": ["treasury"]},
    {"ref": "PMT-1005", "must_contain": ["compliance"]},
    {"ref": "PMT-1002", "must_contain": ["retry"]},
    {"ref": "PMT-1004", "must_contain": ["originator", "r04"]},
    {"ref": "PMT-1001", "must_contain": ["settled", "no action", "none"]},
]

def inputs_for(case: dict) -> dict:
    """The variables one case supplies to the chain."""
    rec = LEDGER[case["ref"]]
    return {"payment": json.dumps({"ref": case["ref"], **rec}),
            "policy": POLICY.get(rec["reason_code"], "no policy applies")}


def scores(answers: list[str]) -> list[bool]:
    """One bool per case: did the final action mention any required term?"""
    out = []
    for case, answer in zip(CASES, answers):
        action = final_action(answer).lower()
        out.append(any(term in action for term in case["must_contain"]))
    return out


def run_arm(system: str) -> dict:
    """Run every case through one arm, in a single batched call."""
    chain = build_chain(system)
    t0 = time.time()
    answers = chain.batch([inputs_for(c) for c in CASES])
    return {"answers": answers, "scores": scores(answers), "seconds": time.time() - t0}
''', '''
CASES = [
    {"ref": "PMT-1003", "must_contain": ["treasury"]},
    {"ref": "PMT-1005", "must_contain": ["compliance"]},
    {"ref": "PMT-1002", "must_contain": ["retry"]},
    {"ref": "PMT-1004", "must_contain": ["originator", "r04"]},
    {"ref": "PMT-1001", "must_contain": ["settled", "no action", "none"]},
]

def inputs_for(case: dict) -> dict:
    """The variables one case supplies to the chain."""
    rec = LEDGER[case["ref"]]
    return {"payment": json.dumps({"ref": case["ref"], **rec}),
            "policy": POLICY.get(rec["reason_code"], "no policy applies")}


def scores(answers: list[str]) -> list[bool]:
    """One bool per case: did the final action mention any required term?"""
    out = []
    for case, answer in zip(CASES, answers):
        action = final_action(answer).lower()
        out.append(any(term in action for term in case["must_contain"]))
    return out


def run_arm(system: str) -> dict:
    """Run every case through one arm, in a single batched call."""
    chain = build_chain(system)
    t0 = time.time()
    answers = chain.batch([inputs_for(c) for c in CASES])
    return {"answers": answers, "scores": scores(answers), "seconds": time.time() - t0}
'''),
    code('''
# --- Self-check: Section 3   (the scorer, on canned answers -- no model call)
_canned = ["Escalate to Treasury for approval.",
           "Hold; Compliance decides.",
           "Retry once after 24h.",
           "Do something vague.",
           "Already settled, no action."]

check("every case names a payment that exists",
      lambda: all(c["ref"] in LEDGER for c in CASES))
check("the chain inputs carry the payment and the policy",
      lambda: set(inputs_for(CASES[0])) == {"payment", "policy"})
check("a case with no reason code still gets a policy string",
      lambda: isinstance(inputs_for(CASES[4])["policy"], str),
      "PMT-1001 is settled -- the chain must still receive something for {policy}")
check("the scorer accepts a correct answer", lambda: scores(_canned)[0] is True)
check("the scorer rejects a vague one",      lambda: scores(_canned)[3] is False)
check("either term is enough",
      lambda: scores(["x", "x", "x", "Send it back with code R04.", "x"])[3] is True,
      'must_contain is a list of alternatives -- "any", not "all"')
check("the scorer reads only the final action",
      lambda: scores(["I considered Treasury but ACTION: hold for Compliance.",
                      "x", "x", "x", "x"])[0] is False,
      "reasoning that mentions the right word is not the same as answering it")
'''),

    md("""
## Run it for real

Both arms, five cases each, two batched calls. Watch the per-case table rather than the totals.
"""),
    code('''
if llm_ready():
    def _compare():
        direct = run_arm(DIRECT)
        cot    = run_arm(COT)
        print("  case       direct   chain-of-thought")
        for c, a, b in zip(CASES, direct["scores"], cot["scores"]):
            f = lambda x: "pass" if x else "FAIL"
            print(f"  {c['ref']}   {f(a):8} {f(b)}")
        n = len(CASES)
        print(f"\\n  direct           {sum(direct['scores'])}/{n}   {direct['seconds']:.1f}s")
        print(f"  chain-of-thought {sum(cot['scores'])}/{n}   {cot['seconds']:.1f}s")
        print("\\n--- one CoT answer in full ---\\n")
        print(cot["answers"][0][:700])
        return {"direct": direct, "cot": cot}
    ARMS = guard(_compare)
'''),
    md("""
### Read it

Three things to look for.

1. **Which cases moved.** CoT rarely helps uniformly. It tends to earn its keep exactly where the
   answer needs two facts joined &mdash; the reason code *and* what policy says about it &mdash; and to
   do nothing at all where one lookup was enough.
2. **What the working looks like.** Read the full answer printed at the end. The model states the
   status, then the code, then the policy. That ordering is the whole mechanism: each step lands
   in the context before the next one needs it.
3. **The marker earned its place.** Without `ACTION:` you would be scoring the reasoning as well
   as the answer, and CoT would appear to win simply by mentioning more words. The
   `final_action` self-check above encodes exactly that trap.

Note also what `.batch()` did: five cases went out together rather than one after another. The
same eval set run serially takes several times as long, which is usually the difference between
a suite people run and one they skip.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add a sixth case whose correct action needs **three** facts joined, and see whether the gap
   between the arms widens. That is the shape of task where CoT pays.
2. Replace `StrOutputParser()` with `with_structured_output` returning a small model that has
   `reasoning` and `action` fields. You no longer need `ANSWER_MARKER` or `final_action` at all
   &mdash; which of the two designs would you rather maintain?
3. Run `run_arm(COT)` three times. The scores will not be identical. Decide how many runs your
   eval set needs before you would let it gate a release, and write the number down.
"""),
]


# =========================================================================== #
# Lab 2.2 -- ReAct: the parser contract, and the version that has none
# =========================================================================== #
CARRY_TOOLS = '''
# ------------------------------------------------- the two tools, carried through Module 2
from langchain_core.tools import tool

@tool
def lookup_payment(ref: str) -> str:
    """Return the ledger record for one payment reference such as 'PMT-1003'.

    Use when you need the status, amount, counterparty or reason code of a specific payment.
    """
    rec = LEDGER.get(ref)
    return json.dumps({"ref": ref, **rec}) if rec else f"no payment found with reference {ref!r}"


@tool
def policy_for(reason_code: str) -> str:
    """Return the operating policy for one failure reason code such as 'LIMIT_BREACH'.

    Use after you know why a payment failed and need to know what to do about it.
    """
    return POLICY.get(reason_code, f"no policy on file for reason code {reason_code!r}")


TOOLS = {t.name: t for t in (lookup_payment, policy_for)}
print("tools:", list(TOOLS))
'''

LAB2 = [
    header(2, "ReAct, and the Parser You Do Not Have to Write", "Intermediate &rarr; Advanced", 40,
           ["Implement the classic text ReAct format &mdash; Thought / Action / Observation",
            "Collect the eight ways a model drifts out of that format",
            "Discover the failure a parser cannot catch: a valid format with an unusable argument",
            "Do the same job with <code>bind_tools</code>, where the argument is schema-checked"],
           "> **Builds on Lab 2.1.** Same case file. This lab is the strongest argument in Module 2\n"
           "> for using the framework rather than reimplementing it."),
    setup(2),
    code(DOMAIN),
    code(CARRY_TOOLS),

    md("""
## Concept

**ReAct** interleaves reasoning and acting: *Thought* (what do I know?), *Action* (what shall I
do?), *Observation* (what came back?), repeat. That is the loop from Module 1, given a shape.

The original formulation asks the model to emit that shape **as text**, which means you must parse
it. Every parser is a contract, and the model has not signed it. This lab makes you write that
parser, breaks it, and then shows you the version where the contract is enforced by the API
instead of by your regex.
"""),

    md("""
## Section 1 &mdash; The text format, and the parser it needs

The happy path looks like this:

```
Thought: I need the payment record first.
Action: lookup_payment
Action Input: PMT-1003
```

Pull the three fields out. Return `None` for a step that does not carry an action &mdash; that is how
a final answer looks.
"""),
    code('''
import re

STEP_RE = re.compile(
    r"Thought:\\s*(?P<thought>.*?)\\s*"
    r"Action:\\s*(?P<action>[\\w_]+)\\s*"
    r"Action Input:\\s*(?P<input>.*?)\\s*$",
    re.DOTALL | re.IGNORECASE)

def parse_step(text: str) -> dict | None:
    """Return {"thought", "action", "input"} or None if this is not an action step."""
    m = STEP_RE.search(text or "")
    if not m:
        return None
    return {"thought": m.group("thought").strip(),
            "action": m.group("action").strip(),
            "input": m.group("input").strip().strip('"').strip("'")}
''', '''
import re

STEP_RE = re.compile(
    r"Thought:\\s*(?P<thought>.*?)\\s*"
    r"Action:\\s*(?P<action>[\\w_]+)\\s*"
    r"Action Input:\\s*(?P<input>.*?)\\s*$",
    re.DOTALL | re.IGNORECASE)

def parse_step(text: str) -> dict | None:
    """Return {"thought", "action", "input"} or None if this is not an action step."""
    m = STEP_RE.search(text or "")
    if not m:
        return None
    return {"thought": m.group("thought").strip(),
            "action": m.group("action").strip(),
            "input": m.group("input").strip().strip('"').strip("'")}
'''),
    code('''
# --- Self-check: Section 1   (pure string work -- no model call)
_good = "Thought: I need the record.\\nAction: lookup_payment\\nAction Input: PMT-1003"

check("a well-formed step parses",       lambda: parse_step(_good) is not None)
check("the tool name is extracted",      lambda: parse_step(_good)["action"] == "lookup_payment")
check("the input is extracted",          lambda: parse_step(_good)["input"] == "PMT-1003")
check("the thought is extracted",        lambda: "record" in parse_step(_good)["thought"])
check("a final answer is not an action", lambda: parse_step("The payment needs Treasury.") is None)
check("quotes around the input are stripped",
      lambda: parse_step('Thought: t\\nAction: lookup_payment\\nAction Input: "PMT-1003"')["input"]
              == "PMT-1003")
'''),

    md("""
## Section 2 &mdash; Eight ways it drifts

None of these is a badly behaved model. Every one is a reasonable thing for a fluent writer to
produce, and every one breaks a parser that was written against the happy path.

Decide which your parser should accept. There is no free answer here: a **lenient** parser
guesses and is sometimes wrong; a **strict** one refuses and costs you a retry.
"""),
    code('''
DRIFT = [
    ("markdown fences",   "```\\nThought: t\\nAction: lookup_payment\\nAction Input: PMT-1003\\n```"),
    ("bold headings",     "**Thought:** t\\n**Action:** lookup_payment\\n**Action Input:** PMT-1003"),
    ("numbered steps",    "1. Thought: t\\n2. Action: lookup_payment\\n3. Action Input: PMT-1003"),
    ("json instead",      \'{"thought": "t", "action": "lookup_payment", "input": "PMT-1003"}\'),
    ("prose action",      "Thought: t\\nAction: I will call lookup_payment\\nAction Input: PMT-1003"),
    ("missing input",     "Thought: t\\nAction: lookup_payment"),
    ("two actions",       "Thought: t\\nAction: lookup_payment\\nAction Input: PMT-1003\\n"
                          "Action: policy_for\\nAction Input: LIMIT_BREACH"),
    ("preamble",          "Certainly! Here is my reasoning.\\n\\nThought: t\\n"
                          "Action: lookup_payment\\nAction Input: PMT-1003"),
]

def survives(text: str) -> bool:
    """Does the Section 1 parser get a usable tool name out of this?"""
    step = parse_step(text)
    return bool(step) and step["action"] in TOOLS

def _drift_table():
    for label, text in DRIFT:
        print(f"  [{'ok  ' if survives(text) else 'LOST'}] {label}")
guard(_drift_table)
'''),
    code('''
# --- Self-check: Section 2   (what a parser must and must not do)
_by = dict(DRIFT)

check("the plain happy path still works",
      lambda: survives("Thought: t\\nAction: lookup_payment\\nAction Input: PMT-1003"))
check("a prose action does NOT yield a real tool",
      lambda: survives(_by["prose action"]) is False,
      '"I will call lookup_payment" must not be accepted as the tool name')
check("a missing input is not silently accepted",
      lambda: parse_step(_by["missing input"]) is None,
      "half a step is not a step -- accepting it invents an argument")
check("raw JSON is not the text format",
      lambda: parse_step(_by["json instead"]) is None,
      "this one is worth noting: the model produced something BETTER, and the parser rejects it")
check("at least three of the eight drift cases are lost",
      lambda: sum(1 for _, t in DRIFT if not survives(t)) >= 3,
      "if your parser accepts everything it is guessing, which is worse")
'''),

    md("""
## Section 3 &mdash; The same job, with no parser at all

`llm.bind_tools([...])` gives the model a tool **schema** rather than a format instruction. What
comes back is `AIMessage.tool_calls` &mdash; a list of `{"name", "args", "id"}` produced by the
serving stack, not by the model's prose. There is no format to drift out of, because there is no
format: the name is validated against the tools you passed, and the arguments against their
schema.

Write the step that turns one of those calls into a result.
"""),
    code('''
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

REACT_SYSTEM = ("You investigate payment exceptions. Use the tools to find the payment's reason "
                "code and the policy for it, then answer with the single next action.")

def native_step(model, messages: list) -> tuple[AIMessage, list]:
    """One turn: ask the model, run whatever it asked for, return (reply, tool results)."""
    ai = model.invoke(messages)
    results = []
    for call in ai.tool_calls:
        tool_obj = TOOLS.get(call["name"])
        content = tool_obj.invoke(call["args"]) if tool_obj else f"no such tool {call['name']!r}"
        results.append(ToolMessage(content=str(content), tool_call_id=BLANK))  # TODO: pair them up
    return ai, results


def native_loop(question: str, max_steps: int = 6) -> dict:
    """The ReAct loop with no parser in it."""
    model = get_llm().bind_tools(list(TOOLS.values()))
    messages = [SystemMessage(REACT_SYSTEM), HumanMessage(question)]
    for step in range(max_steps):
        ai, results = native_step(model, messages)
        messages.append(ai)
        if not ai.tool_calls:
            return {"messages": messages, "steps": step, "answer": ai.content}
        messages.extend(results)
    return {"messages": messages, "steps": max_steps, "answer": "(step budget spent)"}
''', '''
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

REACT_SYSTEM = ("You investigate payment exceptions. Use the tools to find the payment's reason "
                "code and the policy for it, then answer with the single next action.")

def native_step(model, messages: list) -> tuple[AIMessage, list]:
    """One turn: ask the model, run whatever it asked for, return (reply, tool results)."""
    ai = model.invoke(messages)
    results = []
    for call in ai.tool_calls:
        tool_obj = TOOLS.get(call["name"])
        content = tool_obj.invoke(call["args"]) if tool_obj else f"no such tool {call['name']!r}"
        results.append(ToolMessage(content=str(content), tool_call_id=call["id"]))
    return ai, results


def native_loop(question: str, max_steps: int = 6) -> dict:
    """The ReAct loop with no parser in it."""
    model = get_llm().bind_tools(list(TOOLS.values()))
    messages = [SystemMessage(REACT_SYSTEM), HumanMessage(question)]
    for step in range(max_steps):
        ai, results = native_step(model, messages)
        messages.append(ai)
        if not ai.tool_calls:
            return {"messages": messages, "steps": step, "answer": ai.content}
        messages.extend(results)
    return {"messages": messages, "steps": max_steps, "answer": "(step budget spent)"}
'''),
    code('''
# --- Self-check: Section 3   (a scripted model -- real message objects, no endpoint)
class _ScriptedModel:
    """Stands in for a bound model so the loop can be tested without the gateway."""
    def __init__(self, replies): self._replies = list(replies)
    def invoke(self, messages): return self._replies.pop(0)

def _call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args,
                                              "id": cid, "type": "tool_call"}])

def _scripted():
    return _ScriptedModel([
        _call("lookup_payment", {"ref": "PMT-1003"}, "c1"),
        _call("policy_for", {"reason_code": "LIMIT_BREACH"}, "c2"),
        AIMessage("Escalate to Treasury for approval."),
    ])

check("a tool call produces one ToolMessage",
      lambda: len(native_step(_scripted(), [HumanMessage("q")])[1]) == 1)
check("the ToolMessage carries the call's id",
      lambda: native_step(_scripted(), [HumanMessage("q")])[1][0].tool_call_id == "c1",
      "tool_call_id must be call['id'] -- that is what pairs result to request")
check("the tool actually ran",
      lambda: "LIMIT_BREACH" in native_step(_scripted(), [HumanMessage("q")])[1][0].content)
check("an unknown tool is reported, not raised",
      lambda: "no such tool" in native_step(
          _ScriptedModel([_call("delete_everything", {}, "c1")]), [HumanMessage("q")])[1][0].content,
      "the model can only name tools you gave it, but defend the boundary anyway")
check("there is no parser in this path at all",
      lambda: "parse_step" not in native_step.__code__.co_names,
      "that is the entire point of the section")
'''),

    md("""
## Section 4 &mdash; ...and the version you would actually ship

`create_agent` is `native_loop` with the budget, the dispatch, the error handling and the message
bookkeeping already written. You have now built it twice, so you know exactly what it is doing.
"""),
    code('''
from langchain.agents import create_agent

def built_agent():
    """The same ReAct behaviour, as one call."""
    return create_agent(model=get_llm(), tools=list(TOOLS.values()),
                        system_prompt=REACT_SYSTEM)
'''),

    md("""
## Run it for real

Three paths, one question. The first asks the model to produce the text format and parses it; the
second and third let the serving stack carry the structure.
"""),
    code('''
if llm_ready():
    STRICT = ("You investigate payment exceptions. Reply in EXACTLY this format and nothing "
              "else:\\n"
              "Thought: <your reasoning>\\n"
              "Action: <one of: " + ", ".join(TOOLS) + ">\\n"
              "Action Input: <the argument, a bare value with no quotes or braces>")

    # The same instruction after six months of well-meaning edits. Nobody writes the loose one
    # on purpose; prompts drift towards "be helpful and thorough" one review at a time.
    LOOSE  = ("You investigate payment exceptions. Think step by step, explain your reasoning "
              "clearly for the operations team, and use the Thought / Action / Action Input "
              "format to call one of these tools: " + ", ".join(TOOLS) + ". Be thorough.")

    def _text_path():
        attempts = 4
        for label, instruction in (("strict", STRICT), ("loose", LOOSE)):
            parsed = usable = 0
            first = None
            for i in range(attempts):
                reply = ask("Why is PMT-1003 held, and what must we do about it?",
                            system=instruction)
                step = parse_step(reply)
                ok = bool(step) and step["action"] in TOOLS
                parsed += ok
                good = False
                if ok:
                    arg = list(TOOLS[step["action"]].args)[0]
                    result = TOOLS[step["action"]].invoke({arg: step["input"]})
                    good = "no payment found" not in str(result)
                usable += good
                if i == 0:
                    first = (reply, step, good)
            print(f"=== {label} instruction: {parsed}/{attempts} parsed, "
                  f"{usable}/{attempts} usable ===")
            reply, step, good = first
            print("  Action Input as written:",
                  repr(step["input"]) if step else "(did not parse)")
            print(f"  the tool could use it : {good}\\n")
        return None
    TEXT_RESULT = guard(_text_path)
'''),
    code('''
if llm_ready():
    def _native_path():
        out = native_loop("Why is PMT-1003 held, and what must we do about it?")
        show_messages(out["messages"])
        print(f"\\nsteps: {out['steps']}   answer: {out['answer'][:160]}")
        print("tool calls parsed by hand: 0")
    guard(_native_path)
'''),
    code('''
if llm_ready():
    def _built():
        out = built_agent().invoke(
            {"messages": [("human", "Why is PMT-1005 held, and what must we do about it?")]})
        show_messages(out["messages"])
    guard(_built)
'''),
    md("""
### Read the trace

The result here is probably not the one you expected, and it is better than the one you expected.

**The text path parsed fine.** Given an explicit format instruction, the model very likely obeyed
every time. So the lesson is *not* "models cannot follow a format" &mdash; on a good day they can.

**Look at the second number.** `parsed` counts replies where the regex found a tool name.
`usable` counts replies where the extracted argument actually worked when passed to the tool. If
those two numbers differ, read the `Action Input` in the raw reply: models frequently write

```
Action Input: {"payment_id": "PMT-1003"}
```

instead of `PMT-1003`. The format is perfect. The tool name is right. The regex is delighted. And
the argument is a JSON object with a key your tool has never heard of &mdash; so the lookup returns
nothing, the agent concludes the payment does not exist, and **nothing anywhere reports an
error**. Your parser cannot catch this, because validating the argument was never its job.

That is the real cost of the text format, and it is worse than a parse failure. A parse failure is
loud. This is silent.

**The native path cannot fail that way.** `bind_tools` sends the tool's argument *schema*, and the
serving stack validates against it before your code sees anything. A wrong key is rejected at the
boundary. What is left to go wrong &mdash; wrong tool, wrong value &mdash; are real mistakes you can
see and measure.

**The built agent** is the native path with the budget and the bookkeeping done.

Keep the parser in your notes for the day you meet a model with no tool-calling API. That is the
only situation in which you should write one, and now you know exactly what you would be taking
on: not eight formatting edge cases, but every argument your tools accept, forever.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Make `parse_step` lenient enough to survive the markdown-fence and bold-heading cases. Then
   feed it `"Action: I will call lookup_payment"` and decide whether your leniency has started
   guessing.
2. The text path retries the same prompt six times. Add a **repair** turn instead: when parsing
   fails, send the model its own bad output and ask it to restate. Count the extra calls. That is
   the real cost of the text format.
3. Give `built_agent` a `response_format` (Lab 1.3) so it returns a typed action rather than a
   sentence, and check whether the answer to PMT-1005 still says Compliance.
"""),
]


# =========================================================================== #
# Lab 2.3 -- sub-goals that finish, and knowing when to re-plan
# =========================================================================== #
LAB3 = [
    header(3, "Sub-goals That Finish, and Knowing When to Re-plan", "Advanced", 40,
           ["Have the model return a <code>Plan</code> object with declared dependencies",
            "Reject sub-goals no agent could ever call finished",
            "Tell a transient failure from a wrong plan &mdash; retry one, re-plan the other",
            "Run a plan against a tool that fails on purpose, and watch it recover"],
           "> **Builds on Lab 2.2.** You have tools and a loop. This lab is about deciding what to\n"
           "> do with them before you start, and what to do when the world disagrees."),
    setup(3),
    code(DOMAIN),
    code(CARRY_TOOLS),

    md("""
## Concept

An agent that plans badly fails in one of two ways, and they need opposite responses:

- the **world** misbehaved &mdash; a timeout, a rate limit, a blip. Retry.
- the **plan** was wrong &mdash; the tool does not exist, the argument is invalid, the step depends
  on something that never happened. Retrying is a loop that burns budget and finishes nowhere.

Getting that distinction wrong is one of the commonest production failures in agents, and it is
almost always the same bug: a blanket `except: retry`.
"""),

    md("""
## Section 1 &mdash; A plan the model returns as an object

`with_structured_output(Plan)` sends the schema to the model and validates what comes back, so
you get a `Plan`, not a paragraph about a plan. Fill in the field descriptions &mdash; they are the
only instructions the model gets about what each field means.
"""),
    code('''
from typing import List, Literal
from pydantic import BaseModel, Field

class Step(BaseModel):
    """One step of an investigation plan."""
    name: str = Field(description="Short snake_case name for this step")
    tool: str = Field(description="BLANK")          # TODO: how does the model know what may go here?
    argument: str = Field(description="The single argument to pass to the tool")
    depends_on: List[str] = Field(default_factory=list,
                                  description="Names of steps that must finish before this one")


class Plan(BaseModel):
    """An investigation plan for one payment exception."""
    goal: str = Field(description="The question this plan answers, in one line")
    steps: List[Step] = Field(description="The steps, which may be listed in any order")


def plan_for(goal: str) -> Plan:
    """Ask the model for a Plan object."""
    return get_llm().with_structured_output(Plan).invoke(
        "Produce an investigation plan for this goal. Every step must name a tool from the list "
        "and declare the steps it depends on.\\n"
        f"TOOLS: {list(TOOLS)}\\nGOAL: {goal}")
''', '''
from typing import List, Literal
from pydantic import BaseModel, Field

class Step(BaseModel):
    """One step of an investigation plan."""
    name: str = Field(description="Short snake_case name for this step")
    tool: str = Field(description="The tool this step calls. Must be one of the tool names given "
                                  "in the prompt, or the literal 'none' for a reasoning step.")
    argument: str = Field(description="The single argument to pass to the tool")
    depends_on: List[str] = Field(default_factory=list,
                                  description="Names of steps that must finish before this one")


class Plan(BaseModel):
    """An investigation plan for one payment exception."""
    goal: str = Field(description="The question this plan answers, in one line")
    steps: List[Step] = Field(description="The steps, which may be listed in any order")


def plan_for(goal: str) -> Plan:
    """Ask the model for a Plan object."""
    return get_llm().with_structured_output(Plan).invoke(
        "Produce an investigation plan for this goal. Every step must name a tool from the list "
        "and declare the steps it depends on.\\n"
        f"TOOLS: {list(TOOLS)}\\nGOAL: {goal}")
'''),

    md("""
### A sub-goal an agent cannot finish

"Understand the payment thoroughly" has no completion test. An agent handed it will either stop
arbitrarily or never stop. Reject those before you run them.
"""),
    code('''
VAGUE = ("understand", "thoroughly", "make sure", "as needed", "properly",
         "investigate fully", "look into", "analyse", "review", "consider")

def finishable(step: Step) -> bool:
    """A step is finishable when it names a real tool and does not describe an attitude."""
    if step.tool != "none" and step.tool not in TOOLS:
        return False
    return not any(word in step.name.lower().replace("_", " ") for word in VAGUE)
''', '''
VAGUE = ("understand", "thoroughly", "make sure", "as needed", "properly",
         "investigate fully", "look into", "analyse", "review", "consider")

def finishable(step: Step) -> bool:
    """A step is finishable when it names a real tool and does not describe an attitude."""
    if step.tool != "none" and step.tool not in TOOLS:
        return False
    return not any(word in step.name.lower().replace("_", " ") for word in VAGUE)
'''),
    code('''
# --- Self-check: Section 1   (Pydantic objects only -- no model call)
_ok    = Step(name="read_payment", tool="lookup_payment", argument="PMT-1003")
_vague = Step(name="understand_the_case", tool="none", argument="")
_ghost = Step(name="email_the_desk", tool="send_email", argument="ops@example.com")

def _tool_desc():
    d = Step.model_fields["tool"].description
    if not d or d == "BLANK":
        raise NameError("Step.tool still has no description")
    return d

check("the plan schema declares a goal and steps",
      lambda: set(Plan.model_fields) == {"goal", "steps"})
check("Step.tool tells the model where the valid names come from",
      lambda: len(_tool_desc()) > 40 and "none" in _tool_desc(),
      "the model cannot guess an enum it was never shown -- name the source and the escape hatch")
check("a concrete step is finishable",   lambda: finishable(_ok) is True)
check("a vague step is rejected",        lambda: finishable(_vague) is False,
      '"understand the case" has no completion test, so an agent cannot stop')
check("a step naming a tool that does not exist is rejected",
      lambda: finishable(_ghost) is False,
      "this is a WRONG PLAN, not a transient failure -- Section 2 depends on the difference")
check("a reasoning step with tool='none' is allowed",
      lambda: finishable(Step(name="decide_action", tool="none", argument="")) is True)
'''),

    md("""
## Section 2 &mdash; Transient, or wrong?

Look at what came back and decide which kind of failure it is. Retry the world; re-plan the plan.
"""),
    code('''
TRANSIENT = ("timeout", "timed out", "503", "502", "429", "connection reset",
             "temporarily unavailable", "rate limit", "try again")

PERMANENT = ("no such tool", "no payment found", "invalid", "not permitted",
             "no policy on file", "unknown reason code")

def diagnose(observation: str) -> Literal["transient", "wrong_plan", "ok"]:
    """What kind of thing just happened?"""
    low = (observation or "").lower()
    if any(w in low for w in TRANSIENT):
        return "transient"
    if any(w in low for w in BLANK):  # TODO: retrying will never fix these -- which list is it?
        return "wrong_plan"
    return "ok"


def response_to(kind: str, attempts: int, max_retries: int = 2) -> str:
    """What should the agent do about it?"""
    if kind == "ok":
        return "continue"
    if kind == "transient":
        return "retry" if attempts < max_retries else "give_up"
    return "replan"
''', '''
TRANSIENT = ("timeout", "timed out", "503", "502", "429", "connection reset",
             "temporarily unavailable", "rate limit", "try again")

PERMANENT = ("no such tool", "no payment found", "invalid", "not permitted",
             "no policy on file", "unknown reason code")

def diagnose(observation: str) -> Literal["transient", "wrong_plan", "ok"]:
    """What kind of thing just happened?"""
    low = (observation or "").lower()
    if any(w in low for w in TRANSIENT):
        return "transient"
    if any(w in low for w in PERMANENT):
        return "wrong_plan"
    return "ok"


def response_to(kind: str, attempts: int, max_retries: int = 2) -> str:
    """What should the agent do about it?"""
    if kind == "ok":
        return "continue"
    if kind == "transient":
        return "retry" if attempts < max_retries else "give_up"
    return "replan"
'''),
    code('''
# --- Self-check: Section 2
check("a timeout is transient",        lambda: diagnose("upstream timed out after 30s") == "transient")
check("a 429 is transient",            lambda: diagnose("HTTP 429 rate limit") == "transient")
check("a missing record is a wrong plan",
      lambda: diagnose("no payment found with reference 'PMT-9999'") == "wrong_plan",
      "retrying this forever is the classic burn -- the reference will not appear")
check("an unknown tool is a wrong plan", lambda: diagnose("no such tool 'send_email'") == "wrong_plan")
check("a good observation is ok",
      lambda: diagnose(\'{"ref": "PMT-1003", "status": "held"}\') == "ok")
check("a transient failure is retried, then abandoned",
      lambda: [response_to("transient", i) for i in range(4)]
              == ["retry", "retry", "give_up", "give_up"])
check("a wrong plan is never retried",
      lambda: all(response_to("wrong_plan", i) == "replan" for i in range(5)),
      "retrying a wrong plan is a loop that finishes nowhere")
'''),

    md("""
## Section 3 &mdash; Run the plan, and recover

`execute()` walks the ordered steps, runs each tool, and reacts to what comes back. The flaky
wrapper below fails the first *n* calls with a timeout, so you can watch both paths without
waiting for a real outage.
"""),
    code('''
def order_steps(plan: Plan) -> list[Step]:
    """Dependency order. Raises ValueError on a cycle or a missing dependency."""
    by_name = {s.name: s for s in plan.steps}
    ordered, done = [], set()
    while len(ordered) < len(by_name):
        progressed = False
        for name, step in by_name.items():
            if name in done or not all(d in done for d in step.depends_on):
                continue
            ordered.append(step); done.add(name); progressed = True
        if not progressed:
            raise ValueError("cycle or missing dependency in plan")
    return ordered


def flaky(fail_times: int):
    """A tool runner that fails with a timeout the first `fail_times` calls."""
    state = {"n": 0}
    def run(step: Step) -> str:
        state["n"] += 1
        if state["n"] <= fail_times:
            return "upstream timed out after 30s"
        tool_obj = TOOLS.get(step.tool)
        if tool_obj is None:
            return f"no such tool {step.tool!r}"
        arg = list(tool_obj.args)[0]
        return str(tool_obj.invoke({arg: step.argument}))
    return run


def execute(plan: Plan, run, max_retries: int = 2) -> dict:
    """Run the plan. Retry transient failures; stop and report a wrong plan."""
    trace, attempts = [], 0
    for step in order_steps(plan):
        attempts = 0
        while True:
            observation = run(step)
            kind = diagnose(observation)
            action = response_to(kind, attempts, max_retries)
            trace.append((step.name, kind, action))
            if action == "continue":
                break
            if action == "retry":
                attempts += 1
                continue
            return {"trace": trace, "outcome": BLANK, "failed_at": step.name}  # TODO: what ended it?
    return {"trace": trace, "outcome": "completed", "failed_at": None}
''', '''
def order_steps(plan: Plan) -> list[Step]:
    """Dependency order. Raises ValueError on a cycle or a missing dependency."""
    by_name = {s.name: s for s in plan.steps}
    ordered, done = [], set()
    while len(ordered) < len(by_name):
        progressed = False
        for name, step in by_name.items():
            if name in done or not all(d in done for d in step.depends_on):
                continue
            ordered.append(step); done.add(name); progressed = True
        if not progressed:
            raise ValueError("cycle or missing dependency in plan")
    return ordered


def flaky(fail_times: int):
    """A tool runner that fails with a timeout the first `fail_times` calls."""
    state = {"n": 0}
    def run(step: Step) -> str:
        state["n"] += 1
        if state["n"] <= fail_times:
            return "upstream timed out after 30s"
        tool_obj = TOOLS.get(step.tool)
        if tool_obj is None:
            return f"no such tool {step.tool!r}"
        arg = list(tool_obj.args)[0]
        return str(tool_obj.invoke({arg: step.argument}))
    return run


def execute(plan: Plan, run, max_retries: int = 2) -> dict:
    """Run the plan. Retry transient failures; stop and report a wrong plan."""
    trace, attempts = [], 0
    for step in order_steps(plan):
        attempts = 0
        while True:
            observation = run(step)
            kind = diagnose(observation)
            action = response_to(kind, attempts, max_retries)
            trace.append((step.name, kind, action))
            if action == "continue":
                break
            if action == "retry":
                attempts += 1
                continue
            return {"trace": trace, "outcome": action, "failed_at": step.name}
    return {"trace": trace, "outcome": "completed", "failed_at": None}
'''),
    code('''
# --- Self-check: Section 3   (a hand-written plan and a fake runner -- no model call)
HAND_PLAN = Plan(goal="Decide what to do about PMT-1003", steps=[
    Step(name="read_policy",  tool="policy_for",     argument="LIMIT_BREACH",
         depends_on=["read_payment"]),
    Step(name="read_payment", tool="lookup_payment", argument="PMT-1003"),
])
BAD_PLAN = Plan(goal="g", steps=[Step(name="email_desk", tool="send_email", argument="x")])

check("dependencies are ordered first",
      lambda: [s.name for s in order_steps(HAND_PLAN)] == ["read_payment", "read_policy"])
check("a clean run completes",
      lambda: execute(HAND_PLAN, flaky(0))["outcome"] == "completed")
check("one timeout is retried and then succeeds",
      lambda: execute(HAND_PLAN, flaky(1))["outcome"] == "completed")
check("the retry is visible in the trace",
      lambda: any(a == "retry" for _, _, a in execute(HAND_PLAN, flaky(1))["trace"]))
check("endless timeouts are eventually abandoned",
      lambda: execute(HAND_PLAN, flaky(99))["outcome"] == "give_up",
      "max_retries has to be a number, not a hope")
check("a wrong plan triggers a replan, not a retry",
      lambda: execute(BAD_PLAN, flaky(0))["outcome"] == "replan")
check("a wrong plan is caught on the FIRST attempt",
      lambda: len(execute(BAD_PLAN, flaky(0))["trace"]) == 1,
      "no point calling a tool that does not exist twice")
'''),

    md("""
## Run it for real

The model plans; your code decides whether the plan is runnable; then it runs against a tool that
fails twice before it works.
"""),
    code('''
if llm_ready():
    def _plan_and_run():
        plan = plan_for("Find out why PMT-1003 is held and what policy requires.")
        print("goal:", plan.goal)
        for s in plan.steps:
            flag = "ok  " if finishable(s) else "DROP"
            print(f"  [{flag}] {s.name:24} tool={s.tool:16} arg={s.argument:16} after={s.depends_on}")

        runnable = Plan(goal=plan.goal, steps=[s for s in plan.steps if finishable(s)])
        if not runnable.steps:
            print("\\nnothing runnable in that plan -- which is itself a result")
            return None

        print("\\n--- executing against a tool that times out twice ---")
        result = execute(runnable, flaky(2))
        for name, kind, action in result["trace"]:
            print(f"  {name:24} {kind:12} -> {action}")
        print(f"\\noutcome: {result['outcome']}")
        return result
    RESULT = guard(_plan_and_run)
'''),
    md("""
### Read it

Look at the plan the model produced *before* anything ran. Some of the time it will invent a step
naming a tool that does not exist &mdash; `check_sanctions`, `notify_desk`, something plausible. That
is not the model being bad; it is the model doing what planners do. `finishable()` catching it
before execution is the entire value of validating a plan.

Then look at the trace. Two timeouts, two retries, then progress &mdash; and the retries cost you
nothing but time. Had `diagnose` mislabelled the missing-tool case as transient, you would have
watched it retry a step that can never succeed until the budget ran out. That single misjudgement
is the bug this lab exists to prevent, and in real code it always looks like:

```python
except Exception:
    retry()          # what kind of exception? nobody asked
```
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `execute` gives up on a wrong plan. Make it actually **re-plan**: feed the failed step and its
   observation back to `plan_for` and run the new plan. Cap the number of re-plans, and say why
   your cap is the right one.
2. Add a third diagnosis, `needs_human`, for observations that name a sanctions hold. What should
   `response_to` return for it, and why is that different from `give_up`?
3. `finishable()` uses a word list, which is crude. Replace it with a small
   `with_structured_output` call that asks the model whether a step has a completion test. Run
   both over ten invented steps and see which you trust more.
"""),
]


# =========================================================================== #
# Lab 2.4 -- branch, score, prune; and knowing when to stop reflecting
# =========================================================================== #
LAB4 = [
    header(4, "Branch, Score, Prune: and When to Stop Reflecting", "Advanced", 40,
           ["Generate several candidate actions concurrently with <code>.batch()</code>",
            "Score them with a structured judge instead of reading them yourself",
            "Prune, and look at what you threw away",
            "Build a reflection loop that stops when it stops improving"],
           "> **Builds on Lab 2.1.** Same chain machinery, used two ways that both trade extra\n"
           "> calls for a better answer &mdash; when they work."),
    setup(4),
    code(DOMAIN),
    code(CARRY_TOOLS),

    md("""
## Concept

**Tree-of-Thought** explores several routes, scores them, and keeps the promising ones.
**Reflection** drafts, criticises and revises. Both buy quality with extra calls, and both have a
point past which the extra calls buy nothing.

The framework parts that matter here:

- `chain.batch([...])` runs the branches **concurrently**, so width costs latency once, not
  *n* times;
- `with_structured_output(Score)` makes the judge return numbers you can sort, instead of a
  paragraph you have to read.

A scorer that returns prose is not a scorer. It is a second opinion you still have to interpret.
"""),

    md("""
## Section 1 &mdash; Generate the branches, concurrently

One prompt, *n* different framings, one batched call.
"""),
    code('''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

ANGLES = [
    "Answer strictly from the policy text, quoting its operative words.",
    "Answer as the operations desk: what do we physically do next, and who do we tell?",
    "Answer as the control function: what must NOT happen, and who owns the decision?",
]

def branch_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a payments operations analyst. {angle} Answer in two sentences."),
        ("human", "PAYMENT: {payment}\\nPOLICY: {policy}\\n\\nWhat must happen next?"),
    ])
    return prompt | get_llm() | StrOutputParser()


def branches(ref: str) -> list[str]:
    """One candidate answer per angle, generated concurrently."""
    rec = LEDGER[ref]
    base = {"payment": json.dumps({"ref": ref, **rec}),
            "policy": POLICY.get(rec["reason_code"], "no policy applies")}
    return branch_chain().batch([{**base, "angle": a} for a in ANGLES])   # one call, three branches
''', '''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

ANGLES = [
    "Answer strictly from the policy text, quoting its operative words.",
    "Answer as the operations desk: what do we physically do next, and who do we tell?",
    "Answer as the control function: what must NOT happen, and who owns the decision?",
]

def branch_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a payments operations analyst. {angle} Answer in two sentences."),
        ("human", "PAYMENT: {payment}\\nPOLICY: {policy}\\n\\nWhat must happen next?"),
    ])
    return prompt | get_llm() | StrOutputParser()


def branches(ref: str) -> list[str]:
    """One candidate answer per angle, generated concurrently."""
    rec = LEDGER[ref]
    base = {"payment": json.dumps({"ref": ref, **rec}),
            "policy": POLICY.get(rec["reason_code"], "no policy applies")}
    return branch_chain().batch([{**base, "angle": a} for a in ANGLES])   # one call, three branches
'''),
    code('''
# --- Self-check: Section 1   (input construction only -- no model call)
def _inputs():
    rec = LEDGER["PMT-1003"]
    base = {"payment": json.dumps({"ref": "PMT-1003", **rec}),
            "policy": POLICY[rec["reason_code"]]}
    return [{**base, "angle": a} for a in ANGLES]

check("one input per angle",         lambda: len(_inputs()) == len(ANGLES))
check("each input carries all three variables",
      lambda: all(set(d) == {"payment", "policy", "angle"} for d in _inputs()))
check("the angles really differ",    lambda: len({d["angle"] for d in _inputs()}) == len(ANGLES),
      "three copies of one prompt is not a tree")
check("the case data is shared across branches",
      lambda: len({d["payment"] for d in _inputs()}) == 1,
      "branches must differ in approach only, or you are comparing different questions")
'''),

    md("""
## Section 2 &mdash; The scorer is the design

Whatever the judge rewards is what the tree will select for. Declare it as a schema so the
scores come back sortable.
"""),
    code('''
class Score(BaseModel):
    """A judge's verdict on one candidate answer."""
    grounded: int = Field(description="BLANK", ge=0, le=5)  # TODO: what is this scoring, 0 to 5?
    actionable: int = Field(description="0-5: is there a concrete next action a person could take?",
                            ge=0, le=5)
    safe: int = Field(description="0-5: does it respect who is allowed to decide? 0 if it "
                                  "proposes acting where policy reserves the decision for a human",
                      ge=0, le=5)
    why: str = Field(description="One short sentence justifying the lowest of the three scores")

    @property
    def total(self) -> int:
        return self.grounded + self.actionable + self.safe


def judge(ref: str, candidate: str) -> Score:
    rec = LEDGER[ref]
    return get_llm().with_structured_output(Score).invoke(
        "Score this proposed action against the policy. Be strict.\\n"
        f"PAYMENT: {json.dumps({'ref': ref, **rec})}\\n"
        f"POLICY: {POLICY.get(rec['reason_code'], 'no policy applies')}\\n"
        f"CANDIDATE: {candidate}")
''', '''
class Score(BaseModel):
    """A judge's verdict on one candidate answer."""
    grounded: int = Field(description="0-5: is every claim supported by the payment record or "
                                      "the policy text, with nothing invented?", ge=0, le=5)
    actionable: int = Field(description="0-5: is there a concrete next action a person could take?",
                            ge=0, le=5)
    safe: int = Field(description="0-5: does it respect who is allowed to decide? 0 if it "
                                  "proposes acting where policy reserves the decision for a human",
                      ge=0, le=5)
    why: str = Field(description="One short sentence justifying the lowest of the three scores")

    @property
    def total(self) -> int:
        return self.grounded + self.actionable + self.safe


def judge(ref: str, candidate: str) -> Score:
    rec = LEDGER[ref]
    return get_llm().with_structured_output(Score).invoke(
        "Score this proposed action against the policy. Be strict.\\n"
        f"PAYMENT: {json.dumps({'ref': ref, **rec})}\\n"
        f"POLICY: {POLICY.get(rec['reason_code'], 'no policy applies')}\\n"
        f"CANDIDATE: {candidate}")
'''),
    code('''
# --- Self-check: Section 2   (schema + arithmetic -- no model call)
def _grounded_desc():
    d = Score.model_fields["grounded"].description
    if not d or d == "BLANK":
        raise NameError("Score.grounded still has no description")
    return d

check("the judge returns three numbers and a reason",
      lambda: set(Score.model_fields) == {"grounded", "actionable", "safe", "why"})
check("grounded says what it measures",
      lambda: len(_grounded_desc()) > 40)
check("the range is declared to the model",
      lambda: "0" in _grounded_desc() and "5" in _grounded_desc(),
      "a scale the model has to guess is a scale you cannot compare across runs")
def _rejects_out_of_range():
    try:
        Score(grounded=9, actionable=1, safe=1, why="x")
        return False
    except Exception:
        return True

check("out-of-range scores are rejected by the schema",
      lambda: _rejects_out_of_range())
check("total adds the three",
      lambda: Score(grounded=5, actionable=4, safe=3, why="x").total == 12)
'''),

    md("""
## Section 3 &mdash; Prune, and look at what you lost

Keeping the top *k* is easy. Looking at what you dropped is the part people skip, and it is where
you find out your scorer is rewarding the wrong thing.
"""),
    code('''
def prune(candidates: list[str], scored: list[Score], keep: int = 1):
    """Return (kept, dropped) as lists of (candidate, score), best first."""
    pairs = sorted(zip(candidates, scored), key=lambda p: BLANK, reverse=True)  # TODO: by what?
    return pairs[:keep], pairs[keep:]
''', '''
def prune(candidates: list[str], scored: list[Score], keep: int = 1):
    """Return (kept, dropped) as lists of (candidate, score), best first."""
    pairs = sorted(zip(candidates, scored), key=lambda p: p[1].total, reverse=True)
    return pairs[:keep], pairs[keep:]
'''),
    code('''
# --- Self-check: Section 3
_cands = ["a", "b", "c"]
_scored = [Score(grounded=2, actionable=2, safe=2, why="x"),     # 6
           Score(grounded=5, actionable=5, safe=5, why="x"),     # 15
           Score(grounded=4, actionable=4, safe=4, why="x")]     # 12

check("the best candidate is kept",     lambda: prune(_cands, _scored)[0][0][0] == "b")
check("the rest are returned, not lost", lambda: len(prune(_cands, _scored)[1]) == 2)
check("the dropped list is also ordered",
      lambda: [c for c, _ in prune(_cands, _scored)[1]] == ["c", "a"],
      "you cannot review what you pruned if it comes back shuffled")
check("keep=2 keeps two",               lambda: len(prune(_cands, _scored, keep=2)[0]) == 2)
'''),

    md("""
## Section 4 &mdash; Reflection, and the knee

Reflection improves a draft by criticising it. The gain per round drops fast &mdash; usually a real
improvement on round one, a small one on round two, and noise after that. Stop when the critic
stops finding anything.
"""),
    code('''
CRITIC = ("You are a strict reviewer. List the specific faults in this proposed action against "
          "the policy: invented facts, missing next step, or acting where a human must decide. "
          "If there is nothing material to fix, reply with exactly: NO ISSUES")

def critic_is_done(critique: str) -> bool:
    """Has reflecting stopped paying? True when the critic found nothing material."""
    return BLANK                      # TODO: how does the critic say "nothing to fix"?


def reflect(ref: str, draft: str, rounds: int = 3) -> dict:
    """Draft -> critique -> revise, stopping when the critic has nothing left."""
    rec = LEDGER[ref]
    context = (f"PAYMENT: {json.dumps({'ref': ref, **rec})}\\n"
               f"POLICY: {POLICY.get(rec['reason_code'], 'no policy applies')}")
    history = [("draft", draft)]
    for i in range(rounds):
        critique = ask(f"{context}\\n\\nPROPOSED ACTION: {draft}", system=CRITIC)
        history.append((f"critique {i+1}", critique))
        if critic_is_done(critique):
            history.append(("stopped", f"critic found nothing on round {i+1}"))
            break
        draft = ask(f"{context}\\n\\nPROPOSED ACTION: {draft}\\n\\nFAULTS FOUND: {critique}\\n\\n"
                    "Rewrite the action in two sentences, fixing only what was faulted.")
        history.append((f"revision {i+1}", draft))
    return {"final": draft, "history": history}
''', '''
CRITIC = ("You are a strict reviewer. List the specific faults in this proposed action against "
          "the policy: invented facts, missing next step, or acting where a human must decide. "
          "If there is nothing material to fix, reply with exactly: NO ISSUES")

def critic_is_done(critique: str) -> bool:
    """Has reflecting stopped paying? True when the critic found nothing material."""
    return "NO ISSUES" in (critique or "").upper()


def reflect(ref: str, draft: str, rounds: int = 3) -> dict:
    """Draft -> critique -> revise, stopping when the critic has nothing left."""
    rec = LEDGER[ref]
    context = (f"PAYMENT: {json.dumps({'ref': ref, **rec})}\\n"
               f"POLICY: {POLICY.get(rec['reason_code'], 'no policy applies')}")
    history = [("draft", draft)]
    for i in range(rounds):
        critique = ask(f"{context}\\n\\nPROPOSED ACTION: {draft}", system=CRITIC)
        history.append((f"critique {i+1}", critique))
        if critic_is_done(critique):
            history.append(("stopped", f"critic found nothing on round {i+1}"))
            break
        draft = ask(f"{context}\\n\\nPROPOSED ACTION: {draft}\\n\\nFAULTS FOUND: {critique}\\n\\n"
                    "Rewrite the action in two sentences, fixing only what was faulted.")
        history.append((f"revision {i+1}", draft))
    return {"final": draft, "history": history}
'''),
    code('''
# --- Self-check: Section 4   (the stop rule, on canned critiques -- no model call)
check("the critic is told how to say 'nothing to fix'",
      lambda: "NO ISSUES" in CRITIC,
      "a critic with no way to pass will always invent a fault, and you will loop forever")
check("an all-clear critique stops the loop",
      lambda: critic_is_done("NO ISSUES") is True)
check("a critique with faults does not stop it",
      lambda: critic_is_done("It proposes releasing a sanctions hold.") is False)
check("the check is case-insensitive",
      lambda: critic_is_done("no issues") is True,
      "models do not honour your capitalisation")
check("an empty critique does not stop it",
      lambda: critic_is_done("") is False,
      "a model that returned nothing has not told you the draft is good")
'''),

    md("""
## Run it for real &mdash; the tree
"""),
    code('''
if llm_ready():
    def _tree():
        ref = "PMT-1005"          # a sanctions hold: the safe score is the one that separates them
        cands = branches(ref)
        print(f"{len(cands)} branches generated concurrently\\n")
        scored = [judge(ref, c) for c in cands]
        kept, dropped = prune(cands, scored, keep=1)

        for label, group in (("KEPT", kept), ("dropped", dropped)):
            for cand, sc in group:
                print(f"[{label:7}] total={sc.total:2}  g={sc.grounded} a={sc.actionable} s={sc.safe}")
                print(f"          {cand.strip()[:200]}")
                print(f"          why: {sc.why[:160]}\\n")
        return kept, dropped
    TREE = guard(_tree)
'''),
    md("""
## Run it for real &mdash; the reflection loop
"""),
    code('''
if llm_ready():
    def _reflect():
        weak = "Release the payment once the counterparty confirms by email."
        out = reflect("PMT-1005", weak)
        for label, text in out["history"]:
            print(f"--- {label} ---")
            print(text.strip()[:400] + "\\n")
        print("=== final ===")
        print(out["final"].strip()[:400])
    guard(_reflect)
'''),
    md("""
### Read it

**The tree.** Read the `why` on the branch that lost. PMT-1005 is a sanctions hold, so any answer
proposing to release or cancel it should score 0 on `safe` no matter how fluent it is &mdash; and the
operations-desk angle is the one most likely to write exactly that. If the judge scored it well
anyway, your scorer is the problem, not the branch. Fix the scorer before you widen the tree.

**The reflection loop.** The deliberately weak draft proposes releasing a sanctions hold on an
email confirmation, which is precisely what the policy forbids. Watch round one demolish it and
round two do much less. That shape &mdash; a large first gain, then a knee &mdash; is what you should
expect, and it is why `rounds=3` with an early stop beats `rounds=10`.

And note the failure mode built into the stop rule: a critic with no way to say "nothing to fix"
will always find something, because that is what you asked it for. `NO ISSUES` is not a nicety.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add a fourth angle that is deliberately reckless &mdash; "answer as the client relationship
   manager who wants this paid today" &mdash; and confirm the judge scores it 0 on `safe`. If it does
   not, tighten the field description until it does.
2. Score all three branches in **one** batched call instead of three serial ones. You will need
   `with_structured_output(Score).batch(...)`. Measure the wall-clock difference.
3. Run `reflect` on an answer that is already correct. How many rounds before the critic starts
   inventing faults? That number is your real ceiling on reflection.
"""),
]


# =========================================================================== #
# Lab 2.5 -- challenge: the architecture bake-off
# =========================================================================== #
LAB5 = [
    header(5, "Challenge &mdash; The Architecture Bake-Off", "Advanced", 45,
           ["Put all four Module 2 architectures behind one interface",
            "Write the acceptance bar before you look at a single result",
            "Run one eval set through all four and pick a winner on evidence",
            "Find the case that no architecture gets right, and say why"],
           "> **The take-home artifact.** A harness you can point at your own task, and a habit:\n"
           "> choose the reasoning architecture with a number, not a preference."),
    setup(5),
    code(DOMAIN),
    code(CARRY_TOOLS),

    md("""
## Concept

You have built four ways to answer the same question:

| Arm | What it is | What it costs |
|---|---|---|
| **direct** | one chain, no reasoning asked for | one call |
| **cot** | one chain, working shown | one call, longer |
| **react** | `create_agent` with tools | several calls |
| **reflect** | draft, critique, revise | two to six calls |

The right answer is task-dependent and it is often **direct**. This lab is the harness that tells
you which, and the discipline of writing the bar down first so the harness can overrule you.
"""),

    md("""
## Section 1 &mdash; One interface, four arms

Every arm is a function `(ref) -> str`. That is the whole contract, and it is what makes them
comparable.
"""),
    code('''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent

SYSTEM = "You are a payments operations analyst."
QUESTION = "What must happen next with {ref}? Answer with the single next action."

def _context(ref: str) -> str:
    rec = LEDGER[ref]
    return (f"PAYMENT: {json.dumps({'ref': ref, **rec})}\\n"
            f"POLICY: {POLICY.get(rec['reason_code'], 'no policy applies')}")

def arm_direct(ref: str) -> str:
    chain = ChatPromptTemplate.from_messages(
        [("system", SYSTEM), ("human", "{ctx}\\n\\n" + QUESTION.format(ref=ref))]
    ) | get_llm() | StrOutputParser()
    return chain.invoke({"ctx": _context(ref)})

def arm_cot(ref: str) -> str:
    chain = ChatPromptTemplate.from_messages(
        [("system", SYSTEM + " Work through the case step by step, then give the action."),
         ("human", "{ctx}\\n\\n" + QUESTION.format(ref=ref))]
    ) | get_llm() | StrOutputParser()
    return chain.invoke({"ctx": _context(ref)})

def arm_react(ref: str) -> str:
    agent = create_agent(model=get_llm(), tools=list(TOOLS.values()),
                         system_prompt=SYSTEM + " Use the tools to find the reason code and its "
                                                "policy before answering.")
    out = agent.invoke({"messages": [("human", QUESTION.format(ref=ref))]})
    return out["messages"][-1].content

def arm_reflect(ref: str) -> str:
    draft = arm_direct(ref)
    critique = ask(f"{_context(ref)}\\n\\nPROPOSED: {draft}",
                   system="List material faults against the policy, or reply exactly: NO ISSUES")
    if BLANK:                         # TODO: when is the draft already good enough to return?
        return draft
    return ask(f"{_context(ref)}\\n\\nPROPOSED: {draft}\\nFAULTS: {critique}\\n\\n"
               "Rewrite the action in one sentence, fixing only what was faulted.")

ARMS = {"direct": arm_direct, "cot": arm_cot, "react": arm_react, "reflect": arm_reflect}
''', '''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent

SYSTEM = "You are a payments operations analyst."
QUESTION = "What must happen next with {ref}? Answer with the single next action."

def _context(ref: str) -> str:
    rec = LEDGER[ref]
    return (f"PAYMENT: {json.dumps({'ref': ref, **rec})}\\n"
            f"POLICY: {POLICY.get(rec['reason_code'], 'no policy applies')}")

def arm_direct(ref: str) -> str:
    chain = ChatPromptTemplate.from_messages(
        [("system", SYSTEM), ("human", "{ctx}\\n\\n" + QUESTION.format(ref=ref))]
    ) | get_llm() | StrOutputParser()
    return chain.invoke({"ctx": _context(ref)})

def arm_cot(ref: str) -> str:
    chain = ChatPromptTemplate.from_messages(
        [("system", SYSTEM + " Work through the case step by step, then give the action."),
         ("human", "{ctx}\\n\\n" + QUESTION.format(ref=ref))]
    ) | get_llm() | StrOutputParser()
    return chain.invoke({"ctx": _context(ref)})

def arm_react(ref: str) -> str:
    agent = create_agent(model=get_llm(), tools=list(TOOLS.values()),
                         system_prompt=SYSTEM + " Use the tools to find the reason code and its "
                                                "policy before answering.")
    out = agent.invoke({"messages": [("human", QUESTION.format(ref=ref))]})
    return out["messages"][-1].content

def arm_reflect(ref: str) -> str:
    draft = arm_direct(ref)
    critique = ask(f"{_context(ref)}\\n\\nPROPOSED: {draft}",
                   system="List material faults against the policy, or reply exactly: NO ISSUES")
    if "NO ISSUES" in critique.upper():
        return draft
    return ask(f"{_context(ref)}\\n\\nPROPOSED: {draft}\\nFAULTS: {critique}\\n\\n"
               "Rewrite the action in one sentence, fixing only what was faulted.")

ARMS = {"direct": arm_direct, "cot": arm_cot, "react": arm_react, "reflect": arm_reflect}
'''),
    code('''
# --- Self-check: Section 1   (structure only -- no model call)
check("all four arms are registered",
      lambda: set(ARMS) == {"direct", "cot", "react", "reflect"})
check("every arm is callable",
      lambda: all(callable(f) for f in ARMS.values()),
      "the whole comparison rests on the four having the same interface")
check("every arm takes exactly one argument",
      lambda: all(f.__code__.co_argcount == 1 for f in ARMS.values()))
check("the prompt-only arms share one context builder",
      lambda: all("_context" in ARMS[n].__code__.co_names for n in ("direct", "cot")),
      "if the arms build their input differently you are comparing prompts, not architectures")
check("the react arm gets its context from the tools instead",
      lambda: "create_agent" in ARMS["react"].__code__.co_names,
      "that IS the architectural difference -- it fetches rather than being handed the answer")
'''),

    md("""
## Section 2 &mdash; The bar, written first

Fill this in before you run anything. If you write it afterwards you will write down whatever
you got.
"""),
    code('''
BAR = {
    "min_pass_rate":   0.80,          # a candidate must answer four of five
    "must_never_fail": ["PMT-1005"],  # the sanctions hold: getting this wrong is disqualifying
    "max_seconds_case": 20.0,
}

CASES = [
    {"ref": "PMT-1003", "must_contain": ["treasury"]},
    {"ref": "PMT-1005", "must_contain": ["compliance"]},
    {"ref": "PMT-1002", "must_contain": ["retry"]},
    {"ref": "PMT-1004", "must_contain": ["originator", "r04"]},
    {"ref": "PMT-1001", "must_contain": ["settled", "no action", "none"]},
]

def passes(case: dict, answer: str) -> bool:
    low = (answer or "").lower()
    return any(term in low for term in case["must_contain"])


def accepts(result: dict) -> tuple[bool, str]:
    """result: {"rate", "failed_refs", "seconds_case"}. Return (ok, first failing reason)."""
    if result["rate"] < BAR["min_pass_rate"]:
        return False, f"pass rate {result['rate']:.0%} below {BAR['min_pass_rate']:.0%}"
    banned = set(result["failed_refs"]) & set(BAR["must_never_fail"])
    if banned:                        # it cleared the 80% bar, but it got the sanctions hold wrong
        # TODO: does an arm that fails a must-never-fail case still pass? decide, then defend it
        return BLANK, f"failed a disqualifying case: {sorted(banned)}"
    if result["seconds_case"] > BAR["max_seconds_case"]:
        return False, f"{result['seconds_case']:.1f}s/case over {BAR['max_seconds_case']}"
    return True, "accepted"
''', '''
BAR = {
    "min_pass_rate":   0.80,          # a candidate must answer four of five
    "must_never_fail": ["PMT-1005"],  # the sanctions hold: getting this wrong is disqualifying
    "max_seconds_case": 20.0,
}

CASES = [
    {"ref": "PMT-1003", "must_contain": ["treasury"]},
    {"ref": "PMT-1005", "must_contain": ["compliance"]},
    {"ref": "PMT-1002", "must_contain": ["retry"]},
    {"ref": "PMT-1004", "must_contain": ["originator", "r04"]},
    {"ref": "PMT-1001", "must_contain": ["settled", "no action", "none"]},
]

def passes(case: dict, answer: str) -> bool:
    low = (answer or "").lower()
    return any(term in low for term in case["must_contain"])


def accepts(result: dict) -> tuple[bool, str]:
    """result: {"rate", "failed_refs", "seconds_case"}. Return (ok, first failing reason)."""
    if result["rate"] < BAR["min_pass_rate"]:
        return False, f"pass rate {result['rate']:.0%} below {BAR['min_pass_rate']:.0%}"
    banned = set(result["failed_refs"]) & set(BAR["must_never_fail"])
    if banned:
        return False, f"failed a disqualifying case: {sorted(banned)}"
    if result["seconds_case"] > BAR["max_seconds_case"]:
        return False, f"{result['seconds_case']:.1f}s/case over {BAR['max_seconds_case']}"
    return True, "accepted"
'''),
    code('''
# --- Self-check: Section 2
_good     = {"rate": 1.0, "failed_refs": [],            "seconds_case": 3.0}
_low      = {"rate": 0.6, "failed_refs": ["PMT-1002"],  "seconds_case": 3.0}
_unsafe   = {"rate": 0.8, "failed_refs": ["PMT-1005"],  "seconds_case": 3.0}
_slow     = {"rate": 1.0, "failed_refs": [],            "seconds_case": 99.0}

check("a clean result is accepted",   lambda: accepts(_good)[0] is True)
check("a low pass rate is rejected",  lambda: accepts(_low)[0] is False)
check("failing the sanctions case is disqualifying even at 80%",
      lambda: accepts(_unsafe)[0] is False,
      "some cases are not worth 20% -- they are worth the whole decision")
check("the rejection names the case",
      lambda: "PMT-1005" in accepts(_unsafe)[1])
check("a slow arm is rejected",       lambda: accepts(_slow)[0] is False)
check("the scorer accepts a right answer",
      lambda: passes(CASES[1], "Hold; Compliance decides.") is True)
check("the scorer rejects a wrong one",
      lambda: passes(CASES[1], "Release once Treasury approves.") is False)
'''),

    md("""
## Section 3 &mdash; The bake-off
"""),
    code('''
def run_arm(name: str) -> dict:
    """Run every case through one arm and score it."""
    fn = ARMS[name]
    t0, answers, results, failed = time.time(), [], [], []
    for case in CASES:
        try:
            answer = fn(case["ref"])
        except Exception as exc:
            answer = f"<error: {type(exc).__name__}: {exc}>"
        ok = passes(case, answer)
        answers.append(answer); results.append(ok)
        if not ok:
            failed.append(case["ref"])
    seconds = time.time() - t0
    return {"name": name, "answers": answers, "results": results, "failed_refs": failed,
            "rate": sum(results) / len(CASES), "seconds_case": seconds / len(CASES)}
'''),
    code('''
# --- Self-check: Section 3   (shape of the result, on a stub arm -- no model call)
ARMS["_stub"] = lambda ref: {"PMT-1005": "Hold; Compliance decides."}.get(ref, "do something")
_stub = run_arm("_stub")
del ARMS["_stub"]

check("one answer per case",       lambda: len(_stub["answers"]) == len(CASES))
check("the rate is a fraction",    lambda: 0.0 <= _stub["rate"] <= 1.0)
check("failed refs are recorded",  lambda: "PMT-1003" in _stub["failed_refs"])
check("a passing case is not listed as failed",
      lambda: "PMT-1005" not in _stub["failed_refs"])
def _raises_is_scored():
    def boom(ref): raise RuntimeError("nope")
    ARMS["_boom"] = boom
    try:
        return run_arm("_boom")["rate"] == 0.0
    finally:
        del ARMS["_boom"]

check("an arm that raises is scored, not crashed",
      lambda: _raises_is_scored(),
      "one broken arm must not take the whole bake-off down with it")
'''),

    md("""
## Run it for real

Four arms, five cases. `react` and `reflect` make several calls per case, so allow a minute or two.
"""),
    code('''
if llm_ready():
    def _bakeoff():
        results = {name: run_arm(name) for name in ("direct", "cot", "react", "reflect")}

        print("  arm       " + "  ".join(f"{c['ref']:9}" for c in CASES) + "   rate    s/case  verdict")
        print("  " + "-" * 96)
        for name, r in results.items():
            cells = "  ".join(("pass     " if ok else "FAIL     ") for ok in r["results"])
            ok, why = accepts(r)
            print(f"  {name:9} {cells}  {r['rate']:5.0%}  {r['seconds_case']:6.1f}  "
                  f"{'ACCEPT' if ok else 'reject'}")
            if not ok:
                print(f"             -> {why}")

        accepted = [(n, r) for n, r in results.items() if accepts(r)[0]]
        if accepted:
            # Cheapest that cleared the bar -- but a tie on cost is broken by pass rate.
            # Paying nothing extra for a better answer is not a reason to refuse it.
            winner = min(accepted, key=lambda p: (round(p[1]["seconds_case"], 1), -p[1]["rate"]))
            print(f"\\nwinner: {winner[0]} -- the CHEAPEST arm that cleared the bar, "
                  f"not the best-scoring one (cost ties broken on pass rate)")
        else:
            print("\\nno arm cleared the bar. That is a result: the task needs better tools or "
                  "a better prompt, not a fancier architecture.")

        hard = [c["ref"] for i, c in enumerate(CASES)
                if not any(r["results"][i] for r in results.values())]
        print(f"cases no arm answered: {hard or 'none'}")
        return results
    BAKEOFF = guard(_bakeoff)
'''),
    md("""
### Read it

**The winner is the cheapest arm that cleared the bar**, not the highest-scoring one. That line in
the code is the whole lab. Once an arm meets the requirement, further quality is something you are
paying for and not using &mdash; and `direct` clearing the bar is the most common outcome on tasks
where the context already contains the answer.

Note the tie-break, though: when two arms cost the *same*, the higher pass rate wins. "Prefer the
cheaper design" is an argument about what you are willing to pay for, not a reason to accept a
worse answer that costs nothing extra. Those are different claims and it is worth being able to
tell them apart in a design review.

Then look at the last line. A case that **no** arm answers is telling you something no
architecture can fix: the information is missing, the scorer is wrong, or the question is
ambiguous. Reaching for a bigger architecture at that point is the mistake this module exists to
prevent.

**What you take from Module 2:** chains you compose rather than hand-roll; a parser you are now
entitled never to write again; plans that are validated before they run and failures that are
diagnosed before they are retried; a judge that returns numbers; and a harness that picks the
architecture for you. Module 3 gives all of this somewhere durable to live.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add a `react_reflect` arm &mdash; the agent's answer, then one critique round. Does it clear the
   bar, and does it beat `react` by enough to justify doubling the calls?
2. Move `PMT-1001` (already settled) to the front of `CASES` and re-run. If any arm's score
   changes, you have found order dependence, which means your harness is measuring the wrong
   thing. Fix it.
3. Replace the substring scorer with a `with_structured_output` judge from Lab 2.4 and re-run the
   bake-off. Which arms change rank? A scorer swap that reorders the results tells you the
   original ranking was never about the architectures.
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-2-01-chain-of-thought-measured",      LAB1),
    ("lab-2-02-react-parser-contract",          LAB2),
    ("lab-2-03-subgoals-and-replanning",        LAB3),
    ("lab-2-04-tree-of-thought-reflection",     LAB4),
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
