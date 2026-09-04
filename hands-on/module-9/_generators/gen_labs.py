#!/usr/bin/env python3
"""
Generate Module 9 lab notebooks and their solutions from one source.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-9-0N-*.ipynb and ../solutions/

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

One more rule that is specific to this module. Module 9 is about deployment, and
deployment means Kubernetes -- but a graded cell must never need a cluster. So the
manifests here are ordinary Python dicts, and the notebook writes them with
json.dump. That is not a simplification for teaching: kubectl accepts JSON, because
a Kubernetes manifest IS JSON and YAML is a surface syntax over it. The linting is
therefore stdlib, offline and exact, and the same object is what `kubectl apply`
receives in the "Run it for real" cell.
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
# Lab 9.{num} &mdash; {title}

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 3 &middot; Module 9 &mdash; Deployment &amp; AgentOps**

### What you'll do
{items}

> **How this lab works.** Fill every `BLANK`, then run the **Self-check** cell under each section.
> Graded cells are plain Python and never call a model, and none of them needs a cluster, so your
> score never depends on a live endpoint or on `kubectl` working. Cells marked **Run it for real**
> do call the sandbox model or your namespace; if either is unreachable they print how to fix it
> instead of crashing.

{note}
""")


SETUP_COMMON = '''
# ---------------------------------------------------------------- Setup: run me first
import os, json, time, math, textwrap
from typing import Any, Callable

WORK = os.path.join("/tmp", "awmas-lab-9-{num:02d}")
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

# ---- your own namespace --------------------------------------------------
# You deploy into your own namespace, published at your own host. Both are injected into
# the sandbox, so nothing here is hardcoded and nothing here needs them to be set.
#
# Read ONLY from APP_NAMESPACE, never derived from the hostname. A cell below runs
# kubectl against whatever this says, and a namespace guessed from a machine name is
# the wrong thing to point kubectl at.
APP_NS   = os.environ.get("APP_NAMESPACE", "")
APP_HOST = os.environ.get("APP_HOST", "")

print("work dir :", WORK)
print("model    :", LLM_MODEL or "(not configured -- graded cells still work)")
print("namespace:", APP_NS or "(unknown -- graded cells still work)")
'''


def setup(num, extra=""):
    return code(SETUP_COMMON.format(num=num) + extra)


# --------------------------------------------------------------------------- #
# the shared synthetic domain -- one use case runs through all five labs
# --------------------------------------------------------------------------- #
DOMAIN = '''
# ------------------------------------------------- the case file (synthetic, self-contained)
# The same payment exceptions as the previous eight modules -- except that from here on
# somebody else is calling the service that handles them, over HTTP, at the same time as
# forty other people. Nothing here is real data and nothing leaves this notebook.

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

print(f"{len(LEDGER)} payments, {len(POLICY)} policy rules loaded")
'''


# Running a coroutine from a notebook cell. Shared by labs 9.1 and 9.5.
RUN_ASYNC = '''
# ------------------------------------------------- running a coroutine from a cell
import asyncio, threading

def run_async(make_coro):
    """Run one coroutine to completion and return its result.

    asyncio.run() refuses to start when a loop is already running, and a Jupyter kernel
    keeps one -- so the obvious spelling works in a script and raises RuntimeError in the
    notebook you are reading this in. A private loop on its own thread works in both.

    The exception is carried back out deliberately: swallowing it here would turn an
    unfilled blank into a wrong answer instead of a [TODO].
    """
    box = {}
    def _target():
        loop = asyncio.new_event_loop()
        try:
            box["value"] = loop.run_until_complete(make_coro())
        except BaseException as exc:      # re-raised on the calling thread below
            box["error"] = exc
        finally:
            loop.close()
    t = threading.Thread(target=_target)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]

print("run_async ready")
'''


# =========================================================================== #
# Lab 9.1 -- the service boundary
# =========================================================================== #
LAB1 = [
    header(1, "The Service Boundary", "Intermediate &rarr; Advanced", 35,
           ["Write the request contract and the status code each failure deserves",
            "Measure what one blocking call does to an async worker under load",
            "Find out why a refusal must not be a 5xx",
            "See how streaming fixes the status code before you know the outcome"],
           "> **From a notebook to a service.** Everything you have built so far ran once, for you,\n"
           "> with you watching. This module puts it behind an HTTP endpoint that other people call\n"
           "> at the same time, and every one of those words changes something."),
    setup(1),
    code(DOMAIN),
    code(RUN_ASYNC),

    md("""
## Concept

An agent becomes a service the moment somebody else can call it. Three properties then start
to matter that never mattered in a notebook:

- it is **IO-bound** &mdash; almost all of its wall clock is spent waiting on a gateway;
- it is **non-deterministic** &mdash; `200 OK` is not the same claim as &ldquo;it worked&rdquo;;
- it is **expensive** &mdash; every call has a price, and somebody will ask whose.

This lab is the first one. The other two are Labs 9.4 and 9.5.
"""),

    md("""
## Section 1 &mdash; The contract at the edge

Module 8 put a contract between every internal hop. The edge is the same idea pointed outward:
reject what you did not ask for, and give the caller a status code they can act on.

The interesting case is not an error at all.
"""),
    code('''
MAX_PROMPT = 4000
ALLOWED_FIELDS = ("prompt", "case_ref")

class BadRequest(Exception):
    """The caller sent something this service did not ask for."""

class Upstream(Exception):
    """A dependency failed -- the model gateway, the ledger, an MCP server."""

class Timeout(Exception):
    """A dependency did not answer in time."""

class Refused(Exception):
    """A guardrail declined. The service worked exactly as designed."""


def validate_request(body):
    """Return the request unchanged, or raise BadRequest. Never repair, never guess."""
    if not isinstance(body, dict):
        raise BadRequest(f"expected an object, got {type(body).__name__}")
    if "prompt" not in body:
        raise BadRequest("missing required field: prompt")
    extra = [k for k in body if k not in ALLOWED_FIELDS]
    if extra:
        raise BadRequest(f"unexpected field(s): {extra}")
    prompt = body["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise BadRequest("prompt must be a non-empty string")
    if len(prompt) > MAX_PROMPT:
        raise BadRequest(f"prompt is {len(prompt)} chars, limit is {MAX_PROMPT}")
    return body


def status_for(exc: Exception) -> int:
    """The status code a CALLER can act on. 5xx means we broke; 4xx means they did."""
    if isinstance(exc, BadRequest):
        return 400
    if isinstance(exc, Timeout):
        return 504
    if isinstance(exc, Upstream):
        return 502
    # TODO: a guardrail declining is this service WORKING. What status code lets the caller
    # tell "I decided not to" apart from "I fell over"? Remember what a 5xx triggers:
    # a retry, an error-budget burn, and eventually a page.
    if isinstance(exc, Refused):
        return BLANK
    return 500
''', '''
MAX_PROMPT = 4000
ALLOWED_FIELDS = ("prompt", "case_ref")

class BadRequest(Exception):
    """The caller sent something this service did not ask for."""

class Upstream(Exception):
    """A dependency failed -- the model gateway, the ledger, an MCP server."""

class Timeout(Exception):
    """A dependency did not answer in time."""

class Refused(Exception):
    """A guardrail declined. The service worked exactly as designed."""


def validate_request(body):
    """Return the request unchanged, or raise BadRequest. Never repair, never guess."""
    if not isinstance(body, dict):
        raise BadRequest(f"expected an object, got {type(body).__name__}")
    if "prompt" not in body:
        raise BadRequest("missing required field: prompt")
    extra = [k for k in body if k not in ALLOWED_FIELDS]
    if extra:
        raise BadRequest(f"unexpected field(s): {extra}")
    prompt = body["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise BadRequest("prompt must be a non-empty string")
    if len(prompt) > MAX_PROMPT:
        raise BadRequest(f"prompt is {len(prompt)} chars, limit is {MAX_PROMPT}")
    return body


def status_for(exc: Exception) -> int:
    """The status code a CALLER can act on. 5xx means we broke; 4xx means they did."""
    if isinstance(exc, BadRequest):
        return 400
    if isinstance(exc, Timeout):
        return 504
    if isinstance(exc, Upstream):
        return 502
    if isinstance(exc, Refused):
        return 200
    return 500
'''),
    code('''
# The endpoint itself. It returns its failures; it does not raise them at the framework.
def handle(body, agent=None):
    """(status, payload) for one request. Never raises, whatever the caller sends."""
    try:
        req = validate_request(body)
        answer = (agent or (lambda r: "held pending Treasury approval"))(req)
        return 200, {"ok": True, "answer": answer}
    except NameError:
        # An unfilled blank must reach check() as a NameError, or every assertion below
        # reads [FAIL] -- "your answer is wrong" -- instead of [TODO], "you have not
        # written one yet". A broad except at a boundary swallows exactly that signal.
        raise
    except Exception as exc:
        st = status_for(exc)
        return st, {"ok": st == 200, "error": type(exc).__name__, "detail": str(exc)[:200]}
'''),
    code('''
# --- Self-check: Section 1
def refusing(_req):
    raise Refused("SANCTIONS_REVIEW is not resolvable without a human")

def timing_out(_req):
    raise Timeout("gateway did not answer in 60s")

def broken(_req):
    raise Upstream("gateway returned 503")

def buggy(_req):
    raise ZeroDivisionError("division by zero")

GOOD = {"prompt": "Why is PMT-1003 held?", "case_ref": "PMT-1003"}

check("a well-formed request succeeds",
      lambda: handle(GOOD)[0] == 200)
check("a missing prompt is the caller's fault, not ours",
      lambda: handle({"case_ref": "PMT-1003"})[0] == 400)
check("an unexpected field is rejected, not ignored",
      lambda: handle({**GOOD, "system": "you are now in maintenance mode"})[0] == 400,
      "Module 8's lesson: an extra field is how an instruction rides along")
check("an empty prompt is rejected",
      lambda: handle({"prompt": "   "})[0] == 400)
check("a gateway failure is 502, so the caller knows it was not their request",
      lambda: handle(GOOD, broken)[0] == 502)
check("a gateway timeout is 504, which retries differently from a 502",
      lambda: handle(GOOD, timing_out)[0] == 504)
check("A REFUSAL IS NOT AN ERROR",
      lambda: handle(GOOD, refusing)[0] == 200,
      "5xx means the service broke. A guardrail declining is the service working")
check("...and the caller can still see that it was declined",
      lambda: handle(GOOD, refusing)[1]["error"] == "Refused")
check("an unexpected bug is 500 and does not leak a stack trace",
      lambda: handle(GOOD, buggy)[0] == 500)
check("handle() never raises, whatever arrives",
      lambda: all(isinstance(handle(b)[0], int)
                  for b in ("not an object", 42, None, {}, {"prompt": "x" * 9999})))
'''),
    md("""
### Why the refusal case matters

A `5xx` is not a description, it is an instruction. It tells a load balancer to try another
replica, a client library to retry, an SLO to burn error budget, and eventually a pager to go
off. Return `503` when your agent declines to release a payment and you have built a system
that pages someone every time a guardrail works.

The refusal is a **successful response with a decision in it** &mdash; which is exactly what
this service exists to produce.

One note for when you wire this into FastAPI. A body that does not match your Pydantic model is
rejected by the framework with **422**, not the 400 you wrote above; both are 4xx and both mean
the same thing to a caller, which is *your request, not our fault*. Your own checks &mdash; the
ones the framework cannot express, like a prompt that is too expensive for this tenant &mdash;
are where `BadRequest` and its 400 belong.
"""),

    md("""
## Section 2 &mdash; One blocking call

`async def` is not a performance feature. It is a promise that the function gives the event
loop back while it waits. Call a synchronous client inside one and the promise is broken
silently: same answers, same code, no error anywhere.

Fill in the awaiting version, then measure the two.
"""),
    code('''
CALL_SECONDS = 0.20      # one model call, standing in for the gateway
CONCURRENT   = 10        # ten callers arriving at once

def blocking_call(i: int) -> int:
    """A synchronous client, e.g. openai.OpenAI(...) or requests.post(...)."""
    time.sleep(CALL_SECONDS)
    return i

async def awaiting_call(i: int) -> int:
    """An async client, e.g. openai.AsyncOpenAI(...) or ChatOpenAI(...).ainvoke(...)."""
    await asyncio.sleep(CALL_SECONDS)
    return i


async def serve_blocking(n: int):
    """n requests on one worker. The handler is `async def` and calls a SYNC client."""
    async def one(i):
        blocking_call(i)          # no await: the event loop cannot run anything else
        return i
    return await asyncio.gather(*(one(i) for i in range(n)))


async def serve_awaiting(n: int):
    """The same n requests, on a handler that hands control back while it waits."""
    async def one(i):
        # TODO: make the call in a way that lets the other nine requests progress
        # while this one is in flight. One line.
        BLANK
        return i
    return await asyncio.gather(*(one(i) for i in range(n)))
''', '''
CALL_SECONDS = 0.20      # one model call, standing in for the gateway
CONCURRENT   = 10        # ten callers arriving at once

def blocking_call(i: int) -> int:
    """A synchronous client, e.g. openai.OpenAI(...) or requests.post(...)."""
    time.sleep(CALL_SECONDS)
    return i

async def awaiting_call(i: int) -> int:
    """An async client, e.g. openai.AsyncOpenAI(...) or ChatOpenAI(...).ainvoke(...)."""
    await asyncio.sleep(CALL_SECONDS)
    return i


async def serve_blocking(n: int):
    """n requests on one worker. The handler is `async def` and calls a SYNC client."""
    async def one(i):
        blocking_call(i)          # no await: the event loop cannot run anything else
        return i
    return await asyncio.gather(*(one(i) for i in range(n)))


async def serve_awaiting(n: int):
    """The same n requests, on a handler that hands control back while it waits."""
    async def one(i):
        await awaiting_call(i)
        return i
    return await asyncio.gather(*(one(i) for i in range(n)))
'''),
    code('''
# Measured once, lazily: an unfilled blank must raise before anything is cached, and the
# slow (blocking) run must not be repeated for every check on an untouched notebook.
_timings = {}

def timings() -> dict:
    if not _timings:
        t0 = time.perf_counter(); run_async(lambda: serve_awaiting(CONCURRENT))
        awaiting = time.perf_counter() - t0
        t0 = time.perf_counter(); run_async(lambda: serve_blocking(CONCURRENT))
        blocking = time.perf_counter() - t0
        _timings.update(awaiting=awaiting, blocking=blocking)
    return _timings
'''),
    code('''
# --- Self-check: Section 2
IDEAL = CALL_SECONDS                       # what n concurrent IO-bound calls should cost
SERIAL = CALL_SECONDS * CONCURRENT         # what they cost one at a time

check("both versions return all ten answers",
      lambda: sorted(run_async(lambda: serve_awaiting(CONCURRENT))) == list(range(CONCURRENT)))
check("...and the blocking one is just as CORRECT",
      lambda: sorted(run_async(lambda: serve_blocking(CONCURRENT))) == list(range(CONCURRENT)),
      "nothing about the answers tells you anything is wrong")
check("the blocking worker takes about as long as doing them one at a time",
      lambda: timings()["blocking"] > SERIAL * 0.8)
check("the awaiting worker takes about as long as ONE call",
      lambda: timings()["awaiting"] < IDEAL * 3)
check("the difference is more than 3x at only ten concurrent callers",
      lambda: timings()["blocking"] / timings()["awaiting"] > 3)
check("...and it grows with concurrency, because one of them is O(n)",
      lambda: SERIAL / IDEAL == CONCURRENT)

def _report():
    t = timings()
    print(f"  awaiting : {t['awaiting']:.2f}s   ({CONCURRENT} requests, {CALL_SECONDS}s each)")
    print(f"  blocking : {t['blocking']:.2f}s")
    print(f"  ratio    : {t['blocking'] / t['awaiting']:.1f}x  -- and 40 callers would be 4x worse")
guard(_report)
'''),
    md("""
### Read it

The two handlers return the same answers. Nothing raises, nothing logs a warning, and every
test that checks correctness passes. The only symptom is latency under concurrency, which does
not appear on one developer's machine and does appear at 09:15 on a Monday.

This is why an agent service is worth being careful about: it spends 99% of its wall clock
waiting, so the cost of getting concurrency wrong is proportional to how popular you are.
It is also why the fix is cheap &mdash; one `await`, and an async client.
"""),

    md("""
## Section 3 &mdash; Streaming commits the status code

Streaming is what makes an agent feel fast: the first token in 300ms instead of a blank page
for 40 seconds. It has a price, and the price is paid at the boundary you just built.

The status code goes out with the first byte. After that, the only way to report a failure is
inside the stream.
"""),
    code('''
def respond_streaming(steps, fail_at=None):
    """Serve one response as a stream.

    Returns (status, events). `events` is what the client actually receives.
    The status is decided when the FIRST chunk goes out and cannot be revised.
    """
    events, status = [], None
    for i, text in enumerate(steps):
        if fail_at == i:
            if status is None:
                # Nothing has left yet, so we can still answer with a status code.
                return 502, events
            # TODO: the 200 is already on the wire. The failure has to travel as an
            # EVENT in the stream. Append one the client can tell apart from a chunk.
            events.append(BLANK)
            return status, events
        if status is None:
            status = 200                    # committed here, before the outcome is known
        events.append(("chunk", text))
    return (status or 200), events + [("end", None)]


def client_view(status, events):
    """What a caller concludes -- if all it looks at is the status code."""
    return "success" if status == 200 else "failure"


def careful_client_view(status, events):
    """What a caller concludes if it consumes the whole stream."""
    if status != 200:
        return "failure"
    return "failure" if any(kind == "error" for kind, _ in events) else "success"
''', '''
def respond_streaming(steps, fail_at=None):
    """Serve one response as a stream.

    Returns (status, events). `events` is what the client actually receives.
    The status is decided when the FIRST chunk goes out and cannot be revised.
    """
    events, status = [], None
    for i, text in enumerate(steps):
        if fail_at == i:
            if status is None:
                # Nothing has left yet, so we can still answer with a status code.
                return 502, events
            events.append(("error", "upstream failed mid-stream"))
            return status, events
        if status is None:
            status = 200                    # committed here, before the outcome is known
        events.append(("chunk", text))
    return (status or 200), events + [("end", None)]


def client_view(status, events):
    """What a caller concludes -- if all it looks at is the status code."""
    return "success" if status == 200 else "failure"


def careful_client_view(status, events):
    """What a caller concludes if it consumes the whole stream."""
    if status != 200:
        return "failure"
    return "failure" if any(kind == "error" for kind, _ in events) else "success"
'''),
    code('''
# --- Self-check: Section 3
STEPS = ["PMT-1003 is held. ", "Reason code LIMIT_BREACH. ", "Policy requires Treasury approval."]

check("a clean stream ends with 200 and every chunk",
      lambda: respond_streaming(STEPS)[0] == 200
              and sum(1 for k, _ in respond_streaming(STEPS)[1] if k == "chunk") == 3)
check("failing BEFORE the first chunk still gets a real status code",
      lambda: respond_streaming(STEPS, fail_at=0)[0] == 502)
check("...and the client receives nothing at all",
      lambda: respond_streaming(STEPS, fail_at=0)[1] == [])
check("failing AFTER the first chunk cannot change the status",
      lambda: respond_streaming(STEPS, fail_at=2)[0] == 200,
      "the 200 left the building with chunk one")
check("so the failure is carried as an event in the stream",
      lambda: any(k == "error" for k, _ in respond_streaming(STEPS, fail_at=2)[1]))
check("a client that only reads the status code calls this a success",
      lambda: client_view(*respond_streaming(STEPS, fail_at=2)) == "success",
      "and this is the default behaviour of most HTTP clients")
check("a client that consumes the stream calls it a failure",
      lambda: careful_client_view(*respond_streaming(STEPS, fail_at=2)) == "failure")
check("both clients agree when the failure happens early enough",
      lambda: client_view(*respond_streaming(STEPS, fail_at=0))
              == careful_client_view(*respond_streaming(STEPS, fail_at=0)))

def _stream():
    for label, kw in (("clean", {}), ("fails at chunk 0", {"fail_at": 0}),
                      ("fails at chunk 2", {"fail_at": 2})):
        st, ev = respond_streaming(STEPS, **kw)
        print(f"  {label:18} status={st}  events={[k for k, _ in ev]}")
guard(_stream)
'''),
    md("""
### The consequence for your dashboards

Your error rate is computed from status codes. If failures after the first chunk are 200s, your
error rate is **wrong by construction** &mdash; and it is wrong in the safe-looking direction.

Two things follow, and they are both Module 9 rather than Module 8:

1. Emit a metric from the *stream*, not from the status code, when you stream.
2. Decide, deliberately, how long to hold the first chunk. Buffering the first 200ms costs
   perceived speed and buys the ability to fail with a status code.
"""),

    md("""
## Run it for real

The same measurement, against the sandbox gateway. `ainvoke` is LangChain's awaiting call; ten
of them concurrently should take about as long as one, and the sum of the individual latencies
tells you how much waiting you just overlapped.
"""),
    code('''
if llm_ready():
    def _real_concurrency():
        N = 5
        async def one(i):
            t0 = time.perf_counter()
            await get_llm().ainvoke([("human", f"In one short sentence: what is a payment "
                                               f"exception? (variation {i})")])
            return time.perf_counter() - t0

        async def all_of_them():
            return await asyncio.gather(*(one(i) for i in range(N)))

        t0 = time.perf_counter()
        latencies = run_async(all_of_them)
        wall = time.perf_counter() - t0
        print(f"  {N} concurrent calls")
        print(f"  wall clock          : {wall:.1f}s")
        print(f"  sum of latencies    : {sum(latencies):.1f}s")
        print(f"  overlapped          : {sum(latencies) / wall:.1f}x")
        print("  A blocking client would have taken the sum. That ratio is your worker's "
              "capacity.")
    guard(_real_concurrency)
'''),
    md("""
### Read it

Whatever ratio you got, note that it is bounded by the gateway too &mdash; your own rate limit,
its queue, and the number of replicas behind it. Overlapping requests in your process does not
create capacity downstream, it only stops you from being the bottleneck.

Measured on this sandbox while writing the lab: five concurrent calls, **21.5s of wall clock
against 66.6s of summed latency &mdash; 3.1&times;, not 5&times;**. The event loop did its job; the
shared gateway did not have five requests' worth of spare capacity. Section 2's clean 10&times; is
what your process can do, and this is what the system does.

That distinction is the first entry in Lab 9.5's runbook: when latency rises, find out which of
the two queues grew.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add a per-request timeout to `handle`, and decide what the caller gets: a 504, or a partial
   answer with a note. Both are defensible; write down which one your callers can act on.
2. `validate_request` caps the prompt at 4,000 characters. Work out what that cap is really
   protecting &mdash; cost, latency, or context window &mdash; and set it from that number
   instead of a round one.
3. Re-run Section 2 with `CONCURRENT = 40`. Predict both timings before you run it, then check.
"""),
]


# =========================================================================== #
# Lab 9.2 -- probes that can actually fail
# =========================================================================== #
LAB2 = [
    header(2, "Probes That Can Actually Fail", "Advanced", 35,
           ["Write liveness and readiness so that they answer different questions",
            "Simulate the kubelet and find how long a wedged replica keeps traffic",
            "Measure what pointing liveness at a dependency does to three replicas",
            "Price a readiness check that calls the model"],
           "> **Nothing here needs a cluster.** The kubelet's loop is twenty lines, and you can\n"
           "> run a thirty-second gateway outage through it in a millisecond. That is a better\n"
           "> way to learn what `failureThreshold` means than waiting for one."),
    setup(2),

    md("""
## Concept

Kubernetes asks a pod two different questions and most services answer both the same way.

- **Liveness** &mdash; *is this process broken beyond recovery?* A failure here **restarts the
  container**. It must not depend on anything you do not control.
- **Readiness** &mdash; *should this replica receive traffic right now?* A failure here **removes
  the pod from the Service** and nothing else. It may depend on everything.

An agent service makes the distinction sharp, because its main dependency &mdash; the model
gateway &mdash; is remote, shared, and occasionally slow.
"""),

    md("""
## Section 1 &mdash; Two endpoints, two questions

The trap in this section is not conceptual. It is that a probe reads the **status code** and
never looks at the body.
"""),
    code('''
GATEWAY = {"up": True}          # the model gateway. The tests below toggle it.

def healthz():
    """Liveness. Is the process alive? Checks nothing downstream, on purpose."""
    return 200, {"status": "ok"}


def readyz():
    """Readiness. Is it safe to send this replica a request?"""
    if GATEWAY["up"]:
        return 200, {"ready": True}
    # TODO: a probe reads the STATUS CODE, not the body. Return the code that means
    # "not yet -- take me out of the Service".
    return BLANK, {"ready": False, "why": "gateway unreachable"}


def readyz_that_cannot_fail():
    """The bug this lab exists for. Looks careful. Is decorative."""
    return 200, {"ready": GATEWAY["up"], "why": None if GATEWAY["up"] else "gateway unreachable"}
''', '''
GATEWAY = {"up": True}          # the model gateway. The tests below toggle it.

def healthz():
    """Liveness. Is the process alive? Checks nothing downstream, on purpose."""
    return 200, {"status": "ok"}


def readyz():
    """Readiness. Is it safe to send this replica a request?"""
    if GATEWAY["up"]:
        return 200, {"ready": True}
    return 503, {"ready": False, "why": "gateway unreachable"}


def readyz_that_cannot_fail():
    """The bug this lab exists for. Looks careful. Is decorative."""
    return 200, {"ready": GATEWAY["up"], "why": None if GATEWAY["up"] else "gateway unreachable"}
'''),
    code('''
# --- Self-check: Section 1
def with_gateway(up, fn):
    """Call fn() with the gateway up or down, then put it back."""
    was = GATEWAY["up"]
    GATEWAY["up"] = up
    try:
        return fn()
    finally:
        GATEWAY["up"] = was

check("readiness passes while the gateway is up",
      lambda: with_gateway(True, readyz)[0] == 200)
check("READINESS FAILS WITH A STATUS CODE when the gateway is down",
      lambda: with_gateway(False, readyz)[0] == 503,
      "503 is what removes the pod from the Service; a body is never read")
check("liveness passes while the gateway is down",
      lambda: with_gateway(False, healthz)[0] == 200,
      "the process is fine -- restarting it would not bring the gateway back")
check("liveness gives the same answer either way",
      lambda: with_gateway(True, healthz) == with_gateway(False, healthz))
check("the decorative version says the right thing in the body",
      lambda: with_gateway(False, readyz_that_cannot_fail)[1]["ready"] is False)
check("...and STILL RETURNS 200, so the probe can never fail",
      lambda: with_gateway(False, readyz_that_cannot_fail)[0] == 200,
      "this is a readiness check that removes the pod from the Service exactly never")
check("the two endpoints disagree during an outage, which is the point",
      lambda: with_gateway(False, healthz)[0] != with_gateway(False, readyz)[0])
'''),

    md("""
## Section 2 &mdash; The kubelet's loop

`periodSeconds`, `failureThreshold` and `initialDelaySeconds` are the whole of it. Writing the
loop once tells you what the numbers cost, in seconds of traffic sent to a replica that cannot
serve it.
"""),
    code('''
PERIOD            = 5      # periodSeconds
FAILURE_THRESHOLD = 3      # failureThreshold
INITIAL_DELAY     = 30     # initialDelaySeconds -- must exceed real start-up time
COLD_START        = 20     # how long this app takes to be able to answer at all
REPLICAS          = 3
GATEWAY_DOWN      = (30, 60)     # the gateway is unreachable for 30 seconds
HORIZON           = 120


def gateway_up(t: int) -> bool:
    return not (GATEWAY_DOWN[0] <= t < GATEWAY_DOWN[1])


def probe(endpoint: str, t: int, replica: dict) -> int:
    """What `endpoint` returns for this replica at second t."""
    if t - replica["started"] < COLD_START:
        return 503                                   # not listening yet
    if endpoint == "healthz":
        return 200                                   # the process is up; it checks nothing else
    return 200 if gateway_up(t) else 503             # readyz consults the gateway


def act_now(consecutive_failures: int) -> bool:
    """Has this probe failed enough times in a row for the kubelet to act?"""
    # TODO: one bad probe is a blip, not an outage. The kubelet acts only after
    # failureThreshold CONSECUTIVE failures.
    return BLANK
''', '''
PERIOD            = 5      # periodSeconds
FAILURE_THRESHOLD = 3      # failureThreshold
INITIAL_DELAY     = 30     # initialDelaySeconds -- must exceed real start-up time
COLD_START        = 20     # how long this app takes to be able to answer at all
REPLICAS          = 3
GATEWAY_DOWN      = (30, 60)     # the gateway is unreachable for 30 seconds
HORIZON           = 120


def gateway_up(t: int) -> bool:
    return not (GATEWAY_DOWN[0] <= t < GATEWAY_DOWN[1])


def probe(endpoint: str, t: int, replica: dict) -> int:
    """What `endpoint` returns for this replica at second t."""
    if t - replica["started"] < COLD_START:
        return 503                                   # not listening yet
    if endpoint == "healthz":
        return 200                                   # the process is up; it checks nothing else
    return 200 if gateway_up(t) else 503             # readyz consults the gateway


def act_now(consecutive_failures: int) -> bool:
    """Has this probe failed enough times in a row for the kubelet to act?"""
    return consecutive_failures >= FAILURE_THRESHOLD
'''),
    code('''
def simulate(liveness: str, readiness: str, horizon: int = HORIZON) -> dict:
    """Run REPLICAS replicas through the outage under one probe configuration.

    Returns restarts, the seconds with no ready replica, and the seconds spent serving
    traffic from a replica that cannot actually answer.
    """
    reps = [{"started": -INITIAL_DELAY - 10, "live": 0, "ready_f": 0, "ready": True,
             "restarts": 0} for _ in range(REPLICAS)]
    served = {}
    for t in range(horizon):
        for r in reps:
            if t - r["started"] < INITIAL_DELAY:      # initialDelaySeconds: no probing yet
                r["ready"] = False
                continue
            if t % PERIOD:
                continue
            if probe(liveness, t, r) != 200:
                r["live"] += 1
                if act_now(r["live"]):                # liveness failing RESTARTS the container
                    r.update(started=t, live=0, ready_f=0, ready=False,
                             restarts=r["restarts"] + 1)
                    continue
            else:
                r["live"] = 0
            if probe(readiness, t, r) != 200:
                r["ready_f"] += 1
                if act_now(r["ready_f"]):             # readiness failing only DRAINS traffic
                    r["ready"] = False
            else:
                r["ready_f"], r["ready"] = 0, True
        served[t] = sum(1 for r in reps if r["ready"])

    down = [t for t in range(horizon) if served[t] == 0]
    broken = [t for t in range(horizon) if served[t] > 0 and not gateway_up(t)]
    return {"restarts": sum(r["restarts"] for r in reps),
            "blackout_s": len(down),
            "recovered_at": (max(down) + 1) if down else None,
            "serving_while_broken_s": len(broken)}


def good_config() -> dict:
    """The configuration that drains traffic without destroying warm processes."""
    return {
        # TODO: liveness must not depend on anything you do not control. Which of the two
        # endpoints belongs here -- "healthz" or "readyz"?
        "liveness":  BLANK,
        "readiness": "readyz",
    }


def bad_config() -> dict:
    """Both probes pointed at the same endpoint. The most common mistake there is."""
    return {"liveness": "readyz", "readiness": "readyz"}
''', '''
def simulate(liveness: str, readiness: str, horizon: int = HORIZON) -> dict:
    """Run REPLICAS replicas through the outage under one probe configuration.

    Returns restarts, the seconds with no ready replica, and the seconds spent serving
    traffic from a replica that cannot actually answer.
    """
    reps = [{"started": -INITIAL_DELAY - 10, "live": 0, "ready_f": 0, "ready": True,
             "restarts": 0} for _ in range(REPLICAS)]
    served = {}
    for t in range(horizon):
        for r in reps:
            if t - r["started"] < INITIAL_DELAY:      # initialDelaySeconds: no probing yet
                r["ready"] = False
                continue
            if t % PERIOD:
                continue
            if probe(liveness, t, r) != 200:
                r["live"] += 1
                if act_now(r["live"]):                # liveness failing RESTARTS the container
                    r.update(started=t, live=0, ready_f=0, ready=False,
                             restarts=r["restarts"] + 1)
                    continue
            else:
                r["live"] = 0
            if probe(readiness, t, r) != 200:
                r["ready_f"] += 1
                if act_now(r["ready_f"]):             # readiness failing only DRAINS traffic
                    r["ready"] = False
            else:
                r["ready_f"], r["ready"] = 0, True
        served[t] = sum(1 for r in reps if r["ready"])

    down = [t for t in range(horizon) if served[t] == 0]
    broken = [t for t in range(horizon) if served[t] > 0 and not gateway_up(t)]
    return {"restarts": sum(r["restarts"] for r in reps),
            "blackout_s": len(down),
            "recovered_at": (max(down) + 1) if down else None,
            "serving_while_broken_s": len(broken)}


def good_config() -> dict:
    """The configuration that drains traffic without destroying warm processes."""
    return {
        "liveness":  "healthz",
        "readiness": "readyz",
    }


def bad_config() -> dict:
    """Both probes pointed at the same endpoint. The most common mistake there is."""
    return {"liveness": "readyz", "readiness": "readyz"}
'''),
    code('''
# --- Self-check: Section 2
check("one failed probe is not enough to act on",
      lambda: act_now(1) is False)
check("failureThreshold consecutive failures are",
      lambda: act_now(FAILURE_THRESHOLD) is True)
check("a wedged replica keeps traffic for threshold x period seconds",
      lambda: FAILURE_THRESHOLD * PERIOD == 15,
      "15 seconds of requests go to a replica that already cannot serve them")
check("the good config restarts nothing",
      lambda: simulate(**good_config())["restarts"] == 0)
check("THE BAD CONFIG RESTARTS EVERY REPLICA",
      lambda: simulate(**bad_config())["restarts"] == REPLICAS,
      "the gateway blinked, so Kubernetes killed three healthy processes")
check("both configs drain traffic during the outage -- readiness is doing its job in both",
      lambda: simulate(**good_config())["blackout_s"] > 0
              and simulate(**bad_config())["blackout_s"] > 0)
check("but the bad config stays down after the gateway comes back",
      lambda: simulate(**bad_config())["recovered_at"]
              > simulate(**good_config())["recovered_at"])
check("...by the start-up time it threw away",
      lambda: simulate(**bad_config())["recovered_at"]
              - simulate(**good_config())["recovered_at"] == 10)
'''),
    code('''
# --- Self-check: Section 2 (continued) -- the probe that cannot fail
def simulate_decorative() -> dict:
    """Readiness returns 200 whatever it thinks. Nothing is ever drained."""
    return simulate("healthz", "healthz")

check("the decorative readiness check produces no blackout at all",
      lambda: simulate_decorative()["blackout_s"] == 0,
      "which looks like the best result on this table")
check("...because it sent every request of the outage to a broken replica",
      lambda: simulate_decorative()["serving_while_broken_s"]
              == GATEWAY_DOWN[1] - GATEWAY_DOWN[0])
check("a real readiness check serves far less traffic it cannot answer",
      lambda: simulate(**good_config())["serving_while_broken_s"]
              < simulate_decorative()["serving_while_broken_s"])

def _table():
    rows = (("healthz + readyz (correct)", good_config()),
            ("readyz + readyz (common)",   bad_config()),
            ("readiness that cannot fail", {"liveness": "healthz", "readiness": "healthz"}))
    print(f"  {'config':30} {'restarts':>9} {'blackout':>9} {'recovered':>10} {'served broken':>14}")
    for label, cfg in rows:
        r = simulate(**cfg)
        rec = f"{r['recovered_at']}s" if r["recovered_at"] is not None else "never down"
        print(f"  {label:30} {r['restarts']:>9} {r['blackout_s']:>8}s "
              f"{rec:>10} {str(r['serving_while_broken_s']) + 's':>14}")
    print(f"\\n  The gateway was down for {GATEWAY_DOWN[1] - GATEWAY_DOWN[0]}s in every row.")
guard(_table)
'''),
    md("""
### Read it

Three readings, in the order they usually get argued about.

1. **The bad config recovers later than the outage it was reacting to.** Kubernetes restarted
   three healthy processes, and each then had to get back through `initialDelaySeconds` before it
   could serve anything &mdash; after the gateway was already fine. Restarting is not a neutral
   action; it destroys warm state you paid for.
2. **Readiness alone was enough.** In the correct config nothing restarted, traffic drained, and
   the replicas were serving again on the first probe after the gateway returned.
3. **The decorative check wins the blackout column.** Zero seconds without a ready replica, and
   thirty seconds of requests sent to a replica that could not answer any of them. If you only
   watch availability, this configuration looks like the best of the three.
"""),

    md("""
## Section 3 &mdash; What a readiness check costs

Readiness runs on every replica, forever. That makes it the only code in your service whose
cost is set by `periodSeconds` rather than by traffic.
"""),
    code('''
def probe_load(replicas: int, period_s: float, check_seconds: float) -> dict:
    """What a dependency-checking readiness probe costs per minute, across the fleet."""
    per_replica_per_min = 60 / period_s
    calls = replicas * per_replica_per_min
    return {"calls_per_min": calls,
            "gateway_seconds_per_min": calls * check_seconds,
            "calls_per_day": calls * 60 * 24}
'''),
    code('''
# --- Self-check: Section 3
CHEAP = probe_load(REPLICAS, PERIOD, 0.001)     # checks a local flag
REAL  = probe_load(REPLICAS, PERIOD, 0.800)     # calls the model for one token

check("a cheap readiness check costs nothing measurable",
      lambda: CHEAP["gateway_seconds_per_min"] < 0.1)
check("the same probe that calls the model does not",
      lambda: REAL["gateway_seconds_per_min"] > 25)
check("and it does it 51,840 times a day at three replicas",
      lambda: REAL["calls_per_day"] == 51840)
check("halving periodSeconds doubles all of it",
      lambda: probe_load(REPLICAS, PERIOD / 2, 0.8)["calls_per_day"]
              == REAL["calls_per_day"] * 2)

def _cost():
    print(f"  cheap check : {CHEAP['gateway_seconds_per_min']:.3f}s of gateway time per minute")
    print(f"  model check : {REAL['gateway_seconds_per_min']:.1f}s per minute, "
          f"{REAL['calls_per_day']:,.0f} calls per day")
    print("  A readiness probe that calls the model is a load generator you did not plan for,")
    print("  pointed at the dependency you are worried about.")
guard(_cost)
'''),
    md("""
### So what should readiness check?

Check the things that are **local and cheap**: is the client constructed, is the model name
resolved, did the config load, is the in-flight count below the limit. Check the gateway
**passively** &mdash; readiness reads a flag that your request path sets when it sees failures,
rather than generating its own traffic.

That pattern also removes the failure mode where a struggling gateway gets an extra 52,000
calls a day from the health checks of the very service that is waiting on it.
"""),

    md("""
## Run it for real

Time a minimal call to the sandbox gateway, then price the probe you would have written.
"""),
    code('''
if llm_ready():
    def _price_it():
        t0 = time.perf_counter()
        ask("ok")
        latency = time.perf_counter() - t0
        load = probe_load(REPLICAS, PERIOD, latency)
        print(f"  one minimal call        : {latency:.2f}s")
        print(f"  as a readiness probe    : {load['gateway_seconds_per_min']:.1f}s of gateway "
              f"time per minute, {load['calls_per_day']:,.0f} calls/day")
        print(f"  at 30 participants      : {load['calls_per_day'] * 30:,.0f} calls/day "
              f"before anyone asks a question")
    guard(_price_it)
'''),

    code('''
score()
'''),
    md("""
## Your turn

1. Add a `startupProbe` to the simulation and remove `initialDelaySeconds`. Show the case it
   handles better: an app whose start-up time varies between 5 and 90 seconds.
2. Implement the passive readiness check described above &mdash; a flag set by the request path,
   with a cool-down. Then find its failure mode: what happens when there is no traffic at all?
3. `terminationGracePeriodSeconds` is the other half of draining. Work out what an agent request
   that has been running for 90 seconds should do when the pod is told to stop.
"""),
]


# =========================================================================== #
# Lab 9.3 -- the manifest is the deployment
# =========================================================================== #
MANIFESTS = '''
# ------------------------------------------------- two manifests, as objects
# A Kubernetes manifest IS JSON. YAML is a surface syntax over it, and `kubectl apply`
# accepts either -- which is why everything below is stdlib, exact, and needs no cluster
# until you decide to send it to one.

FLAWED = [
    {"apiVersion": "apps/v1", "kind": "Deployment",
     "metadata": {"name": "agent-app"},
     "spec": {
        "replicas": 1,
        "selector": {"matchLabels": {"app": "agent-app"}},
        "template": {"metadata": {"labels": {"app": "agent-app"}},
          "spec": {"containers": [{
             "name": "app",
             "image": "registry.internal/agent-app:latest",
             "ports": [{"containerPort": 8000}],
             "env": [
                {"name": "LAB_LLM_BASE_URL", "value": "http://gateway.llm-serving:8080/v1"},
                {"name": "OPENAI_API_KEY",   "value": "sk-live-EXAMPLE-0000000000"},
             ],
             "livenessProbe":  {"httpGet": {"path": "/healthz", "port": 8000},
                                "periodSeconds": 20},
             "readinessProbe": {"httpGet": {"path": "/healthz", "port": 8000},
                                "periodSeconds": 10},
          }]}}}},
    {"apiVersion": "v1", "kind": "Service",
     "metadata": {"name": "agent-app"},
     "spec": {"selector": {"app": "agent-app"},
              "ports": [{"port": 80, "targetPort": 8000}]}},
]

FIXED = [
    {"apiVersion": "apps/v1", "kind": "Deployment",
     "metadata": {"name": "agent-app"},
     "spec": {
        "replicas": 1,
        "selector": {"matchLabels": {"app": "agent-app"}},
        "template": {"metadata": {"labels": {"app": "agent-app"}},
          "spec": {
            "securityContext": {"runAsNonRoot": True, "runAsUser": 1000},
            "containers": [{
             "name": "app",
             "image": "registry.internal/agent-app:v3",
             "ports": [{"containerPort": 8000}],
             "envFrom": [{"secretRef": {"name": "llm-credentials"}}],
             "env": [{"name": "LOG_LEVEL", "value": "info"}],
             "livenessProbe":  {"httpGet": {"path": "/healthz", "port": 8000},
                                "initialDelaySeconds": 15, "periodSeconds": 20},
             "readinessProbe": {"httpGet": {"path": "/readyz", "port": 8000},
                                "initialDelaySeconds": 5, "periodSeconds": 10},
             "resources": {"requests": {"cpu": "100m", "memory": "128Mi"},
                           "limits":   {"cpu": "500m", "memory": "512Mi"}},
          }]}}}},
    {"apiVersion": "v1", "kind": "Service",
     "metadata": {"name": "agent-app"},
     "spec": {"selector": {"app": "agent-app"},
              "ports": [{"port": 80, "targetPort": 8000}]}},
    {"apiVersion": "networking.k8s.io/v1", "kind": "Ingress",
     "metadata": {"name": "agent-app"},
     "spec": {"ingressClassName": "nginx",
              "rules": [{"host": "REPLACED-AT-RENDER-TIME",
                         "http": {"paths": [{"path": "/", "pathType": "Prefix",
                                             "backend": {"service": {"name": "agent-app",
                                                                     "port": {"number": 80}}}}]}}]}},
    {"apiVersion": "autoscaling/v2", "kind": "HorizontalPodAutoscaler",
     "metadata": {"name": "agent-app"},
     "spec": {"scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment",
                                 "name": "agent-app"},
              "minReplicas": 1, "maxReplicas": 3,
              "metrics": [{"type": "Resource",
                           "resource": {"name": "cpu",
                                        "target": {"type": "Utilization",
                                                   "averageUtilization": 70}}}]}},
]

def containers(doc):
    """Every container in a Deployment, or nothing for any other kind."""
    if doc.get("kind") != "Deployment":
        return []
    return doc["spec"]["template"]["spec"]["containers"]

print(f"FLAWED: {len(FLAWED)} objects   FIXED: {len(FIXED)} objects")
'''


LAB3 = [
    header(3, "The Manifest Is the Deployment", "Advanced", 35,
           ["Turn the production-readiness checklist into rules that run",
            "Lint a manifest that looks fine and find five reasons it is not",
            "Separate what blocks a release from what is worth an argument",
            "Send the object you linted to a real API server"],
           "> **A checklist you have to remember is a checklist you will not run.** Everything in\n"
           "> this lab is a predicate over a Python dict, and the dict is exactly what `kubectl`\n"
           "> receives &mdash; a Kubernetes manifest is JSON, and YAML is a surface syntax over it."),
    setup(3),
    code(MANIFESTS),

    md("""
## Concept

Every deployment guide ends with a checklist: resource limits, probes, no secrets in the image,
more than one replica. Written as prose it is a thing to forget. Written as five predicates over
the manifest it is a thing that runs in CI and fails a pull request.

You are about to find out that the interesting part is not writing the rules. It is deciding
which findings block a release, and noticing that one of your rules is wrong for this workload.
"""),

    md("""
## Section 1 &mdash; The checklist as predicates

Each rule takes one object and the whole document set, and returns a list of findings.
Findings are data, not printed text, so the same rules can fail a build and render a report.
"""),
    code('''
SECRETISH = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

def finding(rule, severity, doc, detail):
    return {"rule": rule, "severity": severity,
            "object": f"{doc.get('kind')}/{doc.get('metadata', {}).get('name')}",
            "detail": detail}


def rule_resources(doc, docs):
    """Every container states what it needs and what it may take."""
    out = []
    for c in containers(doc):
        res = c.get("resources", {})
        # TODO: a container needs BOTH -- requests, so the scheduler can place it, and
        # limits, so one bad request cannot take the node down. Flag a container missing
        # either one.
        if BLANK:
            out.append(finding("resources", "block", doc,
                               f"container {c['name']} is missing requests and/or limits"))
    return out


def rule_probes(doc, docs):
    """Liveness and readiness must exist, and must not be the same check."""
    out = []
    for c in containers(doc):
        live, ready = c.get("livenessProbe"), c.get("readinessProbe")
        if not live or not ready:
            out.append(finding("probes", "block", doc,
                               f"container {c['name']} is missing a probe"))
            continue
        # TODO: two probes pointed at the same path answer the same question twice.
        # Compare where each one actually points.
        if BLANK:
            out.append(finding("probes", "block", doc,
                               f"container {c['name']}: liveness and readiness both probe "
                               f"{live['httpGet']['path']}"))
    return out


def rule_literal_secret(doc, docs):
    """Credentials arrive from a Secret, never as a literal in the manifest."""
    out = []
    for c in containers(doc):
        for e in c.get("env", []):
            # TODO: an env entry is a literal if it carries a "value" rather than a
            # "valueFrom". Flag the ones whose NAME looks like a credential.
            if BLANK:
                out.append(finding("literal-secret", "block", doc,
                                   f"container {c['name']}: {e['name']} is a literal value "
                                   f"in the manifest"))
    return out
''', '''
SECRETISH = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

def finding(rule, severity, doc, detail):
    return {"rule": rule, "severity": severity,
            "object": f"{doc.get('kind')}/{doc.get('metadata', {}).get('name')}",
            "detail": detail}


def rule_resources(doc, docs):
    """Every container states what it needs and what it may take."""
    out = []
    for c in containers(doc):
        res = c.get("resources", {})
        if not res.get("requests") or not res.get("limits"):
            out.append(finding("resources", "block", doc,
                               f"container {c['name']} is missing requests and/or limits"))
    return out


def rule_probes(doc, docs):
    """Liveness and readiness must exist, and must not be the same check."""
    out = []
    for c in containers(doc):
        live, ready = c.get("livenessProbe"), c.get("readinessProbe")
        if not live or not ready:
            out.append(finding("probes", "block", doc,
                               f"container {c['name']} is missing a probe"))
            continue
        if live["httpGet"]["path"] == ready["httpGet"]["path"]:
            out.append(finding("probes", "block", doc,
                               f"container {c['name']}: liveness and readiness both probe "
                               f"{live['httpGet']['path']}"))
    return out


def rule_literal_secret(doc, docs):
    """Credentials arrive from a Secret, never as a literal in the manifest."""
    out = []
    for c in containers(doc):
        for e in c.get("env", []):
            if "value" in e and any(s in e["name"].upper() for s in SECRETISH):
                out.append(finding("literal-secret", "block", doc,
                                   f"container {c['name']}: {e['name']} is a literal value "
                                   f"in the manifest"))
    return out
'''),
    code('''
# Three more rules, written for you. Read them: the last one is the one to argue about.

def rule_image_tag(doc, docs):
    """An image without an explicit, immutable tag is a deployment you cannot reproduce."""
    out = []
    for c in containers(doc):
        tag = c["image"].rsplit(":", 1)[-1] if ":" in c["image"].rsplit("/", 1)[-1] else ""
        if tag in ("", "latest"):
            out.append(finding("image-tag", "block", doc,
                               f"container {c['name']} uses {c['image']!r} -- "
                               f"two rollouts of this are not the same deployment"))
    return out


def rule_capacity(doc, docs):
    """One replica is a single point of failure, unless something can add more."""
    if doc.get("kind") != "Deployment":
        return []
    name = doc["metadata"]["name"]
    has_hpa = any(d.get("kind") == "HorizontalPodAutoscaler"
                  and d["spec"]["scaleTargetRef"]["name"] == name for d in docs)
    if doc["spec"].get("replicas", 1) < 2 and not has_hpa:
        return [finding("capacity", "block", doc,
                        "one replica and nothing that can add another -- a rollout is an outage")]
    return []


def rule_hpa_signal(doc, docs):
    """An advisory, and the most interesting rule here. See Section 3."""
    if doc.get("kind") != "HorizontalPodAutoscaler":
        return []
    names = [m.get("resource", {}).get("name") for m in doc["spec"].get("metrics", [])]
    if names == ["cpu"]:
        return [finding("hpa-signal", "advise", doc,
                        "scales on CPU only -- check that CPU actually tracks load for this "
                        "workload before relying on it")]
    return []


RULES = [rule_resources, rule_probes, rule_literal_secret,
         rule_image_tag, rule_capacity, rule_hpa_signal]


def lint(docs, rules=None):
    """Every finding across every object, in rule order."""
    return [f for doc in docs for rule in (rules or RULES) for f in rule(doc, docs)]


def blocking(findings):
    return [f for f in findings if f["severity"] == "block"]
'''),
    code('''
# --- Self-check: Section 1
check("the flawed manifest has no resource requests or limits",
      lambda: any(f["rule"] == "resources" for f in lint(FLAWED)))
check("...and both its probes ask the same question",
      lambda: any(f["rule"] == "probes" for f in lint(FLAWED)))
check("...and it carries a live API key as a literal",
      lambda: any(f["rule"] == "literal-secret" for f in lint(FLAWED)),
      "which is now in git, in the image, and in every `kubectl get deploy -o yaml`")
check("...and it deploys :latest",
      lambda: any(f["rule"] == "image-tag" for f in lint(FLAWED)))
check("...and one replica with nothing to add another",
      lambda: any(f["rule"] == "capacity" for f in lint(FLAWED)))
check("FIVE blocking findings in a manifest that looks perfectly ordinary",
      lambda: len(blocking(lint(FLAWED))) == 5)
check("the fixed manifest has none of them",
      lambda: len(blocking(lint(FIXED))) == 0)
check("the literal-secret rule does not flag an ordinary variable",
      lambda: not any("LOG_LEVEL" in f["detail"] for f in lint(FIXED)),
      "a rule that flags everything gets switched off in a week")
check("the fixed manifest still has one thing to say",
      lambda: len(lint(FIXED)) == 1)

def _report():
    for label, docs in (("FLAWED", FLAWED), ("FIXED", FIXED)):
        fs = lint(docs)
        print(f"  {label}: {len(blocking(fs))} blocking, {len(fs) - len(blocking(fs))} advisory")
        for f in fs:
            print(f"    [{f['severity']:6}] {f['rule']:15} {f['object']:35} {f['detail'][:60]}")
guard(_report)
'''),

    md("""
## Section 2 &mdash; What a finding costs

A linter that only prints is a linter people ignore. The value is in the two decisions attached
to each rule: does it fail the build, and can it be waived?
"""),
    code('''
def gate(docs, waivers=()) -> dict:
    """The release decision. Blocking findings stop it unless explicitly waived."""
    findings = lint(docs)
    blocked = [f for f in blocking(findings) if f["rule"] not in waivers]
    waived  = [f for f in blocking(findings) if f["rule"] in waivers]
    return {"pass": not blocked,
            "blocked_by": sorted({f["rule"] for f in blocked}),
            "waived": sorted({f["rule"] for f in waived}),
            "advisories": [f["rule"] for f in findings if f["severity"] == "advise"]}
'''),
    code('''
# --- Self-check: Section 2
check("the flawed manifest does not ship",
      lambda: gate(FLAWED)["pass"] is False)
check("and the gate says exactly which rules stopped it",
      lambda: gate(FLAWED)["blocked_by"]
              == ["capacity", "image-tag", "literal-secret", "probes", "resources"])
check("the fixed manifest ships",
      lambda: gate(FIXED)["pass"] is True)
check("an advisory never blocks",
      lambda: gate(FIXED)["advisories"] == ["hpa-signal"] and gate(FIXED)["pass"] is True)
check("a waiver is recorded, not silent",
      lambda: gate(FLAWED, waivers=("capacity",))["waived"] == ["capacity"])
check("waiving one rule does not ship a manifest that fails four others",
      lambda: gate(FLAWED, waivers=("capacity",))["pass"] is False,
      "the usual failure of a checklist is that one waiver becomes a blanket one")
check("waiving everything ships anything, which is why waivers need a name on them",
      lambda: gate(FLAWED, waivers=tuple(gate(FLAWED)["blocked_by"]))["pass"] is True)
'''),

    md("""
## Section 3 &mdash; The rule that is wrong

`rule_hpa_signal` is an advisory rather than a block, and it is the only rule here that is
about *this* workload rather than about deployments in general.

Autoscaling on CPU is the default because for most web services CPU is load: more requests, more
parsing, rendering and serialising, more CPU. An agent service does almost none of that. It sends
a request to a gateway and waits, and waiting consumes no CPU at all.
"""),
    code('''
def cpu_under_load(concurrent: int, call_seconds: float = 8.0,
                   cpu_seconds_per_request: float = 0.015) -> float:
    """CPU utilisation of one replica serving `concurrent` IO-bound agent requests.

    Each request spends call_seconds waiting on the gateway and cpu_seconds_per_request
    actually running code -- parsing JSON, building the prompt, formatting the answer.
    """
    busy = concurrent * cpu_seconds_per_request
    return 100.0 * busy / call_seconds


def hpa_would_scale(utilisation: float, target: int = 70) -> bool:
    return utilisation > target
'''),
    code('''
# --- Self-check: Section 3
check("one request in flight is invisible to the CPU metric",
      lambda: cpu_under_load(1) < 1)
check("forty concurrent requests are still under 10% CPU",
      lambda: cpu_under_load(40) < 10)
check("...so an HPA targeting 70% CPU does not scale",
      lambda: hpa_would_scale(cpu_under_load(40)) is False)
check("nor at a hundred and twenty",
      lambda: hpa_would_scale(cpu_under_load(120)) is False)
check("it finally crosses 70% at four hundred concurrent on one replica",
      lambda: hpa_would_scale(cpu_under_load(400)) is True,
      "a concurrency at which every request has been queueing for minutes -- the "
      "autoscaler fires long after the callers gave up")
check("the CPU metric only moves for work the agent does not do",
      lambda: hpa_would_scale(cpu_under_load(40, cpu_seconds_per_request=1.5)) is True)

def _cpu():
    print(f"  {'concurrent':>11} {'CPU %':>7} {'HPA scales?':>12}")
    for n in (1, 10, 40, 100, 400):
        u = cpu_under_load(n)
        print(f"  {n:>11} {u:>6.1f}% {str(hpa_would_scale(u)):>12}")
guard(_cpu)
'''),
    md("""
### Read it

The HPA is correctly configured, correctly deployed, and will never fire. Latency will go to
forty seconds and the dashboard will show a replica that is 4% busy.

Two things follow.

1. **The advisory is right and the rule cannot be a block**, because the same HPA is exactly
   right for a service that renders templates. A checklist encodes assumptions about the
   workload; this one names the assumption instead of hiding it.
2. **The signal has to be something that grows with load.** In-flight requests, queue depth, or
   time-to-first-token. Lab 9.5 picks one and tests it.

The starter manifest shipped with this module has this HPA in it, on purpose. It demonstrates the
object, and it is the wrong signal for the workload &mdash; which is a more useful thing for you
to have found here than to discover on a Monday.
"""),

    md("""
## Run it for real

Your linted manifest, sent to the real API server with `--dry-run=server`. That runs
authentication, RBAC, admission control, quota and schema validation, and changes nothing.
"""),
    code('''
import shutil, subprocess

def render(docs, namespace: str, host: str = "") -> str:
    """Write the objects to a JSON file kubectl can apply. Nothing is templated by hand."""
    out = []
    for d in docs:
        d = json.loads(json.dumps(d))                 # a copy; never mutate the source
        d.setdefault("metadata", {})["namespace"] = namespace
        if d["kind"] == "Ingress" and host:
            d["spec"]["rules"][0]["host"] = host
        out.append(d)
    path = os.path.join(WORK, "agent-app.json")
    with open(path, "w") as fh:
        json.dump({"apiVersion": "v1", "kind": "List", "items": out}, fh, indent=1)
    return path


def _dry_run():
    if not APP_NS:
        print("APP_NAMESPACE is not set, so there is nothing safe to point kubectl at.")
        print("In a sandbox terminal it is already exported; check with `env | grep APP_`.")
        print("Your namespace is your pod name without the trailing -0.")
        return
    if not shutil.which("kubectl"):
        print("kubectl is not on PATH in this kernel -- open a terminal in the sandbox instead.")
        return
    if not gate(FIXED)["pass"]:
        print("The gate says no. Fix the findings before deploying.")
        return
    path = render(FIXED, APP_NS, APP_HOST or f"{APP_NS}-app.example")
    print("wrote", path)
    r = subprocess.run(["kubectl", "apply", "-n", APP_NS, "--dry-run=server", "-f", path],
                       capture_output=True, text=True, timeout=60)
    print(r.stdout.strip() or r.stderr.strip()[:600])
    print("\\nNothing was created. To deploy for real, in a sandbox TERMINAL:")
    print(f"  kubectl apply -n {APP_NS} -f {path}")
    print(f"  kubectl get pods,svc,ingress,hpa -n {APP_NS}")

guard(_dry_run)
'''),
    md("""
### Read it

Whatever the dry run said, notice which failures it can find and which it cannot. It validates
the schema, your RBAC, the namespace quota and every admission webhook &mdash; and it says
nothing at all about whether your probes are the right way round, whether the image exists, or
whether the HPA will ever fire.

That is the division of labour: the API server checks that the object is legal, and your linter
checks that it is a good idea.

A full, commented starter manifest &mdash; the same objects with the Ingress, the Secret
references and the security context filled in &mdash; ships beside this notebook as
`app-deploy-example.yaml`. Use it for the capstone.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add the rule that catches the thing this lab did not: a `Secret` referenced by `envFrom` that
   does not exist in the namespace. Note that it cannot be checked offline, and decide where it
   belongs &mdash; the linter, the dry run, or readiness.
2. Write the waiver format. A waiver needs a rule, a reason, an owner and an expiry, or it is a
   permanent silence with a comment on it.
3. Run the linter over the starter manifest shipped with this module. It should pass. If it does
   not, one of you is wrong and it is worth finding out which.
"""),
]


# =========================================================================== #
# Lab 9.4 -- spans you can bill
# =========================================================================== #
LAB4 = [
    header(4, "Spans You Can Bill", "Advanced", 35,
           ["Price a call and put the number on the span",
            "Work out which questions your instrumentation can answer, and which it cannot",
            "Count the time series a label adds before you add it",
            "Do the arithmetic that shows head sampling loses the incident"],
           "> **Instrumentation is a decision made before the incident.** Every question in this\n"
           "> lab is answerable or not depending on an attribute somebody chose to record weeks\n"
           "> earlier, when nothing was wrong."),
    setup(4),

    md("""
## Concept

OpenTelemetry gives three signals and one vocabulary.

- A **trace** is one request, as a tree of **spans**. It answers *where did the time go, on this
  one*.
- A **metric** is a number over a window, cut by **labels**. It answers *how often, how bad,
  across all of them*.
- A **log** is an event with a timestamp. It answers *what exactly happened at 14:07*.

They compose: the trace ID goes in the log line, the span carries the attributes, the metric is
derived from the spans. An agent adds a fourth thing that none of the three pillars gives you
for free &mdash; **what it decided and what that cost** &mdash; and that is what this lab is about.
"""),

    md("""
## Section 1 &mdash; The number that has to be on the span

Cost is a per-request property. It cannot be recovered later from a monthly invoice, and it
cannot be divided by request count &mdash; the whole point is that requests differ.
"""),
    code('''
# USD per 1,000 tokens. Illustrative rates; the shape is what matters.
RATES = {
    "qwen-lab":  {"in": 0.0002, "out": 0.0006},
    "big-model": {"in": 0.0030, "out": 0.0150},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """What one model call cost."""
    r = RATES[model]
    # TODO: input and output tokens are priced differently, and the rates above are per
    # 1,000 tokens. Return the cost of this one call.
    return BLANK


def model_span(name: str, model: str, input_tokens: int, output_tokens: int,
               duration_s: float, **extra) -> dict:
    """One span in OpenTelemetry's shape, with the attributes an agent needs."""
    return {
        "name": name,
        "duration_s": duration_s,
        "attrs": {
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
            "app.cost_usd": cost_usd(model, input_tokens, output_tokens),
            **extra,
        },
    }
''', '''
# USD per 1,000 tokens. Illustrative rates; the shape is what matters.
RATES = {
    "qwen-lab":  {"in": 0.0002, "out": 0.0006},
    "big-model": {"in": 0.0030, "out": 0.0150},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """What one model call cost."""
    r = RATES[model]
    return input_tokens / 1000 * r["in"] + output_tokens / 1000 * r["out"]


def model_span(name: str, model: str, input_tokens: int, output_tokens: int,
               duration_s: float, **extra) -> dict:
    """One span in OpenTelemetry's shape, with the attributes an agent needs."""
    return {
        "name": name,
        "duration_s": duration_s,
        "attrs": {
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
            "app.cost_usd": cost_usd(model, input_tokens, output_tokens),
            **extra,
        },
    }
'''),
    code('''
# --- Self-check: Section 1
check("a thousand tokens each way on the lab model costs 0.0008",
      lambda: round(cost_usd("qwen-lab", 1000, 1000), 6) == 0.0008)
check("output tokens cost three times input on that model",
      lambda: round(cost_usd("qwen-lab", 0, 1000), 10)
              == round(3 * cost_usd("qwen-lab", 1000, 0), 10))
check("the same call on the big model costs 0.018",
      lambda: round(cost_usd("big-model", 1000, 1000), 6) == 0.018)
check("...which is 22.5x, and that ratio is a routing decision",
      lambda: round(cost_usd("big-model", 1000, 1000) / cost_usd("qwen-lab", 1000, 1000), 1)
              == 22.5)
check("a call with no tokens costs nothing",
      lambda: cost_usd("qwen-lab", 0, 0) == 0)
check("the span carries the cost as an attribute, not as a note in a log line",
      lambda: "app.cost_usd" in model_span("plan", "qwen-lab", 900, 120, 2.1)["attrs"])
check("...and the token counts too, so the cost can be re-derived when rates change",
      lambda: model_span("plan", "qwen-lab", 900, 120, 2.1)["attrs"]
              ["gen_ai.usage.input_tokens"] == 900,
      "prices change; recording only the dollar figure makes the history unusable")
'''),

    md("""
## Section 2 &mdash; What your instrumentation can answer

Here is one hour of a deployed service. Every request is a trace; the spans carry what somebody
decided to record.
"""),
    code('''
import random

def build_window(n: int = 240, seed: int = 9) -> list:
    """One hour of traffic. Deterministic, so everyone's numbers match."""
    rng = random.Random(seed)
    tenants = ["ops-emea", "ops-apac", "ops-us", "treasury", "client-desk"]
    out = []
    for i in range(n):
        model = "big-model" if rng.random() < 0.15 else "qwen-lab"
        pt, ct = rng.randint(600, 2400), rng.randint(80, 700)
        ok = rng.random() > 0.005
        out.append({
            "trace_id": f"{i:08x}",
            "tenant": rng.choice(tenants),
            "endpoint": rng.choice(["/ask", "/ask", "/ask", "/investigate"]),
            "payment_ref": f"PMT-{rng.randint(1000, 2200)}",
            "model": model,
            "input_tokens": pt,
            "output_tokens": ct,
            "cost_usd": cost_usd(model, pt, ct),
            "duration_s": round(rng.uniform(1.5, 12.0), 2),
            "ok": ok,
        })
    return out


# Built lazily: cost_usd is blanked above, so building at import time would crash the cell
# rather than report a [TODO].
_window = []
def window() -> list:
    if not _window:
        _window.extend(build_window())
    return _window


def spend_by(field: str) -> dict:
    """Total cost grouped by one recorded field."""
    out = {}
    for r in window():
        out[r[field]] = round(out.get(r[field], 0.0) + r["cost_usd"], 4)
    return out
'''),
    code('''
# What this instrumentation recorded -- and therefore what it can be asked.
RECORDED = {"tenant", "endpoint", "model", "input_tokens", "output_tokens",
            "cost_usd", "duration_s", "ok", "trace_id", "payment_ref"}

QUESTIONS = {
    "what did treasury spend this hour?":        {"tenant", "cost_usd"},
    "which model is the money going to?":        {"model", "cost_usd"},
    "how slow is the 95th percentile?":          {"duration_s"},
    "how often did a guardrail refuse?":         {"decision"},
    "did the answer cite a policy document?":    {"cited_policy"},
    "was the answer any good?":                  {"score"},
}

def can_answer(question: str) -> bool:
    """A question is answerable only if every attribute it needs was recorded."""
    return QUESTIONS[question] <= RECORDED
'''),
    code('''
# --- Self-check: Section 2
check("cost per tenant is answerable",
      lambda: can_answer("what did treasury spend this hour?"))
check("...and treasury is not the biggest spender",
      lambda: max(spend_by("tenant"), key=spend_by("tenant").get) != "treasury")
check("the model split is answerable",
      lambda: can_answer("which model is the money going to?"))
check("AN EIGHTH OF THE CALLS ARE THREE QUARTERS OF THE BILL",
      lambda: spend_by("model")["big-model"] > 2 * spend_by("model")["qwen-lab"],
      "a routing decision worth finding, and only visible because the model is on the span")
check("latency percentiles are answerable",
      lambda: can_answer("how slow is the 95th percentile?"))
check("but the refusal rate is NOT",
      lambda: not can_answer("how often did a guardrail refuse?"),
      "Module 8's control is invisible here -- nobody recorded the decision")
check("nor whether the answer cited anything",
      lambda: not can_answer("did the answer cite a policy document?"),
      "Module 6's grounding check, missing from Module 9's telemetry")
check("nor whether it was any good",
      lambda: not can_answer("was the answer any good?"))
check("three of six questions cannot be answered at any price",
      lambda: sum(1 for q in QUESTIONS if not can_answer(q)) == 3,
      "not slowly, not expensively -- the data does not exist")

def _pillars():
    s = spend_by("model")
    share = s["big-model"] / sum(s.values())
    n_big = sum(1 for r in window() if r["model"] == "big-model")
    print(f"  spend by model : {s}")
    print(f"  big-model      : {n_big}/{len(window())} calls, {share:.0%} of the spend")
    print(f"  spend by tenant: {spend_by('tenant')}")
    print(f"  total this hour: ${sum(s.values()):.2f}   "
          f"-> ${sum(s.values()) * 24 * 30:,.0f}/month at this rate")
    print()
    for q in QUESTIONS:
        print(f"  {'yes' if can_answer(q) else 'NO ':4} {q}")
guard(_pillars)
'''),
    md("""
### Read it

The three unanswerable questions are the agent-specific ones. Latency, cost and error rate come
free with any HTTP instrumentation; *did a guardrail fire*, *did the answer cite its source* and
*was it right* have to be recorded on purpose, by you, as span attributes.

That is the whole of the difference between observability for a web service and AgentOps. The
system can be perfectly healthy on all three pillars and be answering wrongly &mdash; and Module
8 closed with exactly that slide.
"""),

    md("""
## Section 3 &mdash; Labels multiply, so count before you add one

`payment_ref` is on the span and that is correct. Putting it on a **metric** is a different act
with a different cost, because every distinct value creates a time series that is stored,
indexed and queried forever.
"""),
    code('''
CARDINALITY = {"service": 1, "endpoint": 4, "status": 3, "model": 2,
               "tenant": 5, "payment_ref": 1200}

def series_count(labels) -> int:
    """How many time series does one metric with these labels produce?"""
    # TODO: labels do not add. Each one multiplies the number of series by the number
    # of distinct values it can take.
    return BLANK


def safe_to_label(labels, budget: int = 500) -> bool:
    """Would this label set stay inside the series budget for one metric?"""
    return series_count(labels) <= budget
''', '''
CARDINALITY = {"service": 1, "endpoint": 4, "status": 3, "model": 2,
               "tenant": 5, "payment_ref": 1200}

def series_count(labels) -> int:
    """How many time series does one metric with these labels produce?"""
    return math.prod(CARDINALITY[l] for l in labels)


def safe_to_label(labels, budget: int = 500) -> bool:
    """Would this label set stay inside the series budget for one metric?"""
    return series_count(labels) <= budget
'''),
    code('''
# --- Self-check: Section 3
BASE = ["service", "endpoint", "status"]

check("the base label set is twelve series",
      lambda: series_count(BASE) == 12)
check("adding the model doubles it",
      lambda: series_count(BASE + ["model"]) == 24)
check("adding the tenant is still fine",
      lambda: series_count(BASE + ["model", "tenant"]) == 120)
check("ADDING THE PAYMENT REFERENCE IS 144,000 SERIES",
      lambda: series_count(BASE + ["model", "tenant", "payment_ref"]) == 144000,
      "for one metric -- and payment_ref is unbounded, so that number only grows")
check("the budget check catches it",
      lambda: safe_to_label(BASE + ["model", "tenant"])
              and not safe_to_label(BASE + ["payment_ref"]))
check("the same field on a SPAN costs nothing extra",
      lambda: "payment_ref" in RECORDED,
      "spans are stored per request; labels are stored per distinct combination, forever")

def _labels():
    for extra in ([], ["model"], ["model", "tenant"], ["model", "tenant", "payment_ref"]):
        ls = BASE + extra
        print(f"  {series_count(ls):>7,} series  {'ok ' if safe_to_label(ls) else 'NO '}"
              f" {'+'.join(ls)}")
guard(_labels)
'''),

    md("""
## Section 4 &mdash; The trace you need is the one you did not keep

Traces are the expensive signal, so everybody samples. Head sampling &mdash; decide at the start
of the request, keep 10% &mdash; is the default because it is the cheapest to implement.

Do the arithmetic on it once and you will not use it for an agent service.
"""),
    code('''
DAILY_REQUESTS = 2000
FAILURE_RATE   = 1 / 200          # the thing you will be asked about

def expected_captured(p: float, requests: int = DAILY_REQUESTS,
                      failure_rate: float = FAILURE_RATE) -> float:
    """How many of the day's failures head sampling at rate p keeps."""
    return requests * failure_rate * p


def p_miss_everything(p: float, requests: int = DAILY_REQUESTS,
                      failure_rate: float = FAILURE_RATE) -> float:
    """The chance that a whole day of head sampling keeps NOT ONE failing trace."""
    failures = requests * failure_rate
    # TODO: each failure is kept independently with probability p. What is the chance
    # that every single one of them is dropped?
    return BLANK


def tail_kept_fraction(p_normal: float, failure_rate: float = FAILURE_RATE) -> float:
    """Tail sampling: decide when the request ENDS. Keep every failure, and p of the rest."""
    return failure_rate + p_normal * (1 - failure_rate)
''', '''
DAILY_REQUESTS = 2000
FAILURE_RATE   = 1 / 200          # the thing you will be asked about

def expected_captured(p: float, requests: int = DAILY_REQUESTS,
                      failure_rate: float = FAILURE_RATE) -> float:
    """How many of the day's failures head sampling at rate p keeps."""
    return requests * failure_rate * p


def p_miss_everything(p: float, requests: int = DAILY_REQUESTS,
                      failure_rate: float = FAILURE_RATE) -> float:
    """The chance that a whole day of head sampling keeps NOT ONE failing trace."""
    failures = requests * failure_rate
    return (1 - p) ** failures


def tail_kept_fraction(p_normal: float, failure_rate: float = FAILURE_RATE) -> float:
    """Tail sampling: decide when the request ENDS. Keep every failure, and p of the rest."""
    return failure_rate + p_normal * (1 - failure_rate)
'''),
    code('''
# --- Self-check: Section 4
check("there are ten failures in a day at this rate",
      lambda: DAILY_REQUESTS * FAILURE_RATE == 10)
check("head sampling at 10% expects to keep exactly one of them",
      lambda: expected_captured(0.10) == 1.0)
check("...and on 35% of days it keeps none at all",
      lambda: round(p_miss_everything(0.10), 4) == 0.3487,
      "one day in three, the trace the incident review asks for was never stored")
check("keeping everything misses nothing, and costs everything",
      lambda: p_miss_everything(1.0) == 0.0)
check("even 50% head sampling loses every failure on 1 day in 1000",
      lambda: round(p_miss_everything(0.50), 4) == 0.001)
check("TAIL SAMPLING AT 5% KEEPS EVERY FAILURE",
      lambda: tail_kept_fraction(0.05) > FAILURE_RATE,
      "the decision moves to the end of the request, when the outcome is known")
check("...while storing less than head sampling at 10%",
      lambda: tail_kept_fraction(0.05) < 0.10)
check("...about 45% less",
      lambda: round(1 - tail_kept_fraction(0.05) / 0.10, 2) == 0.45)

def _sampling():
    print(f"  {'strategy':28} {'stored':>8} {'failures kept':>14} {'blind days':>11}")
    for p in (0.01, 0.10, 0.50):
        print(f"  head sampling at {p:>4.0%}         {p:>7.1%} "
              f"{expected_captured(p):>13.1f} {p_miss_everything(p):>10.1%}")
    for p in (0.01, 0.05):
        f = tail_kept_fraction(p)
        print(f"  tail sampling, {p:>3.0%} of clean {f:>7.1%} "
              f"{DAILY_REQUESTS * FAILURE_RATE:>13.1f} {0.0:>10.1%}")
guard(_sampling)
'''),
    md("""
### Read it

Head sampling at 10% is the industry default and it loses the entire day's evidence about one
day in three. That is not a tail risk, it is a coin you flip every incident review.

Tail sampling costs more to run &mdash; the collector must buffer each trace until the request
finishes, which is why this decision belongs in the **collector** and not in your application.
That is the practical reason the OTLP collector exists between your process and your backend: it
is the one place where sampling, redaction and fan-out to several destinations can happen
without a redeploy of the service.

For an agent, extend the keep rule past errors: keep every trace that **refused**, every one that
**escalated to a human**, and every one in the slowest 1%. Those are the three that get asked
about, and none of them is an error.
"""),

    md("""
## Run it for real

Send one trace to LangFuse, with the cost attributes on it. Your sandbox is pointed at a shared
project and separated by environment, so you will see your own traces and not anybody else's.
"""),
    code('''
def send_trace():
    host = os.environ.get("LANGFUSE_HOST")
    if not (host and os.environ.get("LANGFUSE_PUBLIC_KEY")
            and os.environ.get("LANGFUSE_SECRET_KEY")):
        print("LangFuse is not configured here. To point at one, set LANGFUSE_HOST,")
        print("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY. Nothing above needed it.")
        return
    from langfuse import Langfuse
    lf = Langfuse()                     # reads LANGFUSE_HOST / _PUBLIC_KEY / _SECRET_KEY
    if not lf.auth_check():
        print("LangFuse credentials rejected.")
        return
    spans = [model_span("plan",     "qwen-lab",  900, 120, 2.1, step="plan"),
             model_span("retrieve", "qwen-lab",  240,  40, 0.4, step="retrieve"),
             model_span("answer",   "big-model", 2100, 480, 6.8, step="answer")]
    # SDK 4.x: observations nest by being entered inside one another. client.trace(...)
    # is the v3 API and does not exist here.
    with lf.start_as_current_observation(name="investigate-payment", as_type="span") as root:
        root.update(metadata={"app.cost_usd": round(sum(s["attrs"]["app.cost_usd"]
                                                        for s in spans), 6),
                              "tenant": "ops-emea", "payment_ref": "PMT-1003"})
        for s in spans:
            with lf.start_as_current_observation(name=s["name"], as_type="span") as obs:
                obs.update(metadata=s["attrs"])
    lf.flush()
    env = os.environ.get("LANGFUSE_TRACING_ENVIRONMENT", "(unset)")
    print(f"sent one trace to {host}")
    print(f"environment tag: {env}  -- filter on it in the UI to see only your own")
    print("Open the trace and check the cost is on the root as well as the leaves.")

guard(send_trace)
'''),
    md("""
### Read it

Note what separated your traces from everyone else's: an environment variable, read by the SDK,
with no code of yours involved. That is worth copying &mdash; per-tenant or per-environment
separation that depends on every call site remembering to pass a field is separation that lasts
until the first new call site.

The same trace, sent through an OTLP collector to a vendor backend, would carry the same
attributes under the same `gen_ai.*` names. That naming convention is why the choice of backend
is reversible and the choice of *what to record* is not.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Add `decision` and `cited_policy` to `RECORDED` and to `model_span`, then re-run Section 2.
   Three questions become answerable; work out what it would have cost to add them after the
   incident instead of before.
2. Cost is on the span. Write the aggregation that turns it into a per-tenant daily figure, and
   decide which of the two &mdash; span attribute or metric label &mdash; `tenant` should be.
   (You may want both, for different reasons.)
3. Design the keep rule for tail sampling in your own words: errors, refusals, escalations,
   slowest 1%. Then estimate the stored fraction using `tail_kept_fraction` with a failure rate
   that includes all four.
"""),
]


# =========================================================================== #
# Lab 9.5 -- challenge: the service that is up and wrong
# =========================================================================== #
LAB5 = [
    header(5, "Challenge: The Service That Is Up and Wrong", "Advanced &middot; challenge", 40,
           ["Find an incident in which every conventional signal is green",
            "Show why CPU autoscaling never fires, and pick a signal that does",
            "Write alarms that catch it, and prove they stay quiet on a good day",
            "Leave with the runbook page for an agentic service"],
           "> **The last lab of the course.** It uses Module 6's citations, Module 7's measurements\n"
           "> and Module 8's controls, and asks the Module 9 question about all three: how would\n"
           "> you know, at 09:15, from a dashboard?"),
    setup(5),

    md("""
## Concept

An ordinary service fails by erroring or by slowing down, and both are visible in the four golden
signals &mdash; latency, traffic, errors, saturation. An agent service has a third failure mode:
it answers every request, quickly, with a 200, and the answers are wrong.

Nothing in the golden signals moves. Something else does, and only if you recorded it.
"""),

    md("""
## Section 1 &mdash; Two days that look identical

Here is yesterday and today. Same traffic, same code, one deploy in between.
"""),
    code('''
import random

def build_day(seed: int, refusal_rate: float, citation_rate: float,
              error_rate: float = 0.02, n: int = 2000) -> list:
    """One day of requests. The same seed gives the same latencies, costs and errors,
    so any difference between two days below is a difference in BEHAVIOUR, not noise."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        r = rng.random()
        decision = ("refused" if r < refusal_rate
                    else "escalated" if r < refusal_rate + 0.04
                    else "answered")
        out.append({
            "ok": rng.random() > error_rate,
            "duration_s": round(rng.uniform(1.5, 12.0), 2),
            "cited": rng.random() < citation_rate,
            "cost_usd": round(rng.uniform(0.0008, 0.0032), 5),
            "decision": decision,
        })
    return out


YESTERDAY = build_day(11, refusal_rate=0.08, citation_rate=0.92)
TODAY     = build_day(11, refusal_rate=0.01, citation_rate=0.55)


def pct(values, p):
    """The p-th percentile, nearest-rank. Stdlib, and exact enough for a dashboard."""
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


def metrics(day: list) -> dict:
    n = len(day)
    return {
        "requests":        n,
        "error_rate":      round(sum(1 for r in day if not r["ok"]) / n, 4),
        "p95_latency_s":   pct([r["duration_s"] for r in day], 95),
        "cost_per_req":    round(sum(r["cost_usd"] for r in day) / n, 5),
        "refusal_rate":    round(sum(1 for r in day if r["decision"] == "refused") / n, 4),
        "escalation_rate": round(sum(1 for r in day if r["decision"] == "escalated") / n, 4),
        "citation_rate":   round(sum(1 for r in day if r["cited"]) / n, 4),
    }


GOLDEN = ("error_rate", "p95_latency_s", "cost_per_req", "requests")
AGENTIC = ("refusal_rate", "escalation_rate", "citation_rate")

print("yesterday:", metrics(YESTERDAY))
print("today    :", metrics(TODAY))
'''),
    code('''
def moved(before: float, after: float, tolerance: float = 0.25) -> bool:
    """Did this metric change materially between the two windows?"""
    if before == 0:
        return after != 0
    # TODO: a control that STOPS WORKING makes its own metric go down, not up. A test
    # written as `after > before * (1 + tolerance)` finds nothing today. Compare the
    # size of the change, whichever way it went.
    return BLANK


def what_changed(before: dict, after: dict, tolerance: float = 0.25) -> list:
    """Every metric that moved, in the order they are defined."""
    return [k for k in before if moved(before[k], after[k], tolerance)]
''', '''
def moved(before: float, after: float, tolerance: float = 0.25) -> bool:
    """Did this metric change materially between the two windows?"""
    if before == 0:
        return after != 0
    return abs(after - before) / before > tolerance


def what_changed(before: dict, after: dict, tolerance: float = 0.25) -> list:
    """Every metric that moved, in the order they are defined."""
    return [k for k in before if moved(before[k], after[k], tolerance)]
'''),
    code('''
# --- Self-check: Section 1
Y, T = metrics(YESTERDAY), metrics(TODAY)

check("traffic is identical",
      lambda: Y["requests"] == T["requests"])
check("the error rate did not move",
      lambda: not moved(Y["error_rate"], T["error_rate"]))
check("p95 latency did not move",
      lambda: not moved(Y["p95_latency_s"], T["p95_latency_s"]))
check("cost per request did not move",
      lambda: not moved(Y["cost_per_req"], T["cost_per_req"]))
check("NOT ONE OF THE FOUR GOLDEN SIGNALS MOVED",
      lambda: not any(moved(Y[k], T[k]) for k in GOLDEN),
      "every dashboard the team owns is green")
check("the refusal rate collapsed",
      lambda: moved(Y["refusal_rate"], T["refusal_rate"]))
check("...downwards, which is why a one-sided alarm never fired",
      lambda: T["refusal_rate"] < Y["refusal_rate"])
check("the citation rate fell too",
      lambda: moved(Y["citation_rate"], T["citation_rate"]))
check("exactly the agent-specific metrics moved, and only those",
      lambda: set(what_changed(Y, T)) == {"refusal_rate", "citation_rate"})
check("a one-sided test finds nothing at all",
      lambda: [k for k in Y if T[k] > Y[k] * 1.25] == [],
      "which is how this runs for three weeks")

def _diff():
    print(f"  {'metric':18} {'yesterday':>10} {'today':>10}   moved?")
    for k in Y:
        flag = "  <-- MOVED" if moved(Y[k], T[k]) else ""
        print(f"  {k:18} {Y[k]:>10} {T[k]:>10}{flag}")
guard(_diff)
'''),
    md("""
### What happened

A deploy changed a prompt. The guardrail that used to hold sanctions cases for a human now
answers most of them, and the retriever's grounding check stopped rejecting ungrounded answers.

The service is up. It is fast. It costs the same. It answers every request with a 200, and one
payment in twelve that should have gone to a human did not.

This is the failure mode Module 8 closed on, seen from the dashboard, and the reason those two
metrics have to exist as **first-class signals with alarms on them**, next to latency and errors
rather than in a weekly report.
"""),

    md("""
## Section 2 &mdash; The signal that actually moves with load

The second half of the incident: at 09:15 the same service went from four concurrent requests to
sixty, and the HPA did nothing at all.
"""),
    code('''
CALL_SECONDS   = 8.0      # one agent request, mostly spent waiting on the gateway.
                          # Measured on this sandbox: 7.5-10s for a one-line answer.
CPU_PER_REQ    = 0.015    # the CPU it actually uses
PER_REPLICA    = 8        # concurrent requests one replica serves without queueing
TARGET_UTIL    = 0.70

def cpu_percent(in_flight: int) -> float:
    """CPU utilisation of the fleet's replicas at this concurrency."""
    return 100.0 * in_flight * CPU_PER_REQ / CALL_SECONDS


def replicas_from_cpu(in_flight: int, current: int = 1, target: int = 70) -> int:
    """What an HPA on CPU utilisation asks for. This is the shipped default."""
    util = cpu_percent(in_flight) / current
    return max(1, math.ceil(current * util / target))


def replicas_from_inflight(in_flight: int) -> int:
    """What an HPA on in-flight requests asks for."""
    # TODO: each replica serves PER_REPLICA concurrent requests, and you do not want them
    # running at 100% -- aim for TARGET_UTIL of that. Return the replica count, never
    # fewer than one.
    return BLANK


def latency_at(in_flight: int, replicas: int) -> float:
    """Wall clock per request once the queue forms. Crude, and the right shape."""
    capacity = replicas * PER_REPLICA
    return CALL_SECONDS * math.ceil(max(1, in_flight) / capacity)
''', '''
CALL_SECONDS   = 8.0      # one agent request, mostly spent waiting on the gateway.
                          # Measured on this sandbox: 7.5-10s for a one-line answer.
CPU_PER_REQ    = 0.015    # the CPU it actually uses
PER_REPLICA    = 8        # concurrent requests one replica serves without queueing
TARGET_UTIL    = 0.70

def cpu_percent(in_flight: int) -> float:
    """CPU utilisation of the fleet's replicas at this concurrency."""
    return 100.0 * in_flight * CPU_PER_REQ / CALL_SECONDS


def replicas_from_cpu(in_flight: int, current: int = 1, target: int = 70) -> int:
    """What an HPA on CPU utilisation asks for. This is the shipped default."""
    util = cpu_percent(in_flight) / current
    return max(1, math.ceil(current * util / target))


def replicas_from_inflight(in_flight: int) -> int:
    """What an HPA on in-flight requests asks for."""
    return max(1, math.ceil(in_flight / (PER_REPLICA * TARGET_UTIL)))


def latency_at(in_flight: int, replicas: int) -> float:
    """Wall clock per request once the queue forms. Crude, and the right shape."""
    capacity = replicas * PER_REPLICA
    return CALL_SECONDS * math.ceil(max(1, in_flight) / capacity)
'''),
    code('''
# --- Self-check: Section 2
check("at four in flight one replica is right, and both rules agree",
      lambda: replicas_from_cpu(4) == 1 and replicas_from_inflight(4) == 1)
check("at sixty in flight the CPU rule still asks for one",
      lambda: replicas_from_cpu(60) == 1)
check("...because the fleet is under 16% busy while it queues",
      lambda: cpu_percent(60) < 16)
check("THE IN-FLIGHT RULE ASKS FOR ELEVEN",
      lambda: replicas_from_inflight(60) == 11)
check("one replica at sixty in flight is a 64-second request",
      lambda: latency_at(60, 1) == 64.0)
check("eleven replicas bring it back to one call time",
      lambda: latency_at(60, replicas_from_inflight(60)) == CALL_SECONDS)
check("the CPU rule leaves latency 8x worse than the in-flight rule",
      lambda: latency_at(60, replicas_from_cpu(60))
              == 8 * latency_at(60, replicas_from_inflight(60)))
check("neither rule scales down below one replica",
      lambda: replicas_from_cpu(0) == 1 and replicas_from_inflight(0) == 1)
check("the in-flight rule is bounded by maxReplicas, which is a budget decision",
      lambda: min(replicas_from_inflight(400), 3) == 3,
      "at 400 in flight it wants 72; your quota says 3, so the answer is a queue and a 429")

def _scaling():
    print(f"  {'in flight':>10} {'CPU %':>7} {'cpu rule':>9} {'inflight rule':>14} "
          f"{'latency (cpu)':>14} {'latency (inflight)':>19}")
    for n in (4, 12, 30, 60, 120):
        rc, ri = replicas_from_cpu(n), replicas_from_inflight(n)
        print(f"  {n:>10} {cpu_percent(n):>6.1f}% {rc:>9} {ri:>14} "
              f"{latency_at(n, rc):>13.0f}s {latency_at(n, ri):>18.0f}s")
guard(_scaling)
'''),
    md("""
### Read it

The HPA in the starter manifest &mdash; and in most agent deployments &mdash; scales on CPU at
70%. On this workload it reaches 16% at sixty concurrent requests, so it never fires, and the
replica sitting at 16% busy is serving 64-second requests.

The fix is not a lower CPU target. It is a **different signal**: in-flight requests, queue depth,
or time-to-first-token, exported by your own service and scraped as a custom metric. All three
grow with load because all three are about waiting, which is what this service does.

And note the last check. Scaling has a ceiling that is a budget, not a technical limit. Past it,
the correct behaviour is to shed load with a `429` and a `Retry-After`, not to accept a request
you will answer in four minutes.
"""),

    md("""
## Section 3 &mdash; Alarms that would have caught it

An alarm has two jobs, and the second one is why most alarms get switched off: fire on the
incident, and stay quiet on every good day.
"""),
    code('''
def alarm_golden_signals(before: dict, after: dict) -> bool:
    """The alarms the team already has. Provided so you can see them not fire."""
    return any(moved(before[k], after[k]) for k in GOLDEN)


def alarm_control_drift(before: dict, after: dict) -> bool:
    """A control's own metric moved. Either direction, because down is the dangerous one."""
    return any(moved(before[k], after[k]) for k in AGENTIC)


def alarm_saturation(in_flight: int, replicas: int) -> bool:
    """Fires while requests are queueing, whatever the CPU says."""
    # TODO: the fleet can serve replicas x PER_REPLICA concurrent requests. Fire when
    # in-flight is above the share of that you are willing to run at (TARGET_UTIL).
    return BLANK
''', '''
def alarm_golden_signals(before: dict, after: dict) -> bool:
    """The alarms the team already has. Provided so you can see them not fire."""
    return any(moved(before[k], after[k]) for k in GOLDEN)


def alarm_control_drift(before: dict, after: dict) -> bool:
    """A control's own metric moved. Either direction, because down is the dangerous one."""
    return any(moved(before[k], after[k]) for k in AGENTIC)


def alarm_saturation(in_flight: int, replicas: int) -> bool:
    """Fires while requests are queueing, whatever the CPU says."""
    return in_flight > replicas * PER_REPLICA * TARGET_UTIL
'''),
    code('''
# --- Self-check: Section 3
QUIET_DAY = metrics(build_day(12, refusal_rate=0.08, citation_rate=0.92))

check("the alarms the team already has do not fire on the incident",
      lambda: alarm_golden_signals(Y, T) is False,
      "this is not a criticism of them -- they are measuring something else")
check("THE CONTROL-DRIFT ALARM FIRES",
      lambda: alarm_control_drift(Y, T) is True)
check("...and stays quiet comparing two ordinary days",
      lambda: alarm_control_drift(Y, QUIET_DAY) is False,
      "an alarm that fires on a good day is an alarm somebody mutes")
check("the golden-signal alarms are also quiet on a good day",
      lambda: alarm_golden_signals(Y, QUIET_DAY) is False)
check("the saturation alarm fires at sixty in flight on one replica",
      lambda: alarm_saturation(60, 1) is True)
check("...and not once it has scaled out",
      lambda: alarm_saturation(60, replicas_from_inflight(60)) is False)
check("it fires before latency doubles, not after",
      lambda: alarm_saturation(9, 1) is True and latency_at(9, 1) == 2 * CALL_SECONDS)
check("a CPU alarm at 70% is silent at every concurrency worth alarming on",
      lambda: all(cpu_percent(n) < 70 for n in (10, 60, 120, 200)),
      "it crosses 70% only near 400 in flight, where a request already takes 400 seconds")

def _alarms():
    for label, pair in (("incident (yesterday -> today)", (Y, T)),
                        ("ordinary day vs ordinary day",  (Y, QUIET_DAY))):
        print(f"  {label:32} golden={str(alarm_golden_signals(*pair)):5} "
              f"control_drift={alarm_control_drift(*pair)}")
    print()
    for n, reps in ((4, 1), (60, 1), (60, 11)):
        print(f"  {n:>3} in flight on {reps:>2} replica(s): saturation="
              f"{str(alarm_saturation(n, reps)):5} latency={latency_at(n, reps):.0f}s")
guard(_alarms)
'''),

    md("""
## The runbook page

Everything above is one page of an on-call runbook. Yours will differ; the shape will not.

**Page on**

| Signal | Threshold | Because |
|---|---|---|
| 5xx rate | above 1% for 5 min | the ordinary one; keep it |
| p95 latency | above 3&times; the baseline | the ordinary one; keep it |
| in-flight per replica | above 70% of capacity | CPU will not tell you this |
| refusal / escalation rate | moved &plusmn;25% vs the last 7 days | **a control stopped working** |
| citation rate | moved &plusmn;25% | grounding stopped working |
| cost per request | above 2&times; the baseline | a retry loop, or a routing change |

**First three things to do**

1. **Read one trace, not the logs.** Find a slow or wrong request by trace ID and look at the
   span tree. Which hop grew, and did it grow in count or in duration?
2. **Compare the last deploy.** A prompt is a deploy. So is a model version change made by
   somebody else on the gateway you depend on.
3. **Check the dependency before restarting anything.** Lab 9.2's whole point: restarting a
   healthy process because a remote gateway blinked makes the outage longer.

**Do not**

- Do not raise the CPU target to make the HPA fire. It is the wrong signal, not a mistuned one.
- Do not turn off the control that is alarming. Its metric moving is the alarm.
- Do not conclude anything from a green dashboard. Today's incident had one.
"""),

    md("""
## Run it for real

Your own numbers, from the sandbox gateway. Six sequential calls, then the replica count the
in-flight rule would ask for at sixty concurrent users on *your* measured latency.
"""),
    code('''
if llm_ready():
    def _budget():
        lat = []
        for i in range(6):
            t0 = time.perf_counter()
            ask(f"In one sentence, what is a payment exception? (v{i})")
            lat.append(time.perf_counter() - t0)
        p95 = pct(lat, 95)
        per_replica = max(1, int(PER_REPLICA))
        want = math.ceil(60 / (per_replica * TARGET_UTIL))
        print(f"  measured  : mean {sum(lat) / len(lat):.1f}s, p95 {p95:.1f}s over 6 calls")
        print(f"  at 60 concurrent users the in-flight rule asks for {want} replicas")
        print(f"  the CPU rule asks for {replicas_from_cpu(60)}")
        print(f"  and one replica would answer in about {latency_at(60, 1):.0f}s")
        print("\\n  Six calls is not a latency distribution. It is enough to know which")
        print("  order of magnitude you are budgeting in, which is the decision here.")
    guard(_budget)
'''),

    code('''
score()
'''),
    md("""
## Your turn

1. The control-drift alarm compares two windows. Write the version that compares today with a
   **trailing seven-day median**, and work out what it does on the Monday after a long weekend.
2. Add a `429` path to Lab 9.1's `handle`: shed load when in-flight is above capacity, with a
   `Retry-After`. Then decide which is worse for your callers &mdash; a 429 now, or a 200 in
   four minutes.
3. Take one control from Module 8 &mdash; the approval gate, the contract, the detector &mdash;
   and write the metric that proves it is still running. If you cannot, that control is
   unmonitored, and Section 1 is what that looks like on the day it stops.

**What you take from Module 9:** a service boundary that returns its failures, probes that answer
two different questions, a readiness checklist that executes, spans that carry cost and decisions,
and the two signals &mdash; control drift and saturation &mdash; that an agent needs and a web
service does not.

That is the last lab. The capstone puts all nine modules behind one endpoint.
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-9-01-the-service-boundary",        LAB1),
    ("lab-9-02-probes-that-can-fail",        LAB2),
    ("lab-9-03-the-manifest-is-the-deploy",  LAB3),
    ("lab-9-04-spans-you-can-bill",          LAB4),
    ("lab-9-05-challenge-up-and-wrong",      LAB5),
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
