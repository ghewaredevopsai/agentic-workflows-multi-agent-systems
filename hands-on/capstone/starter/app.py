"""Your capstone service. A skeleton that RUNS -- and fails the gate.

Start here, not from a blank file. This version already serves the three endpoints and
already returns a contract-valid response, so you can score it in the first five minutes:

    uvicorn app:app --port 8000                 # in one terminal
    python3 ../acceptance.py --url http://127.0.0.1:8000 --limit 5

You will get NOT ACCEPTED with every number at zero. That is the point: you now have a
loop -- run, read the scorecard, change one thing, run again -- instead of a demo you
hope works at the end of the afternoon.

WHAT IS ALREADY HERE
  * the response contract, so the harness can talk to you from the start
  * the MCP client, pointed at ledger_mcp.py (Lab 4.4, made persistent)
  * cost accounting from tokens (Lab 9.4 -- the gateway reports tokens and no money)
  * the two probes, answering different questions (Lab 9.2)

WHAT YOU WRITE
  * retrieval over the policy corpus                          (Module 6)
  * the specialists and the supervisor that routes between them (Module 5)
  * the decision, grounded in what you retrieved                (Module 6)
  * a critic                                                    (Module 5)
  * the approval gate                                           (Module 8)

ONE MEASURED WARNING BEFORE YOU START. The lab model reasons before it answers, and the
reasoning is billed as completion tokens: a one-line JSON reply costs ~250 of them. If you
cap max_tokens at 200 to save money you do not get a truncated answer, you get
`content = None` and finish_reason='length'. Budget ~1500 and handle None.
"""
import json, os, re, sys, time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from domain import (LEDGER, POLICY, WATCHLIST_NOTE, NEEDS_HUMAN, SANCTIONS_WATCH,   # noqa
                    RECOMMENDATIONS, NEEDS_APPROVAL)
from mcp_client import MCPClient, MCPError                                          # noqa

BASE  = os.environ.get("LAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
MODEL = os.environ.get("LAB_LLM_MODEL")    or os.environ.get("OPENAI_MODEL")

# AsyncOpenAI, not OpenAI -- Lab 9.1 measured what a synchronous client does to a worker
# serving four callers at once, and the acceptance harness uses four by default.
client = AsyncOpenAI(base_url=BASE, api_key=os.environ.get("OPENAI_API_KEY", "sandbox"),
                     timeout=90.0, max_retries=1)

RATES = {"default": {"in": 0.0002, "out": 0.0006}}      # USD per 1,000 tokens


def cost_usd(model, input_tokens, output_tokens):
    r = RATES.get(model, RATES["default"])
    return input_tokens / 1000 * r["in"] + output_tokens / 1000 * r["out"]


class Usage:
    """Tokens and money for one request. Record BOTH: rates change, history should not."""
    def __init__(self):
        self.input_tokens = self.output_tokens = self.calls = 0
        self.cost = 0.0

    def add(self, response):
        u = response.usage
        self.input_tokens += u.prompt_tokens
        self.output_tokens += u.completion_tokens
        self.cost += cost_usd(MODEL, u.prompt_tokens, u.completion_tokens)
        self.calls += 1

    def as_dict(self):
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost, 6), "model_calls": self.calls}


_mcp = None


def mcp():
    """One long-lived MCP server for the process."""
    global _mcp
    if _mcp is None:
        _mcp = MCPClient(cwd=_HERE)
    return _mcp


# ===========================================================================
# TODO 1 -- retrieval over the policy corpus                        (Module 6)
#
# There is no embedding model on this gateway, so this is lexical retrieval, which is
# what Lab 6.1 built. Return the top k with their scores: a caller that cannot see the
# score cannot judge adequacy, and Lab 6.3 is about exactly that.
# ===========================================================================
DOCS = [{"id": code, "text": f"{code}. {text}"} for code, text in POLICY.items()]
DOCS.append({"id": "SANCTIONS_SCREENING", "text": WATCHLIST_NOTE})


def retrieve(query, k=3):
    raise NotImplementedError("TODO 1: return [{'id':..., 'text':..., 'score':...}, ...]")


# ===========================================================================
# TODO 2 -- one model call that must return an object              (Modules 2, 8)
#
# Return None rather than guessing. A service that coerces a bad reply into a decision
# is Lab 8.2's coercing validator, one layer out.
# ===========================================================================
async def ask_json(system, prompt, usage, retries=1):
    raise NotImplementedError("TODO 2: call the model, parse an object, or return None")


# ===========================================================================
# TODO 3 -- the specialists                                        (Modules 4, 5, 6)
#
# facts   : the ledger record and the screening result, both over MCP
# policy  : what to retrieve, and note the query is built from the CASE, not from the
#           user's words -- Lab 6.2
# decide  : the recommendation, grounded in the two above
# critic  : a second opinion that may overturn the first -- Lab 5.3
#
# Read the eval set before you write these. Seven of the forty-five cases turn on a rule
# that is not in the reason code, and an agent that never screens the counterparty gets
# 84% and fails the approval gate on every one of them.
# ===========================================================================
async def agent_facts(ref, trace):
    raise NotImplementedError("TODO 3a: lookup_payment and screen_counterparty over MCP")


def agent_policy(record, screening, trace):
    raise NotImplementedError("TODO 3b: build the query from the case, then retrieve")


async def agent_decide(record, screening, docs, question, usage, trace):
    raise NotImplementedError("TODO 3c: the recommendation, with citations")


async def agent_critic(record, screening, decision, usage, trace):
    raise NotImplementedError("TODO 3d: a second opinion")


# ===========================================================================
# the service
# ===========================================================================
app = FastAPI(title="payment-exception investigation")


class Ask(BaseModel):
    ref: str = Field(min_length=3, max_length=32)
    question: str = Field(default="what should we do with it?", min_length=1, max_length=2000)


@app.get("/healthz")
def healthz():
    """Liveness. Local only: restarting this process will not fix the gateway."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness. A STATUS CODE, not a body -- Lab 9.2."""
    if BASE and MODEL:
        return JSONResponse({"ready": True, "model": MODEL})
    return JSONResponse({"ready": False, "why": "LLM env not set"}, status_code=503)


@app.post("/investigate")
async def investigate(body: Ask):
    """One case. Returns the contract the acceptance harness validates.

    It answers "unknown" for everything until you write the agents above, which is a
    contract-valid, honest, and completely useless service. Score it anyway -- a failing
    number you can watch move beats a demo you hope works.
    """
    t0 = time.perf_counter()
    usage, trace = Usage(), ["supervisor"]

    recommendation, reason, citations = "unknown", "not implemented yet", []
    try:
        # ===================================================================
        # TODO 4 -- the supervisor. Route between your specialists, then apply
        # THE GATE (Module 8): anything in NEEDS_APPROVAL must set
        # requires_approval and must leave actions_taken empty. That criterion
        # is checked at 100%, not as a rate -- it is a control, not a score.
        # ===================================================================
        pass
    except Exception as exc:
        # A service returns its failures. Lab 9.1: 5xx means WE broke, and a caller
        # cannot act on a stack trace.
        reason = f"{type(exc).__name__}: {str(exc)[:160]}"

    requires_approval = recommendation in NEEDS_APPROVAL
    return {"ref": body.ref, "recommendation": recommendation, "reason": reason,
            "citations": citations, "requires_approval": requires_approval,
            "actions_taken": [], "usage": usage.as_dict(), "trajectory": trace,
            "latency_s": round(time.perf_counter() - t0, 3)}
