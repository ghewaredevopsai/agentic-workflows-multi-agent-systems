"""A working capstone service. READ THIS AFTER YOU HAVE BUILT YOURS.

It exists so the acceptance gate can be shown to discriminate -- a gate nobody has ever
passed is a wish, and a gate nobody can fail is decoration. It is also the trainer's demo.
It is not the only shape that passes, and it is not the best one: the "Where this is weak"
notes at the foot are real.

Run locally:   uvicorn app:app --port 8000        (from this directory)
Deploy:        see ../../module-9/app-deploy-example.yaml
Score it:      python3 ../acceptance.py --url http://127.0.0.1:8000
"""
import asyncio, json, os, re, sys, time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# Two layouts have to work: this repo (app.py here, the rest in ../starter) and the
# container, where the ConfigMap flattens all four files into /app.
_HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = _HERE if os.path.exists(os.path.join(_HERE, "ledger_mcp.py")) \
    else os.path.abspath(os.path.join(_HERE, "..", "starter"))
sys.path.insert(0, CODE_DIR)
from domain import (LEDGER, POLICY, WATCHLIST_NOTE, NEEDS_HUMAN,   # noqa: E402
                    RECOMMENDATIONS, NEEDS_APPROVAL)
from mcp_client import MCPClient, MCPError                          # noqa: E402

BASE  = os.environ.get("LAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
MODEL = os.environ.get("LAB_LLM_MODEL")    or os.environ.get("OPENAI_MODEL")

# AsyncOpenAI, not OpenAI. Lab 9.1 measured what the synchronous client does to a worker
# serving four callers: the answers are identical and the fourth one waits for the first three.
client = AsyncOpenAI(base_url=BASE, api_key=os.environ.get("OPENAI_API_KEY", "sandbox"),
                     timeout=90.0, max_retries=1)

# ---------------------------------------------------------------- cost, per Lab 9.4
# The gateway reports TOKENS and no money, so cost is derived here. That is the whole
# reason Lab 9.4 insisted on recording the token counts alongside the dollar figure: the
# rate is a stand-in for a price sheet and will change, and the history has to survive it.
RATES = {"default": {"in": 0.0002, "out": 0.0006}}       # USD per 1,000 tokens


def cost_usd(model, input_tokens, output_tokens):
    r = RATES.get(model, RATES["default"])
    return input_tokens / 1000 * r["in"] + output_tokens / 1000 * r["out"]


# ---------------------------------------------------------------- retrieval, per Module 6
# There is no embedding model on this gateway, so retrieval is lexical -- which is what
# Lab 6.1 built and, at a corpus of seven documents, is not the weak link.
DOCS = [{"id": code, "text": f"{code}. {text}"} for code, text in POLICY.items()]
DOCS.append({"id": "SANCTIONS_SCREENING", "text": WATCHLIST_NOTE})


def tokenise(s):
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if len(w) > 2]


def similarity(query, doc):
    q, d = set(tokenise(query)), set(tokenise(doc))
    return len(q & d) / len(q | d) if q | d else 0.0


def retrieve(query, k=3):
    """Top-k, and the scores come back with them so a caller can judge adequacy."""
    scored = sorted(((similarity(query, d["text"]), d) for d in DOCS),
                    key=lambda p: -p[0])[:k]
    return [{"id": d["id"], "text": d["text"], "score": round(s, 3)} for s, d in scored]


# ---------------------------------------------------------------- the ledger, over MCP
_mcp = None
_mcp_lock = asyncio.Lock()


async def mcp():
    """One long-lived server for the process. Spawning an interpreter per tool call is a
    latency budget nobody has."""
    global _mcp
    if _mcp is None:
        async with _mcp_lock:
            if _mcp is None:
                _mcp = MCPClient(cwd=CODE_DIR)
    return _mcp


# ---------------------------------------------------------------- the model call
class Usage:
    """Tokens and money for one request, accumulated across every call it makes."""
    def __init__(self):
        self.input_tokens = self.output_tokens = 0
        self.cost = 0.0
        self.calls = 0

    def add(self, response):
        u = response.usage
        self.input_tokens += u.prompt_tokens
        self.output_tokens += u.completion_tokens
        self.cost += cost_usd(MODEL, u.prompt_tokens, u.completion_tokens)
        self.calls += 1

    def as_dict(self):
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost, 6), "model_calls": self.calls}


# Thinking on or off. Measured on this gateway, for one trivial JSON reply:
#
#   thinking on   24.1s   980 completion tokens (955 of them reasoning)
#   thinking off   0.7s    29 completion tokens (0 reasoning)
#
# Same answer, 34x the latency and 34x the tokens. `reasoning_effort="low"` -- the
# OpenAI-standard knob -- is silently ignored by this backend and produced 1,430 tokens,
# MORE than the default; `/no_think` in the prompt half-works. The one that works is
# vendor-specific and goes through extra_body.
#
# Turning it off is not free: see "Where this is weak". Measure it on YOUR eval set.
THINKING = os.environ.get("LAB_LLM_THINKING", "off").lower() == "on"
NO_THINK = {} if THINKING else {"chat_template_kwargs": {"enable_thinking": False}}


async def ask_json(system, prompt, usage, retries=1):
    """One model call that must return an object. Returns None rather than guessing.

    NOTE THE max_tokens. With thinking on, this model spends its completion budget
    reasoning before it answers: a trivial JSON reply costs ~250 tokens and can cost
    1,400. Cap it at 200 to save money and the reply is not truncated -- `content` comes
    back as None with finish_reason='length', and you get no answer at all.
    """
    for attempt in range(retries + 1):
        try:
            r = await client.chat.completions.create(
                model=MODEL, max_tokens=1500, temperature=0, extra_body=NO_THINK,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}])
        except Exception:
            continue
        usage.add(r)
        text = r.choices[0].message.content
        if not text:
            continue                       # reasoning consumed the budget; try once more
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ---------------------------------------------------------------- the agents
async def agent_facts(ref, trace):
    """Ledger and screening, both over MCP. No model: these are lookups with a schema."""
    c = await mcp()
    trace.append("ledger")
    try:
        record = json.loads(c.call("lookup_payment", ref=ref))
    except MCPError:
        return None, None
    trace.append("sanctions")
    screening = json.loads(c.call("screen_counterparty",
                                  counterparty=record["counterparty"]))
    return record, screening


def agent_policy(record, screening, trace):
    """Agentic RAG: the query is built from the case, not from the user's words."""
    trace.append("policy")
    query = " ".join(filter(None, [record.get("reason_code") or "",
                                   record.get("status", ""),
                                   "sanctions screening watchlist"
                                   if screening.get("listed") else ""]))
    return retrieve(query or "payment exception", k=3)


# THREE MEASURED REVISIONS LIVE IN THIS PROMPT, and each one is a lab in miniature.
#
#   1. The rules as prose, thinking off            -> 46.7% accuracy
#   2. The same rules as a numbered procedure      -> 86.7%
#   3. The checks stated against NAMED FACT FIELDS,
#      each with the citation it implies           -> see the calibration numbers
#
# Revision 2 also collapsed grounding to 16.1%, and the cause was one careless token:
# the reply template said `"citations": ["POLICY_CODE", ...]` and eleven cases dutifully
# cited a document called POLICY_CODE. A placeholder in an example IS an instruction.
#
# The model is told to decide from the FACTS and to use the retrieved policy for citation
# and explanation. That sounds backwards for a RAG system and it is deliberate: the facts
# are exact and the retrieval is fuzzy, so letting retrieval decide the outcome was how
# PMT-1003 -- a LIMIT_BREACH on a counterparty nobody has ever screened -- came back as
# "proceed" citing SANCTIONS_SCREENING.
DECIDE_SYSTEM = (
    "You are a payments operations analyst. Decide what happens to ONE payment.\n\n"
    "Decide from the FACTS and the SCREENING RESULT. The policy extracts are there so you "
    "can cite and explain -- do not search them for the answer.\n\n"
    "Work through these checks IN ORDER and STOP at the first one that is true:\n"
    '  C1  facts.status is "settled"\n'
    '        -> recommendation "no action",        citations []\n'
    "  C2  screening.listed is true\n"
    '        -> recommendation "hold for a human", citations ["SANCTIONS_SCREENING"]\n'
    '  C3  facts.reason_code is "LIMIT_BREACH" or "SANCTIONS_REVIEW"\n'
    '        -> recommendation "hold for a human", citations [that reason code]\n'
    "  C4  otherwise\n"
    '        -> recommendation "proceed",          citations [facts.reason_code]\n\n'
    "C1 is first because a settled payment needs nothing even when its counterparty is "
    "listed. C2 is before C3 because a listed counterparty is held even when the reason "
    "code is routine -- screening does not depend on the reason code.\n\n"
    "Reply with ONLY a JSON object and nothing else:\n"
    '{"check": "C1"|"C2"|"C3"|"C4", "recommendation": "...", "reason": "one sentence", '
    '"citations": [...]}\n'
    "Put real document ids in citations, exactly as the checks above specify.")


async def agent_decide(record, screening, docs, question, usage, trace):
    trace.append("decide")
    prompt = (f"Question: {question}\n\n"
              f"Payment facts (from the ledger):\n{json.dumps(record, indent=1)}\n\n"
              f"Sanctions screening result:\n{json.dumps(screening, indent=1)}\n\n"
              "Policy extracts:\n"
              + "\n".join(f"- [{d['id']}] {d['text']}" for d in docs))
    return await ask_json(DECIDE_SYSTEM, prompt, usage)


CRITIC_SYSTEM = (
    "You review a payments decision for ONE failure: an analyst who read the reason code "
    "and forgot that a counterparty on the sanctions screening list is held regardless "
    "of it.\n"
    "Reply with ONLY a JSON object:\n"
    '{"agree": true|false, "why": "one sentence"}\n'
    "Answer agree:false ONLY when the screening result says listed: true AND the payment "
    "is not settled AND the proposed recommendation is not \"hold for a human\". "
    "In every other case answer agree:true.")


async def agent_critic(record, screening, decision, usage, trace):
    trace.append("critic")
    prompt = (f"Payment: {json.dumps(record)}\n"
              f"Screening: {json.dumps(screening)}\n"
              f"Proposed decision: {json.dumps(decision)}")
    return await ask_json(CRITIC_SYSTEM, prompt, usage)


# ---------------------------------------------------------------- tracing, per Module 9
def tracer():
    """LangFuse if it is configured, and a no-op object if it is not. A service that
    only works with its observability stack wired up is a service you cannot run locally."""
    if not (os.environ.get("LANGFUSE_HOST") and os.environ.get("LANGFUSE_PUBLIC_KEY")):
        return None
    try:
        from langfuse import Langfuse
        return Langfuse()
    except Exception:
        return None


LF = tracer()


class span:
    """`with span("name", **attrs):` -- a LangFuse observation, or nothing at all."""
    def __init__(self, name, **attrs):
        self.name, self.attrs, self.cm, self.obs = name, attrs, None, None

    def __enter__(self):
        if LF is not None:
            self.cm = LF.start_as_current_observation(name=self.name, as_type="span")
            self.obs = self.cm.__enter__()
        return self

    def update(self, **attrs):
        self.attrs.update(attrs)

    def __exit__(self, *exc):
        if self.cm is not None:
            try:
                self.obs.update(metadata=self.attrs)
            except Exception:
                pass
            self.cm.__exit__(*exc)
        return False


# ---------------------------------------------------------------- the service
app = FastAPI(title="payment-exception investigation")


class Ask(BaseModel):
    ref: str = Field(min_length=3, max_length=32)
    question: str = Field(default="what should we do with it?", min_length=1, max_length=2000)


@app.get("/healthz")
def healthz():
    """Liveness. Local only -- Lab 9.2: restarting this process will not fix the gateway."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness. A status code, not a body -- and cheap, so it is not a load generator."""
    if BASE and MODEL:
        return JSONResponse({"ready": True, "model": MODEL})
    return JSONResponse({"ready": False, "why": "LLM env not set"}, status_code=503)


@app.post("/investigate")
async def investigate(body: Ask):
    t0 = time.perf_counter()
    usage, trace = Usage(), ["supervisor"]
    with span("investigate", ref=body.ref, question=body.question) as root:
        record, screening = await agent_facts(body.ref, trace)

        # Not in the ledger. Say so and stop -- Module 6's lesson, and five of the
        # forty-five cases are exactly this.
        if record is None:
            out = {"ref": body.ref, "recommendation": "unknown",
                   "reason": f"no payment found with reference {body.ref!r}",
                   "citations": [], "requires_approval": False, "actions_taken": []}
        else:
            docs = agent_policy(record, screening, trace)
            decision = await agent_decide(record, screening, docs, body.question, usage, trace)

            # The model produced nothing usable. Do not guess and do not fall back to a
            # rule -- send it to a person. This costs accuracy on the easy cases, which
            # is the correct direction to be wrong in.
            if decision is None:
                decision = {"recommendation": "hold for a human",
                            "reason": "the analyst step produced no usable answer",
                            "citations": []}
            elif decision.get("recommendation") in ("proceed", "no action"):
                # The critic runs only where being wrong is expensive -- on a decision to
                # act. It is not asked on a hold, because escalating an escalation buys
                # nothing and Lab 7.4 costs it.
                verdict = await agent_critic(record, screening, decision, usage, trace)
                if verdict and verdict.get("agree") is False:
                    # AND IT MAY ONLY ESCALATE. A critic that can turn a hold into a
                    # proceed is a second opinion with the safety property removed --
                    # Module 8's point, in the shape of a code path.
                    decision["recommendation"] = "hold for a human"
                    decision["reason"] = (str(decision.get("reason", "")) + " Critic: "
                                          + str(verdict.get("why", ""))[:160]).strip()

            rec = decision.get("recommendation")
            if rec not in RECOMMENDATIONS:
                rec = "hold for a human"

            # THE GATE. Module 8: a control, not a measurement. Nothing that needs a
            # person is actioned here, whatever the model recommended and however
            # confident it sounded.
            needs_human = rec in NEEDS_APPROVAL
            out = {"ref": body.ref, "recommendation": rec,
                   "reason": str(decision.get("reason", ""))[:400] or "no reason given",
                   "citations": [str(c) for c in (decision.get("citations") or [])][:6],
                   "requires_approval": needs_human,
                   "actions_taken": []}      # this service proposes; it never writes

        out["usage"] = usage.as_dict()
        out["trajectory"] = trace
        out["latency_s"] = round(time.perf_counter() - t0, 3)
        root.update(recommendation=out["recommendation"], **out["usage"])
    return out


# ---------------------------------------------------------------- Where this is weak
#
# Calibrated 2026-09-04: 93.3% accuracy, a clean approval gate, 100% grounding, $0.0002
# per case and p95 3.4s at four concurrent callers. Four consecutive runs, identical.
# ACCEPTED -- and here is everything wrong with it.
#
# 1. IT MISSES EVERY SETTLED-AND-LISTED CASE. The three wrong answers are not three
#    unlucky ones: they are ALL THREE of the payments that are settled AND whose
#    counterparty is on the screening list, and it gets all three wrong the same way.
#    C1 says a settled payment needs nothing and C2 says a listed counterparty is held;
#    the prompt says C1 comes first and explains why, and the model applies C2 anyway.
#    A systematic rule-ordering failure looks exactly like random noise in an aggregate,
#    and looks like what it is the moment you read three traces. Lab 7.4, on this file.
#    It is at least wrong in the safe direction: it over-holds a settled payment.
#
# 2. THINKING ON IS MORE ACCURATE AND FAILS THE GATE. With reasoning enabled it scored
#    15/15 on the first fifteen cases, against 14/15 for this configuration -- a
#    one-case difference, which Lab 7.1 says establishes nothing -- and its p95 was
#    65.9s against a 30s ceiling. Being better is not the same as being shippable, and
#    a gate with only a quality criterion would have taken the wrong one.
#
# 3. THE OPERATING PROCEDURE IS IN THE PROMPT, not derived from the retrieved documents.
#    That is why it passes and it is the least interesting thing about it: a decision
#    table with a language model in front. The harder version reads the policy corpus
#    and works out the ordering itself, and the eval set does not care which you do --
#    which is the point of scoring outcomes rather than implementations.
#
# 4. `actions_taken` is always empty because this service has no write tool at all. That
#    makes the approval gate trivially safe and also makes it untested: the interesting
#    version has a write tool the gate stands in front of.
#
# 5. The critic is prompted for one specific failure -- the watchlist one. Honest
#    engineering for a known weakness, and also overfitting to this eval set. Note that
#    it does not catch failure 1, because it is only asked about decisions to ACT.
#
# 6. Retrieval is lexical over seven documents, so it always retrieves the right thing.
#    Nothing here would survive a corpus of seven hundred, and Lab 6.3's adequacy check
#    is missing entirely.
#
# 7. There is no cache. The same reference asked twice costs twice.
