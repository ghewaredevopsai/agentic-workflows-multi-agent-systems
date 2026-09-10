#!/usr/bin/env python3
"""
Generate Module 6 lab notebooks and their solutions from one source.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-6-0N-*.ipynb and ../solutions/

Design rules:
  * The participant writes REAL LangChain code in every lab -- an Embeddings model, a
    text splitter, a Chroma collection, a retriever, an LCEL chain. The framework is
    the learning, not an optional appendix.
  * Self-checks assert on framework OBJECTS -- a Document, a collection, a retriever,
    a chain, a formatted context -- which is deterministic and needs no gateway. Only
    CHAT-model invocation needs the gateway, and that lives in "Run it for real" cells.
    The EMBEDDING model is not the chat model: it runs on this pod's CPU, costs nothing
    and is billed to nobody, so a graded cell may freely embed and search.
  * A graded cell may build a @tool, an Embeddings, a splitter, a Chroma collection or
    a Pydantic schema -- but NOT a chat model. get_llm() raises ValidationError when
    LAB_LLM_* is unset, which is exactly the state the verifiers run in.
  * Blanks ask a DESIGN DECISION -- which method, which chunk size, which filter, which
    refusal clause. Where the answer is a Python idiom (a comprehension, a set operation,
    an f-string) the code is given and the question moves to what only understanding
    answers.
  * "BLANK" marks a blank; an unfilled blank raises NameError and prints [TODO].
    NOT three underscores: IPython PREDEFINES _, __ and ___ as its output history
    (they start as ""), so under a real Jupyter kernel that token is a defined empty
    string, not an undefined name. The NameError never fires, [TODO] silently becomes
    [FAIL], and a blank used as a loop guard never stops its loop.
  * Blanks live INSIDE function bodies, and every framework object built at module level
    is built lazily or wrapped in guard(), so an untouched lab survives Run All.

Ported from the flagship session-5 labs (2026-09-10). Two substitutions were forced and
both are real:
  * Groq -> the sandbox gateway. ChatOpenAI against LAB_LLM_BASE_URL / LAB_LLM_MODEL,
    i.e. qwen through LiteLLM. No key to register, no vendor to sign up with.
  * HuggingFaceEmbeddings -> a ~10-line Embeddings adapter over chromadb's
    ONNXMiniLM_L6_V2. Same model (all-MiniLM-L6-v2) and same vectors, reached through
    onnxruntime rather than torch: ~170 MB per kernel instead of ~840, against a 2.5 GB
    sandbox. The cache is pre-warmed in every sandbox PVC.
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
def header(num, title, level, minutes, bullets, note=""):
    items = "\n".join("- " + b for b in bullets)
    return md(f"""
# Lab 6.{num} &mdash; {title}

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 2 &middot; Module 6 &mdash; Agentic RAG**

### What you'll do
{items}

> **How this lab works.** You write real LangChain code. Fill every `BLANK`, then run the
> **Self-check** cell under each section &mdash; those check the *objects you built* (a
> `Document`, a Chroma collection, a retriever, a chain), so they are deterministic and do not
> depend on the chat model. Cells marked **Run it for real** put your code in front of the
> sandbox model; that is the part worth watching. The score line is feedback, not a grade.

> **Two different models are in play, and only one of them is billed.** The **chat model**
> (`qwen36-35b-a3b-lab`) answers questions and is reached over the gateway. The **embedding
> model** (`all-MiniLM-L6-v2`, 384 dimensions) turns text into vectors and runs on this pod's
> own CPU &mdash; no key, no gateway, no tokens. Keeping them straight is most of Module 6.

{note}
""")


SETUP_COMMON = '''
# ---------------------------------------------------------------- Setup: run me first
import os, json, time, math, textwrap, warnings
from typing import Any, Callable

warnings.filterwarnings("ignore")     # sentence-transformers is chatty on first import

WORK = os.path.join("/tmp", "awmas-lab-6-{num:02d}")
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

# ---- the CHAT model: qwen, through the sandbox gateway -------------------
# Already configured -- nothing to install, no key to register. Read from the
# environment so this notebook never hardcodes an endpoint.
LLM_BASE_URL = (os.environ.get("LAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("LITELLM_BASE_URL"))
LLM_MODEL    = (os.environ.get("LAB_LLM_MODEL") or os.environ.get("OPENAI_MODEL")
                or os.environ.get("LITELLM_MODEL"))
LLM_API_KEY  = os.environ.get("OPENAI_API_KEY", "sandbox")

# The served model reasons before it answers, and the reasoning is billed as completion
# tokens. Thinking is off by default here because you will make a lot of calls today;
# pass think=True to any call below to see the difference for yourself.
NO_THINK = {{"chat_template_kwargs": {{"enable_thinking": False}}}}

def llm_ready() -> bool:
    if not LLM_BASE_URL or not LLM_MODEL:
        print("Chat model not configured. In a sandbox terminal run `env | grep -i llm` and set:")
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

# ---- the EMBEDDING model: local, free, nothing to configure --------------
# all-MiniLM-L6-v2, 384 dimensions. It runs on this pod's CPU and has nothing to do with
# the chat model above: no gateway, no key, no tokens billed. The cache is already warm
# in your sandbox, so the first call is a second or two, not a download.
#
# It is reached through onnxruntime rather than torch, and that is a measured choice
# rather than a taste: same model, same vectors, ~170 MB of memory instead of ~840. Your
# whole sandbox has 2.5 GB for every notebook you leave open, and a kernel you have
# forgotten about is still holding its share.
from langchain_core.embeddings import Embeddings

class MiniLMEmbeddings(Embeddings):
    """all-MiniLM-L6-v2 behind LangChain's Embeddings interface.

    Two methods is the whole contract -- which is why a store, a splitter and a chain
    never need to know which model is underneath, or what runtime it uses."""

    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        self._fn = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: list) -> list:
        return [[float(x) for x in v] for v in self._fn(list(texts))]

    def embed_query(self, text: str) -> list:
        return [float(x) for x in self._fn([text])[0]]


_emb_cache = {{}}
def get_embeddings():
    """The embedding model, built once per kernel."""
    if "model" not in _emb_cache:
        _emb_cache["model"] = MiniLMEmbeddings()
    return _emb_cache["model"]

print("work dir   :", WORK)
print("chat model :", LLM_MODEL or "(not configured -- the object-level self-checks still work)")
'''


def setup(num, extra=""):
    return code(SETUP_COMMON.format(num=num) + extra)


# --------------------------------------------------------------------------- #
# the shared corpus -- a company employee handbook
# --------------------------------------------------------------------------- #
# Module 6 runs its own domain, the way each Day 1 module does. A handbook is the right
# corpus for RAG teaching because it has natural METADATA -- a category, a source file and
# a page -- and filtering on those is half of what a production retriever does. The
# LEDGER / POLICY payment exceptions of Modules 4, 5, 7-9 and the capstone are untouched.

CORPUS_CELL = code('''
# ------------------------------------------------- the corpus (synthetic, self-contained)
# Ten short passages from a company handbook. Note what each one carries besides its text:
# a category, a source file and a page. Those three are what Lab 6.3 filters on and what
# Lab 6.7 cites -- metadata is not decoration, it is the half of retrieval that is exact.
#
# Note also what is NOT here: nothing mentions salary, notice period or the share price.
# Labs 6.6 and 6.8 need that gap, because refusing is a feature.

HANDBOOK = [
    {"text": "Annual leave is 24 days per year for full-time employees. Leave must be applied "
             "for at least 3 working days in advance. Unused annual leave cannot be carried "
             "forward to the next financial year.",
     "category": "leave", "source": "handbook.pdf", "page": 5},
    {"text": "Sick leave is 12 days per year. Notify your manager by 10 AM on the day of "
             "absence. A medical certificate is required for absences of more than 2 "
             "consecutive days.",
     "category": "leave", "source": "handbook.pdf", "page": 5},
    {"text": "Maternity leave is 26 weeks of paid leave. Paternity leave is 2 weeks. Both "
             "must be applied for at least 30 days before the expected date.",
     "category": "leave", "source": "handbook.pdf", "page": 6},
    {"text": "Employees may work from home up to 3 days per week with team lead approval. "
             "Core hours are 10 AM to 4 PM IST, and you must be reachable during them.",
     "category": "wfh", "source": "handbook.pdf", "page": 8},
    {"text": "A VPN connection is mandatory for reaching internal systems from home. "
             "Contact the IT helpdesk for VPN setup.",
     "category": "wfh", "source": "handbook.pdf", "page": 8},
    {"text": "Internet reimbursement is 1,500 per month for employees working from home. "
             "Submit the broadband bill to finance by the 5th of each month.",
     "category": "expense", "source": "handbook.pdf", "page": 9},
    {"text": "Travel expenses must be submitted with original receipts within 7 working days "
             "of travel. The meal allowance during client visits is 500 per day.",
     "category": "expense", "source": "handbook.pdf", "page": 12},
    {"text": "Laptops are provided by the company and replaced every 3 years. Software "
             "licence requests go through the IT helpdesk and must not be bought directly.",
     "category": "tech", "source": "tech-guide.pdf", "page": 7},
    {"text": "The backend stack is Python with FastAPI, and Java with Spring Boot. New "
             "services should use Python unless there is a specific reason not to. "
             "PostgreSQL is the primary database.",
     "category": "tech", "source": "tech-guide.pdf", "page": 3},
    {"text": "The Bangalore office is the headquarters, on the 5th floor, with 200+ staff. "
             "The Mumbai office is in the Worli business district, Tower A, 12th floor.",
     "category": "office", "source": "office-directory.pdf", "page": 15},
]

print(f"{len(HANDBOOK)} passages, "
      f"{len({d['category'] for d in HANDBOOK})} categories, "
      f"{len({d['source'] for d in HANDBOOK})} source files")
''')


def docs_cell():
    """Turn the corpus into LangChain Documents. Given, not blanked -- it is a dict copy."""
    return code('''
# ------------------------------------------------- the corpus as LangChain Documents
from langchain_core.documents import Document

def handbook_documents() -> list:
    """One Document per passage: the text, and everything else as metadata."""
    return [Document(page_content=d["text"],
                     metadata={"category": d["category"],
                               "source": d["source"],
                               "page": d["page"]})
            for d in HANDBOOK]

print(len(handbook_documents()), "Document objects")
''')


# =========================================================================== #
# Lab 6.1 -- Hello Qwen: the sandbox chat model
# =========================================================================== #
LAB1 = [
    header(1, "Hello Qwen &mdash; the Sandbox Model", "Beginner", 15, [
        "Send your first call to the sandbox model and read what comes back",
        "Decide what belongs in the <strong>system</strong> role and what belongs in the <strong>human</strong> role",
        "Choose the two settings that decide whether today costs you cents or dollars",
        "Watch the model's own reasoning appear in the token bill &mdash; and switch it off",
    ], "> **Nothing to install and no key to register.** The model is already wired into this\n"
       "> sandbox. If a live cell says it is not configured, run `env | grep -i llm` in a\n"
       "> terminal and export the two values it names."),

    setup(1),

    md("""
## Concept

Every call in this module is the same three things:

| | |
|---|---|
| **a system message** | the standing instruction &mdash; who the model is, what shape the answer takes |
| **a human message** | the thing you actually want to know, which changes every call |
| **settings** | `temperature`, and whether the model reasons before answering |

The chat model is reached over the sandbox gateway and every token is billed against your
own daily budget. The embedding model you meet in Lab 6.2 is not: it runs here, on this pod.
"""),

    md("""
## Section 1 &mdash; Two roles, one call

The instruction and the question go in different places. Put the question in the system
message and it becomes part of the model's standing character &mdash; which is exactly the
bug you get when a prompt template is built by string concatenation.
"""),

    code('''
from langchain_core.messages import SystemMessage, HumanMessage

INSTRUCTION = "You are a concise assistant. Answer in at most two sentences."

def build_messages(question: str) -> list:
    """One standing instruction, one question. Which content goes in which role?"""
    return [
        SystemMessage(content=BLANK),
        HumanMessage(content=BLANK),
    ]
''', '''
from langchain_core.messages import SystemMessage, HumanMessage

INSTRUCTION = "You are a concise assistant. Answer in at most two sentences."

def build_messages(question: str) -> list:
    """One standing instruction, one question. Which content goes in which role?"""
    return [
        SystemMessage(content=INSTRUCTION),
        HumanMessage(content=question),
    ]
'''),

    code('''
# --- Self-check: Section 1   (message objects only -- no call yet)
Q = "What is retrieval-augmented generation?"

check("build_messages returns exactly two messages",
      lambda: len(build_messages(Q)) == 2)
check("the first is a SystemMessage and the second a HumanMessage",
      lambda: isinstance(build_messages(Q)[0], SystemMessage)
              and isinstance(build_messages(Q)[1], HumanMessage))
check("the standing instruction is in the SYSTEM message",
      lambda: build_messages(Q)[0].content == INSTRUCTION)
check("the question is in the HUMAN message, and only there",
      lambda: build_messages(Q)[1].content == Q and Q not in build_messages(Q)[0].content,
      "a question baked into the system message becomes part of every later turn")
'''),

    code('''
# --- Run it for real -------------------------------------------------------
def first_call():
    llm = get_llm()
    reply = llm.invoke(build_messages("What is retrieval-augmented generation?"))
    print(reply.content.strip())
    print("\\ntokens:", reply.usage_metadata)

if llm_ready():
    guard(first_call)
'''),

    md("""
## Section 2 &mdash; The two settings that decide the bill

`temperature` decides how much the model varies between identical calls. Reasoning
(&ldquo;thinking&rdquo;) decides how much it writes before it answers &mdash; and that
preamble is billed as completion tokens exactly like the answer is.

Both are decisions, not defaults. You are about to run a few hundred calls today.
"""),

    code('''
def grading_temperature() -> float:
    """You are about to compare two runs of the same question and score the difference.
    Which temperature makes that comparison mean anything?"""
    return BLANK

def thinking_for_labs() -> bool:
    """Reasoning is billed as completion tokens. Across a day of small factual lookups
    against a handbook, should the labs leave it on?"""
    return BLANK

def body_for(think: bool) -> dict:
    """The extra_body the gateway wants. Given -- it is a dict shape, not a decision."""
    return {"chat_template_kwargs": {"enable_thinking": think}}
''', '''
def grading_temperature() -> float:
    """You are about to compare two runs of the same question and score the difference.
    Which temperature makes that comparison mean anything?"""
    return 0.0

def thinking_for_labs() -> bool:
    """Reasoning is billed as completion tokens. Across a day of small factual lookups
    against a handbook, should the labs leave it on?"""
    return False

def body_for(think: bool) -> dict:
    """The extra_body the gateway wants. Given -- it is a dict shape, not a decision."""
    return {"chat_template_kwargs": {"enable_thinking": think}}
'''),

    code('''
# --- Self-check: Section 2   (values and dict shapes -- still no call)
check("a run you intend to compare is deterministic",
      lambda: grading_temperature() == 0.0,
      "anything above 0 and the difference you measure might just be sampling")
check("thinking is OFF for the labs",
      lambda: thinking_for_labs() is False,
      "it is the right default for a few hundred handbook lookups, not a universal one")
check("body_for builds what the gateway expects",
      lambda: body_for(False) == {"chat_template_kwargs": {"enable_thinking": False}})
check("and it matches the NO_THINK the setup cell already defined",
      lambda: body_for(thinking_for_labs()) == NO_THINK)
'''),

    code('''
# --- Run it for real -------------------------------------------------------
# The same question twice: once with reasoning off, once with it on. Watch the
# completion-token count, not the answer.
def thinking_costs_what():
    q = "A payment of 900,000 needs approval. Who approves it? Answer in one line."
    for think in (False, True):
        t0 = time.time()
        reply = get_llm(temperature=grading_temperature(), think=think).invoke(q)
        u = reply.usage_metadata or {}
        print(f"thinking={str(think):5}  completion={u.get('output_tokens', '?'):>5}  "
              f"total={u.get('total_tokens', '?'):>5}  {time.time() - t0:.1f}s")

if llm_ready():
    guard(thinking_costs_what)
'''),

    code('''
score()
'''),

    md("""
## Your turn

1. Move the instruction into the human message instead, joined to the question with a
   newline. Does the answer change? Now ask a *second* question on the same model object
   and notice what the system message would have done that this does not.
2. Run the same question five times at `temperature=0.8` and five times at `0.0`. Count the
   distinct answers. That number is why every measurement in this course pins temperature.
3. `reply.usage_metadata` also carries `input_tokens`. Add a long system message and watch
   which half of the bill grows. That is the number Module 6 is about to make you pay
   attention to, because retrieved context lands in exactly that half.
"""),
]


# =========================================================================== #
# Lab 6.2 -- Understanding embeddings
# =========================================================================== #
LAB2 = [
    header(2, "Understanding Embeddings", "Beginner", 25, [
        "Turn a sentence into 384 numbers and look at them",
        "Score two texts by meaning with cosine similarity, and predict the answer first",
        "Decide which of the two <code>Embeddings</code> methods a query needs, and which a corpus needs",
        "Build a search engine in fifteen lines &mdash; which is all a vector store does",
    ], "> **This lab makes no gateway calls at all.** Everything here runs on the pod's own\n"
       "> CPU, costs nothing, and would work with the network unplugged."),

    setup(2),

    md("""
## Concept

An **embedding** is a position. The model reads a piece of text and returns a point in a
384-dimensional space, chosen so that text meaning similar things lands nearby.

Two texts are then compared by the *angle* between their vectors &mdash; **cosine
similarity**, which runs from 1.0 (same direction) down through 0 (unrelated). That is
the whole mechanism. A vector store is this, plus an index so you do not have to score
every document one at a time.

`Embeddings` has exactly two methods, and the difference matters:

| | |
|---|---|
| `embed_query(text)` | one string &rarr; one vector. For the thing being searched *with* |
| `embed_documents(texts)` | a list &rarr; a list of vectors. For the things being searched *over* |
"""),

    md("""
## Section 1 &mdash; One sentence, 384 numbers

The first call loads the model. It is already cached in your sandbox, so this is a second
or two rather than a download.
"""),

    code('''
def embed_one(text: str) -> list:
    """A single search string. Which of the two methods is that?"""
    emb = get_embeddings()
    method = BLANK
    return method(text)


def cosine(a: list, b: list) -> float:
    """Given -- this is arithmetic, not a design decision."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
''', '''
def embed_one(text: str) -> list:
    """A single search string. Which of the two methods is that?"""
    emb = get_embeddings()
    method = emb.embed_query
    return method(text)


def cosine(a: list, b: list) -> float:
    """Given -- this is arithmetic, not a design decision."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
'''),

    code('''
# --- Self-check: Section 1   (the embedding model runs locally -- no gateway)
check("embed_one returns a vector, not a list of vectors",
      lambda: isinstance(embed_one("annual leave")[0], float),
      "embed_documents would give you a list of lists here")
check("the vector has 384 dimensions",
      lambda: len(embed_one("annual leave")) == 384)
check("the same text always gives the same vector",
      lambda: embed_one("annual leave") == embed_one("annual leave"),
      "embeddings are deterministic -- unlike the chat model, nothing is sampled")
check("a text about something else lands somewhere else",
      lambda: cosine(embed_one("annual leave"), embed_one("annual leave")) >
              cosine(embed_one("annual leave"), embed_one("the Mumbai office")))

def _look():
    v = embed_one("What is the refund policy?")
    print("first 8 of 384:", [round(x, 4) for x in v[:8]])
guard(_look)
'''),

    md("""
## Section 2 &mdash; Predict, then measure

Three phrasings of a question about getting money back, and one about the weather. Write
down which pair you expect to score highest *before* you run it &mdash; the point of the
exercise is the gap between the guess and the number.
"""),

    code('''
PAIRS = {
    "refund/return":  ("What is the refund policy?", "How do I return a product?"),
    "refund/weather": ("What is the refund policy?", "What is the weather in Mumbai?"),
    "refund/money":   ("What is the refund policy?", "How do I get my money back?"),
}

def most_similar_pair() -> str:
    """Which of the three keys above do you expect to score highest? No shared words
    is allowed to win -- that is the whole claim embeddings make."""
    return BLANK

def score_pair(name: str) -> float:
    """Given."""
    a, b = PAIRS[name]
    return cosine(embed_one(a), embed_one(b))
''', '''
PAIRS = {
    "refund/return":  ("What is the refund policy?", "How do I return a product?"),
    "refund/weather": ("What is the refund policy?", "What is the weather in Mumbai?"),
    "refund/money":   ("What is the refund policy?", "How do I get my money back?"),
}

def most_similar_pair() -> str:
    """Which of the three keys above do you expect to score highest? No shared words
    is allowed to win -- that is the whole claim embeddings make."""
    return "refund/money"

def score_pair(name: str) -> float:
    """Given."""
    a, b = PAIRS[name]
    return cosine(embed_one(a), embed_one(b))
'''),

    code('''
# --- Self-check: Section 2
check("your prediction names one of the three pairs",
      lambda: most_similar_pair() in PAIRS)
check("and the measurement agrees with it",
      lambda: max(PAIRS, key=score_pair) == most_similar_pair(),
      "run the cell below, read the three numbers, and change your answer")
check("the unrelated pair scores lowest",
      lambda: min(PAIRS, key=score_pair) == "refund/weather")

def _table():
    for name in PAIRS:
        a, b = PAIRS[name]
        print(f"  {score_pair(name):.4f}  {name:16} {a!r} vs {b!r}")
guard(_table)
'''),

    code('''
def embed_corpus(texts: list) -> list:
    """Many documents at once. Which method, and why is it not the same one?"""
    emb = get_embeddings()
    method = BLANK
    return method(texts)


def rank(query: str, texts: list, vectors: list) -> list:
    """Score every document against the query, best first. Given -- a sort is a sort."""
    qv = embed_one(query)
    return sorted(((cosine(qv, v), t) for t, v in zip(texts, vectors)), reverse=True)
''', '''
def embed_corpus(texts: list) -> list:
    """Many documents at once. Which method, and why is it not the same one?"""
    emb = get_embeddings()
    method = emb.embed_documents
    return method(texts)


def rank(query: str, texts: list, vectors: list) -> list:
    """Score every document against the query, best first. Given -- a sort is a sort."""
    qv = embed_one(query)
    return sorted(((cosine(qv, v), t) for t, v in zip(texts, vectors)), reverse=True)
'''),

    code('''
# --- Self-check: the fifteen-line search engine
DOCS = [
    "Refund within 30 days of purchase",
    "Free shipping on orders above 500",
    "Contact support on the internal helpdesk",
    "Return items in their original packaging",
]

def _vecs():
    return embed_corpus(DOCS)

check("embed_corpus returns one vector per document",
      lambda: len(_vecs()) == len(DOCS))
check("each of them is 384 long",
      lambda: all(len(v) == 384 for v in _vecs()))
check("'how do I get my money back' finds the refund line",
      lambda: rank("How do I get my money back?", DOCS, _vecs())[0][1].startswith("Refund"),
      "not one word of the query appears in that document")
check("and the shipping line is not the top hit for it",
      lambda: "shipping" not in rank("How do I get my money back?", DOCS, _vecs())[0][1])

def _search():
    for s, t in rank("How do I get my money back?", DOCS, _vecs()):
        print(f"  [{s:.4f}] {t}")
guard(_search)
'''),

    code('''
score()
'''),

    md("""
## Your turn

1. Add `"Refunds are processed within 5-7 business days"` to `DOCS` and search again. Two
   documents now deserve to be returned. Does the ranking put them 1 and 2? This is the
   first hint of why `k` is a decision and not a constant.
2. Score `"Python is a programming language"` against `"Python is a snake"`. The number is
   higher than you want it to be. Embeddings capture *topic* at least as much as meaning,
   and that is a failure mode you will meet in production.
3. Swap `cosine` for a plain dot product and re-run Section 2. The ranking barely moves
   here &mdash; because these vectors are already close to unit length. On a corpus of very
   uneven document lengths it moves a lot.
"""),
]


# =========================================================================== #
# Lab 6.3 -- ChromaDB basics
# =========================================================================== #
LAB3 = [
    header(3, "ChromaDB Basics", "Intermediate", 25, [
        "Create a Chroma collection and add documents that carry metadata",
        "Decide what metadata to attach &mdash; before you need it, because you cannot filter on what you did not store",
        "Query by meaning and read the <code>distances</code> nobody reads",
        "Scope a search with a <code>where</code> filter, which is the half of retrieval that is exact",
    ], "> **Raw `chromadb` here, not the LangChain wrapper.** One level down is worth seeing\n"
       "> once; Lab 6.5 puts the wrapper back on top."),

    setup(3),
    CORPUS_CELL,

    md("""
## Concept

A **collection** is the unit of separation &mdash; one per corpus, or per tenant. Inside it,
every document is four things:

| | |
|---|---|
| `id` | how you update or delete it later. People forget it exists until re-indexing day |
| `embedding` | the vector it is ranked by |
| `document` | the original text |
| `metadata` | anything else: source, page, category, entitlement |

Two of those you choose. The embedding is computed for you &mdash; here by the same
`all-MiniLM-L6-v2` you used in Lab 6.2, handed to Chroma as an *embedding function* so it
does not go looking for a model of its own.
"""),

    md("""
## Section 1 &mdash; A collection, and the metadata you will wish you had stored

You cannot filter on an attribute you did not attach. Deciding the metadata schema is the
one irreversible decision in this lab: changing it later means re-embedding the corpus.
"""),

    code('''
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# The same all-MiniLM-L6-v2 as Lab 6.2 -- here it is simply Chroma's own default, which
# is why raw chromadb needs no embedding configuration at all.
EMBED_FN = DefaultEmbeddingFunction()

def metadata_for(passage: dict) -> dict:
    """What has to travel with the text so a later query can scope to it?

    Lab 6.7 needs to cite the answer, and this lab needs to filter by topic.
    Both of those are only possible if the value is in here."""
    return {"category": passage["category"], "source": BLANK, "page": BLANK}


def build_collection():
    """Given -- one add() call. Note that ids are ours to choose, not Chroma's."""
    client = chromadb.Client()                       # in-memory: fast, nothing to clean up
    col = client.get_or_create_collection("handbook", embedding_function=EMBED_FN)
    if col.count() == 0:
        col.add(documents=[d["text"] for d in HANDBOOK],
                metadatas=[metadata_for(d) for d in HANDBOOK],
                ids=[f"doc{i}" for i in range(len(HANDBOOK))])
    return col
''', '''
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# The same all-MiniLM-L6-v2 as Lab 6.2 -- here it is simply Chroma's own default, which
# is why raw chromadb needs no embedding configuration at all.
EMBED_FN = DefaultEmbeddingFunction()

def metadata_for(passage: dict) -> dict:
    """What has to travel with the text so a later query can scope to it?

    Lab 6.7 needs to cite the answer, and this lab needs to filter by topic.
    Both of those are only possible if the value is in here."""
    return {"category": passage["category"], "source": passage["source"],
            "page": passage["page"]}


def build_collection():
    """Given -- one add() call. Note that ids are ours to choose, not Chroma's."""
    client = chromadb.Client()                       # in-memory: fast, nothing to clean up
    col = client.get_or_create_collection("handbook", embedding_function=EMBED_FN)
    if col.count() == 0:
        col.add(documents=[d["text"] for d in HANDBOOK],
                metadatas=[metadata_for(d) for d in HANDBOOK],
                ids=[f"doc{i}" for i in range(len(HANDBOOK))])
    return col
'''),

    code('''
# --- Self-check: Section 1   (a real Chroma collection -- still no gateway)
check("every passage is in the collection",
      lambda: build_collection().count() == len(HANDBOOK))
check("metadata carries the source file",
      lambda: all("source" in m for m in build_collection().get()["metadatas"]),
      "without it, Lab 6.7 has nothing to cite")
check("metadata carries the page",
      lambda: all(isinstance(m.get("page"), int)
                  for m in build_collection().get()["metadatas"]))
check("and the category, which is what the filter below needs",
      lambda: {m["category"] for m in build_collection().get()["metadatas"]}
              == {"leave", "wfh", "expense", "tech", "office"})
'''),

    md("""
## Section 2 &mdash; Asking, and scoping

Two things to notice in this section, and the second one is the important one.

First, the query and the document share no words and match anyway. Second, `n_results=3`
returns three results whatever is in the corpus &mdash; ask about the share price and you
still get three handbook passages back, each with a distance that is the only clue anything
is wrong.
"""),

    code('''
def search(col, question: str, k: int = 3, where: dict | None = None) -> list:
    """Given. Returns (distance, text, metadata) triples, closest first."""
    res = col.query(query_texts=[question], n_results=k,
                    **({"where": where} if where else {}))
    return list(zip(res["distances"][0], res["documents"][0], res["metadatas"][0]))


def expense_only() -> dict:
    """A `where` clause that keeps the search inside expense passages.

    Chroma takes a dict of {field: value} against the metadata you stored above."""
    return BLANK


def from_the_tech_guide() -> dict:
    """And one that keeps it inside a single source FILE, whatever the topic."""
    return BLANK
''', '''
def search(col, question: str, k: int = 3, where: dict | None = None) -> list:
    """Given. Returns (distance, text, metadata) triples, closest first."""
    res = col.query(query_texts=[question], n_results=k,
                    **({"where": where} if where else {}))
    return list(zip(res["distances"][0], res["documents"][0], res["metadatas"][0]))


def expense_only() -> dict:
    """A `where` clause that keeps the search inside expense passages.

    Chroma takes a dict of {field: value} against the metadata you stored above."""
    return {"category": "expense"}


def from_the_tech_guide() -> dict:
    """And one that keeps it inside a single source FILE, whatever the topic."""
    return {"source": "tech-guide.pdf"}
'''),

    code('''
# --- Self-check: Section 2
def _col():
    return build_collection()

check("a SPECIFIC question finds the annual-leave passage",
      lambda: "24 days" in search(_col(), "How much annual leave do I get?")[0][1],
      "neither 'much' nor 'get' appears in it -- the match is on meaning, not words")
check("a VAGUE question does not",
      lambda: "24 days" not in search(_col(), "How many holidays do I get?")[0][1],
      "'holidays' fits 'Sick leave is 12 days per year' just as well, and that one wins")
check("...and it loses by a rounding error, so top-1 was a coin flip",
      lambda: abs(search(_col(), "How many holidays do I get?", k=2)[1][0]
                  - search(_col(), "How many holidays do I get?", k=2)[0][0]) < 0.05,
      "two passages 0.001 apart -- this is the argument for k > 1, made by the data")
check("the expense filter returns expense passages and nothing else",
      lambda: all(m["category"] == "expense"
                  for _, _, m in search(_col(), "What can I claim?", where=expense_only())))
check("the source filter returns only the tech guide",
      lambda: all(m["source"] == "tech-guide.pdf"
                  for _, _, m in search(_col(), "What do we use?", k=2,
                                        where=from_the_tech_guide())))
check("an off-topic question STILL returns k results",
      lambda: len(search(_col(), "What is the company share price?", k=3)) == 3,
      "nothing refuses -- the only signal that this went wrong is the distance")
check("...and they are further away than a real hit",
      lambda: search(_col(), "What is the company share price?")[0][0]
              > search(_col(), "How many holidays do I get?")[0][0])

def _show():
    for label, q, w in [("specific  ", "How much annual leave do I get?", None),
                        ("vague     ", "How many holidays do I get?", None),
                        ("off topic ", "What is the company share price?", None),
                        ("filtered  ", "What can I claim?", expense_only())]:
        for d, t, m in search(_col(), q, k=2, where=w)[:2]:
            print(f"  {label} [{d:.4f}] ({m['source']} p.{m['page']}) {t[:52]}...")
        label = "          "
guard(_show)
'''),

    code('''
score()
'''),

    md("""
## Your turn

1. Print the distance of the best hit for ten questions, five of which the handbook cannot
   answer. Where would you put a threshold? Now notice that the two groups overlap, and that
   any threshold you pick costs you something in one direction or the other.
2. Chroma also takes `{"$or": [...]}`. Write a filter for leave **or** wfh and ask
   &ldquo;what are my benefits?&rdquo;. Compare with no filter at all.
3. Add a `sensitivity` field to `metadata_for` and set it to `"restricted"` on the office
   passages. Now write the filter that a search would need if the person asking is not
   allowed to see restricted material. That is entitlement-aware retrieval, and it is the
   same `where` clause.
"""),
]


# =========================================================================== #
# Lab 6.4 -- Document loading and splitting
# =========================================================================== #
LAB4 = [
    header(4, "Document Loading and Splitting", "Intermediate", 25, [
        "Load a real file off disk with <code>TextLoader</code> and see what a <code>Document</code> is",
        "Split it with <code>RecursiveCharacterTextSplitter</code> and choose the two numbers that matter",
        "Watch a rule get separated from the exception that qualifies it &mdash; and put it back",
        "See what overlap actually buys, at the boundary where it matters",
    ], "> **This lab makes no gateway calls.** Splitting is arithmetic on strings; the model\n"
       "> never sees any of it. It also decides more about retrieval quality than the model does."),

    setup(4),

    md("""
## Concept

A `Document` is two fields: `page_content` and `metadata`. Loaders produce them, splitters
cut them up, vector stores index them, retrievers hand them back. Everything downstream is
the same object.

The splitter has two numbers and both are decisions:

| | |
|---|---|
| `chunk_size` | too large and the vector averages several ideas, so it is close to nothing in particular. Too small and a rule gets separated from its exception |
| `chunk_overlap` | how much of the end of one chunk is repeated at the start of the next, so a sentence that straddles a cut is still findable |

`RecursiveCharacterTextSplitter` tries a list of separators in order &mdash; paragraphs
first, then lines, then spaces &mdash; so a paragraph survives whole wherever it can.
"""),

    code('''
# ------------------------------------------------- the raw document, written to disk
# The two sentences to watch are in CHAPTER 1: annual leave is 24 days, and unused leave
# cannot be carried forward. The second qualifies the first. Cut between them and no
# retriever on earth can return them together.

HANDBOOK_TEXT = """Employee Handbook

CHAPTER 1: LEAVE

Annual leave is 24 days per year for all full-time employees. Leave must be applied for at least 3 working days in advance through the HR portal. Unused annual leave cannot be carried forward to the next financial year.

Sick leave is 12 days per year. Notify your manager by 10 AM on the day of absence. A medical certificate is required for absences exceeding 2 consecutive days.

Maternity leave is 26 weeks of paid leave. Paternity leave is 2 weeks. Both must be applied for at least 30 days before the expected date.

CHAPTER 2: WORKING FROM HOME

Employees may work from home up to 3 days per week with team lead approval. Core hours are 10 AM to 4 PM IST and you must be reachable during them.

A VPN connection is mandatory for reaching internal systems from home. Contact the IT helpdesk for setup.

CHAPTER 3: EXPENSES

Travel expenses must be submitted with original receipts within 7 working days of travel. The meal allowance during client visits is 500 per day.

Internet reimbursement is 1,500 per month for employees working from home. Submit the broadband bill to finance by the 5th of each month.
"""

HANDBOOK_PATH = os.path.join(WORK, "handbook.txt")
with open(HANDBOOK_PATH, "w") as fh:
    fh.write(HANDBOOK_TEXT)

print(f"wrote {len(HANDBOOK_TEXT)} characters to {HANDBOOK_PATH}")
'''),

    md("""
## Section 1 &mdash; A file becomes Documents

`TextLoader` returns a **list** of one `Document` for a text file. A PDF loader returns one
per page. Either way the next stage does not care, which is the point of the abstraction.
"""),

    code('''
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_handbook() -> list:
    """Given."""
    return TextLoader(HANDBOOK_PATH).load()


def extra_metadata() -> dict:
    """TextLoader records only `source`, and it records the temp path it read.

    A citation needs a name a human recognises, and a filter needs something to filter
    on. Add both: a `title` of "Employee Handbook" and a `doc_type` of "policy"."""
    return BLANK
''', '''
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_handbook() -> list:
    """Given."""
    return TextLoader(HANDBOOK_PATH).load()


def extra_metadata() -> dict:
    """TextLoader records only `source`, and it records the temp path it read.

    A citation needs a name a human recognises, and a filter needs something to filter
    on. Add both: a `title` of "Employee Handbook" and a `doc_type` of "policy"."""
    return {"title": "Employee Handbook", "doc_type": "policy"}
'''),

    code('''
# --- Self-check: Section 1   (Document objects -- no store yet, no model)
check("TextLoader returns a list of Documents",
      lambda: isinstance(load_handbook(), list)
              and isinstance(load_handbook()[0], Document))
check("a text file loads as exactly one Document",
      lambda: len(load_handbook()) == 1,
      "a PDF loader would give you one per page -- same object either way")
check("the loader recorded where it came from",
      lambda: "source" in load_handbook()[0].metadata)
check("you added a human-readable title",
      lambda: extra_metadata()["title"] == "Employee Handbook")
check("and something a filter can use",
      lambda: extra_metadata()["doc_type"] == "policy")
'''),

    md("""
## Section 2 &mdash; The cut

Now the decision. Section 1 of the handbook states the annual-leave allowance and then, two
sentences later, says unused leave cannot be carried forward. Someone asking
&ldquo;can I carry my leave over?&rdquo; needs both.

Pick a `chunk_size` that keeps that paragraph whole, and an overlap that keeps the boundary
recoverable.
"""),

    code('''
def chunk_size() -> int:
    """Big enough that the annual-leave paragraph survives in one piece, small enough that
    a chunk is still about one thing. The paragraph is a little over 200 characters."""
    return BLANK


def chunk_overlap() -> int:
    """How much of the end of one chunk to repeat at the start of the next. Roughly a
    sentence is the usual answer; zero is the usual mistake."""
    return BLANK


def split(docs: list) -> list:
    """Given -- your two numbers, wired into the splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size(),
        chunk_overlap=chunk_overlap(),
        separators=["\\n\\n", "\\n", " ", ""])
    return splitter.split_documents(docs)
''', '''
def chunk_size() -> int:
    """Big enough that the annual-leave paragraph survives in one piece, small enough that
    a chunk is still about one thing. The paragraph is a little over 200 characters."""
    return 300


def chunk_overlap() -> int:
    """How much of the end of one chunk to repeat at the start of the next. Roughly a
    sentence is the usual answer; zero is the usual mistake."""
    return 50


def split(docs: list) -> list:
    """Given -- your two numbers, wired into the splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size(),
        chunk_overlap=chunk_overlap(),
        separators=["\\n\\n", "\\n", " ", ""])
    return splitter.split_documents(docs)
'''),

    code('''
# --- Self-check: Section 2
def leave_chunk(chunks):
    """The chunk that states the 24-day allowance."""
    return next(c for c in chunks if "24 days" in c.page_content)

def chunks_now():
    return split(load_handbook())

check("the handbook splits into several chunks",
      lambda: 4 <= len(chunks_now()) <= 20)
check("chunk_size is a sane size for a policy paragraph",
      lambda: 200 <= chunk_size() <= 1000,
      "under 200 splits the paragraph; over 1000 and the vector means nothing in particular")
check("overlap is non-zero but not most of the chunk",
      lambda: 0 < chunk_overlap() <= chunk_size() // 4)
check("the allowance and the carry-forward rule are in ONE chunk",
      lambda: "carried forward" in leave_chunk(chunks_now()).page_content,
      "at a smaller chunk_size these separate, and the answer becomes unreachable")
check("every chunk kept the parent document's metadata",
      lambda: all("source" in c.metadata for c in chunks_now()))

def _cut_it_too_small():
    """The same document at chunk_size=120 -- the failure this lab is about."""
    small = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0)
    chunks = small.split_documents(load_handbook())
    hit = next(c for c in chunks if "24 days" in c.page_content)
    print("  at 120 chars, the allowance chunk reads:")
    print("   ", repr(hit.page_content[:110]))
    print("   ...and 'carried forward' is in it:",
          "carried forward" in hit.page_content)
    print("    (it is still indexed -- just no longer attached to the rule it qualifies)")
guard(_cut_it_too_small)
'''),

    code('''
# --- What overlap actually buys, at the boundary
def _boundary():
    chunks = chunks_now()
    if len(chunks) < 2:
        print("(only one chunk -- lower chunk_size to see a boundary)")
        return
    tail = chunks[0].page_content[-chunk_overlap():]
    print("  end of chunk 1  :", repr(tail))
    print("  start of chunk 2:", repr(chunks[1].page_content[:chunk_overlap()]))
    print("  the tail reappears in chunk 2:", tail[:20] in chunks[1].page_content)

guard(_boundary)

def _sizes():
    for size in (120, 300, 800, 2000):
        s = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=50)
        c = s.split_documents(load_handbook())
        avg = sum(len(x.page_content) for x in c) // len(c)
        print(f"  chunk_size={size:>5} -> {len(c):>2} chunks, {avg:>4} chars each")
guard(_sizes)
'''),

    code('''
score()
'''),

    md("""
## Your turn

1. Set `chunk_overlap()` to 0 and re-run. Which self-check fails, and which one *should*
   have failed but did not? Overlap protects a boundary you have not thought of yet.
2. Drop `"\\n\\n"` from the separator list. The splitter now cuts on single newlines and
   never respects a paragraph. Look at the allowance chunk again.
3. Split by heading instead: `MarkdownHeaderTextSplitter` on `CHAPTER` lines keeps a whole
   chapter together and writes the heading into the metadata. On this document that is
   strictly better. On a 400-page contract it is strictly worse. Why?
"""),
]


# =========================================================================== #
# Lab 6.5 -- LangChain + Chroma
# =========================================================================== #
LAB5 = [
    header(5, "LangChain and Chroma", "Intermediate", 25, [
        "Embed and store a corpus in one call with <code>Chroma.from_documents()</code>",
        "Read the scores that <code>similarity_search_with_score</code> gives you and top-k does not",
        "Build a retriever, and choose between <code>similarity</code> and <code>mmr</code> on a corpus that needs the choice",
        "Scope a retriever with a metadata filter, so the chain in Lab 6.6 inherits it",
    ], "> **This lab makes no gateway calls.** A retriever is not a model &mdash; it embeds,\n"
       "> ranks and returns. The chat model arrives in Lab 6.6."),

    setup(5),
    CORPUS_CELL,
    docs_cell(),

    md("""
## Concept

`Chroma.from_documents()` does three things in one call: embeds every document, writes the
vectors, text and metadata into a collection, and hands back a store object.

A **retriever** is then a thin wrapper with one job &mdash; `str` in, `list[Document]` out.
That signature is the whole reason it drops into an LCEL chain in the next lab without
ceremony.

Two dials, and both are decisions:

| | |
|---|---|
| `k` | how many chunks come back. Always exactly `k`, however bad the last one is |
| `search_type` | `similarity` takes the top k. `mmr` trades a little relevance for variety, and earns its keep when the corpus repeats itself |
"""),

    md("""
## Section 1 &mdash; From Documents to a searchable store

Note what you do **not** pass: an embedding function for Chroma to guess at. You hand it
the model, so there is no ambiguity about what indexed the corpus and what will embed the
query. They must be the same model, and this is how you guarantee it.
"""),

    code('''
from langchain_chroma import Chroma

def build_store():
    """Embed the corpus and store it. Which model embeds it?

    It has to be the same one that will embed the queries -- a store indexed with one
    model and searched with another returns noise, and nothing raises."""
    return Chroma.from_documents(
        documents=handbook_documents(),
        embedding=BLANK,
        collection_name="handbook_lc")


_store = {}
def store():
    """Given -- build once per kernel, not once per query."""
    if "s" not in _store:
        _store["s"] = build_store()
    return _store["s"]
''', '''
from langchain_chroma import Chroma

def build_store():
    """Embed the corpus and store it. Which model embeds it?

    It has to be the same one that will embed the queries -- a store indexed with one
    model and searched with another returns noise, and nothing raises."""
    return Chroma.from_documents(
        documents=handbook_documents(),
        embedding=get_embeddings(),
        collection_name="handbook_lc")


_store = {}
def store():
    """Given -- build once per kernel, not once per query."""
    if "s" not in _store:
        _store["s"] = build_store()
    return _store["s"]
'''),

    code('''
# --- Self-check: Section 1   (a real Chroma store -- embeddings are local, no gateway)
check("the store holds every passage",
      lambda: store()._collection.count() == len(HANDBOOK))
check("similarity_search returns Documents, not strings",
      lambda: isinstance(store().similarity_search("annual leave", k=1)[0], Document))
check("a specific question finds the annual-leave passage",
      lambda: "24 days" in store().similarity_search("How much annual leave do I get?",
                                                     k=1)[0].page_content)
check("the metadata survived the round trip",
      lambda: store().similarity_search("annual leave", k=1)[0].metadata["source"] == "handbook.pdf")

def _scores():
    print("  scores are DISTANCES here -- lower is closer, unlike the cosine in Lab 6.2")
    for doc, s in store().similarity_search_with_score("How do I work from home?", k=3):
        print(f"  [{s:.4f}] ({doc.metadata['category']}) {doc.page_content[:56]}...")
guard(_scores)
'''),

    md("""
## Section 2 &mdash; The retriever, and the two dials

The corpus has **three** leave passages that all look alike to an embedding model. Ask a
broad question about benefits with plain similarity and you get three variations on leave
and nothing about working from home or expenses.

That is the case `mmr` exists for.
"""),

    code('''
def broad_search_type() -> str:
    """A user asks "what leave can I take?". The corpus has three leave passages, and
    plain similarity hands back all three -- annual, sick and maternity -- with nothing
    else. Which search_type trades a little relevance for a wider spread?"""
    return BLANK


def leave_only() -> dict:
    """A filter that pins a retriever to leave passages, whatever the question.

    LangChain passes this through to Chroma as the `where` clause you wrote in Lab 6.3."""
    return BLANK


def make_retriever(k: int = 3, search_type: str = "similarity", flt: dict | None = None):
    """Given -- your decisions, wired in."""
    kwargs = {"k": k}
    if search_type == "mmr":
        kwargs["fetch_k"] = k * 3        # look at 3k, return a diverse k
    if flt:
        kwargs["filter"] = flt
    return store().as_retriever(search_type=search_type, search_kwargs=kwargs)
''', '''
def broad_search_type() -> str:
    """A user asks "what leave can I take?". The corpus has three leave passages, and
    plain similarity hands back all three -- annual, sick and maternity -- with nothing
    else. Which search_type trades a little relevance for a wider spread?"""
    return "mmr"


def leave_only() -> dict:
    """A filter that pins a retriever to leave passages, whatever the question.

    LangChain passes this through to Chroma as the `where` clause you wrote in Lab 6.3."""
    return {"category": "leave"}


def make_retriever(k: int = 3, search_type: str = "similarity", flt: dict | None = None):
    """Given -- your decisions, wired in."""
    kwargs = {"k": k}
    if search_type == "mmr":
        kwargs["fetch_k"] = k * 3        # look at 3k, return a diverse k
    if flt:
        kwargs["filter"] = flt
    return store().as_retriever(search_type=search_type, search_kwargs=kwargs)
'''),

    code('''
# --- Self-check: Section 2   (retriever objects and what they return)
def cats(docs):
    return [d.metadata["category"] for d in docs]

check("a retriever takes a string and returns Documents",
      lambda: all(isinstance(d, Document)
                  for d in make_retriever().invoke("What is the expense policy?")))
check("k really does control how many come back",
      lambda: len(make_retriever(k=2).invoke("annual leave")) == 2)
check("you chose a search_type LangChain knows",
      lambda: broad_search_type() in ("similarity", "mmr"))
check("plain similarity answers 'what leave can I take?' with three leave passages",
      lambda: cats(make_retriever(k=3).invoke("What leave can I take?")) == ["leave"] * 3,
      "all three are relevant -- and the user learns nothing they could not have guessed")
check("your choice returns more than one category for the same question",
      lambda: len(set(cats(make_retriever(k=3, search_type=broad_search_type())
                           .invoke("What leave can I take?")))) > 1,
      "similarity keeps handing back the same topic; mmr is the dial for that")
check("the filtered retriever never leaves the leave passages",
      lambda: set(cats(make_retriever(k=3, flt=leave_only())
                       .invoke("What can I claim for travel?"))) == {"leave"},
      "note the question is about expenses and it STILL only returns leave")

def _compare():
    for label, st in (("similarity", "similarity"), (broad_search_type(), broad_search_type())):
        got = cats(make_retriever(k=3, search_type=st).invoke("What leave can I take?"))
        print(f"  {label:11} -> {got}")
    print("  ...and read what that cost: two of the three LEAVE passages are gone.")
    print("  mmr bought variety by giving up relevance. On this question that is")
    print("  probably a bad trade -- which is the point. It is a dial, not a fix.")
guard(_compare)
'''),

    code('''
score()
'''),

    md("""
## Your turn

1. The filtered retriever answered a travel-expenses question with leave passages and did
   not complain. Build the same retriever with `k=3` and print the scores. Is there a
   distance at which you would rather return nothing? Write that rule down &mdash; Lab 6.6
   makes the model act on it.
2. Raise `fetch_k` on the mmr retriever from `3k` to `10k`. Where does the spread stop
   improving, and what does it cost?
3. Build a *second* store from the same documents with `collection_name="handbook_alt"` but
   chunked differently, and ask both the same question. Two stores, one corpus, different
   answers &mdash; which is the honest version of &ldquo;we upgraded retrieval&rdquo;.
"""),
]


# =========================================================================== #
# Lab 6.6 -- Your first RAG chain
# =========================================================================== #
LAB6 = [
    header(6, "Your First RAG Chain", "Intermediate", 30, [
        "Wire a retriever, a prompt, the model and a parser into one LCEL chain",
        "Work out what <code>RunnablePassthrough</code> is actually for",
        "Write the refusal clause &mdash; and measure what happens without it",
        "Ask the corpus something it does not contain, and watch which version makes something up",
    ], "> **This lab calls the chat model.** Everything up to the last cell of each section\n"
       "> is still offline: a chain is an object, and building it needs no gateway."),

    setup(6),
    CORPUS_CELL,
    docs_cell(),

    code('''
# ------------------------------------------------- carried forward from Lab 6.5 (given)
from langchain_chroma import Chroma

_store = {}
def store():
    if "s" not in _store:
        _store["s"] = Chroma.from_documents(documents=handbook_documents(),
                                            embedding=get_embeddings(),
                                            collection_name="handbook_rag")
    return _store["s"]

def retriever(k: int = 3):
    return store().as_retriever(search_kwargs={"k": k})

print("store and retriever ready")
'''),

    md("""
## Concept

The chain is one pipe with a dictionary at the front:

```
{"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
```

Invoke it with a string. That one string goes to **both** keys: down the left branch it hits
the retriever and comes back as chunks; down the right branch `RunnablePassthrough` hands it
straight to the template. The prompt then has both `{context}` and `{question}` to fill.

Take the passthrough out and the template has nothing to put in `{question}`. That is all it
is for, and it is the piece people cannot explain in interviews.
"""),

    md("""
## Section 1 &mdash; The chain

`format_docs` is given &mdash; joining strings is not the lesson. The decision is what fills
`"question"`.
"""),

    code('''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs) -> str:
    """Given -- Documents to one string for the prompt."""
    return "\\n\\n".join(d.page_content for d in docs)


def question_branch():
    """The chain is invoked with a plain string. The left branch sends it to the retriever.
    What does the right branch need, so the template's {question} is the ORIGINAL string?"""
    return BLANK


def build_chain(prompt):
    """Given -- your branch, wired in."""
    return ({"context": retriever() | format_docs, "question": question_branch()}
            | prompt | get_llm() | StrOutputParser())
''', '''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs) -> str:
    """Given -- Documents to one string for the prompt."""
    return "\\n\\n".join(d.page_content for d in docs)


def question_branch():
    """The chain is invoked with a plain string. The left branch sends it to the retriever.
    What does the right branch need, so the template's {question} is the ORIGINAL string?"""
    return RunnablePassthrough()


def build_chain(prompt):
    """Given -- your branch, wired in."""
    return ({"context": retriever() | format_docs, "question": question_branch()}
            | prompt | get_llm() | StrOutputParser())
'''),

    code('''
# --- Self-check: Section 1   (the branch and the formatter -- the chain needs a gateway)
check("the question branch is a Runnable",
      lambda: hasattr(question_branch(), "invoke"))
check("it passes a string through UNCHANGED",
      lambda: question_branch().invoke("How many days of leave?") == "How many days of leave?",
      "anything that transforms the question here and the template gets the wrong text")
check("format_docs turns retrieved Documents into one string",
      lambda: isinstance(format_docs(retriever().invoke("annual leave")), str))
check("and the retrieved text really is in it",
      lambda: "24 days" in format_docs(retriever().invoke("How many days off do I get?")))
'''),

    md("""
## Section 2 &mdash; The refusal clause

The corpus says nothing about salaries, notice periods or the share price. Ask about them
anyway.

A prompt without an explicit instruction to refuse will get an answer invented out of
whatever chunks happened to come back &mdash; because three chunks always come back. The
clause that fixes it is one sentence, and it is the difference between a demo and something
you would put in front of staff.
"""),

    code('''
def refusal_clause() -> str:
    """One sentence telling the model what to do when the context does not contain the
    answer. It has to name the behaviour you want, not just discourage the one you don't."""
    return BLANK


LOOSE = ChatPromptTemplate.from_template(
    "Answer the question using the context below.\\n\\n"
    "Context:\\n{context}\\n\\nQuestion: {question}\\nAnswer:")

def strict_prompt():
    """Given -- the same prompt, plus your clause."""
    return ChatPromptTemplate.from_template(
        "Answer the question using ONLY the context below. " + refusal_clause() +
        "\\n\\nContext:\\n{context}\\n\\nQuestion: {question}\\nAnswer:")
''', '''
def refusal_clause() -> str:
    """One sentence telling the model what to do when the context does not contain the
    answer. It has to name the behaviour you want, not just discourage the one you don't."""
    return ("If the context does not contain the answer, reply exactly "
            "\\"I don't have that information in the handbook.\\" and nothing else.")


LOOSE = ChatPromptTemplate.from_template(
    "Answer the question using the context below.\\n\\n"
    "Context:\\n{context}\\n\\nQuestion: {question}\\nAnswer:")

def strict_prompt():
    """Given -- the same prompt, plus your clause."""
    return ChatPromptTemplate.from_template(
        "Answer the question using ONLY the context below. " + refusal_clause() +
        "\\n\\nContext:\\n{context}\\n\\nQuestion: {question}\\nAnswer:")
'''),

    code('''
# --- Self-check: Section 2   (prompt objects -- no call yet)
check("your clause is a sentence, not a word",
      lambda: len(refusal_clause().split()) >= 6)
check("it says what to do, not only what to avoid",
      lambda: any(w in refusal_clause().lower()
                  for w in ("say", "reply", "respond", "answer", "state")),
      "'do not make things up' tells the model what not to write, not what to write instead")
check("the strict prompt still has both variables",
      lambda: set(strict_prompt().input_variables) == {"context", "question"})
check("and the clause really is in the rendered prompt",
      lambda: refusal_clause()[:24] in
              strict_prompt().format(context="c", question="q"))
check("the loose prompt does NOT contain it",
      lambda: refusal_clause()[:24] not in LOOSE.format(context="c", question="q"))
'''),

    code('''
# --- Run it for real -------------------------------------------------------
# Four questions. Three the handbook answers, one it does not.
IN_SCOPE = ["How many days of annual leave do I get?",
            "Can I work from home, and how often?",
            "What is the meal allowance on a client visit?"]
OUT_OF_SCOPE = "What is the company's share price?"

def answers():
    chain = build_chain(strict_prompt())
    for q in IN_SCOPE:
        print(f"Q: {q}\\nA: {chain.invoke(q).strip()}\\n")

if llm_ready():
    guard(answers)
'''),

    code('''
# --- Run it for real: the clause, measured ---------------------------------
# The same out-of-scope question through both prompts. Three handbook chunks are
# retrieved either way -- none of them about the share price.
def refusal_matters():
    for label, prompt in (("no clause", LOOSE), ("with clause", strict_prompt())):
        out = build_chain(prompt).invoke(OUT_OF_SCOPE).strip()
        print(f"  [{label:11}] {out[:150]}")
    print("\\n  retrieved for that question:")
    for d in retriever().invoke(OUT_OF_SCOPE):
        print(f"    ({d.metadata['category']}) {d.page_content[:56]}...")

if llm_ready():
    guard(refusal_matters)
'''),

    code('''
score()
'''),

    md("""
## Your turn

1. Delete `"question": question_branch()` from the dictionary and invoke the chain. Read the
   error carefully &mdash; it names exactly what the passthrough was doing.
2. Run the out-of-scope question through the loose prompt five times. Count how many answers
   invent a number. One run is an anecdote; five is the beginning of an eval set, and
   Module 7 turns it into one.
3. Your clause asks for an exact sentence. Now write a version that also says *which*
   category it searched, so the user learns something from the refusal. Does the model
   comply reliably? Instructions that require the model to report on its own retrieval are
   a common source of confident nonsense.
"""),
]


# =========================================================================== #
# Lab 6.7 -- RAG with citations
# =========================================================================== #
LAB7 = [
    header(7, "RAG with Citations", "Intermediate", 25, [
        "Put the source and page INTO the context, because a model cannot cite what it cannot see",
        "Prompt for citations in a fixed format, and check they point at real documents",
        "Build a category-scoped chain from a filtered retriever",
        "Catch the failure that matters: a citation that is right next to an answer it does not support",
    ], "> **The citation is the deliverable.** An answer nobody can check is worth less than\n"
       "> no answer, and this is the one lab in the module a compliance reviewer would ask for."),

    setup(7),
    CORPUS_CELL,
    docs_cell(),

    code('''
# ------------------------------------------------- carried forward from Labs 6.5-6.6 (given)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

_store = {}
def store():
    if "s" not in _store:
        _store["s"] = Chroma.from_documents(documents=handbook_documents(),
                                            embedding=get_embeddings(),
                                            collection_name="handbook_cite")
    return _store["s"]

def retriever(k: int = 3, flt: dict | None = None):
    kwargs = {"k": k}
    if flt:
        kwargs["filter"] = flt
    return store().as_retriever(search_kwargs=kwargs)

print("store and retriever ready")
'''),

    md("""
## Concept

A citation is not a footnote you add afterwards. It is a **binding**: the model can only
name a source that was in the context it was given, so the source has to be *in* the
context.

That is two changes to Lab 6.6, both small:

1. `format_docs` writes `[handbook.pdf, page 5]` above each chunk instead of just joining text.
2. The prompt asks for the citation in a fixed shape.

And one thing that does not follow: a citation being *present* does not make the sentence
next to it *true*. The last section of this lab is about telling those apart.
"""),

    md("""
## Section 1 &mdash; Putting the source in the context

Every `Document` already carries `source` and `page` &mdash; you decided that back in
Lab 6.3, which is why it is available now.
"""),

    code('''
def label_for(doc) -> str:
    """The one-line header that goes above a chunk in the context.

    Return it in the form:  [handbook.pdf, page 5]
    Use doc.metadata; fall back to "?" for a missing page."""
    return BLANK


def format_docs_with_sources(docs) -> str:
    """Given -- your label, then the text, blank line between chunks."""
    return "\\n\\n".join(f"{label_for(d)}\\n{d.page_content}" for d in docs)
''', '''
def label_for(doc) -> str:
    """The one-line header that goes above a chunk in the context.

    Return it in the form:  [handbook.pdf, page 5]
    Use doc.metadata; fall back to "?" for a missing page."""
    return f"[{doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', '?')}]"


def format_docs_with_sources(docs) -> str:
    """Given -- your label, then the text, blank line between chunks."""
    return "\\n\\n".join(f"{label_for(d)}\\n{d.page_content}" for d in docs)
'''),

    code('''
# --- Self-check: Section 1   (string shapes -- no gateway)
def sample():
    return retriever().invoke("How many sick days do I get?")

check("the label names the source file",
      lambda: "handbook.pdf" in label_for(sample()[0]))
check("the label names the page",
      lambda: "page" in label_for(sample()[0]).lower()
              and any(ch.isdigit() for ch in label_for(sample()[0])))
check("a Document with no page does not crash the label",
      lambda: isinstance(label_for(Document(page_content="x", metadata={"source": "a.pdf"})), str),
      "real corpora have gaps -- a loader that skipped a page number must not break citing")
check("the formatted context carries every retrieved chunk's source",
      lambda: all(d.metadata["source"] in format_docs_with_sources(sample()) for d in sample()))
check("and still carries the text itself",
      lambda: all(d.page_content[:30] in format_docs_with_sources(sample()) for d in sample()))

guard(lambda: print(format_docs_with_sources(sample())[:400], "..."))
'''),

    md("""
## Section 2 &mdash; Asking for the citation, and scoping the search

Two decisions left: how to ask for the citation, and what a scoped chain looks like when
you want an assistant that only ever answers from one part of the corpus.
"""),

    code('''
def citation_instruction() -> str:
    """Tell the model to cite. Name the FORMAT you want -- "cite your sources" gets you
    a different shape every call, which nothing downstream can parse."""
    return BLANK


def leave_filter() -> dict:
    """The filter for a leave-only assistant."""
    return BLANK


def citation_prompt():
    """Given."""
    return ChatPromptTemplate.from_template(
        "Answer using ONLY the context below. " + citation_instruction() +
        " If the context does not contain the answer, say you do not have that information."
        "\\n\\nContext:\\n{context}\\n\\nQuestion: {question}\\nAnswer:")


def chain_over(rtv):
    """Given -- any retriever, same chain."""
    return ({"context": rtv | format_docs_with_sources, "question": RunnablePassthrough()}
            | citation_prompt() | get_llm() | StrOutputParser())
''', '''
def citation_instruction() -> str:
    """Tell the model to cite. Name the FORMAT you want -- "cite your sources" gets you
    a different shape every call, which nothing downstream can parse."""
    return ("After each fact, cite the source it came from in parentheses, "
            "like (handbook.pdf, p.5).")


def leave_filter() -> dict:
    """The filter for a leave-only assistant."""
    return {"category": "leave"}


def citation_prompt():
    """Given."""
    return ChatPromptTemplate.from_template(
        "Answer using ONLY the context below. " + citation_instruction() +
        " If the context does not contain the answer, say you do not have that information."
        "\\n\\nContext:\\n{context}\\n\\nQuestion: {question}\\nAnswer:")


def chain_over(rtv):
    """Given -- any retriever, same chain."""
    return ({"context": rtv | format_docs_with_sources, "question": RunnablePassthrough()}
            | citation_prompt() | get_llm() | StrOutputParser())
'''),

    code('''
# --- Self-check: Section 2   (prompt and retriever objects)
check("the instruction names a citation format, not just the idea",
      lambda: "(" in citation_instruction() and ")" in citation_instruction(),
      "show the model the shape you want -- an example is worth a paragraph of description")
check("it mentions a real file from the corpus",
      lambda: any(s in citation_instruction()
                  for s in {d["source"] for d in HANDBOOK}))
check("the prompt still takes context and question",
      lambda: set(citation_prompt().input_variables) == {"context", "question"})
check("the leave-scoped retriever returns only leave passages",
      lambda: {d.metadata["category"]
               for d in retriever(k=3, flt=leave_filter()).invoke("What are my options?")}
              == {"leave"})
check("it finds all three leave passages, not just the one about annual leave",
      lambda: len(retriever(k=3, flt=leave_filter()).invoke("types of leave")) == 3)
'''),

    code('''
# --- Run it for real -------------------------------------------------------
def cited_answers():
    chain = chain_over(retriever())
    for q in ["How many sick days do I get, and do I need a doctor's note?",
              "How do I claim travel expenses?"]:
        print(f"Q: {q}\\nA: {chain.invoke(q).strip()}\\n")

if llm_ready():
    guard(cited_answers)
'''),

    code('''
# --- Run it for real: a scoped assistant -----------------------------------
def leave_desk():
    scoped = chain_over(retriever(k=3, flt=leave_filter()))
    q = "What kinds of leave can I take?"
    print(f"[leave-only] Q: {q}\\nA: {scoped.invoke(q).strip()}\\n")
    # ...and the same assistant asked something outside its scope. The retriever still
    # returns three leave passages, because a filter narrows -- it never returns nothing.
    q2 = "How much is the internet reimbursement?"
    print(f"[leave-only] Q: {q2}\\nA: {scoped.invoke(q2).strip()}")
    print("\\n  what it retrieved for that second question:")
    for d in retriever(k=3, flt=leave_filter()).invoke(q2):
        print(f"    ({d.metadata['category']}) {d.page_content[:56]}...")

if llm_ready():
    guard(leave_desk)
'''),

    code('''
# --- Run it for real: is the citation real? --------------------------------
# A citation is checkable, so check it. Every source the answer names must be one
# that was actually in the context -- a model that invents "policy-2019.pdf" is
# doing the most dangerous thing in this module.
def citations_are_real():
    q = "How many sick days do I get?"
    docs = retriever().invoke(q)
    answer = chain_over(retriever()).invoke(q)
    given = {d.metadata["source"] for d in docs}
    named = {s for s in {d["source"] for d in HANDBOOK} if s in answer}
    print("  in the context:", sorted(given))
    print("  named in the answer:", sorted(named) or "(none)")
    print("  every cited source was really there:", named <= given)

if llm_ready():
    guard(citations_are_real)
'''),

    code('''
score()
'''),

    md("""
## Your turn

1. Add the category to `label_for` and ask the model to cite it too. Longer citations are
   not obviously better &mdash; at what point does the citation crowd out the answer?
2. Write the check from the last cell as a function that returns `True`/`False` and run it
   over ten questions. You have just built the first metric of Module 7, and it is the one
   that catches a fabricated source.
3. Harder: ask &ldquo;how many sick days, and can I carry them forward?&rdquo;. The handbook
   answers the first half and not the second. Read the answer very carefully &mdash; a
   correct citation sitting beside an unsupported clause is the failure mode that survives
   every demo, because everything on the screen looks right.
"""),
]


# =========================================================================== #
# Lab 6.8 -- Challenge
# =========================================================================== #
LAB8 = [
    header(8, "Challenge &mdash; A Handbook Q&amp;A Bot", "Advanced", 30, [
        "Build the whole pipeline yourself: load, split, embed, store, retrieve, answer, cite",
        "Choose every parameter you were given in Labs 6.1&ndash;6.7, and defend each one",
        "Add a confidence band from the retrieval distance, and decide where to put the line",
        "Ship something that refuses well, because that is the part your users will test first",
    ], "> **Less scaffolding here.** Every piece you need appeared in an earlier lab; this\n"
       "> one asks you to put them in order and pick the numbers."),

    setup(8),

    md("""
## Concept

Everything in this module, in one object:

```
raw text -> split -> embed -> store -> retrieve (k, filter) -> format with sources
         -> prompt (refuse + cite) -> model -> answer
```

You have built every stage. What is left is the part that is actually hard: choosing the
numbers, and deciding what the thing does when it does not know.
"""),

    code('''
# ------------------------------------------------- a bigger corpus, as raw text
# Five policy documents. Longer and messier than Lab 6.3's ten neat passages -- which is
# why this one has to be split before it can be indexed.

POLICY_DOCS = [
    {"source": "leave-policy.pdf", "category": "leave", "text": """Leave Policy

Annual Leave: full-time employees receive 24 days per year. Apply at least 3 working days in advance through the HR portal. Unused annual leave cannot be carried forward. Employees with less than 6 months tenure receive prorated leave.

Sick Leave: 12 days per year. Notify your manager by 10 AM on the day of absence. For absences exceeding 2 consecutive days a medical certificate is mandatory. Unused sick leave carries forward up to a maximum of 30 days.

Maternity Leave: 26 weeks paid, and it may start up to 8 weeks before the expected date. Applies after 80 days of continuous employment. Paternity Leave: 2 weeks, to be taken within 6 months of the birth."""},

    {"source": "wfh-policy.pdf", "category": "wfh", "text": """Work From Home Policy

Eligibility: employees who have completed the 6-month probation period. New joiners work from the office for their first 6 months, without exception.

Schedule: up to 3 days per week with team lead approval. Core hours are 10 AM to 4 PM IST. Friday is a mandatory in-office day for every team.

Equipment: the company laptop must be used. A VPN connection is mandatory for internal systems. Reimbursement: 1,500 per month for internet, and up to 10,000 once for an ergonomic chair."""},

    {"source": "expense-policy.pdf", "category": "expense", "text": """Expense Policy

Travel: business travel must be pre-approved by your manager. Submit original receipts within 7 working days of completing the travel. Economy class for domestic flights; business class is allowed for international flights over 6 hours.

Meals: 500 per day during client visits within the country, 3,000 per day for international travel. Team dinners up to 1,000 per person with manager approval.

Equipment: laptops are replaced every 3 years. External monitors up to 15,000. Software licences must be requested through the IT helpdesk and must never be bought directly."""},

    {"source": "tech-guide.pdf", "category": "tech", "text": """Technology Guide

Backend: Python with FastAPI, and Java with Spring Boot. New microservices should use Python unless there is a specific reason otherwise. All APIs follow REST conventions.

Frontend: React is the standard for new work. Angular is maintained for the existing dashboard and admin portal. TypeScript is mandatory.

Data: PostgreSQL is the primary relational database. MongoDB where the schema must flex. Redis for caching and sessions. Infrastructure runs on AWS, described in Terraform, deployed by GitHub Actions."""},

    {"source": "office-directory.pdf", "category": "office", "text": """Office Directory

Bangalore is the headquarters: 5th floor, 200+ staff, every department represented. Cafeteria on the 3rd floor. Basement parking on application to admin. Hours are 9 AM to 6 PM, Monday to Friday.

Mumbai: Worli business district, Tower A, 12th floor. 50 staff, mostly sales, client success and marketing.

Hyderabad: 8th floor, 80 staff. The engineering hub for backend and data services, with 24/7 access for on-call engineers.

Pune: Phase 2, Building C, 4th floor. 40 staff in QA, DevOps and SRE."""},
]

print(f"{len(POLICY_DOCS)} documents, "
      f"{sum(len(d['text']) for d in POLICY_DOCS)} characters")
'''),

    md("""
## Part A &mdash; Build the knowledge base

Turn the five documents into `Document` objects, split them, embed them and store them.
Every one of these appeared in Labs 6.3&ndash;6.5.
"""),

    code('''
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def to_documents() -> list:
    """One Document per entry. The text goes in page_content; source and category are
    metadata, and Lab 6.7 will need both."""
    return BLANK


def splitter():
    """These documents are 600-900 characters and each covers several distinct rules.
    Choose a chunk_size and chunk_overlap that keep one rule per chunk."""
    return RecursiveCharacterTextSplitter(chunk_size=BLANK, chunk_overlap=BLANK)


_kb = {}
def knowledge_base():
    """Given -- build once."""
    if "s" not in _kb:
        chunks = splitter().split_documents(to_documents())
        _kb["chunks"] = chunks
        _kb["s"] = Chroma.from_documents(documents=chunks, embedding=get_embeddings(),
                                         collection_name="handbook_challenge")
    return _kb["s"]
''', '''
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def to_documents() -> list:
    """One Document per entry. The text goes in page_content; source and category are
    metadata, and Lab 6.7 will need both."""
    return [Document(page_content=d["text"],
                     metadata={"source": d["source"], "category": d["category"]})
            for d in POLICY_DOCS]


def splitter():
    """These documents are 600-900 characters and each covers several distinct rules.
    Choose a chunk_size and chunk_overlap that keep one rule per chunk."""
    return RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)


_kb = {}
def knowledge_base():
    """Given -- build once."""
    if "s" not in _kb:
        chunks = splitter().split_documents(to_documents())
        _kb["chunks"] = chunks
        _kb["s"] = Chroma.from_documents(documents=chunks, embedding=get_embeddings(),
                                         collection_name="handbook_challenge")
    return _kb["s"]
'''),

    code('''
# --- Self-check: Part A
check("five Documents, one per policy file",
      lambda: len(to_documents()) == 5
              and all(isinstance(d, Document) for d in to_documents()))
check("each carries its source and category",
      lambda: all({"source", "category"} <= set(d.metadata) for d in to_documents()))
check("splitting produces more chunks than documents",
      lambda: (knowledge_base(), len(_kb["chunks"]) > 8)[1],
      "if this is 5 your chunk_size is bigger than the documents")
check("no chunk is so big it covers the whole document",
      lambda: (knowledge_base(),
               max(len(c.page_content) for c in _kb["chunks"]) <= 600)[1])
check("the store holds every chunk",
      lambda: knowledge_base()._collection.count() == len(_kb["chunks"]))
check("a specific fact is findable",
      lambda: any("80 days" in d.page_content
                  for d in knowledge_base().similarity_search("maternity eligibility", k=3)))

guard(lambda: print(f"  {len(to_documents())} documents -> {len(_kb['chunks'])} chunks"))
'''),

    md("""
## Part B &mdash; The chain, with citations and a refusal

Same shape as Lab 6.7. Everything here is yours to write.
"""),

    code('''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def top_k() -> int:
    """Five documents, split into a dozen-odd chunks, and some questions span two rules.
    How many chunks should reach the prompt?"""
    return BLANK


def format_with_sources(docs) -> str:
    """Label each chunk with its source, then its text -- as in Lab 6.7."""
    return BLANK


def bot_prompt():
    """A prompt that (a) answers only from the context, (b) refuses when it cannot, and
    (c) cites the source of each fact in a fixed format."""
    return ChatPromptTemplate.from_template(BLANK)


def bot_chain():
    """Given -- your pieces, assembled."""
    rtv = knowledge_base().as_retriever(search_kwargs={"k": top_k()})
    return ({"context": rtv | format_with_sources, "question": RunnablePassthrough()}
            | bot_prompt() | get_llm() | StrOutputParser())
''', '''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def top_k() -> int:
    """Five documents, split into a dozen-odd chunks, and some questions span two rules.
    How many chunks should reach the prompt?"""
    return 4


def format_with_sources(docs) -> str:
    """Label each chunk with its source, then its text -- as in Lab 6.7."""
    return "\\n\\n".join(f"[{d.metadata['source']}]\\n{d.page_content}" for d in docs)


def bot_prompt():
    """A prompt that (a) answers only from the context, (b) refuses when it cannot, and
    (c) cites the source of each fact in a fixed format."""
    return ChatPromptTemplate.from_template(
        "You are the company handbook assistant. Answer using ONLY the context below.\\n"
        "Cite the source of each fact in parentheses, like (leave-policy.pdf).\\n"
        "If the context does not contain the answer, reply exactly "
        "\\"I don't have that information in the handbook.\\"\\n\\n"
        "Context:\\n{context}\\n\\nQuestion: {question}\\nAnswer:")


def bot_chain():
    """Given -- your pieces, assembled."""
    rtv = knowledge_base().as_retriever(search_kwargs={"k": top_k()})
    return ({"context": rtv | format_with_sources, "question": RunnablePassthrough()}
            | bot_prompt() | get_llm() | StrOutputParser())
'''),

    code('''
# --- Self-check: Part B
def _docs():
    return knowledge_base().similarity_search("annual leave", k=2)

check("k is between 2 and 8",
      lambda: 2 <= top_k() <= 8,
      "one chunk cannot answer a two-rule question; ten will bury the answer in noise")
check("the formatter names each chunk's source",
      lambda: all(d.metadata["source"] in format_with_sources(_docs()) for d in _docs()))
check("and keeps the text",
      lambda: all(d.page_content[:30] in format_with_sources(_docs()) for d in _docs()))
check("the prompt takes exactly context and question",
      lambda: set(bot_prompt().input_variables) == {"context", "question"})
check("the prompt restricts the model to the context",
      lambda: "only" in bot_prompt().format(context="c", question="q").lower())
check("the prompt asks for a citation",
      lambda: "cite" in bot_prompt().format(context="c", question="q").lower())
check("the prompt says what to do when the answer is not there",
      lambda: any(w in bot_prompt().format(context="c", question="q").lower()
                  for w in ("do not have", "don't have", "not contain", "cannot answer")))
'''),

    md("""
## Part C &mdash; Confidence, and where you put the line

The retriever always returns `k` chunks. `similarity_search_with_score` gives you the
distance, and that is the only number that knows the difference between a good hit and the
closest of a bad lot.

Look at the printed distances before you pick your threshold. They are properties of this
corpus and this embedding model, and they do not transfer.
"""),

    code('''
def confident_below() -> float:
    """A Chroma distance below this means the retrieval genuinely matched.

    Run the calibration cell below FIRST and read the two groups of numbers."""
    return BLANK


def answer_with_confidence(question: str) -> dict:
    """Given -- your threshold, applied."""
    hits = knowledge_base().similarity_search_with_score(question, k=top_k())
    best = hits[0][1] if hits else 99.0
    return {"question": question,
            "confident": best < confident_below(),
            "best_distance": round(best, 4),
            "answer": bot_chain().invoke(question) if llm_ready() else "(model not configured)"}
''', '''
def confident_below() -> float:
    """A Chroma distance below this means the retrieval genuinely matched.

    Run the calibration cell below FIRST and read the two groups of numbers."""
    return 1.2


def answer_with_confidence(question: str) -> dict:
    """Given -- your threshold, applied."""
    hits = knowledge_base().similarity_search_with_score(question, k=top_k())
    best = hits[0][1] if hits else 99.0
    return {"question": question,
            "confident": best < confident_below(),
            "best_distance": round(best, 4),
            "answer": bot_chain().invoke(question) if llm_ready() else "(model not configured)"}
'''),

    code('''
# --- Calibration: run this BEFORE choosing your threshold (no gateway needed)
ANSWERABLE = ["How many sick days do I get?",
              "Can a new joiner work from home?",
              "What is the meal allowance abroad?",
              "Which database should a new service use?"]
UNANSWERABLE = ["What is the company share price?",
                "What is my notice period?",
                "Who is the CEO?"]

def _calibrate():
    for label, qs in (("answerable  ", ANSWERABLE), ("unanswerable", UNANSWERABLE)):
        for q in qs:
            d = knowledge_base().similarity_search_with_score(q, k=1)[0][1]
            print(f"  {label} [{d:.4f}] {q}")
guard(_calibrate)
'''),

    code('''
# --- Self-check: Part C
check("your threshold is a number in the range these distances occupy",
      lambda: 0.2 < confident_below() < 2.0)
check("every answerable question clears it",
      lambda: all(answer_with_confidence(q)["best_distance"] < confident_below()
                  for q in ANSWERABLE),
      "too tight -- you are about to refuse questions the handbook can answer")
check("and the out-of-scope ones do not",
      lambda: not any(answer_with_confidence(q)["best_distance"] < confident_below()
                      for q in UNANSWERABLE),
      "too loose -- an off-topic question is being reported as a confident hit")
'''),

    code('''
# --- Run it for real -------------------------------------------------------
def run_the_bot():
    for q in ANSWERABLE[:2] + UNANSWERABLE[:1]:
        r = answer_with_confidence(q)
        flag = "OK " if r["confident"] else "LOW"
        print(f"[{flag} {r['best_distance']:.3f}] Q: {q}")
        print(f"           A: {r['answer'].strip()[:220]}\\n")

if llm_ready():
    guard(run_the_bot)
'''),

    code('''
score()
'''),

    md("""
## Your turn

1. **The threshold moves.** Re-split the corpus at `chunk_size=200` and re-run the
   calibration cell. The distances change, and your threshold is now wrong. Any number you
   tune against a corpus is a property of that corpus *and* that chunking &mdash; write that
   on the wall next to the number.
2. **Scoped assistants.** Build a leave-only and a tech-only chain from the same store,
   the way Lab 6.7 did. Ask the tech bot a leave question. It answers from tech chunks
   without complaining, which is worth seeing once.
3. **Refuse before you retrieve.** Right now the chain retrieves, spends the tokens, and
   then the model refuses. Use `confident_below()` to short-circuit *before* the model call.
   Measure what you saved on the three unanswerable questions &mdash; and then argue the
   other side, because a threshold that refuses early also refuses silently.
4. **The honest evaluation.** You have seven labelled questions in `ANSWERABLE` and
   `UNANSWERABLE`. Write the loop that scores your bot on both, and report two numbers, not
   one: how often it answered correctly, and how often it refused when it should have. A
   single accuracy number hides which half broke, and Module 7 is about exactly that.
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-6-01-hello-qwen-the-sandbox-model",  LAB1),
    ("lab-6-02-understanding-embeddings",      LAB2),
    ("lab-6-03-chromadb-basics",               LAB3),
    ("lab-6-04-loading-and-splitting",         LAB4),
    ("lab-6-05-langchain-and-chroma",          LAB5),
    ("lab-6-06-your-first-rag-chain",          LAB6),
    ("lab-6-07-rag-with-citations",            LAB7),
    ("lab-6-08-challenge-handbook-qa-bot",     LAB8),
]


def main():
    os.makedirs(SOLDIR, exist_ok=True)
    # Drop any notebook from a previous generation -- the lab names changed in the
    # 2026-09-10 port, and a stale file would still be picked up by the verifiers.
    keep = {name + ".ipynb" for name, _ in LABS}
    for folder in (LABDIR, SOLDIR):
        for fn in os.listdir(folder):
            if fn.endswith(".ipynb") and fn not in keep:
                os.remove(os.path.join(folder, fn))
                print("removed stale " + os.path.relpath(os.path.join(folder, fn), LABDIR))
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
