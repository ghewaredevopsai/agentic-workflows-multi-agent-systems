# Agentic Workflows & Multi-Agent Systems

**3 days · Advanced · 9 modules · 49 labs · capstone**

Course material for an instructor-led course by Rajesh Gheware (Gheware UniGPS Solutions LLP).
Participants build, evaluate and operate multi-agent systems — and learn when *not* to build one.

This repository holds everything participants use: the outline, the delivery decks, the lab
notebooks with their solutions, and a post-session video list.

## Modules

| Day | Modules |
|---|---|
| 1 | 1 Agents vs. Multi-Agent Systems &middot; 2 Agentic Planning & Reasoning &middot; 3 LangGraph: Stateful Agent Workflows |
| 2 | 4 Tool Calling & MCP &middot; 5 Multi-Agent Collaboration & Orchestration &middot; 6 Agentic RAG |
| 3 | 7 Multi-Agent System Evaluation &middot; 8 Safety & Guardrails &middot; 9 Deployment & AgentOps |

Day 3 closes with a capstone: a payment-exception investigation service, accepted against a
45-case eval set inside a cost and latency budget — not on a demo.

## Layout

| Path | What it is |
|---|---|
| `course-outline-agentic-workflows-multi-agent-systems.html` | The course outline — one page, self-contained |
| `presentation/module-N-*.html` | Delivery decks. Self-contained single-file slide runners, no CDN |
| `hands-on/module-N/` | Five lab notebooks per module, plus `solutions/`, `_generators/` and `RUNTIME.md` |
| `hands-on/capstone/` | Brief, eval set, `acceptance.py` (the gate), a starter and a reference service |
| `presentation/faq-day-1-*.html` | Day 1 FAQ and quick revision — searchable, printable |
| `resources/video-resources.html` | Post-session curated videos, mapped to the modules |

Everything is plain HTML and Jupyter notebooks. There is no build, no package manager and no test
framework at the repository level.

## How the labs work

Each notebook is self-contained and self-grading:

- `BLANK` marks a placeholder you fill in.
- Self-check cells print `[PASS]` / `[FAIL]` / `[TODO]` and a `Self-check: passed/total` tally.
- **Self-checks assert on the objects you build** — a bound tool, a compiled graph, a validated
  schema — so they are deterministic and do not need the model. A flaky endpoint cannot break your
  score, and you can work through a whole module while your LLM access is being sorted out.
- Cells marked *Run it for real* put your code in front of the model. That is the half worth
  watching; if the endpoint is unreachable they print how to fix it rather than crashing, so
  *Run All* is always safe on an untouched notebook.
- One synthetic case file runs through every lab in a module, and later labs carry forward
  earlier labs' code. Each module picks a domain everyone already knows, so the only unfamiliar
  thing in a notebook is the framework: a support-ticket queue in Module 1, an employee help desk
  in Module 2, HR leave requests in Module 3, a company handbook in Module 6, and payment
  exceptions in Module 4 and from Module 7 to the capstone. **Module 5 is the exception**: each of
  its three labs is a whole system, so each carries its own small case file &mdash; a customer
  support desk, a vendor research brief, and a production incident.

**Two labs are deliberately different.** Labs 4.1 and 4.2 are *walkthroughs*: there is nothing to
fill in and nothing to score, and their solution notebooks are the same files. They exist so you
watch real MCP servers do real work before the rest of Module 4 asks you to build one. Run the
cells in order and read the output &mdash; that is the lab.

- **4.1** connects your agent to a Jira board and has it raise a ticket from a sentence.
- **4.2** adds a second server to *the same* agent &mdash; your Langfuse project &mdash; and leaves
  you with a library of prompts for getting answers out of your own traces. Keep that config; it is
  genuinely useful on Day 3.

Some of their steps run in a JupyterLab terminal rather than in the notebook, because `opencode` is
a terminal agent and watching the tool calls scroll past is most of the point.

Open `hands-on/module-N/index.html` for the lab landing page of a module.

## Runtime

Notebooks run in a browser-based JupyterLab sandbox provided for the course; the LLM is
pre-configured there, so nothing needs setting up. `hands-on/module-1/RUNTIME.md` is the contract
if you want to run them elsewhere: Python 3.12, the package list, and the three environment
variables the notebooks read —

```
LAB_LLM_BASE_URL   # OpenAI-compatible base URL, ending in /v1
LAB_LLM_MODEL      # model name the gateway serves
OPENAI_API_KEY     # any non-empty value if the gateway does not authenticate
```

`OPENAI_BASE_URL` / `OPENAI_MODEL` are accepted as fallbacks. **No endpoint is hardcoded anywhere.**
If a value is unset, every live cell prints the `export` lines it needs and continues.

Module 9 and the capstone additionally read `APP_NAMESPACE` and `APP_HOST`.

## Working on this material

Preview any HTML by serving the folder — decks and lab pages do not load from `file://` cleanly:

```bash
python3 -m http.server 8099 --bind 127.0.0.1     # then open http://127.0.0.1:8099/<path>
```

**Never edit an `.ipynb` by hand.** Labs and solutions are generated from a single source, so a
blank cannot drift from the answer that grades it — edit `_generators/gen_labs.py` and re-run:

```bash
hands-on/module-N/_generators/regenerate.sh
```

That rebuilds both and runs the two verifiers: every solution must score full marks, and every
untouched lab must survive *Run All* with no uncaught exception. Both run offline — no cluster and
no model needed. Decks are checked with `python3 presentation/check-deck-layout.py <deck.html>`.

## Stack

Python 3.12, LangChain 1.x (`create_agent`), LangGraph 1.x, MCP Python SDK, ChromaDB, FastAPI,
LangFuse v4, OpenTelemetry, Docker, Kubernetes.

## Contact

Gheware UniGPS Solutions LLP · [devops.gheware.com](https://devops.gheware.com) ·
training@gheware.com
