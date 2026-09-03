# Agentic Workflows & Multi-Agent Systems

**3 days · Advanced · 45% theory / 55% labs · 9 modules · 43 labs · capstone**

An instructor-led course by Rajesh Gheware (Gheware UniGPS Solutions LLP). Participants build,
evaluate and operate multi-agent systems, and — as much to the point — learn when *not* to build one.

## This is a compression of the flagship, not a new course

Source: `~/Training/courses/agentic-ai/` — 5 days, 15 sessions, 119 labs + solutions
(`brainupgrade-in/aiagentic-comp`). Nothing here is invented: modules map to flagship sessions,
labs are **selected from the existing 119** and delivered unchanged, and the stack is the flagship's.

> ⚠️ **The flagship lineage is deliberately NOT shown on the outline.** An earlier version carried a
> "condensed from the flagship / 119 labs / 5,000+ trained" header, per-module *Flagship S&lt;n&gt;*
> tags and a stack + prerequisites paragraph. **All of it was cut on request (2026-08-05).
> Do not reinstate it.** Provenance lives in this README. The lab-environment / participant-setup
> block at the foot of the outline is the one exception, added by request 2026-09-03.

### Brief → module → flagship session

| Brief item | Module | Flagship source |
|---|---|---|
| Agents vs Multi-Agent Systems | **M1** | S1 (Intro to Agentic AI), S9 (Multi-Agent Systems) |
| Agentic Planning and Reasoning | **M2** | S3 (Reasoning, Planning & Tool Use) |
| Agentic Memory and Perception | **M3** | S3, S6 (LangChain Agents & Memory) |
| MCP and tool calling | **M4** | S13 (MCP), S3, S6 |
| Multi-agent collaboration / orchestration | **M5** | S7 (LangGraph), S8 (Advanced LangGraph), S9 |
| Agentic RAG | **M6** | S5 (Building RAG Applications), S6 |
| Multi-agent system evaluation | **M7** | S12 (LangFuse Observability), S10 (Observability Fundamentals) |
| Agent Deployment and AgentOps | **M9** | S11 (Production Dev & Deployment), S10, S12 |
| *proposed addition* | **M8 — Safety & Guardrails** | S14 (AI Safety & Guardrails) |
| — | **Capstone** | S15 (Capstone Project) |

**Why M8 was added.** The requested topics teach how to *build* a multi-agent system but not how to
keep one from causing harm — the question a regulated organisation will actually be asked. Flagship
S14 already exists and drops straight in: prompt-injection detection, jailbreak resistance, output
validation, guardrails integration, red-team testing, safety monitoring. It costs nothing to add
because it is built and proven.

### What the 5-day drops to become 3

Stated openly on the outline, so the trade is visible rather than a silent trim:

- **S2 — AI Coding Assistants & Vibe Coding (9 labs).** Out of brief; this cohort is building
  agents, not learning agent-assisted development.
- **S4 — LangChain Fundamentals (8 labs).** Compressed to a fast on-ramp inside M1 — `create_agent`
  and enough LCEL to build. Advanced cohort; a full session would drag.
- **Docker/Kubernetes** compresses from most of a flagship day to the capstone's deployment path
  (S11 + S15 lab 06). Deployment survives because it is explicitly in the brief — and it is the part
  most competitors' agent courses skip.
- Roughly **30 of 119 labs** are selected, plus the two authored below.

## Settled decisions

- **Derive, don't invent.** The first draft (v1.0) was written from scratch and rejected. The
  correct basis is the flagship, whose labs are built, tested and delivered. **If a module needs
  content the flagship doesn't have, that is a signal to reconsider the module, not to write new labs.**
- **Flagship stack, unchanged** — Python 3.12, LangChain 1.x `create_agent`, LangGraph 1.x, MCP
  Python SDK, ChromaDB, FastAPI, LangFuse v4, OpenTelemetry, Docker, Kubernetes. Do not substitute;
  the labs are written against these versions (see the flagship `CLAUDE.md` migration table —
  `create_react_agent` and the Langfuse v2 callback API are removed, not deprecated).
- **Kubernetes deployment stays in.** It is in the brief, it is the trainer's differentiator, and
  the sandbox already provisions a cluster.
- **9 modules, 3 per day**, capstone on Day 3 — the flagship's rhythm, so timing estimates transfer
  directly (~60–75 min per module).

### v3.0 changes (2026-08-05) — six fixes applied

1. **Lab count corrected.** The page claimed "~30 labs" while the day strips listed 40. Now **43**,
   header matching the listing exactly. **Re-verify with the counting script below whenever the
   strips change.**
2. **LangGraph moved to Day 1; days rebalanced 13 / 17 / 13** (was 10 / 18 / 12). Day 1 Module 3
   became **"Memory, State & the LangGraph Substrate"** — StateGraph, reducers and checkpointing sit
   with memory, which is the same subject (agent state), leaving Day 2's Module 5 purely
   collaboration patterns. **Rationale: M5 previously compressed S7 + S8 + S9 (24 flagship labs)
   into one module with the entire LangGraph substrate as a single bullet — you cannot teach
   supervisor/worker orchestration to people who have never built a StateGraph.** Do not merge back.
3. **Measurement thread added.** Day 1 closes by building an eval set + baseline pass rate; Day 2
   scores the multi-agent graph against that same single agent; the capstone is accepted against the
   eval set inside a cost and latency budget, not on a demo. This is what makes "evaluation" a
   discipline rather than a LangFuse feature tour.
4. **Cost elevated to a spine.** Coordination tax in M1 → per-agent **model routing** and the 3–10×
   token multiple in M5 → cost/latency as **release gates** in M7 → budget as a capstone acceptance
   criterion.
5. **"When *not* to build multi-agent" promoted** from a bullet to a named take-home **decision
   rubric** (M1), and a **take-home artifacts** strip added to the outline.
6. **Capstone anchored** in payment-exception investigation (was a generic "observable multi-agent
   service"). Finance-plausible, fully synthetic, and it exercises every module including the
   approval gate.

Smaller: **Perception** strengthened in M3 (was mostly memory); **human-in-the-loop** deliberately
split — *mechanism* in M5, *control* in M8 — so the repetition is framed rather than accidental;
**MCP security & governance** moved from Day 2 to Day 3 M8, where it belongs thematically and where
it helps the day balance.

**Two labs are not straight selections from the 119** — the only new authoring sanctioned:

- **Day 1 · baseline eval set & pass rate** — *new*. The flagship folds evaluation into LangFuse
  (S12), which is too late and too tool-shaped for this thread. Needs a small harness: ~10 cases,
  assertions, a pass-rate number.
- **Day 2 · multi-agent vs. single-agent scorecard** — *reshaped* from S9 lab08 challenge. Must be
  written so that **the graph is allowed to lose** on cost or latency; that outcome is the lesson,
  not a lab failure.

## Delivery risks

⚠️ **External LLM API access is the biggest delivery risk.** Days 2–3 of the flagship run on the
Groq free tier with each participant registering their own key. A cohort behind a corporate network
may be unable to reach a public LLM endpoint, or to self-register for one. **Confirm before dates
are committed.** Fallbacks, in order: a sandbox-hosted model (what this course uses — see the
outline's setup block); Ollama-only (works, but small local models weaken the tool-calling and
multi-agent labs — the flagship's own measurements show `llama3.2:1b` at 33–50% tool-routing
accuracy vs. 100% for `llama-3.3-70b`); or a client-provided OpenAI-compatible internal endpoint.

**LangFuse Cloud** — Day 3 evaluation labs send traces to each participant's own cloud project. An
offline fallback exists (`langfuse-server.sh`, local FastAPI + SQLite) but must be dry-run before
delivery if egress is blocked.

**Density.** Three flagship sessions per day was designed with a 5-day pace and an on-ramp. This
track opens at the agent loop and adds MCP, multi-agent and RAG all on Day 2. If the cohort is
weaker on Python than assumed, Day 2 is where it shows. Have the drop order ready.

## Contents

| Path | Role |
|---|---|
| `course-outline-agentic-workflows-multi-agent-systems.html` | **The course outline — sole source of truth.** CSS inlined, 9 modules + lab strips + capstone + take-aways + participant setup block. **Prints to exactly 1 A4 page.** |
| `presentation/` | Delivery decks — self-contained single-file slide runners, no CDN, hand-authored SVG |
| `hands-on/module-N/` | Lab notebooks + `solutions/` + `_generators/` (single-source generation) + `RUNTIME.md` |
| `resources/video-resources.html` | Post-session curated video resources, mapped to the modules |

Removed 2026-09-03 (in git history at `ae94c05`): the `.md` outline, the exported outline PDF, and a
separate participant-setup sheet whose content now lives at the foot of the HTML. Render a PDF only
when one needs to be sent, and don't commit it.

## Verification

**The outline must stay 1 A4 page — hard constraint, verify after every edit.** Run as one block
from inside this folder (prints `PAGES: 1`):

```bash
nohup python3 -m http.server 8905 >/dev/null 2>&1 &
SRV=$!
sleep 1.5
google-chrome --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf=/tmp/check.pdf \
  "http://localhost:8905/course-outline-agentic-workflows-multi-agent-systems.html?r=$RANDOM"
python3 -c "import re;d=open('/tmp/check.pdf','rb').read();print('PAGES:',len(re.findall(rb'/Type\s*/Page[^s]',d)))"
kill $SRV
```

Print font is `11.2px` in the `@media print` block — the lever if content grows. Measured on v3.1
content (two-column setup block): `11.2px` fits, `11.6px` spills to 2 pages. **Little slack — any
addition needs a matching removal.** Before the setup block, v3.0 fit at `12.2px`. Cache-bust the
URL (`?r=…`). Kill via the captured `$SRV` PID, **not** `pkill -f "http.server"` — that pattern also
matches other servers. If you sweep font sizes in a loop, **check the file is left at a value that
fits**; the loop leaves the last value tested.

**Lab-count check** — the header claim and the day strips must agree (they didn't in v2.0):

```bash
python3 - <<'EOF'
import re
s = open("course-outline-agentic-workflows-multi-agent-systems.html").read()
tot = 0
for m in re.finditer(r'<strong class="lead">(Day \d Labs)</strong>(.*?)</div>', s, re.S):
    n = len([x for x in re.sub(r'<[^>]+>', '', m.group(2)).split('&middot;') if x.strip()])
    tot += n; print(m.group(1), n)
print("listed:", tot, "| header:", re.search(r'(\d+) labs', s).group(1))
EOF
```

**Labs** — rebuild and check both directions (offline; no cluster, no model):

```bash
hands-on/module-1/_generators/regenerate.sh
```

Every solution must score full marks, and every untouched lab must survive *Run All* with no
uncaught exception.

## Build state

- [x] Outline v3.1 — 9 modules, 43 labs, capstone, participant setup block
- [x] Module 1 deck — 51 slides, 26 SVG diagrams, 8-question knowledge check
- [x] Module 1 labs — 5 notebooks + solutions, generated and verified
- [x] Post-session video resources — 57 links, all verified live
- [ ] Modules 2–9 decks
- [ ] Modules 2–9 labs
- [ ] **Author the two non-selected labs** (Day 1 baseline eval set; Day 2 multi-agent vs.
      single-agent scorecard). These carry the measurement thread — if they slip, the thread is just
      a claim on a page
- [ ] Confirm the eval set and the fixed task set are **the same artifact** across Day 1 → Day 2 →
      capstone; the whole thread breaks if they drift
- [ ] Pin the exact lab list (43 of 119, + the 2 above) with per-lab timings, and confirm it fits
      three 3-session days — **Day 2 at 17 labs is still the densest**
- [ ] Dry-run the selected labs end to end on the sandbox image as a standalone 3-day sequence — the
      flagship's labs assume the days before them; check nothing selected depends on a dropped
      session (**S4 is the likely trap**)
- [ ] Agreed drop order for when the clock slips — Day 2 is the pressure point
