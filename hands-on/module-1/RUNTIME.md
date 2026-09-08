# Runtime prerequisites — Day 1 labs (Modules 1–3)

The notebooks run on the **K3s cluster, namespace `agenticai`**. The cluster admin provides the
runtime; this file is the contract the notebooks expect. It lives under `module-1/` for historical
reasons and covers **all of Day 1** — Modules 2 and 3 need exactly the same packages. Nothing here is hardcoded in a notebook —
the endpoint is read from the environment, so the admin can point it anywhere without editing labs.

## 1. Python packages

| Package | Needed by | Notes |
|---|---|---|
| `langchain` (1.x) | **every Day 1 lab** | `langchain.agents.create_agent`, `langchain_core.tools.tool`, message types, `trim_messages`, `ChatPromptTemplate`, `StrOutputParser` |
| `langchain-openai` | **every Day 1 lab** | `ChatOpenAI`, pointed at the in-cluster gateway |
| `langgraph` (1.x) | 1.1, 1.2, and **all of Module 3** | `StateGraph`, `START`/`END`, `InMemorySaver`; Module 3 builds and checkpoints real graphs |
| `pydantic` (2.x) | most labs from 1.2 on | `BaseModel` / `Field` for `with_structured_output` and `response_format` |
| `typing_extensions` | Module 3 | `TypedDict` for graph state |

**Nothing new is required for the Day 1 rebuild.** Every import above is already pinned in the lab
image's `requirements.txt` and present in the running sandbox. ⚠️ The pins are floors and very
loose (`langchain>=0.3` while **1.4.0** is what is installed and what the labs are written
against). A future image rebuild that resolves to a 2.x would break `create_agent`'s signature
silently. Raising those floors to the 1.x line is worth doing before the next rebuild.

Verified against the sandbox image on 2026-09-08: langchain **1.4.0**, langchain-core 1.6.2,
langchain-openai 1.6.0, langgraph 1.2.11, pydantic 2.11.7, Python 3.12.11.

⚠️ **This changed on 2026-09-08.** Module 1 was rebuilt so the participant writes real LangChain
and LangGraph code in every lab, and every lab now makes live model calls. What used to be true —
"graded cells are stdlib-only and never call a model" — is no longer the design. The invariant
that replaced it:

> **Self-checks assert on framework _objects_; only the "Run it for real" cells invoke the model.**

Constructing a `@tool`, a Pydantic schema, a `bind_tools` list or a compiled graph needs no
endpoint, so the self-checks stay deterministic and verify offline. Invoking any of them does.
That is why `verify.py` still runs with the gateway down, and why `verify_live.py` exists.

⚠️ **Lab 1.4 is the one lab in Module 1 that looks at tokens**, and it does so as one column among
several — the lab's finding is about accuracy and context fragmentation, not cost. It is also the
most expensive lab in the module: three architectures plus a handoff audit is roughly fifty agent
runs, about 80 seconds and ~11k tokens per participant.

⚠️ **`trim_messages(token_counter=llm)` raises `NotImplementedError` on this model.** `tiktoken`
has no encoding for `qwen36-35b-a3b-lab`, so the model class cannot count. Lab 1.2 teaches the fix
(`count_tokens_approximately`) deliberately, because it is what every self-hosted or
gateway-served model does.

⚠️ **`response_format` can return `None`.** `create_agent(..., response_format=X)` asks the model
to finish by calling a tool that fills `X`; a model that replies in prose instead leaves
`result["structured_response"]` as `None` and raises nothing. It is intermittent on this model.
Labs 1.3 and 1.4 both handle it explicitly and 1.3 teaches it.

⚠️ **`create_agent` takes `system_prompt=`, not `prompt=`.** The sandbox ships **langchain
1.4.0**; `prompt=` was the 1.0-preview name and now raises `TypeError: create_agent() got an
unexpected keyword argument 'prompt'`. Pin-sensitive — re-check this signature after any
langchain bump, because the graded cells will not catch it (they never call the model).

Python **3.12** (the course stack). A participant whose LLM access is not yet wired can still fill
in every blank and pass every self-check — the object-level assertions do not need the gateway —
but they will miss the "Run it for real" cells, which are the point of the module.

## 2. Environment variables

Set these in the notebook pod (JupyterHub `singleuser.extraEnv`, a ConfigMap, or the image):

```
LAB_LLM_BASE_URL   # OpenAI-compatible base URL, ending in /v1
LAB_LLM_MODEL      # model name the gateway serves
OPENAI_API_KEY     # any non-empty value if the in-cluster gateway does not authenticate
```

`OPENAI_BASE_URL` / `OPENAI_MODEL` are accepted as fallbacks, so if the image already exports the
standard OpenAI variables nothing further is needed.

An in-cluster gateway is reachable at the usual service DNS, for example:

```
LAB_LLM_BASE_URL=http://<service>.agenticai.svc.cluster.local:<port>/v1
```

**These values are deliberately not baked into the notebooks.** If they are unset, every live cell
prints the two `export` lines it needs and continues — it never raises.

Module 9's labs and the capstone additionally read `APP_NAMESPACE` and `APP_HOST` — the namespace
a participant deploys into and the hostname it is published at. Both are injected into the sandbox;
if they are unset, nothing breaks and every cell that would have used them says so instead.

### If the model reasons (measured 2026-09-04 on `qwen36-35b-a3b-lab`)

The served model thinks before it answers, and the thinking is billed as **completion tokens**.
This is not a detail — it changes latency, cost and one common bug, by more than an order of
magnitude:

| | latency | completion tokens | of which reasoning |
|---|---|---|---|
| default | 24.1s | 980 | 955 |
| `extra_body={"chat_template_kwargs": {"enable_thinking": false}}` | **0.7s** | **29** | 0 |
| `/no_think` appended to the prompt | 8.3s | 363 | 338 |
| `reasoning_effort="low"` | 28.6s | 1430 | 1405 |

All four returned the same one-line JSON answer. Three things follow.

1. **`max_tokens` is a trap.** Cap it at 200 to control cost and the reply is not truncated —
   `content` comes back as `None` with `finish_reason="length"`, so a service that does not check
   for it silently produces nothing. Budget ~1500, or turn thinking off.
2. **The standard knob does not work here.** `reasoning_effort` is the OpenAI-documented parameter
   and this backend ignores it, returning *more* tokens than the default. The one that works is
   vendor-specific and travels through `extra_body`.
3. **Turning it off is a trade, not a free win.** On the capstone's own eval set, thinking off took
   the reference service from 25.2s to 2.1s per case and from $0.0011 to $0.0002 — and dropped its
   accuracy to 46.7% until the prompts were rewritten as an explicit ordered procedure.

Throughput is shared. Output tokens are generated serially per request, so concurrent callers divide
the gateway's token throughput between them: the same 45-case run that takes 86 seconds alone will
take considerably longer when thirty people start one at once.

## 3. Egress

None required. The labs reach only the in-cluster LLM service; the case file is synthetic and
inline, and no lab downloads anything.

## 4. Writable path

Each lab creates `/tmp/awmas-lab-1-0N/` for scratch output. `/tmp` inside the pod is enough — no
persistent volume is needed for Module 1.

## 5. Smoke test

From a notebook terminal in the `agenticai` namespace:

```bash
python3 -c "
import os
from langchain_openai import ChatOpenAI
m = ChatOpenAI(model=os.environ['LAB_LLM_MODEL'],
               base_url=os.environ['LAB_LLM_BASE_URL'],
               api_key=os.environ.get('OPENAI_API_KEY', 'sandbox'))
print(m.invoke('Reply with just: OK').content)
"
```

`OK` means the live cells will work. If it fails, the graded cells still do.

## 6. Verifying the labs themselves

```bash
_generators/regenerate.sh
```

Rebuilds all ten notebooks from the single source and checks both directions: every solution must
score full marks, and every untouched lab must survive *Run All* without an uncaught exception.
Runs offline — no cluster, no model, no network (it does start a local Jupyter kernel).

**That is no longer sufficient on its own.** Since the labs became framework-forward, the offline
pass proves only that the objects are built correctly. To prove the agents actually run:

```bash
_generators/verify_live.py            # all five solutions, real kernel, live model
_generators/verify_live.py lab-1-04   # just one
```

Run it **on the cluster** (or anywhere `LAB_LLM_*` points at a live gateway). It fails on any
`[FAIL]`, any `[TODO]`, any cell error, and — importantly — on any cell that *degraded* to
"model unavailable" or "not importable here", which is how a broken live cell used to hide. A full
Module 1 pass is about 130 seconds and roughly a cent.

`verify_labs.py` executes each lab **twice — plain `exec()` and a real Jupyter kernel — and requires
the two to agree** on (todo, pass, fail). Agreement is the actual check. A notebook must not depend
on IPython-specific semantics, and the reason is concrete: IPython predefines `_`, `__` and `___` as
its output history, initialised to `""`. A blank spelled with underscores is therefore a *defined
empty string* in a notebook and raises no `NameError`, so `[TODO]` silently becomes `[FAIL]` and a
blank used as a loop guard never stops its loop. That is why the blank marker is **`BLANK`**, which
is undefined under both. Do not change it back.
