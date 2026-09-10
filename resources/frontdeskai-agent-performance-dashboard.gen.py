import json
import os

DS = {"type": "prometheus", "uid": "prometheus-nuc"}
NS = 'namespace=~"$namespace"'
JOB = f'job="frontdeskai-participants", {NS}'
panels, _id = [], [0]
def nid(): _id[0] += 1; return _id[0]

def row(title, y):
    panels.append({"type": "row", "title": title, "collapsed": False, "id": nid(),
                   "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": []})

def tgt(expr, refid="A", legend=None, table=False, instant=False):
    t = {"expr": expr, "refId": refid, "datasource": DS}
    if legend is not None: t["legendFormat"] = legend
    if table: t["format"] = "table"
    if instant: t["instant"] = True
    return t

def stat(title, expr, x, y, w=4, h=4, unit="short", decimals=1, steps=None, desc=None, graph="none"):
    return {"type": "stat", "title": title, "id": nid(), "datasource": DS,
            "description": desc, "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "targets": [tgt(expr)],
            "fieldConfig": {"defaults": {
                "unit": unit, "decimals": decimals, "mappings": [],
                "color": {"mode": "thresholds"} if steps else {"mode": "fixed", "fixedColor": "text"},
                "thresholds": {"mode": "absolute", "steps": steps or [{"color": "text", "value": None}]}},
                "overrides": []},
            "options": {"colorMode": "background" if steps else "value", "graphMode": graph,
                        "justifyMode": "center", "textMode": "auto",
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}

def ts(title, targets, x, y, w, h, unit="short", decimals=None, steps=None, thresh_line=False, desc=None, stack=False):
    d = {"unit": unit, "custom": {"fillOpacity": 10 if not stack else 40, "lineWidth": 2,
                                  "showPoints": "auto", "pointSize": 5,
                                  **({"stacking": {"mode": "normal"}} if stack else {})}}
    if decimals is not None: d["decimals"] = decimals
    if steps:
        d["thresholds"] = {"mode": "absolute", "steps": steps}
        if thresh_line: d["custom"]["thresholdsStyle"] = {"mode": "dashed"}
    return {"type": "timeseries", "title": title, "id": nid(), "datasource": DS,
            "description": desc, "gridPos": {"h": h, "w": w, "x": x, "y": y}, "targets": targets,
            "fieldConfig": {"defaults": d, "overrides": []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                        "tooltip": {"mode": "multi", "sort": "desc"}}}

def bargauge(title, expr, x, y, w, h, unit="short", decimals=2, desc=None, legend="{{agent}}"):
    return {"type": "bargauge", "title": title, "id": nid(), "datasource": DS,
            "description": desc, "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "targets": [tgt(expr, "A", legend=legend, instant=True)],
            "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals,
                                         "color": {"mode": "continuous-BlPu"},
                                         "thresholds": {"mode": "absolute",
                                                        "steps": [{"color": "text", "value": None}]}},
                            "overrides": []},
            "options": {"orientation": "horizontal", "displayMode": "gradient",
                        "showUnfilled": True, "valueMode": "color", "minVizWidth": 8,
                        "sortBy": "Value", "sortOrder": "Descending",
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}

def text(title, md, x, y, w, h):
    return {"type": "text", "title": title, "id": nid(),
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "options": {"mode": "markdown", "content": md}}

CNT = f'frontdeskai_llm_call_duration_seconds_count{{{NS}}}'
SUMD = f'frontdeskai_llm_call_duration_seconds_sum{{{NS}}}'
TOK = f'frontdeskai_llm_tokens_total{{{NS}}}'
REQC = f'frontdeskai_request_duration_seconds_count{{{NS}}}'
REQS = f'frontdeskai_request_duration_seconds_sum{{{NS}}}'
CAT = f'frontdeskai_category_total{{{NS}}}'
REQB = f'frontdeskai_request_duration_seconds_bucket{{{NS}}}'
CALLB = f'frontdeskai_llm_call_duration_seconds_bucket{{{NS}}}'
ESC = f'frontdeskai_escalations_total{{{NS}}}'
FB = f'frontdeskai_fallbacks_total{{{NS}}}'
ERR = f'frontdeskai_agent_errors_total{{{NS}}}'

# ============================================ 1. the headline number
row("Overall — and whether it is a number or an anecdote", 0)
panels += [
    stat("Requests handled", f'sum({CAT}) or vector(0)', 0, 1, w=3, decimals=0,
         desc="Total since the pod started. Counters reset on redeploy — that is your experiment boundary."),
    stat("Mean end-to-end latency", f'sum({REQS}) / sum({REQC})', 3, 1, w=3, unit="s", decimals=2,
         desc="Total time over total requests, since pod start. Always correct, and the number to lead with; the p95 beside it is what a gate should actually use."),
    stat("p95 end-to-end", f'histogram_quantile(0.95, sum by (le) (rate({REQB}[$__range])))', 6, 1, w=3,
         unit="s", decimals=2,
         steps=[{"color": "green", "value": None}, {"color": "orange", "value": 20}, {"color": "red", "value": 40}],
         desc="Over the dashboard's time range. Needs an app image built 2026-09-11 or later — before that the histogram used millisecond-scale buckets for second-valued observations and every percentile was a bucket edge. TWO WAYS THIS READS WRONG: a pod still on an older image, and a time range that straddles the upgrade — histogram_quantile over a window containing BOTH boundary sets returns a bucket edge (10000), not a latency. If this reads 10000, shorten the range until it only covers the current image."),
    stat("Tokens per request", f'sum({TOK}) / sum({CAT})', 9, 1, w=3, decimals=0,
         desc="The bill for one user question, across every agent it touched."),
    stat("LLM calls per request", f'sum({CNT}) / sum({CAT})', 12, 1, w=3, decimals=2,
         desc="Fan-out. One question, this many model calls. The cheapest thing to tune is usually the number of calls, not the model."),
    stat("Escalation rate", f'(sum({ESC}) or vector(0)) / sum({CAT})', 15, 1, w=3, unit="percentunit", decimals=1,
         steps=[{"color": "green", "value": None}, {"color": "orange", "value": 0.2}, {"color": "red", "value": 0.4}],
         desc="Requests the manager had to take. Rising means the workers are losing ground."),
    stat("Fallback rate", f'(sum({FB}) or vector(0)) / sum({CAT})', 18, 1, w=3, unit="percentunit", decimals=1,
         steps=[{"color": "green", "value": None}, {"color": "orange", "value": 0.1}, {"color": "red", "value": 0.25}],
         desc="Static template answers — the QA gate refused the worker twice. Every one of these is a case for your eval set."),
    stat("Agent errors", f'sum({ERR}) or vector(0)', 21, 1, w=3, decimals=0,
         steps=[{"color": "green", "value": None}, {"color": "red", "value": 1}],
         desc="Not a slow request — a request that did not work. Attribution question two in the deck."),
    ts("End-to-end latency — mean, p50, p95",
       [tgt(f'sum(rate({REQS}[$__rate_interval])) / sum(rate({REQC}[$__rate_interval]))', "A", legend="mean"),
        tgt(f'histogram_quantile(0.5, sum by (le) (rate({REQB}[$__rate_interval])))', "B", legend="p50"),
        tgt(f'histogram_quantile(0.95, sum by (le) (rate({REQB}[$__rate_interval])))', "C", legend="p95")],
       0, 5, 12, 8, unit="s", decimals=2,
       desc="Three lines, not one. The gap between p50 and p95 IS the interval — a mean on its own hides it, and a single run tells you nothing about either. Watch the shape, not the last value. (These use a short rate window, so they are unaffected by a histogram-boundary change further back in the range.)"),
    ts("Requests per minute by category",
       [tgt(f'sum by (category) (rate({CAT}[$__rate_interval]) * 60)', "A", legend="{{category}}")],
       12, 5, 12, 8, decimals=2, stack=True,
       desc="Which desks are actually being exercised. A category with no traffic is a category you have not evaluated."),
]

# ============================================ 2. slowest vs dearest
row("The slowest step and the dearest step are not the same step", 13)
panels += [
    bargauge("Where the time goes — mean seconds per call",
             f'sum by (agent) ({SUMD}) / sum by (agent) ({CNT})', 0, 14, 8, 11, unit="s", decimals=2,
             desc="Mean duration of one call to this agent, since pod start. The tallest bar is where the wall clock goes."),
    bargauge("Where the tail is — p95 seconds per call",
             f'histogram_quantile(0.95, sum by (le, agent) (rate({CALLB}[$__range])))', 8, 14, 8, 11,
             unit="s", decimals=2,
             desc="The same agents at the 95th percentile, over the dashboard's time range. An agent whose mean is fine and whose p95 is not is the one your users complain about."),
    bargauge("Where the tokens go — total tokens",
             f'sum by (agent) ({TOK})', 16, 14, 8, 11, decimals=0,
             desc="Cumulative tokens attributed to each agent. The tallest bar is where the bill goes — and it is usually not the same agent as either bar to the left."),
]

# ============================================ 3. trajectory
row("Trajectory — not just the answer, but how it got there", 25)
panels += [
    bargauge("LLM calls by agent — every ReAct iteration is its own bar",
             f'sum by (agent) ({CNT})', 0, 26, 14, 10, decimals=0,
             desc="Agent names encode the path: <worker>_react_iter_<n> is one think-act-observe round, _worker_final is the answer, _fb is the tool-calling error fallback. Reading down the bars tells you how many steps each worker took."),
    stat("Calls that hit the 3-iteration cap",
         f'sum({CNT.replace("{"+NS+"}", "{"+NS+', agent=~".*_react_iter_2"}')}) or vector(0)',
         14, 26, w=5, h=5, decimals=0,
         steps=[{"color": "green", "value": None}, {"color": "orange", "value": 1}],
         desc="MAX_TOOL_ITERATIONS is 3, so a call tagged _react_iter_2 used the last one it had. A worker that routinely reaches this is not converging — look at its tools and its prompt, not at the model."),
    stat("Tool-calling error fallbacks",
         f'sum({CNT.replace("{"+NS+"}", "{"+NS+', agent=~".*_fb"}')}) or vector(0)',
         19, 26, w=5, h=5, decimals=0,
         steps=[{"color": "green", "value": None}, {"color": "red", "value": 1}],
         desc="The worker's tool call raised, and it answered without tools. This is the deck's second attribution question — a tool returning an error, not the model hallucinating."),
    {"type": "piechart", "title": "Requests by category", "id": nid(), "datasource": DS,
     "description": "Where the supervisor sent the traffic. Compare against what you believe your traffic mix is.",
     "gridPos": {"h": 5, "w": 10, "x": 14, "y": 31},
     "targets": [tgt(f'sum by (category) ({CAT})', "A", legend="{{category}}", instant=True)],
     "fieldConfig": {"defaults": {"unit": "short", "decimals": 0}, "overrides": []},
     "options": {"legend": {"displayMode": "table", "placement": "right", "showLegend": True,
                            "values": ["value", "percent"]},
                 "pieType": "donut", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}},
]

# ============================================ 4. the tuning table
row("Per-agent scorecard — the tuning worksheet", 36)
panels.append({
    "type": "table", "title": "Agent scorecard", "id": nid(), "datasource": DS,
    "description": "One row per agent. Sort by any column. 'Share of time' and 'Share of tokens' disagreeing is the whole point of the module: fix the one you were actually asked to fix.",
    "gridPos": {"h": 12, "w": 24, "x": 0, "y": 37},
    "targets": [
        tgt(f'sum by (agent) ({CNT})', "A", table=True, instant=True),
        tgt(f'sum by (agent) ({SUMD}) / sum by (agent) ({CNT})', "B", table=True, instant=True),
        tgt(f'sum by (agent) ({TOK})', "C", table=True, instant=True),
        tgt(f'sum by (agent) ({TOK}) / sum by (agent) ({CNT})', "D", table=True, instant=True),
        tgt(f'sum by (agent) ({SUMD}) / scalar(sum({SUMD}))', "E", table=True, instant=True),
        tgt(f'sum by (agent) ({TOK}) / scalar(sum({TOK}))', "F", table=True, instant=True),
    ],
    "transformations": [
        {"id": "joinByField", "options": {"byField": "agent", "mode": "outer"}},
        {"id": "organize", "options": {
            "excludeByName": {f"Time{s}": True for s in ("", " 1", " 2", " 3", " 4", " 5", " 6")},
            "renameByName": {"agent": "Agent", "Value #A": "Calls", "Value #B": "Mean seconds",
                             "Value #C": "Tokens", "Value #D": "Tokens per call",
                             "Value #E": "Share of time", "Value #F": "Share of tokens"}}},
        {"id": "sortBy", "options": {"fields": {}, "sort": [{"field": "Share of tokens", "desc": True}]}},
    ],
    "fieldConfig": {"defaults": {"custom": {"align": "auto", "inspect": False}}, "overrides": [
        {"matcher": {"id": "byName", "options": "Mean seconds"},
         "properties": [{"id": "unit", "value": "s"}, {"id": "decimals", "value": 2}]},
        {"matcher": {"id": "byName", "options": "Tokens per call"},
         "properties": [{"id": "decimals", "value": 0}]},
        {"matcher": {"id": "byRegexp", "options": "Share of .*"},
         "properties": [{"id": "unit", "value": "percentunit"}, {"id": "decimals", "value": 1},
                        {"id": "custom.cellOptions",
                         "value": {"type": "color-background", "mode": "gradient"}},
                        {"id": "thresholds", "value": {"mode": "absolute", "steps": [
                            {"color": "green", "value": None}, {"color": "orange", "value": 0.3},
                            {"color": "red", "value": 0.5}]}}]},
    ]},
    "options": {"showHeader": True, "footer": {"show": False}},
})

# ============================================ 5. gates
row("Gates — the thresholds you agree in a calm week", 49)
panels += [
    ts("p95 end-to-end latency against its ceiling",
       [tgt(f'histogram_quantile(0.95, sum by (le) (rate({REQB}[$__rate_interval])))', "A", legend="p95"),
        tgt(f'sum(rate({REQS}[$__rate_interval])) / sum(rate({REQC}[$__rate_interval]))', "B", legend="mean")],
       0, 50, 8, 8, unit="s", decimals=2, thresh_line=True,
       steps=[{"color": "green", "value": None}, {"color": "red", "value": 20}],
       desc="The deck gates on p95, not the mean, because the mean hides the requests people actually notice. The dashed line is a PLACEHOLDER ceiling of 20s — replace it with a number your team agreed before anybody was under pressure."),
    ts("Tokens per request against its ceiling",
       [tgt(f'sum(rate({TOK}[$__rate_interval])) / sum(rate({CAT}[$__rate_interval]))', "A", legend="tokens/request")],
       8, 50, 8, 8, decimals=0, thresh_line=True,
       steps=[{"color": "green", "value": None}, {"color": "red", "value": 6000}],
       desc="Placeholder ceiling of 6000. Cost per case is the number that moves quietly when a prompt grows."),
    ts("Errors and fallbacks",
       [tgt(f'sum(rate({ERR}[$__rate_interval]) * 60) or vector(0)', "A", legend="agent errors/min"),
        tgt(f'sum(rate({FB}[$__rate_interval]) * 60) or vector(0)', "B", legend="fallbacks/min"),
        tgt(f'sum(rate({ESC}[$__rate_interval]) * 60) or vector(0)', "C", legend="escalations/min")],
       16, 50, 8, 8, decimals=2,
       desc="The behaviour column. An agent error is not a slow request — it is a request that did not work."),
    text("How to use this dashboard", """
**This is a report. A gate is a decision.** Nothing here blocks a deploy. To turn a number on this
page into a gate, pick the panel, agree the ceiling while nobody is under pressure, and fail the
build on it.

**Read every number as a range.** These panels are means since the pod started. Redeploy resets the
counters, which is what makes a redeploy a clean experiment boundary: change one thing, replay the
same cases, compare. One run of ten cases cannot separate 70% from 80% — repeat the run, widen the
case set, and report the range.

**When something is wrong, ask which step caused it, in this order:** did the right evidence reach
the agent (retrieval) · did a tool return an error the agent ignored (the `_fb` bars) · did the
supervisor route it to the wrong desk (category donut) · did a handoff drop a constraint · and only
then, was the evidence there and the model still got it wrong (generation). Four of those five get
reported as "the agent hallucinated" and are not.

**Two panels you should expect to disagree:** *Where the time goes* and *Where the tokens go*.
Caching the slow agent makes the app fast and no cheaper; trimming the expensive agent's context
makes it cheaper and nobody notices the speed. Know which one you were asked to fix.

**Metrics say which agent. Traces say why.** Nothing on this page can tell you whether the
right evidence reached the agent — only the span tree can. Go to **Explore → Tempo** and search
the service name `frontdeskai-agenticaiu<N>`; one Tempo serves all 31 participants, which is why
the namespace is in the service name. `chat.send` is the parent span, `llm.<agent>` the children.

*Percentiles need an app image built 2026-09-11 or later; before that the histogram used
millisecond-scale buckets for values recorded in seconds, and every quantile was a bucket edge.
The means are correct on any image, which is why they are still shown alongside.*
""".strip(), 0, 58, 24, 9),
]

# ============================================ 6. pod health
row("App and pod health", 67)
panels += [
    stat("Instances up", f'sum(up{{{JOB}}}) or vector(0)', 0, 68, w=6, decimals=0,
         steps=[{"color": "red", "value": None}, {"color": "green", "value": 1}]),
    stat("Pod restarts (1h)",
         f'sum(increase(kube_pod_container_status_restarts_total{{{NS}}}[1h])) or vector(0)', 6, 68, w=6,
         decimals=0, steps=[{"color": "green", "value": None}, {"color": "orange", "value": 1},
                            {"color": "red", "value": 3}],
         desc="A restart resets every counter on this page."),
    stat("Resident memory", f'sum(process_resident_memory_bytes{{{JOB}}})', 12, 68, w=6, unit="bytes",
         desc="The participant namespace grants 1Gi of memory limits in total, so this is the whole budget."),
    stat("Process CPU", f'sum(rate(process_cpu_seconds_total{{{JOB}}}[$__rate_interval]))', 18, 68, w=6,
         decimals=3, desc="Cores. The app is I/O-bound on the model, so this stays low even when latency is high."),
    ts("Resident memory against the 1Gi namespace ceiling",
       [tgt(f'process_resident_memory_bytes{{{JOB}}}', "A", legend="{{namespace}}")],
       0, 72, 24, 7, unit="bytes", thresh_line=True,
       steps=[{"color": "green", "value": None}, {"color": "red", "value": 1073741824}],
       desc="The namespace ResourceQuota grants 1Gi of limits.memory in total, which one pod at a 1Gi limit consumes entirely."),
]

dash = {
    "title": "FrontDesk AI — Agent Performance",
    "uid": "frontdeskai-agent-performance",
    "description": "Per-agent evaluation and tuning for the FrontDesk AI app each participant deploys into their own namespace. Built against Module 7: outcome and trajectory, the slowest step vs the dearest step, and gates rather than reports. Queries the NUC Prometheus (uid prometheus-nuc); Spark's own Prometheus is agent-mode.",
    "tags": ["agenticai", "workshop", "frontdeskai", "evaluation"],
    "timezone": "Asia/Kolkata",
    "schemaVersion": 39,
    "editable": True,
    "graphTooltip": 1,
    "refresh": "30s",
    "time": {"from": "now-3h", "to": "now"},
    "templating": {"list": [{
        "name": "namespace", "label": "Participant", "type": "query",
        "datasource": DS,
        "query": {"query": 'label_values(up{job="frontdeskai-participants"}, namespace)', "refId": "var"},
        "definition": 'label_values(up{job="frontdeskai-participants"}, namespace)',
        "refresh": 2, "sort": 7, "multi": True, "includeAll": True, "allValue": ".*",
        "current": {"text": ["All"], "value": ["$__all"]}, "options": [],
    }]},
    "annotations": {"list": [{"builtIn": 1, "datasource": {"type": "datasource", "uid": "grafana"},
                              "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
                              "name": "Annotations & Alerts", "type": "dashboard"}]},
    "panels": panels,
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontdeskai-agent-performance-dashboard.json")
json.dump(dash, open(out, "w"), indent=2)
open(out, "a").write("\n")
print("panels:", len(panels), "->", out)
