import json
import os

NS = 'agenticai'
DS = {"type": "prometheus", "uid": "prometheus-nuc"}
panels = []
_id = [0]

def nid():
    _id[0] += 1
    return _id[0]

def row(title, y):
    panels.append({"type": "row", "title": title, "collapsed": False, "id": nid(),
                   "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": []})

def tgt(expr, refid="A", legend=None, table=False, instant=False, step=None):
    t = {"expr": expr, "refId": refid, "datasource": DS}
    if legend is not None: t["legendFormat"] = legend
    if table: t["format"] = "table"
    if instant: t["instant"] = True
    if step: t["step"] = step
    return t

def stat(title, expr, x, y, w=4, h=4, unit="short", decimals=0, steps=None, desc=None, mappings=None):
    p = {"type": "stat", "title": title, "id": nid(), "datasource": DS,
         "gridPos": {"h": h, "w": w, "x": x, "y": y},
         "targets": [tgt(expr)],
         "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals,
                                      "mappings": mappings or [],
                                      "color": {"mode": "thresholds" if steps else "fixed",
                                                **({} if steps else {"fixedColor": "text"})},
                                      "thresholds": {"mode": "absolute",
                                                     "steps": steps or [{"color": "text", "value": None}]}},
                         "overrides": []},
         "options": {"colorMode": "background" if steps else "value", "graphMode": "none",
                     "justifyMode": "center", "textMode": "auto",
                     "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}
    if desc: p["description"] = desc
    return p

def ts(title, targets, x, y, w, h, unit="short", decimals=None, steps=None, thresh_line=False, desc=None, maxv=None):
    d = {"unit": unit, "custom": {"fillOpacity": 8, "lineWidth": 1, "showPoints": "never"}}
    if decimals is not None: d["decimals"] = decimals
    if steps:
        d["thresholds"] = {"mode": "absolute", "steps": steps}
        if thresh_line: d["custom"]["thresholdsStyle"] = {"mode": "dashed"}
    if maxv is not None: d["max"] = maxv
    p = {"type": "timeseries", "title": title, "id": nid(), "datasource": DS,
         "gridPos": {"h": h, "w": w, "x": x, "y": y}, "targets": targets,
         "fieldConfig": {"defaults": d, "overrides": []},
         "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                     "tooltip": {"mode": "multi", "sort": "desc"}}}
    if desc: p["description"] = desc
    return p

def table(title, targets, transformations, x, y, w, h, overrides=None, sort=None, desc=None):
    p = {"type": "table", "title": title, "id": nid(), "datasource": DS,
         "gridPos": {"h": h, "w": w, "x": x, "y": y}, "targets": targets,
         "transformations": transformations,
         "fieldConfig": {"defaults": {"custom": {"align": "auto", "inspect": False}},
                         "overrides": overrides or []},
         "options": {"showHeader": True, "footer": {"show": False},
                     **({"sortBy": [sort]} if sort else {})}}
    if desc: p["description"] = desc
    return p

STS_R = f'kube_statefulset_replicas{{namespace="{NS}"}}'
STS_K = f'kube_statefulset_status_replicas_ready{{namespace="{NS}"}}'
JUP = f'{{namespace="{NS}", container="jupyter"}}'

# ---------------------------------------------------------------- overview
row("Sandbox overview", 0)
panels += [
    stat("Sandboxes defined", f'count({STS_R})', 0, 1),
    stat("UP", f'count({STS_K} == {STS_R} and {STS_R} > 0) or vector(0)', 4, 1,
         steps=[{"color": "dark-gray", "value": None}, {"color": "green", "value": 1}],
         desc="All desired replicas ready."),
    stat("DOWN", f'count({STS_K} < {STS_R} and {STS_R} > 0) or vector(0)', 8, 1,
         steps=[{"color": "green", "value": None}, {"color": "red", "value": 1}],
         desc="Scaled up but not ready — the one that needs you during a session."),
    stat("Scaled to 0", f'count({STS_R} == 0) or vector(0)', 12, 1,
         steps=[{"color": "dark-gray", "value": None}],
         desc="Expected between cohorts; a problem during one."),
    stat("Container restarts (24h)",
         f'sum(increase(kube_pod_container_status_restarts_total{{namespace="{NS}"}}[24h])) or vector(0)', 16, 1,
         decimals=0, steps=[{"color": "green", "value": None}, {"color": "orange", "value": 1},
                            {"color": "red", "value": 3}]),
    stat("OOMKilled containers",
         f'count(kube_pod_container_status_last_terminated_reason{{namespace="{NS}", reason="OOMKilled"}} == 1) or vector(0)',
         20, 1, steps=[{"color": "green", "value": None}, {"color": "red", "value": 1}],
         desc="Last termination reason was OOMKilled. Accumulated Jupyter kernels are the usual cause."),
]

# ------------------------------------------------------------ status table
row("Per-sandbox status", 5)
panels.append(table(
    "Sandbox health",
    [tgt(STS_R, "A", table=True, instant=True),
     tgt(STS_K, "B", table=True, instant=True),
     tgt(f'sum by (statefulset) (label_replace(kube_pod_container_status_restarts_total{{namespace="{NS}"}}, "statefulset", "$1", "pod", "(.*)-[0-9]+"))',
         "C", table=True, instant=True)],
    [{"id": "seriesToColumns", "options": {"byField": "statefulset"}},
     {"id": "organize", "options": {
         "excludeByName": {f"{k}{s}": True for k in ("Time", "__name__", "job", "instance", "namespace",
                                                     "cluster", "source_cluster", "uid")
                           for s in ("", " 1", " 2", " 3")},
         "renameByName": {"statefulset": "Sandbox", "Value #A": "Desired",
                          "Value #B": "Ready", "Value #C": "Restarts"}}},
     {"id": "calculateField", "options": {"alias": "Status", "mode": "binary",
                                          "binary": {"left": "Ready", "operator": "/", "right": "Desired"},
                                          "replaceFields": False}}],
    0, 6, 24, 12,
    overrides=[{"matcher": {"id": "byName", "options": "Status"},
                "properties": [
                    {"id": "custom.cellOptions", "value": {"type": "color-background", "mode": "basic"}},
                    {"id": "mappings", "value": [
                        {"type": "value", "options": {"0": {"text": "DOWN", "color": "red", "index": 0},
                                                      "1": {"text": "UP", "color": "green", "index": 1}}},
                        {"type": "special", "options": {"match": "nan",
                                                        "result": {"text": "SCALED DOWN", "color": "dark-gray", "index": 2}}}]},
                    {"id": "custom.width", "value": 140}]},
               {"matcher": {"id": "byName", "options": "Sandbox"},
                "properties": [{"id": "custom.width", "value": 180}]}],
    sort={"displayName": "Sandbox", "desc": False},
    desc="Desired vs ready replicas per sandbox, with lifetime container restarts."))

# --------------------------------------------------------------- timeline
row("Availability over time", 18)
panels.append(ts("Sandbox availability", [tgt(
    f'clamp_min({STS_K}, 0) / clamp_min({STS_R}, 1) * ({STS_R} > bool 0) + ({STS_R} == bool 0) * -1',
    "A", legend="{{statefulset}}", step="60")], 0, 19, 24, 10))
panels[-1].update({
    "type": "state-timeline",
    "description": "1 = ready, 0 = not ready, -1 = scaled to 0.",
    "fieldConfig": {"defaults": {
        "color": {"mode": "thresholds"},
        "custom": {"fillOpacity": 70, "lineWidth": 0, "spanNulls": False},
        "mappings": [
            {"type": "value", "options": {"0": {"text": "DOWN", "color": "red", "index": 0},
                                          "1": {"text": "UP", "color": "green", "index": 1}}},
            {"type": "range", "options": {"from": -1, "to": 0,
                                          "result": {"text": "SCALED DOWN", "color": "dark-gray", "index": 2}}}],
        "thresholds": {"mode": "absolute", "steps": [
            {"color": "red", "value": None}, {"color": "dark-gray", "value": -0.5},
            {"color": "green", "value": 0.5}]}}, "overrides": []},
    "options": {"mergeValues": True, "showValue": "never", "alignValue": "center", "rowHeight": 0.8,
                "legend": {"displayMode": "list", "placement": "bottom", "showLegend": False},
                "tooltip": {"mode": "single"}}})

# --------------------------------------------------------------- downtime
row("Downtime", 29)
DOWN = f'(({STS_R} - {STS_K}) > bool 0)'
panels += [
    stat("Total sandbox downtime (7d)", f'sum(sum_over_time({DOWN}[7d:1m])) or vector(0)',
         0, 30, w=6, h=8, unit="m", steps=[{"color": "green", "value": None},
                                           {"color": "orange", "value": 30}, {"color": "red", "value": 120}],
         desc="Sandbox-minutes where a scaled-up sandbox was not ready. Scaled-to-0 does not count."),
    table("Downtime by sandbox (7d)",
          [tgt(f'sum by (statefulset) (sum_over_time({DOWN}[7d:1m])) > 0', "A", table=True, instant=True)],
          [{"id": "organize", "options": {
              "excludeByName": {"Time": True},
              "renameByName": {"statefulset": "Sandbox", "Value": "Downtime (min)"}}},
           {"id": "sortBy", "options": {"fields": {}, "sort": [{"field": "Downtime (min)", "desc": True}]}}],
          6, 30, 18, 8),
]

# ------------------------------------------------------- memory & cpu (kubelet)
row("Memory & CPU per sandbox", 38)
MEM = f'sum by (pod) (container_memory_working_set_bytes{JUP})'
LIM = f'sum by (pod) (kube_pod_container_resource_limits{{namespace="{NS}", container="jupyter", resource="memory"}})'
panels += [
    ts("Jupyter memory — share of its limit",
       [tgt(f'{MEM} / on(pod) {LIM}', "A", legend="{{pod}}")], 0, 39, 12, 9,
       unit="percentunit", decimals=1, maxv=1,
       steps=[{"color": "green", "value": None}, {"color": "orange", "value": 0.8}, {"color": "red", "value": 0.92}],
       thresh_line=True,
       desc="Working set over the pod's memory limit (2560Mi). Above the red line the kernel culler is losing to accumulated notebooks."),
    ts("CPU used per sandbox",
       [tgt(f'sum by (pod) (rate(container_cpu_usage_seconds_total{JUP}[5m]))', "A", legend="{{pod}}")],
       12, 39, 12, 9, unit="short", decimals=2, desc="Cores, 5m rate."),
    table("Memory right now",
          [tgt(MEM, "A", table=True, instant=True), tgt(LIM, "B", table=True, instant=True),
           tgt(f'{MEM} / on(pod) {LIM}', "C", table=True, instant=True)],
          [{"id": "joinByField", "options": {"byField": "pod", "mode": "outer"}},
           {"id": "organize", "options": {
               "excludeByName": {f"Time{s}": True for s in ("", " 1", " 2", " 3")},
               "renameByName": {"pod": "Sandbox", "Value #A": "Working set",
                                "Value #B": "Limit", "Value #C": "Share"}}},
           {"id": "sortBy", "options": {"fields": {}, "sort": [{"field": "Share", "desc": True}]}}],
          0, 48, 14, 9,
          overrides=[{"matcher": {"id": "byRegexp", "options": "Working set|Limit"},
                      "properties": [{"id": "unit", "value": "bytes"}]},
                     {"matcher": {"id": "byName", "options": "Share"},
                      "properties": [{"id": "unit", "value": "percentunit"}, {"id": "decimals", "value": 1},
                                     {"id": "custom.cellOptions",
                                      "value": {"type": "color-background", "mode": "gradient"}},
                                     {"id": "thresholds", "value": {"mode": "absolute", "steps": [
                                         {"color": "green", "value": None}, {"color": "orange", "value": 0.8},
                                         {"color": "red", "value": 0.92}]}}]}]),
    table("OOMKills — last termination reason",
          [tgt(f'kube_pod_container_status_last_terminated_reason{{namespace="{NS}", reason="OOMKilled"}} == 1',
               "A", table=True, instant=True)],
          [{"id": "organize", "options": {
              "excludeByName": {k: True for k in ("Time", "__name__", "job", "instance", "namespace",
                                                  "cluster", "source_cluster", "uid", "Value", "container")},
              "renameByName": {"pod": "Sandbox", "reason": "Reason"}}}],
          14, 48, 10, 9,
          desc="A row here means that container was last killed for memory. It clears when the pod is replaced."),
]

# ------------------------------------------------------------- quota + node
row("Namespace quota and the DGX node", 57)
QH = f'kube_resourcequota{{namespace="{NS}", resource="limits.memory", type="hard"}}'
QU = f'kube_resourcequota{{namespace="{NS}", resource="limits.memory", type="used"}}'
panels += [
    ts("Namespace memory limits committed, against the quota",
       [tgt(QU, "A", legend="committed"), tgt(QH, "B", legend="quota (hard)")],
       0, 58, 12, 8, unit="bytes",
       desc="participants x per-sandbox limit must stay under the hard quota. This is the ceiling on cohort size."),
    stat("Quota committed", f'{QU} / ignoring(type) {QH}', 12, 58, w=4, h=4, unit="percentunit", decimals=1,
         steps=[{"color": "green", "value": None}, {"color": "orange", "value": 0.85}, {"color": "red", "value": 0.97}]),
    stat("Node memory available", f'node_memory_MemAvailable_bytes{{cluster="spark"}}', 16, 58, w=4, h=4,
         unit="bytes", decimals=1),
    stat("Node load1", f'node_load1{{cluster="spark"}}', 20, 58, w=4, h=4, decimals=2),
    stat("Root fs available", f'node_filesystem_avail_bytes{{cluster="spark", mountpoint="/"}}',
         12, 62, w=4, h=4, unit="bytes", decimals=1),
    stat("GPU utilisation", "DCGM_FI_DEV_GPU_UTIL", 16, 62, w=4, h=4, unit="percent"),
    stat("GPU temp", "DCGM_FI_DEV_GPU_TEMP", 20, 62, w=4, h=4, unit="celsius",
         steps=[{"color": "green", "value": None}, {"color": "orange", "value": 75}, {"color": "red", "value": 85}]),
    ts("DGX — GPU utilisation and power",
       [tgt("DCGM_FI_DEV_GPU_UTIL", "A", legend="GPU util %"),
        tgt("DCGM_FI_DEV_MEM_COPY_UTIL", "B", legend="memory copy %"),
        tgt("DCGM_FI_DEV_POWER_USAGE", "C", legend="power (W)")],
       0, 66, 12, 8, desc="Zero while the vLLM failover leg is scaled down."),
    ts("Node memory available and namespace working set",
       [tgt('node_memory_MemAvailable_bytes{cluster="spark"}', "A", legend="node available"),
        tgt(f'sum(container_memory_working_set_bytes{{namespace="{NS}"}})', "B", legend="agenticai working set")],
       12, 66, 12, 8, unit="bytes"),
]

# ---------------------------------------------------------- pods and storage
row("Pods and storage", 74)
panels += [
    table("Pods not Running",
          [tgt(f'kube_pod_status_phase{{namespace="{NS}", phase=~"Pending|Failed|Unknown"}} == 1',
               "A", table=True, instant=True)],
          [{"id": "organize", "options": {
              "excludeByName": {k: True for k in ("Time", "__name__", "job", "instance", "namespace",
                                                  "cluster", "source_cluster", "uid", "Value")},
              "renameByName": {"pod": "Pod", "phase": "Phase"}}}],
          0, 75, 12, 8, desc="Empty is the healthy state."),
    table("PVC status",
          [tgt(f'kube_persistentvolumeclaim_status_phase{{namespace="{NS}"}} == 1',
               "A", table=True, instant=True)],
          [{"id": "organize", "options": {
              "excludeByName": {k: True for k in ("Time", "__name__", "job", "instance", "namespace",
                                                  "cluster", "source_cluster", "uid", "Value",
                                                  "storageclass", "volumename")},
              "renameByName": {"persistentvolumeclaim": "PVC", "phase": "Phase"}}},
           {"id": "sortBy", "options": {"fields": {}, "sort": [{"field": "PVC", "desc": False}]}}],
          12, 75, 12, 8, desc="Retained between cohorts — a PVC that is not Bound means a sandbox will not come back."),
]

dash = {
    "title": "Agentic AI Workshop — Sandbox Monitor",
    "uid": "agenticai-sandbox-monitor",
    "description": "Health of the 31 agenticai JupyterLab sandboxes on Spark K3s. Queries the NUC Prometheus (uid prometheus-nuc) — Spark's own Prometheus is agent-mode and answers no queries.",
    "tags": ["agenticai", "workshop", "sandbox"],
    "timezone": "Asia/Kolkata",
    "schemaVersion": 39,
    "editable": True,
    "graphTooltip": 1,
    "refresh": "30s",
    "time": {"from": "now-6h", "to": "now"},
    "templating": {"list": []},
    "annotations": {"list": [{"builtIn": 1, "datasource": {"type": "datasource", "uid": "grafana"},
                              "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
                              "name": "Annotations & Alerts", "type": "dashboard"}]},
    "panels": panels,
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agenticai-sandbox-dashboard.json")
json.dump(dash, open(out, "w"), indent=2)
open(out, "a").write("\n")
print("panels:", len(panels), "->", out)
