#!/usr/bin/env python3
"""servetune analyze: raw benchmark results -> cost at SLO -> ranked recommendations -> report.

Reads runs/<name>/ produced by audit.py (vLLM `bench serve` result JSONs) and writes:
  results.json   one row per candidate x rate, plus the per-candidate summary
  report.md      the client-facing findings
  report.html    same content with latency/throughput charts (inline SVG, no dependencies)

Definitions used throughout the report:
  SLO pass        p95 TTFT <= slo.ttft_p95_ms and p95 TPOT <= slo.tpot_p95_ms and every request completed
  capacity        the highest tested request rate that passes the SLO (measured, not interpolated)
  cost / 1M tok   GPU $/h x GPUs per replica / (output tokens/s at capacity x 3600) x 1e6
  fleet           replicas needed for the client's peak request rate = ceil(peak / capacity)
"""
from __future__ import annotations

import argparse
import html
import json
import math
import tomllib
from pathlib import Path


def pick(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def load_rows(run: Path, cfg: dict) -> list[dict]:
    rows = []
    n_expected = int(cfg["workload"].get("num_prompts", 400))
    for cdir in sorted(p for p in run.iterdir() if p.is_dir()):
        for f in sorted(cdir.glob("rate_*.json")):
            d = json.loads(f.read_text())
            rate_s = f.stem.split("_", 1)[1]
            rate = float(rate_s) if rate_s not in ("inf", "0", "0.0") else math.inf
            rows.append({
                "candidate": cdir.name, "rate": rate,
                "completed": pick(d, "completed", default=0),
                "req_s": pick(d, "request_throughput", default=0.0),
                "out_tok_s": pick(d, "output_throughput", default=0.0),
                "total_tok_s": pick(d, "total_token_throughput", default=0.0),
                "ttft_p50": pick(d, "median_ttft_ms", "p50_ttft_ms"),
                "ttft_p95": pick(d, "p95_ttft_ms", "p99_ttft_ms"),
                "tpot_p50": pick(d, "median_tpot_ms", "p50_tpot_ms"),
                "tpot_p95": pick(d, "p95_tpot_ms", "p99_tpot_ms"),
                "itl_p95": pick(d, "p95_itl_ms", "p99_itl_ms"),
                "goodput_req_s": pick(d, "request_goodput"),
                "n_expected": int(d.get("num_prompts") or n_expected),
            })
    return rows


def passes(r: dict, slo: dict) -> bool:
    return (r["completed"] >= r["n_expected"] and r["ttft_p95"] is not None and r["tpot_p95"] is not None
            and r["ttft_p95"] <= slo["ttft_p95_ms"] and r["tpot_p95"] <= slo["tpot_p95_ms"])


def summarize(rows: list[dict], cfg: dict) -> list[dict]:
    slo = cfg["slo"]
    gpu = cfg["hardware"]
    usd_h = float(gpu["usd_per_gpu_hour"]) * int(gpu.get("gpus_per_replica", 1))
    client = cfg.get("client", {})
    peak = float(client.get("peak_requests_per_s", 0) or 0)
    out = []
    for name in dict.fromkeys(r["candidate"] for r in rows):
        rs = sorted((r for r in rows if r["candidate"] == name), key=lambda r: r["rate"])
        ok = [r for r in rs if passes(r, slo) and math.isfinite(r["rate"])]
        best = max(ok, key=lambda r: r["req_s"]) if ok else None
        sat = next((r for r in rs if not math.isfinite(r["rate"])), rs[-1] if rs else None)
        s = {"candidate": name, "capacity_req_s": best["req_s"] if best else 0.0,
             "capacity_out_tok_s": best["out_tok_s"] if best else 0.0,
             "capacity_rate_tested": best["rate"] if best else None,
             "saturation_out_tok_s": sat["out_tok_s"] if sat else 0.0,
             "saturation_goodput_req_s": sat["goodput_req_s"] if sat else None,
             "ttft_p95_at_capacity": best["ttft_p95"] if best else None,
             "tpot_p95_at_capacity": best["tpot_p95"] if best else None}
        s["usd_per_1m_out_tok"] = (usd_h / (s["capacity_out_tok_s"] * 3600) * 1e6) if s["capacity_out_tok_s"] else None
        if peak and s["capacity_req_s"]:
            s["replicas_for_peak"] = math.ceil(peak / s["capacity_req_s"])
            s["monthly_usd"] = s["replicas_for_peak"] * usd_h * 730
        out.append(s)
    base = next((s for s in out if s["candidate"] == cfg.get("baseline", "baseline")), None)
    for s in out:
        if base and base.get("usd_per_1m_out_tok") and s.get("usd_per_1m_out_tok"):
            s["cost_vs_baseline"] = s["usd_per_1m_out_tok"] / base["usd_per_1m_out_tok"] - 1
            s["capacity_vs_baseline"] = s["capacity_req_s"] / base["capacity_req_s"] - 1
        if base and base.get("monthly_usd") and s.get("monthly_usd"):
            s["monthly_savings_usd"] = base["monthly_usd"] - s["monthly_usd"]
    return sorted(out, key=lambda s: (s["usd_per_1m_out_tok"] is None, s["usd_per_1m_out_tok"] or 0))


# ---------- report ----------

def fmt(x, nd=0, pct=False, money=False):
    if x is None:
        return "n/a"
    if pct:
        return f"{x * 100:+.0f}%"
    if money:
        return f"${x:,.{nd}f}"
    return f"{x:,.{nd}f}"


def svg_curves(rows: list[dict], slo: dict, title: str) -> str:
    """Latency (p95 TTFT) against delivered request throughput, one line per candidate."""
    W, H, L, B, T, R = 640, 320, 56, 40, 28, 150
    pts = [r for r in rows if r["ttft_p95"] is not None]
    if not pts:
        return ""
    xmax = max(r["req_s"] for r in pts) * 1.05 or 1
    ymax = min(max(r["ttft_p95"] for r in pts), slo["ttft_p95_ms"] * 4) * 1.05
    sx = lambda v: L + (W - L - R) * v / xmax
    sy = lambda v: H - B - (H - B - T) * min(v, ymax) / ymax
    palette = ["#2a6fdb", "#e0663a", "#1f9d6b", "#9b51e0", "#c9a227", "#d94f8a", "#4a5568", "#00a3a3"]
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}" class="chart">',
             f'<text x="{L}" y="18" class="ct">{html.escape(title)}</text>',
             f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" class="ax"/><line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" class="ax"/>',
             f'<line x1="{L}" y1="{sy(slo["ttft_p95_ms"]):.1f}" x2="{W-R}" y2="{sy(slo["ttft_p95_ms"]):.1f}" class="slo"/>',
             f'<text x="{W-R-4}" y="{sy(slo["ttft_p95_ms"])-4:.1f}" class="sl" text-anchor="end">TTFT SLO {slo["ttft_p95_ms"]} ms</text>',
             f'<text x="{(L+W-R)/2:.0f}" y="{H-8}" class="al" text-anchor="middle">delivered requests / s</text>',
             f'<text x="14" y="{(T+H-B)/2:.0f}" class="al" transform="rotate(-90 14 {(T+H-B)/2:.0f})" text-anchor="middle">p95 TTFT (ms)</text>']
    for i in range(5):
        xv = xmax * i / 4
        yv = ymax * i / 4
        parts.append(f'<text x="{sx(xv):.0f}" y="{H-B+14}" class="tk" text-anchor="middle">{xv:.1f}</text>')
        parts.append(f'<text x="{L-6}" y="{sy(yv)+4:.0f}" class="tk" text-anchor="end">{yv:,.0f}</text>')
    for i, name in enumerate(dict.fromkeys(r["candidate"] for r in pts)):
        c = palette[i % len(palette)]
        rs = sorted((r for r in pts if r["candidate"] == name), key=lambda r: r["req_s"])
        d = " ".join(f"{'M' if j == 0 else 'L'}{sx(r['req_s']):.1f},{sy(r['ttft_p95']):.1f}" for j, r in enumerate(rs))
        parts.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2"/>')
        for r in rs:
            parts.append(f'<circle cx="{sx(r["req_s"]):.1f}" cy="{sy(r["ttft_p95"]):.1f}" r="3" fill="{c}"/>')
        parts.append(f'<rect x="{W-R+12}" y="{T+i*18}" width="10" height="10" fill="{c}"/>'
                     f'<text x="{W-R+28}" y="{T+i*18+9}" class="lg">{html.escape(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def workload_line(w: dict) -> str:
    if w.get("mode") == "replay":
        import csv as _csv
        import statistics as _st
        try:
            tr = list(_csv.DictReader(open(w["trace"])))
            ins = [int(r["input_tokens"]) for r in tr]
            outs = [int(r["output_tokens"]) for r in tr]
            span = float(tr[-1]["arrival_s"]) if tr else 0
            groups = len({r.get("prefix_id") for r in tr if r.get("prefix_id")})
            return (f"- Workload: replay of {len(tr)} requests spanning {span:.0f} s (median {int(_st.median(ins))} input / "
                    f"{int(_st.median(outs))} output tokens, {groups} shared-prefix groups of {w.get('shared_prefix_len', 0)} "
                    f"tokens), played at {w['speeds']}x speed")
        except Exception:
            return f"- Workload: replay of {w.get('trace')} at {w.get('speeds')}x speed"
    return (f"- Workload: ~{w['input_len']} input / ~{w['output_len']} output tokens (range ratio {w.get('range_ratio', 0.5)}), "
            f"{w.get('shared_prefix_len', 0)} shared prefix tokens, {w.get('num_prompts', 400)} requests per rate, rates {w['request_rates']}")


def write_reports(run: Path, cfg: dict, rows: list[dict], summ: list[dict]) -> None:
    slo, hw, eng, w = cfg["slo"], cfg["hardware"], cfg["engine"], cfg["workload"]
    client = cfg.get("client", {})
    base_name = cfg.get("baseline", "baseline")
    base = next((s for s in summ if s["candidate"] == base_name), None)
    best = next((s for s in summ if s.get("usd_per_1m_out_tok")), None)
    notes = {c["name"]: c.get("why", "") for c in cfg["candidates"]}
    head = cfg.get("report", {}).get("title", f"Inference cost audit: {eng['model']} on {hw['gpu']}")
    md = [f"# {head}", ""]
    if base and best and base.get("usd_per_1m_out_tok") and best is not base:
        md += [f"**Headline.** Of the tested configurations, `{best['candidate']}` serves "
               f"{fmt(best['capacity_req_s'], 2)} requests/s within the SLO against {fmt(base['capacity_req_s'], 2)} "
               f"for the current setup, which cuts cost per million output tokens from "
               f"{fmt(base['usd_per_1m_out_tok'], 2, money=True)} to {fmt(best['usd_per_1m_out_tok'], 2, money=True)} "
               f"({fmt(best.get('cost_vs_baseline'), pct=True)})."]
        if best.get("monthly_savings_usd") is not None:
            md[-1] += (f" At the stated peak of {client.get('peak_requests_per_s')} requests/s that is "
                       f"{best['replicas_for_peak']} replicas instead of {base['replicas_for_peak']}, about "
                       f"{fmt(best['monthly_savings_usd'], money=True)} a month.")
        md.append("")
    md += ["## Setup", "",
           (f"- Engine: {eng['name']} {eng.get('version', '')}, model `{eng['model']}`, baseline flags `{eng.get('baseline_flags', '')}`"
            if eng["name"] != "external" else f"- Endpoints: client-provided, model `{eng['model']}`"),
           f"- Hardware: {hw.get('gpus_per_replica', 1)} x {hw['gpu']} per replica at ${hw['usd_per_gpu_hour']}/GPU-hour",
           workload_line(w),
           f"- SLO: p95 TTFT <= {slo['ttft_p95_ms']} ms and p95 TPOT <= {slo['tpot_p95_ms']} ms", "",
           "## Results, cheapest first", "",
           "| Config | Capacity at SLO (req/s) | vs current | Output tok/s at capacity | $ per 1M output tok | vs current | p95 TTFT / TPOT at capacity (ms) |",
           "|---|---|---|---|---|---|---|"]
    for s in summ:
        if not s["capacity_req_s"]:
            md.append(f"| `{s['candidate']}` | none passed | n/a | n/a | n/a | n/a | see notes |")
            continue
        md.append(f"| `{s['candidate']}` | {fmt(s['capacity_req_s'], 2)} | {fmt(s.get('capacity_vs_baseline'), pct=True)} | "
                  f"{fmt(s['capacity_out_tok_s'])} | {fmt(s['usd_per_1m_out_tok'], 3, money=True)} | "
                  f"{fmt(s.get('cost_vs_baseline'), pct=True)} | {fmt(s['ttft_p95_at_capacity'])} / {fmt(s['tpot_p95_at_capacity'], 1)} |")
    md += ["", "## What each config changes", ""]
    for c in cfg["candidates"]:
        what = c.get("flags") or (f"endpoint {c['base_url']}" if c.get("base_url") else "") or "(baseline)"
        if c.get("model") and c.get("model") != eng.get("model"):
            what = f"model {c['model']}" + (f", {c['flags']}" if c.get("flags") else "")
        md.append(f"- `{c['name']}`: `{what}`. {notes.get(c['name'], '')}")
    md += ["", "## Method and limits", "",
           "servetune reports capacity as the highest tested request rate that met both SLOs with every request "
           "completed. True capacity lies between that rate and the next one tested. Synthetic prompts come from "
           "the engine's own random dataset with the stated length ranges and shared prefix. In a client audit, "
           "servetune replays the client's own traffic instead. Quality effects of quantization are outside this "
           "run's scope, so check any FP8 change on your own eval set before rollout.", ""]
    for extra in cfg.get("report", {}).get("notes", []):
        md.append(f"- {extra}")
    (run / "report.md").write_text("\n".join(md) + "\n")

    chart = svg_curves(rows, slo, "p95 time to first token vs delivered throughput")
    body = "\n".join(md)
    # minimal Markdown to HTML for this report's own constructs
    html_lines, in_table = [], False
    for line in body.splitlines():
        esc = html.escape(line)
        esc = esc.replace("**Headline.**", "<strong>Headline.</strong>")
        import re
        esc = re.sub(r"`([^`]*)`", r"<code>\1</code>", esc)
        if line.startswith("|"):
            cells = [c.strip() for c in esc.strip("|").split("|")]
            if set(line.replace("|", "").strip()) <= {"-"}:
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                html_lines.append("<table>")
                in_table = True
            html_lines.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            html_lines.append("</table>")
            in_table = False
        if line.startswith("# "):
            html_lines.append(f"<h1>{esc[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{esc[3:]}</h2>")
            if line.startswith("## Results"):
                html_lines.append(chart)
        elif line.startswith("- "):
            html_lines.append(f"<li>{esc[2:]}</li>")
        elif line.strip():
            html_lines.append(f"<p>{esc}</p>")
    if in_table:
        html_lines.append("</table>")
    css = """
:root{--bg:#fff;--fg:#1a1f2b;--mut:#5b6474;--line:#e3e7ee;--acc:#2a6fdb;--code:#f3f5f9}
@media (prefers-color-scheme:dark){:root{--bg:#11151c;--fg:#e6e9ef;--mut:#9aa3b2;--line:#2a313d;--acc:#6ea0ff;--code:#1b212b}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;max-width:900px;margin:32px auto;padding:0 16px}
h1{font-size:24px;margin:0 0 12px}h2{font-size:18px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}
th{color:var(--mut);font-weight:600}code{background:var(--code);padding:1px 4px;border-radius:4px;font-size:12.5px}
li{margin:4px 0}.chart{width:100%;height:auto;margin:8px 0 4px}.chart .ax{stroke:var(--mut)}.chart .slo{stroke:#d94f4f;stroke-dasharray:4 3}
.chart text{fill:var(--mut);font-size:11px}.chart .ct{font-size:13px;fill:var(--fg)}.chart .sl{fill:#d94f4f}
"""
    (run / "report.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                                     f"<title>{html.escape(head)}</title><style>{css}</style></head><body>{''.join(html_lines)}</body></html>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = tomllib.loads((run / "audit.toml").read_text())
    rows = load_rows(run, cfg)
    summ = summarize(rows, cfg)
    (run / "results.json").write_text(json.dumps({"rows": rows, "summary": summ}, indent=1, default=str))
    write_reports(run, cfg, rows, summ)
    for s in summ:
        print(f"{s['candidate']:22s} cap={s['capacity_req_s']:.2f} req/s  $/1M={s['usd_per_1m_out_tok']}  vs_base={s.get('cost_vs_baseline')}")


if __name__ == "__main__":
    main()
