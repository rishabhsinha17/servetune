#!/usr/bin/env python3
"""Release diff: what changed in vLLM / SGLang that should trigger a re-tune.

For every release in a window, pull the release notes through the GitHub API,
keep the lines that touch serving performance or defaults, tag each with the
knob it affects, and pull out any quantified claim ("1.4x", "+18% throughput").
Output is a one-page Markdown note per release plus a JSON summary, which is
the per-release deliverable of the re-tune retainer.

Usage:
  python3 release_diff.py --repo vllm-project/vllm --since 2026-07-01 --out notes/
  python3 release_diff.py --repo sgl-project/sglang --since 2026-07-01 --out notes/
Requires the `gh` CLI (authenticated) on PATH.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

# knob -> patterns. A line can hit several knobs.
KNOBS = {
    "defaults": r"\bdefault(s|ed)?\b.*\b(enable|disable|change|switch|now|turn)|\b(enable|disable|switch|turn)\w*\b.*\bby default\b|\bnew default\b",
    "scheduler": r"\bschedul(er|ing)\b|\basync[- ]?schedul|\bchunked[- ]?prefill\b|\bmax[-_]num[-_](seqs|batched)",
    "prefix_cache": r"\bprefix[- ]?cach|\bradix\b|\bkv[- ]?cache[- ]?(reuse|hit)",
    "kv_offload": r"\bkv[- ]?(cache )?offload|\blmcache\b|\bmooncake\b|\bkvcached\b|\bcpu offload|\bhicache\b|\bnixl\b",
    "disagg": r"\bdisaggregat|\bp/?d\b|\bprefill[- ]decode\b|\bpd[- ]?disagg",
    "spec_decode": r"\bspeculative\b|\bspec[- ]?decod|\beagle\d?\b|\bmtp\b|\bngram\b|\bdraft model\b",
    "quantization": r"\bfp8\b|\bfp4\b|\bnvfp4\b|\bmxfp\w*|\bint4\b|\bint8\b|\bawq\b|\bgptq\b|\bquantiz",
    "attention_kernels": r"\bflashinfer\b|\bflash[- ]?attn|\bflashattention|\bflashmla\b|\btriton\b.*\bkernel|\battention backend\b|\bcutlass\b|\bdeepgemm\b",
    "cuda_graphs": r"\bcuda[- ]?graph|\btorch\.compile\b|\bcompil(e|ation)\b",
    "moe_parallel": r"\bmoe\b|\bexpert[- ]parallel|\b(ep|tp|dp|pp)\b ?(size|parallel)|\bdata[- ]parallel|\btensor[- ]parallel|\ball2all\b|\bdeepep\b",
    "hardware": r"\bb200\b|\bgb200\b|\bh200\b|\bh100\b|\bblackwell\b|\bhopper\b|\bmi3\d\d\b|\brocm\b|\btpu\b|\bgaudi\b",
    "memory": r"\bgpu[-_ ]memory[-_ ]utilization\b|\bmemory (usage|footprint|leak)|\boom\b",
}
PERF = re.compile(
    r"\b(perf(ormance)?|throughput|latency|ttft|tpot|itl|speed ?up|faster|slower|regression|optimi[sz]\w*|"
    r"overhead|efficien\w*|tok(ens)?/s)\b", re.I)
CLAIM = re.compile(r"(\d+(\.\d+)?\s?x\b|\b\d+(\.\d+)?\s?%|\b\d+(\.\d+)?\s?(ms|tok/s|tokens/s))", re.I)
PR_REF = re.compile(r"(#\d{4,6}|/pull/\d+)")
COMPILED = {k: re.compile(v, re.I) for k, v in KNOBS.items()}


def releases(repo: str, since: str) -> list[dict]:
    out = subprocess.run(
        ["gh", "api", f"repos/{repo}/releases", "--paginate", "-q",
         '.[] | {tag: .tag_name, date: .published_at, pre: .prerelease, body: .body}'],
        capture_output=True, text=True, check=True).stdout
    rel = [json.loads(line) for line in out.splitlines() if line.strip()]
    rel = [r for r in rel if r["date"] and r["date"][:10] >= since and not r["pre"]]
    return sorted(rel, key=lambda r: r["date"])


def classify(body: str) -> list[dict]:
    items = []
    for raw in body.splitlines():
        line = raw.strip(" -*\t")
        if len(line) < 12:
            continue
        knobs = [k for k, rx in COMPILED.items() if rx.search(line)]
        perf = bool(PERF.search(line))
        if not knobs and not perf:
            continue
        claims = [m.group(0) for m in CLAIM.finditer(line)]
        # weight: defaults and quantified perf claims matter most to a running deployment
        score = (3 if "defaults" in knobs else 0) + (2 if perf and claims else 1 if perf else 0) + min(len(knobs), 2)
        items.append({"line": line[:400], "knobs": knobs, "perf": perf, "claims": claims,
                      "prs": PR_REF.findall(line), "score": score})
    return items


def retune_flags(items: list[dict]) -> list[dict]:
    """Changes that plausibly move throughput or latency >10% for a common config:
    a changed default on a hot-path knob, or a quantified perf claim >= 10% / >= 1.1x."""
    hot = {"scheduler", "prefix_cache", "spec_decode", "quantization", "attention_kernels",
           "cuda_graphs", "moe_parallel", "kv_offload", "disagg"}
    flagged = []
    for it in items:
        big = False
        for c in it["claims"]:
            num = float(re.findall(r"\d+(?:\.\d+)?", c)[0])
            if c.strip().lower().endswith("x") and num >= 1.1:
                big = True
            elif c.strip().endswith("%") and num >= 10:
                big = True
        if ("defaults" in it["knobs"] and hot & set(it["knobs"])) or (big and it["perf"]):
            flagged.append(it)
    return flagged


HW_TOKENS = {
    "hopper": r"\bh100\b|\bh200\b|\bh20\b|\bhopper\b|\bsm90\b",
    "blackwell": r"\bb200\b|\bgb200\b|\bb300\b|\bblackwell\b|\bsm100\b|\bsm103\b|\bsm12\d\b|\bgb10\b|rtx pro 6000",
    "ampere": r"\ba100\b|\ba10g?\b|\bl4\b|\bampere\b|\bsm8\d\b|\bl40s?\b",
    "amd": r"\brocm\b|\bmi3\d\d\w*\b|\bgfx9\d\d\b|\baiter\b|\bhip\w*\b",
    "intel": r"\bxpu\b|\bgaudi\b|\bhpu\b|\bintel\b",
    "cpu": r"\bcpu\b(?! offload)|\bavx\w*\b|\bamx\b|\barm\b|\bpower\b|\bzen\b",
    "tpu": r"\btpu\b|\bjax\b",
    "npu": r"\bascend\b|\bnpu\b|\b910[bc]?\b",
}
HW_RX = {k: re.compile(v, re.I) for k, v in HW_TOKENS.items()}


def for_client(items: list[dict], hardware: set[str], models: list[str]) -> list[dict]:
    """Keep lines that apply to the client's hardware family and boost lines naming their models.
    A line that names only other hardware families is dropped; hardware-agnostic lines stay."""
    if not hardware and not models:
        return items
    out = []
    for it in items:
        fams = {k for k, rx in HW_RX.items() if rx.search(it["line"])}
        if fams and hardware and not (fams & hardware):
            continue
        it = dict(it)
        if models and any(m.lower() in it["line"].lower() for m in models):
            it["score"] += 3
            it["knobs"] = it["knobs"] + ["your_model"]
        out.append(it)
    return out


def note(repo: str, rel: dict, items: list[dict], flagged: list[dict]) -> str:
    by_knob: dict[str, int] = {}
    for it in items:
        for k in it["knobs"] or ["other_perf"]:
            by_knob[k] = by_knob.get(k, 0) + 1
    lines = [f"# {repo} {rel['tag']} ({rel['date'][:10]}): what to re-tune", "",
             f"{len(items)} performance or default-related lines; {len(flagged)} likely to move a common config by more than 10%.", "",
             "## Re-benchmark triggers", ""]
    for it in sorted(flagged, key=lambda x: -x["score"])[:15]:
        lines.append(f"- [{', '.join(it['knobs']) or 'perf'}] {it['line']}")
    if not flagged:
        lines.append("- None flagged. Spot-check only if the client runs a knob listed below.")
    lines += ["", "## Touch count by knob", ""]
    for k, v in sorted(by_knob.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}")
    lines += ["", "## Other notable lines", ""]
    rest = [it for it in sorted(items, key=lambda x: -x["score"]) if it not in flagged][:12]
    for it in rest:
        lines.append(f"- [{', '.join(it['knobs']) or 'perf'}] {it['line']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--since", default="2026-07-01")
    ap.add_argument("--out", default="notes")
    ap.add_argument("--hardware", default="", help="client hardware families, e.g. hopper or hopper,ampere")
    ap.add_argument("--models", default="", help="client model families, e.g. qwen,llama")
    a = ap.parse_args()
    hw = {h.strip().lower() for h in a.hardware.split(",") if h.strip()}
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = []
    for rel in releases(a.repo, a.since):
        items = for_client(classify(rel["body"] or ""), hw, models)
        flagged = retune_flags(items)
        slug = a.repo.split("/")[1] + (f"_{'-'.join(sorted(hw))}" if hw else "")
        (out / f"{slug}_{rel['tag']}.md").write_text(note(a.repo, rel, items, flagged))
        summary.append({"repo": a.repo, "tag": rel["tag"], "date": rel["date"][:10],
                        "perf_lines": len(items), "retune_triggers": len(flagged)})
        print(f"{a.repo} {rel['tag']:10s} {rel['date'][:10]} perf_lines={len(items):3d} retune_triggers={len(flagged):2d}")
    (out / f"{a.repo.split('/')[1]}_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
