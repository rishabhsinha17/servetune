#!/usr/bin/env python3
"""servetune replay: drive any OpenAI-compatible endpoint with a client's own traffic shape.

Input is request metadata only (no prompt content), one row per request:
  arrival_s,input_tokens,output_tokens[,prefix_id]
arrival_s is seconds since the start of the trace; rows with the same prefix_id share
their first `--prefix-tokens` tokens, which is how a common system prompt or tool schema
shows up in prefix-cache hit rates.

Prompts are synthesized to the requested token counts with the model's tokenizer when
`transformers` is available (exact), else with a words-per-token estimate. Each request
streams from /v1/completions with max_tokens = output_tokens and ignore_eos, and the
script records time to first token, time per output token and end-to-end latency.

`--speed k` replays the trace k times faster, which is how capacity is found: raise k until
the SLO breaks. The output JSON uses the same keys as `vllm bench serve`, so analyze.py
reads it unchanged.

Usage:
  python3 replay.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B \
      --trace trace.csv --speed 2 --prefix-tokens 2000 --out results/rate_x2.json
  python3 replay.py --make-trace synthetic.csv --rate 4 --duration 120 \
      --input-mean 3000 --output-mean 300 --prefix-groups 3          # synthetic Poisson trace
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
         "upsilon phi chi psi omega river stone cloud field signal vector matrix kernel buffer packet").split()


class PromptMaker:
    """Builds prompts with an exact token count when a tokenizer is available."""

    def __init__(self, model: str, seed: int = 0):
        self.rng = random.Random(seed)
        self.tok = None
        try:
            from transformers import AutoTokenizer  # type: ignore
            self.tok = AutoTokenizer.from_pretrained(model)
        except Exception:
            self.tok = None
        self.prefixes: dict[str, str] = {}

    def _text(self, n_tokens: int, rng: random.Random) -> str:
        if n_tokens <= 0:
            return ""
        if self.tok is None:  # about 1.3 tokens per short English word
            return " ".join(rng.choice(WORDS) for _ in range(max(1, int(n_tokens / 1.3))))
        ids = [rng.randrange(1000, min(self.tok.vocab_size, 30000)) for _ in range(n_tokens)]
        text = self.tok.decode(ids)
        # re-encode and trim so the round trip lands on the requested count
        enc = self.tok.encode(text, add_special_tokens=False)[:n_tokens]
        return self.tok.decode(enc)

    def prompt(self, n_tokens: int, prefix_id: str | None, prefix_tokens: int) -> str:
        if prefix_id and prefix_tokens > 0:
            if prefix_id not in self.prefixes:
                self.prefixes[prefix_id] = self._text(prefix_tokens, random.Random(hash(prefix_id) & 0xFFFF))
            body = self._text(max(0, n_tokens - prefix_tokens), self.rng)
            return self.prefixes[prefix_id] + " " + body
        return self._text(n_tokens, self.rng)


async def one(session, url: str, model: str, prompt: str, max_tokens: int, t0: float, rec: dict) -> None:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "stream": True,
               "ignore_eos": True, "temperature": 0.0}
    start = time.perf_counter()
    first = None
    last = start
    n_chunks = 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}"
                return
            async for raw in resp.content:
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                text = "".join(c.get("text", "") for c in obj.get("choices", []))
                if text:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    n_chunks += 1
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens"):
                    rec["completion_tokens"] = usage["completion_tokens"]
    except Exception as e:  # network errors count as failures, not crashes
        rec["error"] = type(e).__name__
        return
    end = time.perf_counter()
    rec["start_s"] = start - t0
    rec["ttft_ms"] = (first - start) * 1000 if first else None
    out_toks = rec.get("completion_tokens") or n_chunks
    rec["output_tokens"] = out_toks
    rec["tpot_ms"] = ((last - first) * 1000 / (out_toks - 1)) if first and out_toks > 1 else None
    rec["e2el_ms"] = (end - start) * 1000


def pct(xs: list[float], p: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


async def replay(args) -> dict:
    rows = list(csv.DictReader(open(args.trace)))
    if args.limit:
        rows = rows[: args.limit]
    pm = PromptMaker(args.tokenizer or args.model, seed=args.seed)
    reqs = []
    for r in rows:
        reqs.append({"arrival": float(r["arrival_s"]) / args.speed, "in": int(r["input_tokens"]),
                     "out": int(r["output_tokens"]), "prefix": r.get("prefix_id") or None})
    prompts = [pm.prompt(q["in"], q["prefix"], args.prefix_tokens) for q in reqs]
    url = args.base_url.rstrip("/") + "/v1/completions"
    recs = [dict() for _ in reqs]
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_s)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        t0 = time.perf_counter()
        tasks = []
        for q, p, rec in zip(reqs, prompts, recs):
            delay = q["arrival"] - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(session, url, args.model, p, q["out"], t0, rec)))
        await asyncio.gather(*tasks)
        duration = time.perf_counter() - t0
    ok = [r for r in recs if "error" not in r and r.get("ttft_ms") is not None]
    total_out = sum(r["output_tokens"] for r in ok)
    total_in = sum(q["in"] for q, r in zip(reqs, recs) if r in ok)
    ttft = [r["ttft_ms"] for r in ok]
    tpot = [r["tpot_ms"] for r in ok]
    e2el = [r["e2el_ms"] for r in ok]
    slo_ok = [r for r in ok if (args.slo_ttft_ms is None or r["ttft_ms"] <= args.slo_ttft_ms)
              and (args.slo_tpot_ms is None or (r["tpot_ms"] or 0) <= args.slo_tpot_ms)]
    res = {
        "tool": "servetune.replay", "trace": str(args.trace), "speed": args.speed,
        "num_prompts": len(reqs), "completed": len(ok), "failed": len(reqs) - len(ok),
        "duration": duration,
        "request_rate": len(reqs) / (reqs[-1]["arrival"] or 1) if reqs else 0,
        "request_throughput": len(ok) / duration, "output_throughput": total_out / duration,
        "total_token_throughput": (total_out + total_in) / duration,
        "request_goodput": len(slo_ok) / duration,
        "mean_ttft_ms": statistics.mean(ttft) if ttft else None, "median_ttft_ms": pct(ttft, 50),
        "p95_ttft_ms": pct(ttft, 95), "p99_ttft_ms": pct(ttft, 99),
        "mean_tpot_ms": statistics.mean([x for x in tpot if x]) if any(tpot) else None,
        "median_tpot_ms": pct(tpot, 50), "p95_tpot_ms": pct(tpot, 95), "p99_tpot_ms": pct(tpot, 99),
        "median_e2el_ms": pct(e2el, 50), "p95_e2el_ms": pct(e2el, 95),
        "errors": sorted({r["error"] for r in recs if "error" in r}),
        "tokenizer_exact": pm.tok is not None,
    }
    return res


def make_trace(args) -> None:
    rng = random.Random(args.seed)
    t, rows = 0.0, []
    while t < args.duration:
        t += rng.expovariate(args.rate)
        rows.append({"arrival_s": round(t, 4),
                     "input_tokens": max(16, int(rng.lognormvariate(math.log(args.input_mean), 0.4))),
                     "output_tokens": max(8, int(rng.lognormvariate(math.log(args.output_mean), 0.5))),
                     "prefix_id": f"p{rng.randrange(args.prefix_groups)}" if args.prefix_groups else ""})
    with open(args.make_trace, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} requests over {args.duration}s to {args.make_trace}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model")
    ap.add_argument("--tokenizer")
    ap.add_argument("--trace")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--prefix-tokens", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--slo-ttft-ms", type=float)
    ap.add_argument("--slo-tpot-ms", type=float)
    ap.add_argument("--request-timeout-s", type=float, default=600)
    ap.add_argument("--out")
    ap.add_argument("--make-trace")
    ap.add_argument("--rate", type=float, default=4)
    ap.add_argument("--duration", type=float, default=120)
    ap.add_argument("--input-mean", type=float, default=3000)
    ap.add_argument("--output-mean", type=float, default=300)
    ap.add_argument("--prefix-groups", type=int, default=0)
    a = ap.parse_args()
    if a.make_trace:
        return make_trace(a)
    if aiohttp is None:
        raise SystemExit("pip install aiohttp (it ships with vllm and sglang)")
    res = asyncio.run(replay(a))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("completed", "failed", "request_throughput", "output_throughput",
                                          "p95_ttft_ms", "p95_tpot_ms")}, default=str))


if __name__ == "__main__":
    main()
