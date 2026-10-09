#!/usr/bin/env python3
"""A fake OpenAI-compatible streaming engine for testing servetune without a GPU.

It models one replica as a fixed token budget per scheduling step: each step processes
up to `--step-tokens` prefill tokens plus one decode token for every running sequence,
and takes `--step-ms` plus a per-token cost. A small prefix cache makes repeated prefixes
cheaper. Numbers are made up; the point is that latency rises with load the way a real
engine's does, so replay.py and analyze.py can be exercised end to end.

  python3 mock_engine.py --port 8011 --max-seqs 64 --step-tokens 4096
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from aiohttp import web


class Engine:
    def __init__(self, a):
        self.a = a
        self.queue: list[dict] = []
        self.running: list[dict] = []
        self.cache: set[str] = set()
        self.wake = asyncio.Event()

    async def loop(self):
        while True:
            if not self.queue and not self.running:
                self.wake.clear()
                await self.wake.wait()
            budget = self.a.step_tokens
            while self.queue and len(self.running) < self.a.max_seqs and budget > 0:
                r = self.queue[0]
                need = r["prefill_left"]
                take = min(need, budget)
                r["prefill_left"] -= take
                budget -= take
                if r["prefill_left"] == 0:
                    self.running.append(self.queue.pop(0))
                else:
                    break
            decode = len(self.running)
            used = self.a.step_tokens - budget + decode
            await asyncio.sleep((self.a.step_ms + used * self.a.us_per_token / 1000) / 1000)
            for r in list(self.running):
                r["emitted"] += 1
                r["q"].put_nowait(1)
                if r["emitted"] >= r["max_tokens"]:
                    self.running.remove(r)
                    r["q"].put_nowait(None)

    async def completions(self, request: web.Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        words = prompt.split()
        n_in = int(len(words) * 1.3)
        key = " ".join(words[:200])
        cached = int(min(n_in, 2000) * 0.9) if key in self.cache else 0
        self.cache.add(key)
        r = {"prefill_left": max(1, n_in - cached), "max_tokens": int(body.get("max_tokens", 16)),
             "emitted": 0, "q": asyncio.Queue()}
        self.queue.append(r)
        self.wake.set()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        while True:
            x = await r["q"].get()
            if x is None:
                break
            chunk = {"choices": [{"text": " tok", "index": 0}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        usage = {"choices": [], "usage": {"completion_tokens": r["emitted"], "prompt_tokens": n_in}}
        await resp.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
        return resp


async def health(_):
    return web.Response(text="ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--max-seqs", type=int, default=64)
    ap.add_argument("--step-tokens", type=int, default=4096)
    ap.add_argument("--step-ms", type=float, default=12.0)
    ap.add_argument("--us-per-token", type=float, default=4.0)
    a = ap.parse_args()
    eng = Engine(a)
    app = web.Application()
    app.router.add_post("/v1/completions", eng.completions)
    app.router.add_get("/health", health)

    async def start(app):
        app["loop"] = asyncio.create_task(eng.loop())
    app.on_startup.append(start)
    web.run_app(app, port=a.port, print=None)


if __name__ == "__main__":
    main()
