#!/usr/bin/env python3
"""cache_probe: does multi-turn agent traffic keep its prefix cache on this endpoint?

An agent sends the whole conversation every turn, so each request should start with the
previous request's tokens. When a chat template or an API compatibility layer re-renders
history differently (moves a system message, strips reasoning, reorders tool schemas),
the shared prefix ends early and every turn pays prefill again.

The probe uses vLLM's render endpoints (/v1/chat/completions/render, /v1/messages/render),
which return the exact prompt token IDs without generating, so it is fast, deterministic
and does not depend on GPU speed. For each scenario and turn k it reports:
  prev_kept   share of turn k's prompt that turn k+1 repeats verbatim (1.00 is ideal)
  out_kept    share of the model's raw turn-k output that turn k+1 repeats (what decode
              left in the cache; below 1.00 means generated tokens are re-prefilled)

Usage:
  python3 cache_probe.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-0.6B --out probe.md
"""
from __future__ import annotations

import argparse
import json
import urllib.request

SYSTEM = ("You are a coding agent working in a large repository. Follow the project's conventions, "
          "explain each change briefly, run tests before declaring success, and never edit files "
          "outside the working directory. ") * 12   # a long, stable system prompt like real agents use
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file from the repository",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"}},
                   "required": ["path"]}}}]
USERS = ["The test in tests/test_cache.py fails with a KeyError. Find out why.",
         "Good. Now fix it without changing the public API.",
         "Run the whole test suite and summarize the result.",
         "Add a regression test for the bug you fixed.",
         "Write a two-line changelog entry."]
REPLIES = ["The KeyError comes from the cache lookup when the key was evicted; get() assumes presence.",
           "I changed get() to return None on a miss and updated the two callers to handle it.",
           "All 214 tests pass. One test is skipped because it needs a GPU.",
           "Added test_get_after_evict, which evicts a key and checks get() returns None.",
           "Fixed: cache.get() no longer raises KeyError after eviction. Added a regression test."]
THINK = "<think>\nLet me look at the traceback and the cache code before answering.\n</think>\n\n"


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Probe:
    def __init__(self, base: str, model: str):
        self.base = base.rstrip("/")
        self.model = model
        from transformers import AutoTokenizer  # tokenizes the raw model output for out_kept
        self.tok = AutoTokenizer.from_pretrained(model)

    def chat(self, messages, tools=None):
        body = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "none"  # still renders the schemas; avoids needing a tool-call parser on the server
        return post(self.base + "/v1/chat/completions/render", body)["token_ids"]

    def messages(self, system, messages):
        body = {"model": self.model, "max_tokens": 64, "system": system, "messages": messages}
        return post(self.base + "/v1/messages/render", body)["token_ids"]

    def run(self, name: str, render_turn, raw_output) -> dict:
        """render_turn(k) -> token ids of the prompt at turn k; raw_output(k) -> text the model emitted."""
        rows, error = [], None
        try:
            prompts = [render_turn(k) for k in range(len(USERS))]
        except Exception as e:  # e.g. a template that rejects a mid-conversation system message
            return {"name": name, "rows": [], "error": f"{type(e).__name__}: {str(e)[:200]}"}
        for k in range(len(prompts) - 1):
            p, q = prompts[k], prompts[k + 1]
            kept = lcp(p, q)
            out_ids = self.tok.encode(raw_output(k), add_special_tokens=False)
            tail = q[len(p):] if kept == len(p) else []
            out_kept = lcp(tail, out_ids) / max(1, len(out_ids)) if tail else 0.0
            rows.append({"turn": k + 1, "prompt_tokens": len(p), "prev_kept": kept / len(p), "out_kept": out_kept,
                         "first_diff_at": kept})
        return {"name": name, "rows": rows, "error": error}


def scenarios(pr: Probe) -> list[dict]:
    def hist(k, think=False, tools=False):
        msgs = [{"role": "system", "content": SYSTEM}]
        for i in range(k + 1):
            msgs.append({"role": "user", "content": USERS[i]})
            if i < k:
                if tools and i == 0:
                    msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                                 "function": {"name": "read_file", "arguments": json.dumps({"path": "cache.py"})}}]})
                    msgs.append({"role": "tool", "tool_call_id": "call_1", "content": "def get(self, k):\n    return self._d[k]\n"})
                msgs.append({"role": "assistant", "content": (THINK if think else "") + REPLIES[i]})
        return msgs

    def hist_midsystem(k):
        msgs = hist(k)
        if k >= 2:  # a client-side reminder injected as a system message partway through
            msgs.insert(5, {"role": "system", "content": "Reminder: the user prefers short answers."})
        return msgs

    def anth(k, changing_system=False):
        msgs = []
        for i in range(k + 1):
            msgs.append({"role": "user", "content": USERS[i]})
            if i < k:
                msgs.append({"role": "assistant", "content": REPLIES[i]})
        system = SYSTEM if not changing_system else f"Current time: 2026-10-08T19:{10 + k:02d}:00Z. " + SYSTEM
        return pr.messages(system, msgs)

    return [
        pr.run("chat, plain replies", lambda k: pr.chat(hist(k)), lambda k: REPLIES[k]),
        pr.run("chat, reasoning replies kept with <think>", lambda k: pr.chat(hist(k, think=True)), lambda k: THINK + REPLIES[k]),
        pr.run("chat, reasoning model, client drops <think> from history", lambda k: pr.chat(hist(k)), lambda k: THINK + REPLIES[k]),
        pr.run("chat with tools and a tool call", lambda k: pr.chat(hist(k, tools=True), tools=TOOLS), lambda k: REPLIES[k]),
        pr.run("chat, system reminder inserted mid-conversation", lambda k: pr.chat(hist_midsystem(k)), lambda k: REPLIES[k]),
        pr.run("Anthropic /v1/messages, fixed system", lambda k: anth(k), lambda k: REPLIES[k]),
        pr.run("Anthropic /v1/messages, timestamp in system prompt", lambda k: anth(k, changing_system=True), lambda k: REPLIES[k]),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    pr = Probe(a.base_url, a.model)
    res = scenarios(pr)
    lines = [f"# Prefix-cache probe: {a.model} at {a.base_url}", "",
             "prev_kept: share of each turn's prompt that the next turn repeats verbatim (1.00 is ideal). "
             "out_kept: share of the model's raw output that the next turn repeats.", "",
             "| Scenario | Turn | Prompt tokens | prev_kept | out_kept | First differing token |",
             "|---|---|---|---|---|---|"]
    for s in res:
        if s["error"]:
            lines.append(f"| {s['name']} | - | - | - | - | rejected: {s['error']} |")
            continue
        for r in s["rows"]:
            lines.append(f"| {s['name']} | {r['turn']} | {r['prompt_tokens']} | {r['prev_kept']:.2f} | {r['out_kept']:.2f} | {r['first_diff_at']} |")
    text = "\n".join(lines) + "\n"
    if a.out:
        open(a.out, "w").write(text)
        open(a.out.rsplit(".", 1)[0] + ".json", "w").write(json.dumps(res, indent=1))
    print(text)


if __name__ == "__main__":
    main()
