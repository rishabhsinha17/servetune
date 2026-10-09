#!/usr/bin/env python3
"""servetune audit runner: sweep serving configs against one workload and record results.

For each candidate config in the audit file:
  1. start the engine (vLLM or SGLang) with baseline flags plus the candidate's flags
  2. wait for /health
  3. run the engine's own load generator at each request rate in the workload
  4. save the raw benchmark JSON and the server log
  5. stop the server and move on

Nothing here estimates cost; analyze.py turns the raw results into the report.

Usage:
  python3 audit.py audit.toml --out runs/<name>
  python3 audit.py audit.toml --out runs/<name> --only baseline,fp8_kv   # subset
  python3 audit.py audit.toml --out runs/<name> --dry-run                # print commands
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path


def wait_healthy(port: int, proc: subprocess.Popen, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def server_cmd(cfg: dict, flags: str, port: int, model: str) -> list[str]:
    eng = cfg["engine"]["name"]
    base = cfg["engine"].get("baseline_flags", "")
    if eng == "vllm":
        return ["vllm", "serve", model, "--port", str(port), *shlex.split(base), *shlex.split(flags)]
    if eng == "sglang":
        return [sys.executable, "-m", "sglang.launch_server", "--model-path", model, "--port", str(port),
                *shlex.split(base), *shlex.split(flags)]
    if eng == "external":  # client-provided endpoint; nothing to launch
        return ["(external endpoint)"]
    raise SystemExit(f"unknown engine {eng}")


def replay_cmd(cfg: dict, base_url: str, model: str, speed: float, out: Path) -> list[str]:
    w, slo = cfg["workload"], cfg["slo"]
    here = Path(__file__).resolve().parent
    return [sys.executable, str(here / "replay.py"), "--base-url", base_url, "--model", model,
            "--trace", str(w["trace"]), "--speed", str(speed), "--prefix-tokens", str(w.get("shared_prefix_len", 0)),
            "--slo-ttft-ms", str(slo["ttft_p95_ms"]), "--slo-tpot-ms", str(slo["tpot_p95_ms"]), "--out", str(out)]
    # tokenizer defaults to the model name; set [engine] tokenizer when serving a local path


def bench_cmd(cfg: dict, port: int, rate: float, out_dir: Path, tag: str, model: str) -> list[str]:
    eng = cfg["engine"]["name"]
    w = cfg["workload"]
    slo = cfg["slo"]
    n = int(w.get("num_prompts", 400))
    rate_s = "inf" if rate == 0 else str(rate)
    if eng == "vllm":
        cmd = ["vllm", "bench", "serve", "--backend", "vllm", "--model", model, "--port", str(port),
               "--dataset-name", "random",
               "--random-input-len", str(w["input_len"]), "--random-output-len", str(w["output_len"]),
               "--random-range-ratio", str(w.get("range_ratio", 0.5)),
               "--random-prefix-len", str(w.get("shared_prefix_len", 0)),
               "--num-prompts", str(n), "--request-rate", rate_s, "--ignore-eos",
               "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99",
               "--goodput", f"ttft:{slo['ttft_p95_ms']}", f"tpot:{slo['tpot_p95_ms']}",
               "--save-result", "--result-dir", str(out_dir), "--result-filename", f"{tag}.json"]
        if w.get("seed") is not None:
            cmd += ["--seed", str(w["seed"])]
        return cmd
    # SGLang's bench_serving uses its own flag names
    return [sys.executable, "-m", "sglang.bench_serving", "--backend", "sglang", "--port", str(port),
            "--dataset-name", "random", "--random-input-len", str(w["input_len"]),
            "--random-output-len", str(w["output_len"]), "--random-range-ratio", str(w.get("range_ratio", 0.5)),
            "--num-prompts", str(n), "--request-rate", rate_s, "--output-file", str(out_dir / f"{tag}.jsonl")]


def run_candidate(cfg: dict, cand: dict, out: Path, dry: bool) -> dict:
    port = int(cfg["engine"].get("port", 8000))
    name = cand["name"]
    cdir = out / name
    cdir.mkdir(parents=True, exist_ok=True)
    model = cand.get("model", cfg["engine"]["model"])  # e.g. a pre-quantized checkpoint
    scmd = server_cmd(cfg, cand.get("flags", ""), port, model)
    env = os.environ | {k: str(v) for k, v in cand.get("env", {}).items()}
    record = {"name": name, "model": model, "flags": cand.get("flags", ""), "env": cand.get("env", {}),
              "server_cmd": " ".join(scmd), "rates": {}, "status": "ok"}
    if dry:
        print("SERVER:", " ".join(scmd))
        if cfg["workload"].get("mode") == "replay":
            base_url = cand.get("base_url") or cfg["engine"].get("base_url") or f"http://127.0.0.1:{port}"
            for sp in cfg["workload"]["speeds"]:
                print("  REPLAY:", " ".join(replay_cmd(cfg, base_url, model, sp, cdir / f"rate_{sp}.json")))
        else:
            for r in cfg["workload"]["request_rates"]:
                print("  BENCH:", " ".join(bench_cmd(cfg, port, r, cdir, f"rate_{r}", model)))
        return record
    external = cfg["engine"]["name"] == "external"
    log = open(cdir / "server.log", "w")
    proc = None if external else subprocess.Popen(scmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                                  start_new_session=True)
    try:
        if proc is not None and not wait_healthy(port, proc, int(cfg["engine"].get("startup_timeout_s", 900))):
            record["status"] = "server_failed"
            return record
        if cfg["workload"].get("mode") == "replay":
            base_url = cand.get("base_url") or cfg["engine"].get("base_url") or f"http://127.0.0.1:{port}"
            for sp in cfg["workload"]["speeds"]:
                tag = f"rate_{sp}"
                t0 = time.time()
                res = subprocess.run(replay_cmd(cfg, base_url, model, sp, cdir / f"{tag}.json"), capture_output=True,
                                     text=True, timeout=int(cfg["workload"].get("bench_timeout_s", 1800)))
                (cdir / f"{tag}.stdout").write_text(res.stdout[-20000:] + "\n---stderr---\n" + res.stderr[-20000:])
                record["rates"][str(sp)] = {"returncode": res.returncode, "seconds": round(time.time() - t0, 1)}
                print(f"[{name}] speed={sp} rc={res.returncode} {time.time() - t0:.0f}s", flush=True)
            return record
        # one short warmup so CUDA graph capture and JIT do not land in the first rate
        subprocess.run(bench_cmd(cfg | {"workload": cfg["workload"] | {"num_prompts": 20}}, port, 0, cdir, "warmup", model),
                       capture_output=True, timeout=600)
        for r in cfg["workload"]["request_rates"]:
            tag = f"rate_{r}"
            t0 = time.time()
            res = subprocess.run(bench_cmd(cfg, port, r, cdir, tag, model), capture_output=True, text=True,
                                 timeout=int(cfg["workload"].get("bench_timeout_s", 1800)))
            (cdir / f"{tag}.stdout").write_text(res.stdout[-20000:] + "\n---stderr---\n" + res.stderr[-20000:])
            record["rates"][str(r)] = {"returncode": res.returncode, "seconds": round(time.time() - t0, 1)}
            print(f"[{name}] rate={r} rc={res.returncode} {time.time() - t0:.0f}s", flush=True)
    finally:
        if proc is not None:
            if proc.poll() is not None:  # the server exited on its own; note it rather than crash
                record["server_exit_code"] = proc.returncode
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(5)  # let the GPU memory free before the next server
        log.close()
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.audit).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.toml").write_text(Path(a.audit).read_text())
    only = {s for s in a.only.split(",") if s}
    manifest_f = out / "manifest.json"
    manifest = json.loads(manifest_f.read_text()) if manifest_f.exists() else {"candidates": {}}
    for cand in cfg["candidates"]:
        if only and cand["name"] not in only:
            continue
        rec = run_candidate(cfg, cand, out, a.dry_run)
        manifest["candidates"][cand["name"]] = rec
        if not a.dry_run:
            manifest_f.write_text(json.dumps(manifest, indent=1))  # checkpoint after every candidate


if __name__ == "__main__":
    main()
