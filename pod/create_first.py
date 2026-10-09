"""Try GPU/cloud options in order; print the first created pod id (or nothing). Needs RUNPOD_API_KEY."""
import json, subprocess, sys
pub = sys.argv[1]
image = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
specs = [("NVIDIA H100 80GB HBM3", "COMMUNITY", "12.8,12.9,13.0"),
         ("NVIDIA H100 80GB HBM3", "SECURE", "12.8,12.9,13.0"),
         ("NVIDIA H100 NVL", "COMMUNITY", "12.8,12.9,13.0"),
         ("NVIDIA H100 PCIe", "COMMUNITY", "12.8,12.9,13.0")]
for gpu, cloud, cuda in specs:
    out = subprocess.run([sys.executable, "rp.py", "create", gpu, cloud, image, "120", pub, "8", "48", cuda],
                         capture_output=True, text=True).stdout
    d = json.loads(out or "{}")
    p = ((d.get("data") or {}).get("podFindAndDeployOnDemand") or {})
    if p.get("id"):
        print(json.dumps({"id": p["id"], "gpu": gpu, "cloud": cloud, "costPerHr": p.get("costPerHr")}))
        sys.exit(0)
    msg = (d.get("errors") or [{}])[0].get("message", "")[:90]
    print(f"skip {gpu} {cloud}: {msg}", file=sys.stderr)
sys.exit(1)
