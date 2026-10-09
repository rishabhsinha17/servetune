"""Minimal RunPod helper for the servetune sample run (pod name venture-servetune), adapted from ../wb2_20261007/rp.py.
The key is read from RUNPOD_API_KEY (set by `export RUNPOD_API_KEY=... (your choice of key)`), sent only in
the Authorization header of requests to api.runpod.io / rest.runpod.io, and never printed, logged or written.

  python3 rp_p2.py balance
  python3 rp_p2.py stock "NVIDIA A100-SXM4-80GB"
  python3 rp_p2.py create <gpuTypeId> <cloudType> <image> <containerDiskGb> <pubkey_file> [vcpu] [memGb] [cuda_csv|-]
  python3 rp_p2.py status <podId>
  python3 rp_p2.py terminate <podId>
  python3 rp_p2.py pods          (GraphQL myself.pods)
  python3 rp_p2.py restpods      (REST GET /v1/pods, a second listing)
Every call appends a one-line JSON record (no key, no headers) to logs/rp_calls.jsonl.
"""
import json, os, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://api.runpod.io/graphql"
NAME = "venture-servetune"
LOG = os.path.join(HERE, "logs", "rp_calls.jsonl")


def _key():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY not set")
    return key


def http(url, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"),
                                 headers={"Authorization": "Bearer " + _key(), "Content-Type": "application/json",
                                          "User-Agent": "venture-servetune/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        return {"errors": [{"message": f"HTTP {e.code}: {e.read().decode()[:500]}"}]}


def gql(query, variables=None):
    return http(URL, {"query": query, "variables": variables or {}})


def record(action, out):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "action": action,
                            "out": out}) + "\n")


def main():
    a = sys.argv[1]
    if a == "balance":
        d = gql("query { myself { clientBalance currentSpendPerHr pods { id name desiredStatus costPerHr } } }")
    elif a == "pods":
        d = gql("query { myself { pods { id name desiredStatus costPerHr gpuCount machine { gpuDisplayName } "
                "runtime { uptimeInSeconds gpus { id gpuUtilPercent memoryUtilPercent } } } } }")
    elif a == "restpods":
        d = http("https://rest.runpod.io/v1/pods")
    elif a == "stock":
        d = gql("query G($id: String) { gpuTypes(input: {id: $id}) { id displayName memoryInGb communityPrice "
                "securePrice communityCloud secureCloud lowestPrice(input: {gpuCount: 1}) { minimumBidPrice "
                "uninterruptablePrice stockStatus } } }", {"id": sys.argv[2]})
    elif a == "create":
        gpu, cloud, image, disk, pubf = sys.argv[2:7]
        vcpu = int(sys.argv[7]) if len(sys.argv) > 7 else 8
        mem = int(sys.argv[8]) if len(sys.argv) > 8 else 48
        cuda = sys.argv[9] if len(sys.argv) > 9 else "-"
        pub = open(pubf).read().strip()
        assert pub.startswith("ssh-ed25519 "), "public key expected"
        inp = {"cloudType": cloud, "gpuCount": 1, "volumeInGb": 0, "containerDiskInGb": int(disk),
               "minVcpuCount": vcpu, "minMemoryInGb": mem, "gpuTypeId": gpu, "name": NAME, "imageName": image,
               "dockerArgs": "", "ports": "22/tcp", "supportPublicIp": True, "startSsh": True,
               "env": [{"key": "PUBLIC_KEY", "value": pub}]}
        if cuda != "-":
            inp["allowedCudaVersions"] = cuda.split(",")
        d = gql("mutation D($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) "
                "{ id name imageName costPerHr machineId machine { gpuDisplayName } } }", {"input": inp})
    elif a == "status":
        d = gql("query P($id: String!) { pod(input: {podId: $id}) { id name desiredStatus costPerHr lastStatusChange "
                "imageName machine { gpuDisplayName } runtime { uptimeInSeconds ports { ip isIpPublic privatePort "
                "publicPort type } gpus { id gpuUtilPercent memoryUtilPercent } } } }", {"id": sys.argv[2]})
    elif a == "terminate":
        d = gql("mutation T($id: String!) { podTerminate(input: {podId: $id}) }", {"id": sys.argv[2]})
    else:
        raise SystemExit("unknown action")
    record(a if a != "create" else f"create {sys.argv[2]} {sys.argv[3]}", d)
    print(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
