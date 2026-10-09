#!/bin/bash
# Local dead-man switch: terminate the pod after $2 minutes no matter what. Needs RUNPOD_API_KEY in the environment.
POD=$1; MIN=$2
sleep $((MIN*60))
python3 "$(cd "$(dirname "$0")" && pwd)/rp.py" terminate "$POD"
echo "watchdog terminated $POD at $(date -u +%FT%TZ)"
