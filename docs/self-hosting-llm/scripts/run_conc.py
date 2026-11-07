"""Một model, nhiều mức đồng thời: TTFT, tốc độ mỗi request, throughput tổng."""
import asyncio
import json
import sys

from bench import run
from run_quant import URL, start

model, np_slots = sys.argv[1], sys.argv[2]
prompts = json.load(open("prompts.json", encoding="utf-8"))[:16]
p = start(model, "-c", str(2048 * int(np_slots)), "-np", np_slots)
try:
    for conc in map(int, sys.argv[3:]):
        r = asyncio.run(run(URL, "m", prompts, conc, max_tokens=128, extra={"ignore_eos": True, "cache_prompt": False}))
        print(json.dumps({"slots": int(np_slots), **r}), flush=True)
finally:
    p.terminate()
    p.wait()
