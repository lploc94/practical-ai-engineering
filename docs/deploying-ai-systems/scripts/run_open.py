"""Chạy open_loop ở nhiều tốc độ đến trên llama-server (Qwen2.5-3B Q4_K_M, 8 slot)."""
import asyncio
import json
import sys

from loadgen import open_loop
from run_quant import URL, start

model, duration = sys.argv[1], float(sys.argv[2])
rates = [float(x) for x in sys.argv[3:]]
prompts = json.load(open("prompts.json", encoding="utf-8"))
p = start(model, "-c", "16384", "-np", "8")
try:
    for rate in rates:
        r = asyncio.run(open_loop(URL, "m", prompts, rate, duration,
                                  extra={"ignore_eos": True, "cache_prompt": False}, seed=1))
        print(json.dumps(r), flush=True)
finally:
    p.terminate()
    p.wait()
