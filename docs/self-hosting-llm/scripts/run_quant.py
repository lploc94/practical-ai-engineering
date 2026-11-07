"""Với mỗi file GGUF: khởi động llama-server, đo tốc độ một luồng, độ chính xác trên tác vụ
trích xuất intent (toy.jsonl của bài fine-tuning) và thời gian. In một dòng JSON mỗi file."""
import asyncio
import glob
import json
import os
import subprocess
import sys
import time

import httpx2

from bench import run

PORT = 8095
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"


def start(model, *args):
    p = subprocess.Popen(["llama-server", "-m", model, "--port", str(PORT), *args],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            if httpx2.get(f"http://127.0.0.1:{PORT}/health").status_code == 200:
                return p
        except httpx2.HTTPError:
            pass
        time.sleep(0.5)
    p.kill()
    raise RuntimeError("server không lên")


async def accuracy(rows, concurrency=8):
    sem = asyncio.Semaphore(concurrency)
    async with httpx2.AsyncClient(timeout=120) as c:
        async def one(r):
            async with sem:
                resp = await c.post(URL, json={"model": "m", "messages": r["messages"][:-1],
                                               "max_tokens": 64, "temperature": 0})
                text = resp.json()["choices"][0]["message"]["content"].strip()
                try:
                    return json.loads(text) == json.loads(r["messages"][-1]["content"])
                except json.JSONDecodeError:
                    return False
        hits = await asyncio.gather(*(one(r) for r in rows))
    return sum(hits) / len(rows)


def main(models, toy_path, prompts_path):
    seen, rows = set(), []
    for line in open(toy_path, encoding="utf-8"):
        r = json.loads(line)
        key = r["messages"][1]["content"]
        if key not in seen:
            seen.add(key)
            rows.append(r)
    prompts = json.load(open(prompts_path, encoding="utf-8"))[:8]
    for m in models:
        p = start(m, "-c", "16384", "-np", "8")
        try:
            speed = asyncio.run(run(URL, "m", prompts, 1, max_tokens=128, extra={"ignore_eos": True, "cache_prompt": False}))
            acc = asyncio.run(accuracy(rows))
        finally:
            p.terminate()
            p.wait()
        size = sum(os.path.getsize(f) for f in glob.glob(m.replace("-00001-of-", "-*-of-")))
        print(json.dumps({"model": os.path.basename(m), "gib": round(size / 2**30, 2), "rows": len(rows),
                          "exact": round(acc, 3), **speed}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[3:], sys.argv[1], sys.argv[2])
