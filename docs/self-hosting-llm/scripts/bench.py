"""Đo TTFT, tốc độ sinh mỗi request và throughput tổng của một endpoint OpenAI-compatible
(vLLM, SGLang, llama-server, Ollama) ở nhiều mức đồng thời. Mỗi request stream và đếm token
từ `usage` của chunk cuối."""
import asyncio
import json
import statistics
import time

import httpx2


async def one(client, url, model, messages, max_tokens, extra):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, **extra}
    t0 = time.perf_counter()
    first = last = first_content = None
    usage = None
    async with client.stream("POST", url, json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for c in chunk.get("choices", []):
                d = c["delta"]
                # completion_tokens gồm cả token suy nghĩ, nên đo thời gian theo mọi loại chữ;
                # vLLM gọi trường đó là `reasoning`, nhiều engine khác gọi `reasoning_content`
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    last = time.perf_counter()
                    first = first or last
                    if d.get("content"):
                        first_content = first_content or last
    if usage is None or first is None:
        raise RuntimeError("server không trả usage hoặc không sinh chữ nào")
    n = usage["completion_tokens"]
    return {"ttft": first - t0, "ttft_content": first_content and first_content - t0,
            "e2e": last - t0, "out": n, "in": usage["prompt_tokens"],
            "decode_tps": (n - 1) / (last - first) if n > 1 and last > first else None}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


async def run(url, model, prompts, concurrency, max_tokens=128, extra=None):
    """Chạy hết `prompts`, tối đa `concurrency` request cùng lúc."""
    sem = asyncio.Semaphore(concurrency)
    async with httpx2.AsyncClient(timeout=600) as client:
        async def task(p):
            async with sem:
                return await one(client, url, model, p, max_tokens, extra or {})
        t0 = time.perf_counter()
        res = await asyncio.gather(*(task(p) for p in prompts))
        wall = time.perf_counter() - t0
    ttft = [r["ttft"] for r in res]
    tps = [r["decode_tps"] for r in res if r["decode_tps"]]
    return {"concurrency": concurrency, "requests": len(res),
            "prompt_tokens_mean": round(statistics.mean(r["in"] for r in res)),
            "ttft_p50": round(pct(ttft, 50), 3), "ttft_p95": round(pct(ttft, 95), 3),
            "decode_tps_p50": round(statistics.median(tps), 1),
            "output_tps_total": round(sum(r["out"] for r in res) / wall, 1),
            "wall_s": round(wall, 1)}
