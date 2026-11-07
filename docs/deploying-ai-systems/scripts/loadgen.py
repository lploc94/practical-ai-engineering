"""Tải open-loop cho endpoint OpenAI-compatible: request đến theo tiến trình Poisson với tốc độ cố
định, không chờ request trước xong, giống người dùng thật. TTFT và thời gian tính từ lúc request
lẽ ra được gửi, nên hàng đợi ở bất kỳ đâu (client, proxy, server) đều bị tính vào."""
import asyncio
import json
import random
import time

import httpx2

TEXT_FIELDS = ("content", "reasoning", "reasoning_content")


async def _one(client, url, body, scheduled):
    """Một request. Lỗi (HTTP 503 khi server từ chối, timeout, stream không có token) được trả về
    như một kết quả, không raise: vùng quá tải là chỗ cần đo nhất, và tỉ lệ lỗi cũng là một SLI."""
    first = None
    try:
        async with client.stream("POST", url, json=body) as r:
            if r.status_code != 200:
                return {"error": f"http_{r.status_code}"}
            async for line in r.aiter_lines():
                if first is None and line.startswith("data: ") and line != "data: [DONE]":
                    choices = json.loads(line[6:]).get("choices", [])
                    if any(c["delta"].get(f) for c in choices for f in TEXT_FIELDS):
                        first = time.perf_counter()
    except httpx2.HTTPError as e:
        return {"error": type(e).__name__}
    if first is None:
        return {"error": "no_tokens"}
    return {"ttft": first - scheduled, "e2e": time.perf_counter() - scheduled}


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 2)


async def open_loop(url, model, prompts, rate, duration, *, max_tokens=128, extra=None, seed=0):
    """Gửi request trong `duration` giây, trung bình `rate` request mỗi giây, rồi chờ tất cả xong."""
    rng = random.Random(seed)
    limits = httpx2.Limits(max_connections=None, max_keepalive_connections=None)
    tasks, t = [], 0.0
    async with httpx2.AsyncClient(timeout=900, limits=limits) as client:
        start = time.perf_counter()
        while (t := t + rng.expovariate(rate)) < duration:
            await asyncio.sleep(max(0.0, start + t - time.perf_counter()))
            body = {"model": model, "messages": prompts[len(tasks) % len(prompts)],
                    "max_tokens": max_tokens, "temperature": 0, "stream": True, **(extra or {})}
            tasks.append(asyncio.create_task(_one(client, url, body, start + t)))
        res = await asyncio.gather(*tasks)
        span = time.perf_counter() - start
    ok = [r for r in res if "error" not in r]
    errors = {}
    for r in res:
        if "error" in r:
            errors[r["error"]] = errors.get(r["error"], 0) + 1
    ttft, e2e = [r["ttft"] for r in ok], [r["e2e"] for r in ok]
    # phân vị chỉ tính trên request thành công; đọc kèm error_rate
    return {"rate": rate, "requests": len(res),
            "error_rate": round(1 - len(ok) / len(res), 3) if res else 0.0, "errors": errors,
            "ttft_p50": pct(ttft, 50), "ttft_p95": pct(ttft, 95),
            "e2e_p50": pct(e2e, 50), "e2e_p95": pct(e2e, 95),
            # số request trung bình nằm trong hệ thống = tổng thời gian ở lại / thời gian đo
            "in_system_mean": round(sum(e2e) / span, 1),
            "drain_s": round(span - duration, 1)}
