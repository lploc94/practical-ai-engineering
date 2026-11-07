"""bench.py trên SSE giả lập: đếm token từ usage, bỏ qua chunk không có chữ, báo lỗi khi thiếu usage."""
import asyncio
import json

import httpx2

import bench


def sse(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"


def client(body):
    return httpx2.AsyncClient(transport=httpx2.MockTransport(
        lambda req: httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)))


async def main():
    role = {"choices": [{"delta": {"role": "assistant"}}]}
    text = lambda t: {"choices": [{"delta": {"content": t}}]}
    usage = {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 3}}
    # 1. chunk role và chunk usage không tính là chữ; token lấy từ usage
    async with client(sse([role, text("a"), text("b"), text("c"), usage])) as c:
        r = await bench.one(c, "http://x/v1/chat/completions", "m", [], 3, {})
    assert r["out"] == 3 and r["in"] == 50 and r["ttft"] <= r["e2e"], r
    # 2. server không trả usage: báo lỗi thay vì đếm sai
    async with client(sse([role, text("a")])) as c:
        try:
            await bench.one(c, "http://x/v1/chat/completions", "m", [], 3, {})
        except RuntimeError:
            pass
        else:
            raise AssertionError("thiếu usage phải báo lỗi")
    # 3. model reasoning: 100 token suy nghĩ rồi 10 token trả lời, mỗi token 10 ms. Tốc độ phải tính
    #    trên cả hai phần (khoảng 100 token/giây), TTFT tính từ token suy nghĩ đầu tiên
    class Slow(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield sse([role])[:-len(b"data: [DONE]\n\n")]
            for i in range(110):
                await asyncio.sleep(0.01)
                d = {"reasoning": "r"} if i < 100 else {"content": "c"}
                yield f"data: {json.dumps({'choices': [{'delta': d}]})}\n\n".encode()
            yield sse([{"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 110}}])
    slow = httpx2.AsyncClient(transport=httpx2.MockTransport(
        lambda req: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=Slow())))
    async with slow as c:
        r = await bench.one(c, "http://x/v1/chat/completions", "m", [], 110, {})
    assert 50 < r["decode_tps"] < 110, r
    assert r["ttft"] < 0.1 < r["ttft_content"], r
    # 4. phân vị
    assert bench.pct([5, 1, 3, 2, 4], 50) == 3 and bench.pct([5, 1, 3, 2, 4], 95) == 5
    print("BENCH OK")


asyncio.run(main())
