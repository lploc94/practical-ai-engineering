"""loadgen trên server giả lập: một "slot" duy nhất, mỗi request chiếm slot 0,1 giây."""
import asyncio
import json

import httpx2

import loadgen


def make_client(service_s, reasoning=False, reject_every=0):
    lock = asyncio.Lock()
    count = [0]

    class Stream(httpx2.AsyncByteStream):
        async def __aiter__(self):
            async with lock:                       # server một slot: request sau phải xếp hàng
                await asyncio.sleep(service_s)
                field = "reasoning" if reasoning else "content"
                yield f"data: {json.dumps({'choices': [{'delta': {field: 'x'}}]})}\n\n".encode()
            yield b"data: [DONE]\n\n"

    def handler(req):
        count[0] += 1
        if reject_every and count[0] % reject_every == 0:     # server có giới hạn hàng đợi trả 503
            return httpx2.Response(503)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())

    return httpx2.MockTransport(handler)


async def run(rate, duration, service_s, **kw):
    orig = httpx2.AsyncClient
    transport = make_client(service_s, **kw)
    loadgen.httpx2.AsyncClient = lambda **a: orig(transport=transport, **a)
    try:
        return await loadgen.open_loop("http://x/v1/chat/completions", "m", [[]], rate, duration, seed=3)
    finally:
        loadgen.httpx2.AsyncClient = orig


async def main():
    # 1. tải nhẹ (2 request/giây, phục vụ 0,1 giây): gần như không xếp hàng, TTFT xấp xỉ thời gian phục vụ
    r = await run(2, 5, 0.1)
    assert 5 <= r["requests"] <= 20 and r["ttft_p50"] < 0.3, r
    # 2. quá tải (20 request/giây vào server chỉ làm được 10): hàng đợi dài dần, TTFT p95 vượt xa
    #    thời gian phục vụ và còn phải xả hàng đợi sau khi ngừng gửi
    r = await run(20, 2, 0.1)
    assert r["ttft_p95"] > 1.0 and r["drain_s"] > 1.0, r
    # 3. token reasoning cũng tính là token đầu
    r = await run(2, 2, 0.05, reasoning=True)
    assert r["requests"] > 0 and r["error_rate"] == 0, r
    # 4. một nửa request bị 503: lượt đo không dừng, lỗi được đếm, phân vị tính trên request thành công
    r = await run(4, 3, 0.05, reject_every=2)
    assert r["errors"] == {"http_503": r["requests"] // 2}, r
    assert 0.4 <= r["error_rate"] <= 0.6 and r["ttft_p50"] is not None, r
    # 5. mọi request đều 503: không có phân vị, error_rate 1
    r = await run(4, 2, 0.05, reject_every=1)
    assert r["error_rate"] == 1 and r["ttft_p95"] is None, r
    print("LOADGEN OK")


asyncio.run(main())
