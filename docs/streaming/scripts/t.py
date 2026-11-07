import asyncio, contextlib, json

def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

async def with_heartbeat(agen, interval: float = 15.0, deadline: float = 600.0):
    """Chèn ': ping' khi upstream im lặng quá `interval` giây; cắt khi vượt `deadline` giây."""
    loop = asyncio.get_running_loop()
    end = loop.time() + deadline
    it = agen.__aiter__()
    nxt = None
    try:
        while True:
            remaining = end - loop.time()
            if remaining <= 0:
                yield sse("error", {"code": "deadline_exceeded", "retryable": False})
                return
            if nxt is None:
                nxt = asyncio.ensure_future(it.__anext__())
            done, _ = await asyncio.wait({nxt}, timeout=min(interval, remaining))
            if not done:
                yield ": ping\n\n"
                continue
            task, nxt = nxt, None
            try:
                item = task.result()
            except StopAsyncIteration:
                return
            yield item
    finally:
        if nxt is not None:
            nxt.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await nxt
        await agen.aclose()            # đóng upstream (và kết nối tới provider) trong mọi trường hợp

state = {}
def mk(name, items, stall=False, delay=0):
    async def src():
        state[name] = "open"
        try:
            for x in items:
                await asyncio.sleep(delay); yield x
            if stall: await asyncio.Future()
        finally:
            state[name] = "closed"
    return src()

async def main():
    out = [x async for x in with_heartbeat(mk("normal", ["a","b"], delay=0.01), interval=0.005)]
    print("normal", out.count(": ping\n\n")>0, [x for x in out if not x.startswith(":")], state["normal"])
    g = with_heartbeat(mk("early", ["a"], stall=True), interval=0.01)
    print("first", await g.__anext__()); await g.aclose(); print("early close ->", state["early"])
    g = with_heartbeat(mk("mid", ["a"], stall=True), interval=0.01)
    await g.__anext__(); await g.__anext__()   # one ping while pending
    await g.aclose(); print("pending close ->", state["mid"])
    out = [x async for x in with_heartbeat(mk("stall", [], stall=True), interval=0.01, deadline=0.05)]
    print("stall pings", sum(x.startswith(":") for x in out), "last", out[-1].split("\n")[0], state["stall"])
    # cancellation of consuming task (Starlette disconnect)
    async def consume():
        async for _ in with_heartbeat(mk("cancel", ["a"], stall=True), interval=0.01): pass
    t = asyncio.create_task(consume()); await asyncio.sleep(0.05); t.cancel()
    with contextlib.suppress(asyncio.CancelledError): await t
    print("task cancel ->", state["cancel"])
asyncio.run(main())
