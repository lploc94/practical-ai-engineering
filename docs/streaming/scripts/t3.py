import asyncio, contextlib, json

def sse(event, data): return f"event: {event}\ndata: {json.dumps(data)}\n\n"

_END = object()

async def with_heartbeat(agen, interval: float = 15.0, deadline: float = 600.0):
    queue: asyncio.Queue = asyncio.Queue(maxsize=32)
    timed_out = False

    async def pump():
        nonlocal timed_out
        cm = asyncio.timeout(deadline)
        try:
            async with cm:
                async for item in agen:
                    await queue.put(item)
        except TimeoutError:
            if not cm.expired():
                raise
            timed_out = True
        finally:
            await agen.aclose()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(_END)

    task = asyncio.create_task(pump())
    try:
        while not (task.done() and queue.empty()):
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            if item is _END:
                break
            yield item
        await task
        if timed_out:
            yield sse("error", {"code": "deadline_exceeded", "retryable": False})
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

state = {}
def src(name, items, gap=0.0, hang=False, boom=False):
    async def g():
        state[name] = "open"
        try:
            for i in items:
                await asyncio.sleep(gap); yield i
            if boom: raise ValueError("boom")
            if hang: await asyncio.Future()
        finally:
            state[name] = "closed"
    return g()

async def collect(g, limit=None):
    out = []
    async for x in g:
        out.append(x)
        if limit and len(out) >= limit: break
    return out

async def main():
    # 1 normal
    print("normal", await collect(with_heartbeat(src("a", ["x","y"]), .05, 1)), state["a"])
    # 2 slow -> pings
    out = await collect(with_heartbeat(src("b", ["x"], gap=.12), .05, 1))
    print("slow pings", out.count(": ping\n\n"), out[-1], state["b"])
    # 3 silent upstream -> deadline
    out = await collect(with_heartbeat(src("c", ["x"], hang=True), .05, .2))
    print("deadline", "deadline_exceeded" in out[-1], state["c"])
    # 4 stalled consumer past deadline
    g = with_heartbeat(src("d", ["x"], hang=True), .05, .1)
    print("first", await g.__anext__())
    await asyncio.sleep(.3)
    print("stalled consumer: upstream", state["d"])
    rest = await collect(g); print("rest", rest)
    # 5 client disconnect
    g = with_heartbeat(src("e", ["x"], hang=True), .05, 5)
    await g.__anext__(); await g.aclose(); await asyncio.sleep(0)
    print("disconnect upstream", state["e"])
    # 6 exception propagates
    try:
        await collect(with_heartbeat(src("f", ["x"], boom=True), .05, 1)); print("no raise!")
    except ValueError: print("raised", state["f"])
    # 7 many items > queue size
    out = await collect(with_heartbeat(src("h", list(range(100))), .05, 1))
    print("many", len(out), state["h"])
    # 8 break early
    out = await collect(with_heartbeat(src("i", list(range(100)), gap=.001), .05, 1), limit=3)
    await asyncio.sleep(.01); print("early", out, state["i"])
    async def tmo():
        state['t']='open'
        try:
            yield 'x'; raise TimeoutError('provider read timeout')
        finally: state['t']='closed'
    try:
        print(await collect(with_heartbeat(tmo(), .05, 1)))
    except TimeoutError as e: print('upstream timeout propagated:', e, state['t'])
    print("tasks left", len(asyncio.all_tasks()) - 1)

asyncio.run(main())
