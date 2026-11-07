import json
import random
import sys

import anthropic
import httpx2

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from router import AllFailed, Breaker, Target, call_with_fallback, classify, FATAL, NEXT, RETRY


def ok(model, stop="end_turn"):
    return httpx2.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": model,
        "content": [] if stop == "refusal" else [{"type": "text", "text": "ok"}],
        "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1}})


def err(status, etype, headers=None, code=None):
    e = {"type": etype, "message": "x"}
    if code:
        e["details"] = {"error_code": code}
    return httpx2.Response(status, headers=headers or {}, json={"type": "error", "error": e, "request_id": "req_1"})


class Script:
    """Trả lần lượt các response cho từng model; ghi lại body đã gửi."""
    def __init__(self, per_model):
        self.per_model = {m: list(v) for m, v in per_model.items()}
        self.sent = []

    def __call__(self, request):
        body = json.loads(request.content)
        self.sent.append(body)
        return self.per_model[body["model"]].pop(0)


def client(script):
    return anthropic.Anthropic(api_key="k", max_retries=0,
                               http_client=httpx2.Client(transport=httpx2.MockTransport(script)))


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def sleep(self, s): self.slept.append(s); self.t += s


def run(per_model, models, *, request=None, breakers=None, overrides=None, **kw):
    s = Script(per_model)
    c = client(s)
    clk = Clock(); clk.slept = []
    targets = [Target(m, c, (overrides or {}).get(m, {})) for m in models]
    breakers = breakers or {t.key: Breaker(threshold=5, cooldown_s=30, clock=clk) for t in targets}
    req = request or {"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
    kw.setdefault("deadline_s", 60); kw.setdefault("max_attempts", 6)
    kw.setdefault("attempts_per_target", 2); kw.setdefault("max_wait_s", 10)
    try:
        resp, log = call_with_fallback(req, targets, breakers, sleep=clk.sleep, clock=clk, **kw)
        return resp, log, s, clk
    except Exception as e:
        return e, getattr(e, "log", None), s, clk


A, B = "claude-opus-5-5", "claude-opus-4-8"
random.seed(1)

# 1. 529 -> retry once on A (backoff), then A ok
r, log, s, clk = run({A: [err(529, "overloaded_error"), ok(A)]}, [A, B])
assert r.model == A and [m["model"] for m in s.sent] == [A, A] and len(clk.slept) == 1 and clk.slept[0] <= 0.5, (log, clk.slept)

# 2. 529 twice on A -> fallback to B
r, log, s, clk = run({A: [err(529, "overloaded_error")] * 2, B: [ok(B)]}, [A, B])
assert r.model == B and [m["model"] for m in s.sent] == [A, A, B], log

# 3. spend cap 429 (no retry-after) -> FATAL, raise immediately, no fallback
r, log, s, clk = run({A: [err(429, "rate_limit_error", code="enforced_spend_limit_reached")]}, [A, B])
assert isinstance(r, anthropic.RateLimitError) and len(s.sent) == 1 and clk.slept == [], r

# 4. 429 with retry-after 30 > max_wait 10 -> don't wait, go to B (different bucket)
r, log, s, clk = run({A: [err(429, "rate_limit_error", headers={"retry-after": "30"})], B: [ok(B)]}, [A, B])
assert r.model == B and clk.slept == [], (log, clk.slept)

# 5. 429 with retry-after 2 -> wait exactly 2, retry A
r, log, s, clk = run({A: [err(429, "rate_limit_error", headers={"retry-after": "2"}), ok(A)]}, [A, B])
assert r.model == A and clk.slept == [2.0], clk.slept

# 6. 400 -> FATAL, no fallback (request bug would fail on B too)
r, log, s, clk = run({A: [err(400, "invalid_request_error")]}, [A, B])
assert isinstance(r, anthropic.BadRequestError) and len(s.sent) == 1

# 7. 404 -> NEXT without retry
r, log, s, clk = run({A: [err(404, "not_found_error")], B: [ok(B)]}, [A, B])
assert r.model == B and [m["model"] for m in s.sent] == [A, B] and clk.slept == []

# 8. refusal (HTTP 200) -> next model; all refuse -> last refusal returned, not raised
r, log, s, clk = run({A: [ok(A, "refusal")], B: [ok(B)]}, [A, B])
assert r.model == B and log[0] == (A, "refusal")
r, log, s, clk = run({A: [ok(A, "refusal")], B: [ok(B, "refusal")]}, [A, B])
assert r.stop_reason == "refusal" and r.model == B

# 9. max_attempts caps whole chain
r, log, s, clk = run({A: [err(500, "api_error")] * 2, B: [err(500, "api_error")] * 2}, [A, B], max_attempts=3)
assert isinstance(r, AllFailed) and len(s.sent) == 3, log

# 10. deadline: backoff that would exceed the deadline -> stop waiting, move on; timeout passed per attempt
r, log, s, clk = run({A: [err(503, "api_error", headers={"retry-after": "5"})], B: [ok(B)]}, [A, B], deadline_s=4)
assert r.model == B and clk.slept == [], (log, clk.slept)

# 11. forced tool_choice skips models that reject it
req = {"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}],
       "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
       "tool_choice": {"type": "tool", "name": "f"}}
H = "claude-haiku-5-5"
r, log, s, clk = run({H: [ok(H)]}, [A, H], request=req)
assert r.model == H and [m["model"] for m in s.sent] == [H] and "forced" in log[0][1], log

# 12. overrides: None removes a parameter for that model only
S = "claude-sonnet-5-5"
req = {"max_tokens": 10, "thinking": {"type": "between_tools"}, "messages": [{"role": "user", "content": "hi"}]}
r, log, s, clk = run({S: [err(529, "overloaded_error")] * 2, B: [ok(B)]}, [S, B], request=req,
                     overrides={B: {"thinking": None, "max_tokens": 40}})
assert s.sent[0]["thinking"] == {"type": "between_tools"} and "thinking" not in s.sent[2] and s.sent[2]["max_tokens"] == 40, s.sent

# 13. breaker opens after threshold, skips model, half-open allows one probe after cooldown
clk = Clock(); clk.slept = []
b = Breaker(threshold=2, cooldown_s=30, clock=clk)
b.failure(); assert b.allow()
b.failure(); assert not b.allow()
clk.t = 31; assert b.allow() and not b.allow()      # một probe duy nhất
b.failure(); assert not b.allow()                   # probe lỗi -> mở lại
clk.t = 62; assert b.allow(); b.success(); assert b.allow() and b.allow()
# 429 retry-after longer than cooldown holds the breaker open that long
b = Breaker(threshold=1, cooldown_s=5, clock=clk); clk.t = 0
b.failure(hold_s=60); clk.t = 30; assert not b.allow(); clk.t = 61; assert b.allow()

# 14. open breaker on A -> straight to B
clk0 = Clock(); clk0.slept = []
brs = {("claude-api", A): Breaker(threshold=1, cooldown_s=30, clock=clk0), ("claude-api", B): Breaker(threshold=5, cooldown_s=30, clock=clk0)}
brs[("claude-api", A)].failure()
r, log, s, clk = run({B: [ok(B)]}, [A, B], breakers=brs)
assert r.model == B and [m["model"] for m in s.sent] == [B] and log[0] == (A, "skip: breaker mở")

# 15. connection error -> retry
def boom(request): raise httpx2.ConnectError("down", request=request)
c = anthropic.Anthropic(api_key="k", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(boom)))
s2 = Script({B: [ok(B)]})
clk = Clock(); clk.slept = []
resp, log = call_with_fallback({"max_tokens": 5, "messages": [{"role": "user", "content": "x"}]},
                               [Target(A, c), Target(B, client(s2))], {t: Breaker(threshold=5, cooldown_s=30, clock=clk) for t in (("claude-api", A), ("claude-api", B))},
                               deadline_s=60, max_attempts=5, attempts_per_target=2, max_wait_s=10, sleep=clk.sleep, clock=clk)
assert resp.model == B and [x[1] for x in log[:2]] == ["retry: APIConnectionError"] * 2, log

# classify() against real SDK exceptions
assert classify(anthropic.APITimeoutError(request=httpx2.Request("POST", "https://x")))[0] == RETRY

# 16. same model on two providers: breakers are independent
from router import _retry_after
s3 = Script({A: [err(529, "overloaded_error")]}); s4 = Script({A: [ok(A)]})
clk = Clock(); clk.slept = []
tA, tB = Target(A, client(s3)), Target(A, client(s4), provider="bedrock")
brs = {t.key: Breaker(threshold=1, cooldown_s=30, clock=clk) for t in (tA, tB)}
resp, log = call_with_fallback({"max_tokens": 5, "messages": [{"role": "user", "content": "x"}]}, [tA, tB], brs,
                               deadline_s=60, max_attempts=5, attempts_per_target=1, max_wait_s=10, sleep=clk.sleep, clock=clk)
assert resp.model == A and len(s4.sent) == 1 and not brs[tA.key].allow() and brs[tB.key].allow(), log

# 17. 404 does not count toward the breaker
clk = Clock(); clk.slept = []
b404 = {("claude-api", A): Breaker(threshold=1, cooldown_s=30, clock=clk), ("claude-api", B): Breaker(threshold=1, cooldown_s=30, clock=clk)}
r, log, s, clk = run({A: [err(404, "not_found_error")], B: [ok(B)]}, [A, B], breakers=b404)
assert r.model == B and b404[("claude-api", A)].failures == 0, log

# 18. 409 conflict is retried, like the SDK
r, log, s, clk = run({A: [err(409, "conflict_error"), ok(A)]}, [A, B])
assert r.model == A and [m["model"] for m in s.sent] == [A, A], log

# 19. retry-after as HTTP-date; invalid and negative values ignored
import email.utils
now = 1_800_000_000.0
assert _retry_after({"retry-after": email.utils.formatdate(now + 7, usegmt=True)}, now=lambda: now) == 7
assert _retry_after({"retry-after": email.utils.formatdate(now - 7, usegmt=True)}, now=lambda: now) is None
assert _retry_after({"retry-after": "0"}) is None and _retry_after({"retry-after-ms": "0"}) is None
assert _retry_after({"retry-after": "soon"}) is None and _retry_after({"retry-after": "-3"}) is None
assert _retry_after({"retry-after": "nan"}) is None and _retry_after({"retry-after-ms": "1500"}) == 1.5
# 20. 429 lặp lại với retry-after đã hết hạn: vẫn backoff, không retry ngay
import time
past = {"retry-after": email.utils.formatdate(time.time() - 7, usegmt=True)}
r, log, s, clk = run({A: [err(429, "rate_limit_error", headers=past), ok(A)]}, [A, B])
assert r.model == A and len(s.sent) == 2 and len(clk.slept) == 1 and clk.slept[0] > 0, (log, clk.slept)
# 21. mọi 5xx đều retry như SDK, kể cả 501, 505, 599
for st in (501, 505, 599):
    r, log, s, clk = run({A: [err(st, "api_error"), ok(A)]}, [A, B])
    assert r.model == A and [m["model"] for m in s.sent] == [A, A], (st, log)
# 22. x-should-retry của server thắng quy tắc theo status, như SDK; spend cap vẫn dừng
import anthropic, httpx2 as _h
def _e(st, headers, code=None):
    res = err(st, "api_error", headers=headers, code=code); res.request = _h.Request("POST", "https://x")
    return anthropic.APIStatusError(message="x", response=res, body=res.json())
assert classify(_e(503, {"x-should-retry": "false"})) == ("fatal", None)
assert classify(_e(400, {"x-should-retry": "true"})) == ("retry", None)
assert classify(_e(404, {"x-should-retry": "false"})) == ("next", None)
assert classify(_e(429, {"x-should-retry": "true"}, code="enforced_spend_limit_reached")) == ("fatal", None)
r, log, s, clk = run({A: [err(400, "invalid_request_error", headers={"x-should-retry": "true"}), ok(A)]}, [A, B])
assert r.model == A and len(s.sent) == 2, log
print("ALL OK")
