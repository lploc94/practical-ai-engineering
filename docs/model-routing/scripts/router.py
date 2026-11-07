import email.utils
import math
import random
import threading
import time
from dataclasses import dataclass, field

import anthropic

RETRY, NEXT, FATAL = "retry", "next", "fatal"

# Model không nhận tool_choice any/tool (trang lỗi của Claude, tháng 11/2026).
NO_FORCED_TOOL = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1"}


def classify(err: anthropic.APIError) -> tuple[str, float | None]:
    """Trả về (hành động, số giây server yêu cầu chờ hoặc None)."""
    if isinstance(err, anthropic.APIConnectionError):     # gồm cả APITimeoutError
        return RETRY, None
    if not isinstance(err, anthropic.APIStatusError):
        return FATAL, None
    status = err.status_code
    retry_after = _retry_after(err.response.headers)
    if status == 429 and err.error_code == "enforced_spend_limit_reached":
        return FATAL, None            # hết spend cap của tổ chức: model nào cũng bị chặn
    should_retry = err.response.headers.get("x-should-retry")    # server quyết thay, như SDK
    if should_retry == "true":
        return RETRY, retry_after
    if status == 404:
        return NEXT, None             # model ID sai hoặc model đã ngừng phục vụ
    if should_retry != "false" and (status in (408, 409, 429) or status >= 500):   # cùng quy tắc SDK
        return RETRY, retry_after
    return FATAL, None                # 400, 401, 402, 403, 413...: lỗi của request hoặc tài khoản


def _retry_after(headers, now=time.time) -> float | None:
    """`retry-after-ms`, rồi `retry-after` dạng số giây hoặc HTTP-date, như SDK. Giá trị
    không dương (kể cả ngày đã qua) trả None để caller tự backoff, cũng như SDK."""
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        try:
            v = float(headers[name]) / scale
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0:
            return v
    try:
        when = email.utils.parsedate_to_datetime(headers["retry-after"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:                       # HTTP-date luôn là GMT
        return None
    wait = when.timestamp() - now()
    return wait if wait > 0 else None


class Breaker:
    """Circuit breaker cho một target: mở sau `threshold` lỗi liên tiếp, thử lại sau `cooldown_s`."""

    def __init__(self, *, threshold: int, cooldown_s: float, clock=time.monotonic):
        self.threshold, self.cooldown_s, self.clock = threshold, cooldown_s, clock
        self.failures = 0
        self.open_until = 0.0
        self.probe_started: float | None = None
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            now = self.clock()
            if self.failures < self.threshold:
                return True
            if now < self.open_until:
                return False
            # half-open: mỗi lần chỉ một request thăm dò
            if self.probe_started is not None and now - self.probe_started < self.cooldown_s:
                return False
            self.probe_started = now
            return True

    def success(self) -> None:
        with self.lock:
            self.failures, self.probe_started = 0, None

    def failure(self, hold_s: float | None = None) -> None:
        with self.lock:
            self.failures += 1
            self.probe_started = None
            if self.failures >= self.threshold:
                self.open_until = self.clock() + max(self.cooldown_s, hold_s or 0.0)


@dataclass
class Target:
    model: str
    client: anthropic.Anthropic                     # tạo với max_retries=0
    overrides: dict = field(default_factory=dict)   # giá trị None nghĩa là bỏ tham số đó
    provider: str = "claude-api"                    # cùng model qua Bedrock, Vertex: provider khác

    @property
    def key(self) -> tuple[str, str]:               # breaker theo (provider, model)
        return self.provider, self.model


class AllFailed(Exception):
    def __init__(self, log, last_error):
        super().__init__(f"mọi model đều lỗi: {log}")
        self.log, self.last_error = log, last_error


def _backoff(attempt: int, base: float = 0.5, cap: float = 8.0) -> float:
    return random.uniform(0, min(cap, base * 2 ** attempt))     # full jitter


def call_with_fallback(request: dict, targets: list[Target], breakers: dict[tuple[str, str], Breaker], *,
                       deadline_s: float, max_attempts: int, attempts_per_target: int,
                       max_wait_s: float, sleep=time.sleep, clock=time.monotonic):
    start = clock()
    attempts, log = 0, []
    last_error, last_refusal = None, None
    forced = (request.get("tool_choice") or {}).get("type") in ("any", "tool")
    for t in targets:
        if forced and t.model in NO_FORCED_TOOL:
            log.append((t.model, "skip: không hỗ trợ forced tool_choice"))
            continue
        body = {k: v for k, v in {**request, **t.overrides}.items() if v is not None}
        for i in range(attempts_per_target):
            remaining = deadline_s - (clock() - start)
            if attempts >= max_attempts or remaining <= 0:
                return _give_up(log, last_error, last_refusal)
            if not breakers[t.key].allow():
                log.append((t.model, "skip: breaker mở"))
                break
            attempts += 1
            try:
                resp = t.client.messages.create(**body, model=t.model, timeout=remaining)
            except anthropic.APIError as e:
                action, wait = classify(e)
                last_error = e
                log.append((t.model, f"{action}: {getattr(e, 'status_code', type(e).__name__)}"))
                if action == FATAL:
                    raise
                if action == NEXT:
                    break             # 404 là lỗi cấu hình, không tính vào breaker
                breakers[t.key].failure(hold_s=wait)
                if wait is None:
                    wait = _backoff(i)
                if i + 1 == attempts_per_target or wait > min(max_wait_s, deadline_s - (clock() - start)):
                    break             # không retry sớm hơn server yêu cầu; sang model kế
                sleep(wait)
                continue
            breakers[t.key].success()
            if resp.stop_reason == "refusal":
                log.append((t.model, "refusal"))
                last_refusal = resp
                break                 # gửi lại cùng model thường bị từ chối tiếp
            log.append((t.model, "ok"))
            return resp, log
    return _give_up(log, last_error, last_refusal)


def _give_up(log, last_error, last_refusal):
    if last_refusal is not None:
        return last_refusal, log      # mọi model đều từ chối: trả refusal cuối, không coi là lỗi hạ tầng
    raise AllFailed(log, last_error)
