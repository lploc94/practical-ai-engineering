import asyncio
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind, Status, StatusCode

tracer = trace.get_tracer("app.llm")
meter = metrics.get_meter("app.llm")

DURATION_BUCKETS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56,
                    5.12, 10.24, 20.48, 40.96, 81.92]
duration_hist = meter.create_histogram(
    "gen_ai.client.inference.duration", unit="s",
    explicit_bucket_boundaries_advisory=DURATION_BUCKETS)
ttfc_hist = meter.create_histogram(
    "gen_ai.client.inference.time_to_first_chunk", unit="s",
    explicit_bucket_boundaries_advisory=DURATION_BUCKETS)
usage_counters = {
    name: meter.create_counter(f"gen_ai.client.inference.usage.{name}", unit="{token}")
    for name in ("input_tokens", "output_tokens",
                 "cache_read.input_tokens", "cache_write.input_tokens",
                 "reasoning.output_tokens")
}
cost_counter = meter.create_counter("app.llm.cost", unit="USD")  # không có trong semconv

# Lý do kết thúc bình thường. Mọi lý do khác (bị cắt, refusal, safety, lý do mới chưa biết) đều giữ trace.
# OpenAI Responses: ghi "completed" khi status=completed, ngược lại ghi incomplete_details.reason.
NORMAL_FINISH = {"stop", "end_turn", "stop_sequence", "tool_use", "tool_calls", "pause_turn", "completed"}
log = logging.getLogger("app.llm.telemetry")
telemetry_errors = meter.create_counter("app.llm.telemetry.errors")


def normalize_usage(provider: str, u: dict) -> dict:
    """Đưa usage của provider về một quy ước: input gồm cả cache, output gồm cả reasoning."""
    if provider == "anthropic":
        read = u.get("cache_read_input_tokens") or 0
        write = u.get("cache_creation_input_tokens") or 0
        details = u.get("output_tokens_details") or {}
        return {"input_tokens": u["input_tokens"] + read + write,   # input_tokens chỉ là phần sau breakpoint
                "output_tokens": u["output_tokens"],               # đã gồm thinking
                "cache_read.input_tokens": read,
                "cache_write.input_tokens": write,
                "reasoning.output_tokens": details.get("thinking_tokens") or 0}
    if provider == "openai":                                       # Responses API
        idet = u.get("input_tokens_details") or {}
        odet = u.get("output_tokens_details") or {}
        return {"input_tokens": u["input_tokens"],                 # đã gồm cached
                "output_tokens": u["output_tokens"],               # đã gồm reasoning
                "cache_read.input_tokens": idet.get("cached_tokens") or 0,
                "cache_write.input_tokens": idet.get("cache_write_tokens") or 0,
                "reasoning.output_tokens": odet.get("reasoning_tokens") or 0}
    if provider == "gcp.gemini":                                   # generateContent, usageMetadata
        thoughts = u.get("thoughtsTokenCount") or 0
        return {"input_tokens": u.get("promptTokenCount", 0) + (u.get("toolUsePromptTokenCount") or 0),
                "output_tokens": (u.get("candidatesTokenCount") or 0) + thoughts,  # thoughts tách riêng
                "cache_read.input_tokens": u.get("cachedContentTokenCount") or 0,
                "cache_write.input_tokens": 0,
                "reasoning.output_tokens": thoughts}
    raise ValueError(f"chưa có mapping usage cho provider {provider!r}")


@dataclass
class Price:            # USD / 1 triệu token
    input: float
    cache_read: float
    cache_write: float
    output: float


def find_price(prices: dict[str, Price], response_model: str) -> Price | None:
    """Khớp tiền tố dài nhất: 'model-a-2026-05-01' dùng giá của 'model-a'."""
    keys = [k for k in prices if response_model == k or response_model.startswith(k + "-")]
    return prices[max(keys, key=len)] if keys else None


def cost_usd(n: dict, p: Price) -> float:
    uncached = n["input_tokens"] - n["cache_read.input_tokens"] - n["cache_write.input_tokens"]
    return (uncached * p.input + n["cache_read.input_tokens"] * p.cache_read
            + n["cache_write.input_tokens"] * p.cache_write
            + n["output_tokens"] * p.output) / 1_000_000          # reasoning đã nằm trong output


@dataclass
class Inference:
    provider: str
    response_id: str | None = None
    response_model: str | None = None
    finish_reasons: list[str] = field(default_factory=list)
    usage: dict | None = None            # usage thô của provider
    first_chunk_at: float | None = None

    def mark_first_chunk(self):
        if self.first_chunk_at is None:
            self.first_chunk_at = time.monotonic()


@contextmanager
def traced_inference(*, provider, operation, request_model, server_address,
                     prices, stream=False, app_attrs=None, metric_attrs=None):
    """app_attrs: tenant, feature, prompt version... gắn lên span.
    metric_attrs: chỉ những chiều có số giá trị nhỏ (feature, tier), gắn lên metric."""
    attrs = {"gen_ai.operation.name": operation,
             "gen_ai.provider.name": provider,
             "gen_ai.request.model": request_model,
             "server.address": server_address}
    if stream:
        attrs["gen_ai.request.stream"] = True
    rec = Inference(provider=provider)
    start = time.monotonic()
    # các attribute dùng cho sampling phải có ngay lúc tạo span
    with tracer.start_as_current_span(f"{operation} {request_model}", kind=SpanKind.CLIENT,
                                      attributes={**attrs, **(app_attrs or {})},
                                      record_exception=False, set_status_on_exception=False) as span:
        error_type, cancelled = None, False
        try:
            yield rec
        except (GeneratorExit, asyncio.CancelledError):
            cancelled = True                   # người dùng ngắt stream: không phải lỗi provider
            span.set_attribute("app.llm.cancelled", True)
            raise
        except BaseException as e:
            error_type = getattr(e, "status_code", None) or type(e).__qualname__
            span.set_attribute("error.type", str(error_type))
            span.set_status(Status(StatusCode.ERROR, str(e)[:200]))
            span.record_exception(e)
            span.set_attribute("app.llm.keep", True)
            raise
        finally:
            # telemetry lỗi thì ghi nhận rồi bỏ qua, không được thay exception của ứng dụng
            try:
                _finish(span, rec, attrs, metric_attrs, provider, request_model, prices,
                        start, error_type, cancelled)
            except Exception:
                _report_telemetry_error(span, provider)


def _report_telemetry_error(span, provider):
    # chính đường báo lỗi cũng có thể hỏng (logger, metric SDK): mỗi bước tự nuốt lỗi của nó
    for report in (lambda: log.exception("telemetry finalize failed"),
                   lambda: telemetry_errors.add(1, {"gen_ai.provider.name": provider}),
                   lambda: span.set_attribute("app.llm.telemetry_error", True)):
        try:
            report()
        except Exception:
            pass


def _finish(span, rec, attrs, metric_attrs, provider, request_model, prices,
            start, error_type, cancelled):
    elapsed = time.monotonic() - start
    m_attrs = {**attrs, **(metric_attrs or {})}
    m_attrs.pop("gen_ai.request.stream", None)
    if error_type is not None:
        m_attrs["error.type"] = str(error_type)
    if rec.response_model:
        m_attrs["gen_ai.response.model"] = rec.response_model
        span.set_attribute("gen_ai.response.model", rec.response_model)
    if rec.response_id:
        span.set_attribute("gen_ai.response.id", rec.response_id)
    if rec.finish_reasons:
        span.set_attribute("gen_ai.response.finish_reasons", rec.finish_reasons)
        if {r.lower() for r in rec.finish_reasons} - NORMAL_FINISH:
            span.set_attribute("app.llm.keep", True)
    if rec.first_chunk_at is not None:
        ttfc = rec.first_chunk_at - start
        span.set_attribute("gen_ai.response.time_to_first_chunk", ttfc)
        ttfc_hist.record(ttfc, m_attrs)
    if not cancelled:                          # thời lượng của lượt bị huỷ làm lệch phân phối
        duration_hist.record(elapsed, m_attrs)
    if rec.usage is not None:
        n = normalize_usage(provider, rec.usage)
        for k, v in n.items():
            if v:
                span.set_attribute(f"gen_ai.usage.{k}", v)
                tok_attrs = m_attrs if k.startswith("reasoning") else {**m_attrs, "gen_ai.token.modality": "text"}
                usage_counters[k].add(v, tok_attrs)
        price = find_price(prices, rec.response_model or request_model)
        if price is None:
            span.set_attribute("app.llm.cost.unpriced", True)   # đừng ghi 0 cho có
        else:
            c = cost_usd(n, price)
            span.set_attribute("app.llm.cost_usd", c)
            cost_counter.add(c, m_attrs)
    elif error_type is None and not cancelled:
        span.set_attribute("app.llm.usage_missing", True)
        span.set_attribute("app.llm.keep", True)
