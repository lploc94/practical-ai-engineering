from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
exp=InMemorySpanExporter(); tp=TracerProvider(); tp.add_span_processor(SimpleSpanProcessor(exp)); trace.set_tracer_provider(tp)
metrics.set_meter_provider(MeterProvider(metric_readers=[InMemoryMetricReader()]))
import logging; logging.disable(logging.CRITICAL)
import llm_telemetry as lt
P={}
def run(fr):
    with lt.traced_inference(provider="openai",operation="chat",request_model="m",server_address="a",prices=P) as r:
        r.finish_reasons=[fr]; r.usage={"input_tokens":1,"output_tokens":1}
    return dict(exp.get_finished_spans()[-1].attributes).get("app.llm.keep")
for fr in ["max_output_tokens","length","max_tokens","MAX_TOKENS","SAFETY","SPII","content_filter","refusal","RECITATION"]:
    assert run(fr) is True, fr
for fr in ["stop","end_turn","STOP","completed","tool_use","tool_calls"]:
    assert run(fr) is None, fr
# telemetry error does not mask domain error
orig=lt.normalize_usage
lt.normalize_usage=lambda p,u:(_ for _ in ()).throw(ValueError("norm"))
try:
    with lt.traced_inference(provider="openai",operation="chat",request_model="m",server_address="a",prices=P) as r:
        r.usage={"x":1}; raise RuntimeError("domain")
except RuntimeError as e: assert str(e)=="domain"
assert dict(exp.get_finished_spans()[-1].attributes)["app.llm.telemetry_error"] is True
# telemetry error on success path: no raise
with lt.traced_inference(provider="openai",operation="chat",request_model="m",server_address="a",prices=P) as r:
    r.usage={"x":1}
lt.normalize_usage=orig
# cancellation
def gen():
    with lt.traced_inference(provider="anthropic",operation="chat",request_model="m",server_address="a",prices=P,stream=True) as r:
        r.mark_first_chunk(); yield 1; yield 2
g=gen(); next(g); g.close()
s=exp.get_finished_spans()[-1]; a=dict(s.attributes)
assert s.status.status_code.name=="UNSET" and a["app.llm.cancelled"] is True and "error.type" not in a and "app.llm.usage_missing" not in a, (s.status, a)
import asyncio
async def coro():
    with lt.traced_inference(provider="anthropic",operation="chat",request_model="m",server_address="a",prices=P) as r:
        await asyncio.sleep(10)
async def main():
    t=asyncio.create_task(coro()); await asyncio.sleep(0.01); t.cancel()
    try: await t
    except asyncio.CancelledError: pass
asyncio.run(main())
a=dict(exp.get_finished_spans()[-1].attributes); assert a["app.llm.cancelled"] is True
print("FIX OK")
