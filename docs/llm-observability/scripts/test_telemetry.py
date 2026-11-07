from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exp = InMemorySpanExporter()
tp = TracerProvider(); tp.add_span_processor(SimpleSpanProcessor(exp)); trace.set_tracer_provider(tp)
reader = InMemoryMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))

import llm_telemetry as lt  # noqa: E402

PRICES = {"model-a": lt.Price(input=2.0, cache_read=0.2, cache_write=2.5, output=10.0)}

# 1. normalize
n = lt.normalize_usage("anthropic", {"input_tokens": 100, "cache_read_input_tokens": 4000,
    "cache_creation_input_tokens": 500, "output_tokens": 300, "output_tokens_details": {"thinking_tokens": 120}})
assert n == {"input_tokens": 4600, "output_tokens": 300, "cache_read.input_tokens": 4000,
             "cache_write.input_tokens": 500, "reasoning.output_tokens": 120}, n
n = lt.normalize_usage("openai", {"input_tokens": 12000, "input_tokens_details": {"cached_tokens": 4000,
    "cache_write_tokens": 7000}, "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 20}})
assert n["input_tokens"] == 12000 and n["cache_read.input_tokens"] == 4000 and n["reasoning.output_tokens"] == 20
n = lt.normalize_usage("gcp.gemini", {"promptTokenCount": 1000, "cachedContentTokenCount": 800,
    "candidatesTokenCount": 100, "thoughtsTokenCount": 400, "totalTokenCount": 1500})
assert n["output_tokens"] == 500 and n["input_tokens"] == 1000

# 2. price lookup
assert lt.find_price(PRICES, "model-a-2026-05-01") is PRICES["model-a"]
assert lt.find_price(PRICES, "model-ab") is None
assert lt.find_price(PRICES, "model-b") is None
# cost: uncached = 12000-4000-7000=1000
c = lt.cost_usd({"input_tokens": 12000, "cache_read.input_tokens": 4000, "cache_write.input_tokens": 7000,
                 "output_tokens": 50, "reasoning.output_tokens": 20}, PRICES["model-a"])
assert abs(c - (1000*2 + 4000*0.2 + 7000*2.5 + 50*10)/1e6) < 1e-12, c

# 3. happy path with stream + refusal
with lt.traced_inference(provider="anthropic", operation="chat", request_model="model-a",
        server_address="api.anthropic.com", prices=PRICES, stream=True,
        app_attrs={"app.tenant.id": "t1", "app.feature": "support"}, metric_attrs={"app.feature": "support"}) as rec:
    rec.mark_first_chunk()
    rec.response_id = "msg_1"; rec.response_model = "model-a-2026-05-01"
    rec.finish_reasons = ["refusal"]
    rec.usage = {"input_tokens": 100, "output_tokens": 10}
s = exp.get_finished_spans()[-1]
a = dict(s.attributes)
assert s.name == "chat model-a" and s.kind.name == "CLIENT"
assert a["gen_ai.usage.input_tokens"] == 100 and a["app.llm.keep"] is True
assert a["gen_ai.response.finish_reasons"] == ("refusal",)
assert "app.llm.cost_usd" in a and "error.type" not in a
assert s.status.status_code.name == "UNSET"

# 4. error path
class APIStatusError(Exception):
    status_code = 529
try:
    with lt.traced_inference(provider="anthropic", operation="chat", request_model="model-a",
            server_address="x", prices=PRICES) as rec:
        raise APIStatusError("overloaded")
except APIStatusError:
    pass
s = exp.get_finished_spans()[-1]; a = dict(s.attributes)
assert a["error.type"] == "529" and s.status.status_code.name == "ERROR" and a["app.llm.keep"]
assert "app.llm.usage_missing" not in a
assert any(e.name == "exception" for e in s.events)

# 5. unpriced model, missing usage
with lt.traced_inference(provider="openai", operation="chat", request_model="model-z",
        server_address="x", prices=PRICES) as rec:
    rec.response_model = "model-z-1"
    rec.usage = {"input_tokens": 5, "output_tokens": 5}
a = dict(exp.get_finished_spans()[-1].attributes)
assert a["app.llm.cost.unpriced"] is True and "app.llm.cost_usd" not in a
with lt.traced_inference(provider="openai", operation="chat", request_model="model-a",
        server_address="x", prices=PRICES) as rec:
    rec.response_model = "model-a"
a = dict(exp.get_finished_spans()[-1].attributes)
assert a["app.llm.usage_missing"] is True

# 6. metrics
names = {m.name for rm in reader.get_metrics_data().resource_metrics for sm in rm.scope_metrics for m in sm.metrics}
assert {"gen_ai.client.inference.duration", "gen_ai.client.inference.time_to_first_chunk",
        "gen_ai.client.inference.usage.input_tokens", "app.llm.cost"} <= names, names
for rm in reader.get_metrics_data().resource_metrics:
    for sm in rm.scope_metrics:
        for m in sm.metrics:
            for dp in m.data.data_points:
                assert "app.tenant.id" not in dp.attributes
print("ALL OK")
