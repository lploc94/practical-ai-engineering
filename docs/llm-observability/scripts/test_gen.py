exec(open("test_telemetry.py").read().split("# 1. normalize")[0])
def answer():
    with lt.traced_inference(provider="anthropic", operation="chat", request_model="model-a",
            server_address="x", prices=PRICES, stream=True) as rec:
        for t in ["a","b","c"]:
            rec.mark_first_chunk(); yield t
        rec.usage={"input_tokens":1,"output_tokens":3}; rec.finish_reasons=["end_turn"]
g=answer(); next(g); g.close()
s=exp.get_finished_spans()[-1]; print(s.status.status_code, dict(s.attributes))
print(list(answer())); s=exp.get_finished_spans()[-1]; print(s.status.status_code, dict(s.attributes).get("gen_ai.usage.output_tokens"))
