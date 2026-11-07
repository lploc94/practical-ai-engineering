import time, anthropic, httpx2
n=[0]
def h(req):
    n[0]+=1
    if n[0]==1: return httpx2.Response(429, headers={"retry-after":"3"}, json={"type":"error","error":{"type":"rate_limit_error","message":"x"}})
    return httpx2.Response(200, json={"id":"m","type":"message","role":"assistant","model":"m","content":[],"stop_reason":"end_turn","stop_sequence":None,"usage":{"input_tokens":1,"output_tokens":1}})
c=anthropic.Anthropic(api_key="k", http_client=httpx2.Client(transport=httpx2.MockTransport(h)))
t=time.monotonic(); c.messages.create(model="m",max_tokens=5,messages=[{"role":"user","content":"x"}]); print("waited", round(time.monotonic()-t,2), "calls", n[0])
