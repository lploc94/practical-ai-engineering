import asyncio, os, json
os.environ["ANTHROPIC_API_KEY"]="x"
import httpx2 as httpx, anthropic
from orders_server import mcp
import bridge_snip
seen=[]
def handler(req: httpx.Request):
    body=json.loads(req.content); seen.append(body)
    print("REQ", req.url.path, list(body))
    last=body["messages"][-1]
    print(json.dumps(body,ensure_ascii=False)[:400])
    if not (isinstance(last["content"],list) and last["content"][0].get("type")=="tool_result"):
        content=[{"type":"tool_use","id":"tu1","name":"get_order","input":{"order_id":"A1001"}}]; sr="tool_use"
    else:
        tr=last["content"][0]
        content=[{"type":"text","text":"KQ: "+json.dumps(tr["content"],ensure_ascii=False)[:80]}]; sr="end_turn"
    return httpx.Response(200,json={"id":"m","type":"message","role":"assistant","model":"claude-sonnet-5-5","content":content,"stop_reason":sr,"stop_sequence":None,"usage":{"input_tokens":1,"output_tokens":1}})
orig=anthropic.AsyncAnthropic
bridge_snip.anthropic.AsyncAnthropic=lambda: orig(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
print(asyncio.run(bridge_snip.ask(mcp,"Đơn A1001?")))
print("tools sent:", [t["name"] for t in seen[0].get("tools",[])])
