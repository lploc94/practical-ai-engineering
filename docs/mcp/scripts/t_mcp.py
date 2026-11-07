import asyncio
from mcp import Client
from orders_server import mcp

async def main():
    async with Client(mcp) as c:
        print("proto", c.protocol_version)
        tools = await c.list_tools()
        for t in tools.tools:
            print(t.name, t.annotations, t.output_schema is not None)
            print(" input", t.input_schema)
        r = await c.call_tool("get_order", {"order_id": "a1001"})
        print("get", r.is_error, r.structured_content, r.content[0].text)
        r = await c.call_tool("get_order", {"order_id": "B9"})
        print("miss", r.is_error, r.content[0].text)
        r = await c.call_tool("refund_order", {"order_id": "A1001", "reason": "hỏng"})
        print("refund", r.is_error, r.content[0].text)
        r = await c.call_tool("refund_order", {"order_id": "A1001", "reason": "x"*300})
        print("toolong", r.is_error, r.content[0].text[:150])
asyncio.run(main())
