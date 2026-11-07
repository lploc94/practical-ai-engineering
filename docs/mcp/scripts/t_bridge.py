import asyncio, os
os.environ.setdefault("ANTHROPIC_API_KEY", "x")
from mcp import Client
from anthropic.lib.tools.mcp import async_mcp_tool
from orders_server import mcp

async def main():
    async with Client(mcp) as c:
        listed = await c.list_tools()
        tools = [async_mcp_tool(t, c) for t in listed.tools]
        for t in tools:
            print(t.to_dict()["name"], list(t.to_dict()["input_schema"]["properties"]))
        print(await tools[0].call({"order_id": "a1001"}))
        try:
            print(await tools[0].call({"order_id": "B9"}))
        except Exception as e:
            print("EXC", type(e).__name__, e)
asyncio.run(main())
