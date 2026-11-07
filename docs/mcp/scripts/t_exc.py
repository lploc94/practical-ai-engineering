import asyncio
from mcp import Client
from mcp.server import MCPServer
m = MCPServer("x")
@m.tool()
def boom() -> str:
    """b"""
    raise ConnectionError("db-internal.prod:5432 refused, user=svc_orders")
async def main():
    async with Client(m) as c:
        r = await c.call_tool("boom", {})
        print(r.is_error, r.content[0].text)
asyncio.run(main())
