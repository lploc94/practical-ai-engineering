import asyncio
from mcp import Client
from orders_server import mcp

async def main():
    async with Client(mcp) as c:
        r = await c.call_tool("get_order", {"order_id": "a1001"})
        assert not r.is_error and r.structured_content["status"] == "delivered"

        r = await c.call_tool("get_order", {"order_id": "B9"})
        assert r.is_error and "Không có đơn B9" in r.content[0].text

        r = await c.call_tool("get_order", {"order_id": " a1001 "})
        assert r.structured_content["order_id"] == "A1001"   # chuẩn hoá một lần, dùng cho mọi chỗ

        r = await c.call_tool("refund_order", {"order_id": "A1001", "reason": ""})
        assert r.is_error                       # vi phạm min_length, bị chặn trước khi vào hàm

asyncio.run(main())
