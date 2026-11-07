from typing import Annotated

from pydantic import BaseModel, Field

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

mcp = MCPServer("orders", instructions="Tra cứu và hoàn tiền đơn hàng của cửa hàng.")

ORDERS = {"A1001": {"status": "delivered", "total_vnd": 450_000, "refunded": False}}


class Order(BaseModel):
    order_id: str
    status: str
    total_vnd: int
    refunded: bool


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def get_order(order_id: Annotated[str, Field(description="Mã đơn, dạng A1234")]) -> Order:
    """Lấy trạng thái và tổng tiền của một đơn hàng."""
    order_id = order_id.strip().upper()
    o = ORDERS.get(order_id)
    if o is None:
        # lỗi nghiệp vụ: model đọc được thông báo và có thể hỏi lại người dùng
        raise ToolError(f"Không có đơn {order_id}. Mã đơn có dạng A + 4 chữ số.")
    return Order(order_id=order_id, **o)


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True, openWorldHint=False))
def refund_order(order_id: str, reason: Annotated[str, Field(min_length=1)]) -> str:
    """Hoàn toàn bộ tiền một đơn đã giao. Gọi lại với cùng mã đơn không hoàn tiền lần hai."""
    order_id = order_id.strip().upper()
    o = ORDERS.get(order_id)
    if o is None:
        raise ToolError(f"Không có đơn {order_id}.")
    if o["status"] != "delivered":
        raise ToolError(f"Đơn {order_id} đang ở trạng thái {o['status']}, chỉ hoàn được đơn đã giao.")
    if not o["refunded"]:
        o["refunded"] = True          # thay bằng lệnh gọi hệ thống thanh toán kèm idempotency key
    return f"Đã hoàn {o['total_vnd']:,} đồng cho đơn {order_id}."


if __name__ == "__main__":
    mcp.run()                         # mặc định stdio
