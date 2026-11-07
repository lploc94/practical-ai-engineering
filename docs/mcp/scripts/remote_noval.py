from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import TransportSecuritySettings

RESOURCE = "https://mcp.shop.example/mcp"


class JwtVerifier:
    """Xác thực access token do authorization server của bạn phát hành."""

    async def verify_token(self, token: str) -> AccessToken | None:
        claims = decode_and_verify_jwt(token)      # kiểm chữ ký, iss, exp; None nếu sai
        if claims is None:
            return None
        return AccessToken(token=token, client_id=claims["client_id"],
                           scopes=claims["scope"].split(), expires_at=claims["exp"],
                           resource=claims["aud"], subject=claims["sub"])


mcp = MCPServer(
    "orders",
    token_verifier=JwtVerifier(),
    auth=AuthSettings(
        issuer_url="https://auth.shop.example",
        resource_server_url=RESOURCE,
        #          # từ chối token phát hành cho resource khác
        required_scopes=["orders:read"],
    ),
)


@mcp.tool()
def my_orders() -> list[str]:
    """Liệt kê mã các đơn hàng của người dùng đang đăng nhập."""
    user = get_access_token().subject          # danh tính lấy từ token, không lấy từ tham số
    return orders_of(user)


app = mcp.streamable_http_app(
    host="0.0.0.0",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["mcp.shop.example"],
        allowed_origins=["https://shop.example"],
    ),
)
