# MCP: kết nối model với công cụ và dữ liệu

Một công ty có ba ứng dụng AI: chatbot hỗ trợ khách hàng, trợ lý nội bộ trong Slack, và agent viết code trong IDE. Cả ba cần đọc ticket trong hệ thống hỗ trợ, tra đơn hàng, tìm tài liệu trong Drive. Mỗi đội tự viết tool `get_ticket` cho ứng dụng của mình: ba bộ schema, ba cách xác thực, ba nơi sửa khi API của hệ thống ticket đổi. Khi một khách hàng muốn dùng Claude Desktop hay ChatGPT để tra đơn của chính họ, đội lại viết thêm lần thứ tư.

MCP (Model Context Protocol) chuẩn hoá chỗ nối đó. Bạn viết một **MCP server** cho hệ thống ticket một lần; mọi ứng dụng nói được MCP (IDE, Claude, ChatGPT, agent tự viết) khám phá và gọi tool của nó theo cùng một giao thức. MCP không thay đổi cách model chọn tool: phía client vẫn đưa tool definition vào prompt và model vẫn sinh ra yêu cầu gọi tool như trong [Tool use](../tool-use/). Cái MCP chuẩn hoá là khâu khám phá tool, gọi tool qua mạng hoặc qua process, xác thực, và vòng đời của kết nối.

Cái giá đi kèm cũng cụ thể. Tool definition do bên thứ ba viết đi thẳng vào prompt của bạn, nên một server độc hại có thể ra lệnh cho model qua mô tả tool. Mỗi server nối thêm làm phình context trước khi người dùng nói câu đầu tiên. Và giao thức đổi nhanh: bản đặc tả 2026-07-28 bỏ hẳn bước `initialize` và session mà mọi hướng dẫn năm 2025 đều dựa vào.

Tài liệu giả định bạn đã đọc [Tool use](../tool-use/). Ba phần:

- **Nền tảng — Phần 1:** kiến trúc host/client/server, ba primitive, vòng đời request theo bản 2026-07-28, transport, và MCP khác function calling ở đâu.
- **Tra cứu — Phần 2:** các phiên bản đặc tả, SDK, OAuth cho remote server, các tính năng nâng cao và trạng thái deprecated, client hỗ trợ MCP, MCP connector trong API của Claude, OpenAI, Gemini, Grok.
- **Thiết kế và vận hành — Phần 3:** chọn tool để expose, kích thước tool definition và output, deploy remote server, bảo mật kèm các sự cố đã xảy ra, test, failure modes và checklist.

> Bản tháng 11/2026, theo đặc tả MCP 2026-07-28 và Python SDK `mcp` 2.3.0. Code trong bài đã chạy với SDK này và với MCP Inspector 2.13.0. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu đặc tả và API.

Tài liệu liên quan: [Tool use](../tool-use/) (vòng lặp tool call, description, tool result), [Agents](../agents/#25-mcp-và-tích-hợp-hệ-thống-bên-ngoài) (MCP trong thiết kế agent), [Bảo mật LLM](../llm-security/) (prompt injection, least privilege).

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 MCP là gì: host, client, server](#11-mcp-là-gì-host-client-server)
  - [1.2 Tools, resources và prompts](#12-tools-resources-và-prompts)
  - [1.3 Vòng đời một request: từ initialize đến stateless](#13-vòng-đời-một-request-từ-initialize-đến-stateless)
  - [1.4 Transport: stdio và Streamable HTTP](#14-transport-stdio-và-streamable-http)
  - [1.5 MCP khác function calling thế nào](#15-mcp-khác-function-calling-thế-nào)
- [Phần 2 — Tra cứu giao thức và SDK](#phần-2--tra-cứu-giao-thức-và-sdk)
  - [2.1 Đặc tả và các phiên bản](#21-đặc-tả-và-các-phiên-bản)
  - [2.2 SDK chính thức](#22-sdk-chính-thức)
  - [2.3 Authorization: OAuth cho remote server](#23-authorization-oauth-cho-remote-server)
  - [2.4 Sampling, elicitation và các tính năng nâng cao](#24-sampling-elicitation-và-các-tính-năng-nâng-cao)
  - [2.5 Client hỗ trợ MCP](#25-client-hỗ-trợ-mcp)
  - [2.6 MCP connector trong API của provider](#26-mcp-connector-trong-api-của-provider)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Thiết kế MCP server: tool nào nên expose](#31-thiết-kế-mcp-server-tool-nào-nên-expose)
  - [3.2 Mô tả tool và kích thước output](#32-mô-tả-tool-và-kích-thước-output)
  - [3.3 Remote server: deploy, scale, multi-tenant](#33-remote-server-deploy-scale-multi-tenant)
  - [3.4 Bảo mật: tool poisoning, quyền hạn, server bên thứ ba](#34-bảo-mật-tool-poisoning-quyền-hạn-server-bên-thứ-ba)
  - [3.5 Test và debug MCP server](#35-test-và-debug-mcp-server)
  - [3.6 Failure modes thường gặp](#36-failure-modes-thường-gặp)
  - [3.7 Checklist](#37-checklist)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

### Khi nào dùng MCP

| Tình huống | Nên dùng | Lý do |
|---|---|---|
| Một ứng dụng, vài tool gọi hệ thống của chính bạn | Function calling trực tiếp | Ít lớp, ít bề mặt tấn công, debug trong một process |
| Nhiều ứng dụng hoặc nhiều đội cùng cần một hệ thống (ticket, DB, Drive) | MCP server dùng chung | Viết tích hợp một lần, mọi client MCP dùng được |
| Muốn người dùng gọi hệ thống của bạn từ Claude, ChatGPT, IDE của họ | Remote MCP server có OAuth | Đó là giao thức các client này hỗ trợ |
| Dùng tích hợp sẵn của bên thứ ba (GitHub, Stripe, Sentry) | MCP server chính chủ, lọc tool, có bước duyệt | Không phải tự viết, nhưng tool description của họ vào prompt của bạn |
| Agent của công ty A nói chuyện với agent của công ty B | A2A, không phải MCP | MCP nối agent với tool và dữ liệu, A2A nối agent với agent ([§1.5](#15-mcp-khác-function-calling-thế-nào)) |

### Transport

| Transport | Dùng khi | Ghi chú |
|---|---|---|
| stdio | Server chạy trên máy người dùng, client khởi chạy nó như process con | Không dùng OAuth; lấy credential từ biến môi trường [5]. Không ghi log ra stdout |
| Streamable HTTP | Server remote, nhiều người dùng | Bản 2026-07-28: chỉ POST, mỗi request độc lập, không session [3] |
| HTTP+SSE (bản 2024-11-05) | Chỉ để nói chuyện với server cũ | Deprecated [1] |

### Checklist bảo mật tối thiểu cho server

1. Remote: HTTPS, xác thực mọi request, kiểm `Origin` (sai trả 403) [3].
2. Chỉ nhận access token phát hành cho chính server (audience); không chuyển tiếp token của client sang API phía sau [6]. Với Python SDK, bật `validate_token_resource=True` ([§2.3](#23-authorization-oauth-cho-remote-server)).
3. Danh tính người dùng lấy từ token, không lấy từ tham số tool.
4. Lỗi nghiệp vụ trả `isError: true` với thông báo model đọc được; không lộ stack trace.
5. Tool ghi dữ liệu: idempotent, có annotation đúng, và client có bước duyệt. Annotation chỉ là gợi ý, không phải kiểm soát [4].
6. Phía client: lọc tool tối thiểu, ghim mô tả tool đã duyệt, cảnh báo khi mô tả đổi ([§3.4](#34-bảo-mật-tool-poisoning-quyền-hạn-server-bên-thứ-ba)).

---

# Phần 1 — Nền tảng

## 1.1 MCP là gì: host, client, server

Đặc tả chia ba vai:

- **Host**: ứng dụng người dùng tương tác, chứa model và vòng lặp tool call. Claude Desktop, Claude Code, VS Code, ChatGPT, hoặc agent bạn tự viết.
- **Client**: thành phần trong host giữ kết nối tới **một** server. Host nối năm server thì có năm client.
- **Server**: chương trình expose tool, resource, prompt. Có thể là process local (đọc file, chạy git) hoặc dịch vụ remote (GitHub, Sentry, hệ thống nội bộ của bạn).

Model không nói chuyện với server. Host lấy danh sách tool từ các server qua client, gửi chúng vào prompt như tool definition bình thường, nhận yêu cầu gọi tool từ model, rồi chuyển yêu cầu đó tới đúng server. Kết quả từ server đi ngược lại vào context của model. Mọi quyết định "có gọi hay không", "có hỏi người dùng trước không" nằm ở host.

Bản 2026-07-28 đổi một điểm quan trọng ở quan hệ client-server: **server không còn gửi JSON-RPC request ngược cho client** [3]. Trước đây server có thể chủ động hỏi client (xin model sinh text qua sampling, hỏi người dùng qua elicitation, hỏi danh sách thư mục qua roots) trên một kênh SSE mở sẵn. Giờ server muốn hỏi gì thì trả về một result đặc biệt và client gửi lại request kèm câu trả lời ([§1.3](#13-vòng-đời-một-request-từ-initialize-đến-stateless)). Mọi luồng đều do client khởi đầu, nên server có thể chạy stateless sau một load balancer bình thường.

## 1.2 Tools, resources và prompts

Server cung cấp ba loại primitive, khác nhau ở chỗ ai quyết định dùng:

| Primitive | Ai quyết định | Dùng cho | Method |
|---|---|---|---|
| Tools | Model (qua host) | Hành động và truy vấn: tra đơn, tạo ticket, chạy query | `tools/list`, `tools/call` |
| Resources | Ứng dụng hoặc người dùng | Dữ liệu đọc được, định danh bằng URI: file, bản ghi, log | `resources/list`, `resources/read`, `resources/templates/list` |
| Prompts | Người dùng | Template tin nhắn có tham số, thường hiện thành slash command | `prompts/list`, `prompts/get` |

Trong thực tế, tools là phần mọi client hỗ trợ; resources và prompts thì không. MCP connector của Claude API chỉ hỗ trợ tool [12], và remote MCP của OpenAI, Gemini, Grok cũng chỉ nhập tool [13][14][15]. Nếu dữ liệu cần đến được model qua mọi client, expose nó bằng tool đọc, đừng chỉ bằng resource.

### Tool definition

Một tool trong `tools/list` có `name`, `title`, `description`, `inputSchema` (bắt buộc, JSON Schema dạng object), và tuỳ chọn `outputSchema`, `annotations`, `icons`, `_meta` [4]. Tên nên dài 1–128 ký tự, chỉ gồm chữ, số, `_`, `-`, `.` [4]. Tool không có tham số nên khai `{"type": "object", "additionalProperties": false}` [4].

**Annotations** mô tả hành vi tool cho client [10]:

| Field | Mặc định | Ý nghĩa |
|---|---|---|
| `readOnlyHint` | `false` | Tool không sửa đổi môi trường |
| `destructiveHint` | `true` | Tool có thể xoá hoặc ghi đè (chỉ có nghĩa khi `readOnlyHint` là `false`) |
| `idempotentHint` | `false` | Gọi lại với cùng tham số không gây thêm tác dụng |
| `openWorldHint` | `true` | Tool tương tác với thực thể bên ngoài (web, hệ thống khác) |

Chú ý giá trị mặc định: tool không khai gì bị coi là "có thể phá huỷ" và "mở ra thế giới bên ngoài". Đó là mặc định an toàn cho client. Ngược lại, đặc tả nói rõ client **MUST** coi annotation là không đáng tin trừ khi server đáng tin [4]. Server độc hại hoàn toàn có thể gắn `readOnlyHint: true` cho tool xoá dữ liệu. Annotation giúp client hiển thị và quyết định có hỏi người dùng hay không với server của chính mình; nó không phải cơ chế bảo mật.

### Kết quả và lỗi

Kết quả `tools/call` có `content` (danh sách block `text`, `image`, `audio`, `resource_link`, `resource`) và tuỳ chọn `structuredContent` (JSON). Nếu tool khai `outputSchema`, server **MUST** trả `structuredContent` khớp schema; tool trả structured content **SHOULD** kèm bản JSON đó dưới dạng text để client cũ vẫn đọc được [4]. `structuredContent` là dữ liệu server sinh ra, không liên quan đến structured output (ràng buộc decoding) của model.

Lỗi có hai tầng [4]:

- **Protocol error**: JSON-RPC error. Tool không tồn tại, request sai cấu trúc, lỗi server.
- **Tool execution error**: result bình thường với `isError: true`, nội dung giải thích lỗi. Đặc tả khuyên client đưa loại lỗi này cho model để nó tự sửa.

Lỗi nghiệp vụ (mã đơn không tồn tại, ngày nằm ngoài khoảng cho phép, thiếu quyền với bản ghi) phải là tool execution error. Nếu trả protocol error, nhiều client chỉ báo "tool lỗi" và model mất thông tin để hỏi lại người dùng hay thử cách khác. Đây là cùng nguyên tắc với [Tool use §3.4](../tool-use/#34-lỗi-từ-tool-trả-về-cho-model-hay-xử-lý-trong-code).

### Một server hoàn chỉnh

Server dưới đây có hai tool: một tool đọc, một tool hoàn tiền. Viết bằng Python SDK chính thức `mcp` 2.x:

```python
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
```

SDK sinh `inputSchema` từ type hint và `Field`, sinh `outputSchema` từ kiểu trả về `Order`, và lấy docstring làm `description`. Khi chạy thật:

- `get_order("a1001")` trả `structuredContent` `{"order_id": "A1001", "status": "delivered", ...}` kèm block text chứa cùng JSON.
- `get_order("B9")` trả `isError: true` với text `Error executing tool get_order: Không có đơn B9. Mã đơn có dạng A + 4 chữ số.` SDK tự thêm tiền tố "Error executing tool".
- `refund_order` với `reason` rỗng bị pydantic chặn trước khi vào hàm, cũng trả về `isError: true` kèm thông báo validation. Validate ở schema thì model thấy ràng buộc ngay trong tool definition; validate trong code thì bắt được ràng buộc phụ thuộc dữ liệu (trạng thái đơn). Cần cả hai.

`refund_order` khai `idempotentHint=True` và code thực sự idempotent: gọi lần hai không hoàn thêm. Khai annotation mà code không làm đúng thì client tin nhầm, ví dụ tự retry một tool "idempotent" sau timeout.

## 1.3 Vòng đời một request: từ initialize đến stateless

### Bản 2025-11-25 trở về trước

Client mở kết nối bằng request `initialize` (gửi `protocolVersion`, `capabilities`, `clientInfo`), server trả capabilities của nó, client gửi `notifications/initialized`. Trên HTTP, server cấp header `Mcp-Session-Id` và mọi request sau phải mang header này; phiên kết thúc bằng HTTP DELETE. Trạng thái (capabilities đã thương lượng, subscription, đôi khi cả dữ liệu ứng dụng) gắn với session. Hệ quả vận hành: load balancer cần sticky session hoặc server cần session store chung.

### Bản 2026-07-28 (hiện hành)

Bản này bỏ `initialize`, `notifications/initialized` và `Mcp-Session-Id` [1]. Mỗi request tự mang thông tin cần thiết trong `params._meta`:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_order",
    "arguments": { "order_id": "a1001" },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": { "name": "curl", "version": "0" },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Server **MUST** hỗ trợ method `server/discover`; client **MAY** gọi nó trước để biết server hỗ trợ version và capability nào, nhưng không bắt buộc [2]. Gọi `server/discover` tới server ở §1.2 (chạy qua HTTP) trả về:

```json
{"jsonrpc":"2.0","id":1,"result":{
  "cacheScope":"private",
  "capabilities":{"prompts":{"listChanged":true},"resources":{"listChanged":true,"subscribe":true},"tools":{"listChanged":true}},
  "instructions":"Tra cứu và hoàn tiền đơn hàng của cửa hàng.",
  "resultType":"complete",
  "supportedVersions":["2026-07-28"],
  "ttlMs":0,
  "_meta":{"io.modelcontextprotocol/serverInfo":{"name":"orders","version":""}}}}
```

Ba field mới trong result này:

- **`resultType`**: bắt buộc trên mọi result, `"complete"` hoặc `"input_required"`. Client gặp server cũ không gửi field này thì **MUST** coi là `"complete"` [1].
- **`ttlMs` và `cacheScope`**: server cho client biết kết quả được cache bao lâu và cache dùng chung giữa người dùng (`public`) hay riêng (`private`). Spec chỉ bắt buộc hai field này trên result của `tools/list`, `prompts/list`, `resources/list`, `resources/read` và `resources/templates/list` [1]; SDK Python gắn thêm vào result của `server/discover` như ở trên. Result của `tools/call` không có chúng, và client không nên đòi chúng ở method khác.

Version không được hỗ trợ trả lỗi `-32022` kèm danh sách version server nhận:

```json
{"jsonrpc":"2.0","id":3,"error":{"code":-32022,"message":"Unsupported protocol version",
 "data":{"supported":["2026-07-28"],"requested":"1900-01-01"}}}
```

### Server cần hỏi giữa chừng: MRTR

Khi tool cần thêm thông tin từ người dùng (ví dụ xác nhận tài khoản GitHub), server không mở kênh ngược mà trả result có `resultType: "input_required"` [4]:

```json
{"jsonrpc":"2.0","id":2,"result":{
  "resultType":"input_required",
  "inputRequests":{"github_login":{"method":"elicitation/create",
    "params":{"mode":"form","message":"Please provide your GitHub username",
      "requestedSchema":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}},
  "requestState":"eyJsb2NhdGlvbiI6Ik5ldyBZb3JrIn0..."}}
```

Client hỏi người dùng, rồi gửi lại **cùng tool call với `id` mới** (MUST khác id cũ), kèm `inputResponses` (ví dụ `{"github_login": {"action": "accept", "content": {"name": "octocat"}}}`) và `requestState` nguyên văn [4]. Đặc tả gọi cơ chế này là multi round-trip request (MRTR). Server không phải nhớ gì giữa hai lượt: mọi trạng thái cần thiết nằm trong `requestState` mà client gửi lại. Vì `requestState` đi qua client, server phải coi nó là input không tin cậy (ký hoặc mã hoá nếu nó chứa thứ client không được sửa).

### Trạng thái ứng dụng: handle thay cho session

Không còn session thì server giữ trạng thái giữa các lần gọi bằng **handle**: tool tạo trạng thái trả về một mã (`basket_id`, `browser_id`), model truyền mã đó vào các lần gọi sau [4]. Đặc tả đưa bốn hướng dẫn:

- Với server có xác thực, handle là **tên**, không phải **quyền**: mỗi lần gọi phải kiểm người gọi có quyền với handle đó. Với server không xác thực, handle chính là bearer token nên phải đủ ngẫu nhiên (UUIDv4 trở lên) và có hạn.
- Handle không nên mang cấu trúc đoán được.
- Thời hạn của handle ghi trong description của tool tạo ra nó ("giỏ hàng hết hạn sau 24 giờ không hoạt động"), để model biết khi quyết định tạo.
- Handle hết hạn hoặc không tồn tại trả tool execution error, để model tạo cái mới.

### Client mới gặp server cũ

Client theo bản 2026 và server theo bản 2025 không nói chuyện được với nhau, và ngược lại [2]. Đặc tả mô tả cách phát hiện: trên stdio, client gửi `server/discover` trước, nếu lỗi không phải lỗi kiểu mới thì quay về `initialize`; trên HTTP, client thử request kiểu mới, nếu nhận `400` mà body không phải lỗi kiểu mới thì quay về cách cũ. Kết quả phát hiện nên cache theo process (stdio) hoặc origin (HTTP) [2]. Server có thể phục vụ cả hai thời kỳ cùng lúc. Python SDK 2.3.0 làm cả hai việc: `Client` mặc định probe `server/discover` rồi fallback, và server HTTP của nó vẫn nhận `initialize` kiểu 2025-11-25 và cấp `mcp-session-id`.

## 1.4 Transport: stdio và Streamable HTTP

### stdio

Client khởi chạy server như process con và trao đổi JSON-RPC qua stdin/stdout, mỗi message một dòng. Hai hệ quả:

- **Không ghi gì ra stdout ngoài message giao thức.** Một dòng `print("debug")` làm hỏng framing và client ngắt kết nối với lỗi khó hiểu. Log ra stderr [1].
- **Không dùng OAuth.** Đặc tả authorization nói transport stdio SHOULD NOT theo phần OAuth, mà lấy credential từ môi trường [5]. Trong thực tế là biến môi trường do client truyền vào khi khởi chạy (`claude mcp add my-server -e API_KEY=... -- ...`).

stdio phù hợp cho server cần truy cập máy người dùng: file, git, trình duyệt local. Nó cũng là lựa chọn an toàn hơn cho server local so với mở cổng HTTP trên máy, vì không process nào khác nối vào được.

### Streamable HTTP (bản 2026-07-28)

Một endpoint duy nhất (thường là `/mcp`) nhận **POST**. Các quy tắc chính [3]:

- Mọi POST mang `Accept: application/json, text/event-stream`, `MCP-Protocol-Version`, và **`Mcp-Method`**. Các method có tham số `name` (`tools/call`, `prompts/get`) và `resources/read` (tham số `uri`) mang thêm **`Mcp-Name`**. Header phải khớp body; lệch thì server trả `400` với mã `-32020` (HeaderMismatch).
- Server trả `application/json` (một object) hoặc `text/event-stream` (SSE cho riêng request đó, dùng khi có progress). Client **MUST** xử lý được cả hai.
- Notification từ client nhận `202 Accepted` không body.
- **GET bị bỏ.** Thông báo thay đổi danh sách tool đi qua request `subscriptions/listen` mà client mở khi cần [1].
- **Không còn resume stream.** `Last-Event-ID` bị bỏ; stream đứt thì request đang chạy mất, client **MUST** gửi lại với id mới [1]. Tool có tác dụng phụ vì vậy càng phải idempotent.
- **Đóng response stream là cancellation**: server không được gửi thêm gì cho request đó và nên dừng công việc sớm nhất có thể [3].
- Server **MUST** kiểm header `Origin` và trả `403` nếu không hợp lệ, để chặn DNS rebinding; server local **SHOULD** chỉ bind `127.0.0.1` [3].

Header `Mcp-Method` và `Mcp-Name` tồn tại để gateway, load balancer và rate limiter định tuyến mà không phải parse body JSON. Tool có thể đẩy thêm tham số lên header: property kiểu `string`, `integer` hoặc `boolean` khai `"x-mcp-header": "Region"` thì client gửi kèm `Mcp-Param-Region: us-west1`; giá trị ngoài ASCII mã hoá dạng `=?base64?...?=` [3]. Đừng đặt `x-mcp-header` cho tham số nhạy cảm, vì header hiện ra với mọi proxy trên đường đi [3].

Chạy server ở §1.2 qua HTTP bằng `uvicorn` với `app = mcp.streamable_http_app()`, rồi gọi bằng curl:

```bash
curl -s -X POST http://127.0.0.1:8765/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: tools/call' \
  -H 'Mcp-Name: get_order' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_order",
       "arguments":{"order_id":"a1001"},
       "_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28",
                "io.modelcontextprotocol/clientCapabilities":{}}}}'
```

```json
{"jsonrpc":"2.0","id":2,"result":{
  "content":[{"text":"{\n  \"order_id\": \"A1001\",\n  \"status\": \"delivered\",\n  \"total_vnd\": 450000,\n  \"refunded\": false\n}","type":"text"}],
  "isError":false,"resultType":"complete",
  "structuredContent":{"order_id":"A1001","status":"delivered","total_vnd":450000,"refunded":false},
  "_meta":{"io.modelcontextprotocol/serverInfo":{"name":"orders","version":""}}}}
```

Đổi `Mcp-Name` thành giá trị khác `name` trong body, hoặc bỏ header đó, đều nhận `400`:

```json
{"jsonrpc":"2.0","id":3,"error":{"code":-32020,"message":"mcp-name header does not match the request body's 'name' parameter"}}
```

Gửi kèm `Origin: https://evil.example` nhận `403`. Gửi với `Host` không phải localhost nhận `421 Misdirected Request`: khi bạn không truyền cấu hình, SDK tự bật chống DNS rebinding và chỉ chấp nhận host `127.0.0.1`, `localhost`, `[::1]`. Deploy sau domain thật mà quên cấu hình này thì mọi request bị `421`; cách cấu hình ở [§3.3](#33-remote-server-deploy-scale-multi-tenant).

### HTTP+SSE cũ

Transport của bản 2024-11-05 dùng hai endpoint: GET mở SSE, POST gửi message. Nó deprecated từ bản 2025-03-26 khi Streamable HTTP ra đời và được xếp vào danh sách deprecated chính thức trong bản 2026-07-28 [1]. Bạn chỉ cần nó để nói chuyện với server cũ. MCP connector của Claude API vẫn nhận cả SSE lẫn Streamable HTTP [12]; Gemini chỉ nhận Streamable HTTP [14].

## 1.5 MCP khác function calling thế nào

| | Function calling trực tiếp | MCP |
|---|---|---|
| Ai định nghĩa tool | Code của bạn, gửi trong mỗi request | Server, client lấy qua `tools/list` |
| Ai thực thi | Code của bạn, cùng process | Server (process khác hoặc máy khác); hoặc provider gọi server thay bạn |
| Ai thấy tool | Ứng dụng của bạn | Mọi client MCP được cấu hình nối tới server |
| Ai kiểm soát mô tả tool | Bạn | Người viết server, có thể đổi bất cứ lúc nào |
| Chi phí context | Bạn chọn từng tool | Mặc định mọi tool của server; phải lọc chủ động |
| Xác thực | Credential của ứng dụng | OAuth theo người dùng (remote) hoặc biến môi trường (stdio) |

MCP không thay function calling. Host lấy tool từ MCP server rồi đưa vào model bằng chính cơ chế function calling. Vấn đề của tool use (model gọi nhầm tool, bịa tham số, lặp vô hạn, tool result chứa prompt injection) vẫn y nguyên, và MCP thêm một vấn đề: tool definition giờ do bên khác viết.

Có ba cách một ứng dụng dùng MCP:

1. **Host là client.** Claude Desktop, Claude Code, IDE: người dùng cấu hình server, host tự lo vòng lặp.
2. **Ứng dụng của bạn là client.** Bạn dùng MCP SDK để nối tới server, chuyển tool sang định dạng của model, tự chạy vòng lặp tool call. Toàn quyền kiểm soát: lọc tool, duyệt, log, server local qua stdio.
3. **Provider là client.** Bạn khai URL của MCP server trong request API; provider gọi server từ hạ tầng của họ ([§2.6](#26-mcp-connector-trong-api-của-provider)). Ít code nhất, nhưng server phải public trên internet và dữ liệu đi qua provider.

**A2A (Agent2Agent)** là giao thức khác, cho agent nói chuyện với agent: một agent gửi task cho agent khác (có thể của công ty khác) và nhận artifact về. Google đóng góp A2A cho Linux Foundation; trang chính thức liệt kê ban kỹ thuật gồm AWS, Cisco, Google, IBM Research, Microsoft, Salesforce, SAP, ServiceNow [29]. Hai giao thức bổ sung cho nhau: agent nhận task qua A2A có thể dùng MCP bên trong để gọi tool.

---

# Phần 2 — Tra cứu giao thức và SDK

## 2.1 Đặc tả và các phiên bản

Tên phiên bản là ngày có thay đổi phá vỡ tương thích gần nhất; thay đổi tương thích ngược không tăng phiên bản [2].

| Phiên bản | Thay đổi chính |
|---|---|
| 2024-11-05 | Bản đầu tiên. HTTP+SSE |
| 2025-03-26 | OAuth 2.1; Streamable HTTP thay HTTP+SSE; tool annotations; audio content |
| 2025-06-18 | Structured tool output (`outputSchema`, `structuredContent`); elicitation; resource link; bỏ JSON-RPC batching; header `MCP-Protocol-Version` |
| 2025-11-25 | Tasks (thử nghiệm); Client ID Metadata Documents; elicitation dạng URL; sampling có tool; icons; extensions; JSON Schema 2020-12 mặc định [10] |
| **2026-07-28** (hiện hành) | Stateless: bỏ `initialize`, session, GET stream, resume SSE; thêm `server/discover`, MRTR, `resultType`, `subscriptions/listen`; Tasks thành extension; header `Mcp-Method`/`Mcp-Name`; chính sách deprecation [1] |

Bản 2026-07-28 deprecate các phần sau [1]:

| Tính năng | Thay bằng |
|---|---|
| Sampling (`sampling/createMessage`) | Server gọi thẳng API của LLM provider |
| Roots (`roots/list`) | Truyền thư mục qua tham số tool, URI của resource, hoặc cấu hình server |
| Logging (`notifications/message`) | Log ra stderr (stdio) hoặc OpenTelemetry |
| HTTP+SSE transport | Streamable HTTP |
| `includeContext: "thisServer"/"allServers"` trong sampling | — |
| Đăng ký client động (DCR, RFC 7591) | Client ID Metadata Documents |

Bản này cũng **xoá hẳn** `ping`, `logging/setLevel` (log level đặt theo từng request qua `_meta`), `notifications/roots/list_changed`, `resources/subscribe`/`unsubscribe` (thay bằng `subscriptions/listen`) [1]. Mã lỗi đổi số: dải `-32000` đến `-32019` để implementation tự dùng, dải `-32020` đến `-32099` dành cho đặc tả; lỗi "resource not found" chuyển từ `-32002` sang `-32602` [1].

Tính năng deprecated còn chạy ít nhất 12 tháng trước khi bị xoá; ngoại lệ là khi có rủi ro bảo mật đang bị khai thác, thời hạn rút xuống tối thiểu 90 ngày [2]. Nếu server của bạn dựa vào sampling hoặc roots, đó là hạn chót để chuyển.

Về quản trị: Anthropic chuyển MCP sang Agentic AI Foundation thuộc Linux Foundation, công bố ngày 9/12/2025; các dự án khởi đầu của quỹ gồm MCP, goose (Block) và AGENTS.md (OpenAI) [26].

## 2.2 SDK chính thức

Trang SDK của MCP chia SDK theo tier [7]:

| Tier | SDK |
|---|---|
| Tier 1 | TypeScript, Python, C#, Go, Rust, Ruby |
| Tier 2 | Java |
| Tier 3 | Swift, PHP, Kotlin |

SDK áp dụng phiên bản đặc tả mới theo lịch riêng [7]. Bài công bố bản 2026-07-28 nói bốn SDK Tier 1 lúc đó (TypeScript, Python, Go, C#) hỗ trợ bản mới và Rust ở mức beta [11]. Trước khi chọn SDK cho server mới, kiểm tra changelog của SDK đó xem đã hỗ trợ `server/discover` và MRTR chưa.

### Python

Gói `mcp` bản 2.x dùng class `MCPServer` (`from mcp.server import MCPServer`) [8]. Nhiều bài hướng dẫn năm 2025 dùng `FastMCP` từ bản 1.x; khi đọc code cũ, đối chiếu API với tài liệu bản 2. Những gì đã kiểm với `mcp` 2.3.0:

- `@mcp.tool()` sinh schema từ type hint; `annotations=ToolAnnotations(...)` gắn annotation.
- Lỗi nghiệp vụ: `raise ToolError(...)` (từ `mcp.server.mcpserver.exceptions`) thành `isError: true`, kèm thông báo của bạn. Exception khác cũng thành `isError: true`, nhưng SDK giấu chi tiết: tool ném `ConnectionError("db-internal.prod:5432 refused")` thì model chỉ thấy `Error executing tool boom`. Chi tiết nội bộ không lộ ra, nhưng model cũng không biết nên thử lại, đổi tham số hay báo người dùng. Lỗi nào model xử lý được thì bắt và ném lại thành `ToolError` với thông báo bạn viết.
- `mcp.run()` mặc định chạy stdio; `mcp.streamable_http_app()` trả ứng dụng ASGI (Starlette) với endpoint `/mcp`.
- `from mcp import Client`; `Client(mcp)` chạy server ngay trong process, hữu ích cho test ([§3.5](#35-test-và-debug-mcp-server)).

### Nối MCP vào vòng lặp tool call của bạn

Khi ứng dụng của bạn là client (cách 2 ở §1.5), bạn cần chuyển tool MCP sang định dạng của model. Anthropic SDK có sẵn helper `async_mcp_tool` [12][28]:

```python
import anthropic
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import Client

ALLOWED = {"get_order"}               # tool ghi như refund_order không đưa thẳng vào runner


async def ask(server, question: str) -> str:
    client = anthropic.AsyncAnthropic()
    # server: một MCPServer (chạy trong process), URL "https://.../mcp", hoặc StdioServerParameters
    async with Client(server) as mcp_client:
        listed = await mcp_client.list_tools()
        tools = [async_mcp_tool(t, mcp_client) for t in listed.tools if t.name in ALLOWED]
        runner = client.beta.messages.tool_runner(
            model="claude-sonnet-5-5", max_tokens=4096, tools=tools,
            messages=[{"role": "user", "content": question}])
        final = await runner.until_done()
        return "".join(b.text for b in final.content if b.type == "text")
```

Đã chạy với `anthropic` 2.6.0 và `mcp` 2.3.0, API Anthropic được giả lập: request đầu chỉ chứa tool `get_order`, kết quả tool đi về model dưới dạng `tool_result`. Helper gọi `tools/call` thẳng khi model yêu cầu; tool trả `isError: true` thành `ToolError` phía Anthropic SDK và tới model như tool result lỗi. Helper **không** hỏi người dùng trước khi chạy tool nào. Lọc theo danh sách cho phép như dòng trên, và với tool ghi dữ liệu, bọc thêm bước duyệt của bạn (xem [Agents §3.3](../agents/#33-human-in-the-loop)) thay vì đưa thẳng vào `tool_runner`.

## 2.3 Authorization: OAuth cho remote server

Authorization trong MCP là tuỳ chọn. Server HTTP có xác thực SHOULD theo phần này; server stdio thì không [5]. Vai trò:

- **MCP server** là OAuth 2.1 resource server: nhận và kiểm access token.
- **MCP client** là OAuth client: lấy token thay mặt người dùng.
- **Authorization server** (AS) phát token; có thể là IdP sẵn có (Auth0, Okta, Entra, Keycloak) hoặc dịch vụ của bạn.

### Luồng khi client gặp server lần đầu

1. Client gọi server không token. Server trả `401` với `WWW-Authenticate: Bearer resource_metadata="https://.../.well-known/oauth-protected-resource/mcp"` và SHOULD kèm `scope` [5].
2. Client đọc **Protected Resource Metadata** (RFC 9728) ở URL đó, lấy `authorization_servers`. Server **MUST** cung cấp tài liệu này [5].
3. Client lấy metadata của AS qua RFC 8414 hoặc OpenID Connect Discovery.
4. Client đăng ký (xem dưới), mở trình duyệt với PKCE và tham số `resource` là URI chuẩn của MCP server.
5. Client đổi `code` lấy token, cũng kèm `resource`. Client **MUST** gửi `resource` ở cả hai bước, kể cả khi AS không hỗ trợ RFC 8707 [5].
6. Mọi request sau mang `Authorization: Bearer <token>`. Token không bao giờ nằm trong query string [5].

Thứ tự đăng ký client mà đặc tả khuyến nghị: dùng thông tin đăng ký trước nếu có; nếu không thì **Client ID Metadata Documents** (client dùng một URL HTTPS làm `client_id`, AS tải JSON metadata từ URL đó); DCR (RFC 7591) chỉ còn để tương thích với AS chưa hỗ trợ CIMD; cuối cùng là hỏi người dùng nhập [5].

### Phía server: kiểm audience, không chuyển tiếp token

Hai quy tắc bị vi phạm nhiều nhất:

- Server **MUST** chỉ chấp nhận token phát hành cho chính nó [5][6]. Token hợp lệ nhưng phát cho dịch vụ khác cùng AS (audience khác) phải bị từ chối. Nếu không, một server khác dùng chung AS, nhận token của người dùng, có thể đem token đó gọi server của bạn.
- Server **MUST NOT** chuyển tiếp token nhận từ client sang API phía sau (token passthrough) [6]. Gọi API phía sau thì dùng token riêng phát cho server, hoặc đổi token qua token exchange.

Python SDK hỗ trợ phía resource server. Server dưới đây lấy danh tính người dùng từ token và chỉ chấp nhận token phát cho đúng URL của nó:

```python
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
        validate_token_resource=True,          # từ chối token phát hành cho resource khác
        required_scopes=["orders:read"],
    ),
)


@mcp.tool()
def my_orders() -> list[str]:
    """Liệt kê mã các đơn hàng của người dùng đang đăng nhập."""
    user = get_access_token().subject          # danh tính lấy từ token, không lấy từ tham số
    return orders_of(user)


app = mcp.streamable_http_app(
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["mcp.shop.example"],
        allowed_origins=["https://shop.example"],
    ),
)
```

`decode_and_verify_jwt` và `orders_of` là phần của bạn: thư viện JWT với khoá công khai của AS, và truy vấn dữ liệu. Kết quả khi gọi server này:

| Request | Kết quả |
|---|---|
| Không có token, hoặc token sai | `401`, `WWW-Authenticate: Bearer error="invalid_token", ..., resource_metadata="https://mcp.shop.example/.well-known/oauth-protected-resource/mcp"` |
| Token hợp lệ nhưng `aud` là `https://other.example/mcp` | `401` |
| Token đúng audience, thiếu scope `orders:read` | `403`, `error="insufficient_scope"` |
| Token đúng | `200`, tool chạy với `subject` của token |
| `Origin: https://evil.example` | `403` |
| `GET /.well-known/oauth-protected-resource/mcp` | `{"resource":"https://mcp.shop.example/mcp","authorization_servers":["https://auth.shop.example"],"scopes_supported":["orders:read"],"bearer_methods_supported":["header"]}` |

**Điểm dễ sai:** ở `mcp` 2.3.0, nếu bỏ dòng `validate_token_resource=True`, SDK chỉ phát cảnh báo và **chấp nhận** token có audience khác: dòng thứ hai trong bảng trả `200`. Cảnh báo nói mặc định sẽ thành `True` ở bản 3.0. Cho tới lúc đó, hoặc đặt `True`, hoặc tự kiểm `aud` trong `verify_token`.

### Scope

- `scopes_supported` trong metadata nên là tập tối thiểu; liệt kê mọi scope có thể có là một lỗi được nêu tên trong hướng dẫn bảo mật [6].
- Khi thiếu quyền, server trả `403` `insufficient_scope` kèm scope cần thiết. Client xin lại quyền với **hợp** của scope cũ và scope mới [5]. Server nên đòi đủ scope cho một thao tác trong một lần, không bắt người dùng đăng nhập lại nhiều lượt.

### Phía client: mix-up và `iss`

Client nối nhiều server, mỗi server dùng một AS khác nhau, có thể bị lừa gửi authorization code cho nhầm AS. Đặc tả yêu cầu client ghi lại `issuer` trước khi chuyển hướng, so sánh chính xác (không chuẩn hoá chuỗi) với tham số `iss` trong response theo RFC 9207, và từ chối response thiếu `iss` khi AS tuyên bố có hỗ trợ [6]. Hướng dẫn bảo mật nói rõ PKCE một mình không chặn được tấn công này [6].

## 2.4 Sampling, elicitation và các tính năng nâng cao

### Elicitation: hỏi người dùng giữa chừng

Server cần người dùng cung cấp thông tin hoặc xác nhận thì trả `input_required` với request `elicitation/create` (§1.3). Có hai mode [30]:

- `form`: client hiện form theo `requestedSchema`, dữ liệu người dùng nhập đi qua client về server.
- `url`: client mở một URL để người dùng làm việc ngoài client (đăng nhập dịch vụ bên thứ ba, nhập API key, thanh toán). Ngoài chính URL, dữ liệu không đi qua client. Client phải hiện rõ domain đích và xin đồng ý trước khi mở.

Server **MUST NOT** dùng form để xin mật khẩu, API key, access token hay thông tin thanh toán; những thứ đó **MUST** đi qua mode `url` [30]. Tên, email, username thì không bị cấm. Mode `url` cũng không dùng để client xin quyền truy cập chính MCP server; việc đó thuộc luồng OAuth ở §2.3 [30].

### Sampling: đừng dùng cho server mới

Sampling cho server mượn model của client để sinh text. Nó deprecated từ bản 2026-07-28 [1]. Server cần LLM thì gọi thẳng API của provider, tự quản lý key và chi phí.

### Tasks: tool chạy lâu

Tool chạy vài phút (build, xuất báo cáo, crawl) không nên giữ một HTTP request mở suốt thời gian đó, vì stream đứt là mất request (§1.4). Bản 2025-11-25 đưa Tasks vào lõi ở dạng thử nghiệm; bản 2026-07-28 chuyển nó thành extension `io.modelcontextprotocol/tasks`: server trả về task handle, client hỏi trạng thái bằng `tasks/get`, gửi thêm input bằng `tasks/update`; `tasks/list` bị bỏ [1]. Code viết theo Tasks bản 2025-11-25 phải sửa lại.

### Extensions và MCP Apps

Bản 2025-11-25 thêm cơ chế extension: client và server khai báo extension hỗ trợ trong `capabilities.extensions`, khoá theo định danh có tiền tố như `io.modelcontextprotocol/...` [2]. Extension chính thức đầu tiên là **MCP Apps** (`io.modelcontextprotocol/ui`), ra mắt 26/1/2026: tool gắn với một resource HTML (`_meta.ui.resourceUri`, ví dụ `ui://charts/interactive`), host hiển thị UI đó trong iframe sandbox ngay trong hội thoại [27]. Lúc ra mắt có Claude (web, desktop, mobile), goose, VS Code Insiders và ChatGPT hỗ trợ [27]. UI gọi tool phải đi qua cùng luồng duyệt như tool call thường.

### Progress, cancellation, trace

- Progress: client đặt `progressToken` trong `_meta` của request; server gửi `notifications/progress` trên chính response stream của request đó [3].
- Cancellation: trên stdio gửi `notifications/cancelled`; trên HTTP đóng stream [3].
- Trace: bản 2026-07-28 chuẩn hoá `traceparent`, `tracestate`, `baggage` trong `_meta`, để một trace OpenTelemetry đi từ host qua client, server tới các dịch vụ phía sau [1]. Cách gắn span cho LLM call ở [LLM Observability](../llm-observability/).

## 2.5 Client hỗ trợ MCP

| Client | Cách thêm server | Ghi chú |
|---|---|---|
| Claude Code | `claude mcp add --transport http sentry https://mcp.sentry.dev/mcp`; stdio: `claude mcp add my-server -e API_KEY=xxx -- npx my-mcp-server` | Scope `local` (mặc định), `user`, `project`. Scope `project` ghi vào `.mcp.json` ở gốc repo để chia sẻ qua git; server trong file này phải được người dùng duyệt trước khi chạy, và repo clone về không tự duyệt được server của chính nó [16] |
| Claude (web, desktop, mobile) | Connector trong cài đặt | Hỗ trợ MCP Apps [27] |
| ChatGPT | Apps và connector | Hỗ trợ MCP Apps [27] |
| VS Code | Cấu hình MCP server của workspace | Hỗ trợ MCP Apps (bản Insiders khi ra mắt) [27] |
| goose | Extension | Dự án khởi đầu của Agentic AI Foundation [26] |
| Agent tự viết | MCP SDK làm client | Toàn quyền lọc, duyệt, log ([§2.2](#22-sdk-chính-thức)) |

Claude Code có vài giới hạn đáng biết khi viết server cho nó: cảnh báo khi output của một tool vượt 10.000 token, và mặc định cắt ở 25.000 token (đổi bằng biến môi trường `MAX_MCP_OUTPUT_TOKENS`) [16]. Tool search bật mặc định với các model từ Opus 4 và Sonnet 4 trở đi: tool definition của MCP server được hoãn nạp, chỉ vào context khi model tìm đến; biến môi trường `ENABLE_TOOL_SEARCH` đổi hành vi này [16].

## 2.6 MCP connector trong API của provider

Bốn provider lớn cho bạn khai MCP server ngay trong request; provider gọi server từ hạ tầng của họ.

| | Claude | OpenAI | Gemini | Grok |
|---|---|---|---|---|
| API | Messages API, beta | Responses API | Interactions API | Responses-style API |
| Khai báo | `mcp_servers` + tool `mcp_toolset` | tool `type: "mcp"` | tool `type: "mcp_server"` | tool `type: "mcp"` |
| Transport | Streamable HTTP và SSE, URL `https://` public | Streamable HTTP và HTTP/SSE; server private qua Secure MCP Tunnel | Chỉ Streamable HTTP | Streamable HTTP và SSE |
| Lọc tool | `default_config` / `configs` với `enabled`, `defer_loading` | `allowed_tools`, `defer_loading` | — | `allowed_tools`; không đặt thì mọi tool vào context |
| Duyệt trước khi gọi | Không có trong API; tự lọc tool ghi | **Mặc định có**, `require_approval` | — | `require_approval` không hỗ trợ |
| Xác thực | `authorization_token` | `authorization`, `headers` | — | `authorization`, `headers` |
| Ghi chú | Chỉ tool, không resources/prompts; không thuộc phạm vi ZDR | `authorization` không được lưu, gửi lại mỗi request | Tên server không chứa `-` | |
| Nguồn | [12] | [13] | [14] | [15] |

"—" nghĩa là tài liệu của provider không đề cập.

### Claude

```json
{
  "model": "claude-sonnet-5-5",
  "max_tokens": 2048,
  "messages": [{"role": "user", "content": "Đơn A1001 đã giao chưa?"}],
  "mcp_servers": [{"type": "url", "url": "https://mcp.shop.example/mcp",
                   "name": "orders", "authorization_token": "<access token>"}],
  "tools": [{"type": "mcp_toolset", "mcp_server_name": "orders",
             "default_config": {"enabled": false},
             "configs": {"get_order": {"enabled": true}}}]
}
```

Gửi với header `anthropic-beta: mcp-client-2025-11-20`. Mẫu trên là allowlist: tắt mọi tool, bật riêng `get_order`. Response chứa block `mcp_tool_use` và `mcp_tool_result` [12]. Các điểm cần biết [12]:

- Mỗi server trong `mcp_servers` phải được đúng một `mcp_toolset` tham chiếu. Tên tool trong `configs` không tồn tại trên server chỉ sinh cảnh báo phía backend, không báo lỗi; gõ sai tên tool trong allowlist thì tool đó im lặng không xuất hiện.
- Bạn tự chạy luồng OAuth và tự refresh token; API chỉ nhận token đã có.
- Beta `mcp-client-2026-09-15` thêm khả năng ghim danh sách tool: response mở đầu bằng block `mcp_tool_listing` cho mỗi server được hỏi; gửi lại message assistant nguyên vẹn (kể cả block đó) thì các request sau dùng danh sách đã ghi thay vì hỏi lại server. Bạn cũng có thể chép danh sách vào field `tools` của toolset. Code đọc `content[0]` phải bỏ qua block này.
- Connector không thuộc phạm vi Zero Data Retention: tool definition và kết quả lưu theo chính sách lưu trữ tiêu chuẩn.
- Không có cơ chế duyệt trong API. Với trợ lý chỉ đọc, tài liệu khuyên denylist các tool ghi; cần người duyệt trước thao tác ghi thì tự chạy client (§2.2).

### OpenAI

```json
{"type": "mcp", "server_label": "orders",
 "server_url": "https://mcp.shop.example/mcp",
 "authorization": "<access token>",
 "allowed_tools": ["get_order", "refund_order"],
 "require_approval": {"never": {"tool_names": ["get_order"]}}}
```

Lần đầu, API lấy danh sách tool và ghi vào item `mcp_list_tools`; chừng nào item đó còn trong context, API không hỏi lại server [13]. Mặc định mọi lần gọi cần duyệt: response chứa item `mcp_approval_request` (có `name`, `arguments`), bạn trả lời bằng item `mcp_approval_response` với `approve` và `approval_request_id` trong request tiếp theo [13]. Mẫu trên bỏ duyệt cho `get_order` và giữ duyệt cho `refund_order`. Mỗi lần gọi tạo item `mcp_call` có `arguments`, `output`, `error` [13].

Các điểm cần biết [13]: `connector_id` (connector có sẵn của OpenAI) deprecated với model ra sau 1/9/2026; dùng `server_url`, hoặc `tunnel_id` cho server private qua Secure MCP Tunnel. Với `store=true`, dữ liệu gửi tới MCP server được log 30 ngày trừ khi tổ chức bật Zero Data Retention. Tài liệu khuyên chỉ nối server chính chủ (ví dụ server Stripe do Stripe host) và cảnh giác với "aggregator".

### Gemini và Grok

Gemini nhận remote MCP trong Interactions API, không trong `generateContent` thông thường: tool `{"type": "mcp_server", "url": ...}`, chỉ Streamable HTTP, tên server không được chứa `-` [14]. Grok dùng `type: "mcp"` với `server_url`, `server_label`, `allowed_tools`; SDK native đổi tên thành `allowed_tool_names`, `extra_headers`. Grok không hỗ trợ `require_approval`, nên với Grok, tool ghi phải lọc khỏi `allowed_tools` hoặc tự chạy client [15].

### Khi nào để provider làm client

Để provider gọi server khi server đã public, tool chỉ đọc hoặc provider có bước duyệt bạn tin được, và bạn không cần can thiệp giữa các lần gọi. Tự làm client khi server chạy local hoặc trong mạng nội bộ, khi cần duyệt tool ghi theo luật của bạn, khi cần log đầy đủ mọi tham số theo định dạng của bạn, hoặc khi dữ liệu không được phép đi qua hạ tầng của provider.

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Thiết kế MCP server: tool nào nên expose

**Tool theo tác vụ người dùng, không theo endpoint.** Bọc nguyên REST API thành 40 tool (`list_issues`, `get_issue`, `list_comments`, `get_user`...) bắt model tự ghép bốn lần gọi cho một câu hỏi đơn giản và đẩy cả bốn kết quả thô vào context. Một tool `get_issue_summary(issue_id)` trả issue kèm comment gần nhất và người được giao làm việc đó trong server. Nguyên tắc thiết kế bộ tool chung ở [Tool use §3.1](../tool-use/#31-thiết-kế-bộ-tool-ít-mà-rõ); MCP làm nó quan trọng hơn vì server của bạn sẽ chạy cạnh server của người khác trong cùng một context.

**Ít tool.** Mỗi tool definition tốn token ở mọi request. Anthropic đo một cấu hình 5 server với 58 tool tốn khoảng 55.000 token trước khi hội thoại bắt đầu, và từng thấy tool definition chiếm 134.000 token trước khi tối ưu [17]. Server của bạn không phải nơi duy nhất trong context.

**Đặt tên tránh trùng.** Tên tool chỉ cần duy nhất trong một server. Client nối nhiều server sẽ gặp hai tool `search`; đặc tả khuyên client thêm tiền tố định danh server, và lưu ý `name` trong `serverInfo` không đảm bảo duy nhất [4]. Phía server, tên cụ thể (`search_tickets` thay vì `search`) giảm nhầm lẫn bất kể client xử lý thế nào.

**Luật nghiệp vụ nằm trong code của tool.** "Chỉ hoàn tiền đơn đã giao" nằm trong `refund_order` (§1.2), không nằm trong description để model tự tuân theo. Description nói cho model biết khi nào nên gọi; code quyết định có được làm hay không.

**Danh tính và quyền lấy từ token.** Tool không nhận `user_id` làm tham số. Model điền tham số từ hội thoại, mà hội thoại có thể chứa prompt injection. `my_orders()` ở §2.3 lấy `subject` từ token.

**Thứ tự `tools/list` cố định.** Đặc tả bản 2026-07-28 khuyên server trả tool theo thứ tự xác định [1]. Danh sách đổi thứ tự giữa các lần gọi làm thay đổi prompt và làm mất prompt cache của client (xem [Multi-turn Conversation & Prompt Caching](../multi-turn-conversation/)).

## 3.2 Mô tả tool và kích thước output

### Tool definition

Description và schema của tool là một phần prompt của mọi request; cách viết ở [Tool use §3.2](../tool-use/#32-viết-description-và-schema-cho-tool). Với MCP, thêm hai việc:

- **Đo.** Lấy `tools/list` của server, chuyển sang định dạng tool của model, đếm token bằng token counting API của provider. Làm lại mỗi khi thêm tool. `outputSchema` lớn cũng tính vào.
- **Cho phép client hoãn nạp.** Client lớn đã có tool search: tool definition chỉ vào context khi model tìm đến. Claude Code bật mặc định [16]; Claude API có `defer_loading` trong `mcp_toolset` [12]; OpenAI có `defer_loading` kèm tool search [13]. Với tool search của Anthropic, một cấu hình trên 50 tool giảm từ khoảng 77.000 xuống khoảng 8.700 token, và độ chính xác chọn tool trong đánh giá nội bộ tăng (Opus 4 từ 49% lên 74%, Opus 4.5 từ 79,5% lên 88,1%) [17]. Tool search tìm theo tên và description, nên tool mô tả mơ hồ sẽ khó được tìm thấy.

### Tool output

Tool result vào context và được model đọc như mọi input khác ([Tool use §3.3](../tool-use/#33-tool-result-là-prompt)). Một tool trả 5.000 dòng log làm đầy context và đẩy chi phí của mọi lượt sau. Cách giảm:

- **Lọc trong server.** Nhận tham số lọc (`since`, `status`, `limit`) và trả phần cần thiết. Mặc định của `limit` là quyết định của bạn; ghi nó trong description.
- **Trả tham chiếu thay dữ liệu.** `resource_link` trỏ tới dữ liệu đầy đủ, hoặc handle để tool khác đọc tiếp từng phần. Đặc tả lưu ý resource link do tool trả về không nhất thiết có trong `resources/list` [4].
- **Có `structuredContent` khi client cần xử lý bằng code**, kèm text tóm tắt cho model.
- **Biết giới hạn của client.** Claude Code cảnh báo ở 10.000 token và cắt ở 25.000 token output của một tool [16]. Output bị cắt giữa chừng thường tệ hơn output ngắn có chủ đích.

### Code execution với MCP

Anthropic mô tả một cách tiếp cận khác cho agent dùng nhiều tool: trình bày MCP server thành thư viện code (mỗi tool một file), agent đọc file của tool cần dùng rồi viết code gọi tool, lọc và nối kết quả trong sandbox, chỉ đưa phần cuối cùng vào model [18]. Trong ví dụ của bài viết (lấy transcript từ Google Drive rồi ghi vào Salesforce), lượng token giảm từ khoảng 150.000 xuống khoảng 2.000, tức 98,7% [18]. Đó là số của một ví dụ chọn để minh hoạ, không phải mức giảm chung. Bài viết cũng nêu cái giá: cần môi trường chạy code an toàn với sandbox, giới hạn tài nguyên và giám sát [18].

## 3.3 Remote server: deploy, scale, multi-tenant

### Stateless

Server theo bản 2026-07-28 không có session, nên mọi instance phục vụ được mọi request: load balancer round-robin bình thường, không sticky session, không session store [11]. Cloudflare mô tả server MCP chạy trong một Worker theo cách này, và đã đưa `createMcpHandler` vào TypeScript SDK chính thức [25]. Trạng thái cần giữ giữa các lần gọi đi vào store chung, khoá bằng handle (§1.3).

Server còn phục vụ client đời 2025 thì vẫn có session cho nhóm client đó. Python SDK 2.3.0 có tham số `stateless_http` cho `streamable_http_app()`, và giới hạn session mặc định (`session_idle_timeout`, `max_sessions`) cho phần legacy. Theo dõi tỉ lệ request đời cũ để biết khi nào tắt được.

### Host, Origin và reverse proxy

Như §1.4: khi không truyền `transport_security`, Python SDK chỉ nhận host localhost. Sau domain thật, khai `allowed_hosts` và `allowed_origins` như ví dụ ở §2.3. Đừng tắt `enable_dns_rebinding_protection` để "cho chạy được"; kiểm `Origin` là MUST của đặc tả [3].

Server có response dạng SSE (progress, `subscriptions/listen`) đặt sau nginx thì tắt buffering cho endpoint MCP, bằng `proxy_buffering off` hoặc header response `X-Accel-Buffering: no` [31]. Với stream sống lâu, đặc tả khuyến khích server gửi định kỳ một dòng comment SSE (dòng bắt đầu bằng `:`) để giữ kết nối [3]. Nhớ rằng stream đứt là mất request (§1.4); timeout của proxy phải dài hơn thời gian chạy của tool chậm nhất, hoặc tool đó phải chuyển sang Tasks.

### Multi-tenant

Ba quy tắc:

- **Tenant và người dùng lấy từ token ở mỗi request.** Không từ tham số, không từ cache của request trước.
- **Mọi cache khoá theo tenant.** Kết quả `tools/list` có thể khác nhau theo người dùng; đánh `cacheScope: "private"` khi đúng như vậy [1].
- **Handle khoá theo người dùng.** Lưu `<user_id>:<handle>` và kiểm ở mỗi lần gọi (§1.3). Hướng dẫn bảo mật gọi lỗi này là chiếm handle trạng thái [6].

Sự cố của Asana cho thấy hậu quả. Asana ra mắt MCP server ngày 1/5/2025, phát hiện lỗi ngày 4/6, tắt tính năng từ 5/6 đến 17/6; thông báo gửi khách hàng nói lỗi "could have potentially exposed certain information from your Asana domain to other Asana MCP users" [24]. Asana reset mọi kết nối MCP và không công bố chi tiết lỗi trong bài postmortem mà The Register trích [24]. Không có dấu hiệu dữ liệu bị khai thác [24]. Trước khi mở server cho nhiều tổ chức, viết test gọi với token của tenant A và mã đối tượng của tenant B cho từng tool.

### Observability

- Log mỗi `tools/call` với: tool, tham số (đã che dữ liệu nhạy cảm), `subject` và `client_id` của token, tenant, thời gian, `isError`. Khi có sự cố như Asana, đây là cách duy nhất biết ai đã thấy gì.
- Đọc `traceparent` từ `_meta` để nối span của server vào trace của client (§2.4).
- Metric theo tool: số lần gọi, tỉ lệ `isError`, latency, kích thước output. Tool có tỉ lệ lỗi nghiệp vụ cao thường là tool có description hoặc schema khó hiểu với model.
- Log ra stderr hoặc OpenTelemetry, không qua `notifications/message` (deprecated) [1].

## 3.4 Bảo mật: tool poisoning, quyền hạn, server bên thứ ba

Phần này nói về rủi ro riêng của MCP. Nguyên lý chung về prompt injection, least privilege và bước duyệt ở [Bảo mật LLM](../llm-security/) và [Tool use §3.6](../tool-use/#36-hành-động-có-tác-dụng-phụ-và-bảo-mật).

### Tool definition là input không tin cậy

Khi nối server của bên thứ ba, description của tool họ viết đi thẳng vào prompt của bạn. Invariant Labs mô tả ba dạng tấn công (1/4/2025) [19]:

- **Tool poisoning**: description chứa chỉ dẫn ẩn người dùng không thấy nhưng model đọc được. Ví dụ của họ: tool `add` cộng hai số, description có khối `<IMPORTANT>` yêu cầu model đọc `~/.cursor/mcp.json` và khoá SSH rồi nhét nội dung vào một tham số phụ tên `sidenote`.
- **Rug pull**: server đổi description sau khi người dùng đã duyệt.
- **Tool shadowing**: description của tool độc hại chỉ dẫn model thay đổi cách dùng tool của một server **khác**, đáng tin. Ví dụ: chuyển mọi email gửi qua `send_email` của server thật sang địa chỉ của kẻ tấn công.

Shadowing cho thấy một server độc hại ảnh hưởng được tới server khác trong cùng context, nên kiểm từng server riêng là chưa đủ. Phòng thủ phía client:

- Chỉ nối server chính chủ hoặc đã review; lọc tool tối thiểu.
- Khi người dùng duyệt server, lưu hash của name, description, inputSchema của từng tool. Mỗi lần lấy lại danh sách (khi khởi động, khi nhận thông báo `list_changed`), so hash; khác thì tắt tool đó và hỏi lại. Beta ghim danh sách tool của Claude API (§2.6) giải quyết một phần: danh sách không đổi giữa chừng một hội thoại.
- Hiển thị tham số đầy đủ của tool call khi xin duyệt, gồm cả tham số "phụ" như `sidenote`.

### Prompt injection qua tool result

Tool result chứa dữ liệu do người ngoài viết (issue, email, ticket, trang web) là kênh injection chính. Hai trường hợp được ghi lại kỹ:

- **GitHub MCP (Invariant Labs, 26/5/2025)** [20]. Kẻ tấn công mở một issue trong repo public của nạn nhân, chứa chỉ dẫn. Nạn nhân nhờ agent "xem các issue mở". Agent đọc issue, làm theo chỉ dẫn: lấy dữ liệu từ repo private của nạn nhân và đăng vào một pull request ở repo public. Demo dùng Claude Desktop với Claude 4 Opus và chế độ "Always Allow". Invariant nói không có lỗi nào trong code của GitHub MCP server; đây là vấn đề kiến trúc phải xử lý ở tầng hệ thống agent [20].
- **Supabase MCP (Simon Willison mô tả, 6/7/2025)** [21]. Kịch bản trình diễn: agent kết nối database bằng `service_role` (bỏ qua row-level security), đọc một ticket hỗ trợ chứa chỉ dẫn, rồi ghi dữ liệu nhạy cảm vào chính ticket đó nơi kẻ tấn công đọc được. Đây là demo, không phải sự cố đã xảy ra với khách hàng.

Willison gọi tổ hợp này là **lethal trifecta**: agent có (1) quyền đọc dữ liệu riêng tư, (2) tiếp xúc nội dung không tin cậy, (3) kênh đưa dữ liệu ra ngoài [21]. Có đủ ba thì một injection thành công là dữ liệu bị lấy. Cắt một chân là đủ chặn lớp tấn công này: chế độ chỉ đọc, token chỉ có quyền một project, không cho tool ghi ra nơi công khai trong cùng phiên đã đọc nội dung từ người ngoài.

MCP làm trifecta dễ xảy ra hơn vì người dùng tự ghép server: một server đọc email (nội dung không tin cậy), một server đọc Drive (dữ liệu riêng tư), một server gửi Slack (kênh ra ngoài). Không server nào sai, tổ hợp mới nguy hiểm.

### Server tấn công client

Client cũng là bề mặt tấn công. **CVE-2025-6514** trong `mcp-remote` (công cụ nối client chỉ hỗ trợ stdio tới server remote): server độc hại trả `authorization_endpoint` được tạo đặc biệt, `mcp-remote` mở URL đó theo cách dẫn tới thực thi lệnh tuỳ ý trên máy người dùng. CVSS 9.6, ảnh hưởng 0.0.5 đến 0.1.15, sửa ở 0.1.16; JFrog công bố 9/7/2025 [22]. Người dùng chỉ cần kết nối tới server độc hại.

Hướng dẫn bảo mật của MCP rút ra các quy tắc cho client [6]:

- Kiểm URL nhận từ server (OAuth endpoint, link) trước khi mở: chỉ `https` (hoặc `http` cho loopback khi dev), không mở URL bằng lệnh shell.
- Khi fetch URL trong luồng OAuth (`resource_metadata`, `authorization_servers`, các endpoint trong metadata của AS), chặn địa chỉ private, link-local (`169.254.169.254`) và loopback, kiểm redirect, cân nhắc đi qua egress proxy. Authorization server hỗ trợ CIMD phải fetch URL do client cung cấp nên cũng chịu rủi ro SSRF này.
- Cài đặt server local "một cú bấm" phải hiện nguyên văn lệnh sẽ chạy, không cắt bớt, và chờ người dùng đồng ý.

### Server proxy: confused deputy

Server MCP làm proxy tới API của bên thứ ba (dùng một `client_id` cố định với AS của bên đó, cho client MCP đăng ký động) có thể bị lợi dụng: AS của bên thứ ba nhớ consent qua cookie, kẻ tấn công gửi cho nạn nhân một link đăng ký với `redirect_uri` của chúng, consent được bỏ qua vì cookie, và authorization code về tay kẻ tấn công [6]. Đặc tả yêu cầu server proxy **MUST** xin consent cho từng client, với cookie consent có tiền tố `__Host-`, `Secure`, `HttpOnly`, `SameSite=Lax`, gắn với `client_id` [6].

### Chuỗi cung ứng và hosting

- Server local chạy với quyền của người dùng: đọc được file, khoá SSH, biến môi trường. Cài server từ npm hay PyPI là chạy code của người khác. Ghim phiên bản, ưu tiên server chính chủ, đọc code server nhỏ trước khi dùng.
- Nền tảng host MCP server cũng là mục tiêu. GitGuardian (công bố 22/10/2025) tìm thấy ở Smithery: tham số `dockerBuildPath` không được kiểm, đặt `".."` thì build context là thư mục home của máy build, nơi có file chứa token fly.io quyền rộng; token đó điều khiển được khoảng 3.000 ứng dụng [23]. Smithery sửa trong hai ngày sau khi nhận báo cáo, và không có bằng chứng bị khai thác [23].

Chi tiết về chuỗi cung ứng ở [Bảo mật LLM §3.4](../llm-security/#34-chuỗi-cung-ứng-model-mcp-server-plugin-bên-thứ-ba).

### Bước duyệt

Đặc tả: SHOULD luôn có người trong vòng lặp, có khả năng từ chối tool call; ứng dụng SHOULD hiện tool nào được expose, hiện chỉ báo khi tool được gọi, và hỏi xác nhận cho thao tác nhạy cảm [4]. Thực tế các client khác nhau xa: OpenAI duyệt mặc định, Claude API và Grok không có bước duyệt trong API (§2.6), Claude Code hỏi theo cấu hình quyền. Chế độ "luôn cho phép" là điều kiện của demo GitHub MCP [20]. Nếu ứng dụng của bạn cho phép "luôn cho phép", giới hạn nó cho tool chỉ đọc của server đáng tin.

## 3.5 Test và debug MCP server

### Unit test trong process

`Client(mcp)` của Python SDK chạy server trong cùng process, không cần mạng hay subprocess:

```python
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
```

Test những gì model sẽ thấy: nội dung thông báo lỗi có đủ để model tự sửa không, `structuredContent` có khớp `outputSchema` không, tool definition (`list_tools`) có đúng description và annotation không.

### MCP Inspector

MCP Inspector là công cụ debug chính thức, một gói với ba giao diện: web (mặc định), CLI và TUI. Cần Node 22.19.0 trở lên [9]:

```bash
# web UI, nối tới server stdio
npx @modelcontextprotocol/inspector python orders_server.py

# CLI: gọi một tool rồi thoát, phù hợp cho CI
npx @modelcontextprotocol/inspector --cli python orders_server.py \
  --method tools/call --tool-name get_order --tool-arg order_id=A1001 --format json

# server HTTP
npx @modelcontextprotocol/inspector --cli http://127.0.0.1:8765/mcp --transport http \
  --method tools/list
```

Với server ở §1.2, lệnh CLI thứ hai in ra `{"result":{"content":[...],"structuredContent":{"order_id":"A1001",...},"isError":false}}`. Gọi `get_order` với `B9` thì in result có `isError: true` kèm một dòng `{"error":{"code":"tool_is_error",...}}`, và CLI thoát với mã lỗi, nên dùng được trong CI. Inspector tự thương lượng giữa thời kỳ cũ và mới (§1.3) [9]. Web UI in một URL có API token riêng cho mỗi lần chạy [9]; đừng chia sẻ URL đó, vì Inspector khởi chạy được lệnh trên máy bạn.

### curl

Với Streamable HTTP, curl đủ để kiểm từng request (§1.4). Bốn header hay quên: `Accept` có cả hai content type, `MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name`. Thêm `_meta` có `protocolVersion` trong body.

### Eval

Unit test kiểm server đúng; eval kiểm model dùng server đúng: chọn đúng tool, điền đúng tham số, dừng đúng lúc. Lấy câu hỏi thật, chạy qua host với server của bạn, chấm cả kết quả lẫn chuỗi tool call. Cách làm chung ở [Tool use §3.9](../tool-use/#39-eval-cho-tool-use) và [Evaluation](../evaluation/). Chạy lại eval mỗi khi đổi description hay thêm tool: một tool mới mô tả chồng lấn có thể làm model chọn nhầm tool cũ vốn đang chạy tốt.

## 3.6 Failure modes thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Client ngắt kết nối server stdio ngay khi khởi động, lỗi parse JSON | Server in log hoặc banner ra stdout | Mọi log ra stderr |
| Mọi request qua domain thật trả `421 Misdirected Request` | Python SDK chỉ nhận host localhost khi không cấu hình | Khai `allowed_hosts`, `allowed_origins` (§3.3) |
| `400`, mã `-32020` | Thiếu hoặc sai `Mcp-Method`/`Mcp-Name`/`Mcp-Param-*`; header không khớp body | Gửi header khớp body; giá trị ngoài ASCII mã hoá base64 (§1.4) |
| `400`, mã `-32022` | Client gửi phiên bản server không hỗ trợ | Chọn phiên bản trong `data.supported` |
| `400` "Missing session ID" khi gửi request kiểu mới | Header `MCP-Protocol-Version` vẫn là bản cũ, server xử lý như client đời 2025 | Header và `_meta` cùng một phiên bản |
| Client mới không nói chuyện được với server cũ | Server chỉ hiểu `initialize` | Fallback theo §1.3, hoặc nâng cấp server |
| Mất trạng thái khi scale hoặc deploy | State gắn với session hay bộ nhớ của instance | Handle + store chung, kiểm quyền theo người dùng |
| Request dài mất giữa chừng | Stream đứt, bản 2026 không còn resume | Client gửi lại với id mới; tool phải idempotent; việc dài chuyển sang Tasks |
| Token của dịch vụ khác vẫn gọi được server | Không kiểm audience (Python SDK: chưa bật `validate_token_resource`) | Bật kiểm audience, test bằng token có `aud` khác |
| `403 insufficient_scope` lặp đi lặp lại | Client xin lại chỉ scope mới, mất scope cũ | Xin hợp của scope cũ và mới; server đòi đủ scope trong một lần |
| Context đầy trước câu đầu tiên của người dùng | Mọi tool của mọi server vào prompt | Lọc tool, tool search / `defer_loading`, bớt server |
| Model gọi nhầm tool giữa hai server | Tên chung chung (`search`), description chồng lấn | Tên cụ thể, client thêm tiền tố server |
| Prompt cache hit thấp dù prompt không đổi | `tools/list` trả thứ tự khác nhau | Thứ tự cố định |
| Gõ sai tên tool trong allowlist của Claude connector, tool không xuất hiện, không có lỗi | Tên lạ chỉ sinh cảnh báo phía backend | So `configs` với `tools/list` trong test |
| OpenAI không thấy tool mới thêm vào server | `mcp_list_tools` còn trong context nên API không hỏi lại | Bỏ item đó khỏi context khi cần làm mới (đổi lại độ trễ) |
| Gemini từ chối cấu hình MCP | Tên server có dấu `-` | Đổi tên |
| Claude connector không nối được server | Server chạy local hoặc stdio | Server public qua `https`, hoặc tự làm client |
| Tool đã duyệt bắt đầu làm việc lạ | Rug pull: description đổi sau khi duyệt | Ghim hash description, tắt tool khi hash đổi |
| Người dùng tenant A thấy dữ liệu tenant B | Cache hoặc truy vấn không khoá theo tenant | Tenant từ token mỗi request, cache khoá theo tenant, test chéo tenant |
| Tool bị huỷ nhưng vẫn ghi dữ liệu | Server không dừng khi stream đóng | Kiểm cancellation trong việc dài; idempotency key cho thao tác ghi |
| Model trả lời bằng thông tin không có trong dữ liệu sau khi tool lỗi | Lỗi nghiệp vụ trả bằng protocol error, model không thấy lý do | `isError: true` với thông báo cụ thể |

## 3.7 Checklist

**Thiết kế server**

- [ ] Mỗi tool là một tác vụ người dùng; ít tool; tên cụ thể, không trùng với tên phổ biến.
- [ ] Description nói khi nào dùng và khi nào không; ràng buộc tham số nằm trong schema.
- [ ] Luật nghiệp vụ nằm trong code; danh tính lấy từ token, không từ tham số.
- [ ] Lỗi nghiệp vụ trả `isError: true` với thông báo model dùng được; lỗi hạ tầng không lộ stack trace.
- [ ] Tool trả dữ liệu có `outputSchema` và text tương đương; output có giới hạn kích thước, có tham số lọc.
- [ ] Annotation khớp hành vi thật; tool khai idempotent thì thật sự idempotent.
- [ ] `tools/list` có thứ tự cố định.
- [ ] Không dùng sampling, roots, logging cho server mới; trạng thái đi bằng handle có hạn, khoá theo người dùng.

**Remote server**

- [ ] HTTPS; `allowed_hosts`/`allowed_origins` đúng domain; `Origin` sai trả 403.
- [ ] Protected Resource Metadata có `authorization_servers`; `scopes_supported` tối thiểu.
- [ ] Kiểm audience của token (Python SDK: `validate_token_resource=True`); có test với token audience khác.
- [ ] Không chuyển tiếp token của client sang API phía sau.
- [ ] Proxy tới bên thứ ba: consent theo từng client.
- [ ] Cache và truy vấn khoá theo tenant; có test chéo tenant cho từng tool.
- [ ] Log mỗi tool call kèm `subject`, `client_id`, tenant; nối trace qua `traceparent`.
- [ ] Proxy không buffer SSE; timeout dài hơn tool chậm nhất, hoặc tool đó dùng Tasks.

**Phía client và ứng dụng**

- [ ] Chỉ nối server chính chủ hoặc đã review; ghim phiên bản server local.
- [ ] Lọc tool tối thiểu cho mỗi server; tool ghi cần duyệt, không "luôn cho phép".
- [ ] Ghim hash tool definition đã duyệt; cảnh báo và tắt khi đổi.
- [ ] Không có tổ hợp đủ ba yếu tố: dữ liệu riêng tư, nội dung không tin cậy, kênh ra ngoài trong cùng một phiên.
- [ ] URL từ server được kiểm scheme, không mở bằng shell; fetch trong luồng OAuth chặn địa chỉ nội bộ.
- [ ] Đo token của tool definition; bật tool search hoặc `defer_loading` khi nhiều tool.
- [ ] Biết chính sách lưu dữ liệu của connector (Claude connector không thuộc ZDR; OpenAI log 30 ngày khi `store=true`).
- [ ] Theo dõi CVE của SDK, `mcp-remote`, MCP Inspector và các server đang dùng.

---

## Nguồn tham khảo

| # | Nguồn |
|---|---|
| 1 | Model Context Protocol, "Changelog" (bản 2026-07-28). https://modelcontextprotocol.io/specification/2026-07-28/changelog |
| 2 | Model Context Protocol, "Versioning" và "Versioning and Compatibility" (bản 2026-07-28). https://modelcontextprotocol.io/docs/learn/versioning ; https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning |
| 3 | Model Context Protocol, "Transports" và "Streamable HTTP" (bản 2026-07-28). https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http |
| 4 | Model Context Protocol, "Tools" (bản 2026-07-28). https://modelcontextprotocol.io/specification/2026-07-28/server/tools |
| 5 | Model Context Protocol, "Authorization" (bản 2026-07-28). https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization |
| 6 | Model Context Protocol, "Security Best Practices". https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices |
| 7 | Model Context Protocol, "SDKs". https://modelcontextprotocol.io/docs/sdk |
| 8 | MCP Python SDK v2 (tài liệu và mã nguồn gói `mcp` 2.3.0). https://py.sdk.modelcontextprotocol.io/ ; https://github.com/modelcontextprotocol/python-sdk |
| 9 | Model Context Protocol, "MCP Inspector". https://modelcontextprotocol.io/docs/tools/inspector |
| 10 | Model Context Protocol, "Schema Reference" (bản 2025-11-25), `ToolAnnotations`. https://modelcontextprotocol.io/specification/2025-11-25/schema |
| 11 | MCP Blog, công bố bản đặc tả 2026-07-28. https://blog.modelcontextprotocol.io/posts/2026-07-28/ |
| 12 | Anthropic, "MCP connector". https://platform.claude.com/docs/en/agents-and-tools/mcp-connector |
| 13 | OpenAI, "Connectors and MCP servers". https://developers.openai.com/api/docs/guides/tools-connectors-mcp |
| 14 | Google, "Function calling", mục Remote MCP trong Interactions API. https://aistudio.google.com/docs/function-calling |
| 15 | xAI, "Remote MCP Tools". https://docs.x.ai/developers/tools/remote-mcp |
| 16 | Anthropic, "Connect Claude Code to tools via MCP". https://code.claude.com/docs/en/mcp |
| 17 | Anthropic Engineering, "Introducing advanced tool use on the Claude Developer Platform" (24/11/2025). https://www.anthropic.com/engineering/advanced-tool-use |
| 18 | Anthropic Engineering, "Code execution with MCP" (4/11/2025). https://www.anthropic.com/engineering/code-execution-with-mcp |
| 19 | Invariant Labs, "MCP Security Notification: Tool Poisoning Attacks" (1/4/2025). https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks |
| 20 | Invariant Labs, "GitHub MCP Exploited: Accessing private repositories via MCP" (26/5/2025). https://invariantlabs.ai/blog/mcp-github-vulnerability |
| 21 | Simon Willison, "Supabase MCP can leak your entire SQL database" (6/7/2025). https://simonwillison.net/2025/Jul/6/supabase-mcp-lethal-trifecta/ |
| 22 | JFrog, "Critical RCE Vulnerability in mcp-remote: CVE-2025-6514" (9/7/2025). https://jfrog.com/blog/2025-6514-critical-mcp-remote-rce-vulnerability/ |
| 23 | GitGuardian, "Breaking MCP Server Hosting" (22/10/2025). https://blog.gitguardian.com/breaking-mcp-server-hosting/ |
| 24 | The Register, "Asana MCP server back online after plugging a data-leak hole" (18/6/2025). https://www.theregister.com/security/2025/06/18/asana-mcp-server-back-online-after-plugging-a-data-leak-hole/1199951 |
| 25 | Cloudflare Blog, "MCP v2" (6/8/2026). https://blog.cloudflare.com/mcp-v2/ |
| 26 | Linux Foundation, "Linux Foundation Announces the Formation of the Agentic AI Foundation" (9/12/2025). https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation |
| 27 | MCP Blog, "MCP Apps" (26/1/2026). https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/ |
| 28 | Anthropic Python SDK, `anthropic/lib/tools/mcp.py` (bản 2.6.0). https://github.com/anthropics/anthropic-sdk-python |
| 29 | A2A Protocol. https://a2a-protocol.org/latest/ |
| 30 | Model Context Protocol, "Elicitation" (bản 2026-07-28). https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation |
| 31 | nginx, "Module ngx_http_proxy_module", `proxy_buffering`. https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_buffering |
