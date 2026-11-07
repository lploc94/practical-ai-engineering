# Streaming: trả từng phần response qua SSE

Một công cụ tóm tắt hợp đồng trả về bản tóm tắt khoảng 1.500 token. Ở chế độ không stream, người dùng nhìn spinner gần nửa phút rồi mới thấy chữ đầu tiên. Nhiều người bấm tải lại trang ở giây thứ mười lăm, và mỗi lần bấm lại tạo ra thêm một request tốn tiền. Đội phát triển bật `stream: true`. Trên máy local, chữ hiện ra sau chưa tới một giây.

Lên production thì các lỗi mới bắt đầu xuất hiện:

- Qua nginx, chữ vẫn hiện ra cùng một lúc ở cuối, y như chưa bật stream.
- Câu trả lời dài bị cắt ngang ở giây thứ 60, không kèm thông báo lỗi nào.
- Lúc provider quá tải, stream đã trả HTTP 200 rồi nhận thêm một event `error`. App bỏ qua event đó, lưu câu trả lời cụt như một câu trả lời hoàn chỉnh, và cơ chế fallback sang model khác không được kích hoạt.
- Người dùng đóng tab nhưng backend vẫn đọc nốt stream, nên token vẫn bị tính.
- Một tính năng trả JSON không parse được cho đến token cuối cùng, nên stream không giúp gì cho phần đó.

Streaming làm thời gian chờ *cảm nhận* ngắn đi, còn tổng thời gian sinh thì gần như giữ nguyên. Đổi lại, một lần gọi request–response trở thành một kết nối mở kéo dài hàng chục giây, đi qua nhiều tầng mạng. Tầng nào cũng có thể gom buffer, cắt kết nối hoặc nuốt mất lỗi. Tài liệu này đi theo đúng chuỗi đó:

- **Nền tảng — Phần 1:** TTFT và tốc độ sinh token, giao thức SSE theo đặc tả, so sánh với WebSocket và chunked, cách provider chia output thành event, usage khi stream.
- **Tra cứu theo provider — Phần 2:** event model của OpenAI (Responses và Chat Completions), Claude, Gemini (`streamGenerateContent` và Interactions API), Grok, vLLM, Ollama, cùng các helper trong SDK và Vercel AI SDK.
- **Thiết kế và vận hành — Phần 3:** kiến trúc chuyển tiếp, cấu hình proxy và CDN, stream kèm tool call, JSON và reasoning, xử lý lỗi giữa chừng, huỷ request, UI, moderation, observability. Khi cần tìm nguyên nhân một lỗi cụ thể, xem [bảng ở §3.11](#311-failure-modes-thường-gặp).

> Bản tháng 11/2026. Tên event, field và tham số giữ nguyên tiếng Anh để tiện đối chiếu API. Những điểm chỉ thấy ở nguồn thứ cấp hoặc chưa đối chiếu được với tài liệu chính thức được đánh dấu tại chỗ.

Tài liệu liên quan: [Tool Use §3.7](../tool-use/#37-streaming-khi-có-tool-call) (gom tool call khi stream), [Structured Output §3.4](../structured-output/#34-streaming-partial-json) (parse JSON dở dang), [Reasoning Models](../reasoning-models/), [Latency](../latency/), [LLM Observability](../llm-observability/).

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Vì sao cần stream: TTFT, tốc độ sinh token và thời gian chờ cảm nhận](#11-vì-sao-cần-stream-ttft-tốc-độ-sinh-token-và-thời-gian-chờ-cảm-nhận)
  - [1.2 Server-Sent Events hoạt động thế nào](#12-server-sent-events-hoạt-động-thế-nào)
  - [1.3 SSE, WebSocket, HTTP chunked và gRPC streaming](#13-sse-websocket-http-chunked-và-grpc-streaming)
  - [1.4 Từ token đến event: delta, content block, usage](#14-từ-token-đến-event-delta-content-block-usage)
  - [1.5 Stream không thay đổi chi phí, nhưng thay đổi cách tính usage](#15-stream-không-thay-đổi-chi-phí-nhưng-thay-đổi-cách-tính-usage)
- [Phần 2 — Streaming theo provider](#phần-2--streaming-theo-provider)
  - [2.1 Bức tranh chung](#21-bức-tranh-chung)
  - [2.2 OpenAI: Chat Completions và Responses API](#22-openai-chat-completions-và-responses-api)
  - [2.3 Claude (Anthropic)](#23-claude-anthropic)
  - [2.4 Google Gemini](#24-google-gemini)
  - [2.5 xAI Grok](#25-xai-grok)
  - [2.6 Self-hosted: vLLM, SGLang, Ollama](#26-self-hosted-vllm-sglang-ollama)
  - [2.7 SDK và thư viện: helper stream, Vercel AI SDK](#27-sdk-và-thư-viện-helper-stream-vercel-ai-sdk)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Kiến trúc chuyển tiếp: provider → backend → client](#31-kiến-trúc-chuyển-tiếp-provider--backend--client)
  - [3.2 Stream qua proxy, load balancer và CDN](#32-stream-qua-proxy-load-balancer-và-cdn)
  - [3.3 Stream kèm tool call](#33-stream-kèm-tool-call)
  - [3.4 Stream kèm structured output: parse JSON chưa hoàn chỉnh](#34-stream-kèm-structured-output-parse-json-chưa-hoàn-chỉnh)
  - [3.5 Stream kèm reasoning/thinking](#35-stream-kèm-reasoningthinking)
  - [3.6 Lỗi giữa chừng: đứt kết nối, lỗi sau khi đã gửi một phần, retry và resume](#36-lỗi-giữa-chừng-đứt-kết-nối-lỗi-sau-khi-đã-gửi-một-phần-retry-và-resume)
  - [3.7 Người dùng huỷ: dừng request phía provider để không tốn token](#37-người-dùng-huỷ-dừng-request-phía-provider-để-không-tốn-token)
  - [3.8 Phía UI: render markdown từng phần, cuộn, hiển thị trạng thái](#38-phía-ui-render-markdown-từng-phần-cuộn-hiển-thị-trạng-thái)
  - [3.9 Moderation và guardrail trên output đang stream](#39-moderation-và-guardrail-trên-output-đang-stream)
  - [3.10 Observability: đo TTFT, tokens/giây, tỉ lệ stream bị cắt](#310-observability-đo-ttft-tokensgiây-tỉ-lệ-stream-bị-cắt)
  - [3.11 Failure modes thường gặp](#311-failure-modes-thường-gặp)
  - [3.12 Checklist](#312-checklist)
- [Nguồn tham khảo](#nguồn-tham-khảo)

---

## Tra cứu nhanh

### Có nên stream không

| Tình huống | Stream? | Lý do |
|---|---|---|
| Chat, trợ lý, viết nội dung dài cho người đọc | Có | Người dùng thấy chữ sau vài trăm ms thay vì chờ toàn bộ |
| Request có `max_tokens` lớn, kể cả khi không cần hiển thị dần | Có, dùng helper gom kết quả | Tránh HTTP timeout; SDK của Claude yêu cầu stream với `max_tokens` lớn [9] |
| Reasoning model trả lời lâu | Có, kèm trạng thái "đang suy nghĩ" | Thời gian suy luận nằm trong TTFT; không stream thì kết nối dễ bị proxy cắt |
| Output là JSON mà code cần đủ mới xử lý được | Thường không | Không quyết định nghiệp vụ được trên JSON dở dang; chỉ stream nếu cần xem trước |
| Batch job, pipeline nền, phân loại ngắn | Không | Không có người chờ; stream chỉ thêm độ phức tạp |
| Nội dung bắt buộc kiểm duyệt trước khi hiển thị | Cân nhắc | Phần đã hiển thị không thu hồi được ([§3.9](#39-moderation-và-guardrail-trên-output-đang-stream)) |

### Event model theo provider (snapshot 11/2026)

| | Text delta | Tool arguments delta | Event kết thúc | Usage nằm ở đâu | Lỗi giữa stream |
|---|---|---|---|---|---|
| OpenAI Responses | `response.output_text.delta` | `response.function_call_arguments.delta` | `response.completed` / `.incomplete` / `.failed` | `response.completed` → `response.usage` | Event `error` hoặc `response.failed` |
| OpenAI Chat Completions | `choices[].delta.content` | `choices[].delta.tool_calls[i].function.arguments` | `data: [DONE]` | Chunk cuối khi bật `stream_options.include_usage` | Chunk có field `error` |
| Claude | `content_block_delta` / `text_delta` | `content_block_delta` / `input_json_delta` | `message_stop` | `message_start` (input) và `message_delta` (cộng dồn) | `event: error` sau HTTP 200 |
| Gemini `streamGenerateContent` | `candidates[].content.parts[].text` | Function call nằm trọn trong part | `finishReason` khác rỗng, rồi đóng kết nối | `usageMetadata` | Chưa xác minh dạng lỗi giữa stream |
| Gemini Interactions | `step.delta` | `step.delta` với `arguments_delta` | `interaction.completed` | `interaction.usage` trong event kết thúc | Chưa xác minh |
| Grok | `choices[].delta.content` | Chưa xác minh | `data: [DONE]` | `usage` trong chunk (theo ví dụ docs) | Chưa xác minh |
| vLLM | Như Chat Completions | Như Chat Completions | `data: [DONE]` | Chunk cuối khi `include_usage` | Chunk lỗi |
| Ollama (`/api/chat`) | `message.content` mỗi dòng JSON | Trong `message` | Dòng có `"done": true` | Dòng `done: true` | Dòng `{"error": ...}` |

### Cấu hình tối thiểu để SSE không bị buffer hoặc cắt

| Tầng | Cấu hình |
|---|---|
| Response header | `Content-Type: text/event-stream; charset=utf-8`, `Cache-Control: no-cache, no-transform`, `X-Accel-Buffering: no` |
| Nội dung | Gửi comment `: ping` mỗi 15–30 giây khi chưa có token (đang xếp hàng, đang suy luận, đang chạy tool) |
| nginx | `proxy_buffering off` cho route stream (hoặc dựa vào `X-Accel-Buffering: no`), `proxy_read_timeout` lớn hơn khoảng im lặng dài nhất, `gzip off` |
| Middleware nén | Loại `text/event-stream` khỏi gzip/brotli |
| AWS ALB | `idle_timeout.timeout_seconds` (mặc định 60) lớn hơn khoảng heartbeat |
| Cloudflare | Gửi header ngay khi nhận request, heartbeat dưới 125 giây (Proxy Read Timeout, lỗi 524); docs không bảo đảm heartbeat reset timeout, cần test |
| API Gateway REST | Response transfer mode `STREAM`; tối đa 15 phút |
| Serverless (Vercel) | `maxDuration` đủ cho cả thời gian stream |
| Backend | Truyền abort signal của request client vào request gọi provider |

Chi tiết và nguồn ở [§3.2](#32-stream-qua-proxy-load-balancer-và-cdn).

---

# Phần 1 — Nền tảng

## 1.1 Vì sao cần stream: TTFT, tốc độ sinh token và thời gian chờ cảm nhận

Model sinh output từng token một. Ở chế độ không stream, provider giữ toàn bộ output đến token cuối rồi mới trả về. Stream thì gửi từng phần ngay khi chúng được sinh ra. Muốn biết stream giúp được bao nhiêu, cần tách độ trễ thành các thành phần. NVIDIA định nghĩa như sau trong tài liệu benchmark [35]:

- **TTFT (time to first token):** thời gian từ lúc gửi request đến lúc nhận token đầu tiên. Gồm thời gian xếp hàng, thời gian xử lý prompt (*prefill*) và, với reasoning model, thời gian suy luận trước khi có chữ hiển thị.
- **ITL (inter-token latency):** thời gian trung bình giữa hai token liên tiếp, `ITL = (e2e_latency − TTFT) / (output_tokens − 1)`. NVIDIA dùng TPOT như một tên khác của ITL. Một số công cụ benchmark định nghĩa TPOT hơi khác, nên khi so số liệu, kiểm tra công thức trước.
- **Tokens/giây** của một request xấp xỉ `1 / ITL`.

Tổng thời gian:

```text
e2e ≈ TTFT + (output_tokens − 1) × ITL
```

Ví dụ minh hoạ: TTFT 0,8 giây, ITL 20 ms, output 1.500 token. Tổng thời gian khoảng 0,8 + 1.499 × 0,02 ≈ 30,8 giây ở cả hai chế độ. Khác biệt nằm ở thời điểm người dùng thấy chữ đầu tiên: giây 0,8 khi stream, giây 30,8 khi không stream. Người đọc trung bình đọc chậm hơn tốc độ sinh của model, nên với câu trả lời dài, stream gần như xoá được cảm giác chờ.

Bốn hệ quả thực tế:

**Stream không sửa được TTFT cao.** Prompt dài, hàng đợi đông hay reasoning lâu đều làm TTFT tăng, và stream không rút ngắn phần đó. Muốn giảm TTFT phải rút gọn prompt, dùng prompt caching, đổi model hoặc giảm mức reasoning ([Latency](../latency/)).

**Với reasoning model, chữ đầu tiên có thể đến sau hàng chục giây.** Phần suy luận được sinh trước câu trả lời. Nếu provider không trả nội dung suy luận (hoặc chỉ trả bản tóm tắt), kết nối có thể im lặng rất lâu. Đây là lúc proxy hay cắt kết nối nhất ([§3.2](#32-stream-qua-proxy-load-balancer-và-cdn)), và cũng là lúc UI cần hiển thị trạng thái ([§3.5](#35-stream-kèm-reasoningthinking)).

**Stream còn là cách tránh timeout.** Tài liệu Claude ghi rằng SDK yêu cầu stream với request có `max_tokens` lớn để tránh HTTP timeout, và cung cấp helper gom stream thành một `Message` hoàn chỉnh cho những trường hợp không cần xử lý từng phần [9]. xAI khuyến nghị đặt timeout client rất cao (ví dụ 3.600 giây) cho reasoning model [16]. Vì vậy, kể cả job nền không hiển thị gì cho người dùng, stream vẫn có thể là lựa chọn đúng.

**Stream đổi kiểu lỗi.** Không stream thì lỗi đến dưới dạng HTTP status, và mọi thư viện HTTP đều xử lý được. Khi stream, status 200 đã được gửi đi trước khi model sinh token đầu tiên, nên lỗi xảy ra sau đó phải đi trong chính nội dung stream. Phần lớn lỗi khó chẩn đoán trong tài liệu này bắt nguồn từ đây ([§3.6](#36-lỗi-giữa-chừng-đứt-kết-nối-lỗi-sau-khi-đã-gửi-một-phần-retry-và-resume)).

## 1.2 Server-Sent Events hoạt động thế nào

OpenAI, Claude, Gemini, Grok và vLLM đều stream qua Server-Sent Events (SSE); API native của Ollama là ngoại lệ, dùng NDJSON ([§1.3](#13-sse-websocket-http-chunked-và-grpc-streaming)). SSE là một định dạng văn bản đơn giản, được định nghĩa trong HTML Living Standard [1]. Hiểu đúng định dạng này giúp bạn viết được parser đúng và đọc được raw stream khi debug.

### Định dạng

Server trả `Content-Type: text/event-stream`, luôn mã hoá UTF-8. Body là một chuỗi dòng, mỗi dòng kết thúc bằng LF, CRLF hoặc CR. Một dòng trống đánh dấu kết thúc một event.

```text
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Xin"}}

: đây là comment, client bỏ qua

data: dòng thứ nhất
data: dòng thứ hai
id: 42

```

Bốn field có nghĩa [1]:

| Field | Hành vi |
|---|---|
| `data` | Nối giá trị vào buffer dữ liệu, kèm một ký tự LF. Nhiều dòng `data:` liên tiếp được nối bằng LF; LF cuối bị bỏ khi dispatch |
| `event` | Tên loại event. Mặc định là `message` |
| `id` | Đặt *last event ID* (bỏ qua nếu chứa ký tự NULL). Giá trị được giữ cho các event sau cho tới khi gặp `id:` khác |
| `retry` | Thời gian chờ trước khi reconnect, tính bằng ms; chỉ nhận khi giá trị toàn chữ số |

Các quy tắc parse dễ làm sai khi tự viết parser [1]:

- Dòng bắt đầu bằng `:` là **comment** và bị bỏ qua. Đây là cơ chế keep-alive chuẩn.
- Nếu giá trị sau dấu `:` bắt đầu bằng một dấu cách, dấu cách đó bị bỏ đúng một lần. Vì vậy `data:abc` và `data: abc` cho cùng kết quả.
- Tên field phân biệt chữ hoa chữ thường. Field lạ bị bỏ qua.
- **Event chưa có dòng trống kết thúc khi stream đóng sẽ bị bỏ**, không được dispatch. Một parser "tiện tay" vẫn dispatch phần dở dang này sẽ đưa ra JSON cụt.
- Byte order mark ở đầu stream bị bỏ.

Chính vì những quy tắc này mà cách `buffer.split("\n\n")` rồi lấy dòng bắt đầu bằng `data: ` hay gặp trong các bài hướng dẫn sẽ hỏng với CRLF, với event có nhiều dòng `data:` và với dòng comment. Hãy dùng một parser đã được kiểm thử như `eventsource-parser` [25], hoặc viết theo đặc tả ([§2.7](#27-sdk-và-thư-viện-helper-stream-vercel-ai-sdk)).

### Reconnect và `Last-Event-ID`

Client `EventSource` của trình duyệt tự kết nối lại khi gặp lỗi mạng, sau một khoảng thời gian mà đặc tả mô tả là "implementation-defined, probably in the region of a few seconds" [1]. Khi kết nối lại, nếu đã nhận một `id`, client gửi header `Last-Event-ID`. Đặc tả không định nghĩa server phải replay thế nào: server tự lưu event và tự quyết định gửi lại từ đâu. Ba trường hợp khiến client dừng hẳn, không reconnect: status khác 200, `Content-Type` khác `text/event-stream`, hoặc server trả `204 No Content` [1].

Không provider LLM nào trong tài liệu này dùng cơ chế `Last-Event-ID` của SSE để resume. OpenAI có cơ chế resume riêng cho background response, Claude dùng request tiếp nối. Chi tiết ở [§3.6](#36-lỗi-giữa-chừng-đứt-kết-nối-lỗi-sau-khi-đã-gửi-một-phần-retry-và-resume).

### Keep-alive

Đặc tả ghi rằng một số proxy cũ đóng kết nối HTTP sau một khoảng timeout ngắn, và khuyên gửi một dòng comment "every 15 seconds or so" [1]. Claude gửi event `ping` xen kẽ trong stream [9]. vLLM có flag `--sse-keep-alive-interval` để gửi comment khi stream đang chờ (xếp hàng, prefill, hoặc giữa hai token), nhưng mặc định là 0, tức tắt [18]. Nếu backend của bạn chuyển tiếp stream, hãy tự gửi heartbeat ([§3.2](#32-stream-qua-proxy-load-balancer-và-cdn)).

### Giới hạn của `EventSource`

`EventSource` chỉ gửi được request `GET`: không có body, không có header tuỳ ý, ngoài cờ `withCredentials` [1]. API chat thường cần `POST` với body là lịch sử hội thoại, và cần header xác thực. Vì vậy client web hầu như luôn đọc SSE bằng `fetch` + `ReadableStream` + một parser, không dùng `EventSource`.

Trên HTTP/1.1, trình duyệt chỉ cho mở tối đa 6 kết nối đồng thời tới một domain. Giới hạn này tính theo trình duyệt cộng domain, nên người dùng mở nhiều tab sẽ hết slot. Chrome và Firefox đã đánh dấu vấn đề này là "Won't fix". Trên HTTP/2, số stream đồng thời được thương lượng, mặc định là 100 [2].

## 1.3 SSE, WebSocket, HTTP chunked và gRPC streaming

| | SSE | WebSocket | HTTP chunked / NDJSON | gRPC streaming |
|---|---|---|---|---|
| Chiều dữ liệu | Server → client | Hai chiều | Server → client | Một chiều hoặc hai chiều |
| Nền tảng | HTTP thường | Nâng cấp từ HTTP (`101 Switching Protocols`) | HTTP thường | HTTP/2 |
| Reconnect và ID | Có trong đặc tả (`EventSource`) | Tự làm | Tự làm | Tự làm |
| Trình duyệt | `fetch` hoặc `EventSource` | `WebSocket` | `fetch` | Cần gRPC-Web |
| Qua proxy/CDN | Dễ bị buffer nếu cấu hình sai | Proxy phải hỗ trợ Upgrade | Dễ bị buffer | Cần HTTP/2 xuyên suốt |
| Dùng ở đâu trong LLM API | OpenAI, Claude, Gemini, Grok, vLLM | API realtime/voice (ví dụ OpenAI Realtime, Gemini Live) | Ollama native API | Hiếm gặp ở API công khai |

Chat với LLM chỉ cần dữ liệu đi một chiều: client gửi một request, server trả về một dòng token. SSE hợp với kiểu này vì nó là HTTP thường, đi qua được hầu hết hạ tầng, debug được bằng `curl -N`, và mọi provider đều dùng. WebSocket đáng chọn khi cần trao đổi hai chiều liên tục: voice, người dùng chen ngang khi model đang nói, hoặc nhiều luồng trên một kết nối. Để huỷ một câu trả lời đang stream thì không cần WebSocket: đóng kết nối là đủ ([§3.7](#37-người-dùng-huỷ-dừng-request-phía-provider-để-không-tốn-token)).

Ollama trả một chuỗi object JSON, mỗi object một dòng, chứ không dùng SSE [22]. Định dạng này đơn giản hơn, nhưng không có tên event, không có comment để giữ kết nối, và parser phải tự tách dòng.

## 1.4 Từ token đến event: delta, content block, usage

Một response của LLM không chỉ có chữ. Nó có thể gồm phần suy luận, nhiều đoạn văn bản, nhiều tool call song song, trích dẫn và thống kê usage. Vì vậy provider không gửi "từng token" mà gửi các event có cấu trúc. Các provider chia cấu trúc này khác nhau:

| Provider | Đơn vị cấp cao | Đơn vị con | Khoá để gom |
|---|---|---|---|
| OpenAI Responses | Output item (`message`, `function_call`, `reasoning`, tool phía server) | Content part | `item_id`, `output_index`, `content_index` |
| OpenAI Chat Completions | Choice | `delta` (content, tool_calls) | `choices[].index`, `tool_calls[].index` |
| Claude | Content block (`text`, `tool_use`, `thinking`, ...) | Delta theo loại block | `index` của block |
| Gemini Interactions | Step | Delta | Step |
| Gemini `streamGenerateContent` | Mỗi event là một `GenerateContentResponse` chứa phần mới | Part | Thứ tự event |

Không phải provider nào cũng gửi event mở và đóng khối. Claude, OpenAI Responses và Gemini Interactions có event riêng cho việc bắt đầu và kết thúc mỗi khối. Chat Completions chỉ gửi delta rồi `finish_reason`. Gemini `streamGenerateContent` gửi các `GenerateContentResponse` từng phần và kết thúc bằng `finishReason`. Ollama gửi từng dòng JSON đến dòng `"done": true`. Vì vậy, nên viết một adapter cho mỗi provider để chuyển event của nó về một khuôn nội bộ: **bắt đầu khối → nhiều delta → kết thúc khối**, lặp lại cho từng khối, rồi một **event kết thúc response** mang lý do dừng và usage. Với provider không có event mở và đóng khối, adapter tự sinh chúng (ví dụ khi gặp part hoặc index mới, và khi gặp `finishReason`). Phần còn lại của code chỉ làm việc với khuôn nội bộ này:

```python
blocks: dict[str, Block] = {}         # khoá: index / item_id tuỳ provider

for ev in normalize(provider_stream): # adapter theo provider: chuyển event về 4 loại nội bộ
    match ev.kind:
        case "block_start":  blocks[ev.key] = Block(type=ev.block_type, meta=ev.meta)
        case "block_delta":  blocks[ev.key].append(ev.delta)   # text, JSON dở, thinking
        case "block_stop":   blocks[ev.key].close()           # chỉ lúc này mới parse JSON / thực thi tool
        case "response_end": finish(ev.stop_reason, ev.usage); break
else:
    raise StreamTruncated()           # hết stream mà không có response_end: coi là lỗi
```

Ba quy tắc rút ra từ khuôn này:

- **Gom theo khoá, không theo thứ tự nhận.** Khi model gọi nhiều tool song song, delta của các tool có thể đến xen kẽ nhau. Với Chat Completions, `id` của tool call thường chỉ có ở delta đầu tiên, nên khoá đúng là `index` ([Tool Use §3.7](../tool-use/#37-streaming-khi-có-tool-call)).
- **Không xử lý nghiệp vụ trên khối chưa đóng.** JSON dở dang có thể vô tình hợp lệ: `{"order_id": "DH-2"` thêm `}` là JSON đúng cú pháp nhưng sai mã đơn.
- **Hết stream mà không có event kết thúc là lỗi**, không phải thành công. Nhiều bug "câu trả lời dừng giữa câu" mà không ai biết đều đến từ việc bỏ qua quy tắc này ([§3.6](#36-lỗi-giữa-chừng-đứt-kết-nối-lỗi-sau-khi-đã-gửi-một-phần-retry-và-resume)).

## 1.5 Stream không thay đổi chi phí, nhưng thay đổi cách tính usage

Giá token khi stream và khi không stream là như nhau. Điểm khác là usage chỉ đến **ở cuối stream**, hoặc rải rác qua nhiều event:

| Provider | Usage khi stream |
|---|---|
| OpenAI Responses | Trong `response.completed`, field `response.usage` [6] |
| OpenAI Chat Completions | Chỉ khi đặt `stream_options: {"include_usage": true}`. Một chunk bổ sung trước `data: [DONE]` chứa usage tổng, với `choices` là mảng rỗng; các chunk khác có `usage: null` [5] |
| Claude | `message_start` có `input_tokens`; `message_delta` có `output_tokens`. Số trong `message_delta` là **cộng dồn**, không phải phần tăng thêm [9] |
| Gemini | `usageMetadata` trong `GenerateContentResponse`, gồm `promptTokenCount`, `cachedContentTokenCount`, `candidatesTokenCount`, `thoughtsTokenCount`, `toolUsePromptTokenCount`, `totalTokenCount` [13] |
| Gemini Interactions | Trong `interaction.completed`, field `usage` với `total_input_tokens`, `total_output_tokens`, `total_tokens` [14] |
| vLLM | Như Chat Completions; `continuous_usage_stats` để có usage ở mọi chunk; `--enable-force-include-usage` để luôn gửi usage [17][18] |
| Ollama | Object cuối có `done: true` kèm `prompt_eval_count`, `eval_count` và các thời gian xử lý [22] |

Hệ quả cho việc đo chi phí:

**Stream bị cắt thì có thể mất usage, nhưng hoá đơn thì không mất.** OpenAI ghi rõ: "If the stream is interrupted, you may not receive the final usage chunk" [5]. Trung tâm hỗ trợ của Claude ghi rằng nếu client ngắt kết nối hoặc timeout giữa một request đang chạy bình thường, request đó vẫn bị tính tiền [11]. Code đo chi phí chỉ dựa vào event cuối sẽ đếm thiếu đúng những request có vấn đề.

**Cộng dồn hay phần tăng thêm phải đọc kỹ.** Với Claude, lấy giá trị `output_tokens` của `message_delta` cuối cùng nhận được, đừng cộng các giá trị lại. Với vLLM khi bật `continuous_usage_stats`, mỗi chunk mang usage tính đến thời điểm đó.

**Ghi lại cờ thay vì đoán im lặng.** Khi thiếu usage, lưu `usage_status = "missing"`, input token đã biết (Claude gửi input token ngay trong `message_start`), và một ước lượng output từ số ký tự đã nhận. Đối chiếu định kỳ với số liệu billing của provider ([§3.10](#310-observability-đo-ttft-tokensgiây-tỉ-lệ-stream-bị-cắt)).

---

# Phần 2 — Streaming theo provider

Phần này dùng để tra cứu khi viết client cho từng provider. Mỗi mục ghi event, cách kết thúc, usage, lỗi giữa stream, và những hành vi không có trong ví dụ quickstart.

## 2.1 Bức tranh chung

| Provider | Transport | Event kết thúc đáng tin | Resume khi đứt | Ghi chú |
|---|---|---|---|---|
| OpenAI Responses | SSE, mỗi event có `type` và `sequence_number` | `response.completed` / `.incomplete` / `.failed` | Có, chỉ với `background: true` | Khoảng 50 loại event trong SDK |
| OpenAI Chat Completions | SSE, `chat.completion.chunk` | `data: [DONE]` sau chunk có `finish_reason` | Không | Usage phải bật riêng |
| Claude | SSE, có cả dòng `event:` và field `type` | `message_stop` | Không; tạo request tiếp nối | Lỗi có thể đến sau HTTP 200 |
| Gemini `streamGenerateContent` | SSE khi thêm `?alt=sse` | `finishReason` khác rỗng, rồi đóng kết nối | Không | Google gọi generateContent là API legacy, khuyên dùng Interactions cho dự án mới [15] |
| Gemini Interactions | SSE, `event:` khớp `event_type` | `interaction.completed` | Chưa xác minh | Đổi tên event từ 05/2026 [14] |
| Grok | SSE, `chat.completion.chunk` | `data: [DONE]` | Không | Có `reasoning_content` trong delta [16] |
| vLLM | SSE tương thích OpenAI | `data: [DONE]` | Không | Huỷ khi client ngắt kết nối |
| Ollama native | Dòng JSON | `"done": true` | Không | Lỗi là dòng `{"error": ...}` |

## 2.2 OpenAI: Chat Completions và Responses API

### Responses API: các nhóm event

Python SDK liệt kê các event mà trang hướng dẫn streaming nêu ra [3], còn SDK TypeScript bản master có khoảng 50 loại [6]. Gom theo nhóm:

| Nhóm | Event |
|---|---|
| Vòng đời response | `response.created`, `response.queued`, `response.in_progress`, `response.completed`, `response.incomplete`, `response.failed`, `error` |
| Output item | `response.output_item.added`, `response.output_item.done`, `response.content_part.added`, `response.content_part.done` |
| Văn bản | `response.output_text.delta`, `response.output_text.done`, `response.output_text.annotation.added`, `response.refusal.delta`, `response.refusal.done` |
| Function call | `response.function_call_arguments.delta`, `response.function_call_arguments.done`, `response.custom_tool_call_input.delta`, `.done` |
| Reasoning | `response.reasoning_summary_part.added`, `.done`, `response.reasoning_summary_text.delta`, `.done`, `response.reasoning_text.delta`, `.done` |
| Tool phía server | `response.web_search_call.in_progress` / `.searching` / `.completed`, `response.file_search_call.*`, `response.code_interpreter_call.*`, `response.mcp_call.*`, `response.mcp_list_tools.*`, `response.image_generation_call.*` (có `.partial_image`) |

Những chi tiết đáng chú ý:

- **Mọi event có `sequence_number` tăng dần.** Đây là con trỏ để resume background response ([§3.6](#36-lỗi-giữa-chừng-đứt-kết-nối-lỗi-sau-khi-đã-gửi-một-phần-retry-và-resume)).
- **Lấy reasoning item từ `response.output_item.done`, không lấy từ `.added`.** Docstring của SDK ghi rằng với reasoning item, `encrypted_content` có thể chưa đầy đủ khi item còn đang chạy, và chỉ nên dùng item từ event `.done` khi gửi lại ở request sau [6]. Code nào dựng lịch sử từ `.added` sẽ gặp lỗi khó hiểu ở lượt tiếp theo.
- **`response.function_call_arguments.done` có sẵn chuỗi `arguments` đầy đủ**, kèm `item_id` và `output_index` [6]. Nếu chỉ cần arguments hoàn chỉnh, đọc từ event này, không cần tự nối delta. Tên function nằm trong item của `response.output_item.added`.
- **Event delta có field `obfuscation`.** Đây là chuỗi đệm ngẫu nhiên, mặc định bật. Đặt `stream_options: {"include_obfuscation": false}` nếu muốn tiết kiệm băng thông và bạn tin cậy đường truyền giữa server và OpenAI [6].
- **Thứ tự event giữa các item không được tài liệu cam kết.** Trang streaming chỉ ghi rằng có event phát một lần và có event phát nhiều lần [3]. Đừng viết code giả định `response.output_text.delta` luôn đến sau mọi function call hay ngược lại.

```python
from openai import OpenAI

client = OpenAI()
stream = client.responses.create(
    model="gpt-6.1-sol",
    input=[{"role": "user", "content": "Tóm tắt hợp đồng sau: ..."}],
    tools=tools,
    stream=True,
)

final = None
for ev in stream:
    match ev.type:
        case "response.output_text.delta":
            send_to_client(ev.delta)
        case "response.output_item.added" if ev.item.type == "function_call":
            show_status(f"Đang gọi {ev.item.name}…")      # tên tool có ngay từ đầu
        case "response.function_call_arguments.done":
            pending_calls[ev.item_id] = ev.arguments       # chuỗi JSON đầy đủ
        case "response.completed":
            final = ev.response                            # gồm output, usage
        case "response.incomplete":
            final = ev.response                            # xem incomplete_details.reason
        case "response.failed" | "error":
            raise StreamFailed(ev)                         # SDK không tự raise, xem dưới

if final is None:
    raise StreamTruncated()                                # hết stream mà không có event kết thúc
```

**SDK Python không raise với event `error` của Responses.** Trong `_streaming.py` của `openai-python`, stream chỉ raise `APIError` khi dữ liệu của chunk có field `error` [7]. Event `error` của Responses có dạng `{"type": "error", "code": ..., "message": ..., "sequence_number": ...}`, không có field `error` lồng bên trong, nên nó được yield như mọi event khác. `response.failed` cũng vậy. Nếu vòng lặp không có nhánh cho hai event này, lỗi trôi qua im lặng.

### Chat Completions

- `stream: true`; mỗi chunk là một `chat.completion.chunk` với `choices[].delta`. Chunk đầu thường có `role`, các chunk sau có `content` hoặc `tool_calls` [5].
- `finish_reason` là `null` ở các chunk giữa, có giá trị (`stop`, `length`, `tool_calls`, `content_filter`) ở chunk cuối của choice đó.
- Stream kết thúc bằng `data: [DONE]`.
- Usage: đặt `stream_options: {"include_usage": true}` [5]. Không bật thì không có usage khi stream.
- Tool call: delta đầu tiên của mỗi tool call có `id` và `function.name`, các delta sau chỉ có mảnh `function.arguments`. Gom theo `tool_calls[i].index`.
- Lỗi giữa stream: SDK Python raise `APIError` khi chunk chứa field `error` [7]. Exception này không mang HTTP status, vì status 200 đã gửi đi từ trước.

```python
stream = client.chat.completions.create(
    model="gpt-6.1-sol",
    messages=messages,
    stream=True,
    stream_options={"include_usage": True},
)
usage = None
for chunk in stream:
    if chunk.usage:                       # chunk cuối, choices rỗng
        usage = chunk.usage
        continue
    delta = chunk.choices[0].delta
    if delta.content:
        send_to_client(delta.content)
```

### Background mode: stream có thể nối lại

Responses API có chế độ `background: true`: response chạy bất đồng bộ phía OpenAI, không phụ thuộc vào kết nối HTTP [4]. Kết hợp với stream:

1. Tạo response với cả `background: true` và `stream: true`. Lưu `sequence_number` của event cuối cùng đã xử lý.
2. Khi mất kết nối, gọi `GET /v1/responses/{id}?stream=true&starting_after=<sequence_number>` để nhận tiếp các event sau con trỏ đó. Keep-alive event không được tính là vị trí resume.
3. Chỉ resume được response được tạo với `stream: true`.
4. Huỷ: `POST /v1/responses/{id}/cancel`. Gọi nhiều lần vẫn an toàn và trả về response cuối. Với response *không* chạy nền, cách huỷ là đóng kết nối [4].

Đánh đổi: background response **không tự dừng khi client ngắt kết nối**. Đó là mục đích của chế độ này, nên nút "Dừng" phải gọi endpoint cancel ([§3.7](#37-người-dùng-huỷ-dừng-request-phía-provider-để-không-tốn-token)). Về lưu trữ: với project dùng Zero Data Retention, background request vẫn được lưu tạm khoảng 10 phút để polling và resume hoạt động [4].

## 2.3 Claude (Anthropic)

### Luồng event

Mỗi event SSE của Claude có cả dòng `event:` lẫn field `type` cùng tên trong `data` [9]. Thứ tự:

1. `message_start`: một `Message` với `content` rỗng. `usage` đã có `input_tokens`.
2. Lặp lại cho từng content block: `content_block_start` → một hoặc nhiều `content_block_delta` → `content_block_stop`. `index` của block khớp với vị trí của nó trong mảng `content` cuối cùng.
3. Một hoặc nhiều `message_delta`, mang `stop_reason`, `stop_sequence` và `usage` cộng dồn.
4. `message_stop`.

`ping` có thể xuất hiện ở bất kỳ đâu. Tài liệu yêu cầu xử lý êm các loại event chưa biết, vì Anthropic có thể thêm event mới theo chính sách versioning [9]. Một trường hợp đã có: khi bật *server-side fallback*, ở ranh giới giữa hai model sẽ có một block `fallback`, gồm `content_block_start` và `content_block_stop` mà không có delta nào ở giữa [9]. Parser nào giả định mọi block đều có ít nhất một delta sẽ hỏng ở chỗ này.

### Các loại delta

| `delta.type` | Field | Block | Ghi chú |
|---|---|---|---|
| `text_delta` | `text` | `text` | |
| `input_json_delta` | `partial_json` | `tool_use` | Block bắt đầu với `input: {}`. Gom các chuỗi rồi parse ở `content_block_stop` [9] |
| `thinking_delta` | `thinking` | `thinking` | Khi `display: "omitted"`, block chỉ nhận một `thinking_delta` rỗng rồi tới signature [9] |
| `signature_delta` | `signature` | `thinking` | Đến ngay trước `content_block_stop`; phải lưu nguyên để gửi lại ở lượt sau |

### Lỗi trong stream

Tài liệu Claude ghi rằng API có thể gửi lỗi ngay trong event stream. Ví dụ, lúc tải cao bạn có thể nhận `overloaded_error`, lỗi mà ở chế độ không stream sẽ tương ứng với HTTP 529 [9]:

```text
event: error
data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
```

Event này đến **sau khi** status 200 đã được gửi. Hậu quả nằm ở cách SDK đóng gói lỗi. Trong `anthropic-sdk-python` bản hiện tại, khi gặp `event: error`, stream gọi `_make_status_error(..., response=self.response)` với chính response HTTP 200 [12]. Hàm ánh xạ lỗi tra bảng theo `(status_code, error_code)`: 529 ứng với `OverloadedError`, 429 với `RateLimitError`, mọi status từ 500 trở lên với `InternalServerError`. Status 200 không có trong bảng nên rơi về `APIStatusError` cơ sở [12]. Kết quả là `overloaded_error` giữa stream thành `APIStatusError` với `status_code == 200`, không phải `OverloadedError`. Issue #1258 mô tả đúng hệ quả này: `FallbackModel` của pydantic-ai kiểm tra `status_code >= 500` nên không chuyển sang model dự phòng [37]. SDK TypeScript có báo cáo tương tự, nhưng với `status: undefined` (issue #1262) [36].

Vì vậy, phân loại lỗi stream theo loại lỗi trong body, không theo status hay class của exception:

```python
import anthropic

RETRYABLE = {"overloaded_error", "rate_limit_error", "api_error"}

def error_type(e: anthropic.APIStatusError) -> str | None:
    body = e.body if isinstance(e.body, dict) else {}
    return (body.get("error") or {}).get("type")

async def answer(client: anthropic.AsyncAnthropic, messages: list[dict]) -> None:
    try:
        async with client.messages.stream(model="claude-sonnet-5-5", max_tokens=4096,
                                          messages=messages) as stream:
            async for text in stream.text_stream:
                await send_to_client(text)
            final = await stream.get_final_message()  # stop_reason, usage đầy đủ
    except anthropic.APIStatusError as e:
        kind = error_type(e)                          # đúng cho cả lỗi trước stream và giữa stream
        if kind in RETRYABLE and nothing_sent_yet():
            ...                                       # retry hoặc fallback (§3.6)
        else:
            ...                                       # báo người dùng, giữ phần đã nhận
    except anthropic.APIConnectionError:
        ...                                           # đứt mạng: không có event kết thúc
```

`send_to_client` và `nothing_sent_yet` là phần của ứng dụng. Gọi bằng `await answer(...)` trong một hàm async, hoặc `asyncio.run(...)` từ code đồng bộ.

Nhiều dự án tự parse SSE còn mắc một lỗi khác: không có nhánh xử lý event `error`. Khi đó vòng đọc dừng lúc kết nối đóng, không có `message_stop`, nhưng code vẫn coi như đã xong. Người dùng thấy câu trả lời dừng giữa chừng không rõ lý do, còn circuit breaker không biết provider đang quá tải [38].

### Helper của SDK

`client.messages.stream(...)` (Python) giữ kết nối SSE, cung cấp `text_stream` cho văn bản, và `get_final_message()` để gom mọi event thành một `Message` hoàn chỉnh. Tương đương ở TypeScript là `stream.finalMessage()` [9]. Khi không cần hiển thị từng phần nhưng request có `max_tokens` lớn, dùng helper này thay vì `messages.create()` không stream.

### Khôi phục khi stream bị cắt

Claude không có API resume. Tài liệu hướng dẫn tạo một request tiếp nối, và cách làm khác nhau theo thế hệ model [9]:

- **Claude 4.5 trở về trước:** đặt phần đã nhận vào đầu một assistant message mới (prefill) để model viết tiếp.
- **Claude 4.6 trở đi:** prefill không còn được hỗ trợ ([Prompt Engineering §2.3](../prompt-engineering/#23-kiểm-soát-độ-dài-giọng-văn-và-format)). Thay vào đó, thêm một user message chứa phần đã nhận và yêu cầu viết tiếp, ví dụ: "Your previous response was interrupted and ended with [previous_response]. Continue from where you left off."
- Block `tool_use` và `thinking` không khôi phục từng phần được. Chỉ tiếp nối được từ text block gần nhất.

Request tiếp nối là một request mới: toàn bộ prompt và phần đã sinh được tính lại như input. Prompt caching giảm được chi phí phần prefix nếu prefix không đổi ([Multi-turn Conversation](../multi-turn-conversation/)).

### Fine-grained tool streaming

Mặc định, Claude buffer và kiểm tra từng giá trị tham số của tool trước khi gửi xuống. Đặt `eager_input_streaming: true` trên từng tool để nhận các mảnh JSON ngay khi model sinh ra. Tài liệu cảnh báo rằng khi đó có thể nhận JSON dở hoặc không hợp lệ, nhất là khi response dừng vì `max_tokens` [10]. Chi tiết và cách trả `INVALID_JSON` cho model có ở [Tool Use §2.3](../tool-use/#23-claude-anthropic).

## 2.4 Google Gemini

Gemini có hai API sinh nội dung. Google hiện gọi `generateContent` là API legacy và khuyên dùng Interactions API cho dự án mới [15].

### `streamGenerateContent`

```bash
curl -N "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse" \
  -H "x-goog-api-key: $GEMINI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"role":"user","parts":[{"text":"Tóm tắt hợp đồng sau: ..."}]}]}'
```

- **Phải có `?alt=sse`** để nhận SSE. Không có tham số này, theo một gateway bên thứ ba, endpoint trả một mảng JSON được stream dần thay vì SSE; tài liệu Google mà chúng tôi đọc không mô tả định dạng đó (chưa xác minh) [13].
- Mỗi `data:` là một `GenerateContentResponse` đầy đủ cấu trúc, nhưng chỉ chứa phần nội dung mới [13].
- `finishReason` để trống khi model còn đang sinh [13]. Ngoài `STOP` và `MAX_TOKENS`, enum còn nhiều giá trị khác như `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `MALFORMED_FUNCTION_CALL`, `MISSING_THOUGHT_SIGNATURE`, `TOO_MANY_TOOL_CALLS`. Client chỉ xử lý `STOP` sẽ hiểu sai những lần stream bị chặn giữa chừng.
- Khi dùng grounding, `groundingChunks` của mỗi event chỉ chứa phần mới so với event trước, trong khi chỉ số tham chiếu được đánh liên tục qua toàn stream. Client phải gom theo thứ tự [13].
- **Đừng dựa vào `data: [DONE]`.** Các nguồn mâu thuẫn nhau về việc Gemini có gửi sentinel này hay không, và tài liệu chính thức không ghi. Coi stream xong khi đã nhận `finishReason` khác rỗng và kết nối đóng.

### Interactions API

Interactions API stream qua SSE với tên event nằm cả ở dòng `event:` lẫn field `event_type` trong payload. Đợt breaking change tháng 05/2026 đã đổi tên toàn bộ event [14]:

| Trước 05/2026 | Hiện tại |
|---|---|
| `interaction.start` | `interaction.created` |
| `content.start` | `step.start` |
| `content.delta` | `step.delta` |
| `content.stop` | `step.stop` |
| `interaction.complete` | `interaction.completed` |

`interaction.status_update` giữ nguyên. Usage nằm trong object `interaction` của event `interaction.completed`. Với function call, `step.start` mang tên function, còn `step.delta` stream arguments dưới dạng chuỗi JSON từng phần (`arguments_delta`) mà client phải tự nối [14].

Đây là một ví dụ thực tế cho quy tắc ở [§3.1](#31-kiến-trúc-chuyển-tiếp-provider--backend--client): nếu frontend đọc thẳng event của provider, đợt đổi tên này làm hỏng cả frontend. Nếu backend chuẩn hoá event về một schema riêng, chỉ cần sửa một adapter.

## 2.5 xAI Grok

Theo tài liệu streaming của xAI [16]:

- Bật bằng `"stream": true`. Mọi model sinh text đều hỗ trợ stream, model sinh ảnh thì không.
- Mỗi event là `data:` chứa một `chat.completion.chunk`. Text nằm ở `choices[0].delta.content`. Reasoning model có thêm `delta.reasoning_content`.
- Kết thúc bằng `data: [DONE]`.
- Ví dụ trong docs có `usage` ở chunk; trang không nhắc tới `stream_options.include_usage`. Hành vi này khác với OpenAI, nên hãy kiểm tra bằng request thật trước khi viết code đếm usage.
- Với reasoning model, xAI khuyến nghị đặt timeout client cao, ví dụ 3.600 giây.

Cách stream tool call và lỗi giữa stream của Grok không được mô tả trong trang này (chưa xác minh).

## 2.6 Self-hosted: vLLM, SGLang, Ollama

### vLLM

Server của vLLM tương thích OpenAI, nên stream theo định dạng Chat Completions và Responses. Các điểm riêng:

- `stream_options.include_usage` hoạt động như OpenAI. `stream_options.continuous_usage_stats: true` gắn usage tích luỹ vào **mọi** chunk, rất hữu ích khi cần biết đã sinh bao nhiêu token tại thời điểm stream bị cắt [17]. Flag `--enable-force-include-usage` của server bắt buộc gửi usage ở mọi request [18].
- **Per-request metrics** (thời gian xếp hàng, prefill, decode) được gắn vào chunk usage cuối, nên chỉ có khi `include_usage` được bật hoặc server bật force include usage [19].
- `--sse-keep-alive-interval` gửi comment SSE khi stream đang chờ (xếp hàng, prefill, hoặc giữa hai token). Mặc định là 0, tức tắt [18]. Nếu vLLM đứng sau proxy có read timeout và request có thể xếp hàng lâu, hãy bật.
- **Client ngắt kết nối thì request bị huỷ.** Server của vLLM bọc route bằng `with_cancellation`: handler chạy song song với một listener chờ `http.disconnect`, và khi client đi mất thì handler bị huỷ, kéo theo việc engine abort request [20]. Hai lỗ hổng đã được báo cáo:
  - Thêm middleware dạng `BaseHTTPMiddleware` (ví dụ qua `--middleware`) từng khiến `request.is_disconnected()` luôn trả `False`, nên request không bị huỷ (issue #10087) [20].
  - Một bug report tháng 10/2026 trên bản nightly ghi rằng với request **không stream**, client ngắt kết nối nhưng engine vẫn sinh đến `max_tokens` (issue #61918, đang được thảo luận) [21].

  Bài học: với self-hosted, việc huỷ khi ngắt kết nối phải được **kiểm tra bằng test** sau mỗi lần nâng cấp hoặc thêm middleware, không thể chỉ giả định.

### SGLang

SGLang cũng có endpoint tương thích OpenAI. Đợt biên soạn này chưa đối chiếu được tài liệu về usage khi stream và hành vi huỷ của SGLang. Hãy kiểm tra trực tiếp trên phiên bản bạn chạy.

### Ollama

API native của Ollama (`/api/generate`, `/api/chat`) trả về một chuỗi object JSON, mỗi object một dòng [22]. Đặt `"stream": false` để nhận một object duy nhất. Object cuối có `"done": true`, `done_reason` và các thống kê: `total_duration`, `load_duration`, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`. Vì không phải SSE, proxy nào chỉ nhận diện `text/event-stream` để tắt buffer sẽ không nhận ra loại stream này.

## 2.7 SDK và thư viện: helper stream, Vercel AI SDK

### Helper trong SDK chính thức

| SDK | Gom thành response hoàn chỉnh | Chỉ lấy text |
|---|---|---|
| `openai` (Python) | Event `response.completed` mang `response` đầy đủ; Chat Completions có `client.chat.completions.stream(...)` | Lọc `response.output_text.delta` |
| `anthropic` (Python) | `stream.get_final_message()` | `stream.text_stream` |
| `@anthropic-ai/sdk` (TS) | `stream.finalMessage()` | Event `text` của helper |
| `google-genai` | Tự gom từ `generate_content_stream` | `chunk.text` |

Dùng helper khi có thể. Helper xử lý giúp phần gom block theo index, usage cộng dồn và các event lạ. Nhưng đừng giả định helper xử lý lỗi giữa stream đúng như bạn muốn: hai SDK ở trên đều có cách đóng gói lỗi giữa stream khác với lỗi HTTP thường ([§2.2](#22-openai-chat-completions-và-responses-api), [§2.3](#23-claude-anthropic)).

### Parser SSE cho phía client

`eventsource-parser` là parser thuần, không kèm fetch: tạo bằng `createParser({ onEvent, onRetry, onComment, onError })`, đưa từng chunk byte đã decode vào `feed()`, và gọi `reset({ consume: true })` khi stream kết thúc để xử lý phần còn lại [25]. Phía trình duyệt nên dùng nó cùng `fetch` và `ReadableStream`:

```ts
import { createParser, type EventSourceMessage } from "eventsource-parser";

export async function streamChat(body: unknown, signal: AbortSignal,
                                 onEvent: (e: EventSourceMessage) => void) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,                                    // nút "Dừng" gọi controller.abort()
  });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

  let finished = false;
  let failure = null as { code: string; retryable: boolean } | null; // `as`: TS không theo dõi gán trong callback
  const parser = createParser({
    onEvent(e) {
      if (e.event === "done") finished = true;               // event kết thúc do backend định nghĩa (§3.1)
      else if (e.event === "error") failure = JSON.parse(e.data); // lỗi đã được backend phân loại
      onEvent(e);
    },
  });
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    parser.feed(value);
  }
  parser.reset({ consume: true });                            // xử lý phần còn lại trong buffer
  if (failure) throw new StreamError(failure.code, failure.retryable);
  if (!finished) throw new StreamError("stream_truncated", true); // đóng kết nối mà không có done hay error
}

export class StreamError extends Error {
  constructor(public code: string, public retryable: boolean) { super(code); }
}
```

### Vercel AI SDK

AI SDK định nghĩa một giao thức riêng giữa server và `useChat` gọi là *UI message stream*. Giao thức này chạy trên SSE, mỗi chunk là một dòng `data:` JSON, kết thúc bằng `data: [DONE]`. Backend tự viết (không dùng `toUIMessageStream`) phải gửi header `x-vercel-ai-ui-message-stream: v1` [23]. Các loại chunk gồm:

- Vòng đời: `start`, `start-step`, `finish-step`, `finish`, `abort`, `error` (có `errorText`).
- Văn bản và suy luận: `text-start`, `text-delta`, `text-end`, `reasoning-start`, `reasoning-delta`, `reasoning-end`.
- Tool: `tool-input-start`, `tool-input-delta`, `tool-input-available`, `tool-output-available`, `tool-output-denied`, `tool-approval-request`.
- Nguồn và dữ liệu: `source-url`, `source-document`, `file`, `data-*` cho dữ liệu tuỳ biến.

Về huỷ: truyền `abortSignal: req.signal` vào `streamText` để đóng tab là huỷ luôn request gọi model. `onAbort` nhận các step đã hoàn thành để lưu kết quả dở dang [24]. AI SDK cũng hỗ trợ *resumable streams* qua package `resumable-stream` và Redis, kèm một đánh đổi quan trọng được thảo luận ở [§3.6](#36-lỗi-giữa-chừng-đứt-kết-nối-lỗi-sau-khi-đã-gửi-một-phần-retry-và-resume).

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Kiến trúc chuyển tiếp: provider → backend → client

Gần như mọi sản phẩm đều đặt một backend giữa trình duyệt và provider. Backend giữ API key, xác thực người dùng, áp quota, ghi log usage, và tuỳ trường hợp chạy tool hoặc moderation. Khi stream, backend trở thành một mắt xích trong chuỗi chuyển tiếp, và mắt xích này có thể làm hỏng stream theo ba cách: đọc hết rồi mới gửi, không chuyển lỗi xuống client, và không huỷ upstream khi client đi mất.

### Chuẩn hoá event thay vì chuyển nguyên

Có hai lựa chọn:

- **Chuyển nguyên** (pass-through) stream của provider xuống client. Đơn giản và giữ độ trễ thấp nhất, nhưng frontend phải biết event model của từng provider. Đổi provider, hoặc provider đổi tên event như Gemini Interactions làm vào 05/2026 [14], thì frontend phải sửa theo.
- **Chuẩn hoá** về một schema riêng của bạn, chẳng hạn `text`, `status`, `tool`, `error`, `done`. Thêm một tầng mapping, đổi lại frontend độc lập với provider, và backend kiểm soát được lỗi lẫn event kết thúc.

Với sản phẩm có nhiều model hoặc có fallback, chuẩn hoá gần như là bắt buộc. Một schema tối thiểu:

```text
event: status   data: {"phase": "thinking" | "tool", "label": "Đang tra cứu đơn hàng"}
event: text     data: {"delta": "..."}
event: error    data: {"code": "overloaded" | "rate_limited" | "truncated" | ..., "retryable": true}
event: done     data: {"stop_reason": "end" | "max_tokens" | "refusal", "message_id": "..."}
```

Điểm mấu chốt là **`done` là event bắt buộc do chính bạn phát ra**, và chỉ phát sau khi đã nhận event kết thúc hợp lệ từ provider. Client không nhận được `done` thì coi là lỗi.

### Không gom response để log

Một lỗi hay gặp: middleware log đọc hết body (`await upstream.text()`) để ghi lại nội dung, vô tình biến stream thành response thường. Nếu cần log hay đếm, tách nhánh stream (dùng `tee()` hoặc `TransformStream` ở JavaScript, hoặc ghi vào buffer trong vòng lặp ở Python) và ghi log sau khi stream kết thúc.

### Ví dụ: FastAPI chuyển tiếp Claude thành SSE chuẩn hoá

```python
import asyncio, contextlib, json
import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
client = anthropic.AsyncAnthropic()

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

_END = object()

async def with_heartbeat(agen, interval: float = 15.0, deadline: float = 600.0):
    """Chèn ': ping' khi upstream im lặng quá `interval` giây; đóng upstream khi vượt `deadline` giây."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=32)
    timed_out = False

    async def pump():                          # task riêng: deadline chạy kể cả khi client ngừng đọc
        nonlocal timed_out
        cm = asyncio.timeout(deadline)
        try:
            async with cm:
                async for item in agen:
                    await queue.put(item)      # client chậm thì chờ ở đây, vẫn trong deadline
        except TimeoutError:
            if not cm.expired():               # TimeoutError của SDK/upstream, không phải deadline
                raise
            timed_out = True
        finally:
            await agen.aclose()                # đóng generator upstream (và stream tới provider)
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(_END)

    task = asyncio.create_task(pump())
    try:
        while not (task.done() and queue.empty()):
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            if item is _END:
                break
            yield item
        await task                             # ném lại lỗi của upstream nếu có
        if timed_out:
            yield sse("error", {"code": "deadline_exceeded", "retryable": False})
    finally:
        if not task.done():                    # client ngắt kết nối: huỷ pump, pump tự đóng upstream
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

async def relay(messages):
    try:
        async with client.messages.stream(model="claude-sonnet-5-5", max_tokens=8192,
                                          messages=messages) as stream:
            async for text in stream.text_stream:
                yield sse("text", {"delta": text})
            final = await stream.get_final_message()
        record_usage(final.usage)
        yield sse("done", {"stop_reason": final.stop_reason, "message_id": final.id})
    except anthropic.APIStatusError as e:
        kind = (e.body or {}).get("error", {}).get("type") if isinstance(e.body, dict) else None
        yield sse("error", {"code": kind or "api_error",
                            "retryable": kind in {"overloaded_error", "rate_limit_error", "api_error"}})
    except anthropic.APIConnectionError:
        yield sse("error", {"code": "upstream_disconnected", "retryable": True})
    # Client đóng kết nối: Starlette huỷ generator, `async with` đóng stream tới Claude.

@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    return StreamingResponse(
        with_heartbeat(relay(body["messages"])),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
```

`with_heartbeat` đọc upstream trong một task riêng (`pump`) và chuyển dữ liệu qua một hàng đợi có giới hạn. Tách như vậy vì generator chỉ chạy khi có người đọc: nếu client ngừng đọc (mạng chậm, tab bị treo), một generator đang dừng ở `yield` sẽ không bao giờ kiểm tra deadline, và kết nối tới provider bị giữ mãi. Với task riêng, `asyncio.timeout` (Python 3.11 trở lên) vẫn chạy khi `pump` đang chờ hàng đợi đầy, và `finally` đóng generator upstream trong mọi trường hợp: hết stream, lỗi, vượt `deadline`, hoặc client ngắt kết nối (task bị huỷ). `deadline` là giới hạn tổng cho cả request, độc lập với heartbeat. Không có nó, một upstream treo vẫn nhận ping đều đặn và giữ kết nối mãi. Chọn giá trị theo request dài nhất bạn chấp nhận.

Ví dụ này dựa vào việc Starlette huỷ generator khi client ngắt kết nối. Như trường hợp của vLLM ở [§2.6](#26-self-hosted-vllm-sglang-ollama) cho thấy, middleware kiểu `BaseHTTPMiddleware` có thể làm tín hiệu ngắt kết nối không đến được [20]. Hãy viết một test: mở stream, đóng client sau vài giây, rồi kiểm tra request tới provider đã bị đóng.

### Ví dụ: Next.js Route Handler chuyển nguyên

```ts
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const { input } = await req.json();
  const upstream = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model: "gpt-6.1-sol", input, stream: true }),
    signal: req.signal,                        // client đóng tab → huỷ request tới OpenAI
  });
  if (!upstream.ok || !upstream.body) {
    return new Response(await upstream.text(), { status: upstream.status });
  }
  return new Response(upstream.body, {         // không đọc body: giữ backpressure và độ trễ
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}
```

Lỗi trước khi stream bắt đầu (401, 429, 400) được trả về với status gốc. Lỗi giữa stream đi xuống client nguyên dạng của provider, nên frontend vẫn phải xử lý event `error` và `response.failed`.

### Gắn generation với kết nối, hay tách ra

Hai ví dụ trên **gắn** việc sinh nội dung với kết nối HTTP: client đi thì generation dừng. Cách này đơn giản và tiết kiệm, nhưng người dùng chuyển từ Wi-Fi sang 4G là mất câu trả lời đang sinh.

Cách thứ hai là **tách** hai thứ ra. Một worker gọi provider và ghi từng event vào nơi lưu trữ (Redis stream, database) kèm ID tăng dần. Endpoint SSE chỉ đọc từ nơi lưu trữ đó, gửi `id:` cho mỗi event, và dùng `Last-Event-ID` để gửi tiếp khi client kết nối lại. Background mode của OpenAI và resumable streams của AI SDK đều theo mô hình này [4][24].

| | Gắn với kết nối | Tách generation |
|---|---|---|
| Mất mạng giữa chừng | Mất câu trả lời; phải gửi lại | Kết nối lại và nhận tiếp |
| Đóng tab | Huỷ, ngừng tốn token | Vẫn chạy đến hết, trừ khi gọi stop |
| Nút "Dừng" | Đóng kết nối là đủ | Cần endpoint stop riêng |
| Hạ tầng | Không cần thêm | Queue/worker và nơi lưu event |

AI SDK nêu rõ đánh đổi này: khi bật resumable streams, `stop()` phía client chỉ đóng kết nối HTTP hiện tại, không huỷ generation bên dưới. Muốn có nút dừng thật thì phải viết endpoint riêng: lưu phần đã sinh, huỷ worker, xoá ID stream đang hoạt động [24]. Chọn cách tách khi câu trả lời dài và đắt (agent chạy nhiều phút, deep research), và người dùng hay đổi mạng. Chọn cách gắn cho chat thông thường.

## 3.2 Stream qua proxy, load balancer và CDN

Mỗi tầng giữa provider và trình duyệt có thể làm một trong ba việc: gom buffer (chữ đến cùng lúc ở cuối), cắt kết nối khi im lặng quá lâu (stream dừng sau đúng một con số giây), hoặc nén body (gzip phải chờ đủ dữ liệu). Cách debug nhanh nhất là gọi lần lượt qua từng tầng bằng `curl -N` và so thời điểm byte đầu tiên đến:

```bash
# -N: tắt buffer của curl; so origin với domain production
curl -N -sS -X POST https://origin.internal/api/chat -H 'Content-Type: application/json' -d @req.json | ts '%.s'
curl -N -sS -X POST https://app.example.com/api/chat -H 'Content-Type: application/json' \
     -H 'Accept-Encoding: gzip' -d @req.json | ts '%.s'
```

Nếu origin trả từng dòng đều đặn mà domain production trả cả khối ở cuối, có một tầng ở giữa đang buffer.

### nginx

Theo tài liệu `ngx_http_proxy_module` [27]:

- `proxy_buffering` **mặc định bật**. Khi bật, nginx đọc response của upstream vào buffer; khi tắt, response được chuyển cho client "synchronously, immediately as it is received".
- Upstream có thể tắt buffer cho từng response bằng header `X-Accel-Buffering: no`. Header này bị bỏ qua nếu cấu hình có `proxy_ignore_headers X-Accel-Buffering`.
- `proxy_read_timeout` mặc định **60 giây**, tính giữa hai lần đọc liên tiếp chứ không phải cho toàn bộ response. Stream dài vẫn ổn, miễn là không có khoảng im lặng nào dài quá 60 giây.

```nginx
location /api/chat {
    proxy_pass http://app_upstream;
    proxy_http_version 1.1;
    proxy_buffering off;        # hoặc giữ on và để backend gửi X-Accel-Buffering: no
    proxy_cache off;
    proxy_read_timeout 300s;    # lớn hơn khoảng im lặng dài nhất (kể cả lúc reasoning)
    gzip off;
}
```

Một bài đo trên hệ thống thật ghi nhận 195/214 lượt (91%) đến trình duyệt thành một khối khi `proxy_buffering` bật, và median thời gian tới event đầu tiên giảm từ 3,4 giây xuống 0,46 giây sau khi tắt buffer [42]. Đây là số liệu của một hệ thống, nhưng cho thấy mức độ ảnh hưởng.

### Nén gzip và brotli

Thuật toán nén cần gom đủ một cửa sổ dữ liệu mới xuất ra được, nên middleware nén biến stream thành từng khối lớn, hoặc thành một khối duy nhất ở cuối. README của middleware `compression` cho Express ghi rằng module này "does not work out of the box with server-sent events" [40]. Next.js chạy bằng `next start` có bật nén mặc định, và đã có nhiều báo cáo về SSE bị giữ lại vì gzip. Hành vi này khác nhau giữa các phiên bản Next.js, nên hãy kiểm tra bằng `curl -H 'Accept-Encoding: gzip'` trên chính phiên bản bạn chạy [40]. Cách xử lý: loại route stream ra khỏi middleware nén, và gửi `Cache-Control: no-transform` để báo các tầng trung gian không biến đổi body.

### Cloudflare

Tài liệu giới hạn kết nối của Cloudflare liệt kê [28]:

| Giới hạn | Giá trị | Mã lỗi khi chạm | Cấu hình được |
|---|---|---|---|
| Proxy Read Timeout | 125 giây | 524 | Có, với zone Enterprise |
| Proxy Write Timeout | 30 giây | 524 | Không |
| Proxy Idle Timeout | 900 giây | 520 | Không |

Trang lỗi 524 mô tả trường hợp origin "did not provide an HTTP response before the default 125 seconds", và gợi ý dùng polling hoặc đi vòng qua proxy cho request chạy lâu [29]. Nhiều bài viết cũ hơn ghi con số 100 giây. Tài liệu hiện tại không nói rõ timeout này tính giữa hai lần đọc hay cho toàn response, và không nhắc tới SSE. Vì vậy heartbeat chỉ là biện pháp *giả định* đúng, không phải bảo đảm:

- Gửi header response và một comment đầu tiên ngay khi nhận request, trước khi gọi provider. Trường hợp "chưa có HTTP response" trong 125 giây là lúc origin chưa gửi gì cả, và heartbeat gửi sau khi đã có response không giúp được trường hợp đó.
- Sau đó gửi heartbeat đều đặn (15–30 giây), và kiểm tra bằng một request thật đi qua đúng đường Cloudflare của bạn, với một khoảng im lặng phía provider dài hơn 125 giây.
- Với tác vụ có thể chạy rất lâu (agent chạy tool nhiều phút, deep research), đừng phụ thuộc vào kết nối giữ mở: chuyển sang mô hình tách generation ở [§3.1](#31-kiến-trúc-chuyển-tiếp-provider--backend--client) kèm polling hoặc resume, hoặc đặt route đó ở subdomain không qua proxy (DNS-only). Zone Enterprise tăng được timeout 524 lên tối đa 6.000 giây [29].

### AWS Application Load Balancer

`idle_timeout.timeout_seconds` mặc định **60 giây**: hết khoảng này mà không có dữ liệu đi qua theo chiều nào thì ALB đóng kết nối [30]. Tăng giá trị này, hoặc giữ heartbeat dưới 60 giây.

### AWS API Gateway (REST API)

Mặc định, API Gateway chờ toàn bộ response (`BUFFERED`). Muốn stream, đặt response transfer mode thành `STREAM` [31]. Các giới hạn:

- Chỉ áp dụng cho integration `HTTP_PROXY` và `AWS_PROXY` (gồm Lambda proxy), chỉ với REST API.
- Stream tối đa **15 phút**.
- Idle timeout: 5 phút với endpoint regional hoặc private, **30 giây** với endpoint edge-optimized.
- Không dùng được cùng endpoint caching, content encoding và VTL response transformation.
- 10 MB đầu không giới hạn băng thông, sau đó giới hạn 2 MB/s.
- Khi client đóng kết nối, Lambda có thể vẫn chạy đến hết timeout của nó.

### Vercel và serverless

Với Fluid compute, thời gian chạy tối đa của function là 300 giây ở gói Hobby và mặc định 300, tối đa 800 giây ở Pro và Enterprise. Thời gian stream response được tính vào giới hạn này [32]. Edge runtime phải bắt đầu gửi response trong 25 giây, sau đó được stream tối đa 300 giây [32]. Agent hay tác vụ reasoning dài có thể vượt các con số này. Khi đó cần job nền và một endpoint stream đọc từ nơi lưu trữ, như mô hình tách generation ở [§3.1](#31-kiến-trúc-chuyển-tiếp-provider--backend--client).

### HTTP/2

Trên HTTP/1.1, trình duyệt chỉ mở tối đa 6 kết nối mỗi domain [2]. Ứng dụng mở nhiều luồng stream (nhiều tab, hoặc nhiều panel stream song song) nên dùng HTTP/2 ở edge, hoặc gộp các luồng vào một kết nối, chẳng hạn qua SharedWorker như đặc tả gợi ý [1].

### Header và heartbeat

| Header / hành vi | Mục đích |
|---|---|
| `Content-Type: text/event-stream; charset=utf-8` | Bắt buộc với `EventSource`; nhiều proxy dựa vào content type để quyết định có buffer hay không |
| `Cache-Control: no-cache, no-transform` | Không cache; báo tầng trung gian không biến đổi (nén) body |
| `X-Accel-Buffering: no` | Tắt buffer của nginx cho response này [27] |
| Comment `: ping` mỗi 15–30 giây | Giữ kết nối dưới ngưỡng idle của nginx (60 giây), ALB (60 giây), Cloudflare (125 giây) |
| Gửi header ngay (flush) trước khi gọi provider | Một số tầng tính timeout tới byte đầu tiên; header sớm giúp kết nối được coi là đang hoạt động |

## 3.3 Stream kèm tool call

Cách gom tool call theo từng provider có ở [Tool Use §3.7](../tool-use/#37-streaming-khi-có-tool-call). Ở góc nhìn streaming, có thêm ba điểm:

**Thời gian chạy tool là khoảng im lặng của stream.** Trong vòng lặp agent, sau khi model trả tool call, backend chạy tool rồi gọi model lần nữa. Trong lúc đó không có token nào đi xuống client. Hãy phát event `status` ("Đang tra cứu đơn hàng…") ngay khi biết tên tool. Tên tool có từ event bắt đầu: `response.output_item.added` với OpenAI, `content_block_start` với Claude, `step.start` với Gemini Interactions. Duy trì heartbeat trong suốt thời gian tool chạy.

**Tool phía server có event tiến trình riêng.** OpenAI phát `response.web_search_call.searching`, `response.code_interpreter_call.interpreting` [6]. Dùng các event này để cập nhật trạng thái thay vì để màn hình đứng yên.

**Đứt kết nối giữa tool call thì bỏ tool call dở dang.** Không lưu nó vào lịch sử, không thực thi. Tool call không có kết quả trong lịch sử sẽ làm request sau trả `400` ([Tool Use §3.4](../tool-use/#34-lỗi-từ-tool-trả-về-cho-model-hay-xử-lý-trong-code)).

## 3.4 Stream kèm structured output: parse JSON chưa hoàn chỉnh

Structured output được stream như văn bản thường: các mảnh của một chuỗi JSON. Kỹ thuật parse JSON dở dang (sửa JSON thiếu dấu đóng, thư viện partial parser, BAML `@stream.done`) có ở [Structured Output §3.4](../structured-output/#34-streaming-partial-json). Quy tắc khi kết hợp với streaming:

- **Partial object chỉ để hiển thị**, không dùng để ra quyết định, gọi API hay ghi database. Vercel AI SDK ghi rõ partial object (`DeepPartial`) không thể validate theo schema. Ngược lại, `elementStream` của `Output.array()` chỉ phát phần tử đã hoàn chỉnh và đã qua validate [23].
- **Validate một lần, sau event kết thúc.** Nếu stream dừng với `max_tokens` / `length` / `incomplete`, JSON chắc chắn không đầy đủ; xử lý như lỗi, đừng cố sửa.
- **Nếu UI không cần xem trước**, đừng stream JSON. Dùng helper gom kết quả ([§2.7](#27-sdk-và-thư-viện-helper-stream-vercel-ai-sdk)) để giữ được lợi thế tránh timeout mà không phải xử lý JSON dở dang.

## 3.5 Stream kèm reasoning/thinking

Reasoning model sinh phần suy luận trước câu trả lời. Khi stream, phần này đến qua các event riêng, hoặc không đến chút nào:

| Provider | Event của phần suy luận | Lưu ý |
|---|---|---|
| OpenAI Responses | `response.reasoning_summary_text.delta` (tóm tắt), `response.reasoning_text.delta` | Reasoning item để gửi lại lấy từ `response.output_item.done` [6] |
| Claude | `thinking_delta`, rồi `signature_delta` trước `content_block_stop` | Khi `display: "omitted"`, không có văn bản suy luận nhưng vẫn có block và signature [9] |
| Gemini | Part có `thought: true` khi bật `includeThoughts`; số token ở `thoughtsTokenCount` | [13] |
| Grok | `delta.reasoning_content` | [16] |

Hệ quả thiết kế:

- **Kết nối có thể im lặng lâu.** Nếu không có tóm tắt suy luận được stream, client không nhận token nào trong suốt thời gian model suy nghĩ. Heartbeat của backend và event `status: thinking` là bắt buộc.
- **Signature và encrypted content phải được lưu nguyên.** Đây là lý do không nên dựng lịch sử hội thoại từ văn bản đã hiển thị cho người dùng. Hãy lưu message mà SDK gom được (`get_final_message()`, `response.output`) ([Tool Use §1.6](../tool-use/#16-reasoning-model-trạng-thái-ẩn-đi-kèm-tool-call)).
- **Phần suy luận hiển thị cho người dùng nên được trình bày khác câu trả lời**, chẳng hạn thu gọn hoặc chữ mờ. Tóm tắt suy luận không phải câu trả lời, và có thể không phản ánh đầy đủ những gì model thực sự đã làm. Chi tiết ở [Reasoning Models](../reasoning-models/).

## 3.6 Lỗi giữa chừng: đứt kết nối, lỗi sau khi đã gửi một phần, retry và resume

### Ba loại lỗi

| Loại | Dấu hiệu | Ví dụ |
|---|---|---|
| 1. Lỗi trước stream | HTTP status khác 200 | 400 sai tham số, 401, 429, 529 trước khi có byte nào |
| 2. Lỗi trong stream | Status 200, sau đó có event lỗi | Claude `event: error` (`overloaded_error`); OpenAI `error`, `response.failed`; chunk có `error`; Ollama dòng `{"error": ...}` |
| 3. Đứt kết nối | Kết nối đóng, không có event lỗi, cũng không có event kết thúc | Proxy timeout, mạng di động, server restart |

Loại 1 được SDK xử lý tốt, kể cả retry tự động. Loại 2 và 3 mới là nơi phát sinh bug. Các báo cáo thực tế cho thấy ba kiểu sai lặp lại:

- **Bỏ qua event lỗi**, coi việc kết nối đóng là thành công. Câu trả lời cụt được lưu như câu trả lời đầy đủ [38].
- **Gán sai loại lỗi.** Lỗi quá tải giữa stream mang status 200 hoặc không có status, nên logic retry và fallback dựa trên status bỏ sót nó [36][37].
- **Không phát event kết thúc khi upstream lỗi.** Router hoặc gateway ở giữa nuốt lỗi; client tưởng còn đang chờ, hoặc retry và tạo thêm chi phí. Có trường hợp văn bản lỗi `[error: overloaded_error] Overloaded` còn lọt vào nội dung assistant [39].

### Quy tắc xử lý

1. **Stream chỉ hoàn thành khi nhận được event kết thúc hợp lệ** (`message_stop`, `response.completed`, `[DONE]` sau `finish_reason`, `interaction.completed`, `done: true`). Mọi trường hợp khác là lỗi.
2. **Phân loại lỗi theo loại trong body**, không theo HTTP status: `overloaded_error` → quá tải, `rate_limit_error` → rate limit, `api_error` → lỗi server. Ba loại này retry được. Lỗi validation thì không.
3. **Chỉ retry im lặng khi chưa gửi gì xuống client.** Nếu đã có token hiển thị, retry im lặng sẽ in ra một câu trả lời mới chồng lên câu cũ, hoặc lặp lại đoạn đầu. Thay vào đó, hiển thị "Câu trả lời bị gián đoạn", giữ phần đã nhận, và cho người dùng chọn thử lại hoặc viết tiếp.
4. **Không lưu tool call dở dang và block thinking thiếu signature** vào lịch sử.
5. **Retry có giới hạn và jitter.** Mỗi lần retry là một request mới, tốn tiền như request đầu. Khi provider quá tải, retry dồn dập làm tình hình tệ hơn. Chiến lược retry và fallback giữa model có ở [Model Routing](../model-routing/).

### Ba cơ chế "resume" khác nhau

Chữ "resume" được dùng cho ba cơ chế không liên quan gì đến nhau:

| Cơ chế | Ai giữ trạng thái | Chi phí | Khi nào dùng |
|---|---|---|---|
| SSE `Last-Event-ID` | Server của bạn tự lưu event và tự replay | Chỉ chi phí lưu trữ | Mô hình tách generation ([§3.1](#31-kiến-trúc-chuyển-tiếp-provider--backend--client)) |
| OpenAI background + `starting_after` | OpenAI lưu event của response | Không sinh lại | Response dài trên OpenAI, cần chịu được mất kết nối [4] |
| Claude continuation | Không ai giữ; tạo request mới chứa phần đã nhận | Tính lại input, sinh tiếp phần còn lại | Khi stream Claude bị cắt và muốn giữ phần đã có [9] |

Continuation của Claude không cho kết quả giống hệt việc stream không bị cắt. Model viết tiếp dựa trên phần đã có, nên chỗ nối có thể lệch về giọng văn hoặc lặp ý. Với nội dung quan trọng, sinh lại từ đầu thường an toàn hơn.

## 3.7 Người dùng huỷ: dừng request phía provider để không tốn token

Huỷ một câu trả lời đang stream là một chuỗi gồm bốn mắt xích, và chỉ cần đứt một mắt xích là việc huỷ không có tác dụng:

```text
[1] Client đóng kết nối   →  [2] Proxy chuyển việc đóng   →  [3] Backend phát hiện   →  [4] Request tới provider bị đóng
    AbortController           (nginx, CDN, LB)                 req.signal / disconnect      SDK abort / thoát context
```

- **[1]** `controller.abort()` trên `fetch`. Chỉ ngừng đọc mà không abort thì kết nối vẫn mở.
- **[3]** Node/Next.js: `req.signal`. Python/Starlette: generator bị huỷ, hoặc kiểm tra `await request.is_disconnected()`; middleware có thể làm hỏng tín hiệu này [20].
- **[4]** Truyền signal vào SDK hoặc `fetch` gọi provider, hoặc thoát khối `with ... stream` để SDK đóng kết nối. AI SDK: `abortSignal: req.signal` [24].

### Provider có dừng sinh và dừng tính tiền không

Đây là điểm tài liệu chính thức nói ít nhất:

- **Claude:** trung tâm hỗ trợ ghi rằng request bị client ngắt kết nối hoặc timeout giữa chừng, nếu đang chạy bình thường, **vẫn bị tính tiền** [11]. Tài liệu không nói rõ số output token được tính là phần đã sinh đến lúc ngắt hay nhiều hơn.
- **OpenAI:** với response không chạy nền, cách huỷ là đóng kết nối [4]. Background response chạy tiếp cho đến khi bạn gọi endpoint cancel.
- **OpenRouter** (gateway, không phải provider gốc) ghi trong trang hỗ trợ rằng tại thời điểm viết, OpenAI, Azure, Anthropic, DeepSeek, Together, DeepInfra và xAI hỗ trợ huỷ (dừng xử lý và dừng tính tiền phần chưa sinh), còn AWS Bedrock, Google AI Studio, Groq, Mistral và Perplexity thì không. Với nhóm sau, ngắt kết nối không dừng việc sinh ở upstream, và có thể bị tính tiền cả completion [41].
- **Azure OpenAI:** khi content filter dừng stream, token của prompt và completion sinh trước thời điểm lọc vẫn bị tính [34].
- **Self-hosted:** vLLM huỷ request khi client ngắt kết nối ([§2.6](#26-self-hosted-vllm-sglang-ollama)). Ở đây "chi phí" là GPU bị chiếm, và một request không được huỷ sẽ giữ KV cache, làm giảm thông lượng của mọi request khác.

Kết luận thực tế: **luôn nối đủ bốn mắt xích**, vì ít nhất bạn ngừng trả tiền cho phần chưa sinh với các provider hỗ trợ huỷ. **Luôn đặt `max_tokens`**, vì đó là trần chi phí duy nhất được bảo đảm cho mỗi request. Và **đo bằng usage thật**: huỷ một số request ở các thời điểm khác nhau, rồi so số token bị tính với số token đã nhận.

Một nguồn chi phí ít ai để ý: người dùng bấm "Dừng" rồi "Tạo lại". Nếu request đầu chưa thực sự bị huỷ ở upstream, bạn trả tiền cho cả hai câu trả lời.

## 3.8 Phía UI: render markdown từng phần, cuộn, hiển thị trạng thái

**Markdown chưa đóng.** Giữa chừng stream, văn bản thường chứa cú pháp chưa đóng: `**đậm` thiếu dấu đóng, link chưa có `)`, code block mới có ba dấu backtick mở. Renderer markdown thông thường sẽ hiển thị toàn bộ phần sau thành code, hoặc làm layout nhảy liên tục. Streamdown là một thư viện React thay cho `react-markdown`, được viết để tự phát hiện và tạm đóng các cú pháp chưa kết thúc [26]. Nếu tự viết, ít nhất hãy tạm đóng code fence khi số dấu ``` là số lẻ.

**Hiệu năng render.** Parse lại toàn bộ markdown ở mỗi delta tốn CPU theo bình phương độ dài câu trả lời. Gom delta lại và chỉ render một lần mỗi frame (`requestAnimationFrame`) hoặc mỗi 30–50 ms. Mắt người không phân biệt được 60 lần cập nhật mỗi giây với 20 lần.

**Cuộn.** Chỉ tự cuộn xuống cuối khi người dùng đang ở cuối. Nếu họ đã cuộn lên để đọc lại, đừng kéo họ xuống.

**Trạng thái.** Một câu trả lời đang stream có ít nhất năm trạng thái mà UI nên phân biệt:

| Trạng thái | Dấu hiệu | Hiển thị |
|---|---|---|
| Đang chờ | Đã gửi, chưa có event | Spinner |
| Đang suy nghĩ / chạy tool | Event `status` | Nhãn cụ thể ("Đang tra cứu đơn hàng") |
| Đang viết | Có delta văn bản | Văn bản và con trỏ nhấp nháy |
| Xong | Event `done` | Bật nút sao chép, tạo lại |
| Bị gián đoạn | Lỗi, hoặc đóng kết nối mà không có `done` | Giữ phần đã nhận, báo rõ, nút thử lại |

**An toàn.** Output của model là dữ liệu không tin cậy. Không render HTML thô từ model. Chỉ biến URL thành link khi scheme là `https`, và cẩn thận với ảnh markdown trỏ tới domain ngoài: chúng có thể được dùng để rò rỉ dữ liệu qua URL ([Bảo mật LLM](../llm-security/)).

## 3.9 Moderation và guardrail trên output đang stream

Stream và kiểm duyệt output kéo nhau theo hai hướng ngược nhau: stream muốn hiển thị ngay từng token, kiểm duyệt muốn xem đủ nội dung rồi mới quyết định. Phần đã hiển thị cho người dùng không thu hồi được.

**OpenAI** ghi trong hướng dẫn streaming rằng stream làm việc kiểm duyệt khó hơn, rằng moderation score (nếu yêu cầu) chỉ đến sau khi toàn bộ output đã sinh xong và không nằm trong các delta, và rằng điều này có thể ảnh hưởng đến việc được phê duyệt sử dụng [3].

**Azure OpenAI** mô tả rõ nhất hai chế độ [34]:

- **Mặc định:** content filter chạy trên từng đoạn output, nên nội dung được buffer và trả theo từng khối, không phải từng token.
- **Asynchronous Filter:** trả token ngay, không buffer; tín hiệu lọc đến trễ hơn. Microsoft ghi rằng ở chế độ này, một phần nội dung lẽ ra bị lọc ngay có thể đã hiển thị cho người dùng.
- Khi filter chặn, response vẫn có status 200, kèm `finish_reason: "content_filter"`.
- Trong thử nghiệm của Microsoft với blocklist, một từ bị chặn nằm ở 15–30 từ cuối của stream có thể được trả về với `finish_reason: "stop"`, trong khi cùng câu trả lời ở chế độ không stream bị chặn.

**Gemini** dừng stream với `finishReason` như `SAFETY`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII` [13]. Client phải xử lý các giá trị này thay vì chỉ chờ `STOP`.

Các lựa chọn thiết kế, xếp từ rủi ro thấp đến cao:

1. **Không stream** với nội dung bắt buộc phải kiểm duyệt trước (ví dụ nội dung gửi đi thay mặt doanh nghiệp).
2. **Buffer theo câu hoặc theo đoạn** ở backend: chạy classifier trên từng đoạn hoàn chỉnh rồi mới gửi xuống. TTFT tăng lên bằng thời gian sinh một đoạn cộng thời gian chạy classifier.
3. **Stream ngay và kiểm duyệt song song**, có cơ chế "rút lại": ẩn đoạn vi phạm, thay bằng thông báo, và ghi log. Hợp với chatbot nội bộ, nơi rủi ro một vài câu lọt qua là chấp nhận được.

Classifier, phát hiện PII và cách xử lý refusal có ở [Guardrails](../guardrails/).

## 3.10 Observability: đo TTFT, tokens/giây, tỉ lệ stream bị cắt

| Metric | Cách đo | Vì sao cần |
|---|---|---|
| TTFT phía client | Từ lúc gửi request đến event nội dung đầu tiên, đo ở trình duyệt/app | Log phía server không thấy được buffer của proxy; một bài viết thực tế tóm lại: "Server-side logs will lie to you" [42] |
| TTFT phía backend | Từ lúc gọi provider đến delta đầu tiên | Tách độ trễ của provider khỏi độ trễ hạ tầng của bạn |
| ITL / tokens mỗi giây | `(e2e − TTFT) / (output_tokens − 1)` [35] | Phát hiện provider chậm đi, hoặc tầng mạng gom buffer |
| Tỉ lệ stream bị cắt | Số stream không có event kết thúc / tổng số stream | Chỉ số sức khoẻ quan trọng nhất của streaming; tăng đột ngột thường do proxy, deploy hoặc provider quá tải |
| Lỗi giữa stream theo loại | Đếm theo `error.type` (`overloaded_error`, ...) | Phân biệt với lỗi trước stream |
| Tỉ lệ huỷ bởi người dùng | Số lần client abort / tổng số stream | Câu trả lời quá dài hoặc sai hướng; nguồn chi phí bị bỏ phí |
| Usage thiếu | Số stream có `usage_status = missing` | Sai số khi đối soát chi phí ([§1.5](#15-stream-không-thay-đổi-chi-phí-nhưng-thay-đổi-cách-tính-usage)) |
| `stop_reason` | Phân phối `end`, `max_tokens`, `refusal`, `content_filter` | `max_tokens` tăng nghĩa là câu trả lời bị cắt |

Về tracing: span của một lần gọi model khi stream phải kết thúc **khi stream kết thúc**, không phải khi hàm trả về iterator. Các thư viện instrument tự động từng có lỗi ở đúng chỗ này: span không được đóng khi stream ném lỗi giữa chừng [43]. Kiểm tra trace của một request bị lỗi giữa stream để chắc rằng span có trạng thái lỗi và thời gian đúng. Định nghĩa thuộc tính theo OpenTelemetry GenAI có ở [LLM Observability](../llm-observability/).

## 3.11 Failure modes thường gặp

| Triệu chứng | Nguyên nhân khả dĩ | Cách kiểm tra / xử lý |
|---|---|---|
| Chữ hiện ra cùng lúc ở cuối | nginx `proxy_buffering` mặc định bật; middleware nén; proxy doanh nghiệp | `curl -N` qua từng tầng; `X-Accel-Buffering: no`; tắt nén cho route stream [27][40] |
| Local chạy tốt, production thì không | Proxy, CDN, gzip chỉ có ở production | So origin với domain production bằng `curl -N` |
| Stream dừng sau đúng khoảng 60 giây | nginx `proxy_read_timeout` hoặc ALB idle timeout mặc định 60 giây | Heartbeat 15–30 giây; tăng timeout [27][30] |
| Lỗi 524 sau khoảng 2 phút | Cloudflare Proxy Read Timeout 125 giây | Heartbeat; gửi header sớm; tách generation [28][29] |
| Stream dừng ở phút thứ 15 | Giới hạn stream của API Gateway REST | Đổi kiến trúc, hoặc chia nhỏ tác vụ [31] |
| Function timeout trên Vercel | Stream dài hơn `maxDuration` | Tăng `maxDuration`, hoặc dùng job nền [32] |
| Chỉ mở được khoảng 6 tab chat | Giới hạn kết nối của HTTP/1.1 | HTTP/2; gộp kết nối [2] |
| Câu trả lời dừng giữa câu, không có lỗi | Không xử lý event `error`; coi đóng kết nối là xong | Bắt buộc có event kết thúc; xử lý `error`, `response.failed` [9][38] |
| Fallback không kích hoạt khi provider quá tải | Lỗi giữa stream mang status 200 hoặc không có status | Phân loại theo `error.type` trong body [36][37] |
| Responses API lỗi nhưng code không raise | SDK yield event `error` / `response.failed` thay vì raise | Thêm nhánh xử lý cho hai event này [7] |
| Lượt sau lỗi do reasoning item thiếu `encrypted_content` | Lấy item từ `output_item.added` | Lấy từ `output_item.done` [6] |
| Không có usage cho một số request | Stream bị cắt trước chunk usage; quên `include_usage` | Ghi `usage_status = missing`; bật `include_usage` [5] |
| Usage Claude bị đếm gấp nhiều lần | Cộng `output_tokens` của mọi `message_delta` | Lấy giá trị cuối cùng; số liệu là cộng dồn [9] |
| Tool chạy với tham số sai | Thực thi trước event kết thúc block | Chỉ thực thi sau `content_block_stop` / `.done` |
| Parser hỏng khi gặp block `fallback` | Giả định block nào cũng có delta | Xử lý block không có delta; bỏ qua event lạ [9] |
| Client Gemini Interactions ngừng nhận chữ sau khi nâng SDK | Tên event đổi từ 05/2026 | Cập nhật adapter `step.*`, `interaction.completed` [14] |
| Chi phí vẫn tăng khi người dùng đóng tab | Không truyền abort signal tới upstream; middleware chặn tín hiệu ngắt kết nối | Nối đủ bốn mắt xích; test ngắt kết nối [20][24] |
| GPU self-hosted bận với request không ai chờ | Engine không abort; request không stream không được huỷ | Kiểm tra `num_requests_running` sau khi ngắt client [21] |
| Code block chưa đóng làm cả đoạn sau thành code | Render markdown chưa hoàn chỉnh | Renderer cho streaming; tạm đóng fence [26] |
| Nội dung vi phạm lọt ra trước khi bị chặn | Lọc output trễ hơn stream | Không stream, hoặc buffer theo đoạn [34] |
| Stream của `streamGenerateContent` trả JSON lạ, không có `data:` | Thiếu `?alt=sse` | Thêm `alt=sse` [13] |

## 3.12 Checklist

### Giao thức và client

- [ ] Parser SSE xử lý đúng CRLF, comment, event nhiều dòng `data:`, và bỏ event chưa có dòng trống kết thúc.
- [ ] Client web dùng `fetch` + `ReadableStream` + `AbortController`, không dùng `EventSource` cho chat.
- [ ] Stream chỉ được coi là xong khi có event kết thúc hợp lệ; thiếu event này thì hiện trạng thái "bị gián đoạn".
- [ ] Có nhánh xử lý cho mọi loại event lỗi: Claude `error`; OpenAI `error`, `response.failed`, `response.incomplete`; Gemini `finishReason` khác `STOP`; Ollama `error`.
- [ ] Event chưa biết được bỏ qua, không làm crash parser.

### Backend

- [ ] Event được chuẩn hoá về schema riêng, với event `done` do backend phát.
- [ ] Không có middleware nào đọc hết body để log hoặc nén response stream.
- [ ] Heartbeat `: ping` mỗi 15–30 giây, kể cả lúc chạy tool hoặc chờ reasoning.
- [ ] Abort signal của client được truyền tới request gọi provider; có test ngắt kết nối.
- [ ] Lỗi được phân loại theo `error.type` trong body; chỉ retry im lặng khi chưa gửi gì xuống client.
- [ ] Mọi request đều có `max_tokens`.
- [ ] Đã chọn rõ giữa gắn generation với kết nối và tách generation; nếu tách thì có endpoint stop.

### Hạ tầng

- [ ] Header: `Content-Type: text/event-stream`, `Cache-Control: no-cache, no-transform`, `X-Accel-Buffering: no`.
- [ ] Timeout của mọi tầng (nginx, LB, CDN, gateway, serverless) lớn hơn khoảng heartbeat và đủ cho thời gian stream dài nhất.
- [ ] Đã kiểm tra bằng `curl -N` qua domain production, có và không có `Accept-Encoding: gzip`.

### Provider

- [ ] **OpenAI:** `include_usage` với Chat Completions; reasoning item lấy từ `output_item.done`; background mode thì lưu `sequence_number` và có nút gọi cancel.
- [ ] **Claude:** usage lấy từ `message_delta` cuối; lưu nguyên thinking block và signature; continuation bằng user message với model từ 4.6.
- [ ] **Gemini:** `?alt=sse`; không chờ `[DONE]`; adapter Interactions dùng tên event mới.
- [ ] **Self-hosted:** bật `--sse-keep-alive-interval` nếu đứng sau proxy; `continuous_usage_stats` khi cần đếm token lúc bị cắt; test huỷ khi ngắt kết nối sau mỗi lần nâng cấp.

### Đo lường

- [ ] TTFT đo ở client và ở backend; ITL; tỉ lệ stream bị cắt; lỗi giữa stream theo loại; tỉ lệ huỷ.
- [ ] Span tracing kết thúc khi stream kết thúc, có trạng thái lỗi khi stream lỗi.
- [ ] Đối soát định kỳ usage ghi nhận với billing của provider, nhất là với request bị huỷ.

---

# Nguồn tham khảo

## Đặc tả

| # | Resource | URL |
|---|---|---|
| 1 | WHATWG HTML Standard §9.2: Server-sent events | [html.spec.whatwg.org/multipage/server-sent-events.html](https://html.spec.whatwg.org/multipage/server-sent-events.html) |
| 2 | MDN: Using server-sent events | [developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) |

## OpenAI

| # | Resource | URL |
|---|---|---|
| 3 | Streaming API responses | [developers.openai.com/api/docs/guides/streaming-responses](https://developers.openai.com/api/docs/guides/streaming-responses) |
| 4 | Background mode | [developers.openai.com/api/docs/guides/background](https://developers.openai.com/api/docs/guides/background) |
| 5 | Chat Completions: create (`stream_options`) | [developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) |
| 6 | SDK types: Responses stream events (`openai-node` `responses.ts`, `openai-python` `types/responses`) | [github.com/openai/openai-node/blob/master/src/resources/responses/responses.ts](https://github.com/openai/openai-node/blob/master/src/resources/responses/responses.ts) |
| 7 | `openai-python` `_streaming.py` | [github.com/openai/openai-python/blob/main/src/openai/_streaming.py](https://github.com/openai/openai-python/blob/main/src/openai/_streaming.py) |
| 8 | Models | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) |

## Claude (Anthropic)

| # | Resource | URL |
|---|---|---|
| 9 | Streaming messages | [platform.claude.com/docs/en/build-with-claude/streaming](https://platform.claude.com/docs/en/build-with-claude/streaming) |
| 10 | Fine-grained tool streaming | [platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming](https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming) |
| 11 | Claude Help Center: How do I pay for my Claude API usage? | [support.claude.com/en/articles/8977456-how-do-i-pay-for-my-claude-api-usage](https://support.claude.com/en/articles/8977456-how-do-i-pay-for-my-claude-api-usage) |
| 12 | `anthropic-sdk-python`: `_streaming.py`, `_error_classes.py` | [github.com/anthropics/anthropic-sdk-python/tree/main/src/anthropic](https://github.com/anthropics/anthropic-sdk-python/tree/main/src/anthropic) |

## Google Gemini

| # | Resource | URL |
|---|---|---|
| 13 | Gemini API reference: Generating content | [ai.google.dev/api/generate-content](https://ai.google.dev/api/generate-content) |
| 14 | Interactions API: Breaking changes migration guide (May 2026) | [ai.google.dev/gemini-api/docs/interactions-breaking-changes-may-2026](https://ai.google.dev/gemini-api/docs/interactions-breaking-changes-may-2026) |
| 15 | Gemini API reference (overview: Interactions được khuyến nghị) | [ai.google.dev/api](https://ai.google.dev/api) |

## xAI

| # | Resource | URL |
|---|---|---|
| 16 | Streaming response | [docs.x.ai/docs/guides/streaming-response](https://docs.x.ai/docs/guides/streaming-response) |

## Self-hosted

| # | Resource | URL |
|---|---|---|
| 17 | vLLM: OpenAI-compatible server (extra parameters) | [docs.vllm.ai/en/latest/serving/openai_compatible_server/](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) |
| 18 | vLLM: `vllm serve` CLI (`--sse-keep-alive-interval`, `--enable-force-include-usage`) | [docs.vllm.ai/en/stable/cli/serve/](https://docs.vllm.ai/en/stable/cli/serve/) |
| 19 | vLLM: Per-request metrics | [docs.vllm.ai/en/latest/features/per_request_metrics/](https://docs.vllm.ai/en/latest/features/per_request_metrics/) |
| 20 | vLLM issue #10087: requests aren't aborted when a middleware is provided | [github.com/vllm-project/vllm/issues/10087](https://github.com/vllm-project/vllm/issues/10087) |
| 21 | vLLM issue #61918: client disconnect not detected on non-streaming chat completions (10/2026) | [github.com/vllm-project/vllm/issues/61918](https://github.com/vllm-project/vllm/issues/61918) |
| 22 | Ollama API | [github.com/ollama/ollama/blob/main/docs/api.md](https://github.com/ollama/ollama/blob/main/docs/api.md) |

## Thư viện

| # | Resource | URL |
|---|---|---|
| 23 | AI SDK: Stream protocols | [ai-sdk.dev/docs/ai-sdk-ui/stream-protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol) |
| 24 | AI SDK: Stopping streams; Chatbot resume streams | [ai-sdk.dev/docs/advanced/stopping-streams](https://ai-sdk.dev/docs/advanced/stopping-streams), [ai-sdk.dev/docs/ai-sdk-ui/chatbot-resume-streams](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-resume-streams) |
| 25 | eventsource-parser | [github.com/rexxars/eventsource-parser](https://github.com/rexxars/eventsource-parser) |
| 26 | Streamdown | [streamdown.ai/docs](https://streamdown.ai/docs) |

## Proxy, CDN, cloud

| # | Resource | URL |
|---|---|---|
| 27 | nginx: `ngx_http_proxy_module` | [nginx.org/en/docs/http/ngx_http_proxy_module.html](https://nginx.org/en/docs/http/ngx_http_proxy_module.html) |
| 28 | Cloudflare: Connection limits | [developers.cloudflare.com/fundamentals/reference/connection-limits/](https://developers.cloudflare.com/fundamentals/reference/connection-limits/) |
| 29 | Cloudflare: Error 524 | [developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/) |
| 30 | AWS: Application Load Balancer attributes | [docs.aws.amazon.com/elasticloadbalancing/latest/application/application-load-balancers.html](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/application-load-balancers.html) |
| 31 | AWS API Gateway: Response transfer mode | [docs.aws.amazon.com/apigateway/latest/developerguide/response-transfer-mode.html](https://docs.aws.amazon.com/apigateway/latest/developerguide/response-transfer-mode.html) |
| 32 | Vercel: Functions limitations | [vercel.com/docs/functions/limitations](https://vercel.com/docs/functions/limitations) |
| 33 | Vercel: Streaming functions | [vercel.com/docs/functions/streaming-functions](https://vercel.com/docs/functions/streaming-functions) |
| 34 | Microsoft Learn: Content streaming in Azure OpenAI | [learn.microsoft.com/en-us/azure/foundry/openai/concepts/content-streaming](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/content-streaming) |

## Metric

| # | Resource | URL |
|---|---|---|
| 35 | NVIDIA NIM: LLM benchmarking metrics | [docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) |

## Issue, báo cáo cộng đồng

| # | Resource | URL |
|---|---|---|
| 36 | anthropic-sdk-typescript #1262: mid-stream `overloaded_error` có `status: undefined` | [github.com/anthropics/anthropic-sdk-typescript/issues/1262](https://github.com/anthropics/anthropic-sdk-typescript/issues/1262) |
| 37 | anthropic-sdk-python #1258: mid-stream error có `status_code=200` | [github.com/anthropics/anthropic-sdk-python/issues/1258](https://github.com/anthropics/anthropic-sdk-python/issues/1258) |
| 38 | use-agent-os/agent-os #2118: mid-stream error của Anthropic bị bỏ qua | [github.com/use-agent-os/agent-os/issues/2118](https://github.com/use-agent-os/agent-os/issues/2118) |
| 39 | vllm-project/semantic-router #3063: lỗi upstream giữa stream không kết thúc stream | [github.com/vllm-project/semantic-router/issues/3063](https://github.com/vllm-project/semantic-router/issues/3063) |
| 40 | vercel/next.js #9965: SSE trong API route bị giữ lại do nén (trích README `compression`) | [github.com/vercel/next.js/issues/9965](https://github.com/vercel/next.js/issues/9965) |
| 41 | OpenRouter Help Center: How do I cancel a streaming request, and which providers stop billing when I do? | [openrouter.zendesk.com/hc/en-us/articles/51691588409883](https://openrouter.zendesk.com/hc/en-us/articles/51691588409883-How-do-I-cancel-a-streaming-request-and-which-providers-stop-billing-when-I-do) |
| 42 | dev.to: nginx `proxy_buffering` broke my LLM SSE stream (số liệu một hệ thống) | [dev.to/ji_ai/nginx-proxybuffering-broke-my-llm-sse-stream-04s-became-34s-1fhb](https://dev.to/ji_ai/nginx-proxybuffering-broke-my-llm-sse-stream-04s-became-34s-1fhb) |
| 43 | traceloop/openllmetry #4643: span không kết thúc khi stream lỗi giữa chừng | [github.com/traceloop/openllmetry/issues/4643](https://github.com/traceloop/openllmetry/issues/4643) |

---

Thông tin API trong tài liệu được ghi nhận tại **tháng 11/2026**, từ tài liệu chính thức, source code của SDK (`openai-python`, `openai-node`, `anthropic-sdk-python`) và repo của dự án. Báo cáo từ issue tracker, bài blog và gateway bên thứ ba được ghi rõ là nguồn cộng đồng.

Các điểm chưa xác minh được trên endpoint thật: định dạng `streamGenerateContent` khi không có `alt=sse`, và việc Gemini có gửi `[DONE]` hay không; dạng lỗi giữa stream của Gemini và Grok; cách Grok stream tool call và trả usage; hành vi usage và huỷ của SGLang; số output token bị tính khi huỷ stream trên Claude và OpenAI (tài liệu chính thức không nêu con số); cơ chế buffer của content filter phía OpenAI. Hành vi nén mặc định của Next.js với Route Handler thay đổi theo phiên bản. Giá trị timeout của CDN và cloud có thể thay đổi; kiểm tra lại trang giới hạn của nhà cung cấp hạ tầng bạn dùng.
