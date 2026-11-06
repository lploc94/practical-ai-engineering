# Streaming: trả từng phần response qua SSE

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng tình huống người dùng nhìn màn hình trắng 20 giây chờ một câu trả lời dài, rồi bật stream và gặp lỗi mới: proxy gom buffer, kết nối đứt giữa chừng, JSON chưa parse được. Nêu streaming cải thiện thời gian chờ cảm nhận chứ không làm tổng thời gian ngắn hơn, và chỉ đường đọc theo ba phần. -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật và tên event giữ nguyên tiếng Anh để tiện đối chiếu API.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Streaming theo provider](#phần-2--streaming-theo-provider)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Khi nào nên và không nên stream; bảng so sánh event model giữa các provider; cấu hình proxy/CDN cần chỉnh để SSE không bị buffer. -->

---

# Phần 1 — Nền tảng

## 1.1 Vì sao cần stream: TTFT, tốc độ sinh token và thời gian chờ cảm nhận

## 1.2 Server-Sent Events hoạt động thế nào

<!-- TODO: Format `event:` / `data:` / `id:` / dòng trống, `Content-Type: text/event-stream`, keep-alive, `retry`, `Last-Event-ID`. -->

## 1.3 SSE, WebSocket, HTTP chunked và gRPC streaming

## 1.4 Từ token đến event: delta, content block, usage

## 1.5 Stream không thay đổi chi phí, nhưng thay đổi cách tính usage

---

# Phần 2 — Streaming theo provider

## 2.1 Bức tranh chung

## 2.2 OpenAI: Chat Completions và Responses API

## 2.3 Claude (Anthropic)

## 2.4 Google Gemini

## 2.5 xAI Grok

## 2.6 Self-hosted: vLLM, SGLang, Ollama

## 2.7 SDK và thư viện: helper stream, Vercel AI SDK

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Kiến trúc chuyển tiếp: provider → backend → client

<!-- TODO: Backend làm proxy stream, giữ API key phía server, biến đổi event trước khi gửi xuống client. -->

## 3.2 Stream qua proxy, load balancer và CDN

<!-- TODO: Nginx `proxy_buffering`, `X-Accel-Buffering`, timeout idle, nén gzip làm gom buffer, Cloudflare, serverless function. -->

## 3.3 Stream kèm tool call

## 3.4 Stream kèm structured output: parse JSON chưa hoàn chỉnh

<!-- TODO: Liên kết với tài liệu Structured Output. -->

## 3.5 Stream kèm reasoning/thinking

## 3.6 Lỗi giữa chừng: đứt kết nối, lỗi sau khi đã gửi một phần, retry và resume

## 3.7 Người dùng huỷ: dừng request phía provider để không tốn token

## 3.8 Phía UI: render markdown từng phần, cuộn, hiển thị trạng thái

## 3.9 Moderation và guardrail trên output đang stream

## 3.10 Observability: đo TTFT, tokens/giây, tỉ lệ stream bị cắt

## 3.11 Failure modes thường gặp

## 3.12 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
