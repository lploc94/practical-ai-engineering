# Practical AI Engineering

Tài liệu tiếng Việt cho kỹ sư đang đưa LLM và các hệ thống AI vào sản phẩm thật.

Mở API ra gọi thử một model thì không khó. Phần khó đến sau: chọn embedding model nào cho search, vì sao hoá đơn API tăng theo số lượt hội thoại, làm sao để model trả về JSON mà code không phải đoán, và phải làm gì khi hệ thống bị tấn công lúc nửa đêm. Repo này gom những câu hỏi đó thành các tài liệu chuyên sâu, mỗi tài liệu đi từ cơ chế đến quyết định thiết kế và vận hành.

## Có gì trong repo

| Chủ đề | Bạn sẽ nắm được | Cập nhật |
|---|---|---|
| [**Embeddings**](./docs/embeddings/) | Embedding hoạt động thế nào, chọn model và similarity metric, semantic search, RAG, recommendation, chọn vector database, chunking, hybrid search, evaluation và observability | 11/2026 |
| [**Multi-turn Conversation & Prompt Caching**](./docs/multi-turn-conversation/) | Lịch sử hội thoại được gửi lên model ra sao, vì sao cache khớp theo prefix, cấu hình cache và giá theo từng provider, rút gọn context, tìm nguyên nhân cache miss | 11/2026 |
| [**Structured Output**](./docs/structured-output/) | Các mức đảm bảo từ prompt đến constrained decoding, JSON Schema trên OpenAI, Claude, Gemini, Grok và engine self-hosted, validate, retry, streaming, eval | 11/2026 |
| [**Under Attack**](./docs/under-attack/) | Playbook xử lý DDoS/flood cho server vật lý đặt sau Cloudflare: 5 phút đầu, incident response, cấu hình khuyến nghị, cứng hoá origin | 06/2026 |

**Đang soạn:** [Deploying AI Systems](./docs/deploying-ai-systems/) — model serving, batching, autoscaling, observability, cost và capacity planning.

## Cách đọc

Mỗi tài liệu viết để đọc độc lập, không cần đọc theo thứ tự giữa các chủ đề. Bên trong, nội dung thường chia ba phần:

1. **Nền tảng** — cơ chế và lý do đằng sau, đủ để bạn tự suy luận khi gặp trường hợp tài liệu không nhắc tới.
2. **Tra cứu** — tham số, giới hạn và khác biệt giữa các provider hoặc công cụ.
3. **Thiết kế & vận hành** — đánh đổi khi đưa vào production, failure mode thường gặp, checklist.

Nếu đang cần giải quyết một vấn đề cụ thể, bắt đầu từ phần **TL;DR** ở đầu tài liệu hoặc mục **failure modes thường gặp** ở Phần 3.

## Nguyên tắc biên soạn

- **Dựa trên nguồn gốc.** Nội dung đối chiếu với tài liệu chính thức, bài báo nghiên cứu và benchmark; nguồn được liệt kê trong từng tài liệu.
- **Ghi rõ thời điểm.** Giá, giới hạn và tham số API thay đổi nhanh. Mỗi tài liệu ghi tháng cập nhật, và những chỗ chưa xác minh được đánh dấu tại chỗ.
- **Tiếng Việt, giữ thuật ngữ gốc.** Thuật ngữ kỹ thuật và tên field giữ nguyên tiếng Anh để bạn dễ đối chiếu với docs và code.
- **Ưu tiên quyết định hơn định nghĩa.** Mục tiêu là giúp bạn chọn đúng và biết vì sao, không chỉ biết khái niệm.

## Cấu trúc repo

```text
docs/
  embeddings/
  multi-turn-conversation/
  structured-output/
  under-attack/
  deploying-ai-systems/
```

Mỗi chủ đề là một thư mục `kebab-case` dưới `docs/`, tài liệu chính là `README.md` để GitHub hiển thị ngay khi mở thư mục. Khi một chủ đề lớn dần, các phần phụ như glossary, references hay hình ảnh được tách ra trong cùng thư mục.

## Bản quyền

© 2026 Thorn Lee. Bảo lưu mọi quyền.

Mọi người được đọc tài liệu thoải mái. Nếu chưa có văn bản cho phép của tác giả, bạn không được sao chép, đăng lại, chỉnh sửa hay đứng tên nội dung này, dù toàn bộ hay một phần. Xem chi tiết ở [LICENSE](./LICENSE).
