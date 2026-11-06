# Routing và fallback giữa các model

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng tình huống provider chính trả 529/overloaded lúc cao điểm và toàn bộ sản phẩm ngừng trả lời. Nêu hai bài toán: chọn model theo request để tiết kiệm chi phí, và chuyển model khi lỗi để giữ uptime; chỉ đường đọc theo ba phần. -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Công cụ](#phần-2--công-cụ)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Bảng mã lỗi theo provider và nên retry hay fallback; cấu hình retry/backoff khuyến nghị. -->

---

# Phần 1 — Nền tảng

## 1.1 Vì sao một model, một provider là một điểm lỗi

## 1.2 Rate limit: RPM, TPM, và cách provider tính

## 1.3 Retry, backoff và jitter

## 1.4 Routing theo độ khó, theo loại tác vụ, theo chi phí

## 1.5 Model khác nhau trả lời khác nhau: hệ quả khi fallback

---

# Phần 2 — Công cụ

## 2.1 LLM gateway: LiteLLM, Portkey, OpenRouter, Cloudflare AI Gateway

## 2.2 Cloud provider: Bedrock, Vertex AI, Azure OpenAI và cross-region inference

## 2.3 Router dùng classifier hoặc model nhỏ

## 2.4 Priority tier, provisioned throughput và batch

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Thiết kế chuỗi fallback

## 3.2 Giữ prompt, tool và schema tương thích giữa các model

## 3.3 Circuit breaker và health check

## 3.4 Quản lý quota nội bộ theo tenant

## 3.5 Đo và eval khi route sang model khác

## 3.6 Failure modes thường gặp

## 3.7 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
