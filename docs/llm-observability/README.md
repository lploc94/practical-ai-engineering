# Observability cho ứng dụng LLM

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng tình huống người dùng báo "bot trả lời sai" mà không ai tìm lại được request, prompt hay context lúc đó. Nêu những gì observability truyền thống (metric, log, trace) bỏ sót với LLM, và chỉ đường đọc theo ba phần. Ghi rõ ranh giới với Deploying AI Systems (hạ tầng serving). -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Công cụ](#phần-2--công-cụ)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Danh sách field cần log cho mỗi lượt gọi model; metric cần dashboard; bảng so sánh công cụ. -->

---

# Phần 1 — Nền tảng

## 1.1 Vì sao log request/response thôi chưa đủ

## 1.2 Trace, span và một lượt gọi LLM

## 1.3 Metric riêng của LLM: token, TTFT, tokens/giây, cache hit, tỉ lệ refusal

## 1.4 Chất lượng là một tín hiệu cần quan sát

## 1.5 OpenTelemetry GenAI semantic conventions

---

# Phần 2 — Công cụ

## 2.1 Langfuse

## 2.2 LangSmith

## 2.3 Arize Phoenix

## 2.4 Helicone và các gateway có logging

## 2.5 Datadog, Grafana và stack observability sẵn có

## 2.6 Tự xây trên OpenTelemetry

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Log gì, lưu bao lâu, che PII thế nào

## 3.2 Trace cho RAG, tool use và agent nhiều bước

## 3.3 Theo dõi chi phí theo user, feature, tenant

## 3.4 Thu feedback người dùng và nối với trace

## 3.5 Phát hiện drift khi provider đổi model

## 3.6 Từ log production về eval dataset

<!-- TODO: Liên kết với tài liệu Evaluation. -->

## 3.7 Alert: ngưỡng nào đáng đánh thức người trực

## 3.8 Failure modes thường gặp

## 3.9 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
