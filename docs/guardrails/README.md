# Guardrails và content moderation

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng tình huống chatbot hỗ trợ khách hàng bị dẫn dắt đưa ra cam kết hoàn tiền hoặc nói về chủ đề ngoài phạm vi. Nêu guardrail là gì, đặt ở đâu trong pipeline, và ranh giới với tài liệu Bảo mật LLM (tấn công có chủ đích); chỉ đường đọc theo ba phần. -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Công cụ](#phần-2--công-cụ)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Bảng loại guardrail → vị trí (input, output, tool) → công cụ; ngưỡng và độ trễ chấp nhận được. -->

---

# Phần 1 — Nền tảng

## 1.1 Guardrail chặn cái gì: an toàn, phạm vi, chính sách, chất lượng

## 1.2 Input guardrail, output guardrail và guardrail trên hành động

## 1.3 Precision và recall: chặn nhầm cũng có giá

## 1.4 Model tự từ chối và guardrail bên ngoài

---

# Phần 2 — Công cụ

## 2.1 Moderation API của provider

## 2.2 Llama Guard, ShieldGemma và classifier open-weight

## 2.3 NeMo Guardrails, Guardrails AI

## 2.4 Phát hiện PII: Presidio và các dịch vụ cloud

## 2.5 Rule, regex và classifier tự huấn luyện

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Đặt guardrail ở đâu để không tăng độ trễ quá nhiều

## 3.2 Guardrail khi output đang stream

<!-- TODO: Liên kết với tài liệu Streaming. -->

## 3.3 Xử lý khi bị chặn: thông báo, fallback, chuyển người

## 3.4 Xử lý refusal của model trong sản phẩm

## 3.5 Đánh giá và tinh chỉnh ngưỡng

## 3.6 Nội dung tiếng Việt: điểm yếu của classifier

## 3.7 Failure modes thường gặp

## 3.8 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
