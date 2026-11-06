# Multimodal: ảnh, PDF và âm thanh

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng tình huống trích xuất dữ liệu từ hoá đơn scan hoặc PDF nhiều bảng, nơi OCR truyền thống và text extraction đều hỏng. Nêu model đọc ảnh thế nào và chi phí token của ảnh; chỉ đường đọc theo ba phần. -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Multimodal theo provider](#phần-2--multimodal-theo-provider)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Bảng giới hạn kích thước, số ảnh, số trang PDF và cách tính token theo provider. -->

---

# Phần 1 — Nền tảng

## 1.1 Model đọc ảnh thế nào: patch, tile và token

## 1.2 Ảnh tốn bao nhiêu token

## 1.3 PDF: ảnh trang, text layer hay cả hai

## 1.4 Âm thanh: speech-to-text, model audio native

## 1.5 Những gì vision model vẫn làm kém

---

# Phần 2 — Multimodal theo provider

## 2.1 OpenAI

## 2.2 Claude (Anthropic)

## 2.3 Google Gemini

## 2.4 Model open-weight: Qwen-VL và các model vision khác

## 2.5 OCR và document parsing chuyên dụng

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Tiền xử lý ảnh: độ phân giải, cắt, xoay

## 3.2 Pipeline xử lý tài liệu: parse trước hay đưa thẳng cho model

## 3.3 Trích xuất có cấu trúc từ ảnh và PDF

<!-- TODO: Liên kết với tài liệu Structured Output. -->

## 3.4 RAG đa phương thức

<!-- TODO: Liên kết với tài liệu RAG nâng cao. -->

## 3.5 Chi phí và caching khi gửi lại ảnh nhiều lượt

## 3.6 Eval cho tác vụ vision

## 3.7 Failure modes thường gặp

## 3.8 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
