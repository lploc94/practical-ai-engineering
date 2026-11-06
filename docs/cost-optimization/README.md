# Tối ưu chi phí cho ứng dụng LLM

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng tình huống hoá đơn API tăng nhanh hơn số người dùng. Nêu các đòn bẩy chính (chọn model, số token, caching, batch) và cách đo trước khi tối ưu; chỉ đường đọc theo ba phần. Liên kết với tài liệu Multi-turn (prompt caching). -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Đòn bẩy theo provider](#phần-2--đòn-bẩy-theo-provider)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Bảng đòn bẩy → mức tiết kiệm điển hình → đánh đổi; công thức ước lượng chi phí cho một feature. -->

---

# Phần 1 — Nền tảng

## 1.1 Chi phí đến từ đâu: input, output, cache, reasoning, tool

## 1.2 Đo trước khi tối ưu: chi phí theo request, user, feature

## 1.3 Vì sao output token thường đắt hơn input token

## 1.4 Đánh đổi chi phí, chất lượng và độ trễ

---

# Phần 2 — Đòn bẩy theo provider

## 2.1 Bảng giá và cách tính token theo provider

## 2.2 Prompt caching

<!-- TODO: Tóm tắt và liên kết với tài liệu Multi-turn. -->

## 2.3 Batch API

## 2.4 Model nhỏ, distillation và fine-tune

## 2.5 Self-host: khi nào rẻ hơn API

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Chọn model nhỏ nhất đạt yêu cầu

## 3.2 Rút ngắn prompt và context

## 3.3 Giới hạn output và reasoning

## 3.4 Semantic cache và cache kết quả

## 3.5 Giới hạn chi phí theo user và chống lạm dụng

## 3.6 Dự báo chi phí khi scale

## 3.7 Failure modes thường gặp

## 3.8 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
