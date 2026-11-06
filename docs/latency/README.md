# Giảm độ trễ cho ứng dụng LLM

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng tình huống một lượt trả lời mất 15 giây vì gồm ba lần gọi model nối tiếp và một lần retrieval. Nêu các thành phần của độ trễ và cái nào người dùng cảm nhận; chỉ đường đọc theo ba phần. -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Kỹ thuật](#phần-2--kỹ-thuật)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Bảng thành phần độ trễ → kỹ thuật giảm; số liệu TTFT và tokens/giây tham khảo theo model. -->

---

# Phần 1 — Nền tảng

## 1.1 Độ trễ gồm những gì: mạng, hàng đợi, prefill, decode

## 1.2 TTFT, tokens/giây và tổng thời gian

## 1.3 Độ trễ cảm nhận và độ trễ thực

## 1.4 Vì sao độ trễ dao động: p50 và p99

---

# Phần 2 — Kỹ thuật

## 2.1 Streaming

<!-- TODO: Liên kết với tài liệu Streaming. -->

## 2.2 Prompt caching và prefill

## 2.3 Chọn model và chế độ nhanh (fast mode, priority tier)

## 2.4 Chạy song song và speculative execution

## 2.5 Speculative decoding và các kỹ thuật phía serving

## 2.6 Semantic cache

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Đo độ trễ từng tầng trong pipeline

## 3.2 Giảm số lượt gọi model nối tiếp

## 3.3 Đặt region gần provider và người dùng

## 3.4 Timeout, hedging và retry

## 3.5 UX khi phải chờ: trạng thái, kết quả từng phần

## 3.6 Failure modes thường gặp

## 3.7 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
