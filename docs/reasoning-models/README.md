# Reasoning models: khi nào cho model suy nghĩ lâu hơn

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng tình huống bật reasoning cho mọi request và thấy độ trễ, chi phí tăng gấp nhiều lần nhưng chất lượng chỉ tăng ở một số loại tác vụ. Nêu câu hỏi chính: tác vụ nào đáng trả thêm token suy nghĩ, và chỉ đường đọc theo ba phần. -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Reasoning theo provider](#phần-2--reasoning-theo-provider)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Bảng tham số reasoning (effort, budget, adaptive) theo provider; khi nào bật, khi nào tắt; reasoning token có bị tính phí và có trả về không. -->

---

# Phần 1 — Nền tảng

## 1.1 Reasoning model khác model thường ở đâu

## 1.2 Thinking token: sinh ra, tính phí, nhưng không phải lúc nào cũng thấy

<!-- TODO: Summarized thinking, encrypted reasoning, redacted thinking. -->

## 1.3 Tác vụ nào hưởng lợi, tác vụ nào không

## 1.4 Effort, budget và adaptive thinking

## 1.5 Reasoning trong hội thoại nhiều lượt

<!-- TODO: Gửi lại hay bỏ reasoning của lượt trước; ảnh hưởng đến prompt caching. Liên kết với tài liệu Multi-turn. -->

---

# Phần 2 — Reasoning theo provider

## 2.1 Bức tranh chung

## 2.2 OpenAI

## 2.3 Claude (Anthropic)

## 2.4 Google Gemini

## 2.5 xAI Grok

## 2.6 DeepSeek, Qwen và các model open-weight

## 2.7 Self-hosted: tách reasoning content trong vLLM, SGLang

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Chọn mức effort theo tác vụ

## 3.2 Prompt cho reasoning model: viết ít hơn hay nhiều hơn

## 3.3 Reasoning kết hợp tool use

## 3.4 Reasoning kết hợp structured output và streaming

## 3.5 Chi phí và độ trễ: max_tokens, timeout, ước lượng ngân sách

## 3.6 Eval: đo xem reasoning có thật sự giúp

## 3.7 Failure modes thường gặp

## 3.8 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
