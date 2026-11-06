# Evaluation: biết hệ thống LLM tốt lên hay tệ đi

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng tình huống đổi prompt hoặc đổi model rồi "thử vài câu thấy ổn" và hỏng ở production. Nêu vì sao output không tất định khiến unit test truyền thống không đủ, và chỉ đường đọc theo ba phần. -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và công cụ.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Phương pháp và công cụ](#phần-2--phương-pháp-và-công-cụ)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Chọn loại grader theo loại tác vụ; quy mô dataset tối thiểu; checklist dựng eval đầu tiên trong một ngày. -->

---

# Phần 1 — Nền tảng

## 1.1 Vì sao "thử vài câu" không phải là eval

## 1.2 Eval offline, eval online và human review

## 1.3 Định nghĩa "tốt": tiêu chí, rubric và metric

## 1.4 Phương sai: chạy bao nhiêu lần, cần bao nhiêu mẫu

<!-- TODO: Non-determinism, khoảng tin cậy, khi nào chênh lệch là có ý nghĩa. -->

## 1.5 Benchmark công khai khác eval của bạn thế nào

---

# Phần 2 — Phương pháp và công cụ

## 2.1 Grader dựa trên code: exact match, regex, schema, unit test

## 2.2 LLM-as-judge

<!-- TODO: Pointwise và pairwise, rubric, các thiên lệch (vị trí, độ dài, tự ưu ái), cách hiệu chỉnh judge với nhãn của người. -->

## 2.3 Eval cho RAG: retrieval và generation

<!-- TODO: Liên kết với phần evaluation trong tài liệu Embeddings. -->

## 2.4 Eval cho tool use và agent

## 2.5 Eval cho structured output và extraction

## 2.6 Công cụ: OpenAI Evals, promptfoo, Braintrust, LangSmith, Langfuse, Inspect

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Xây golden dataset từ đâu

## 3.2 Gán nhãn và đồng thuận giữa người gán nhãn

## 3.3 Regression test khi đổi prompt hoặc model

## 3.4 Đưa eval vào CI

## 3.5 Eval online: A/B test, feedback người dùng, sampling log

## 3.6 Chi phí và thời gian chạy eval

## 3.7 Failure modes thường gặp

## 3.8 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
