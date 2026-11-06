# Self-hosting LLM: chạy model open-weight

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng các lý do thường gặp để tự host (dữ liệu không được rời hạ tầng, chi phí ở quy mô lớn, cần tuỳ biến) và cái giá vận hành đi kèm. Ghi rõ ranh giới với Deploying AI Systems (bài này tập trung vào chọn model, phần cứng và inference engine); chỉ đường đọc theo ba phần. -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Inference engine và phần cứng](#phần-2--inference-engine-và-phần-cứng)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Bảng ước lượng VRAM theo kích thước model và quantization; bảng so sánh inference engine. -->

---

# Phần 1 — Nền tảng

## 1.1 Khi nào nên tự host, khi nào không

## 1.2 Bộ nhớ cần cho weights và KV cache

## 1.3 Prefill, decode và vì sao batching quan trọng

## 1.4 Quantization: FP8, INT4, GGUF, AWQ, GPTQ

## 1.5 License của model open-weight

---

# Phần 2 — Inference engine và phần cứng

## 2.1 vLLM

## 2.2 SGLang

## 2.3 TensorRT-LLM và TGI

## 2.4 llama.cpp và Ollama

## 2.5 Chọn GPU: NVIDIA, AMD, Apple Silicon

## 2.6 Cloud GPU và dịch vụ inference serverless

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Chọn model theo tác vụ và phần cứng

## 3.2 Benchmark throughput và độ trễ trên tải thật

## 3.3 Tool calling và structured output trên model tự host

## 3.4 Cập nhật model và rollback

## 3.5 So sánh tổng chi phí với API

## 3.6 Failure modes thường gặp

## 3.7 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
