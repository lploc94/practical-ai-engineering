# Fine-tuning: khi nào và làm thế nào

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng câu hỏi thường gặp "có nên fine-tune không" và vì sao câu trả lời thường là thử prompt, RAG và eval trước. Nêu các trường hợp fine-tune thật sự đáng làm; chỉ đường đọc theo ba phần. -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Phương pháp và nền tảng](#phần-2--phương-pháp-và-nền-tảng)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Cây quyết định prompt / RAG / fine-tune; quy mô dữ liệu tham khảo theo loại tác vụ. -->

---

# Phần 1 — Nền tảng

## 1.1 Fine-tune thay đổi gì trong model và không thay đổi gì

## 1.2 Fine-tune, RAG và prompt: giải quyết bài toán khác nhau

## 1.3 SFT, preference tuning (DPO), reinforcement fine-tuning

## 1.4 Full fine-tune, LoRA và QLoRA

## 1.5 Distillation: dạy model nhỏ bắt chước model lớn

---

# Phần 2 — Phương pháp và nền tảng

## 2.1 Fine-tune qua API: OpenAI, Gemini/Vertex, Bedrock

## 2.2 Fine-tune open-weight: Hugging Face TRL, Unsloth, Axolotl

## 2.3 Chọn base model

## 2.4 Phần cứng và chi phí huấn luyện

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Chuẩn bị dữ liệu: nguồn, làm sạch, format

## 3.2 Sinh dữ liệu tổng hợp từ model lớn

## 3.3 Eval trước và sau fine-tune

<!-- TODO: Liên kết với tài liệu Evaluation. -->

## 3.4 Overfitting và catastrophic forgetting

## 3.5 Deploy và serving LoRA adapter

## 3.6 Bảo trì: khi base model ra phiên bản mới

## 3.7 Failure modes thường gặp

## 3.8 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
