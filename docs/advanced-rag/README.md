# RAG nâng cao: khi retrieval cơ bản không đủ

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng tình huống pipeline RAG cơ bản (chunk, embed, top-k) trả về đúng tài liệu nhưng sai đoạn, hoặc không trả lời được câu hỏi cần ghép nhiều nguồn. Ghi rõ tài liệu giả định người đọc đã nắm phần RAG trong tài liệu Embeddings, và chỉ đường đọc theo ba phần. -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Kỹ thuật](#phần-2--kỹ-thuật)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Bảng triệu chứng → kỹ thuật nên thử; thứ tự ưu tiên khi nâng cấp pipeline. -->

---

# Phần 1 — Nền tảng

## 1.1 Pipeline RAG có những điểm hỏng nào

## 1.2 Retrieval, reranking và generation: tối ưu từng tầng

## 1.3 Long context có thay thế được RAG không

## 1.4 Agentic RAG: model tự quyết khi nào và tìm gì

---

# Phần 2 — Kỹ thuật

## 2.1 Query rewriting, HyDE, multi-query, decomposition

## 2.2 Reranking: cross-encoder và reranker API

## 2.3 Contextual retrieval và chunk có ngữ cảnh

## 2.4 Parent–child, sentence window và late chunking

## 2.5 GraphRAG và knowledge graph

## 2.6 Metadata filtering và structured retrieval (text-to-SQL)

## 2.7 RAG đa phương thức: bảng, ảnh, PDF phức tạp

## 2.8 Trích dẫn nguồn và grounding

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Chọn kỹ thuật theo triệu chứng

## 3.2 Cập nhật index khi dữ liệu thay đổi

## 3.3 Phân quyền tài liệu trong retrieval

## 3.4 Độ trễ và chi phí của từng tầng

## 3.5 Eval cho RAG nâng cao

<!-- TODO: Liên kết với tài liệu Evaluation và Embeddings. -->

## 3.6 Failure modes thường gặp

## 3.7 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
