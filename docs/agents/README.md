# Agents: khi LLM tự quyết bước tiếp theo

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng ranh giới giữa workflow (code quyết định luồng) và agent (model quyết định luồng). Nêu câu hỏi đầu tiên luôn là "có cần agent không", và chỉ đường đọc theo ba phần. Tài liệu giả định người đọc đã nắm Tool use. -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và framework.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Kiến trúc và công cụ](#phần-2--kiến-trúc-và-công-cụ)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Cây quyết định workflow hay agent; một agent hay nhiều agent; bảng framework. -->

---

# Phần 1 — Nền tảng

## 1.1 Workflow và agent

## 1.2 Agent loop: quan sát, suy luận, hành động

## 1.3 Các pattern workflow: chaining, routing, parallelization, evaluator–optimizer

## 1.4 Context là tài nguyên khan hiếm

<!-- TODO: Context dài làm giảm chất lượng; compaction, ghi chú ra file, sub-agent để cô lập context. Liên kết với tài liệu Multi-turn. -->

## 1.5 Memory: trong phiên, giữa các phiên

---

# Phần 2 — Kiến trúc và công cụ

## 2.1 Một agent với nhiều tool

## 2.2 Orchestrator–worker và multi-agent

## 2.3 Agent chạy code và sandbox

## 2.4 Computer use và browser agent

## 2.5 MCP và tích hợp hệ thống bên ngoài

## 2.6 Framework: Claude Agent SDK, OpenAI Agents SDK, LangGraph, Pydantic AI

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Khi nào không nên dùng agent

## 3.2 Điều kiện dừng, giới hạn bước và ngân sách

## 3.3 Human-in-the-loop

## 3.4 Chạy dài: checkpoint, resume, chạy nền

## 3.5 Observability: trace từng bước

## 3.6 Eval cho agent

## 3.7 Chi phí và độ trễ

## 3.8 Bảo mật

<!-- TODO: Liên kết với tài liệu Bảo mật LLM. -->

## 3.9 Failure modes thường gặp

## 3.10 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
