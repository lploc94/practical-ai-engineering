# MCP: kết nối model với công cụ và dữ liệu

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng tình huống mỗi ứng dụng AI phải viết lại tích hợp cho cùng một hệ thống (GitHub, database, Drive). Nêu MCP giải quyết bài toán gì và không giải quyết gì so với tool tự định nghĩa, và chỉ đường đọc theo ba phần. Tài liệu giả định người đọc đã nắm Tool use. -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Tra cứu giao thức và SDK](#phần-2--tra-cứu-giao-thức-và-sdk)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Khi nào dùng MCP, khi nào tool tự định nghĩa là đủ; bảng transport; checklist bảo mật cho MCP server. -->

---

# Phần 1 — Nền tảng

## 1.1 MCP là gì: host, client, server

## 1.2 Tools, resources và prompts

## 1.3 Vòng đời một phiên: initialize, capability negotiation

## 1.4 Transport: stdio và Streamable HTTP

## 1.5 MCP khác function calling thế nào

---

# Phần 2 — Tra cứu giao thức và SDK

## 2.1 Đặc tả và các phiên bản

## 2.2 SDK chính thức: Python, TypeScript và các ngôn ngữ khác

## 2.3 Authorization: OAuth cho remote server

## 2.4 Sampling, elicitation và các tính năng nâng cao

## 2.5 Client hỗ trợ MCP: Claude, ChatGPT, IDE, agent SDK

## 2.6 MCP connector trong API của provider

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Thiết kế MCP server: tool nào nên expose

## 3.2 Mô tả tool và kích thước output

## 3.3 Remote server: deploy, scale, multi-tenant

## 3.4 Bảo mật: tool poisoning, quyền hạn, server bên thứ ba

<!-- TODO: Liên kết với tài liệu Bảo mật LLM. -->

## 3.5 Test và debug MCP server

## 3.6 Failure modes thường gặp

## 3.7 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
