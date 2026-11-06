# Bảo mật LLM: prompt injection và những gì đi kèm

> **Đang soạn.** Đây là khung nội dung; các mục sẽ được viết dần.

<!-- TODO: Mở bài bằng tình huống indirect injection (một email hoặc trang web chứa chỉ thị khiến agent gửi dữ liệu ra ngoài). Nêu vì sao không thể "lọc sạch" injection bằng prompt, và chỉ đường đọc theo ba phần. Phân biệt với tài liệu Under Attack (bảo vệ hạ tầng). -->

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và công cụ.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
- [Phần 2 — Các lớp phòng thủ](#phần-2--các-lớp-phòng-thủ)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

<!-- TODO: Bảng mối đe doạ → biện pháp; "lethal trifecta" (dữ liệu riêng tư + nội dung không tin cậy + kênh gửi ra ngoài). -->

---

# Phần 1 — Nền tảng

## 1.1 Vì sao model không phân biệt được chỉ thị và dữ liệu

## 1.2 Direct injection và jailbreak

## 1.3 Indirect injection qua tài liệu, web, email, tool result

## 1.4 Rò rỉ dữ liệu: system prompt, context, exfiltration qua link và ảnh

## 1.5 OWASP Top 10 cho ứng dụng LLM

---

# Phần 2 — Các lớp phòng thủ

## 2.1 Giới hạn quyền: least privilege cho tool và agent

## 2.2 Tách nguồn tin cậy: đánh dấu dữ liệu không tin cậy, dual-LLM, CaMeL

## 2.3 Xác nhận của con người cho hành động nhạy cảm

## 2.4 Classifier và guardrail cho input/output

## 2.5 Kiểm soát output: chặn URL, markdown image, render an toàn

## 2.6 PII: phát hiện, che, và lưu log

## 2.7 Tính năng bảo mật từ provider

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Threat modeling cho ứng dụng LLM

## 3.2 Red teaming và eval bảo mật

## 3.3 Logging, phát hiện và phản ứng sự cố

## 3.4 Chuỗi cung ứng: model, MCP server, plugin bên thứ ba

## 3.5 Failure modes thường gặp

## 3.6 Checklist

---

## Nguồn tham khảo

<!-- TODO -->
