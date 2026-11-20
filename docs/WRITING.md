# Hướng dẫn biên soạn

Quy tắc khi viết mới hoặc sửa tài liệu trong repo. `README.md` ở gốc repo dành cho người đọc; file này dành cho người viết. Quy tắc làm hình nằm ở [ILLUSTRATIONS.md](./ILLUSTRATIONS.md).

## Nguyên tắc

- **Dựa trên nguồn gốc.** Mọi nhận định về hành vi API, giá, giới hạn, benchmark phải đối chiếu được với tài liệu chính thức, bài báo nghiên cứu hoặc số đo tự chạy. Nguồn ghi ở mục Nguồn tham khảo cuối bài.
- **Ghi rõ thời điểm.** Giá, giới hạn và tham số API thay đổi nhanh. Đầu bài ghi tháng của bản hiện tại. Chỗ nào chưa xác minh được thì đánh dấu ngay tại chỗ, không để người đọc tưởng là đã kiểm.
- **Tiếng Việt, giữ thuật ngữ gốc.** Thuật ngữ kỹ thuật, tên field, tham số và model giữ nguyên tiếng Anh để người đọc dễ đối chiếu với docs và code. Viết tiếng Việt có dấu đầy đủ.
- **Ưu tiên quyết định hơn định nghĩa.** Mỗi phần nên giúp người đọc chọn đúng và biết vì sao, không chỉ biết khái niệm.
- **Code và số liệu kiểm lại được.** Code ví dụ đặt trong `scripts/` của chủ đề và có test. Con số trong hình được script sinh hình kiểm với nội dung bài.

## Khung một tài liệu

1. **Tiêu đề** (`#`) và đoạn mở đầu bằng một tình huống thật cho thấy vì sao chủ đề này đáng đọc.
2. **Bài này đi theo thứ tự:** danh sách ngắn mô tả từng phần.
3. **Dòng phiên bản**, dạng trích dẫn: `> Bản tháng 11/2026. ...`. Ghi thêm ngày đối chiếu nguồn, phiên bản SDK hoặc engine đã chạy code, và môi trường đo nếu có.
4. **Tài liệu liên quan:** link sang các chủ đề khác trong repo.
5. **Mục lục.**
6. **Tra cứu nhanh:** bảng cho người cần câu trả lời ngay, mỗi dòng trỏ về mục chi tiết.
7. **Phần 1 — Nền tảng:** cơ chế và lý do đằng sau.
8. **Phần 2:** tham số, giới hạn, khác biệt giữa provider, engine hoặc công cụ. Đặt tên theo chủ đề, ví dụ "Streaming theo provider".
9. **Phần 3 — Thiết kế và vận hành:** đánh đổi khi đưa vào production. Hai mục cuối là **Failure modes thường gặp** (bảng triệu chứng, nguyên nhân, cách xử lý) và **Checklist**.
10. **Nguồn tham khảo:** bảng đánh số `[n]`, kèm ghi chú nguồn đó dùng cho ý nào. Thân bài trích bằng số `[n]`.

Mục trong mỗi phần đánh số `## 1.1`, `## 1.2`, ... để link và trích dẫn ổn định.

Chủ đề không hợp với khung này thì đổi khung, rồi ghi ngoại lệ vào mục "Cách đọc" của `README.md` ở gốc repo. Ví dụ hiện có: Under Attack là runbook xếp theo thời gian, Decision Models có thêm phần hướng dẫn theo bài toán.

## Văn phong

- Câu ngắn, mỗi câu một ý. Mở đoạn bằng ý chính.
- Dùng ví dụ cụ thể có số liệu thay cho mô tả chung chung.
- Bảng dùng cho so sánh và tra cứu; đoạn văn dùng cho lập luận.
- Khi số liệu là giả định để minh họa, ghi rõ là giả định.

## Sửa tài liệu đã có

- Sửa một nhận định thì mở lại nguồn, cập nhật hoặc thêm dòng ở Nguồn tham khảo. Nguồn mới chèn giữa các số cũ thì dùng hậu tố (`[23a]`) thay vì đánh số lại toàn bài.
- Một thay đổi về hành vi API thường xuất hiện ở nhiều bài. Tìm (`grep`) trong toàn bộ `docs/` và sửa mọi chỗ cùng lúc.
- Sửa code trong khối code của bài thì sửa cả bản trong `scripts/` và chạy lại test.

## Tổ chức file

```text
docs/
  README.md             danh sách chủ đề và quy ước
  WRITING.md            file này
  ILLUSTRATIONS.md      quy tắc làm hình minh họa
  assets/svgkit.py      bảng màu và helper dùng chung cho các script sinh hình
  scripts/              công cụ cho cả repo, ví dụ check_links.py
  <chủ-đề>/
    README.md           tài liệu chính
    images/             hình SVG và script make_*.py sinh ra chúng
    scripts/            code ví dụ chạy được và test
```

- Tên thư mục chủ đề là `kebab-case` tiếng Anh. Tài liệu chính là `README.md` để GitHub hiển thị ngay khi mở thư mục.
- Hình sửa qua script `make_*.py` rồi chạy lại, không sửa file SVG bằng tay.
- Thêm chủ đề mới thì cập nhật cả [docs/README.md](./README.md) và bảng "Có gì trong repo" trong `README.md` ở gốc repo.

## Trước khi commit

- Chạy `python3 docs/scripts/check_links.py .` ở gốc repo để kiểm link nội bộ và anchor trong mọi `docs/*/README.md`.
- Chạy test trong `scripts/` của chủ đề nếu đã sửa code.
- Chạy lại script `make_*.py` nếu đã sửa con số mà hình dùng; assert trong script sẽ báo khi hình lệch với bài.
