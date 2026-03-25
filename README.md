# Practical AI Engineering

Kho tài liệu về chủ đề xây dựng, triển khai và vận hành các hệ thống AI. Nội dung được chia theo từng chủ đề để bạn có thể bắt đầu từ một vấn đề cụ thể, hoặc đi dần từ nền tảng sang hệ thống và vận hành.


## Bắt đầu từ đâu

- [Embeddings](./docs/embeddings/)

## Sắp có
...

## Cấu trúc repo

```text
docs/
  embeddings/
    README.md
```

## Repo được tổ chức thế nào?

- Mỗi chủ đề nằm trong một thư mục riêng dưới `docs/`
- Tên thư mục dùng `kebab-case` bằng tiếng Anh để ổn định lâu dài, ví dụ: `embeddings`, `deploying-ai-systems`
- File chính của mỗi chủ đề là `README.md` để GitHub tự hiển thị khi mở thư mục
- Nếu một chủ đề lớn dần lên, có thể tách thêm các file phụ như:
  - `glossary.md`
  - `patterns.md`
  - `references.md`
  - `checklist.md`
  - `assets/` cho hình ảnh hoặc file phụ trợ

## Nếu sau này thêm chủ đề mới

Khi repo có thêm nhiều tài liệu, nên giữ naming pattern như sau:

- `docs/embeddings/README.md`
- `docs/deploying-ai-systems/README.md`
- `docs/rag-systems/README.md`
- `docs/model-serving/README.md`
- `docs/ai-observability/README.md`
