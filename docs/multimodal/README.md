# Multimodal: ảnh, PDF và âm thanh

Một đội kế toán cần trích số tiền, mã số thuế và danh sách dòng hàng từ vài nghìn hoá đơn mỗi tháng. Một phần là PDF xuất từ phần mềm, có text layer sạch. Phần còn lại là ảnh chụp bằng điện thoại: nghiêng, có bóng tay, có con dấu đỏ đè lên chữ. Thư viện trích text từ PDF đọc được nhóm đầu nhưng làm vỡ bảng, cột "đơn giá" dồn sang cột "thành tiền". OCR truyền thống đọc được nhóm sau nhưng mất dấu tiếng Việt và nhầm `8` với `B`.

Đội thử đưa thẳng ảnh và PDF cho một vision model, kèm schema JSON. Kết quả tốt hơn hẳn trên bộ mẫu. Lên production thì có vấn đề mới:

- Hoá đơn chụp bằng điện thoại 12 megapixel bị provider tự thu nhỏ, chữ nhỏ ở phần chân trang không còn đọc được, và model *đoán* mã số thuế thay vì báo không đọc được.
- Chi phí mỗi hoá đơn cao gấp ba ước tính. PDF được tính cả token cho ảnh trang lẫn text trích ra, và ảnh chụp lớn tốn token nhiều hơn ảnh nhỏ.
- Request có 25 ảnh bị từ chối, dù mỗi ảnh đều dưới giới hạn dung lượng.
- Ảnh chụp dọc lưu xoay bằng metadata EXIF được gửi đi nằm ngang.
- Model đếm sai số dòng hàng trên hoá đơn dài.

Tài liệu này giải thích model đọc ảnh, PDF và âm thanh thế nào, mỗi loại tốn bao nhiêu token, và cách dựng pipeline để những lỗi trên không xảy ra âm thầm.

- **Nền tảng — Phần 1:** patch, tile và token; cách tính token ảnh của từng provider; PDF được đưa vào model ra sao; âm thanh và video; những việc vision model vẫn làm kém theo nghiên cứu.
- **Tra cứu theo provider — Phần 2:** OpenAI, Claude, Gemini, model open-weight (Qwen-VL và các model vision khác), và công cụ OCR, parse tài liệu chuyên dụng.
- **Thiết kế và vận hành — Phần 3:** tiền xử lý ảnh, chọn giữa parse trước và đưa thẳng cho model, trích xuất có cấu trúc, RAG đa phương thức, chi phí và cache, eval, và [bảng failure mode ở §3.7](#37-failure-modes-thường-gặp).

> Bản tháng 11/2026. Tên tham số, field và model giữ nguyên tiếng Anh để tiện đối chiếu API. Công thức tính token ảnh đã thay đổi nhiều lần trong hai năm qua, kể cả trong cùng một provider. Những điểm chỉ thấy ở nguồn thứ cấp hoặc chưa đo được bằng request thật được đánh dấu tại chỗ.

Tài liệu liên quan: [Structured Output](../structured-output/) (schema và validation cho dữ liệu trích xuất), [RAG nâng cao §2.7](../advanced-rag/#27-rag-đa-phương-thức-bảng-ảnh-pdf-phức-tạp), [Embeddings §2.8](../embeddings/#28-multimodal-embedding), [Multi-turn §1.4](../multi-turn-conversation/#14-kinh-tế-học-của-prompt-caching) (cache khi gửi lại nội dung lớn), [Bảo mật LLM §1.3](../llm-security/#13-indirect-injection-qua-tài-liệu-web-email-tool-result) (prompt injection qua tài liệu).

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Model đọc ảnh thế nào: patch, tile và token](#11-model-đọc-ảnh-thế-nào-patch-tile-và-token)
  - [1.2 Ảnh tốn bao nhiêu token](#12-ảnh-tốn-bao-nhiêu-token)
  - [1.3 PDF: ảnh trang, text layer hay cả hai](#13-pdf-ảnh-trang-text-layer-hay-cả-hai)
  - [1.4 Âm thanh: speech-to-text, model audio native](#14-âm-thanh-speech-to-text-model-audio-native)
  - [1.5 Những gì vision model vẫn làm kém](#15-những-gì-vision-model-vẫn-làm-kém)
- [Phần 2 — Multimodal theo provider](#phần-2--multimodal-theo-provider)
  - [2.1 OpenAI](#21-openai)
  - [2.2 Claude (Anthropic)](#22-claude-anthropic)
  - [2.3 Google Gemini](#23-google-gemini)
  - [2.4 Model open-weight: Qwen-VL và các model vision khác](#24-model-open-weight-qwen-vl-và-các-model-vision-khác)
  - [2.5 OCR và document parsing chuyên dụng](#25-ocr-và-document-parsing-chuyên-dụng)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Tiền xử lý ảnh: độ phân giải, cắt, xoay](#31-tiền-xử-lý-ảnh-độ-phân-giải-cắt-xoay)
  - [3.2 Pipeline xử lý tài liệu: parse trước hay đưa thẳng cho model](#32-pipeline-xử-lý-tài-liệu-parse-trước-hay-đưa-thẳng-cho-model)
  - [3.3 Trích xuất có cấu trúc từ ảnh và PDF](#33-trích-xuất-có-cấu-trúc-từ-ảnh-và-pdf)
  - [3.4 RAG đa phương thức](#34-rag-đa-phương-thức)
  - [3.5 Chi phí và caching khi gửi lại ảnh nhiều lượt](#35-chi-phí-và-caching-khi-gửi-lại-ảnh-nhiều-lượt)
  - [3.6 Eval cho tác vụ vision](#36-eval-cho-tác-vụ-vision)
  - [3.7 Failure modes thường gặp](#37-failure-modes-thường-gặp)
  - [3.8 Checklist](#38-checklist)
- [Nguồn tham khảo](#nguồn-tham-khảo)

---

## Tra cứu nhanh

### Giới hạn ảnh (snapshot 11/2026)

| | OpenAI | Claude | Gemini | Grok |
|---|---|---|---|---|
| Định dạng | PNG, JPEG, WEBP, GIF không động | JPEG, PNG, GIF (frame đầu), WebP | PNG, JPEG, WEBP, HEIC, HEIF | JPG/JPEG, PNG |
| Dung lượng mỗi ảnh | Không nêu riêng; giới hạn theo số patch | 10 MB (base64, Claude API); 5 MB trên Bedrock; tối đa 8000×8000 px | Gộp trong giới hạn request inline 20 MB | 20 MiB |
| Số ảnh mỗi request | 1.500 | 600; 100 với model context 200k | 3.600 | Docs ghi không giới hạn |
| Tổng request | 512 MB | 128 MB | 20 MB inline; lớn hơn dùng Files API | Không nêu |
| Điều khiển độ chi tiết | `detail`: `low`, `high`, `auto`, `original` | Không có tham số; tier phân giải theo model | `media_resolution` / `resolution`: `low` → `ultra_high` | `detail` (docs không liệt kê giá trị) |
| Cách gửi | URL, base64 data URL, `file_id` | base64, URL, `file_id` | Inline base64, Files API | base64 data URL, URL công khai |

Nguồn: [1][4][7][14]. Claude còn một giới hạn dễ vấp: request có **hơn 20** ảnh hoặc document block thì cạnh dài mỗi ảnh phải ≤ 3.000 px [4].

### Công thức token ảnh

| Provider | Công thức | Ví dụ |
|---|---|---|
| OpenAI, GPT-5.x và `gpt-6-astra` | `ceil(w/32) × ceil(h/32)` patch, thu nhỏ nếu vượt ngân sách patch, rồi × multiplier theo model (1,2 với `gpt-6-astra`, GPT-5.6) | 2048×2048 ở `high` trên `gpt-6-astra` → 1600×1600, 2.500 patch → 3.000 token [1] |
| OpenAI, model cũ (gpt-4o, gpt-4.1, gpt-5.1, o-series) | Tile 512 px: `base + tile × số_tile`, hoặc chỉ `base` ở `low` | gpt-5.1: base 70, mỗi tile 140 [1] |
| Claude | `⌈w/28⌉ × ⌈h/28⌉`; standard tier: cạnh dài ≤ 1.568 px, tối đa 1.568 token; high-res (Claude 4.7 trở lên): ≤ 2.576 px, tối đa 4.784 token | 1920×1080: 1.560 token (standard), 2.691 (high-res) [4] |
| Gemini 3, 4 | Cố định theo mức `media_resolution`: 280 / 560 / 1.120 / 2.240 token | Mặc định 1.120 token, bất kể kích thước [9] |
| Gemini model cũ hơn | 258 token nếu cả hai cạnh ≤ 384 px; lớn hơn chia tile 768×768, mỗi tile 258 token | [7][12] |

### PDF, âm thanh, video

| | OpenAI | Claude | Gemini 3, 4 |
|---|---|---|---|
| PDF tối đa | 50 MB mỗi file và tổng mỗi request | 128 MB request; 600 trang (100 trang nếu context < 1M) | 50 MB hoặc 1.000 trang |
| Token mỗi trang | Text trích ra + ảnh trang (docs không nêu con số) | Text khoảng 1.500–3.000 token + ảnh trang theo công thức ảnh | 560 token ảnh trang ở mặc định; text gốc trong PDF không tính tiền |
| Âm thanh | Model transcribe và realtime, tính theo phút hoặc token | Không nhận audio | 25 token/giây, tối đa 9,5 giờ mỗi prompt |
| Video | Không nhận trực tiếp | Không nhận | 70 token/frame (`low`, `medium`), 280 (`high`), 1 FPS mặc định |

Nguồn: [2][6][8][9][10][11]. Chi tiết và các mâu thuẫn giữa trang docs của Gemini ở [§2.3](#23-google-gemini).

---

# Phần 1 — Nền tảng

## 1.1 Model đọc ảnh thế nào: patch, tile và token

Language model chỉ xử lý chuỗi token. Để "đọc" ảnh, ảnh được cắt thành các ô vuông nhỏ (*patch*), mỗi patch đi qua một vision encoder để thành vector, rồi các vector này được chiếu vào cùng không gian với token văn bản. Từ góc nhìn của language model, một ảnh là vài trăm đến vài nghìn token nằm xen giữa các token chữ.

Từ cơ chế này có ba hệ quả thực tế:

**Ảnh tốn token theo diện tích.** Số patch tỉ lệ với số pixel. Ảnh gấp đôi mỗi chiều tốn gần gấp bốn lần token, cho đến khi chạm trần mà provider đặt ra.

**Provider thu nhỏ ảnh trước khi đưa vào encoder.** Mỗi provider có trần riêng về cạnh dài hoặc tổng số patch. Ảnh vượt trần bị thu nhỏ âm thầm. Ảnh 12 megapixel từ điện thoại có thể chỉ còn khoảng 1,5 megapixel khi model thực sự nhìn thấy nó. Docs Claude ghi rõ việc thu nhỏ có thể làm chữ khó đọc hơn [4]. Chữ nhỏ ở chân trang hoá đơn là thứ mất đầu tiên.

**Có hai kiểu chia ảnh.** Kiểu cũ chia ảnh thành *tile* cố định (512 px của OpenAI thế hệ trước, 768 px của Gemini thế hệ trước), mỗi tile có giá token cố định. Kiểu mới đếm thẳng số patch nhỏ (28 px của Claude, 32 px của OpenAI GPT-5.x, 32 px sau khi gộp của Qwen3-VL) hoặc cấp một ngân sách token cố định theo mức độ chi tiết (Gemini 3). Hai kiểu cho kết quả rất khác với cùng một ảnh, nên con số "một ảnh tốn bao nhiêu" chỉ có nghĩa khi gắn với model cụ thể.

Model open-weight cho thấy rõ cơ chế này. Với Qwen2.5-VL, mỗi vùng 28×28 pixel ứng với một token, và số token mỗi ảnh mặc định nằm trong khoảng 4 đến 16.384. Model card gợi ý đặt `min_pixels` và `max_pixels` (ví dụ 256×28×28 đến 1280×28×28) để giới hạn chi phí [16]. Qwen3-VL đổi sang patch 16 px gộp 2×2, tức mỗi token ứng với vùng 32×32 pixel, và tách ngân sách pixel của ảnh với video [15]. Provider đóng làm điều tương tự ở phía server, chỉ khác là bạn không chỉnh trực tiếp được.

## 1.2 Ảnh tốn bao nhiêu token

### Theo từng provider

**OpenAI** có hai họ công thức, chọn theo model [1]:

- *Patch 32 px* (GPT-5.x, `gpt-6-astra`): `patch_count = ceil(w/32) × ceil(h/32)`. Nếu vượt ngân sách patch của mức `detail`, ảnh được thu nhỏ cho vừa rồi đếm lại. Token tính phí là `ceil(patch_count × multiplier)`. Multiplier là 1,2 với `gpt-6-astra`, GPT-5.6, GPT-5.5, GPT-5.4; 1,5 với `gpt-5-nano`; 1,72 với `o4-mini`. Ví dụ trong docs: ảnh 2048×2048 ở `detail: "high"` trên `gpt-6-astra` được thu về 1600×1600, tức 2.500 patch, thành 3.000 token.
- *Tile 512 px* (gpt-4o, gpt-4.1, gpt-5.1, o-series): `low` chỉ tính base token bất kể kích thước. `high` thu ảnh vừa 2048×2048, rồi thu cạnh ngắn về 768 px nếu lớn hơn, đếm số tile 512 px, rồi tính `base + tile × số_tile`. Với gpt-5.1, base là 70 và mỗi tile 140.

Mức `detail` có bốn giá trị: `low`, `high`, `auto` (mặc định) và `original`. Với `gpt-6-astra`, `auto` tương đương `original`; với GPT-5.4, `auto` tương đương `high` [1]. `original` cho phép ảnh lớn hơn nhiều (tới 65.535 px mỗi cạnh trên GPT-5.6 và `gpt-6-astra`), tức có thể tốn rất nhiều token nếu không kiểm soát kích thước ảnh gửi lên. Docs cũng lưu ý ảnh được tính vào giới hạn token mỗi phút (TPM), và ảnh vượt ngưỡng patch tuyệt đối bị từ chối chứ không được thu nhỏ [1].

**Claude** đếm patch 28 px: `⌈w/28⌉ × ⌈h/28⌉` visual token, với hai tier [4]:

| Tier | Model | Cạnh dài tối đa | Token tối đa |
|---|---|---|---|
| High-resolution | Claude 4.7 trở lên | 2.576 px | 4.784 |
| Standard | Các model khác | 1.568 px | 1.568 |

Ví dụ trong docs [4]:

| Ảnh gốc | Standard | High-resolution |
|---|---|---|
| 1000×1000 | 1.296 | 1.296 |
| 1920×1080 | thu về 1456×819 → 1.560 | 2.691 |
| 2000×1500 | thu về 1269×952 → 1.564 | 3.888 |
| 3840×2160 | thu về 1456×819 → 1.560 | thu về 2576×1449 → 4.784 |

Docs ghi tier high-res có thể dùng khoảng gấp ba lần visual token so với cùng ảnh trên model standard [4]. Nâng từ một model Claude cũ lên model 4.7+ mà giữ nguyên ảnh đầu vào có thể làm chi phí ảnh tăng vọt, dù đơn giá token không đổi. Một số tài liệu cũ dùng công thức `width × height / 750`. Công thức này không còn xuất hiện trong docs hiện tại.

Hàm sau ước lượng token ảnh của Claude. Nó cho đúng tám con số trong bảng trên, nhưng quy tắc thu nhỏ chính xác nằm ở trang vision coordinates [5], nên hãy đối chiếu với `usage.input_tokens` trên request thật:

```python
import math

def claude_image_tokens(w: int, h: int, high_res: bool) -> int:
    """Ước lượng visual token của Claude theo công thức patch 28 px (docs 11/2026)."""
    max_edge, max_tokens = (2576, 4784) if high_res else (1568, 1568)

    def tokens(scale: float) -> int:
        return math.ceil(math.floor(w * scale) / 28) * math.ceil(math.floor(h * scale) / 28)

    scale = min(1.0, max_edge / max(w, h))
    if tokens(scale) > max_tokens:            # vẫn vượt trần token: thu tiếp
        lo, hi = 0.0, scale
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if tokens(mid) <= max_tokens else (lo, mid)
        scale = lo
    return tokens(scale)

assert claude_image_tokens(1920, 1080, high_res=False) == 1560
assert claude_image_tokens(3840, 2160, high_res=True) == 4784
```

**Gemini 3 và 4** không tính theo kích thước ảnh mà theo mức `media_resolution` [9]: `low` 280 token, `medium` 560, `high` 1.120, `ultra_high` 2.240 (chỉ đặt được theo từng ảnh). Mặc định (`unspecified`) là 1.120. Ảnh 4K và ảnh 1000 px tốn như nhau ở cùng một mức. Với model Gemini cũ hơn, ảnh có cả hai cạnh ≤ 384 px tính 258 token, ảnh lớn hơn được chia tile 768×768, mỗi tile 258 token [7][12].

**Grok:** trang image understanding không có công thức token [14]. Đọc số token từ `usage` của response.

### So sánh trên cùng ảnh

| Ảnh | OpenAI `gpt-6-astra`, `high` (tính theo công thức) | Claude high-res (docs) | Gemini 3, mặc định (docs) |
|---|---|---|---|
| 1000×1000 | 32×32 = 1.024 patch × 1,2 ≈ 1.229 | 1.296 | 1.120 |
| 1920×1080 | 60×34 = 2.040 patch × 1,2 ≈ 2.448 | 2.691 | 1.120 |
| 3840×2160 | Vượt 2.500 patch, thu nhỏ rồi tính lại | 4.784 | 1.120 |

Cột OpenAI là phép tính từ công thức trong docs, chưa đối chiếu với request thật. Hai điều rút ra: (1) chi phí ảnh khác nhau tới vài lần giữa các provider với cùng một ảnh; (2) với OpenAI và Claude, thu nhỏ ảnh trước khi gửi là cách giảm chi phí trực tiếp nhất, còn với Gemini 3 thì chọn mức `media_resolution` mới là cách đó.

### Quy ra tiền

Chi phí ảnh = token ảnh × đơn giá input của model. Docs Claude đưa sẵn vài con số: ảnh 1000×1000 trên Haiku 4.5 ($1 / 1M token input) khoảng $1,30 cho 1.000 ảnh; cùng ảnh trên Opus 5.5 ($4 / 1M, high-res) khoảng $5,18; ảnh 4K trên Opus 5.5 khoảng $19,14 cho 1.000 ảnh [4]. Với pipeline xử lý hàng trăm nghìn ảnh mỗi tháng, chênh lệch giữa ảnh 4K và ảnh đã thu về đúng kích thước cần dùng là khoản tiền đáng kể.

## 1.3 PDF: ảnh trang, text layer hay cả hai

PDF có thể chứa text thật (xuất từ Word, phần mềm kế toán) hoặc chỉ là ảnh của trang giấy (scan, chụp). Cả ba provider lớn đều xử lý PDF bằng cách kết hợp hai nguồn: ảnh render của từng trang và text trích từ text layer.

| | Claude [6] | OpenAI [2] | Gemini 3, 4 [8][9] |
|---|---|---|---|
| Cách xử lý | Mỗi trang thành ảnh, text trích ra gửi kèm | Model có vision: text và ảnh trang; file khác (DOCX, PPTX): chỉ text | Ảnh trang + text gốc trích ra |
| Tính phí | Text khoảng 1.500–3.000 token/trang tuỳ mật độ, cộng token ảnh trang | Text + ảnh trang; docs không nêu số token mỗi trang | Ảnh trang theo `media_resolution` (mặc định 560); text gốc **không tính tiền** |
| Giới hạn | 128 MB request; 600 trang (100 nếu context < 1M) | 50 MB mỗi file và tổng mỗi request | 50 MB hoặc 1.000 trang |
| Ghi chú | Không nhận PDF có mật khẩu; .docx, .xlsx phải chuyển sang PDF hoặc text | `detail` của ảnh trang mặc định `auto`: `high` trên GPT-5.6 trở lên, `low` trên model cũ hơn; `file_url` chỉ có trên Responses API | Token ảnh trang được báo dưới modality `IMAGE`, không phải `DOCUMENT` như vài bản trước |

Ba điều thường bị bỏ sót:

- **PDF dày đặc chữ đầy context trước khi chạm giới hạn số trang.** Docs Claude ghi rõ điều này với tài liệu nhiều chữ nhỏ và bảng phức tạp [6]. 600 trang × khoảng 3.000 token chữ đã là 1,8 triệu token, chưa tính ảnh.
- **Text layer có thể sai mà nhìn bằng mắt không thấy.** PDF xuất từ một số phần mềm có font nhúng ánh xạ ký tự sai: trang hiển thị đúng chữ "Hoá đơn", nhưng text trích ra là chuỗi ký tự lạ hoặc mất dấu. Khi model nhận cả ảnh lẫn text, hai nguồn mâu thuẫn nhau. Với tài liệu tiếng Việt, hãy kiểm tra text layer trước khi tin nó.
- **Gemini thu ảnh trang về khung 3072×3072** (trang lớn) hoặc phóng lên 768×768 (trang nhỏ). Trang nhỏ hơn không làm giảm chi phí, chỉ giảm băng thông [8]. Docs Gemini cũng khuyên xoay trang đúng chiều trước khi upload và tránh trang mờ [8].

Với Gemini 3, docs khuyên dùng mức `medium` cho PDF: chất lượng thường bão hoà ở mức này, và tăng lên `high` hiếm khi cải thiện OCR với tài liệu thông thường [9].

## 1.4 Âm thanh: speech-to-text, model audio native

Có hai cách đưa âm thanh vào hệ thống LLM:

**Chuyển thành text trước (speech-to-text).** Một model chuyên dụng chép lời, rồi LLM xử lý văn bản. Cách này rẻ, dễ debug (đọc được bản chép), và tách được hai bài toán. OpenAI tính phí theo phút: `gpt-4o-transcribe` và `whisper-1` khoảng $0,006/phút, `gpt-4o-mini-transcribe` khoảng $0,003/phút [3]. Đổi lại, mất những thông tin không nằm trong chữ: giọng điệu, cảm xúc, ai đang nói khi transcript không có diarization, tiếng ồn nền.

**Đưa âm thanh thẳng vào model audio native.** Model nhận audio như một modality, nên trả lời được câu hỏi về giọng nói, nhận diện người nói, hoặc tóm tắt cuộc họp mà không cần bước chép lời riêng.

- Gemini nhận audio trong prompt: tối đa 9,5 giờ mỗi prompt; định dạng WAV, MP3, AIFF, AAC, OGG, FLAC, M4A, Opus, WebM và một số định dạng khác. Prompt có thể tham chiếu mốc thời gian dạng `MM:SS`. Inline tối đa 20 MB tổng request, lớn hơn thì dùng Files API [10].
- Token audio của Gemini: trang media resolution và trang giá ghi **25 token/giây** cho Gemini 3 và 4, cố định ở mọi mức phân giải [9][13]. Trang audio và trang token counting vẫn ghi 32 token/giây (1 phút = 1.920 token) [10][12]. Con số 32 là của thế hệ model trước. Hãy đo bằng `countTokens` trên model bạn dùng.
- OpenAI có model realtime cho hội thoại giọng nói. Giá audio input của `gpt-realtime` trên trang giá chính thức là $32 / 1M token, output $64 / 1M token [3]. Kiến trúc realtime (WebSocket, WebRTC, chen ngang) nằm ngoài phạm vi tài liệu này.
- Claude không nhận audio trong Messages API. Muốn dùng Claude với âm thanh thì chép lời trước.

Chọn cách nào: nếu đầu ra cần là văn bản (biên bản, tìm kiếm, phân loại cuộc gọi theo nội dung), chép lời trước thường rẻ và dễ kiểm soát hơn. Nếu thông tin nằm ở chính âm thanh (cảm xúc khách hàng, người nói, âm thanh không phải lời) thì dùng model audio native.

**Video** (Gemini) được xử lý như chuỗi frame lấy mẫu 1 FPS mặc định, cộng audio. Với Gemini 3 và 4, mỗi frame tốn 70 token ở `low` và `medium`, 280 token ở `high` [9]. Trang video understanding và trang token counting ghi các con số của thế hệ trước: 66 hoặc 258 token mỗi frame, cộng 32 token/giây audio, tổng khoảng 100 token/giây ở độ phân giải mặc định và 300 token/giây ở độ phân giải cao [11][12]. Chế độ *agentic* (Gemini 4 Argon, 3.8 Flash, 3.6 Flash, 3.5 Flash Lite) cho model tự tải transcript, frame hoặc audio khi cần. Docs ghi chế độ này dùng ít hơn tới 88% token với video dài, nhưng có thể tăng TTFT và tổng token với clip ngắn dưới 5 phút [11].

## 1.5 Những gì vision model vẫn làm kém

Vision model đọc tài liệu và mô tả ảnh tốt đến mức dễ quên rằng chúng có những điểm yếu có hệ thống. Các nghiên cứu sau đo trên model của 2023–2025. Model 2026 đã tốt hơn ở nhiều chỗ, nhưng các nhóm lỗi vẫn là danh sách cần kiểm tra:

- **Đếm và hình đơn giản.** "Vision language models are blind" (Rahmanzadehgervi và cộng sự, 2024) thử bốn VLM trên bảy tác vụ người làm được ngay, như đếm số điểm giao nhau của hai đường, xác định hai hình tròn có chạm nhau không, đếm hàng và cột trong lưới. Độ chính xác trung bình chỉ khoảng 58%. Model tốt nhất lúc đó là Claude 3.5 Sonnet, đạt khoảng 75–78% tuỳ phiên bản paper [22]. Docs Claude hiện tại vẫn ghi việc đếm có thể không chính xác, nhất là với nhiều vật nhỏ [4].
- **Quan hệ không gian.** VLM yếu với trái/phải, trên/dưới (Kamath và cộng sự, 2023) [23]. "Mind the Gap" (2025) thử 13 VLM và ghi độ chính xác trung bình trên các tác vụ suy luận không gian gần mức đoán ngẫu nhiên [24].
- **Chữ nhỏ sau khi bị thu nhỏ.** Không phải lỗi của model mà là hệ quả của bước thu nhỏ ở [§1.1](#11-model-đọc-ảnh-thế-nào-patch-tile-và-token). Model không biết mình đang nhìn bản mờ, và thường đoán ra chữ hợp lý thay vì báo không đọc được.
- **Bịa chi tiết.** Khi câu hỏi giả định một thứ không có trong ảnh, hoặc kiến thức nền mâu thuẫn với ảnh, model có xu hướng trả lời theo kiến thức nền.
- **Dấu tiếng Việt.** Chưa có benchmark công khai đủ tin cậy cho OCR tiếng Việt trên model 2026. Những lỗi cần đo trên dữ liệu của bạn: mất dấu khi ảnh bị nén mạnh hoặc chữ nhỏ, nhầm `đ`/`d`, `ơ`/`o`, `ư`/`u`, và lỗi ở bảng trộn tiếng Anh với tiếng Việt. Đo tỉ lệ sai ở mức ký tự có dấu, không chỉ tỉ lệ đúng từ.

Nguyên tắc rút ra: đừng giao cho vision model những việc mà một bước xác định có thể làm chính xác hơn. Đếm số dòng hàng thì đếm phần tử trong JSON đã trích, rồi so với tổng tiền. Kiểm tra vị trí thì yêu cầu bounding box rồi kiểm bằng code. Đọc mã số thì kiểm checksum.

---

# Phần 2 — Multimodal theo provider

## 2.1 OpenAI

Responses API nhận ảnh qua content part `input_image` và file qua `input_file`:

```python
import base64
from openai import OpenAI

client = OpenAI()

def data_url(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

resp = client.responses.create(
    model="gpt-6-astra",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Trích số hoá đơn, ngày và tổng tiền."},
            {"type": "input_image", "image_url": data_url("hoadon.jpg", "image/jpeg"), "detail": "high"},
            {"type": "input_file", "filename": "phu-luc.pdf",
             "file_data": data_url("phu-luc.pdf", "application/pdf")},
        ],
    }],
)
print(resp.output_text)
print(resp.usage.input_tokens)   # gồm cả token ảnh và PDF
```

Các điểm cần nhớ [1][2]:

- Ảnh gửi bằng URL công khai, base64 data URL, hoặc `file_id` từ Files API. File dùng làm input cho model upload với `purpose: "user_data"`; `purpose: "vision"` dành cho dữ liệu fine-tune vision, không phải cho input thường [2]. PDF gửi bằng `file_data`, `file_id`, hoặc `file_url` (chỉ Responses API; Chat Completions không hỗ trợ `file_url`).
- Tối đa 1.500 ảnh và 512 MB payload mỗi request. PDF tối đa 50 MB mỗi file và tổng mỗi request.
- Đặt `detail` rõ ràng. Với `gpt-6-astra`, mặc định `auto` tương đương `original`, tức ảnh lớn được giữ nhiều chi tiết hơn và tốn nhiều token hơn.
- Ảnh và chart nhúng trong DOCX, PPTX không được trích; chỉ PDF với model có vision mới có ảnh trang.

## 2.2 Claude (Anthropic)

```python
import base64
import anthropic

client = anthropic.Anthropic()

def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()

resp = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=4096,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/jpeg", "data": b64("hoadon.jpg")}},
            {"type": "document",
             "source": {"type": "base64", "media_type": "application/pdf", "data": b64("phu-luc.pdf")}},
            {"type": "text", "text": "Trích số hoá đơn, ngày và tổng tiền từ cả hai tài liệu."},
        ],
    }],
)
print(resp.usage.input_tokens)
```

Các điểm cần nhớ [4][5][6]:

- Ảnh gửi bằng base64, URL (`source.type: "url"`) hoặc Files API (`source.type: "file"`). Trên Bedrock chỉ có base64, mỗi ảnh tối đa 5 MB.
- Giới hạn: 10 MB mỗi ảnh (base64), 8000×8000 px, 600 ảnh mỗi request (100 với model context 200k), 128 MB mỗi request. Request lớn có thể lỗi trước khi chạm giới hạn số ảnh, kể cả khi dùng Files API.
- **Quy tắc nhiều ảnh:** request có hơn 20 ảnh hoặc document block, tính cả ảnh ở các lượt trước và ảnh trong `tool_result`, thì cạnh dài mỗi ảnh phải ≤ 3.000 px. Vi phạm trả `invalid_request_error` nhắc tới "many-image requests" [4]. Agent chụp màn hình qua nhiều lượt rất dễ chạm quy tắc này.
- **Ảnh quá cỡ** mặc định bị thu nhỏ. Đặt `"transformations": {"oversized_image": "error"}` trên image block để API trả lỗi 400 thay vì âm thầm thu nhỏ; thông báo lỗi cho biết kích thước lớn nhất được chấp nhận. Block `document` (PDF) không nhận field này vì trang PDF được rasterize phía server [5]. Ảnh nằm trong `tool_result` (screenshot từ computer use, browser) vượt cỡ thì bị từ chối bằng lỗi validation thay vì thu nhỏ [5].
- **Tier phân giải theo model.** Model 4.7 trở lên dùng tier high-res (2.576 px, 4.784 token), model khác dùng tier standard (1.568 px, 1.568 token). Đổi model là đổi chi phí ảnh ([§1.2](#12-ảnh-tốn-bao-nhiêu-token)).
- Docs Claude khuyên đặt ảnh trước câu hỏi trong content.
- PDF: 600 trang mỗi request, 100 trang khi context window của request dưới 1M token. Không nhận PDF mã hoá. File text (.txt, .csv, .md) gửi qua Files API với MIME `text/plain` [6].

## 2.3 Google Gemini

### `media_resolution`

Gemini 3 và 4 cho đặt độ phân giải theo từng media part [9]. Token tối đa mỗi part:

| Mức | Ảnh | Video (mỗi frame) | Audio | PDF (mỗi trang) |
|---|---|---|---|---|
| `unspecified` (mặc định) | 1.120 | 70 | 25/giây | 560 |
| `low` | 280 | 70 | 25/giây | 280 + text gốc |
| `medium` | 560 | 70 | 25/giây | 560 + text gốc |
| `high` | 1.120 | 280 | 25/giây | 1.120 + text gốc |
| `ultra_high` | 2.240 | — | — | — |

Khuyến nghị trong docs [9]: ảnh dùng `high`; PDF dùng `medium`; video thông thường `low` (hoặc `medium`, hai mức như nhau với video); video cần đọc chữ dày đặc dùng `high`; audio để mặc định. `ultra_high` chỉ đặt được theo từng part, dành cho computer use hoặc khi test cho thấy rõ lợi ích so với `high`.

Ví dụ với Interactions API, đặt `resolution` cho từng ảnh [9]:

```python
from google import genai

client = genai.Client()
diagram = client.files.upload(file="so-do-mach.png")
photo = client.files.upload(file="anh-hien-truong.jpg")

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=[
        {"type": "text", "text": "Đối chiếu sơ đồ với ảnh hiện trường, liệt kê điểm khác."},
        {"type": "image", "uri": diagram.uri, "mime_type": diagram.mime_type, "resolution": "high"},
        {"type": "image", "uri": photo.uri, "mime_type": photo.mime_type, "resolution": "low"},
    ],
)
print(interaction.output_text)
print(interaction.usage)
```

### Các trang docs mâu thuẫn nhau

Docs Gemini có nhiều con số token khác nhau cho cùng loại media. Lý do là một số trang vẫn mô tả thế hệ model trước:

| Loại | Gemini 3, 4 (trang media resolution, giá) | Trang khác (thế hệ trước) |
|---|---|---|
| Audio | 25 token/giây [9][13] | 32 token/giây (trang audio, token counting) [10][12] |
| Video | 70 / 280 token mỗi frame [9] | 66 / 258 token mỗi frame + 32 token/giây audio (trang video) [11] |
| PDF | 560 token mỗi trang ở mặc định, text gốc miễn phí [8][9] | 258 token mỗi trang (phần "Technical details" của trang document processing) [8] |
| Ảnh | Theo mức `media_resolution` [9] | 258 token mỗi ảnh nhỏ hoặc mỗi tile 768 px [7][12] |

Trước khi đưa con số vào dự toán, gọi `countTokens` với file thật trên đúng model bạn dùng.

### Giới hạn và cách gửi

- Inline (base64) tối đa 20 MB cho cả request, gồm prompt và mọi file. File lớn hơn, hoặc file dùng lại nhiều lần, upload qua Files API [7].
- Tối đa 3.600 ảnh mỗi request [7].
- PDF: 50 MB hoặc 1.000 trang, áp dụng cho cả inline và Files API [8]. Các định dạng văn bản khác (TXT, HTML, Markdown) được đọc như text thuần. Docs ghi "document vision" chỉ hiểu có ý nghĩa với PDF [8].
- Video: Files API tối đa 20 GB (gói trả phí), 2 GB (gói miễn phí). Docs khuyên dùng Files API cho file trên 100 MB, video trên 10 phút, hoặc khi dùng lại. Phải chờ file chuyển sang trạng thái `ACTIVE` trước khi dùng [11].
- File trên Files API tự xoá sau 48 giờ [8]. Không lưu `file.name` vào database như một tham chiếu lâu dài; cần dùng lại sau 48 giờ thì upload lại.

Vòng đời một file video trong job xử lý: upload, chờ `ACTIVE` có hạn chót, coi `FAILED` là lỗi vĩnh viễn, và xoá file khi xong việc thay vì chờ hết 48 giờ:

```python
import logging
import time
from google import genai

client = genai.Client()
log = logging.getLogger(__name__)

def with_uploaded_file(path: str, use, *, deadline_s: float, poll_s: float):
    if deadline_s <= 0 or poll_s <= 0:
        raise ValueError("deadline_s và poll_s phải dương")
    f = client.files.upload(file=path)
    try:
        deadline = time.monotonic() + deadline_s
        while f.state.name == "PROCESSING":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{f.name} vẫn PROCESSING sau {deadline_s}s")
            time.sleep(min(poll_s, remaining))
            f = client.files.get(name=f.name)
        if f.state.name != "ACTIVE":           # FAILED: dừng, báo lỗi rõ ràng
            raise RuntimeError(f"{f.name} ở trạng thái {f.state.name}")
        return use(f)
    finally:
        try:
            client.files.delete(name=f.name)
        except Exception:
            # không ném lại để khỏi che lỗi chính của use(); file còn tồn tại tới khi hết 48 giờ
            log.exception("không xoá được %s", f.name)
```

`deadline_s` và `poll_s` cố ý không có giá trị mặc định: docs không công bố thời gian xử lý của Files API, nên hạn chót phải lấy từ thời gian xử lý đo được trên file thật của bạn (độ dài video, kích thước) cộng biên an toàn. Lỗi xoá được ghi log thay vì nuốt im lặng; nếu dữ liệu nhạy cảm, gắn alert vào log này, vì file sẽ nằm trên server tới 48 giờ.

Nếu job được retry, mỗi lần retry upload một file mới, nên xoá trong `finally` để retry không để lại file rác chiếm quota lưu trữ.

## 2.4 Model open-weight: Qwen-VL và các model vision khác

Từ 2026, vision không còn là một dòng model riêng ở nhiều họ open-weight: mọi kích cỡ Qwen3.5 nhận ảnh và video, Gemma 4 đọc ảnh ở mọi kích cỡ, Ministral 3 có vision encoder đi kèm. Đây là tổng hợp của một bài so sánh độc lập đã đối chiếu với model card tháng 10/2026, nên hãy kiểm tra lại model card của model bạn chọn [26]. Dòng Qwen3-VL (ra mắt nửa sau 2025) vẫn là một trong các lựa chọn chuyên cho vision. Model card ghi OCR 32 ngôn ngữ, context gốc 256K mở rộng tới 1M, có bản Instruct và bản Thinking, kiến trúc dense và MoE [15].

Khi tự host model vision, ba thứ cần cấu hình mà API đóng làm thay bạn:

**Ngân sách pixel mỗi ảnh.** Processor của Qwen thu hoặc phóng ảnh về khoảng `[min_pixels, max_pixels]`. Qwen3-VL nén 32 lần theo mỗi chiều (patch 16 px, gộp 2×2), nên 256–1.280 token mỗi ảnh tương ứng `min_pixels = 256×32×32` và `max_pixels = 1280×32×32`. Ngân sách của video đặt riêng [15]. Để mặc định, một ảnh lớn có thể chiếm hàng nghìn token và đẩy KV cache ra khỏi GPU.

**Số media mỗi request.** vLLM giới hạn bằng `--limit-mm-per-prompt` (ví dụ `image=4,video=1`) và cấp phát bộ nhớ theo giới hạn đó. Đặt modality không dùng về 0 để không tốn bộ nhớ cho nó. Từ bản gần đây, có thể khai báo cả kích thước dự kiến (`{"count": ..., "width": ..., "height": ...}`) để vLLM dự trù bộ nhớ theo media thật thay vì theo mức tối đa của model [17]. Model vision chỉ dùng cho text thì bật `--language-model-only` để bỏ phần vision và lấy bộ nhớ cho KV cache [17].

**Tham số processor theo request.** vLLM nhận `mm_processor_kwargs` theo request, nhưng có một cờ để từ chối giá trị lạ từ client, vì các tham số này thay đổi được lượng tài nguyên mà bước tiền xử lý dùng [17]. Với API công khai, giữ cờ này ở chế độ từ chối và đặt ngân sách pixel ở phía server.

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --mm-processor-kwargs '{"max_pixels": 1310720}'
```

`1310720 = 1280 × 32 × 32`, tức khoảng 1.280 token mỗi ảnh. Cú pháp cờ đối chiếu với docs vLLM [17]. Kiểm tra tên khoá `max_pixels` với processor của phiên bản model bạn dùng: Qwen3-VL lưu giới hạn này trong `size.longest_edge` của `preprocessor_config.json` [15].

## 2.5 OCR và document parsing chuyên dụng

Trước khi đưa tài liệu cho LLM, có thể chuyển nó thành text có cấu trúc (Markdown, JSON có bảng) bằng một công cụ chuyên dụng:

| Công cụ | Loại | Điểm đáng chú ý |
|---|---|---|
| Docling (IBM) | Mã nguồn mở, chạy local | Code MIT. Đọc PDF, DOCX, PPTX, XLSX, HTML, ảnh, audio. Có pipeline dùng VLM nhỏ (`granite-docling-258M`) [18] |
| Marker (Datalab) | Mã nguồn mở + chế độ LLM | Code Apache-2.0; trọng số model theo giấy phép OpenRAIL-M sửa đổi, miễn phí cho nghiên cứu, cá nhân và startup dưới $5 triệu doanh thu hoặc vốn gọi được. `--use_llm` dùng thêm một LLM (Gemini, Claude, OpenAI-compatible, Ollama...) để sửa bảng và công thức. Tự báo 76,0% ở chế độ cân bằng trên olmOCR-bench [19] |
| Mistral OCR | API | Bản ra mắt 03/2025 (`mistral-ocr-latest`) đã được thay bằng thế hệ mới theo trang tin của Mistral. Giá ban đầu khoảng 1.000 trang/USD [20] |
| AWS Textract | Managed | Giá theo trang, theo loại tính năng: text khoảng $0,0015/trang, bảng $0,015, form $0,05 ở mức 1 triệu trang đầu (giá ví dụ ở us-west-2) [21] |
| Azure AI Document Intelligence, LlamaParse, Unstructured | Managed / mã nguồn mở | Chưa xác minh giá trong đợt biên soạn này |

Số liệu benchmark trên đây là do nhà phát triển tự báo cáo. So sánh giữa các công cụ chỉ có ý nghĩa khi chạy trên tài liệu của chính bạn ([§3.6](#36-eval-cho-tác-vụ-vision)).

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Tiền xử lý ảnh: độ phân giải, cắt, xoay

Mục tiêu của tiền xử lý: ảnh model nhìn thấy phải đủ rõ cho tác vụ, và không lớn hơn mức cần thiết.

**Xoay theo EXIF trước tiên.** Ảnh chụp bằng điện thoại thường lưu pixel theo chiều cảm biến, kèm một cờ EXIF `Orientation` để trình xem ảnh tự xoay. Docs của các provider không nêu việc họ có áp dụng cờ này hay không. Tự xoay ở phía bạn rồi bỏ EXIF là cách chắc chắn nhất.

**Thu nhỏ về đúng trần của provider, hoặc nhỏ hơn.** Gửi ảnh lớn hơn trần không làm model thấy rõ hơn, chỉ tốn băng thông và có nguy cơ vượt giới hạn dung lượng. Tự thu nhỏ còn giúp bạn biết chính xác model nhìn thấy gì, và chọn được thuật toán resize tốt hơn.

**Cắt vùng cần đọc thay vì gửi toàn bộ.** Với chữ nhỏ, cắt vùng chứa chữ (chân trang hoá đơn, bảng) rồi gửi riêng với độ phân giải đầy đủ thường hiệu quả hơn gửi ảnh lớn bị thu nhỏ. Hai ảnh: một ảnh toàn cảnh nhỏ cho ngữ cảnh, một ảnh cắt cho chi tiết.

**Nén hợp lý.** Với ảnh chụp tài liệu, JPEG chất lượng khoảng 85–90 là điểm xuất phát hợp lý, không phải con số đã kiểm chứng: chọn mức cuối cùng bằng eval OCR trên ảnh thật của bạn, so độ chính xác trích xuất ở vài mức quality. Nén quá mạnh tạo nhiễu khối quanh chữ, và dấu tiếng Việt thường hỏng trước chữ cái. Ảnh chụp màn hình giữ PNG: chữ render sắc nét, nền phẳng, JPEG làm nhoè viền chữ mà file lại không nhỏ hơn bao nhiêu.

```python
from io import BytesIO
from PIL import Image, ImageOps

def prepare_image(raw: bytes, *, fmt: str = "JPEG", max_long_edge: int = 1568,
                  quality: int = 88) -> tuple[bytes, str]:
    """Xoay theo EXIF, bỏ metadata, thu nhỏ cạnh dài về max_long_edge.
    fmt="JPEG" cho ảnh chụp, fmt="PNG" cho screenshot. Trả về (bytes, media_type)."""
    img = ImageOps.exif_transpose(Image.open(BytesIO(raw)))   # áp Orientation rồi bỏ cờ
    if max(img.size) > max_long_edge:
        img.thumbnail((max_long_edge, max_long_edge), Image.Resampling.LANCZOS)
    out = BytesIO()
    if fmt == "PNG":
        if img.mode not in ("RGB", "RGBA", "L", "LA"):
            img = img.convert("RGBA")
        img.save(out, format="PNG", optimize=True)            # giữ alpha, không ghi EXIF
        return out.getvalue(), "image/png"
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")                             # vùng trong suốt thành nền trắng,
        bg = Image.new("RGB", rgba.size, (255, 255, 255))      # không phải nền đen
        bg.paste(rgba, mask=rgba.getchannel("A"))
        img = bg
    img = img.convert("RGB")
    img.save(out, format="JPEG", quality=quality, optimize=True)  # không ghi lại EXIF
    return out.getvalue(), "image/jpeg"
```

`max_long_edge` chọn theo provider và model: 1.568 cho Claude standard tier, 2.576 cho Claude high-res, 2.048 cho OpenAI `detail: "high"`. Với Gemini 3, kích thước ảnh không đổi số token, nhưng gửi ảnh nhỏ hơn vẫn giảm băng thông và giúp giữ dưới giới hạn 20 MB inline. Bỏ metadata còn có lợi về bảo mật: EXIF có thể chứa toạ độ GPS và tên thiết bị của người dùng.

## 3.2 Pipeline xử lý tài liệu: parse trước hay đưa thẳng cho model

Có ba kiến trúc chính:

| Kiến trúc | Cách làm | Hợp với | Rủi ro |
|---|---|---|---|
| **Parse trước** | Docling, Marker hoặc OCR chuyển tài liệu thành Markdown/JSON, LLM chỉ đọc text | PDF số có text layer tốt; khối lượng lớn; cần chạy local; cần kiểm tra lại được từng bước | Lỗi parse (bảng lệch cột, mất thứ tự đọc) đi thẳng vào LLM mà không ai thấy |
| **Đưa thẳng cho VLM** | Gửi ảnh hoặc PDF cho model kèm schema | Ảnh chụp, scan, bố cục phức tạp, chart; khối lượng vừa phải | Chi phí token ảnh; model đoán khi không đọc rõ; khó biết lỗi đến từ đâu |
| **Kết hợp** | Parse lấy text, đồng thời gửi ảnh trang cho VLM; hoặc parse trước, chỉ gửi trang khó cho VLM | Tài liệu hỗn hợp; cần độ chính xác cao | Phức tạp hơn; phải xử lý khi hai nguồn mâu thuẫn |

Cách chọn dựa trên dữ liệu thật:

1. **Phân loại tài liệu đầu vào trước.** PDF có text layer hay không thì kiểm tra được bằng một thư viện PDF trong vài mili giây. Tài liệu có text layer tốt đi đường parse. Tài liệu scan hoặc ảnh chụp đi đường VLM.
2. **Kiểm tra chất lượng text layer**, không chỉ có hay không. Tỉ lệ ký tự lạ, tỉ lệ chữ tiếng Việt có dấu, số trang có text rỗng là các tín hiệu rẻ để phát hiện font nhúng hỏng.
3. **Dùng VLM cho trang khó, không phải cho mọi trang.** Một báo cáo 200 trang có thể chỉ có 15 trang bảng phức tạp. Gửi 15 trang đó cho VLM, phần còn lại dùng text parse, rẻ hơn nhiều lần.
4. **Giữ số trang và vị trí** trong mọi bước để trích dẫn được và để người kiểm tra mở đúng chỗ.

Với dữ liệu nhạy cảm (hồ sơ y tế, giấy tờ tuỳ thân), pipeline chạy local (Docling, Marker, model open-weight) hoặc hợp đồng zero data retention với provider là yêu cầu trước khi bàn đến chất lượng.

## 3.3 Trích xuất có cấu trúc từ ảnh và PDF

Trích xuất từ ảnh dùng cùng kỹ thuật với trích xuất từ text: schema, structured output và validation nhiều lớp ([Structured Output §3.3](../structured-output/#33-validation-nhiều-lớp-và-retry)). Phần riêng của ảnh là xử lý trường hợp model *không đọc được*:

- **Cho phép giá trị "không đọc được" trong schema.** Mỗi trường quan trọng nên nhận `null` hoặc có một cờ `legible: false`. Schema bắt buộc mọi trường phải có giá trị sẽ đẩy model đến chỗ đoán.
- **Yêu cầu trích nguyên văn trước khi chuẩn hoá.** Ví dụ trường `total_raw: "1.250.000đ"` đi kèm `total: 1250000`. Khi sai, bạn biết lỗi nằm ở bước đọc hay bước chuẩn hoá.
- **Yêu cầu vị trí.** Số trang, và bounding box nếu model hỗ trợ. Người kiểm tra mở đúng chỗ, và code kiểm tra được giá trị có nằm trong vùng hợp lý không.
- **Kiểm tra chéo bằng code.** Tổng các dòng hàng phải bằng tổng tiền trước thuế; tiền thuế phải bằng thuế suất nhân giá trị trước thuế; mã số thuế phải đúng độ dài và checksum; ngày phải hợp lệ. Hoá đơn sai một trong các điều kiện này thì đưa vào hàng đợi người duyệt thay vì ghi thẳng vào hệ thống.
- **Đừng giao việc đếm cho model.** Số dòng hàng là `len(items)` của JSON đã trích. Nếu nghi model bỏ sót dòng, so tổng tiền.

```python
from decimal import Decimal
from pydantic import BaseModel, Field

class LineItem(BaseModel):
    description: str
    quantity: Decimal | None
    unit_price: Decimal | None
    amount: Decimal | None

class Invoice(BaseModel):
    invoice_number: str | None = Field(description="Nguyên văn như trên chứng từ; null nếu không đọc được")
    tax_code: str | None
    issue_date: str | None = Field(description="YYYY-MM-DD")
    subtotal: Decimal | None
    vat_amount: Decimal | None
    total: Decimal | None
    total_raw: str | None = Field(description="Chuỗi tổng tiền nguyên văn trên chứng từ")
    items: list[LineItem]
    page: int
    unreadable_fields: list[str] = Field(description="Tên các trường không đọc rõ")

def needs_review(inv: Invoice) -> list[str]:
    issues = list(inv.unreadable_fields)
    amounts = [i.amount for i in inv.items]
    if inv.subtotal is not None and None not in amounts and sum(amounts) != inv.subtotal:
        issues.append("tổng dòng hàng khác subtotal")
    if None not in (inv.subtotal, inv.vat_amount, inv.total) and inv.subtotal + inv.vat_amount != inv.total:
        issues.append("subtotal + VAT khác total")
    return issues
```

Schema này dùng được với structured output của cả ba provider sau khi chuyển sang JSON Schema. Các ràng buộc riêng của từng provider (field optional, kiểu số thập phân) ở [Structured Output §3.2](../structured-output/#32-schema-chạy-được-trên-nhiều-provider).

## 3.4 RAG đa phương thức

Khi kho tài liệu có nhiều bảng, biểu đồ và ảnh, có ba cách đưa chúng vào retrieval:

- **Chuyển mọi thứ về text.** Parse bảng thành Markdown, nhờ VLM viết mô tả cho ảnh và chart, rồi embed như text thường. Đơn giản và dùng lại được hạ tầng sẵn có. Đổi lại, mô tả do VLM viết có thể sai, và số liệu trong chart dễ bị làm tròn hoặc bịa. Lưu mô tả kèm liên kết tới ảnh gốc, và khi trả lời câu hỏi về số liệu, đưa ảnh gốc vào context thay vì chỉ dùng mô tả.
- **Embedding đa phương thức.** Embed ảnh và text vào cùng một không gian vector (ví dụ `gemini-embedding-2`, Cohere embed v5) để truy vấn bằng text tìm được ảnh. Chi tiết model và giá ở [Embeddings §2.8](../embeddings/#28-multimodal-embedding).
- **Retrieval trên ảnh trang.** ColPali (Faysse và cộng sự, 2024) embed thẳng ảnh của từng trang bằng VLM, mỗi trang thành nhiều vector, và so khớp kiểu late interaction như ColBERT. Cách này bỏ qua OCR và chunking [25]. Đổi lại, index lớn hơn nhiều so với embedding text, vì mỗi trang là hàng trăm vector.

Phần chunking bảng, giữ header khi chia bảng dài, và trích dẫn nguồn nằm ở [RAG nâng cao §2.7](../advanced-rag/#27-rag-đa-phương-thức-bảng-ảnh-pdf-phức-tạp).

## 3.5 Chi phí và caching khi gửi lại ảnh nhiều lượt

Trong hội thoại nhiều lượt, ảnh ở lượt đầu được gửi lại ở mọi lượt sau như một phần của lịch sử, và mỗi lần đều tính token. Một trợ lý mà người dùng gửi năm ảnh chụp màn hình rồi hỏi thêm mười câu sẽ trả tiền cho năm ảnh đó mười lần.

- **Prompt caching giải quyết phần lớn chi phí này**, miễn là ảnh nằm trong phần prefix không đổi. Ảnh được cache như mọi nội dung khác. Đổi một ảnh ở đầu lịch sử, hoặc gửi lại ảnh đã được xử lý lại (resize khác, nén khác cho ra byte khác) là mất cache cho mọi thứ phía sau. Cơ chế cache theo provider ở [Multi-turn §1.4](../multi-turn-conversation/#14-kinh-tế-học-của-prompt-caching) và [§3.3](../multi-turn-conversation/#33-những-thứ-âm-thầm-làm-vỡ-cache).
- **Tiền xử lý phải xác định.** Cùng một ảnh đầu vào phải cho ra cùng byte đầu ra, nếu không cache sẽ không bao giờ hit. Lưu ảnh đã xử lý (hoặc `file_id`) và gửi lại đúng bản đó thay vì xử lý lại từ ảnh gốc ở mỗi lượt.
- **Dùng Files API cho ảnh và tài liệu dùng lại nhiều lần.** Không giảm token, nhưng giảm băng thông và tránh giới hạn dung lượng request (128 MB của Claude, 20 MB inline của Gemini).
- **Thay ảnh cũ bằng bản trích xuất.** Khi ảnh đã được trích xuất thành dữ liệu và cuộc hội thoại không cần nhìn lại ảnh, thay ảnh trong lịch sử bằng kết quả trích xuất ở thời điểm cắt context. Việc này làm mất cache một lần, nhưng giảm chi phí cho mọi lượt sau.
- **Agent chụp màn hình:** giới hạn số ảnh giữ lại trong context (ví dụ chỉ vài ảnh gần nhất). Claude còn có quy tắc hơn 20 ảnh thì cạnh dài ≤ 3.000 px ([§2.2](#22-claude-anthropic)).

## 3.6 Eval cho tác vụ vision

Bộ eval cho vision cần đại diện cho *điều kiện chụp* chứ không chỉ cho *loại tài liệu*. Hoá đơn in sạch scan ở 300 DPI và hoá đơn nhàu chụp dưới đèn vàng là hai bài toán khác nhau.

- **Thu mẫu theo nguồn thật.** Lấy từ dữ liệu production, giữ nguyên phân bố: tỉ lệ ảnh chụp và PDF số, tỉ lệ ảnh nghiêng, mờ, có dấu đỏ, chữ viết tay. Bộ mẫu chọn toàn ảnh đẹp sẽ cho số đẹp hơn thực tế.
- **Chấm theo trường, không theo cả tài liệu.** Đo độ chính xác riêng cho từng trường (số hoá đơn, ngày, tổng tiền, mã số thuế, từng dòng hàng). Một tài liệu đúng 9/10 trường vẫn là một tài liệu sai, nhưng con số theo trường cho bạn biết cần sửa ở đâu.
- **Tách ba loại lỗi:** đọc sai (model trả giá trị khác thật), bỏ sót (trả `null` khi có giá trị), và bịa (trả giá trị khi tài liệu không có). Loại cuối nguy hiểm nhất cho pipeline tự động. Nên có mẫu cố ý thiếu trường để đo nó.
- **Đo trên tiếng Việt có dấu.** Tỉ lệ sai ký tự có dấu là metric riêng, vì một tên công ty sai một dấu vẫn có thể "gần đúng" theo các metric so khớp mờ.
- **Đo theo cấu hình độ phân giải.** Chạy cùng bộ eval ở vài mức `detail` / `media_resolution` / kích thước ảnh. Thường có một điểm mà tăng thêm không cải thiện chất lượng nhưng chi phí vẫn tăng. Docs Gemini nêu điểm này cho PDF ở mức `medium` [9].
- **Benchmark công khai chỉ để tham khảo.** DocVQA, ChartQA, OmniDocBench, olmOCR-bench đo trên tài liệu tiếng Anh và tiếng Trung là chủ yếu, và nhiều số liệu là do nhà phát triển tự báo cáo.

Phương pháp xây golden dataset và đọc kết quả có ý nghĩa thống kê nằm ở [Evaluation](../evaluation/).

## 3.7 Failure modes thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Đọc sai chữ nhỏ, bỏ dòng cuối bảng | Ảnh bị thu nhỏ về trần của provider (Claude 1.568 / 2.576 px; OpenAI theo ngân sách patch; Gemini theo `media_resolution`) | Cắt vùng chữ và gửi riêng; tăng `detail` / `media_resolution`; dùng `transformations.oversized_image: "error"` (Claude) để biết khi ảnh bị thu nhỏ [4][5] |
| Model trả giá trị hợp lý nhưng sai cho trường mờ | Schema bắt buộc trường; model đoán | Cho phép `null` và danh sách trường không đọc được; kiểm tra chéo bằng code |
| Claude: `invalid_request_error` nhắc "many-image requests" | Hơn 20 ảnh/document block (kể cả lượt trước và `tool_result`), có ảnh cạnh dài > 3.000 px | Thu cạnh dài ≤ 3.000 px hoặc giảm số ảnh trong context [4] |
| Request bị từ chối vì quá lớn dù ít ảnh | Ảnh gốc nặng (PNG scan, ảnh 12 MP) | Tiền xử lý và nén; dùng Files API; chú ý 128 MB (Claude), 20 MB inline (Gemini), 512 MB (OpenAI) |
| Ảnh chụp dọc bị model đọc như ảnh ngang | Cờ EXIF `Orientation` không được áp dụng | `ImageOps.exif_transpose` trước khi gửi ([§3.1](#31-tiền-xử-lý-ảnh-độ-phân-giải-cắt-xoay)) |
| Chi phí ảnh tăng mạnh sau khi đổi model Claude | Model 4.7 trở lên dùng tier high-res, tới khoảng gấp ba token | Thu nhỏ ảnh về kích thước tác vụ thật sự cần [4] |
| OpenAI: token ảnh cao hơn dự tính trên `gpt-6-astra` | `detail` mặc định `auto` tương đương `original` | Đặt `detail: "high"` hoặc `"low"` tường minh [1] |
| Chi phí PDF cao gấp nhiều lần ước tính | Claude tính cả text lẫn ảnh trang; PDF dày chữ khoảng 3.000 token text mỗi trang | Đo `usage` trên mẫu; chỉ gửi trang cần thiết; parse trước với trang chỉ có chữ [6] |
| Gemini: số token audio, video, PDF không khớp tài liệu | Trang docs mô tả thế hệ model khác | Đo bằng `countTokens` trên model thật ([§2.3](#23-google-gemini)) |
| Text trích từ PDF lỗi font, mất dấu | Font nhúng ánh xạ ký tự sai | Kiểm tra chất lượng text layer; với trang lỗi, dùng ảnh trang thay vì text |
| Bảng bị lệch cột sau khi parse | Parser text không giữ cấu trúc bảng | Parser có mô hình bảng (Docling, Marker `--use_llm`) hoặc gửi ảnh trang cho VLM kèm schema |
| Đếm sai số dòng, số vật | Điểm yếu có hệ thống của VLM [22] | Đếm phần tử JSON đã trích; đối chiếu tổng tiền |
| Trả lời sai về vị trí trái/phải, trên/dưới | Điểm yếu suy luận không gian [23][24] | Yêu cầu bounding box rồi kiểm bằng code |
| Cache không hit dù gửi lại cùng ảnh | Tiền xử lý không xác định, tạo byte khác ở mỗi lượt | Lưu và gửi lại đúng bản đã xử lý hoặc `file_id` |
| Gemini: video dài tốn nhiều token | 1 FPS mặc định, token mỗi frame cộng audio | Giảm FPS, cắt đoạn, hoặc dùng chế độ agentic với video dài [11] |
| vLLM: OOM hoặc KV cache nhỏ khi bật model vision | Bộ nhớ dự trù theo số media và kích thước tối đa | `--limit-mm-per-prompt` với số lượng và kích thước thật; tắt modality không dùng [17] |
| Model làm theo chữ ghi trong ảnh ("bỏ qua chỉ dẫn trước") | Prompt injection qua ảnh hoặc tài liệu | Coi ảnh là dữ liệu không tin cậy; giới hạn quyền tool ([Bảo mật LLM §1.3](../llm-security/#13-indirect-injection-qua-tài-liệu-web-email-tool-result)) |

## 3.8 Checklist

**Đầu vào**

- [ ] Ảnh được xoay theo EXIF, bỏ metadata, thu nhỏ về trần của provider hoặc nhỏ hơn.
- [ ] Vùng chữ nhỏ được cắt và gửi riêng ở độ phân giải đầy đủ.
- [ ] Tiền xử lý xác định: cùng đầu vào cho cùng byte đầu ra.
- [ ] PDF được phân loại theo có text layer hay không, và text layer được kiểm tra chất lượng.
- [ ] Đặt `detail` (OpenAI) hoặc `media_resolution` (Gemini) tường minh.

**Giới hạn**

- [ ] Kiểm tra số ảnh, dung lượng mỗi ảnh và tổng request trước khi gửi.
- [ ] Claude: hơn 20 ảnh/document trong context thì cạnh dài ≤ 3.000 px.
- [ ] File dùng lại nhiều lần đi qua Files API.

**Trích xuất**

- [ ] Schema cho phép `null` và có danh sách trường không đọc được.
- [ ] Có giá trị nguyên văn cạnh giá trị đã chuẩn hoá cho các trường số và ngày.
- [ ] Có kiểm tra chéo bằng code (tổng tiền, thuế, checksum, ngày hợp lệ) và hàng đợi người duyệt.
- [ ] Không giao việc đếm hay xác định vị trí cho model khi kiểm tra được bằng code.

**Vận hành**

- [ ] Log token ảnh/PDF/audio mỗi request; đối chiếu với ước tính.
- [ ] Đo lại bằng `countTokens` hoặc `usage` sau mỗi lần đổi model.
- [ ] Eval theo trường, có mẫu ảnh xấu, mẫu thiếu trường và mẫu tiếng Việt có dấu.
- [ ] Ảnh và tài liệu từ người dùng được coi là dữ liệu không tin cậy.

---

## Nguồn tham khảo

### Tài liệu provider

| # | Tài liệu | URL |
|---|---|---|
| 1 | OpenAI: Images and vision | [developers.openai.com/api/docs/guides/images-vision](https://developers.openai.com/api/docs/guides/images-vision) |
| 2 | OpenAI: File inputs (PDF) | [developers.openai.com/api/docs/guides/pdf-files](https://developers.openai.com/api/docs/guides/pdf-files) |
| 3 | OpenAI: API pricing | [developers.openai.com/api/docs/pricing](https://developers.openai.com/api/docs/pricing) |
| 4 | Claude: Vision | [platform.claude.com/docs/en/build-with-claude/vision](https://platform.claude.com/docs/en/build-with-claude/vision) |
| 5 | Claude: Vision coordinates (cách resize, `oversized_image`) | [platform.claude.com/docs/en/build-with-claude/vision-coordinates](https://platform.claude.com/docs/en/build-with-claude/vision-coordinates) |
| 6 | Claude: PDF support | [platform.claude.com/docs/en/build-with-claude/pdf-support](https://platform.claude.com/docs/en/build-with-claude/pdf-support) |
| 7 | Gemini API: Image understanding | [aistudio.google.com/docs/image-understanding](https://aistudio.google.com/docs/image-understanding) |
| 8 | Gemini API: Document processing | [aistudio.google.com/docs/document-processing](https://aistudio.google.com/docs/document-processing) |
| 9 | Gemini API: Media resolution | [aistudio.google.com/docs/media-resolution](https://aistudio.google.com/docs/media-resolution) |
| 10 | Gemini API: Audio understanding | [aistudio.google.com/docs/audio](https://aistudio.google.com/docs/audio) |
| 11 | Gemini API: Video understanding | [aistudio.google.com/docs/video-understanding](https://aistudio.google.com/docs/video-understanding) |
| 12 | Gemini API: Understand and count tokens | [aistudio.google.com/docs/tokens](https://aistudio.google.com/docs/tokens) |
| 13 | Gemini API: Pricing | [aistudio.google.com/docs/pricing](https://aistudio.google.com/docs/pricing) |
| 14 | xAI: Image understanding | [docs.x.ai/developers/model-capabilities/images/understanding](https://docs.x.ai/developers/model-capabilities/images/understanding) |

### Model open-weight và công cụ

| # | Tài liệu | URL |
|---|---|---|
| 15 | Qwen3-VL: model card và README (ngân sách pixel, `preprocessor_config.json`) | [huggingface.co/Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [github.com/QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) |
| 16 | Qwen2.5-VL: model card | [huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| 17 | vLLM: Conserving memory (multi-modal input limits); MultiModalConfig; `vllm serve` | [docs.vllm.ai/en/latest/configuration/conserving_memory/](https://docs.vllm.ai/en/latest/configuration/conserving_memory/), [docs.vllm.ai/en/latest/api/vllm/config/multimodal/](https://docs.vllm.ai/en/latest/api/vllm/config/multimodal/), [docs.vllm.ai/en/stable/cli/serve/](https://docs.vllm.ai/en/stable/cli/serve/) |
| 18 | Docling | [github.com/docling-project/docling](https://github.com/docling-project/docling) |
| 19 | Marker (Datalab) | [github.com/datalab-to/marker](https://github.com/datalab-to/marker) |
| 20 | Mistral OCR (trang tin ra mắt) | [mistral.ai/news/mistral-ocr](https://mistral.ai/news/mistral-ocr) |
| 21 | AWS Textract pricing | [aws.amazon.com/textract/pricing/](https://aws.amazon.com/textract/pricing/) |
| 26 | vdf.ai: Best local vision models (2026), so sánh model card tháng 10/2026 (nguồn thứ cấp) | [vdf.ai/blog/best-local-vision-models/](https://vdf.ai/blog/best-local-vision-models/) |

### Paper

| # | Paper | URL |
|---|---|---|
| 22 | Rahmanzadehgervi et al. (2024), *Vision language models are blind*, arXiv:2407.06581 | [arxiv.org/abs/2407.06581](https://arxiv.org/abs/2407.06581) |
| 23 | Kamath et al. (2023), *What's "up" with vision-language models?*, arXiv:2310.19785 | [arxiv.org/abs/2310.19785](https://arxiv.org/abs/2310.19785) |
| 24 | Stogiannidis et al. (2025), *Mind the Gap: Benchmarking Spatial Reasoning in Vision-Language Models*, arXiv:2503.19707 | [arxiv.org/abs/2503.19707](https://arxiv.org/abs/2503.19707) |
| 25 | Faysse et al. (2024), *ColPali: Efficient Document Retrieval with Vision Language Models*, arXiv:2407.01449 | [arxiv.org/abs/2407.01449](https://arxiv.org/abs/2407.01449) |

---

Thông tin API trong tài liệu được ghi nhận tại **tháng 11/2026** từ tài liệu chính thức của provider, model card và docs của vLLM. Các con số benchmark của công cụ OCR là do nhà phát triển tự báo cáo.

Các điểm chưa xác minh được: số token mỗi trang PDF của OpenAI (docs không nêu); công thức token và giá trị `detail` hợp lệ của Grok; việc provider có tự áp dụng EXIF orientation hay không; giá Azure AI Document Intelligence, LlamaParse và Unstructured; số liệu OCR tiếng Việt của model 2026. Cột OpenAI trong bảng so sánh token ở [§1.2](#12-ảnh-tốn-bao-nhiêu-token) là phép tính theo công thức, chưa đối chiếu với request thật.
