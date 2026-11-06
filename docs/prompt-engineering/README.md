# Prompt engineering cho production

Một đội đang vận hành chatbot chăm sóc khách hàng cho cửa hàng điện tử. Có lần bot trả lời "đơn của anh đang trên đường giao" trong khi chưa hề tra cứu đơn. Để sửa, một kỹ sư thêm vào system prompt: `QUAN TRỌNG: LUÔN LUÔN tra cứu đơn hàng bằng tool trước khi trả lời.` Bot không còn bịa trạng thái đơn nữa, và bản sửa được merge.

Tuần sau, khách hỏi "cửa hàng mở cửa mấy giờ?" thì bot đòi mã đơn trước khi trả lời. Số lần gọi tool tra cứu đơn tăng gấp ba. Không ai nghĩ đến câu vừa thêm, vì không có test nào bao phủ câu hỏi về giờ mở cửa. Một tháng sau đó, đội nâng lên model mới. Đoạn code điền sẵn `{` vào lượt assistant để ép JSON bắt đầu trả lỗi 400, tham số `temperature: 0` cũng bị từ chối, và câu trả lời dài gấp rưỡi so với trước.

Cả ba sự cố đều có chung một nguyên nhân: prompt là một phần của chương trình, nhưng không có compiler, không có type, và hành vi của nó phụ thuộc vào đúng model đang chạy nó. Viết prompt cho bản demo nghĩa là tìm câu chữ chạy được trên vài ví dụ. Viết prompt cho production nghĩa là quản lý một thành phần có thể hỏng âm thầm khi bạn sửa một câu, khi đổi model, hoặc khi provider bỏ đi một tính năng.

Tài liệu này không lặp lại các mẹo đã có trong mọi bài nhập môn. Nó tập trung vào những điều ít được nhắc tới: cơ chế đằng sau các mẹo đó, những chỗ các provider khuyên ngược nhau, những kỹ thuật từng phổ biến nhưng nay đã bị gỡ khỏi API, và cách quản lý prompt để thay đổi nào cũng đo được.

- **Phần 1 — Nền tảng:** vì sao prompt nhạy cảm đến vậy, phân quyền giữa các loại message, điều gì thật sự giúp model làm đúng, few-shot dạy model những gì, và vì sao prompt gắn chặt với model.
- **Phần 2 — Kỹ thuật và công cụ:** cấu trúc prompt, xử lý tài liệu dài, kiểm soát format khi không còn prefill và temperature, chaining, tự động tối ưu prompt, công cụ quản lý.
- **Phần 3 — Thiết kế và vận hành:** lưu prompt ở đâu, versioning, regression test, chuyển model, prompt tiếng Việt. Khi cần tìm nguyên nhân một lỗi cụ thể, bắt đầu từ [bảng ở §3.7](#37-failure-modes-thường-gặp).

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API. Hướng dẫn của provider thay đổi theo từng thế hệ model; những điểm chỉ xác minh qua nguồn thứ cấp được đánh dấu tại chỗ.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Prompt là một phần của chương trình](#11-prompt-là-một-phần-của-chương-trình)
  - [1.2 System prompt, user message và vai trò của từng phần](#12-system-prompt-user-message-và-vai-trò-của-từng-phần)
  - [1.3 Cụ thể, có ngữ cảnh, có lý do: điều model cần để làm đúng](#13-cụ-thể-có-ngữ-cảnh-có-lý-do-điều-model-cần-để-làm-đúng)
  - [1.4 Few-shot: khi nào giúp, khi nào làm model bắt chước sai](#14-few-shot-khi-nào-giúp-khi-nào-làm-model-bắt-chước-sai)
  - [1.5 Prompt và model: vì sao prompt tốt cho model này chưa chắc tốt cho model khác](#15-prompt-và-model-vì-sao-prompt-tốt-cho-model-này-chưa-chắc-tốt-cho-model-khác)
- [Phần 2 — Kỹ thuật và công cụ](#phần-2--kỹ-thuật-và-công-cụ)
  - [2.1 Cấu trúc prompt: section, tag XML, markdown](#21-cấu-trúc-prompt-section-tag-xml-markdown)
  - [2.2 Đưa tài liệu dài vào prompt: thứ tự, trích dẫn](#22-đưa-tài-liệu-dài-vào-prompt-thứ-tự-trích-dẫn)
  - [2.3 Kiểm soát độ dài, giọng văn và format](#23-kiểm-soát-độ-dài-giọng-văn-và-format)
  - [2.4 Prompt chaining, chain-of-thought và tách tác vụ](#24-prompt-chaining-chain-of-thought-và-tách-tác-vụ)
  - [2.5 Tự động tối ưu prompt: DSPy, GEPA, công cụ của provider](#25-tự-động-tối-ưu-prompt-dspy-gepa-công-cụ-của-provider)
  - [2.6 Công cụ quản lý prompt: registry, versioning, template](#26-công-cụ-quản-lý-prompt-registry-versioning-template)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Lưu prompt ở đâu: trong code, file template hay prompt registry](#31-lưu-prompt-ở-đâu-trong-code-file-template-hay-prompt-registry)
  - [3.2 Versioning và review thay đổi prompt](#32-versioning-và-review-thay-đổi-prompt)
  - [3.3 Thứ tự nội dung để tận dụng prompt caching](#33-thứ-tự-nội-dung-để-tận-dụng-prompt-caching)
  - [3.4 Regression test khi đổi prompt](#34-regression-test-khi-đổi-prompt)
  - [3.5 Di chuyển prompt khi đổi model hoặc nâng version](#35-di-chuyển-prompt-khi-đổi-model-hoặc-nâng-version)
  - [3.6 Đa ngôn ngữ: prompt tiếng Anh hay tiếng Việt](#36-đa-ngôn-ngữ-prompt-tiếng-anh-hay-tiếng-việt)
  - [3.7 Failure modes thường gặp](#37-failure-modes-thường-gặp)
  - [3.8 Checklist](#38-checklist)
- [Nguồn tham khảo](#nguồn-tham-khảo)

---

## Tra cứu nhanh

### Khung một system prompt

Thứ tự dưới đây kết hợp khuyến nghị của các provider ([§2.1](#21-cấu-trúc-prompt-section-tag-xml-markdown), [§2.2](#22-đưa-tài-liệu-dài-vào-prompt-thứ-tự-trích-dẫn)) với yêu cầu của prompt caching ([§3.3](#33-thứ-tự-nội-dung-để-tận-dụng-prompt-caching)): phần ổn định đứng trước, phần thay đổi theo request đứng sau, câu hỏi đứng cuối.

```text
[system — ổn định, dùng chung cho mọi request, được cache]
1. Bối cảnh: sản phẩm là gì, người dùng là ai, output được dùng vào việc gì
2. Nhiệm vụ và tiêu chí "làm tốt" (đo được nếu có thể)
3. Quy tắc, mỗi quy tắc kèm lý do; nói điều cần làm thay vì chỉ điều cấm
4. Cách xử lý khi không chắc / thiếu thông tin / ngoài phạm vi
5. Format output (hoặc dùng structured output thay cho đoạn này)
6. Ví dụ (nếu dùng), bọc trong tag riêng, đa dạng
7. Tài liệu tham chiếu tĩnh (chính sách, catalog)

[phần động — thay đổi theo phiên hoặc request]
8. Dữ liệu của phiên: ngày hiện tại (render một lần đầu phiên), hồ sơ khách
9. Tài liệu dài của request (retrieval, file đính kèm), bọc tag kèm nguồn
10. Câu hỏi / yêu cầu cụ thể — đặt cuối cùng
```

### Bảy điều ít được nhắc nhưng hay gây sự cố

1. **Một thay đổi nhỏ về format có thể làm kết quả chênh lớn.** Trên model cũ, mức chênh đo được lên tới 76 điểm accuracy, và format tốt cho model này không tương quan nhiều với format tốt cho model khác ([§1.1](#11-prompt-là-một-phần-của-chương-trình)).
2. **Ngôn ngữ nhấn mạnh làm model mới kích hoạt quá mức.** `CRITICAL`, `MUST`, "nếu nghi ngờ thì cứ gọi tool" từng cần thiết với model cũ, nhưng nay khiến model gọi tool cả khi không cần ([§1.3](#13-cụ-thể-có-ngữ-cảnh-có-lý-do-điều-model-cần-để-làm-đúng)).
3. **Few-shot dạy format và phân phối nhãn nhiều hơn dạy cách làm.** Thứ tự ví dụ, nhãn chiếm đa số và nhãn ở cuối đều làm lệch output. Với reasoning model, few-shot còn có thể làm kết quả kém đi ([§1.4](#14-few-shot-khi-nào-giúp-khi-nào-làm-model-bắt-chước-sai)).
4. **Prefill và `temperature` đang biến mất.** Claude từ 4.6 và Gemini 3.6 trả 400 khi request có prefill. Claude Opus 4.7, 4.8, Sonnet 5 và Haiku 5.5 trả 400 với `temperature` khác mặc định; Gemini 3.6 bỏ qua tham số này ([§2.3](#23-kiểm-soát-độ-dài-giọng-văn-và-format)).
5. **Persona không làm model giỏi hơn.** "Bạn là chuyên gia 20 năm kinh nghiệm" không cải thiện độ chính xác. Tip tiền hay đe doạ cũng không có tác dụng trung bình, dù có thể làm từng câu riêng lẻ lệch mạnh theo cả hai chiều ([§1.3](#13-cụ-thể-có-ngữ-cảnh-có-lý-do-điều-model-cần-để-làm-đúng)).
6. **Công cụ prompt do provider host có thể bị tắt.** OpenAI tắt reusable prompt objects vào 30/11/2026. Anthropic đã gỡ prompt generator/improver API và saved prompts của Workbench từ 17/08/2026 ([§2.6](#26-công-cụ-quản-lý-prompt-registry-versioning-template)).
7. **Tiếng Việt dạng NFD tốn gấp 2,8 lần token so với NFC**, và cũng làm vỡ cache. Viết không dấu không rẻ hơn viết có dấu ([§3.6](#36-đa-ngôn-ngữ-prompt-tiếng-anh-hay-tiếng-việt)).

### Khi nào cần few-shot

| Tình huống | Few-shot? |
|---|---|
| Output phải theo một format, giọng văn hoặc độ dài khó mô tả bằng lời | Có, 3–5 ví dụ đa dạng |
| Phân loại với nhãn mà tên không tự giải thích được | Có, phủ mọi nhãn, cân bằng, xáo thứ tự |
| Cấu trúc JSON cố định | Không cần, dùng structured output ([tài liệu Structured Output](../structured-output/)) |
| Reasoning model, bài toán suy luận | Thử zero-shot trước; chỉ thêm ví dụ nếu eval cho thấy có ích |
| Ca biên mà quy tắc diễn đạt bằng lời rất dài | Có, một ví dụ thường ngắn hơn một đoạn quy tắc |
| Tác vụ có nhiều ví dụ gán nhãn và context dài, cache được | Cân nhắc many-shot (hàng chục đến hàng trăm ví dụ) |

### Các tham số đã bị thay đổi hoặc gỡ (tháng 11/2026)

| Thứ đã quen dùng | Hiện trạng | Thay bằng |
|---|---|---|
| Prefill lượt assistant (Claude) | 400 từ Claude 4.6 trở đi | Structured output, chỉ dẫn trong system, chuyển phần tiếp nối sang user message |
| Prefill lượt model (Gemini) | 400 trên Gemini 3.6 Flash, 3.5 Flash-Lite và các model sau | `system_instruction`, structured output |
| `temperature` / `top_p` / `top_k` (Claude) | 400 khi khác mặc định trên Opus 4.7, 4.8, Sonnet 5, Haiku 5.5; Python SDK 1.0 gỡ hẳn | Ràng buộc bằng prompt, structured output, eval nhiều lần |
| `temperature` (Gemini) | Gemini 3: khuyến nghị giữ 1.0. Gemini 3.6: bị bỏ qua, model sau sẽ trả 400 | Quy tắc rõ trong system instruction |
| `temperature` (OpenAI GPT-5.x) | Chỉ nhận khi `reasoning_effort` là `none` (GPT-5.1+); gpt-5 gốc không nhận* | Như trên |
| `budget_tokens` (Claude thinking) | 400 từ Claude 4.7 | `thinking: {type: "adaptive"}` + `effort` |
| Reusable prompts (`prompt` id + version, OpenAI) | Tắt 30/11/2026 | Prompt trong code, gửi qua `input` |
| `/v1/experimental/generate_prompt`, `improve_prompt` (Anthropic) | Gỡ từ 17/08/2026 | Tự viết, hoặc tối ưu bằng DSPy / GEPA ([§2.5](#25-tự-động-tối-ưu-prompt-dspy-gepa-công-cụ-của-provider)) |

\* Thông tin về OpenAI lấy từ ghi chú tương thích GPT-5.2 được các nguồn thứ cấp trích lại, chưa đối chiếu trực tiếp với trang gốc.

---

# Phần 1 — Nền tảng

Ở ví dụ đầu bài, câu "LUÔN LUÔN tra cứu đơn hàng" trông như một quy tắc nghiệp vụ bình thường. Muốn hiểu vì sao nó làm hỏng một tác vụ không liên quan, cần nhìn prompt như input của một hàm xác suất, không phải như một bản đặc tả được thực thi từng dòng.

## 1.1 Prompt là một phần của chương trình

Code thông thường có ba thứ mà prompt không có. Compiler báo lỗi khi bạn viết sai cú pháp, type system cho biết thay đổi này ảnh hưởng đến đâu, và cùng một input luôn cho cùng một output. Khi sửa prompt, bạn không có thứ nào trong ba thứ đó.

**Không có ranh giới ảnh hưởng.** Mọi câu trong prompt đều có thể tác động đến mọi câu trả lời. Câu "LUÔN tra cứu đơn" được viết cho câu hỏi về giao hàng, nhưng model không biết điều đó. Nó chỉ thấy một chỉ thị mạnh, không có điều kiện áp dụng, và áp dụng nó ở mọi nơi có thể.

**Nhạy với những thay đổi tưởng như vô nghĩa.** Sclar và cộng sự (ICLR 2024) đo ảnh hưởng của các thay đổi format giữ nguyên nghĩa: dấu phân cách, chữ hoa thường, khoảng trắng giữa các trường. Với LLaMA-2-13B, mức chênh lên tới **76 điểm accuracy** giữa format tốt nhất và tệ nhất trên cùng tác vụ. Độ nhạy này vẫn còn khi tăng kích thước model, thêm few-shot hay instruction tuning. Một kết quả quan trọng hơn cho production: format tốt trên model này **chỉ tương quan yếu** với format tốt trên model khác [27]. Nhóm tác giả đề xuất báo cáo một *khoảng* hiệu năng trên nhiều format, thay vì một con số duy nhất.

Model mới ít nhạy hơn, nhưng không hết hẳn. Hua và cộng sự (EMNLP 2025) thử 7 model gồm các họ GPT và Gemini với 12 template và thấy độ lệch giữa template khá thấp. Họ cho rằng phần lớn "độ nhạy với prompt" trong các nghiên cứu cũ là artifact của cách chấm điểm [28]. Tuy vậy, độ nhạy với *thứ tự* nội dung vẫn còn trên model mới [29]. Kết luận thực tế: với model hiện tại, bạn ít khi mất 76 điểm chỉ vì đổi dấu hai chấm thành gạch ngang. Nhưng bạn vẫn không thể biết một thay đổi là vô hại nếu chưa đo.

**Không deterministic, kể cả khi bạn muốn.** Thinking Machines Lab chạy Qwen3-235B 1.000 lần với cùng prompt, temperature 0, và nhận được 80 output khác nhau. Các output giống nhau ở 102 token đầu, rồi tách ra từ token 103. Nguyên nhân không phải race condition, mà là kích thước batch trên server thay đổi theo tải, làm thứ tự phép cộng số thực thay đổi theo [41]. Chỉ khi dùng kernel batch-invariant, cả 1.000 output mới giống hệt nhau. Các provider thương mại không hứa điều này. Hơn nữa, như bảng ở phần Tra cứu nhanh cho thấy, nhiều model mới không còn cho bạn đặt temperature.

Hệ quả: so sánh một output trước và sau khi sửa prompt không nói lên được gì. Một thay đổi prompt cần được đánh giá như một thay đổi code trên một hệ thống ngẫu nhiên. Nghĩa là chạy trên một bộ dữ liệu, nhiều lần, rồi so sánh phân phối kết quả ([§3.4](#34-regression-test-khi-đổi-prompt)).

## 1.2 System prompt, user message và vai trò của từng phần

### Ai có quyền cao hơn

API chat chia input thành các message theo vai trò. OpenAI mô tả rõ nhất thứ bậc này trong Model Spec, gọi là *chain of command*. Thứ tự từ cao xuống thấp là Root (quy tắc của OpenAI), System, Developer, User, Guideline. Assistant message, tool message và nội dung được trích dẫn hay không đáng tin thì **không có quyền mặc định** [15]. Từ o1-2024-12-17, reasoning model của OpenAI dùng `developer` message thay cho `system` [12]. Trong Responses API, chỉ dẫn ở tham số `instructions` được ưu tiên hơn nội dung trong `input`, nhưng không được giữ lại qua `previous_response_id` [13]. Nếu bạn dựng hội thoại nhiều lượt bằng `previous_response_id`, phải gửi lại `instructions` ở mọi request.

Trên Claude, `system` là top-level. Các model mới còn cho chèn system message vào giữa `messages`, để đổi chỉ dẫn giữa phiên mà không làm mất cache của phần trước ([tài liệu Multi-turn, §2.2](../multi-turn-conversation/#22-claude-anthropic)).

### Thứ bậc đó yếu hơn bạn nghĩ

Thứ bậc vai trò là thứ model được *huấn luyện* để tôn trọng, không phải cơ chế được *thực thi*. Nghiên cứu "Control Illusion" (AAAI-26) thử 6 LLM với các xung đột đơn giản, ví dụ system yêu cầu viết chữ thường còn user yêu cầu viết hoa. Kết quả: thứ tự ưu tiên giữa system và user không ổn định, ngay cả với những xung đột chỉ liên quan đến format. Prompt engineering và fine-tuning chỉ cải thiện nhẹ. Đáng chú ý hơn, các khung xã hội như "theo chuyên gia" hay "mọi người đều đồng ý" ảnh hưởng đến hành vi nhiều hơn chính vai trò của message [33].

Có ba hệ quả cho thiết kế:

- **System prompt không phải ranh giới bảo mật.** Quy tắc như "không bao giờ hoàn tiền quá 2 triệu" phải được kiểm tra trong code của tool hoàn tiền, không chỉ nằm trong prompt. Xem [tài liệu Bảo mật LLM](../llm-security/) (đang soạn).
- **System prompt không bí mật.** Hãy coi mọi thứ trong prompt là có thể bị người dùng đọc được. Đừng đặt API key, tên nội bộ nhạy cảm hay logic chống gian lận vào đó.
- **Tool result không có quyền ra lệnh**, và model mới được huấn luyện để nghi ngờ chỉ thị nằm trong đó. Trang hướng dẫn Haiku 5.5 ghi rằng tin nhắn của user được chèn vào giữa lượt qua `tool_result` có thể bị model coi là không đáng tin [3]. Nếu cần chuyển lời người dùng vào giữa một vòng tool, hãy đặt nó vào lượt user.

### Prompt nằm ở nhiều chỗ hơn system prompt

Tên tool, description của tool, tên field và `description` trong JSON Schema, thông báo lỗi tool trả về: tất cả đều là văn bản model đọc để quyết định làm gì. Anthropic ghi nhận rằng chỉ một chỉnh sửa nhỏ trong tool description cũng có thể làm kết quả thay đổi rõ rệt [5]. Khi review một thay đổi prompt, hãy tính cả những chỗ này. Một kỹ sư đổi description của tool `lookup_order` từ "Tra cứu đơn hàng" sang "Dùng cho mọi câu hỏi của khách liên quan đến cửa hàng" cũng có thể gây ra sự cố như ở đầu bài, dù system prompt không đổi một chữ.

## 1.3 Cụ thể, có ngữ cảnh, có lý do: điều model cần để làm đúng

### Lời giải thích tổng quát hoá được, mệnh lệnh thì không

Anthropic đưa một ví dụ trong hướng dẫn chính thức. Câu `NEVER use ellipses` kém hiệu quả hơn câu: "Câu trả lời sẽ được đọc bằng engine text-to-speech, nên không dùng dấu ba chấm vì engine không biết đọc chúng." Lý do được nêu ngắn gọn: "Claude can generalize from the explanation" [1].

Cơ chế ở đây khá dễ hiểu. Một mệnh lệnh chỉ phủ đúng trường hợp được nêu. Một lý do cho model biết *mục tiêu*, nên nó tự suy ra những trường hợp bạn không viết ra. Khi biết output sẽ được đọc thành tiếng, model cũng sẽ tránh bảng, URL dài và ký hiệu toán, dù bạn không liệt kê chúng.

Quay lại ví dụ đầu bài, một quy tắc tốt hơn sẽ như sau:

```text
Khi khách hỏi về trạng thái, vị trí hoặc thời gian giao của một đơn cụ thể, tra cứu
bằng lookup_order trước khi trả lời. Thông tin giao hàng chỉ đến từ hệ thống đơn;
nếu trả lời theo phỏng đoán, khách có thể chờ một đơn đã bị huỷ.
Câu hỏi chung về cửa hàng (giờ mở cửa, địa chỉ, chính sách) không cần mã đơn.
```

Quy tắc này có điều kiện áp dụng, có lý do, và nói rõ khi nào *không* áp dụng.

### Bớt nhấn mạnh với model mới

Các model đời trước hay "lười" gọi tool, nên prompt thường phải viết `CRITICAL: You MUST use this tool when...`. Theo Anthropic, các model từ Opus 4.5 trở đi phản hồi mạnh hơn với system prompt, và chính lối viết đó khiến model kích hoạt quá mức. Cách sửa là "soften any aggressive language", tức viết đơn giản là "Use this tool when...". Họ cũng nêu riêng câu "If in doubt, use [tool]" là một nguyên nhân gây over-trigger [1].

Điều này có nghĩa là một prompt được tinh chỉnh kỹ cho model cũ có thể chạy *tệ hơn* trên model mới chính vì những chỗ đã được tinh chỉnh. Opus 5 còn có thêm một trường hợp: model tự kiểm tra lại công việc khá tốt mà không cần nhắc, nên câu "trước khi kết thúc, hãy kiểm tra lại..." mang từ prompt cũ sang có thể gây *over-verification*, tốn thêm token mà không thêm chất lượng [1].

### Nói điều cần làm, không chỉ điều cấm

"Không dùng markdown" cho model biết một thứ cần tránh, nhưng không cho biết nên viết thế nào. Hướng dẫn của Anthropic đề xuất viết thành "Câu trả lời gồm các đoạn văn xuôi liền mạch" [1]. Câu phủ định còn có một vấn đề khác: nó nhắc đến chính thứ bạn muốn tránh. Một danh sách dài kiểu "không nói X, không nhắc Y" đưa X và Y vào context của mọi request. Khi cần cấm, hãy viết lệnh cấm kèm hành vi thay thế: "Không hứa ngày giao hàng. Nếu khách hỏi, đọc ngày dự kiến từ kết quả `lookup_order`; nếu không có, nói rằng hệ thống chưa xác nhận."

### Chỉ dẫn mâu thuẫn có giá

Hai provider mô tả hai hệ quả khác nhau. Với GPT-4.1, khi hai chỉ dẫn mâu thuẫn, model "tends to follow the one closer to the end of the prompt" [10]. Một quy tắc thêm vào cuối prompt để vá lỗi sẽ âm thầm đè lên quy tắc ở đầu. Với GPT-5, model dùng reasoning token để tìm cách dung hoà hai chỉ dẫn, và OpenAI viết rằng prompt mâu thuẫn hoặc mơ hồ "can be more damaging to GPT-5" so với model trước [11]. Bạn mất cả chi phí lẫn độ ổn định.

Mâu thuẫn trong prompt production thường xuất hiện theo thời gian, không phải ngay từ đầu. Ví dụ, đoạn đầu viết "trả lời ngắn gọn", ba tháng sau có người thêm "luôn giải thích đầy đủ chính sách đổi trả". Khi review một thay đổi, hãy đọc lại toàn bộ prompt, không chỉ phần diff ([§3.2](#32-versioning-và-review-thay-đổi-prompt)).

### Số lượng chỉ dẫn có giới hạn

Thêm một quy tắc mỗi khi gặp lỗi là cách phổ biến nhất để prompt phình to. IFScale (2025) thử 20 model với 10 đến 500 chỉ dẫn đồng thời trong một tác vụ viết báo cáo. Độ chính xác giảm khi số chỉ dẫn tăng, và model tốt nhất chỉ đạt khoảng 68% ở 500 chỉ dẫn. Model có xu hướng ưu tiên các chỉ dẫn xuất hiện sớm, và lỗi phổ biến nhất là *bỏ sót* chứ không phải làm sai [35]. Arize chạy lại benchmark này trên các model frontier tháng 05/2026 và thấy ngưỡng tăng khoảng một bậc độ lớn, lên hàng nghìn chỉ dẫn. Đây là kết quả chưa qua phản biện, và chính Arize lưu ý IFScale chỉ đo một loại chỉ dẫn rất đơn giản (đưa một từ khoá vào văn bản), không chứng minh mọi loại chỉ dẫn đều được tuân thủ [36]. Một quy tắc nghiệp vụ có điều kiện khó hơn nhiều so với một từ khoá.

Anthropic gọi đây là bài toán chọn đúng "altitude": tránh cả hai cực, if-else cứng nhắc cho từng trường hợp và chỉ dẫn mơ hồ chung chung. Cách làm họ đề xuất là bắt đầu từ prompt tối thiểu, rồi chỉ thêm chỉ dẫn khi quan sát được lỗi cụ thể. Họ cũng nhấn mạnh rằng "minimal does not necessarily mean short" [4]. Mỗi quy tắc nên có một lỗi thật mà nó sửa, và một ca test chứng minh nó còn cần thiết.

### Persona, tip và đe doạ

"Bạn là chuyên gia chăm sóc khách hàng 20 năm kinh nghiệm" xuất hiện trong rất nhiều prompt. Zheng, Pei và Jurgens thử 162 vai trò trên 2.457 câu hỏi factual. Bản cập nhật của bài báo đổi tên thành "Personas in System Prompts Do Not Improve Performances of Large Language Models": thêm persona không cải thiện độ chính xác so với không có persona, và chọn persona tự động cho từng câu hỏi gần như không hơn chọn ngẫu nhiên [37].

Nhóm Wharton (Prompting Science Report) đo các mẹo khác. Hứa "tip $100" hay đe doạ model không tạo khác biệt có ý nghĩa khi tính trung bình trên benchmark. Nói "please" cũng vậy [38][39]. Phát hiện đáng chú ý nhất lại là một chi tiết phụ: **ở từng câu hỏi riêng lẻ**, các biến thể này có thể đẩy kết quả lên hoặc xuống rất mạnh, dù trung bình gần như bằng không [38]. Bài học thực tế không nằm ở chuyện tip có tác dụng hay không. Nó nằm ở chỗ: nếu bạn thử một thay đổi prompt trên 5 câu và thấy tốt hơn, rất có thể bạn chỉ đang thấy nhiễu ở cấp từng câu.

Vậy role có vô dụng không? Không hẳn. Anthropic vẫn khuyên đặt role trong system prompt, vì nó "focuses Claude's behavior and tone" [1]. Role hữu ích để định giọng văn, mức độ trang trọng và phạm vi. Nó không thêm kiến thức. Một role tốt mô tả *bối cảnh*, chẳng hạn "Bạn trả lời khách hàng của một cửa hàng điện tử ở Việt Nam qua chat; khách thường hỏi trên điện thoại, giữa giờ làm", chứ không phải một danh hiệu.

## 1.4 Few-shot: khi nào giúp, khi nào làm model bắt chước sai

### Ví dụ dạy model điều gì

Trực giác thông thường cho rằng few-shot dạy model *ánh xạ* từ input sang output. Min và cộng sự (EMNLP 2022) kiểm tra điều này bằng cách thay nhãn đúng trong ví dụ bằng nhãn ngẫu nhiên. Trên 12 model, kể cả GPT-3, hiệu năng giảm rất ít. Những gì thật sự quan trọng là **không gian nhãn** (có những nhãn nào), **phân phối của input** và **format** [21]. Một nghiên cứu sau đó phản biện rằng nhãn đúng vẫn quan trọng hơn mức Min đánh giá trong một số thiết lập [22]. Dù vậy, kết luận chung vẫn đứng vững: phần lớn những gì model lấy từ ví dụ là *hình dạng* của output.

Điều này giải thích một hiện tượng hay gặp. Model bắt chước các đặc điểm bề mặt mà bạn không cố ý dạy. Nếu cả ba ví dụ trả lời khách đều mở đầu bằng "Dạ, em chào anh/chị", mọi câu trả lời sẽ mở đầu như vậy, kể cả khi đang giữa hội thoại. Nếu cả ba ví dụ đều dài khoảng 60 từ, câu trả lời cho "cảm ơn em" cũng sẽ dài 60 từ. Anthropic khuyên ví dụ cần "vary enough that Claude doesn't pick up unintended patterns" [1].

### Ba nguồn thiên lệch

Zhao và cộng sự (ICML 2021) chỉ ra ba thiên lệch trong few-shot [19]:

- **Majority label bias:** model nghiêng về nhãn xuất hiện nhiều nhất trong ví dụ.
- **Recency bias:** nhãn của ví dụ cuối có trọng số lớn. Với chuỗi ví dụ P P P N, model dự đoán N cho phần lớn input.
- **Common token bias:** nhãn là từ phổ biến trong dữ liệu pretraining được chọn nhiều hơn.

Lu và cộng sự (ACL 2022) cho thấy chỉ riêng *thứ tự* ví dụ đã đủ đẩy kết quả từ gần state-of-the-art xuống gần ngẫu nhiên. Hiệu ứng này vẫn còn ở model lớn nhất họ thử, và thứ tự tốt trên model này không chuyển được sang model khác [20].

Những nghiên cứu này làm trên GPT-2, GPT-3, tức model cũ hơn nhiều so với hiện tại. Biên độ trên model mới nhỏ hơn, nhưng hướng của thiên lệch vẫn là thứ cần kiểm tra. Với tác vụ phân loại, có ba việc nên làm: cân bằng số ví dụ giữa các nhãn, không để ví dụ cuối luôn thuộc cùng một nhãn, và đo phân phối nhãn của output so với phân phối thật.

### Reasoning model và few-shot

Ở điểm này, các provider khuyên ngược nhau:

| Nguồn | Khuyến nghị |
|---|---|
| DeepSeek-R1 (paper) | "Few-shot prompting consistently degrades its performance." Nên mô tả vấn đề trực tiếp và chỉ định format output, zero-shot [23] |
| OpenAI (reasoning best practices, o-series) | Reasoning model thường không cần few-shot; thử zero-shot trước; nếu có ví dụ thì phải khớp rất sát với chỉ dẫn [12] |
| Google (Gemini prompting strategies) | "We recommend to always include few-shot examples in your prompts"; cảnh báo quá nhiều ví dụ gây overfitting [16] |
| Anthropic | 3–5 ví dụ đa dạng, bọc trong `<example>` [1] |

Khuyến nghị của DeepSeek là quan sát của nhóm phát triển, không phải một thí nghiệm có kiểm soát đầy đủ. Các nghiên cứu sau còn thấy one-shot tốt nhất với một số model distill từ R1 [24]. Cách hiểu hợp lý: với reasoning model, ví dụ có thể *neo* quá trình suy luận vào một lối giải cụ thể. Điều đó giúp nếu lối giải đúng, và hại nếu bài toán mới cần cách khác. Không có quy tắc chung, chỉ có kết quả eval trên tác vụ của bạn.

### Many-shot

Với context dài và prompt caching, bạn không còn bị giới hạn ở 3–5 ví dụ. Agarwal và cộng sự (2024) cho thấy chuyển từ few-shot sang many-shot, tức hàng trăm đến hàng nghìn ví dụ, cải thiện rõ trên nhiều tác vụ. Many-shot còn có thể ghi đè thiên lệch từ pretraining [25]. Chi phí tăng tuyến tính theo số ví dụ, nhưng nếu các ví dụ nằm ở phần đầu prompt và được cache, chi phí thực tế giảm còn khoảng một phần mười ([tài liệu Multi-turn, §1.4](../multi-turn-conversation/#14-kinh-tế-học-của-prompt-caching)). Many-shot là một lựa chọn trung gian giữa prompt và fine-tuning, đáng thử trước khi quyết định fine-tune ([tài liệu Fine-tuning](../fine-tuning/) (đang soạn)).

## 1.5 Prompt và model: vì sao prompt tốt cho model này chưa chắc tốt cho model khác

Mỗi model được post-train với dữ liệu và mục tiêu khác nhau, nên cùng một câu chữ kích hoạt những hành vi khác nhau. Các bằng chứng ở trên đều chỉ về một hướng: format tốt chỉ tương quan yếu giữa các model [27], thứ tự ví dụ tốt không chuyển được giữa các model [20]. Hướng dẫn chính thức của các provider cũng khác nhau, kể cả trong cùng một provider qua các thế hệ:

| Vấn đề | Khác biệt |
|---|---|
| Vị trí chỉ dẫn khi context dài | Anthropic, Google: tài liệu trước, câu hỏi cuối [1][16]. OpenAI (GPT-4.1): chỉ dẫn ở cả đầu và cuối context; nếu chỉ đặt một chỗ thì đặt *trên* context [10] |
| Độ dài prompt | Gemini 3: "Be concise in your input prompts", model "may over-analyze verbose or overly complex prompt engineering techniques" [17]. GPT-4.1: một câu nói rõ hành vi mong muốn thường là đủ [10] |
| Mức độ literal | GPT-4.1 làm theo "more literally than its predecessors" [10]. Claude đời mới "trained for precise instruction following": hỏi "gợi ý cách cải thiện hàm này" thì model chỉ gợi ý, không sửa [1] |
| Độ dài câu trả lời | Opus 5 mặc định trả lời dài hơn model trước, và đổi `effort` không làm thay đổi độ dài một cách tin cậy; phải yêu cầu ngắn gọn trong prompt [1]. GPT-5 có tham số `verbosity` riêng [11] |
| Tự kiểm tra | Model cũ cần câu "kiểm tra lại trước khi kết thúc"; Opus 5 thì câu đó gây over-verification [1] |

Hướng dẫn GPT-5.5 của OpenAI (theo trích dẫn của Simon Willison, chưa đối chiếu trực tiếp trang gốc) nói điều này rõ nhất. Họ khuyên coi model mới là "a new model family to tune for, not a drop-in replacement", bắt đầu từ "the smallest prompt that preserves the product contract", rồi tinh chỉnh lại effort, verbosity, tool description và format trên các ví dụ đại diện [14].

Hệ quả cho cách quản lý: **một prompt chỉ có nghĩa khi đi cùng một model ID cụ thể.** Version của prompt phải gồm cả model, tham số và định nghĩa tool ([§3.2](#32-versioning-và-review-thay-đổi-prompt)). Đổi model là một thay đổi prompt, cần chạy lại toàn bộ eval ([§3.5](#35-di-chuyển-prompt-khi-đổi-model-hoặc-nâng-version)).

---

# Phần 2 — Kỹ thuật và công cụ

## 2.1 Cấu trúc prompt: section, tag XML, markdown

### Mục đích của cấu trúc

Prompt production thường trộn bốn loại nội dung: chỉ dẫn, dữ liệu tham chiếu, ví dụ và input của người dùng. Cấu trúc giúp model phân biệt chúng. Nếu không có ranh giới, một đoạn email khách viết "vui lòng bỏ qua yêu cầu trước" có thể bị đọc như chỉ dẫn. Anthropic khuyên bọc từng loại trong tag riêng (`<instructions>`, `<context>`, `<input>`), đặt tên tag nhất quán giữa các prompt, và lồng tag khi nội dung có thứ bậc tự nhiên [1]. OpenAI đề xuất thứ tự Identity, Instructions, Examples, Context cho developer message, "usually in this order", và đặt context gần cuối [13].

```xml
<instructions>
Trả lời câu hỏi của khách dựa trên chính sách trong <policies>.
...
</instructions>

<policies>
  <policy id="return">Đổi trả trong 30 ngày, sản phẩm chưa mở hộp...</policy>
  <policy id="warranty">...</policy>
</policies>

<examples>
  <example>...</example>
</examples>

<customer_message>
{{message}}
</customer_message>
```

Đánh dấu dữ liệu không đáng tin bằng tag giúp model phân biệt, nhưng **không phải là biện pháp phòng chống prompt injection**: kẻ tấn công có thể viết `</customer_message>` ngay trong nội dung. Các lớp phòng thủ thật sự (giới hạn quyền của tool, tách nguồn tin cậy, xác nhận của con người) sẽ được trình bày trong [tài liệu Bảo mật LLM](../llm-security/), hiện đang soạn.

### XML, markdown hay JSON

He và cộng sự (2024) so sánh plain text, Markdown, JSON và YAML để định dạng cùng một prompt. Không format nào thắng ở mọi tác vụ và mọi model, và model lớn hơn ít nhạy hơn [26]. Cách chọn thực tế:

- **Tag XML** để phân ranh giới giữa các khối nội dung, đặc biệt khi khối chứa dữ liệu dài hoặc dữ liệu từ người dùng. Tag có tên mở và đóng rõ ràng, nên model biết chính xác một khối kết thúc ở đâu. Hướng dẫn GPT-5 cũng ghi nhận Cursor thấy spec dạng XML giúp model tuân thủ chỉ dẫn tốt hơn [11].
- **Markdown heading và danh sách** cho phần chỉ dẫn mà con người cũng cần đọc và review.
- **JSON** cho dữ liệu có cấu trúc mà model cần tra cứu, chẳng hạn danh sách sản phẩm. Đừng viết *chỉ dẫn* dưới dạng JSON.

### Style của prompt lan sang style của output

Đây là một hiệu ứng ít được nhắc. Anthropic ghi rằng "the formatting style used in your prompt might influence Claude's response style", và "removing markdown from your prompt can reduce the volume of markdown in the output" [1]. Một system prompt viết toàn bullet và heading sẽ kéo câu trả lời về phía bullet và heading, kể cả khi bạn đã dặn "trả lời bằng văn xuôi". Nếu sản phẩm cần câu trả lời dạng hội thoại tự nhiên, chẳng hạn chat CSKH trên Zalo hay Messenger vốn không render markdown, hãy viết chính phần chỉ dẫn bằng văn xuôi.

## 2.2 Đưa tài liệu dài vào prompt: thứ tự, trích dẫn

### Thứ tự

Với input dài, từ khoảng 20.000 token trở lên, Anthropic khuyên đặt tài liệu ở đầu prompt, phía trên câu hỏi, chỉ dẫn và ví dụ. Theo họ, đặt câu hỏi ở cuối "can improve response quality by up to 30 percent in tests, especially with complex, multidocument inputs" [1]. Đây là con số "lên tới", và Anthropic không công bố phương pháp đo. Google khuyên tương tự: "Place your specific instructions or questions at the very end of the prompt" [16]. OpenAI, với GPT-4.1, lại khuyên đặt chỉ dẫn ở **cả đầu và cuối** context; nếu chỉ chọn một vị trí thì đặt *trên* context [10].

Hai cách này không loại trừ nhau. Một bố cục chạy ổn trên cả ba provider: chỉ dẫn chung ở đầu system (vốn ổn định và được cache), tài liệu ở giữa, rồi nhắc lại câu hỏi kèm chỉ dẫn quan trọng nhất ở cuối. Bố cục này cũng khớp với cách prompt caching hoạt động ([§3.3](#33-thứ-tự-nội-dung-để-tận-dụng-prompt-caching)).

### Thông tin ở giữa bị bỏ sót

"Lost in the Middle" (Liu và cộng sự, 2023) đo độ chính xác khi thông tin cần thiết nằm ở các vị trí khác nhau trong context. Kết quả cao nhất khi thông tin nằm ở đầu hoặc cuối, và giảm rõ khi nằm ở giữa, kể cả với model được thiết kế cho context dài [30]. Các nghiên cứu sau cho thấy vấn đề không chỉ nằm ở vị trí:

- **Độ dài tự nó đã có hại.** Levy, Jacoby và Goldberg (ACL 2024) giữ nguyên tác vụ suy luận và chỉ thêm văn bản không liên quan. Độ chính xác giảm khi input dài ra, dù còn cách rất xa giới hạn context [31].
- **Context window được quảng cáo khác context dùng được.** NoLiMa (2025) loại bỏ trùng lặp từ khoá giữa câu hỏi và thông tin cần tìm, buộc model phải suy luận để nối hai bên. Trong điều kiện đó, GPT-4o đạt 99,3 điểm ở context ngắn, nhưng "effective length", tức độ dài còn giữ được 85% điểm gốc, chỉ là 8K trên 128K được công bố. Ở 32K, điểm còn 69,7 [32].
- **Chroma "Context Rot" (2025)** thử 18 model. Hiệu năng giảm theo độ dài ngay cả với tác vụ đơn giản. Câu hỏi càng ít giống về mặt từ ngữ với thông tin cần tìm thì càng giảm nhanh [34].

Từ đó, đưa thêm tài liệu vào prompt không miễn phí, kể cả khi context window còn trống. Với RAG, retrieve 5 đoạn đúng thường tốt hơn retrieve 30 đoạn "cho chắc". Xem [tài liệu RAG nâng cao](../advanced-rag/) (đang soạn) về reranking.

### Trích dẫn trước, trả lời sau

Với tài liệu dài, Anthropic khuyên yêu cầu model trích các đoạn liên quan trước, bọc trong tag `<quotes>`, rồi mới trả lời dựa trên các đoạn đó [1]. Cách này có ba tác dụng: buộc model tìm thông tin trước khi kết luận, để lại bằng chứng bạn có thể kiểm tra bằng code (đoạn trích có thật trong tài liệu không), và đưa thông tin liên quan về cuối context, gần câu trả lời. Bọc mỗi tài liệu trong `<document>` có `<source>` và `<document_content>` giúp model trích dẫn đúng nguồn [1].

```xml
<documents>
  <document index="1">
    <source>chinh-sach-doi-tra-2026.pdf</source>
    <document_content>{{RETURN_POLICY}}</document_content>
  </document>
</documents>

Trích nguyên văn các câu trong tài liệu liên quan đến câu hỏi vào <quotes>.
Nếu không có câu nào liên quan, ghi <quotes>không có</quotes> và nói rằng
chính sách không đề cập. Sau đó trả lời trong <answer>, chỉ dựa trên <quotes>.

Câu hỏi: {{question}}
```

## 2.3 Kiểm soát độ dài, giọng văn và format

### Khi không còn prefill

Prefill, tức điền sẵn phần đầu của lượt assistant, từng là kỹ thuật kiểm soát format mạnh nhất trên Claude: điền `{` để ép JSON, điền `<answer>` để bỏ câu dẫn. Từ Claude 4.6, request có prefill ở lượt assistant cuối trả lỗi 400 [1]. Gemini 3.6 Flash, 3.5 Flash-Lite và các model sau cũng trả 400 với request kết thúc bằng lượt model không rỗng [18]. Anthropic liệt kê cách thay thế cho từng mục đích dùng prefill [1]:

| Mục đích cũ | Cách thay |
|---|---|
| Ép JSON / YAML / nhãn phân loại | Structured output ([tài liệu Structured Output](../structured-output/)); với phân loại, enum trong schema |
| Bỏ câu dẫn "Here is..." | Chỉ dẫn trong system: trả lời trực tiếp, không mở đầu bằng "Dưới đây là...", "Dựa trên..."; hoặc output trong tag rồi cắt bằng code; câu dẫn còn sót thì xoá ở bước hậu xử lý |
| Tiếp tục một response bị cắt | Đưa phần đã có vào lượt user: "Câu trả lời trước bị ngắt ở `...`. Viết tiếp từ đó." |
| Nhắc lại ngữ cảnh trong hội thoại dài | Chèn lời nhắc vào lượt user, hoặc qua tool, hoặc lúc compaction |

### Khi không còn temperature

Nhiều đội dùng `temperature: 0` như một cách để "làm model ổn định". Như đã nói ở §1.1, cách này chưa bao giờ cho output deterministic. Giờ trên nhiều model, nó còn không được phép: Claude Opus 4.7, 4.8, Sonnet 5 và Haiku 5.5 trả 400 khi temperature khác mặc định [2], Gemini 3.6 bỏ qua tham số này và cảnh báo các model sau sẽ trả 400 [18]. Google viết thẳng cách thay thế: "To improve determinism, define a system instruction with explicit rules for your specific use case" [18].

Trên thực tế, sự "ổn định" mà đội cần thường nằm ở một trong ba thứ, và mỗi thứ có công cụ riêng:

- **Ổn định về format** → structured output, không phải temperature.
- **Ổn định về quyết định** (cùng một input thì cùng một nhãn) → quy tắc rõ ràng hơn, ví dụ cho các ca biên, và đo tỷ lệ đồng thuận khi chạy nhiều lần.
- **Ổn định để test** → so sánh phân phối thay vì so khớp chuỗi ([§3.4](#34-regression-test-khi-đổi-prompt)).

### Độ dài

Mỗi provider dùng một núm điều khiển khác nhau. GPT-5 có tham số `verbosity`, điều khiển độ dài câu trả lời cuối chứ không phải độ dài phần suy luận [11]. Trên Claude Opus 5, `effort` không làm thay đổi độ dài phần hiển thị một cách tin cậy, nên phải yêu cầu ngắn gọn ngay trong prompt [1]. Gemini 3 mặc định trả lời ngắn; muốn câu trả lời dài hơn thì phải yêu cầu rõ [17].

Một kinh nghiệm thực tế, chưa có nghiên cứu xác nhận: chỉ định độ dài bằng đơn vị có cấu trúc ("tối đa 3 câu", "một đoạn") thường được tuân thủ tốt hơn chỉ định bằng số từ ("khoảng 80 từ"), vì model không đếm từ chính xác. Với tiếng Việt, "từ" còn mơ hồ hơn: một từ là một âm tiết, hay một từ ghép?

### Format bị trôi trong hội thoại dài

Hướng dẫn GPT-5 ghi nhận rằng chỉ dẫn về markdown có thể mất tác dụng sau nhiều lượt, và đề xuất nhắc lại chỉ dẫn đó mỗi 3–5 tin nhắn của user [11]. Trên Claude, cách tương đương mà không làm vỡ cache là chèn system message vào giữa hội thoại ([tài liệu Multi-turn, §2.2](../multi-turn-conversation/#22-claude-anthropic)).

## 2.4 Prompt chaining, chain-of-thought và tách tác vụ

### Khi nào nên tách

Một prompt làm năm việc thì khó đo xem việc nào hỏng. Tách tác vụ thành chuỗi các bước có giá trị khi:

- Các bước cần **chỉ dẫn mâu thuẫn nhau**. Ví dụ, bước trích xuất cần "chỉ dùng thông tin có trong email", còn bước soạn trả lời cần "giọng thân thiện, chủ động gợi ý".
- Một bước cần **model khác**, rẻ hơn cho phân loại hoặc mạnh hơn cho suy luận ([tài liệu Model Routing](../model-routing/) (đang soạn)).
- Bạn cần **kiểm tra bằng code giữa hai bước**, ví dụ mã đơn trích ra có tồn tại không, trước khi soạn câu trả lời.
- Bạn muốn **eval riêng từng bước** để biết lỗi nằm ở đâu.

Cái giá là độ trễ cộng dồn và nhiều điểm hỏng hơn. Nếu các bước không cần kiểm tra giữa chừng và dùng chung một model, một prompt có cấu trúc tốt kèm reasoning thường đủ ([tài liệu Latency](../latency/) (đang soạn)).

### Chain-of-thought đang mất giá trị

"Let's think step by step" từng là mẹo nổi tiếng nhất. Prompting Science Report 2 (Wharton, 06/2025) đo lại trên các model hiện tại [40]:

- Với model không có reasoning, CoT chỉ cải thiện trung bình một chút, vì nhiều model đã tự suy luận từng bước dù không được yêu cầu.
- Với reasoning model, yêu cầu CoT gần như không thay đổi độ chính xác, nhưng tốn thêm token và thời gian.

CoT còn có thể gây hại. "Mind Your Step (by Step)" (ICML 2025) tìm ra các nhóm tác vụ mà CoT làm hiệu năng giảm rõ. Đó là những tác vụ mà ngay con người cũng làm kém đi khi phải suy nghĩ thành lời, chẳng hạn học quy luật ngầm từ ví dụ [42].

Với reasoning model, Anthropic khuyên dùng chỉ dẫn chung thay vì các bước viết sẵn: "A prompt such as 'think thoroughly' often produces better reasoning than a hand-written step-by-step plan" [1]. OpenAI cũng viết rằng yêu cầu reasoning model "think step by step" là không cần thiết, và có thể làm kết quả kém đi [12]. Một quy trình nghiệp vụ năm bước viết sẵn trong prompt có thể *giới hạn* model vào lối giải bạn nghĩ ra. Hãy mô tả mục tiêu và tiêu chí đúng, để model tự lập kế hoạch. Xem thêm [tài liệu Reasoning Models](../reasoning-models/) (đang soạn).

### Đừng dùng phần suy luận làm lời giải thích

Khi CoT xuất hiện trong output, người ta hay dùng nó làm "lý do" của quyết định, ghi vào log để audit hoặc hiển thị cho người dùng. Nghiên cứu về *faithfulness* cho thấy đây là một giả định nguy hiểm. Anthropic (2025) cài gợi ý vào câu hỏi rồi kiểm tra xem reasoning model có thừa nhận đã dùng gợi ý không. Trong phần lớn trường hợp model dùng gợi ý, chuỗi suy luận không nhắc đến nó [43]. Phần suy luận là văn bản model sinh ra, không phải bản ghi quá trình tính toán thật. Nếu cần lý do có thể kiểm tra, hãy yêu cầu bằng chứng cụ thể (đoạn trích, mã đơn) rồi xác minh bằng code.

## 2.5 Tự động tối ưu prompt: DSPy, GEPA, công cụ của provider

### Cách hoạt động

Các optimizer coi prompt là tham số cần tối ưu. Bạn cung cấp một chương trình (một hoặc nhiều lời gọi LLM), một bộ dữ liệu và một metric. Optimizer thử các biến thể chỉ dẫn và ví dụ, giữ lại biến thể cho điểm cao nhất.

- **MIPROv2 (DSPy)** tối ưu đồng thời chỉ dẫn và ví dụ few-shot cho chương trình nhiều bước. Trên 7 chương trình thử nghiệm với Llama-3-8B, nó tốt hơn baseline ở 5 chương trình, với mức cải thiện tới 13% độ chính xác [44].
- **GEPA (2025)** dùng LLM đọc lại trace của các lần chạy thất bại, viết ra nhận xét bằng ngôn ngữ tự nhiên, rồi sửa prompt dựa trên nhận xét đó. Bài báo báo cáo GEPA vượt GRPO, một phương pháp RL cập nhật trọng số, trung bình 6% và tối đa 20%, trong khi dùng ít hơn tới 35 lần số lần chạy. Nó cũng vượt MIPROv2 hơn 10% [45].

Điểm ít được nhắc trong kết quả của GEPA: trên một số tác vụ, tối ưu prompt cạnh tranh được với fine-tuning bằng RL, với chi phí thấp hơn nhiều. Trước khi quyết định fine-tune, đáng thử một optimizer với cùng bộ dữ liệu.

### Những rủi ro cần biết

- **Không có metric thì không có optimizer.** Phần khó nhất là bộ dữ liệu và metric, không phải chạy optimizer. Nếu chưa có eval, hãy làm eval trước ([tài liệu Evaluation](../evaluation/) (đang soạn)).
- **Overfit bộ dev.** Optimizer tìm ra prompt tốt nhất *trên dữ liệu bạn đưa vào*. Với bộ dev 50 câu và nhiễu ở cấp từng câu lớn như §1.3 mô tả, prompt "tốt nhất" có thể chỉ là prompt may mắn nhất. Luôn giữ một bộ test riêng mà optimizer không thấy.
- **Prompt tối ưu chuyển sang model khác được một phần.** PromptAgent (ICLR 2024) lấy prompt tối ưu trên GPT-3.5 đem sang GPT-4 và PaLM 2: prompt vẫn tốt hơn baseline ở 11/12 tác vụ với GPT-4, nhưng chỉ 7/12 với PaLM 2 [64]. Mức chuyển được phụ thuộc vào model đích và tác vụ, khớp với các bằng chứng ở §1.5. Khi đổi model, chạy eval cho prompt đã tối ưu trước; chỉ chạy lại optimizer nếu kết quả không đạt.
- **Prompt khó đọc và khó review.** Prompt do optimizer viết có thể chứa ví dụ và câu chữ mà không ai trong đội hiểu vì sao lại ở đó. Hãy lưu kèm điểm eval và cấu hình optimizer, để lần sau có thể chạy lại thay vì sửa tay.

### Công cụ của provider đang bị gỡ

Hai provider lớn đều đã gỡ hoặc đang gỡ công cụ tối ưu prompt trong năm 2026:

- **Anthropic** gỡ các endpoint `/v1/experimental/generate_prompt`, `improve_prompt`, `templatize_prompt` cùng lúc với Workbench cũ, từ 17/08/2026. Saved prompts, biến và eval của Workbench không được chuyển sang playground mới [2].
- **OpenAI** vẫn giữ chat tối ưu prompt trên dashboard, nhưng workflow tối ưu dựa trên dataset đi cùng Evals. Evals chuyển sang chỉ đọc từ 31/10/2026 và dự kiến tắt vào 30/11/2026 [9].

Bài học không nằm ở việc công cụ nào tốt hơn. Nó nằm ở chỗ **đừng đặt dữ liệu eval và lịch sử prompt duy nhất trong công cụ do provider host**. Hãy dùng chúng để khám phá, nhưng lưu kết quả vào repo của bạn.

## 2.6 Công cụ quản lý prompt: registry, versioning, template

### Prompt registry

Registry lưu prompt ngoài code, có version, và cho phép đổi version đang chạy mà không cần deploy. Cơ chế của Langfuse là ví dụ điển hình [6][7]:

- Mỗi lần lưu tạo một version mới. **Label** như `production` hay `staging` trỏ đến một version cụ thể; `latest` tự trỏ đến version mới nhất. Gọi mà không chỉ định label thì nhận version gắn `production`.
- Rollback bằng cách chuyển label `production` về version cũ. Label có thể được bảo vệ để chỉ admin mới đổi được (tính năng trả phí).
- SDK **cache prompt phía client, TTL mặc định 60 giây**. Khi cache hết hạn, SDK trả bản cũ và làm mới ở nền. Khi cache trống, chẳng hạn lúc process vừa khởi động, mà registry không truy cập được, bạn phải tự truyền `fallback`. Nếu không, request lỗi.

Chi tiết cuối cùng là điều ít ai để ý khi bắt đầu dùng registry: **registry trở thành một dependency lúc runtime**. Khi registry gặp sự cố đúng lúc bạn deploy hoặc autoscale, các instance mới không có prompt nào để chạy. Prompt fallback trong code cần được giữ đồng bộ với version production. Nếu không, khi registry lỗi, bạn đang chạy một prompt cũ mà không ai biết.

MLflow Prompt Registry và PromptLayer có mô hình tương tự: version bất biến, alias hoặc release label, gắn version vào trace [8].

### Prompt object của provider

OpenAI từng cho lưu prompt trên dashboard và gọi bằng `prompt: {id, version, variables}` trong Responses API. Tính năng này bị ngừng khuyến khích từ 03/06/2026, và `v1/prompts` tắt vào **30/11/2026**. Hướng dẫn chuyển đổi chính thức là đưa nội dung prompt vào code, dựng messages trong ứng dụng, rồi gửi qua `input` [9]. Nếu hệ thống của bạn còn gọi theo prompt ID, đây là việc cần làm trong tháng này.

### Template và injection

Prompt production gần như luôn là template có biến. Có hai loại rủi ro khác nhau:

**Template do người dùng kiểm soát.** Nếu người dùng (hoặc model, hoặc metadata của file model) có thể cung cấp *chính template*, Jinja2 `Environment()` thông thường cho phép thực thi code tuỳ ý. Đây là lỗi server-side template injection (SSTI). Lỗi này đã xuất hiện thật trong các công cụ AI: llama-cpp-python parse chat template trong metadata của file GGUF bằng Jinja2 không sandbox (CVE-2024-34359) [46], và IBM mcp-context-forge cùng RAGFlow có lỗi tương tự trong phần render prompt [47]. `SandboxedEnvironment` giảm rủi ro nhưng không loại bỏ hoàn toàn. Advisory của LangChain lưu ý sandbox chặn các thuộc tính dunder nhưng vẫn cho truy cập các thuộc tính khác [48]. Nguyên tắc an toàn: template là code, chỉ đến từ repo; dữ liệu người dùng chỉ được đưa vào dưới dạng giá trị của biến.

**Dữ liệu người dùng phá cấu trúc.** Ngay cả khi template an toàn, giá trị của biến vẫn có thể chứa `</customer_message>` hoặc `{{`. Với Python `str.format`, giá trị chứa dấu ngoặc nhọn không gây lỗi; chỉ *chuỗi format* chứa ngoặc nhọn mới gây lỗi. Nhưng nếu bạn format hai lần, chẳng hạn chèn input vào template rồi đưa kết quả qua `.format()` thêm một lần nữa, input sẽ thành một phần của chuỗi format. Hãy render đúng một lần, và escape hoặc loại bỏ các tag ranh giới trong dữ liệu người dùng.

### Ngôn ngữ chuyên cho prompt

Có các định dạng riêng cho prompt, như POML của Microsoft (08/2025), một ngôn ngữ markup kiểu HTML có biến, vòng lặp và điều kiện [49], hay BAML (xem [tài liệu Structured Output, §2.8](../structured-output/#28-thư-viện)). Chúng giúp tách prompt khỏi code và review dễ hơn. Đổi lại, bạn có thêm một bước build và một dependency. Với đa số đội, một hàm Python hoặc TypeScript có type cho các biến là đủ.

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Lưu prompt ở đâu: trong code, file template hay prompt registry

| Cách lưu | Phù hợp khi | Rủi ro chính |
|---|---|---|
| **Hàm trong code** (builder có type) | Mặc định cho phần lớn đội. Prompt thay đổi cùng logic xử lý output | Người ngoài đội kỹ thuật khó sửa |
| **File template trong repo** (`.md`, `.jinja`, `.poml`) | Prompt dài, nhiều người review; muốn diff sạch | Thêm bước load; cần test render |
| **Prompt registry** (Langfuse, PromptLayer, MLflow) | Người không viết code (PM, chuyên gia nghiệp vụ) sửa prompt thường xuyên, *và* đã có eval tự động chặn trước khi lên production | Thêm dependency runtime; sửa prompt không qua review code; lệch với code xử lý output |
| **Prompt object của provider** | Không khuyến nghị | Lock-in; đã có hai trường hợp bị tắt trong năm 2026 ([§2.5](#25-tự-động-tối-ưu-prompt-dspy-gepa-công-cụ-của-provider), [§2.6](#26-công-cụ-quản-lý-prompt-registry-versioning-template)) |

Lý lẽ chính cho việc giữ prompt trong repo: prompt và code đọc output của nó phải thay đổi cùng nhau. Nếu prompt thêm một trường vào output mà code chưa xử lý, hoặc ngược lại, chỉ một commit chung mới giữ hai bên khớp nhau. Prompt trong database hay UI không diff được trong pull request và không revert cùng code được.

Lý lẽ cho registry: vòng lặp sửa-thử của người làm nghiệp vụ nhanh hơn nhiều. Nhưng tốc độ đó đi kèm rủi ro. Tháng 05/2025, xAI công bố rằng một "unauthorized modification" vào system prompt của Grok trên X đã bỏ qua quy trình review, khiến bot đưa một chủ đề chính trị vào các câu trả lời không liên quan. Sau sự cố, họ công bố system prompt lên GitHub và bổ sung giám sát [50][51]. Nếu dùng registry, ít nhất cần: label `production` được bảo vệ, mỗi lần đổi label phải qua eval tự động, và có audit log ai đổi gì, lúc nào.

## 3.2 Versioning và review thay đổi prompt

### Version là một bộ, không phải một chuỗi

Từ §1.5, hành vi phụ thuộc vào nhiều thứ ngoài câu chữ của prompt. Một "version" có ý nghĩa cần ghi lại cả bộ:

```python
@dataclass(frozen=True)
class PromptConfig:
    template_version: str        # hash của template hoặc git SHA
    model: str                   # model ID cụ thể, không dùng alias trôi
    params: dict                 # effort, verbosity, max_tokens...
    tools_hash: str              # hash của định nghĩa tool (tên, description, schema)
    output_schema_version: str   # nếu dùng structured output
```

Ghi `PromptConfig` (hoặc hash của nó) vào trace của mọi request. Anthropic postmortem tháng 09/2025 về ba lỗi hạ tầng có câu: "we lacked a clear way to connect these to each of our recent changes" [52]. Nếu bạn không ghi version vào trace, khi chất lượng giảm, bạn cũng sẽ ở vào đúng tình huống đó. Xem [tài liệu LLM Observability](../llm-observability/) (đang soạn).

### Ghim model

Các provider xử lý alias khác nhau, và khác biệt này ảnh hưởng trực tiếp đến việc prompt có bị "đổi model" mà bạn không biết hay không:

- **Anthropic:** từ thế hệ 4.6, mọi model ID, kể cả ID không có ngày như `claude-opus-5-5`, đều là snapshot cố định. Với model trước 4.6, alias trỏ đến một ID có ngày. Anthropic báo trước ít nhất 60 ngày trước khi retire model [53][54].
- **Google:** alias dạng `-latest` "will get hot-swapped with every new release", có email báo trước 2 tuần. Bản preview có thể bị deprecate với 2 tuần báo trước [55].
- **OpenAI:** các alias `chat-latest` trỏ đến model mới theo thời gian và cũng bị deprecate theo lịch riêng [56].

Production nên dùng ID cố định. Alias trôi chỉ dùng trong môi trường thử nghiệm.

### Review một thay đổi prompt

Diff của prompt khó review hơn diff của code, vì một câu thêm vào có thể ảnh hưởng đến mọi thứ. Khi review, hỏi:

1. **Thay đổi này sửa lỗi nào, và ca test nào chứng minh?** Mỗi quy tắc mới nên đi kèm ít nhất một ca test cho trường hợp nó cần áp dụng *và* một ca cho trường hợp nó không được áp dụng (câu hỏi giờ mở cửa ở đầu bài).
2. **Có mâu thuẫn với chỗ nào khác không?** Đọc lại toàn bộ prompt, kể cả tool description và schema.
3. **Có ngôn ngữ tuyệt đối không cần thiết không?** `LUÔN`, `KHÔNG BAO GIỜ`, `QUAN TRỌNG`, chữ in hoa.
4. **Có làm vỡ cache không?** Thay đổi có chèn nội dung động vào phần đầu prompt không ([§3.3](#33-thứ-tự-nội-dung-để-tận-dụng-prompt-caching)).
5. **Kết quả eval trước và sau** trên toàn bộ bộ dữ liệu, không chỉ các ca liên quan.

## 3.3 Thứ tự nội dung để tận dụng prompt caching

Prompt caching tái sử dụng phần *đầu* prompt khi nó giống hệt lần trước. Vì vậy thứ tự nội dung quyết định cả chất lượng lẫn chi phí. Cơ chế và cấu hình chi tiết theo từng provider có trong [tài liệu Multi-turn và Prompt Caching](../multi-turn-conversation/). Ở đây chỉ nêu các điểm giao với việc viết prompt:

- **Ổn định trước, động sau.** Tool, chỉ dẫn, ví dụ và tài liệu tham chiếu tĩnh đứng trước. Ngày giờ, hồ sơ người dùng, kết quả retrieval đứng sau ([Multi-turn §3.2](../multi-turn-conversation/#32-bố-cục-prompt-theo-độ-ổn-định)). Trên Claude, cache phân tầng theo `tools` → `system` → `messages`: đổi định nghĩa tool làm mất toàn bộ cache phía sau [57].
- **Ngày hiện tại render một lần lúc bắt đầu phiên**, không render lại mỗi request. Anthropic khuyên đúng điều này trong hướng dẫn Haiku 5.5 [3]. Một dòng `Bây giờ là 14:32:05` ở đầu system prompt làm cache miss ở *mọi* request.
- **Ví dụ few-shot nằm trong phần tĩnh.** Chọn ví dụ động theo từng request (retrieve ví dụ giống input nhất) có thể cải thiện chất lượng, nhưng làm mất cache cho mọi thứ phía sau. Nếu cần ví dụ động, đặt chúng sau phần tĩnh và đo xem chất lượng tăng có đáng với chi phí không.
- **Thứ tự này trùng với khuyến nghị về chất lượng.** Tài liệu dài trước, câu hỏi cuối ([§2.2](#22-đưa-tài-liệu-dài-vào-prompt-thứ-tự-trích-dẫn)) cũng là thứ tự tốt cho cache. Hiếm khi bạn phải chọn giữa hai mục tiêu.
- **Byte giống hệt, không chỉ nội dung giống nhau.** Thứ tự key trong JSON, khoảng trắng cuối dòng, chuẩn Unicode NFC hay NFD ([§3.6](#36-đa-ngôn-ngữ-prompt-tiếng-anh-hay-tiếng-việt)) đều có thể khiến hai prompt "giống nhau" khác nhau về byte.

## 3.4 Regression test khi đổi prompt

Phần này chỉ nêu những điểm đặc thù của prompt. Phương pháp eval đầy đủ (golden dataset, LLM-as-judge, số mẫu cần thiết) sẽ có trong [tài liệu Evaluation](../evaluation/), hiện đang soạn.

### Vì sao vài ca thử là không đủ

Ba hiện tượng ở Phần 1 cộng lại khiến việc "thử vài câu" đặc biệt dễ sai với prompt:

- **Nhiễu ở cấp từng câu lớn.** Thay đổi không có tác dụng trung bình vẫn có thể làm từng câu riêng lẻ lệch mạnh [38]. Năm câu thử tốt hơn có thể chỉ là nhiễu.
- **Không deterministic.** Cùng một prompt chạy hai lần có thể cho hai kết quả khác nhau [41]. Cần chạy mỗi ca nhiều lần với các tác vụ mà quyết định quan trọng, ví dụ phân loại khiếu nại, và đo tỷ lệ đồng thuận.
- **Ảnh hưởng không cục bộ.** Thay đổi nhắm vào một loại câu hỏi có thể làm hỏng loại khác (đầu bài). Bộ test phải phủ tất cả loại câu hỏi, không chỉ loại vừa sửa.

Anthropic postmortem tháng 09/2025 cũng thừa nhận: "The evaluations we ran simply didn't capture the degradation users were reporting" [52]. Eval tự động là điều kiện cần, chưa phải điều kiện đủ. Cần thêm tín hiệu từ production như feedback người dùng và tỷ lệ chuyển cho nhân viên.

### Một bộ test prompt tối thiểu

- **Ca theo từng quy tắc**: mỗi quy tắc trong prompt có ca nó phải áp dụng và ca nó không được áp dụng.
- **Ca từ production**: các câu hỏi thật đã gây lỗi, đã ẩn danh. Mỗi sự cố thêm ít nhất một ca.
- **Ca đối kháng**: người dùng cố đổi vai trò, chèn chỉ dẫn, hỏi ngoài phạm vi.
- **Biến thể cách hỏi**: cùng một ý hỏi theo nhiều cách, viết có dấu và không dấu, viết tắt kiểu chat ("sp", "ship", "đc"). Ý tưởng này lấy từ FormatSpread [27]: đo một khoảng, không đo một điểm.
- **Kiểm tra bằng code trước, LLM-as-judge sau**: tool nào được gọi, có gọi tool không cần thiết không, độ dài, ngôn ngữ của output, có nhắc đến nội dung bị cấm không.

### Đưa vào CI

promptfoo là một lựa chọn phổ biến để chạy eval trong CI. Workflow chạy khi file prompt hoặc config eval thay đổi [58]. Mặc định, `promptfoo eval` trả exit code `100` khi có dù chỉ một ca thất bại, tức ngưỡng pass rate là 100%. Với hệ thống không deterministic, ngưỡng này thường quá chặt: một ca dao động là CI đỏ. Đặt `PROMPTFOO_PASS_RATE_THRESHOLD` (theo phần trăm, ví dụ `95`) để chấp nhận một tỷ lệ lỗi nhỏ; `PROMPTFOO_FAILED_TEST_EXIT_CODE` đổi exit code khi thất bại [58][63]. Chỉ cần tự đọc output JSON khi ngưỡng phức tạp hơn, chẳng hạn ngưỡng riêng cho từng nhóm ca. Cache kết quả gọi model giữa các lần chạy để CI không tốn tiền cho các ca không đổi, nhưng tắt cache khi muốn đo độ ổn định.

## 3.5 Di chuyển prompt khi đổi model hoặc nâng version

Đổi model là thay đổi lớn nhất mà một prompt phải chịu. Một quy trình dựa trên khuyến nghị của các provider:

1. **Chốt baseline trước khi đổi.** Chạy toàn bộ eval trên model hiện tại, lưu kết quả. Không có baseline thì không biết model mới tốt hơn hay tệ hơn.
2. **Sửa các lỗi API trước.** Tìm prefill, `temperature` / `top_p` / `top_k`, `budget_tokens`, prompt object ID, tham số đã bị gỡ. Những lỗi này trả 400 ngay, nên dễ phát hiện, nhưng sẽ chặn mọi thứ khác.
3. **Gỡ các "miếng vá" cho model cũ.** Ngôn ngữ nhấn mạnh (`CRITICAL`, `MUST`), câu thúc model gọi tool ("nếu nghi ngờ thì cứ gọi"), câu chống lười ("hãy làm thật đầy đủ"), câu tự kiểm tra, quy trình từng bước viết sẵn cho reasoning model. Anthropic khuyên giảm các chỉ dẫn kiểu này khi chuyển sang Claude 4.6 trở đi, vì model mới chủ động hơn và dễ kích hoạt quá mức [1].
4. **Chạy eval, so với baseline.** Xem từng loại câu hỏi, không chỉ điểm trung bình.
5. **Nếu kết quả kém, cân nhắc bắt đầu lại từ prompt tối thiểu.** Hướng dẫn GPT-5.5 khuyên bắt đầu từ "the smallest prompt that preserves the product contract" rồi thêm dần theo lỗi quan sát được [14]. Cách này thường nhanh hơn việc vá prompt cũ.
6. **Tinh chỉnh lại few-shot.** Thứ tự và lựa chọn ví dụ tốt không chuyển được giữa các model [20]. Với reasoning model, thử bỏ hẳn ví dụ.
7. **Kiểm tra độ dài và format.** Model mới thường có độ dài mặc định khác: Opus 5 dài hơn, Gemini 3 ngắn hơn. Đây là thay đổi người dùng thấy ngay.
8. **Rollout dần.** A/B hoặc canary theo phần trăm traffic, theo dõi các tín hiệu production.

Lịch deprecation cho bạn biết khi nào *phải* làm việc này. Tính đến tháng 11/2026: `claude-sonnet-4-5-20250929` retire ngày 30/11/2026 [54], OpenAI `v1/prompts` tắt ngày 30/11/2026 [9], và `gemini-2.5-flash` cùng `gemini-2.5-pro` bị deprecate từ 01/12/2026 [55] (nguồn thứ cấp, cần kiểm tra trang gốc). Hãy theo dõi trang deprecation của từng provider trong lịch vận hành, đừng đợi đến khi request trả lỗi.

## 3.6 Đa ngôn ngữ: prompt tiếng Anh hay tiếng Việt

### Chi phí token: tự đo, đừng đoán

Petrov và cộng sự (NeurIPS 2023) cho thấy cùng một nội dung dịch sang các ngôn ngữ khác nhau có thể chênh tới khoảng 15 lần về số token, tuỳ tokenizer [59]. Với tiếng Việt và tokenizer hiện tại, chưa có nghiên cứu có phản biện nào đưa ra con số cụ thể. Bảng dưới là kết quả đo bằng `tiktoken` trên đoạn system prompt CSKH khoảng 80 từ, viết bằng tiếng Anh và bản dịch tiếng Việt tương đương (tháng 11/2026):

| Phiên bản | `o200k_base` (GPT-4o trở đi) | `cl100k_base` (GPT-4, GPT-3.5) |
|---|---|---|
| Tiếng Anh | 96 token (1,00×) | 98 token (1,00×) |
| Tiếng Việt, Unicode NFC | 123 token (1,28×) | 209 token (2,13×) |
| Tiếng Việt, **không dấu** | 137 token (1,43×) | 160 token (1,63×) |
| Tiếng Việt, Unicode **NFD** | 339 token (3,53×) | 431 token (4,40×) |

<details>
<summary>Input và cách đo</summary>

Tiếng Anh:

```text
You are a customer support assistant for an online electronics store. Answer questions about orders, shipping, returns and warranty. If the customer asks for a refund, check that the order was delivered within the last 30 days and that the product is unopened. If you are not sure about a policy, say so and offer to transfer the customer to a human agent. Keep answers short, polite and in the customer's language. Never promise a delivery date that the order system has not confirmed.
```

Tiếng Việt (NFC):

```text
Bạn là trợ lý chăm sóc khách hàng cho một cửa hàng điện tử trực tuyến. Trả lời các câu hỏi về đơn hàng, giao hàng, đổi trả và bảo hành. Nếu khách yêu cầu hoàn tiền, hãy kiểm tra đơn hàng đã được giao trong vòng 30 ngày gần nhất và sản phẩm chưa mở hộp. Nếu bạn không chắc về một chính sách, hãy nói rõ và đề nghị chuyển khách sang nhân viên. Trả lời ngắn gọn, lịch sự và bằng ngôn ngữ của khách. Không bao giờ hứa ngày giao hàng mà hệ thống đơn hàng chưa xác nhận.
```

Các biến thể được tạo từ bản NFC: NFD bằng `unicodedata.normalize("NFD", vi)`; không dấu bằng cách đổi `đ`/`Đ` thành `d`/`D` trước, rồi `unicodedata.normalize("NFKD", ...)` và bỏ các ký tự không phải ASCII. Bước đổi `đ` là bắt buộc, vì `đ` không phân rã được trong NFKD và sẽ bị xoá mất thay vì thành `d`. Số token đếm bằng `tiktoken.get_encoding(name).encode(text)`.

</details>

Ba điều rút ra, phần lớn không có trong các bài hướng dẫn:

1. **Tokenizer mới thu hẹp khoảng cách rất nhiều.** Từ 2,13× xuống 1,28×. Các con số "tiếng Việt tốn gấp đôi tiếng Anh" lan truyền trên mạng thường đến từ tokenizer cũ.
2. **Bỏ dấu không tiết kiệm token.** Với `o200k_base`, bản không dấu còn tốn hơn bản có dấu, vì tokenizer đã học các âm tiết tiếng Việt có dấu phổ biến như một token. Bỏ dấu còn làm mất nghĩa ("bàn", "bán", "bạn" đều thành "ban").
3. **NFD đắt gấp 2,8 lần NFC.** Ở dạng NFD, mỗi chữ có dấu bị tách thành chữ cái gốc cộng một hoặc hai ký tự dấu kết hợp, ví dụ "ệ" thành "e" + dấu mũ + dấu nặng. Văn bản trông hoàn toàn giống nhau trên màn hình. Văn bản NFD thường đến từ tên file trên macOS, copy từ một số file PDF, hoặc bộ gõ cũ. Ngoài chi phí, NFD còn làm vỡ prompt cache (khác byte), làm hỏng so khớp chuỗi trong eval, và có thể làm giảm chất lượng vì model ít thấy dạng này khi huấn luyện.

Cách xử lý: chuẩn hoá mọi văn bản về NFC (`unicodedata.normalize("NFC", s)` trong Python, `s.normalize("NFC")` trong JavaScript) ở ranh giới hệ thống, gồm input người dùng, tài liệu được ingest và chính file template prompt. Với Claude và Gemini, tokenizer khác `tiktoken`, nên hãy đo bằng API đếm token của provider trên dữ liệu thật của bạn.

### Model trả lời sai ngôn ngữ

Marchisio và cộng sự (EMNLP 2024) gọi hiện tượng này là *language confusion*: model trả lời bằng ngôn ngữ khác ngôn ngữ người dùng, hoặc chuyển ngôn ngữ giữa câu. Họ đo trên 15 ngôn ngữ. Ngay cả model mạnh nhất cũng không ổn định hoàn toàn; prompt phức tạp và temperature cao làm tình trạng tệ hơn; few-shot trong đúng ngôn ngữ đích giúp giảm một phần [60].

Trong production tiếng Việt, hiện tượng này thường xuất hiện theo những cách cụ thể:

- System prompt tiếng Anh, khách hỏi bằng tiếng Việt không dấu: model trả lời bằng tiếng Anh, vì "khong dau" trông không giống tiếng Việt.
- Tài liệu retrieve được bằng tiếng Anh: câu trả lời trộn câu tiếng Anh, hoặc giữ nguyên thuật ngữ tiếng Anh không cần thiết.
- Hội thoại dài: lượt đầu đúng tiếng Việt, các lượt sau trôi dần sang tiếng Anh.

Chỉ dẫn hiệu quả cần nói rõ quy tắc, không chỉ nói "trả lời bằng ngôn ngữ của khách": "Luôn trả lời bằng tiếng Việt có dấu, kể cả khi khách viết không dấu hoặc tài liệu tham chiếu bằng tiếng Anh. Giữ nguyên tên sản phẩm và mã lỗi." Thêm ca test ngôn ngữ của output vào bộ regression. Dùng một bộ phát hiện ngôn ngữ hoặc grader đã được kiểm tra trên chính output của bạn, và loại riêng các output chỉ gồm số, mã đơn, tên sản phẩm hay mã lỗi. Với các output đó, mọi bộ phát hiện đều không đáng tin. Tỷ lệ ký tự có dấu chỉ nên là tín hiệu để lọc ca cần xem lại, không phải điều kiện pass/fail: `1,000,000 VND` và `Here is...` đều có tỷ lệ bằng 0, dù chỉ ca thứ hai là lỗi.

### Viết chỉ dẫn bằng tiếng Anh hay tiếng Việt

Có nghiên cứu cho thấy model suy luận tốt hơn khi tự dịch input sang tiếng Anh rồi mới giải [61], và nghiên cứu năm 2025 xác nhận model thường suy luận tốt nhất bằng tiếng Anh, nhưng bước dịch cũng có thể gây lỗi [62]. Đây là kết quả về *suy luận trên input*, không trực tiếp trả lời câu hỏi chỉ dẫn nên viết bằng ngôn ngữ nào. Không có câu trả lời chung, nhưng có các tiêu chí để chọn:

| Viết chỉ dẫn bằng tiếng Anh khi | Viết chỉ dẫn bằng tiếng Việt khi |
|---|---|
| Tác vụ nặng về suy luận, output là dữ liệu (phân loại, trích xuất) | Output là văn bản tiếng Việt cho người dùng cuối, cần đúng giọng văn |
| Đội phát triển đa quốc gia cần review prompt | Người review và người sửa prompt là đội nghiệp vụ Việt Nam |
| Prompt dài, nhiều khối tĩnh, chi phí token quan trọng | Quy tắc có khái niệm khó dịch (xưng hô "anh/chị/em", "dạ", mức độ lịch sự) |

Thường cách làm kết hợp chạy tốt: chỉ dẫn cấu trúc bằng tiếng Anh, còn ví dụ, quy tắc xưng hô và giọng văn bằng tiếng Việt. Ví dụ few-shot bằng tiếng Việt kéo output về tiếng Việt mạnh hơn bất kỳ câu chỉ dẫn nào ([§1.4](#14-few-shot-khi-nào-giúp-khi-nào-làm-model-bắt-chước-sai)). Dù chọn cách nào, quyết định nên dựa trên eval cho cả hai phiên bản, không dựa trên cảm nhận.

### Những chi tiết riêng của tiếng Việt

- **Xưng hô** là một phần của prompt, không phải chi tiết phụ. "Anh/chị", "bạn", "quý khách" cho ra những giọng văn rất khác nhau. Khi không biết giới tính khách, quy định rõ cách xưng hô trung tính, đừng để model đoán từ tên.
- **Định dạng số và ngày**: `1.000.000 đ` hay `1,000,000 VND`, `06/11/2026` hay `2026-11-06`. Quy định rõ trong prompt và kiểm tra bằng code, nhất là khi output được parse lại.
- **Viết tắt và teencode** trong input ("ko", "đc", "sp", "ib") nên có trong bộ test, vì model có thể hiểu sai hoặc trả lời bằng chính lối viết đó.

## 3.7 Failure modes thường gặp

| Triệu chứng | Nguyên nhân khả dĩ | Cách kiểm tra / xử lý |
|---|---|---|
| Model gọi tool cho cả câu hỏi không cần | Ngôn ngữ tuyệt đối (`LUÔN`, `CRITICAL`, "nếu nghi ngờ thì gọi"); tool description quá rộng | Viết lại quy tắc có điều kiện và lý do; thêm ca "không được gọi tool" vào eval ([§1.3](#13-cụ-thể-có-ngữ-cảnh-có-lý-do-điều-model-cần-để-làm-đúng)) |
| Sửa một lỗi, lỗi khác xuất hiện | Ảnh hưởng không cục bộ; quy tắc mới mâu thuẫn quy tắc cũ | Chạy toàn bộ bộ test, không chỉ ca liên quan; đọc lại toàn prompt khi review ([§3.2](#32-versioning-và-review-thay-đổi-prompt)) |
| 400 `does not support assistant message prefill` | Code prefill trên Claude 4.6+ hoặc Gemini 3.6+ | Structured output hoặc chỉ dẫn trong system ([§2.3](#23-kiểm-soát-độ-dài-giọng-văn-và-format)) |
| 400 khi gửi `temperature` | Claude Opus 4.7, 4.8, Sonnet 5, Haiku 5.5; GPT-5.x khi reasoning khác `none` | Bỏ tham số; dùng ràng buộc trong prompt và structured output |
| Mọi câu trả lời mở đầu giống hệt nhau | Ví dụ few-shot cùng một câu mở đầu | Đa dạng hoá ví dụ; xem model bắt chước đặc điểm bề mặt nào ([§1.4](#14-few-shot-khi-nào-giúp-khi-nào-làm-model-bắt-chước-sai)) |
| Phân loại lệch về một nhãn | Majority label bias, recency bias trong ví dụ | Cân bằng ví dụ, xáo thứ tự; so phân phối nhãn output với phân phối thật |
| Câu trả lời đầy bullet dù đã dặn viết văn xuôi | System prompt tự nó viết bằng bullet và heading | Viết phần chỉ dẫn bằng văn xuôi ([§2.1](#21-cấu-trúc-prompt-section-tag-xml-markdown)) |
| Câu trả lời dài hơn sau khi đổi model | Model mới có độ dài mặc định khác; effort không điều khiển độ dài | Chỉ định độ dài theo câu/đoạn; dùng `verbosity` nếu có |
| Bỏ sót thông tin có trong tài liệu | Thông tin ở giữa context dài; quá nhiều tài liệu không liên quan | Câu hỏi đặt cuối; trích dẫn trước khi trả lời; retrieve ít mà đúng ([§2.2](#22-đưa-tài-liệu-dài-vào-prompt-thứ-tự-trích-dẫn)) |
| Prompt cũ chạy kém hơn trên model mới | Miếng vá cho model cũ gây over-trigger, over-verification | Quy trình ở [§3.5](#35-di-chuyển-prompt-khi-đổi-model-hoặc-nâng-version); bắt đầu lại từ prompt tối thiểu |
| Chất lượng giảm mà không có deploy nào | Alias model trôi sang model mới; ai đó đổi label trong registry | Ghim model ID; ghi `PromptConfig` vào trace; audit log của registry ([§3.2](#32-versioning-và-review-thay-đổi-prompt)) |
| Instance mới khởi động không có prompt | Registry không truy cập được, cache trống, không có fallback | Fallback trong code, đồng bộ với version production ([§2.6](#26-công-cụ-quản-lý-prompt-registry-versioning-template)) |
| Request theo prompt ID của OpenAI bắt đầu lỗi | `v1/prompts` tắt ngày 30/11/2026 | Chuyển prompt vào code, gửi qua `input` |
| Bot trả lời tiếng Anh cho khách viết không dấu | Language confusion; system prompt tiếng Anh | Quy tắc ngôn ngữ cụ thể; ví dụ tiếng Việt; ca test ngôn ngữ output ([§3.6](#36-đa-ngôn-ngữ-prompt-tiếng-anh-hay-tiếng-việt)) |
| Số token tiếng Việt cao bất thường; cache miss dù nội dung giống | Văn bản ở dạng Unicode NFD | Chuẩn hoá NFC ở ranh giới hệ thống |
| Kết quả eval dao động giữa các lần chạy với cùng prompt | Không deterministic; bộ test quá nhỏ | Chạy nhiều lần; so phân phối; tăng số ca ([§3.4](#34-regression-test-khi-đổi-prompt)) |
| Lời giải thích của model không khớp với lý do thật | CoT không faithful | Yêu cầu bằng chứng kiểm tra được thay vì dựa vào phần suy luận ([§2.4](#24-prompt-chaining-chain-of-thought-và-tách-tác-vụ)) |
| Dữ liệu người dùng bị hiểu như chỉ dẫn | Không có ranh giới giữa dữ liệu và chỉ dẫn | Tag cho dữ liệu; nhưng phòng thủ thật nằm ở [tài liệu Bảo mật LLM](../llm-security/) (đang soạn) |

## 3.8 Checklist

### Khi viết hoặc sửa prompt

- [ ] Mỗi quy tắc có điều kiện áp dụng và lý do. Không dùng chữ in hoa hay `CRITICAL` / `MUST` để nhấn mạnh.
- [ ] Mỗi lệnh cấm đi kèm hành vi thay thế.
- [ ] Có chỉ dẫn cho trường hợp không chắc, thiếu thông tin, ngoài phạm vi.
- [ ] Không có mâu thuẫn giữa các phần của prompt, tool description và schema.
- [ ] Ví dụ few-shot đa dạng về độ dài, câu mở đầu, nhãn; nhãn cân bằng; được bọc tag riêng.
- [ ] Style của phần chỉ dẫn khớp với style output mong muốn.
- [ ] Với context dài: tài liệu trước, câu hỏi cuối; tài liệu có tag và nguồn.
- [ ] Không có quy tắc bảo mật hay giới hạn nghiệp vụ *chỉ* nằm trong prompt; mọi giới hạn đều được kiểm tra trong code.
- [ ] Không có bí mật trong prompt.
- [ ] Văn bản tiếng Việt ở dạng NFC; quy tắc ngôn ngữ và xưng hô được ghi rõ.

### Khi đưa vào production

- [ ] Prompt nằm trong repo, hoặc trong registry có label bảo vệ, eval chặn và audit log. Không phụ thuộc vào prompt object của provider.
- [ ] Model ID cố định, không dùng alias trôi.
- [ ] `PromptConfig` (template, model, params, tools, schema) được ghi vào trace của mọi request.
- [ ] Không dùng prefill, `temperature`, `budget_tokens` trên model không còn hỗ trợ.
- [ ] Phần tĩnh đứng trước phần động; không có timestamp hay dữ liệu theo request ở đầu prompt.
- [ ] Registry (nếu dùng) có fallback trong code, đồng bộ với version production.
- [ ] Template chỉ đến từ repo; dữ liệu người dùng chỉ là giá trị của biến; render đúng một lần.

### Khi thay đổi prompt hoặc model

- [ ] Có baseline eval trước khi đổi.
- [ ] Bộ test có ca áp dụng và ca không áp dụng cho mỗi quy tắc, ca từ sự cố production, ca đối kháng, biến thể cách hỏi.
- [ ] Mỗi ca quan trọng chạy nhiều lần; so phân phối, không so một output.
- [ ] Khi đổi model: đã gỡ các miếng vá cho model cũ, tinh chỉnh lại few-shot, kiểm tra độ dài và format.
- [ ] Rollout dần, theo dõi tín hiệu production.
- [ ] Đã kiểm tra lịch deprecation của model và tính năng đang dùng.

---

## Nguồn tham khảo

### Anthropic

| # | Resource | URL |
|---|---|---|
| 1 | Prompting best practices | [platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) |
| 2 | Release notes (Workbench sunset, prompt tools API, sampling parameters) | [platform.claude.com/docs/en/release-notes/overview](https://platform.claude.com/docs/en/release-notes/overview) |
| 3 | Prompting Claude Haiku 5.5 | [platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-haiku-5-5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-haiku-5-5) |
| 4 | Effective context engineering for AI agents (09/2025) | [anthropic.com/engineering/effective-context-engineering-for-ai-agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) |
| 5 | Writing effective tools for agents (09/2025) | [anthropic.com/engineering/writing-tools-for-agents](https://www.anthropic.com/engineering/writing-tools-for-agents) |

### Công cụ quản lý prompt

| # | Resource | URL |
|---|---|---|
| 6 | Langfuse: Prompt version control | [langfuse.com/docs/prompt-management/features/prompt-version-control](https://langfuse.com/docs/prompt-management/features/prompt-version-control) |
| 7 | Langfuse: Caching | [langfuse.com/docs/prompt-management/features/caching](https://langfuse.com/docs/prompt-management/features/caching) |
| 8 | MLflow Prompt Registry | [mlflow.org/docs/latest/genai/prompt-registry/](https://mlflow.org/docs/latest/genai/prompt-registry/) |

### OpenAI

| # | Resource | URL |
|---|---|---|
| 9 | Migrate from prompt objects; Deprecations; Prompt optimizer | [developers.openai.com/api/docs/guides/prompting/migrate-from-prompt-object](https://developers.openai.com/api/docs/guides/prompting/migrate-from-prompt-object), [developers.openai.com/api/docs/deprecations](https://developers.openai.com/api/docs/deprecations), [developers.openai.com/api/docs/guides/prompt-optimizer](https://developers.openai.com/api/docs/guides/prompt-optimizer) |
| 10 | GPT-4.1 prompting guide | [developers.openai.com/cookbook/examples/gpt4-1_prompting_guide](https://developers.openai.com/cookbook/examples/gpt4-1_prompting_guide) |
| 11 | GPT-5 prompting guide | [developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide) |
| 12 | Reasoning best practices | [developers.openai.com/api/docs/guides/reasoning-best-practices](https://developers.openai.com/api/docs/guides/reasoning-best-practices) |
| 13 | Prompt engineering guide | [developers.openai.com/api/docs/guides/prompt-engineering](https://developers.openai.com/api/docs/guides/prompt-engineering) |
| 14 | Simon Willison: GPT-5.5 prompting guide (04/2026, nguồn thứ cấp) | [simonwillison.net/2026/apr/25/gpt-5-5-prompting-guide/](https://simonwillison.net/2026/apr/25/gpt-5-5-prompting-guide/) |
| 15 | Model Spec (2026-10-29) | [model-spec.openai.com/2026-10-29.html](https://model-spec.openai.com/2026-10-29.html) |

### Google

| # | Resource | URL |
|---|---|---|
| 16 | Gemini prompting strategies | [aistudio.google.com/docs/prompting-strategies](https://aistudio.google.com/docs/prompting-strategies) |
| 17 | Get started with Gemini 3 (Google Cloud) | [docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/get-started-with-gemini-3](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/get-started-with-gemini-3) |
| 18 | What's new in Gemini 3.6 | [aistudio.google.com/docs/whats-new-gemini-3.6](https://aistudio.google.com/docs/whats-new-gemini-3.6) |

### Nghiên cứu: few-shot

| # | Resource | URL |
|---|---|---|
| 19 | Zhao et al.: Calibrate Before Use (ICML 2021) | [arxiv.org/abs/2102.09690](https://arxiv.org/abs/2102.09690) |
| 20 | Lu et al.: Fantastically Ordered Prompts (ACL 2022) | [arxiv.org/abs/2104.08786](https://arxiv.org/abs/2104.08786) |
| 21 | Min et al.: Rethinking the Role of Demonstrations (EMNLP 2022) | [arxiv.org/abs/2202.12837](https://arxiv.org/abs/2202.12837) |
| 22 | Yoo et al.: Ground-Truth Labels Matter (2022) | [arxiv.org/abs/2205.12685](https://arxiv.org/abs/2205.12685) |
| 23 | DeepSeek-AI: DeepSeek-R1 (2025) | [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948) |
| 24 | Few-shot với các model distill từ R1 (2025) | [arxiv.org/abs/2503.19602](https://arxiv.org/abs/2503.19602) |
| 25 | Agarwal et al.: Many-Shot In-Context Learning (2024) | [arxiv.org/abs/2404.11018](https://arxiv.org/abs/2404.11018) |

### Nghiên cứu: độ nhạy, context dài, chỉ dẫn

| # | Resource | URL |
|---|---|---|
| 26 | He et al.: Does Prompt Formatting Have Any Impact on LLM Performance? (2024) | [arxiv.org/abs/2411.10541](https://arxiv.org/abs/2411.10541) |
| 27 | Sclar et al.: Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design (ICLR 2024) | [arxiv.org/abs/2310.11324](https://arxiv.org/abs/2310.11324) |
| 28 | Hua et al.: Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs (EMNLP 2025) | [aclanthology.org/2025.emnlp-main.1006](https://aclanthology.org/2025.emnlp-main.1006/) |
| 29 | The Order Effect (2025) | [arxiv.org/abs/2502.04134](https://arxiv.org/abs/2502.04134) |
| 30 | Liu et al.: Lost in the Middle (TACL 2024) | [arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172) |
| 31 | Levy et al.: Same Task, More Tokens (ACL 2024) | [arxiv.org/abs/2402.14848](https://arxiv.org/abs/2402.14848) |
| 32 | Modarressi et al.: NoLiMa (2025) | [arxiv.org/abs/2502.05167](https://arxiv.org/abs/2502.05167) |
| 33 | Geng et al.: Control Illusion (AAAI-26) | [arxiv.org/abs/2502.15851](https://arxiv.org/abs/2502.15851) |
| 34 | Chroma: Context Rot (07/2025) | [research.trychroma.com/context-rot](https://research.trychroma.com/context-rot) |
| 35 | Jaroslawicz et al.: How Many Instructions Can LLMs Follow at Once? (IFScale, 2025) | [arxiv.org/abs/2507.11538](https://arxiv.org/abs/2507.11538) |
| 36 | Arize: Models got an order of magnitude better at following instructions in one year (2026, chưa qua phản biện) | [arize.com/blog/llm-instruction-following-benchmark-2026/](https://arize.com/blog/llm-instruction-following-benchmark-2026/) |

### Nghiên cứu: persona, mẹo prompt, chain-of-thought, nondeterminism

| # | Resource | URL |
|---|---|---|
| 37 | Zheng et al.: Personas in System Prompts Do Not Improve Performances of LLMs | [arxiv.org/abs/2311.10054](https://arxiv.org/abs/2311.10054) |
| 38 | Meincke et al.: Prompting Science Report 3 — tip và đe doạ (08/2025) | [arxiv.org/abs/2508.00614](https://arxiv.org/abs/2508.00614) |
| 39 | Meincke et al.: Prompting Science Report 1 — lịch sự và định dạng (03/2025) | [arxiv.org/abs/2503.04818](https://arxiv.org/abs/2503.04818) |
| 40 | Meincke et al.: Prompting Science Report 2 — The Decreasing Value of Chain of Thought (06/2025) | [arxiv.org/abs/2506.07142](https://arxiv.org/abs/2506.07142) |
| 41 | Thinking Machines Lab: Defeating Nondeterminism in LLM Inference (09/2025) | [thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) |
| 42 | Liu et al.: Mind Your Step (by Step) (ICML 2025) | [arxiv.org/abs/2410.21333](https://arxiv.org/abs/2410.21333) |
| 43 | Chen et al. (Anthropic): Reasoning Models Don't Always Say What They Think (2025) | [arxiv.org/abs/2505.05410](https://arxiv.org/abs/2505.05410) |

### Nghiên cứu: tối ưu prompt

| # | Resource | URL |
|---|---|---|
| 44 | Opsahl-Ong et al.: MIPRO (EMNLP 2024) | [arxiv.org/abs/2406.11695](https://arxiv.org/abs/2406.11695) |
| 45 | Agrawal et al.: GEPA (ICLR 2026) | [arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457) |
| 64 | Wang et al.: PromptAgent (ICLR 2024) | [proceedings.iclr.cc/paper_files/paper/2024/file/686a3f32067838c8dbb68da6e9e3cf69-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2024/file/686a3f32067838c8dbb68da6e9e3cf69-Paper-Conference.pdf) |

### Template và bảo mật

| # | Resource | URL |
|---|---|---|
| 46 | llama-cpp-python: Jinja2 SSTI trong chat template (CVE-2024-34359) | [github.com/advisories/GHSA-56xg-wfcc-g829](https://github.com/advisories/GHSA-56xg-wfcc-g829) |
| 47 | IBM mcp-context-forge (GHSA-vwf3-4xxj-qg6h); RAGFlow (GHSA-wpg4-h5g2-jxm6) | [github.com/advisories/GHSA-vwf3-4xxj-qg6h](https://github.com/advisories/GHSA-vwf3-4xxj-qg6h), [github.com/advisories/GHSA-wpg4-h5g2-jxm6](https://github.com/advisories/GHSA-wpg4-h5g2-jxm6) |
| 48 | LangChain advisory về Jinja2 sandbox (GHSA-6qv9-48xg-fc7f) | [github.com/advisories/GHSA-6qv9-48xg-fc7f](https://github.com/advisories/GHSA-6qv9-48xg-fc7f) |
| 49 | Microsoft POML | [microsoft.github.io/poml/latest/](https://microsoft.github.io/poml/latest/) |

### Sự cố, model và deprecation

| # | Resource | URL |
|---|---|---|
| 50 | xAI: grok-prompts | [github.com/xai-org/grok-prompts](https://github.com/xai-org/grok-prompts) |
| 51 | xAI: thông báo về thay đổi system prompt trái phép (05/2025) | [x.com/xai/status/1923183620606619649](https://x.com/xai/status/1923183620606619649) |
| 52 | Anthropic: A postmortem of three recent issues (09/2025) | [anthropic.com/engineering/a-postmortem-of-three-recent-issues](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues) |
| 53 | Anthropic: Models overview | [platform.claude.com/docs/en/models/overview](https://platform.claude.com/docs/en/models/overview) |
| 54 | Anthropic: Model deprecations | [platform.claude.com/docs/en/about-claude/model-deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations) |
| 55 | Google AI Studio: Models (stable, preview, latest) | [aistudio.google.com/docs/models](https://aistudio.google.com/docs/models) |
| 56 | OpenAI: Deprecations | [developers.openai.com/api/docs/deprecations](https://developers.openai.com/api/docs/deprecations) |
| 57 | Anthropic: Prompt caching | [platform.claude.com/docs/en/build-with-claude/prompt-caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) |
| 58 | promptfoo: CI/CD integration | [promptfoo.dev/docs/integrations/ci-cd/](https://www.promptfoo.dev/docs/integrations/ci-cd/) |
| 63 | promptfoo: Command line (exit code, `PROMPTFOO_PASS_RATE_THRESHOLD`) | [promptfoo.dev/docs/usage/command-line/](https://www.promptfoo.dev/docs/usage/command-line/) |

### Nghiên cứu: đa ngôn ngữ

| # | Resource | URL |
|---|---|---|
| 59 | Petrov et al.: Language Model Tokenizers Introduce Unfairness Between Languages (NeurIPS 2023) | [arxiv.org/abs/2305.15425](https://arxiv.org/abs/2305.15425) |
| 60 | Marchisio et al.: Understanding and Mitigating Language Confusion in LLMs (EMNLP 2024) | [arxiv.org/abs/2406.20052](https://arxiv.org/abs/2406.20052) |
| 61 | Etxaniz et al.: Do Multilingual Language Models Think Better in English? (NAACL 2024) | [arxiv.org/abs/2308.01223](https://arxiv.org/abs/2308.01223) |
| 62 | The Reasoning Lingua Franca (2025) | [arxiv.org/abs/2510.20647](https://arxiv.org/abs/2510.20647) |

---

Thông tin API và hướng dẫn của provider được ghi nhận tại **tháng 11/2026**. Phần lớn nghiên cứu về few-shot và độ nhạy với format được thực hiện trên các model đời 2021–2024. Hướng của các hiệu ứng vẫn đáng kiểm tra, nhưng biên độ trên model hiện tại thường nhỏ hơn.

Các điểm chưa đối chiếu trực tiếp với nguồn gốc: hướng dẫn GPT-5.5 của OpenAI (qua bài của Simon Willison), điều kiện `temperature` trên GPT-5.x (qua nguồn thứ cấp), lịch deprecate Gemini 2.5, chi tiết thông báo của xAI tháng 05/2025, và kết quả chạy lại IFScale của Arize. Số liệu token tiếng Việt ở §3.6 được đo trực tiếp trên một đoạn văn bản khoảng 80 từ. Tỷ lệ trên văn bản của bạn có thể khác, nên hãy đo lại trên dữ liệu thật.
