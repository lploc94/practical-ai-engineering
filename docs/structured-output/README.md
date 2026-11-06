# Structured output: để LLM trả về dữ liệu, không chỉ văn bản

Giả sử bạn cần đọc email khiếu nại và đưa ba thông tin lên dashboard: tên sản phẩm, mã đơn hàng và mức độ bực bội của khách, chấm từ 1 đến 5. Bạn mô tả các field trong prompt, thêm câu "chỉ trả về JSON", rồi thử với vài email. Kết quả parse được, các cột trên dashboard có dữ liệu.

Đến một email khác, model trả về:

````text
Dưới đây là thông tin bạn cần:
```json
{"product": "Máy lọc nước A3", "order_id": null, "anger": "khá cao"}
```
````

`json.loads` dừng ngay ở câu dẫn. Bạn bỏ câu dẫn và code fence, nhưng `anger` vẫn là chuỗi, trong khi dashboard cần số. Ngay cả khi sửa được lỗi này, một response dùng `orderId` thay cho `order_id` vẫn có thể khiến cột mã đơn bị bỏ trống.

Ba lỗi ấy đòi hỏi ba điều kiện: output phải parse được, giá trị phải đúng kiểu, tên field phải đúng như code đang đọc. **Structured output** là tên chung cho các kỹ thuật giúp LLM đáp ứng những điều kiện này, từ hướng dẫn bằng prompt đến ràng buộc từng token theo JSON Schema.

Khi cấu trúc đã đúng, bạn còn một câu hỏi: mã đơn và mức độ bực bội có phản ánh đúng email không? Schema cho phép `anger: 5`, nhưng không thể xác nhận khách đang giận đến mức đó. Phân biệt hai việc này sẽ giúp bạn biết nên sửa schema, sửa prompt hay bổ sung kiểm tra trong code.

Bạn có thể đọc tài liệu theo thứ tự nếu đang xây pipeline trích xuất hoặc phân loại đầu tiên, hoặc mở thẳng phần liên quan khi làm agent, sinh UI hay LLM-as-judge:

- **Nền tảng — Phần 1:** chọn mức đảm bảo phù hợp, hiểu cách constrained decoding hoạt động và ảnh hưởng của schema đến câu trả lời.
- **Tra cứu API — Phần 2:** tham số, tập con JSON Schema và giới hạn của từng provider, engine self-hosted, thư viện.
- **Thiết kế và vận hành — Phần 3:** viết schema, kiểm tra dữ liệu, retry, streaming, reasoning, tools và eval. Khi cần tìm nguyên nhân một lỗi cụ thể, xem [bảng triệu chứng ở §3.12](#312-failure-modes-thường-gặp).

> Bản tháng 11/2026. Thuật ngữ kỹ thuật và tên field giữ nguyên tiếng Anh để tiện đối chiếu API. Tham số và giới hạn được ghi nhận tại thời điểm biên soạn; các mục chưa xác minh được đánh dấu tại chỗ.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Vì sao LLM không tự trả về dữ liệu đúng format](#11-vì-sao-llm-không-tự-trả-về-dữ-liệu-đúng-format)
  - [1.2 Phổ các mức đảm bảo](#12-phổ-các-mức-đảm-bảo)
  - [1.3 Constrained decoding hoạt động thế nào](#13-constrained-decoding-hoạt-động-thế-nào)
  - [1.4 JSON Schema: ngôn ngữ chung, nhưng mỗi nơi nói một phương ngữ](#14-json-schema-ngôn-ngữ-chung-nhưng-mỗi-nơi-nói-một-phương-ngữ)
  - [1.5 Ép format có làm model kém đi không?](#15-ép-format-có-làm-model-kém-đi-không)
  - [1.6 Đúng schema chưa phải là đúng](#16-đúng-schema-chưa-phải-là-đúng)
- [Phần 2 — Structured output theo provider](#phần-2--structured-output-theo-provider)
  - [2.1 Bức tranh chung](#21-bức-tranh-chung)
  - [2.2 OpenAI](#22-openai)
  - [2.3 Claude (Anthropic)](#23-claude-anthropic)
  - [2.4 Google Gemini](#24-google-gemini)
  - [2.5 xAI Grok](#25-xai-grok)
  - [2.6 Các provider Trung Quốc](#26-các-provider-trung-quốc)
  - [2.7 Self-hosted: vLLM, SGLang, llama.cpp, Ollama](#27-self-hosted-vllm-sglang-llamacpp-ollama)
  - [2.8 Thư viện](#28-thư-viện)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Thiết kế schema](#31-thiết-kế-schema)
  - [3.2 Schema chạy được trên nhiều provider](#32-schema-chạy-được-trên-nhiều-provider)
  - [3.3 Validation nhiều lớp và retry](#33-validation-nhiều-lớp-và-retry)
  - [3.4 Streaming partial JSON](#34-streaming-partial-json)
  - [3.5 Structured output với reasoning model](#35-structured-output-với-reasoning-model)
  - [3.6 Structured output trong agent và MCP](#36-structured-output-trong-agent-và-mcp)
  - [3.7 Các ứng dụng điển hình](#37-các-ứng-dụng-điển-hình)
  - [3.8 Chọn format: JSON, YAML, XML, TOON](#38-chọn-format-json-yaml-xml-toon)
  - [3.9 Chi phí, độ trễ và cache](#39-chi-phí-độ-trễ-và-cache)
  - [3.10 Eval cho structured output](#310-eval-cho-structured-output)
  - [3.11 Observability](#311-observability)
  - [3.12 Failure modes thường gặp](#312-failure-modes-thường-gặp)
  - [3.13 Checklist cho AI agent](#313-checklist-cho-ai-agent)
- [Tổng hợp Sources](#tổng-hợp-sources)

---

<a id="tldr--cheat-sheet"></a>

## Tra cứu nhanh

### Những quyết định cần chốt trước khi viết code

1. Chọn strict JSON Schema hoặc strict tool nếu provider hỗ trợ. JSON mode chỉ bảo đảm cú pháp; hướng dẫn trong prompt không bảo đảm cả cú pháp lẫn schema.
2. Ghi đơn vị, thang đo và cách xử lý dữ liệu thiếu ngay trong tên field hoặc `description`. Model đọc những phần này khi điền giá trị.
3. Cho phép `null` khi input có thể thiếu thông tin, và thêm `other` nếu danh sách enum chưa bao quát mọi trường hợp. Nếu mọi giá trị hợp lệ đều sai với input, model vẫn phải chọn một giá trị để hoàn thành output.
4. Với tác vụ cần suy luận, dùng thinking hoặc đặt field bằng chứng trước field kết luận. Nếu dùng cách thứ hai, kiểm tra thứ tự key thực tế của provider.
5. Xử lý output bị cắt, refusal và content rỗng trước khi parse. Cả ba đều có thể đi kèm HTTP 200.
6. Validate bằng Pydantic / Zod model gốc, sau đó chạy business rules. Schema gửi đến provider có thể đã bị bỏ bớt ràng buộc.
7. Đánh giá độ đúng theo từng field. Theo dõi riêng những lần model điền giá trị khi đáp án phải là `null`.

### Chọn cơ chế nào

| Tình huống | Cơ chế |
|---|---|
| Câu trả lời cuối là dữ liệu (extraction, classification, judge) | Response format `json_schema` strict |
| Model gọi function của bạn | Tool với `strict: true` (Gemini: mode `VALIDATED`) |
| Output là ngôn ngữ khác JSON (SQL, DSL, format riêng) | Grammar: custom tool Lark / regex (OpenAI), `grammar` / `ebnf` (vLLM, SGLang), GBNF (llama.cpp) |
| Provider chỉ có JSON mode | JSON mode + schema trong prompt + validate + retry (Instructor, BAML) |
| Bài toán suy luận nặng | Reasoning model + schema cho câu trả lời cuối, hoặc trả lời tự do rồi chuyển sang schema ở bước hai |
| Cần chạy trên nhiều provider | Schema theo mẫu số chung ([§3.2](#32-schema-chạy-được-trên-nhiều-provider)) |

### So sánh nhanh (snapshot tháng 11/2026)

| Provider | JSON Schema ép khi giải mã | Strict tools | Optional field | Đệ quy | Ràng buộc số / độ dài | Lưu ý lớn nhất |
|---|---|---|---|---|---|---|
| **OpenAI** | `json_schema` + `strict` | Có | Không, dùng union `null` | Có | Số: có; độ dài chuỗi: không | `strict` mặc định `false` trên Chat Completions |
| **Claude** | `output_config.format` | Có | Có | Không | Không (400) | Không prefill, không citations cùng JSON; giới hạn 20 / 24 / 16 |
| **Gemini** | `response_format.schema`, `responseJsonSchema` | `VALIDATED` | Có, mặc định | Có điều kiện | Số: có; độ dài: bỏ qua lặng lẽ | Thứ tự key; thinking ăn hết `max_output_tokens` |
| **Grok** | `json_schema` | Luôn bật | Có | Không vòng | Có, tới ngưỡng | `additionalProperties` mặc định `false` |
| **DeepSeek** | Không (chỉ JSON mode) | Có, qua `/beta` | Không | Có (strict tool) | Số: có; độ dài, mảng: không | Content toàn khoảng trắng khi JSON hỏng |
| **Qwen** | Một số model 3.7 / 3.8 | Không ghi | — | — | Một số keyword 400 | Input ảnh / audio không được áp schema |
| **Kimi** | Có, dialect MFJS | Không | Nullable | — | — | K3 luôn thinking; 200 chưa chắc schema được áp |
| **GLM** | Không (chỉ JSON mode) | Không | — | — | — | Validate phía client |
| **MiniMax** | Docs mâu thuẫn | Không | — | — | — | Chỉ `tool_choice: auto` |
| **Doubao** | Beta, model từ 250615 | Có | Không, dùng union `null` | — | — | Phụ thuộc hậu tố phiên bản model |
| **vLLM / SGLang** | Có (xgrammar, guidance...) | Qua JSON Schema / structural tag | Tuỳ backend | Tuỳ backend | Tuỳ backend | vLLM gỡ `guided_*` từ v0.12.0 |

"—" là tài liệu không ghi rõ. Chi tiết và nguồn ở Phần 2.

---

# Phần 1 — Nền tảng

Với email ở đầu bài, thêm "không có câu dẫn" vào prompt có thể sửa được response vừa gặp. Nhưng để quyết định có thể tin cậy cách sửa đó đến đâu, cần biết điều gì kiểm soát token mà model được phép sinh.

## 1.1 Vì sao LLM không tự trả về dữ liệu đúng format

Model sinh từng token theo phân phối xác suất. Dù prompt yêu cầu JSON, các token mở đầu như `Dưới`, `Here` hay `` ``` `` vẫn có thể được chọn thay cho `{`. Một prompt chạy tốt trên vài mẫu chưa cho biết nó sẽ ổn định trên hàng nghìn email. Các lỗi thường xuất hiện từ ba nguồn:

- **Thói quen hội thoại.** Model được tinh chỉnh để trò chuyện, nên hay thêm câu mở đầu, giải thích và markdown.
- **Output dài và lồng sâu.** Mỗi token thêm vào là thêm một cơ hội sai: quên đóng ngoặc, thừa dấu phẩy, field bị lặp.
- **Prompt không định nghĩa đủ.** "Trả về JSON có field `anger`" không nói `anger` là số hay chuỗi, có được null không, thang đo là gì.

Lỗi parse thường dễ phát hiện vì chương trình ném exception. Lỗi thiếu field khó thấy hơn: nếu code dùng `data.get("order_id")`, một response có key `orderId` sẽ được xử lý như email không có mã đơn. Dashboard vẫn chạy, nhưng dữ liệu đã sai.

## 1.2 Phổ các mức đảm bảo

Một chỉ dẫn trong prompt làm thay đổi xác suất chọn token. Constrained decoding đi xa hơn: bộ giải mã loại các token vi phạm ràng buộc khỏi tập lựa chọn. Những cơ chế thường gặp nằm trong bảng sau.

| Mức | Cơ chế | Đảm bảo | Không đảm bảo |
|---|---|---|---|
| 0. Prompt | Dặn "chỉ trả JSON", cho ví dụ, few-shot | Không gì cả | Parse được, đúng schema |
| 1. Prefill / stop sequence | Mở sẵn `{` trong lượt assistant, dừng ở `}` cuối | Output bắt đầu đúng chỗ | Đóng ngoặc, đúng schema |
| 2. JSON mode | `response_format: {"type": "json_object"}` | JSON hợp lệ về cú pháp | Tên field, kiểu, field bắt buộc |
| 3. Tool / function calling (không strict) | Model điền arguments cho một tool có schema | Thường đúng schema, có trường hợp không | Đúng schema 100%, model có gọi tool |
| 4. Strict schema (constrained decoding) | `json_schema` + `strict: true`, `strict` trên tool | Output khớp schema đã khai báo (trong subset được hỗ trợ) | Giá trị đúng ngữ nghĩa, output không bị cắt |
| 5. Grammar tuỳ ý | Regex, CFG (Lark, GBNF, EBNF) | Output khớp ngữ pháp, kể cả format không phải JSON | Như trên |

Ở mức 4, nếu `anger` có kiểu `integer`, bộ giải mã chặn các token tạo ra chuỗi `"khá cao"`. Bạn không còn phải dựa vào việc model làm theo lời dặn về kiểu dữ liệu. Cơ chế chặn token được giải thích ở [§1.3](#13-constrained-decoding-hoạt-động-thế-nào).

Ràng buộc này chỉ có ích khi model hoàn thành câu trả lời. Output vẫn có thể bị cắt do `max_tokens`, bị thay bằng refusal, hoặc chứa giá trị đúng kiểu nhưng sai sự thật. Code nhận response cần xử lý riêng cả ba trường hợp.

### JSON mode khác structured output thế nào

`{"foo": 1}` là JSON hợp lệ, nhưng không có thông tin nào mà dashboard đang cần. **JSON mode** chấp nhận output đó vì nó chỉ kiểm soát cú pháp. **Structured output theo schema** còn ràng buộc tên field, kiểu và các quy tắc thuộc tập con mà provider hỗ trợ. OpenAI gọi tính năng này là Structured Outputs; các provider khác dùng tên riêng. Với API chỉ có JSON mode, bạn vẫn phải mô tả schema trong prompt và tự validate.

### Tool calling cũng là structured output

Khi model gọi tool, arguments là một JSON object theo `parameters` / `input_schema` của tool đó. Trước khi các provider có `json_schema` riêng, cách phổ biến để trích dữ liệu có cấu trúc là khai báo một tool giả như `record_extraction` rồi ép model gọi nó. Ngày nay hai cơ chế thường dùng chung một bộ máy constrained decoding phía provider, nhưng mục đích khác nhau:

- **Response format** quy định *câu trả lời cuối* của model.
- **Strict tool** quy định *arguments* khi model quyết định gọi tool. Model vẫn có thể không gọi tool nào.

Một agent thường dùng cả hai: tool strict cho các bước trung gian, response format cho kết quả cuối mà code downstream đọc.

## 1.3 Constrained decoding hoạt động thế nào

### Che token trước khi sample

Ở mỗi bước, model trả về một logit cho mỗi token trong vocabulary (thường 100K–200K token). Bình thường bộ sampler chọn token từ phân phối đó. Với constrained decoding, một **bộ ràng buộc** chạy song song, giữ trạng thái của một automaton biểu diễn schema. Trước khi sample, nó đặt logit của mọi token không hợp lệ ở trạng thái hiện tại thành `-inf`. Token bị che không thể được chọn, dù model đánh giá nó cao đến đâu [45].

```text
Schema: {"anger": integer 1..5}
Output đến lúc này:  {"anger":
Token model muốn:    " \"" (0.41)   " 4" (0.38)   " \"kh" (0.12)  ...
Sau khi che:         " 4" (0.38 → chuẩn hoá lại)   " 3"   " 5"  ...
```

Sau khi chọn token, automaton cập nhật trạng thái để tính tập token hợp lệ cho bước tiếp theo. Cơ chế này giải thích hai đặc điểm của constrained decoding:

- Bộ ràng buộc chỉ loại lựa chọn; nó không giải thích ý nghĩa của field cho model. Nếu model định trả một chuỗi nhưng schema yêu cầu số, model phải chọn trong số các token còn được phép. Con số đó chưa chắc đúng. Provider vì vậy thường chèn schema vào prompt để model đọc được cả tên field lẫn `description`.
- Phần lớn token cố định có thể được sinh mà không cần gọi model. Sau `{"`, nếu chỉ còn một key hợp lệ là `anger`, engine có thể ghi luôn `anger":` thay vì chạy model qua từng token. SGLang gọi kỹ thuật này là *compressed FSM*, cộng đồng hay gọi là *jump-forward decoding* [51].

### Từ schema đến automaton

**Regex và schema phẳng** chuyển được thành máy trạng thái hữu hạn (FSM). Outlines (Willard & Louf, 2023) có đóng góp then chốt là xây sẵn một chỉ mục: với mỗi trạng thái FSM, token nào trong vocabulary hợp lệ. Lúc giải mã, việc che token chỉ còn là tra bảng [45].

**JSON có lồng nhau và đệ quy** cần một ngăn xếp để nhớ đang ở trong bao nhiêu cấp object, array. Ngữ pháp phi ngữ cảnh (CFG) và pushdown automaton làm được việc này, nhưng số trạng thái kết hợp với ngăn xếp không còn hữu hạn, nên không tính sẵn mọi mask được. Các engine hiện đại giải bài toán này theo nhiều cách:

- **XGrammar** chia token thành hai nhóm: token *độc lập ngữ cảnh*, kiểm tra trước một lần và cache theo từng trạng thái, và token *phụ thuộc ngữ cảnh*, chỉ nhóm nhỏ này phải kiểm tra lúc chạy, dùng một persistent stack. Công việc trên CPU được chạy chồng lên forward pass của GPU. Bài báo ghi tăng tốc tới 100 lần so với các giải pháp trước đó [46][47]. XGrammar là backend mặc định của SGLang và có trong vLLM, TensorRT-LLM.
- **llguidance** (Microsoft, viết bằng Rust) nhận JSON Schema, regex và CFG kiểu Lark, với chi phí khoảng 50 µs CPU mỗi token cho tokenizer 128K và gần như không tốn thời gian khởi tạo [49]. Nó được dùng trong Guidance, llama.cpp, vLLM, SGLang và Chromium. Nhóm phát triển cho biết OpenAI chuyển sang llguidance từ tháng 5/2025 [50]; custom tools với grammar Lark của OpenAI dùng "biến thể Lark của LLGuidance" ([§2.2](#22-openai)).

### Vấn đề ranh giới token

Grammar định nghĩa trên ký tự, còn model sinh token, mỗi token có thể gồm nhiều ký tự. Token `27` hợp lệ khi đang ở đầu một năm `YYYY`, nhưng không hợp lệ khi chỉ còn thiếu một chữ số. Engine phải mô phỏng từng token qua automaton để biết token đó có đưa output ra khỏi tập hợp lệ không. Một số token còn là chuỗi byte UTF-8 dang dở, đặc biệt với tiếng Việt, tiếng Trung, emoji, nên engine phải làm việc ở mức byte [50].

Ràng buộc đôi khi buộc một từ được tách thành chuỗi token khác với cách model thường gặp trong huấn luyện, và chất lượng có thể giảm nhẹ. Với field chứa văn bản dài, nên tránh `pattern` phức tạp nếu không có yêu cầu cụ thể cần kiểm soát.

### Cái giá về hiệu năng

| Chi phí | Khi nào | Mức độ |
|---|---|---|
| Compile grammar | Lần đầu gặp một schema | Vài ms đến vài giây với schema lớn. Claude cache grammar 24 giờ, timeout 180 giây; OpenAI ghi request đầu với schema mới chậm hơn |
| Tính mask mỗi token | Mỗi bước giải mã | Hàng chục µs với XGrammar, llguidance; đáng kể hơn với engine cũ hoặc schema nhiều union |
| Token tiết kiệm được | Key và dấu câu cố định | Jump-forward bỏ qua được nhiều bước model |
| Token thêm vào prompt | Mỗi request | Provider chèn schema hoặc hướng dẫn format vào context |

Khi dùng API, chi phí dễ nhận ra nhất là request đầu với schema mới chậm hơn các request sau. Giữ schema ổn định giúp tái sử dụng grammar đã compile; chỉ sinh schema theo từng request khi cấu trúc dữ liệu thực sự cần thay đổi.

### Engine không bao phủ hết JSON Schema

JSONSchemaBench (01/2025) chạy khoảng 10.000 schema thật lấy từ GitHub, Kubernetes, Snowplow, JSON Schema Store qua sáu framework. Một số kết quả về *empirical coverage* (tỉ lệ schema được xử lý và sinh output hợp lệ) [73]:

| Bộ dữ liệu | LM không ràng buộc | Guidance | llama.cpp | Outlines | XGrammar | OpenAI |
|---|---|---|---|---|---|---|
| GitHub Easy | 0,65 | 0,86 | 0,75 | 0,59 | 0,79 | 0,29 |
| GitHub Hard | 0,13 | 0,41 | 0,39 | 0,03 | 0,28 | 0,09 |
| Kubernetes | 0,56 | 0,91 | 0,76 | 0,57 | 0,07 | 0,21 |

Coverage thấp của OpenAI chủ yếu do API từ chối schema có keyword ngoài tập con được hỗ trợ; với schema đã nhận, tỉ lệ tuân thủ rất cao. Bài báo phân biệt lỗi **over-constrained**, khi engine chặn cả output hợp lệ, và **under-constrained**, khi engine để lọt output sai. XGrammar ít lỗi compile nhất nhưng nhiều lỗi under-constrained nhất ở thời điểm khảo sát [73].

Đây là số liệu đầu năm 2025; các engine đã thay đổi, trong đó XGrammar-2 ra đầu 2026 [48]. Dùng bảng để hiểu hai loại lỗi cần kiểm tra, còn quyết định chọn engine nên dựa trên thử nghiệm với schema và phiên bản sẽ chạy production, nhất là khi schema dùng keyword ít phổ biến.

## 1.4 JSON Schema: ngôn ngữ chung, nhưng mỗi nơi nói một phương ngữ

Hầu như mọi API đều nhận schema dưới dạng JSON Schema hoặc một biến thể. Trong code, bạn hiếm khi viết tay schema này. Bạn khai báo một Pydantic model hay một Zod schema, và SDK chuyển nó thành JSON Schema:

```python
from typing import Literal
from pydantic import BaseModel, Field

class Complaint(BaseModel):
    product: str = Field(description="Tên sản phẩm như khách viết, không chuẩn hoá")
    order_id: str | None = Field(description="Mã đơn dạng DH-123456, null nếu email không nhắc tới")
    anger: int = Field(ge=1, le=5, description="1 = bình thường, 5 = doạ huỷ đơn / bóc phốt")
    category: Literal["delivery", "defect", "billing", "other"]
```

```json
{
  "type": "object",
  "properties": {
    "product":  {"type": "string", "description": "Tên sản phẩm như khách viết, không chuẩn hoá"},
    "order_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Mã đơn dạng DH-123456, null nếu email không nhắc tới"},
    "anger":    {"type": "integer", "minimum": 1, "maximum": 5, "description": "1 = bình thường, 5 = doạ huỷ đơn / bóc phốt"},
    "category": {"type": "string", "enum": ["delivery", "defect", "billing", "other"]}
  },
  "required": ["product", "order_id", "anger", "category"],
  "additionalProperties": false
}
```

Trong schema này, `minimum` và `maximum` giới hạn `anger` trong khoảng 1–5. Còn `description` giải thích số 1 và số 5 có nghĩa gì. Bộ giải mã dùng phần ràng buộc để kiểm soát cấu trúc; model dùng tên field và mô tả trong context để chọn giá trị. Hai phần bổ sung cho nhau: có khoảng số mà thiếu định nghĩa thang đo thì những email tương tự vẫn có thể bị chấm khác nhau.

Một schema hợp lệ theo chuẩn JSON Schema chưa chắc được API chấp nhận. Mỗi provider chỉ hỗ trợ một tập con; khi chuyển provider, cần kiểm tra các điểm sau:

| Khía cạnh | Câu hỏi cần kiểm tra với từng provider |
|---|---|
| Field bắt buộc | Có bắt mọi field nằm trong `required` không? Optional phải biểu diễn bằng union với `null`? |
| `additionalProperties` | Có bắt buộc `false` không? |
| Ràng buộc số và chuỗi | `minimum`, `maximum`, `pattern`, `format`, `minLength` được *ép* khi giải mã, chỉ được *gợi ý* cho model, hay bị từ chối? |
| Mảng | `minItems`, `maxItems`, `uniqueItems` có được hỗ trợ không? |
| Union | `anyOf`, `oneOf` được phép ở đâu? Root có được là union không? |
| Đệ quy | `$ref`, `$defs`, schema tự tham chiếu có được không? |
| Kích thước | Giới hạn số property, độ sâu lồng, số giá trị enum, tổng độ dài chuỗi? |
| Thứ tự key | Output giữ thứ tự khai báo trong schema hay theo thứ tự khác? |

[§2.1](#21-bức-tranh-chung) tổng hợp câu trả lời cho từng provider. [§3.2](#32-schema-chạy-được-trên-nhiều-provider) bàn cách viết một schema chạy được ở nhiều nơi.

## 1.5 Ép format có làm model kém đi không?

Với email khiếu nại, phần lớn dữ liệu cần trích đã có trong input. Nhưng nếu yêu cầu model giải một bài toán rồi trả kết quả theo schema, thứ tự sinh output có thể ảnh hưởng đến đáp án. Các nghiên cứu dưới đây cho thấy vì sao cần phân biệt hai loại tác vụ này.

### Các nghiên cứu cho kết quả khác nhau

**"Let Me Speak Freely?"** (Tam et al., 08/2024) báo cáo khả năng suy luận của LLM giảm đáng kể khi bị ràng buộc format, và format càng chặt thì giảm càng nhiều trên các bài toán suy luận như GSM8K. Với tác vụ phân loại, ràng buộc lại có thể giúp [69].

**"Say What You Mean"** (nhóm .txt, tác giả Outlines) chạy lại thí nghiệm và cho kết quả ngược lại: trên Llama-3-8B-Instruct, structured generation bằng hoặc tốt hơn unstructured trên GSM8K, Last Letter và Shuffle Object (ví dụ Last Letter 0,73 so với 0,77). Họ chỉ ra ba vấn đề trong thí nghiệm gốc: dùng Claude 3 Haiku làm "AI parser" để chấm output tự do (regex viết tay cho điểm cao hơn), prompt cho JSON mode không hề nhắc tới JSON hay schema, và prompt giữa hai điều kiện không tương đương [70]. Một bản tái lập độc lập xác nhận phần lớn phản biện, nhưng ghi nhận rằng kết quả phụ thuộc vào model và cách prompt [71].

**CRANE** (ICML 2025) đưa ra giải thích cơ chế: grammar quá chặt loại bỏ không gian cho model suy luận. Thêm vào grammar các quy tắc cho phép đoạn suy luận trước phần output có cấu trúc giúp giữ được khả năng suy luận mà output vẫn đúng cú pháp, cải thiện tới 10 điểm phần trăm trên GSM-symbolic và FOLIO so với baseline [72].

### Điều rút ra cho thực tế

Từ các kết quả này, một lựa chọn thiết kế cần kiểm thử là vị trí của đáp án. Nếu output bắt đầu ngay bằng `"answer": `, model phải chọn đáp án trước khi sinh phần giải thích. Với tác vụ cần nhiều bước suy luận, bạn có ba cách bố trí:

1. **Dùng reasoning model hoặc bật thinking.** Với Claude, OpenAI reasoning models, Gemini thinking, phần suy luận nằm ngoài vùng bị ràng buộc. Grammar chỉ áp dụng cho câu trả lời cuối, nên không cần thêm phần suy luận vào schema chỉ để model có bước này.
2. **Đặt field suy luận trước field kết luận.** Vì nhiều provider sinh key theo thứ tự trong schema ([§2.1](#21-bức-tranh-chung)), một field `reasoning` hay `analysis` đặt trước `answer` cho model suy nghĩ rồi mới kết luận. Giữ field này ngắn, có giới hạn độ dài trong `description`, và lưu ý trên Claude, yêu cầu "lập luận từng bước" trong một property có thể bị từ chối ([§2.3](#23-claude-anthropic)).
3. **Tách hai bước.** Bước một để model trả lời tự do, bước hai dùng một model rẻ chuyển câu trả lời sang schema. Tốn thêm một request nhưng cô lập hoàn toàn suy luận khỏi format.

Với trích xuất và phân loại đơn giản, ràng buộc format thường hữu ích vì câu trả lời đã nằm trong input. Khi tác vụ cần suy luận, hãy so sánh các cách bố trí trên cùng tập dữ liệu trước khi chọn.

## 1.6 Đúng schema chưa phải là đúng

Giả sử response đã parse được và mọi field đều đúng kiểu. Dashboard nhận `order_id: "DH-123456"` mà không báo lỗi. Nếu email không hề nhắc đến mã đơn này, schema vẫn không phát hiện được. Những lỗi như vậy thường bắt nguồn từ cách biểu diễn dữ liệu:

- **Field bắt buộc nhưng input không có dữ liệu.** Nếu `order_id` chỉ nhận `string`, model không thể trả `null` khi email thiếu mã đơn và có thể điền một mã trông hợp lý. Cho phép `null` hoặc `"unknown"` để biểu diễn trường hợp thiếu dữ liệu; một field `confidence` có thể ghi thêm mức độ chắc chắn.
- **Ép vào enum sai.** Email nói về bảo hành, nhưng enum chỉ có `delivery`, `defect`, `billing`. Model buộc phải chọn một, và lựa chọn đó trông chắc chắn như mọi lựa chọn khác. Thêm `other` và một field mô tả cho trường hợp `other`.
- **Đúng kiểu, sai đơn vị hoặc thang đo.** `anger: 4` không cho biết model hiểu thang 1–5 theo nghĩa nào. `price: 1200` có thể là nghìn đồng hay đồng. Đơn vị và thang đo thuộc về `description`, hoặc tốt hơn, nằm trong tên field (`price_vnd`).
- **Mất thông tin vì schema quá hẹp.** Khách phàn nàn về hai sản phẩm, schema chỉ có một field `product`. Model chọn một, cái còn lại biến mất mà không có lỗi nào.

Sau khi kiểm tra cấu trúc, code vẫn cần đối chiếu mã đơn với dữ liệu hệ thống, cộng lại các dòng tiền và kiểm tra giá trị trích xuất với văn bản gốc. [§3.3](#33-validation-nhiều-lớp-và-retry) trình bày thứ tự các bước kiểm tra; [§3.10](#310-eval-cho-structured-output) hướng dẫn đo những lỗi còn lại trên dữ liệu thật.

---

# Phần 2 — Structured output theo provider

## 2.1 Bức tranh chung

Khi chuyển provider, request có thể vẫn được chấp nhận trong khi mức ràng buộc đã thay đổi. Một nơi từ chối `minLength`, nơi khác nhận nhưng bỏ qua; một nơi luôn sinh đủ key, nơi khác cho phép bỏ field optional. Bảng dưới đây giúp đối chiếu những khác biệt này trước khi dùng các ví dụ API ở từng mục.

| | OpenAI | Claude | Gemini | Grok |
|---|---|---|---|---|
| Response format | `response_format` / `text.format` `json_schema` | `output_config.format` `json_schema` | `response_format.schema` (Interactions), `responseJsonSchema` | `response_format` / `text.format` `json_schema` |
| Strict tool args | `strict: true` | `strict: true` | Mode `VALIDATED` | Luôn bật |
| Field optional | Không, mọi field `required`; dùng union `null` | Có | Có (mặc định optional) | Có |
| `additionalProperties: false` | Bắt buộc ghi | Bắt buộc ghi | Tuỳ chọn | Mặc định |
| Đệ quy | Có | **Không** | Có, qua field optional | Không (chỉ `$ref` không vòng) |
| `minimum` / `maximum` | Ép | **Không**, 400 | Ép | Ép |
| `minLength` / `maxLength` | Không nằm trong danh sách hỗ trợ | **Không**, 400 | Bỏ qua lặng lẽ | Ép tới 2.048 |
| `minItems` / `maxItems` | Ép | Chỉ `minItems` 0 hoặc 1 | Ép | Ép tới 256 |
| Thứ tự key | Theo schema | Bắt buộc trước, optional sau | `propertyOrdering`; không có thì xem [§2.4](#24-google-gemini) | Không ghi |
| Keyword không hỗ trợ | 400 | 400 | Phần lớn bỏ qua lặng lẽ | 400 hoặc best-effort |
| Grammar tuỳ ý | Custom tool Lark / regex | Không | Không | Không |

Các khác biệt thường ảnh hưởng trực tiếp đến code đọc output là:

1. **Optional.** Schema theo kiểu OpenAI (mọi field `required`, optional là union `null`) chạy được ở mọi nơi. Schema có field ngoài `required` sẽ bị OpenAI strict mode từ chối.
2. **Ràng buộc số và độ dài.** Claude từ chối `minimum` và `minLength`; Gemini ép `minimum` nhưng bỏ qua `minLength` mà không báo; OpenAI ép `minimum` nhưng không liệt kê `minLength` trong danh sách hỗ trợ; Grok ép cả hai. Cùng một Pydantic model `Field(ge=1, le=5)` cho ba hành vi khác nhau. Ràng buộc kiểu này luôn phải được validate lại phía client.
3. **Đệ quy.** Schema cây (comment lồng comment, org chart, AST) chạy trên OpenAI và Gemini nhưng không chạy trên Claude và Grok. Cách thay: làm phẳng thành danh sách node có `parent_id`.

Nếu cần fallback hoặc so sánh model giữa các provider, xem cách tổ chức schema và validation ở [§3.2](#32-schema-chạy-được-trên-nhiều-provider).

## 2.2 OpenAI

OpenAI giới thiệu Structured Outputs vào tháng 8/2024. Trên bộ eval schema phức tạp trong bài ra mắt, `gpt-4o-2024-08-06` với Structured Outputs đạt 100%, còn `gpt-4-0613` đạt dưới 40% [6]. Các endpoint tương thích OpenAI thường dùng lại cú pháp request này, dù tập con schema và mức đảm bảo có thể khác.

### Các cơ chế

| Cơ chế | Vị trí | Ghi chú |
|---|---|---|
| JSON mode | `response_format: {"type": "json_object"}` (Chat Completions), `text.format` (Responses) | Chỉ bảo đảm JSON hợp lệ. Bắt buộc có chữ "JSON" trong context, không có thì API báo lỗi [1] |
| Structured Outputs | `response_format: {"type": "json_schema", "json_schema": {...}}` (Chat Completions), `text.format` (Responses) | `strict` mặc định là `false` trên Chat Completions, phải bật tường minh [3] |
| Strict function calling | `strict: true` trên function tool | Ràng buộc arguments [2] |
| Custom tool với grammar | `{"type": "custom", "format": {"type": "grammar", "syntax": "lark" \| "regex", ...}}` (Responses) | Input là text tự do theo grammar, không phải JSON [2] |

```json
// Chat Completions
"response_format": {
  "type": "json_schema",
  "json_schema": {"name": "invoice", "strict": true, "schema": {"type": "object", "...": "..."}}
}

// Responses API
"text": {
  "format": {"type": "json_schema", "name": "invoice", "strict": true, "schema": {"...": "..."}}
}

// Function tool (Responses API, dạng phẳng)
{
  "type": "function",
  "name": "get_weather",
  "parameters": {"type": "object", "properties": {"...": "..."}, "required": ["..."], "additionalProperties": false},
  "strict": true
}
```

Với JSON mode, cần chỉ dẫn rõ trong prompt rằng output phải là JSON. Nếu thiếu, model có thể sinh khoảng trắng cho đến khi hết ngân sách token [1]. Cách nhận biết và xử lý nằm trong [bảng lỗi ở §3.12](#312-failure-modes-thường-gặp).

### Quy tắc strict mode

Schema gửi đến strict mode phải đáp ứng ba quy tắc sau [1][2]:

1. **Mọi field phải nằm trong `required`.** Field optional được biểu diễn bằng union với `null`: `"type": ["string", "null"]`. Model luôn sinh key đó, chỉ là giá trị có thể là `null`.
2. **`additionalProperties: false` ở mọi object**, kể cả object lồng trong `$defs`. Thiếu ở một chỗ là cả request bị 400.
3. **Root phải là object và không được là `anyOf`.** Một Zod discriminated union đặt ở root sẽ không hợp lệ. Cách xử lý: bọc union vào một field, ví dụ `{"result": {"anyOf": [...]}}`. Riêng `parameters` của function cho phép root `anyOf` nếu mọi nhánh là object [2].

| Nhóm | Hỗ trợ | Không hỗ trợ |
|---|---|---|
| Kiểu | `string`, `number`, `integer`, `boolean`, `object`, `array`, `enum`, `anyOf` | |
| Chuỗi | `pattern`, `format`: `date-time`, `time`, `date`, `duration`, `email`, `hostname`, `ipv4`, `ipv6`, `uuid` | |
| Số | `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf` | |
| Mảng | `minItems`, `maxItems` | |
| Đệ quy | `$defs` + `$ref`, kể cả `"$ref": "#"` | |
| Kết hợp | | `allOf`, `not`, `if` / `then` / `else`, `dependentRequired`, `dependentSchemas` |

Với model fine-tuned, danh sách bị thu hẹp thêm: không có `minLength`, `maxLength`, `pattern`, `format`, `minimum`, `maximum`, `multipleOf`, `patternProperties`, `minItems`, `maxItems` [1].

Giới hạn kích thước [1]:

| Giới hạn | Giá trị |
|---|---|
| Tổng số property | 5.000 |
| Độ sâu lồng | 10 cấp |
| Tổng độ dài tên property, tên definition, giá trị enum và const | 120.000 ký tự |
| Tổng số giá trị enum | 1.000 |
| Một enum chuỗi có trên 250 giá trị | Tổng tối đa 15.000 ký tự |
| Số tools mỗi request | 350 [2] |

Output giữ thứ tự key trong schema [1]. Nếu muốn model sinh bằng chứng trước điểm số, hãy khai báo các field theo thứ tự đó.

### Refusal, truncation và độ trễ lần đầu

- **Refusal**: khi model từ chối vì lý do an toàn, Chat Completions trả `message.refusal` (chuỗi), Responses API trả content item `type: "refusal"`. Output lúc đó không theo schema [1][3].
- **Truncation**: `finish_reason: "length"` (Chat Completions) hoặc `status: "incomplete"` với `incomplete_details.reason: "max_output_tokens"` (Responses). JSON bị cắt dở [1][3].
- **Độ trễ lần đầu**: request đầu tiên với một schema mới chậm hơn vì API phải xử lý schema, các request sau với cùng schema thì không [1]. Với model fine-tuned, schema đã cache không đủ điều kiện zero data retention [2].

SDK helper xử lý hai trường hợp đầu cho bạn. Python `parse()` raise `LengthFinishReasonError` khi bị cắt, và đặt kết quả parse là `None` khi có refusal [5]:

```python
from pydantic import BaseModel

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.responses.parse(
    model="gpt-6-astra",
    input=[{"role": "user", "content": "Alice và Bob đi hội chợ khoa học thứ Sáu này."}],
    text_format=CalendarEvent,
)
event = response.output_parsed  # None nếu model từ chối
```

```typescript
import { zodTextFormat } from "openai/helpers/zod";

const response = await client.responses.parse({
  model: "gpt-6-astra",
  input: [{ role: "user", content: "..." }],
  text: { format: zodTextFormat(CalendarEvent, "event") },
});
response.output_parsed;
```

Với tools, `openai.pydantic_function_tool(Model)` tạo function tool strict từ Pydantic; `parse()` chỉ nhận strict function tools và trả arguments đã parse ở `tool_call.function.parsed_arguments` [5].

### Đổi schema giữa các lượt: `configuration_update`

Trên các model GPT-6 trở lên (chế độ single-agent), Responses API cho đổi format giữa hai lượt bằng cách nối một item `configuration_update` vào `input` trước user message tiếp theo [1]:

```json
{"type": "configuration_update", "text": {"format": {"type": "text"}}}
```

Mỗi update thay thế toàn bộ format và giữ hiệu lực qua các lượt sau. Lợi ích là phần lịch sử phía trước không đổi, nên không phải dựng lại context như khi đổi `text.format` ở top-level.

### Custom tools với grammar

Khi output không phải JSON, như một câu SQL, một lệnh DSL hay một ngày theo format riêng, custom tool cho phép ràng buộc bằng grammar Lark hoặc regex [2]:

```json
{
  "type": "custom",
  "name": "timestamp",
  "description": "Ghi lại mốc thời gian dạng YYYY-MM-DD HH:MM",
  "format": {
    "type": "grammar",
    "syntax": "regex",
    "definition": "^\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01]) ([01]\\d|2[0-3]):[0-5]\\d$"
  }
}
```

Hai cú pháp này có giới hạn riêng. Lark ở đây là biến thể của LLGuidance: không có lookaround, không có lazy modifier trong regex của lexer, không có terminal priority, không import ngoài `%import common`. Regex dùng cú pháp Rust (không phải Python `re`) và chỉ một dòng. Grammar quá phức tạp có thể bị từ chối [2].

### Model hỗ trợ

Structured Outputs có trên mọi model từ GPT-4o trở đi, cụ thể `gpt-4o-mini`, `gpt-4o-2024-08-06` và các model sau [1]. Tại thời điểm viết, dòng flagship là `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6.1-terra` và `gpt-6-luna`; tài liệu khuyên bắt đầu dự án mới với `gpt-6-astra` [1][4].

### Các lỗi được báo trên issue tracker và forum

Những báo cáo dưới đây hữu ích khi chẩn đoán lỗi SDK; chúng chưa phải cam kết hành vi trong tài liệu API:

- `z.record()` của Zod sinh `propertyNames`, keyword strict mode không nhận, gây 400 [7].
- Dùng `response_format` loại `json_schema` cùng một function tool **không** strict có thể bị 400; cho tools strict luôn [8].
- Lỗi "maximum schema size exceeded" với enum dài; cách xử lý thường là rút ngắn giá trị enum hoặc tách schema [9].

## 2.3 Claude (Anthropic)

Claude có hai cơ chế dùng chung một bộ constrained decoding: **JSON outputs** cho câu trả lời cuối và **strict tool use** cho arguments của tool. Cả hai đã GA, không cần beta header [10][11].

### JSON outputs: `output_config.format`

```json
{
  "model": "claude-opus-5-5",
  "max_tokens": 1024,
  "messages": [{"role": "user", "content": "Trích thông tin liên hệ từ email sau: ..."}],
  "output_config": {
    "format": {
      "type": "json_schema",
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "demo_requested": {"type": "boolean"}
        },
        "required": ["name", "demo_requested"],
        "additionalProperties": false
      }
    }
  }
}
```

Kết quả nằm trong text block đầu tiên của `content`, dạng chuỗi JSON. Tham số top-level `output_format` của thời beta đã deprecated: không có header `structured-outputs-2025-11-13` thì trả 400, và Python SDK v1+ raise `TypeError` nếu truyền nó vào `create()` [10]. Code viết từ cuối 2025 cần chuyển sang `output_config.format`.

Với SDK, dùng helper `parse()`:

```python
from pydantic import BaseModel

class ContactInfo(BaseModel):
    name: str
    email: str
    demo_requested: bool

response = client.messages.parse(
    model="claude-opus-5-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": email_text}],
    output_format=ContactInfo,   # helper tự chuyển thành output_config.format
)
contact = response.parsed_output
```

```typescript
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";

const response = await client.messages.parse({
  model: "claude-opus-5-5",
  max_tokens: 1024,
  messages: [{ role: "user", content: emailText }],
  output_config: { format: zodOutputFormat(ContactInfo) },
});
response.parsed_output;
```

Helper chuyển các ràng buộc API không hỗ trợ, như `minimum` và `maxLength`, vào `description`, rồi kiểm tra chúng phía client khi parse. Bạn có thể dùng `transform_schema()` trong Python để xem schema sau biến đổi mà không gọi API. Riêng `jsonSchemaOutputFormat()` trong TypeScript nhận JSON Schema thuần nhưng không validate response; nếu dùng helper này, cần tự kiểm tra những ràng buộc như `minimum` [10].

### Strict tool use

Đặt `"strict": true` ở cấp tool definition, cạnh `name` và `input_schema`, không đặt trong `tool_choice` [11]:

```json
{
  "name": "get_weather",
  "description": "Lấy thời tiết hiện tại của một thành phố",
  "strict": true,
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {"type": "string"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location"],
    "additionalProperties": false
  }
}
```

Có thể dùng chung strict tools và `output_config.format` trong một request. Các toolset dựng sẵn như computer use và browser use không nhận `strict` [11].

### Subset JSON Schema được hỗ trợ

| Nhóm | Hỗ trợ | Không hỗ trợ (trả 400) |
|---|---|---|
| Kiểu | `object`, `array`, `string`, `integer`, `number`, `boolean`, `null` | |
| Giá trị | `enum` (string, number, boolean, null), `const`, `default` | `enum` chứa object/array |
| Kết hợp | `anyOf`, `allOf` | `allOf` cùng `$ref` |
| Tham chiếu | `$ref`, `$defs`, `definitions` nội bộ | `$ref` ra ngoài, **schema đệ quy** |
| Object | `required`; `additionalProperties: false` (bắt buộc) | `additionalProperties` khác `false` |
| Số | | `minimum`, `maximum`, `multipleOf` |
| Chuỗi | `format`: `date-time`, `time`, `date`, `duration`, `email`, `hostname`, `uri`, `ipv4`, `ipv6`, `uuid`; `pattern` với regex đơn giản | `minLength`, `maxLength`; backreference, lookaround, `\b` |
| Mảng | `minItems` bằng 0 hoặc 1 | `maxItems`, `minItems` > 1, `uniqueItems` |

Claude cho phép bỏ field khỏi `required`; model có thể không sinh key đó. Mỗi request cũng chịu các giới hạn về độ phức tạp sau [10]:

| Giới hạn | Giá trị |
|---|---|
| Số strict tools | 20 |
| Số tham số optional (tính trên mọi schema) | 24 |
| Số tham số có union type (`anyOf` hoặc type array) | 16 |
| Thời gian compile grammar | 180 giây, quá thì trả 400 `Schema is too complex for compilation.` |

Thứ tự key trong output: field bắt buộc trước, field optional sau, trong mỗi nhóm theo thứ tự khai báo [10]. Điều này ảnh hưởng đến kỹ thuật đặt field "lý do" trước field "kết luận" ([§3.1](#31-thiết-kế-schema)): nếu field lý do là optional còn field kết luận là bắt buộc, kết luận sẽ được sinh trước.

### Compile, cache và chi phí

Request đầu tiên với một schema mới chậm hơn vì grammar phải được compile. Grammar đã compile được cache 24 giờ kể từ lần dùng cuối. Đổi cấu trúc schema, hoặc đổi tập tools khi dùng chung với structured outputs, sẽ compile lại. Đổi `name` hay `description` thì không [10].

Cache grammar và prompt cache có vòng đời riêng. Đổi `output_config.format` làm mất prompt cache của hội thoại vì API chèn vào system prompt một đoạn giải thích format; đoạn này cũng được tính vào input token [10]. Giữ schema cố định trong hội thoại nhiều lượt giúp tránh chi phí đó.

### Tương tác với các tính năng khác

| Tính năng | Hành vi |
|---|---|
| Extended / adaptive thinking | Grammar chỉ áp cho output trực tiếp. Thinking và tool calls không bị ràng buộc, nên Claude suy luận tự do trước khi sinh JSON [10] |
| Streaming | Hỗ trợ. Phải gom đủ response mới deserialize được [10] |
| Prefill assistant | Không dùng chung được với JSON outputs. Các model mới (Opus 4.6+, Sonnet 4.6+, Opus 5.5, Sonnet 5.5) từ chối prefill hoàn toàn với lỗi 400 [10][12][13] |
| Citations | Bật cùng `output_config.format` trả 400 [10] |
| Batch API | Hỗ trợ, giảm 50% giá [10] |
| Managed Agents | Không có structured outputs [10] |
| Refusal | `stop_reason: "refusal"`, HTTP 200, vẫn tính tiền, output **có thể không khớp schema** [10][14] |
| Truncation | `stop_reason: "max_tokens"`: output bị cắt, không khớp schema [10] |
| Nền tảng | Claude API, Amazon Bedrock, Google Cloud, Microsoft Foundry [10] |

Ngoài các giới hạn trong bảng, cần xử lý hai trường hợp sau:

- **Hoa thường của enum có thể bị đổi.** Tài liệu ghi Claude có thể trả `"Conversation topic 3"` cho enum `"Conversation Topic 3"`. Tránh các giá trị enum chỉ khác nhau về hoa thường và so sánh không phân biệt hoa thường [10].
- **Field "giải thích từng bước" có thể bị từ chối.** Yêu cầu chain-of-thought chi tiết trong một property có thể kích hoạt refusal loại `reasoning_extraction`. Tài liệu khuyên xin "giải thích ngắn" thay vì "lập luận từng bước"; muốn model suy luận kỹ thì bật thinking [10].

Với dữ liệu nhạy cảm: khi bật ZDR, prompt và output không được lưu nhưng **JSON schema được cache tối đa 24 giờ**. Không đặt PHI hay dữ liệu cá nhân vào tên property, `enum`, `const` hay `pattern` [10].

### Các pattern cũ: prefill và forced tool choice

Trước structured outputs, hai cách phổ biến để lấy JSON từ Claude là prefill `{` vào lượt assistant và ép gọi tool bằng `tool_choice: {"type": "tool", "name": "..."}`. Cả hai đang bị loại bỏ:

- Prefill trả 400 `This model does not support assistant message prefill.` trên Opus 4.6+, Sonnet 4.6+, Sonnet 5 / 5.5. Sonnet 4.5, Haiku 4.5 và model cũ hơn vẫn nhận [12][13].
- `tool_choice` loại `any` hoặc `tool` trả 400 trên Fable 5.1, Opus 5.5, Sonnet 5.5 [12][13].

Cách thay thế: dùng `output_config.format` khi chỉ cần JSON, hoặc `tool_choice: {"type": "auto"}` cùng `strict: true` và prompt nói rõ khi nào gọi tool.

## 2.4 Google Gemini

Gemini Developer API cung cấp hai API với cách khai báo structured output khác nhau [15]:

- **Interactions API** (`POST /v1beta/interactions`), được khuyên dùng cho dự án mới, GA từ 06/2026.
- **generateContent** (`POST /v1beta/models/{model}:generateContent`), legacy nhưng vẫn được hỗ trợ đầy đủ.

### Interactions API: `response_format`

```json
{
  "model": "gemini-3.8-flash",
  "input": "Trích công thức nấu ăn từ đoạn sau: ...",
  "response_format": {
    "type": "text",
    "mime_type": "application/json",
    "schema": {
      "type": "object",
      "properties": {"name": {"type": "string"}, "ingredients": {"type": "array", "items": {"type": "string"}}},
      "required": ["name", "ingredients"]
    }
  }
}
```

```python
from google import genai

client = genai.Client()
interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=prompt,
    response_format={
        "type": "text",
        "mime_type": "application/json",
        "schema": Recipe.model_json_schema(),
    },
)
recipe = Recipe.model_validate_json(interaction.output_text)
```

Tham số `response_mime_type` cũ của Interactions đã bị gỡ; schema legacy ngừng hoạt động từ 08/06/2026 [16]. Code viết cho bản preview của Interactions cần chuyển sang `response_format`. Trên JavaScript, tài liệu dùng `z.fromJSONSchema(schema).parse(JSON.parse(interaction.output_text))` của Zod v4 để validate [15].

### generateContent: `responseJsonSchema` và `responseSchema`

| REST | Python (`google-genai`) | Trạng thái |
|---|---|---|
| `generationConfig.responseMimeType` | `response_mime_type` | `text/plain` (mặc định), `application/json`, `text/x.enum` |
| `generationConfig.responseJsonSchema` | `response_json_schema` | JSON Schema, **được khuyên dùng** |
| `generationConfig.responseSchema` | `response_schema` | Subset OpenAPI 3.0, legacy. Docs ghi: "Use `responseJsonSchema` rather than this field" [17] |

Hai field schema khác nhau ở cú pháp: `responseSchema` dùng tên kiểu viết hoa (`STRING`, `OBJECT`) và `nullable: true`, không có `$ref` hay `additionalProperties`; `responseJsonSchema` dùng JSON Schema chuẩn với kiểu viết thường và union với `null` [17]. Python SDK vẫn nhận một Pydantic class trực tiếp ở `response_schema` và trả kết quả đã parse ở `response.parsed`.

`text/x.enum` cho phép model chỉ trả một giá trị enum dạng chuỗi trơn, tiện cho phân loại. Interactions API **không** có mime type này; dùng `application/json` với một field `{"type": "string", "enum": [...]}` [18].

### Subset JSON Schema được hỗ trợ

| Nhóm | Hỗ trợ (ép khi giải mã) | Bị lặng lẽ bỏ qua |
|---|---|---|
| Kiểu | `string`, `number`, `integer`, `boolean`, `object`, `array`, `null`; union qua `["string", "null"]` | |
| Object | `properties`, `required`, `additionalProperties` (boolean hoặc schema) | `minProperties`, `maxProperties` |
| Chuỗi | `enum`, `format` (`date-time`, `date`, `time`) | `minLength`, `maxLength` |
| Số | `enum`, `minimum`, `maximum` | |
| Mảng | `items`, `prefixItems`, `minItems`, `maxItems` | `uniqueItems`, `contains`, `minContains`, `maxContains` |
| Kết hợp, tham chiếu | `anyOf`, `oneOf` (xử lý như `anyOf`), `$defs`, `$ref` (kể cả đệ quy) | |
| Mô tả | `title`, `description` | |

Các keyword ở cột cuối bị xoá khi API chuẩn hoá request, nên không gây lỗi [15]. Chẳng hạn, request từ một Pydantic model có `Field(min_length=...)` vẫn có thể thành công dù giới hạn độ dài không được áp dụng. Hãy validate output bằng model gốc để kiểm tra phần ràng buộc đã mất.

Đệ quy được hỗ trợ nhưng có điều kiện: tham chiếu đệ quy phải nằm ở property optional hoặc nullable, không đi qua property bắt buộc hay mảng có `minItems > 0` [15]. Tài liệu Vertex nói rõ hơn: vòng tham chiếu chỉ được "trải ra ở mức giới hạn" [23].

Field không nằm trong `required` có thể bị bỏ qua [23]. Nếu code luôn đọc một key, hãy đưa key đó vào `required`; việc giá trị có được là `null` hay không được quy định riêng trong kiểu của field.

### Giới hạn độ phức tạp

Gemini compile schema thành state machine, với giới hạn về tổng số trạng thái, số nhánh và độ sâu lồng. Request vượt giới hạn trả `400 INVALID_ARGUMENT` [15]. Các thành phần làm tăng độ phức tạp gồm:

- Tên property dài: mỗi ký tự tốn trạng thái.
- `minItems` / `maxItems`: schema của item bị nhân bản cho từng vị trí được đếm.
- `minimum` / `maximum`, enum nhiều giá trị, nhiều property optional, nhiều `anyOf`.

Cách xử lý được khuyên: chuyển quy tắc không mang tính cấu trúc vào `description`, làm phẳng object lồng, rút ngắn tên, bỏ giới hạn số lượng và kiểm tra trong code [15][23].

### Thứ tự key

Với `responseSchema` (legacy và Vertex), `propertyOrdering` quy định thứ tự sinh key. Nếu không có field này, tài liệu Vertex ghi rằng Python SDK giữ thứ tự khai báo; các trường hợp khác sinh theo thứ tự alphabet, field bắt buộc trước [23]. Tài liệu `responseJsonSchema` và Interactions không ghi rõ cơ chế thứ tự. Nếu schema cần `reasoning` trước `answer`, hãy kiểm thử trên đúng API đang dùng. Đặt tên như `a_reasoning`, `b_answer` phù hợp với thứ tự alphabet; dùng thinking thì không cần dựa vào thứ tự hai field này để tạo bước suy luận.

Tài liệu Vertex còn lưu ý: ví dụ và mô tả trong prompt nên trình bày field theo đúng thứ tự trong schema, lệch thứ tự có thể làm model lúng túng. Mục troubleshooting ghi output lặp chữ có thể do thứ tự field lệch với schema; cách xử lý là không chỉ định thứ tự trong prompt và đặt mọi field là bắt buộc [20][23].

### Function calling

| Mode (generateContent / Interactions) | Hành vi |
|---|---|
| `AUTO` / `auto` (mặc định) | Model chọn gọi function hay trả lời text |
| `ANY` / `any` | Luôn gọi function, có thể giới hạn bằng `allowedFunctionNames` |
| `NONE` / `none` | Không gọi function |
| `VALIDATED` / `validated` | Function call được kiểm tra bằng constrained decoding; model vẫn có thể trả text. Từ Gemini 3, mode này còn bảo đảm có đủ tham số bắt buộc [17][24] |

Tài liệu chỉ ghi rõ `VALIDATED` (và `ANY` trên Vertex) bảo đảm arguments theo schema. Với `AUTO`, validate arguments trước khi thực thi [17][24]. Trên generateContent, tool khai báo schema qua `parameters` (OpenAPI) hoặc `parametersJsonSchema` (JSON Schema), hai field loại trừ nhau [17].

### Kết hợp với tools và thinking

Structured output dùng chung với built-in tools (Google Search, URL Context, Code Execution, File Search) và function calling đang ở dạng **preview, chỉ cho dòng Gemini 3** [15]; ví dụ trong tài liệu dùng `gemini-4-argon` với `google_search` và `url_context`. Mẫu điển hình: tìm kiếm web rồi trả kết quả theo schema trong một request.

Thinking bật mặc định trên hầu hết model (`gemini-4-argon`, `gemini-3.8-flash`: medium). `max_output_tokens` tính cả token thinking, nên nếu thinking dùng hết ngân sách, interaction có `status: "incomplete"` và text **có thể rỗng** [20][21]. Với structured output, rỗng nghĩa là `json.loads("")` lỗi. Tăng `max_output_tokens` hoặc giảm `thinking_level`. Trong hội thoại stateless, gửi lại nguyên các `thought` block và signature (xem [tài liệu multi-turn](../multi-turn-conversation/#26-google-gemini)).

Streaming trả các event `step.delta` với `delta.text`; ghép đủ rồi mới parse [15].

### Model hỗ trợ và endpoint tương thích OpenAI

Theo trang model, structured output có trên `gemini-4-argon`, dòng 3.x (`gemini-3.8-flash`, `3.7-flash`, `3.5-flash-lite`, `3.1-pro-preview`...), 2.5 và 2.0. Các model Live, TTS và image không hỗ trợ [19].

Endpoint tương thích OpenAI (`https://generativelanguage.googleapis.com/v1beta/openai/`) nhận `client.beta.chat.completions.parse(response_format=PydanticModel)` và `zodResponseFormat` [22]. Cách này tiện để chạy cùng code trên nhiều provider, nhưng schema vẫn chịu subset của Gemini ở trên.

### Những lưu ý khi gửi request

- `application/json` không kèm schema chỉ là "gợi ý mạnh", vẫn có rủi ro JSON hỏng; luôn dùng cả hai [23].
- Tài liệu Vertex khuyên **không lặp lại schema trong prompt** khi đã truyền `responseSchema`, vì có thể làm giảm chất lượng [23]. Điều này ngược với lời khuyên ở nhiều engine self-hosted.
- Prompt chứa escape `\u` hoặc `\t` có thể khiến model sinh `\n` lặp lại; dùng ký tự UTF-8 trực tiếp trong ví dụ [20].
- Structured output trên model đã tuning có thể làm giảm chất lượng [23].

## 2.5 xAI Grok

Cả Chat Completions và Responses API của xAI dùng cú pháp tương thích OpenAI. Bạn thường có thể tái sử dụng code gọi API bằng cách đổi `base_url`, sau đó điều chỉnh schema theo tập con của xAI [25].

```json
// Chat Completions
"response_format": {
  "type": "json_schema",
  "json_schema": {"name": "collatz_result", "strict": true, "schema": {"...": "..."}}
}

// Responses API
"text": {"format": {"type": "json_schema", "name": "collatz_result", "strict": true, "schema": {"...": "..."}}}
```

`response_format` cũng nhận `"json_object"` và `"text"` (mặc định). SDK `xai_sdk` có `chat.parse(Model)` trả về cặp `(response, parsed)`; với `stream()`, các chunk ghép thành JSON và được parse sau khi stream kết thúc [25].

### Khác biệt so với OpenAI

| Chủ đề | OpenAI | xAI |
|---|---|---|
| `additionalProperties` | Phải ghi `false` ở mọi object | Mặc định `false`; ghi `true` nếu muốn cho thêm field |
| Field optional | Phải nằm trong `required`, dùng union với `null` | Bỏ khỏi `required` là được, hoặc dùng union với `null` |
| Đệ quy | Hỗ trợ | `$ref` / `$defs` chỉ khi **không vòng** |
| Kết hợp | Không có `allOf` | `anyOf`, `oneOf` (xử lý như `anyOf`), `allOf` chỉ với một subschema |
| Strict trên tools | `strict: true` tường minh | Arguments của tool luôn theo schema, như thể `strict` luôn bật |
| `format` được ép | Có `duration`, `hostname` | `date`, `time`, `date-time`, `email`, `uuid`, `ipv4`, `ipv6`, `uri`; format khác được nhận nhưng không ép |

xAI công bố rõ ngưỡng mà ràng buộc còn được **bảo đảm** [25]:

| Ràng buộc | Bảo đảm tới |
|---|---|
| `minLength` / `maxLength` | 2.048 |
| `minItems` / `maxItems` | 256 |
| `minProperties` / `maxProperties` | 64 |
| Giới hạn số | Không giới hạn |

Vượt các ngưỡng này, request vẫn được nhận nhưng việc tuân thủ phụ thuộc vào model. Tương tự, `not`, `if` / `then` / `else`, `allOf` nhiều subschema và format không có trong danh sách chỉ là **best-effort**. Những trường hợp bị từ chối hẳn (400): `enum` hoặc `anyOf` rỗng, property có schema là `true` / `false`, `minContains` / `maxContains`, và `items` dạng mảng (dùng `prefixItems`) [25].

`pattern` dùng một subset ECMAScript: không backreference, không `\p{}`, không `\b`, không lookaround. `^` và `$` được ngầm hiểu, tức là luôn so khớp toàn chuỗi, và `.` khớp cả xuống dòng [25].

### Model hỗ trợ

`grok-4.7` có structured outputs và function calling, context 500K, reasoning effort `low` / `medium` / `high` / `xhigh` [27]. Tài liệu ghi structured outputs dùng chung với tools được hỗ trợ trên "các model Grok 4 được hỗ trợ" mà không liệt kê cụ thể [25]. Với các ID khác như `grok-4.6`, `grok-4.5`, `grok-4.3`, `grok-4.20-*` [26], hãy thử trực tiếp trước khi dựa vào. Trên Responses API, `grok-4.7` luôn trả `reasoning.encrypted_content`; khi tự quản lý lịch sử, gửi lại nguyên phần này (xem [tài liệu multi-turn](../multi-turn-conversation/#25-xai-grok)).

`tool_choice` nhận `auto`, `required`, `none` hoặc tên một function; `parallel_tool_calls` mặc định bật [28].

## 2.6 Các provider Trung Quốc

Các provider dưới đây phần lớn có endpoint tương thích OpenAI, nhưng hỗ trợ `response_format` ở mức khác nhau. Một request có schema được chấp nhận chưa đủ để xác nhận bộ giải mã đang dùng schema đó. Bảng ghi lại khả năng được mô tả trong tài liệu chính thức; những chỗ tài liệu thiếu hoặc mâu thuẫn được nêu riêng để bạn kiểm thử trên endpoint đang dùng.

| Khả năng | DeepSeek | Qwen | Kimi | GLM | MiniMax | Doubao |
|---|---|---|---|---|---|---|
| JSON mode `json_object` | Có | Có | Có | Có | Không ghi | Có (Responses API) |
| Bắt buộc chữ "json" trong prompt | Có | Có | ? | ? | — | ? |
| `json_schema` ép schema | Không ghi | Có, một số model Qwen3.7 / 3.8 | Có, dialect MFJS | Không | Docs mâu thuẫn | Có, beta, model từ bản 250615 |
| Strict function calling | Có, beta, qua `/beta` | Không ghi | Không ghi | Không ghi | Không | Có, `strict` trên tool |
| `tool_choice` | ? | `auto`, `none`, tên function; **không** có `required` | `auto`, `required` | ? | Chỉ `auto` | ? |
| Tắt được thinking | Có | Có | K3: không | ? | ? | Có |

"?" là chưa tìm thấy trong tài liệu chính thức. Với những ô đó, thử trên endpoint thật trước khi dựa vào.

### DeepSeek

**JSON mode** dùng `response_format: {"type": "json_object"}`, kèm hai điều kiện: prompt phải có chữ "json" và nên có ví dụ JSON mong muốn; `max_tokens` phải đủ lớn để không bị cắt [29].

Một hành vi cần xử lý riêng: khi output không phải JSON hợp lệ, API thay phần không hợp lệ bằng khoảng trắng. Kết quả là `content` có thể chỉ toàn dấu cách trong khi `finish_reason` vẫn là `stop` [29]:

```python
resp = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {"role": "system", "content": 'Trả về json dạng {"name": str, "age": int}'},
        {"role": "user", "content": "Tom 30 tuổi."},
    ],
    response_format={"type": "json_object"},
    max_tokens=512,
)
content = resp.choices[0].message.content
if not content.strip():
    ...  # output không hợp lệ đã bị xoá, retry
```

DeepSeek không ghi hỗ trợ `json_schema`. Muốn có ràng buộc schema thật, dùng **strict function calling** (beta) [31]:

- Đổi `base_url` thành `https://api.deepseek.com/beta`.
- Đặt `strict: true` trên **mọi** function trong `tools`. Server kiểm tra schema và trả lỗi nếu dùng kiểu không hỗ trợ.
- Subset: `object` (mọi field phải `required`, bắt buộc `additionalProperties: false`), `string` có `pattern` và `format` (`email`, `hostname`, `ipv4`, `ipv6`, `uuid`) nhưng không có `minLength` / `maxLength`, số có `minimum`, `maximum`, `exclusive*`, `multipleOf`, mảng **không** có `minItems` / `maxItems`, có `anyOf`, `$ref` / `$defs` kể cả đệ quy.
- Hoạt động ở cả thinking và non-thinking mode.

Lỗi dễ gặp: wrapper không chuyển sang `/beta`, request đi vào `/v1`, và `strict` bị bỏ qua mà không báo lỗi. Một issue của LangChain mô tả đúng trường hợp này [42]. Hãy kiểm tra URL thực tế mà SDK gọi.

Trong thinking mode, request có `tools` phải gửi lại `reasoning_content` của mọi lượt trước, thiếu là 400 [30]. Tài liệu thinking mode không nhắc đến `response_format`, nên chưa rõ JSON mode có hoạt động khi thinking bật hay không.

### Alibaba Qwen (Model Studio)

Qwen có cả hai chế độ [32]:

- **JSON Object** (`json_object`): có trên hầu hết model text và multimodal, kể cả một số model DeepSeek, Kimi, GLM do Model Studio host. Thiếu chữ "json" trong system hoặc user message sẽ bị lỗi.
- **JSON Schema** (`json_schema` với `strict: true`): chỉ trên dòng Qwen3.7-Plus / Flash / Max và Qwen3.8-Max / Flash; vùng Singapore chưa hỗ trợ. Bảng so sánh trên cùng trang lại ghi "một số model qwen-plus", nên kiểm tra trong console.

Giới hạn của JSON Schema mode:

- `multipleOf`, `uniqueItems`, `contains`, `patternProperties`, `propertyNames` trả 400. `$ref` chỉ dùng được với tham chiếu nội bộ.
- **Input có ảnh, video hoặc audio thì schema không được áp dụng**: request lặng lẽ chuyển về JSON Object mode. Pipeline trích xuất từ ảnh hoá đơn trên Qwen vì vậy phải tự validate.
- Tài liệu khuyên không đặt `max_tokens` vì giới hạn có thể cắt dở JSON.
- Model chỉ có non-thinking chỉ hỗ trợ structured output ở non-thinking mode. Tắt thinking bằng `extra_body={"enable_thinking": False}` [33].

Function calling không có `strict` trong tài liệu và `tool_choice` không nhận `"required"` [33]. Muốn ép gọi tool, truyền tên function cụ thể.

### Moonshot Kimi

Kimi hỗ trợ `json_schema` với `strict` theo **MFJS** (Moonshot Flavored JSON Schema), một biến thể riêng. Tài liệu giới thiệu công cụ `walle` để kiểm tra tĩnh thay vì liệt kê đầy đủ keyword bị cấm [34]. Khi kiểm thử cần phân biệt lỗi schema và mức tuân thủ của model:

- Schema sai cú pháp trả 400, nhưng **200 chưa chứng minh schema tương thích**. Tài liệu khuyên kiểm tra output trên model thật.
- Độ tin cậy khác nhau theo model: `kimi-k3` và `kimi-k3.1-pro` ổn với object lồng, mảng, `anyOf`; `kimi-k2.7-code` ổn nhất, kể cả `oneOf`, `$ref`; `kimi-k2.6` đôi khi không ổn với schema phức tạp, có thể bọc output trong code fence markdown làm `json.loads` lỗi, và có thể sinh field thừa dù có `additionalProperties: false` [34].

`kimi-k3` **luôn bật thinking**, chỉ điều chỉnh được `reasoning_effort` (`low`, `high`, `max`, mặc định `max`). Reasoning nằm ở `reasoning_content`; chỉ parse `message.content` [35]. Với structured output, thinking luôn bật có nghĩa là độ trễ cao hơn cho các tác vụ trích xuất đơn giản.

Strict chỉ áp cho `response_format`, không có trên tool definition. `tool_choice: "required"` có trên K3; quickstart khuyên dùng nó ở lượt đầu để ép gọi tool rồi chuyển về `auto` [35].

### Zhipu GLM (Z.ai)

Tài liệu chính thức chỉ ghi `json_object`. Schema được đưa vào system message làm hướng dẫn và validate phía client bằng thư viện `jsonschema` [36]. Function calling dùng định dạng OpenAI, không có `strict`, và các tài liệu bên thứ ba mâu thuẫn nhau về `tool_choice` [37]. Với GLM, coi structured output là mức 2 (JSON mode) cộng validate và retry phía client.

### MiniMax

Trang API reference liệt kê `response_format` loại `json_schema` (`name` khớp `^\w+$`, tối đa 64 ký tự; root là `object`). Tuy nhiên, mô tả tham số chỉ ghi hỗ trợ `MiniMax-Text-01`, còn danh sách model trên cùng trang chỉ có `MiniMax-M3` và `MiniMax-M3.1-Flash` [38]. Bảng tính năng MiniMax-M2.5 trên Alibaba Model Studio lại ghi không hỗ trợ structured output [44].

Với tài liệu chưa thống nhất như vậy, cần thử đúng model trên endpoint sẽ dùng và chuẩn bị fallback bằng prompt kèm validation. `tool_choice` chỉ có `auto`.

### ByteDance Doubao (Volcengine Ark)

Doubao có structured output **beta** trên Responses API [40]:

```json
"text": {
  "format": {"type": "json_schema", "name": "ticket", "strict": true, "schema": {"...": "..."}}
}
```

- Áp dụng cho LLM từ phiên bản 250615 trở đi, trừ khi danh sách model ghi khác; danh sách model có cột "结构化输出能力（beta）" [40][41].
- Tài liệu ghi tính năng đang beta và có thể dao động theo tải. Không dùng được với gói TPM bảo đảm [40].
- Hậu tố phiên bản quyết định hỗ trợ. `doubao-seed-2-0-code-preview-260215` từ chối cả `json_object` lẫn `json_schema` trên Ark [43].

Function calling có `strict` chính thức, và tài liệu khuyên luôn bật; mọi field trong `properties` phải nằm trong `required`, field optional dùng `["string", "null"]` [39]. Tài liệu cũng ghi tắt thinking giúp function calling hiệu quả hơn, và các ví dụ `json_schema` đều đặt `thinking: {"type": "disabled"}` [39][40].

### Model open-weight tự host

DeepSeek, Qwen, Kimi, GLM và MiniMax đều phát hành open weights. Khi tự host bằng vLLM hay SGLang, structured output do engine đảm nhận qua guided decoding, không liên quan đến các giới hạn API ở trên. Thường đây là cách có ràng buộc schema chặt nhất cho các model này. Xem [§2.7](#27-self-hosted-vllm-sglang-llamacpp-ollama).

## 2.7 Self-hosted: vLLM, SGLang, llama.cpp, Ollama

Khi tự host, bạn có thể chọn backend ràng buộc và dùng regex, grammar hoặc danh sách lựa chọn. Khả năng hỗ trợ schema và chi phí giải mã phụ thuộc vào backend đó, nên cần ghi cả engine lẫn backend khi cấu hình và đo hiệu năng.

| Engine | Backend mặc định | Backend khác | Tham số chính |
|---|---|---|---|
| vLLM | `auto` | `xgrammar`, `guidance`, `lm-format-enforcer` (deprecated) | `structured_outputs` hoặc `response_format` |
| SGLang | `xgrammar` | `outlines`, `llguidance` | `response_format`, `extra_body.regex`, `extra_body.ebnf` |
| llama.cpp | GBNF tích hợp | llguidance (build flag) | `json_schema`, `grammar`, `response_format` |
| Ollama | Không công bố | | `format` |
| TensorRT-LLM | XGrammar | | `GuidedDecodingParams` |

### vLLM

Tham số đã đổi tên: các field `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`, `guided_decoding_backend` **bị gỡ từ v0.12.0**, thay bằng một object `structured_outputs` [52]. Code và tutorial cũ dùng `guided_*` sẽ không chạy trên bản mới.

```python
completion = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Phân loại cảm xúc: 'Giao hàng chậm quá'"}],
    extra_body={"structured_outputs": {"choice": ["positive", "neutral", "negative"]}},
)

# Hoặc theo chuẩn OpenAI
completion = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[...],
    response_format={"type": "json_schema", "json_schema": {"name": "complaint", "schema": schema}},
)
```

Các loại ràng buộc: `json`, `regex`, `choice`, `grammar` (EBNF), `structural_tag`, cùng `whitespace_pattern` để giới hạn khoảng trắng giữa các token JSON. Backend chọn bằng `--structured-outputs-config.backend` [52].

Khi cấu hình vLLM, kiểm tra thêm backend và reasoning parser [52]:

- **Đổi backend có thể đổi output.** Mỗi backend dùng dialect regex khác nhau (lm-format-enforcer theo Python, xgrammar và guidance theo Rust), nên tập token được phép khác nhau.
- **Model reasoning.** Với model có `<think>`, bật `--structured-outputs-config.enable_in_reasoning=True` (từ v0.11.2) để ràng buộc chỉ áp vào phần sau reasoning. Nếu không cấu hình reasoning parser đúng, ràng buộc có thể áp vào cả phần suy luận hoặc bị tắt.

### SGLang

Mỗi request chọn một trong `json_schema`, `regex`, `ebnf` hoặc `structural_tag`. Qua API tương thích OpenAI, JSON Schema đi trong `response_format`, regex và EBNF trong `extra_body`. `structural_tag` cho phép ràng buộc chỉ kích hoạt khi model sinh một tag nhất định, ví dụ JSON bên trong `<function=...>`, phần còn lại tự do [53]. Đây là cơ chế hữu ích cho agent: model nói tự nhiên, nhưng mỗi khi gọi tool thì arguments bị ràng buộc.

### llama.cpp

llama.cpp dùng **GBNF**, một dạng BNF riêng, và có công cụ chuyển JSON Schema sang GBNF (`--json-schema` / `-j` ở CLI, field `json_schema` ở server) [54]:

```text
root   ::= "{" ws "\"sentiment\":" ws sentiment ws "}"
sentiment ::= "\"positive\"" | "\"neutral\"" | "\"negative\""
ws     ::= [ \t\n]{0,2}
```

llama.cpp hỗ trợ một tập con JSON Schema: `minimum` / `maximum` chỉ áp dụng cho integer, `$ref` lồng nhau chưa chạy đúng, `pattern` phải bắt đầu bằng `^` và kết thúc bằng `$`; không có `uniqueItems`, `not` hay điều kiện. Keyword không hỗ trợ bị bỏ qua mà không báo lỗi, nên vẫn cần validate phía client [54]. Schema cũng không tự được đưa vào prompt: bạn phải mô tả cấu trúc mong muốn. Để tránh lặp không giới hạn, tài liệu khuyên dùng `x{0,N}` thay cho `x*`.

### Ollama

Từ 12/2024, tham số `format` nhận một JSON Schema [55]:

```python
from ollama import chat

response = chat(
    model="qwen3",
    messages=[{"role": "user", "content": "Kể về Việt Nam."}],
    format=Country.model_json_schema(),
    options={"temperature": 0},
)
country = Country.model_validate_json(response.message.content)
```

Blog của Ollama khuyên vẫn thêm chỉ dẫn "trả về JSON" vào prompt và đặt temperature 0. Vision model cũng nhận `format`.

### TGI và TensorRT-LLM

Text Generation Inference compile grammar thành FSM và áp như một logits processor, qua `grammar` trên `/generate` hoặc qua tools trên `/chat/completions`. Tài liệu khuyên đưa schema vào prompt và dùng `repetition_penalty` khi gặp token lặp [56]. TensorRT-LLM có guided decoding qua `GuidedDecodingParams` với các loại `kJSON`, `kJSON_SCHEMA`, `kREGEX`, `kEBNF_GRAMMAR`, chạy trên backend XGrammar [57].

## 2.8 Thư viện

Khi đã chọn API, bạn có thể dùng thư viện để chuyển Pydantic / Zod thành schema phù hợp, validate và retry, hoặc stream object từng phần. Cần biết thư viện đang yêu cầu provider ràng buộc lúc giải mã hay đang parse và sửa output phía client: hai cách cho mức đảm bảo khác nhau, dù cùng trả về một object có kiểu.

| Thư viện | Ngôn ngữ | Cơ chế chính | Điểm mạnh |
|---|---|---|---|
| SDK chính hãng (`parse()`) | Python, TS | Structured output native | Không thêm dependency, bám sát API |
| Instructor | Python (có bản TS, PHP, Elixir...) | Native / tool calling + retry với lỗi validation | Validator Pydantic tuỳ ý, reask tự động |
| Pydantic AI | Python | Tool output (mặc định), native, hoặc prompted | Agent framework có output có kiểu |
| LangChain | Python, TS | `ProviderStrategy` / `ToolStrategy` | Tự chọn chiến lược theo model |
| Vercel AI SDK | TS | `Output.object()` trong `generateText` / `streamText` | Streaming partial object vào UI |
| BAML | DSL, sinh client nhiều ngôn ngữ | Schema-aligned parsing (sửa output phía client) | Chạy được cả với model không có structured output |
| Outlines, Guidance | Python | Constrained decoding trên model local | Regex, CFG, kiểm soát ở mức token |

### Instructor

Instructor gói client của provider và thêm `response_model` cùng `max_retries`. Khi Pydantic validation thất bại, kể cả validator tuỳ chỉnh không thể diễn đạt bằng JSON Schema, lỗi được gửi lại cho model kèm yêu cầu sửa [59][60]:

```python
import instructor
from pydantic import BaseModel, Field, field_validator

class Product(BaseModel):
    name: str
    price_vnd: int = Field(gt=0)

    @field_validator("name")
    @classmethod
    def not_generic(cls, v: str) -> str:
        if v.lower() in {"sản phẩm", "hàng"}:
            raise ValueError("Tên sản phẩm phải cụ thể, không được chung chung")
        return v

client = instructor.from_provider("openai/gpt-6-luna", max_retries=2)
product = client.create(
    response_model=Product,
    messages=[{"role": "user", "content": "..."}],
)
```

Retry do validation khác với retry do lỗi mạng. Hết lượt retry, Instructor raise `InstructorRetryException` [59].

### Pydantic AI

Pydantic AI có nhiều chế độ output [58]:

- `ToolOutput` (mặc định): schema được gửi như tham số của một "output tool". Nhiều kiểu output thành nhiều output tool.
- `NativeOutput`: dùng structured output native của provider.
- `PromptedOutput`: chèn schema vào instructions rồi parse text. Model không bị ép.
- Các chế độ khác: `TextOutput`, `StructuredDict()`, `Choices()`.

Với OpenAI, schema vượt giới hạn strict mode sẽ được gửi ở chế độ không strict [58]. Nếu cần bảo đảm ở mức giải mã, hãy kiểm tra schema và request thực tế; việc nhận được object đã parse chưa chứng minh strict mode đang bật.

### LangChain

Trong LangChain 1.x, agent tạo bằng `create_agent` nhận `response_format` [61]:

- `ProviderStrategy(schema, strict=...)`: structured output native. `strict` cần `langchain>=1.2`.
- `ToolStrategy(schema, handle_errors=...)`: qua tool calling, cho model không có native; hỗ trợ cả union.
- Truyền thẳng một kiểu (Pydantic, dataclass, TypedDict): LangChain tự chọn `ProviderStrategy` nếu model hỗ trợ (OpenAI, Claude, Grok), ngược lại dùng `ToolStrategy`. JSON Schema dạng dict phải bọc trong một strategy tường minh.

Kết quả nằm ở `result["structured_response"]`. Với chat model đơn lẻ, `model.with_structured_output(Schema)` vẫn là cách quen thuộc.

### Vercel AI SDK

Từ AI SDK 6, `generateObject` và `streamObject` bị deprecated; structured output chuyển vào tham số `output` của `generateText` và `streamText` [62][63]:

```typescript
import { generateText, streamText, Output } from "ai";
import { z } from "zod";

const { output } = await generateText({
  model,
  output: Output.object({ schema: z.object({ name: z.string(), tags: z.array(z.string()) }) }),
  prompt: "...",
});

const { partialOutputStream } = streamText({
  model,
  output: Output.object({ schema }),
  prompt: "...",
});
for await (const partial of partialOutputStream) render(partial);
```

Các loại: `Output.object()`, `Output.array()` (có `elementStream` trả từng phần tử hoàn chỉnh), `Output.choice()`, `Output.json()`, `Output.text()`. Có một issue báo rằng `Output.object()` không yêu cầu JSON mode ở provider như `generateObject` cũ mà parse text phía client [64]. Nếu bạn cần ràng buộc ở mức giải mã, kiểm tra request thực tế gửi đi với provider của mình.

### Zod v4 và Pydantic: nguồn của schema

Zod 4 có sẵn `z.toJSONSchema(schema)`, mặc định theo Draft 2020-12, có tuỳ chọn Draft 7, Draft 4, OpenAPI 3.0. `z.date()`, `z.bigint()` và `z.transform()` không có tương đương trong JSON Schema và mặc định sẽ throw [65]. Pydantic có `Model.model_json_schema()`, sinh `$defs` / `$ref` cho model lồng và các keyword như `minLength` từ `Field(...)`. Schema sinh tự động thường dùng keyword mà provider không nhận hoặc bỏ qua, nên SDK chính hãng mới có các bước transform như `transform_schema()` của Anthropic.

### BAML: sửa output thay vì ràng buộc

BAML cho phép khai báo hàm LLM có kiểu bằng một DSL và sinh client cho Python, TypeScript, Go, Ruby... **Schema-Aligned Parsing** xử lý output sau khi sinh: bỏ code fence, sửa dấu phẩy thừa, chuẩn hoá hoa thường của enum và bỏ qua đoạn suy luận trước object [66]. Vì sửa ở phía client, BAML dùng được với model không có structured output và không ràng buộc phần suy luận. Cách này không bảo đảm ở mức token; nên log những lần parser phải sửa để nhận ra khi chất lượng output giảm.

---

# Phần 3 — Thiết kế và vận hành

Trở lại pipeline đọc email: giờ bạn đã chọn được API và có thể nhận JSON đúng schema. Trước khi đưa dữ liệu lên dashboard, còn phải quyết định email thiếu mã đơn được biểu diễn ra sao, khi nào cần thử lại, và làm sao phát hiện model đang phân loại sai. Những quyết định đó bắt đầu từ schema.

## 3.1 Thiết kế schema

Khi đặt tên một cột database, bạn chủ yếu nghĩ đến code sẽ đọc và ghi nó. Với schema gửi cho LLM, tên field còn giúp model hiểu phải tìm thông tin gì trong input. `description` và các giá trị enum cũng tham gia vào quá trình đó, nên cần viết đủ rõ để người đọc và model hiểu cùng một nghĩa.

### Tên field và description

`price` chưa cho biết đơn vị tiền; `refund` chưa cho biết đó là số tiền hay yêu cầu hoàn tiền. Các tên như `price_vnd`, `duration_minutes`, `is_refund_requested` giải quyết được phần mơ hồ này ngay tại field. Với boolean, tiền tố `is_` hoặc `has_` giúp diễn đạt thành một câu hỏi có đáp án đúng/sai.

Dùng `description` cho những điều tên không thể nói hết: thang đo, format, trường hợp biên và ví dụ. Mô tả `"1 = bình thường, 5 = doạ huỷ đơn / bóc phốt"` giúp người đọc dashboard hiểu cách chấm `anger`. Với `order_id`, câu `"null nếu email không nhắc tới"` cho model cách biểu diễn dữ liệu thiếu thay vì phải đoán.

Tên field cũng có chi phí. Gemini tính số trạng thái theo độ dài tên; OpenAI giới hạn tổng ký tự tên và enum ở 120.000. Khi schema có vài trăm field, hãy giữ tên vừa đủ rõ và chuyển phần giải thích dài sang `description`.

### Biểu diễn dữ liệu thiếu và trường hợp chưa biết

Email có thể không nhắc mã đơn, nói về một loại khiếu nại chưa có trong enum, hoặc diễn đạt quá mơ hồ để phân loại chắc chắn. Schema cần biểu diễn được cả ba trường hợp:

```python
class Extraction(BaseModel):
    order_id: str | None = Field(description="Mã đơn dạng DH-XXXXXX. null nếu không có")
    category: Literal["delivery", "defect", "billing", "warranty", "other"]
    category_other: str | None = Field(description="Mô tả ngắn nếu category là other, ngược lại null")
    confidence: Literal["high", "medium", "low"] = Field(
        description="low nếu email mơ hồ hoặc phải đoán"
    )
```

Ở đây, `null` cho biết input không có mã đơn; `other` giữ lại những loại khiếu nại chưa được đặt tên. Sau một thời gian, bạn có thể nhóm các giá trị `category_other` để xem có cần bổ sung nhãn, chẳng hạn `warranty`, hay không.

`confidence` phục vụ một việc khác: giúp định tuyến các trường hợp mơ hồ sang người duyệt. Một giá trị như `0.73` dễ bị hiểu là xác suất đã được hiệu chỉnh, trong khi model không bảo đảm điều đó. Các mức `high`, `medium`, `low` kèm định nghĩa rõ thường dễ dùng hơn; chúng vẫn cần được đối chiếu với dữ liệu đã gán nhãn.

### Thứ tự field: suy luận trước, kết luận sau

Với OpenAI và vLLM/SGLang, thứ tự khai báo trong schema được dùng để sinh output. Một schema chấm điểm có thể yêu cầu trích bằng chứng, đánh giá ngắn rồi mới cho điểm:

```python
class Judgement(BaseModel):
    evidence: list[str] = Field(description="Tối đa 3 câu trích từ câu trả lời, liên quan đến tiêu chí")
    analysis: str = Field(description="1–3 câu đánh giá dựa trên evidence")
    score: Literal[1, 2, 3, 4, 5]
```

Để cách bố trí này có tác dụng, cần kiểm tra thứ tự key trên provider thực tế: Claude sinh field bắt buộc trước optional; Gemini có thể sắp theo alphabet nếu không có `propertyOrdering` ([§2.1](#21-bức-tranh-chung)). Các field đứng trước điểm số nên là bắt buộc. Nếu model đã có thinking và ứng dụng không cần phần đánh giá bằng văn bản, có thể bỏ `analysis`; giữ `evidence` để đối chiếu khi kiểm tra điểm.

### Hình dạng của schema

Nếu một email nói về hai sản phẩm, `items: list[Item]` biểu diễn được cả hai mà không cần dự đoán trước số field `item_1`, `item_2`, `item_3`. Cấu trúc cũng nên ít cấp lồng khi có thể: mỗi cấp làm tăng trạng thái của bộ giải mã và khiến output khó kiểm tra hơn. OpenAI cho phép tối đa 10 cấp; 3–4 cấp thường dễ quản lý hơn trong ứng dụng.

Khi kết quả có nhiều dạng, dùng union với một field `Literal` để nhận biết từng nhánh. Ví dụ, hoàn tiền cần số tiền, còn đổi hàng cần SKU mới:

```python
class Refund(BaseModel):
    kind: Literal["refund"]
    amount_vnd: int

class Exchange(BaseModel):
    kind: Literal["exchange"]
    new_sku: str

class Resolution(BaseModel):
    action: Refund | Exchange   # bọc trong object: OpenAI không cho root là anyOf
```

Các giá trị enum nên ngắn và phân biệt được bằng nghĩa, tránh chỉ khác hoa thường hay dấu câu. Với danh mục hàng trăm giá trị, có thể chia thành hai bước chọn nhóm rồi chọn mục, hoặc dùng retrieval để lấy các ứng viên liên quan trước khi dựng schema.

Với ID nội bộ, thường nên trích tên rồi tra cứu bằng code. Email có thể viết `"Máy lọc nước A3"` nhưng không chứa SKU `LN-A3-2026-WH`; yêu cầu model tự điền SKU sẽ biến một tác vụ trích xuất thành một lần đoán dữ liệu danh mục.

### Ổn định schema

Code phía sau phụ thuộc vào tên field, kiểu và ý nghĩa giá trị. Ghi `schema_version` trong log hoặc tên format để biết một response được tạo theo phiên bản nào. Khi thêm field optional / nullable, vẫn cần kiểm tra code đang đọc output có chấp nhận thay đổi đó không.

Giữ schema cố định khi cấu trúc dữ liệu không đổi. Mỗi schema mới có thể cần compile grammar ([§1.3](#13-constrained-decoding-hoạt-động-thế-nào)); trên Claude, đổi schema còn làm mất prompt cache ([§2.3](#23-claude-anthropic)). Dùng một Pydantic hoặc Zod model làm nguồn để sinh schema và validate output sẽ tránh việc hai bản định nghĩa lệch nhau.

## 3.2 Schema chạy được trên nhiều provider

Giả sử pipeline chuyển sang provider dự phòng khi API chính lỗi. Nếu provider mới bỏ qua `order_id` khi thiếu dữ liệu, trong khi code quen nhận `order_id: null`, lần đọc `data["order_id"]` có thể gây `KeyError`. Để giảm những khác biệt như vậy, bắt đầu với tập quy tắc chung dưới đây rồi kiểm tra subset của từng endpoint:

| Quy tắc | Lý do |
|---|---|
| Root là `object` | OpenAI không cho root `anyOf`; MiniMax bắt root là object |
| Mọi field nằm trong `required`, optional là `["T", "null"]` | Bắt buộc với OpenAI strict, DeepSeek strict, Doubao; hợp lệ ở mọi nơi khác |
| `additionalProperties: false` ở mọi object | Bắt buộc với OpenAI, Claude, DeepSeek; vô hại ở nơi khác |
| Chỉ dùng `type`, `properties`, `required`, `items`, `enum`, `anyOf`, `$defs` / `$ref`, `description` | Giao của các subset |
| Không đệ quy | Claude, Grok không hỗ trợ; Gemini có điều kiện |
| Không `minLength` / `maxLength` / `minimum` / `maximum` / `pattern` trong schema gửi đi | Claude trả 400; Gemini bỏ qua `minLength`; llama.cpp chỉ hỗ trợ một phần. Giữ chúng trong Pydantic / Zod để validate phía client |
| `enum` chỉ chứa chuỗi | Gemini `responseSchema` và nhiều engine chỉ nhận enum chuỗi |
| Không `allOf`, `not`, `if` / `then` / `else`, `oneOf` | Phần lớn không hỗ trợ hoặc chỉ best-effort |
| `format` chỉ dùng `date-time`, `date`, `time`, `email`, `uuid` | Giao của các danh sách format |

Giữ đầy đủ ràng buộc trong một Pydantic / Zod model, rồi biến đổi schema gửi đi theo từng provider: bỏ keyword không hỗ trợ, chuyển mô tả ràng buộc vào `description`, thêm `additionalProperties: false`, chuyển optional thành nullable. SDK của Anthropic (`transform_schema()`) và OpenAI đã làm một phần việc này.

Sau khi nhận response, validate bằng model gốc. Nhờ vậy, đổi provider chỉ thay cách gửi schema; quy tắc chấp nhận dữ liệu của ứng dụng vẫn nằm ở một nơi. Chẳng hạn, một giới hạn độ dài bị bỏ khỏi request vẫn được kiểm tra trước khi ghi database.

## 3.3 Validation nhiều lớp và retry

### Bốn lớp kiểm tra

```text
response từ API
  │
  ├─ 1. Trạng thái: stop_reason / finish_reason / status
  │     refusal? max_tokens? content_filter? content rỗng?
  │
  ├─ 2. Parse + schema: json.loads + Model.model_validate
  │     (bắt buộc kể cả khi đã dùng strict: ràng buộc bị bỏ qua lặng lẽ, provider fallback)
  │
  ├─ 3. Business rules: validator trong code
  │     mã đơn tồn tại? tổng tiền = tổng các dòng? ngày không ở tương lai?
  │
  └─ 4. Grounding: giá trị có thật trong input không?
        tên, số tiền, mã có xuất hiện trong văn bản gốc không?
```

HTTP 200 cho biết request đã được xử lý, nhưng chưa cho biết có dữ liệu để parse hay không. Output bị cắt vì `max_tokens` và refusal của Claude đều có thể trả HTTP 200. DeepSeek JSON mode còn có thể trả chuỗi toàn khoảng trắng với `finish_reason: "stop"`. Vì vậy, bước đầu tiên là kiểm tra trạng thái và content; ví dụ với Claude:

```python
def check_response(resp) -> str:
    # Claude
    if resp.stop_reason == "refusal":
        raise Refused(resp.stop_details)
    if resp.stop_reason == "max_tokens":
        raise Truncated()
    text = resp.content[0].text
    if not text.strip():
        raise EmptyOutput()
    return text
```

Khi response đã hoàn tất và có content, parse rồi validate bằng schema gốc. Bước này kiểm tra cả những ràng buộc có thể đã bị bỏ qua: `minLength` trên Gemini, keyword không hỗ trợ trên llama.cpp, schema với input ảnh trên Qwen hoặc `strict` khi DeepSeek không được gọi qua `/beta`. Nó cũng bắt lỗi khi SDK chuyển sang non-strict, như Pydantic AI với schema quá lớn.

Tiếp đó mới đến nội dung: mã đơn có tồn tại trong hệ thống không, tổng tiền có bằng tổng các dòng không, dữ liệu trích xuất có xuất hiện trong input không. Với tên và mã cần trích nguyên văn, có thể so chuỗi sau khi chuẩn hoá khoảng trắng và hoa thường. Với số tiền, đối chiếu số trong văn bản. Nhiều kiểm tra như vậy làm được bằng code mà không cần thêm một lượt gọi LLM.

### Retry thế nào

| Lỗi | Retry? | Cách |
|---|---|---|
| Truncated | Có | Tăng `max_tokens`; với Gemini, giảm `thinking_level`. Nếu vẫn bị cắt, schema hoặc input quá lớn: chia nhỏ |
| Content rỗng, JSON hỏng (JSON mode) | Có | Gửi lại nguyên request; nếu lặp lại, chuyển sang strict hoặc tool calling |
| Schema / business rule sai | Có, có giới hạn | Gửi lại kèm thông báo lỗi cụ thể (kiểu Instructor reask): "`total_vnd` là 1.200.000 nhưng tổng các dòng là 1.150.000" |
| Refusal | Thường không | Ghi log, chuyển người duyệt. Retry nguyên văn hiếm khi thay đổi kết quả |
| Grounding sai | Có thể | Reask với chỉ dẫn trích nguyên văn; hoặc đánh dấu để người duyệt |
| Schema bị từ chối (400) | Không | Lỗi lập trình. Sửa schema |

Đặt giới hạn retry, thường là 2–3 lần, và ghi lại lỗi của từng lần. Khi tỉ lệ retry tăng, các bản ghi này giúp phân biệt ảnh hưởng của prompt, phiên bản model và dữ liệu đầu vào mới.

Reask thêm output lỗi và phản hồi sửa lỗi vào hội thoại, nên request sau dài hơn request đầu. Nếu object lớn mà chỉ sai một phần, cân nhắc yêu cầu sửa phần đó để giảm token và tránh làm thay đổi những giá trị đã đúng.

## 3.4 Streaming partial JSON

Một bảng 50 dòng trích từ PDF có thể mất đủ lâu để người dùng muốn xem kết quả trước khi request kết thúc. Streaming giúp hiển thị dần, nhưng dữ liệu đang nhận chưa chắc đã dùng được để tính toán hay ghi database. Có ba cách xử lý tuỳ đơn vị bạn muốn hiển thị: object tạm thời, từng phần tử, hoặc từng field.

**Parse từng phần.** Thư viện như Vercel AI SDK (`partialOutputStream`), BAML hay các parser "partial JSON" đóng tạm các ngoặc còn mở để có một object hợp lệ ở mỗi chunk:

```text
chunk 1:  {"title": "Báo cáo Q3", "rows": [{"name": "Hà N
partial:  {"title": "Báo cáo Q3", "rows": [{"name": "Hà N"}]}
chunk 2:  ội", "revenue": 120
partial:  {"title": "Báo cáo Q3", "rows": [{"name": "Hà Nội", "revenue": 120}]}
```

Trong chunk thứ hai, `120` có thể mới là phần đầu của `1200`. Field bắt buộc cũng có thể chưa đến, và chuỗi có thể dừng giữa một từ. Partial object vì vậy chỉ là bản tạm để hiển thị; chờ dữ liệu hoàn chỉnh và validate trước khi dùng cho quyết định của ứng dụng.

**Stream theo phần tử.** Với output là danh sách, chờ mỗi phần tử hoàn chỉnh rồi mới phát. Vercel `Output.array()` có `elementStream` làm đúng việc này [62]. Mỗi phần tử có thể validate riêng ngay khi đến.

**Đánh dấu field nào được stream.** BAML có annotation như `@stream.done` (chỉ phát khi field hoàn chỉnh) và `@stream.not_null` (không phát object cho đến khi field này có giá trị) [66]. Với UI, ví dụ một thẻ sản phẩm chỉ nên hiện khi đã có `id`, còn `description` có thể hiện dần.

Ở mức API, mọi provider đều stream structured output như text thường, và đều khuyên gom đủ rồi mới parse ([§2.3](#23-claude-anthropic), [§2.4](#24-google-gemini)). Tool arguments stream dưới dạng delta chuỗi (`response.function_call_arguments.delta` ở OpenAI, `arguments_delta` ở Gemini Interactions), cùng nguyên tắc.

## 3.5 Structured output với reasoning model

Khi provider áp grammar riêng cho câu trả lời cuối, reasoning model có thể dùng thinking trước khi sinh JSON ([§1.5](#15-ép-format-có-làm-model-kém-đi-không)). Bạn không cần thêm field suy luận chỉ để tạo bước này, nhưng vẫn cần tính ngân sách token và kiểm tra cách API hoặc engine phân tách thinking với output:

| Vấn đề | Provider | Xử lý |
|---|---|---|
| Thinking dùng hết `max_output_tokens`, output rỗng | Gemini (`status: incomplete`), Claude, OpenAI | Đặt ngân sách đủ cho cả thinking và JSON; giảm effort cho tác vụ trích xuất đơn giản |
| Không tắt được thinking | Kimi K3, GLM-5.3 | Chấp nhận độ trễ, hoặc dùng model khác cho extraction số lượng lớn |
| Ví dụ của provider tắt thinking khi dùng schema | Doubao, Qwen (model non-thinking) | Kiểm tra kỹ nếu bật cả hai; có model chỉ hỗ trợ structured output ở non-thinking |
| Engine self-hosted ràng buộc cả phần `<think>` | vLLM, SGLang | Cấu hình reasoning parser và `enable_in_reasoning` |
| Reasoning phải gửi lại ở lượt sau | DeepSeek (có tools), Kimi, Gemini, Grok | Xem [tài liệu multi-turn](../multi-turn-conversation/#37-reasoning-và-thinking-blocks) |
| Field "lập luận" trong schema bị từ chối | Claude (`reasoning_extraction`) | Dùng thinking thay vì field lập luận dài |

Chỉ thêm phần giải thích vào schema nếu ứng dụng thực sự dùng nó. Một field `reasoning` dài tốn thêm output token và trên Claude có thể kích hoạt refusal. Nếu mục đích là kiểm tra kết quả, field `evidence` ngắn thường cung cấp thông tin cần đối chiếu trực tiếp hơn.

Với tác vụ cần suy luận nặng và output phức tạp, phương án tách hai bước ở [§1.5](#điều-rút-ra-cho-thực-tế) vẫn có giá trị: reasoning model mạnh trả lời tự do, model rẻ chuyển sang schema.

## 3.6 Structured output trong agent và MCP

Trong một agent, structured output xuất hiện ở ba chỗ:

```text
user ──► agent loop ──► tool call (arguments: JSON theo input schema)   ◄── strict tool
                  ◄── tool result (structuredContent theo outputSchema) ◄── MCP outputSchema
              ...
          ──► final answer (JSON theo response schema)                  ◄── response format
```

**Tool arguments.** Với tool thay đổi dữ liệu như `transfer_money`, arguments sai kiểu có thể dẫn đến thao tác sai, nên bật strict khi provider hỗ trợ. Cần tính cả giới hạn số tool và cách chọn tool: Claude nhận tối đa 20 strict tools mỗi request, OpenAI nhận 350 tools; Claude Opus 5.5, Sonnet 5.5 và Fable 5.1 không cho ép `tool_choice` loại `tool` / `any` ([§2.3](#23-claude-anthropic)), Qwen không có `required`, MiniMax chỉ có `auto`. Nếu workflow cần model luôn trả dữ liệu cho một bước cụ thể, có thể dùng response format ở bước đó rồi để code xử lý kết quả.

**Tool results với MCP.** Từ spec 2025-06-18, MCP tool có thể khai báo `outputSchema`. Khi có, server **phải** trả kết quả khớp schema trong `structuredContent`, client **nên** validate, và để tương thích ngược, server nên trả thêm bản JSON đã serialize trong một `TextContent` block [67]:

```json
{
  "name": "get_order",
  "inputSchema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
  "outputSchema": {
    "type": "object",
    "properties": {"status": {"type": "string", "enum": ["pending", "shipped", "delivered"]}, "eta": {"type": "string", "format": "date"}},
    "required": ["status"]
  }
}
```

`outputSchema` mô tả dữ liệu server trả cho client, nên không tham gia ràng buộc quá trình model sinh token. Client và model biết trước cấu trúc kết quả; code đọc được dữ liệu trực tiếp từ `structuredContent`, và client có thể phát hiện lỗi cấu trúc ngay khi nhận response từ server.

**Final answer.** Khi agent là một bước trong pipeline lớn hơn, kết quả cuối nên có schema. LangChain `create_agent(response_format=...)`, Pydantic AI `output_type`, OpenAI Agents SDK đều có cơ chế này. Kiểm tra provider có cho dùng response format **cùng lúc** với tools không: Claude, OpenAI, Grok có; Gemini với built-in tools đang preview và chỉ cho Gemini 3 trở lên ([§2.4](#24-google-gemini)).

**Handoff giữa các agent.** Trong hệ multi-agent, dữ liệu truyền giữa agent nên có kiểu như dữ liệu giữa các service. Một agent nghiên cứu trả `ResearchReport(findings: list[Finding], open_questions: list[str])` cho agent viết; agent viết không phải đoán cấu trúc từ văn bản tự do. Lỗi ở biên được bắt ngay, thay vì lan sang agent sau.

## 3.7 Các ứng dụng điển hình

### Trích xuất thông tin từ tài liệu

Khi chuyển hoá đơn, hợp đồng, CV, email hoặc biên bản thành dữ liệu để nạp vào database, mỗi giá trị sai có thể được lưu lại mà không gây lỗi cấu trúc. Ngoài schema, pipeline cần cách chia tài liệu và đối chiếu nguồn:

- Input thường là PDF hoặc ảnh. Kiểm tra provider có áp schema khi input là ảnh không (Qwen thì không, [§2.6](#26-các-provider-trung-quốc)).
- Tài liệu dài: trích theo trang hoặc theo section, rồi gộp. Một schema khổng lồ cho tài liệu 50 trang dễ bị cắt và khó retry.
- Kèm **vị trí nguồn** cho mỗi giá trị quan trọng (số trang, câu trích) để người duyệt kiểm tra nhanh. Claude không cho dùng citations cùng JSON outputs; một field `source_quote` trong schema là cách thay thế.
- Các benchmark gần đây đo cả độ đúng của giá trị, không chỉ tính hợp lệ của schema, và đều cho thấy lỗi ngữ nghĩa vẫn còn dù cấu trúc đúng 100% [76][77].

### Phân loại và định tuyến

Với gán nhãn ticket, phát hiện ý định hoặc chọn agent con, enum giới hạn output vào các nhãn mà code biết xử lý ([§1.5](#15-ép-format-có-làm-model-kém-đi-không)). Khi tập nhãn nhỏ, có thể dùng `text/x.enum` (Gemini), `choice` (vLLM) hoặc `Output.choice()` (Vercel) để nhận một giá trị gọn. Thêm `other` / `unclear` nếu input có thể nằm ngoài tập nhãn, rồi theo dõi tỉ lệ này để phát hiện loại yêu cầu mới.

### LLM-as-judge

Chấm điểm câu trả lời, đánh giá an toàn, so sánh hai output. Schema điển hình là evidence trước, điểm sau, điểm dạng `Literal[1, 2, 3, 4, 5]` hoặc enum có tên (`"pass"`, `"fail"`) thay vì số thực. Định nghĩa từng mức điểm trong `description`. Structured output bảo đảm điểm đọc được bằng máy, không bảo đảm điểm được calibrate; vẫn cần so với nhãn của người trên một tập mẫu.

### Generative UI

Agent sinh mô tả giao diện, client render bằng component native. A2UI của Google (mã nguồn mở 12/2025) là một ví dụ: agent gửi JSON mô tả component từ một catalog đã duyệt, client render, không có code thực thi nào từ agent [68]. Catalog component chính là schema. Cách này an toàn hơn để model sinh HTML/JS, và streaming partial object ([§3.4](#34-streaming-partial-json)) cho UI hiện dần.

### Text-to-SQL và DSL

Khi output là một ngôn ngữ chứ không phải JSON, grammar mạnh hơn schema. Custom tool với Lark (OpenAI), `grammar` / `ebnf` (vLLM, SGLang), GBNF (llama.cpp) bảo đảm câu SQL đúng cú pháp hoặc chỉ dùng tập lệnh cho phép. Grammar không biết bảng hay cột nào tồn tại; vẫn cần chạy thử với `EXPLAIN` hoặc kiểm tra với catalog. Một phương án nhẹ hơn là để model sinh JSON mô tả truy vấn (`{"table": ..., "filters": [...]}`) rồi code dựng SQL; enum `table` và `column` lấy từ catalog thật.

### Sinh dữ liệu tổng hợp

Tạo dữ liệu test, dữ liệu huấn luyện, dữ liệu few-shot. Schema bảo đảm mỗi dòng nạp được; cần thêm kiểm tra phân phối (model hay lặp lại cùng một kiểu tên, cùng một khoảng giá trị) và kiểm tra giá trị (dữ liệu hợp lệ về cấu trúc vẫn có thể sai). `Output.array()` hay sinh theo batch nhỏ giúp tránh output quá dài bị cắt.

## 3.8 Chọn format: JSON, YAML, XML, TOON

JSON được các provider hỗ trợ rộng rãi cho constrained decoding và dễ đưa vào code hiện có. Với dữ liệu dạng bảng, việc lặp lại key ở mỗi dòng có thể tốn nhiều token; đó là lý do để cân nhắc format khác, nhất là khi đưa dữ liệu vào prompt.

**TOON** (Token-Oriented Object Notation, 2025) viết mảng object cùng cấu trúc như một bảng, chỉ khai báo key một lần:

```text
JSON:  [{"id": 1, "name": "Hà Nội", "revenue": 120}, {"id": 2, "name": "Huế", "revenue": 45}]
TOON:  rows[2]{id,name,revenue}:
         1,Hà Nội,120
         2,Huế,45
```

Mức tiết kiệm token và độ chính xác thay đổi theo việc model đọc hay sinh dữ liệu:

- Benchmark của dự án TOON báo độ chính xác ngang hoặc nhỉnh hơn JSON khi model **đọc** dữ liệu, với ít token hơn khoảng 40% [79].
- Một benchmark độc lập trên 209 câu hỏi truy xuất cho kết quả ngược lại: TOON 47,5% so với JSON 52,3%, dù TOON dùng ít token hơn nhiều; Markdown key-value đạt cao nhất (60,7%) [80].
- Khi model phải **sinh** dữ liệu, JSON thuần cho độ chính xác tốt nhất; "chi phí prompt" để giải thích TOON xoá mất phần tiết kiệm trên context ngắn [81].
- Trong vòng lặp agent gọi tool, TOON giảm 2–18% tổng token nhưng có thể giảm độ chính xác, và làm hỏng parallel tool calls trên một số model open-weight [82].

Với output cần nạp vào code, JSON có lợi thế về constrained decoding và công cụ xử lý. Với input là bảng lớn, đồng nhất và được đưa vào context nhiều lần, hãy thử TOON hoặc CSV rồi đo cả token lẫn độ chính xác trên tác vụ của bạn.

XML tag hữu ích để phân chia prompt và các khối text dài, đặc biệt khi dùng Claude; vai trò ở đây là đánh dấu nội dung trong prompt. YAML có thể gọn hơn JSON nhưng nhạy cảm với thụt lề và thiếu hỗ trợ constrained decoding từ provider.

## 3.9 Chi phí, độ trễ và cache

| Yếu tố | Ảnh hưởng | Gợi ý |
|---|---|---|
| Compile grammar | Request đầu với schema mới chậm hơn | Schema cố định; warm-up khi deploy với một request mẫu |
| Schema trong prompt | Provider chèn schema / hướng dẫn format vào input | Description ngắn gọn; bỏ field không dùng |
| Output token | Key JSON, dấu ngoặc, khoảng trắng đều là output token (đắt hơn input) | Tên key ngắn với schema lặp nhiều lần; tránh field `reasoning` dài khi đã có thinking |
| Prompt cache | Đổi schema làm mất cache trên Claude; schema nằm ở phần đầu request | Ghim schema theo phiên; đổi giữa phiên trên OpenAI bằng `configuration_update` |
| Retry | Mỗi retry là một request đầy đủ | Theo dõi tỉ lệ retry như một metric chi phí |
| Batch | Structured output chạy được với Batch API (Claude giảm 50%) | Extraction offline nên chạy batch |

Khi tính chi phí, cộng cả những lần phải sinh lại output. Constrained decoding có thể tốn thêm ở bước compile nhưng giảm số lần retry do sai cấu trúc. Trên engine self-hosted, jump-forward còn có thể bỏ qua các bước sinh token cố định.

## 3.10 Eval cho structured output

Một pipeline có thể đạt gần 100% JSON hợp lệ mà vẫn gán sai loại khiếu nại hoặc bịa mã đơn. Sau khi strict mode đã xử lý phần lớn lỗi cấu trúc, eval cần chỉ ra field nào còn sai và sai theo cách nào.

### Metric theo field

| Loại field | Metric | Ghi chú |
|---|---|---|
| Enum, boolean | Accuracy, confusion matrix theo lớp | Nhìn riêng lớp `other` / `unclear` |
| Chuỗi trích nguyên văn (tên, mã) | Exact match sau chuẩn hoá (khoảng trắng, hoa thường, dấu) | Chuẩn hoá tiếng Việt: có dấu / không dấu, Unicode NFC |
| Chuỗi tự do (tóm tắt, mô tả) | LLM judge hoặc similarity | Không dùng exact match |
| Số, tiền, ngày | Exact match hoặc sai số cho phép | Tách lỗi đơn vị (nghìn / đồng) khỏi lỗi giá trị |
| Field nullable | Precision / recall của việc trả `null` | Model trả giá trị khi đáng lẽ `null` = bịa; trả `null` khi có dữ liệu = bỏ sót |
| Mảng | Precision / recall theo phần tử, sau khi ghép cặp | Ghép theo key tự nhiên (mã dòng, tên) trước khi so |

Theo dõi riêng hai lỗi của field nullable: trả `null` khi có dữ liệu là bỏ sót; điền giá trị khi input không có dữ liệu là bịa. Gộp cả hai vào một chỉ số accuracy sẽ khó biết cần chỉnh cách trích xuất hay cách biểu diễn thông tin thiếu.

### Bộ dữ liệu

- Lấy từ **production**, gồm cả input xấu: email lẫn hai ngôn ngữ, ảnh mờ, tài liệu thiếu trang. Lưu các case từng gây lỗi làm regression test.
- Ground truth do người gán phải được kiểm tra lại. Cleanlab báo rằng **mọi** benchmark structured output công khai họ xem xét đều có ground truth sai, và phát hành bốn bộ đã làm sạch [78]. Dữ liệu nội bộ của bạn cũng không ngoại lệ.
- Chạy eval khi đổi model, đổi schema (kể cả chỉ đổi `description`), đổi prompt, đổi provider.

### Benchmark công khai

Các benchmark dưới đây đo những khía cạnh khác nhau. Chọn benchmark gần với tác vụ để tham khảo model hoặc engine, rồi kiểm tra lại trên dữ liệu ứng dụng:

- **JSONSchemaBench**: độ phủ và tuân thủ của engine constrained decoding trên schema thật [73].
- **StructEval**: sinh và chuyển đổi 18 format (JSON, YAML, CSV, HTML, SVG, Mermaid...) [74].
- **SchemaBench**: khoảng 40.000 schema; cho thấy model mới vẫn gặp khó, và RL với validator chi tiết giúp cải thiện [75].
- **LLMStructBench**: 22 model, 5 chiến lược prompt; cách prompt quan trọng hơn kích thước model, và lỗi ngữ nghĩa vẫn còn khi cấu trúc đã đúng [76].
- **SO-Bench**: structured output từ input đa phương thức (UI, tài liệu, biểu đồ) [77].

## 3.11 Observability

Khi một mã đơn sai xuất hiện trên dashboard, cần biết nó đến từ model nào, phiên bản schema nào và đã qua những bước kiểm tra nào. Một bản ghi cho mỗi request có thể gồm:

```json
{
  "schema_name": "complaint_extraction",
  "schema_version": "2026-11-01",
  "schema_hash": "sha256:9f2c…",
  "provider": "anthropic",
  "model": "claude-sonnet-5-5",
  "mode": "json_schema_strict",
  "stop_reason": "end_turn",
  "parse_ok": true,
  "validation_errors": [],
  "business_rule_errors": ["order_id_not_found"],
  "retries": 1,
  "null_fields": ["order_id"],
  "latency_ms": 1840,
  "input_tokens": 2310,
  "output_tokens": 142
}
```

Từ các bản ghi này, theo dõi:

- **Tỉ lệ theo `stop_reason`**: tăng `max_tokens` hay `refusal` là tín hiệu đầu tiên.
- **Tỉ lệ lỗi theo lớp** (parse, schema, business rule, grounding) và tỉ lệ retry.
- **Phân phối giá trị** của các field enum và tỉ lệ `null` theo field. Một field đột ngột `null` nhiều hơn, hay enum `other` tăng vọt, thường là dấu hiệu input đã đổi (template email mới, nhà cung cấp mới) trước khi ai đó báo lỗi.
- **Độ trễ p50 / p99 theo `schema_hash`**, để thấy chi phí compile khi deploy schema mới.

Giữ raw output của request lỗi để đối chiếu với kết quả sau parse. Hai bản này giúp xác định giá trị sai từ lúc model sinh hay từ bước xử lý phía client.

## 3.12 Failure modes thường gặp

Khi output lỗi, bắt đầu từ triệu chứng quan sát được. Chẳng hạn, `json.loads` báo lỗi có thể do JSON mode, nhưng cũng có thể chỉ vì response đã bị cắt trước khi đóng ngoặc.

| Triệu chứng | Nguyên nhân khả dĩ | Cách kiểm tra / xử lý |
|---|---|---|
| `json.loads` lỗi dù đã bật JSON mode | Output bị cắt vì `max_tokens`; DeepSeek trả chuỗi toàn khoảng trắng | Kiểm tra `finish_reason` / `stop_reason` và content rỗng trước khi parse |
| Output chỉ toàn khoảng trắng hoặc `\n` đến hết token | JSON mode không có chỉ dẫn JSON trong prompt (OpenAI); escape `\u` / `\t` trong prompt (Gemini) | Thêm chỉ dẫn; dùng UTF-8 trực tiếp; giới hạn whitespace (`whitespace_pattern` ở vLLM, `x{0,N}` ở GBNF) |
| Output rỗng, `status: incomplete` | Thinking dùng hết `max_output_tokens` (Gemini và reasoning model khác) | Tăng ngân sách, giảm thinking level |
| 400 ngay khi gửi request | Keyword ngoài subset; thiếu `additionalProperties: false` ở object lồng; root là `anyOf` (OpenAI); schema đệ quy (Claude, Grok); `propertyNames` từ `z.record()` | Đọc nguyên thông báo lỗi; chạy hàm transform cho provider |
| 400 `Schema is too complex for compilation` / `INVALID_ARGUMENT` | Quá nhiều field optional, union, enum lớn, tên dài, `minItems` / `maxItems` | Làm phẳng; tách schema; chuyển ràng buộc vào description |
| Request đầu sau deploy rất chậm | Compile grammar cho schema mới | Warm-up; không sinh schema động |
| Ràng buộc `minLength`, `maxLength` không có tác dụng | Gemini bỏ qua lặng lẽ; llama.cpp bỏ qua keyword không hỗ trợ | Validate phía client |
| Schema không được áp với input ảnh | Qwen chuyển về JSON Object mode với input đa phương thức | Validate phía client; chọn provider khác cho vision extraction |
| `strict` dường như không có tác dụng (DeepSeek) | Request đi vào `/v1` thay vì `/beta` | Log URL thực tế mà SDK gọi |
| Schema đúng nhưng giá trị bịa | Field bắt buộc không có đường lui | Nullable, `other`, confidence; grounding check |
| Enum sai hoa thường | Claude có thể đổi hoa thường của enum | So sánh không phân biệt hoa thường; tránh enum chỉ khác hoa thường |
| Kết luận mâu thuẫn với lập luận | Field kết luận sinh trước field lập luận (thứ tự key của provider) | Kiểm tra thứ tự key thực tế; dùng thinking |
| Output bị bọc trong code fence markdown | JSON mode không chặt (Kimi K2.6); prompt-only | Strict mode; parser bỏ fence; BAML |
| `refusal` với schema có field "giải thích từng bước" | Claude `reasoning_extraction` | Xin giải thích ngắn; dùng thinking |
| Hit rate prompt cache giảm sau khi thêm structured output | Đổi schema giữa phiên làm mất cache (Claude) | Ghim schema theo phiên |
| Pipeline đổi provider thì `KeyError` | Field optional bị bỏ qua (Claude, Gemini) so với luôn có mặt (OpenAI) | Mẫu số chung ở [§3.2](#32-schema-chạy-được-trên-nhiều-provider) |
| Output khác sau khi đổi backend vLLM | Dialect regex và độ phủ schema khác nhau giữa xgrammar, guidance | Chạy eval khi đổi backend |
| Code cũ không chạy sau khi nâng cấp | vLLM gỡ `guided_*` (v0.12.0); Claude `output_format` deprecated; Gemini Interactions gỡ `response_mime_type`; Vercel `generateObject` deprecated | Chuyển sang tham số mới ở Phần 2 |
| 400 `does not support assistant message prefill` | Code cũ prefill `{` trên Claude model mới | Dùng `output_config.format` |

## 3.13 Checklist cho AI agent

Dùng checklist này khi giao AI agent dựng hoặc sửa pipeline structured output. Mỗi mục cần được đối chiếu với schema, code xử lý response hoặc request thực tế gửi đi.

### Bắt buộc

- [ ] Dùng cơ chế mạnh nhất provider hỗ trợ: strict JSON Schema > strict tool > tool calling > JSON mode > prompt. Không dùng prompt-only khi có structured output native.
- [ ] Schema sinh từ một model Pydantic / Zod duy nhất, và chính model đó validate output.
- [ ] Root là `object`. Mọi object có `additionalProperties: false`. Với OpenAI và DeepSeek strict, mọi field nằm trong `required`, optional dùng union `null`.
- [ ] Không dùng keyword ngoài subset của provider đang dùng (bảng [§2.1](#21-bức-tranh-chung)). Ràng buộc bị loại khỏi schema gửi đi vẫn được validate phía client.
- [ ] Mỗi field có thể vắng trong input là nullable. Mỗi enum có giá trị `other` hoặc `unknown` nếu tập giá trị không đóng tuyệt đối.
- [ ] Field có đơn vị hoặc thang đo ghi rõ trong tên hoặc `description`.
- [ ] Trước khi parse, kiểm tra `stop_reason` / `finish_reason` / `status`: xử lý riêng refusal, truncation, content rỗng.
- [ ] Sau khi parse, chạy validation schema đầy đủ, rồi business rules. Không coi strict mode là đủ.
- [ ] Retry có giới hạn (≤ 3), gửi kèm thông báo lỗi cụ thể khi lỗi là validation. Không retry lỗi 400 do schema.
- [ ] Schema cố định trong suốt một phiên hội thoại; không sinh schema mới mỗi request.
- [ ] Log `schema_version`, `stop_reason`, lỗi theo lớp, số lần retry, raw output khi lỗi.

### Theo provider

- [ ] **OpenAI**: `strict: true` tường minh (mặc định `false` trên Chat Completions). Tools dùng cùng `json_schema` cũng strict. JSON mode phải có chữ "JSON" trong context. Đổi format giữa phiên dùng `configuration_update`.
- [ ] **Claude**: dùng `output_config.format`, không dùng `output_format` hay prefill. Không có `minimum`, `maxLength`, đệ quy trong schema gửi đi. Không bật citations cùng JSON outputs. Tối đa 20 strict tools, 24 tham số optional, 16 union. So sánh enum không phân biệt hoa thường.
- [ ] **Gemini**: Interactions dùng `response_format` với `mime_type: "application/json"` và `schema`; generateContent dùng `responseJsonSchema`. Biết rằng `minLength`, `uniqueItems`... bị bỏ qua lặng lẽ. Kiểm tra thứ tự key thực tế nếu dựa vào nó. Đặt `max_output_tokens` đủ cho cả thinking.
- [ ] **Grok**: không dùng `$ref` vòng. Ràng buộc độ dài vượt 2.048, mảng vượt 256 phần tử chỉ là best-effort.
- [ ] **DeepSeek**: JSON mode cần chữ "json" và ví dụ; xử lý content toàn khoảng trắng. Strict tool cần base URL `/beta` và `strict` trên mọi tool.
- [ ] **Qwen**: JSON Schema chỉ trên model được liệt kê; input ảnh / audio / video không được áp schema. `tool_choice` không có `required`.
- [ ] **Kimi, GLM, MiniMax, Doubao**: xác minh trên endpoint thật model có ép schema hay chỉ JSON mode; validate phía client trong mọi trường hợp.
- [ ] **Self-hosted**: vLLM dùng `structured_outputs`, không dùng `guided_*`. Cấu hình reasoning parser cho model có `<think>`. llama.cpp: tự mô tả cấu trúc trong prompt và validate vì keyword không hỗ trợ bị bỏ qua.

---

# Tổng hợp Sources

## OpenAI

| # | Resource | URL |
|---|---|---|
| 1 | Structured Outputs guide | [developers.openai.com/api/docs/guides/structured-outputs](https://developers.openai.com/api/docs/guides/structured-outputs) |
| 2 | Function calling guide (strict, custom tools, grammar) | [developers.openai.com/api/docs/guides/function-calling](https://developers.openai.com/api/docs/guides/function-calling) |
| 3 | Chat Completions API reference | [developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) |
| 4 | Models | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) |
| 5 | openai-python helpers (parse, pydantic_function_tool) | [github.com/openai/openai-python/blob/main/helpers.md](https://github.com/openai/openai-python/blob/main/helpers.md) |
| 6 | Introducing Structured Outputs in the API (08/2024) | [openai.com/index/introducing-structured-outputs-in-the-api/](https://openai.com/index/introducing-structured-outputs-in-the-api/) |
| 7 | Mastra issue #25235: `propertyNames` từ `z.record()` (cộng đồng) | [github.com/mastra-ai/mastra/issues/25235](https://github.com/mastra-ai/mastra/issues/25235) |
| 8 | openai-python issue #1733: json_schema với tool không strict (cộng đồng) | [github.com/openai/openai-python/issues/1733](https://github.com/openai/openai-python/issues/1733) |
| 9 | OpenAI forum: enum restrictions in strict schemas (cộng đồng) | [community.openai.com/t/enum-restrictions-in-strict-json-schemas-for-responses-api/1366742](https://community.openai.com/t/enum-restrictions-in-strict-json-schemas-for-responses-api/1366742) |

## Claude (Anthropic)

| # | Resource | URL |
|---|---|---|
| 10 | Structured outputs | [platform.claude.com/docs/en/build-with-claude/structured-outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) |
| 11 | Strict tool use | [platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use) |
| 12 | Claude Opus 5.5 migration guide | [platform.claude.com/docs/en/models/opus-5-5/migration-guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide) |
| 13 | Claude Sonnet 5.5 migration guide | [platform.claude.com/docs/en/models/sonnet-5-5/migration-guide](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide) |
| 14 | Handling stop reasons | [platform.claude.com/docs/en/build-with-claude/handling-stop-reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons) |

## Google Gemini

| # | Resource | URL |
|---|---|---|
| 15 | Structured output | [aistudio.google.com/docs/structured-output](https://aistudio.google.com/docs/structured-output) |
| 16 | Interactions API breaking changes (05/2026) | [aistudio.google.com/docs/interactions-breaking-changes-may-2026](https://aistudio.google.com/docs/interactions-breaking-changes-may-2026) |
| 17 | API reference: generateContent (Schema, GenerationConfig, FunctionCallingConfig) | [aistudio.google.com/api-reference/generate-content](https://aistudio.google.com/api-reference/generate-content) |
| 18 | API reference: Interactions (ResponseFormat, ToolChoiceConfig) | [aistudio.google.com/api-reference/interactions-api](https://aistudio.google.com/api-reference/interactions-api) |
| 19 | Models | [aistudio.google.com/docs/models](https://aistudio.google.com/docs/models) |
| 20 | Troubleshooting | [aistudio.google.com/docs/troubleshooting](https://aistudio.google.com/docs/troubleshooting) |
| 21 | Long decode | [aistudio.google.com/docs/long-decode](https://aistudio.google.com/docs/long-decode) |
| 22 | OpenAI compatibility | [aistudio.google.com/docs/openai](https://aistudio.google.com/docs/openai) |
| 23 | Vertex AI: Control generated output | [docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output) |
| 24 | Vertex AI: Function calling | [docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) |

## xAI Grok

| # | Resource | URL |
|---|---|---|
| 25 | Structured outputs | [docs.x.ai/developers/model-capabilities/text/structured-outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs) |
| 26 | Models | [docs.x.ai/developers/models](https://docs.x.ai/developers/models) |
| 27 | grok-4.7 | [docs.x.ai/developers/models/grok-4.7](https://docs.x.ai/developers/models/grok-4.7) |
| 28 | Function calling | [docs.x.ai/developers/tools/function-calling](https://docs.x.ai/developers/tools/function-calling) |

## Các provider Trung Quốc

| # | Resource | URL |
|---|---|---|
| 29 | DeepSeek: JSON Output | [api-docs.deepseek.com/guides/json_mode](https://api-docs.deepseek.com/guides/json_mode) |
| 30 | DeepSeek: Thinking mode | [api-docs.deepseek.com/guides/thinking_mode](https://api-docs.deepseek.com/guides/thinking_mode) |
| 31 | DeepSeek: Tool calls (strict beta) | [api-docs.deepseek.com/guides/tool_calls](https://api-docs.deepseek.com/guides/tool_calls) |
| 32 | Alibaba Model Studio: Qwen structured output | [alibabacloud.com/help/en/model-studio/qwen-structured-output](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output) |
| 33 | Alibaba Model Studio: Qwen function calling | [alibabacloud.com/help/en/model-studio/qwen-function-calling](https://www.alibabacloud.com/help/en/model-studio/qwen-function-calling) |
| 34 | Kimi: response_format | [platform.kimi.ai/docs/guide/response_format](https://platform.kimi.ai/docs/guide/response_format) |
| 35 | Kimi K3 quickstart | [platform.kimi.ai/docs/guide/kimi-k3-quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart) |
| 36 | Z.ai: Structured output | [docs.z.ai/guides/capabilities/struct-output](https://docs.z.ai/guides/capabilities/struct-output) |
| 37 | Z.ai: Function calling | [docs.z.ai/guides/capabilities/function-calling](https://docs.z.ai/guides/capabilities/function-calling) |
| 38 | MiniMax: Text generation API reference | [platform.minimax.io/docs/api-reference/text-post](https://platform.minimax.io/docs/api-reference/text-post) |
| 39 | Volcengine Ark: Function calling | [docs.volcengine.com/docs/ark/function-calling?lang=zh](https://docs.volcengine.com/docs/ark/function-calling?lang=zh) |
| 40 | Volcengine Ark: Structured output (beta) | [docs.volcengine.com/docs/ark/responses-api-structured-output?lang=zh](https://docs.volcengine.com/docs/ark/responses-api-structured-output?lang=zh) |
| 41 | Volcengine Ark: Model list | [volcengine.com/docs/82379/1593703](https://www.volcengine.com/docs/82379/1593703) |
| 42 | LangChain issue #40029: DeepSeek strict không tới endpoint beta (cộng đồng) | [github.com/langchain-ai/langchain/issues/40029](https://github.com/langchain-ai/langchain/issues/40029) |
| 43 | models.dev PR #5772: Ark từ chối response_format với Seed 2.0 Code (cộng đồng) | [github.com/anomalyco/models.dev/pull/5772](https://github.com/anomalyco/models.dev/pull/5772) |
| 44 | Alibaba Model Studio: MiniMax API (bảng tính năng M2.5) | [alibabacloud.com/help/en/model-studio/minimax-api](https://www.alibabacloud.com/help/en/model-studio/minimax-api) |

## Constrained decoding và engine

| # | Resource | URL |
|---|---|---|
| 45 | Willard & Louf: Efficient Guided Generation for LLMs (2023) | [arxiv.org/abs/2307.09702](https://arxiv.org/abs/2307.09702) |
| 46 | Dong et al.: XGrammar (2024) | [arxiv.org/abs/2411.15100](https://arxiv.org/abs/2411.15100) |
| 47 | MLC blog: XGrammar (11/2024) | [blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar) |
| 48 | XGrammar-2 (2026) | [arxiv.org/abs/2601.04426](https://arxiv.org/abs/2601.04426) |
| 49 | llguidance | [github.com/guidance-ai/llguidance](https://github.com/guidance-ai/llguidance) |
| 50 | llguidance: Making Structured Outputs Go Brrr | [guidance-ai.github.io/llguidance/llg-go-brrr](https://guidance-ai.github.io/llguidance/llg-go-brrr) |
| 51 | Zheng et al.: SGLang (2023) | [arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104) |
| 52 | vLLM: Structured outputs | [docs.vllm.ai/en/latest/features/structured_outputs.html](https://docs.vllm.ai/en/latest/features/structured_outputs.html) |
| 53 | SGLang: Structured outputs | [docs.sglang.io/advanced_features/structured_outputs.html](https://docs.sglang.io/advanced_features/structured_outputs.html) |
| 54 | llama.cpp: GBNF grammars | [github.com/ggml-org/llama.cpp/blob/master/grammars/README.md](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) |
| 55 | Ollama: Structured outputs (12/2024) | [ollama.com/blog/structured-outputs](https://ollama.com/blog/structured-outputs) |
| 56 | Hugging Face TGI: Guidance | [huggingface.co/docs/text-generation-inference/conceptual/guidance](https://huggingface.co/docs/text-generation-inference/conceptual/guidance) |
| 57 | TensorRT-LLM: Executor API (guided decoding) | [nvidia.github.io/TensorRT-LLM/advanced/executor.html](https://nvidia.github.io/TensorRT-LLM/advanced/executor.html) |

## Thư viện và giao thức

| # | Resource | URL |
|---|---|---|
| 58 | Pydantic AI: Output | [pydantic.dev/docs/ai/core-concepts/output/](https://pydantic.dev/docs/ai/core-concepts/output/) |
| 59 | Instructor: Retrying | [python.useinstructor.com/concepts/retrying/](https://python.useinstructor.com/concepts/retrying/) |
| 60 | Instructor: Retry mechanisms | [github.com/567-labs/instructor/blob/main/docs/learning/validation/retry_mechanisms.md](https://github.com/567-labs/instructor/blob/main/docs/learning/validation/retry_mechanisms.md) |
| 61 | LangChain: Structured output | [docs.langchain.com/oss/python/langchain/structured-output](https://docs.langchain.com/oss/python/langchain/structured-output) |
| 62 | Vercel AI SDK: Generating structured data | [ai-sdk.dev/docs/ai-sdk-core/generating-structured-data](https://ai-sdk.dev/docs/ai-sdk-core/generating-structured-data) |
| 63 | Vercel AI issue #10025: deprecate generateObject | [github.com/vercel/ai/issues/10025](https://github.com/vercel/ai/issues/10025) |
| 64 | Vercel AI issue #12491: Output.object và JSON mode của provider | [github.com/vercel/ai/issues/12491](https://github.com/vercel/ai/issues/12491) |
| 65 | Zod: JSON Schema | [zod.dev/json-schema](https://zod.dev/json-schema) |
| 66 | BAML | [docs.boundaryml.com/guide/introduction/what-is-baml](https://docs.boundaryml.com/guide/introduction/what-is-baml) |
| 67 | MCP specification 2025-06-18: Tools (outputSchema, structuredContent) | [modelcontextprotocol.io/specification/2025-06-18/server/tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools) |
| 68 | Google: Introducing A2UI (12/2025) | [developers.googleblog.com/introducing-a2ui-an-open-project-for-agent-driven-interfaces/](https://developers.googleblog.com/introducing-a2ui-an-open-project-for-agent-driven-interfaces/) |

## Papers, benchmark và bài phân tích

| # | Resource | URL |
|---|---|---|
| 69 | Tam et al.: Let Me Speak Freely? (2024) | [arxiv.org/abs/2408.02442](https://arxiv.org/abs/2408.02442) |
| 70 | .txt: Say What You Mean | [blog.dottxt.ai/say-what-you-mean.html](https://blog.dottxt.ai/say-what-you-mean.html) |
| 71 | Dylan Castillo: Structured outputs can hurt the performance of LLMs | [dylancastillo.co/posts/say-what-you-mean-sometimes.html](https://dylancastillo.co/posts/say-what-you-mean-sometimes.html) |
| 72 | Banerjee et al.: CRANE (ICML 2025) | [arxiv.org/abs/2502.09061](https://arxiv.org/abs/2502.09061) |
| 73 | Geng et al.: JSONSchemaBench (2025) | [arxiv.org/abs/2501.10868](https://arxiv.org/abs/2501.10868) |
| 74 | Yang et al.: StructEval (2025) | [arxiv.org/abs/2505.20139](https://arxiv.org/abs/2505.20139) |
| 75 | Lu et al.: Learning to Generate Structured Output with Schema Reinforcement Learning (SchemaBench, 2025) | [arxiv.org/abs/2502.18878](https://arxiv.org/abs/2502.18878) |
| 76 | Tenckhoff et al.: LLMStructBench (2026) | [arxiv.org/abs/2602.14743](https://arxiv.org/abs/2602.14743) |
| 77 | SO-Bench (2025) | [arxiv.org/abs/2511.21750](https://arxiv.org/abs/2511.21750) |
| 78 | Cleanlab: Structured output benchmark | [cleanlab.ai/blog/structured-output-benchmark/](https://cleanlab.ai/blog/structured-output-benchmark/) |
| 79 | TOON format | [github.com/toon-format/toon](https://github.com/toon-format/toon) |
| 80 | ImprovingAgents: TOON benchmarks | [improvingagents.com/blog/toon-benchmarks](https://www.improvingagents.com/blog/toon-benchmarks) |
| 81 | Matveev: TOON vs JSON generation (2026) | [arxiv.org/abs/2603.03306](https://arxiv.org/abs/2603.03306) |
| 82 | Kutschka & Geiger: Notation Matters (2026) | [arxiv.org/abs/2605.29676](https://arxiv.org/abs/2605.29676) |


---

Thông tin API trong tài liệu được ghi nhận tại **tháng 11/2026**. Tên tham số, model ID, tập con JSON Schema và giới hạn được lấy từ tài liệu chính thức; những chỗ chưa xác minh được đã được ghi ngay trong mục liên quan.

Các điểm còn cần kiểm tra trên endpoint gồm: model MiniMax nào hỗ trợ `json_schema`; danh sách model JSON Schema của Qwen; hỗ trợ trên từng phiên bản Doubao; `tool_choice` của DeepSeek, GLM và Doubao; thứ tự key của Gemini khi dùng JSON Schema; và hỗ trợ structured output của các model Grok ngoài `grok-4.7`.

Báo cáo từ issue tracker và forum được ghi riêng, chưa đối chiếu với changelog chính thức. Số liệu JSONSchemaBench phản ánh các engine đầu năm 2025; kết quả TOON phụ thuộc nghiên cứu và phiên bản spec. Khi chọn một khả năng hoặc giới hạn làm điều kiện thiết kế, hãy đối chiếu nguồn và thử trên đúng phiên bản sẽ triển khai.
