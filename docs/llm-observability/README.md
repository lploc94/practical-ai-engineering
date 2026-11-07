# Observability cho ứng dụng LLM

Thứ Hai, bộ phận chăm sóc khách hàng chuyển lên một ảnh chụp màn hình. Chatbot nói với khách rằng đơn hàng được hoàn tiền trong 3 ngày, trong khi chính sách ghi 14 ngày. Kỹ sư trực mở dashboard: tỉ lệ lỗi 0,02%, p95 latency 4,1 giây, mọi thứ xanh. Log ứng dụng có dòng `POST /chat 200 3812ms`. Không ai biết lúc đó prompt được render thế nào, retriever trả về tài liệu nào, model nào thực sự trả lời, hay câu trả lời có bị cắt giữa chừng không. Ảnh chụp màn hình là bằng chứng duy nhất, và nó không đủ để sửa lỗi.

Đây là khoảng trống mà observability truyền thống để lại khi gặp LLM. Metric HTTP đo request có thành công không, còn với LLM câu hỏi là câu trả lời có đúng không. Một lượt gọi model trả về `200` vẫn có thể sai nội dung, bị cắt vì hết `max_tokens`, bị từ chối, hoặc gọi tool với tham số sai. Chi phí không cố định theo request mà theo token, và token phụ thuộc vào cache, reasoning, độ dài hội thoại. Hành vi của model có thể đổi mà tên model không đổi.

Bài này đi theo ba phần. Phần 1 giải thích cần quan sát cái gì và vì sao: trace, metric riêng của LLM, chất lượng như một tín hiệu, và chuẩn OpenTelemetry GenAI. Phần 2 so sánh các công cụ đang dùng phổ biến. Phần 3 là thiết kế cụ thể: log gì, che PII thế nào, trace agent ra sao, tính chi phí theo tenant, nối feedback với trace, phát hiện drift, alert.

Bài chỉ nói về phía ứng dụng: trace, log, chi phí, chất lượng của các lượt gọi model. Hạ tầng serving (GPU, autoscale, metric của vLLM) thuộc bài [Deploying AI Systems](../deploying-ai-systems/) và [Self-hosting LLM](../self-hosting-llm/).

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API. Giá và gói của công cụ đọc ngày 06/11/2026, có thể đã đổi.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Vì sao log request/response thôi chưa đủ](#11-vì-sao-log-requestresponse-thôi-chưa-đủ)
  - [1.2 Trace, span và một lượt gọi LLM](#12-trace-span-và-một-lượt-gọi-llm)
  - [1.3 Metric riêng của LLM](#13-metric-riêng-của-llm-token-ttft-tokensgiây-cache-hit-tỉ-lệ-refusal)
  - [1.4 Chất lượng là một tín hiệu cần quan sát](#14-chất-lượng-là-một-tín-hiệu-cần-quan-sát)
  - [1.5 OpenTelemetry GenAI semantic conventions](#15-opentelemetry-genai-semantic-conventions)
- [Phần 2 — Công cụ](#phần-2--công-cụ)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

### Field cần có cho mỗi lượt gọi model

Tên attribute theo OpenTelemetry GenAI semantic conventions [1][2]. Tất cả `gen_ai.*` đang ở trạng thái Development, tức là có thể đổi tên.

| Nhóm | Field | Vì sao cần |
|---|---|---|
| Định danh | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `server.address` | Phải có ngay lúc tạo span vì sampler cần [2] |
| Model thực tế | `gen_ai.response.model`, `gen_ai.response.id` | Alias có thể trỏ sang snapshot khác; `response.id` để nối feedback và tra log của provider |
| Kết thúc | `gen_ai.response.finish_reasons`, `error.type` | Phân biệt trả lời xong, bị cắt, bị từ chối, lỗi |
| Token | `gen_ai.usage.input_tokens`, `output_tokens`, `cache_read.input_tokens`, `cache_write.input_tokens`, `reasoning.output_tokens` | Chi phí, cache hit rate, phát hiện prompt phình |
| Thời gian | thời lượng span, `gen_ai.response.time_to_first_chunk` (khi stream) | TTFT và tổng thời gian |
| Prompt | `gen_ai.prompt.name`, `gen_ai.prompt.version` | Biết lỗi thuộc phiên bản prompt nào |
| Hội thoại | `gen_ai.conversation.id` (chỉ khi có id thật) | Nối các lượt của một hội thoại |
| Của bạn | tenant, feature, user (đã hash), deployment version | Chia chi phí và chất lượng theo chiều kinh doanh |
| Nội dung | `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions` | **Opt-in**. Mặc định tắt vì chứa dữ liệu người dùng |

### Metric nên có trên dashboard

| Metric | Cách tính | Dấu hiệu bất thường |
|---|---|---|
| TTFT p50/p95 | `gen_ai.client.inference.time_to_first_chunk` | Tăng khi provider quá tải hoặc prompt dài ra |
| Thời lượng p95 | `gen_ai.client.inference.duration` | Tăng khi output dài ra hoặc effort reasoning tăng |
| Token theo loại | counter `gen_ai.client.inference.usage.*` | Input tăng dần theo tuần: context phình |
| Cache hit rate | `cache_read.input_tokens / input_tokens` | Giảm đột ngột sau deploy: prefix bị đổi |
| Chi phí | tự tính theo `gen_ai.response.model` | Chi phí trên mỗi request tăng mà traffic không tăng |
| Tỉ lệ bị cắt | `finish_reasons` chứa một trong các giá trị "bị cắt" của provider: `length`, `max_output_tokens`, `max_tokens`, `MAX_TOKENS` | Output dài hơn giới hạn, code nhận câu trả lời dở |
| Tỉ lệ refusal | `stop_reason=refusal` (Claude) hoặc tương đương | Đổi model, đổi prompt, hoặc bị tấn công |
| Lỗi theo provider | `error.type` theo `gen_ai.provider.name` | 429, 529, timeout tăng |
| Stream thiếu usage | span không có usage dù không lỗi | Stream bị ngắt hoặc thiếu cấu hình |
| Model trả về | phân phối `gen_ai.response.model` | Giá trị mới xuất hiện |

### Chọn công cụ

| Công cụ | Mô hình | Self-host | Gói miễn phí (06/11/2026) | Ghi chú |
|---|---|---|---|---|
| Langfuse | Tracing, prompt management, eval, score | Có (Docker Compose, Helm) | Hobby 50k units/tháng, giữ 30 ngày | Nhận OTLP/HTTP; server hiện là v4 |
| LangSmith | Tracing, dataset, eval | Chỉ gói Enterprise | Developer 5k trace/tháng, giữ 14 ngày | Hợp nhất nếu đã dùng LangChain/LangGraph |
| Arize Phoenix | Tracing, eval, embedding | Có, license ELv2 | Tự host không mất phí | Dùng OpenInference |
| Helicone | Proxy/gateway ghi log | Có (Docker, Helm) | Hobby 10k request, giữ 7 ngày | Nằm trên đường đi của request |
| Braintrust | Eval và log | Chỉ Enterprise | 1 GB + 10k score, giữ 14 ngày | Tính phí theo GB và score |
| Datadog | APM + LLM Observability | Không | 40k LLM span/tháng cho khách hàng Datadog | Nhận `gen_ai.*` từ semconv v1.37 |
| Tự xây trên OTel | Collector + backend sẵn có | Có | — | Tốn công nhất, linh hoạt nhất |

Chi tiết ở [Phần 2](#phần-2--công-cụ).

---

# Phần 1 — Nền tảng

## 1.1 Vì sao log request/response thôi chưa đủ

APM thường trả lời ba câu hỏi: request có lỗi không, mất bao lâu, ở đâu chậm. Với LLM, ba câu đó vẫn cần nhưng không đủ, vì những lỗi gây hại nhất lại trông như request thành công.

**Lỗi không hiện ra trên HTTP status.** Những trường hợp sau đều trả về `200`:

- Câu trả lời sai nội dung, bịa chính sách, trích nhầm tài liệu.
- Output bị cắt vì chạm `max_tokens`. Claude trả `stop_reason: "max_tokens"`, OpenAI Responses trả `status: "incomplete"`. Nếu code chỉ đọc text, nó nhận nửa câu trả lời như thể đã đủ.
- Model từ chối trả lời. Với Claude, refusal là response bình thường có `stop_reason: "refusal"`, không phải lỗi HTTP [35].
- Tool call có arguments sai schema, hoặc gọi tool không tồn tại.
- Model reasoning dùng hết ngân sách token cho phần suy luận và không viết được chữ nào (xem [Reasoning Models](../reasoning-models/)).

**Chi phí không tỉ lệ với số request.** Hai request giống nhau có thể chênh nhau mười lần về chi phí vì một cái đọc được cache, cái kia không. Một agent có thể gọi model 40 lần cho một câu hỏi. Log `200 3812ms` không cho biết gì về những điều này.

**Model đổi mà tên không đổi.** Chen, Zaharia và Zou (2023) đo cùng một bài toán nhận diện số nguyên tố trên GPT-4: độ chính xác 84% với bản tháng 3/2023 và 51% với bản tháng 6/2023, trong khi tên gọi qua API vẫn là `gpt-4` [27]. Nếu chỉ log tên model bạn yêu cầu, bạn không thấy thay đổi này. Log `gen_ai.response.model` (tên snapshot provider trả về) và chạy canary (xem [§3.5](#35-phát-hiện-drift-khi-provider-đổi-model)) là hai cách để thấy.

**Muốn debug phải phát lại được.** Để trả lời câu "vì sao bot nói 3 ngày", cần biết đúng prompt đã render (system prompt phiên bản nào, biến nào được chèn), tài liệu retriever trả về, lịch sử hội thoại, tham số, và output đầy đủ. Đó đều là nội dung người dùng, nên việc lưu chúng là một quyết định về quyền riêng tư (xem [§3.1](#31-log-gì-lưu-bao-lâu-che-pii-thế-nào)), không phải mặc định.

Tóm lại, observability cho LLM cần thêm ba thứ so với APM: **nội dung** (có kiểm soát), **token và chi phí**, và **tín hiệu chất lượng**.

## 1.2 Trace, span và một lượt gọi LLM

Một trace là toàn bộ công việc để phục vụ một yêu cầu, gồm nhiều span lồng nhau. Mỗi span là một bước có thời điểm bắt đầu, kết thúc, attribute và trạng thái. Với ứng dụng LLM, cây span thường trông như sau:

```text
POST /chat                                        (SERVER, span HTTP sẵn có)
└── invoke_agent support-bot                      (INTERNAL)
    ├── retrieval policies                        (truy xuất tài liệu)
    │   ├── embeddings text-embedding-x            (CLIENT)
    │   └── vector search                          (CLIENT, span của DB client)
    ├── chat claude-sonnet-5-5                     (CLIENT, model quyết định gọi tool)
    ├── execute_tool get_order                     (INTERNAL)
    └── chat claude-sonnet-5-5                     (CLIENT, đọc kết quả tool, viết câu trả lời)
```

Mấy điểm khác so với trace của một API thường:

- **Span gọi model mang nhiều dữ liệu hơn.** Ngoài thời gian và trạng thái, nó có token theo loại, model thực tế, lý do kết thúc, và tuỳ cấu hình là nội dung prompt và output.
- **Cây span của agent không cố định.** Cùng một câu hỏi, lần này agent gọi 2 tool, lần sau gọi 7. Số span và độ sâu là dữ liệu cần theo dõi (vòng lặp vô hạn hiện ra ở đây).
- **Span stream kết thúc khi stream kết thúc**, không phải khi hàm trả về iterator. Instrumentation đóng span sớm sẽ ghi thời lượng sai và mất usage, vì usage thường đến ở chunk cuối (xem [Streaming §3.10](../streaming/#310-observability-đo-ttft-tokensgiây-tỉ-lệ-stream-bị-cắt)).
- **Một client request có thể ứng với nhiều inference phía server.** Spec OTel nói rõ provider có thể ghi các span inference lồng nhau, và khi đếm token nên nhóm theo `service.name` để không đếm hai lần [2].

Trace giải quyết đúng bài toán ở đầu bài: từ ảnh chụp màn hình, tra theo user và thời điểm ra trace, thấy retriever trả về tài liệu chính sách cũ, thấy prompt version, thấy model trả lời. Muốn tra được như vậy, mỗi response hiển thị cho người dùng nên mang theo một id (trace id hoặc `gen_ai.response.id`) để bộ phận hỗ trợ gửi kèm khi báo lỗi.

## 1.3 Metric riêng của LLM: token, TTFT, tokens/giây, cache hit, tỉ lệ refusal

Trace để điều tra từng trường hợp. Metric để biết có vấn đề hay không, trên toàn bộ traffic, rẻ và lưu lâu. Spec OTel định nghĩa một số metric cho client [2][3]:

| Metric | Loại | Ý nghĩa |
|---|---|---|
| `gen_ai.client.inference.duration` | Histogram (s) | Từ lúc gửi request đến khi nhận đủ response hoặc lỗi. Bằng thời lượng span |
| `gen_ai.client.inference.time_to_first_chunk` | Histogram (s) | TTFT, chỉ với stream |
| `gen_ai.client.inference.time_per_output_chunk` | Histogram (s) | Khoảng cách giữa các chunk, chỉ với stream |
| `gen_ai.client.inference.usage.input_tokens` | Counter | Tổng input token, **đã gồm** cache read và cache write |
| `gen_ai.client.inference.usage.output_tokens` | Counter | Tổng output token, **đã gồm** reasoning |
| `gen_ai.client.inference.usage.cache_read.input_tokens` | Counter | Phần input đọc từ cache |
| `gen_ai.client.inference.usage.cache_write.input_tokens` | Counter | Phần input ghi vào cache |
| `gen_ai.client.inference.usage.reasoning.output_tokens` | Counter | Phần output dùng cho suy luận |
| `gen_ai.client.inference.operation.input_tokens` / `output_tokens` | Histogram | Phân phối token mỗi request, để xem p95/p99 |

Spec tách hai họ metric token có chủ đích. Counter `usage.*` dùng để tính tổng và ước chi phí. Histogram `operation.*` chỉ dùng để xem phân phối, spec ghi rõ không dùng chúng để tính tổng hay chi phí [3].

Nếu bạn đã đọc bài cũ hay code cũ dùng `gen_ai.client.token.usage` và `gen_ai.client.operation.duration` cho lượt gọi model: hai tên này thuộc bản cũ. Bản hiện tại không còn `gen_ai.client.token.usage`, còn `gen_ai.client.operation.duration` vẫn tồn tại nhưng spec ghi không dùng cho inference, embeddings, execute tool, invoke agent [4].

Spec cũng có `gen_ai.server.time_to_first_token` và `gen_ai.server.time_per_output_token`, nhưng đó là metric phía **model server** (khi bạn tự host model), không phải phía ứng dụng gọi API [4].

Những chỉ số sau không có trong spec, phải tự tính:

| Chỉ số | Công thức | Ghi chú |
|---|---|---|
| Cache hit rate | Σ `cache_read.input_tokens` / Σ `input_tokens` | Tính trên token, không trên số request. OpenAI cũng khuyên tính như vậy và gộp theo user, workspace, ngày [32] |
| Tokens/giây | `output_tokens / (duration − TTFT)` | Theo request, rồi lấy p50/p95 |
| Tỉ lệ bị cắt | số span có `finish_reasons` chứa `length` (Chat Completions), `max_output_tokens` (Responses), `max_tokens` (Claude) hoặc `MAX_TOKENS` (Gemini) / tổng | Tên giá trị khác nhau giữa provider; định nghĩa tập này một chỗ và dùng chung cho dashboard và alert |
| Tỉ lệ refusal | số span kết thúc bằng refusal / tổng | Claude: `stop_reason: "refusal"`, kèm `stop_details` cho biết danh mục [35] |
| Chi phí | Σ token × giá theo `gen_ai.response.model` | Xem [§3.3](#33-theo-dõi-chi-phí-theo-user-feature-tenant) |
| Tỉ lệ stream thiếu usage | stream kết thúc không lỗi nhưng không có usage / tổng stream | OpenAI Chat Completions chỉ gửi usage khi đặt `stream_options.include_usage`, và chunk usage có thể không đến nếu stream bị ngắt [36] |

**Chuẩn hoá token giữa các provider là chỗ dễ sai nhất.** Quy ước của OTel là `input_tokens` gồm cả phần cache và `output_tokens` gồm cả phần reasoning [2]. Các provider báo khác nhau:

| Provider | Input | Cache | Output và reasoning |
|---|---|---|---|
| Claude | `input_tokens` **chỉ** là phần sau cache breakpoint cuối; tổng input = `input_tokens + cache_read_input_tokens + cache_creation_input_tokens` [31] | `cache_read_input_tokens`, `cache_creation_input_tokens` | `output_tokens` đã gồm thinking; `output_tokens_details.thinking_tokens` là phần suy luận |
| OpenAI Responses | `input_tokens` đã gồm cache [32] | `input_tokens_details.cached_tokens`, `input_tokens_details.cache_write_tokens` | `output_tokens` đã gồm reasoning; `output_tokens_details.reasoning_tokens` [33] |
| Gemini generateContent | `promptTokenCount` đã gồm phần cached content [34] | `cachedContentTokenCount` | `candidatesTokenCount` **không** gồm `thoughtsTokenCount`; `totalTokenCount` = prompt + tool-use prompt + thoughts + candidates [34] |

Ghi thẳng `input_tokens` của Claude vào `gen_ai.usage.input_tokens` sẽ làm cache hit rate vượt 100% và chi phí lệch. Ghi `candidatesTokenCount` của Gemini làm output token, bạn bỏ sót toàn bộ phần thinking, trong khi Google tính tiền output là tổng của output và thinking (xem [Reasoning Models](../reasoning-models/)). Hàm `normalize_usage` ở [§2.6](#26-tự-xây-trên-opentelemetry) làm việc này. Bài [Multi-turn §3.11](../multi-turn-conversation/#311-observability-cho-cache) có bảng tên field cache cho thêm DeepSeek, Qwen và các API tương thích OpenAI.

## 1.4 Chất lượng là một tín hiệu cần quan sát

Latency và lỗi đo được tự động. Chất lượng thì không: không có field nào trong response cho biết câu trả lời đúng. Nhưng chất lượng vẫn phải là một đường trên dashboard, nếu không bạn chỉ biết nó giảm khi người dùng phàn nàn.

Ba nguồn tín hiệu chất lượng, xếp theo độ rẻ:

1. **Tín hiệu có sẵn trong response.** Tỉ lệ bị cắt, refusal, tool call sai schema, structured output không parse được, câu trả lời rỗng. Rẻ, chính xác, nhưng chỉ bắt được lỗi về hình thức.
2. **Tín hiệu từ người dùng.** Thumbs up/down, bấm "tạo lại", sửa output, sao chép câu trả lời, hỏi lại cùng ý, chuyển cho nhân viên. Có sẵn nhưng nhiễu và thiên lệch (xem [§3.4](#34-thu-feedback-người-dùng-và-nối-với-trace)).
3. **Chấm điểm online.** Chạy judge (LLM-as-judge đã hiệu chỉnh) hoặc grader code trên một phần trace. Đắt hơn, nhưng đo được thứ hai nguồn trên không đo được: câu trả lời có bám tài liệu không, có lộ thông tin nội bộ không. Cách xây và hiệu chỉnh judge ở bài [Evaluation](../evaluation/), phần chạy trên production ở [Evaluation §3.5](../evaluation/#35-eval-online-ab-test-feedback-người-dùng-sampling-log).

Kết quả chấm điểm nên gắn vào trace. OTel có event `gen_ai.evaluation.result` cho việc này, với `gen_ai.evaluation.name`, `gen_ai.evaluation.score.value`, `gen_ai.evaluation.score.label`, `gen_ai.evaluation.explanation` và `gen_ai.response.id` để trỏ về response được chấm [7]. Event này không chứa nội dung prompt hay output, nên lưu được lâu hơn nội dung.

Hai sự cố công khai cho thấy vì sao không thể chỉ tin một nguồn:

- **OpenAI, GPT-4o tháng 4/2025.** Bản cập nhật ngày 25/4 làm model nịnh người dùng quá mức và bị rollback. Theo bài phân tích của OpenAI, eval offline (kể cả eval về hành vi) và A/B test đều tốt; một số chuyên gia thử nghiệm nói model "felt slightly off" nhưng OpenAI vẫn ra mắt dựa trên tín hiệu tích cực từ người dùng A/B. Bản cập nhật có thêm reward signal từ thumbs up/down, và OpenAI viết rằng feedback người dùng "can sometimes favor more agreeable responses". Họ cũng không có eval riêng về sycophancy trong quy trình deploy [28].
- **Anthropic, tháng 8–9/2025.** Ba lỗi hạ tầng (định tuyến context window, output bị hỏng trên TPU, lỗi compiler ở approximate top-k) làm giảm chất lượng Claude trong nhiều tuần. Các lỗi chỉ xuất hiện trên một số nền tảng, batch size, cấu hình. Anthropic viết: "The evaluations we ran simply didn't capture the degradation users were reporting" [29].

Bài học cho observability: số liệu tổng hợp có thể đẹp trong khi một nhóm người dùng đang gặp lỗi. Cần cắt metric chất lượng theo chiều đủ nhỏ (provider, region, model snapshot, phiên bản prompt, loại phiên), và khi người đánh giá nói "có gì đó sai" mà số liệu nói ngược lại, đó là lúc điều tra, không phải lúc tin số liệu.

## 1.5 OpenTelemetry GenAI semantic conventions

OpenTelemetry (OTel) là chuẩn mở cho trace, metric, log. Semantic conventions là phần quy định tên attribute, tên span, tên metric, để dữ liệu từ thư viện khác nhau đổ về một backend vẫn query được chung. Nhóm GenAI quy định cho các lượt gọi model, tool, agent.

**Trạng thái hiện tại.** Các trang `opentelemetry.io/docs/specs/semconv/gen-ai/*` đã chuyển thành trang "Moved". Nội dung đang được duy trì ở repo riêng `open-telemetry/semantic-conventions-genai` [1]. Mọi attribute, span, metric `gen_ai.*` trong repo này đều ở trạng thái **Development**, chỉ `error.type`, `server.address`, `server.port` là Stable vì dùng chung với semconv chung [2]. Development nghĩa là tên có thể đổi giữa các bản. Thực tế đã đổi: `gen_ai.system` thành `gen_ai.provider.name`, `gen_ai.client.token.usage` thành họ `gen_ai.client.inference.usage.*`.

**Các span chính:**

| Span | Tên span | Kind | Attribute bắt buộc |
|---|---|---|---|
| Gọi model | `{gen_ai.operation.name} {gen_ai.request.model}`, ví dụ `chat gpt-6-astra` [2] | CLIENT | `gen_ai.operation.name`, `gen_ai.provider.name` |
| Thực thi tool | `execute_tool {gen_ai.tool.name}` [6] | INTERNAL | `gen_ai.operation.name`, `gen_ai.tool.name` |
| Chạy agent | `invoke_agent {gen_ai.agent.name}` [5] | INTERNAL | `gen_ai.operation.name` |

Giá trị của `gen_ai.operation.name` gồm `chat`, `generate_content`, `text_completion`, `embeddings`, `retrieval`, `execute_tool`, `invoke_agent`, `invoke_workflow`, `plan`, và các thao tác với memory [2]. `gen_ai.provider.name` có danh sách giá trị chuẩn, ví dụ `openai`, `anthropic`, `gcp.gemini`, `gcp.vertex_ai`, `aws.bedrock`, `azure.ai.openai`, `x_ai`, `deepseek` [1].

**Nội dung là opt-in.** Các attribute chứa nội dung (`gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `gen_ai.retrieval.query.text`...) mặc định không được ghi. Mỗi attribute có biến thể `*_ref` chứa URI trỏ ra storage ngoài, để nội dung nằm ở nơi có phân quyền và thời hạn lưu riêng [1][2].

**Biến môi trường: spec và thư viện đang lệch nhau.** Đây là chỗ cần cẩn thận khi cấu hình:

| | Theo spec [1] | Theo `opentelemetry-util-genai` 1.3b0 và `opentelemetry-instrumentation-openai-v2` 2.4b0 [8][9] |
|---|---|---|
| Bật ghi nội dung | `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT` | `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` |
| Giá trị | `NO_CONTENT` (mặc định), `SPAN_ONLY`, `EVENT_ONLY`, `SPAN_AND_EVENT` | Như bên trái; riêng instrumentation OpenAI còn nhận `true` (kiểu cũ) |
| Hook upload | `OTEL_INSTRUMENTATION_GENAI_CONTENT_HOOK=upload` | `OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload` |
| Đích upload | `OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`, `..._UPLOAD_FORMAT` (`json`/`jsonl`), `..._UPLOAD_MAX_QUEUE_SIZE` (mặc định 20) | Như bên trái |
| Dùng convention mới nhất | Không nhắc | Phải đặt `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`; không đặt thì instrumentation OpenAI theo semconv v1.30.0 |

Hệ quả thực tế: đặt đúng biến theo spec nhưng thư viện bạn dùng đọc biến khác thì nội dung không được ghi (an toàn), hoặc tệ hơn, bạn tưởng đã tắt mà thực ra đang bật qua biến kia. Đọc README của đúng phiên bản instrumentation đang dùng, rồi kiểm tra bằng một span thật trước khi tin cấu hình.

**Instrumentation có sẵn.** Không phải tự viết mọi thứ:

- `opentelemetry-instrumentation-openai-v2` (thuộc `opentelemetry-python-contrib`) cho OpenAI SDK, và các API tương thích OpenAI như Azure OpenAI, DeepSeek, Groq [8]. Có `opentelemetry-instrumentation-google-genai` cho Google Gen AI SDK.
- OpenLLMetry của Traceloop có instrumentation cho nhiều provider và framework [38]. Lưu ý gói `opentelemetry-instrumentation-anthropic` trên PyPI là của OpenLLMetry, không thuộc dự án OpenTelemetry.
- OpenInference của Arize là một bộ quy ước **khác** với `gen_ai.*`: dùng `openinference.span.kind` (`LLM`, `TOOL`, `AGENT`, `RETRIEVER`, `EMBEDDING`...), `llm.model_name`, `llm.token_count.prompt`, `input.value`, `output.value` [10]. Phoenix dùng bộ này.

Nếu dùng nhiều nguồn instrumentation, cùng một lượt gọi có thể có hai bộ tên attribute. Chọn một bộ làm chuẩn nội bộ (khuyên dùng `gen_ai.*` vì nhiều backend nhận trực tiếp), và kiểm tra backend map bộ còn lại thế nào.

**Backend nhận theo phiên bản semconv nào.** Datadog nhận `gen_ai.*` theo semconv v1.37 trở lên [23]. LangSmith map cả tên cũ `gen_ai.system` [17]. Langfuse đọc `gen_ai.*`, `llm.*` của OpenInference, và các attribute `langfuse.*` riêng [13]. Khi spec đổi tên, backend đổi theo chậm hơn. Pin phiên bản instrumentation và kiểm tra lại mapping khi nâng cấp.

---

# Phần 2 — Công cụ

Không có công cụ nào tốt nhất cho mọi đội. Câu hỏi đầu tiên là dữ liệu được phép đi đâu: nếu prompt chứa dữ liệu khách hàng không được rời hạ tầng của bạn, danh sách thu hẹp còn các công cụ tự host được. Câu hỏi thứ hai là bạn đã có stack observability chưa. Câu hỏi thứ ba là cần gì ngoài trace: prompt management, dataset, eval.

Một nguyên tắc giúp đổi công cụ sau này dễ hơn: instrument bằng OpenTelemetry với `gen_ai.*`, rồi export sang công cụ. Code của bạn không phụ thuộc SDK của vendor, và có thể gửi cùng lúc tới hai nơi trong giai đoạn chuyển đổi.

## 2.1 Langfuse

Langfuse gom tracing, prompt management, dataset, eval và score trong một sản phẩm. Mô hình dữ liệu: trace chứa các observation (span, generation, event...), score gắn vào trace hoặc observation, session nhóm các trace theo `sessionId`.

**Phiên bản.** Trang self-hosting hiện ghi "Version: v4" [14]. Python SDK trên PyPI là `langfuse` 5.x. Tài liệu hoặc bài viết nói về "Langfuse v3" là bản cũ.

**Nhận OpenTelemetry.** Endpoint `/api/public/otel` (trace ở `/api/public/otel/v1/traces`), chỉ OTLP qua HTTP (JSON hoặc protobuf), chưa hỗ trợ gRPC. Xác thực bằng Basic auth từ public key và secret key [13]:

```bash
AUTH_STRING=$(echo -n "pk-lf-...:sk-lf-..." | base64)
export OTEL_EXPORTER_OTLP_ENDPOINT="https://cloud.langfuse.com/api/public/otel"   # EU; US: us.cloud.langfuse.com
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic ${AUTH_STRING},x-langfuse-ingestion-version=4"
```

Header `x-langfuse-ingestion-version=4` quan trọng: thiếu nó, dữ liệu OTel có thể mất tới khoảng 10 phút mới hiện [13]. Khi debug "trace không lên", kiểm tra header này trước.

**Mapping attribute.** Langfuse đọc `gen_ai.*`, `llm.*` của OpenInference, và attribute riêng `langfuse.*` (ưu tiên cao nhất). User và session: `user.id`/`session.id` hoặc `langfuse.user.id`/`langfuse.session.id`. Metadata muốn lọc được phải đặt với tiền tố `langfuse.trace.metadata.*`; attribute OTel thường không có tiền tố sẽ rơi vào `metadata.attributes`, không lọc trực tiếp được [13]. Trace-level field phải có trên **mọi** span, không chỉ span gốc; dùng OTel Baggage hoặc `propagate_attributes()` của SDK [13].

**Che dữ liệu.** Python SDK có `mask_otel_spans`, chạy lúc export, sửa được cả attribute do instrumentation bên thứ ba ghi. Giới hạn: chỉ đổi được attribute (không đổi tên span, event, resource), và nếu hàm mask ném lỗi thì **cả batch bị bỏ** [15]. Test hàm mask như test code production.

**Self-host.** Docker Compose trên một VM (trang ghi rõ không có HA, scaling, backup) hoặc Kubernetes qua Helm, có hướng dẫn cho AWS, Azure, GCP. Thành phần: PostgreSQL, Redis/Valkey, ClickHouse, S3/Blob storage, container web và worker [14]. Sự kiện được ghi vào S3 trước khi vào DB, nên DB ngừng tạm thời không làm mất dữ liệu; đó là độ bền khi ingest, không thay cho backup. Một số tính năng quản trị (Organization Creators, Instance Management API, UI Customization) cần license key [14]. Tự host production nghĩa là tự vận hành ClickHouse và S3.

**Giá cloud** (06/11/2026) [16]: Hobby miễn phí 50k units/tháng, giữ 30 ngày; Core $29/tháng, 100k units, 90 ngày; Pro $199/tháng, 100k units, 3 năm; Enterprise $2.499/tháng. Mỗi trace, observation hoặc score là một unit. Một agent 10 bước có thể tốn hàng chục unit mỗi request.

## 2.2 LangSmith

LangSmith của LangChain gồm tracing, dataset, eval, annotation queue. Tích hợp sâu nhất với LangChain và LangGraph, nhưng nhận được trace từ code bất kỳ.

**Gói** (06/11/2026) [18]: Developer $0/seat, 5k base trace/tháng; Plus $39/seat/tháng, 10k base trace; Enterprise theo thoả thuận. Base trace giữ 14 ngày; extended trace giữ 180 ngày với phí cao hơn. **Self-host chỉ có ở Enterprise**, dưới hai dạng Hybrid (control plane SaaS, data plane của bạn) và Self-Hosted [18].

**Nhận OpenTelemetry** [17]:

- Endpoint là API URL cộng `/otel`, ví dụ `https://api.smith.langchain.com/otel`. Không tự thêm `/v1/traces` vào `OTEL_EXPORTER_OTLP_ENDPOINT` vì exporter tự thêm; thêm hai lần sẽ bị 404.
- Header `x-api-key`, và `Langsmith-Project` để chọn project.
- Map `gen_ai.*`: `gen_ai.operation.name` thành run type, `gen_ai.request.model`/`gen_ai.response.model` thành model, `gen_ai.usage.*` thành usage. Vẫn map tên cũ `gen_ai.system` và `gen_ai.usage.prompt_tokens`.

Hai giới hạn cần biết trước khi gặp [17]:

- **Span con bị bỏ âm thầm** nếu span cha không bao giờ được export tới LangSmith. Endpoint trả 200 trước khi xử lý, nên phía bạn không thấy lỗi. Trường hợp hay gặp: span gốc là span HTTP do một exporter khác xử lý, hoặc bị sampler loại.
- Chỉ exception event đầu tiên được điền vào trường `error`.

## 2.3 Arize Phoenix

Phoenix là công cụ tracing và eval của Arize, xây trên OpenTelemetry với quy ước OpenInference [10][19]. Chạy local, trong container, hoặc trên cloud của bạn. License là **Elastic License 2.0 (ELv2)** [19]: xem được mã nguồn và tự host được, nhưng ELv2 không phải license open source theo định nghĩa OSI và có hạn chế về việc cung cấp phần mềm như dịch vụ managed cho bên thứ ba. Nếu bạn định nhúng Phoenix vào sản phẩm bán cho khách, đọc kỹ license. Arize AX là sản phẩm managed thương mại riêng.

Điểm mạnh thường được nhắc là trực quan hoá embedding và drift, hợp với hệ thống RAG (xem [Embeddings §3.7](../embeddings/#37-observability-cho-embedding--rag-systems)). Vì dùng OpenInference, nếu bạn instrument bằng `gen_ai.*` thì kiểm tra Phoenix hiển thị các attribute đó thế nào, hoặc dùng instrumentation OpenInference cho phần gửi về Phoenix.

## 2.4 Helicone và các gateway có logging

Helicone là proxy: bạn đổi base URL của SDK sang Helicone, Helicone chuyển tiếp request tới provider và ghi log trên đường đi. Không cần thêm code instrumentation, đổi một dòng cấu hình là có log.

**Gói** (06/11/2026) [21]: Hobby miễn phí 10k request, giữ 7 ngày; Pro $79/tháng; Team $799/tháng; Enterprise theo thoả thuận. **Self-host** có Docker Compose, Helm/Kubernetes, cài thủ công [20]. Trang self-host không ghi tên license; kiểm tra file LICENSE của repo trước khi kết luận.

Mô hình proxy có đánh đổi rõ:

- **Ưu:** không sửa code, thấy đúng request/response đi qua mạng, thường kèm cache, rate limit, fallback.
- **Nhược:** nằm trên đường đi của request. Proxy chậm thì ứng dụng chậm; proxy sập thì ứng dụng không gọi được model, trừ khi bạn có đường vòng (gọi thẳng provider khi proxy lỗi). Proxy chỉ thấy lượt gọi model, không thấy retriever, tool, logic nghiệp vụ, nên không dựng được trace đầy đủ của một agent. Đo độ trễ thêm vào của proxy trong môi trường của bạn trước khi dùng cho đường nóng.

Các LLM gateway khác (LiteLLM proxy, gateway của cloud provider) có cùng mô hình. Gateway hợp để có log tập trung cho nhiều đội, kiểm soát key và quota; trace chi tiết của ứng dụng vẫn cần instrumentation trong code. Gateway và routing được bàn ở bài [Model Routing](../model-routing/).

## 2.5 Datadog, Grafana và stack observability sẵn có

Nếu tổ chức đã dùng Datadog, New Relic, Grafana, Honeycomb, thêm trace LLM vào đó có lợi lớn: span gọi model nằm chung trace với span HTTP, DB, queue; cùng hệ thống alert, cùng người trực, cùng quy trình phân quyền.

**Datadog LLM Observability** (trang docs hiện gọi là Agent Observability). Có SDK Python tự instrument OpenAI, Anthropic, LangChain, Bedrock, và nhận trực tiếp span theo OTel GenAI semconv v1.37 trở lên qua Collector, Datadog Agent hoặc OTLP intake, không cần SDK riêng [22][23]. Có dashboard chi phí, latency, usage dựng sẵn, quét dữ liệu nhạy cảm, phát hiện prompt injection [22].

Cách tính phí theo LLM span: mỗi request tới provider là một span tính phí, nên một agent gọi model 15 lần cho một câu hỏi tạo 15 span [22]. Blog của Datadog ghi miễn phí tới 40.000 LLM span/tháng cho khách hàng Datadog [23]; giá phần vượt không có trên các trang đã đọc, cần hỏi Datadog. Datadog LLM Observability không có trên các site gov [22].

**Grafana stack** (Tempo cho trace, Prometheus/Mimir cho metric, Loki cho log) không có sản phẩm LLM riêng theo cùng kiểu, nhưng nhận OTLP. Bạn dựng dashboard từ metric `gen_ai.client.inference.*`. Đây là lựa chọn hợp lý khi đã vận hành Grafana và không cần giao diện xem prompt/response đẹp.

Hạn chế chung của APM tổng quát: giao diện xem hội thoại, so sánh phiên bản prompt, gán nhãn trace để đưa vào dataset thường kém hơn công cụ chuyên cho LLM. Nhiều đội dùng hai nơi: APM cho metric và alert, một công cụ LLM cho đọc trace và eval, cùng nhận từ một pipeline OTel.

**Các công cụ khác.** Braintrust gom eval và log, tính phí theo GB dữ liệu và số score (Starter $0 với 1 GB và 10k score, giữ 14 ngày; Pro $249/tháng), self-host chỉ ở Enterprise [25]. Pydantic Logfire xây trên OTel, tính theo record (Personal miễn phí 10M record/tháng, giữ 30 ngày; Team $49/tháng) [26]. W&B Weave có tracing theo session/turn, scorer dựng sẵn và alert [37]; trang sản phẩm không nêu license và giá.

## 2.6 Tự xây trên OpenTelemetry

Tự xây nghĩa là: instrument bằng OTel SDK, đẩy qua OTel Collector, lưu ở backend bạn đã có hoặc gửi tiếp sang công cụ ở trên. Dưới đây là một wrapper tối thiểu cho một lượt gọi model: tạo span theo `gen_ai.*`, chuẩn hoá usage của ba provider, tính chi phí theo model thực trả về, ghi metric, và đánh dấu trace cần giữ lại khi sampling.

```python
import asyncio
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind, Status, StatusCode

tracer = trace.get_tracer("app.llm")
meter = metrics.get_meter("app.llm")

DURATION_BUCKETS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56,
                    5.12, 10.24, 20.48, 40.96, 81.92]
duration_hist = meter.create_histogram(
    "gen_ai.client.inference.duration", unit="s",
    explicit_bucket_boundaries_advisory=DURATION_BUCKETS)
ttfc_hist = meter.create_histogram(
    "gen_ai.client.inference.time_to_first_chunk", unit="s",
    explicit_bucket_boundaries_advisory=DURATION_BUCKETS)
usage_counters = {
    name: meter.create_counter(f"gen_ai.client.inference.usage.{name}", unit="{token}")
    for name in ("input_tokens", "output_tokens",
                 "cache_read.input_tokens", "cache_write.input_tokens",
                 "reasoning.output_tokens")
}
cost_counter = meter.create_counter("app.llm.cost", unit="USD")  # không có trong semconv

# Lý do kết thúc bình thường. Mọi lý do khác (bị cắt, refusal, safety, lý do mới chưa biết) đều giữ trace.
# OpenAI Responses: ghi "completed" khi status=completed, ngược lại ghi incomplete_details.reason.
NORMAL_FINISH = {"stop", "end_turn", "stop_sequence", "tool_use", "tool_calls", "pause_turn", "completed"}
log = logging.getLogger("app.llm.telemetry")
telemetry_errors = meter.create_counter("app.llm.telemetry.errors")


def normalize_usage(provider: str, u: dict) -> dict:
    """Đưa usage của provider về một quy ước: input gồm cả cache, output gồm cả reasoning."""
    if provider == "anthropic":
        read = u.get("cache_read_input_tokens") or 0
        write = u.get("cache_creation_input_tokens") or 0
        details = u.get("output_tokens_details") or {}
        return {"input_tokens": u["input_tokens"] + read + write,   # input_tokens chỉ là phần sau breakpoint
                "output_tokens": u["output_tokens"],               # đã gồm thinking
                "cache_read.input_tokens": read,
                "cache_write.input_tokens": write,
                "reasoning.output_tokens": details.get("thinking_tokens") or 0}
    if provider == "openai":                                       # Responses API
        idet = u.get("input_tokens_details") or {}
        odet = u.get("output_tokens_details") or {}
        return {"input_tokens": u["input_tokens"],                 # đã gồm cached
                "output_tokens": u["output_tokens"],               # đã gồm reasoning
                "cache_read.input_tokens": idet.get("cached_tokens") or 0,
                "cache_write.input_tokens": idet.get("cache_write_tokens") or 0,
                "reasoning.output_tokens": odet.get("reasoning_tokens") or 0}
    if provider == "gcp.gemini":                                   # generateContent, usageMetadata
        thoughts = u.get("thoughtsTokenCount") or 0
        return {"input_tokens": u.get("promptTokenCount", 0) + (u.get("toolUsePromptTokenCount") or 0),
                "output_tokens": (u.get("candidatesTokenCount") or 0) + thoughts,  # thoughts tách riêng
                "cache_read.input_tokens": u.get("cachedContentTokenCount") or 0,
                "cache_write.input_tokens": 0,
                "reasoning.output_tokens": thoughts}
    raise ValueError(f"chưa có mapping usage cho provider {provider!r}")


@dataclass
class Price:            # USD / 1 triệu token
    input: float
    cache_read: float
    cache_write: float
    output: float


def find_price(prices: dict[str, Price], response_model: str) -> Price | None:
    """Khớp tiền tố dài nhất: 'model-a-2026-05-01' dùng giá của 'model-a'."""
    keys = [k for k in prices if response_model == k or response_model.startswith(k + "-")]
    return prices[max(keys, key=len)] if keys else None


def cost_usd(n: dict, p: Price) -> float:
    uncached = n["input_tokens"] - n["cache_read.input_tokens"] - n["cache_write.input_tokens"]
    return (uncached * p.input + n["cache_read.input_tokens"] * p.cache_read
            + n["cache_write.input_tokens"] * p.cache_write
            + n["output_tokens"] * p.output) / 1_000_000          # reasoning đã nằm trong output


@dataclass
class Inference:
    provider: str
    response_id: str | None = None
    response_model: str | None = None
    finish_reasons: list[str] = field(default_factory=list)
    usage: dict | None = None            # usage thô của provider
    first_chunk_at: float | None = None

    def mark_first_chunk(self):
        if self.first_chunk_at is None:
            self.first_chunk_at = time.monotonic()


@contextmanager
def traced_inference(*, provider, operation, request_model, server_address,
                     prices, stream=False, app_attrs=None, metric_attrs=None):
    """app_attrs: tenant, feature, prompt version... gắn lên span.
    metric_attrs: chỉ những chiều có số giá trị nhỏ (feature, tier), gắn lên metric."""
    attrs = {"gen_ai.operation.name": operation,
             "gen_ai.provider.name": provider,
             "gen_ai.request.model": request_model,
             "server.address": server_address}
    if stream:
        attrs["gen_ai.request.stream"] = True
    rec = Inference(provider=provider)
    start = time.monotonic()
    # các attribute dùng cho sampling phải có ngay lúc tạo span
    with tracer.start_as_current_span(f"{operation} {request_model}", kind=SpanKind.CLIENT,
                                      attributes={**attrs, **(app_attrs or {})},
                                      record_exception=False, set_status_on_exception=False) as span:
        error_type, cancelled = None, False
        try:
            yield rec
        except (GeneratorExit, asyncio.CancelledError):
            cancelled = True                   # người dùng ngắt stream: không phải lỗi provider
            span.set_attribute("app.llm.cancelled", True)
            raise
        except BaseException as e:
            error_type = getattr(e, "status_code", None) or type(e).__qualname__
            span.set_attribute("error.type", str(error_type))
            span.set_status(Status(StatusCode.ERROR, str(e)[:200]))
            span.record_exception(e)
            span.set_attribute("app.llm.keep", True)
            raise
        finally:
            # telemetry lỗi thì ghi nhận rồi bỏ qua, không được thay exception của ứng dụng
            try:
                _finish(span, rec, attrs, metric_attrs, provider, request_model, prices,
                        start, error_type, cancelled)
            except Exception:
                _report_telemetry_error(span, provider)


def _report_telemetry_error(span, provider):
    # chính đường báo lỗi cũng có thể hỏng (logger, metric SDK): mỗi bước tự nuốt lỗi của nó
    for report in (lambda: log.exception("telemetry finalize failed"),
                   lambda: telemetry_errors.add(1, {"gen_ai.provider.name": provider}),
                   lambda: span.set_attribute("app.llm.telemetry_error", True)):
        try:
            report()
        except Exception:
            pass


def _finish(span, rec, attrs, metric_attrs, provider, request_model, prices,
            start, error_type, cancelled):
    elapsed = time.monotonic() - start
    m_attrs = {**attrs, **(metric_attrs or {})}
    m_attrs.pop("gen_ai.request.stream", None)
    if error_type is not None:
        m_attrs["error.type"] = str(error_type)
    if rec.response_model:
        m_attrs["gen_ai.response.model"] = rec.response_model
        span.set_attribute("gen_ai.response.model", rec.response_model)
    if rec.response_id:
        span.set_attribute("gen_ai.response.id", rec.response_id)
    if rec.finish_reasons:
        span.set_attribute("gen_ai.response.finish_reasons", rec.finish_reasons)
        if {r.lower() for r in rec.finish_reasons} - NORMAL_FINISH:
            span.set_attribute("app.llm.keep", True)
    if rec.first_chunk_at is not None:
        ttfc = rec.first_chunk_at - start
        span.set_attribute("gen_ai.response.time_to_first_chunk", ttfc)
        ttfc_hist.record(ttfc, m_attrs)
    if not cancelled:                          # thời lượng của lượt bị huỷ làm lệch phân phối
        duration_hist.record(elapsed, m_attrs)
    if rec.usage is not None:
        n = normalize_usage(provider, rec.usage)
        for k, v in n.items():
            if v:
                span.set_attribute(f"gen_ai.usage.{k}", v)
                tok_attrs = m_attrs if k.startswith("reasoning") else {**m_attrs, "gen_ai.token.modality": "text"}
                usage_counters[k].add(v, tok_attrs)
        price = find_price(prices, rec.response_model or request_model)
        if price is None:
            span.set_attribute("app.llm.cost.unpriced", True)   # đừng ghi 0 cho có
        else:
            c = cost_usd(n, price)
            span.set_attribute("app.llm.cost_usd", c)
            cost_counter.add(c, m_attrs)
    elif error_type is None and not cancelled:
        span.set_attribute("app.llm.usage_missing", True)
        span.set_attribute("app.llm.keep", True)
```

Dùng với Anthropic SDK khi stream:

```python
import anthropic

client = anthropic.Anthropic()
PRICES = load_prices()      # từ config có ngày hiệu lực, không hardcode trong code

def answer(question: str, tenant_id: str):
    with traced_inference(provider="anthropic", operation="chat",
                          request_model="claude-sonnet-5-5", server_address="api.anthropic.com",
                          prices=PRICES, stream=True,
                          app_attrs={"app.tenant.id": tenant_id, "app.feature": "support",
                                     "gen_ai.prompt.name": "support-answer", "gen_ai.prompt.version": "v14"},
                          metric_attrs={"app.feature": "support"}) as rec:
        with client.messages.stream(model="claude-sonnet-5-5", max_tokens=2048,
                                    messages=[{"role": "user", "content": question}]) as stream:
            for text in stream.text_stream:
                rec.mark_first_chunk()
                yield text
            msg = stream.get_final_message()
        rec.response_id, rec.response_model = msg.id, msg.model
        rec.finish_reasons = [msg.stop_reason]
        rec.usage = msg.usage.model_dump()
```

Vài quyết định trong đoạn code đáng giải thích:

- **Span bao trọn vòng đọc stream.** Span chỉ đóng khi vòng `for` xong và đã có message cuối, nên thời lượng và usage đúng. Nếu người dùng ngắt và generator bị đóng (`GeneratorExit`, hoặc `CancelledError` với code async), span được ghi `app.llm.cancelled=true`, không có trạng thái ERROR, nên không làm tăng error rate. Lượt bị huỷ không có usage cuối; provider vẫn có thể tính tiền phần đã sinh, nên tỉ lệ `app.llm.cancelled` cao là một nguồn lệch khi đối soát hoá đơn ([§3.3](#33-theo-dõi-chi-phí-theo-user-feature-tenant)).
- **Telemetry không được làm hỏng request.** Phần ghi attribute, metric và tính chi phí chạy trong `try/except` riêng. Mapping usage thiếu một provider, hay bảng giá có lỗi, chỉ ghi log và tăng `app.llm.telemetry.errors`; exception gốc của ứng dụng (nếu có) vẫn được ném ra nguyên vẹn. Không có lớp này, một `ValueError` trong `normalize_usage` sẽ thay thế lỗi thật của request trong khối `finally`.
- **Attribute dùng cho sampling được đặt lúc tạo span** (`operation`, `provider`, `request.model`, `server.address`), theo khuyến nghị của spec [2]. Sampler ở SDK chỉ thấy attribute có lúc tạo span.
- **Giữ trace theo danh sách lý do bình thường, không theo danh sách lý do bất thường.** Mỗi provider đặt tên khác nhau cho cùng một chuyện: bị cắt là `length` (Chat Completions), `max_output_tokens` (Responses, trong `incomplete_details.reason`), `max_tokens` (Claude), `MAX_TOKENS` (Gemini); bị chặn có `content_filter`, `refusal`, `SAFETY`, `RECITATION`, `PROHIBITED_CONTENT`, `SPII`... [34][35][42]. Liệt kê hết lý do bất thường thì dễ sót, và sót nghĩa là trace bị cắt rơi vào mẫu 5%. Làm ngược lại: mọi lý do không nằm trong `NORMAL_FINISH` đều giữ, kể cả lý do provider mới thêm.
- **Refusal không phải lỗi.** `stop_reason: "refusal"` vào `finish_reasons`, span không có trạng thái ERROR, nhưng `app.llm.keep` được bật để trace được giữ lại.
- **`error.type` lấy mã HTTP khi có** (`status_code` của exception trong SDK Anthropic và OpenAI), nếu không thì lấy tên lớp exception. Spec chấp nhận cả hai kiểu giá trị [2].
- **Model không có giá thì đánh dấu, không ghi 0.** Chi phí bằng 0 trên dashboard trông như tiết kiệm, trong khi thực ra là thiếu dữ liệu.
- **Tenant ở trên span, không ở trên metric.** Mỗi giá trị attribute khác nhau trên metric tạo một time series riêng. 50.000 tenant nhân với số model, số provider sẽ làm backend metric quá tải. Chi phí theo tenant tính từ span (xem [§3.3](#33-theo-dõi-chi-phí-theo-user-feature-tenant)).
- **`gen_ai.token.modality` cố định là `text`** vì ví dụ chỉ có văn bản. Spec yêu cầu chia counter usage theo modality [3]; nếu gửi ảnh hoặc audio, tách token theo modality từ usage của provider (OTel có `gen_ai.usage.image.input_tokens`, `gen_ai.usage.audio.input_tokens`...).

Đoạn code trên chạy được với `opentelemetry-sdk` 1.45 và đã được kiểm tra bằng exporter in-memory: chuẩn hoá usage của cả ba provider, tính chi phí khi có cache write, nhánh lỗi, nhánh refusal, các finish reason bị cắt hoặc bị chặn của cả ba provider, huỷ stream (sync và async), lỗi trong chính phần telemetry, model chưa có giá, thiếu usage, và metric không chứa tenant.

**Pipeline phía sau** thường là: SDK trong ứng dụng → OTel Collector → một hay nhiều backend. Collector là nơi đặt tail sampling, che dữ liệu bổ sung, và fan-out sang nhiều đích. Cấu hình gửi trace sang Langfuse từ Collector:

```yaml
exporters:
  otlphttp/langfuse:
    endpoint: "https://cloud.langfuse.com/api/public/otel"
    headers:
      Authorization: "Basic ${env:LANGFUSE_AUTH}"
      x-langfuse-ingestion-version: "4"
```

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Log gì, lưu bao lâu, che PII thế nào

Bắt đầu bằng hai lớp dữ liệu với chính sách khác nhau:

| Lớp | Gồm | Lưu bao lâu | Ai xem được |
|---|---|---|---|
| Metadata | provider, model yêu cầu và thực trả về, token, chi phí, thời gian, finish reason, `error.type`, response id, tenant, feature, prompt version, số tool call | Lâu (vài tháng tới hơn một năm), vì nhẹ và không chứa nội dung người dùng | Kỹ sư, product, tài chính |
| Nội dung | prompt đã render, lịch sử hội thoại, tài liệu retriever trả về, tool arguments và result, output | Ngắn (vài ngày tới vài tuần), theo chính sách dữ liệu | Nhóm nhỏ, có audit log |

Metadata đủ để trả lời phần lớn câu hỏi vận hành: cái gì chậm, cái gì tốn, cái gì lỗi, bắt đầu từ khi nào. Nội dung cần cho debug từng trường hợp và để xây eval dataset.

**Tách nội dung ra khỏi span.** Spec OTel có sẵn cách: attribute `*_ref` chứa URI tới object storage, cùng hook upload trong thư viện instrumentation (xem bảng biến môi trường ở [§1.5](#15-opentelemetry-genai-semantic-conventions)) [1]. Span giữ con trỏ, nội dung nằm ở bucket có phân quyền và lifecycle rule riêng. Xoá dữ liệu của một người dùng khi họ yêu cầu khi đó là xoá object trong bucket, không phải tìm trong backend trace. Spec cũng lưu ý bản thân URI có thể lộ tên bucket, tenant, id [2].

**Che trước khi dữ liệu rời ứng dụng.** Thứ tự an toàn:

1. Trong code, trước khi đặt attribute: thay số điện thoại, email, số thẻ, số định danh bằng placeholder. Đây là chỗ duy nhất bạn chắc dữ liệu chưa đi đâu.
2. Ở bước export trong SDK: ví dụ `mask_otel_spans` của Langfuse, chạy được trên cả attribute do instrumentation bên thứ ba ghi [15].
3. Ở Collector: dữ liệu đã rời process ứng dụng. Hữu ích như lớp thứ hai, không thay cho lớp một.
4. Ở backend: Datadog có quét và che dữ liệu nhạy cảm [22]. Dữ liệu đã nằm ở hệ thống của bên thứ ba.

Che bằng regex bắt được định dạng có cấu trúc (email, số thẻ) nhưng không bắt được "tôi bị tiểu đường, đang dùng thuốc X". Nếu ứng dụng nhận loại nội dung đó, giải pháp là không ghi nội dung ở môi trường production, hoặc chỉ ghi trên mẫu nhỏ đã qua bộ phát hiện PII tốt hơn, có người duyệt quyền truy cập.

**Tool arguments và result** cũng là nội dung. Spec ghi rõ chúng có thể chứa thông tin nhạy cảm và để opt-in [6]. Một tool `get_customer` trả về địa chỉ, số điện thoại; ghi result vào span là ghi những thứ đó.

**Sampling.** Không cần giữ mọi trace. Hai kiểu [11]:

- **Head sampling**: quyết định ở đầu trace, thường theo trace id (ví dụ giữ 5%). Rẻ, giữ nguyên trace, nhưng không biết trace nào sẽ lỗi.
- **Tail sampling**: quyết định sau khi đã thấy các span của trace. Giữ được mọi trace lỗi, chậm, hoặc có cờ đặc biệt. Đắt hơn: Collector phải giữ span trong bộ nhớ chờ quyết định, và mọi span của một trace phải tới cùng một instance Collector [12].

Với LLM, nên giữ 100% trace có lỗi, bị cắt, bị từ chối, thiếu usage, và lấy mẫu phần còn lại. Wrapper ở [§2.6](#26-tự-xây-trên-opentelemetry) đặt `app.llm.keep=true` cho các trường hợp đó ngay trong lúc request chạy, để Collector chỉ cần một policy. Feedback tiêu cực thì khác: nó đến sau, wrapper không biết, và cần cơ chế riêng (đoạn cuối mục này).

```yaml
processors:
  tail_sampling:
    decision_wait: 30s          # agent chạy lâu thì tăng lên; span đến sau quyết định không được xét lại
    num_traces: 50000           # số trace giữ trong bộ nhớ: tính từ traffic, xem đoạn dưới
    policies:
      - name: errors
        type: status_code
        status_code: {status_codes: [ERROR]}
      - name: flagged
        type: boolean_attribute
        boolean_attribute: {key: app.llm.keep, value: true}
      - name: slow
        type: latency
        latency: {threshold_ms: 30000}
      - name: baseline
        type: probabilistic
        probabilistic: {sampling_percentage: 5}
```

Các policy được kết hợp theo kiểu OR: trace khớp bất kỳ policy nào thì được giữ [12]. Chạy nhiều instance Collector thì cần một tầng load balancing exporter phía trước để định tuyến theo trace id [12].

`num_traces` giới hạn số trace, không giới hạn byte. Trace LLM có thể rất lớn khi capture nội dung: một agent run với vài chục span, mỗi span mang prompt và tool result dài. Ước lượng bộ nhớ cần bằng số trace mới mỗi giây × `decision_wait` × kích thước trung bình một trace, đo kích thước đó trên traffic thật thay vì đoán. Hai lớp chặn khác:

- `maximum_trace_size_bytes`: trace vượt ngưỡng bị bỏ ngay khỏi tail sampler để bảo vệ Collector [12]. Đánh đổi: trace bị bỏ thường là agent run dài nhất, đúng loại bạn muốn xem. Nếu đặt, theo dõi số trace bị bỏ, và cân nhắc cắt nội dung trên span (hoặc đẩy nội dung ra object storage như ở trên) để trace nhỏ lại thay vì để mất cả trace.
- Processor `memory_limiter` đặt đầu pipeline của Collector, để khi bộ nhớ gần đầy Collector từ chối dữ liệu mới thay vì bị OOM kill và mất mọi trace đang chờ; docs khuyên đặt nó là processor đầu tiên [43].

Feedback tiêu cực thường đến sau khi quyết định sampling đã xong. Nếu muốn giữ trace của mọi feedback tiêu cực, phải lưu nội dung ngắn hạn cho 100% request (ví dụ 72 giờ, ở bucket riêng), rồi khi có feedback thì chuyển bản ghi đó sang vùng lưu dài hơn.

Mẫu dùng cho chấm điểm online phải lấy **sau khi đã che PII**, vì nó sẽ được gửi tới một model judge.

## 3.2 Trace cho RAG, tool use và agent nhiều bước

**RAG.** Mỗi bước là một span con trong cùng trace: embedding câu hỏi, truy vấn vector DB, rerank, gọi model sinh câu trả lời. Trên span retrieval, ghi số tài liệu lấy về, id tài liệu và score; chỉ ghi nội dung tài liệu khi đã bật capture nội dung. Khi câu trả lời sai, câu hỏi đầu tiên là retriever có lấy đúng tài liệu không, và id tài liệu trên span trả lời được câu đó mà không cần nội dung. Phần này đã được viết chi tiết cho hệ thống embedding ở [Embeddings §3.7](../embeddings/#37-observability-cho-embedding--rag-systems) (latency theo từng bước, drift của embedding).

**Tool use.** Span `execute_tool {gen_ai.tool.name}` là con của agent run hoặc của span model đã yêu cầu nó, có `gen_ai.tool.call.id` khi provider cung cấp để khớp với tool call trong response của model [6]. Thứ tự đọc được trên trace: model quyết định (span `chat`) → tool chạy (`execute_tool`) → model đọc kết quả (`chat` tiếp). Dashboard theo tool đã được liệt kê ở [Tool Use §3.10](../tool-use/#310-observability). Metric `gen_ai.execute_tool.duration` đo thời gian tool [6].

**Agent.** Span `invoke_agent` bọc toàn bộ một lần chạy agent [5]. Ba metric đáng có alert:

- `gen_ai.invoke_agent.duration`: tổng thời gian một agent run [4].
- `gen_ai.invoke_agent.inference_calls`: số lượt gọi model trong một run [4].
- `gen_ai.invoke_agent.tool_calls`: số lượt gọi tool trong một run [4].

Phân phối của hai metric sau là cách rẻ nhất để phát hiện agent lặp: p50 là 4 lượt gọi model mà p99 lên 60 thì có những run đang chạy vòng. Đặt giới hạn cứng trong code (xem [Tool Use §3.5](../tool-use/#35-giới-hạn-vòng-lặp-timeout-và-chi-phí)) và alert khi tỉ lệ run chạm giới hạn tăng.

Những thứ hay thiếu trên trace agent:

- **Context bị đứt khi tool chạy ở process khác.** Tool gọi qua HTTP, queue hay MCP server mà không truyền trace context thì span của tool thành trace riêng. Dùng W3C Trace Context (`traceparent`) khi gọi qua mạng, và kiểm tra trên một trace thật rằng `execute_tool` có cha là `invoke_agent`.
- **Bước người duyệt.** Tool ghi cần người xác nhận thì bước chờ xác nhận nên là một span (hoặc event) có kết quả duyệt/từ chối/hết hạn. Không có nó, trace cho thấy một khoảng trống 20 phút không giải thích được.
- **Trace-level attribute trên mọi span.** Tenant, user, session phải có trên mọi span con, vì nhiều backend chỉ lọc theo attribute có trên span đó (Langfuse là một ví dụ [13]). Dùng Baggage hoặc gắn tay ở wrapper.
- **`gen_ai.conversation.id`**: chỉ đặt khi bạn có id hội thoại thật. Spec không cho dùng trace id hay UUID tự sinh mỗi lượt để thay [5]. Có id này, nối được các lượt của một hội thoại dài, xem được cache hit rate theo lượt (xem [Multi-turn §3.11](../multi-turn-conversation/#311-observability-cho-cache)).

## 3.3 Theo dõi chi phí theo user, feature, tenant

Hoá đơn của provider cho biết tổng tiền theo ngày và theo model. Nó không cho biết tính năng nào, khách hàng nào tiêu số tiền đó. Muốn biết, phải tính chi phí cho từng lượt gọi và gắn với các chiều kinh doanh.

**Tính chi phí ở đâu.** Có hai cách, nên làm cả hai:

- **Lúc ghi span**, như wrapper ở [§2.6](#26-tự-xây-trên-opentelemetry): chi phí là một attribute, cộng dồn được theo bất kỳ chiều nào có trên span.
- **Lúc query**, từ token đã lưu và bảng giá: tính lại được khi phát hiện bảng giá sai, hoặc khi provider đổi giá.

**Bảng giá theo model thực trả về.** Dùng `gen_ai.response.model`, không dùng model bạn yêu cầu. Gọi alias `gpt-6-astra` có thể nhận về một snapshot có tên dài hơn; gọi qua gateway có fallback thì model trả lời có thể là model khác hẳn. Bảng giá cần:

- Ngày hiệu lực cho mỗi dòng giá, để tính lại chi phí quá khứ đúng giá lúc đó.
- Đủ loại token: input không cache, cache read, cache write (có khi nhiều mức theo TTL), output. Với OpenAI, cache write có hai mức TTL, 30 phút tính 1,25× và 12 giờ tính 2,5× giá input; `input_tokens_details.cache_write_tokens_details` tách hai phần này [32]. Wrapper ở trên dùng một giá cache write duy nhất cho gọn; nếu dùng cả hai mức TTL, tách riêng.
- Các khoản không theo token: batch API thường rẻ hơn, tool chạy phía server (web search, code execution) tính theo lượt, input dài có thể có giá khác. Kiểm tra trang giá của từng provider.

**Không cộng reasoning hai lần.** Theo quy ước OTel và cả ba provider ở bảng [§1.3](#13-metric-riêng-của-llm-token-ttft-tokensgiây-cache-hit-tỉ-lệ-refusal) sau khi chuẩn hoá, reasoning là tập con của output. `output_tokens × giá output` đã gồm phần suy luận.

**Gắn chiều kinh doanh.** Tối thiểu: tenant (hoặc account), feature, môi trường. Thêm user (đã hash) nếu cần phát hiện lạm dụng. Đặt các chiều này lên span. Trên metric, chỉ đặt chiều có ít giá trị (feature, gói dịch vụ), như đã giải thích ở [§2.6](#26-tự-xây-trên-opentelemetry). Chi phí theo tenant query từ span store (ClickHouse, BigQuery, backend trace có query tổng hợp), hoặc từ một bảng sự kiện riêng mỗi lượt gọi ghi một dòng.

**Đối soát với hoá đơn.** Mỗi tháng so tổng chi phí tự tính với hoá đơn của provider. Lệch vài phần trăm là bình thường (làm tròn, request ngoài ứng dụng). Lệch nhiều thường do:

- Bỏ sót cache write, hoặc ghi `input_tokens` của Claude mà không cộng phần cache.
- Bỏ sót thinking của Gemini (`thoughtsTokenCount` không nằm trong `candidatesTokenCount`).
- Stream không có usage, nên những lượt đó không có chi phí. Tỉ lệ `app.llm.usage_missing` cho biết phần thiếu lớn tới đâu.
- Có code path gọi model không qua wrapper (script batch, job nền, eval).
- Bảng giá lỗi thời.

## 3.4 Thu feedback người dùng và nối với trace

Feedback chỉ có ích khi tìm lại được trace mà nó nói tới. Thiết kế:

1. Mỗi response trả về client mang theo một id: trace id, hoặc id nội bộ map tới trace id và `gen_ai.response.id`.
2. Khi người dùng bấm thumbs down, gửi lên id đó kèm lý do (nếu có chọn lý do) và nguồn feedback.
3. Lưu feedback như một score gắn với trace. Langfuse có score gắn vào trace hoặc observation; LangSmith có `create_feedback()` gắn được vào trace hoặc bất kỳ run con nào [40]. Nếu tự xây, một bảng `(trace_id, response_id, source, value, reason, created_at)`.
4. Feedback tiêu cực đi vào hàng đợi đọc trace. Trace đó chỉ còn để đọc nếu bạn đã lưu ngắn hạn 100% request rồi chuyển bản ghi sang vùng lưu dài khi có feedback (xem cuối [§3.1](#31-log-gì-lưu-bao-lâu-che-pii-thế-nào)); tail sampling đã quyết định xong từ trước khi người dùng bấm.

**Phân biệt nguồn feedback.**

| Loại | Ví dụ | Độ tin cậy |
|---|---|---|
| Explicit | Thumbs up/down, chấm sao, chọn lý do | Rõ ràng nhưng ít người bấm, và người bấm thường là người rất hài lòng hoặc rất bực |
| Implicit | Tạo lại, sửa output, sao chép, hỏi lại cùng ý, bỏ dở, chuyển cho nhân viên | Có trên mọi lượt, nhưng mỗi tín hiệu có nhiều cách hiểu: sao chép có thể là hài lòng, cũng có thể là sao chép để sửa ở chỗ khác |
| Từ nhân viên | Nhân viên hỗ trợ đánh dấu câu trả lời sai khi xử lý ticket | Ít nhưng chất lượng cao, gần với nhãn |

**Đừng tối ưu thẳng theo thumbs.** OpenAI đã thêm reward signal từ thumbs up/down vào bản cập nhật GPT-4o tháng 4/2025, và viết rằng feedback người dùng "can sometimes favor more agreeable responses", góp phần vào việc model trở nên nịnh hót [28]. Nếu bạn dùng tỉ lệ thumbs up để chọn giữa hai prompt, cùng cơ chế đó có thể đẩy bạn tới prompt chiều người dùng hơn là prompt đúng hơn. Dùng feedback để tìm trace cần đọc và để làm ca eval, quyết định cuối dựa trên eval có nhãn. Chi tiết hơn về giới hạn của feedback ở [Evaluation §3.5](../evaluation/#35-eval-online-ab-test-feedback-người-dùng-sampling-log).

## 3.5 Phát hiện drift khi provider đổi model

"Drift" ở đây có ba nghĩa, cần cách phát hiện khác nhau.

**1. Model đổi phiên bản mà bạn không đổi code.** Khi bạn gọi bằng alias, provider có thể trỏ alias sang snapshot mới. Phát hiện:

- Ghi `gen_ai.response.model` trên mọi span. Alert khi xuất hiện một giá trị chưa từng thấy trong 30 ngày qua.
- Pin snapshot cho tính năng quan trọng, và đổi snapshot như một lần deploy có eval (xem [Evaluation §3.3](../evaluation/#33-regression-test-khi-đổi-prompt-hoặc-model)).

**2. Hành vi đổi mà tên không đổi.** Đây là trường hợp của Chen et al. [27] và của các sự cố hạ tầng phía provider [29]: cùng tên model, chất lượng khác. Field nào trong response cũng không thấy được. Cách phát hiện là **canary**: một bộ câu hỏi cố định có đáp án kiểm tra được bằng code, chạy theo lịch (mỗi giờ hoặc mỗi ngày) với cùng tham số, ghi điểm theo thời gian.

```python
CANARIES = [
    {"id": "refund-days", "prompt": "Theo chính sách đính kèm, hoàn tiền trong bao nhiêu ngày? ...",
     "check": lambda out: "14" in out},
    {"id": "json-shape", "prompt": "Trích xuất thành JSON theo schema ...",
     "check": lambda out: valid_invoice_json(out)},      # hàm validate schema của bạn
    # ... phủ các tác vụ chính của sản phẩm; số ca và số lần chạy quyết định độ nhạy, xem dưới
]

def run_canaries(call_model, model: str, runs: int) -> dict:
    if type(runs) is not int or runs < 1:
        raise ValueError(f"runs phải là số nguyên dương, nhận {runs!r}")
    results = {}
    for c in CANARIES:
        passed = sum(c["check"](call_model(model, c["prompt"])) for _ in range(runs))
        results[c["id"]] = passed / runs
    return results          # ghi thành metric app.canary.pass_rate{model, canary_id}
```

Alert khi pass rate của bộ canary giảm vượt dao động bình thường. Dao động bình thường phải đo trước: chạy bộ canary vài chục lần trên cùng model trong điều kiện ổn định, xem pass rate dao động bao nhiêu. Từ đó chọn số ca và `runs` theo mức giảm bạn cần bắt: ví dụ 50 ca, mỗi ca 3 lần, cho khoảng tin cậy rất rộng (xem cách tính ở [Evaluation §1.4](../evaluation/#14-phương-sai-chạy-bao-nhiêu-lần-cần-bao-nhiêu-mẫu)); cấu hình đó bắt được thay đổi lớn, không bắt được giảm 2%. Muốn nhạy hơn thì tăng mẫu, và trả thêm tiền cho mỗi lần chạy.

Chạy canary ở nhiều đường đi nếu có: nhiều region, nhiều nền tảng (API trực tiếp, Bedrock, Vertex). Sự cố của Anthropic năm 2025 ảnh hưởng các nền tảng khác nhau ở mức khác nhau [29].

**3. Dữ liệu vào đổi.** Người dùng bắt đầu hỏi về sản phẩm mới, ngôn ngữ khác, kiểu câu hỏi khác. Chất lượng giảm vì bộ prompt và tài liệu không còn khớp, không phải vì model đổi. Theo dõi phân phối độ dài input, ngôn ngữ, chủ đề (phân cụm embedding của câu hỏi), tỉ lệ retriever không tìm được tài liệu nào đủ điểm. Kỹ thuật cho embedding drift ở [Embeddings §3.7](../embeddings/#37-observability-cho-embedding--rag-systems).

**Cắt metric đủ nhỏ.** Bài học từ sự cố Claude Code tháng 3–4/2026 của Anthropic: ba thay đổi khác nhau ảnh hưởng ba nhóm traffic khác nhau theo ba lịch khác nhau, và "the aggregate effect looked like broad, inconsistent degradation" [30]. Một lỗi chỉ xảy ra với phiên cũ (stale session) đã lọt qua code review, unit test, end-to-end test và dogfooding [30]. Metric tổng hợp không đủ. Cần cắt theo phiên bản ứng dụng, phiên bản prompt, model snapshot, loại phiên (mới/cũ, ngắn/dài) để thấy nhóm nào đang tệ đi.

## 3.6 Từ log production về eval dataset

Eval dataset tốt nhất được xây từ những gì người dùng thật hỏi và những chỗ hệ thống thật sai. Observability là nguồn cấp cho việc đó.

Quy trình:

1. **Chọn trace.** Feedback tiêu cực, trace bị cắt, refusal, tool error, câu hỏi mà retriever không tìm thấy tài liệu, trace được judge online chấm thấp, và một mẫu ngẫu nhiên để không chỉ có ca khó.
2. **Đọc và gán nhãn.** Người có hiểu biết về nghiệp vụ đọc trace, ghi lỗi là gì (phân loại lỗi), câu trả lời đúng phải thế nào. Langfuse và LangSmith đều có annotation queue cho bước này: hàng đợi trace để người duyệt đọc và gắn nhãn trên giao diện [39][41].
3. **Che và chuyển thành ca eval.** Bỏ thông tin cá nhân, giữ lại đúng phần cần để tái hiện: input, context (tài liệu, lịch sử), tiêu chí chấm. Ghi nguồn (trace id, ngày) để truy lại.
4. **Thêm vào bộ regression.** Lỗi đã sửa thành ca phải pass mãi về sau.

Để làm được bước 3, trace phải chứa đủ để phát lại: prompt version, tài liệu retriever trả về (hoặc id của chúng ở đúng phiên bản index), tham số model. Thiếu một thứ, ca eval không tái hiện được lỗi.

Cách tổ chức golden dataset, gán nhãn, đồng thuận giữa người gán nhãn ở [Evaluation §3.1](../evaluation/#31-xây-golden-dataset-từ-đâu) và [§3.2](../evaluation/#32-gán-nhãn-và-đồng-thuận-giữa-người-gán-nhãn).

## 3.7 Alert: ngưỡng nào đáng đánh thức người trực

Các con số trong bảng dưới chỉ để minh hoạ dạng của một luật alert, không lấy từ nghiên cứu hay chuẩn nào. Ngưỡng thật đặt theo baseline của hệ thống bạn: lấy vài tuần số liệu, xem phân phối của từng chỉ số theo giờ trong ngày, chọn ngưỡng sao cho số lần page sai mỗi tháng nằm trong mức đội trực chịu được, rồi chỉnh sau mỗi lần alert sai hoặc sự cố không được alert.

**Đánh thức người trực (page).** Ảnh hưởng người dùng ngay, cần người xử lý:

| Điều kiện | Ví dụ ngưỡng | Hành động thường gặp |
|---|---|---|
| Tỉ lệ lỗi gọi model theo provider | > 5% trong 5 phút | Chuyển provider dự phòng, kiểm tra status page |
| TTFT p95 | > 3× baseline trong 10 phút | Kiểm tra provider, kiểm tra prompt có phình không |
| Tỉ lệ stream bị cắt | > 2× baseline trong 10 phút | Proxy, deploy, provider quá tải ([Streaming §3.11](../streaming/#311-failure-modes-thường-gặp)) |
| Chi phí theo giờ | > 3× cùng giờ tuần trước | Vòng lặp agent, lạm dụng, cache vỡ |

**Ticket hoặc kênh chat (không page).** Cần xem trong ngày:

- `gen_ai.response.model` có giá trị mới.
- Tỉ lệ bị cắt (cùng tập giá trị như ở [§1.3](#13-metric-riêng-của-llm-token-ttft-tokensgiây-cache-hit-tỉ-lệ-refusal): `length`, `max_output_tokens`, `max_tokens`, `MAX_TOKENS`) tăng sau deploy hoặc đổi model.
- Tỉ lệ refusal tăng.
- Cache hit rate giảm mạnh sau deploy, thường do prefix bị đổi (xem [Multi-turn §3.3](../multi-turn-conversation/#33-những-thứ-âm-thầm-làm-vỡ-cache)).
- Tool error rate hoặc tỉ lệ structured output không hợp lệ tăng.
- p99 của `gen_ai.invoke_agent.inference_calls` tăng.
- Pass rate của canary hoặc điểm judge online giảm.
- Tỉ lệ stream thiếu usage tăng.

**Alert cho chính hệ thống observability.** Pipeline trace hỏng thì mọi thứ khác im lặng, và im lặng trông giống như ổn:

- Số span nhận được giảm mạnh so với số request HTTP.
- Exporter báo lỗi hoặc rớt span (queue đầy).
- Backend từ chối một phần span: OTLP có `partial_success` trong response, Braintrust ghi rõ có thể trả HTTP 200 nhưng từ chối một số span qua `partialSuccess.rejectedSpans` [24].
- Dữ liệu đến trễ: Langfuse không có header version, LangSmith bỏ span con khi thiếu span cha (xem [Phần 2](#phần-2--công-cụ)).

Alert về chất lượng (canary, judge) hiếm khi đáng page lúc 3 giờ sáng: người trực thường không sửa được gì ngay. Ngoại lệ là khi có hành động khẩn cấp rõ ràng, ví dụ rollback prompt vừa deploy, hoặc pin lại snapshot model cũ.

## 3.8 Failure modes thường gặp

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| Có trace nhưng không có nội dung prompt | Capture nội dung tắt (mặc định), hoặc đặt sai tên biến môi trường cho thư viện đang dùng | Đối chiếu README của instrumentation; spec và thư viện Python dùng tên biến khác nhau ([§1.5](#15-opentelemetry-genai-semantic-conventions)) |
| Instrumentation không ghi attribute mới | Thiếu `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`, thư viện chạy theo semconv cũ | Bật opt-in, kiểm tra backend có nhận tên mới không |
| Cache hit rate trên 100% | Ghi `input_tokens` của Claude (chỉ phần sau breakpoint) làm tổng input | Chuẩn hoá: cộng cache read và cache write |
| Chi phí thấp hơn hoá đơn | Thiếu cache write, thiếu thinking của Gemini, stream không có usage, code path không qua wrapper | Đối soát hằng tháng, theo dõi `app.llm.usage_missing` ([§3.3](#33-theo-dõi-chi-phí-theo-user-feature-tenant)) |
| Span gọi model ngắn bất thường khi stream | Span đóng khi hàm trả về iterator | Đóng span khi stream kết thúc; test với stream lỗi giữa chừng |
| Stream không có usage | OpenAI Chat Completions thiếu `stream_options.include_usage`; stream bị ngắt trước chunk cuối [36] | Bật tuỳ chọn; đánh dấu thiếu usage, không ghi 0 |
| Refusal bị đếm là lỗi, hoặc bị bỏ qua | Claude trả refusal với HTTP 200 [35] | Đọc `stop_reason`, ghi vào `finish_reasons`, không đặt `error.type` |
| Span con biến mất trong LangSmith | Span cha không được export tới LangSmith [17] | Đảm bảo span gốc cũng đi tới LangSmith |
| Trace Langfuse hiện sau 10 phút | Thiếu header `x-langfuse-ingestion-version=4` [13] | Thêm header |
| Mất cả batch span ở Langfuse | Hàm `mask_otel_spans` ném lỗi [15] | Test hàm mask; xử lý mọi kiểu dữ liệu attribute |
| Mất span khi process thoát | Exporter gửi theo batch nền, chưa flush | Gọi `shutdown()`/`force_flush()` của provider trước khi thoát, nhất là với job ngắn và serverless |
| Backend metric quá tải | Đặt tenant, user, trace id lên metric | Chiều có nhiều giá trị chỉ để trên span |
| Trace agent đứt thành nhiều mảnh | Không truyền trace context khi gọi tool qua mạng | `traceparent` khi gọi HTTP/queue/MCP; kiểm tra quan hệ cha con trên trace thật |
| Tail sampling giữ trace thiếu span | Span đến sau `decision_wait`, hoặc span của một trace đi tới nhiều Collector [12] | Tăng `decision_wait` cho agent chạy lâu; load balancing theo trace id |
| Chi phí Datadog tăng theo agent | Mỗi lượt gọi model là một span tính phí [22] | Ước tính số lượt gọi mỗi request trước khi bật cho toàn bộ traffic |
| Nội dung nhạy cảm nằm trong backend trace | Bật capture ở production, tool result chứa dữ liệu khách | Tách nội dung ra `*_ref` có phân quyền; che trước khi export |
| Dashboard xanh nhưng người dùng phàn nàn | Metric tổng hợp che lỗi của một nhóm; không có tín hiệu chất lượng | Cắt theo model snapshot, prompt version, loại phiên; canary; judge online [28][29][30] |

## 3.9 Checklist

**Instrumentation**

- [ ] Mọi lượt gọi model đi qua một wrapper hoặc instrumentation, kể cả job nền, script batch, eval.
- [ ] Span có `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `server.address` ngay lúc tạo.
- [ ] Ghi `gen_ai.response.model`, `gen_ai.response.id`, `gen_ai.response.finish_reasons`, `error.type`.
- [ ] Usage được chuẩn hoá theo quy ước OTel (input gồm cache, output gồm reasoning) cho từng provider.
- [ ] Span stream kết thúc khi stream kết thúc; có TTFT; test với stream bị ngắt.
- [ ] Phiên bản instrumentation được pin; biến môi trường đã đối chiếu với README của đúng phiên bản.
- [ ] Trace context được truyền qua mọi lời gọi tool ngoài process.

**Dữ liệu và quyền riêng tư**

- [ ] Metadata và nội dung có chính sách lưu khác nhau.
- [ ] Nội dung mặc định tắt ở production, hoặc đi ra storage riêng qua `*_ref` với phân quyền và TTL.
- [ ] Che PII trong process trước khi export; hàm che có test.
- [ ] Mẫu cho judge online lấy sau khi che.

**Chi phí**

- [ ] Bảng giá có ngày hiệu lực, tra theo `gen_ai.response.model`.
- [ ] Model không có giá được đánh dấu, không ghi 0.
- [ ] Chi phí cắt được theo tenant và feature từ span store.
- [ ] Đối soát với hoá đơn hằng tháng.

**Chất lượng và drift**

- [ ] Mỗi response hiển thị cho người dùng có id tra được trace.
- [ ] Feedback gắn với trace id, phân biệt explicit và implicit.
- [ ] Canary chạy theo lịch, có baseline dao động.
- [ ] Alert khi `gen_ai.response.model` có giá trị mới.
- [ ] Trace lỗi, bị cắt, refusal được giữ 100%; trace có feedback tiêu cực được giữ qua lưu ngắn hạn toàn bộ request rồi chuyển sang lưu dài.
- [ ] Có quy trình đưa trace production vào eval dataset.

**Vận hành**

- [ ] Phân biệt alert page và alert ticket.
- [ ] Có alert cho chính pipeline observability.
- [ ] Exporter flush trước khi process thoát.

---

## Nguồn tham khảo

Đọc ngày 06/11/2026.

| # | Nguồn |
|---|---|
| [1] | OpenTelemetry. [semantic-conventions-genai: GenAI spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) (content capture, upload, provider names) |
| [2] | OpenTelemetry. [Client inference](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md) (span `gen_ai.client.inference`, attribute, metric inference) |
| [3] | OpenTelemetry. [Inference token metrics](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-token-metrics.md) |
| [4] | OpenTelemetry. [GenAI metrics](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md) (server, agent metrics) |
| [5] | OpenTelemetry. [GenAI agent spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md) |
| [6] | OpenTelemetry. [Execute tool](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/execute-tool.md) |
| [7] | OpenTelemetry. [GenAI events](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md) (`gen_ai.evaluation.result`) |
| [8] | PyPI. [opentelemetry-instrumentation-openai-v2](https://pypi.org/project/opentelemetry-instrumentation-openai-v2/) 2.4b0 |
| [9] | PyPI. [opentelemetry-util-genai](https://pypi.org/project/opentelemetry-util-genai/) 1.3b0 |
| [10] | Arize. [OpenInference semantic conventions](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md) |
| [11] | OpenTelemetry. [Sampling](https://opentelemetry.io/docs/concepts/sampling/) |
| [12] | OpenTelemetry Collector Contrib. [Tail sampling processor](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/processor/tailsamplingprocessor/README.md) |
| [13] | Langfuse. [OpenTelemetry: get started](https://langfuse.com/docs/opentelemetry/get-started) |
| [14] | Langfuse. [Self-hosting](https://langfuse.com/self-hosting) |
| [15] | Langfuse. [Masking](https://langfuse.com/docs/observability/features/masking) |
| [16] | Langfuse. [Pricing](https://langfuse.com/pricing) |
| [17] | LangChain. [Trace with OpenTelemetry (LangSmith)](https://docs.langchain.com/langsmith/trace-with-opentelemetry) |
| [18] | LangChain. [Pricing](https://www.langchain.com/pricing) |
| [19] | Arize. [Phoenix (GitHub, README và license)](https://github.com/Arize-ai/phoenix) |
| [20] | Helicone. [Self-host](https://docs.helicone.ai/getting-started/self-host) |
| [21] | Helicone. [Pricing](https://www.helicone.ai/pricing) |
| [22] | Datadog. [LLM Observability](https://docs.datadoghq.com/llm_observability/) |
| [23] | Datadog. [Native support for OpenTelemetry GenAI Semantic Conventions](https://www.datadoghq.com/blog/llm-otel-semantic-convention) |
| [24] | Braintrust. [Traces](https://docs.braintrust.dev/guides/traces) |
| [25] | Braintrust. [Pricing](https://www.braintrust.dev/pricing) |
| [26] | Pydantic. [Logfire pricing](https://pydantic.dev/pricing) |
| [27] | Chen, L., Zaharia, M., Zou, J. (2023). [How is ChatGPT's behavior changing over time?](https://arxiv.org/abs/2307.09009) |
| [28] | OpenAI (02/05/2025). [Expanding on what we missed with sycophancy](https://openai.com/index/expanding-on-sycophancy/) |
| [29] | Anthropic (09/2025). [A postmortem of three recent issues](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues) |
| [30] | Anthropic (23/04/2026). [An update on recent Claude Code quality reports](https://www.anthropic.com/engineering/april-23-postmortem) |
| [31] | Anthropic. [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) (các field usage) |
| [32] | OpenAI. [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) (cache hit rate, cache write theo TTL) |
| [33] | OpenAI. [Python SDK: `ResponseUsage`](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_usage.py) |
| [34] | Google. [Gemini API: generateContent, `UsageMetadata`](https://ai.google.dev/api/generate-content#UsageMetadata) |
| [35] | Anthropic. [Handling stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons) |
| [36] | OpenAI. [Chat Completions streaming events](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events) (`stream_options.include_usage`) |
| [37] | Weights & Biases. [Weave](https://wandb.ai/site/weave) |
| [38] | Traceloop. [OpenLLMetry](https://www.traceloop.com/openllmetry) |
| [39] | Langfuse. [Annotation queues](https://langfuse.com/docs/evaluation/evaluation-methods/annotation-queues) |
| [40] | LangChain. [Log user feedback using the SDK (LangSmith)](https://docs.langchain.com/langsmith/attach-user-feedback) |
| [41] | LangChain. [Use annotation queues (LangSmith)](https://docs.langchain.com/langsmith/annotation-queues) |
| [42] | OpenAI. [Python SDK: `Response`](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response.py) (`status`, `incomplete_details.reason`) |
| [43] | OpenTelemetry Collector. [Memory limiter processor](https://github.com/open-telemetry/opentelemetry-collector/blob/main/processor/memorylimiterprocessor/README.md) |
