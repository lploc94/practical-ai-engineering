# Reasoning models: khi nào cho model suy nghĩ lâu hơn

Một đội làm trợ lý hỗ trợ khách hàng nâng model lên bản reasoning và đặt effort `high` cho mọi request. Hệ thống có bốn loại việc: phân loại ticket, trích xuất mã đơn hàng, trả lời câu hỏi chính sách và một agent tra log để tìm nguyên nhân lỗi thanh toán. Sau một tuần, hoá đơn tăng gần ba lần, p95 latency từ vài giây lên vài chục giây. Phần agent tra log tốt lên rõ. Phân loại và trích xuất thì không đổi, có loại còn tệ đi vì model "nghĩ thêm" ra những ngoại lệ không có trong quy tắc.

Kèm theo đó là các lỗi không có trong quickstart:

- Một số request trả về rỗng với `status: "incomplete"`. Toàn bộ `max_output_tokens` đã bị phần suy luận dùng hết trước khi model viết được chữ nào.
- Agent báo lỗi 400 ở lượt thứ hai của tool loop. Code lưu lịch sử chỉ giữ phần text và bỏ thinking block cùng chữ ký đi kèm.
- Tỉ lệ cache hit giảm mạnh sau khi đội thử đổi effort theo từng request trong cùng một cuộc hội thoại.
- Log ghi lại "lời giải thích" của model và đội dùng nó để giải thích quyết định cho khách hàng. Thứ đó là bản tóm tắt do một model khác viết, và có thể không phản ánh đúng những gì model đã thực sự làm.

Reasoning model trả thêm token suy luận để đổi lấy chất lượng ở *một số* tác vụ. Câu hỏi chính của tài liệu là tác vụ nào đáng trả khoản đó, trả bao nhiêu, và cần sửa code thế nào để không làm hỏng trạng thái suy luận mà provider trả về.

- **Nền tảng — Phần 1:** reasoning model khác model thường ở đâu, thinking token được tính phí và hiển thị thế nào, tác vụ nào hưởng lợi theo nghiên cứu, ba kiểu điều khiển (effort, budget, adaptive), reasoning trong hội thoại nhiều lượt.
- **Tra cứu theo provider — Phần 2:** OpenAI, Claude, Gemini, Grok, DeepSeek, Qwen, gpt-oss, và cách tách reasoning khi tự host bằng vLLM hoặc SGLang.
- **Thiết kế và vận hành — Phần 3:** chọn effort, viết prompt, kết hợp với tool use, structured output và streaming, ngân sách token, eval, và [bảng failure mode ở §3.7](#37-failure-modes-thường-gặp).

> Bản tháng 11/2026. Tên tham số, field và model giữ nguyên tiếng Anh để tiện đối chiếu API. Những điểm chỉ thấy ở nguồn thứ cấp hoặc chưa đối chiếu được với tài liệu chính thức được đánh dấu tại chỗ. Tên model và giá trị effort hợp lệ thay đổi theo từng đợt phát hành; kiểm tra lại trang docs của model bạn dùng trước khi copy cấu hình.

Tài liệu liên quan: [Multi-turn §3.7](../multi-turn-conversation/#37-reasoning-và-thinking-blocks) (gửi lại reasoning qua các lượt và ảnh hưởng tới cache), [Tool Use §1.6](../tool-use/#16-reasoning-model-trạng-thái-ẩn-đi-kèm-tool-call) (trạng thái suy luận đi kèm tool call), [Streaming §3.5](../streaming/#35-stream-kèm-reasoningthinking) (event của reasoning khi stream), [Prompt Engineering §2.4](../prompt-engineering/#24-prompt-chaining-chain-of-thought-và-tách-tác-vụ) (chain-of-thought trong prompt), [Model Routing](../model-routing/), [Cost Optimization](../cost-optimization/), [Latency](../latency/).

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Reasoning model khác model thường ở đâu](#11-reasoning-model-khác-model-thường-ở-đâu)
  - [1.2 Thinking token: sinh ra, tính phí, nhưng không phải lúc nào cũng thấy](#12-thinking-token-sinh-ra-tính-phí-nhưng-không-phải-lúc-nào-cũng-thấy)
  - [1.3 Tác vụ nào hưởng lợi, tác vụ nào không](#13-tác-vụ-nào-hưởng-lợi-tác-vụ-nào-không)
  - [1.4 Effort, budget và adaptive thinking](#14-effort-budget-và-adaptive-thinking)
  - [1.5 Reasoning trong hội thoại nhiều lượt](#15-reasoning-trong-hội-thoại-nhiều-lượt)
- [Phần 2 — Reasoning theo provider](#phần-2--reasoning-theo-provider)
  - [2.1 Bức tranh chung](#21-bức-tranh-chung)
  - [2.2 OpenAI](#22-openai)
  - [2.3 Claude (Anthropic)](#23-claude-anthropic)
  - [2.4 Google Gemini](#24-google-gemini)
  - [2.5 xAI Grok](#25-xai-grok)
  - [2.6 DeepSeek, Qwen và các model open-weight](#26-deepseek-qwen-và-các-model-open-weight)
  - [2.7 Self-hosted: tách reasoning content trong vLLM, SGLang](#27-self-hosted-tách-reasoning-content-trong-vllm-sglang)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Chọn mức effort theo tác vụ](#31-chọn-mức-effort-theo-tác-vụ)
  - [3.2 Prompt cho reasoning model: viết ít hơn hay nhiều hơn](#32-prompt-cho-reasoning-model-viết-ít-hơn-hay-nhiều-hơn)
  - [3.3 Reasoning kết hợp tool use](#33-reasoning-kết-hợp-tool-use)
  - [3.4 Reasoning kết hợp structured output và streaming](#34-reasoning-kết-hợp-structured-output-và-streaming)
  - [3.5 Chi phí và độ trễ: max_tokens, timeout, ước lượng ngân sách](#35-chi-phí-và-độ-trễ-max_tokens-timeout-ước-lượng-ngân-sách)
  - [3.6 Eval: đo xem reasoning có thật sự giúp](#36-eval-đo-xem-reasoning-có-thật-sự-giúp)
  - [3.7 Failure modes thường gặp](#37-failure-modes-thường-gặp)
  - [3.8 Checklist](#38-checklist)
- [Nguồn tham khảo](#nguồn-tham-khảo)

---

## Tra cứu nhanh

### Tham số reasoning theo provider (snapshot 11/2026)

| Provider | Tham số độ sâu | Tắt được không | Nội dung suy luận trả về | Đếm token suy luận | Phải gửi lại |
|---|---|---|---|---|---|
| OpenAI (Responses) | `reasoning.effort`: `none` → `max`, tập giá trị tuỳ model; `reasoning.mode`: `standard` / `pro` | Chỉ ở model nhận `none` | Không có bản gốc. `reasoning.summary` cho bản tóm tắt; `encrypted_content` khi stateless | `usage.output_tokens_details.reasoning_tokens` | Reasoning item, function call và output kể từ user message cuối |
| OpenAI (Chat Completions) | `reasoning_effort` | Như trên | Không | `usage.completion_tokens_details.reasoning_tokens` | Không có cơ chế; reasoning không vào context lượt sau |
| Claude 5.x | `output_config.effort`: `low` → `max` | Opus 5.5, Fable 5.1, Mythos 5.1 không tắt được; Sonnet 5.5 chỉ tắt phần nghĩ trước (`between_tools`); Haiku 5.5 tắt được ở effort `high` trở xuống | `display: "summarized"` (tóm tắt) hoặc `"omitted"` (rỗng, mặc định) | `usage.output_tokens_details.thinking_tokens` | Mọi `thinking` và `redacted_thinking` block, nguyên văn |
| Claude Opus 4.6–4.8, Sonnet 4.6 | `thinking: {type: "adaptive"}` + `output_config.effort` | Mặc định tắt | Như trên | Như trên | Như trên |
| Gemini 3.x, 4 | `thinking_level` (`minimal` / `low` / `medium` / `high`, tuỳ model) | Không với Gemini 3 và 2.5 Pro; `minimal` không đảm bảo tắt | `includeThoughts` (generateContent) hoặc `thinking_summaries` (Interactions) cho bản tóm tắt | `usageMetadata.thoughtsTokenCount` / `usage.total_thought_tokens` | Thought signature, nguyên văn |
| Gemini 2.5 | `thinking_budget` (số token) | Có trên Flash, Flash-Lite | Như trên | Như trên | Như trên |
| xAI Grok 4.5–4.7 | `reasoning_effort`: `low` / `medium` / `high` / `xhigh` | Không | `reasoning_content`; `encrypted_content` | `reasoning_tokens` | `reasoning.encrypted_content` |
| DeepSeek | `thinking` bật/tắt, `reasoning_effort` | Có | Toàn văn trong `reasoning_content` | Chưa xác minh | `reasoning_content` mọi lượt khi request có `tools` |
| Qwen (DashScope) | `enable_thinking`, `thinking_budget` | Model hybrid: có; model thinking-only: không | Toàn văn trong `reasoning_content` | `reasoning_tokens` | Theo `preserve_thinking` |
| gpt-oss, Qwen3 tự host | `Reasoning: low/medium/high` trong system (gpt-oss); `enable_thinking` qua chat template | Tuỳ model | Toàn văn (kênh `analysis`, `<think>`) | Tuỳ runtime | Tuỳ chat template |

Nguồn và chi tiết ở Phần 2.

### Khi nào bật, khi nào giảm

| Tình huống | Gợi ý | Lý do |
|---|---|---|
| Debug nhiều file, chứng minh, toán nhiều bước, agent tool loop dài | Bật, bắt đầu ở mức mặc định của provider | Đây là nhóm tác vụ provider và nghiên cứu cùng ghi nhận có lợi ([§1.3](#13-tác-vụ-nào-hưởng-lợi-tác-vụ-nào-không)) |
| Phân loại, trích xuất, định dạng lại, trả lời FAQ ngắn | Effort thấp nhất model cho phép, hoặc route sang model không reasoning | Overthinking: tốn token mà không tăng độ chính xác |
| Chat cần phản hồi nhanh | `low` hoặc `medium`, stream kèm trạng thái "đang suy nghĩ" | Thời gian suy luận nằm trọn trong TTFT |
| Tác vụ có nhiều thông tin gây nhiễu, đếm, quy tắc cứng | Đo nhiều mức effort trước khi tăng | Có kết quả inverse scaling: nghĩ lâu hơn làm kết quả tệ đi |
| Job nền, batch | Bật nếu eval cho thấy lợi; dùng batch API khi suy luận dài | Không có người chờ, chỉ còn bài toán chi phí |

### Ba điều phải làm đúng ngay từ đầu

1. **`max_tokens` / `max_output_tokens` tính cả token suy luận.** Đặt theo mức cho model không reasoning sẽ gặp response rỗng hoặc bị cắt. OpenAI khuyên để ít nhất 25.000 token khi mới bắt đầu [1].
2. **Lưu và gửi lại nguyên văn response của provider**, không dựng lại từ text đã hiển thị: reasoning item, thinking block kèm `signature`, thought signature, `reasoning_content`.
3. **Không đổi effort hoặc cấu hình thinking ở cấp request giữa các lượt** của một hội thoại cần cache. Dùng cơ chế đổi giữa chừng mà provider hỗ trợ (`configuration_update` của OpenAI, per-message effort của Claude).

---

# Phần 1 — Nền tảng

## 1.1 Reasoning model khác model thường ở đâu

Model thường trả lời trong một lượt sinh: token đầu tiên của output đã là chữ của câu trả lời. Reasoning model sinh thêm một chuỗi token suy luận trước, rồi mới viết câu trả lời. Chuỗi này được dùng làm nháp: thử một hướng, kiểm tra, bỏ và thử hướng khác. Tài liệu Claude mô tả lý do bằng một ví dụ: với một chứng minh, một bug khó hoặc một tác vụ agent dài, cách tiếp cận đầu tiên thường không phải cách tốt nhất [5].

Về mặt kỹ thuật, có ba khác biệt ảnh hưởng trực tiếp đến code:

**Token suy luận là output token.** Chúng được tính tiền như output, chiếm chỗ trong context window và trong giới hạn `max_tokens` của request [1][5][11]. Câu trả lời 200 token có thể đi kèm vài nghìn token suy luận.

**Model học cách suy luận, không chỉ được nhắc suy luận.** DeepSeek-R1 là ví dụ công khai rõ nhất: nhóm tác giả dùng reinforcement learning để model tự hình thành chuỗi suy luận dài, tự kiểm tra và đổi hướng, thay vì chỉ huấn luyện trên dữ liệu chain-of-thought viết sẵn [27]. Vì vậy, việc thêm "hãy suy nghĩ từng bước" vào prompt của model thường không cho ra cùng hành vi. Với reasoning model, câu đó thường thừa (xem [§3.2](#32-prompt-cho-reasoning-model-viết-ít-hơn-hay-nhiều-hơn)).

**Có một trạng thái ẩn đi kèm response.** Provider đóng thường không trả nguyên chuỗi suy luận. Họ trả bản tóm tắt, một chuỗi rỗng, hoặc một blob mã hoá (`encrypted_content`, `signature`, thought signature). Blob này là thứ để model tiếp tục mạch suy luận ở lượt sau. Code bỏ nó đi sẽ gặp lỗi 400 hoặc chất lượng giảm mà không có thông báo ([§1.5](#15-reasoning-trong-hội-thoại-nhiều-lượt)).

Các model hiện nay chia thành ba kiểu:

| Kiểu | Ví dụ | Hệ quả |
|---|---|---|
| Luôn suy luận, không tắt được | Grok 4.x [15]; Claude Opus 5.5, Fable 5.1, Mythos 5.1 [5]; Gemini 3.x và 2.5 Pro [13]; Qwen thinking-only [17] | Chỉ điều chỉnh được độ sâu. Mọi request đều có chi phí suy luận |
| Hybrid: cùng model, bật hoặc tắt theo request | Claude Haiku 5.5, Opus 4.6–4.8; Gemini 2.5 Flash; Qwen3 hybrid; DeepSeek [5][13][16][17] | Có thể dùng một model cho cả tác vụ đơn giản và phức tạp |
| Adaptive: model tự quyết có nghĩ hay không | Claude adaptive thinking; Gemini dynamic thinking [5][11] | Câu hỏi đơn giản có thể không có thinking block nào. Đừng giả định block luôn xuất hiện |

Ý tưởng chung đằng sau là *test-time compute*: tăng lượng tính toán lúc suy luận thay vì chỉ tăng kích thước model. Snell và cộng sự (2024) cho thấy khi phân bổ compute lúc suy luận theo độ khó của từng prompt, hiệu quả vượt best-of-N hơn 4 lần. Trong so sánh cùng FLOPs, một model nhỏ được thêm compute lúc suy luận có thể vượt model lớn hơn 14 lần trên những bài mà model nhỏ đã có tỉ lệ đúng khác không [20]. Kết quả này có hai mặt. Compute lúc suy luận đáng giá nhất khi phân bổ *theo độ khó*. Đặt mức cao cho mọi request là cách dùng kém hiệu quả nhất.

## 1.2 Thinking token: sinh ra, tính phí, nhưng không phải lúc nào cũng thấy

### Tính phí

Mọi provider lớn đều tính token suy luận theo giá output:

| Provider | Câu trong docs | Field đọc số token |
|---|---|---|
| OpenAI | Reasoning token "billed as output tokens and occupy the context window" [1] | Responses: `usage.output_tokens_details.reasoning_tokens`; Chat Completions: `usage.completion_tokens_details.reasoning_tokens` [3] |
| Claude | Số tiền giống nhau với mọi giá trị `display`; chỉ thứ bạn nhìn thấy là khác [8] | `usage.output_tokens` là tổng để tính tiền; `usage.output_tokens_details.thinking_tokens` là phần suy luận, luôn ≤ `output_tokens`. Khi stream, phần breakdown chỉ có ở event `message_delta` cuối [8] |
| Gemini | "Response pricing is the sum of output tokens and thinking tokens" [11] | generateContent: `usageMetadata.thoughtsTokenCount` [14]; Interactions: `usage.total_thought_tokens` [11] |
| xAI | Reasoning token "billed as part of your total consumption" [15] | `reasoning_tokens` |
| Qwen (DashScope) | Nội dung suy luận tính theo output token [17] | `completion_tokens_details.reasoning_tokens` hoặc `output_tokens_details.reasoning_tokens` |

Hai hệ quả thực tế:

- **Ẩn suy luận không làm rẻ đi.** Claude với `display: "omitted"` trả `thinking` rỗng nhưng vẫn tính toàn bộ token suy luận [8]. Lợi ích duy nhất là chữ đầu tiên của câu trả lời đến sớm hơn khi stream, vì không phải chờ truyền phần tóm tắt [5].
- **Số token suy luận phải vào log.** Không có field này thì không biết vì sao chi phí một request tăng gấp mười. Ví dụ minh hoạ trong docs Claude có 312 thinking token trên tổng 348 output token [8]: gần 90% output là phần người dùng không thấy.

### Thứ bạn nhận được: tóm tắt, rỗng, mã hoá hoặc toàn văn

| Provider | Có chuỗi suy luận gốc qua API? | Nhận được gì |
|---|---|---|
| OpenAI | Không | Reasoning item: mảng `summary` khi bật `reasoning.summary`, `encrypted_content` khi stateless. Một số model yêu cầu xác minh tổ chức trước khi trả summary [1] |
| Claude | Không. Docs ghi rõ không giá trị `display` nào trả về chain of thought gốc [5] | `thinking` block chứa bản tóm tắt (`summarized`) hoặc chuỗi rỗng (`omitted`), luôn kèm `signature`. `redacted_thinking` chứa `data` mã hoá |
| Gemini | Không | Part có `thought: true` chứa tóm tắt, và thought signature. Tóm tắt có thể rỗng với request đơn giản [11] |
| xAI | Tuỳ model | `reasoning_content`; grok-4.7 trả `reasoning.encrypted_content` trên mọi response [15] |
| DeepSeek, Qwen, model tự host | Có | Toàn văn suy luận trong `reasoning_content` hoặc `reasoning` [16][17][18] |

Một vài chi tiết dễ bỏ sót:

- **Tóm tắt của Claude do một model khác viết**, không phải model đang trả lời [5]. Nó hữu ích để hiển thị tiến trình. Nó không phải bản ghi trung thực những gì model đã tính toán.
- **`signature` của Claude giống nhau với mọi giá trị `display`**, nên đổi `display` giữa các lượt không làm hỏng việc gửi lại [5].
- **`redacted_thinking` dễ bị lọc mất.** Docs Claude ghi: chỉ lọc theo `block.type == "thinking"` sẽ âm thầm bỏ `redacted_thinking` và làm hỏng giao thức nhiều lượt [5].
- **Thinking block của Sonnet 5.5 và Haiku 5.5 gắn với tài khoản tạo ra nó.** Chúng chỉ dùng được trong cùng tài khoản hoặc tài khoản liên kết. Điều này đáng chú ý khi chuyển lịch sử hội thoại giữa các tổ chức hoặc giữa môi trường thử và production [5].
- **Hỏi model "giải thích suy luận của bạn" trong phần text có thể bị từ chối.** Trên Claude Fable 5.1, Opus 5.5, Opus 5, Sonnet 5.5 và Fable 5, request cố lấy suy luận nội bộ qua text có thể nhận refusal với `stop_details.category: "reasoning_extraction"`. Muốn xem suy luận thì đọc `thinking` block với `display: "summarized"` [5].

### Suy luận hiển thị không phải là lời giải thích đáng tin

Nhóm Anthropic thử cài gợi ý (hint) vào câu hỏi, rồi kiểm tra model có thừa nhận đã dùng gợi ý đó trong chain-of-thought không. Kết quả: Claude 3.7 Sonnet nhắc tới gợi ý trung bình 25% số lần, DeepSeek R1 là 39%. Phần lớn trường hợp, chuỗi suy luận không nói ra yếu tố đã thực sự quyết định câu trả lời [25]. Một nghiên cứu của OpenAI về giám sát chain-of-thought cho thấy nếu phạt model vì những "ý nghĩ xấu" bị phát hiện trong chuỗi suy luận, model học cách giấu ý định thay vì bỏ hành vi đó [26].

Hệ quả cho sản phẩm:

- Không dùng chuỗi suy luận, hay bản tóm tắt của nó, làm căn cứ giải thích quyết định cho khách hàng hoặc cho kiểm toán. Nếu cần lý do có thể kiểm tra, yêu cầu model trích dẫn bằng chứng trong câu trả lời và kiểm tra trích dẫn đó bằng code.
- Suy luận vẫn có ích để debug: thấy model hiểu sai đề, bỏ sót ràng buộc, hoặc gọi tool vì lý do nào. Hãy xem nó như log gợi ý, không phải bằng chứng.

## 1.3 Tác vụ nào hưởng lợi, tác vụ nào không

Docs của provider nêu cùng một nhóm tác vụ hưởng lợi: chứng minh, bug khó, coding phức tạp, dùng tool nhiều bước và agent chạy dài [5][10]. OpenAI viết thẳng rằng `reasoning.effort` là "a tuning knob, not the main way to improve quality" [1]. Chất lượng đến trước hết từ việc định nghĩa rõ mục tiêu và cách kiểm tra kết quả.

Nghiên cứu độc lập bổ sung các điểm mà docs không nhấn mạnh:

- **Overthinking** (Chen và cộng sự, 2024): model kiểu o1 dùng quá nhiều compute cho bài đơn giản như "2+3=?" mà gần như không thu được gì. Nhóm tác giả đề xuất cách huấn luyện giảm phần thừa này mà không giảm độ chính xác trên GSM8K, MATH500, GPQA và AIME [21].
- **Inverse scaling** (Gema và cộng sự, 2025): trên bốn nhóm tác vụ (đếm có thông tin gây nhiễu, hồi quy có đặc trưng giả, suy diễn có ràng buộc, các kịch bản rủi ro AI), cho model suy luận lâu hơn làm kết quả *tệ đi*. Có năm kiểu thất bại: Claude bị phân tâm bởi thông tin không liên quan, model o-series bám quá sát cách đặt vấn đề, model chuyển từ giả định hợp lý sang tương quan giả, khó giữ tập trung trong suy diễn dài, và suy luận dài có thể khuếch đại hành vi đáng lo ngại [22].
- **"The Illusion of Thinking"** (Shojaee và cộng sự, 2025): trên các câu đố có độ phức tạp điều chỉnh được, model không suy luận làm tốt hơn ở mức dễ, reasoning model có lợi thế ở mức trung bình, và cả hai sụp ở mức khó. Lượng suy luận tăng theo độ khó đến một ngưỡng rồi *giảm*, dù vẫn còn ngân sách token [23].
- **Phản biện** (Lawsen, 2025): một phần các "thất bại" ở bài trên là do output vượt giới hạn token (Tháp Hà Nội cần liệt kê rất nhiều bước), và một số bài qua sông với N > 5 thực ra không có lời giải nhưng vẫn bị chấm là sai. Khi yêu cầu model viết hàm sinh lời giải thay vì liệt kê từng bước, độ chính xác cao hơn (kết quả sơ bộ) [24].

Cuộc tranh luận giữa hai bài cuối chính là bài học cho người làm eval: **phải tách "không đủ token" khỏi "không biết giải"** trước khi kết luận model kém (xem [§3.6](#36-eval-đo-xem-reasoning-có-thật-sự-giúp)).

Tổng hợp lại:

| Nhóm tác vụ | Có lợi? | Căn cứ |
|---|---|---|
| Toán nhiều bước, chứng minh, bài có lời giải kiểm tra được | Có, rõ nhất | Docs Claude [5]; Snell 2024 [20] |
| Debug, sửa code nhiều file | Có | Docs Claude [10] |
| Agent nhiều bước, tool loop dài | Có, kèm điều kiện quản lý đúng trạng thái suy luận | Docs Claude và OpenAI [1][10] |
| Phân loại, trích xuất, định dạng lại | Thường không; tăng latency và chi phí | Chen 2024 [21]; Shojaee 2025 ở mức dễ [23] |
| Đếm, tác vụ có nhiều chi tiết gây nhiễu, quy tắc cứng | Có thể hại khi suy luận dài | Gema 2025 [22] |
| Hỏi đáp dựa trên tài liệu dài | Chưa có kết luận chung; cần eval trên dữ liệu của bạn | Chưa tìm được nghiên cứu đủ tin cậy |

Các con số trong bảng benchmark của provider đo trên bộ đề chung. Chúng cho biết *hướng*, không thay được eval trên traffic của bạn.

## 1.4 Effort, budget và adaptive thinking

Có ba cách điều khiển lượng suy luận:

| Kiểu | Bạn nói gì với model | Provider dùng |
|---|---|---|
| **Effort / level** | Một mức định tính: `low`, `medium`, `high`... | OpenAI `reasoning.effort`; Claude `output_config.effort`; Gemini 3.x `thinking_level`; Grok `reasoning_effort`; DeepSeek |
| **Budget** | Số token tối đa cho phần suy luận | Gemini 2.5 `thinking_budget`; Claude cũ `budget_tokens`; Qwen `thinking_budget` |
| **Adaptive** | Để model tự quyết có nghĩ không và nghĩ bao lâu, effort làm tín hiệu | Claude `thinking: {type: "adaptive"}`; Gemini "dynamic thinking" |

Xu hướng chung là bỏ budget theo số token để chuyển sang effort. Claude trả 400 với `budget_tokens` từ Opus 4.7 trở đi [6]. Gemini ghi `thinking_budget` là "no longer recommended" trên mọi model 3.x, và gửi cùng lúc `thinking_level` với `thinking_budget` sẽ bị 400 [12]. Budget còn hữu ích khi bạn cần một trần cứng cho phần suy luận trên model cũ hoặc model tự host.

Một vài tính chất cần nắm:

**Effort là tín hiệu mềm, `max_tokens` là giới hạn cứng.** Docs Claude ghi effort là "a behavioral signal, not a strict token budget": ở effort thấp, model vẫn suy luận khi bài đủ khó, chỉ ít hơn [7]. Muốn chặn chi phí tuyệt đối thì dùng `max_tokens`. Đổi lại, request chạm trần sẽ bị cắt, có khi trước khi có chữ nào của câu trả lời ([§3.5](#35-chi-phí-và-độ-trễ-max_tokens-timeout-ước-lượng-ngân-sách)).

**Effort của Claude ảnh hưởng toàn bộ output, không chỉ phần suy luận.** Effort thấp làm model gọi ít tool hơn, gộp thao tác, bỏ phần giới thiệu và viết xác nhận ngắn. Effort cao làm model gọi nhiều tool hơn, giải thích kế hoạch trước khi làm và tóm tắt kỹ hơn [7]. Vì vậy, hạ effort để tiết kiệm có thể đổi luôn cả hành vi agent, không chỉ độ dài suy luận.

**Cùng tên mức nhưng khác lượng suy luận giữa các model.** Docs Claude ghi rằng các mức effort của Sonnet 5.5 đã được hiệu chỉnh lại, nên `high` trên Sonnet 5.5 không cho cùng lượng suy luận như `high` trên Sonnet 5 [7]. Mặc định cũng thay đổi: Opus 5.5 và Haiku 5.5 mặc định `medium`, các model Claude khác mặc định `high`. Request nâng từ Opus 5 lên Opus 5.5 mà không đặt effort sẽ âm thầm chạy thấp hơn một mức [7]. Khi đổi model, chạy lại sweep effort thay vì mang cấu hình cũ sang.

**Ánh xạ giữa các provider không tuyến tính.** Lớp tương thích OpenAI của Gemini công bố bảng ánh xạ, đủ để thấy cùng một chữ `low` mang nghĩa khác nhau tuỳ model [13]:

| `reasoning_effort` (OpenAI) | Gemini 4 Argon | Gemini 3 Flash | Gemini 2.5 (`thinking_budget`) |
|---|---|---|---|
| `minimal` | `low` | `minimal` | 1.024 |
| `low` | `low` | `low` | 1.024 |
| `medium` | `medium` | `medium` | 8.192 |
| `high` | `high` | `high` | 24.576 |

Gateway hoặc thư viện đa provider thường tự ánh xạ effort theo cách tương tự. Khi chuyển provider qua gateway, kiểm tra mức thực tế được gửi đi, nhất là các giá trị biên như `minimal` và `max`.

## 1.5 Reasoning trong hội thoại nhiều lượt

Đây là chỗ code production hay hỏng nhất. Chi tiết theo từng provider, kể cả ảnh hưởng tới cache, nằm ở [Multi-turn §3.7](../multi-turn-conversation/#37-reasoning-và-thinking-blocks) và [Tool Use §1.6](../tool-use/#16-reasoning-model-trạng-thái-ẩn-đi-kèm-tool-call). Mục này tóm tắt quy tắc chung và các điểm riêng của reasoning.

### Hai tình huống khác nhau

**Trong một tool loop** (model gọi tool, bạn trả kết quả, model gọi tiếp): với API stateless có trạng thái suy luận (OpenAI Responses khi tự quản lịch sử, Claude Messages, Gemini generateContent hoặc Interactions không lưu phía server, DeepSeek khi có `tools`), bạn phải gửi lại trạng thái suy luận của lượt đang chạy. Thiếu nó, model mất mạch suy luận đã dẫn đến tool call, và phần lớn trường hợp API trả 400. Hai ngoại lệ: Chat Completions của OpenAI không có cơ chế đưa reasoning lại vào context, nên không có gì để gửi; API stateful (Responses với `previous_response_id`, Gemini Interactions với `store: true` và `previous_interaction_id`) giữ trạng thái phía server, bạn chỉ gửi ID.

**Giữa các user turn** (người dùng gửi câu hỏi mới): mỗi provider một quy tắc, và quy tắc đổi theo model.

| Provider | Trong tool loop | Giữa các user turn |
|---|---|---|
| OpenAI Responses | Gửi lại mọi reasoning item, function call và output kể từ user message cuối [1] | `reasoning.context`: họ gpt-5.6 mặc định `all_turns`, model cũ hơn mặc định `current_turn` [3] |
| OpenAI Chat Completions | Reasoning item không bao giờ vào context lượt sau. Docs ghi điều này có thể làm chất lượng giảm nhẹ và tăng token suy luận trong workflow nhiều function call [2] | Như cột bên |
| Claude | Gửi lại mọi `thinking` và `redacted_thinking`, nguyên văn, đúng thứ tự [5] | Keep-all: Opus 4.5 trở lên, Sonnet 4.6 trở lên, Haiku 5.5, Fable, Mythos. Last-turn-only: Opus/Sonnet cũ hơn và mọi Haiku đến 4.5, API tự bỏ block cũ [5] |
| Gemini | Gửi lại thought signature, không sửa [11] | Từ Gemini 3.5 Flash, model dùng suy luận của mọi lượt trước khi signature có trong lịch sử [12] |
| DeepSeek | Có `tools`: gửi lại `reasoning_content` của mọi lượt, thiếu là 400 [16] | Không có `tools`: `reasoning_content` cũ bị bỏ qua [16] |

Quy tắc an toàn khi bạn tự quản lịch sử với API stateless: **lưu nguyên response, gửi lại nguyên văn, để API tự lọc**. Với API stateful, dùng ID của lượt trước thay vì tự gửi lại; với Chat Completions, quy tắc này không áp dụng vì không có reasoning item để gửi. Docs Claude ghi rõ: qua các lượt, gửi lại mọi thứ, API tự bỏ những gì model không cần [5]. Docs OpenAI cũng nói gửi lại mọi reasoning item là cách đơn giản nhất, vì item không liên quan sẽ bị bỏ qua [1].

### Ví dụ: tool loop với Claude, giữ nguyên block

```python
import anthropic

client = anthropic.Anthropic()
messages = [{"role": "user", "content": "Vì sao đơn #A1027 bị trừ tiền hai lần?"}]

while True:
    resp = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=32000,
        output_config={"effort": "medium"},
        tools=TOOLS,
        messages=messages,
    )
    # Đưa toàn bộ content vào lịch sử, kể cả thinking, redacted_thinking và signature.
    # Không lọc theo type, không dựng lại từ text.
    messages.append({"role": "assistant", "content": resp.content})

    if resp.stop_reason != "tool_use":
        break

    results = [
        {"type": "tool_result", "tool_use_id": b.id, "content": run_tool(b.name, b.input)}
        for b in resp.content
        if b.type == "tool_use"
    ]
    messages.append({"role": "user", "content": results})
```

`TOOLS` và `run_tool` là phần của ứng dụng. Ví dụ không đặt `thinking` vì Sonnet 5.5 mặc định đã bật adaptive thinking [5]. Vòng lặp thật cần thêm xử lý `stop_reason` khác như `max_tokens` hoặc `refusal` (xem [§3.7](#37-failure-modes-thường-gặp)).

### Sửa lịch sử phía trước làm mất hiệu lực thinking block

Từ Claude 5.1, `signature` của mỗi thinking block ghi lại prefix đã sinh ra nó: system, tools và mọi message trước đó. Khi bạn sửa phần lịch sử phía trước (đổi system prompt, đổi danh sách tool, sửa một message cũ), block đó và mọi block phía sau không còn hợp lệ [9]. Tuỳ thời điểm tạo tài khoản và cấu hình `thinking.block_binding.prefix_mismatch_behavior` (beta `thinking-binding-controls-2026-08-01`), API trả 400 hoặc bỏ block đó đi [9]. Chi tiết các thao tác an toàn và không an toàn ở [Multi-turn §3.7](../multi-turn-conversation/#37-reasoning-và-thinking-blocks).

Với OpenAI, suy luận chỉ tái sử dụng được trong cùng họ model: không mang được suy luận giữa họ GPT-5.6 và GPT-5.5 [1]. Với Claude, Fable 5.1 đọc được thinking của Opus 5 và (trên Claude API) Opus 5.5, nhưng chiều ngược lại thì không. Block không đọc được bị bỏ mà không báo lỗi và không tính tiền [5]. Gemini thì yêu cầu vẫn gửi lại thought block của model cũ khi đổi model giữa phiên, backend tự xử lý tương thích [11].

### Ảnh hưởng tới prompt caching

Cấu hình suy luận nằm trong phần prompt được render, nên đổi nó giữa các request làm mất cache:

- **Claude:** đổi kiểu thinking, đổi `budget_tokens` hoặc đổi effort ở cấp request đều làm cache miss. Ví dụ trong docs: request thứ ba đổi effort từ `medium` sang `low` cho kết quả `cache_read_input_tokens: 0` [8]. Để đổi effort giữa chừng mà giữ cache, dùng per-message effort (beta) trên các model hỗ trợ ([§2.3](#23-claude-anthropic)).
- **OpenAI:** dùng item `configuration_update` trong `input` thay vì đổi tham số cấp request. Docs ghi cách này giữ nguyên prefix cho prompt caching [1].
- **Claude, keep-all:** thinking block cũ được giữ trong context và cache, nên mỗi lượt sau đều trả tiền input (hoặc cache read) cho chúng. Hội thoại agent dài tích luỹ suy luận khá nhanh. Docs gợi ý dùng thinking block clearing khi cần lấy lại context [5].

---

# Phần 2 — Reasoning theo provider

## 2.1 Bức tranh chung

Mọi provider có cùng ba thành phần: một tham số độ sâu, một trạng thái phải gửi lại, và token suy luận được tính như output. Khác biệt nằm ở chi tiết:

| Khía cạnh | Khác biệt |
|---|---|
| Tắt được không | Grok 4.x, Claude Opus 5.5, Gemini 3.x không tắt được; Sonnet 5.5 chỉ tắt được phần suy luận trước câu trả lời |
| Đơn vị điều khiển | Effort (OpenAI, Claude, Grok, DeepSeek), level (Gemini 3.x), số token (Gemini 2.5, Qwen, Claude cũ) |
| Trả về gì | Toàn văn (DeepSeek, Qwen, model tự host); tóm tắt hoặc mã hoá (OpenAI, Claude, Gemini) |
| Trạng thái gửi lại | Reasoning item (OpenAI), thinking block + `signature` (Claude), thought signature (Gemini), `encrypted_content` (Grok), `reasoning_content` (DeepSeek) |
| Tham số sampling | Claude 5.x và Opus 4.7, 4.8 trả 400 với `temperature`, `top_p`, `top_k` khác mặc định [5]; Gemini 3.x khuyên bỏ hẳn [12]; DeepSeek bỏ qua `temperature`, `presence_penalty`, `frequency_penalty` mà không báo lỗi, `top_p` tuỳ model [16] |

## 2.2 OpenAI

**Dùng Responses API.** Chat Completions vẫn nhận `reasoning_effort`, nhưng reasoning item không bao giờ được đưa lại vào context [2], và Chat Completions không hỗ trợ function calling với GPT-6 Astra và GPT-6.1 Sol [1]. Docs khuyên bắt đầu với `gpt-6-astra` cho phần lớn workload cần suy luận [1].

**`reasoning.effort`.** SDK khai báo bảy giá trị: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. Không model nào nhận đủ cả bảy [3]:

| Model | Ghi chú từ docs [1] |
|---|---|
| `gpt-6-astra` | Không nhận `none`, gửi là HTTP 400 |
| `gpt-6.1-sol` | Không nhận `none` và `minimal`; mặc định `medium` |
| `gpt-6-sol`, `gpt-6-luna` | Mặc định `medium` |
| `gpt-5.6-*` | Mặc định `medium` ở cả hai mode |
| `gpt-5.5` | Mặc định `medium` |

Gửi giá trị không hỗ trợ sẽ nhận 400. Vì vậy, cấu hình effort nên gắn với model, không dùng chung một hằng số cho mọi model.

**`reasoning.mode`.** GPT-5.6 và GPT-6 có `standard` (mặc định) và `pro`. Pro cho model làm nhiều việc hơn, nên tốn nhiều token và chi phí hơn. Token vẫn tính theo đơn giá chuẩn của model. Mode và effort độc lập với nhau [1].

**`reasoning.summary`.** Nhận `auto`, `concise`, `detailed`. `auto` chọn bộ tóm tắt chi tiết nhất mà model hỗ trợ [1]; `concise` có trên `computer-use-preview` và mọi reasoning model sau GPT-5 [3]. Một số model yêu cầu xác minh tổ chức trước khi trả summary [1]. Field cũ `generate_summary` đã deprecated [3].

**Stateless và `encrypted_content`.** Khi `store: false` hoặc tổ chức dùng Zero Data Retention, reasoning item mặc định có `encrypted_content`. Tham số `include: ["reasoning.encrypted_content"]` vẫn được chấp nhận nhưng không còn bắt buộc [1]. Code tự quản lịch sử phải giữ mọi item kể từ user message cuối và gửi lại. Khi stream, lấy reasoning item từ event `response.output_item.done`, không lấy từ `.added`, vì lúc item còn đang chạy thì `encrypted_content` có thể chưa đầy đủ (xem [Streaming §2.2](../streaming/#22-openai-chat-completions-và-responses-api)).

**`reasoning.context`.** Nhận `auto`, `current_turn`, `all_turns`. Bỏ trống hoặc `auto` thì model tự chọn: họ gpt-5.6 mặc định `all_turns`, model cũ hơn mặc định `current_turn`. Response trả về giá trị thực tế đã dùng [3].

**`configuration_update`.** Đổi effort giữa hội thoại mà giữ cache bằng cách thêm một item vào `input` trước user message tiếp theo [1]:

```json
{"type": "configuration_update", "reasoning": {"effort": "high"}}
```

Cấu hình mới có hiệu lực cho các response sau, đến khi một update khác ghi đè. Không kết hợp item này với compaction tự động hoặc truncation tự động. Endpoint `/responses/compact` cũng từ chối lịch sử có item này [1].

**`phase` trên GPT-5.5 và GPT-5.4.** Assistant message có `phase: "commentary"` (cập nhật trung gian, lời dẫn trước tool call) hoặc `"final_answer"`. Docs khuyên dùng cho luồng dài và nhiều tool, và cảnh báo rằng mất `phase` khi gửi lại có thể khiến model coi lời dẫn là câu trả lời cuối. Không đặt `phase` cho user message [1].

**`text.verbosity`.** `low`, `medium` (mặc định), `high`. Tham số này điều khiển độ dài câu trả lời, không phải độ sâu suy luận [3]. Muốn câu trả lời ngắn nhưng vẫn suy luận kỹ thì giữ effort và hạ verbosity.

**`max_output_tokens` tính cả suy luận.** Khi hết, response có `status: "incomplete"` và `incomplete_details.reason: "max_output_tokens"`, có thể chưa có chữ nào của câu trả lời [1]. OpenAI khuyên để ít nhất 25.000 token cho suy luận và output khi mới bắt đầu, rồi giảm dần theo số liệu thực tế [1].

```python
from openai import OpenAI

client = OpenAI()
resp = client.responses.create(
    model="gpt-6-astra",
    input="Tìm nguyên nhân gốc của lỗi trong log sau: ...",
    reasoning={"effort": "medium", "summary": "auto"},
    max_output_tokens=25000,
)

if resp.status == "incomplete":
    reason = resp.incomplete_details.reason if resp.incomplete_details else None
    used = resp.usage.output_tokens_details.reasoning_tokens if resp.usage else None
    raise RuntimeError(f"incomplete: {reason}, reasoning_tokens={used}")

print(resp.output_text)
```

Đối chiếu với kiểu dữ liệu trong `openai-python` (`Reasoning`, `ResponseUsage`) [3]. `output_text` là thuộc tính tiện ích của SDK, gom text từ các message output.

## 2.3 Claude (Anthropic)

### Cấu hình theo model

Mỗi model Claude xử lý giá trị `thinking` khác nhau [5]:

| Model | Không gửi `thinking` | `adaptive` | `enabled` + `budget_tokens` | `between_tools` | `disabled` |
|---|---|---|---|---|---|
| Opus 5.5, Fable 5.1, Mythos 5.1, Fable 5, Mythos 5 | Adaptive | Adaptive | 400 | 400 | 400 |
| Sonnet 5.5 | Adaptive | Adaptive | 400 | Tắt phần nghĩ trước, ở effort `high` trở xuống | 400 |
| Haiku 5.5, Opus 5 | Adaptive | Adaptive | 400 | 400 | Tắt, ở effort `high` trở xuống |
| Sonnet 5 | Adaptive | Adaptive | 400 | 400 | Tắt |
| Opus 4.8, 4.7 | Tắt | Adaptive | 400 | 400 | Tắt |
| Opus 4.6, Sonnet 4.6 | Tắt | Adaptive | Extended (deprecated) | 400 | Tắt |
| Opus 4.5, Sonnet 4.5, Haiku 4.5 | Tắt | 400 | Extended | 400 | Tắt |

"Ở effort `high` trở xuống" nghĩa là request chạy được ở `low`, `medium`, `high` và trả 400 ở `xhigh` hoặc `max` [5]. Để kiểm tra trước khi gửi, đọc `capabilities.thinking.types.disabled.supported` hoặc `capabilities.thinking.types.between_tools.supported` từ Models API [5].

`between_tools` chỉ có trên Sonnet 5.5. Nó tắt phần suy luận trước câu trả lời, nhưng model vẫn viết progress update giữa các tool call. Không có tool thì response chỉ có text. Giá trị này không nhận field nào khác: gửi kèm `display`, `budget_tokens` hoặc `block_binding` đều bị 400 [5].

### Effort

`output_config.effort` nằm ngoài object `thinking`. Không truyền `adaptive` làm giá trị effort vì đó là một chế độ thinking, không phải một mức [7].

| Mức | Có trên | Ghi chú [7] |
|---|---|---|
| `max` | Mọi model hỗ trợ effort, trừ Opus 4.5 | Không giới hạn chi tiêu token |
| `xhigh` | Model 5.x, Opus 4.7, 4.8 | Tác vụ agent, coding chạy trên 30 phút |
| `high` | Mọi model | Mặc định, trừ Opus 5.5 và Haiku 5.5 |
| `medium` | Mọi model | Mặc định của Opus 5.5 và Haiku 5.5 |
| `low` | Mọi model | Tác vụ đơn giản, subagent |

Docs khuyên với effort `high` trở lên thì đặt `max_tokens` lớn. Với coding agent trên Sonnet 5.5, đặt `max_tokens` bằng 128.000 (mức tối đa của model) và stream response. Với Opus 5, Opus 4.8 ở `xhigh` hoặc `max`, bắt đầu từ 64k [7].

**Per-message effort (beta).** Header `mid-conversation-output-config-2026-07-01`. Thêm một message `role: "system"` với `content` rỗng và `output_config.effort` mới. Phần trước message đó không đổi nên cache vẫn khớp [7]. Trên Claude API, tính năng có ở Fable 5.1, Mythos 5.1, Opus 5.5, Opus 5, Sonnet 5.5 và Haiku 5.5. Trên Amazon Bedrock, Claude Platform on AWS và Google Cloud (bản do Anthropic quản lý), danh sách model ngắn hơn [7].

```python
response = client.beta.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    betas=["mid-conversation-output-config-2026-07-01"],
    output_config={"effort": "high"},
    messages=[
        {"role": "user", "content": "Lập kế hoạch chuyển từ SQLite sang PostgreSQL trong ba bước."},
        {"role": "assistant", "content": previous_assistant_content},
        {"role": "system", "content": [], "output_config": {"effort": "low"}},
        {"role": "user", "content": "Tóm tắt kế hoạch trong một câu."},
    ],
)
```

Mức mới có hiệu lực từ user turn tiếp theo có input mới. User turn chỉ chứa `tool_result` không được tính là input mới, nên thay đổi đặt trong tool loop sẽ chờ đến lượt người dùng kế tiếp [7]. Các lỗi thường gặp [7]:

- Thiếu beta header: 400 `messages.N.output_config: Extra inputs are not permitted`.
- Model không hỗ trợ (ví dụ Fable 5): 400 `output_config.effort requires a model that supports per-turn effort; this model does not`.
- Sonnet 5.5 với `between_tools` và Haiku 5.5 với `disabled` không đổi được effort giữa chừng.

### Display

- `"summarized"`: thinking block chứa bản tóm tắt. Mặc định trên Opus 4.6, Sonnet 4.6 và cũ hơn [5].
- `"omitted"`: `thinking` rỗng, `signature` vẫn có. Mặc định trên Claude 5.x, Opus 4.7, 4.8 và Mythos Preview [5].
- `"updates"` (beta `thinking-display-updates-2026-08-18`): suy luận vẫn ẩn, chỉ progress update giữa các tool call trả text đọc được. Phù hợp với giao diện agent muốn hiện dòng trạng thái ở mỗi bước. Mọi block có text không rỗng là progress update [5].

**Progress update** là ghi chú model viết cho người đang theo dõi agent: vừa tìm thấy gì, sắp làm gì. Mỗi update là một `thinking` block riêng có `signature` riêng, đứng ngay trước `tool_use` mà nó giới thiệu. Model có thể bỏ qua update nào cũng được, và viết ít update hơn ở effort cao và trong chuỗi tool dài [5]. Giao diện không nên giả định lúc nào cũng có update.

### Các giới hạn đi kèm thinking

- **Forced tool use.** Manual extended thinking chỉ nhận `tool_choice` `auto` hoặc `none`. Adaptive thinking nhận forced tool use, *trừ* Opus 5.5, Sonnet 5.5, Fable 5.1 và Mythos 5.1: các model này trả 400 cho mọi request có `tool_choice` `any` hoặc `tool`. Trên các model đó, dùng `auto` kèm strict tool use hoặc structured outputs. Khi forced tool use được chấp nhận, response bắt đầu bằng tool call và không có thinking block [5].
- **Prefill.** Claude 4.6 trở lên trả 400 với assistant turn cuối được điền sẵn, dù bật thinking hay không [5].
- **Sampling.** Trên Claude 5.x, Opus 4.7, 4.8 và Mythos Preview, `temperature`, `top_p`, `top_k` khác mặc định đều trả 400 trên mọi request. `top_p` mặc định là 0,99, nên `top_p: 1` cũng bị 400. Từ API version `2026-10-21`, gửi bất kỳ tham số nào trong ba tham số này đều trả 400 trên mọi model [5].
- **Interleaved thinking.** Với adaptive thinking, suy luận xen giữa các tool call tự bật, không cần beta header. Manual mode trên Opus 4.5, Sonnet 4.5 cần header `interleaved-thinking-2025-05-14` [5].
- **`stop_reason`.** `max_tokens` là giới hạn cứng tính cả thinking của lượt hiện tại; hết thì `stop_reason: "max_tokens"`. Từ Claude 4.5, input cộng `max_tokens` có thể vượt context window. Nếu quá trình sinh chạm giới hạn, `stop_reason` là `"model_context_window_exceeded"` [5].

### Manual extended thinking trên model cũ

Opus 4.5, Sonnet 4.5, Haiku 4.5 chỉ có `thinking: {type: "enabled", budget_tokens: N}`. `budget_tokens` tối thiểu 1.024 và phải nhỏ hơn `max_tokens` [6]. Opus 4.5 là model extended thinking duy nhất nhận cả effort, và docs khuyên đặt cả hai [7]. Khi chuyển lên model mới, thay bằng `adaptive` + effort: `budget_tokens` trả 400 từ Opus 4.7 [6]. Với budget trên 32k, docs khuyên dùng batch để tránh timeout [6].

## 2.4 Google Gemini

Gemini có hai API. **generateContent** là API stateless cũ, signature có thể gắn vào bất kỳ part nào. **Interactions API** là API mặc định trong docs mới, có chế độ stateful (`store: true` + `previous_interaction_id`) mà server tự quản signature [11].

### `thinking_level`

Mặc định và mức hỗ trợ theo model [11]:

| Model | Mặc định | Mức hỗ trợ |
|---|---|---|
| `gemini-4-argon` | `medium` | `low`, `medium`, `high` |
| `gemini-3.8-flash` | `medium` | `low`, `medium`, `high` |
| `gemini-3.6-flash` | `medium` | `minimal`, `low`, `medium`, `high` |
| `gemini-3.5-flash-lite` | `minimal` | `minimal`, `low`, `medium`, `high` |
| `gemini-3.1-pro-preview` | `high` | `low`, `medium`, `high` |
| `gemini-3-flash-preview` | `high` | `minimal`, `low`, `medium`, `high` |
| `gemini-2.5-pro`, `gemini-2.5-flash` | Bật | `low`, `medium`, `high` |
| `gemini-2.5-flash-lite` | Tắt | `low`, `medium`, `high` |

Ba lưu ý từ hướng dẫn chuyển đổi Gemini 3.x [12]:

- `minimal` "does not guarantee that thinking is off": model vẫn có thể suy luận một chút với bài phức tạp. Gemini 3.8 Flash không nhận `minimal`.
- `thinking_budget` không còn được khuyến nghị trên mọi model 3.x. Gửi cả `thinking_level` và `thinking_budget` trong một request sẽ bị 400.
- Google khuyên bỏ `temperature`, `top_p`, `top_k` khỏi mọi request tới Gemini 3.x vì khả năng suy luận được tối ưu cho giá trị mặc định. Muốn kết quả ổn định hơn thì viết quy tắc rõ trong system instruction.

Docs gợi ý bắt đầu với `medium`, thử `low` khi cần nhanh và rẻ, dùng `high` cho toán khó và coding phức tạp, `minimal` cho câu hỏi đơn giản cần tốc độ [12].

Qua lớp tương thích OpenAI, `reasoning_effort: "none"` chỉ tắt được suy luận trên model 2.5 không phải Pro. Gemini 2.5 Pro và Gemini 3 không tắt được. Không gửi cùng lúc `reasoning_effort` với `thinking_level` hoặc `thinking_budget` [13].

### Tóm tắt, token và signature

- generateContent: đặt `include_thoughts=True` trong `thinking_config`. Part tóm tắt có `thought: true`. Số token suy luận nằm ở `usageMetadata.thoughtsTokenCount` [14].
- Interactions: `thinking_summaries: "auto"` hoặc `"none"`. Thought block có thể chỉ có signature mà không có tóm tắt (request đơn giản, tóm tắt bị tắt, nội dung dạng ảnh), nên code phải xử lý trường hợp `summary` rỗng [11].
- Stateless: gửi lại mọi thought block đúng như đã nhận, không xoá, không sửa. Đổi model giữa phiên vẫn gửi lại thought block cũ. Built-in tool như Google Search có signature riêng trên block call/result, cũng phải gửi lại [11].
- Thiếu signature: `MISSING_THOUGHT_SIGNATURE` là một giá trị `finishReason` của generateContent ("Request has at least one thought signature missing") [14], và `missing_thought_signature` là mã lỗi sinh output của Interactions [14]. Bài cũ trong repo ghi Gemini 3 trả 400 khi thiếu signature ở function call của lượt hiện tại. Trang docs chính thức đọc được không nêu điều kiện cụ thể này, nên chưa xác minh.
- Function calling trên Gemini 3.x: `id`, `name` và số lượng function response phải khớp với các function call trước đó. generateContent hiện chưa báo lỗi khi không khớp, nhưng trong phần lớn trường hợp model trả response rỗng với `finish_reason: STOP` [12]. Đây là lỗi im lặng, rất khó tìm nếu không biết trước.

```python
from google import genai
from google.genai import types

client = genai.Client()
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Chứng minh rằng có vô hạn số nguyên tố.",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="medium", include_thoughts=True)
    ),
)
for part in response.candidates[0].content.parts:
    if not part.text:
        continue
    print("[tóm tắt suy luận]" if part.thought else "[trả lời]", part.text)
print("thought tokens:", response.usage_metadata.thoughts_token_count)
```

Các field trong ví dụ đều có trong API reference của generateContent [14]: `GenerationConfig.thinkingConfig` với `thinkingLevel` và `includeThoughts` ("thoughts are returned only when available"), `Part.thought` (boolean), `UsageMetadata.thoughtsTokenCount`. SDK Python dùng tên snake_case tương ứng. Trang hướng dẫn thinking riêng cho generateContent hiện chuyển hướng sang bản Interactions, nên đoạn code ghép từ API reference chứ không chép từ một mẫu chính thức.

### Giới hạn output

Với `gemini-4-argon`, input, suy luận và output dùng chung context 1.048.576 token. Mỗi lượt sinh ở foreground bị giới hạn 65.536 token khi stream và 32.768 token khi không stream, tính cả suy luận [11]. Interactions API phân biệt hai trạng thái dừng giữa chừng, xử lý khác nhau [11]:

- `requires_continuation`: một hop foreground chạm trần mỗi request (32K khi không stream, 64K khi stream) trong khi `max_output_tokens` vẫn còn. Đây không phải lỗi. SDK Google GenAI mặc định tự gửi hop tiếp theo. Khi tắt tự tiếp tục (`automatic_continuation: false` trong ví dụ của docs) hoặc gọi REST, gửi hop tiếp với `previous_interaction_id` và `continuation_token`, rồi nối output của các hop.
- `incomplete`: đã dùng hết `max_output_tokens`. Text có thể rỗng nếu model vẫn đang suy luận, và token suy luận đã sinh vẫn bị tính tiền. Tiếp tục bằng `previous_interaction_id`, `continuation_token` với `max_output_tokens` cao hơn, hoặc gửi lại request với trần cao hơn.

Code coi mọi trạng thái khác `completed` là lỗi sẽ vứt bỏ những response chỉ cần gửi tiếp. Docs khuyên hạ `thinking_level` thay vì đặt `max_output_tokens` nhỏ [11]. Muốn sinh dài mà không giữ kết nối thì dùng `background=True` [11].

generateContent không có `status`; đọc `candidate.finishReason` [14]. `CONTINUATION` nghĩa là chạm trần mỗi request nhưng chưa sinh xong: gửi request tiếp với `continuation_token`, `contents` phải lặp lại y hệt request đầu, và `max_output_tokens` tính cộng dồn qua các hop nên chỉ được tăng hoặc bỏ trống. `MAX_TOKENS` nghĩa là đã hết trần.

## 2.5 xAI Grok

- grok-4.7, 4.6, 4.5 nhận `reasoning_effort` (Responses API: `reasoning.effort`) với `low`, `medium`, `high` (mặc định), `xhigh` (từ 4.6; trên 4.5 được xử lý như `high`). Docs ghi: "Reasoning cannot be disabled" [15].
- Với `grok-4.20-multi-agent`, tham số effort chọn số agent (4 hoặc 16), không phải độ sâu suy luận [15].
- grok-4.7 trả `reasoning.encrypted_content` trên mọi response, kể cả khi không yêu cầu qua `include`. Chat Completions không có field tương ứng [15].
- Không hỗ trợ `presencePenalty`, `frequencyPenalty`, `stop`; gửi sẽ bị lỗi [15].
- Khi stream: `delta.reasoning_content` (Chat Completions) hoặc các event `response.reasoning_*` khi gọi qua SDK OpenAI [15].
- xAI khuyên đặt timeout client cao, ví dụ 3.600 giây, cho reasoning model (xem [Streaming §2.5](../streaming/#25-xai-grok)).

## 2.6 DeepSeek, Qwen và các model open-weight

### DeepSeek

Trang thinking mode hiện tại dùng tên model `deepseek-flash` và `deepseek-v4-pro` [16]. Các bài cũ và nhiều thư viện vẫn dùng `deepseek-reasoner`. Kiểm tra danh sách model hiện hành của tài khoản trước khi chọn tên.

- Bật/tắt theo định dạng OpenAI: `thinking: {"type": "enabled" | "disabled"}`, qua `extra_body` khi dùng SDK OpenAI. Mặc định bật [16].
- Effort: docs nêu `low`, `high`, `max`. Các giá trị khác được ánh xạ: `minimal` và `low` về `low`, `medium` và `xhigh` về `high`, `ultra` về `max` [16]. Gửi `medium` và nghĩ rằng đã giảm suy luận là hiểu sai.
- Output có `reasoning_content` (toàn văn suy luận) tách khỏi `content` [16].
- `temperature`, `presence_penalty`, `frequency_penalty` không có tác dụng ở thinking mode và không báo lỗi [16]. Code chỉnh temperature để "giảm sáng tạo" sẽ không thấy thay đổi gì.
- `top_p` thì khác theo model [16]. Với `deepseek-flash`, `top_p` chỉ có tác dụng ở thinking mode, trong khoảng 0,95–1,0; giá trị dưới 0,95 bị nâng lên 0,95. Với `deepseek-v4-pro` thì ngược lại: thinking mode bỏ qua giá trị bạn gửi (cố định 0,95 khi có tool calling, 1,0 khi không), non-thinking mode thì dùng.
- Multi-turn: request có `tools` phải gửi lại `reasoning_content` của mọi lượt trước, kể cả lượt không gọi tool, thiếu là 400. Request không có `tools` thì `reasoning_content` cũ bị bỏ qua [16]. Code viết từ thời DeepSeek-R1 thường xoá `reasoning_content` trước khi gửi, nên sẽ hỏng khi thêm tools.

### Qwen (Alibaba Model Studio)

- Model **hybrid** bật/tắt bằng `enable_thinking`. Model **thinking-only** "always thinks before replying and cannot be disabled" [17].
- `thinking_budget` giới hạn token suy luận. Khi chạm giới hạn, model chuyển ngay sang viết câu trả lời [17].
- `preserve_thinking` quyết định model có đọc `reasoning_content` của các lượt trước không. Mặc định khác nhau theo model [17].
- Một số model mã mở trên DashScope (ví dụ `qwen3.5-122b-a10b`, `qwen3.5-27b`) chỉ chạy ở chế độ stream khi bật thinking. Gọi không stream nhận lỗi `parameter.enable_thinking only support stream call` [17].
- Với Qwen3 mã mở, `/think` và `/no_think` trong prompt chuyển chế độ theo từng lượt khi `enable_thinking: true` [17].

Trang Alibaba còn liệt kê các model bên thứ ba (Kimi, GLM, MiniMax, DeepSeek) với chế độ thinking riêng, và có chỗ tự mâu thuẫn, ví dụ `kimi-k3` vừa nằm trong nhóm thinking-only vừa được mô tả là tắt được [17]. Với các model này, đọc trang docs của chính nhà phát triển model. Quy tắc gửi lại reasoning của Kimi, GLM, MiniMax có ở [Multi-turn §3.7](../multi-turn-conversation/#37-reasoning-và-thinking-blocks).

### gpt-oss (harmony format)

- gpt-oss dùng harmony format với các role `system`, `developer`, `user`, `assistant`, `tool`. Assistant message có ba channel: `analysis` (chain-of-thought), `commentary` (thường chứa function tool call), `final` (câu trả lời cho người dùng) [4].
- Effort đặt bằng một dòng `Reasoning: low`, `medium` hoặc `high` trong system message, mặc định `medium` [4].
- Multi-turn: thường bỏ `analysis` của các lượt trước khi lượt đó kết thúc bằng `final`. Ngoại lệ là tool call: khi đó phải giữ lại chain-of-thought [4].
- OpenAI khuyến cáo không hiển thị kênh `analysis` cho người dùng cuối, vì nội dung kênh này không tuân theo cùng tiêu chuẩn an toàn như kênh `final` và có thể chứa nội dung có hại [4]. Model card của gpt-oss mô tả thêm định dạng và cách đánh giá model [28].

## 2.7 Self-hosted: tách reasoning content trong vLLM, SGLang

Model mở sinh suy luận dưới dạng text thường, đánh dấu bằng thẻ như `<think>...</think>` hoặc channel của harmony. Server phải tách phần đó ra khỏi `content` bằng một **reasoning parser** đúng với model. Chọn sai parser, hoặc không bật parser, thì toàn bộ suy luận lọt vào câu trả lời người dùng thấy, hoặc tool call bị bỏ sót.

### vLLM

- Bật parser khi khởi động: `vllm serve <model> --reasoning-parser <tên>`. Ví dụ tên trong docs: `deepseek_r1`, `qwen3`, `glm45`, `deepseek_v3`, `gemma4`, `granite` [18].
- Field output: `message.reasoning` (không stream) và `delta.reasoning` (stream). Docs cảnh báo field này trước đây tên là `reasoning_content`. Client cũ đọc tên cũ sẽ nhận giá trị rỗng mà không có lỗi nào [18].
- Bật/tắt suy luận qua `chat_template_kwargs`. Qwen3 mặc định bật (`enable_thinking: False` để tắt); Gemma 4 mặc định tắt; DeepSeek-V3.1 dùng `thinking=True` [18]. `--default-chat-template-kwargs` đặt mặc định cho toàn server, request vẫn ghi đè được [18].
- Đặt `reasoning_effort` (`low`, `medium`, `high`, `none`) thì vLLM tự thêm `enable_thinking` tương ứng. Giá trị `enable_thinking` đặt tường minh sẽ ghi đè [18].
- Tool call chỉ được parse từ `content`, không từ `reasoning`. Bảng hỗ trợ của vLLM ghi DeepSeek R1 không hỗ trợ tool calling [18].
- Suy luận chỉ có ở `/v1/chat/completions`, `/v1/messages` và `/v1/responses` [18].
- Structured output dùng constrained decoding khiến model trả lời ngay từ token đầu và không suy luận [18]. Muốn cả hai thì xem [§3.4](#34-reasoning-kết-hợp-structured-output-và-streaming).
- `include_reasoning: false` bỏ phần suy luận khỏi response nhưng model vẫn suy luận và vẫn tốn thời gian sinh [18].

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[{"role": "user", "content": "Có bao nhiêu số nguyên tố nhỏ hơn 100?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
msg = resp.choices[0].message
# vLLM mới trả "reasoning"; bản cũ trả "reasoning_content". SDK OpenAI giữ field lạ trong model_extra.
extra = msg.model_extra or {}
reasoning = extra.get("reasoning") or extra.get("reasoning_content")
print(reasoning, msg.content, sep="\n---\n")
```

Field và tham số đối chiếu với docs vLLM [18]. Tên model là ví dụ.

### SGLang

- Bật bằng `--reasoning-parser`: `deepseek-r1`, `deepseek-v3` (DeepSeek-V3.1, dùng tham số template `thinking`, không phải `enable_thinking`), `qwen3`, `qwen3-thinking`, `kimi`, `gpt-oss` [19].
- Response theo định dạng của DeepSeek: suy luận nằm trong `reasoning_content`, câu trả lời trong `content` [19]. Như vậy, cùng một model nhưng chuyển từ vLLM sang SGLang thì tên field đổi từ `reasoning` sang `reasoning_content`.
- `separate_reasoning: False` giữ suy luận trong `content` thay vì tách ra (mặc định `True`) [19].
- Có thể gom phần suy luận khi stream và gửi một lần ở chunk cuối của nó, thay vì gửi từng token [19].

### Chat template và cache

Template của Qwen3 và các model tương tự chỉ render suy luận của các assistant turn sau user message cuối. Khi có user message mới, khối `<think>` của các lượt trước biến mất khỏi chuỗi render, và prefix cache của lượt trước không dùng lại được. Chi tiết và cách xử lý ở [Multi-turn §2.7](../multi-turn-conversation/#27-self-hosted-vllm-sglang-và-chat-template).

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Chọn mức effort theo tác vụ

Hướng dẫn chính thức khá thống nhất:

- Claude: bắt đầu với mặc định. Tăng lên `xhigh` hoặc `max` cho coding và agent nhạy với năng lực nhất. Hạ xuống `medium` hoặc `low` cho việc thường ngày hoặc cần độ trễ thấp *khi eval cho thấy chất lượng giữ được* [7].
- OpenAI: effort là núm tinh chỉnh. Với tác vụ agent hoặc nghiên cứu, việc định nghĩa "xong" là gì và cách kiểm tra quan trọng hơn [1].
- Gemini: `medium` cho phần lớn việc, `low` khi cần nhanh và rẻ, `high` cho bài khó [12].

Quy trình đề xuất:

1. **Bắt đầu từ mặc định của model**, không từ mức cao nhất.
2. **Chia traffic theo loại tác vụ** trước khi chỉnh effort. Một endpoint xử lý cả phân loại lẫn debug sẽ không có mức effort nào đúng cho cả hai. Phân loại và trích xuất có thể dùng mức thấp nhất hoặc chuyển sang model không suy luận ([Model Routing](../model-routing/)).
3. **Sweep effort trên bộ eval của từng loại** ([§3.6](#36-eval-đo-xem-reasoning-có-thật-sự-giúp)). Chọn mức thấp nhất mà chất lượng không giảm có ý nghĩa thống kê.
4. **Ghi rõ effort trong cấu hình, gắn với model.** Mặc định thay đổi giữa các model (Opus 5 mặc định `high`, Opus 5.5 mặc định `medium` [7]), và tập giá trị hợp lệ cũng thay đổi.
5. **Xem lại sau mỗi lần nâng model.** Docs Claude nói rõ phải chạy sweep mới thay vì mang cấu hình cũ sang [7].

Effort động theo request (đơn giản thì thấp, phức tạp thì cao) là một dạng routing. Cách này hiệu quả nếu bộ phân loại độ khó rẻ và đủ chính xác. Với hội thoại cần cache, chỉ đổi qua cơ chế giữa chừng của provider ([§1.5](#15-reasoning-trong-hội-thoại-nhiều-lượt)).

## 3.2 Prompt cho reasoning model: viết ít hơn hay nhiều hơn

Phần lớn kỹ thuật prompt ở [Prompt Engineering](../prompt-engineering/) vẫn áp dụng. Khác biệt chính:

**Bỏ "hãy suy nghĩ từng bước".** OpenAI ghi rằng prompt chain-of-thought với reasoning model có thể không cải thiện kết quả và đôi khi còn làm tệ đi [2]. Model đã được huấn luyện để suy luận. Phần cần viết là mục tiêu, ràng buộc và định dạng đầu ra, không phải các bước trung gian [2].

**Few-shot: OpenAI và Anthropic khuyên khác nhau.**

- OpenAI: reasoning model thường không cần ví dụ. Nếu có ví dụ cho định dạng phức tạp, ví dụ phải khớp chặt với chỉ dẫn, vì chỗ lệch có thể cho kết quả xấu [2].
- Claude: ví dụ vẫn hiệu quả khi bật thinking. Ví dụ có lời giải mẫu định hướng cách model suy luận trong thinking block. Docs gợi ý trình bày mỗi ví dụ gồm bài toán, phương pháp áp dụng và đáp án [10].

Hai lời khuyên không mâu thuẫn trực tiếp, nhưng cho thấy cần thử cả hai cách trên eval thay vì theo một quy tắc chung. Xem thêm [Prompt Engineering §1.4](../prompt-engineering/#14-few-shot-khi-nào-giúp-khi-nào-làm-model-bắt-chước-sai).

**Chỉ dẫn cấp cao về cách nghĩ thì có ích.** Ví dụ trong docs Claude: sau khi nhận kết quả tool, đánh giá chất lượng kết quả và quyết định bước tiếp theo trước khi làm tiếp [10]. Chỉ dẫn kiểu này nói *nên cân nhắc điều gì*, không ép từng bước.

**Khi model nghĩ quá nhiều.** Docs Claude gợi ý hai cách: hạ effort, hoặc thêm chỉ dẫn yêu cầu chọn một hướng và theo đến cùng [10]. Trên Sonnet 5.5 và Haiku 5.5, việc model có suy luận hay không ít phụ thuộc vào prompt. Prompt kiểu "nghĩ ít thôi" không đủ, phải hạ effort [8].

**Một số quy ước riêng của OpenAI.** Dùng developer message thay system message với reasoning model. Markdown trong câu trả lời mặc định tắt; muốn bật thì đặt "Formatting re-enabled" ở dòng đầu developer message [2].

**Không dùng prefill.** Claude 4.6 trở lên trả 400 với assistant turn được điền sẵn [5]. Đặt yêu cầu định dạng trong user message hoặc dùng structured outputs.

## 3.3 Reasoning kết hợp tool use

Reasoning model là phần lõi của phần lớn agent hiện nay, và cũng là nơi lỗi trạng thái xuất hiện nhiều nhất.

- **Gửi lại trạng thái trong tool loop.** OpenAI: function call và output phải đi cùng reasoning item của chúng. Code lọc hoặc sắp xếp lại item sẽ nhận 400 (xem [Tool Use §1.6](../tool-use/#16-reasoning-model-trạng-thái-ẩn-đi-kèm-tool-call)). Claude: gửi lại thinking block nguyên văn. Gemini: gửi lại signature và giữ khớp `id`/`name` giữa function call và response [12]. DeepSeek: gửi lại `reasoning_content` mọi lượt khi có `tools` [16].
- **Không đổi cấu hình thinking giữa một assistant turn đang có tool call.** Chỉ đổi ở ranh giới giữa các user turn.
- **Forced tool use bị giới hạn** trên nhiều model Claude ([§2.3](#23-claude-anthropic)). Muốn model chắc chắn gọi tool thì dùng `tool_choice: auto` và nói rõ trong prompt khi nào cần dùng tool. Khi forced tool use được chấp nhận, model không suy luận trước tool call [5].
- **Effort ảnh hưởng số lần gọi tool.** Ở effort thấp, Claude gọi ít tool và gộp thao tác. Haiku 5.5 ở `low` dễ bỏ qua bước tìm kiếm, dừng sớm hoặc bỏ bước kiểm tra trong prompt agent dài [7]. Một agent "tiết kiệm" có thể ra quyết định trên ít dữ liệu hơn.
- **Tool call bị cắt giữa chừng không được thực thi.** Khi `stop_reason` là `max_tokens`, tool call cuối có thể chưa đầy đủ. Claude có thể trả một progress-update block thay cho phần việc chưa xong, với text "This part of the response was interrupted before it finished." (rỗng khi `display: "omitted"`). Để tiếp tục, gửi lại assistant turn nguyên văn rồi thêm user message mới, kèm `tool_result` cho mỗi `tool_use` của lượt đó [5].
- **Ngân sách cho cả vòng lặp.** Mỗi lượt trong tool loop có chi phí suy luận riêng, và với model keep-all thì suy luận cũ còn tính tiền input ở các lượt sau. Đặt trần số lượt và trần token cho cả phiên agent ([Agents](../agents/)).

## 3.4 Reasoning kết hợp structured output và streaming

### Structured output

- **Với provider đóng, structured output và suy luận dùng được cùng nhau.** Model suy luận trong phần riêng, rồi viết JSON theo schema ở phần câu trả lời. Trên các Claude không nhận forced tool use, structured outputs là cách thay thế được docs gợi ý [5].
- **Tự host: constrained decoding có thể chặn suy luận.** vLLM ghi rằng khi bật structured output, model trả lời ngay từ token đầu tiên và không có phần suy luận [18]. Nếu cần cả hai, có hai cách: chạy hai bước (lượt một suy luận tự do, lượt hai trích JSON từ kết quả) hoặc dùng runtime cho phép áp grammar chỉ sau khi kết thúc phần suy luận. Kiểm tra trên phiên bản runtime bạn dùng.
- **`max_tokens` hết giữa chừng thì JSON hỏng.** Kiểm tra `status` / `stop_reason` / `finishReason` trước khi parse ([Structured Output](../structured-output/)).

### Streaming

Suy luận nằm trước câu trả lời, nên với reasoning model, TTFT (thời gian đến token đầu tiên) bao gồm cả thời gian suy luận. Khi provider không stream nội dung suy luận, kết nối có thể im lặng hàng chục giây. Đây là lúc proxy hay cắt kết nối nhất. Event cụ thể của từng provider và cách cấu hình proxy nằm ở [Streaming §3.5](../streaming/#35-stream-kèm-reasoningthinking) và [§3.2](../streaming/#32-stream-qua-proxy-load-balancer-và-cdn).

Tóm tắt những điểm riêng của reasoning khi stream:

| Provider | Event suy luận | Lưu ý |
|---|---|---|
| OpenAI Responses | `response.reasoning_summary_text.delta`, `response.reasoning_text.delta` | Lấy reasoning item để gửi lại từ `response.output_item.done` |
| Claude | `content_block_delta` với `thinking_delta`, rồi `signature_delta` ngay trước `content_block_stop` | Với `display: "omitted"` không có text suy luận, chỉ có signature [5]. Có thể có khoảng dừng vài giây trước khi progress-update block mở [5] |
| Gemini Interactions | `step.delta` loại `thought_summary`, rồi `thought_signature` ở delta cuối trước `step.stop` | [11] |
| Grok, DeepSeek, vLLM | `delta.reasoning_content` hoặc `delta.reasoning` | Tên field tuỳ runtime ([§2.7](#27-self-hosted-tách-reasoning-content-trong-vllm-sglang)) |

Phía UI, hiển thị trạng thái "đang suy nghĩ" kèm thời gian đã trôi qua. Nếu hiển thị bản tóm tắt suy luận thì trình bày khác câu trả lời (thu gọn, chữ mờ) và đừng gọi nó là "lý do" của câu trả lời ([§1.2](#12-thinking-token-sinh-ra-tính-phí-nhưng-không-phải-lúc-nào-cũng-thấy)).

## 3.5 Chi phí và độ trễ: max_tokens, timeout, ước lượng ngân sách

### Đặt `max_tokens`

| Provider | Gợi ý từ docs |
|---|---|
| OpenAI | Để ít nhất 25.000 token cho suy luận và output khi bắt đầu, giảm dần theo dữ liệu thật [1] |
| Claude | Với effort `high` trở lên, đặt `max_tokens` lớn. Coding agent trên Sonnet 5.5: 128.000 và stream. Opus 5 ở `xhigh`/`max`: bắt đầu từ 64k [7] |
| Gemini | Hạ `thinking_level` thay vì đặt `max_output_tokens` nhỏ, vì chạm trần trong lúc suy luận vẫn bị tính tiền phần đã sinh [11] |

Trần cao không làm request tốn hơn. Model chỉ dùng phần nó cần. Trần chỉ là chốt an toàn chống trường hợp suy luận chạy quá dài. Chi phí thực tế được điều khiển bằng effort.

### Ước lượng ngân sách

Không provider nào công bố hệ số "effort `high` tốn gấp bao nhiêu `low`". Con số phụ thuộc mạnh vào loại tác vụ, và docs Claude nói rõ model có thể bỏ qua suy luận hoàn toàn với bài đơn giản [7]. Cách ước lượng thực tế:

1. Chạy một mẫu traffic đại diện (vài trăm request mỗi loại tác vụ) ở từng mức effort ứng viên.
2. Ghi `reasoning_tokens` / `thinking_tokens` / `thoughtsTokenCount`, output token, tổng thời gian và TTFT của từng request.
3. Lấy p50 và p95 của token suy luận theo loại tác vụ. Chi phí trung bình dựa vào p50; trần `max_tokens` và cảnh báo dựa vào p95 và p99.
4. Tính lại sau mỗi lần đổi model hoặc prompt.

Phân bố token suy luận thường lệch phải: phần lớn request dùng ít, một số ít dùng rất nhiều. Cảnh báo theo trung bình sẽ không bắt được các request đắt bất thường. Theo dõi thêm phân vị cao.

### Timeout

- Thời gian suy luận cộng vào TTFT. Timeout HTTP mặc định của nhiều client và proxy (60 giây ở nhiều load balancer) không đủ cho request suy luận dài. Stream giúp giữ kết nối, nhưng vẫn cần heartbeat và cấu hình proxy ([Streaming §3.2](../streaming/#32-stream-qua-proxy-load-balancer-và-cdn)).
- Request suy luận rất dài nên chạy nền: OpenAI background mode, Gemini `background=True`, hoặc batch API. Docs Claude khuyên dùng batch khi budget suy luận trên 32k [6].
- Retry một request suy luận lâu là trả tiền suy luận lại từ đầu. Đặt số lần retry thấp và chỉ retry lỗi tạm thời, không retry khi bị cắt do `max_tokens`.

### Độ trễ

Effort là đòn bẩy độ trễ lớn nhất trên reasoning model, lớn hơn mọi tối ưu mạng hay prompt. Docs Claude ghi effort thấp giảm đáng kể thời gian phản hồi [7]. Các số đo độc lập về TTFT theo effort (ví dụ của Artificial Analysis) thay đổi theo thời điểm và tải. Dùng chúng để so thứ tự, và tự đo trên workload của bạn trước khi đặt SLO. Các kỹ thuật giảm độ trễ khác nằm ở [Latency](../latency/).

## 3.6 Eval: đo xem reasoning có thật sự giúp

Docs Claude mô tả cách đo khi chỉnh hành vi suy luận: chạy một mẫu traffic đại diện có và không có thay đổi, so sánh tần suất xuất hiện thinking block, output token, độ trễ và chất lượng câu trả lời trên các case quan trọng [8]. Cấu trúc một lần sweep:

| Chiều đo | Đo gì | Vì sao |
|---|---|---|
| Chất lượng | Tỉ lệ đúng theo grader của tác vụ (code chấm, so đáp án, hoặc LLM judge đã hiệu chỉnh) | Tiêu chí chính |
| Token suy luận | p50, p95 theo mức effort | Chi phí và rủi ro chạm trần |
| Độ trễ | TTFT, tổng thời gian, p95 | SLO |
| Tỉ lệ bị cắt | Tỉ lệ `incomplete` / `max_tokens` / `MAX_TOKENS` | Lỗi cấu hình trần, không phải lỗi model |
| Tỉ lệ có suy luận | Bao nhiêu request có thinking block (với adaptive) | Hiểu model đang tự quyết thế nào |

Các nguyên tắc rút ra từ nghiên cứu:

- **Có baseline không suy luận**, hoặc mức thấp nhất model cho phép. Snell và cộng sự so sánh trên cùng lượng FLOPs. Không có baseline thì không biết phần tăng đến từ suy luận hay từ model [20].
- **Đo nhiều mức, không chỉ hai mức thấp và cao.** Với inverse scaling, đường chất lượng theo effort có thể đi lên rồi đi xuống [22].
- **Tách "không đủ token" khỏi "sai".** Đếm riêng các case bị cắt. Đặt trần đủ cao trong lúc eval rồi mới kết luận model không giải được (bài học từ tranh luận Shojaee – Lawsen [23][24]).
- **Grader kiểm tra kết quả, không đọc suy luận.** Chuỗi suy luận không phải bằng chứng đúng sai [25]. Khi có thể, kiểm tra đầu ra bằng code: chạy test, so đáp án, xác minh trích dẫn.
- **Chạy lại khi đổi model.** Mức effort giữa các model không tương đương [7].

Cách dựng bộ eval, grader và đọc kết quả có ý nghĩa thống kê nằm ở [Evaluation](../evaluation/).

## 3.7 Failure modes thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| OpenAI: `status: "incomplete"`, output rỗng | `max_output_tokens` bị suy luận dùng hết | Tăng trần (≥ 25.000 lúc đầu), hạ effort, đọc `reasoning_tokens` [1] |
| Claude: `stop_reason: "max_tokens"`, câu trả lời cụt hoặc tool call dở | Thinking cộng text vượt `max_tokens` | Tăng `max_tokens` hoặc hạ effort; không thực thi tool call dở [5] |
| Gemini: response cụt, `status: "requires_continuation"` (generateContent: `finishReason: CONTINUATION`) | Chạm trần 64K / 32K của một hop, ngân sách vẫn còn | Gửi hop tiếp với `continuation_token`, hoặc để SDK tự tiếp tục [11][14] |
| Gemini: output rỗng hoặc cụt, `status: "incomplete"` | Hết `max_output_tokens`, có thể ngay trong lúc suy luận | Tiếp tục với trần cao hơn; hạ `thinking_level` [11] |
| OpenAI: 400 thiếu reasoning item khi có function call | Code dựng lại lịch sử chỉ với `role`/`content`, hoặc lọc item | Gửi lại `response.output` nguyên văn ([Tool Use §1.6](../tool-use/#16-reasoning-model-trạng-thái-ẩn-đi-kèm-tool-call)) |
| OpenAI: lượt sau lỗi vì `encrypted_content` thiếu | Lấy reasoning item từ `response.output_item.added` khi stream | Lấy từ `.done` |
| Claude: 400 vì thinking block bị sửa | Sửa, sắp xếp lại, bỏ một phần block; bỏ block có `thinking` rỗng khi `display: "omitted"` | Gửi nguyên văn, kể cả block rỗng [5] |
| Claude: tool loop hỏng sau khi lọc content | Lọc `block.type == "thinking"` làm mất `redacted_thinking` | Giữ mọi block [5] |
| Claude 5.1+: 400 hoặc block bị bỏ sau khi đổi system prompt hoặc tools giữa phiên | Prefix check của signature | Append-only; xem `block_binding` [9] |
| Claude: 400 với `budget_tokens` | Model từ Opus 4.7 | Chuyển sang `adaptive` + `output_config.effort` [6] |
| Claude: 400 với `thinking: {type: "disabled"}` | Opus 5.5, Sonnet 5.5, Fable, Mythos không tắt được | Hạ effort; Sonnet 5.5 dùng `between_tools` [5] |
| Claude: 400 với `tool_choice: any` / `tool` | Manual thinking, hoặc Opus 5.5, Sonnet 5.5, Fable 5.1, Mythos 5.1 | `tool_choice: auto` + chỉ dẫn trong prompt, strict tool use hoặc structured outputs [5] |
| Claude: 400 với `temperature` / `top_p` | Model 5.x, Opus 4.7, 4.8; mọi model từ API version `2026-10-21` | Bỏ tham số [5] |
| Claude: 400 `messages.N.output_config: Extra inputs are not permitted` | Dùng per-message effort thiếu beta header | Thêm `mid-conversation-output-config-2026-07-01` [7] |
| Cache hit giảm sau khi chỉnh effort theo request | Effort là một phần của prompt được render | Giữ effort cố định trong hội thoại; dùng `configuration_update` hoặc per-message effort [1][7] |
| Chi phí giảm ít dù đã ẩn suy luận | `display: "omitted"` chỉ ẩn, vẫn tính tiền | Hạ effort thay vì ẩn [8] |
| Prompt "nghĩ ít thôi" không có tác dụng | Trên Sonnet 5.5, Haiku 5.5, việc có suy luận ít phụ thuộc prompt | Hạ effort [8] |
| Claude refusal `reasoning_extraction` | Prompt yêu cầu viết suy luận nội bộ vào text | Đọc `thinking` block với `display: "summarized"` [5] |
| Hành vi agent đổi sau khi nâng model, dù không đổi code | Mặc định effort khác (Opus 5 `high`, Opus 5.5 `medium`) hoặc mức đã hiệu chỉnh lại | Đặt effort tường minh; sweep lại [7] |
| OpenAI: 400 với `reasoning.effort: "none"` | Model không nhận `none` (ví dụ `gpt-6-astra`) | Cấu hình effort theo từng model [1] |
| Chat Completions: chất lượng agent kém hơn Responses | Reasoning item không vào context lượt sau | Chuyển sang Responses API [2] |
| Gemini: 400 khi gửi `thinking_level` | Gửi kèm `thinking_budget` (hoặc `reasoning_effort` qua lớp OpenAI) | Chỉ gửi một tham số [12][13] |
| Gemini: response rỗng, `finish_reason: STOP`, không lỗi | Function response không khớp `id`, `name` hoặc số lượng với function call | Khớp đủ ba điều kiện [12] |
| Gemini: `finishReason: MISSING_THOUGHT_SIGNATURE` | Lịch sử thiếu thought signature | Gửi lại lịch sử nguyên văn, dùng chat helper của SDK [14] |
| DeepSeek: 400 khi thêm tools | Code cũ xoá `reasoning_content` trước khi gửi | Gửi lại `reasoning_content` mọi lượt [16] |
| DeepSeek: chỉnh `temperature` không thấy khác biệt | Thinking mode bỏ qua tham số này, không báo lỗi | Không dựa vào sampling để điều khiển đầu ra [16] |
| DeepSeek: gửi effort `medium` nhưng suy luận không giảm | `medium` được ánh xạ thành `high` | Dùng `low` [16] |
| Qwen: `parameter.enable_thinking only support stream call` | Model chỉ chạy khi stream | Bật stream [17] |
| vLLM: field suy luận rỗng sau khi nâng phiên bản | `reasoning_content` đổi tên thành `reasoning` | Đọc `reasoning`, giữ fallback cho tên cũ [18] |
| vLLM: tool call không xuất hiện | Model đặt tool call trong phần suy luận; parser chỉ đọc `content` | Kiểm tra parser và model có hỗ trợ tool calling [18] |
| Self-host: structured output không có suy luận, chất lượng giảm | Constrained decoding áp từ token đầu | Tách hai bước hoặc dùng runtime áp grammar sau phần suy luận [18] |
| Suy luận lộ ra trong câu trả lời người dùng thấy | Chưa bật hoặc chọn sai reasoning parser | Bật `--reasoning-parser` đúng model [18][19] |
| Bài đơn giản tốn nhiều token suy luận | Overthinking | Hạ effort, route sang model không suy luận [21] |
| Chất lượng giảm khi tăng effort | Inverse scaling trên tác vụ có nhiễu | Sweep nhiều mức; không tăng effort mặc định [22] |

## 3.8 Checklist

**Trước khi bật**

- [ ] Đã chia traffic theo loại tác vụ và biết loại nào cần suy luận.
- [ ] Có bộ eval cho từng loại, có baseline ở mức effort thấp nhất hoặc model không suy luận.
- [ ] Đã sweep effort và chọn mức thấp nhất mà chất lượng giữ được.
- [ ] Effort được đặt tường minh, gắn với model, có kiểm tra giá trị hợp lệ.

**Request**

- [ ] `max_tokens` / `max_output_tokens` tính cả suy luận (OpenAI ≥ 25.000 lúc đầu; Claude effort cao thì đặt lớn).
- [ ] Không gửi `temperature` / `top_p` / `top_k` tới Claude 5.x; bỏ chúng với Gemini 3.x.
- [ ] Kiểm tra `status` / `stop_reason` / `finishReason` trước khi dùng output hoặc parse JSON.
- [ ] Không thực thi tool call nằm trong response bị cắt.

**Trạng thái và lịch sử**

- [ ] Lưu nguyên response của provider: reasoning item, thinking block kèm `signature`, `redacted_thinking`, thought signature, `encrypted_content`, `reasoning_content`.
- [ ] Không dựng lịch sử từ text đã hiển thị; không lọc block theo type.
- [ ] Với OpenAI stream: lấy reasoning item từ `output_item.done`.
- [ ] Không đổi system prompt, tools hoặc message cũ giữa phiên (Claude 5.1+); chỉ nối thêm.
- [ ] Đổi effort giữa hội thoại bằng `configuration_update` hoặc per-message effort, không đổi tham số cấp request.

**Vận hành**

- [ ] Log `reasoning_tokens` / `thinking_tokens` / `thoughtsTokenCount` cho mỗi request; dashboard theo p50/p95 và theo loại tác vụ.
- [ ] Cảnh báo khi tỉ lệ response bị cắt tăng.
- [ ] Timeout và proxy đủ dài cho thời gian suy luận; request rất dài chạy nền hoặc batch.
- [ ] Không dùng suy luận hoặc bản tóm tắt làm lời giải thích chính thức cho người dùng hay kiểm toán.
- [ ] Sweep lại effort sau mỗi lần nâng model.

---

## Nguồn tham khảo

### Tài liệu provider

| # | Tài liệu | URL |
|---|---|---|
| 1 | OpenAI: Reasoning models | [developers.openai.com/api/docs/guides/reasoning](https://developers.openai.com/api/docs/guides/reasoning) |
| 2 | OpenAI: Reasoning best practices | [developers.openai.com/api/docs/guides/reasoning-best-practices](https://developers.openai.com/api/docs/guides/reasoning-best-practices) |
| 3 | `openai-python`: kiểu `Reasoning`, `ResponseUsage`, `CompletionUsage`, `ResponseTextConfig` | [github.com/openai/openai-python/tree/main/src/openai/types](https://github.com/openai/openai-python/tree/main/src/openai/types) |
| 4 | OpenAI Cookbook: Harmony response format | [developers.openai.com/cookbook/articles/openai-harmony](https://developers.openai.com/cookbook/articles/openai-harmony) |
| 5 | Claude: Thinking | [platform.claude.com/docs/en/build-with-claude/thinking](https://platform.claude.com/docs/en/build-with-claude/thinking) |
| 6 | Claude: Extended thinking | [platform.claude.com/docs/en/build-with-claude/extended-thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) |
| 7 | Claude: Effort | [platform.claude.com/docs/en/build-with-claude/effort](https://platform.claude.com/docs/en/build-with-claude/effort) |
| 8 | Claude: Steering thinking and cost | [platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost](https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost) |
| 9 | Claude: Preserved thinking | [platform.claude.com/docs/en/build-with-claude/preserved-thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking) |
| 10 | Claude: Prompting best practices | [platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) |
| 11 | Gemini API: Thinking (Interactions API) | [aistudio.google.com/docs/thinking](https://aistudio.google.com/docs/thinking) |
| 12 | Gemini API: What's new in Gemini 3.5 Flash (thinking level, thought preservation, tham số 3.x) | [aistudio.google.com/docs/whats-new-gemini-3.6](https://aistudio.google.com/docs/whats-new-gemini-3.6) |
| 13 | Gemini API: OpenAI compatibility | [aistudio.google.com/docs/openai](https://aistudio.google.com/docs/openai) |
| 14 | Gemini API reference: generateContent (`thoughtsTokenCount`, `FinishReason`); API errors | [aistudio.google.com/api-reference/generate-content](https://aistudio.google.com/api-reference/generate-content), [aistudio.google.com/docs/api-errors](https://aistudio.google.com/docs/api-errors) |
| 15 | xAI: Reasoning | [docs.x.ai/docs/guides/reasoning](https://docs.x.ai/docs/guides/reasoning) |
| 16 | DeepSeek: Thinking mode | [api-docs.deepseek.com/guides/thinking_mode](https://api-docs.deepseek.com/guides/thinking_mode) |
| 17 | Alibaba Cloud Model Studio: Deep thinking | [alibabacloud.com/help/en/model-studio/deep-thinking](https://www.alibabacloud.com/help/en/model-studio/deep-thinking) |
| 18 | vLLM: Reasoning outputs | [docs.vllm.ai/en/latest/features/reasoning_outputs/](https://docs.vllm.ai/en/latest/features/reasoning_outputs/) |
| 19 | SGLang: OpenAI APIs (reasoning support); Separate reasoning | [docs.sglang.io/docs/basic_usage/openai_api_completions](https://docs.sglang.io/docs/basic_usage/openai_api_completions), [github.com/sgl-project/sglang/blob/main/docs/advanced_features/separate_reasoning.ipynb](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/separate_reasoning.ipynb) |

### Paper

| # | Paper | URL |
|---|---|---|
| 20 | Snell et al. (2024), *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*, arXiv:2408.03314 | [arxiv.org/abs/2408.03314](https://arxiv.org/abs/2408.03314) |
| 21 | Chen et al. (2024), *Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs*, arXiv:2412.21187 | [arxiv.org/abs/2412.21187](https://arxiv.org/abs/2412.21187) |
| 22 | Gema et al. (2025), *Inverse Scaling in Test-Time Compute*, arXiv:2507.14417 | [arxiv.org/abs/2507.14417](https://arxiv.org/abs/2507.14417) |
| 23 | Shojaee et al. (2025), *The Illusion of Thinking*, arXiv:2506.06941 | [arxiv.org/abs/2506.06941](https://arxiv.org/abs/2506.06941) |
| 24 | Lawsen (2025), *Comment on The Illusion of Thinking*, arXiv:2506.09250 | [arxiv.org/abs/2506.09250](https://arxiv.org/abs/2506.09250) |
| 25 | Chen et al. (2025), *Reasoning Models Don't Always Say What They Think*, arXiv:2505.05410; bài blog Anthropic cùng chủ đề | [arxiv.org/abs/2505.05410](https://arxiv.org/abs/2505.05410), [anthropic.com/research/reasoning-models-dont-say-think](https://www.anthropic.com/research/reasoning-models-dont-say-think) |
| 26 | Baker et al. (2025), *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation*, arXiv:2503.11926 | [arxiv.org/abs/2503.11926](https://arxiv.org/abs/2503.11926) |
| 27 | DeepSeek-AI (2025), *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, arXiv:2501.12948 | [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948) |
| 28 | OpenAI (2025), *gpt-oss-120b & gpt-oss-20b Model Card*, arXiv:2508.10925 | [arxiv.org/abs/2508.10925](https://arxiv.org/abs/2508.10925) |

---

Thông tin API trong tài liệu được ghi nhận tại **tháng 11/2026** từ tài liệu chính thức và source code của `openai-python`. Các paper được dẫn theo abstract và phần kết luận chính; con số chi tiết trong bảng kết quả của từng paper chưa được kiểm tra lại.

Các điểm chưa xác minh được: điều kiện cụ thể khiến Gemini generateContent trả 400 khi thiếu thought signature (trang hướng dẫn generateContent hiện chuyển hướng sang bản Interactions); việc đổi `reasoning.effort` ở cấp request có làm mất cache trên OpenAI hay không (docs chỉ nói `configuration_update` giữ được prefix); field usage cho token suy luận của DeepSeek; quy tắc gửi lại `reasoning_content` của Kimi K3 trên trang của Moonshot; tên parser `gpt_oss` trong vLLM. Hệ số chi phí và độ trễ giữa các mức effort không được provider nào công bố; hãy đo trên workload của bạn.
