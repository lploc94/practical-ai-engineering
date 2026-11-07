# Hội thoại nhiều lượt và prompt caching

<a id="tài-liệu-toàn-diện-về-multi-turn-conversation-và-prompt-caching"></a>

Bạn nhờ một chatbot sửa hàm đăng nhập. Nó đề xuất một bản sửa. Bạn hỏi tiếp: “Nếu token hết hạn thì sao?” Chatbot trả lời đúng vào đoạn code vừa bàn, như thể nó vẫn đang giữ cuộc hội thoại trong đầu.

Với API stateless, ứng dụng tạo ra sự liên tục ấy bằng cách gửi lại lịch sử: yêu cầu ban đầu, bản sửa của assistant, rồi câu hỏi mới. Sau mười lượt, model nhận cả mười lượt. Trong một agent gọi tool liên tục, lịch sử còn dài ra sau mỗi lần đọc file, chạy test hay nhận kết quả tìm kiếm.

Phần lớn input vì thế là nội dung đã gửi. **Prompt caching** cho phép tái dùng kết quả tính toán của phần đầu prompt, hay *prefix*, khi phần đó không đổi. Cách tổ chức lịch sử quyết định bạn tận dụng được bao nhiêu cache. Chẳng hạn, cập nhật giờ hiện tại ở đầu system prompt có thể khiến một lịch sử dài phải được xử lý lại, dù người dùng chỉ hỏi thêm một câu.

Giữ lịch sử ổn định chỉ giải quyết một nửa bài toán. Một phiên quá dài vẫn có thể khiến model bỏ sót thông tin hoặc bám vào giả định đã lỗi thời. Khi đó cần rút gọn context và chấp nhận dựng lại cache. Tài liệu này đi từ cơ chế đến những lựa chọn như vậy, dành cho người đang xây chatbot hoặc agent gọi LLM qua API.

Bạn có thể đọc theo thứ tự hoặc đi thẳng vào phần đang cần:

- **Phần 1** giải thích lịch sử được gửi lên model thế nào, vì sao cache khớp theo prefix, và chi phí tích luỹ qua các lượt.
- **Phần 2** dùng để tra cứu từng provider: cấu hình cache, giá, TTL, usage và cách gửi lại reasoning.
- **Phần 3** dành cho lúc triển khai: lưu transcript, bố trí dữ liệu động, xử lý tools, rút gọn context và tìm nguyên nhân cache miss. Nếu hệ thống đã chạy nhưng hit rate thấp, bắt đầu ở [§3.13](#313-failure-modes-thường-gặp).

> Bản tháng 11/2026. Thuật ngữ kỹ thuật và tên field giữ nguyên tiếng Anh để tiện đối chiếu API. Giá và giới hạn được ghi nhận tại thời điểm biên soạn; các mục chưa xác minh được đánh dấu tại chỗ.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Multi-turn trên một API stateless](#11-multi-turn-trên-một-api-stateless)
  - [1.2 KV cache và vì sao cache chỉ khớp theo prefix](#12-kv-cache-và-vì-sao-cache-chỉ-khớp-theo-prefix)
  - [1.3 Các vấn đề của multi-turn conversation](#13-các-vấn-đề-của-multi-turn-conversation)
  - [1.4 Kinh tế học của prompt caching](#14-kinh-tế-học-của-prompt-caching)
- [Phần 2 — Prompt caching theo provider](#phần-2--prompt-caching-theo-provider)
  - [2.1 Bức tranh chung: implicit, explicit, routing](#21-bức-tranh-chung-implicit-explicit-routing)
  - [2.2 Claude (Anthropic)](#22-claude-anthropic)
  - [2.3 OpenAI](#23-openai)
  - [2.4 Các provider Trung Quốc](#24-các-provider-trung-quốc)
  - [2.5 xAI Grok](#25-xai-grok)
  - [2.6 Google Gemini](#26-google-gemini)
  - [2.7 Self-hosted: vLLM, SGLang và chat template](#27-self-hosted-vllm-sglang-và-chat-template)
- [Phần 3 — Dựng multi-turn conversation không làm vỡ cache](#phần-3--dựng-multi-turn-conversation-không-làm-vỡ-cache)
  - [3.1 Giữ prefix ổn định giữa các lượt](#31-giữ-prefix-ổn-định-giữa-các-lượt)
  - [3.2 Bố cục prompt theo độ ổn định](#32-bố-cục-prompt-theo-độ-ổn-định)
  - [3.3 Những thứ âm thầm làm vỡ cache](#33-những-thứ-âm-thầm-làm-vỡ-cache)
  - [3.4 Lưu transcript đủ để phát lại](#34-lưu-transcript-đủ-để-phát-lại)
  - [3.5 Thông tin động: đặt ở đâu](#35-thông-tin-động-đặt-ở-đâu)
  - [3.6 Tools trong hội thoại dài](#36-tools-trong-hội-thoại-dài)
  - [3.7 Reasoning và thinking blocks](#37-reasoning-và-thinking-blocks)
  - [3.8 Khi context quá dài: trim, summarize, compact](#38-khi-context-quá-dài-trim-summarize-compact)
  - [3.9 Fork, sub-agent và summarizer](#39-fork-sub-agent-và-summarizer)
  - [3.10 Routing, concurrency và TTL](#310-routing-concurrency-và-ttl)
  - [3.11 Observability cho cache](#311-observability-cho-cache)
  - [3.12 Mã giả: Conversation manager an toàn với cache](#312-mã-giả-conversation-manager-an-toàn-với-cache)
  - [3.13 Failure modes thường gặp](#313-failure-modes-thường-gặp)
  - [3.14 Kiểm tra trước khi đưa vào sử dụng](#314-kiểm-tra-trước-khi-đưa-vào-sử-dụng)
- [Nguồn tham khảo](#nguồn-tham-khảo)

---

<a id="tldr--cheat-sheet"></a>

## Tra cứu nhanh

<a id="7-quy-tắc-nếu-bạn-chỉ-có-2-phút"></a>

### Bảy điểm cần nhớ khi triển khai

1. **Giữ nguyên phần lịch sử đã gửi.** Giữa hai lần rút gọn context, nối lượt mới vào cuối để prefix của lượt trước còn dùng được.
2. **Đặt nội dung ổn định trước nội dung hay đổi.** Bố cục thường dùng là tools → system prompt → ngữ cảnh phiên → lịch sử → lượt hiện tại và dữ liệu mới của lượt đó.
3. **Ghim system prompt và tools theo phiên.** Đưa giờ hiện tại, số token còn lại và trạng thái thay đổi theo request vào message mới. Hồ sơ riêng của user có thể nằm ở đầu lịch sử phiên.
4. **Lưu đủ response để gửi lại.** Ngoài text còn có content blocks, arguments gốc của tool call, reasoning items và thinking blocks kèm signature. Bản lưu chỉ phục vụ hiển thị thường thiếu những phần này.
5. **Serialize nhất quán.** Cố định thứ tự tools, key JSON và khoảng trắng của nội dung do ứng dụng tạo. Giữ nguyên nội dung đã nhận từ provider.
6. **Tính chi phí khi đổi cấu hình giữa phiên.** Model, tools và thinking/effort đều có thể ảnh hưởng cache; một số provider có cơ chế riêng để cập nhật mà giữ được prefix.
7. **Theo dõi cache qua usage.** Ghi `cached_tokens` hoặc field tương đương ở mỗi request. Nếu lượt sau vẫn không hit, kiểm tra độ dài prefix, TTL, routing và cách dựng lịch sử trước khi kết luận có lỗi.

<a id="so-sánh-nhanh-các-provider-snapshot-november-2026"></a>

### So sánh các provider (tháng 11/2026)

| Provider | Cơ chế | Min tokens | TTL | Giá đọc cache so với input | Field đo cache hit | Gợi ý routing |
|---|---|---|---|---|---|---|
| **Claude** | Explicit `cache_control` (hoặc auto ở top-level) | 512–4096 tuỳ model | 5 phút (mặc định) hoặc 1 giờ | 0.1× (Opus 5.5: 0.05×, Fable 5.1: 0.025×); ghi 1.25× / 2× | `cache_read_input_tokens` | Không cần |
| **OpenAI** | Tự động | 1024 | `in_memory` 5–10 phút, `24h`; GPT-5.6+: `30m` / `12h` | 0.1× (GPT-5.x), 0.25× (gpt-4.1, o3), 0.5× (gpt-4o) | `cached_tokens` | `prompt_cache_key` |
| **DeepSeek** | Tự động, lưu trên disk | Không công bố | Vài giờ đến vài ngày | ~0.02–0.03× | `prompt_cache_hit_tokens` | Không |
| **Qwen** (Alibaba) | Implicit + explicit `cache_control` | 1024 | Explicit 5 phút; implicit không xác định | Explicit 0.1× (ghi 1.25×); implicit 0.2× | `cached_tokens` | Không |
| **Kimi** (Moonshot) | Tự động | Chưa xác minh | 5 phút mặc định | ~0.1× (chưa xác minh) | `cached_tokens` | Không |
| **GLM** (Z.ai) | Implicit | Không công bố | Không công bố | ~0.18–0.2× | `cached_tokens` | Không |
| **MiniMax** | Tự động | 512 | Không công bố | 0.1× (M3.1-Flash) | `cache_read_input_tokens` / `cached_tokens` | Không |
| **Doubao** (Volcengine) | Implicit + explicit (Responses API) | 1024 (prefix explicit: 256) | Explicit: mặc định 3 ngày, tối đa 7 | 0.2×, explicit tính thêm phí lưu theo giờ | `cached_tokens` | `previous_response_id` |
| **Grok** (xAI) | Tự động | Không công bố | Không công bố | 0.15–0.25× | `cached_tokens` | `x-grok-conv-id` / `prompt_cache_key` |
| **Gemini** (Google) | Implicit; explicit `cachedContents` (generateContent, model trước Gemini 4) | 2048 (2.5) / 4096 (3.x, 4) | Implicit không công bố; explicit đặt qua `ttl` | 0.1× (3.x), 0.05× (Gemini 4 Argon); explicit tính thêm phí lưu theo giờ | `cachedContentTokenCount` / `total_cached_tokens` | `previous_interaction_id` |

### Thay đổi gì thì mất cache phần nào

| Thay đổi giữa hai request | Hậu quả |
|---|---|
| Nối thêm message ở cuối | Giữ được prefix cũ; cache hit còn phụ thuộc ngưỡng, TTL và routing |
| Sửa / xoá / chèn một message ở giữa | Mất cache từ vị trí đó trở đi |
| Sửa system prompt | Mất cache system và toàn bộ lịch sử |
| Thêm, bớt, đổi thứ tự tools | Mất toàn bộ cache (tools nằm ở đầu prefix) |
| Đổi model | Mất toàn bộ cache (cache gắn với model) |
| Đổi `thinking` / `reasoning.effort` | Thường mất cache phần messages, tuỳ provider có thể mất cả phần đầu |
| Không gửi lại reasoning / thinking của lượt trước | Prefix khác đi, thường mất cache, có provider trả lỗi 400 |
| Trim lịch sử theo cửa sổ trượt mỗi lượt | Đầu prefix đổi mỗi lượt, gần như không bao giờ hit |
| Compact / summarize một lần | Mất cache một lần, phần tóm tắt trở thành prefix mới |

### Reasoning của lượt trước: gửi lại hay bỏ?

| Provider | Quy tắc |
|---|---|
| Claude | Gửi lại thinking blocks y nguyên. API tự bỏ những block model hiện tại không đọc được. Sửa lịch sử phía trước sẽ làm block mất hiệu lực |
| OpenAI (Responses) | Gửi lại toàn bộ item trong `output` (reasoning items, `encrypted_content` khi `store: false`) |
| xAI Grok | Gửi lại `reasoning.encrypted_content` y nguyên. Docs ghi đây là nguyên nhân cache miss số một |
| Gemini | generateContent: gửi lại mọi part với `thoughtSignature` y nguyên (thiếu ở function call trong lượt hiện tại là 400 trên Gemini 3). Interactions API stateless: gửi lại mọi `thought` step |
| DeepSeek | Request có `tools`: bắt buộc gửi lại `reasoning_content` của mọi lượt (thiếu là 400). Không có `tools`: bị bỏ qua |
| Kimi K3 / K3.1 | Bắt buộc gửi lại toàn bộ assistant message kèm `reasoning_content` |
| GLM 5.x / 4.7 | Interleaved thinking: trả thinking cùng tool results. Preserved thinking (`clear_thinking: false`): gửi lại nguyên văn, đúng thứ tự |
| MiniMax M3.x | Giữ nguyên cả assistant message kèm `reasoning_content` |

---

# Phần 1 — Nền tảng

Ở câu hỏi “Nếu token hết hạn thì sao?”, riêng câu mới chưa đủ cho model trả lời. Nó cần biết hàm đăng nhập nào và bản sửa nào đang được bàn tới. Bắt đầu từ cách API nhận ngữ cảnh đó sẽ giúp ta hiểu cả chi phí lẫn cache.

## 1.1 Multi-turn trên một API stateless

### Model không nhớ gì giữa hai request

Với Chat Completions của OpenAI, Messages API của Claude và phần lớn endpoint tương thích OpenAI, client tự gửi lịch sử ở mỗi request. Request trước không tự trở thành ngữ cảnh của request sau. Để tiếp tục cuộc hội thoại, ứng dụng gửi lại tools, system prompt và các message cần thiết theo đúng thứ tự:

```text
Request 1:  [tools][system][user₁]                                  → assistant₁
Request 2:  [tools][system][user₁][assistant₁][user₂]               → assistant₂
Request 3:  [tools][system][user₁][assistant₁][user₂][assistant₂][user₃] → assistant₃
            └──────────── giống hệt request 2 ────────────┘
```

Trong ví dụ này, request 3 giữ phần input của request 2, nối câu trả lời của assistant rồi thêm câu hỏi mới. Phần đầu lặp lại chính là phần có thể dùng cache.

### Lượt hội thoại khác request

Trong chatbot đơn giản, một lượt user tương ứng một request. Trong agent, một lượt user có thể sinh ra hàng chục request:

```text
user: "Sửa bug trong file auth.py"
  request 1 → assistant: tool_use(read_file auth.py)
  request 2 → (tool_result) → assistant: tool_use(run_tests)
  request 3 → (tool_result) → assistant: tool_use(edit_file ...)
  ...
  request 14 → assistant: "Đã sửa xong, nguyên nhân là ..."
```

Sau mỗi lần gọi tool, request kế tiếp mang theo cả lịch sử lẫn kết quả vừa nhận. Manus báo cáo tỉ lệ input:output trung bình khoảng 100:1 trong agent của họ, với khoảng 50 tool call mỗi task. Khi lượng input lớn như vậy, giá input và tỉ lệ đọc từ cache ảnh hưởng đáng kể đến chi phí của một task.

### Các API có state phía server

Một số provider giữ hội thoại hộ bạn:

| API | Cơ chế | Có giảm token phải trả không? |
|---|---|---|
| OpenAI Responses API | `previous_response_id`, `store: true` (mặc định lưu 30 ngày) | Không. Docs ghi rõ toàn bộ input của các response trước trong chuỗi vẫn được tính là input tokens. Phần tiết kiệm đến từ cache |
| OpenAI Conversations API | Truyền `conversation` ID bền vững | Như trên |
| xAI Responses API | `previous_response_id`, lưu 30 ngày | Như trên, cache hit vẫn là thứ giảm giá |
| Doubao Responses API | `previous_response_id`, explicit cache chế độ session | Có cache explicit với phí lưu trữ theo giờ |
| Gemini Interactions API | `previous_interaction_id`, `store: true` (Paid Tier lưu 55 ngày, Free Tier 1 ngày) | Docs không nêu cách tính token của lượt trước; cache implicit vẫn áp dụng |

Với các API này, client gửi ID để server khôi phục lịch sử. Payload qua mạng ngắn đi, còn lịch sử vẫn tham gia vào input của model. Mức tiết kiệm phụ thuộc vào caching và cách tính phí của API đang dùng.

Server quản lý lịch sử giúp giảm lỗi ghép message, nhất là với reasoning items. Đổi lại, ứng dụng phải làm việc theo cơ chế lưu trữ của provider. Khi response gốc hết hạn — chẳng hạn sau 30 ngày ở OpenAI và xAI — code cần một cách khôi phục hoặc bắt đầu lại phiên.

### Chi phí tăng theo bậc hai

Lượt đầu gửi một ít lịch sử; lượt sau gửi lại phần đó cộng phần mới. Để thấy lượng lặp lại tăng ra sao, gọi `S` là số token cố định (tools + system), `t` là số token mỗi lượt thêm vào (user + assistant). Input của request thứ `n` xấp xỉ `S + n·t`. Tổng input của `N` lượt:

```text
Tổng input ≈ N·S + t·N(N+1)/2
```

Số hạng thứ hai tăng theo `N²`: số lượt tăng gấp đôi thì lượng lịch sử gửi lại tăng gần gấp bốn. Trong ví dụ cụ thể ở [§1.4](#14-kinh-tế-học-của-prompt-caching), `S = 8.000`, mỗi lượt thêm 300 token user và 700 token assistant. Sau 30 lượt, tổng input là 684.000 token, dù nội dung mới chỉ khoảng 38.000 token. Hơn 90% input là phần lặp lại. Cache cho phép tính phần lặp lại đủ điều kiện theo giá đọc, thấp hơn giá input thông thường.

Cache còn giảm thời gian đến token đầu tiên (TTFT) vì model xử lý lại ít input hơn. Mức cải thiện phụ thuộc vào lượng cache thực sự đọc được và các yếu tố khác như tải hệ thống; hội thoại dài vẫn cần được đo riêng.

## 1.2 KV cache và vì sao cache chỉ khớp theo prefix

### Prefill và decode

Inference của một transformer decoder có hai pha:

- **Prefill**: model đọc toàn bộ input một lượt, tính ra key và value (KV) của attention cho từng token ở từng layer. Đây là pha tốn compute, chạy song song được.
- **Decode**: model sinh từng token mới, mỗi token cần attend vào KV của tất cả token trước nó.

Trong quá trình decode, model giữ KV của các token trước trong bộ nhớ GPU. Đó là **KV cache** của một request. Prompt caching giữ lại KV của prefix để request khác có cùng prefix có thể dùng tiếp, giảm phần phải prefill.

### Vì sao chỉ tái dùng được prefix

Attention trong decoder là causal: biểu diễn của token thứ `i` ở mỗi layer được tính từ token `i` và mọi token đứng trước nó. Từ layer thứ hai trở đi, KV của token `i` phụ thuộc vào toàn bộ chuỗi `[0, i]`.

```text
Request A: [t0 t1 t2 t3 t4 t5 t6 t7]
Request B: [t0 t1 t2 X  t4 t5 t6 t7]
                     ↑
            khác từ vị trí 3

KV(t0..t2): giống nhau → tái dùng được
KV(t3..t7) của B: phụ thuộc vào X → phải tính lại,
                  dù t4..t7 giống hệt A về mặt token
```

Trong ví dụ trên, `t4` đến `t7` vẫn là những token cũ nhưng ngữ cảnh phía trước chúng đã đổi. Vì vậy cache chỉ dùng được đến trước `X`. Một timestamp thay đổi ở đầu system prompt cũng gây ra hiệu ứng này: phần đứng trước nó còn có thể tái dùng, phần sau phải tính lại.

Điểm cần so sánh là chuỗi token sau khi prompt được render. Hai câu cùng nghĩa chưa chắc có cùng token; khác khoảng trắng, thứ tự key trong JSON được đưa vào prompt hay cách encode Unicode đều có thể làm chuỗi thay đổi. Đây là lý do nên giữ nguyên nội dung đã gửi thay vì viết lại một bản “tương đương”.

### Cache được cài đặt thế nào

Ở các inference engine mã nguồn mở:

- **vLLM automatic prefix caching** chia KV thành block (mặc định 16 token). Mỗi block được hash từ token của chính nó cộng với hash của toàn bộ prefix trước nó. Hai request có block hash trùng nhau thì dùng chung block vật lý. Chỉ block đầy mới được cache, nên phần dư cuối prefix không bao giờ hit. Eviction theo LRU.
- **SGLang RadixAttention** lưu KV trong một radix tree theo chuỗi token. Request nào có chung prefix với một nhánh trong cây thì tái dùng các node đó.

Khi dùng API thương mại, còn hai yếu tố cần tính đến: chính sách giá và routing. Provider có thể tính giá riêng cho việc ghi, đọc cache; request cũng phải đến nơi đang giữ cache để tái dùng nó. Các tham số như `prompt_cache_key` của OpenAI và `x-grok-conv-id` của xAI giúp provider định tuyến những request liên quan.

Hai đặc điểm thực tế đi kèm:

- **Granularity**: cache hit được tính theo khối, không theo từng token. OpenAI làm tròn xuống bội số của 128 token (trừ GPT-5.6+ báo đúng ranh giới), vLLM theo block 16 token, Claude theo breakpoint bạn đặt. Prefix ngắn hơn ngưỡng tối thiểu không được cache, và không có lỗi nào báo điều đó.
- **Best effort**: hầu hết provider không cam kết cache hit. Entry có thể bị evict sớm khi tải cao. Thiết kế đúng tăng xác suất hit, không đảm bảo hit.

## 1.3 Các vấn đề của multi-turn conversation

### Context window có giới hạn, và chất lượng giảm trước khi chạm giới hạn

Một model nhận được 200K, 500K hay 1M token vẫn có thể bỏ sót chi tiết trong đó. Context window cho biết lượng input có thể đưa vào; chất lượng sử dụng lượng input ấy cần được đánh giá riêng.

**Lost in the Middle** (Liu et al., 2023, TACL 2024) cho thấy đường cong hình chữ U: model trả lời đúng nhất khi thông tin nằm ở đầu hoặc cuối context, kém nhất khi nằm ở giữa. Với GPT-3.5-Turbo và 20 tài liệu, độ chính xác khi đáp án nằm giữa còn thấp hơn khi không đưa tài liệu nào (56,1% closed-book). Các phiên bản context dài hơn (16K, 100K) không tốt hơn bản ngắn khi input vừa cả hai.

**Context Rot** (Chroma, 07/2025) thử 18 model, gồm GPT-4.1, Claude 4, Gemini 2.5 và Qwen3. Kết quả:

- Hiệu năng giảm khi input dài ra, kể cả với tác vụ đơn giản. Needle-in-a-haystack không phản ánh đúng cách dùng thực tế.
- Câu hỏi càng ít trùng từ với thông tin cần tìm thì độ giảm theo độ dài càng dốc.
- Chỉ một distractor (đoạn văn trông liên quan nhưng sai) đã làm giảm hiệu năng, bốn distractor giảm thêm nữa.
- Trên LongMemEval (lịch sử hội thoại), cùng câu hỏi nhưng chỉ đưa phần liên quan (~300 token) cho kết quả tốt hơn hẳn so với đưa toàn bộ lịch sử (~113K token).

Trong hội thoại nhiều lượt, lịch sử tích luỹ cả thông tin hữu ích lẫn những đoạn đã hết tác dụng: kết quả tìm kiếm bị loại, thử nghiệm thất bại, giả định đã được sửa. Giữ tất cả giúp tránh mất chi tiết, nhưng cũng tăng lượng nội dung model phải phân biệt. Đây là lý do cần có cách rút gọn lịch sử khi phiên kéo dài.

### Model "lạc" khi yêu cầu đến từng mảnh

Một vấn đề khác xuất hiện ngay cả khi lịch sử chưa dài: người dùng thường bổ sung yêu cầu qua nhiều lượt. **LLMs Get Lost In Multi-Turn Conversation** (Laban et al., Microsoft Research và Salesforce, 2025, ICLR 2026 Outstanding Paper) khảo sát tình huống này. Nhóm tác giả lấy yêu cầu đầy đủ từ các benchmark về code, SQL, API call, toán, data-to-text và tóm tắt, chia thành các mảnh (*shard*), rồi để một LLM đóng vai user lần lượt tiết lộ chúng. Thí nghiệm gồm hơn 200.000 cuộc hội thoại mô phỏng trên 15 model.

| Thiết lập | Cách đưa thông tin | Kết quả so với FULL |
|---|---|---|
| FULL | Toàn bộ yêu cầu trong một lượt | 100% (mốc) |
| CONCAT | Các shard gộp thành một prompt dạng gạch đầu dòng | ~95% |
| SHARDED | Mỗi lượt một shard | Giảm trung bình 39% |
| RECAP | Như SHARDED, lượt cuối nhắc lại toàn bộ shard | Phục hồi một phần |
| SNOWBALL | Mỗi lượt nhắc lại mọi shard trước đó | Phục hồi một phần |

CONCAT gần bằng FULL, trong khi SHARDED giảm rõ rệt. Kết quả cho thấy trong thí nghiệm này, việc thông tin đến qua nhiều lượt gây khó khăn lớn hơn việc cắt nhỏ và diễn đạt lại yêu cầu. Phân tích của bài báo ghi nhận:

- Năng lực tốt nhất (aptitude) chỉ giảm khoảng 15%, nhưng độ dao động giữa lần chạy tốt và tệ (unreliability) tăng khoảng 112%. Model vẫn làm được, chỉ là không ổn định.
- Model đưa ra lời giải hoàn chỉnh khi chưa đủ thông tin, sau đó tiếp tục bám vào giả định ban đầu dù các lượt sau bổ sung yêu cầu.
- Reasoning model không giải quyết được vấn đề. Response dài hơn còn kèm thêm giả định sai.

Sau nhiều lượt làm rõ yêu cầu, có thể thử tổng hợp lại đặc tả hiện tại rồi giao cho model, ngay trong phiên hoặc ở một phiên mới. Bản tổng hợp giúp model tiếp cận yêu cầu đã thống nhất mà không phải tự gỡ từng giả định cũ. Nếu mở phiên mới, prefix đổi và cache phải được dựng lại; nên đánh giá khoản chi phí đó cùng với chất lượng lời giải.

### Tính toàn vẹn của transcript

Multi-turn có các ràng buộc cấu trúc mà single-turn không có:

- **Cặp tool call và tool result**: mỗi lời gọi tool cần kết quả tương ứng, đúng ID và thứ tự API yêu cầu. Với Claude, các `tool_result` của một lượt nằm trong user message kế tiếp. Thiếu kết quả hoặc đóng gói sai có thể gây lỗi 400 hay ảnh hưởng cách model gọi tool song song.
- **Thứ tự role**: Claude yêu cầu user và assistant xen kẽ. Nhiều template self-hosted không chấp nhận hai message cùng role liên tiếp.
- **Reasoning của lượt trước**: mỗi provider có quy tắc riêng (xem bảng ở phần tra cứu nhanh và [§3.7](#37-reasoning-và-thinking-blocks)). Làm sai có thể gây lỗi 400, giảm chất lượng reasoning, hoặc làm vỡ cache.
- **Tham chiếu file và ảnh**: nếu ảnh được tham chiếu qua URL và URL đó trả về nội dung khác ở request sau, prefix đã thay đổi.

### Mâu thuẫn giữa "dọn context" và "giữ cache"

Giả sử một tool result dài 20K token đã hết hữu ích. Xoá nó làm context gọn hơn, nhưng cũng thay đổi prefix của mọi message đứng sau. Giữ nó giúp tái dùng cache, song model vẫn phải làm việc với phần lịch sử thừa.

Một cách cân bằng là rút gọn theo ngưỡng:

- Giữa hai lần rút gọn, lịch sử chỉ nối thêm (*append-only*) để giữ prefix ổn định.
- Khi chạm ngưỡng, compact, tóm tắt hoặc dùng tính năng dọn phía server. Phần lịch sử mới trở thành prefix cho các lượt tiếp theo.
- Nếu dùng cửa sổ trượt hoặc xoá tool result cũ ở mỗi request, prefix sẽ thay đổi liên tục. Cần tính cả chi phí xử lý lại phần đó khi so với phương án rút gọn theo lô.

Các cách rút gọn và rủi ro của từng cách được so sánh ở [§3.8](#38-khi-context-quá-dài-trim-summarize-compact).

## 1.4 Kinh tế học của prompt caching

### Ba loại giá input

Hầu hết provider tách input thành ba loại:

| Loại | Ý nghĩa | Giá điển hình |
|---|---|---|
| Uncached input | Token phải prefill bình thường | 1× |
| Cache write | Token được prefill và ghi vào cache | 1× (OpenAI trước GPT-5.6, DeepSeek, GLM, implicit cache của Qwen) hoặc 1.25×–2.5× (Claude, OpenAI GPT-5.6+, explicit cache của Qwen) |
| Cache read | Token lấy từ cache | 0.02×–0.5× tuỳ provider và model |

Nếu không có phụ phí ghi hoặc lưu trữ, cache hit giảm giá input còn miss được tính như input thường. Khi có phí ghi, kết quả phụ thuộc số lần đọc lại: liên tục ghi entry mới rồi bỏ đi có thể tốn hơn không dùng cache.

### Ví dụ tính toán: 30 lượt chat trên Claude Opus 5.5

Giả định: tools + system = 8.000 token, mỗi lượt user 300 token, mỗi assistant 700 token, các lượt cách nhau dưới 5 phút. Giá Claude Opus 5.5: input $4/MTok, ghi cache 5 phút $5/MTok, đọc cache $0,20/MTok.

Input của request thứ `n` = 8.300 + 1.000·(n−1). Tổng 30 request: **684.000 input tokens**.

| Kịch bản | Cách tính | Chi phí input |
|---|---|---|
| Không dùng cache | 684.000 × $4/M | **$2,74** |
| Cache đúng (append-only, breakpoint ở cuối) | Request 1 ghi 8.300; mỗi request sau đọc toàn bộ prefix trước (tổng 646.700 token đọc) và ghi ~1.000 token mới (tổng 37.300 token ghi) | 646.700 × $0,20/M + 37.300 × $5/M ≈ **$0,32** |
| Có `cache_control` nhưng timestamp trong system prompt | Mỗi request ghi lại toàn bộ input, không đọc gì | 684.000 × $5/M = **$3,42** |

Trong các giả định trên, tái dùng cache giảm chi phí khoảng 8,5 lần. Nếu phải ghi lại toàn bộ ở mỗi request, chi phí lại cao hơn 25% so với không cache. Ứng dụng vẫn có thể trả lời bình thường trong cả hai trường hợp, nên cần nhìn vào `usage` để biết đang trả tiền theo kịch bản nào.

### Break-even và TTL

Với Claude:

- TTL 5 phút: ghi 1.25×, đọc 0.1×. Chỉ cần 2 request dùng chung prefix là hoà vốn (1,25 + 0,1 = 1,35 < 2).
- TTL 1 giờ: ghi 2×. Cần ít nhất 3 request (2 + 0,2 = 2,2 < 3).

Mỗi lần đọc cache làm mới bộ đếm TTL mà không thu thêm phí gia hạn. Với một agent liên tục gửi request cách nhau dưới 5 phút, TTL ngắn thường đã đủ. TTL 1 giờ đáng cân nhắc hơn khi có khoảng nghỉ dài, chẳng hạn người dùng đọc câu trả lời rồi 20 phút sau mới hỏi tiếp. Trong bảng giá trên, phí ghi tương ứng là 1.25× và 2× giá input.

Với OpenAI trước GPT-5.6 và các chế độ không thu thêm phí ghi hoặc lưu trữ, không cần bù phí tạo cache. Khi prefix được tái dùng, phần input đó được tính với giá thấp hơn.

---

# Phần 2 — Prompt caching theo provider

Một request giữ nguyên lịch sử có thể dùng cache tự động ở provider này nhưng cần marker ở provider khác. Cùng một khoảng nghỉ 20 phút, cache có thể còn hoặc đã hết hạn. Khi tích hợp API, cần tra bốn điểm: cách bật cache, thời gian giữ cache, phí ghi và đọc, cùng quy tắc gửi lại reasoning. Các mục dưới đây được tổ chức để có thể đọc riêng theo provider đang dùng.

## 2.1 Bức tranh chung: implicit, explicit, routing

### Implicit và explicit

| | Implicit (tự động) | Explicit (đánh dấu) |
|---|---|---|
| Cách bật | Không cần làm gì | Đặt marker (`cache_control`) hoặc gọi API tạo cache |
| Ai quyết định chỗ cache | Provider | Bạn |
| Phí ghi | Thường không có | Thường có (1.25× hoặc hơn), có nơi tính phí lưu trữ theo giờ |
| Độ chắc chắn | Best effort | Cao hơn, có TTL rõ ràng |
| Provider | OpenAI, DeepSeek, GLM, Kimi, MiniMax, Grok, implicit của Qwen, Doubao và Gemini | Claude, explicit của Qwen, Doubao và Gemini (`cachedContents`) |

Với implicit caching, provider tự tìm phần có thể tái dùng. Ứng dụng vẫn cần giữ prefix ổn định: một system prompt thay đổi mỗi request sẽ hạn chế cache dù tính năng đã được bật sẵn.

### Routing: cache nằm trên một máy cụ thể

Cache phải nằm ở nơi request có thể truy cập. Nếu KV đang ở một máy mà request được chuyển sang máy khác, prefix giống nhau vẫn có thể miss. Provider có thể định tuyến theo hash của phần đầu prompt, kết hợp gợi ý từ client:

| Provider | Gợi ý routing | Ghi chú |
|---|---|---|
| OpenAI | `prompt_cache_key` | Chỉ ảnh hưởng routing, không đảm bảo hit. Trên khoảng 15 request/phút cho cùng prefix + key, traffic có thể tràn sang máy khác |
| xAI | Header `x-grok-conv-id` (Chat Completions), `prompt_cache_key` (Responses API) | Cache entry nằm theo từng server, dùng ID ổn định cho cả phiên |
| Doubao | `previous_response_id` (explicit session cache) | Chuỗi các lượt qua response ID |
| Gemini | `previous_interaction_id` (Interactions API) | Docs ghi chế độ stateful giúp server tối ưu caching |
| Claude | Không cần | Cache cô lập theo workspace (Claude API) hoặc organization (Bedrock, Vertex) |

### Đọc usage để biết cache có hit không

| Provider / API | Field |
|---|---|
| Claude Messages API | `usage.cache_read_input_tokens`, `usage.cache_creation_input_tokens`, `usage.input_tokens` (chỉ phần không cache) |
| OpenAI Chat Completions | `usage.prompt_tokens_details.cached_tokens` |
| OpenAI Responses API | `usage.input_tokens_details.cached_tokens`, `cache_write_tokens` (GPT-5.6+) |
| DeepSeek | `usage.prompt_cache_hit_tokens`, `usage.prompt_cache_miss_tokens` |
| Qwen | `usage.prompt_tokens_details.cached_tokens` (+ `cache_creation_input_tokens` với explicit) |
| GLM, Kimi | `usage.prompt_tokens_details.cached_tokens` |
| MiniMax | `cache_read_input_tokens` (Anthropic-compatible), `prompt_tokens_details.cached_tokens` (OpenAI-compatible) |
| Doubao | `prompt_tokens_details.cached_tokens` (Chat), `input_tokens_details.cached_tokens` (Responses), `cache_read_input_tokens` (Messages) |
| xAI | `prompt_tokens_details.cached_tokens` (Chat), `input_tokens_details.cached_tokens` (Responses) |
| Gemini | `usageMetadata.cachedContentTokenCount` (generateContent), `usage.total_cached_tokens` (Interactions API) |

Ở Claude, cần cộng ba field để có tổng input: `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`. Chỉ đọc `input_tokens` sẽ đếm thiếu phần ghi và đọc cache. Vì vậy một agent chạy hàng giờ có thể báo vài nghìn `input_tokens` trong khi đã sử dụng lượng context lớn hơn nhiều.

## 2.2 Claude (Anthropic)

### Cơ chế

Claude dùng `cache_control` để đánh dấu ranh giới cache, gọi là *breakpoint*. Prompt được render theo thứ tự `tools` → `system` → `messages`. Đặt breakpoint ở một block nghĩa là đề nghị cache phần từ đầu prompt đến hết block đó:

```json
{
  "model": "claude-opus-5-5",
  "tools": [ ... ],
  "system": [
    {"type": "text", "text": "<system prompt tĩnh>", "cache_control": {"type": "ephemeral"}}
  ],
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": [ ... ]},
    {"role": "user", "content": [
      {"type": "text", "text": "<lượt mới>", "cache_control": {"type": "ephemeral"}}
    ]}
  ]
}
```

Các thông số chính:

| Thông số | Giá trị |
|---|---|
| Số breakpoint tối đa | 4 mỗi request |
| TTL | `{"type": "ephemeral"}` = 5 phút; `{"type": "ephemeral", "ttl": "1h"}` = 1 giờ. Mỗi lần đọc làm mới TTL miễn phí. TTL tính từ lúc **bắt đầu** request, nên một lượt generate mất 4 phút chỉ còn khoảng 1 phút cho request tiếp theo |
| Giá ghi | 1.25× input (5 phút), 2× input (1 giờ) |
| Giá đọc | 0.1× input; Claude Opus 5.5: 0.05× ($0,20/MTok); Claude Fable 5.1: 0.025× ($0,25/MTok) |
| Phạm vi cache | Theo workspace trên Claude API; theo organization trên Bedrock và Vertex. Không bao giờ chia sẻ giữa các organization |
| Rate limit | Trên Claude API, token đọc từ cache không tính vào input-token rate limit ở hầu hết model |

Ngưỡng tối thiểu cần được tra theo đúng model. Model mới hơn không nhất thiết có ngưỡng thấp hơn:

| Model | Tối thiểu |
|---|---:|
| Claude Opus 5.5, Opus 5, Sonnet 5.5, Haiku 5.5, Fable 5 / 5.1 | 512 tokens |
| Opus 4.8, Sonnet 5, Sonnet 4.6, Sonnet 4.5 | 1024 tokens |
| Opus 4.7 | 2048 tokens |
| Opus 4.6, Opus 4.5, Haiku 4.5 | 4096 tokens |

Nếu prefix chưa đủ dài, request vẫn chạy nhưng `cache_creation_input_tokens` bằng 0. Đây là một trong những điều nên kiểm tra đầu tiên khi đã đặt marker mà chưa thấy cache.

### Automatic caching cho hội thoại

Với hội thoại nối dài qua từng lượt, việc đặt marker có thể giao cho API: thêm `cache_control` ở top-level request. API đặt breakpoint ở block cuối có thể cache và dịch nó theo lịch sử. Breakpoint tự động này sử dụng một trong bốn slot.

Docs của Anthropic gợi ý kết hợp một breakpoint explicit ở cuối system prompt tĩnh với automatic caching cho phần đuôi hội thoại. Khi phần messages thay đổi, breakpoint ở system vẫn là một điểm để tìm lại cache của tools + system.

Khi kết hợp hai TTL, entry có TTL dài phải đứng trước entry có TTL ngắn. Chẳng hạn, nếu dùng 1 giờ cho phần đuôi thì marker ở system cũng phải dùng 1 giờ.

Automatic caching ít có lợi khi phần cuối request sẽ không xuất hiện lại, chẳng hạn các câu hỏi độc lập với kết quả RAG riêng. Breakpoint nằm sau phần đó khiến ứng dụng trả phí ghi cho nội dung không được đọc lại. Trong trường hợp này, marker explicit ở cuối phần dùng chung sẽ tránh khoản ghi thừa. Còn trong một hội thoại tiếp diễn, câu hỏi và kết quả RAG của lượt hiện tại có thể trở thành lịch sử được tái dùng.

### Invalidation hierarchy

Tác động của một thay đổi phụ thuộc vị trí của nó trong thứ tự `tools` → `system` → `messages`. Bảng dưới đây cho biết phần nào bị invalidation, tức không còn dùng được cache cũ:

| Thay đổi | Cache tools còn | Cache system còn | Cache messages còn |
|---|:---:|:---:|:---:|
| Định nghĩa tools (thêm / bớt / đổi thứ tự) | ✗ | ✗ | ✗ |
| Đổi model | ✗ | ✗ | ✗ |
| `speed`, bật/tắt web search, citations | ✓ | ✗ | ✗ |
| Nội dung system prompt | ✓ | ✗ | ✗ |
| `tool_choice`, ảnh | ✓ | ✓ | ✗ |
| `thinking` hoặc `effort` | tuỳ model | tuỳ model | ✗ |
| Nội dung message | ✓ | ✓ | ✗ (từ vị trí thay đổi) |

### Các cơ chế giữ cache khi cần thay đổi giữa phiên

Có lúc ứng dụng cần đổi chỉ dẫn hoặc bổ sung tool trong một phiên đang chạy. Các tính năng dưới đây cho phép biểu diễn một số thay đổi bằng sự kiện mới trong `messages`, giữ phần đầu request ổn định:

| Bạn muốn | Đừng | Làm thế này |
|---|---|---|
| Thêm chỉ dẫn của operator giữa phiên (đổi mode, cập nhật trạng thái) | Sửa top-level `system` | Nối `{"role": "system", "content": "..."}` vào `messages`. Có trên Opus 5 / 5.5, Opus 4.8, Fable 5 / 5.1, Sonnet 5.5, Haiku 5.5 (không có trên Sonnet 5), không cần beta header. Phải đứng sau một message `user` và không được là `messages[0]` |
| Nhắc nhở model mỗi lượt trong tool loop | Chèn reminder rồi xoá ở request sau | System message với `clear_at: "next_user_message"` (beta `mid-conversation-system-clear-at-2026-08-21`). Nó hiển thị một lượt rồi nằm lại transcript ở dạng đã clear. Không có beta: thêm text block sau các `tool_result`, giữ nguyên các bản cũ |
| Thêm / bớt tool giữa phiên | Sửa mảng `tools` | Khai báo đủ tool từ đầu (tool ẩn dùng `defer_loading: true`), rồi gửi block `tool_addition` / `tool_removal` (beta `mid-conversation-tool-changes-2026-07-01`) |
| Có nhiều tool nhưng mỗi request chỉ cần vài cái | Gửi tool khác nhau mỗi request | Tool search: schema được nối thêm khi cần, không thay thế, prefix giữ nguyên |
| Đổi effort giữa phiên | Đổi `output_config.effort` ở top-level | System message có `content: []` và `output_config: {effort: ...}` (beta `mid-conversation-output-config-2026-07-01`) |
| Dùng model rẻ hơn cho việc phụ | Đổi `model` giữa phiên | Giữ vòng lặp chính trên một model, gọi sub-agent với model khác cho việc phụ |

### Thinking blocks và preserved thinking

Với các model Claude hiện tại, thinking blocks của lượt trước được giữ mặc định. Gửi lại chúng y nguyên giữ cho cache phần messages hợp lệ. Ở các model Opus / Sonnet cũ hơn và Haiku đến 4.5, một user message thường (không phải tool result) đi sau tool use khi bật thinking sẽ khiến API bỏ thinking blocks cũ khỏi context, và mọi thứ sau block đầu tiên bị bỏ rơi khỏi cache.

Từ Claude Fable 5.1 và các model 5.5 (Opus, Sonnet, Haiku), Anthropic thêm **preserved thinking**: chữ ký (`signature`) của mỗi thinking block ghi lại prefix đã sinh ra nó (system, tools, mọi message trước). Khi transcript được gửi lại, API kiểm tra prefix đó không đổi.

- Tài khoản tạo từ 31/08/2026 trở đi: request gửi lại thinking block sau khi lịch sử phía trước bị sửa sẽ bị **400**.
- Tài khoản cũ hơn: API ghi nhận mismatch nhưng chỉ thực thi khi request đặt `thinking.block_binding.prefix_mismatch_behavior` (`"error"` hoặc `"drop_block"`, beta `thinking-binding-controls-2026-08-01`).

Những thao tác làm mất hiệu lực mọi thinking block phía sau:

- Sửa, sắp xếp lại, hoặc xoá một lượt cũ trong khi giữ các lượt sau, kể cả xoá tool result cũ ở phía client.
- Chèn text theo từng request vào một lượt cũ (reminder, dòng trạng thái, số token) rồi xoá hoặc dựng lại ở request sau.
- Dựng lại `system` hoặc `tools` giữa các request trong cùng phiên.
- Xoá một thinking block không nằm ở đầu chuỗi.
- Ảnh hoặc tài liệu ở lượt cũ được tham chiếu qua URL mà trả về byte khác ở request sau.

Những thao tác vẫn an toàn: nối thêm (kể cả system message giữa phiên), xoá một dãy thinking blocks từ đầu (cũ nhất trước), đổi tham số ngoài `system` / `tools` / `messages` (như `max_tokens`, `tool_choice`), thêm / bớt / di chuyển marker `cache_control`, và dọn context phía server (context editing, compaction).

Cache reuse và kiểm tra signature cùng phụ thuộc vào việc giữ nguyên prefix. Tuy vậy, chúng cần được kiểm tra riêng: cache hit cho biết có phần được tái dùng; signature còn ràng buộc thinking block với lịch sử đã sinh ra nó.

### Context editing và compaction phía server

Claude hỗ trợ dọn context trên bản sao phía server. Nhờ đó, client vẫn giữ transcript gốc khi sử dụng các cơ chế sau:

| Tính năng | Làm gì | Beta |
|---|---|---|
| Context editing | Xoá tool result cũ (`clear_tool_uses_20250919`, tuỳ chọn xoá cả input của tool call) hoặc thinking blocks cũ (`clear_thinking_20251015`) theo ngưỡng | `context-management-2025-06-27` |
| Compaction | Khi hội thoại chạm ngưỡng (mặc định 150K token), API tự tóm tắt phần đầu thành một `compaction` block | `compact-2026-01-12` |
| Memory tool | Model đọc/ghi file trong thư mục `/memories` mà bạn cung cấp backend, để lưu trạng thái qua nhiều phiên | `memory_20250818` |

Sau compaction, lưu toàn bộ `response.content` vào lịch sử. Mảng này chứa compaction block để API nhận biết phần đã được tóm tắt ở request sau. Nếu ứng dụng chỉ lấy text hiển thị, nó sẽ bỏ mất trạng thái cần thiết để tiếp tục compaction.

### Các chi tiết dễ bỏ sót

- **Lookback 20 vị trí**: mỗi breakpoint chỉ dò ngược tối đa 20 vị trí để tìm entry cũ. Một chuỗi `tool_use` liên tiếp tính là một vị trí, một chuỗi `tool_result` liên tiếp cũng vậy. Nếu một lượt thêm hơn 20 vị trí nội dung khác (tool loop tuần tự dài, nhiều ảnh), request sau không tìm thấy cache cũ. Cách xử lý: thêm breakpoint trung gian khoảng mỗi 15 vị trí.
- **Request song song**: cache entry chỉ dùng được sau khi response đầu tiên bắt đầu ("after the first response starts"), dù có stream hay không. Gửi N request cùng prefix cùng lúc thì cả N đều trả giá đầy đủ. Với fan-out, tài liệu khuyên gửi 1 request, chờ response đầu tiên, rồi mới gửi N−1 request còn lại.
- **Pre-warm**: gửi request với `max_tokens: 0` để ghi cache mà không sinh output, rồi trả về ngay. Đặt `cache_control` ở block cuối dùng chung với traffic thật (system hoặc tools), không đặt ở placeholder user message. Thinking và effort phải giống traffic thật, vì chúng được render vào prompt.
- **Keep-alive cho Fable 5.1**: vì đọc cache chỉ 0.025×, khi user im lặng 5–60 phút, gửi lại request trước với `max_tokens: 0` ngay trước khi TTL 5 phút hết thường rẻ hơn dùng TTL 1 giờ.
- **Cache diagnostics** (beta `cache-diagnosis-2026-04-07`): gửi header ở **mọi** request và truyền `diagnostics.previous_message_id`. Response cho biết hai request khác nhau ở đâu (model, system, tools hay lịch sử messages). Thiếu header ở request trước thì không có fingerprint để so.
- **Server tools** như web search tự chèn một cache write 5 phút sau tool result khi request đã dùng caching. Đây là hành vi bình thường, không phải bug.

## 2.3 OpenAI

### Cơ chế

OpenAI cache tự động cho mọi model được hỗ trợ, không cần marker:

- Ngưỡng tối thiểu: 1.024 token input hiển thị với GPT-5.6 trở lên. Với model cũ hơn, ngưỡng thay đổi theo cấu hình request (tools, ảnh, schema, reasoning effort, verbosity).
- Granularity: số token cache làm tròn xuống bội số của 128, trừ GPT-5.6+ báo đúng ranh giới. GPT-5.5 và GPT-5.5 Pro đặt breakpoint ngầm mỗi 2.048 token.
- Phạm vi: toàn bộ prefix đã render phải khớp. Lookup đi từ prefix khớp dài nhất về ngắn dần.
- Cache lưu tensor KV, không lưu text. Không chia sẻ giữa các organization hay vùng địa lý. Tương thích với Zero Data Retention (tối đa 24 giờ).

### Routing và `prompt_cache_key`

Request được route theo hash của phần đầu prompt, tải của máy và `prompt_cache_key`. Docs ghi rằng traffic trên 15 request/phút có thể bị tràn sang máy khác. Với model trước GPT-5.6, docs gợi ý giữ khoảng 15 RPM cho mỗi key và chia traffic lớn hơn ra nhiều key.

Có thể đặt `prompt_cache_key` theo conversation ID hoặc theo nhóm request dùng chung một system prompt dài. Key cung cấp gợi ý routing; tải hệ thống vẫn có thể khiến request đến máy khác. Với GPT-5.6+, key là tuỳ chọn và còn dùng để tách số liệu cache theo khách hàng hoặc workspace.

### Retention

| Thế hệ | Tham số | Lựa chọn |
|---|---|---|
| Trước GPT-5.6 | `prompt_cache_retention` | `in_memory`: khoảng 5–10 phút không hoạt động, tối đa 1 giờ. `24h`: thường khoảng 30 phút, tối đa 24 giờ. Organization không ZDR mặc định `24h`, organization ZDR mặc định `in_memory`. GPT-5.5 / 5.5 Pro chỉ nhận `24h` |
| GPT-5.6+ (Responses API) | `prompt_cache_options.ttl` | `"30m"` (mặc định, tối thiểu 30 phút, làm mới mỗi lần đọc) hoặc `"12h"` (tối thiểu 12 giờ, ghi tốn 2.5×) |

Model hỗ trợ `24h`: gpt-5.5, gpt-5.5-pro, gpt-5.4, gpt-5.2, gpt-5.1 (kể cả các bản codex, codex-max, codex-mini, chat-latest), gpt-5, gpt-5-codex, gpt-4.1. Các model khác chỉ có `in_memory`.

### Giá (standard tier, input ≤ 272K, USD / 1M token)

| Model | Input | Cached input | Tỉ lệ | Output |
|---|---|---|---|---|
| gpt-5.6-sol | $4,00 | $0,40 | 0.10× | $20,00 |
| gpt-5.5 | $5,00 | $0,50 | 0.10× | $30,00 |
| gpt-5.4 | $2,50 | $0,25 | 0.10× | $15,00 |
| gpt-4.1 | $2,00 | $0,50 | 0.25× | $8,00 |
| o3 | $2,00 | $0,50 | 0.25× | $8,00 |
| gpt-4o | $2,50 | $1,25 | 0.50× | $10,00 |

Từ GPT-5.6, phí ghi cache là 1.25× với TTL 30 phút và 2.5× với TTL 12 giờ. Vì vậy cần tính số lần một prefix được đọc lại để bù phí ghi, tương tự bài toán break-even ở Claude.

### Những gì làm vỡ cache

Theo docs của OpenAI:

- Đổi `model`.
- Đổi `tools`: tên, mô tả, schema hoặc **thứ tự**.
- Đổi `parallel_tool_calls`, `text.format`, `reasoning.effort`, `text.verbosity`.
- Compaction, summarization hoặc truncation lịch sử.
- Sửa message cũ thay vì nối thêm.
- Đặt timestamp hoặc nội dung riêng theo user trước phần nội dung ổn định.

### Conversation state và reasoning items

Responses API trả về `output` là một mảng item: message, function call, reasoning item. Với reasoning model, reasoning item mang trạng thái suy luận của lượt đó.

- **`store: true` + `previous_response_id`**: server giữ lịch sử, bạn chỉ gửi phần mới. Reasoning items được nối đúng chỗ. Vẫn trả tiền cho toàn bộ input, nhưng prefix do server ghép nên ổn định.
- **`store: false` hoặc ZDR**: bạn tự giữ lịch sử. Phải lấy **mọi item** trong `output` và gửi lại ở `input` của request sau, gồm cả reasoning item với `encrypted_content` (yêu cầu qua `include: ["reasoning.encrypted_content"]`) và giữ nguyên field `phase` của assistant message. Docs có ghi trường hợp thiếu `phase` khiến GPT-5.5 nhầm một cập nhật trung gian là câu trả lời cuối.
- Model hỗ trợ có thể dùng `reasoning.context: "all_turns"` để giữ reasoning của mọi lượt trước.

Khi chuyển từ Chat Completions sang Responses API với `store: false`, hãy kiểm tra code lưu lịch sử. Một schema chỉ có `role` và `text` sẽ bỏ mất reasoning items. Request sau khi đó có prefix khác và thiếu trạng thái suy luận mà API đã trả về.

### Compaction

Responses API hỗ trợ compaction qua `context_management` với `compact_threshold`, cùng endpoint riêng `/responses/compact`. Compaction thay context cũ bằng biểu diễn ngắn hơn, nên giảm input nhưng cũng thay prefix và giảm cache reuse. Docs khuyến nghị so tổng chi phí input trước và sau khi bật để biết phần tiết kiệm có bù được phần cache mất đi hay không.

## 2.4 Các provider Trung Quốc

DeepSeek, Qwen, Kimi, GLM, MiniMax và Doubao đều có lựa chọn caching, nhưng cách tính phí và xử lý reasoning khác nhau. Phần lớn cung cấp endpoint tương thích OpenAI; một số có thêm endpoint tương thích Anthropic. Khả năng nhận cùng định dạng request chưa đủ để suy ra cùng hành vi cache. DeepSeek, chẳng hạn, nhận `cache_control` nhưng bỏ qua field này và dùng implicit caching.

### DeepSeek

| Mục | Chi tiết |
|---|---|
| Cơ chế | Tự động, bật mặc định. Cache theo prefix, lưu trên disk. Các mốc prefix được lưu ở cuối input, cuối output, và theo khoảng cố định với input dài. Best effort |
| TTL | Không cố định. Docs ghi entry thường bị xoá sau "vài giờ đến vài ngày" không dùng |
| Phí lưu trữ | Không có |
| Usage | `prompt_cache_hit_tokens`, `prompt_cache_miss_tokens` |
| Endpoint Anthropic | Có, `https://api.deepseek.com/anthropic`. `cache_control` bị **bỏ qua**, cache implicit vẫn hoạt động |

Giá (USD / 1M token, off-peak / peak; peak là 01:00–04:00 và 06:00–10:00 UTC thứ Hai đến thứ Sáu, trừ ngày lễ Trung Quốc):

| Model | Cache hit | Cache miss | Output |
|---|---|---|---|
| deepseek-flash (V4.1-Flash) | $0,003 / $0,006 | $0,15 / $0,30 | $0,60 / $1,20 |
| deepseek-v4-pro | $0,022 / $0,044 | $0,66 / $1,32 | $1,98 / $3,96 |

Trong thinking mode, cách gửi lại reasoning phụ thuộc việc request có khai báo tools hay không:

- Request **có `tools`**: phải gửi lại `reasoning_content` của **mọi** lượt trước, kể cả lượt không gọi tool. Thiếu sẽ bị 400.
- Request **không có `tools`**: `reasoning_content` bị bỏ qua, không được ghép vào context.

Nếu đang dùng code viết từ thời DeepSeek-R1, kiểm tra xem nó có loại bỏ `reasoning_content` trước khi gửi hay không. Quy tắc cũ đó không áp dụng cho agent có tools trên các model hiện tại được mô tả ở đây.

### Alibaba Qwen (Model Studio / DashScope)

Qwen có hai chế độ, mỗi request chỉ dùng một:

| | Implicit | Explicit |
|---|---|---|
| Bật | Tự động, không tắt được | Khi request có `cache_control` |
| Tối thiểu | 1.024 token | 1.024 token |
| Giá ghi | 1× input | 1.25× input |
| Giá hit | 0.2× input (một số model như DeepSeek host trên Qwen: 0.1×) | 0.1× input |
| TTL | Không xác định | 5 phút, làm mới khi hit |

Explicit dùng cú pháp giống Claude: `"cache_control": {"type": "ephemeral"}`, tối đa 4 marker, mỗi marker cache từ đầu `messages` đến nó. Marker đặt trên `tools` bị bỏ qua.

Usage: `prompt_tokens_details.cached_tokens` và `cache_creation_input_tokens` (OpenAI / DashScope), `cache_read_input_tokens` và `cache_creation_input_tokens` (Anthropic-compatible).

Giá input (USD, vùng Singapore): qwen3.8-max $2, qwen3.7-plus $0,40, qwen3.8-flash $0,15. Giá hit tính theo tỉ lệ ở bảng trên.

Quy tắc reasoning trong multi-turn của Qwen API chưa được tài liệu này xác minh. Với model Qwen self-hosted, xem [§2.7](#27-self-hosted-vllm-sglang-và-chat-template) về chat template.

### Moonshot Kimi

| Mục | Chi tiết |
|---|---|
| Cơ chế | Tự động với prefix lặp lại, TTL mặc định 5 phút |
| Phí ghi | **Chưa rõ.** Trang giá chính thức vừa ghi cache write của K3 được tính riêng theo TTL (5 phút / 1 giờ), vừa ghi request hit không có phí ghi thêm. Kiểm tra lại trước khi tính chi phí |
| Giá | Bảng giá chính thức không hiển thị được khi research. Nguồn thứ ba ghi K3: input $3, cache hit $0,30, output $15 / 1M token. **Chưa xác minh** |
| Usage | `cached_tokens` (nguồn thứ ba) |
| Lưu ý | Dòng K2 dự kiến ngừng vào cuối tháng 12/2026 |

Cách giữ reasoning cần được chọn theo đúng phiên bản Kimi:

- **K3 / K3.1 Pro**: gửi lại toàn bộ assistant message **kèm `reasoning_content`**. FAQ ghi đây là bắt buộc trong hội thoại nhiều lượt và tool-call loop.
- **K2.7 Code**: preserved thinking luôn bật, giữ toàn bộ `reasoning_content` cũ.
- **K2.6**: điều khiển bằng `thinking.keep`. Mặc định `null` bỏ qua reasoning cũ, `"all"` giữ lại.

### Zhipu GLM (Z.ai)

| Mục | Chi tiết |
|---|---|
| Cơ chế | Chỉ implicit, tự nhận diện nội dung lặp lại |
| Tối thiểu, TTL | Không công bố |
| Phí lưu trữ | Đang miễn phí có thời hạn |
| Usage | `prompt_tokens_details.cached_tokens` |

Giá (USD / 1M token, input / cached / output): GLM-5.3, 5.2, 5.1: $1,40 / $0,26 / $4,40. GLM-5: $1,00 / $0,20 / $3,20. GLM-4.7, 4.6: $0,60 / $0,11 / $2,20. Trang hướng dẫn caching của Z.ai ghi hit "thường bằng 50% giá chuẩn", nhưng bảng giá cho thấy khoảng 18–20%. Dùng bảng giá.

Reasoning:

- **Interleaved thinking** bật mặc định trên GLM-5.x và 4.7. Thinking blocks phải được trả lại cùng tool results.
- **Preserved thinking** tắt mặc định trên API chuẩn, bật mặc định trên Coding Plan. Bật bằng `"clear_thinking": false`. Khi bật, phải gửi lại `reasoning_content` **đầy đủ, không sửa**, đúng thứ tự ban đầu.
- GLM-5.3 và GLM-5.3-Flash luôn bật thinking, không tắt được.

### MiniMax

| Mục | Chi tiết |
|---|---|
| Cơ chế | Tự động theo prefix, áp dụng khi input từ 512 token |
| Model hỗ trợ | MiniMax-M3.1-Flash, M3, M2.7, M2.5, M2.1. Bản M2 gốc chưa rõ (bản tiếng Anh và tiếng Trung của docs nói khác nhau) |
| TTL, phí ghi | Không công bố trên trang tiếng Anh |
| Giá | M3.1-Flash: input $0,15, cache read $0,015, output $0,60 / 1M token |
| Usage | `cache_read_input_tokens` (Anthropic-compatible), `prompt_tokens_details.cached_tokens` (OpenAI-compatible; có báo cáo từ bên thứ ba rằng field này đôi khi thiếu) |
| Endpoint Anthropic | Có, `https://api.minimax.io/anthropic` |

Reasoning (M3.1-Flash): phải giữ **toàn bộ** response message, kể cả `reasoning_content`, trong lịch sử và gửi lại. Docs ghi rằng bỏ phần này "phá vỡ tính liên tục".

### ByteDance Doubao (Volcengine Ark / BytePlus ModelArk)

Ở Doubao, trước hết cần chọn giữa cache tự động và cache explicit của Responses API. Lựa chọn này quyết định cách nối các lượt, thời hạn cache và phí lưu trữ.

**Implicit cache**: tự động trên model hỗ trợ, không tắt được, áp dụng cho Chat, Responses, Messages và Batch. Không có phí lưu trữ. Ngưỡng mặc định 1.024 token, tuỳ model từ 128 đến 8.960.

**Explicit cache**: chỉ có trên Responses API, phải bật bảng giá "推理（缓存）定价" trước, mỗi request đặt `caching.type = "enabled"`. Hai chế độ:

- **Session**: nối các lượt qua `previous_response_id`. Phù hợp với hội thoại nhiều lượt.
- **Prefix**: request tạo cache với `caching.prefix = true` (không sinh output, tối thiểu 256 token), các request sau tham chiếu response ID của nó. Phù hợp với một system prompt dài dùng cho nhiều phiên.

TTL của explicit đặt qua `expire_at` (Unix seconds UTC), mặc định 3 ngày sau khi tạo, tối đa 7 ngày. Hit **không** gia hạn TTL. Phí lưu trữ tính theo giờ tròn, lẻ giờ tính tròn lên.

Giá (CNY / 1M token, online thường):

| Model | Input | Cache hit | Lưu trữ / giờ | Output |
|---|---|---|---|---|
| doubao-seed-2.1-pro | 6,00 | 1,20 | 0,017 | 30,00 |
| doubao-seed-2.1-turbo | 3,00 | 0,60 | 0,017 | 15,00 |

Ràng buộc khi dùng explicit cache trong multi-turn:

- Không dùng chung với `instructions`.
- `thinking` phải giữ nhất quán qua các lượt.
- `tools` chỉ nên đặt ở lượt đầu.
- Nên cách khoảng 100 ms giữa các lượt.
- Implicit và explicit không cùng áp dụng cho một response.

Usage: `prompt_tokens_details.cached_tokens` (Chat), `input_tokens_details.cached_tokens` (Responses), `cache_read_input_tokens` (Messages, `cache_creation_input_tokens` luôn là 0). Với Batch, `cached_tokens` luôn là 0, nên không đo được hit rate ở đó.

## 2.5 xAI Grok

| Mục | Chi tiết |
|---|---|
| Cơ chế | Tự động, cache prefix tính từ đầu mảng `messages`. Sửa bất kỳ message cũ nào làm vỡ cache |
| Routing | Header `x-grok-conv-id` (Chat Completions, gRPC metadata) hoặc `prompt_cache_key` trong body (Responses API). Cache entry nằm theo từng server, nên dùng một UUID ổn định cho cả phiên. Muốn cố ý bỏ cache thì đổi ID |
| Tối thiểu, TTL | Không công bố. Entry có thể bị evict bất kỳ lúc nào |
| Usage | `prompt_tokens_details.cached_tokens` (Chat), `input_tokens_details.cached_tokens` (Responses), `cached_prompt_text_tokens` (gRPC SDK) |

Giá (USD / 1M token, standard tier):

| Model | Input | Cached input | Output | Context |
|---|---|---|---|---|
| grok-4.7 | $2,00 | $0,50 | $6,00 | 500K |
| grok-4.6 | $2,00 | $0,50 | $6,00 | 500K |
| grok-4.5 | $2,00 | $0,30 | $6,00 | 500K |
| grok-4.3 | $1,25 | $0,20 | $2,50 | 1M |
| grok-build-0.1 | $1,00 | $0,20 | $2,00 | 256K |

Khi prompt vượt 200K token, **mọi** token trong request được tính theo giá long-context (grok-4.7: input $4, cached $1, output $12). Một hội thoại vượt mốc này sẽ thấy chi phí mỗi request tăng gấp đôi đột ngột, kể cả với phần đã cache.

grok-4-fast, grok-4.1-fast, grok-4-0709, grok-3 và grok-code-fast-1 đã ngừng từ 15/05/2026. Slug cũ được chuyển hướng và tính giá grok-4.3.

**Conversation state**: Responses API hỗ trợ `previous_response_id`, response lưu 30 ngày. Với `store: false`, gửi lại `response.output` làm `input`. Với reasoning model, yêu cầu `include: ["reasoning.encrypted_content"]` (grok-4.7 luôn trả về) và gửi lại y nguyên. Trang multi-turn của xAI ghi việc bỏ encrypted reasoning là **nguyên nhân cache miss hàng đầu**. Prefix có thể cache bao gồm cả tool results.

Endpoint tương thích Anthropic của xAI: trạng thái chưa xác minh được từ docs chính thức. Base URL tương thích OpenAI là `https://api.x.ai/v1`.

## 2.6 Google Gemini

Với Gemini, bắt đầu bằng việc xác định API ứng dụng đang gọi. **generateContent** là API stateless, hiện được docs gắn nhãn “Legacy”. **Interactions API** là API mặc định trong docs mới và hỗ trợ state phía server qua `previous_interaction_id`. Cấu hình caching và cách gửi lại reasoning của hai API được trình bày riêng bên dưới.

### Implicit caching

Bật mặc định cho Gemini 2.5 trở lên, không cần cấu hình, hoạt động cả với hội thoại stateful (`previous_interaction_id`) lẫn stateless.

| Model | Ngưỡng tối thiểu |
|---|---:|
| Gemini 4 Argon, Gemini 3.8 Flash, Gemini 3.6 Flash, Gemini 3.1 Pro Preview | 4.096 tokens |
| Gemini 2.5 Pro, Gemini 2.5 Flash | 2.048 tokens |

Với ngưỡng 4.096 token, một system prompt dài 3.000 token chưa đủ để có cache trên các model Gemini 3.x và 4 trong bảng. Hội thoại phải dài thêm mới đạt ngưỡng, dù cùng prefix đó có thể đã đủ cho model ở provider khác.

Hai gợi ý chính thức để tăng hit: đặt nội dung lớn và dùng chung ở đầu prompt, và gửi các request có prefix giống nhau trong khoảng thời gian ngắn. TTL của implicit cache không được công bố.

**Quy tắc 2.050 token trên Gemini 4 (Argon trở đi).** Trong multi-turn và agent loop, Gemini chỉ cache toàn bộ request trước khi request mới thêm **ít nhất 2.050 token**. Output, thoughts và function results đều được tính vào con số này. Nếu thêm ít hơn, 2.050 token cuối của request trước bị xử lý lại với giá input chuẩn.

```text
Request n:    [............ prefix 80.000 token ............]
Request n+1:  [............ prefix 80.000 token ............][+300 token tool result]

Thêm 300 < 2.050  →  77.950 token đầu: cache hit
                      2.050 token cuối của request n + 300 token mới: giá chuẩn
```

Với agent loop có nhiều tool result nhỏ, mỗi bước vẫn xử lý lại khoảng 2.000 token dù prefix giữ nguyên. Theo giá Gemini 4 Argon $4/MTok từ 01/12/2026, phần này tốn khoảng $0,008 mỗi bước. Khi dự toán, nhân khoản đó với số bước trung bình và số task sẽ cho thấy tác động lên tổng chi phí.

### Explicit caching (`cachedContents`)

Explicit caching tạo một đối tượng cache riêng bằng `POST /v1beta/cachedContents` (SDK: `client.caches.create`), chứa nội dung (file, tài liệu, `systemInstruction`) và `ttl` (ví dụ `"300s"`). Request sau tham chiếu cache qua field `cachedContent: "cachedContents/{id}"`. TTL được gia hạn bằng `PATCH` với `ttl` mới.

Giới hạn quan trọng:

- **Chỉ có trên generateContent**, không có trên Interactions API.
- **Không hỗ trợ Gemini 4 Argon trở đi.** Hướng dẫn migration của Google yêu cầu gỡ mọi lời gọi tạo hoặc tham chiếu `cached_content` khi chuyển sang Argon.
- Có phí lưu trữ theo giờ, tính trên số token được cache.

Explicit cache trên Gemini dễ tính hiệu quả nhất khi một khối nội dung lớn — bộ tài liệu, codebase hoặc video — được nhiều request dùng lại trong vài giờ. Lịch sử hội thoại thay đổi liên tục cần cân nhắc thêm việc quản lý cache. Chẳng hạn, với một tài liệu 100K token trên Gemini 3.8 Flash:

- Phí lưu trữ: 100K × $0,50/MTok/giờ = $0,05 mỗi giờ.
- Mỗi request đọc cache tiết kiệm: 100K × ($0,75 − $0,075)/MTok ≈ $0,068.

Chỉ cần khoảng một request mỗi giờ là đủ bù phí lưu trữ. Ít hơn thế thì implicit cache (không phí lưu trữ) có lợi hơn.

### Giá (USD / 1M token, standard tier)

| Model | Input | Cached input | Lưu trữ explicit / giờ | Output |
|---|---|---|---|---|
| gemini-4-argon | $2,00 → $4,00 từ 01/12/2026 | $0,10 → $0,20 (giảm 95%) | Không hỗ trợ explicit | $10,00 → $20,00 |
| gemini-3.8-flash | $0,75 → $1,50 từ 01/01/2027 | $0,075 → $0,15 | $0,50 → $1,00 | $3,75 → $7,50 |
| gemini-3.6-flash | $0,75 → $1,50 từ 01/01/2027 | $0,075 → $0,15 | $0,50 → $1,00 | $3,75 → $7,50 |
| gemini-3.5-flash-lite | $0,30 | $0,03 | $1,00 | $2,50 |
| gemini-3.1-pro-preview | $2,00 (≤200K) / $4,00 (>200K) | $0,20 / $0,40 | $4,50 | $12,00 / $18,00 |

Giá hiện tại của Argon, 3.8 Flash và 3.6 Flash là giá giới thiệu có ngày hết hạn. Lập dự toán chi phí theo giá sau mốc.

### Usage

- generateContent: `usageMetadata.cachedContentTokenCount`.
- Interactions API: `usage.total_cached_tokens`.

### Thought signatures: reasoning trong multi-turn

Gemini trả trạng thái reasoning dưới dạng **thought signature**, một chuỗi mã hoá mà client phải gửi lại. Cách xử lý khác nhau theo API.

**generateContent** (bạn tự giữ lịch sử trong `contents`):

- Signature gắn vào part của response, thường là part `functionCall`. Part text cuối cũng có thể mang signature.
- **Gemini 3: kiểm tra bắt buộc trong "lượt hiện tại".** Lượt hiện tại gồm mọi bước `functionCall` / `functionResponse` kể từ user message dạng text gần nhất. Thiếu signature ở `functionCall` đầu tiên của bất kỳ bước nào trong lượt hiện tại sẽ bị **400**, thông báo bắt đầu bằng `Function call is missing a thought_signature in functionCall parts`. Các lượt cũ hơn không bị kiểm tra.
- **Gemini 2.5**: thiếu signature không gây lỗi nhưng làm giảm chất lượng.
- **Function call song song**: chỉ `functionCall` đầu tiên có signature. Thứ tự gửi lại phải là `FC1 + signature, FC2, FR1, FR2`. Gửi xen kẽ `FC1 + signature, FR1, FC2, FR2` sẽ bị 400.
- **Function call tuần tự**: mỗi bước có signature riêng, phải gửi lại tất cả cho đến khi user gửi message text mới.
- Signature trên part text: không bắt buộc, nhưng Google khuyến nghị gửi lại.
- **Từ Gemini 3.5 Flash**, model dùng reasoning của mọi lượt trước khi signature có trong lịch sử. Docs yêu cầu gửi **toàn bộ lịch sử, không sửa đổi**, kể cả signature. SDK tự làm việc này khi dùng chat helper.
- Lịch sử chứa function call do model khác sinh ra (không có signature): bản docs generateContent lưu trữ ghi hai giá trị giữ chỗ để bỏ qua kiểm tra, `context_engineering_is_the_way_to_go` và `skip_thought_signature_validator`, kèm khuyến cáo chỉ dùng khi không còn cách nào khác.
- Endpoint tương thích OpenAI của Gemini 3 cũng hỗ trợ thought signature trong chat completions.

Một lỗi đáng kiểm tra ở gateway hoặc thư viện đa provider là mất signature trong quá trình chuyển `functionCall` sang `tool_calls` rồi chuyển ngược. Lượt đầu vẫn chạy vì chưa cần gửi lại signature; đến lượt tiếp theo trong tool loop, API mới báo 400. Đối chiếu response gốc với request thực sự gửi đi sẽ thấy field bị mất ở bước nào.

**Interactions API**:

- **Stateful** (`store: true` + `previous_interaction_id`): server tự giữ thought và signature, client không phải làm gì.
- **Stateless** (`store: false`): signature nằm trong các step `thought` riêng của mảng `steps`, không gắn vào input, output hay function call. Phải gửi lại **mọi** `thought` step đúng như nhận được, không xoá, không sửa. Step của built-in tool (như Google Search) cũng có signature riêng và phải gửi lại.
- Khi đổi model trong phiên, docs vẫn yêu cầu gửi lại thought step của model trước. Backend tự xử lý tương thích.
- `store: false` không dùng được với background execution và không dùng được `previous_interaction_id`.
- Interaction được lưu 55 ngày (Paid Tier) hoặc 1 ngày (Free Tier). Paid Tier có thể rút xuống 7, 14 hoặc 28 ngày, và khi đó không tiếp tục được các hội thoại cũ hơn mốc đó bằng `previous_interaction_id`.

Hướng dẫn migration sang Gemini 4 Argon khuyến nghị dùng `previous_interaction_id` cho multi-turn và function calling. Ở chế độ này, server quản lý thứ tự thought và signature, giảm phần client phải tự ghép. Theo docs, cách lưu state này cũng giúp server tối ưu caching.

## 2.7 Self-hosted: vLLM, SGLang và chat template

Khi tự host, việc giữ nguyên JSON request chưa đủ để kiểm tra cache. Giữa JSON và model còn có **chat template**, nơi quyết định chuỗi token thực sự được đưa vào inference engine.

Chat template (Jinja trong `tokenizer_config.json`) biến danh sách message thành chuỗi token. Nếu template render lịch sử khác với cách model đã thấy khi sinh ra nó, prefix thay đổi dù client gửi lại đúng y nguyên.

Ví dụ điển hình là reasoning model. Template của Qwen3 và các model tương tự chỉ render reasoning của các assistant turn **sau** user message cuối cùng. Khi có user message mới, khối `<think>` của các lượt trước biến mất khỏi chuỗi render:

```text
Lượt 2, khi sinh:      ... <|assistant|><think>phân tích A...</think>Trả lời A
Lượt 3, khi render lại: ... <|assistant|>Trả lời A        ← <think> đã bị bỏ
                                         ↑
                         prefix khác từ đây, mọi thứ sau phải prefill lại
```

Các báo cáo đã ghi nhận:

- Qwen3 issue #1826: với `enable_thinking=false`, template chỉ thêm khối think rỗng vào assistant turn cuối, làm KV không tái dùng được.
- LM Studio mlx-engine issue #176: lượt cũ render không có `<think>` trong khi prompt sinh có, nên mọi token đã sinh bị xử lý lại hai lần.
- vLLM 0.31 đổi tên `reasoning_content` thành `reasoning` trước khi render template, khiến một số template Qwen mất reasoning cũ.
- Template kỳ vọng `arguments` của tool call là object có thể lỗi khi nhận chuỗi JSON kiểu OpenAI, thường xuất hiện ở lượt thứ hai của agent loop.

Một template có thể chủ ý bỏ reasoning cũ để tiết kiệm context hoặc khớp cách model được huấn luyện. Khi cần prefix caching, phải tính thêm tác động của hành vi đó. Có ba việc nên cân nhắc:

1. Chấp nhận mất cache phần sau user message cuối ở mỗi lượt mới. Trong một tool loop (không có user message mới), prefix vẫn ổn định.
2. Dùng template có chế độ giữ reasoning toàn lịch sử, nếu model hỗ trợ (một số nhà cung cấp gọi là "full history" so với "suffix-only").
3. Kiểm tra bằng cách render hai request liên tiếp ra chuỗi token và so sánh, thay vì tin vào JSON message.

---

# Phần 3 — Dựng multi-turn conversation không làm vỡ cache

Trở lại ứng dụng sửa hàm đăng nhập ở đầu bài. Sau vài lượt, nó đã có system prompt, tool definitions, code đã đọc, kết quả test và các yêu cầu bổ sung của người dùng. Conversation manager phải lưu đủ những phần đó, dựng lại request đúng định dạng và quyết định lúc nào nên rút gọn. Phần này đi qua các quyết định ấy theo thứ tự triển khai.

<a id="31-bất-biến-duy-nhất-append-only"></a>

## 3.1 Giữ prefix ổn định giữa các lượt

Giữa hai lần compact, cách đơn giản để giữ prefix là lưu lịch sử theo kiểu append-only: mỗi lượt chỉ nối thêm nội dung. Khi render request mới, phần input của request trước vẫn ở đầu, tiếp đến là output vừa nhận và input mới:

```text
Request n:     [P ─────────────────────────────][output n]
Request n+1:   [P ─────────────────────────────][output n][input mới]
                └───── phải giống hệt ─────┘
```

Nếu phần đầu giữ nguyên, nó đủ điều kiện để được tái dùng; hit thực tế còn phụ thuộc ngưỡng tối thiểu, TTL, routing và chính sách của provider. Nếu lịch sử đổi ở một vị trí, phần từ đó trở đi không còn khớp cache cũ.

Để kiểm tra, đối chiếu tools, system và các message cũ trong hai request liên tiếp. Bỏ qua marker `cache_control` khi so nội dung vì marker có thể dịch chuyển. Với hệ thống self-hosted, so cả chuỗi token sau chat template. Body JSON giúp tìm chỗ nội dung bị sửa, nhưng thứ tự key của lớp vỏ HTTP không tự nó cho biết prefix mà model nhận.

Theo dõi dữ liệu qua cả đường đi: code ghép prompt, SDK, framework agent, middleware, proxy, database và chat template. Chẳng hạn, code đã lưu đúng response nhưng proxy lại bỏ thinking blocks thì request cuối cùng vẫn thay đổi. Điểm khác biệt đầu tiên trong phần lịch sử chung là nơi bắt đầu điều tra.

## 3.2 Bố cục prompt theo độ ổn định

System prompt có thể dùng qua hàng nghìn phiên, hồ sơ user giữ nguyên trong một phiên, còn kết quả RAG thay đổi theo câu hỏi. Đặt chúng theo thứ tự đó giúp mỗi nhóm request chia sẻ được một prefix dài nhất có thể. Bố cục tham khảo:

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Tool definitions        không bao giờ đổi trong phiên    │ ← breakpoint 1 (Claude)
│ 2. System prompt tĩnh       đổi khi deploy phiên bản mới    │
├─────────────────────────────────────────────────────────────┤
│ 3. Ngữ cảnh của phiên       đổi theo phiên (hồ sơ user,     │ ← breakpoint 2 (tuỳ chọn)
│    tài liệu được gắn vào phiên, project context)            │
├─────────────────────────────────────────────────────────────┤
│ 4. Lịch sử hội thoại        chỉ nối thêm                    │
│    user₁, assistant₁, tool_result, ..., assistantₙ          │
├─────────────────────────────────────────────────────────────┤
│ 5. Lượt hiện tại            user message mới                │ ← breakpoint cuối
├─────────────────────────────────────────────────────────────┤
│ 6. Phần đuôi biến động      giờ hiện tại, kết quả RAG riêng │   (không cache)
│    cho lượt này, reminder                                   │
└─────────────────────────────────────────────────────────────┘
```

Nếu system prompt dùng chung cho nhiều phiên, đặt ngữ cảnh riêng của phiên ở message đầu tiên. Khi đó tools + system có thể được tái dùng giữa các request trong phạm vi chia sẻ cache mà provider cho phép; phần hồ sơ và tài liệu riêng vẫn thuộc prefix của từng phiên.

Dữ liệu mới ở đuôi sẽ trở thành lịch sử ở lượt kế tiếp. Giả sử lượt 5 gửi “giờ hiện tại: 14:02”. Đến lượt 6, giữ nguyên dòng đó ở vị trí cũ và nối giờ mới vào lượt 6. Cập nhật lại thành “14:05” trong message cũ sẽ thay prefix của tất cả nội dung phía sau.

Thứ tự này cũng hữu ích với OpenAI, DeepSeek, Grok và các provider tự tìm cache: prefix chung càng dài, phần có thể tái dùng càng lớn.

## 3.3 Những thứ âm thầm làm vỡ cache

Khi cache hit thấp, bắt đầu từ request thực tế. Bảng dưới đây ghép từng kiểu thay đổi với nguyên nhân và cách xử lý để bạn đối chiếu với code đang chạy.

| Lỗi | Vì sao vỡ | Cách sửa |
|---|---|---|
| `datetime.now()` trong system prompt | Prefix đổi mỗi request | Bỏ đi, hoặc đưa giờ vào phần đuôi user message. Nếu chỉ cần ngày, cập nhật bằng message mới khi ngày đổi |
| UUID, request ID, trace ID trong nội dung prompt | Mỗi request là duy nhất | Đưa vào metadata/header, không đưa vào prompt |
| User ID, tên user nhúng vào system prompt | Mỗi user một prefix, không chia sẻ được giữa các user | Đưa vào message đầu của phiên |
| Nhánh điều kiện trong system prompt (`if flag: system += ...`) | Mỗi tổ hợp flag là một prefix | Cố định nội dung, đưa phần điều kiện xuống messages |
| Danh sách tools xây theo quyền user hoặc theo lượt | Tools nằm ở vị trí 0 | Khai báo đủ tool từ đầu phiên, chặn quyền ở tầng thực thi tool. Dùng cơ chế thêm tool kiểu append của provider nếu có |
| Tools không sort | Thứ tự phụ thuộc thứ tự import, dict, plugin load | Sort theo tên trước khi gửi |
| `json.dumps` không `sort_keys`, lặp qua `set` | Serialize không tất định | `sort_keys=True`, `separators` cố định, chuyển `set` thành list đã sort |
| Số token còn lại, "bạn đã dùng X/Y bước" chèn vào lượt cũ | Lượt cũ bị sửa mỗi request | Chỉ nối thêm ở cuối, không quay lại sửa |
| Cửa sổ trượt: giữ N message cuối | Message đầu tiên đổi mỗi lượt, prefix đổi mỗi lượt | Dùng ngưỡng: để lịch sử lớn đến ngưỡng rồi compact một lần ([§3.8](#38-khi-context-quá-dài-trim-summarize-compact)) |
| Xoá tool result cũ ở phía client mỗi lượt | Sửa lịch sử ở giữa | Dùng dọn context phía server (Claude context editing) hoặc gom thành một lần compact |
| Lưu lịch sử dạng text rồi dựng lại message | Mất thinking/reasoning, mất cấu trúc block, khác whitespace | Lưu nguyên object response ([§3.4](#34-lưu-transcript-đủ-để-phát-lại)) |
| Parse arguments của tool call thành dict rồi serialize lại | `{"a":1,"b":2}` thành `{"b": 2, "a": 1}` | Lưu chuỗi arguments gốc, gửi lại chuỗi đó |
| `.strip()` output của assistant, chuẩn hoá Unicode, đổi `\r\n` | Byte khác token khác | Không biến đổi nội dung đã nhận từ model |
| Few-shot examples xoay vòng hoặc chọn ngẫu nhiên | Đầu prompt đổi theo request | Cố định tập ví dụ cho mỗi phiên bản prompt |
| Đưa kết quả RAG vào system prompt | System đổi mỗi lượt | Đưa kết quả RAG vào user message của lượt đó |
| Ảnh qua signed URL sinh lại mỗi lượt mà trả về byte khác | Nội dung ảnh khác | Upload một lần (Files API) và dùng `file_id`, hoặc base64 cố định |
| Đổi model theo độ khó của câu hỏi giữa phiên | Cache gắn với model | Giữ một model cho vòng lặp chính, gọi model khác như sub-agent |
| Đổi thinking / effort theo lượt | Được render vào prompt | Cố định theo route; Claude có system message đổi effort mà không vỡ cache |
| A/B test prompt theo request thay vì theo phiên | Hai biến thể xen kẽ trong một phiên | Gán biến thể theo phiên |
| Framework tự chèn message (memory, "summary so far") ở đầu messages | Đầu lịch sử đổi | Kiểm tra request thực tế framework gửi đi, tắt hoặc chuyển phần chèn xuống cuối |
| Nhiều workspace / API key cho cùng một prefix | Cache cô lập theo workspace (Claude) hay organization | Gom traffic cùng prefix về một workspace |

Nếu code ghép prompt trông đúng, kiểm tra tiếp request ở tầng HTTP và cấu hình workspace/API key. Framework có thể chèn thêm message sau bước ghép prompt; hai request giống nhau cũng có thể thuộc hai phạm vi cache khác nhau.

<a id="34-lưu-transcript-log-không-phải-view"></a>

## 3.4 Lưu transcript đủ để phát lại

Một bảng `messages(id, role, text, created_at)` đủ để vẽ các bong bóng chat. Nhưng khi dùng bảng đó để dựng request tiếp theo, ứng dụng có thể thiếu những phần không hiển thị trên giao diện:

- Thinking blocks và signature (Claude), reasoning items với `encrypted_content` (OpenAI, xAI), `reasoning_content` (DeepSeek, Kimi, GLM, MiniMax).
- Cấu trúc nhiều block trong một message (text + tool_use + thinking).
- Chuỗi arguments gốc của tool call.
- Compaction blocks (Claude), field `phase` (OpenAI).
- Chính xác từng byte của text.

Nên lưu transcript dưới dạng log chứa các object cần để phát lại request, rồi dựng giao diện từ log. Như vậy UI có thể ẩn reasoning, định dạng text hoặc gộp các block để dễ đọc mà không thay đổi dữ liệu gửi lại cho model.

Mô hình dữ liệu tối thiểu:

```text
conversation
  id
  provider, model                      # cố định cho cả phiên
  prompt_version                       # phiên bản tools + system, cố định cho cả phiên
  cache_routing_key                    # prompt_cache_key / x-grok-conv-id
  created_at

conversation_event                     # append-only, không UPDATE, không DELETE
  conversation_id
  seq                                  # thứ tự tuyệt đối
  kind: request_input | response_output | compaction_boundary
  payload_raw                          # JSON đúng như gửi đi / nhận về
  payload_hash                         # để phát hiện ai đó sửa
  usage_raw                            # usage object của response
```

Khi dựng request, lấy `tools` và `system` theo `prompt_version` đã lưu cùng phiên, rồi nối `payload_raw` theo `seq`. Nếu vừa deploy prompt mới, các phiên đang mở vẫn dùng phiên bản cũ; phiên mới mới nhận bản cập nhật. Cách ghim phiên bản này giữ cache và tránh đổi chỉ dẫn giữa một cuộc hội thoại đang dở.

Với hệ thống nhiều provider, giữ payload gốc bên cạnh dữ liệu chung của ứng dụng. Khi chuyển một transcript từ Claude sang OpenAI, cần chuyển đổi định dạng, dựng cache mới và xử lý reasoning không tương thích. Nên coi đây là một bước chuyển phiên có xử lý riêng thay vì để xảy ra trong mỗi lần dựng request.

## 3.5 Thông tin động: đặt ở đâu

Có những dữ liệu phải đổi để câu trả lời còn đúng: ngày hiện tại, quyền vừa được cập nhật, kết quả tìm kiếm mới. Hãy ghi nhận thay đổi bằng message mới, giữ nguyên các bản đã xuất hiện trong lịch sử.

| Loại thông tin | Đặt ở đâu | Ghi chú |
|---|---|---|
| Ngày hiện tại | Message đầu phiên, hoặc message mới khi ngày đổi | Đa số tác vụ chỉ cần ngày, không cần giây |
| Giờ chính xác đến phút | Phần đuôi user message của lượt cần nó | Nằm lại trong lịch sử, đừng sửa ở lượt sau |
| Hồ sơ user, quyền | Message đầu phiên | Nếu đổi giữa phiên, nối message mới mô tả thay đổi |
| Kết quả RAG cho câu hỏi hiện tại | Trong user message của lượt đó, trước câu hỏi | Không đưa vào system |
| Chỉ dẫn operator đổi giữa phiên ("chuyển sang chế độ ngắn gọn") | Message mới ở cuối. Claude: `role: "system"` giữa phiên. Provider khác: user message có đánh dấu rõ nguồn | Không sửa system prompt |
| Reminder mỗi bước trong tool loop | Text block sau tool results, giữ nguyên các bản cũ. Claude có `clear_at` | Không chèn rồi xoá |
| Trạng thái task (todo list, tiến độ) | Nối thêm bản cập nhật ở cuối, hoặc để model đọc/ghi file qua tool | Manus gọi là "recitation": nhắc lại mục tiêu ở cuối context giúp model giữ hướng trong vòng lặp dài |

Khi ghi chỉ dẫn của operator vào user message, cần phân biệt nó với tài liệu được paste hoặc nội dung lấy từ web. Nhãn văn bản có thể bị nội dung khác bắt chước. Nếu provider hỗ trợ kênh chỉ dẫn riêng, như `role: "system"` giữa phiên ở các model Claude tương ứng, ứng dụng nên dùng kênh đó. Quyền thực thi tool vẫn cần được kiểm tra trong code, kể cả khi prompt đã mô tả quyền.

## 3.6 Tools trong hội thoại dài

Tool definitions thường nằm ở đầu prefix, nên thay đổi danh sách này có thể làm mất phần cache lớn. Cách triển khai dễ kiểm soát là chọn tập tool khi tạo phiên, cố định thứ tự theo tên và serialize schema nhất quán. Với Claude preserved thinking, thêm tool bằng cách sửa phần đầu request còn ảnh hưởng hiệu lực của thinking blocks đã có.

Một tool đã khai báo vẫn có thể chưa được phép chạy ở trạng thái hiện tại. Manus xử lý bằng cách giữ định nghĩa và dùng logit masking khi decode. Với API thương mại, các lựa chọn gồm `tool_choice`, chỉ dẫn mới trong messages và kiểm tra ở tầng thực thi. Chẳng hạn, tool có thể trả lỗi rõ ràng khi bị gọi sai trạng thái. Ở Claude, đổi `tool_choice` giữ được cache tools và system. Khi có quá nhiều tool, có thể dùng tool search với `defer_loading` trên Claude hoặc thiết kế lại thành một tập tool gọn hơn.

Kích thước kết quả tool cũng quyết định tốc độ dài lên của lịch sử. Một kết quả 50K token ở lượt 3 sẽ theo các request sau cho đến khi bị rút gọn. Nếu có thể đọc lại qua tool, hãy lưu kết quả lớn vào file hoặc storage và trả về đường dẫn cùng phần tóm tắt. Model vẫn có cách lấy chi tiết khi cần mà context chính chỉ mang phần đang hữu ích.

Với tool call song song, mỗi call phải có kết quả tương ứng, kể cả khi tool báo lỗi. Cách đóng gói phụ thuộc API: Claude nhận các `tool_result` trong một user message kế tiếp. Tách sai cấu trúc có thể gây lỗi hoặc ảnh hưởng việc model tiếp tục gọi tool song song. Hàm dựng tool results nên chịu trách nhiệm chuyển sang đúng định dạng của provider.

## 3.7 Reasoning và thinking blocks

Text hiển thị cho người dùng có thể chỉ là một phần response. Reasoning items, thinking blocks hoặc signatures còn mang trạng thái để tiếp tục lượt sau. Cách gửi lại chúng thay đổi theo API và thế hệ model:

| Provider | Gửi lại? | Hậu quả khi làm sai |
|---|---|---|
| Claude | Có, nguyên block kèm `signature`. API tự bỏ block model hiện tại không đọc được (không tính phí) | Sửa lịch sử trước block: 400 với tài khoản mới trên model có preserved thinking, hoặc block bị bỏ với `drop_block`. Bỏ block ở giữa: block sau mất hiệu lực |
| OpenAI Responses | Có, mọi item trong `output`, gồm reasoning item và `encrypted_content` khi `store: false` | Prefix khác, mất cache; model mất phần suy luận đã làm |
| xAI | Có, `reasoning.encrypted_content` y nguyên | Nguyên nhân cache miss số một theo docs xAI |
| Gemini generateContent | Có, mọi part với `thoughtSignature` y nguyên, đúng thứ tự (song song: các FC trước, các FR sau) | Gemini 3: 400 nếu thiếu ở function call trong lượt hiện tại. Gemini 2.5: giảm chất lượng |
| Gemini Interactions | Stateful: server tự lo. Stateless: gửi lại mọi `thought` step | Model mất reasoning của các bước trước |
| DeepSeek | Có tools: bắt buộc gửi lại mọi lượt. Không tools: bị bỏ qua | 400 nếu thiếu khi có tools |
| Kimi K3 / K3.1 | Có, toàn bộ assistant message | Bắt buộc trong tool loop |
| Kimi K2.6 | Theo `thinking.keep` | Mặc định bỏ |
| GLM 5.x / 4.7 | Interleaved: gửi cùng tool results. Preserved (`clear_thinking: false`): gửi nguyên văn, đúng thứ tự | Sai thứ tự hoặc sửa nội dung: hỏng tính liên tục |
| MiniMax M3.x | Có, toàn bộ message | "Phá vỡ tính liên tục" |
| Self-hosted (Qwen3 và tương tự) | Template thường tự bỏ reasoning của lượt trước user message cuối | Mất cache phần đó mỗi lượt mới ([§2.7](#27-self-hosted-vllm-sglang-và-chat-template)) |

Ở tầng lưu trữ, giữ nguyên output của provider để không mất thông tin trước khi biết có cần dùng lại hay không. Ở tầng dựng request, gửi lại các block hoặc item theo quy tắc của API trong bảng. Nếu một API yêu cầu bỏ phần nào, áp dụng quy tắc đó nhất quán ở mọi lượt.

Khi chuyển sang model khác để fallback hoặc xử lý một bước riêng, kiểm tra khả năng nhận reasoning cũ. Claude tự bỏ block không đọc được; provider khác có thể xử lý khác hoặc trả lỗi. Việc lưu payload gốc giúp bạn xử lý bước chuyển này mà không phải đoán lại những field đã bị lược bỏ.

## 3.8 Khi context quá dài: trim, summarize, compact

Một phiên có thể giữ cache tốt nhưng vẫn quá dài để model làm việc hiệu quả. Khi chọn cách rút gọn, cần xem cả lượng context giảm được, phần cache phải dựng lại và thông tin có nguy cơ mất:

| Chiến lược | Cách làm | Tác động lên cache | Rủi ro |
|---|---|---|---|
| **Cửa sổ trượt** (giữ N message cuối, mỗi lượt) | Bỏ message cũ nhất mỗi khi thêm mới | Vỡ ở **mọi** request | Mất chi tiết đầu hội thoại; tách cặp tool_use / tool_result |
| **Trim theo ngưỡng** | Khi chạm ngưỡng (ví dụ 80% context), bỏ một khối lớn lượt cũ, còn khoảng 40% | Vỡ một lần, rồi ổn định đến lần trim sau | Mất chi tiết đầu hội thoại |
| **Summarize theo ngưỡng** (client) | Khi chạm ngưỡng, tóm tắt toàn bộ hội thoại thành một message, bắt đầu request mới với tóm tắt + lượt mới | Vỡ một lần; tóm tắt thành prefix mới | Tóm tắt bỏ sót; "summary drift" qua nhiều lần tóm tắt |
| **Compaction phía server** (Claude, OpenAI) | Provider tự tóm tắt khi chạm ngưỡng | Vỡ một lần ở điểm compact | Ít kiểm soát nội dung tóm tắt (Claude cho truyền `instructions`) |
| **Context editing phía server** (Claude) | Xoá tool result / thinking cũ trên bản sao phía server | Phần bị dọn thay đổi; dọn theo ngưỡng để giảm số lần | Model mất kết quả tool cũ, phải gọi lại nếu cần |
| **External memory** (file, memory tool, note) | Model ghi trạng thái quan trọng ra ngoài, đọc lại khi cần | Không ảnh hưởng: chỉ nối thêm tool call và kết quả | Model phải biết khi nào cần ghi và đọc |
| **Sub-agent** | Giao việc đọc nhiều cho agent con, nhận về tóm tắt 1–2K token | Context chính chỉ nối thêm tóm tắt | Chi phí của agent con; mất chi tiết không có trong tóm tắt |

Hai ngưỡng tách biệt giúp tránh compact liên tục: một ngưỡng kích hoạt và một mức context muốn giữ lại. Ví dụ, chạm 150K thì rút xuống khoảng 30K. Khoảng cách này, thường gọi là *hysteresis*, để lại chỗ cho nhiều lượt mới trước lần compact tiếp theo.

Thời điểm compact cũng cần rõ ràng. Nếu assistant đã phát `tool_use` và đang chờ `tool_result`, hãy hoàn tất cặp đó trước. Rút gọn ở ranh giới lượt giúp tránh mang một tool call dở dang sang lịch sử mới.

Với cách tóm tắt toàn bộ lịch sử ở client, docs của Anthropic khuyến nghị bắt đầu lại bằng một message tóm tắt cộng lượt mới. Giữ nguyên vài lượt cuối (*keep-tail*) cần xử lý thêm preserved thinking: các thinking block ấy được sinh với prefix cũ, nên có thể bị từ chối khi đặt sau bản tóm tắt. Khi dùng cách này, phải bỏ thinking tương ứng hoặc đặt `drop_block`.

Chất lượng bản tóm tắt quyết định phần công việc nào tiếp tục được. Chỉ dẫn cho summarizer nên nêu rõ những gì phải giữ: mục tiêu hiện tại, ràng buộc người dùng, quyết định đã chốt, file đã sửa, việc còn dở và lỗi đã gặp. Với cách thay toàn bộ lịch sử, những chi tiết bị bỏ sót sẽ không còn trong context của phiên mới.

Đây cũng là dịp hợp nhất yêu cầu sau nhiều lần bổ sung. Thay vì kể lại “đầu tiên đã thử gì, sau đó đã nói gì”, bản tóm tắt có thể ghi đặc tả hiện tại và trạng thái công việc. Cách trình bày này phù hợp với kết quả của Laban et al. ở [§1.3](#13-các-vấn-đề-của-multi-turn-conversation), nơi model gặp khó khi phải tự cập nhật các giả định qua nhiều lượt.

Nếu compact ở background, cần xác định chính xác đoạn lịch sử mà bản tóm tắt thay thế. Các lượt phát sinh trong lúc chờ vẫn dùng prefix cũ. Khi ghép chúng sau bản tóm tắt, thinking blocks của Claude phải được bỏ hoặc xử lý bằng `drop_block`.

## 3.9 Fork, sub-agent và summarizer

Giả sử bạn cần tóm tắt một phiên dài 100K token. Nếu tạo request với system prompt mới rồi chép toàn bộ lịch sử vào, prefix thay đổi ngay từ đầu. Khi lời gọi phụ cần đúng context đang có, có thể tạo một nhánh (*fork*) từ request chính:

- Giữ nguyên `model`, `tools`, `system` và lịch sử, rồi nối chỉ dẫn của lời gọi phụ ở cuối. Nhánh mới có thể dùng cache của phần chung nếu entry còn khả dụng.
- Nếu lời gọi phụ cần system prompt khác hẳn, nó là một prefix mới. Cân nhắc chi phí: với hội thoại 100K token, một lời gọi tóm tắt dựng lại từ đầu có thể đắt hơn cả lượt chat chính.
- Với Claude preserved thinking, request tóm tắt vẫn phải giữ `tools` của hội thoại (bỏ tools là sửa prefix). Thêm một câu dặn model viết tóm tắt thay vì gọi tool.

Một sub-agent đọc 30 file cho một câu hỏi độc lập thường chỉ cần mô tả nhiệm vụ và các đầu vào liên quan. Cho nó context riêng rồi đưa kết quả cô đọng về phiên chính giúp tránh nhập cả 30 file vào lịch sử chung. Vì vậy hãy chọn theo dữ liệu công việc cần đọc: dùng fork khi cần lại lịch sử hiện có, context riêng khi nhiệm vụ có thể tách ra.

## 3.10 Routing, concurrency và TTL

**Routing key.** Với OpenAI và xAI, đặt `prompt_cache_key` / `x-grok-conv-id` theo conversation ID. Với system prompt dài dùng chung cho nhiều phiên trên OpenAI, có thể dùng key theo phiên bản prompt cho request đầu của mỗi phiên, nhưng giữ dưới khoảng 15 request/phút cho mỗi key với model trước GPT-5.6.

**Request song song cùng prefix.** Với Claude, cache chỉ dùng được sau khi response của request ghi nó đã bắt đầu. N worker song song trên cùng context sẽ cùng trả giá đầy đủ. Gửi một request trước, chờ response của nó, rồi mới fan out. Với provider implicit, hiệu ứng tương tự có thể xảy ra vì cache cần thời gian để được ghi.

**User nghỉ giữa chừng.** Hội thoại với người thật có khoảng nghỉ dài hơn nhiều so với agent loop.

| Khoảng cách giữa hai request | Lựa chọn |
|---|---|
| Dưới 5 phút | TTL mặc định của mọi provider là đủ |
| 5–60 phút | Claude: TTL 1 giờ, hoặc keep-alive `max_tokens: 0` (rẻ hơn với model có giá đọc thấp như Fable 5.1). OpenAI: `24h` retention hoặc TTL `30m` / `12h` với GPT-5.6+. Doubao: explicit cache với `expire_at` |
| Trên 1 giờ | Chấp nhận cache miss cho request đầu sau khi user quay lại. Với DeepSeek, cache disk có thể còn sau vài giờ |

Chi phí keep-alive phát sinh cả ở những phiên người dùng không quay lại. Với ứng dụng có hàng nghìn phiên mở, nên dựa vào khoảng nghỉ thực tế và tỉ lệ quay lại để tính hiệu quả, thay vì giữ cache cho tất cả các phiên.

## 3.11 Observability cho cache

Một request cache miss vẫn có thể trả lời đúng và kết thúc thành công. Log lỗi HTTP sẽ không cho biết ứng dụng vừa xử lý lại bao nhiêu token. Muốn phát hiện sớm, cần ghi usage và quan sát tỉ lệ đọc, ghi cache qua các lượt.

### Chuẩn hoá usage

```text
function normalize_usage(provider, usage):
    if provider == "anthropic":
        cached  = usage.cache_read_input_tokens
        written = usage.cache_creation_input_tokens
        total   = usage.input_tokens + cached + written
    elif provider == "deepseek":
        cached  = usage.prompt_cache_hit_tokens
        written = 0
        total   = usage.prompt_cache_hit_tokens + usage.prompt_cache_miss_tokens
    elif provider == "gemini" and api == "generateContent":
        cached  = usage_metadata.cachedContentTokenCount or 0
        written = 0
        total   = usage_metadata.promptTokenCount     # đã bao gồm phần cached
    elif provider == "gemini" and api == "interactions":
        cached  = usage.total_cached_tokens or 0
        written = 0
        total   = usage.total_input_tokens
    elif provider uses Responses API (openai, xai, doubao):
        cached  = usage.input_tokens_details.cached_tokens
        written = usage.input_tokens_details.cache_write_tokens or 0
        total   = usage.input_tokens
    else:  # OpenAI-compatible Chat Completions (qwen, glm, kimi, minimax, xai, doubao)
        cached  = usage.prompt_tokens_details.cached_tokens or 0
        written = usage.cache_creation_input_tokens or 0
        total   = usage.prompt_tokens

    return {cached, written, total, hit_rate: cached / total}
```

### Metric cần theo dõi

| Metric | Ý nghĩa | Ngưỡng cảnh báo gợi ý |
|---|---|---|
| Cache hit rate theo request (`cached / total`) | Tỉ lệ input đi qua cache | Bằng 0 qua nhiều lượt dù prefix đủ dài, ổn định và các lượt nằm trong TTL |
| Hit rate trung bình theo route | Sức khoẻ của từng luồng | Giảm đột ngột sau một lần deploy |
| `written` so với phần mới thêm | Có đang ghi lại toàn bộ prefix không | `written` gần bằng `total` ở nhiều request liên tiếp |
| Chi phí input / task hoàn thành | Thước đo cuối cùng | Tăng mà không có thay đổi traffic |
| Số lần compact / phiên | Compact quá dày làm mất lợi ích cache | Tăng sau khi đổi ngưỡng |

Với một vòng lặp Claude giữ prefix tốt, phần đọc từ cache thường tăng theo lượt; phần ghi mới gần với output của lượt trước cộng input mới; phần chưa cache nằm ở đuôi. Có thể dùng xu hướng này để chẩn đoán, nhưng mức hit kỳ vọng phải tính thêm granularity và quy tắc riêng của từng provider, như phần đuôi 2.050 token ở Gemini 4.

### Tìm chỗ vỡ

```text
function find_cache_break(request_a, request_b):
    # request_a, request_b: body JSON của hai request liên tiếp trong cùng phiên
    a = canonical_render(strip_cache_markers(request_a))
    b = canonical_render(strip_cache_markers(request_b))

    for field in ["model", "tools", "system", thinking_and_effort_params]:
        if a[field] != b[field]:
            return f"{field} đã đổi"

    for i, msg in enumerate(a.messages):
        if i >= len(b.messages) or msg != b.messages[i]:
            return f"message {i} đã đổi: {diff(msg, b.messages[i])}"

    # a.messages là prefix của b.messages; kiểm tra output của a có được nối đúng không
    if b.messages[len(a.messages)] != response_of(request_a):
        return "output của request trước không được nối lại nguyên vẹn"

    return "prefix nguyên vẹn; nếu vẫn miss, xem TTL, routing, ngưỡng tối thiểu, workspace"
```

Với Claude, cache diagnostics làm phép so sánh này ở phía server. Với self-hosted, so sánh ở mức chuỗi token sau chat template, không ở mức JSON.

### Kiểm thử tự động

Một integration test với hai lượt liên tiếp giúp phát hiện khi cách dựng request thay đổi. Chọn prefix vượt ngưỡng tối thiểu và gửi lượt sau trong thời gian cache còn hiệu lực:

```text
test "second turn hits cache":
    conv = new_conversation(prompt_version=CURRENT)
    r1 = conv.send("câu hỏi 1")
    r2 = conv.send("câu hỏi 2")
    assert normalize_usage(r2).cached > 0
    assert normalize_usage(r2).cached >= normalize_usage(r1).total * 0.9
```

Mốc 90% trong ví dụ cần được chỉnh theo provider, độ dài prefix và granularity. Test hữu ích khi thêm timestamp, đổi thư viện serialize hoặc thêm middleware chèn message. Vì cache thường là best effort, một lần miss cần được đối chiếu thêm với request đã render và usage trước khi kết luận regression.

## 3.12 Mã giả: Conversation manager an toàn với cache

Ví dụ sau gom các quyết định trên vào một conversation manager: ghim cấu hình khi tạo phiên, lưu output gốc, nối tool results và compact theo ngưỡng. Các hàm dựng request và message là lớp chuyển đổi cho từng provider; hình dạng `system` / `messages` dưới đây chỉ là mã giả, chưa phải payload dùng chung cho mọi API.

```text
class Conversation:
    def __init__(self, provider, model, prompt_version, session_context):
        self.provider = provider
        self.model = model                                  # cố định cả phiên
        self.tools = sort_by_name(load_tools(prompt_version)) # cố định cả phiên
        self.system = load_system(prompt_version)           # cố định, không chứa dữ liệu động
        self.routing_key = new_uuid()                       # prompt_cache_key / x-grok-conv-id
        self.log = AppendOnlyLog()
        # ngữ cảnh của phiên đi vào message đầu, không vào system
        self.log.append(user_message(render(session_context)))

    def send(self, user_text, volatile=None):
        # 1. Nối lượt mới. Phần biến động (giờ, kết quả RAG) đi kèm lượt này
        #    và từ đây trở thành lịch sử bất biến.
        self.log.append(user_message(user_text, tail=volatile))

        # 2. Dọn context theo ngưỡng, không theo từng lượt.
        if self.estimated_tokens() > COMPACT_THRESHOLD and not self.in_tool_round():
            self.compact()

        while True:
            request = self.build_request()
            response = call_provider(request)
            record_usage(normalize_usage(self.provider, response.usage))

            # 3. Nối NGUYÊN response: text, tool calls (arguments dạng chuỗi gốc),
            #    thinking/reasoning, compaction blocks. Không strip, không parse lại.
            self.log.append(raw_output(response))

            if not has_tool_calls(response):
                return response

            # 4. Mọi tool result của một lượt đi trong MỘT message.
            results = run_tools_in_parallel(tool_calls(response))
            self.log.append(tool_results_message(results))

    def build_request(self):
        req = {
            "model": self.model,
            "tools": self.tools,
            "system": self.system,
            "messages": self.log.replay(),      # đúng thứ tự, đúng byte
        }
        return apply_provider_cache_hints(self.provider, req, self.routing_key)
        # anthropic: cache_control ở cuối system + top-level automatic caching
        # openai:    prompt_cache_key = routing_key, retention phù hợp
        # xai:       header x-grok-conv-id = routing_key
        # gemini:    Interactions API với previous_interaction_id nếu dùng stateful
        # others:    không cần gì thêm

    def add_operator_instruction(self, text):
        # Không sửa self.system. Nối thành sự kiện mới.
        if provider_supports_mid_conversation_system(self.provider, self.model):
            self.log.append(system_message(text))
        else:
            self.log.append(user_message(tagged_operator_note(text)))

    def compact(self):
        # Fork từ đúng prefix hiện tại để tận dụng cache, rồi xin bản tóm tắt.
        summary = call_provider(self.build_request()
                                + user_message(SUMMARY_INSTRUCTIONS))
        # Compaction đơn giản: bắt đầu log mới chỉ với tóm tắt.
        # Cache vỡ một lần ở đây, và tóm tắt trở thành prefix mới.
        self.log = AppendOnlyLog()
        self.log.append(user_message(render_summary(summary)))
```

Trong ví dụ, `prompt_version` được giữ từ lúc tạo phiên, `raw_output` lưu đủ các item nhận được và `compact()` tạo ranh giới lịch sử mới. Khi triển khai, các hàm chuyển đổi cần giữ được những tính chất này, đồng thời tuân theo định dạng tool results và reasoning của API đã chọn.

## 3.13 Failure modes thường gặp

| Triệu chứng | Nguyên nhân khả dĩ | Cách kiểm tra |
|---|---|---|
| Cache hit luôn bằng 0 | Prefix dưới ngưỡng tối thiểu; timestamp/UUID ở đầu prompt; tools đổi mỗi request | So body hai request đầu; đếm token phần tĩnh |
| Hit ở lượt 2–3 rồi về 0 | Cửa sổ trượt bắt đầu cắt; lookback 20 vị trí (Claude) bị vượt trong tool loop dài | Xem lượt nào bắt đầu miss, đếm số block mỗi lượt |
| Hit thấp vào giờ cao điểm | Routing tràn sang máy khác (OpenAI trên ~15 RPM/key); eviction | Kiểm tra RPM theo key, chia key |
| Hit chỉ bằng phần tools + system, không bao giờ gồm lịch sử | Lịch sử được dựng lại khác mỗi lượt (mất reasoning, khác whitespace, arguments bị serialize lại) | Diff `messages` của hai request liên tiếp |
| `cache_creation_input_tokens` gần bằng toàn bộ input mỗi request (Claude) | Prefix bị viết lại trước breakpoint; hoặc model cũ bỏ thinking blocks khi có user message sau tool use | Cache diagnostics; kiểm tra model |
| Chi phí tăng sau khi bật caching | Có phí ghi mà cache vỡ liên tục; TTL 1 giờ cho traffic liên tục | So `written` với `cached` |
| 400 sau khi thêm tính năng "chỉnh sửa tin nhắn" | Claude preserved thinking phát hiện lịch sử bị sửa | Coi chỉnh sửa là fork phiên mới, hoặc bỏ thinking blocks phía sau điểm sửa |
| 400 thiếu `reasoning_content` (DeepSeek) | Request có tools nhưng không gửi lại reasoning | Lưu và gửi lại `reasoning_content` của mọi lượt |
| 400 `Function call is missing a thought_signature` (Gemini 3) | Proxy hoặc code chuyển format làm rơi `thoughtSignature`; gửi FC/FR song song xen kẽ | Giữ nguyên part gốc; gửi các FC trước, các FR sau |
| Gemini 4 agent loop: mỗi bước vẫn tính giá đầy đủ cho ~2.000 token | Mỗi bước thêm dưới 2.050 token | Bình thường theo thiết kế; gộp tool call nếu có thể |
| Gemini 3.x: system prompt 3.000 token không bao giờ cache | Dưới ngưỡng 4.096 token | Chấp nhận, hoặc đưa thêm nội dung tĩnh hữu ích vào phần đầu |
| Hit rate tụt đồng loạt mọi phiên cùng lúc | Deploy prompt / tools mới áp vào phiên đang mở | Ghim `prompt_version` theo phiên |
| Fan-out N worker đều trả giá đầy đủ | Request song song trước khi cache được ghi | Gửi một request, chờ token đầu, rồi fan out |
| Model chất lượng giảm dần trong phiên dài | Context rot; model bám vào giả định sai từ lượt đầu | Compact theo ngưỡng với tóm tắt dạng đặc tả; mở phiên mới với yêu cầu hợp nhất |
| Self-hosted: mỗi lượt user mới đều prefill lại phần lớn lịch sử | Chat template bỏ `<think>` của lượt trước | Render hai request ra token và so sánh |

<a id="314-checklist-cho-ai-agent"></a>

## 3.14 Kiểm tra trước khi đưa vào sử dụng

Danh sách này dùng khi review conversation manager hoặc giao agent kiểm tra code. Đối chiếu từng mục với request thực tế và usage; các mục riêng của provider chỉ áp dụng cho API và model đang dùng.

<a id="bắt-buộc"></a>

### Lịch sử và cấu hình phiên

- [ ] Giữa các lần compact, request mới giữ nguyên prefix đã gửi và nối output, input mới vào cuối. Đã kiểm tra cả chat template nếu tự host.
- [ ] `model`, `tools`, `system` được xác định một lần khi tạo phiên và không đổi đến hết phiên.
- [ ] System prompt không chứa: thời gian, ngày, UUID, request ID, user ID, tên user, số token, feature flag, kết quả RAG, hay bất kỳ giá trị nào đổi theo request hoặc theo user.
- [ ] Tools được sort theo tên. Mọi JSON đưa vào prompt được serialize với key đã sort và separator cố định.
- [ ] Response của provider được lưu nguyên dạng (mọi content block, mọi output item) và gửi lại nguyên dạng. Không lưu riêng text rồi dựng lại.
- [ ] Arguments của tool call được giữ ở dạng chuỗi gốc khi gửi lại.
- [ ] Reasoning / thinking của lượt trước được xử lý đúng quy tắc của provider đang dùng (bảng ở [§3.7](#37-reasoning-và-thinking-blocks)). Mặc định: giữ nguyên và gửi lại.
- [ ] Mọi tool call có tool result tương ứng, đúng ID và thứ tự. Kết quả được đóng gói theo API; với Claude, gom trong một user message kế tiếp.
- [ ] Thông tin động được nối vào cuối, ở lượt hiện tại. Nội dung đã gửi ở lượt trước không bị sửa ở lượt sau.
- [ ] Không có cửa sổ trượt theo số message. Dọn context chỉ xảy ra khi chạm ngưỡng, ở ranh giới lượt, không giữa tool round.
- [ ] Usage được log ở mọi request với field cache của provider tương ứng.

### Theo provider

- [ ] **Claude**: breakpoint explicit ở cuối system + `cache_control` top-level cho phần đuôi. Prefix tĩnh vượt ngưỡng tối thiểu của model. Đổi chỉ dẫn giữa phiên bằng `role: "system"` message trên model hỗ trợ. Với compaction, nối toàn bộ `response.content`. Với model có preserved thinking, không sửa lịch sử; kiểm thử bằng `prefix_mismatch_behavior: "error"` trong CI.
- [ ] **OpenAI**: `prompt_cache_key` theo conversation. Chọn retention phù hợp với khoảng nghỉ của user. Với Responses API và `store: false`, nối lại mọi item trong `output`, gồm reasoning item có `encrypted_content`.
- [ ] **xAI**: `x-grok-conv-id` (Chat Completions) hoặc `prompt_cache_key` (Responses) ổn định cho cả phiên. Gửi lại `reasoning.encrypted_content`. Theo dõi mốc 200K token vì giá đổi cho toàn request.
- [ ] **Gemini**: xác định code dùng generateContent hay Interactions API. generateContent: giữ nguyên mọi part kèm `thoughtSignature`, function call song song gửi các FC trước rồi các FR. Interactions stateless: gửi lại mọi `thought` step. Không dùng `cachedContents` với Gemini 4 Argon hoặc Interactions API. Tính đến ngưỡng 4.096 token và quy tắc 2.050 token khi ước lượng cache hit.
- [ ] **DeepSeek**: nếu request có `tools`, gửi lại `reasoning_content` của mọi lượt. Không dựa vào `cache_control` trên endpoint Anthropic-compatible.
- [ ] **Qwen**: quyết định dùng implicit hay explicit (`cache_control`) cho mỗi route và giữ nhất quán.
- [ ] **Kimi / GLM / MiniMax**: kiểm tra quy tắc giữ `reasoning_content` của đúng model đang dùng (`thinking.keep` cho Kimi K2.6, `clear_thinking` cho GLM).
- [ ] **Doubao**: explicit cache chỉ qua Responses API; giữ `thinking` nhất quán; `tools` chỉ ở lượt đầu; không dùng chung với `instructions`.
- [ ] **Self-hosted**: render hai request liên tiếp qua chat template và xác nhận chuỗi token đầu của request sau trùng với request trước.

<a id="trước-khi-coi-là-xong"></a>

### Xác nhận bằng request và usage

- [ ] Có test hai lượt với prefix đủ dài, TTL còn hiệu lực và mức hit kỳ vọng theo provider. Có dữ liệu để phân biệt lỗi ghép prefix với miss do routing hoặc eviction.
- [ ] Đã đối chiếu tools, system và lịch sử chung giữa hai request liên tiếp; giải thích được mọi khác biệt ngoài phần nối thêm và marker cache.
- [ ] Phiên đang mở không bị ảnh hưởng khi deploy phiên bản prompt mới (prompt được ghim theo phiên).

---

<a id="tổng-hợp-sources"></a>

# Nguồn tham khảo

## Claude (Anthropic)

| # | Resource | URL |
|---|----------|-----|
| 1 | Prompt caching | [platform.claude.com/docs/en/build-with-claude/prompt-caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) |
| 2 | Compaction | [platform.claude.com/docs/en/build-with-claude/compaction](https://platform.claude.com/docs/en/build-with-claude/compaction) |
| 3 | Context editing | [platform.claude.com/docs/en/build-with-claude/context-editing](https://platform.claude.com/docs/en/build-with-claude/context-editing) |
| 4 | Context windows | [platform.claude.com/docs/en/build-with-claude/context-windows](https://platform.claude.com/docs/en/build-with-claude/context-windows) |
| 5 | Effective context engineering for AI agents (09/2025) | [anthropic.com/engineering/effective-context-engineering-for-ai-agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) |

## OpenAI

| # | Resource | URL |
|---|----------|-----|
| 6 | Prompt caching guide | [developers.openai.com/api/docs/guides/prompt-caching](https://developers.openai.com/api/docs/guides/prompt-caching) |
| 7 | Conversation state | [developers.openai.com/api/docs/guides/conversation-state](https://developers.openai.com/api/docs/guides/conversation-state) |
| 8 | Pricing | [developers.openai.com/api/docs/pricing](https://developers.openai.com/api/docs/pricing) |
| 9 | Cookbook — Short-term memory management with sessions | [github.com/openai/openai-cookbook/.../session_memory.ipynb](https://github.com/openai/openai-cookbook/blob/main/examples/agents_sdk/session_memory.ipynb) |

## Provider Trung Quốc

| # | Resource | URL |
|---|----------|-----|
| 10 | DeepSeek — KV cache | [api-docs.deepseek.com/guides/kv_cache](https://api-docs.deepseek.com/guides/kv_cache) |
| 11 | DeepSeek — Thinking mode | [api-docs.deepseek.com/guides/thinking_mode](https://api-docs.deepseek.com/guides/thinking_mode) |
| 12 | DeepSeek — Pricing | [api-docs.deepseek.com/quick_start/pricing](https://api-docs.deepseek.com/quick_start/pricing/) |
| 13 | DeepSeek — Anthropic API | [api-docs.deepseek.com/guides/anthropic_api](https://api-docs.deepseek.com/guides/anthropic_api) |
| 14 | Alibaba Model Studio — Context cache | [alibabacloud.com/help/en/model-studio/context-cache](https://www.alibabacloud.com/help/en/model-studio/context-cache) |
| 15 | Alibaba Model Studio — Pricing | [alibabacloud.com/help/en/model-studio/model-pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing) |
| 16 | Kimi — Pricing | [platform.kimi.com/docs/pricing/chat](https://platform.kimi.com/docs/pricing/chat) |
| 17 | Kimi — Thinking model guide | [platform.kimi.com/docs/guide/use-kimi-k2-thinking-model](https://platform.kimi.com/docs/guide/use-kimi-k2-thinking-model) |
| 18 | Z.ai — Context caching | [docs.z.ai/guides/capabilities/cache](https://docs.z.ai/guides/capabilities/cache) |
| 19 | Z.ai — Thinking mode | [docs.z.ai/guides/capabilities/thinking-mode](https://docs.z.ai/guides/capabilities/thinking-mode) |
| 20 | Z.ai — Pricing | [docs.z.ai/guides/overview/pricing](https://docs.z.ai/guides/overview/pricing) |
| 21 | MiniMax — Prompt caching | [platform.minimax.io/docs/api-reference/text-prompt-caching](https://platform.minimax.io/docs/api-reference/text-prompt-caching) |
| 22 | MiniMax — Text generation | [platform.minimax.io/docs/guides/text-generation](https://platform.minimax.io/docs/guides/text-generation) |
| 23 | Volcengine Ark — Context cache | [docs.volcengine.com/docs/82379/1398933](https://docs.volcengine.com/docs/82379/1398933?lang=zh) |
| 24 | Volcengine Ark — Model pricing | [docs.volcengine.com/docs/ark/model-pricing](https://docs.volcengine.com/docs/ark/model-pricing?lang=zh) |

## xAI

| # | Resource | URL |
|---|----------|-----|
| 25 | Prompt caching — How it works | [docs.x.ai/developers/advanced-api-usage/prompt-caching/how-it-works](https://docs.x.ai/developers/advanced-api-usage/prompt-caching/how-it-works) |
| 26 | Prompt caching — Maximizing cache hits | [docs.x.ai/developers/advanced-api-usage/prompt-caching/maximizing-cache-hits](https://docs.x.ai/developers/advanced-api-usage/prompt-caching/maximizing-cache-hits) |
| 27 | Prompt caching — Multi-turn | [docs.x.ai/developers/advanced-api-usage/prompt-caching/multi-turn](https://docs.x.ai/developers/advanced-api-usage/prompt-caching/multi-turn) |
| 28 | Prompt caching — Usage and pricing | [docs.x.ai/developers/advanced-api-usage/prompt-caching/usage-and-pricing](https://docs.x.ai/developers/advanced-api-usage/prompt-caching/usage-and-pricing) |
| 29 | Pricing | [docs.x.ai/developers/pricing](https://docs.x.ai/developers/pricing) |
| 30 | Reasoning | [docs.x.ai/developers/model-capabilities/text/reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning) |
| 31 | May 15 retirement | [docs.x.ai/developers/migration/may-15-retirement](https://docs.x.ai/developers/migration/may-15-retirement) |

## Google Gemini

| # | Resource | URL |
|---|----------|-----|
| 32 | Context caching | [aistudio.google.com/docs/caching](https://aistudio.google.com/docs/caching) |
| 33 | Pricing | [aistudio.google.com/docs/pricing](https://aistudio.google.com/docs/pricing) |
| 34 | Thinking — Thought signatures (Interactions API) | [aistudio.google.com/docs/thinking#signatures](https://aistudio.google.com/docs/thinking#signatures) |
| 35 | Thought signatures (generateContent, Legacy) | [ai.google.dev/gemini-api/docs/generate-content/thought-signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures) |
| 36 | Gemini 3 developer guide (generateContent) | [ai.google.dev/gemini-api/docs/generate-content/gemini-3](https://ai.google.dev/gemini-api/docs/generate-content/gemini-3) |
| 37 | Interactions API — Data storage and retention | [aistudio.google.com/docs/interactions-overview#data-storage-retention](https://aistudio.google.com/docs/interactions-overview#data-storage-retention) |
| 38 | Full docs dump (dùng để tra `cachedContents`, migration sang Gemini 4 Argon) | [aistudio.google.com/llms-full.txt](https://aistudio.google.com/llms-full.txt) |

## Papers, nghiên cứu và bài kỹ thuật

| # | Resource | URL |
|---|----------|-----|
| 39 | Laban et al. — LLMs Get Lost In Multi-Turn Conversation (2025) | [arxiv.org/abs/2505.06120](https://arxiv.org/abs/2505.06120) |
| 40 | Liu et al. — Lost in the Middle (2023) | [arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172) |
| 41 | Chroma — Context Rot (07/2025) | [trychroma.com/research/context-rot](https://www.trychroma.com/research/context-rot) |
| 42 | Manus — Context Engineering for AI Agents (07/2025) | [manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) |
| 43 | vLLM — Automatic prefix caching | [docs.vllm.ai/en/stable/design/prefix_caching](https://docs.vllm.ai/en/stable/design/prefix_caching/) |
| 44 | SGLang (RadixAttention) | [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) |
| 45 | Qwen3 issue #1826 — empty think block breaks KV reuse | [github.com/QwenLM/Qwen3/issues/1826](https://github.com/QwenLM/Qwen3/issues/1826) |
| 46 | vLLM forum — multi-turn cache reuse with default templates | [discuss.vllm.ai/t/.../2955](https://discuss.vllm.ai/t/decode-kv-writeback-can-multi-turn-chat-preserve-cache-reuse-with-default-templates/2955) |

---

Giá, TTL, ngưỡng và tên tham số trong bài được ghi nhận cho bản tháng 11/2026. Trước khi lập dự toán hoặc triển khai, đối chiếu lại trang chính thức của model đang dùng; các beta header có ngày của Claude cũng có thể đổi tên hoặc lên GA.

Những điểm chưa đủ căn cứ hoặc có nguồn mâu thuẫn đã được đánh dấu tại chỗ: giá và phí ghi cache của Kimi, ngưỡng tối thiểu của DeepSeek, endpoint Anthropic-compatible của xAI, quy tắc reasoning nhiều lượt của Qwen API và TTL mặc định của explicit cache Gemini. Hai giá trị bỏ qua kiểm tra thought signature của Gemini lấy từ docs lưu trữ. Số liệu chi tiết theo từng model trong Laban et al. lấy từ bài tổng hợp, chưa đối chiếu từng dòng với bản arXiv.
