# Tool use: để LLM gọi code của bạn

Một shop online muốn chatbot chăm sóc khách hàng trả lời được câu "đơn DH-208431 của tôi đang ở đâu?". Model không biết đơn hàng nào tồn tại, nên bạn khai báo một tool `get_order(order_id)`. Model đọc câu hỏi, trả về yêu cầu gọi `get_order` với `order_id: "DH-208431"`, code của bạn tra database, gửi kết quả lại, và model viết câu trả lời. Bản demo chạy tốt ngay lần đầu.

Vài tuần sau khi lên production, log bắt đầu có những dòng khó giải thích:

- Khách hỏi "huỷ giúp tôi đơn hôm qua", model gọi `cancel_order` với một mã đơn nó tự nghĩ ra.
- Một phiên hội thoại gọi `search_products` 40 lần liên tiếp với cùng tham số, hoá đơn API của phiên đó bằng cả trăm phiên bình thường.
- Sau khi đổi sang reasoning model, request thứ hai trong vòng lặp trả `400` vì "thiếu reasoning item".
- Chuyển sang một nhà cung cấp rẻ hơn chạy cùng model open-weight, tỉ lệ tool call sai schema tăng từ gần 0 lên vài chục phần trăm.

Không lỗi nào ở trên nằm trong code gọi API. Chúng đến từ việc hiểu sai ranh giới giữa hai bên: **model chỉ sinh ra yêu cầu gọi tool; code của bạn quyết định có thực thi hay không, thực thi thế nào và trả lại gì.** Tool definition là một phần của prompt, tool result là input mà model sẽ đọc và tin, còn mọi tác dụng phụ đều do code của bạn gây ra. Tài liệu này đi theo ranh giới đó:

- **Nền tảng — Phần 1:** vòng lặp tool call, tool definition ảnh hưởng thế nào đến prompt và cache, `tool_choice`, gọi song song, tool phía server, và trạng thái ẩn mà reasoning model gắn vào mỗi tool call.
- **Tra cứu API — Phần 2:** wire format, tham số, giới hạn và lỗi thường gặp của OpenAI, Claude, Gemini, Grok, các provider Trung Quốc, engine self-hosted và thư viện.
- **Thiết kế và vận hành — Phần 3:** thiết kế bộ tool, viết description, định dạng tool result, xử lý lỗi, giới hạn vòng lặp, hành động có tác dụng phụ, bảo mật, streaming, eval và observability. Khi cần tìm nguyên nhân một lỗi cụ thể, xem [bảng triệu chứng ở §3.11](#311-failure-modes-thường-gặp).

> Bản tháng 11/2026. Thuật ngữ kỹ thuật và tên field giữ nguyên tiếng Anh để tiện đối chiếu API. Tham số và giới hạn được ghi nhận tại thời điểm biên soạn; các mục chưa xác minh được đánh dấu tại chỗ.

Tài liệu liên quan: [Structured Output](../structured-output/) (strict tool, constrained decoding), [Multi-turn Conversation & Prompt Caching](../multi-turn-conversation/) (cache và reasoning blocks trong hội thoại dài).

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Model không gọi tool, model sinh ra yêu cầu gọi tool](#11-model-không-gọi-tool-model-sinh-ra-yêu-cầu-gọi-tool)
  - [1.2 Tool definition là một phần của prompt](#12-tool-definition-là-một-phần-của-prompt)
  - [1.3 Tool choice: auto, required, none, chỉ định tool](#13-tool-choice-auto-required-none-chỉ-định-tool)
  - [1.4 Gọi nhiều tool song song và tuần tự](#14-gọi-nhiều-tool-song-song-và-tuần-tự)
  - [1.5 Tool phía client, tool phía server và MCP](#15-tool-phía-client-tool-phía-server-và-mcp)
  - [1.6 Reasoning model: trạng thái ẩn đi kèm tool call](#16-reasoning-model-trạng-thái-ẩn-đi-kèm-tool-call)
- [Phần 2 — Tool use theo provider](#phần-2--tool-use-theo-provider)
  - [2.1 Bức tranh chung](#21-bức-tranh-chung)
  - [2.2 OpenAI](#22-openai)
  - [2.3 Claude (Anthropic)](#23-claude-anthropic)
  - [2.4 Google Gemini](#24-google-gemini)
  - [2.5 xAI Grok](#25-xai-grok)
  - [2.6 Các provider Trung Quốc](#26-các-provider-trung-quốc)
  - [2.7 Self-hosted: vLLM, SGLang, llama.cpp, Ollama](#27-self-hosted-vllm-sglang-llamacpp-ollama)
  - [2.8 Thư viện và framework](#28-thư-viện-và-framework)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Thiết kế bộ tool: ít mà rõ](#31-thiết-kế-bộ-tool-ít-mà-rõ)
  - [3.2 Viết description và schema cho tool](#32-viết-description-và-schema-cho-tool)
  - [3.3 Tool result là prompt](#33-tool-result-là-prompt)
  - [3.4 Lỗi từ tool: trả về cho model hay xử lý trong code](#34-lỗi-từ-tool-trả-về-cho-model-hay-xử-lý-trong-code)
  - [3.5 Giới hạn vòng lặp, timeout và chi phí](#35-giới-hạn-vòng-lặp-timeout-và-chi-phí)
  - [3.6 Hành động có tác dụng phụ và bảo mật](#36-hành-động-có-tác-dụng-phụ-và-bảo-mật)
  - [3.7 Streaming khi có tool call](#37-streaming-khi-có-tool-call)
  - [3.8 Khi JSON tool call không phải lựa chọn tốt nhất](#38-khi-json-tool-call-không-phải-lựa-chọn-tốt-nhất)
  - [3.9 Eval cho tool use](#39-eval-cho-tool-use)
  - [3.10 Observability](#310-observability)
  - [3.11 Failure modes thường gặp](#311-failure-modes-thường-gặp)
  - [3.12 Checklist trước khi lên production](#312-checklist-trước-khi-lên-production)
- [Nguồn tham khảo](#nguồn-tham-khảo)

---

## Tra cứu nhanh

### Những quyết định cần chốt trước khi viết code

1. **Mỗi tool call phải nhận đúng một kết quả**, kể cả khi tool lỗi, bị từ chối hay không được chạy. Thiếu kết quả là lỗi `400` ở mọi provider lớn.
2. **Tham số định danh người dùng không đến từ model.** `user_id`, `tenant_id`, quyền hạn lấy từ session phía server. Model chỉ điền những gì nó thực sự cần chọn.
3. **Tách tool đọc và tool ghi.** Tool ghi có xác nhận (người dùng hoặc policy), idempotency key và log riêng.
4. **Giới hạn vòng lặp bằng code**: số bước, tổng token, thời gian, và phát hiện lặp cùng tool cùng tham số. Đừng dựa vào mặc định của framework; LangGraph mặc định cho phép tới 10.007 super-step.
5. **Cố định danh sách tool trong một phiên**, thứ tự ổn định. Muốn hạn chế tool ở một bước, dùng `tool_choice` / `allowed_tools` thay vì xoá định nghĩa, để không mất prompt cache.
6. **Với reasoning model, gửi lại nguyên văn mọi item / block / signature** mà provider trả về giữa hai lượt user. Đây là nguyên nhân số một của lỗi `400` khi nâng cấp model.
7. **Tool result ngắn, có cấu trúc và có hướng dẫn tiếp theo.** Cắt bớt kèm gợi ý cách lấy phần còn lại; lỗi kèm cách sửa.
8. **Coi tool result và tool description là input không tin cậy.** Một agent vừa đọc dữ liệu riêng tư, vừa đọc nội dung từ bên ngoài, vừa gửi được dữ liệu ra ngoài là đủ điều kiện để bị khai thác ([§3.6](#36-hành-động-có-tác-dụng-phụ-và-bảo-mật)).
9. **Đo bằng pass^k, không chỉ pass@1.** Agent đúng 60% ở lần chạy đầu có thể chỉ đúng dưới 25% khi yêu cầu cùng task phải đúng 8 lần liên tiếp.

### So sánh nhanh (snapshot tháng 11/2026)

| | OpenAI (Responses) | Claude | Gemini | Grok | vLLM / SGLang |
|---|---|---|---|---|---|
| Ép gọi tool | `required`, function cụ thể, `allowed_tools` | `any` / `tool`, **trừ** Opus 5.5, Sonnet 5.5, Fable 5.1, Mythos 5.1 | `ANY` (+ `allowed_function_names` để chỉ định) | `required`, function cụ thể | `required`, named |
| Song song mặc định | Bật | Bật | Bật | Bật | Tuỳ parser; llama.cpp tắt |
| Strict schema | Tự chuẩn hoá khi bỏ trống `strict` (Chat Completions: tắt) | `strict: true`, tối đa 20 tool | `VALIDATED` (không ép gọi; text vẫn hợp lệ) | Có | `strict` / `--tool-strict-level` |
| Tool search / nạp trễ | `tool_search` + `defer_loading` (gpt-5.4+) | Tool search tool + `defer_loading` (tới 10.000 tool) | Không có tương đương | Không | Không |
| Tool gọi từ code | Programmatic tool calling | Programmatic tool calling (`allowed_callers`) | Không | Không | Không |
| Phải gửi lại | Reasoning items + `encrypted_content` | `thinking` blocks + `signature` | `thought_signature` (Gemini 3+: thiếu là `400`) | `reasoning.encrypted_content` | Theo chat template |
| Kết quả tool có ảnh / file | Có (Responses); Chat Completions chỉ text | Text, image, document, search_result | Có (Gemini 3+) | Chỉ string | Theo template |
| MCP từ xa do provider gọi | `mcp`, mặc định cần approval | `mcp_toolset` (beta) | `mcp_server`, chỉ Streamable HTTP | `mcp`, không có approval | Không |

Chi tiết và nguồn ở Phần 2.

---

# Phần 1 — Nền tảng

Nếu chỉ làm bản demo cho chatbot đơn hàng, bạn có thể copy ví dụ trong docs và chạy được. Để biết vì sao model bịa mã đơn, vì sao chi phí tăng nhanh theo số bước, hay vì sao đổi tool definition làm hit rate cache rơi về 0, cần nắm cơ chế bên dưới.

## 1.1 Model không gọi tool, model sinh ra yêu cầu gọi tool

Ở mỗi request, model chỉ sinh token. Khi bạn khai báo tool, provider đưa định nghĩa tool vào prompt và huấn luyện model biết rằng nó có thể kết thúc lượt bằng một khối có cấu trúc thay vì văn bản. API tách khối đó ra thành một tool call có `id`, `name` và `arguments`, đặt `stop_reason` / `finish_reason` thành `tool_use` / `tool_calls`, và trả về cho bạn. Không có gì được thực thi phía provider (trừ server tool, [§1.5](#15-tool-phía-client-tool-phía-server-và-mcp)).

```text
request 1: [tools] [system] [user: "đơn DH-208431 đâu rồi?"]
response 1: assistant → tool_call(id=call_1, get_order, {"order_id": "DH-208431"})
                         stop_reason = tool_use

code của bạn: kiểm tra quyền → gọi DB → {"status": "shipped", "carrier": "GHN", ...}

request 2: [tools] [system] [user] [assistant: tool_call call_1] [tool_result call_1: {...}]
response 2: assistant → "Đơn của bạn đã giao cho GHN hôm 4/11, dự kiến tới trong 2 ngày."
                         stop_reason = end_turn
```

Ba hệ quả thường bị bỏ qua:

**`id` là khoá nối giữa yêu cầu và kết quả.** Kết quả gửi lại phải mang đúng `tool_call_id` / `tool_use_id` / `call_id` mà model sinh. Với model open-weight, format của ID cũng là một phần của prompt mà model đã được huấn luyện. Kimi K2 kỳ vọng mọi ID trong lịch sử có dạng `functions.<tên>:<số thứ tự>`; khi lịch sử chứa ID kiểu khác, model có thể sinh ID sai và parser của vLLM bỏ cả tool call hợp lệ ([§2.7](#27-self-hosted-vllm-sglang-llamacpp-ollama)).

**API stateless nên mỗi bước gửi lại toàn bộ lịch sử.** Một tác vụ cần 20 bước, mỗi tool result khoảng 2.000 token, thì request thứ *i* mang theo *i* kết quả. Tổng input token của riêng phần tool result là 2.000 × (1 + 2 + … + 20) = 420.000 token, gấp 10 lần lượng dữ liệu tool thực sự trả về. Chi phí tăng gần theo bình phương số bước. Prompt caching giảm giá phần lặp lại, nhưng không giảm độ dài context mà model phải đọc ([§3.3](#33-tool-result-là-prompt), [§3.5](#35-giới-hạn-vòng-lặp-timeout-và-chi-phí)).

**Model không biết tool đã chạy hay chưa, ngoài những gì bạn nói với nó.** Nếu code của bạn lỗi giữa chừng và bỏ qua một tool call, model không có cách tự phát hiện. Nếu bạn trả `{"ok": true}` khi thao tác thực ra đã thất bại, model sẽ báo với người dùng là đã xong. Tool result là nguồn sự thật duy nhất của model về thế giới bên ngoài.

Vòng lặp tối thiểu, viết theo kiểu trung lập provider:

```python
MAX_STEPS = 12

def run(messages, tools):
    for step in range(MAX_STEPS):
        resp = llm.create(messages=messages, tools=tools)
        if resp.stop_reason == "max_tokens" and resp.has_incomplete_tool_call:
            raise Truncated()             # không lưu, không thực thi tool call bị cắt
        messages.append(resp.assistant_message_as_returned)   # giữ nguyên mọi block / item

        if not resp.tool_calls:
            return resp.text

        results = []
        for call in resp.tool_calls:      # mọi call đều phải có kết quả
            results.append(execute_safely(call))   # không bao giờ raise, trả lỗi dạng result
        messages.append(tool_results_message(results))  # đúng format của provider

    return give_up_gracefully(messages)   # báo người dùng phần đã làm, không im lặng dừng
```

Phần khó không nằm ở vòng lặp mà ở `execute_safely`, `tool_results_message` và `give_up_gracefully`. Phần 3 đi qua từng hàm.

## 1.2 Tool definition là một phần của prompt

Tool definition không đi qua một kênh riêng. Provider render tên, description và schema của mọi tool thành văn bản rồi đặt vào đầu prompt. OpenAI ghi rõ function definitions "được chèn vào system message", tính vào context limit và tính tiền như input token [1]. Claude tạo một system prompt riêng cho tool use, bắt đầu bằng "In this environment you have access to a set of tools…", theo sau là định nghĩa tool và system prompt của bạn [14].

### Chi phí cố định cho mỗi request

Ngoài token của chính các tool definition, provider còn thêm một đoạn hướng dẫn cố định. Claude công bố con số này theo từng model [11]:

| Model | `auto` / `none` | `any` / `tool` |
|---|---|---|
| Claude Opus 5.5, Sonnet 5.5 | 286 | Không hỗ trợ |
| Claude Haiku 5.5, Opus 5 | 286 | 406 |
| Claude Sonnet 5 | 354 | 474 |
| Claude Opus 4.7 | 675 | 804 |
| Claude Sonnet 4.6, Opus 4.6 | 497 | 589 |

Phần lớn chi phí đến từ chính tool definitions. Anthropic đo một cấu hình 5 MCP server với 58 tool tốn khoảng 55.000 token trước khi người dùng nói câu đầu tiên; cấu hình nội bộ của họ từng lên tới 134.000 token [26]. Với context 200K, đó là hơn một phần tư bộ nhớ làm việc bị chiếm trước.

### Tool definitions nằm đầu cache prefix

Vì tool được render trước system prompt, mọi thay đổi trên tool definitions làm mất cache của toàn bộ phần sau. Claude mô tả thứ tự prefix là `tools` → `system` → `messages` [16]:

| Thay đổi | Cache bị mất |
|---|---|
| Thêm, xoá, sửa một tool definition (kể cả đổi thứ tự) | Toàn bộ: tools, system, messages |
| Đổi `tool_choice` hoặc `disable_parallel_tool_use` | Chỉ messages |
| Đổi tham số thinking | Messages; với một số model cả tools và system |

OpenAI cũng cache theo prefix đã render, gồm cả tool definitions; đổi tên, description, schema, thứ tự tool hay `parallel_tool_calls` đều có thể làm vỡ cache [5]. Hệ quả thực tế: **đừng thêm hoặc bớt tool theo từng bước** để "giúp model tập trung". Giữ danh sách cố định cho cả phiên và giới hạn lựa chọn bằng `tool_choice` / `allowed_tools`, hoặc dùng tool search ([§2.2](#22-openai), [§2.3](#23-claude-anthropic)). Chi tiết về bố cục prompt theo độ ổn định có ở [Multi-turn Conversation §3.6](../multi-turn-conversation/#36-tools-trong-hội-thoại-dài).

### Tên và description quyết định tool nào được chọn

Model chọn tool chủ yếu dựa trên văn bản: tên, description, tên tham số. Hai nghiên cứu năm 2025 cho thấy mức độ nhạy cảm này lớn hơn nhiều người nghĩ:

- *Tool Preferences in Agentic LLMs are Unreliable* (EMNLP 2025): chỉ sửa description, không đổi chức năng, có thể làm một tool được dùng **nhiều hơn 10 lần** so với tool cạnh tranh có cùng chức năng. Thí nghiệm có kiểm soát trên GPT-4.1 và Qwen2.5-7B, kiểm tra tổng quát trên 17 model [61].
- *ToolTweak* (2025): lặp lại việc chỉnh tên và description tăng tỉ lệ được chọn của một tool từ khoảng 20% lên tới 81%, và hiệu ứng chuyển được giữa model open-weight và model đóng [60].

Hai hệ quả. Với bộ tool của chính bạn, chất lượng description ảnh hưởng trực tiếp đến độ chính xác; đây là chỗ đáng đầu tư thời gian và cần eval ([§3.2](#32-viết-description-và-schema-cho-tool)). Với tool từ bên thứ ba, chẳng hạn MCP server công khai, description là bề mặt tấn công: một server có thể viết description để "cướp" lượt gọi của tool khác, hoặc nhét chỉ dẫn ẩn mà người dùng không nhìn thấy ([§3.6](#36-hành-động-có-tác-dụng-phụ-và-bảo-mật)).

## 1.3 Tool choice: auto, required, none, chỉ định tool

| Giá trị (tên OpenAI) | Ý nghĩa | Tên ở provider khác |
|---|---|---|
| `auto` | Model tự quyết gọi tool hay trả lời | Claude `auto`, Gemini `AUTO` |
| `required` | Phải gọi ít nhất một tool | Claude `any`, Gemini `ANY` |
| `{"type": "function", "name": ...}` | Phải gọi đúng tool này | Claude `{"type": "tool", "name": ...}`, Gemini `ANY` + `allowed_function_names` |
| `none` | Không được gọi tool | Claude `none`, Gemini `NONE` |
| `allowed_tools` | Chỉ được chọn trong tập con, giữ nguyên danh sách tools | Gemini `allowed_function_names`, Vercel AI SDK `activeTools` |

Những điều ít được ghi trong ví dụ:

**Ép gọi tool được cài đặt bằng prefill hoặc constrained decoding.** Claude ghi rằng với `any` / `tool`, API prefill lượt assistant để ép tool call, nên model không viết được câu dẫn hay giải thích nào trước `tool_use` [13]. Ở vLLM, `required` và named function mặc định dùng structured outputs backend; `auto` mặc định parse tool call từ văn bản tự do, nên "arguments đôi khi có thể sai format" [43]. vLLM bổ sung ràng buộc cho `auto` khi có tool đặt `strict: true` hoặc server đặt `--tool-strict-level`, nhưng vẫn không ép gọi tool ([§2.7](#27-self-hosted-vllm-sglang-llamacpp-ollama)). Cùng một model, `auto` và `required` có thể cho độ tin cậy schema khác nhau tuỳ cấu hình strict.

**Ép gọi tool xung đột với thinking.** Model buộc phải bắt đầu bằng tool call thì không có chỗ để suy luận trước. Claude từ chối `any` / `tool` khi bật manual extended thinking; với adaptive thinking trên Opus 5 và Haiku 5.5, request được chấp nhận nhưng response bắt đầu thẳng bằng `tool_use`, không có `thinking` block [13]. Trên Claude Opus 5.5, Sonnet 5.5, Fable 5.1 và Mythos 5.1, forced tool use bị bỏ hẳn: request trả `400` với thông báo `tool_choice: type "tool" and "any" are not supported for this model.` [22]. Qwen ở thinking mode cũng chỉ ghi nhận `auto` / `none` trong một số trang docs ([§2.6](#26-các-provider-trung-quốc)).

Nếu bạn đang dùng mẹo cũ "ép gọi tool `record_result` để lấy JSON", hãy chuyển sang structured output của response ([Structured Output §1.2](../structured-output/#12-phổ-các-mức-đảm-bảo)).

**`none` khác với bỏ `tools`.** Bỏ tham số `tools` làm đổi prefix, mất cache. Gửi đủ tools với `tool_choice: none` giữ được phần cache tools. vLLM có flag `--exclude-tools-when-tool-choice-none` để bỏ hẳn định nghĩa khỏi prompt khi `none`, đổi lại là mất cache theo cách tương tự [43].

**`required` không có đường thoát.** Nếu mọi tool đều không phù hợp với câu hỏi, model vẫn phải gọi một tool, và sẽ gọi tool "ít sai nhất" với tham số bịa. Đây cũng là cơ chế khiến model bịa mã đơn khi khách hỏi mơ hồ. Nếu workflow cần ép tool, thêm một tool thoát như `ask_user_for_clarification` hoặc `cannot_help`.

## 1.4 Gọi nhiều tool song song và tuần tự

Khi khách hỏi "so sánh giúp tôi đơn DH-208431 và DH-208432", model có thể trả hai tool call trong cùng một response. Đó là **parallel tool calling**: hai yêu cầu được sinh cùng lúc, bạn có thể thực thi đồng thời, rồi gửi lại cả hai kết quả trong một lượt. Ngược lại, khi kết quả bước trước là input của bước sau (tìm đơn hàng gần nhất, rồi huỷ đơn đó), model phải gọi tuần tự qua nhiều lượt.

Parallel mặc định bật ở OpenAI, Claude, Gemini và Grok; llama.cpp tắt mặc định [44]. Một số quy tắc đóng gói kết quả:

| Provider | Quy tắc |
|---|---|
| OpenAI Responses | Mỗi `function_call` cần một `function_call_output` cùng `call_id` ở request kế tiếp. Built-in tool không nằm trong batch song song với function [1] |
| Claude | Mọi `tool_result` nằm trong **một** user message ngay sau lượt assistant, và đứng **trước** mọi text trong message đó; sai thứ tự là `400` [12] |
| Gemini | Gửi tất cả function call rồi mới đến tất cả function response (`FC1, FC2, FR1, FR2`). Xen kẽ `FC1, FR1, FC2, FR2` là `400` [30] |

**"Song song" không có nghĩa là độc lập.** Model có thể trả `update_address` và `get_shipping_fee` trong cùng một batch, trong khi phí ship phụ thuộc địa chỉ mới. Model cũng có thể trả hai `cancel_order` cho hai đơn khi khách chỉ muốn huỷ một. Code thực thi phải quyết định thứ tự và mức đồng thời: chạy song song các tool chỉ đọc, chạy tuần tự hoặc xin xác nhận với tool ghi. Nếu một call bị bỏ qua vì call trước thất bại, vẫn phải trả kết quả cho nó. Claude gợi ý dùng `is_error: true` với nội dung như "Not executed: the preceding write_file call failed." [15].

**Tắt song song có giá.** `parallel_tool_calls: false` (OpenAI, Grok) hoặc `disable_parallel_tool_use: true` (Claude) bảo đảm tối đa một tool call mỗi lượt, đơn giản hoá code nhưng tăng số round trip: mỗi round trip là một lần gửi lại toàn bộ lịch sử. Dùng khi tool có tác dụng phụ khó đảo ngược hoặc khi bạn chưa sẵn sàng xử lý batch đúng cách. Lưu ý thêm: OpenAI ghi rằng với model fine-tuned, nếu model gọi nhiều function trong một lượt thì strict mode bị tắt cho các call đó [1].

## 1.5 Tool phía client, tool phía server và MCP

Có ba nơi một tool có thể chạy:

| Loại | Ai thực thi | Ví dụ | Bạn thấy gì trong response |
|---|---|---|---|
| Client tool | Code của bạn | `get_order`, `cancel_order` | Tool call, `stop_reason: tool_use`; bạn phải gửi kết quả |
| Server tool (built-in) | Provider | Web search, code execution, file search, tool search | Block / item mô tả lần gọi và kết quả, đã nằm sẵn trong response |
| MCP qua provider | Provider gọi MCP server của bạn hoặc bên thứ ba | `mcp` (OpenAI, Grok), `mcp_toolset` (Claude), `mcp_server` (Gemini) | Item lần gọi và kết quả; có thể kèm yêu cầu approval |

Server tool tiện vì bạn không phải tự dựng search hay sandbox, nhưng có vài hành vi riêng:

- **Vòng lặp phía server có giới hạn.** Claude chạy server tool trong một vòng lặp sampling nội bộ; khi chạm giới hạn (mặc định 10 vòng mỗi request), response trả `stop_reason: "pause_turn"`. Bạn gửi lại nguyên nội dung assistant để model tiếp tục, kèm `container` ID nếu có, nếu không sẽ mất file trong sandbox [20].
- **Trộn server tool và client tool.** Nếu cùng một lượt có cả client tool call và server tool call chưa chạy, Claude trả `stop_reason: "tool_use"` mà chưa thực thi server tool. Bạn gửi kết quả client tool, giữ nguyên `tools`, và server tool chạy ở request sau [20][24].
- **Tính tiền theo lần gọi, cộng token.** Web search của Claude là 10 USD / 1.000 lượt tìm, cộng token nội dung; lượt lỗi không tính tiền [24]. OpenAI web search là 10 USD / 1.000 lượt, cộng token nội dung tìm được tính theo giá model [7]. OpenAI có `max_tool_calls` để giới hạn tổng số lần gọi built-in tool trong một response; tham số này **không** áp dụng cho function tool của bạn [8].
- **MCP qua provider dễ làm phình context.** OpenAI nhập danh sách tool từ MCP server thành một item `mcp_list_tools`. Chừng nào item này còn trong context, API không tải lại danh sách ở mỗi lượt; xoá nó khỏi context sẽ buộc tải lại và tăng độ trễ [4]. Grok ghi rõ nếu không đặt `allowed_tools`, mọi tool của server đều vào context [35].
- **Approval mặc định khác nhau.** OpenAI mặc định yêu cầu approval trước khi chia sẻ dữ liệu với MCP server (`require_approval`), và `authorization` không được lưu nên phải gửi lại ở mỗi request [4]. Grok không hỗ trợ `require_approval` [35]. Gemini chỉ nhận MCP server qua Streamable HTTP, không SSE, và tên server không được chứa dấu `-` [28].

Thiết kế và bảo mật MCP server có tài liệu riêng ở [MCP](../mcp/).

## 1.6 Reasoning model: trạng thái ẩn đi kèm tool call

Reasoning model suy luận trước mỗi tool call và tiếp tục suy luận sau khi nhận kết quả (*interleaved thinking*). Phần suy luận đó thường không trả về dạng văn bản đọc được. Thay vào đó, provider trả một đối tượng mờ (reasoning item, thinking block, thought signature) mà bạn phải gửi lại nguyên văn ở request tiếp theo. Nếu bỏ đi, model mất phần suy luận đã làm, và ở nhiều provider request bị từ chối.

Đây là điểm vỡ phổ biến nhất khi một vòng lặp tool đang chạy tốt với model thường được chuyển sang reasoning model. Code cũ dựng lại lịch sử từ những field nó biết (`role`, `content`, `tool_calls`), vô tình bỏ đúng phần provider cần:

| Provider | Phải gửi lại | Lỗi khi thiếu hoặc sửa |
|---|---|---|
| OpenAI Responses | Mọi reasoning item, function call và function call output kể từ user message cuối; `encrypted_content` khi `store: false` [2] | `400 Item 'fc_…' of type 'function_call' was provided without its required 'reasoning' item: 'rs_…'.` [9] |
| Claude | Mọi `thinking` và `redacted_thinking` block, nguyên văn, kể cả block có `thinking` rỗng [22] | `400` "`thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified" [22] |
| Gemini 3+ | `thought_signature` trên function call part, đúng vị trí [30] | `400`; thông báo cộng đồng ghi nhận: "Function call is missing a thought_signature in functionCall parts" |
| DeepSeek (thinking + tools) | `reasoning_content` của mọi lượt [37] | `400` |
| Kimi, MiniMax, GLM | Toàn bộ assistant message gồm `reasoning_content` [39][40] | Mất tính liên tục của suy luận; Kimi K2.7 Code bắt buộc |

Ba chi tiết ít được nhắc:

- **Lỗi hay xuất hiện khi model gọi song song nhiều tool.** Một báo cáo cộng đồng ghi gpt-5 "thường gọi 5–10 tool song song", và lỗi thiếu reasoning item xuất hiện khi code chia nhỏ hoặc sắp xếp lại các item đó [9]. Khi streaming, dùng reasoning item từ sự kiện `response.output_item.done`; bản trong `response.output_item.added` có thể chưa đủ `encrypted_content` [8].
- **Gemini chỉ gắn signature vào function call đầu tiên** trong một batch song song; các call sau không có. Code tự "chuẩn hoá" bằng cách copy signature sang mọi call, hoặc tách batch thành nhiều lượt, đều làm vỡ quy tắc [30].
- **Claude 5.1 trở lên kiểm tra prefix của thinking block.** Nếu bạn sửa bất cứ thứ gì *trước* block đó (system prompt, tools, message cũ), block bị coi là "bound to a different conversation" và request trả `400`. Cách xử lý là giữ lịch sử chỉ-thêm (append-only), hoặc bật `prefix_mismatch_behavior: "drop_block"` với beta header tương ứng [22].

Quy tắc an toàn cho mọi provider: **lưu nguyên response của provider, và dựng request tiếp theo từ chính các object đó**, không phải từ bản rút gọn trong database của bạn. Bảng đầy đủ hơn theo từng thế hệ model có ở [Multi-turn Conversation §3.7](../multi-turn-conversation/#37-reasoning-và-thinking-blocks).

---

# Phần 2 — Tool use theo provider

Phần này dùng để tra cứu khi viết code cho một provider cụ thể. Mỗi mục ghi wire format, tham số quan trọng và những hành vi không có trong ví dụ quickstart.

## 2.1 Bức tranh chung

| | Tool call | Kết quả gửi lại | Khoá nối | Đánh dấu lỗi |
|---|---|---|---|---|
| OpenAI Responses | Item `function_call` (`arguments` là chuỗi JSON) | Item `function_call_output` | `call_id` (không phải `id` dạng `fc_…`) | Không có field; ghi trong `output` |
| OpenAI Chat Completions | `message.tool_calls[]` | Message `role: "tool"` | `tool_call_id` | Không có field |
| Claude | Block `tool_use` (`input` là object) | Block `tool_result` trong user message | `tool_use_id` | `is_error: true` |
| Gemini (Interactions) | Step `function_call` | Step `function_result` | `call_id` | Ghi trong `result` |
| Gemini (generateContent) | Part `functionCall` | Part `functionResponse` | `id` (nếu có), hoặc tên và thứ tự | Ghi trong `response` |
| Grok | Như OpenAI Responses | Như OpenAI Responses | `call_id` | Không có field |
| Model open-weight | Văn bản theo chat template, parser tách ra | Message `role: "tool"` | `tool_call_id`, format theo model | Không có field |

Hai khác biệt dễ gây bug khi viết lớp trừu tượng cho nhiều provider: `arguments` của OpenAI là **chuỗi** cần `json.loads`, còn `input` của Claude đã là **object**; và chỉ Claude có field `is_error` riêng. Với provider khác, quy ước lỗi nằm trong nội dung kết quả, nên hãy dùng một cấu trúc thống nhất như `{"error": {"code": ..., "message": ..., "hint": ...}}`.

## 2.2 OpenAI

### Vòng lặp trên Responses API

```python
import json
from openai import OpenAI

client = OpenAI()

tools = [{
    "type": "function",
    "name": "get_order",
    "description": "Tra trạng thái một đơn hàng của khách đang chat. Chỉ dùng khi khách đã nêu mã đơn dạng DH-XXXXXX.",
    "parameters": {
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "Mã đơn, ví dụ DH-208431"}},
        "required": ["order_id"],
        "additionalProperties": False,
    },
    "strict": True,
}]

items = [{"role": "user", "content": "Đơn DH-208431 của tôi đâu rồi?"}]
for _ in range(12):
    resp = client.responses.create(model="gpt-6.1-sol", tools=tools, input=items, store=False)
    if resp.status == "incomplete":           # ví dụ max_output_tokens: arguments có thể dở dang
        raise Truncated(resp.incomplete_details)   # không đưa vào lịch sử, không thực thi
    items += resp.output                      # gồm cả reasoning item, không lọc
    calls = [it for it in resp.output if it.type == "function_call"]
    if not calls:
        print(resp.output_text)
        break
    for call in calls:
        result = execute_safely(call.name, call.arguments)   # parse JSON bên trong, lỗi thì trả error result
        items.append({
            "type": "function_call_output",
            "call_id": call.call_id,          # call_id, không phải id
            "output": json.dumps(result, ensure_ascii=False),
        })
else:
    print(give_up_gracefully(items))          # hết số bước: tóm tắt phần đã làm (§3.5)
```

Với `store: False` (hoặc tổ chức dùng Zero Data Retention), API ở chế độ stateless: reasoning item mang `encrypted_content` và bạn phải gửi lại toàn bộ item kể từ user message cuối [2]. Với `store: True`, bạn có thể dùng `previous_response_id` thay vì gửi lại.

Reasoning chỉ dùng lại được trong cùng họ model. `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna` dùng được reasoning của nhau, nhưng reasoning không chuyển giữa họ GPT-5.6 và GPT-5.5; API tự bỏ phần không tương thích khỏi context [2]. Khi fallback sang model khác giữa chừng một tác vụ, model mới không thừa hưởng phần suy luận đã làm.

### Strict mode: mặc định khác nhau theo API

- **Responses:** nếu bỏ trống `strict`, API "cố gắng chuẩn hoá schema sang strict khi có thể" và lặng lẽ quay về non-strict nếu schema không tương thích [1][8].
- **Chat Completions:** mặc định non-strict [1].

Hành vi tự chuẩn hoá của Responses nghe tiện nhưng che lỗi: một schema chứa keyword ngoài subset sẽ không báo lỗi mà chỉ mất bảo đảm. Đặt `strict: true` tường minh để schema không tương thích bị từ chối ngay khi gửi. Yêu cầu của strict: `additionalProperties: false` ở mọi object, mọi property nằm trong `required`, field optional dùng union với `null` [1]. Request đầu tiên với một schema mới chậm hơn vì API phải xử lý schema [6]. Chi tiết subset có ở [Structured Output §2.2](../structured-output/#22-openai).

### Giới hạn lựa chọn mà không đổi tools

```json
{"tool_choice": {"type": "allowed_tools", "mode": "required",
                 "tools": [{"type": "function", "name": "get_order"},
                           {"type": "mcp", "server_label": "deepwiki"}]}}
```

`mode: "auto"` cho phép chọn trong tập con hoặc trả lời bằng text; `"required"` bắt gọi ít nhất một tool trong tập con [8]. Tài liệu prompt caching của OpenAI khuyên dùng cách này thay vì xoá tool definition để giữ cache [5].

### Bao nhiêu tool là nhiều

OpenAI gợi ý "dưới 20 function có sẵn ở đầu một lượt", và nói rõ đây chỉ là gợi ý mềm; với namespace, giữ mỗi namespace dưới 10 function [1][3]. Khi cần nhiều hơn:

- **Namespace** gom tool theo miền (`crm`, `billing`, `shipping`) với một description ngắn cho cả nhóm [3].
- **Tool search** (`{"type": "tool_search"}`, chỉ từ `gpt-5.4`) kết hợp `defer_loading: true` trên function, function trong namespace hoặc tool của MCP server. Có hai chế độ: `execution: "server"` (OpenAI tìm trong các tool bạn khai báo) và `"client"` (model trả `tool_search_call`, bạn tự tìm và trả `tool_search_output`) [3].

Hai chi tiết quan trọng từ docs tool search [3]: tool được tìm thấy **được chèn vào cuối context**, nên phần prefix cũ vẫn giữ cache, nhưng mọi thay đổi tập tool đã nạp làm vỡ cache từ điểm đó trở đi. Và OpenAI khuyên ưu tiên namespace hoặc MCP server cho tool search vì model "chủ yếu được huấn luyện để tìm trên các bề mặt đó".

### Custom tool: input là văn bản, có thể ràng buộc bằng grammar

Khi input tự nhiên của tool là code, SQL hay một DSL, bọc nó trong JSON (`{"code": "print(\"hi\")"}`) buộc model escape mọi dấu ngoặc kép và xuống dòng. Custom tool cho model trả thẳng văn bản trong field `input` của item `custom_tool_call` [1]:

```json
{"type": "custom", "name": "run_sql",
 "description": "Chạy một câu SELECT trên bảng orders",
 "format": {"type": "grammar", "syntax": "lark", "definition": "start: \"SELECT \" ..."}}
```

`format` mặc định là văn bản không ràng buộc; `syntax` nhận `lark` hoặc `regex` [1]. Khi streaming, input đến qua `response.custom_tool_call_input.delta` và kết thúc ở `.done` [8].

### Kết quả tool có ảnh và file

Trong Responses, `output` của `function_call_output` có thể là chuỗi hoặc mảng `input_text` / `input_image` / `input_file` [1][8]. Trong Chat Completions, tool message chỉ nhận text [8]. Nếu tool của bạn chụp màn hình hay trả PDF, đó là lý do cụ thể để dùng Responses.

### Text trước tool call: `phase`

Model có thể viết một câu ngắn trước khi gọi tool ("Để mình kiểm tra đơn hàng..."). OpenAI gọi đó là *preamble* và khuyên gắn `phase: "commentary"` cho các cập nhật trung gian; thiếu hoặc bỏ `phase` có thể khiến preamble bị coi là câu trả lời cuối trong workflow nhiều bước [2]. Đây là nguồn của bug "chatbot trả lời 'Để mình kiểm tra...' rồi dừng".

### Lỗi thường gặp

| Thông báo | Nguyên nhân |
|---|---|
| `No tool output found for function call call_…` | Một `function_call` không có `function_call_output` tương ứng ở request sau [10] |
| `No tool call found for function call output with call_id call_…` | Gửi output cho call không tồn tại trong context, thường do dùng `previous_response_id` trỏ sai response |
| `Item 'fc_…' of type 'function_call' was provided without its required 'reasoning' item` | Bỏ reasoning item đi kèm, xem [§1.6](#16-reasoning-model-trạng-thái-ẩn-đi-kèm-tool-call) [9] |
| `Item 'rs_…' of type 'reasoning' was provided without its required following item` | Giữ reasoning item nhưng bỏ item theo sau nó (thường khi tỉa lịch sử cũ) [9] |

Khi tỉa lịch sử cũ, bỏ **cả nhóm** reasoning item và các item theo sau nó, không bỏ riêng một phần.

## 2.3 Claude (Anthropic)

### Vòng lặp

```python
import anthropic

client = anthropic.Anthropic()
messages = [{"role": "user", "content": "Đơn DH-208431 của tôi đâu rồi?"}]
container_id = None                                # sandbox của code execution, nếu có

for _ in range(12):
    extra = {"container": container_id} if container_id else {}
    resp = client.messages.create(model="claude-sonnet-5-5", max_tokens=4096,
                                  tools=tools, messages=messages, **extra)
    if resp.stop_reason == "max_tokens" and resp.content[-1].type == "tool_use":
        raise Truncated()                          # tool_use dở dang: không lưu, tăng max_tokens rồi thử lại
    if getattr(resp, "container", None):
        container_id = resp.container.id           # mất ID này thì request sau chạy container mới, mất file
    messages.append({"role": "assistant", "content": resp.content})   # nguyên văn, gồm thinking blocks

    if resp.stop_reason == "pause_turn":          # server tool chạm giới hạn vòng lặp: gửi lại để tiếp tục
        continue
    if resp.stop_reason != "tool_use":
        break

    results = []
    for block in resp.content:
        if block.type == "tool_use":
            ok, payload = execute_safely(block.name, block.input)
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": payload, "is_error": not ok})
    messages.append({"role": "user", "content": results})   # chỉ tool_result, đứng trước mọi text
else:
    give_up_gracefully(messages)                   # hết số bước (§3.5)
```

`tool_result.content` nhận chuỗi hoặc mảng block `text`, `image`, `document`, `search_result` [12]. Hai quy tắc về vị trí: `tool_result` phải nằm ngay sau lượt assistant chứa `tool_use` tương ứng, và trong user message đó các `tool_result` phải đứng **trước** mọi text [12]. Thiếu kết quả trả lỗi bắt đầu bằng "tool_use ids were found without tool_result blocks immediately after" [12].

Với lỗi, Claude ghi rằng model thường thử lại 2–3 lần với tham số đã sửa trước khi xin lỗi người dùng [12]. Nghĩa là thông báo lỗi bạn viết sẽ được model đọc và làm theo; [§3.4](#34-lỗi-từ-tool-trả-về-cho-model-hay-xử-lý-trong-code) bàn cách viết.

### `tool_choice` và model mới

`auto` là mặc định khi có tools; `any` và `tool` ép gọi; `none` cấm gọi; `disable_parallel_tool_use: true` đặt **bên trong** `tool_choice` [11][13]. Trên Claude Opus 5.5, Sonnet 5.5, Fable 5.1 và Mythos 5.1, `any` / `tool` trả `400`, kể cả trên endpoint đếm token [22]. Anthropic đề xuất dùng `auto` kết hợp strict tool để input luôn đúng schema, hoặc structured outputs nếu cần chính response có dạng JSON cố định [22].

### Strict tool use

`strict: true` trên tool bảo đảm `input` khớp schema bằng constrained decoding. Giới hạn mỗi request [21]:

| Giới hạn | Giá trị |
|---|---|
| Số strict tool | 20 (tool không strict không tính) |
| Tham số optional trên mọi strict schema cộng lại | 24 |
| Tham số dùng `anyOf` hoặc mảng type | 16 |
| Thời gian compile grammar | 180 giây, quá thì lỗi |

Vượt giới hạn trả `400 Schema is too complex for compilation`. Grammar đã compile được cache 24 giờ kể từ lần dùng cuối; đổi `name` hay `description` không làm mất cache grammar, đổi cấu trúc schema thì có [21].

### Tool search và `defer_loading`

Khi có hàng trăm tool, đánh dấu phần lớn là `defer_loading: true` và thêm một tool search tool (`tool_search_tool_regex_20251119` hoặc `tool_search_tool_bm25_20251119`). Model chỉ thấy tool search và các tool không trì hoãn; khi cần, nó tìm, và API nạp định nghĩa tìm được dưới dạng block `tool_reference` trong lượt hiện tại [17].

- Tối đa 10.000 tool trì hoãn mỗi request; mỗi lần tìm mặc định trả 5 kết quả [17].
- Phải có ít nhất một tool không trì hoãn; trì hoãn tất cả trả `400 At least one tool must have defer_loading=false` [17].
- Bạn vẫn gửi đủ định nghĩa mọi tool ở mỗi request; tiết kiệm nằm ở phần model phải đọc, không ở payload [17].
- Vì tool được nạp nằm trong messages chứ không ở đầu prefix, cache của tools và system được giữ [16].

Anthropic công bố trên bộ eval MCP: Opus 4 tăng từ 49% lên 74%, Opus 4.5 từ 79,5% lên 88,1% khi bật tool search; cấu hình 58 tool giảm từ khoảng 55.000 token xuống khoảng 8.700 token [26]. Lợi ích không chỉ là tiết kiệm token: ít tool trong tầm nhìn hơn thì model chọn đúng hơn.

### `input_examples`

Schema nói được kiểu dữ liệu nhưng khó nói quy ước: ngày theo format nào, khi nào điền field optional, tham số nào hay đi cùng nhau. `input_examples` cho phép đính kèm vài ví dụ input hợp lệ trên từng tool. Mỗi ví dụ phải hợp lệ với `input_schema`, nếu không request trả `400`; chi phí khoảng 20–50 token cho ví dụ đơn giản, 100–200 cho ví dụ lồng sâu [14]. Anthropic ghi độ chính xác trên tác vụ tham số phức tạp tăng từ 72% lên 90% [26]. Không dùng được trên server tool [14].

### Programmatic tool calling

Thay vì mỗi tool call là một round trip qua model, model viết code chạy trong sandbox code execution, và code đó gọi tool của bạn. Bật bằng `"allowed_callers": ["code_execution_20260120"]` trên tool; `tool_use` từ code mang field `caller` [18]. Lợi ích chính: kết quả trung gian nằm trong biến của chương trình, không đi vào context. Docs ghi kết quả của tool gọi theo cách này **không tính vào input hay output token** [18]. Anthropic công bố trên tác vụ nghiên cứu phức tạp: token trung bình giảm 37% (43.588 → 27.297) [26].

Giới hạn cần biết [18]: không dùng chung với `strict: true`, không ép được qua `tool_choice`, không hỗ trợ `disable_parallel_tool_use`; user message trả kết quả cho programmatic call **chỉ** được chứa `tool_result`, không có text. Haiku 4.5 nhận version tool nhưng không chạy PTC.

### Streaming tool input

Mặc định Claude buffer input của tool rồi mới gửi các `input_json_delta`. Đặt `eager_input_streaming: true` trên tool để nhận input sớm hơn; đổi lại, docs cảnh báo bạn có thể nhận JSON dở dang hoặc không hợp lệ, và response kết thúc bằng `max_tokens` có thể cắt một tham số giữa chừng [19]. Nếu JSON không parse được, docs gợi ý trả về cho model `{"INVALID_JSON": "<chuỗi gốc>"}` trong `tool_result` với `is_error: true` [19].

### Thinking với tools

- Adaptive thinking tự interleave giữa các tool call [23]. Sonnet 5.5 không tắt được thinking bằng `disabled`; mức thấp nhất là `thinking: {"type": "between_tools"}`, không suy luận trước nhưng vẫn viết cập nhật ngắn giữa các tool call dưới dạng thinking block [22].
- Trên model mới, `display` mặc định là `"omitted"`: field `thinking` là chuỗi rỗng, chỉ có `signature`. Block rỗng **vẫn phải gửi lại** [22]. Code lọc "bỏ block rỗng cho gọn" sẽ gây `400`.
- Nếu code lọc block theo type trước khi gửi lại, giữ cả `thinking` lẫn `redacted_thinking` [22].

### Tool runner trong SDK

`client.beta.messages.tool_runner(...)` với các hàm gắn `@beta_tool` chạy vòng lặp hộ bạn: thực thi tool khi `stop_reason` là `tool_use`, xử lý `pause_turn`, dừng ở lý do khác; có `max_iterations` để giới hạn số vòng [27]. Đặt `max_iterations` tường minh thay vì dựa vào mặc định.

## 2.4 Google Gemini

Gemini có hai API: **Interactions** (mới, có chế độ stateful) và **generateContent** (cũ). Docs hiện tại trên `aistudio.google.com` viết chủ yếu cho Interactions [28].

### Format

- Interactions: tool `{"type": "function", "name", "description", "parameters"}`; model trả step `function_call` có `id`; bạn trả step `function_result` với `call_id` bằng `id` đó và `result` là mảng content block [28].
- generateContent: part `functionCall` / `functionResponse`. Với nhiều response song song, Vertex ghép theo `id` (nếu có) hoặc theo tên và thứ tự trong mảng `parts` [30].

### Chế độ gọi

`AUTO` (mặc định), `ANY` (luôn gọi một hoặc nhiều function), `NONE`, và `VALIDATED`. `VALIDATED` không ép gọi tool: model vẫn được trả lời bằng text, nhưng nếu gọi thì function call phải khớp schema; từ Gemini 3, mode này còn bắt buộc có mặt các tham số required [31]. Cần chắc chắn có tool call thì dùng `ANY`. `allowed_function_names` giới hạn tập function được chọn [31]. Gọi song song mặc định bật, và gọi tuần tự nhiều bước (*compositional*) được hỗ trợ [28]. Với `any`, schema quá lớn hoặc lồng quá sâu có thể bị từ chối với `400 INVALID_ARGUMENT` [28]. Google khuyên giữ tập tool đang hoạt động ở mức 10–20 [28].

Schema dùng subset của OpenAPI: `type`, `nullable`, `required`, `format`, `description`, `properties`, `items`, `enum`, `anyOf`, `$ref`, `$defs` [31].

### Thought signature

Với Gemini 3 trở lên, thiếu thought signature bắt buộc trả `400` [30]. Các quy tắc [30]:

- Response có `functionCall` thì có signature. Gọi **song song**: chỉ function call **đầu tiên** mang `thought_signature`. Gọi **tuần tự**: mỗi function call mang signature riêng.
- Trả lại part đúng như đã nhận, đúng thứ tự: tất cả FC rồi tất cả FR.
- Với text thường, signature có thể nằm ở part text cuối; khi streaming nó có thể đến trong một part text rỗng, nên đọc đến khi có `finish_reason`.
- Khi cần chèn một function call mà model không sinh ra (chẳng hạn chuyển lịch sử từ model khác), có thể đặt `thought_signature` là `skip_thought_signature_validator`. Vertex ghi đây là "biện pháp cuối cùng" vì ảnh hưởng chất lượng.

Với Interactions ở chế độ stateful (`store: true` + `previous_interaction_id`), server tự quản lý signature; ở chế độ stateless, gửi lại nguyên các step `thought` cùng `id` và `signature` [29].

### Automatic function calling trong Python SDK

Khi bạn truyền thẳng hàm Python vào `tools`, SDK `google-genai` **tự gọi hàm đó** mặc định [33]. Giới hạn `maximum_remote_calls` (mặc định 10) đếm số **request tới model**, không đếm số lần hàm của bạn chạy: muốn cho phép *x* lượt function calling thì đặt `x + 1` [33]. Mỗi lượt, model có thể trả nhiều function call song song và SDK chạy tất cả, nên với 10 request, hàm có thể chạy 9 lần, 18 lần hoặc nhiều hơn tuỳ kích thước batch. Khi hết ngân sách, response cuối có thể còn function call chưa được chạy mà code của bạn phải tự xử lý. Tiện cho demo, nhưng hàm của bạn chạy mà không qua lớp kiểm tra quyền, xác nhận hay log mà vòng lặp của bạn có. Với tool có tác dụng phụ, tắt bằng `automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)` và tự xử lý `response.function_calls` [33].

### Kết hợp built-in tool và function

Google Search, URL Context, File Search, Maps và Code Execution có thể dùng cùng function của bạn trong Interactions API, ở trạng thái preview cho Gemini 3 trở lên. Khi bật tính năng chuyển kết quả giữa các tool (*context circulation*), dùng mode `validated`; `auto` không được hỗ trợ [32]. Google cũng ghi rằng thông tin địa điểm hoặc thời gian mâu thuẫn giữa `system_instruction` và description của function có thể khiến việc kết hợp chạy kém [32].

### Kết quả đa phương thức và streaming

Từ Gemini 3, function result nhận text, ảnh (`image/png`, `image/jpeg`, `image/webp`), tài liệu (PDF, CSV, Office), âm thanh, và video trên `v1beta` / `v1alpha`; MIME không hỗ trợ trả `400` [28]. Khi streaming, arguments đến dưới dạng delta; `thought_signature` đến cuối cùng, trước `step.stop`. Gom đủ delta trước khi thực thi [28].

## 2.5 xAI Grok

Grok dùng Responses API tương thích OpenAI tại `https://api.x.ai/v1/responses`: model trả `function_call` với `call_id`, bạn gửi `function_call_output` [34]. Một số điểm riêng:

- Root của `parameters` phải là `"type": "object"`; sai thì `400` kèm tên tool [34].
- Tối đa 350 tool mỗi request; `parallel_tool_calls` mặc định bật [34].
- Function call trả nguyên khối trong một chunk khi streaming, không stream từng phần arguments [34]. UI không hiển thị được tham số đang được điền như với OpenAI hay Claude.
- Server tool: `web_search`, `x_search`, `code_interpreter`, cùng collections search; tính tiền theo token và số lần gọi tool [34].
- Remote MCP: `{"type": "mcp", "server_url", "server_label"}`, có `allowed_tools`, `authorization`, `headers`; hỗ trợ Streamable HTTP và SSE; **không** hỗ trợ `require_approval` [35]. Muốn có bước xác nhận, bạn phải tự gọi MCP server từ phía client.
- `grok-4.7` luôn trả `reasoning.encrypted_content`, kể cả khi `include` không yêu cầu [36]; gửi lại nguyên văn như với OpenAI.

## 2.6 Các provider Trung Quốc

Phần lớn dùng format tương thích OpenAI Chat Completions, nhưng khác biệt nằm ở cách xử lý nội dung suy luận trong vòng lặp tool.

**DeepSeek.** Khi có `tools` và bật thinking, `reasoning_content` phải được gửi lại đầy đủ ở mọi request tiếp theo, thiếu là `400`; khi không có `tools`, field này bị bỏ qua [37]. Strict mode cho function là beta, chỉ chạy qua `base_url="https://api.deepseek.com/beta"`, cần `"strict": true` trên từng function, `additionalProperties: false` và mọi property trong `required` [38]. Gọi qua `/v1` thì `strict` không có tác dụng.

**Kimi (Moonshot).** Với `kimi-k2.7-code`, giữ `reasoning_content` của mọi assistant message trong lịch sử là bắt buộc; `kimi-k2.6` theo `thinking.keep`; `kimi-k3.1-pro` cần gửi lại nguyên assistant message [39]. Không đặt `temperature` cho `kimi-k2.6` và `kimi-k2.7-code` [39]. ID tool call trong lịch sử nên theo dạng `functions.<tên>:<số>`; API chính thức tự đổi tên ID trước khi gọi model, nhà cung cấp khác thì chưa chắc ([§2.7](#27-self-hosted-vllm-sglang-llamacpp-ollama)) [46].

**MiniMax.** Docs yêu cầu nối **toàn bộ** assistant message vào lịch sử (với OpenAI SDK gồm `reasoning_content` và `tool_calls`; với Anthropic SDK là toàn bộ `content`) để giữ chuỗi suy luận liên tục [40]. Một số bài viết bên thứ ba nêu mức giảm hiệu năng cụ thể khi bỏ phần này; chưa tìm được số liệu chính thức từ MiniMax.

**GLM (Z.ai).** Stream tool call cần cả `stream=True` và `tool_stream=True`; `tool_stream` chỉ hỗ trợ dòng GLM-5.x, GLM-4.7, GLM-4.6 [41].

**Qwen (Alibaba Model Studio).** Các trang docs mâu thuẫn: một trang nói thinking mode chỉ hỗ trợ `tool_choice` `auto` / `none` và phải tắt `enable_thinking` để ép tool, trang khác có ví dụ `required` trong phần thinking [42]. Mặc định của `parallel_tool_calls` cũng chưa thống nhất. Kiểm tra trên endpoint thật trước khi dựa vào.

## 2.7 Self-hosted: vLLM, SGLang, llama.cpp, Ollama

Với model open-weight, "tool calling" là hai việc tách rời: **chat template** render tool definitions và lịch sử vào prompt theo format model đã học, và **tool call parser** đọc văn bản model sinh ra để tách tool call. Hai thành phần này do engine và người deploy cấu hình, không phải do model.

### Cùng model, khác nhà cung cấp, khác độ chính xác

Moonshot xây K2 Vendor Verifier để đo cùng một model Kimi K2 trên nhiều nhà cung cấp. Kết quả ngày 15/11/2025 với `kimi-k2-0905` (temperature 0,6) cho tỉ lệ tool call đúng schema [46]:

| Nhà cung cấp | Tool call trả về | Đúng schema | Tỉ lệ |
|---|---|---|---|
| Moonshot (chính thức) | 1.274 | 1.274 | 100,00% |
| DeepInfra, Fireworks, NovitaAI, Infinigence | — | — | 100,00% |
| SiliconFlow | 1.305 | 1.302 | 99,77% |
| vLLM (tự deploy) | 1.325 | 1.007 | 76,00% |
| SGLang (tự deploy) | 1.269 | 928 | 73,13% |
| Together | 1.266 | 911 | 71,96% |
| Nebius | 644 | 544 | 84,47%, và chỉ 50,60% giống hành vi gọi tool của bản chính thức |

Nguyên nhân chính được vLLM phân tích lại [47]: parser tách tên function bằng `function_id.split('.')[1].split(':')[0]`. Khi lịch sử chứa ID không đúng dạng `functions.<tên>:<số>`, chẳng hạn `search:2`, lệnh split ném `IndexError` và cả tool call hợp lệ bị bỏ. API chính thức của Moonshot đổi tên mọi ID trong lịch sử trước khi gọi model; vLLM thì không. Sau khi sửa parser và cập nhật tokenizer, số tool call thành công tăng từ 218 lên 971 (4,4 lần) [47].

OpenRouter quan sát cùng hiện tượng ở quy mô lớn và ra mắt endpoint `:exacto` (10/2025) chỉ route tới nhóm nhà cung cấp có tỉ lệ tool call hợp lệ cao hơn, cho Kimi K2, DeepSeek V3.1 Terminus, GLM 4.6, gpt-oss-120b và Qwen3 Coder. Họ kiểm tra ba điều trên mỗi tool call: JSON hợp lệ, tên tool có trong danh sách, arguments khớp schema; và ghi nhận độ chính xác "khác nhau giữa các nhà cung cấp nhiều hơn đáng kể so với benchmark tiêu chuẩn". Các nguyên nhân được nêu: nhà cung cấp cần thời gian "burn in" model mới, thay đổi inference stack để giảm chi phí, và quantization [48].

Bài học: khi chọn nhà cung cấp cho model open-weight, hoặc nâng cấp engine tự host, chạy một bộ eval tool call riêng ([§3.9](#39-eval-cho-tool-use)) thay vì tin rằng "cùng weights là cùng kết quả".

### vLLM

```bash
vllm serve Qwen/Qwen3-32B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3
```

- `--enable-auto-tool-choice` bắt buộc để dùng `tool_choice: "auto"`; `--tool-call-parser` chọn parser theo họ model (`hermes`, `mistral`, `llama3_json`, `pythonic`, `qwen3_xml`, `kimi_k2`, `deepseek_v31`, `glm47`, `openai` cho gpt-oss…); `--chat-template` thay template khi template đi kèm model không render tool đúng [43].
- Ba câu hỏi tách biệt: model **có buộc phải gọi** tool không (`tool_choice`), **khung** tool call (markup và tên function) có được ràng buộc không, và **arguments** có bị ép theo schema không. `required` và named function buộc phải gọi và dùng guided decoding. Với `auto`, khung tool call chỉ được ràng buộc khi có ít nhất một tool đặt `strict: true` hoặc `--tool-strict-level` là `function` / `parameter`; text thuần vẫn hợp lệ. Arguments chỉ bị ép theo schema ở tool đặt `strict: true`, hoặc ở mọi tool khi server dùng `--tool-strict-level parameter`; tool không đặt `strict` vẫn để arguments tự do, kể cả khi tool khác trong cùng request là strict [43].
- Ngoại lệ: với model Mistral mới dùng format chính thức của Mistral, `required` và named **không** bảo đảm có tool call; model có thể tiếp tục viết text đến `max_tokens` [43].

### SGLang

SGLang cài ràng buộc tool call bằng EBNF / structural tag. Một issue đang mở ghi nhận khi gửi cả tools và `response_format` JSON schema, một số parser trả `400`, còn với parser GLM-4.7 và DeepSeek, tool call bị **bỏ lặng lẽ** và model "tự bịa câu trả lời" [49]. Nếu dùng cả tool lẫn structured output trên SGLang, kiểm tra hành vi trên phiên bản đang chạy.

### llama.cpp

`llama-server` cần `--jinja` và một template có hỗ trợ tool; kiểm tra `chat_template` / `chat_template_tool_use` ở endpoint `/props` [44]. Với template không nhận diện được, llama.cpp dùng format "generic", tốn token hơn và kém hiệu quả hơn format gốc. `parallel_tool_calls` tắt mặc định. Và một cảnh báo ít ai để ý: KV cache quantization mạnh (ví dụ `-ctk q4_0`) "có thể làm giảm đáng kể" khả năng gọi tool [44].

### Ollama

Khi streaming, `tool_calls` đến trong các chunk cùng `thinking` và `content`; gom cả ba rồi nối assistant message vào lịch sử trước tool result. Tool result dùng `role: "tool"` với `tool_name` [45].

## 2.8 Thư viện và framework

Thư viện agent chạy vòng lặp hộ bạn, nên giới hạn mặc định của chúng trở thành giới hạn của hệ thống:

| Thư viện | Giới hạn vòng lặp mặc định | Ghi chú |
|---|---|---|
| OpenAI Agents SDK | `max_turns=10`, vượt thì `MaxTurnsExceeded` | `max_turns=None` tắt giới hạn [50] |
| Vercel AI SDK 7 | `ToolLoopAgent`: `isStepCount(20)` | `stopWhen` chỉ được xét khi bước cuối có tool result; có `toolApproval`, `activeTools`, `experimental_repairToolCall` [51] |
| LangGraph | `recursion_limit` 10.007 super-step (từ 1.1.4) | Gần như không giới hạn; vượt thì `GraphRecursionError` [52] |
| `google-genai` (Python) | `maximum_remote_calls=10` request tới model (= 9 lượt function calling) | Mặc định **tự chạy** hàm Python; mỗi lượt có thể chạy nhiều hàm song song [33] |
| Anthropic SDK tool runner | `max_iterations` | Beta [27] |

Vài điểm cần kiểm tra khi dùng framework:

- **Exception khi chạm giới hạn.** `MaxTurnsExceeded` hay `GraphRecursionError` mà không được bắt thì người dùng nhận lỗi 500, và phần việc đã làm bị mất. Bắt exception, tóm tắt phần đã làm và trả về.
- **Framework có giữ nguyên reasoning item không.** Nhiều lỗi `400` thiếu reasoning item được báo trên issue tracker của các framework, không phải của API [9].
- **`activeTools` / `allowed_tools` có thật sự giữ danh sách tools gửi lên không**, hay framework lọc rồi gửi danh sách mới (mất cache). Xem request thực tế bằng chế độ log raw request.

---

# Phần 3 — Thiết kế và vận hành

Quay lại chatbot đơn hàng. API đã chạy, vòng lặp đã có. Những lỗi ở đầu bài (mã đơn bịa, 40 lần gọi lặp, huỷ nhầm đơn) không sửa được bằng tham số API. Chúng được quyết định bởi bộ tool bạn đưa cho model, nội dung bạn trả lại, và những gì code của bạn cho phép xảy ra.

## 3.1 Thiết kế bộ tool: ít mà rõ

Cách dễ nhất để có tool là bọc từng endpoint của API nội bộ thành một tool: `list_orders`, `get_order`, `get_order_items`, `get_shipment`, `get_customer`. Một chương trình gọi năm endpoint này không tốn gì. Một agent gọi chúng tốn năm round trip, mỗi round trip gửi lại toàn bộ lịch sử, và mỗi kết quả chiếm thêm context. Agent còn phải tự nhớ phải gọi theo thứ tự nào.

Anthropic khuyên gộp các thao tác thường đi cùng nhau thành một tool ở mức tác vụ: thay vì `list_users`, `list_events`, `create_event`, dùng một `schedule_event` tự tìm lịch trống và tạo sự kiện [25]. Với chatbot đơn hàng, một `get_order` trả luôn trạng thái giao hàng và danh sách sản phẩm sẽ thay được ba tool.

### Số lượng tool ảnh hưởng đến độ chính xác

- *Less-is-More* (2024): Llama 3.1 8B chạy trên thiết bị edge thất bại khi có 46 tool, nhưng thành công khi chỉ đưa 19 tool liên quan; tác giả mô tả số lựa chọn lớn "làm LLM bối rối". Lọc tool động giảm tới 70% thời gian thực thi [59].
- *RAG-MCP* (2025): chọn tool qua retrieval trước khi đưa vào prompt tăng độ chính xác chọn tool từ 13,62% lên 43,13% và giảm hơn 50% prompt token [62].
- Provider: OpenAI gợi ý dưới 20 function ở đầu một lượt [1]; Google gợi ý 10–20 tool đang hoạt động [28]; tool search của Claude tăng độ chính xác MCP eval thêm 8,6–25 điểm phần trăm tuỳ model [26].

| Số tool | Cách tổ chức |
|---|---|
| Dưới ~20 | Đưa hết, thứ tự cố định |
| 20–100 | Namespace theo miền; giới hạn theo trạng thái bằng `allowed_tools` / `activeTools`; cân nhắc tách sub-agent theo miền |
| Hàng trăm trở lên | Tool search với `defer_loading` (OpenAI, Claude), hoặc tự retrieval tool rồi giữ tập đã chọn cố định trong phiên |

### Đặt tên và namespace

Khi agent nối với nhiều dịch vụ, `search` của Jira và `search` của Slack trùng tên. Thêm tiền tố theo dịch vụ và tài nguyên: `jira_search_issues`, `slack_search_messages`. Anthropic ghi nhận đặt namespace ở đầu hay cuối tên cho kết quả khác nhau tuỳ model, nên cần thử với eval của bạn [25]. Sắp xếp tool theo tên khi serialize để thứ tự ổn định giữa các request, giữ được prompt cache.

### Đặt business rule trong tool, không trong prompt

"Chỉ được huỷ đơn khi chưa giao cho đơn vị vận chuyển" là một quy tắc. Viết nó trong system prompt nghĩa là hy vọng model nhớ và tuân theo ở mọi lượt. Viết nó trong `cancel_order` nghĩa là quy tắc luôn đúng: tool kiểm tra trạng thái và trả lỗi rõ ràng nếu không được huỷ. Prompt có thể nhắc quy tắc để model giải thích cho khách trước khi gọi, nhưng code mới là nơi thực thi.

Cùng logic đó, tool quá rộng như `execute_sql(query)` hay `http_request(url, method, body)` mạnh nhưng khó kiểm soát: bạn không validate được ý định, chỉ validate được cú pháp. Chỉ dùng chúng trong sandbox với quyền tối thiểu.

## 3.2 Viết description và schema cho tool

OpenAI đề xuất một phép thử: một thực tập sinh chỉ đọc tên, description và schema có dùng tool đúng được không [1]? Anthropic nói tương tự: viết như đang hướng dẫn một nhân viên mới [25]. Một description tốt trả lời năm câu hỏi: tool làm gì, khi nào dùng, khi nào **không** dùng, trả về gì, và cần điều kiện gì trước.

```json
{
  "name": "cancel_order",
  "description": "Huỷ một đơn hàng của khách đang chat. Chỉ dùng khi khách nói rõ muốn huỷ VÀ đã nêu mã đơn. Nếu khách chưa nêu mã, hỏi lại hoặc dùng list_recent_orders để khách chọn; không đoán mã. Đơn đã giao cho đơn vị vận chuyển không huỷ được, tool sẽ trả lỗi NOT_CANCELLABLE. Trả về trạng thái mới và số tiền hoàn dự kiến (VND).",
  "parameters": {
    "type": "object",
    "properties": {
      "order_id": {"type": "string", "description": "Mã đơn dạng DH-XXXXXX, lấy nguyên văn từ lời khách hoặc từ kết quả list_recent_orders"},
      "reason": {"type": "string", "enum": ["changed_mind", "wrong_item", "late_delivery", "other"]}
    },
    "required": ["order_id", "reason"],
    "additionalProperties": false
  }
}
```

Câu "không đoán mã" và đường thoát `list_recent_orders` giải quyết trực tiếp lỗi bịa mã đơn ở đầu bài: model có một hành động hợp lệ thay vì phải chọn giữa từ chối và bịa.

### Quy tắc cho schema

- **Tên tham số không mơ hồ.** `user_id` thay vì `user`; `amount_vnd` thay vì `amount` [25].
- **Không cho biểu diễn trạng thái vô nghĩa.** OpenAI lấy ví dụ `toggle_light(on: bool, off: bool)` cho phép `on=true, off=true` [1]. Một `state: enum["on", "off"]` loại bỏ trường hợp đó.
- **Không bắt model điền thứ bạn đã biết** [1]. `customer_id`, `tenant_id`, ngôn ngữ, múi giờ của phiên lấy từ session phía server. Đây vừa là chuyện độ chính xác, vừa là chuyện bảo mật: tham số model điền được là tham số bị prompt injection điều khiển được ([§3.6](#36-hành-động-có-tác-dụng-phụ-và-bảo-mật)).
- **ID ngắn và có nghĩa.** Model phải chép lại ID từ tool result sang tool call. UUID 36 ký tự là chỗ dễ chép sai một ký tự hoặc bịa ra một ID có hình dạng hợp lệ. Anthropic ghi rằng trả tên và ID có nghĩa thay vì UUID "cải thiện đáng kể" độ chính xác [25]. Nếu bắt buộc dùng UUID nội bộ, đánh số ngắn trong phạm vi phiên (`order_1`, `order_2`) và ánh xạ lại trong code.
- **Enum có đường lui** (`other`, `unknown`) khi tập giá trị không đóng tuyệt đối. Chi tiết thiết kế schema có ở [Structured Output §3.1](../structured-output/#31-thiết-kế-schema).
- **Ví dụ cho quy ước khó diễn đạt bằng schema**: `input_examples` trên Claude [14], hoặc một câu ví dụ trong description ở provider khác.

Description là code: đặt trong version control, review khi đổi, và chạy eval sau mỗi lần sửa. Một thay đổi nhỏ trong câu chữ có thể làm tần suất chọn tool thay đổi hơn 10 lần [61].

## 3.3 Tool result là prompt

Mọi thứ tool trả về sẽ được model đọc ở lượt tiếp theo và ở mọi lượt sau đó cho đến khi bị rút gọn. Tool result vì vậy cần được thiết kế như một đoạn prompt: chỉ chứa thông tin model cần cho quyết định tiếp theo, ở dạng model đọc đúng.

So sánh hai kết quả của `get_order`:

```json
{"id": "8f14e45f-ceea-467a-9575-6d2c3b6e2f1a", "st": 4, "crr": 2, "ts": 1762243200,
 "items": [{"sku_id": "a1b2…", "q": 1, "p": 459000, "meta": {...}}], "audit": [...], "internal_flags": {...}}
```

```json
{"order_id": "DH-208431", "status": "shipped", "carrier": "GHN", "tracking": "GHN123456",
 "shipped_at": "2026-11-04T09:00:00+07:00", "eta": "2026-11-06",
 "items": [{"name": "Máy lọc nước A3", "qty": 1, "price_vnd": 459000}],
 "can_cancel": false, "next_actions": ["track_shipment", "request_return_after_delivery"]}
```

Bản thứ hai ngắn hơn, nhưng quan trọng hơn là không bắt model giải mã `st: 4` hay `ts` epoch, và nói trước điều model sẽ cần cho câu hỏi tiếp theo (`can_cancel`, `next_actions`). Ngày giờ ghi tuyệt đối kèm múi giờ, vì model không biết "hôm nay" của server là ngày nào trừ khi bạn nói.

### Điều chỉnh độ chi tiết

Anthropic gợi ý thêm tham số `response_format` với hai giá trị `concise` / `detailed` cho tool trả dữ liệu lớn; trong ví dụ của họ, bản chi tiết tốn 206 token còn bản ngắn 72 token [25]. Model tự chọn bản chi tiết khi cần ID để gọi tool tiếp, bản ngắn khi chỉ cần trả lời.

### Cắt bớt có hướng dẫn

Claude Code giới hạn tool response ở 25.000 token mặc định [25]. Bạn cần một giới hạn tương tự, và khi cắt phải nói với model rằng đã cắt và cách lấy phần còn lại:

```json
{"results": [ ... 20 đơn ... ],
 "truncated": true, "shown": 20, "total": 312,
 "hint": "Thu hẹp bằng status hoặc from_date, hoặc gọi lại với page=2."}
```

Cắt im lặng là một dạng nói dối: model sẽ tin 20 đơn là tất cả. Với dữ liệu thật sự lớn, lưu ra file hoặc storage và trả về đường dẫn kèm tóm tắt ([Multi-turn §3.6](../multi-turn-conversation/#36-tools-trong-hội-thoại-dài)).

### Kết quả cũ: che đi thường đủ tốt

Một agent chạy 50 bước mang theo 50 tool result, phần lớn không còn liên quan. Nghiên cứu *The Complexity Trap* (JetBrains, 2025) so sánh hai cách rút gọn trên SWE-bench Verified: **observation masking** (thay tool result cũ bằng một placeholder, giữ nguyên lập luận và tool call) và **tóm tắt bằng LLM**. Với Qwen3-Coder 480B, masking đưa chi phí từ 1,29 xuống 0,61 USD mỗi task (giảm 52,7%), còn tỉ lệ giải được tăng nhẹ từ 53,4% lên 54,8%. Tóm tắt bằng LLM cho kết quả tương đương nhưng thường đắt hơn vì phải gọi thêm model [66].

Một lý do khác để rút gọn là *context rot*: báo cáo của Chroma trên 18 model cho thấy chất lượng giảm khi input dài ra, và giảm thêm khi có nội dung gây nhiễu nằm cạnh thông tin cần tìm [67]. Tool result cũ không liên quan chính là nội dung gây nhiễu đó.

Lưu ý khi áp dụng: masking sửa phần giữa của lịch sử, nên phần cache sau điểm sửa bị mất. Làm theo đợt (ví dụ mỗi 10 bước che các kết quả cũ hơn 10 bước) thay vì mỗi bước một lần, và kiểm tra quy tắc gửi lại thinking block của provider ([§1.6](#16-reasoning-model-trạng-thái-ẩn-đi-kèm-tool-call)).

### Nội dung không tin cậy nằm trong tool result

Email khách gửi, trang web vừa tải, review sản phẩm đều có thể chứa chỉ dẫn nhắm vào model. Claude khuyên đặt nội dung như vậy trong `tool_result`, không đưa vào system prompt hay text của user [12]. Nhưng đặt đúng chỗ chỉ giảm rủi ro, không loại bỏ được nó; [§3.6](#36-hành-động-có-tác-dụng-phụ-và-bảo-mật) bàn cách giới hạn thiệt hại.

## 3.4 Lỗi từ tool: trả về cho model hay xử lý trong code

Câu hỏi đầu tiên cho mỗi loại lỗi: model có thể làm gì khác đi nếu biết lỗi này không? Nếu có, trả lỗi cho model kèm hướng sửa. Nếu không, xử lý trong code và chỉ cho model biết kết luận.

| Loại lỗi | Ai xử lý | Trả cho model |
|---|---|---|
| Tham số sai (sai format, thiếu, không tồn tại) | Model | Lỗi cụ thể + ví dụ đúng: `"order_id phải dạng DH-XXXXXX, nhận được '208431'"` |
| Không tìm thấy | Model | Nói rõ không có, và gợi ý: `"Không có đơn DH-208431 trong tài khoản này. 3 đơn gần nhất: ..."` |
| Không được phép theo business rule | Model | Lý do và việc nên làm: `"NOT_CANCELLABLE: đơn đã giao cho GHN. Có thể yêu cầu trả hàng sau khi nhận."` |
| Không có quyền (authz) | Code, rồi model | `"FORBIDDEN, không thử lại."` Không giải thích chi tiết quyền |
| Lỗi tạm thời (timeout, 429, 503) | Code retry với backoff trước | Nếu vẫn lỗi: `"Hệ thống vận chuyển tạm thời không phản hồi. retryable: true"` |
| Tên tool không tồn tại | Code | `"Không có tool 'get_order_status'. Các tool hợp lệ: get_order, cancel_order, ..."` |
| Arguments không phải JSON hợp lệ (non-strict) | Code | `is_error` kèm chuỗi gốc ([§2.3](#23-claude-anthropic)); hoặc sửa bằng `experimental_repairToolCall` (Vercel) [51] |
| Tool call bị cắt (`max_tokens`) | Code | Không thực thi; gửi lại request với `max_tokens` lớn hơn [20] |
| Bug trong code của bạn (exception) | Code | Log đầy đủ; trả model một lỗi chung, không lộ stack trace |

Anthropic khuyên thông báo lỗi phải "actionable" và có ví dụ input đúng [25]. Dùng một cấu trúc thống nhất cho mọi tool để model học được quy ước:

```json
{"error": {"code": "INVALID_ARGUMENT", "message": "from_date phải là YYYY-MM-DD", "example": "2026-11-01", "retryable": true}}
```

`retryable: false` quan trọng không kém phần còn lại. Thiếu nó, model sẽ thử lại một thao tác bị cấm cho đến khi chạm giới hạn vòng lặp.

### Lịch sử bị "nhiễm độc"

Một lỗi ít được nhắc nhưng nghiêm trọng: code của bạn ném exception sau khi đã lưu assistant message có tool call nhưng trước khi lưu tool result. Lần sau người dùng nhắn tiếp, lịch sử chứa một tool call không có kết quả, và **mọi request từ đó trở đi đều trả `400`**. Cuộc hội thoại không dùng được nữa, dù lỗi ban đầu chỉ xảy ra một lần.

Phòng ngừa ở hai chỗ:

```python
def execute_safely(call):
    try:
        return run_tool(call)                       # có timeout
    except ToolError as e:
        return error_result(e.code, e.message, retryable=e.retryable)
    except Exception:
        log.exception("tool crashed", extra={"tool": call.name, "call_id": call.id})
        return error_result("INTERNAL", "Tool gặp lỗi nội bộ.", retryable=False)

def load_history(conversation_id):
    msgs = store.load(conversation_id)
    for call in orphan_tool_calls(msgs):            # tool call chưa có kết quả
        msgs.append(error_result_for(call, "INTERRUPTED", "Thao tác bị gián đoạn, chưa rõ đã thực hiện hay chưa."))
    return msgs
```

Với tool ghi, thông điệp "chưa rõ đã thực hiện hay chưa" là sự thật. Model cần biết điều đó để kiểm tra trạng thái trước khi thử lại, thay vì huỷ đơn lần thứ hai.

## 3.5 Giới hạn vòng lặp, timeout và chi phí

Vòng lặp tool không có điều kiện dừng tự nhiên ngoài việc model quyết định trả lời. Khi model bị kẹt (tool cứ trả lỗi, kết quả không như kỳ vọng, câu hỏi không có đáp án), nó có thể gọi tiếp vô thời hạn. Đặt nhiều lớp giới hạn bằng code:

| Giới hạn | Mục đích | Giá trị khởi điểm cho chatbot CSKH |
|---|---|---|
| Số bước tối đa | Chặn vòng lặp vô hạn | 8–12 |
| Tổng token / chi phí của tác vụ | Chặn phiên đắt bất thường | Vài lần p95 thực tế |
| Thời gian tổng | Người dùng không chờ mãi | 60–120 giây |
| Timeout từng tool | Một API chậm không treo cả vòng | Nhỏ hơn timeout của request ngoài cùng |
| Số lần gọi một tool | Chặn spam một tool | Theo từng tool: tra cứu 3–5 lần mỗi tác vụ, tool polling cao hơn |
| Gọi trùng tool + tham số + kết quả | Phát hiện vòng lặp kẹt | 3 lần thì can thiệp |

Gọi lại cùng tham số chưa chắc là kẹt: `check_job(job_id)` được gọi lặp chính là để chờ trạng thái thay đổi. Tín hiệu kẹt là **cùng tool, cùng tham số, nhận cùng kết quả**. Bộ đếm dưới đây chạy sau khi tool đã trả kết quả, tạo mới cho mỗi tác vụ, và chuẩn hoá arguments vì OpenAI trả chuỗi JSON còn Claude trả object:

```python
seen = Counter()                        # tạo mới cho mỗi tác vụ
REPEAT_LIMIT = {"check_job": 10}        # tool polling: kết quả "running" lặp lại là bình thường

def canonical(value) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)   # OpenAI: arguments là chuỗi JSON
        except json.JSONDecodeError:
            return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def after_execute(call, result):
    key = (call.name, canonical(call.arguments), canonical(result))
    seen[key] += 1
    if seen[key] >= REPEAT_LIMIT.get(call.name, 3):
        return error_result("REPEATED_CALL",
            f"Đã gọi {call.name} với cùng tham số {seen[key]} lần và lần nào cũng nhận cùng kết quả. "
            "Hãy thử cách khác hoặc trả lời người dùng với thông tin hiện có.", retryable=False)
    return result
```

Đây là cách xử lý trường hợp 40 lần `search_products` ở đầu bài: đến lần thứ ba nhận cùng kết quả, model nhận một thông điệp buộc nó đổi hướng. Bộ đếm này dành cho tool đọc. Với tool ghi, lần chạy thứ hai đã là tác dụng phụ, nên lớp chặn phải là idempotency ([§3.6](#36-hành-động-có-tác-dụng-phụ-và-bảo-mật)), không phải đếm sau khi chạy.

### Khi chạm giới hạn

Dừng vòng lặp rồi trả lỗi chung cho người dùng sẽ lãng phí phần việc đã làm. Cách tốt hơn: gửi thêm một request với `tool_choice: "none"` (giữ nguyên `tools` để không mất cache), kèm chỉ dẫn tóm tắt những gì đã tìm được và những gì còn dang dở. Người dùng nhận được câu trả lời một phần thay vì "đã có lỗi xảy ra". Riêng Claude Opus 5.5, Sonnet 5.5, Fable 5.1 và Mythos 5.1 vẫn nhận `none`, chỉ không nhận `any` / `tool` [22].

### Chi phí tăng theo bình phương số bước

Như đã tính ở [§1.1](#11-model-không-gọi-tool-model-sinh-ra-yêu-cầu-gọi-tool), tổng input token tăng gần theo bình phương số bước. Ba đòn bẩy, theo thứ tự hiệu quả thường gặp:

1. **Giảm số bước**: tool ở mức tác vụ ([§3.1](#31-thiết-kế-bộ-tool-ít-mà-rõ)), parallel call cho các tra cứu độc lập, programmatic tool calling cho chuỗi gọi dài ([§3.8](#38-khi-json-tool-call-không-phải-lựa-chọn-tốt-nhất)).
2. **Giảm kích thước mỗi kết quả**: format gọn, cắt có hướng dẫn, masking kết quả cũ ([§3.3](#33-tool-result-là-prompt)).
3. **Cache phần lặp lại**: giữ tools và system ổn định, lịch sử chỉ-thêm. Cache giảm giá của phần prefix, nhưng không giảm độ dài context mà model phải đọc.

### Tool chạy lâu

Nếu một tool cần vài phút (xuất báo cáo, gọi hệ thống bên thứ ba chậm), đừng giữ vòng lặp chờ. Trả ngay một job ID với thông báo đang xử lý, và cung cấp tool `check_job(job_id)`; hoặc kết thúc lượt, báo người dùng sẽ có thông báo khi xong. Với server tool, provider tự giới hạn: Claude trả `pause_turn` sau 10 vòng nội bộ [20], OpenAI có `max_tool_calls` cho built-in tool [8].

## 3.6 Hành động có tác dụng phụ và bảo mật

### Xác nhận phải đến từ người dùng, không đến từ model

Với `cancel_order`, cách phổ biến là dặn trong prompt: "luôn hỏi khách xác nhận trước khi huỷ". Model sẽ hỏi, khách trả lời "ok", model gọi tool. Nhưng chính model là bên quyết định "khách đã đồng ý". Một câu mơ hồ, hoặc một chỉ dẫn được chèn qua nội dung bên ngoài, cũng có thể khiến model kết luận như vậy.

Xác nhận đáng tin cậy đi qua kênh mà model không giả được:

```text
model gọi cancel_order(DH-208431)
  → code KHÔNG thực thi; tạo pending_action {id, tool, args, expires_at}
  → UI hiện nút [Xác nhận huỷ đơn DH-208431] / [Không]
  → người dùng bấm → backend thực thi → tool_result gửi lại cho model
```

Vercel AI SDK có `toolApproval` cho từng tool [51]; OpenAI remote MCP có `require_approval` [4]. Với tool tự viết, tự cài theo mẫu trên. Mức xác nhận nên tương ứng với rủi ro: tool đọc không cần, tool ghi có thể đảo ngược cần ít, tool chuyển tiền hoặc gửi dữ liệu ra ngoài cần nhiều.

### Idempotency

Một tool ghi có thể bị chạy hai lần: request đến backend bị timeout nhưng thực ra đã thành công, worker crash và chạy lại vòng lặp, hoặc model gọi lại vì không chắc. Hai lớp chống trùng:

- **Theo lần gọi:** dùng `(conversation_id, tool_call_id)` làm idempotency key khi gọi API phía sau. Lần chạy lại cùng một tool call trả kết quả đã lưu thay vì thực hiện lại.
- **Theo nghiệp vụ:** tool call ID chỉ ổn định trong một response. Nếu model sinh lại response (retry request LLM), nó tạo ID mới cho cùng ý định. Vì vậy tool ghi vẫn cần kiểm tra trạng thái nghiệp vụ: đơn đã huỷ thì `cancel_order` trả `ALREADY_CANCELLED` kèm thời điểm huỷ, không báo lỗi chung.

### Quyền hạn: model là "confused deputy"

Tool chạy với quyền của backend, còn yêu cầu đến từ một model đọc dữ liệu từ nhiều nguồn. Nếu `get_order` nhận `customer_id` từ arguments, một prompt injection có thể bảo model tra đơn của người khác. Quy tắc:

- Danh tính và phạm vi (`customer_id`, `tenant_id`, role) gắn vào context thực thi từ session, không nằm trong schema.
- Mỗi tool tự kiểm tra quyền trên từng lần gọi, kể cả khi "model chỉ gọi tool này sau khi đã xác thực".
- Tập tool đưa cho model tuỳ theo quyền người dùng: khách thường không thấy `refund_any_order`.

### Prompt injection qua tool result

Model không phân biệt được chỉ dẫn của bạn với chỉ dẫn nằm trong dữ liệu nó đọc. Benchmark AgentDojo (97 tác vụ, 629 test case bảo mật) đo tỉ lệ tấn công thành công có chủ đích: GPT-4o 53,1%, Claude 3 Opus 12,9%; và utility của GPT-4o giảm từ 69% xuống 45% khi bị tấn công [68]. Các model mới đã tốt hơn, nhưng không model nào đạt 0%.

Simon Willison gọi tên tổ hợp nguy hiểm là **lethal trifecta**: agent có quyền truy cập dữ liệu riêng tư, tiếp xúc với nội dung không tin cậy, và có cách gửi dữ liệu ra ngoài. Có đủ ba yếu tố thì kẻ tấn công có thể lấy dữ liệu; ông lưu ý guardrail chặn được 95% vẫn là "điểm trượt" với bài toán bảo mật [71]. Với chatbot đơn hàng: đọc review sản phẩm (không tin cậy) + tra thông tin khách (riêng tư) + gửi email (kênh ra ngoài) là đủ ba.

Bài báo *Design Patterns for Securing LLM Agents against Prompt Injections* (2025) đề xuất sáu mẫu thiết kế đánh đổi tính linh hoạt lấy khả năng chống injection có thể chứng minh [70]:

| Mẫu | Ý tưởng | Đánh đổi |
|---|---|---|
| Action-selector | Model chỉ chọn trong các hành động định sẵn; tool result không quay lại model | Miễn nhiễm injection, mất tính linh hoạt |
| Plan-then-execute | Chốt danh sách hành động trước khi đọc dữ liệu không tin cậy | Dữ liệu vẫn ảnh hưởng tham số; khó khi bước sau phụ thuộc bước trước |
| LLM map-reduce | Mỗi mục dữ liệu không tin cậy được xử lý bởi một LLM cô lập, không có tool nguy hiểm; bước reduce là code không dùng LLM, hoặc output của bước map bị ràng buộc chặt (ví dụ chỉ boolean hoặc enum) trước khi tới LLM reduce | Nếu bước map trả văn bản tự do cho một LLM reduce, injection vẫn đi tiếp xuống dưới; không phải tác vụ nào cũng chia được |
| Dual LLM | LLM có quyền lập kế hoạch nhưng không thấy dữ liệu; LLM cách ly đọc dữ liệu nhưng không có tool | LLM cách ly vẫn bị injection, nhưng không làm được gì |
| Code-then-execute | Agent viết chương trình gọi tool; dữ liệu chảy qua biến | Mở rộng plan-then-execute; cần ngôn ngữ đủ mạnh |
| Context-minimization | Xoá prompt không tin cậy khỏi context sau khi đã dùng nó để chọn hành động | Kém linh hoạt |

CaMeL (Google DeepMind) hiện thực hoá hướng code-then-execute: tách luồng điều khiển tin cậy khỏi dữ liệu không tin cậy, gắn capability cho dữ liệu và kiểm tra policy khi gọi tool. Trên AgentDojo, CaMeL giải được 77% tác vụ với bảo đảm bảo mật có thể chứng minh, so với 84% của hệ không phòng thủ [69]. Mất 7 điểm utility để đổi lấy bảo đảm là mức giá cụ thể để cân nhắc cho tool có rủi ro cao.

### Tool từ bên thứ ba: poisoning và rug pull

Invariant Labs mô tả ba kiểu tấn công qua MCP [72]:

- **Tool poisoning**: description chứa chỉ dẫn ẩn mà model đọc được nhưng giao diện không hiển thị cho người dùng. Demo của họ khiến agent đọc `~/.ssh/id_rsa` và gửi đi qua một tham số ẩn.
- **Rug pull**: server đổi description sau khi người dùng đã phê duyệt.
- **Shadowing**: description của một server độc hại thay đổi cách model dùng tool của một server tin cậy khác; demo chuyển hướng mọi email tới kẻ tấn công.

Biện pháp: ghim phiên bản và lưu hash của description khi phê duyệt, báo động khi hash đổi; review description như review code; chỉ bật `allowed_tools` cần thiết. Chi tiết hơn ở [Bảo mật LLM](../llm-security/) và [MCP](../mcp/).

## 3.7 Streaming khi có tool call

Khi stream, tool call đến từng mảnh. Cấu trúc sự kiện khác nhau theo provider:

| Provider | Bắt đầu | Arguments | Kết thúc | Khoá để gom |
|---|---|---|---|---|
| OpenAI Responses | `response.output_item.added` | `response.function_call_arguments.delta` | `.done`, rồi `response.output_item.done` | `item_id` / `output_index` [8] |
| OpenAI Chat Completions | Delta đầu có `id`, `name` | `delta.tool_calls[i].function.arguments` | `finish_reason: "tool_calls"` | `index` [8] |
| Claude | `content_block_start` (`input: {}`) | `input_json_delta.partial_json` | `content_block_stop` | Chỉ số content block [19] |
| Gemini Interactions | `step.start` | `step.delta` loại arguments | `step.stop`; signature đến cuối | Step [28] |
| Grok | Cả tool call trong một chunk | — | — | [34] |

Quy tắc:

- **Không thực thi trước sự kiện kết thúc.** JSON dở dang có thể parse được (`{"order_id": "DH-2"` thêm `}` là hợp lệ) nhưng sai giá trị.
- **Gom theo khoá đúng.** Với Chat Completions, nhiều tool call song song chỉ phân biệt bằng `index`; `id` thường chỉ có ở delta đầu tiên.
- **Dùng sự kiện bắt đầu để cập nhật UI.** Tên tool có ngay từ đầu, nên có thể hiện "Đang tra cứu đơn hàng…" trước khi arguments xong.
- **Partial JSON chỉ dùng để xem trước**, chẳng hạn hiển thị bản nháp email đang được viết. Kỹ thuật parse partial có ở [Structured Output §3.4](../structured-output/#34-streaming-partial-json).
- **Mất kết nối giữa tool call:** bỏ tool call dở dang, không lưu vào lịch sử, gửi lại request. Lưu một tool call không có kết quả dẫn đến lịch sử nhiễm độc ([§3.4](#34-lỗi-từ-tool-trả-về-cho-model-hay-xử-lý-trong-code)).

Chi tiết về SSE qua proxy và xử lý đứt kết nối có ở [Streaming](../streaming/).

## 3.8 Khi JSON tool call không phải lựa chọn tốt nhất

Tool call dạng JSON là mặc định, nhưng không phải lúc nào cũng tối ưu. Ba hướng thay thế có số liệu:

**Code làm action.** CodeAct (ICML 2024) để model viết code Python gọi tool thay vì từng JSON call. Trên M3ToolEval với gpt-4-1106-preview, tỉ lệ thành công đạt 74,4% so với 52,4% của JSON, và số lượt trung bình giảm từ 7,6 xuống 5,5 [63]. Lý do: code biểu diễn được vòng lặp, điều kiện và biến trung gian, nên một lượt làm được việc của nhiều tool call. Programmatic tool calling của Claude và OpenAI là phiên bản của ý tưởng này trong API thương mại ([§2.2](#22-openai), [§2.3](#23-claude-anthropic)).

**Ngôn ngữ tự nhiên làm format trung gian.** *Natural Language Tools* (2025) cho model trả lời bằng văn bản tự do việc cần gọi tool nào, rồi tách bước parse ra. Trên 10 model và 6.400 lượt thử, độ chính xác tăng 18,4 điểm phần trăm, phương sai output giảm 70%; model open-weight tăng nhiều nhất (58,7% → 84,8%) [64]. Một nghiên cứu lặp lại năm 2026 trên 14 model xác nhận mức tăng trung bình 14,9 điểm, nhưng ghi nhận với model frontier như GPT-5 và Gemini 2.5 Pro, lợi thế nhỏ hơn hoặc đảo ngược [65].

**Custom tool với grammar** (OpenAI) cho input là SQL, code hay DSL, tránh escape JSON ([§2.2](#22-openai)).

Cách đọc các kết quả này: với model frontier có native tool calling và tác vụ ít bước, JSON tool call vẫn là lựa chọn đúng. Đáng cân nhắc phương án khác khi dùng model nhỏ hoặc open-weight không có tool calling tốt, khi một tác vụ cần hàng chục lời gọi phụ thuộc nhau trên dữ liệu lớn, hoặc khi input tự nhiên của tool không phải JSON. Phương án code làm action đòi hỏi sandbox an toàn cho code do model viết.

## 3.9 Eval cho tool use

Một câu trả lời cuối đúng chưa cho biết agent làm đúng. Agent có thể huỷ nhầm đơn rồi xin lỗi, và câu xin lỗi được chấm là "lịch sự, đúng ngữ cảnh". Eval cho tool use cần nhiều tầng:

| Tầng | Câu hỏi | Cách đo |
|---|---|---|
| Chọn tool | Gọi đúng tool không? Có gọi khi không nên gọi không? | So tên tool với nhãn; tập câu hỏi không cần tool (*irrelevance*) |
| Tham số | Arguments đúng không? Có hỏi lại khi thiếu thông tin không? | So từng field; tập câu thiếu tham số |
| Quỹ đạo | Thứ tự và số bước hợp lý không? | So với quỹ đạo tham chiếu, cho phép nhiều đường đúng |
| Kết quả | Trạng thái hệ thống sau cùng đúng không? | So database / trạng thái mock với trạng thái mục tiêu |
| Độ ổn định | Đúng bao nhiêu lần trên k lần chạy? | pass^k |

### pass^k

τ-bench định nghĩa pass^k là xác suất agent thành công ở **cả k** lần chạy độc lập cùng một tác vụ, và chấm bằng cách so trạng thái database với trạng thái mục tiêu. Ở thời điểm công bố, gpt-4o đạt pass^1 61,2% trên miền retail, nhưng pass^8 dưới 25% [53]. Với một chatbot phục vụ hàng nghìn khách, con số quyết định trải nghiệm là pass^k, không phải pass@1: cùng một yêu cầu, người dùng này được xử lý đúng còn người dùng khác thì không. τ²-bench mở rộng sang tình huống cả agent và người dùng đều có tool (miền telecom), và ghi nhận hiệu năng giảm đáng kể so với khi chỉ agent có tool [54].

### Những tình huống benchmark cho thấy là khó

- **Thiếu thông tin và phụ thuộc trạng thái ẩn.** ToolSandbox chỉ ra ba nhóm khó với cả model mạnh: phụ thuộc trạng thái (phải bật mạng trước khi gửi tin nhắn), chuẩn hoá giá trị (chuyển "chiều mai" thành ngày cụ thể), và thông tin không đủ (phải hỏi lại thay vì đoán) [57]. Nhóm thứ ba chính là lỗi bịa mã đơn.
- **Gọi lồng nhau.** NESTFUL đo chuỗi gọi trong đó output của lần trước là input của lần sau; model tốt nhất là GPT-4o chỉ khớp toàn bộ chuỗi ở 28% [56].
- **Môi trường MCP thật.** MCP-Universe (11 MCP server, 6 miền): GPT-5 đạt 43,72%, Claude-4.0-Sonnet 29,44% [58].
- BFCL tách riêng các hạng mục như irrelevance, multi-turn thiếu tham số hoặc thiếu function, context dài, và độ nhạy với format [55].

### Bộ eval của chính bạn

- **Nguồn dữ liệu**: hội thoại thật từ log (đã ẩn danh), cộng các ca biên tự viết: câu mơ hồ, thiếu mã đơn, hỏi ngoài phạm vi, yêu cầu bị cấm, chỉ dẫn chèn trong review sản phẩm.
- **Tool giả lập có trạng thái**: mock trả kết quả cố định và ghi lại thay đổi trạng thái, để chấm tầng kết quả và để chạy lại được.
- **Chạy k lần mỗi ca** để đo pass^k và phát hiện ca chập chờn.
- **Chạy lại khi**: sửa description, thêm hoặc bớt tool, đổi model, đổi provider, nâng cấp engine self-hosted hoặc chat template.
- **Kiểm tra nhà cung cấp** theo cách của K2 Vendor Verifier: so tần suất gọi tool với nhà cung cấp tham chiếu (*trigger similarity*) và tỉ lệ arguments đúng schema [46]. Hai chỉ số này phát hiện được lỗi template và parser mà eval câu trả lời cuối bỏ sót.

Chi tiết về golden dataset và LLM-as-judge có ở [Evaluation](../evaluation/).

## 3.10 Observability

OpenTelemetry định nghĩa span `execute_tool` cho mỗi lần thực thi tool, ở trạng thái Development [73]:

- Tên span: `execute_tool {gen_ai.tool.name}`, kind `INTERNAL`, `gen_ai.operation.name` là `execute_tool`.
- Bắt buộc: `gen_ai.tool.name`. Bắt buộc có điều kiện: `gen_ai.tool.call.id` (khi model / provider cung cấp), `error.type` (khi lỗi), `gen_ai.conversation.id` (khi có). Khuyến nghị: `gen_ai.tool.type`, `gen_ai.tool.description`.
- `gen_ai.tool.call.arguments` và `gen_ai.tool.call.result` là **opt-in**, vì chúng thường chứa dữ liệu cá nhân.

Spec còn định nghĩa metric `gen_ai.execute_tool.duration` [73]. Đặt span tool làm con của span gọi model để trace đọc được theo thứ tự: model quyết định → tool chạy → model đọc kết quả.

Những chỉ số nên có dashboard:

- **Theo tool**: số lần gọi, tỉ lệ lỗi theo `error.type`, độ trễ p50 / p95, kích thước kết quả (token).
- **Theo tác vụ**: phân phối số bước, tổng token, tỉ lệ chạm giới hạn vòng lặp, số lần phát hiện gọi trùng.
- **Chất lượng tool call**: tỉ lệ arguments sai schema, tên tool không tồn tại, tool call bị cắt. Ba chỉ số này tăng đột ngột sau khi đổi provider hoặc nâng cấp engine là tín hiệu sớm nhất của lỗi template / parser.
- **Tool ghi**: số lần được xác nhận, bị từ chối, hết hạn; số lần idempotency key chặn được lần chạy trùng.

Lưu raw request và response của provider (có kiểm soát truy cập và thời hạn) để phát lại được một phiên lỗi. Nhiều lỗi ở bảng dưới chỉ chẩn đoán được khi xem đúng payload đã gửi đi.

## 3.11 Failure modes thường gặp

| Triệu chứng | Nguyên nhân khả dĩ | Cách kiểm tra / xử lý |
|---|---|---|
| Model gọi tool với ID / mã bịa | Không có đường thoát khi thiếu thông tin; `tool_choice` ép buộc | Thêm hướng dẫn "không đoán", tool hỏi lại hoặc liệt kê; validate tồn tại trong code |
| `400` thiếu tool result | Exception giữa lúc lưu tool call và tool result; bỏ qua một call trong batch song song | `execute_safely` không bao giờ raise; sửa lịch sử khi load ([§3.4](#34-lỗi-từ-tool-trả-về-cho-model-hay-xử-lý-trong-code)) |
| Mọi request của một hội thoại đều `400` | Lịch sử đã lưu chứa tool call mồ côi hoặc thinking block bị sửa | Thêm kết quả lỗi cho call mồ côi; với Claude dùng `drop_block` hoặc bỏ block [22] |
| `400` thiếu reasoning item (OpenAI) | Code dựng lại lịch sử, lọc mất reasoning item; tỉa lịch sử không theo nhóm | Gửi lại nguyên `resp.output`; tỉa cả nhóm [9] |
| `400` thiếu thought signature (Gemini 3+) | Dựng lại part; copy signature sai; xen kẽ FC / FR | Gửi lại part nguyên văn; FC trước, FR sau [30] |
| `400` `tool_choice ... not supported` (Claude) | `any` / `tool` trên Opus 5.5, Sonnet 5.5, Fable 5.1, Mythos 5.1 | `auto` + strict tool, hoặc structured outputs [22] |
| `400` khi có text trong user message chứa tool result (Claude) | Text đứng trước `tool_result`; text trong lượt trả kết quả programmatic call | Đặt `tool_result` trước; với PTC chỉ gửi `tool_result` [12][18] |
| Hit rate cache giảm mạnh khi bật tools | Thêm / bớt / sắp lại tool theo bước; framework lọc tools | Cố định tools; dùng `allowed_tools` / `tool_choice` |
| Gọi cùng tool cùng tham số nhiều lần | Kết quả không đổi nhưng model không biết dừng; lỗi không ghi `retryable` | Phát hiện gọi trùng; ghi `retryable: false` |
| Chi phí một phiên bất thường | Vòng lặp dài, tool result lớn, tăng theo bình phương | Giới hạn token mỗi tác vụ; cắt kết quả; masking |
| Model báo "đã xong" nhưng thao tác thất bại | Tool trả `ok` khi lỗi; cắt im lặng | Kết quả phản ánh đúng trạng thái; ghi rõ `truncated` |
| Chatbot trả "Để mình kiểm tra..." rồi dừng | Preamble bị coi là câu trả lời cuối | OpenAI: `phase: "commentary"` [2]; kiểm tra điều kiện dừng vòng lặp |
| Tool call sai schema tăng sau khi đổi nhà cung cấp / nâng cấp engine | Chat template, tool call parser, quantization, ID lịch sử sai format | Eval kiểu vendor verifier; chuẩn hoá ID; endpoint `:exacto` [46][48] |
| Tool call biến mất khi dùng cùng `response_format` (SGLang) | Grammar tool bị bỏ lặng lẽ với một số parser | Kiểm tra phiên bản; tách bước [49] |
| Model open-weight in tool call ra `content` thay vì `tool_calls` | Thiếu `--enable-auto-tool-choice`, sai `--tool-call-parser` hoặc template | Đối chiếu parser với họ model; xem raw output [43] |
| Tool call hỏng chỉ trên llama.cpp | Template không nhận diện, rơi về generic; KV quantization mạnh | `--jinja`, kiểm tra `/props`; giảm quantization KV [44] |
| Hàm Python chạy mà không qua kiểm tra quyền (Gemini) | Automatic function calling của SDK mặc định bật | `AutomaticFunctionCallingConfig(disable=True)` [33] |
| Tool call bị cắt giữa chừng | `max_tokens` quá nhỏ, nhất là khi có thinking | Không thực thi; tăng `max_tokens` [20] |
| JSON arguments không parse được | Non-strict; Claude `eager_input_streaming` | Strict mode; trả `is_error` với chuỗi gốc [19] |
| Tool bên thứ ba hành xử khác sau khi đã duyệt | Rug pull: description đổi | Ghim hash description, báo khi đổi [72] |
| Người dùng nhận lỗi 500 sau nhiều bước | Exception chạm giới hạn của framework không được bắt | Bắt `MaxTurnsExceeded` / `GraphRecursionError`, tóm tắt phần đã làm |

## 3.12 Checklist trước khi lên production

### Vòng lặp

- [ ] Mọi tool call nhận đúng một kết quả, kể cả khi lỗi, bị từ chối hoặc không chạy. `execute_safely` không raise.
- [ ] Khi load lịch sử, tool call mồ côi được thêm kết quả lỗi `INTERRUPTED`.
- [ ] Assistant message được lưu và gửi lại nguyên văn, gồm reasoning item / thinking block / thought signature.
- [ ] Kiểm tra `stop_reason` / `finish_reason`: không thực thi tool call bị cắt; xử lý `pause_turn`.
- [ ] Có giới hạn số bước, tổng token, thời gian, timeout từng tool, và phát hiện gọi trùng.
- [ ] Chạm giới hạn thì trả câu trả lời một phần (`tool_choice: none`), không trả lỗi chung.

### Bộ tool

- [ ] Tool ở mức tác vụ; số tool phù hợp với cách tổ chức ở [§3.1](#31-thiết-kế-bộ-tool-ít-mà-rõ).
- [ ] Description trả lời: làm gì, khi nào dùng, khi nào không, trả về gì, điều kiện trước.
- [ ] Không có tham số định danh / quyền hạn trong schema; lấy từ session.
- [ ] Strict mode bật tường minh khi provider hỗ trợ.
- [ ] Danh sách tool cố định trong phiên, thứ tự ổn định; giới hạn bằng `tool_choice` / `allowed_tools`.
- [ ] Description nằm trong version control và có eval đi kèm.

### Kết quả và lỗi

- [ ] Kết quả gọn, trường có nghĩa, thời gian tuyệt đối kèm múi giờ, ID ngắn.
- [ ] Có giới hạn kích thước; khi cắt thì ghi `truncated` và cách lấy tiếp.
- [ ] Lỗi theo một cấu trúc chung, có `code`, `message`, ví dụ đúng khi phù hợp, và `retryable`.
- [ ] Lỗi tạm thời được retry trong code trước khi báo model.

### Tác dụng phụ và bảo mật

- [ ] Tool ghi đi qua xác nhận từ người dùng ở kênh model không giả được, theo mức rủi ro.
- [ ] Idempotency key theo `(conversation_id, tool_call_id)` và kiểm tra trạng thái nghiệp vụ.
- [ ] Mỗi tool tự kiểm tra quyền trên từng lần gọi.
- [ ] Đã rà lethal trifecta: nếu có đủ ba yếu tố, áp dụng một mẫu thiết kế ở [§3.6](#36-hành-động-có-tác-dụng-phụ-và-bảo-mật) hoặc bỏ một yếu tố.
- [ ] Tool từ MCP / bên thứ ba: ghim phiên bản, lưu hash description, `allowed_tools` tối thiểu.

### Theo provider

- [ ] **OpenAI**: Responses cho kết quả đa phương thức; `strict: true` tường minh; gửi lại reasoning item theo nhóm; `phase: "commentary"` cho preamble; `require_approval` cho MCP.
- [ ] **Claude**: `tool_result` đứng đầu user message; không dùng `any` / `tool` trên model 5.5 / Fable 5.1 / Mythos 5.1; giữ thinking block rỗng; giới hạn strict 20 / 24 / 16; tool search khi nhiều tool.
- [ ] **Gemini**: thought signature nguyên vị trí; FC trước FR sau; tắt automatic function calling cho tool có tác dụng phụ; `VALIDATED` khi cần đủ tham số required.
- [ ] **Grok**: root `parameters` là object; tự cài approval nếu dùng MCP; gửi lại `encrypted_content`.
- [ ] **DeepSeek / Kimi / MiniMax / GLM**: gửi lại `reasoning_content` / toàn bộ assistant message; DeepSeek strict qua `/beta`; ID Kimi dạng `functions.<tên>:<số>`.
- [ ] **Self-hosted**: đúng `--tool-call-parser` và template cho họ model; chạy eval tool call sau mỗi lần nâng cấp engine hoặc đổi nhà cung cấp.

---

# Nguồn tham khảo

## OpenAI

| # | Resource | URL |
|---|---|---|
| 1 | Function calling guide | [developers.openai.com/api/docs/guides/function-calling](https://developers.openai.com/api/docs/guides/function-calling) |
| 2 | Reasoning models guide | [developers.openai.com/api/docs/guides/reasoning](https://developers.openai.com/api/docs/guides/reasoning) |
| 3 | Tool search | [developers.openai.com/api/docs/guides/tools-tool-search](https://developers.openai.com/api/docs/guides/tools-tool-search) |
| 4 | MCP and connectors | [developers.openai.com/api/docs/guides/tools-connectors-mcp](https://developers.openai.com/api/docs/guides/tools-connectors-mcp) |
| 5 | Prompt caching | [developers.openai.com/api/docs/guides/prompt-caching](https://developers.openai.com/api/docs/guides/prompt-caching) |
| 6 | Structured Outputs | [developers.openai.com/api/docs/guides/structured-outputs](https://developers.openai.com/api/docs/guides/structured-outputs) |
| 7 | Pricing | [developers.openai.com/api/docs/pricing](https://developers.openai.com/api/docs/pricing) |
| 8 | openai-python: Responses và Chat types | [github.com/openai/openai-python/tree/main/src/openai/types](https://github.com/openai/openai-python/tree/main/src/openai/types) |
| 9 | openai-openapi issue #536: ràng buộc ghép cặp reasoning item (cộng đồng) | [github.com/openai/openai-openapi/issues/536](https://github.com/openai/openai-openapi/issues/536) |
| 10 | openai-agents-python issue #1061: No tool output found (cộng đồng) | [github.com/openai/openai-agents-python/issues/1061](https://github.com/openai/openai-agents-python/issues/1061) |

## Claude (Anthropic)

| # | Resource | URL |
|---|---|---|
| 11 | Tool use overview | [platform.claude.com/docs/en/agents-and-tools/tool-use/overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) |
| 12 | Handle tool calls | [platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls) |
| 13 | Implement tool use | [platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use) |
| 14 | Define tools | [platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools) |
| 15 | Parallel tool use | [platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use) |
| 16 | Tool use with prompt caching | [platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching) |
| 17 | Tool search tool | [platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool) |
| 18 | Programmatic tool calling | [platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling) |
| 19 | Fine-grained tool streaming | [platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming](https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming) |
| 20 | Handling stop reasons | [platform.claude.com/docs/en/build-with-claude/handling-stop-reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons) |
| 21 | Structured outputs (giới hạn strict) | [platform.claude.com/docs/en/build-with-claude/structured-outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) |
| 22 | Claude API errors | [platform.claude.com/docs/en/api/errors](https://platform.claude.com/docs/en/api/errors) |
| 23 | Thinking | [platform.claude.com/docs/en/build-with-claude/thinking](https://platform.claude.com/docs/en/build-with-claude/thinking) |
| 24 | Web search tool | [platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) |
| 25 | Anthropic Engineering: Writing effective tools for agents | [anthropic.com/engineering/writing-tools-for-agents](https://www.anthropic.com/engineering/writing-tools-for-agents) |
| 26 | Anthropic Engineering: Advanced tool use | [anthropic.com/engineering/advanced-tool-use](https://www.anthropic.com/engineering/advanced-tool-use) |
| 27 | anthropic-sdk-python: tool helpers | [github.com/anthropics/anthropic-sdk-python/blob/main/tools.md](https://github.com/anthropics/anthropic-sdk-python/blob/main/tools.md) |

## Google Gemini

| # | Resource | URL |
|---|---|---|
| 28 | Function calling | [aistudio.google.com/docs/function-calling](https://aistudio.google.com/docs/function-calling) |
| 29 | Thought signatures | [aistudio.google.com/docs/thought-signatures](https://aistudio.google.com/docs/thought-signatures) |
| 30 | Vertex AI: Thought signatures | [docs.cloud.google.com/vertex-ai/generative-ai/docs/thought-signatures](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/thought-signatures) |
| 31 | Vertex AI: Function calling | [docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) |
| 32 | Tool combination | [aistudio.google.com/docs/tool-combination](https://aistudio.google.com/docs/tool-combination) |
| 33 | python-genai README (automatic function calling) | [github.com/googleapis/python-genai](https://github.com/googleapis/python-genai/blob/main/README.md) |

## xAI Grok

| # | Resource | URL |
|---|---|---|
| 34 | Function calling | [docs.x.ai/developers/tools/function-calling](https://docs.x.ai/developers/tools/function-calling) |
| 35 | Remote MCP | [docs.x.ai/developers/tools/remote-mcp](https://docs.x.ai/developers/tools/remote-mcp) |
| 36 | Models | [docs.x.ai/developers/models](https://docs.x.ai/developers/models) |

## Các provider Trung Quốc

| # | Resource | URL |
|---|---|---|
| 37 | DeepSeek: Thinking mode | [api-docs.deepseek.com/guides/thinking_mode](https://api-docs.deepseek.com/guides/thinking_mode) |
| 38 | DeepSeek: Tool calls | [api-docs.deepseek.com/guides/tool_calls](https://api-docs.deepseek.com/guides/tool_calls) |
| 39 | Kimi: thinking models | [platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model](https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model) |
| 40 | MiniMax: M3 function call | [platform.minimax.io/docs/guides/text-m3-function-call](https://platform.minimax.io/docs/guides/text-m3-function-call) |
| 41 | Z.ai: Stream tool call | [docs.z.ai/guides/tools/stream-tool](https://docs.z.ai/guides/tools/stream-tool) |
| 42 | Alibaba Model Studio: Qwen function calling | [alibabacloud.com/help/en/model-studio/qwen-function-calling](https://www.alibabacloud.com/help/en/model-studio/qwen-function-calling) |

## Self-hosted và nhà cung cấp model open-weight

| # | Resource | URL |
|---|---|---|
| 43 | vLLM: Tool calling | [docs.vllm.ai/en/latest/features/tool_calling/](https://docs.vllm.ai/en/latest/features/tool_calling/) |
| 44 | llama.cpp: Function calling | [github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md) |
| 45 | Ollama: Tool calling | [docs.ollama.com/capabilities/tool-calling](https://docs.ollama.com/capabilities/tool-calling) |
| 46 | MoonshotAI: K2 Vendor Verifier | [github.com/MoonshotAI/K2-Vendor-Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier) |
| 47 | vLLM blog: Kimi K2 tool calling accuracy | [vllm.ai/blog/Kimi-K2-Accuracy](https://vllm.ai/blog/Kimi-K2-Accuracy) |
| 48 | OpenRouter: Provider Variance, Introducing Exacto (10/2025) | [openrouter.ai/blog/announcements/provider-variance-introducing-exacto/](https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/) |
| 49 | SGLang issue #47167: tools + response_format (cộng đồng) | [github.com/sgl-project/sglang/issues/47167](https://github.com/sgl-project/sglang/issues/47167) |

## Thư viện

| # | Resource | URL |
|---|---|---|
| 50 | OpenAI Agents SDK: Running agents | [openai.github.io/openai-agents-python/running_agents](https://openai.github.io/openai-agents-python/running_agents) |
| 51 | Vercel AI SDK: Loop control | [ai-sdk.dev/docs/agents/loop-control](https://ai-sdk.dev/docs/agents/loop-control) |
| 52 | LangGraph: Graph API (recursion limit) | [docs.langchain.com/oss/python/langgraph/graph-api](https://docs.langchain.com/oss/python/langgraph/graph-api) |

## Benchmark, nghiên cứu và bài phân tích

| # | Resource | URL |
|---|---|---|
| 53 | Yao et al.: τ-bench (2024) | [arxiv.org/abs/2406.12045](https://arxiv.org/abs/2406.12045) |
| 54 | Barres et al.: τ²-bench (2025) | [arxiv.org/abs/2506.07982](https://arxiv.org/abs/2506.07982) |
| 55 | Berkeley Function Calling Leaderboard | [gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| 56 | Basu et al.: NESTFUL (2024) | [arxiv.org/abs/2409.03797](https://arxiv.org/abs/2409.03797) |
| 57 | Lu et al.: ToolSandbox (2024) | [arxiv.org/abs/2408.04682](https://arxiv.org/abs/2408.04682) |
| 58 | Luo et al.: MCP-Universe (2025) | [arxiv.org/abs/2508.14704](https://arxiv.org/abs/2508.14704) |
| 59 | Paramanayakam et al.: Less is More (DATE 2025) | [arxiv.org/abs/2411.15399](https://arxiv.org/abs/2411.15399) |
| 60 | Sneh et al.: ToolTweak (2025) | [arxiv.org/abs/2510.02554](https://arxiv.org/abs/2510.02554) |
| 61 | Faghih et al.: Tool Preferences in Agentic LLMs are Unreliable (EMNLP 2025) | [arxiv.org/abs/2505.18135](https://arxiv.org/abs/2505.18135) |
| 62 | Gan & Sun: RAG-MCP (2025) | [arxiv.org/abs/2505.03275](https://arxiv.org/abs/2505.03275) |
| 63 | Wang et al.: Executable Code Actions Elicit Better LLM Agents (CodeAct, ICML 2024) | [arxiv.org/abs/2402.01030](https://arxiv.org/abs/2402.01030) |
| 64 | Johnson, Pain, West: Natural Language Tools (2025) | [arxiv.org/abs/2510.14453](https://arxiv.org/abs/2510.14453) |
| 65 | Somma et al.: Natural Language Tools, replication study (2026) | [arxiv.org/abs/2607.03953](https://arxiv.org/abs/2607.03953) |
| 66 | Lindenbauer et al.: The Complexity Trap (JetBrains, 2025) | [arxiv.org/abs/2508.21433](https://arxiv.org/abs/2508.21433) |
| 67 | Chroma: Context Rot | [trychroma.com/research/context-rot](https://www.trychroma.com/research/context-rot) |
| 68 | Debenedetti et al.: AgentDojo (2024) | [arxiv.org/abs/2406.13352](https://arxiv.org/abs/2406.13352) |
| 69 | Debenedetti et al.: Defeating Prompt Injections by Design (CaMeL, 2025) | [arxiv.org/abs/2503.18813](https://arxiv.org/abs/2503.18813) |
| 70 | Beurer-Kellner et al.: Design Patterns for Securing LLM Agents against Prompt Injections (2025) | [arxiv.org/abs/2506.08837](https://arxiv.org/abs/2506.08837) |
| 71 | Simon Willison: The lethal trifecta for AI agents (06/2025) | [simonwillison.net/2025/Jun/16/the-lethal-trifecta/](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) |
| 72 | Invariant Labs: MCP Security Notification, Tool Poisoning Attacks (04/2025) | [invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks) |
| 73 | OpenTelemetry GenAI semantic conventions: Execute tool | [github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/execute-tool.md](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/execute-tool.md) |

---

Thông tin API trong tài liệu được ghi nhận tại **tháng 11/2026**, từ tài liệu chính thức, type definitions của SDK và repo của dự án. Báo cáo từ issue tracker và forum được ghi rõ là nguồn cộng đồng.

Các điểm còn cần kiểm tra trên endpoint: mặc định `parallel_tool_calls` và giới hạn số tool trên OpenAI Responses (tài liệu tham khảo cũ ghi mặc định `true`); nguyên văn thông báo lỗi thiếu thought signature của Gemini; mặc định `parallel_tool_calls` và `tool_choice` ở thinking mode của Qwen; số liệu về mức giảm hiệu năng khi bỏ nội dung suy luận của MiniMax; danh sách parser hiện tại của SGLang. Số liệu benchmark phản ánh model tại thời điểm công bố của từng nghiên cứu; dùng chúng để hiểu loại lỗi cần đo, còn quyết định chọn model hay nhà cung cấp nên dựa trên eval với dữ liệu của bạn.

