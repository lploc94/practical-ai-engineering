# Giảm độ trễ cho ứng dụng LLM

Một chatbot hỗ trợ khách hàng mất 15 giây cho mỗi câu trả lời. Đo từng bước thì thấy bốn việc chạy nối đuôi nhau:

1. Một lần gọi model viết lại câu hỏi cho đủ ngữ cảnh.
2. Một lần tìm trong vector store.
3. Một lần gọi model sinh câu trả lời 600 token.
4. Một lần gọi model kiểm tra câu trả lời trước khi gửi.

Người dùng nhìn màn hình trống suốt 15 giây, vì câu trả lời chỉ được gửi đi khi bước kiểm tra xong. Không có bước nào chậm bất thường; chúng chỉ cộng lại. Đổi sang model nhanh hơn chỉ cắt được một phần. Phần lớn thời gian nằm ở cách các bước được xếp và ở chỗ người dùng không thấy gì trong lúc chờ.

Bài này đi theo thứ tự:

- **Nền tảng — Phần 1:** một request tốn thời gian ở đâu (mạng, hàng đợi, prefill, reasoning, decode), các số đo TTFT, tokens/giây và tổng thời gian, khác biệt giữa độ trễ thực và độ trễ người dùng cảm nhận, và vì sao phải nhìn p99 chứ không nhìn trung bình.
- **Kỹ thuật — Phần 2:** streaming, prompt caching, chọn model, effort và các tier nhanh, chạy song song và speculative execution, các kỹ thuật phía serving cho model tự host, semantic cache.
- **Thiết kế và vận hành — Phần 3:** đo từng tầng, giảm số lượt gọi nối tiếp, region, timeout, hedging và retry, UX khi phải chờ, failure modes và checklist.

> Bản tháng 11/2026. Số liệu provider lấy từ tài liệu chính thức ngày 06–07/11/2026. Code trong bài chạy với `anthropic` 2.6.0 (Python), test bằng stream SSE giả lập qua `httpx2.MockTransport` và một server HTTP cục bộ, không gọi API thật. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

Tài liệu liên quan: [Streaming](../streaming/) (SSE, proxy, huỷ request), [Cost optimization](../cost-optimization/) (phần lớn đòn bẩy giảm độ trễ cũng giảm chi phí, và ngược lại), [Model routing](../model-routing/) (retry, fallback, priority tier), [Reasoning models](../reasoning-models/) (effort và thinking), [LLM Observability](../llm-observability/#13-metric-riêng-của-llm-token-ttft-tokensgiây-cache-hit-tỉ-lệ-refusal) (metric TTFT, tokens/giây), [Self-hosting LLM](../self-hosting-llm/).

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Độ trễ gồm những gì: mạng, hàng đợi, prefill, decode](#11-độ-trễ-gồm-những-gì-mạng-hàng-đợi-prefill-decode)
  - [1.2 TTFT, tokens/giây và tổng thời gian](#12-ttft-tokensgiây-và-tổng-thời-gian)
  - [1.3 Độ trễ cảm nhận và độ trễ thực](#13-độ-trễ-cảm-nhận-và-độ-trễ-thực)
  - [1.4 Vì sao độ trễ dao động: p50 và p99](#14-vì-sao-độ-trễ-dao-động-p50-và-p99)
- [Phần 2 — Kỹ thuật](#phần-2--kỹ-thuật)
  - [2.1 Streaming](#21-streaming)
  - [2.2 Prompt caching và prefill](#22-prompt-caching-và-prefill)
  - [2.3 Chọn model và chế độ nhanh (fast mode, priority tier)](#23-chọn-model-và-chế-độ-nhanh-fast-mode-priority-tier)
  - [2.4 Chạy song song và speculative execution](#24-chạy-song-song-và-speculative-execution)
  - [2.5 Speculative decoding và các kỹ thuật phía serving](#25-speculative-decoding-và-các-kỹ-thuật-phía-serving)
  - [2.6 Semantic cache](#26-semantic-cache)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Đo độ trễ từng tầng trong pipeline](#31-đo-độ-trễ-từng-tầng-trong-pipeline)
  - [3.2 Giảm số lượt gọi model nối tiếp](#32-giảm-số-lượt-gọi-model-nối-tiếp)
  - [3.3 Đặt region gần provider và người dùng](#33-đặt-region-gần-provider-và-người-dùng)
  - [3.4 Timeout, hedging và retry](#34-timeout-hedging-và-retry)
  - [3.5 UX khi phải chờ: trạng thái, kết quả từng phần](#35-ux-khi-phải-chờ-trạng-thái-kết-quả-từng-phần)
  - [3.6 Failure modes thường gặp](#36-failure-modes-thường-gặp)
  - [3.7 Checklist](#37-checklist)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

### Thành phần độ trễ và cách giảm

| Thành phần | Người dùng thấy ở đâu | Tăng theo | Đòn bẩy chính | Mục |
|---|---|---|---|---|
| Mạng | Trước chữ đầu tiên | Khoảng cách tới region, số lần mở kết nối mới | Giữ kết nối; WebSocket cho agent nhiều lượt tool; region gần | [§3.3](#33-đặt-region-gần-provider-và-người-dùng) |
| Hàng đợi ở provider | Trước chữ đầu tiên, dao động theo giờ | Tải chung của provider | Tier ưu tiên; tránh tier "flex"/"sheddable" cho đường người dùng chờ | [§2.3](#23-chọn-model-và-chế-độ-nhanh-fast-mode-priority-tier) |
| Prefill | Trước chữ đầu tiên | Số input token không đọc từ cache | Prompt caching, pre-warm cache, bớt context | [§2.2](#22-prompt-caching-và-prefill) |
| Reasoning | Trước chữ đầu tiên, có thể hàng chục giây | Effort, độ khó của câu hỏi | Effort thấp hơn, model nhỏ hơn | [§2.3](#23-chọn-model-và-chế-độ-nhanh-fast-mode-priority-tier) |
| Decode | Tốc độ chữ chạy; tổng thời gian | Số output token | Câu trả lời ngắn hơn, fast mode, speculative decoding (tự host) | [§2.3](#23-chọn-model-và-chế-độ-nhanh-fast-mode-priority-tier), [§2.5](#25-speculative-decoding-và-các-kỹ-thuật-phía-serving) |
| Các lượt nối tiếp | Mọi thứ ở trên nhân với số lượt | Số lần gọi model, tool, retrieval tuần tự | Gộp bước, chạy song song, bỏ bước không cần LLM | [§3.2](#32-giảm-số-lượt-gọi-model-nối-tiếp) |

### Đòn bẩy và mức hiệu quả đã công bố

| Đòn bẩy | Hiệu quả theo nguồn | Cái giá |
|---|---|---|
| Streaming | "The single most effective approach": giảm thời gian *chờ* xuống dưới khoảng một giây [9]. Không giảm tổng thời gian | Phải xử lý stream, lỗi giữa chừng, proxy ([Streaming](../streaming/)) |
| Bớt output token | "Cutting 50% of your output tokens may cut ~50% of your latency" (heuristic của OpenAI) [9] | Câu trả lời phải vẫn đủ |
| Bớt input token | "Cutting 50% of your prompt may only result in a 1–5% latency improvement", trừ khi context rất lớn [9] | Ít hiệu quả; ưu tiên cache |
| Prompt caching | Giảm thời gian xử lý input trước khi trả lời; không provider nào công bố con số phần trăm chung [3][14] | Prefix phải ổn định |
| Effort thấp | Opus 5.5: thời gian tới chữ trả lời đầu tiên 19,38 s ở `medium`, 6,34 s ở `low` (Artificial Analysis) [18] | Chất lượng giảm tuỳ tác vụ |
| Claude fast mode | Tới 2,5 lần output tokens/giây; không cải thiện TTFT [2] | Giá gấp đôi trên Opus 5.5 [2] |
| OpenAI Fast mode | "Up to 2.5× faster" với GPT-5.6 Sol [10] | Giá gấp đôi với GPT-6.x [15] |
| OpenAI WebSocket mode | Tới khoảng 40% nhanh hơn từ đầu tới cuối với rollout từ 20 tool call trở lên [12] | Kết nối tối đa 60 phút, cần tự khôi phục |
| Hedged request | Ở Google, hedge sau 10 ms đưa p99,9 từ 1.800 ms xuống 74 ms với 2% request thêm (BigTable, không phải LLM) [20] | Tốn thêm token; chỉ có ích khi độ trễ cao là ngẫu nhiên |
| Speculative decoding (tự host) | 2–3 lần trên T5-XXL [21]; 2,7–3,5 lần trên LLaMA2-Chat 70B [22] | Chỉ khi tự host; lợi ích giảm ở tải cao [26] |

### Số đo tham khảo theo model

Artificial Analysis đo các endpoint chính thức mỗi 6 giờ, dưới tải đồng thời, từ Google Cloud `us-central1`; con số trên trang là trung bình trung vị của 5 ngày gần nhất trên nhiều loại workload [19]. "Time to first answer token" tính cả thời gian reasoning, nên so được model reasoning với model thường [19]. Lát cắt ngày 07/11/2026 [18]:

| Model (mức effort) | Thời gian tới chữ trả lời đầu tiên | Thời gian mỗi tác vụ |
|---|---|---|
| GPT-6.1 Terra (non-reasoning) | 1,69 s | 2,2 phút |
| GPT-6.1 Terra (low) | 2,28 s | 1,1 phút |
| GPT-6.1 Sol (medium) | 6,53 s | 4,2 phút |
| Claude Sonnet 5.5 (low) | 3,55 s | 2,6 phút |
| Claude Sonnet 5.5 (high) | 13,05 s | 7,0 phút |
| Claude Haiku 5.5 (low) | 9,33 s | 2,2 phút |
| Claude Opus 5.5 (low) | 6,34 s | 3,0 phút |
| Claude Opus 5.5 (medium) | 19,38 s | 6,8 phút |
| Claude Opus 5.5 (xhigh) | 81,55 s | 16,2 phút |
| Gemini 3.8 Flash (medium) | 10,01 s | 11,8 phút |
| Gemini 4 Carbon (low) | 8,38 s | 36,3 phút |

Bảng để thấy độ lớn, không để chọn model. Ba điều đáng chú ý. Effort đổi thời gian chờ nhiều hơn đổi model: cùng Opus 5.5, `xhigh` chờ lâu gấp khoảng 13 lần `low`. Model "nhanh nhất" theo tài liệu provider chưa chắc có thời gian chờ ngắn nhất ở mọi cấu hình: Haiku 5.5 ở `low` chờ lâu hơn Sonnet 5.5 ở `low` trong lát cắt này. Và số của bạn sẽ khác, vì region, độ dài prompt, độ dài output và giờ trong ngày đều khác. Đo trên traffic của mình ([§3.1](#31-đo-độ-trễ-từng-tầng-trong-pipeline)).

---

# Phần 1 — Nền tảng

## 1.1 Độ trễ gồm những gì: mạng, hàng đợi, prefill, decode

Một lần gọi model đi qua các đoạn nối tiếp:

```text
gửi ─[mạng]─[hàng đợi]─[prefill]─[reasoning]─[decode token 1 ... token N]─[mạng]─ nhận xong
     └──────────────── TTFT: tới chữ đầu tiên ─────────────┘└──── decode ────┘
```

- **Mạng:** DNS, TCP, TLS khi mở kết nối mới, rồi thời gian đi về giữa ứng dụng và region của provider. SDK chính thức giữ kết nối và dùng lại giữa các request, nên phần mở kết nối thường chỉ trả một lần. Với agent gọi nhiều tool liên tiếp, phần mạng lặp lại ở mọi lượt; OpenAI khuyến nghị WebSocket cho trường hợp này vì "without a persistent connection, network overhead can reduce the latency gains" [11].
- **Hàng đợi:** request chờ tới lượt được xử lý. Provider không công bố con số; nó lộ ra ở chỗ cùng một request có TTFT khác nhau theo giờ trong ngày. Các tier ưu tiên ([§2.3](#23-chọn-model-và-chế-độ-nhanh-fast-mode-priority-tier)) chủ yếu tác động vào đoạn này.
- **Prefill:** model đọc toàn bộ input trong một lượt song song. Đoạn này bị giới hạn bởi sức tính của GPU và tăng theo số input token chưa có trong cache ([Cost optimization §1.3](../cost-optimization/#13-vì-sao-output-token-thường-đắt-hơn-input-token) có ví dụ tính của Hugging Face) [27].
- **Reasoning:** với model có thinking, model sinh token suy luận trước khi viết chữ trả lời đầu tiên. Thinking token sinh ra cũng bằng decode, nên đắt về thời gian như output. Nếu provider không stream nội dung thinking, người dùng chỉ thấy một khoảng lặng.
- **Decode:** model sinh từng token, mỗi token một lượt đọc toàn bộ trọng số từ bộ nhớ. Đoạn này bị giới hạn bởi băng thông bộ nhớ, và thời gian tăng gần tuyến tính theo số output token [27].

Từ đó ra quy tắc mà OpenAI tóm lại trong hướng dẫn tối ưu độ trễ: sinh token "is almost always the highest latency step"; bớt một nửa output có thể bớt khoảng một nửa độ trễ, bớt một nửa prompt thường chỉ được 1–5% [9]. Ngoại lệ là context rất lớn (tài liệu dài, nhiều ảnh), khi prefill đủ lớn để đáng kể.

## 1.2 TTFT, tokens/giây và tổng thời gian

Ba số đo, ba câu hỏi khác nhau:

| Số đo | Câu hỏi | Gồm |
|---|---|---|
| **TTFT** (time to first token) | Người dùng chờ bao lâu trước khi thấy chữ? | Mạng, hàng đợi, prefill, reasoning |
| **Tokens/giây** (hoặc khoảng cách giữa hai token) | Chữ chạy nhanh hay chậm? | Decode |
| **Tổng thời gian** (end-to-end) | Bao lâu thì có kết quả đầy đủ để dùng tiếp? | Tất cả |

Quan hệ gần đúng: `tổng ≈ TTFT + số output token ÷ tokens/giây`. Câu trả lời 600 token ở 60 tokens/giây là 10 giây decode, cộng TTFT.

Hai chỗ hay đo sai:

- **"Token đầu tiên" là token nào.** Với model reasoning, token đầu tiên có thể là token suy luận, không phải chữ người dùng thấy. Artificial Analysis vì vậy tách hai số: "time to first token" (với model trả token reasoning, là token reasoning đầu tiên) và "time to first answer token" (tính cả thời gian reasoning) [19]. Khi ghi metric, đo tới **chữ trả lời đầu tiên**, vì đó là điều người dùng trải nghiệm.
- **Tokens/giây tính trên khoảng nào.** Artificial Analysis tính output speed bằng số output token (kể cả reasoning) chia cho toàn bộ thời gian request, *gồm cả TTFT* [19]. Đó không phải tốc độ decode thuần. Khi so số của bạn với số công bố, kiểm tra cùng định nghĩa.

Tính tokens/giây từ `usage` có một bẫy với model reasoning: `output_tokens` gồm cả thinking token sinh *trước* chữ đầu tiên, nên chia `output_tokens` cho khoảng thời gian sau chữ đầu tiên sẽ ra tốc độ cao hơn thực tế. Code đo ở [§3.1](#31-đo-độ-trễ-từng-tầng-trong-pipeline) trả `None` cho trường hợp này thay vì một con số sai.

## 1.3 Độ trễ cảm nhận và độ trễ thực

Người dùng không đo tổng thời gian. Họ cảm nhận khoảng thời gian không có gì xảy ra. OpenAI phân biệt "waiting" và "watching progress happen" [9]:

- **Streaming và chunking** giảm thời gian *chờ* thật sự: người dùng đọc trong lúc model viết, nên đọc xong sớm hơn dù tổng thời gian sinh không đổi [9].
- **Hiện các bước và trạng thái tải** chủ yếu có tác dụng tâm lý, nhưng vẫn đáng làm khi phải chờ [9].

Vì vậy hai hệ thống có cùng tổng thời gian 10 giây có thể cho trải nghiệm rất khác nhau: một cái trống màn hình 10 giây, một cái hiện chữ sau 1 giây. Tối ưu độ trễ nên bắt đầu ở TTFT của đường người dùng nhìn thấy, rồi mới tới tổng thời gian.

Ngược lại, với bước không có người xem (một lần gọi phân loại giữa pipeline, một bước của agent), chỉ tổng thời gian có ý nghĩa: stream bước đó không làm gì nhanh hơn. Với bước cuối cùng mà người dùng đọc, TTFT là số quan trọng nhất.

## 1.4 Vì sao độ trễ dao động: p50 và p99

Trung bình che mất thứ người dùng nhớ. Một request cứ một trăm lần lại mất 30 giây làm người dùng thường xuyên gặp nó, nhất là khi một lượt trả lời gồm nhiều lần gọi.

Dean và Barroso phân tích hiện tượng này trong "The Tail at Scale". Một server thường trả lời trong 10 ms nhưng p99 là 1 giây. Request chỉ chạm một server thì 1 trên 100 request chậm. Request phải chờ 100 server như thế thì 63% request chậm hơn 1 giây [20]. Trong một dịch vụ thật của Google mà họ đo, p99 của một request lẻ là 10 ms, nhưng p99 để mọi request con xong là 140 ms; riêng việc chờ 5% request con chậm nhất chiếm một nửa con số đó [20].

Ứng dụng LLM có cấu trúc giống vậy theo chiều nối tiếp: một lượt trả lời gồm nhiều bước, và tổng p99 bị kéo bởi bước tệ nhất trong mỗi lần chạy. Nguồn dao động chính:

1. **Độ dài output khác nhau.** Cùng một prompt có thể sinh 50 hoặc 2.000 token. Decode tỉ lệ với số token, nên đuôi phân phối độ trễ thường là đuôi phân phối độ dài.
2. **Reasoning khác nhau.** Model tự quyết định nghĩ bao lâu theo độ khó và effort.
3. **Cache hit hay miss.** Request đầu tiên sau khi cache hết hạn phải prefill lại toàn bộ.
4. **Tải của provider.** Hàng đợi dài hơn ở giờ cao điểm; request có thể bị `429`, `529` hoặc bị hạ tier ([§2.3](#23-chọn-model-và-chế-độ-nhanh-fast-mode-priority-tier)).
5. **Retry.** Một lần retry sau backoff có thể nhân đôi thời gian của request đó.

Báo cáo p50, p95, p99 riêng cho TTFT và tổng thời gian, tách theo model, effort, tier và độ dài output. Khi p99 xấu, xem trước tiên là độ dài output và số lần retry của các request ở đuôi.

---

# Phần 2 — Kỹ thuật

## 2.1 Streaming

Streaming là đòn bẩy đầu tiên cho mọi câu trả lời người dùng đọc. Nó không giảm thời gian model làm việc, nhưng chuyển chữ đầu tiên từ cuối quá trình lên đầu [9]. Anthropic xếp nó cùng chọn model và rút ngắn prompt là ba cách giảm độ trễ chính [1].

Phần kỹ thuật (SSE, event theo provider, proxy, lỗi giữa chừng, huỷ) ở [Streaming](../streaming/). Những điểm liên quan trực tiếp tới độ trễ:

- **Stream chỉ có ích khi đi tới tận người dùng.** Một proxy hay backend gom hết stream rồi mới gửi xuống làm mất toàn bộ lợi ích ([Streaming §3.2](../streaming/#32-stream-qua-proxy-load-balancer-và-cdn)).
- **Bước có hậu xử lý thì xử lý theo đoạn.** Nếu output cần kiểm duyệt hay dịch trước khi hiển thị, stream về backend và xử lý từng đoạn thay vì chờ cả câu trả lời [9].
- **Request dài nên stream dù không ai xem.** Anthropic khuyên dùng streaming hoặc Batch API cho request có thể kéo dài, nhất là trên 10 phút, và tránh `max_tokens` lớn khi không stream; SDK kiểm tra trước và từ chối request không stream nếu ước tính vượt 10 phút [7].
- **Fast mode phát huy rõ nhất khi stream**, vì nó tăng tốc độ chữ chạy, không tăng tốc chữ đầu [2].

## 2.2 Prompt caching và prefill

Cache prefix giúp provider không phải xử lý lại phần đầu prompt đã thấy. Lợi ích về độ trễ có thật nhưng không được lượng hoá công khai:

- Anthropic: 5 phút và 1 giờ "behave the same with respect to latency"; "you generally see improved time-to-first-token for long documents" [3].
- OpenAI: cache "reduce[s] the time spent processing input before the response starts" [14], nhưng thời hạn tối thiểu của cache chỉ quyết định giá đọc cache và "does not guarantee that every request reuses stored KV tensors or has lower latency" [14].

Hai hệ quả. Lợi ích độ trễ lớn khi phần đã cache dài (tài liệu, lịch sử hội thoại dài, nhiều tool) và gần như không đáng kể với prompt ngắn, khớp với nhận xét bớt input chỉ được 1–5% [9]. Và vì không có con số chung, phải đo: so TTFT của request có `cache_read_input_tokens` lớn với request cùng loại bị cache miss, trên traffic của bạn.

### Pre-warm: trả trước lần cache miss

Request đầu tiên của một phiên luôn là cache miss. Với Claude, có thể nạp cache trước khi người dùng gửi gì: gửi request với `max_tokens: 0`. API đọc prompt, ghi cache tại các breakpoint `cache_control`, rồi trả về ngay, không sinh output [3]. Ví dụ: pre-warm system prompt và tool khi người dùng mở cửa sổ chat. Ba điều kiện để request thật dùng được entry vừa ghi [3]:

- Đặt `cache_control` tường minh trên block cuối cùng *dùng chung* với request sau (thường là system prompt hoặc tool), không phải trên message giữ chỗ. Cache tự động đặt breakpoint ở block cuối, tức đúng message giữ chỗ, nên không dùng được cho việc này.
- Dùng cùng cấu hình thinking và `output_config.effort` với request thật, vì chúng là một phần của prompt.
- Không dùng trong Batch: pre-warm nhắm vào TTFT, thứ không có ý nghĩa với batch [3].

`max_tokens: 0` bị từ chối khi request có thinking ở chế độ `enabled`, structured output, hoặc ép gọi tool ([Cost optimization §2.2](../cost-optimization/#hoà-vốn-và-ttl)).

### Những thứ làm mất cache và làm chậm

Chi tiết theo provider ở [Multi-turn §3.3](../multi-turn-conversation/#33-những-thứ-âm-thầm-làm-vỡ-cache). Hai điểm riêng cho độ trễ:

- **Đổi giữa fast mode và tốc độ thường làm mất cache.** Anthropic: "Requests at different speeds do not share cached prefixes" [2]; bảng invalidation của trang prompt caching ghi đổi speed làm mất cache của system và messages [3]. Fallback từ fast sang thường khi bị `429` vì vậy có thể chậm hơn chờ fast mode, vì phải prefill lại từ đầu.
- **Đổi effort giữa hội thoại.** Đặt effort ở cấp request rồi đổi làm vỡ cache; Opus 5.5 hỗ trợ đổi effort theo từng message (beta) để giữ cache [4].

## 2.3 Chọn model và chế độ nhanh (fast mode, priority tier)

### Model và effort

Model nhỏ thường nhanh hơn [9]. Trang tổng quan model của Anthropic xếp độ trễ tương đối: Fable 5.1 "Slower", Opus 5.5 "Moderate", Sonnet 5.5 "Fast", Haiku 5.5 "Fastest", và ghi rõ độ trễ thực tế phụ thuộc độ dài prompt, độ dài output và effort [5].

Với model reasoning, effort thường là biến lớn hơn kích thước model. Bảng ở [Tra cứu nhanh](#số-đo-tham-khảo-theo-model) cho thấy cùng Opus 5.5, thời gian tới chữ trả lời đầu tiên đi từ 6,34 s (`low`) tới 81,55 s (`xhigh`) [18]. Anthropic mô tả effort là nút chính cho đánh đổi giữa trí tuệ, độ trễ và chi phí, và khuyên hạ xuống `medium` hoặc `low` cho việc thường ngày hoặc nhạy với độ trễ khi eval cho thấy chất lượng giữ được [4]. Hướng dẫn cụ thể theo model [4]:

| Model | Effort mặc định | Gợi ý cho việc nhạy độ trễ |
|---|---|---|
| Opus 5.5 | `medium` | Hạ xuống `low` khi eval cho phép; adaptive thinking luôn bật, không tắt được |
| Sonnet 5.5 | `high` | Chat và việc nhạy độ trễ: bắt đầu ở `medium` hoặc `low` |
| Haiku 5.5 | `medium` | `low` cho chat, tác vụ tool ngắn, request đơn giản khối lượng lớn |

Hai bẫy khi nâng model: Opus 5.5 mặc định `medium` trong khi Opus 5 mặc định `high`, nên request không đặt effort sẽ chạy thấp hơn một bậc sau khi nâng [4]; và mức effort của Sonnet 5.5 đã được hiệu chỉnh lại so với Sonnet 5 [4]. Đặt effort tường minh và chạy lại eval cả chất lượng lẫn độ trễ.

Ví dụ của Anthropic cho việc nhạy độ trễ là Haiku 5.5 ở `low`, chừa chỗ trong `max_tokens` cho thinking [1]:

```python
message = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=1024,
    output_config={"effort": "low"},
    messages=[{"role": "user", "content": "Summarize this customer feedback in 2 sentences: ..."}],
)
```

### Claude fast mode

Fast mode chạy cùng model với cấu hình inference nhanh hơn, không đổi chất lượng [2]:

- **Bật:** `speed: "fast"` trên Messages API. Có trên Opus 5.5; research preview trên Opus 5 và Opus 4.8 (cần beta header `fast-mode-2026-02-01`). Fast mode trên Opus 4.8 bị gỡ ngày 30/11/2026 [2].
- **Nhanh ở đâu:** "up to 2.5x higher output tokens per second"; lợi ích tập trung ở tokens/giây, *không* ở TTFT [2]. Câu trả lời dài xong sớm hơn; chữ đầu không đến sớm hơn.
- **Giá:** Opus 5.5 8 USD input, 40 USD output mỗi triệu token, gấp đôi giá thường; hệ số cache và data residency áp lên trên [2].
- **Rate limit riêng.** Vượt thì `429` kèm `retry-after`; SDK tự retry 2 lần mặc định [2].
- **Kiểm tra đã chạy fast chưa:** `usage.speed` là `"fast"` hoặc `"standard"`. Model không hỗ trợ trả lỗi, trừ Opus 4.6: request có `speed: "fast"` và beta header vẫn thành công nhưng chạy tốc độ thường [2].
- **Không dùng được** với Batch API và với cam kết Priority Tier [2].

Nếu muốn chạy tốc độ thường thay vì chờ capacity của fast mode, tài liệu gợi ý đặt `max_retries=0` cho request fast, bắt `RateLimitError` và gửi lại không có `speed` [2]. Nhớ rằng lần gửi lại là một lần cache miss ([§2.2](#những-thứ-làm-mất-cache-và-làm-chậm)).

### Claude Priority Tier

Priority Tier ưu tiên request trước mọi request khác để giảm lỗi quá tải, nhưng đã là tính năng cũ: cam kết capacity không còn bán, tổ chức đã có hợp đồng dùng tiếp tới hết hạn [6]. Cần capacity bảo đảm thì liên hệ sales [6]. Với dự án mới trên Claude API, đòn bẩy tốc độ là model, effort và fast mode.

### OpenAI: Fast mode và Ultrafast mode

- **Fast mode** là tên mới của Priority processing từ 30/07/2026; dùng `service_tier: "fast"` hoặc `"priority"` [10]. Hứa hẹn "up to 2.5× faster speeds and more consistent latency"; tốc độ 2,5 lần được nêu cho `gpt-5.6-sol` [10]. Giá gấp đôi giá chuẩn với GPT-6.x (ví dụ gpt-6.1-sol 4 USD và 20 USD so với 2 và 10), gấp 2,5 lần với gpt-5.5 [15]. Cache discount vẫn áp dụng [10].
- **Ramp rate.** Traffic tăng quá nhanh thì một phần request bị hạ xuống tốc độ chuẩn và tính giá chuẩn, response trả `service_tier: "default"`. Quy tắc gợi ý: khi đã ở mức 1 triệu input token mỗi phút, tăng không quá 50% mỗi 15 phút; không chạy ETL hay batch lớn bằng Fast mode [10].
- **Đọc `service_tier` trong response** để biết request thực sự chạy tier nào. Với GPT-5.6 trở về trước, response luôn ghi `priority` dù request gửi `fast` [10].
- **SLA:** Fast mode cho GPT-6 Astra không kèm SLA độ trễ [10].
- **Ultrafast mode** là tier nhanh nhất, cho `gpt-6-astra` và `gpt-6.1-sol` (`service_tier: "ultrafast"`), rate limit riêng, khuyến nghị mạnh WebSocket [11]. Trang không công bố số tốc độ [11]. Giá gấp 6 lần giá chuẩn: gpt-6.1-sol 12 USD input, 60 USD output mỗi triệu token [15].

### Gemini: Priority và Flex

Gemini có bốn chế độ xử lý. Ba chế độ đầu chọn bằng tham số `service_tier` (mặc định Standard); Batch là Batch API riêng, chạy bất đồng bộ với thời gian trả kết quả tới 24 giờ, không phải một giá trị của `service_tier` [16][17]:

| Tier | Giá | Độ trễ mục tiêu | Khi quá tải |
|---|---|---|---|
| Priority | Cao hơn Standard 75–100% | Vài giây | Tự hạ xuống Standard, tính giá Standard, không trả lỗi |
| Standard | Giá đầy đủ | Vài giây tới vài phút | — |
| Flex | Giảm 50% | 1–15 phút | `503` hoặc `429`; không tự nâng lên Standard |
| Batch | Giảm 50% | Tới 24 giờ | — |

Priority có rate limit mặc định bằng 0,3 lần rate limit chuẩn; header `x-gemini-service-tier` trong response cho biết request có bị hạ xuống `standard` không [16]. Flex dùng capacity "sheddable" và có thể bị đẩy ra khi traffic chuẩn tăng [17]: đừng đặt nó trên đường người dùng chờ.

### Nguyên tắc chung cho các tier nhanh

- **Kiểm tra tier thực tế ở mỗi response** (`usage.speed`, `service_tier`, `x-gemini-service-tier`). Cả ba provider đều có trường hợp request không chạy ở tier bạn yêu cầu mà không báo lỗi.
- **Fast mode tăng tốc decode, không tăng tốc chữ đầu** (Claude ghi rõ [2]). Nếu vấn đề là TTFT, nhìn vào reasoning, prefill và hàng đợi trước.
- **Chỉ dùng cho đường người dùng chờ.** Batch, ETL, eval không cần và làm hoá đơn tăng gấp đôi trở lên.

## 2.4 Chạy song song và speculative execution

### Song song hoá các bước độc lập

Bước không phụ thuộc nhau thì chạy cùng lúc [9]. Ví dụ ở mở bài: phân loại ý định và tìm tài liệu có thể chạy song song nếu tìm kiếm không cần kết quả phân loại. Với tool, Claude và OpenAI đều cho model gọi nhiều tool trong một lượt; ứng dụng nên chạy chúng song song thay vì lần lượt ([Tool use §1.4](../tool-use/#14-gọi-nhiều-tool-song-song-và-tuần-tự)).

OpenAI WebSocket mode cho phép chạy nhiều hội thoại song song trên một kết nối qua `stream_id`: tối đa 16 response đang chạy và 32 `stream_id` có tên mỗi kết nối; request cùng `stream_id` chạy tuần tự, khác `stream_id` thì chạy đồng thời [12].

### Speculative execution: chạy trước bước sau, bỏ nếu đoán sai

Khi bước B phụ thuộc bước A nhưng A gần như luôn cho một kết quả, chạy A và B cùng lúc; nếu A ra kết quả khác dự đoán thì huỷ B [9]. Trường hợp điển hình là kiểm duyệt: chạy kiểm duyệt input cùng lúc với sinh câu trả lời, và huỷ câu trả lời nếu input bị chặn. Đoán đúng thì bước A gần như không thêm độ trễ [9].

Cái giá là token của nhánh bị huỷ. Với Claude, request bị client ngắt giữa chừng vẫn bị tính tiền; các provider khác nhau ở việc có dừng sinh khi bị ngắt không ([Streaming §3.7](../streaming/#37-người-dùng-huỷ-dừng-request-phía-provider-để-không-tốn-token)). Speculative execution đáng làm khi xác suất đoán đúng cao và nhánh bị huỷ thường bị huỷ sớm.

Một trường hợp quan trọng: nếu nhánh chạy trước là câu trả lời đang stream cho người dùng, phải giữ lại output cho tới khi bước kiểm tra xong. Ngược lại, người dùng có thể đã thấy nội dung lẽ ra bị chặn.

### Predicted Outputs (OpenAI)

Khi phần lớn output đã biết trước (sửa một file code, sửa một đoạn văn bản), Chat Completions nhận tham số `prediction` chứa nội dung dự đoán. Model xác nhận phần khớp thay vì sinh lại từng token, nên nhanh hơn [13]. Giới hạn [13]:

- Chỉ hỗ trợ GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano. Không có model GPT-5 hay GPT-6 trong danh sách.
- Không dùng được với tool, `n > 1`, `logprobs`, `max_completion_tokens`, audio.
- Token dự đoán không được dùng (`rejected_prediction_tokens`) vẫn tính tiền như output.

## 2.5 Speculative decoding và các kỹ thuật phía serving

Phần này chỉ áp dụng khi bạn tự host model. Với API, provider đã quyết định thay bạn. Cách chọn inference engine và GPU ở [Self-hosting LLM](../self-hosting-llm/).

### Speculative decoding

Decode chậm vì sinh K token cần K lượt chạy model tuần tự. Speculative decoding dùng một cách đoán rẻ (model nhỏ, hoặc một đầu dự đoán gắn vào model lớn) để đề xuất vài token, rồi model lớn kiểm cả cụm trong một lượt. Leviathan và cộng sự chứng minh cách lấy mẫu này không đổi phân phối output, và đo được 2–3 lần nhanh hơn trên T5-XXL với output giống hệt [21]. EAGLE đạt 2,7–3,5 lần giảm độ trễ trên LLaMA2-Chat 70B mà giữ nguyên phân phối [22].

Thực tế trong vLLM [26]:

- Speculative decoding nhằm giảm độ trễ giữa hai token ở tải thấp và vừa, khi decode bị giới hạn bởi băng thông bộ nhớ.
- Phương pháp dựa trên model (EAGLE, MTP, draft model, PARD) giảm độ trễ nhiều nhất; n-gram và suffix decoding cho mức tăng khiêm tốn nhưng không thêm tải lúc cao điểm.
- Trang tài liệu chỉ đưa bảng định tính và ghi "Real gains depend on your model family, traffic pattern, hardware, and sampling settings".
- Về lý thuyết là lossless trong giới hạn số học dấu phẩy động; nhưng vLLM không bảo đảm logprob ổn định, và đổi batch size có thể làm output khác nhau giữa các lần chạy. Phương pháp FLy cố ý nới bước kiểm và không giữ phân phối.

### Chunked prefill và batching

Một request prefill dài chen vào batch đang decode làm mọi request khác khựng lại. Sarathi-Serve chia prefill thành các khúc gần bằng nhau và xếp chúng xen với decode mà không dừng decode đang chạy; dưới ràng buộc độ trễ đuôi, năng lực phục vụ tăng 2,6 lần với Mistral-7B trên một A100 và tới 5,6 lần với Falcon-180B [23].

Trong vLLM V1, chunked prefill bật mặc định khi có thể; scheduler ưu tiên decode rồi mới xếp prefill vào phần còn lại của ngân sách `max_num_batched_tokens` [25]. Tham số này là núm chỉnh giữa hai số đo [25]:

- Nhỏ (ví dụ 2.048): khoảng cách giữa hai token tốt hơn, vì ít prefill chen vào decode.
- Lớn: TTFT tốt hơn, vì xử lý được nhiều token prefill mỗi batch. Cho throughput tối ưu, vLLM gợi ý trên 8.192, nhất là model nhỏ trên GPU lớn.

PagedAttention quản lý KV cache theo trang để giảm phân mảnh, cho throughput gấp 2–4 lần ở cùng mức độ trễ so với các hệ thống trước nó [24]. Điểm chung của các kỹ thuật này: tăng batch tăng throughput nhưng có thể làm từng request chậm đi. Chọn cấu hình theo SLO p95/p99 của TTFT và khoảng cách giữa token, không theo throughput tối đa.

## 2.6 Semantic cache

Cache hit trả lời ngay, không gọi model: độ trễ gần bằng một lần tra vector store. Đây là cách nhanh nhất để trả lời, và cũng rủi ro nhất: semantic cache trả câu trả lời của một câu hỏi *gần giống*, và trong benchmark của AWS khoảng 8% câu trả lời lấy từ cache không đúng ý người hỏi ngay cả ở ngưỡng tương đồng 0,99. Số liệu, điều kiện dùng được và cách đo ở [Cost optimization §3.4](../cost-optimization/#34-semantic-cache-và-cache-kết-quả).

Cho độ trễ, thêm hai điểm:

- **Tính cả thời gian tra cache vào đường đi.** Miss thì request chậm thêm đúng thời gian embed câu hỏi và tra vector store. Với tỉ lệ hit thấp, semantic cache làm p50 tệ đi.
- **Exact cache trên câu hỏi gợi ý sẵn** (nút "Câu hỏi thường gặp" trên giao diện) không có rủi ro trả sai và cho độ trễ gần 0. OpenAI gợi ý cùng hướng: input bị giới hạn (chọn danh mục) thì có thể sinh trước câu trả lời [9].

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Đo độ trễ từng tầng trong pipeline

### Ghi gì cho mỗi lần gọi model

- Thời điểm gửi, event đầu tiên, **chữ trả lời đầu tiên**, kết thúc.
- `usage`: input, cache read, cache write, output.
- Model, effort, tier thực tế (`usage.speed`, `service_tier`), region hoặc endpoint.
- Số lần retry, có fallback không, request có bị huỷ không.

Mỗi bước khác của pipeline (retrieval, tool, kiểm duyệt, hàng đợi nội bộ) là một span riêng trong trace. Cấu trúc trace cho RAG, tool use và agent ở [LLM Observability §3.2](../llm-observability/#32-trace-cho-rag-tool-use-và-agent-nhiều-bước). Không có trace theo từng span thì không biết 15 giây ở mở bài nằm ở đâu.

### Code: đo một stream Claude và đặt hạn chờ

Hàm dưới đây đọc raw event của `AsyncAnthropic().messages.create(stream=True)`, trả về text, ghi thời điểm, và áp hai hạn chờ:

- `first_text_s`: tối đa bao lâu tới chữ trả lời đầu tiên, gồm cả hàng đợi, prefill và thinking.
- `idle_s`: tối đa bao lâu giữa hai delta, chỉ áp sau chữ đầu tiên. Trước chữ đầu tiên, model có thể đang nghĩ mà không gửi event nào; Claude có gửi event `ping` nhưng tài liệu chỉ nói chúng "might" xuất hiện, không hứa tần suất [8].

```python
import asyncio
import time
from dataclasses import dataclass


class StreamStalled(Exception):
    """Không có chữ đầu tiên trước hạn, hoặc stream ngừng quá lâu giữa hai delta."""


@dataclass
class Timing:
    start: float
    first_event: float | None = None     # event đầu tiên (message_start): đã qua mạng và hàng đợi
    first_text: float | None = None      # text_delta đầu tiên: TTFT người dùng thấy
    end: float | None = None
    output_tokens: int | None = None
    had_thinking: bool = False
    max_gap: float = 0.0                 # khoảng lặng dài nhất sau chữ đầu tiên

    @property
    def ttft(self) -> float | None:
        return None if self.first_text is None else self.first_text - self.start

    @property
    def e2e(self) -> float | None:
        return None if self.end is None else self.end - self.start

    @property
    def tokens_per_s(self) -> float | None:
        """Tốc độ sinh sau chữ đầu tiên. Có thinking thì output_tokens gồm cả token
        sinh trước chữ đầu, nên không tính được từ usage: trả None."""
        if self.had_thinking or not self.output_tokens or self.first_text is None or self.end is None:
            return None
        span = self.end - self.first_text
        return (self.output_tokens - 1) / span if span > 0 and self.output_tokens > 1 else None


async def read_stream(events, timing: Timing, *, first_text_s: float, idle_s: float, clock=time.monotonic):
    """Đọc raw event của `client.messages.create(stream=True)` (AsyncAnthropic), yield text,
    ghi thời điểm vào `timing`. `first_text_s` phải đủ cho cả hàng đợi, prefill và thinking;
    `idle_s` chỉ áp sau chữ đầu tiên, vì lúc model đang nghĩ có thể không có event nào."""
    it = aiter(events)
    last = None
    while True:
        if timing.first_text is None:
            timeout = first_text_s - (clock() - timing.start)
        else:
            timeout = idle_s
        try:
            event = await asyncio.wait_for(anext(it), timeout=max(timeout, 0))
        except StopAsyncIteration:
            break
        except TimeoutError:
            phase = "chữ đầu tiên" if timing.first_text is None else "delta kế tiếp"
            raise StreamStalled(f"quá hạn chờ {phase}") from None
        now = clock()
        if timing.first_event is None:
            timing.first_event = now
        if event.type == "content_block_start" and event.content_block.type in ("thinking", "redacted_thinking"):
            timing.had_thinking = True
        elif event.type == "content_block_delta" and event.delta.type == "text_delta":
            if timing.first_text is None:
                timing.first_text = now
            else:
                timing.max_gap = max(timing.max_gap, now - last)
            last = now
            yield event.delta.text
        elif event.type == "message_delta":
            timing.output_tokens = event.usage.output_tokens   # số cộng dồn, event cuối là tổng
```

Cách dùng:

```python
client = anthropic.AsyncAnthropic()
timing = Timing(start=time.monotonic())
stream = await client.messages.create(model="claude-sonnet-5-5", max_tokens=2048,
                                      messages=messages, stream=True)
async with stream:                       # thoát khối là đóng kết nối, kể cả khi StreamStalled
    async for text in read_stream(stream, timing, first_text_s=30, idle_s=20):
        await send_to_user(text)
log_latency(ttft=timing.ttft, e2e=timing.e2e, tokens_per_s=timing.tokens_per_s,
            max_gap=timing.max_gap, output_tokens=timing.output_tokens)
```

Các con số `first_text_s=30, idle_s=20` chỉ là ví dụ. Lấy `first_text_s` từ p99 TTFT đo được của đúng model và effort đó cộng một khoảng dư; với `xhigh` hay `max` trên model lớn, TTFT có thể lên hàng phút ([Tra cứu nhanh](#số-đo-tham-khảo-theo-model)).

Test đã chạy (`anthropic` 2.6.0, stream SSE giả lập có độ trễ thật giữa các event):

1. Chữ đầu sau 0,2 s, bốn delta cách nhau 0,05 s, `usage` 5 token: TTFT trong khoảng 0,18–0,4 s, khoảng lặng lớn nhất khoảng 0,05 s, tokens/giây trong khoảng 15–30.
2. Có thinking block ẩn (chỉ có `signature_delta`) và `usage` 200 token: có TTFT nhưng `tokens_per_s` là `None`.
3. `message_start` tới ngay nhưng chữ đầu tới sau 0,5 s với `first_text_s=0.2`: ném `StreamStalled` với pha "chữ đầu tiên".
4. Im lặng 0,5 s giữa hai delta với `idle_s=0.2`: ném `StreamStalled` với pha "delta kế tiếp".
5. Thinking kéo dài 0,4 s trước chữ đầu với `idle_s=0.1`: không bị cắt, vì `idle_s` chỉ áp sau chữ đầu tiên.
6. Event `error` (`overloaded_error`) tới sau chữ đầu: SDK ném `anthropic.APIStatusError` ngay trong vòng lặp, `read_stream` không nuốt lỗi và không đặt `timing.end`, nên request không bị ghi là thành công. SDK tự chuyển event `error` thành exception, nên `read_stream` không cần nhánh riêng cho nó.

### Timeout của SDK không phải hạn tổng

`timeout` của SDK `anthropic` (mặc định 10 phút, connect 5 giây [28]) là hạn cho *mỗi lần đọc* từ socket, không phải cho cả request. Kiểm bằng một server HTTP cục bộ gửi 10 event SSE, mỗi event cách nhau 0,15 s: với `timeout=0.5`, stream kéo dài 1,53 s vẫn xong bình thường. Khi khoảng cách tăng lên 0,8 s, request lỗi sau 0,5 s. Hai hệ quả:

- Một stream mỗi giây nhỏ giọt một token sẽ không bao giờ chạm timeout của SDK. Muốn có hạn tổng, bọc bằng `asyncio.timeout(...)` hoặc deadline của riêng bạn.
- Lỗi đọc xảy ra *trong lúc lặp* stream được ném ra dưới dạng `httpx2.ReadTimeout` (thư viện HTTP mà SDK dùng), không phải `anthropic.APIError` hay `APITimeoutError`. Code chỉ bắt `anthropic.APIError` sẽ để lọt lỗi này.

## 3.2 Giảm số lượt gọi model nối tiếp

Mỗi lượt gọi model trả lại toàn bộ chi phí cố định: mạng, hàng đợi, prefill, TTFT. Ba lượt nối tiếp mỗi lượt 1,5 s TTFT là 4,5 s chưa sinh chữ nào. Giảm số lượt thường được nhiều hơn tăng tốc từng lượt.

Ví dụ của OpenAI về một bot hỗ trợ khách hàng đi qua đúng các bước này [9]:

1. **Gộp hai lượt nối tiếp vào một.** Viết lại câu hỏi và quyết định có cần tìm kiếm không là hai prompt riêng; gộp thành một prompt trả JSON hai trường.
2. **Dùng model nhỏ cho bước đơn giản.** Hai việc trên là việc phân loại rõ ràng, chuyển sang model nhỏ hơn.
3. **Tách bước dài thành phần nhỏ chạy model nhỏ và phần chính chạy model lớn,** khi đo thấy lợi hơn. Đây là xung đột giữa "ít request hơn" và "xử lý token nhanh hơn", và chỉ đo trên ví dụ thật mới quyết được.
4. **Chạy song song** bước suy luận (không cần tài liệu) với bước tìm kiếm.
5. **Rút ngắn tên trường JSON** trong output: bớt 19 output token.

Thêm vài cách hay bị bỏ qua:

- **Không dùng LLM khi không cần** [9]. Câu xác nhận, câu từ chối, yêu cầu nhập lại thông tin có thể viết sẵn. Số liệu tổng hợp hợp với một thành phần UI hơn là một đoạn văn do model viết.
- **Cho tool trả đủ dữ liệu trong một lần** để model không phải gọi lại. Tool trả 5 kết quả đầu kèm tổng số, thay vì bắt model gọi trang 2 ([Tool use §3.3](../tool-use/#33-tool-result-là-prompt)).
- **Đưa bước kiểm tra ra khỏi đường đi của người dùng** khi rủi ro cho phép: stream câu trả lời, kiểm tra song song, và thu hồi hoặc sửa nếu bước kiểm tra phát hiện vấn đề. Đánh đổi ở đây là người dùng có thể đã đọc nội dung sai; với nội dung nhạy cảm, giữ bước kiểm tra trên đường đi hoặc kiểm theo từng đoạn ([§2.1](#21-streaming)).
- **Agent loop:** mỗi vòng tool là một lượt gọi model. OpenAI WebSocket mode giữ kết nối và chỉ gửi phần input mới mỗi lượt, đo được tới khoảng 40% nhanh hơn từ đầu tới cuối với rollout từ 20 tool call trở lên [12]. Request `response.create` với `generate: false` chuẩn bị trạng thái trước để lượt sau bắt đầu nhanh hơn [12].

## 3.3 Đặt region gần provider và người dùng

- **Đặt backend gần endpoint của provider.** Mỗi lượt gọi model trả một vòng mạng giữa backend và provider; pipeline nhiều lượt trả nhiều lần. Đặt backend, vector store và các service nó gọi trong cùng region với nhau và gần provider.
- **Số đo công khai phụ thuộc vị trí đo.** Artificial Analysis đo từ `us-central1` và ghi rõ TTFT của họ gồm độ trễ mạng nên nhạy với vị trí server, có thể lợi hoặc thiệt cho provider tuỳ nơi đặt server [19]. Người dùng ở Việt Nam gọi qua backend ở Singapore sẽ thấy số khác.
- **Data residency là quyết định về tuân thủ và giá, không phải về tốc độ.** Claude: `inference_geo: "us"` nhân giá 1,1 lần trên Claude 4.6 trở lên [6]. OpenAI: endpoint xử lý theo vùng cộng 10% với model ra mắt từ 05/03/2026 [15]; Fast mode hỗ trợ data residency EU cho GPT-6.1 Sol, GPT-6.1 Terra và GPT-6 Luna, không cho GPT-6 Astra và GPT-6 Sol [10]. Không provider nào trong số đó hứa độ trễ thấp hơn khi chọn vùng.
- **Với self-host,** đặt GPU cùng region với ứng dụng, và tính cả thời gian kéo model khi scale lên một máy mới ([Self-hosting LLM](../self-hosting-llm/)).

Với model reasoning, vài chục mili-giây mạng là nhỏ so với hàng giây hay hàng chục giây reasoning. Region đáng tối ưu khi model nhanh, câu trả lời ngắn và có nhiều lượt nối tiếp.

## 3.4 Timeout, hedging và retry

### Timeout theo pha

Một timeout tổng cho cả request không đủ: request chết ngay từ đầu phải chờ hết hạn tổng mới bị cắt, còn request đang stream tốt nhưng dài thì bị cắt oan. Tách theo pha:

| Pha | Hạn | Lấy từ đâu |
|---|---|---|
| Kết nối | Vài giây | SDK `anthropic` mặc định connect 5 giây [28] |
| Tới chữ đầu tiên | p99 TTFT của model và effort đó, cộng dư | Đo ([§3.1](#31-đo-độ-trễ-từng-tầng-trong-pipeline)) |
| Giữa hai delta | Vài lần khoảng lặng lớn nhất quan sát được | Đo `max_gap` |
| Tổng | Thời gian người dùng hoặc bước sau chấp nhận chờ | Yêu cầu sản phẩm |

Hai lưu ý từ tài liệu provider. Flex của Gemini có thể nằm trong hàng đợi nhiều phút, nên Google khuyên timeout client từ 10 phút trở lên [17]. Proxy và load balancer có idle timeout riêng (AWS ALB mặc định 60 giây) có thể cắt stream trong lúc model đang nghĩ ([Streaming §3.2](../streaming/#32-stream-qua-proxy-load-balancer-và-cdn)).

### Hedged request

Ý tưởng từ "The Tail at Scale": gửi request, chờ tới khoảng p95 độ trễ của loại request đó, nếu chưa xong thì gửi bản thứ hai và lấy kết quả nào đến trước, huỷ bản còn lại [20]. Chờ tới p95 giới hạn tải thêm ở khoảng 5%, vì chỉ khoảng 5% request chậm hơn ngưỡng đó [20]. Trong benchmark BigTable của Google, hedge sau 10 ms đưa p99,9 khi đọc 1.000 khoá từ 1.800 ms xuống 74 ms, chỉ thêm 2% request [20].

Áp dụng cho LLM cần cẩn thận hơn:

- **Chỉ có ích khi độ trễ cao là ngẫu nhiên.** Hedge thắng vì "the source of latency is often not inherent in the particular request but rather due to other forms of interference" [20]. Request chậm vì output dài hay vì reasoning khó sẽ chậm y hệt ở bản thứ hai. Hedge hợp với TTFT bị kéo bởi hàng đợi, không hợp với decode dài.
- **Hedge theo chữ đầu tiên.** Khi đã stream chữ cho người dùng thì không đổi nguồn được nữa. Điểm quyết định tự nhiên là: bản chính chưa có chữ đầu tiên sau ngưỡng thì gửi bản thứ hai.
- **Bản bị huỷ vẫn có thể bị tính tiền** ([Streaming §3.7](../streaming/#37-người-dùng-huỷ-dừng-request-phía-provider-để-không-tốn-token)), và nó tiêu rate limit đúng lúc hệ thống có thể đang quá tải. Giới hạn tỉ lệ hedge.
- **Hai bản cho hai câu trả lời khác nhau.** Không vấn đề gì nếu bạn chỉ dùng một bản, nhưng đừng ghi log hay lưu kết quả của bản thua như thể nó là câu trả lời.

Code: hàm `hedged` chạy một coroutine, chờ `hedge_after_s`, nếu chưa xong và còn ngân sách thì chạy thêm một bản, lấy bản thành công đến trước, huỷ bản còn lại. `HedgeBudget` giới hạn số hedge ở khoảng `ratio` tổng số request.

```python
class HedgeBudget:
    """Giới hạn số request hedge ở khoảng `ratio` tổng số request (token bucket)."""

    def __init__(self, ratio: float, burst: float):
        self.ratio, self.burst = ratio, burst
        self.tokens = burst

    def on_request(self) -> None:
        self.tokens = min(self.burst, self.tokens + self.ratio)

    def take(self) -> bool:
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


async def hedged(start, *, hedge_after_s: float, budget: HedgeBudget, discard=None):
    """Chạy `start()`; nếu sau `hedge_after_s` chưa xong và budget còn, chạy thêm một bản.
    Trả kết quả thành công đến trước, huỷ bản còn lại. `discard` nhận kết quả của bản thua
    nếu nó đã xong cùng lúc (ví dụ để đóng stream đang mở)."""
    budget.on_request()
    tasks = [asyncio.ensure_future(start())]
    try:
        done, _ = await asyncio.wait(tasks, timeout=hedge_after_s)
        if not done and budget.take():
            tasks.append(asyncio.ensure_future(start()))
        errors = []
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.exception() is None:
                    winner = t
                    break
                errors.append(t.exception())
            else:
                continue
            for other in tasks:
                if other is not winner and other.done() and other.exception() is None and discard:
                    await discard(other.result())
            return winner.result()
        raise errors[-1]
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
```

Hedge theo chữ đầu tiên với stream Claude, dùng `read_stream` ở [§3.1](#31-đo-độ-trễ-từng-tầng-trong-pipeline):

```python
budget = HedgeBudget(ratio=0.05, burst=1)      # sống lâu hơn một request, như breaker

async def open_until_first_text():
    timing = Timing(start=time.monotonic())
    stream = await client.messages.create(**request, stream=True)
    gen = read_stream(stream, timing, first_text_s=30, idle_s=20)
    try:
        first = await anext(gen)
    except BaseException:                      # gồm cả CancelledError khi bản này thua
        await stream.close()
        raise
    return first, gen, stream

first, gen, stream = await hedged(open_until_first_text, hedge_after_s=p95_ttft,
                                  budget=budget, discard=lambda r: r[2].close())
async with stream:
    await send_to_user(first)
    async for text in gen:
        await send_to_user(text)
```

`p95_ttft` lấy từ số đo của chính loại request đó, cập nhật định kỳ. Tỉ lệ 5% là ví dụ theo lập luận của Dean và Barroso; chọn theo ngân sách token và rate limit của bạn.

Test đã chạy:

1. Bản chính xong trước ngưỡng: không hedge.
2. Bản chính chậm: hedge sau 0,1 s, bản hedge thắng, bản chính bị huỷ, tổng thời gian dưới 0,5 s.
3. Hết ngân sách: không hedge, chờ bản chính.
4. Ngân sách tỉ lệ 0,25, burst 1, 100 request cần hedge: đúng 25 hedge.
5. Bản hedge lỗi: bản chính vẫn được trả về.
6. Cả hai lỗi: ném lỗi.
7. Hai bản xong trong cùng một lượt của event loop: bản thua được đưa vào `discard`.
8. Với stream thật của SDK: bản chính chưa có chữ đầu sau 0,1 s, bản hedge có ngay; kết quả là chữ của bản hedge, hai request đã được gửi, tổng thời gian dưới 0,6 s.

### Retry

Retry là nguồn độ trễ đuôi lớn: một lần backoff có thể dài hơn cả request. Cách xếp retry, backoff có jitter, tôn trọng `retry-after`, fallback sang model khác và circuit breaker ở [Model routing §1.3](../model-routing/#13-retry-backoff-và-jitter) và [§3.1](../model-routing/#31-thiết-kế-chuỗi-fallback). Ba điểm riêng cho độ trễ:

- **Retry trong deadline của request, không cộng thêm.** Mỗi lần thử nhận phần thời gian còn lại; hết deadline thì dừng, không thử tiếp.
- **Retry fast mode sang tốc độ thường là cache miss** [2]: có thể chậm hơn chờ `retry-after` ngắn của fast mode, vốn "typically short" vì token được bổ sung liên tục [2].
- **Không retry giữa stream đã gửi chữ cho người dùng** trừ khi bạn đệm output hoặc có cách nối tiếp ([Streaming §3.6](../streaming/#36-lỗi-giữa-chừng-đứt-kết-nối-lỗi-sau-khi-đã-gửi-một-phần-retry-và-resume)).

## 3.5 UX khi phải chờ: trạng thái, kết quả từng phần

Khi không thể nhanh hơn, làm cho thời gian chờ có nội dung [9]:

- **Phản hồi ngay khi nhận request.** Hiện trạng thái trước khi có token đầu tiên; đừng để màn hình trống trong lúc request còn đang xếp hàng.
- **Hiện bước đang làm.** "Đang tìm trong tài liệu", "Đang gọi API đơn hàng". Với agent, hiện từng tool khi nó chạy. OpenAI ghi: "The more real progress you can show, the better" [9].
- **Với reasoning,** hiện "đang suy nghĩ", hoặc stream bản tóm tắt thinking nếu provider có ([Streaming §3.5](../streaming/#35-stream-kèm-reasoningthinking)).
- **Trả phần quan trọng trước.** Yêu cầu model viết kết luận trước, chi tiết sau, để người dùng có thể dừng đọc sớm. Với output có cấu trúc, đặt trường người dùng cần đầu tiên lên đầu schema.
- **Cho đường thoát.** Nút dừng phải huỷ thật request ở provider ([Streaming §3.7](../streaming/#37-người-dùng-huỷ-dừng-request-phía-provider-để-không-tốn-token)). Với tác vụ dài, chạy nền và báo khi xong thay vì giữ người dùng ở màn hình.
- **Đừng giả tiến độ.** Thanh tiến độ không gắn với bước thật làm người dùng mất lòng tin khi nó đứng ở 90%.

## 3.6 Failure modes thường gặp

| # | Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|---|
| 1 | Mỗi câu trả lời chậm dù từng bước đều "bình thường" | Nhiều lượt gọi model, retrieval, kiểm tra chạy nối tiếp | Trace từng span; gộp bước, chạy song song, bỏ bước không cần LLM ([§3.2](#32-giảm-số-lượt-gọi-model-nối-tiếp)) |
| 2 | Màn hình trống lâu dù tổng thời gian chấp nhận được | Không stream, hoặc proxy gom stream | Stream tới tận client; kiểm tra buffering của proxy |
| 3 | TTFT tăng mạnh sau khi nâng model | Effort mặc định khác; reasoning dài hơn | Đặt effort tường minh; đo lại TTFT theo effort [4] |
| 4 | Bật fast mode mà chữ đầu không đến sớm hơn | Fast mode tăng tokens/giây, không giảm TTFT [2] | Đo tokens/giây và tổng thời gian; giảm TTFT bằng effort, cache |
| 5 | Request "fast" nhưng không nhanh | Chạy tốc độ thường: model không hỗ trợ (Opus 4.6), bị hạ tier do ramp rate, bị hạ Priority xuống Standard | Đọc `usage.speed`, `service_tier`, `x-gemini-service-tier` ở mỗi response |
| 6 | TTFT xấu ngay sau khi fallback từ fast sang thường | Đổi speed làm mất cache [2] | Chấp nhận, hoặc chờ `retry-after` của fast mode thay vì fallback |
| 7 | Request đầu tiên của mỗi phiên chậm | Cache miss | Pre-warm bằng `max_tokens: 0` với breakpoint tường minh [3] |
| 8 | Pre-warm không làm request sau nhanh hơn | Breakpoint đặt trên message giữ chỗ, hoặc khác cấu hình thinking/effort [3] | Đặt `cache_control` trên block dùng chung; dùng cùng cấu hình |
| 9 | Stream bị cắt khi model đang nghĩ | Idle timeout của proxy hoặc của chính ứng dụng áp trước chữ đầu | Heartbeat; tách hạn chờ chữ đầu khỏi hạn chờ giữa hai delta |
| 10 | Stream treo rất lâu mà không lỗi | `timeout` của SDK là hạn mỗi lần đọc, không phải hạn tổng | Thêm deadline tổng bằng `asyncio.timeout` |
| 11 | Timeout giữa stream lọt qua `except anthropic.APIError` | Lỗi trong lúc lặp stream là `httpx2.ReadTimeout` | Bắt cả lỗi của thư viện HTTP, hoặc dùng `StreamStalled` của bạn |
| 12 | Tokens/giây đo được cao bất thường với model reasoning | `output_tokens` gồm thinking token sinh trước chữ đầu | Không tính tokens/giây từ `usage` khi có thinking ([§1.2](#12-ttft-tokensgiây-và-tổng-thời-gian)) |
| 13 | Hedge không cải thiện p99 mà tăng chi phí | Đuôi do output dài hoặc reasoning khó, không do ngẫu nhiên | Chỉ hedge theo chữ đầu tiên; xem phân bố độ dài output ở đuôi |
| 14 | Hedge làm tăng `429` | Không giới hạn tỉ lệ hedge | `HedgeBudget`; ngưỡng hedge theo p95 |
| 15 | Request Flex thất bại hàng loạt giờ cao điểm | Flex là capacity sheddable, không tự nâng tier [17] | Không dùng Flex cho đường người dùng chờ |
| 16 | Fast mode của OpenAI bị hạ tier sau khi chuyển traffic | Tăng traffic nhanh hơn ramp rate [10] | Tăng không quá 50% mỗi 15 phút khi đã ở mức 1 triệu TPM |
| 17 | Semantic cache làm p50 tệ hơn | Tỉ lệ hit thấp; mọi miss trả thêm thời gian embed và tra | Đo tỉ lệ hit; bỏ nếu không đủ cao |
| 18 | p99 của model tự host tăng khi tăng batch | Prefill dài chen vào decode | Chunked prefill; chỉnh `max_num_batched_tokens` theo SLO [25] |
| 19 | Speculative decoding không nhanh hơn ở tải cao | Lợi ích lớn nhất ở tải thấp, decode bị giới hạn bởi bộ nhớ [26] | Đo trên tải thật; cân nhắc n-gram hoặc tắt |
| 20 | Output khác nhau giữa các lần chạy sau khi bật speculative decoding | Logprob phụ thuộc batch size; một số phương pháp không giữ phân phối [26] | Không dùng cho eval cần tái lập chặt; chọn phương pháp lossless |

## 3.7 Checklist

**Đo**

- [ ] Trace có span cho từng lượt gọi model, retrieval, tool và bước kiểm tra.
- [ ] Mỗi lượt gọi model ghi thời điểm chữ trả lời đầu tiên, tổng thời gian, `usage`, model, effort, tier thực tế.
- [ ] Báo cáo p50, p95, p99 của TTFT và tổng thời gian, tách theo model, effort, tier.
- [ ] Biết phân bố độ dài output, nhất là ở các request chậm nhất.

**Đường người dùng chờ**

- [ ] Stream tới tận client; proxy không gom.
- [ ] Hiện trạng thái ngay khi nhận request; hiện bước đang làm.
- [ ] Bước không cần LLM thì không dùng LLM; bước độc lập chạy song song; bước nối tiếp gộp lại khi được.
- [ ] Effort đặt tường minh, chọn bằng eval cả chất lượng lẫn độ trễ.
- [ ] Câu trả lời ngắn nhất mà người dùng cần.

**Cache và tier**

- [ ] Prefix ổn định; cache hit rate được theo dõi; pre-warm cho phiên mới nếu TTFT lượt đầu quan trọng.
- [ ] Tier nhanh chỉ cho đường người dùng chờ; tier thực tế được kiểm ở mỗi response.
- [ ] Biết hệ quả cache của fallback giữa các tier.

**Timeout và retry**

- [ ] Hạn chờ riêng cho kết nối, chữ đầu tiên, giữa hai delta và tổng.
- [ ] Hạn chờ chữ đầu tiên đủ cho reasoning ở effort đang dùng.
- [ ] Bắt cả lỗi timeout của thư viện HTTP trong lúc lặp stream.
- [ ] Retry nằm trong deadline của request; không retry sau khi đã gửi chữ cho người dùng.
- [ ] Hedge (nếu dùng) theo chữ đầu tiên, ngưỡng theo p95 đo được, có giới hạn tỉ lệ.

---

## Nguồn tham khảo

Truy cập ngày 06–07/11/2026.

| # | Nguồn | Nội dung dùng trong bài |
|---|---|---|
| [1] | [Claude: Reducing latency](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-latency) | Ba cách giảm độ trễ; ví dụ Haiku 5.5 ở effort `low` |
| [2] | [Claude: Fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode) | `speed: "fast"`, 2,5 lần OTPS, không giảm TTFT, giá, rate limit, `usage.speed`, fallback và cache miss |
| [3] | [Claude: Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) | Độ trễ của cache, pre-warm bằng `max_tokens: 0`, bảng invalidation |
| [4] | [Claude: Effort](https://platform.claude.com/docs/en/build-with-claude/effort) | Mặc định và gợi ý effort theo model, đổi effort giữa hội thoại |
| [5] | [Claude: Models overview](https://platform.claude.com/docs/en/about-claude/models/overview) | Độ trễ tương đối giữa các model |
| [6] | [Claude: Service tiers](https://platform.claude.com/docs/en/api/service-tiers) | Priority Tier đã ngừng bán; `inference_geo` |
| [7] | [Claude API: Errors](https://platform.claude.com/docs/en/api/errors) | Request dài, streaming, kiểm tra 10 phút của SDK |
| [8] | [Claude: Streaming messages](https://platform.claude.com/docs/en/build-with-claude/streaming) | Event `ping`, thinking ẩn trong stream |
| [9] | [OpenAI: Latency optimization](https://developers.openai.com/api/docs/guides/latency-optimization) | Bảy nguyên tắc, heuristic output/input token, speculative execution, ví dụ bot hỗ trợ khách hàng |
| [10] | [OpenAI: Fast mode](https://developers.openai.com/api/docs/guides/fast-mode) | Đổi tên từ Priority processing, `service_tier`, ramp rate, SLA, data residency |
| [11] | [OpenAI: Ultrafast mode](https://developers.openai.com/api/docs/guides/ultrafast-mode) | Tier nhanh nhất, model hỗ trợ, khuyến nghị WebSocket |
| [12] | [OpenAI: WebSocket mode](https://developers.openai.com/api/docs/guides/websocket-mode) | Tới khoảng 40% nhanh hơn với 20+ tool call, `generate: false`, `stream_id`, giới hạn kết nối |
| [13] | [OpenAI: Predicted Outputs](https://developers.openai.com/api/docs/guides/predicted-outputs) | Tham số `prediction`, model hỗ trợ, giới hạn |
| [14] | [OpenAI: Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) | Cache giảm thời gian xử lý input, không bảo đảm độ trễ thấp hơn |
| [15] | [OpenAI: Pricing](https://developers.openai.com/api/docs/pricing) | Giá Fast mode và Ultrafast, phụ phí xử lý theo vùng |
| [16] | [Gemini API: Priority inference](https://ai.google.dev/gemini-api/docs/priority-inference) | Giá, hạ tier khi quá tải, rate limit, header `x-gemini-service-tier` |
| [17] | [Gemini API: Flex inference](https://ai.google.dev/gemini-api/docs/flex-inference) | Bảng bốn tier, capacity sheddable, timeout 10 phút |
| [18] | [Artificial Analysis: LLM Leaderboard](https://artificialanalysis.ai/leaderboards/models) | Thời gian tới chữ trả lời đầu tiên và thời gian mỗi tác vụ, lát cắt 07/11/2026 |
| [19] | [Artificial Analysis: Performance benchmarking methodology](https://artificialanalysis.ai/methodology/performance-benchmarking) | Định nghĩa TTFT, time to first answer token, output speed; tần suất và vị trí đo |
| [20] | [Dean, Barroso, "The Tail at Scale", CACM 56(2), 2013](https://cacm.acm.org/research/the-tail-at-scale/) | Khuếch đại độ trễ đuôi, hedged và tied request, số liệu BigTable |
| [21] | [Leviathan et al., Fast Inference from Transformers via Speculative Decoding (arXiv 2211.17192)](https://arxiv.org/abs/2211.17192) | Speculative decoding, 2–3 lần trên T5-XXL |
| [22] | [Li et al., EAGLE (arXiv 2401.15077)](https://arxiv.org/abs/2401.15077) | 2,7–3,5 lần trên LLaMA2-Chat 70B |
| [23] | [Agrawal et al., Sarathi-Serve (arXiv 2403.02310)](https://arxiv.org/abs/2403.02310) | Chunked prefill, stall-free scheduling |
| [24] | [Kwon et al., PagedAttention (arXiv 2309.06180)](https://arxiv.org/abs/2309.06180) | Quản lý KV cache, throughput 2–4 lần |
| [25] | [vLLM: Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) | Chunked prefill mặc định, `max_num_batched_tokens` |
| [26] | [vLLM: Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/) | Phương pháp, khi nào có lợi, tính lossless |
| [27] | [Hugging Face: LLM Inference at scale with TGI](https://huggingface.co/blog/martinigoyanes/llm-inference-at-scale-with-tgi) | Prefill compute-bound, decode memory-bound |
| [28] | [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) | Bản 2.6.0: `DEFAULT_TIMEOUT` 10 phút, connect 5 giây; hành vi timeout kiểm bằng server cục bộ |
