# Routing và fallback giữa các model

Tám giờ tối, giờ cao điểm. Provider chính bắt đầu trả `529 overloaded_error`. SDK tự retry hai lần, mỗi lần chờ thêm một chút, rồi ném exception. Chatbot hiển thị "Đã có lỗi xảy ra" cho mọi người dùng. Một kỹ sư thêm vội một chuỗi fallback sang model khác. Request đầu tiên sang model mới nhận ngay `400`, vì request body gửi `tool_choice: {"type": "any"}` mà model đó không hỗ trợ. Sửa xong lỗi đó thì chi phí giờ tiếp theo tăng gấp đôi, vì prompt cache của model cũ không dùng được cho model mới và vòng retry của ứng dụng nhân lên với vòng retry trong SDK.

Bài này bàn hai bài toán dùng chung một cơ chế:

- **Fallback**: model hoặc provider chính lỗi, request phải được phục vụ ở chỗ khác. Mục tiêu là uptime.
- **Routing**: chọn model cho từng request, ví dụ câu dễ cho model rẻ, câu khó cho model mạnh. Mục tiêu là chi phí và chất lượng.

Cả hai đều đổi model giữa chừng, và mọi rủi ro của việc đổi model (request không hợp lệ ở model mới, mất cache, mất reasoning, output khác giọng) áp dụng cho cả hai. Ba phần:

- **Nền tảng — Phần 1:** vì sao một model là một điểm lỗi, rate limit được tính thế nào, retry đúng cách, các nghiên cứu về router, và những gì thay đổi khi đổi model.
- **Công cụ — Phần 2:** LLM gateway (LiteLLM, OpenRouter, Portkey, Cloudflare), fallback có sẵn của Claude, cloud provider (Bedrock, Azure, Vertex), router bằng classifier, priority tier và batch.
- **Thiết kế và vận hành — Phần 3:** chuỗi fallback kèm code đã chạy, giữ request tương thích giữa các model, circuit breaker, quota theo tenant, eval router, failure modes và checklist.

> Bản tháng 11/2026. Code trong bài đã chạy với `anthropic` 2.6.0 (Python, API version `2026-10-21`) qua mock transport, không gọi API thật. Số rate limit và tên model thay đổi thường xuyên; đối chiếu trang của provider trước khi dùng. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

Tài liệu liên quan: [Decision models §3.11](../decision-models/#311-routing-giữa-các-llm) (router dùng decision model để chọn model theo độ khó và miền), [Decision models §3.7](../decision-models/#37-cascade-trích-xuất-dữ-liệu) (cascade model nhỏ trước, kiểm, rồi leo thang), [Tool use §3.11](../tool-use/#311-failure-modes-thường-gặp) (tool call sai sau khi đổi model), [Multi-turn](../multi-turn-conversation/) (cache và thinking block qua các lượt), [Streaming](../streaming/) (lỗi giữa stream), [LLM Observability](../llm-observability/), [Evaluation](../evaluation/).

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Vì sao một model, một provider là một điểm lỗi](#11-vì-sao-một-model-một-provider-là-một-điểm-lỗi)
  - [1.2 Rate limit: RPM, TPM, và cách provider tính](#12-rate-limit-rpm-tpm-và-cách-provider-tính)
  - [1.3 Retry, backoff và jitter](#13-retry-backoff-và-jitter)
  - [1.4 Routing theo độ khó, theo loại tác vụ, theo chi phí](#14-routing-theo-độ-khó-theo-loại-tác-vụ-theo-chi-phí)
  - [1.5 Model khác nhau trả lời khác nhau: hệ quả khi fallback](#15-model-khác-nhau-trả-lời-khác-nhau-hệ-quả-khi-fallback)
- [Phần 2 — Công cụ](#phần-2--công-cụ)
  - [2.1 LLM gateway: LiteLLM, Portkey, OpenRouter, Cloudflare AI Gateway](#21-llm-gateway-litellm-portkey-openrouter-cloudflare-ai-gateway)
  - [2.2 Cloud provider: Bedrock, Vertex AI, Azure OpenAI và cross-region inference](#22-cloud-provider-bedrock-vertex-ai-azure-openai-và-cross-region-inference)
  - [2.3 Router dùng classifier hoặc model nhỏ](#23-router-dùng-classifier-hoặc-model-nhỏ)
  - [2.4 Priority tier, provisioned throughput và batch](#24-priority-tier-provisioned-throughput-và-batch)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Thiết kế chuỗi fallback](#31-thiết-kế-chuỗi-fallback)
  - [3.2 Giữ prompt, tool và schema tương thích giữa các model](#32-giữ-prompt-tool-và-schema-tương-thích-giữa-các-model)
  - [3.3 Circuit breaker và health check](#33-circuit-breaker-và-health-check)
  - [3.4 Quản lý quota nội bộ theo tenant](#34-quản-lý-quota-nội-bộ-theo-tenant)
  - [3.5 Đo và eval khi route sang model khác](#35-đo-và-eval-khi-route-sang-model-khác)
  - [3.6 Failure modes thường gặp](#36-failure-modes-thường-gặp)
  - [3.7 Checklist](#37-checklist)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

### Những quyết định cần chốt trước khi viết code

1. **Ba loại lỗi, ba cách xử lý.** Lỗi tạm thời của hạ tầng (quá tải, 5xx, timeout, mất kết nối) thì retry có backoff, hết lượt thì fallback. Rate limit thì chờ đúng thời gian server yêu cầu; nếu thời gian đó quá dài, chuyển sang model nằm ở bucket rate limit khác. Lỗi của request hoặc tài khoản (400, 401, 402, 403, hết hạn mức chi tiêu) thì không retry, không fallback, báo lỗi ngay.
2. **Không phân loại bằng status code một mình.** Claude báo hết spend cap của tháng bằng `429`, cùng `rate_limit_error` như rate limit thường; chỉ `error.details.error_code = "enforced_spend_limit_reached"` và việc không có header `retry-after` cho biết retry vô ích [2]. OpenAI cũng dùng `429` cho cả rate limit lẫn hết credit [10].
3. **Refusal không phải exception.** Claude trả HTTP 200 với `stop_reason: "refusal"` [3]. Code chỉ bắt exception sẽ coi refusal là thành công.
4. **Một nơi retry duy nhất.** SDK, gateway và ứng dụng đều có thể retry. Ba tầng mỗi tầng ba lần thử là 27 request cho một lần bấm. Chọn một tầng làm chủ, tắt retry ở các tầng còn lại (`max_retries=0` với SDK của Claude và OpenAI) [1][11].
5. **Trần cho cả chuỗi, không chỉ cho từng model:** tổng số lần thử và tổng thời gian.
6. **Chuỗi fallback chọn theo tương thích, không chỉ theo giá.** Cùng một request body chạy được trên model A có thể bị `400` trên model B (thinking, `tool_choice`, computer use) [1]. Fallback cũng làm mất prompt cache, vì cache Claude tách theo model [4].
7. **Log model thực sự trả lời.** Trường `model` trong response (và `usage.iterations` khi dùng fallback phía server của Claude) là tín hiệu chính để biết router và fallback đang làm gì [3][13][18].

### Mã lỗi theo provider

| Tình huống | Claude [1][2] | OpenAI [10][11] | Gemini [22] | Xử lý |
|---|---|---|---|---|
| Quá tải tạm thời | `529 overloaded_error`; `500 api_error` | `503 service_unavailable_error`, code `server_is_overloaded` | `503 UNAVAILABLE` | Retry có backoff, rồi fallback |
| Rate limit | `429 rate_limit_error`, thường có `retry-after` | `429`, có thể kèm `Retry-After`; code `slow_down` khi traffic tăng quá nhanh | `429 RESOURCE_EXHAUSTED` | Chờ `retry-after`; quá lâu thì sang bucket khác |
| Hết hạn mức chi tiêu | `429` với `error_code: enforced_spend_limit_reached`, không có `retry-after`; hạn mức tự đặt trả `400` | `429` với code `credit_balance_exhausted`, `organization_spend_limit_exceeded`, `project_spend_limit_exceeded` | `402` (hết credit trả trước) | Không retry, không fallback trong cùng tổ chức; báo động |
| Timeout | `504 timeout_error`, đôi khi `500` kèm `timeout_error` | Timeout phía client | `408`, `5xx` | Retry có trần; request dài dùng streaming |
| Request sai | `400 invalid_request_error`, `413 request_too_large` | `400` | `400` | Không retry. Lỗi chỉ xảy ra ở một model là bug cấu hình chuỗi |
| Xác thực, quyền, thanh toán | `401`, `403`, `402` | `401`, `403` | `403`, `402` | Không retry |
| Model không tồn tại | `404 not_found_error` | `404` | `404` | Sang model kế, kèm alert |
| Model từ chối | HTTP 200, `stop_reason: "refusal"` | | | Model khác; với Claude dùng `fallbacks` phía server ([§2.2](#22-cloud-provider-bedrock-vertex-ai-azure-openai-và-cross-region-inference)) |

Ô trống nghĩa là bài không đối chiếu cơ chế tương ứng của provider đó.

### Retry mặc định của SDK

| SDK | Mặc định | Ghi chú |
|---|---|---|
| `anthropic` (Python 2.6.0) | 2 lần retry; backoff 0,5 s nhân đôi, trần 8 s, jitter tới 25%; retry 408, 409, 429, 5xx và lỗi kết nối | Có `retry-after` thì chờ đúng giá trị đó, không có trần riêng (đo: `retry-after: 3` thì chờ 3,02 s). Server gửi `x-should-retry` thì theo header đó [1][29] |
| OpenAI SDK chính thức | Tự retry 429 và 503 đủ điều kiện | Cách xử lý `Retry-After` dài khác nhau theo phiên bản; tài liệu khuyên kiểm hành vi của bản đang cài [11] |
| Gemini Python SDK | Tối đa 4 lần, chờ ban đầu khoảng 1 s, tối đa 60 s | Retry timeout, lỗi mạng, 429, 5xx [22] |

### Cấu hình khởi điểm cho vòng retry của bạn

Đây là điểm xuất phát để đo, không phải số tối ưu:

- SDK: `max_retries=0`.
- Mỗi model thử tối đa 2 lần; cả chuỗi tối đa 4–6 lần.
- Deadline cho cả chuỗi bằng thời gian người dùng chấp nhận chờ. Mỗi lần thử nhận phần thời gian còn lại làm timeout.
- Backoff: full jitter, `random.uniform(0, min(8, 0.5 * 2**attempt))` giây.
- Server yêu cầu chờ lâu hơn ngưỡng bạn chịu được (ví dụ 10 s) hoặc lâu hơn phần deadline còn lại: không chờ, sang model ở bucket khác. Không retry sớm hơn thời gian server yêu cầu [11].
- Circuit breaker theo (provider, model): mở sau vài lỗi liên tiếp, thử lại sau một khoảng cooldown. Không provider nào công bố ngưỡng hay cooldown nên dùng; code ở [§3.1](#31-thiết-kế-chuỗi-fallback) để chúng là tham số bắt buộc. Chọn từ trace: ngưỡng đủ cao để vài lỗi lẻ không mở breaker, cooldown cỡ thời gian một đợt quá tải thường kéo dài.

---

# Phần 1 — Nền tảng

## 1.1 Vì sao một model, một provider là một điểm lỗi

Một ứng dụng gọi đúng một model ở một provider phụ thuộc vào năm thứ không nằm trong tay bạn:

- **Capacity của provider.** Claude trả `529 overloaded_error` khi "API is temporarily overloaded" [1]. OpenAI trả `503` với code `server_is_overloaded` khi model "does not have enough capacity to process your request at the moment" [10]. Lỗi này không phụ thuộc request của bạn: request hợp lệ vẫn có thể bị từ chối.
- **Rate limit và hạn mức của chính bạn.** Vượt RPM hay TPM thì nhận `429`. Hết hạn mức chi tiêu tháng thì mọi request đều lỗi cho đến khi qua tháng hoặc nâng hạn mức [2].
- **Chính sách của model.** Fable, Opus 5.x, Sonnet 5.5 và Haiku 5.5 có classifier an toàn có thể từ chối request, kể cả request vô hại; tài liệu Claude ghi rõ công việc bảo mật hợp lệ cũng có thể kích hoạt nhóm `cyber` [3]. Từ chối là HTTP 200, không phải lỗi.
- **Vòng đời model.** Model bị deprecate rồi retire. Code còn trỏ vào model ID cũ sẽ nhận `404`.
- **Vùng địa lý.** Một region có sự cố thì mọi request vào region đó lỗi. Cloud provider có cơ chế phân phối request qua nhiều region ([§2.2](#22-cloud-provider-bedrock-vertex-ai-azure-openai-và-cross-region-inference)).

Retry chỉ giải quyết được lỗi kéo dài vài giây. Sự cố capacity có thể kéo dài nhiều phút đến vài giờ. Muốn sản phẩm vẫn trả lời trong lúc đó, bạn cần một chỗ khác để gửi request: model khác cùng provider, cùng model ở provider khác (Claude có trên Claude API, Bedrock, Vertex AI, Foundry), hoặc model của provider khác. Mỗi lựa chọn có cái giá riêng, và phần lớn bài này nói về cái giá đó.

## 1.2 Rate limit: RPM, TPM, và cách provider tính

### Claude

Messages API giới hạn theo ba chiều cho mỗi nhóm model: request mỗi phút (RPM), input token mỗi phút (ITPM), output token mỗi phút (OTPM). Vượt chiều nào cũng nhận `429` nói rõ chiều bị vượt [2]. Vài chi tiết ảnh hưởng trực tiếp tới thiết kế fallback:

- **Token bucket.** Dung lượng được nạp lại liên tục, không reset theo phút tròn [2]. Một đợt burst ngắn có thể vượt giới hạn dù trung bình theo phút vẫn thấp.
- **Cache read không tính vào ITPM.** `input_tokens` và `cache_creation_input_tokens` tính vào ITPM, `cache_read_input_tokens` thì không [2]. Cache hit rate cao làm tăng thông lượng thực. Fallback sang model khác làm mất cache, nên cùng một lượng traffic sẽ tốn ITPM hơn ở model đích.
- **`max_tokens` không ảnh hưởng OTPM.** OTPM đếm token thực sinh ra [2]. OpenAI tính khác (xem dưới).
- **Bucket dùng chung trong họ.** Giới hạn Opus 4.x là tổng traffic của Opus 4.8, 4.7, 4.6 và 4.5; Opus 5.5 và Opus 5 mỗi model có bucket riêng. Fable 5.1 và Fable 5 dùng chung một bucket. Sonnet 4.6 và 4.5 dùng chung; Sonnet 5.5 và Sonnet 5 mỗi model riêng [2]. Fallback từ Opus 4.8 sang Opus 4.7 vì `429` không giải quyết được gì: hai model rút từ cùng một bucket.
- **Fast mode có bucket riêng** cho Opus 5.5, Opus 5 và Opus 4.8; vượt thì nhận `429` có `retry-after` [2].
- **Spend cap theo bậc.** Bậc Start, Build, Scale có trần chi tiêu tháng lần lượt 500, 1.000 và 200.000 USD. Chạm trần thì API trả `429`, `rate_limit_error`, **không có** `retry-after`, `error.details.error_code` là `enforced_spend_limit_reached`, và mọi retry kể cả retry tự động của SDK đều thất bại cho đến 00:00 UTC ngày đầu tháng sau [2].
- **Spend limit tự đặt** thấp hơn trần trả `400 invalid_request_error`, message bắt đầu bằng `You have reached your specified API usage limits` (hoặc `workspace API usage limits`) [2]. Khác hẳn trường hợp trên: cùng là "hết tiền" nhưng một cái là `429`, một cái là `400`.
- **Header.** `retry-after`, `anthropic-ratelimit-requests-*`, `anthropic-ratelimit-input-tokens-*`, `anthropic-ratelimit-output-tokens-*`, và `anthropic-ratelimit-tokens-*` cho giới hạn chặt nhất đang áp dụng (kể cả giới hạn của workspace) [2].

Số cụ thể đổi theo bậc và theo thời gian. Ví dụ ở bậc Start lúc viết bài: Opus 5.5, Sonnet 5.5 và Haiku 5.5 cùng mức 4.000 RPM, 2.000.000 ITPM, 400.000 OTPM; nhóm Fable 5.x là 4.000 RPM, 500.000 ITPM, 100.000 OTPM [2]. Xem số của tổ chức bạn ở trang Rate limits trong Console.

### OpenAI

- Giới hạn theo RPM, RPD, TPM, TPD và IPM (ảnh mỗi phút), áp ở cấp organization và project [11].
- **TPM tính theo `max_tokens`.** "Your rate limit is calculated as the maximum of `max_tokens` and the estimated number of tokens based on the character count of your request" [11]. Đặt `max_tokens` dư nhiều sẽ giữ chỗ TPM dù output thực ngắn. Đây là điểm ngược với Claude.
- **`slow_down`.** Traffic tăng quá nhanh thì nhận `429` với code `slow_down`, khác với `503 server_is_overloaded` khi model quá tải. Một số endpoint trước đây dùng `503 slow_down` cho cả hai trường hợp; code xử lý lỗi cần kiểm cả HTTP status lẫn `error.code` [11].
- **Header** `x-ratelimit-limit-*`, `x-ratelimit-remaining-*`, `x-ratelimit-reset-*` cho request và token; `Retry-After` có thể có trên `429` tạm thời và `503` quá tải, và không có nghĩa lỗi quota hay billing sẽ tự hết nếu retry [11].
- **Hard spend limit** của organization hoặc project trả `429` [11]. "Retrying billing, spend, or quota errors won't restore API access" [10].

### Hệ quả cho fallback

Fallback vì rate limit chỉ có tác dụng khi model đích nằm ở bucket khác: họ model khác, provider khác, hoặc cùng model ở nền tảng khác (Claude trên Bedrock có quota của AWS, không dùng chung với Claude API). Fallback vì hết tiền thì không có tác dụng trong cùng tổ chức. Chuyển sang provider khác để né hạn mức chi tiêu là quyết định kinh doanh, không phải quyết định của vòng retry.

## 1.3 Retry, backoff và jitter

### Ba thành phần

- **Exponential backoff:** lần thử thứ n chờ `base * 2**n`, có trần. Provider quá tải cần thời gian để hồi; retry ngay lập tức chỉ thêm tải.
- **Jitter:** thêm phần ngẫu nhiên vào thời gian chờ. Không có jitter, mọi client gặp lỗi cùng lúc sẽ retry cùng lúc và tạo thêm một đợt quá tải. OpenAI và Gemini đều khuyên thêm jitter [11][22]. Full jitter (`random.uniform(0, cap)`) trải đều nhất.
- **`retry-after`:** khi server nói thời gian chờ, đó là mức tối thiểu. Retry sớm hơn chắc chắn thất bại. OpenAI viết thêm: "If a valid server delay exceeds the supported or configured maximum retry delay, stop retrying and defer the request rather than retrying sooner" [11].

### SDK đã retry sẵn

SDK Python của Claude retry 2 lần với backoff 0,5 s nhân đôi, trần 8 s [1][29]. Đọc mã nguồn bản 2.6.0 thấy thêm hai chi tiết:

- Có `retry-after` (hoặc `retry-after-ms`) thì SDK chờ đúng giá trị đó, không có trần riêng. Đo thử với mock trả `retry-after: 3`: SDK chờ 3,02 s rồi gửi lại. Một header `retry-after: 120` sẽ giữ request của người dùng thêm hai phút.
- Timeout áp cho từng lần thử. Với timeout mặc định 10 phút và 2 lần retry, một lời gọi có thể kéo dài khoảng 30 phút trước khi báo lỗi.

Gemini Python SDK retry tối đa 4 lần, chờ tối đa 60 s [22]. OpenAI SDK tự retry 429 và 503 [11].

### Retry chồng retry

Khi ứng dụng có vòng retry riêng trên một SDK đã retry, số request nhân lên. OpenAI viết thẳng: "If you manage retries in your application, disable SDK retries or account for them in those limits so nested retry loops don't multiply requests" [11]. Gateway thêm một tầng nữa. Ví dụ ba tầng, mỗi tầng ba lần thử, thì một request lỗi có thể thành 27 request lên provider đang quá tải, nhân với số người dùng đang bấm "thử lại".

Vòng fallback cũng có thể tự lặp. LiteLLM issue #19985 (tháng 1/2026) mô tả router có cả `num_retries` lẫn `fallbacks`: model fallback hết lượt retry thì logic fallback chạy lại, và model đó lại được retry `num_retries` lần, lặp không dừng [28]. Issue đã đóng ngày 2/2/2026. Bài học vẫn còn nguyên khi bạn tự viết chuỗi fallback: trần phải đặt cho cả chuỗi, và cần test với mock lỗi để thấy tổng số request thực gửi đi.

### Retry tốn tiền, không chỉ tốn thời gian

Gọi lại LLM không có tác dụng phụ ngoài chi phí. Nhưng chi phí là thật: request lỗi sau khi đã stream một phần output vẫn bị tính tiền phần đã sinh ra. Refusal của Claude giữa stream tính tiền input và phần output đã stream; refusal trước khi có output tính tiền khi thuộc nhóm `bio`, `frontier_llm`, `reasoning_extraction` [3]. Request có tool phía server (web search, code execution) thì retry chạy lại tool đó và tính tiền lại.

Với streaming, retry sau khi đã gửi token cho người dùng sẽ làm lặp nội dung. Quy tắc an toàn: chỉ retry hoặc fallback khi chưa gửi gì cho client, hoặc đệm output đến khi chắc chắn. Chi tiết lỗi giữa stream có ở [Streaming](../streaming/).

## 1.4 Routing theo độ khó, theo loại tác vụ, theo chi phí

Router chọn model cho từng request. Có ba cách tiếp cận chính, và bằng chứng về hiệu quả của chúng không đồng đều.

### Theo độ khó: chọn giữa model mạnh và model yếu

**RouteLLM** (Ong et al., arXiv 2406.18665) huấn luyện router từ dữ liệu preference của Chatbot Arena (khoảng 80.000 trận, sau khi lọc còn 65.000 cặp so sánh giữa 64 model), quyết định gửi mỗi câu cho GPT-4 hay Mixtral 8x7B [23]. Abstract báo cáo giảm chi phí "over 2 times without sacrificing response quality"; phần phân tích chi phí nói tới 3,66 lần, tính từ số lượt gọi GPT-4 vì GPT-4 chiếm gần hết chi phí [23]. Hai chi tiết đáng chú ý hơn con số:

- Trên MMLU, mọi router chỉ huấn luyện bằng dữ liệu Arena đều tệ ngang router ngẫu nhiên. Tác giả giải thích là phần lớn câu MMLU nằm ngoài phân phối dữ liệu huấn luyện. Thêm khoảng 1.500 mẫu có nhãn từ tập validation của MMLU (dưới 2% dữ liệu huấn luyện) thì router mới vượt ngẫu nhiên [23].
- Trên GSM8K cũng vậy: router chỉ học từ Arena gần như ngẫu nhiên, phải bổ sung dữ liệu do GPT-4 chấm mới cải thiện [23].

Nói cách khác, router học được chỉ tốt trên phân phối nó đã thấy. Traffic thật của bạn gần như chắc chắn khác Chatbot Arena.

### Cascade: thử model rẻ trước

**FrugalGPT** (Chen, Zaharia, Zou, arXiv 2305.05176) gọi lần lượt từ model rẻ đến model đắt và dừng khi một scorer đánh giá câu trả lời đủ tin cậy. Abstract báo cáo đạt hiệu năng ngang model tốt nhất (GPT-4) "with up to 98% cost reduction", hoặc tăng 4% độ chính xác với cùng chi phí [24]. Cascade trả giá bằng độ trễ: câu khó đi qua mọi tầng trước khi tới model mạnh. Trong repo này, [Decision models §3.7](../decision-models/#37-cascade-trích-xuất-dữ-liệu) có ví dụ cascade cho trích xuất dữ liệu.

### Theo loại tác vụ

**Lee, "Most of the LLM routing gap is task type"** (arXiv 2608.23023, tháng 8/2026) đưa ra bằng chứng ngược chiều [26]. Abstract dẫn các khảo sát trước: 21 phương pháp routing trên năm benchmark cho kết quả chênh nhau chưa tới một điểm, đều xa mức trần lý thuyết, và "learned routers often fail to beat simply always calling the strongest model". Tác giả cho 14 model trả lời 294 câu thuộc 7 loại tác vụ, bằng tiếng Hàn, tiếng Anh và tiếng Hindi, rồi chạy lại toàn bộ lần hai mà không đổi gì. 5,37% trong 4.116 cặp model-câu hỏi đổi kết quả giữa hai lần chạy. Chỉ tính câu đúng ở cả hai lần thì chỉ 29 câu có thể cải thiện bằng routing. Một bảng cố định gán mỗi loại tác vụ một model, chọn một lần và không cập nhật, cải thiện 21 trong 29 câu đó. Bảng này trả lời đúng 262/294 câu với 3,33 USD mỗi lượt chạy, trong khi model đơn lẻ tốt nhất đúng 245 câu với 7,69 USD. Tác giả tự ghi rằng mọi thứ được khớp và chấm trên cùng 294 câu, không có holdout, nên đây là gợi ý chứ chưa phải kết luận chung.

Hai bài học thực tế từ paper này:

1. Trước khi huấn luyện router, thử một bảng tĩnh: loại tác vụ nào dùng model nào. Ứng dụng thường đã biết loại tác vụ từ endpoint hay tính năng gọi tới (phân loại ticket, tóm tắt, sinh code), không cần đoán từ prompt.
2. Nhiễu giữa các lần chạy có thể lớn hơn phần cải thiện mà router mang lại. So sánh router với baseline phải chạy lặp ([§3.5](#35-đo-và-eval-khi-route-sang-model-khác)).

### Router có sẵn trong sản phẩm

- **Azure model router** là một ML model nhẹ, không phải LLM, phân tích từng prompt để ước lượng độ khó, có ba chế độ Balanced (mặc định), Cost và Quality [18][19]. Microsoft nói router được huấn luyện "on hundreds of thousands of examples" nhưng không công bố số đo chất lượng định tuyến [19].
- **OpenRouter Auto Router** phân loại prompt vào khoảng 30 loại tác vụ, rồi xếp hạng model theo tỷ trọng chi tiêu của cộng đồng OpenRouter trong 7 ngày gần nhất [14]. Đó là tín hiệu "nhiều người đang dùng", không phải chất lượng đo trên tác vụ của bạn.
- **Bedrock Intelligent Prompt Routing** dự đoán chất lượng câu trả lời của hai model cùng họ cho mỗi prompt, chỉ tối ưu cho prompt tiếng Anh [20].

Chi tiết các sản phẩm này ở [Phần 2](#phần-2--công-cụ). Router dùng decision model (chấm độ khó, miền, mức rủi ro) có ở [Decision models §3.11](../decision-models/#311-routing-giữa-các-llm).

### Trước khi dựng cascade nhiều model

Model mới có nút `effort` điều khiển mức suy luận (Claude: `output_config.effort` từ `low` đến `max`). Trước khi dựng router giữa model rẻ và model đắt, hãy đo phương án đơn giản hơn: một model mạnh với effort thấp cho tác vụ dễ. Một model nghĩa là một namespace cache, một bộ tham số, một bộ prompt. Cascade qua nhiều model mất cache ở mỗi lần đổi model [4][8], và mỗi model thêm vào là một tổ hợp tham số phải giữ tương thích ([§3.2](#32-giữ-prompt-tool-và-schema-tương-thích-giữa-các-model)). Tính chi phí theo tác vụ hoàn thành, không theo request: model rẻ cần thêm lượt hoặc thêm retry mới xong việc thì chưa chắc rẻ hơn.

## 1.5 Model khác nhau trả lời khác nhau: hệ quả khi fallback

Đổi model giữa chừng làm thay đổi bảy thứ. Phần lớn ví dụ dưới đây là của Claude vì tài liệu Claude mô tả chi tiết nhất; provider khác có vấn đề tương tự ở dạng khác.

### a. Request body hợp lệ ở model này, `400` ở model kia

Trang lỗi của Claude liệt kê nhiều trường hợp cùng một request chạy được trên model A và bị `400` trên model B [1]:

- **Extended thinking.** `thinking: {"type": "enabled", "budget_tokens": N}` bị từ chối trên Claude 4.7 trở lên. Ngược lại, `{"type": "adaptive"}` bị từ chối trên Claude 4.5 và cũ hơn.
- **Tắt thinking.** `{"type": "disabled"}` bị từ chối trên Fable 5.1, Fable 5, Mythos 5.1, Mythos 5 và Opus 5.5 vì thinking luôn bật. Sonnet 5.5 cũng từ chối `disabled` và dùng `{"type": "between_tools"}` để tắt thinking đầu lượt; gửi `between_tools` cho model nào khác Sonnet 5.5 đều bị `400`.
- **Forced tool use.** Opus 5.5, Sonnet 5.5, Fable 5.1 và Mythos 5.1 từ chối `tool_choice: {"type": "any"}` và `{"type": "tool", ...}`, kể cả trên endpoint đếm token. Haiku 5.5 vẫn nhận forced tool use [1][6]. Chuỗi Opus 5.5 → Sonnet 5.5 cho một request ép gọi tool sẽ `400` ở cả hai bước.
- **Prefill.** Claude 4.6 trở lên từ chối message cuối là assistant [1].
- **Computer use.** Haiku 5.5 từ chối `computer_20250124` và dùng `computer_toolset_20260801` [6].
- **Sampling.** Từ API version `2026-10-21`, request gửi `temperature`, `top_p` hoặc `top_k` bị `400` trên mọi model, kể cả khi giá trị bằng mặc định [9]. SDK Python 2.6.0 mặc định gửi version này và đã bỏ ba tham số khỏi `messages.create`. Gateway hay thư viện trung gian (LiteLLM và các thư viện tương tự) còn chèn `temperature` vào request Claude sẽ gây lỗi; tài liệu migration của Anthropic nhắc kiểm cấu hình của chúng [9].

Hệ quả: request template phải được dựng theo model đích, không dùng chung một body cho cả chuỗi ([§3.2](#32-giữ-prompt-tool-và-schema-tương-thích-giữa-các-model)).

### b. Tokenizer và độ dài

Haiku 5.5 dùng tokenizer của Claude 4.7 trở lên; cùng văn bản tốn khoảng 30% token hơn Haiku 4.5 [6]. Ảnh lớn cũng tốn nhiều visual token hơn. `max_tokens` chỉnh cho model cũ có thể cắt mất output ở model mới, và thinking tính vào `max_tokens`, nên request có `max_tokens` nhỏ có thể dừng với `stop_reason: "max_tokens"` sau block thinking, trước khi có chữ nào [6]. Đếm lại token bằng endpoint đếm token của model đích, đừng nhân hệ số cố định.

### c. Prompt cache tách theo model

"Prompt caches are per-model" [4]. Fallback từ A sang B thì toàn bộ prefix đã cache ở A phải ghi lại vào cache của B. Cache write đắt hơn cache read: với Claude, write TTL 5 phút là 1,25 lần giá input, read khoảng 0,1 lần (rẻ hơn nữa trên một số model mới) [8]. Ngưỡng tối thiểu để cache cũng khác theo model: 512 token cho các model 5.x hiện tại, 1.024 cho Opus 4.8 và Sonnet 5, 2.048 cho Opus 4.7, 4.096 cho Haiku 4.5, Opus 4.6 và 4.5; prompt ngắn hơn ngưỡng không được cache và không báo lỗi [8]. Khi fallback vì refusal, Claude có cơ chế fallback credit bù phần chi phí cache này ([§2.2](#22-cloud-provider-bedrock-vertex-ai-azure-openai-và-cross-region-inference)); fallback vì lỗi hạ tầng thì không có.

### d. Thinking block không đi theo

Mỗi model Claude chỉ đọc được thinking block của chính nó và của một số model khác theo bảng cố định. Block model đích không đọc được sẽ bị bỏ lặng lẽ, request vẫn thành công [7]. Ví dụ: Opus 5.5 đọc block của Opus 5, của các model Opus, Sonnet, Haiku cũ hơn, và (trên Claude API) của Sonnet 5.5 và Haiku 5.5, nhưng không đọc block của Fable hay Mythos. Chiều ngược lại hẹp hơn nhiều: ngoài Fable 5.1 và Mythos 5.1 trên Claude API, không model nào đọc block của Opus 5.5 [7]. Fallback thường đi từ model mới xuống model cũ, ví dụ Opus 5.5 sang Opus 4.8, và đó đúng là chiều mất reasoning. Fallback giữa hội thoại nghĩa là model mới làm tiếp mà không có reasoning của các lượt trước. API không sửa mảng `messages` của bạn, nên block vẫn còn trong lịch sử và dùng lại được nếu hội thoại quay về model cũ [7].

Thinking block của Sonnet 5.5 và Haiku 5.5 còn bị ràng buộc theo tài khoản: gửi block đó qua tài khoản khác (không liên kết) thì API bỏ block, request vẫn thành công [7]. Fallback sang cùng model nhưng qua tài khoản hoặc key của tổ chức khác cũng mất reasoning.

Với Claude 5.1 trở lên, sửa `system`, `tools` hay message cũ sau khi đã có thinking block có thể làm request bị `400 Invalid signature in thinking block`. Chuỗi fallback mà đổi system prompt theo model đích sẽ gặp lỗi này. Cách xử lý (giữ lịch sử append-only, hoặc beta `thinking-binding-controls-2026-08-01` với `prefix_mismatch_behavior: "drop_block"`) có ở [1] và [Multi-turn](../multi-turn-conversation/).

### e. Stop reason và refusal

Refusal của Claude có `stop_details.category` (`cyber`, `bio`, `frontier_llm`, `reasoning_extraction`, `general_harms` hoặc `null`) [3]. Model khác nhau có bộ classifier khác nhau, nên request bị một model từ chối thường được model khác phục vụ; đó là lý do Anthropic có fallback phía server cho refusal. Với `reasoning_extraction` (prompt bắt model viết lại suy luận nội bộ vào output), tài liệu nói không có model fallback khuyến nghị và nên sửa prompt thay vì retry [3].

`stop_reason` khác cũng cần nhánh riêng: `max_tokens` và `model_context_window_exceeded` là output bị cắt, `pause_turn` là server tool chưa xong [5]. Gộp tất cả vào "lỗi" hay "thành công" đều sai.

### f. Giọng và độ dài câu trả lời

Request hợp lệ ở model mới chưa chắc cho output tương đương. Anthropic ghi trong migration guide: "Claude 4 and later models have a more concise, direct communication style and need explicit direction" [6]. Prompt viết cho model cũ có thể cho câu trả lời ngắn hơn, ít giải thích hơn ở model mới. Fallback không đổi prompt nghĩa là chất lượng đổi theo cách khó thấy trên dashboard lỗi.

### g. Tool calling khác provider

Fallback sang provider khác thì định dạng tool, cách ép gọi tool, strict schema và trạng thái reasoning đi kèm tool call đều khác. [Tool use Phần 2](../tool-use/#phần-2--tool-use-theo-provider) so sánh chi tiết. Gateway dịch format giữa các provider giúp request chạy được, nhưng không dịch được hành vi.

---

# Phần 2 — Công cụ

## 2.1 LLM gateway: LiteLLM, Portkey, OpenRouter, Cloudflare AI Gateway

Gateway đứng giữa ứng dụng và provider, gom routing, retry, fallback, quota, log vào một chỗ. Điểm đầu tiên cần biết ở mỗi gateway: **lỗi nào kích hoạt fallback theo mặc định**.

| Gateway | Mặc định kích hoạt fallback khi | Điều chỉnh |
|---|---|---|
| LiteLLM Router | Deployment lỗi (lỗi kết nối, 404, 429...) thì thử deployment `order` kế tiếp, hết thì sang `fallbacks` [12] | `context_window_fallbacks`, `content_policy_fallbacks`, `retry_policy` theo loại lỗi |
| OpenRouter `models` | Bất kỳ lỗi nào, gồm lỗi độ dài context, moderation, rate limit, downtime [13] | Danh sách model theo thứ tự |
| Portkey | Mọi status code không phải 2xx [15] | `strategy.on_status_codes`, ví dụ `[429, 503]` |
| Cloudflare AI Gateway | Request lỗi, hoặc quá timeout đã đặt [16] | Thứ tự trong mảng request |
| Claude `fallbacks` (phía server) | **Chỉ** khi classifier an toàn từ chối. Rate limit, quá tải, lỗi server trả về nguyên trạng [3] | `"default"` hoặc tối đa 3 model |

Hàng cuối không phải gateway nhưng đặt đây để so sánh. Tham số `fallbacks` của OpenRouter trên endpoint Messages có cùng tên và cùng shape với `fallbacks` của Anthropic SDK, nhưng OpenRouter tự xử lý và kích hoạt trên mọi lỗi ở bảng trên, "not only on refusals" [13]. Code chạy đúng trên một nền tảng sẽ hành xử khác khi đổi `base_url` sang nền tảng kia.

### LiteLLM Router

- **Chiến lược chọn deployment** (`routing_strategy`): `simple-shuffle` là mặc định và được khuyến nghị cho production (chọn theo `weight`, `rpm`, `tpm`); ngoài ra có `least-busy`, `usage-based-routing-v2`, `latency-based-routing`, `cost-based-routing` [12].
- **`order`**: số nhỏ được thử trước. "Each order level gets its own set of retries before escalating to the next. If all order levels are exhausted, the router falls through to any configured fallbacks" [12].
- **`context_window_fallbacks`** được áp trước các `order` sau cho lỗi vượt context; `content_policy_fallbacks` xử lý riêng lỗi chính sách nội dung [12].
- **`enable_weighted_failover`**: mặc định tắt; bật thì thử deployment khác trong cùng nhóm trước khi sang nhóm fallback, tối đa `max_fallbacks` (mặc định 5) [12].
- **Cooldown**: khi chưa đặt `allowed_fails`, deployment bị đưa vào trạng thái nguội 5 giây ngay khi gặp `429`, khi quá 50% request lỗi trong phút hiện tại, hoặc khi gặp `401`, `404`, `408` [12]. Đặt `allowed_fails` và `cooldown_time` để chỉnh.
- **`num_retries`** đặt được ở bốn chỗ (request body, deployment, ...), có thứ tự ưu tiên [12]. Cùng một request có thể retry khác nhau tuỳ nơi bạn đặt.
- **`max_parallel_requests`**: nếu không đặt mà có `rpm` hay `tpm`, LiteLLM tự suy ra giới hạn song song theo từng process. Request đến khi hết slot bị `429` ngay, không có hàng đợi; giới hạn này đếm trong process, không chia sẻ giữa các worker hay pod [12]. Một deployment chỉ đặt `rpm: 2` sẽ chỉ chạy song song được 2 request mỗi worker.
- **Session affinity**: có thể ghim mọi request của một hội thoại vào deployment đã phục vụ request đầu, TTL mặc định 3.600 giây [12]. Hữu ích để giữ prompt cache.

### OpenRouter

- **`models`**: mảng model ID theo thứ tự; model đầu lỗi thì thử model kế. Model fallback cũng lỗi thì trả lỗi đó [13].
- **Tính tiền theo model thực phục vụ**, ghi ở trường `model` của response. Model `:free` mà fallback sang model trả phí thì bạn trả giá trả phí; muốn request luôn miễn phí thì chỉ liệt kê model `:free` [13].
- **Endpoint Messages (`/api/v1/messages`)** nhận `fallbacks` với tối đa 3 mục, mỗi mục chỉ có `model`; ghi đè `max_tokens`, `thinking`, `speed`, `output_config` theo từng mục bị `400`; không dùng chung với `models` [13]. Fallback ở đây là của OpenRouter, không dùng fallback phía server của Anthropic [13].
- **Auto Router** (`openrouter/auto`): classifier nhẹ xếp prompt vào khoảng 30 loại tác vụ, xếp hạng model theo tỷ trọng chi tiêu 7 ngày, có thể giới hạn bằng `allowed_models`, `excluded_models`, mức chi phí; trường `model` của response cho biết model đã chọn [14].

### Portkey

Cấu hình `strategy.mode = "fallback"` với danh sách `targets` theo thứ tự. Mặc định mọi response không phải 2xx đều kích hoạt fallback; `on_status_codes: [429]` chỉ fallback khi bị rate limit [15]. Target có thể lồng nhau: một target là load balancer, conditional router hoặc chuỗi fallback khác [15]. Mặc định "mọi non-2xx" có nghĩa `400` do request sai cũng được gửi sang model kế tiếp, tốn thêm một request và che mất bug thật. Nên đặt `on_status_codes` tường minh.

### Cloudflare AI Gateway

- **Fallback qua Universal endpoint**: gửi một mảng request, mỗi phần tử chỉ định provider, endpoint, header và body riêng. Phần tử đầu lỗi hoặc quá timeout thì gateway thử phần tử kế. Header `cf-aig-step` của response cho biết bước nào đã phục vụ [16]. Vì mỗi bước có body riêng, bạn có thể dùng tham số phù hợp với từng model trong cùng chuỗi.
- **Dynamic routes** (beta): dựng luồng routing bằng giao diện hoặc JSON, gồm điều kiện, quota, ngân sách và rate limit theo user, project, team, chia traffic A/B; ứng dụng gọi tên route thay cho tên model, đổi route không cần deploy lại code [17].

### Gateway không giải quyết được gì

Gateway dịch được format request và chuyển request sang provider khác. Nó không biết request của bạn có dùng tính năng mà model đích không hỗ trợ, không giữ được cache hay thinking block qua model, và không biết output ở model fallback có đủ tốt cho tác vụ của bạn hay không. Mục [§1.5](#15-model-khác-nhau-trả-lời-khác-nhau-hệ-quả-khi-fallback) vẫn áp dụng nguyên vẹn khi có gateway.

## 2.2 Cloud provider: Bedrock, Vertex AI, Azure OpenAI và cross-region inference

### Fallback có sẵn của Claude

Claude có hai cơ chế fallback cho refusal, cả hai đang ở beta [3]:

- **Fallback phía server.** Thêm beta header `server-side-fallback-2026-07-01` và tham số `fallbacks: "default"`. Khi model chính từ chối, API chạy lại request trên model Anthropic khuyến nghị cho nhóm refusal đó, trong cùng một lời gọi. Có thể thay `"default"` bằng danh sách tối đa 3 model, mỗi mục có thể ghi đè `max_tokens`, `thinking`, `output_config`, `speed` cho lần thử đó. Có trên Claude API và Claude Models on Google Cloud (Anthropic managed). Haiku 5.5 không có fallback phía server [3].
- **Middleware trong SDK** (`BetaRefusalFallbackMiddleware` cùng `BetaFallbackState`) làm việc tương tự ở phía client, dùng được trên mọi nền tảng, kể cả Bedrock và Foundry. Middleware chỉ xử lý request qua `client.beta.messages` [3][29].

Những điều cần biết khi dùng:

- **Chỉ refusal kích hoạt fallback.** "A rate limit, overload, or server error on the requested model is returned to you as-is" [3]. Muốn fallback khi quá tải, bạn vẫn cần chuỗi fallback riêng ([§3.1](#31-thiết-kế-chuỗi-fallback)).
- **Model fallback bị rate limit hay quá tải thì không thử**, và refusal được trả về kèm `stop_details.recommended_model` gợi ý model để thử trực tiếp [3]. Cần chừa đủ rate limit ở model fallback cho lượng refusal dự kiến.
- **Request phải hợp lệ với mọi model trong danh sách**; model fallback không hỗ trợ một tính năng request dùng thì API từ chối ngay từ đầu [3].
- **Response cho biết ai đã trả lời**: trường `model` ở cấp trên cùng, một content block `fallback` đánh dấu chỗ chuyển model, và `usage.iterations` ghi từng lần thử. Usage ở cấp trên cùng chỉ tính lần thử đã tạo ra message trả về [3]. Tính chi phí phải cộng `usage.iterations`.
- **Sticky routing**: sau khi một hội thoại đã fallback, các request sau của hội thoại đó (có `fallbacks`) đi thẳng tới model fallback, giữ khoảng 1 giờ, lưu dưới dạng hash của prefix, best effort [3].
- **Streaming**: refusal trước khi có output thì `message_start` đã mang tên model fallback, nên time to first byte gồm cả lần thử bị từ chối. Refusal giữa output thì block đang mở được đóng, block `fallback` đánh dấu ranh giới, và model fallback viết tiếp từ phần text đã có [3].
- **Fallback credit**: khi tự viết retry, gửi header `fallback-credit-2026-07-01`, lấy `stop_details.fallback_credit_token` từ refusal và gửi kèm lần thử lại để được tính tiền như thể hội thoại chạy trên model mới từ đầu, tức không trả thêm tiền ghi cache [4]. Token hết hạn sau 5 phút. Lần thử lại phải giữ nguyên `system`, `messages`, `tools`, `tool_choice`, `thinking`, `cache_control` và các beta header; chỉ được đổi `model`, `max_tokens`, `stop_sequences`, `stream`, `metadata`, `service_tier` [4]. Model đích được phép: Fable 5.1, Fable 5 và Opus 5.5 sang Opus 4.8 hoặc Opus 5; Opus 5 sang Opus 4.8; Sonnet 5.5 sang Sonnet 5 [4]. Refusal của Haiku 5.5 và refusal trong Message Batches không có credit [4]. Fallback phía server và middleware tự áp credit [3].

Chi tiết đầy đủ ở [Structured output](../structured-output/) và [Tool use](../tool-use/) cho phần tương tác với tool; bài này chỉ nói phần liên quan tới chọn model.

### AWS Bedrock

- **Cross-region inference** dùng inference profile: profile theo vùng địa lý (US, EU, APAC) chọn region trong vùng đó; profile global chọn bất kỳ region thương mại nào [21]. Profile global rẻ hơn khoảng 10% và không giữ dữ liệu trong một vùng; profile theo vùng dùng khi có yêu cầu lưu trú dữ liệu [21]. Không có phí routing thêm; CloudTrail ghi region thực xử lý ở `additionalEventData.inferenceRegion` [21]. Inference profile hiện không hỗ trợ Provisioned Throughput [21]. Chính AWS ghi trong bảng so sánh: muốn đạt mục tiêu uptime, "add bounded retries and multi-Region or multi-model fallback in your application" [21]. Cross-region inference giảm rủi ro một region, không thay được chuỗi fallback.
- **Intelligent Prompt Routing** chọn giữa đúng hai model cùng họ (ví dụ Nova Lite và Nova Pro). Bạn chọn một model làm fallback làm mốc và đặt `responseQualityDifference`; router chỉ chuyển sang model còn lại khi chất lượng dự đoán của nó hơn model mốc quá ngưỡng. Ví dụ trong tài liệu: ngưỡng 10% với Nova Lite làm mốc [20]. Giới hạn: chỉ tối ưu cho prompt tiếng Anh [20]. Ứng dụng tiếng Việt nên đo trước khi dùng.

### Azure: Microsoft Foundry model router

- Một deployment duy nhất, mặc định chế độ Balanced trên toàn bộ model được hỗ trợ; đổi chế độ hoặc tập model mất tới 5 phút mới có hiệu lực [18].
- Chọn tập con model: "Select at least two models for failover. A single-model subset defeats the purpose of routing and disables automatic failover" [19]. Model mới phát hành sau này bị loại theo mặc định cho đến khi được thêm vào [18].
- Chat Completions là stateless, nên router có thể chọn model khác ở lượt sau của cùng hội thoại. Session affinity (preview) yêu cầu router thử lại model cũ trước, có thể tăng cơ hội tái dùng cache nhưng "doesn't inspect cache state or guarantee a cache hit"; liên kết hết hạn sau 30 phút không có request thành công; chính sách, an toàn, khả năng, quota và fallback được ưu tiên hơn affinity [18].
- Model reasoning bỏ qua `temperature` và `top_p` khi được router chọn [18].
- Chi phí là tổng chi phí của các model bên dưới [18]. Trường `model` trong response là model thực phục vụ.
- Microsoft khuyên đo router trên workload thật so với model hiện dùng về chất lượng, chi phí, độ trễ trước khi chuyển production, và mỗi lần chỉ đổi một đòn bẩy (chế độ hoặc tập model) [18][19].

### Google Cloud Vertex AI

Model Google và model đối tác (gồm Claude) có endpoint theo region và một global endpoint. Theo tài liệu, global endpoint "provide higher availability and reliability than single regions" [22b]. Dùng global endpoint khi không có ràng buộc lưu trú dữ liệu; có ràng buộc thì dùng endpoint multi-region hoặc region cụ thể, kèm chuỗi fallback riêng.

## 2.3 Router dùng classifier hoặc model nhỏ

Router tự dựng có ba dạng, theo chi phí tăng dần:

1. **Bảng tĩnh theo loại tác vụ.** Endpoint `/summarize` dùng model A, `/code-review` dùng model B. Không tốn thêm request, không có độ trễ, dễ debug. Theo [Lee 2026](#theo-loại-tác-vụ), phần lớn lợi ích của routing nằm ở đây [26].
2. **Classifier nhẹ** (embedding cộng logistic regression, BERT, decision model) chấm độ khó hay miền của prompt. Thêm vài chục mili giây. [Decision models §3.11](../decision-models/#311-routing-giữa-các-llm) có ví dụ router hỏi miền, độ khó, mức rủi ro.
3. **LLM làm router**: gọi một model nhỏ để quyết định. Đắt và chậm nhất; khoản tiết kiệm phải lớn hơn chi phí của chính lời gọi đó.

**RouteLLM** (lm-sys) là thư viện mã nguồn mở đi kèm paper ở [§1.4](#theo-độ-khó-chọn-giữa-model-mạnh-và-model-yếu). Bạn chọn một cặp model mạnh và yếu, một loại router (matrix factorization, ranking có trọng số Elo, BERT, causal LLM), và một ngưỡng; ngưỡng quyết định tỷ lệ request đi tới model mạnh và được hiệu chỉnh trên dữ liệu có sẵn [23]. Router đã huấn luyện sẵn học từ Chatbot Arena; như kết quả MMLU ở §1.4 cho thấy, dùng trên traffic khác phân phối thì cần dữ liệu có nhãn của chính bạn.

**RouterBench** (Hu et al., arXiv 2403.12031) là bộ hơn 405.000 kết quả suy luận đã tính sẵn, dùng để so sánh phương pháp routing mà không phải gọi model thật [25]. Model trong bộ dữ liệu là model năm 2024. Hợp để so sánh phương pháp, không hợp để chọn model cho năm 2026.

Ba điều kiện để router tự dựng đáng làm:

- Traffic đa dạng thật: có cả câu rất dễ lẫn câu rất khó trong cùng một luồng, và ứng dụng không biết trước câu nào thuộc loại nào.
- Chênh lệch giá giữa model rẻ và model mạnh đủ lớn để bù chi phí router cộng phần chất lượng mất ở các ca router đoán sai.
- Có dữ liệu có nhãn từ traffic thật để huấn luyện hoặc ít nhất để hiệu chỉnh ngưỡng và đo ([§3.5](#35-đo-và-eval-khi-route-sang-model-khác)).

## 2.4 Priority tier, provisioned throughput và batch

Ba cơ chế này không phải routing, nhưng chúng thay đổi việc bạn có cần fallback hay không.

- **Priority Tier của Claude** ưu tiên request hơn mọi request khác để giảm lỗi quá tải; nhắm mục tiêu uptime 99,5%; chỉ dành cho tổ chức đã có cam kết capacity (số input và output token mỗi phút, thời hạn 1–12 tháng, cho một model cụ thể) [30]. Request vượt phần cam kết tự rơi về standard tier. Tham số `service_tier`: `"auto"` (mặc định, dùng Priority nếu còn) hoặc `"standard_only"` [30]. Priority Tier **không hỗ trợ** Fable 5.1, Haiku 5.5, Mythos 5.1, Mythos 5, Opus 5.5, Opus 5, Sonnet 5.5 và Sonnet 5 [30]. Chuyển sang model mới có thể làm mất cam kết capacity; migration guide của Haiku 5.5 nhắc riêng điểm này [6].
- **OpenAI** có Scale Tier, và Reserved Tier cho GPT-5.6 trở lên, cho capacity dễ dự đoán hơn; tài liệu nói chúng không thay đổi cách xử lý `slow_down`: vẫn theo `Retry-After` và tăng traffic từ từ [11].
- **Provisioned Throughput** trên Bedrock không dùng được với inference profile cross-region [21]. Phải chọn giữa capacity đặt trước ở một region và phân phối qua nhiều region.
- **Batch** đổi độ trễ lấy giá thấp hơn, và chạy ngoài capacity thông thường [30]. Batch không phải đích fallback cho request tương tác, vì kết quả có sau nhiều phút đến nhiều giờ. Lưu ý riêng của Claude: refusal trong batch không có fallback credit, và tham số `fallbacks` không dùng được trong batch (request có `fallbacks` cho kết quả errored); muốn chạy lại các item bị từ chối thì gom lại và gửi batch mới trên model khác [3].

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Thiết kế chuỗi fallback

### Phân loại lỗi trước, chuỗi sau

Hàm `classify` dưới đây chuyển exception của SDK `anthropic` thành một trong ba hành động. Bảng ở [Tra cứu nhanh](#mã-lỗi-theo-provider) là nguồn của các nhánh. Viết cho provider khác thì giữ ba hành động và đổi phần nhận diện lỗi.

| Hành động | Lỗi | Lý do |
|---|---|---|
| `RETRY` | Lỗi kết nối, timeout, `408`, `409`, `429` (trừ spend cap), mọi `5xx`; hoặc server gửi `x-should-retry: true` | Có thể tự hết; cùng quy tắc mà SDK tự retry. Header `x-should-retry: false` thì không retry [29] |
| `NEXT` | `404` | Model ID sai hoặc model đã retire; model khác có thể vẫn chạy. Không tính vào breaker: đây là lỗi cấu hình, sửa chuỗi chứ không chờ |
| `FATAL` | `400`, `401`, `402`, `403`, `413`, `429` có `enforced_spend_limit_reached` | Lỗi của request hoặc tài khoản; model khác sẽ lỗi y hệt |

`400` để `FATAL` có chủ ý. Request sai ở model A thường sai ở model B. Trường hợp `400` chỉ xảy ra ở một model (ví dụ forced `tool_choice`) là bug cấu hình chuỗi: cần sửa chuỗi, không nên để vòng fallback che đi.

### Code

Code dùng SDK `anthropic` 2.6.0. Mỗi `Target` là một model kèm client, các tham số ghi đè riêng và tên provider; một client có thể trỏ tới Claude API, client khác trỏ tới Bedrock hay Vertex cho cùng model. Breaker giữ theo cặp (provider, model), nên Claude API quá tải không chặn luôn cùng model đó trên Bedrock.

```python
import email.utils
import math
import random
import threading
import time
from dataclasses import dataclass, field

import anthropic

RETRY, NEXT, FATAL = "retry", "next", "fatal"

# Model không nhận tool_choice any/tool (trang lỗi của Claude, tháng 11/2026).
NO_FORCED_TOOL = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1"}


def classify(err: anthropic.APIError) -> tuple[str, float | None]:
    """Trả về (hành động, số giây server yêu cầu chờ hoặc None)."""
    if isinstance(err, anthropic.APIConnectionError):     # gồm cả APITimeoutError
        return RETRY, None
    if not isinstance(err, anthropic.APIStatusError):
        return FATAL, None
    status = err.status_code
    retry_after = _retry_after(err.response.headers)
    if status == 429 and err.error_code == "enforced_spend_limit_reached":
        return FATAL, None            # hết spend cap của tổ chức: model nào cũng bị chặn
    should_retry = err.response.headers.get("x-should-retry")    # server quyết thay, như SDK
    if should_retry == "true":
        return RETRY, retry_after
    if status == 404:
        return NEXT, None             # model ID sai hoặc model đã ngừng phục vụ
    if should_retry != "false" and (status in (408, 409, 429) or status >= 500):   # cùng quy tắc SDK
        return RETRY, retry_after
    return FATAL, None                # 400, 401, 402, 403, 413...: lỗi của request hoặc tài khoản


def _retry_after(headers, now=time.time) -> float | None:
    """`retry-after-ms`, rồi `retry-after` dạng số giây hoặc HTTP-date, như SDK. Giá trị
    không dương (kể cả ngày đã qua) trả None để caller tự backoff, cũng như SDK."""
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        try:
            v = float(headers[name]) / scale
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0:
            return v
    try:
        when = email.utils.parsedate_to_datetime(headers["retry-after"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:                       # HTTP-date luôn là GMT
        return None
    wait = when.timestamp() - now()
    return wait if wait > 0 else None


class Breaker:
    """Circuit breaker cho một target: mở sau `threshold` lỗi liên tiếp, thử lại sau `cooldown_s`."""

    def __init__(self, *, threshold: int, cooldown_s: float, clock=time.monotonic):
        self.threshold, self.cooldown_s, self.clock = threshold, cooldown_s, clock
        self.failures = 0
        self.open_until = 0.0
        self.probe_started: float | None = None
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            now = self.clock()
            if self.failures < self.threshold:
                return True
            if now < self.open_until:
                return False
            # half-open: mỗi lần chỉ một request thăm dò
            if self.probe_started is not None and now - self.probe_started < self.cooldown_s:
                return False
            self.probe_started = now
            return True

    def success(self) -> None:
        with self.lock:
            self.failures, self.probe_started = 0, None

    def failure(self, hold_s: float | None = None) -> None:
        with self.lock:
            self.failures += 1
            self.probe_started = None
            if self.failures >= self.threshold:
                self.open_until = self.clock() + max(self.cooldown_s, hold_s or 0.0)


@dataclass
class Target:
    model: str
    client: anthropic.Anthropic                     # tạo với max_retries=0
    overrides: dict = field(default_factory=dict)   # giá trị None nghĩa là bỏ tham số đó
    provider: str = "claude-api"                    # cùng model qua Bedrock, Vertex: provider khác

    @property
    def key(self) -> tuple[str, str]:               # breaker theo (provider, model)
        return self.provider, self.model


class AllFailed(Exception):
    def __init__(self, log, last_error):
        super().__init__(f"mọi model đều lỗi: {log}")
        self.log, self.last_error = log, last_error


def _backoff(attempt: int, base: float = 0.5, cap: float = 8.0) -> float:
    return random.uniform(0, min(cap, base * 2 ** attempt))     # full jitter


def call_with_fallback(request: dict, targets: list[Target], breakers: dict[tuple[str, str], Breaker], *,
                       deadline_s: float, max_attempts: int, attempts_per_target: int,
                       max_wait_s: float, sleep=time.sleep, clock=time.monotonic):
    start = clock()
    attempts, log = 0, []
    last_error, last_refusal = None, None
    forced = (request.get("tool_choice") or {}).get("type") in ("any", "tool")
    for t in targets:
        if forced and t.model in NO_FORCED_TOOL:
            log.append((t.model, "skip: không hỗ trợ forced tool_choice"))
            continue
        body = {k: v for k, v in {**request, **t.overrides}.items() if v is not None}
        for i in range(attempts_per_target):
            remaining = deadline_s - (clock() - start)
            if attempts >= max_attempts or remaining <= 0:
                return _give_up(log, last_error, last_refusal)
            if not breakers[t.key].allow():
                log.append((t.model, "skip: breaker mở"))
                break
            attempts += 1
            try:
                resp = t.client.messages.create(**body, model=t.model, timeout=remaining)
            except anthropic.APIError as e:
                action, wait = classify(e)
                last_error = e
                log.append((t.model, f"{action}: {getattr(e, 'status_code', type(e).__name__)}"))
                if action == FATAL:
                    raise
                if action == NEXT:
                    break             # 404 là lỗi cấu hình, không tính vào breaker
                breakers[t.key].failure(hold_s=wait)
                if wait is None:
                    wait = _backoff(i)
                if i + 1 == attempts_per_target or wait > min(max_wait_s, deadline_s - (clock() - start)):
                    break             # không retry sớm hơn server yêu cầu; sang model kế
                sleep(wait)
                continue
            breakers[t.key].success()
            if resp.stop_reason == "refusal":
                log.append((t.model, "refusal"))
                last_refusal = resp
                break                 # gửi lại cùng model thường bị từ chối tiếp
            log.append((t.model, "ok"))
            return resp, log
    return _give_up(log, last_error, last_refusal)


def _give_up(log, last_error, last_refusal):
    if last_refusal is not None:
        return last_refusal, log      # mọi model đều từ chối: trả refusal cuối, không coi là lỗi hạ tầng
    raise AllFailed(log, last_error)
```

Cách dùng:

```python
primary = anthropic.Anthropic(max_retries=0)
targets = [
    Target("claude-sonnet-5-5", primary),
    Target("claude-sonnet-5", primary, {"thinking": None}),  # between_tools chỉ Sonnet 5.5 nhận
]
# Các số dưới đây là ví dụ, không phải mặc định an toàn: suy ra từ deadline của request
# và tần suất lỗi đo được (xem "Cấu hình khởi điểm" ở Tra cứu nhanh).
breakers = {t.key: Breaker(threshold=5, cooldown_s=30) for t in targets}
resp, log = call_with_fallback(request, targets, breakers, deadline_s=30, max_attempts=4,
                               attempts_per_target=2, max_wait_s=10)
```

`breakers` phải sống lâu hơn một request (biến cấp module, hoặc store dùng chung giữa các worker); tạo mới cho mỗi request thì breaker không bao giờ mở. `log` nên đi vào trace của request ([§3.5](#35-đo-và-eval-khi-route-sang-model-khác)).

### Đã kiểm những gì

Test chạy SDK `anthropic` 2.6.0 thật, thay transport bằng `httpx2.MockTransport` để trả response giả lập, và dùng đồng hồ giả để đo thời gian chờ. Các trường hợp đã qua:

1. `529` một lần: chờ backoff (≤ 0,5 s ở lần đầu) rồi thành công trên cùng model.
2. `529` hai lần: sang model kế.
3. `429` có `enforced_spend_limit_reached`: ném `RateLimitError` ngay sau một request, không chờ, không fallback.
4. `429` với `retry-after: 30` lớn hơn `max_wait_s`: không chờ, sang model kế.
5. `429` với `retry-after: 2`: chờ đúng 2 s rồi retry.
6. `400`: ném ngay, không thử model khác.
7. `404`: sang model kế, không retry.
8. Refusal (HTTP 200): sang model kế; mọi model đều từ chối thì trả refusal cuối thay vì ném lỗi.
9. `max_attempts=3` với bốn lần lỗi xếp sẵn: chỉ ba request được gửi.
10. `retry-after` dài hơn phần deadline còn lại: không chờ, sang model kế.
11. Request có forced `tool_choice`: bỏ qua Opus 5.5, gửi thẳng Haiku 5.5.
12. Ghi đè `{"thinking": None, "max_tokens": 40}` chỉ áp cho model fallback; request tới model chính giữ `between_tools`.
13. Breaker: mở sau ngưỡng; sau cooldown chỉ cho một request thăm dò; thăm dò lỗi thì mở lại; `retry-after` dài hơn cooldown giữ breaker mở lâu bằng `retry-after`.
14. Breaker của model chính đang mở: request đi thẳng tới model kế.
15. Lỗi kết nối (`httpx2.ConnectError`): được xếp `RETRY`, hết lượt thì sang model kế.
16. Cùng model trên hai provider: provider đầu `529` làm breaker của nó mở, request đi tiếp sang provider thứ hai và breaker của provider thứ hai vẫn đóng.
17. `404` không tăng bộ đếm lỗi của breaker.
18. `409 conflict_error`: retry trên cùng model, như SDK.
19. `retry-after` dạng HTTP-date được đổi ra số giây; ngày đã qua, giá trị 0, âm, không hợp lệ hoặc `nan` bị bỏ qua.
20. `429` kèm `retry-after` là ngày đã qua: vẫn chờ theo backoff rồi mới thử lại, không gửi lại ngay.
21. `501`, `505`, `599`: retry trên cùng model, như SDK retry mọi `5xx`.
22. `x-should-retry` thắng quy tắc theo status: `503` kèm `false` dừng, `400` kèm `true` được retry; `404` vẫn sang model kế; spend cap vẫn dừng dù header là `true`.

### Những gì code này chưa làm

- **Streaming.** Hàm trên dùng request không stream. Với streaming, chỉ được retry hoặc fallback khi chưa gửi byte nào cho client; lỗi xảy ra sau đó phải được báo cho client như một lỗi, hoặc output phải được đệm lại.
- **Fallback credit** cho refusal. Khi chuỗi có model Claude hỗ trợ fallback phía server, đặt `fallbacks` trong request để refusal được xử lý trong cùng lời gọi và được bù chi phí cache [3]; chuỗi của bạn lo phần lỗi hạ tầng.
- **Provider khác.** Thêm provider nghĩa là thêm một `classify` cho exception của SDK đó và một bước dịch request.
- **Async.** Bản async cần `asyncio.sleep` và `anthropic.AsyncAnthropic`; logic giữ nguyên.
- **Ghim model cho hội thoại.** Hội thoại đã fallback sang model B nên ở lại B cho các lượt sau, thay vì quay về A ở mỗi lượt. Anthropic khuyên làm như vậy khi tự viết retry cho refusal [3]. Lưu model đang phục vụ vào state của hội thoại.

## 3.2 Giữ prompt, tool và schema tương thích giữa các model

### Bảng capability nằm trong code

Đừng hard-code một request template cho mọi model. Giữ một bảng capability theo model, dựng request từ bảng đó, và kiểm chuỗi fallback lúc khởi động. Một phần bảng cho các model Claude hiện tại, theo [1][6][9]:

| Model | `thinking` hợp lệ | Forced `tool_choice` | Sampling params | Priority Tier |
|---|---|---|---|---|
| Opus 5.5 | `adaptive` hoặc bỏ trống | Không | Không (API `2026-10-21`) | Không |
| Sonnet 5.5 | `adaptive`, `between_tools` | Không | Không | Không |
| Haiku 5.5 | `adaptive`, `disabled` | Có | Không | Không |
| Fable 5.1 | `adaptive` hoặc bỏ trống | Không | Không | Không |
| Opus 4.8 | `adaptive`, `disabled` | Có | Không | Có |
| Sonnet 5 | `adaptive`, `disabled` | Có | Không | Không |

Bảng này cũ đi theo từng đợt phát hành model. Đừng tin bảng hơn API: test CI ở mục dưới gửi request thật của từng model trong chuỗi, và lỗi `400` ở đó là tín hiệu bảng đã lỗi thời.

### Danh sách kiểm khi thêm một model vào chuỗi

1. **Tham số.** Chạy request thật (hoặc endpoint đếm token, endpoint này cũng kiểm `tool_choice` [1]) với từng model trong chuỗi, trong CI. Lỗi `400` phải lộ ra ở CI, không phải lúc sự cố.
2. **Forced tool use.** Request nào ép gọi tool thì chỉ đưa vào chuỗi các model hỗ trợ, hoặc đổi sang `tool_choice: auto` cộng `strict: true` cho mọi model [1].
3. **Thinking.** Dùng `adaptive` khi có thể; nó được nhận trên mọi model Claude 4.6 trở lên. Cấu hình đặc thù (`between_tools`, `disabled`) phải được ghi đè ở model fallback.
4. **`max_tokens` và chi phí.** Đếm token lại với model đích; tính lại ước lượng chi phí và `max_tokens` [6].
5. **Cache.** Đo cache hit rate theo model. Chấp nhận rằng fallback làm mất cache, hoặc dùng fallback credit khi lý do là refusal [4].
6. **Lịch sử append-only** khi có thinking block. Không đổi `system` hay `tools` theo model đích giữa hội thoại [1][7].
7. **Prompt.** Chạy eval của tác vụ trên model fallback như chạy trên model chính. Model fallback là model production, chỉ ít traffic hơn.
8. **Schema output.** Structured output và strict tool use có subset JSON Schema khác nhau giữa provider; xem [Structured output](../structured-output/).
9. **Tool format khác provider.** Xem [Tool use Phần 2](../tool-use/#phần-2--tool-use-theo-provider) và các lỗi tool sau khi đổi model ở [Tool use §3.11](../tool-use/#311-failure-modes-thường-gặp).

## 3.3 Circuit breaker và health check

Không có circuit breaker, mỗi request trong lúc provider chính sập vẫn thử model chính trước, chờ hết các lần retry, rồi mới sang fallback. Người dùng nào cũng chịu đủ độ trễ đó. Circuit breaker nhớ rằng model chính đang lỗi và cho request đi thẳng tới fallback trong một khoảng thời gian.

Ba trạng thái, như `Breaker` ở [§3.1](#31-thiết-kế-chuỗi-fallback):

- **Đóng**: cho mọi request qua, đếm lỗi liên tiếp.
- **Mở**: đạt ngưỡng thì chặn mọi request trong `cooldown_s`.
- **Nửa mở**: hết cooldown thì cho đúng một request thăm dò qua. Thành công thì đóng lại; lỗi thì mở tiếp.

Các lựa chọn thiết kế:

- **Lỗi nào được đếm.** Chỉ đếm lỗi nói về sức khoẻ của model (`RETRY` trong bảng). `400` hay `401` không nói gì về model; đếm chúng thì một request sai lặp lại có thể mở breaker cho mọi người.
- **`retry-after` đặt cooldown.** Server nói chờ 60 giây thì mở breaker ít nhất 60 giây, không thăm dò sớm hơn.
- **Đếm liên tiếp hay theo tỷ lệ.** Đếm lỗi liên tiếp đơn giản và phản ứng nhanh khi model sập hẳn. Lưu lượng lớn và lỗi lác đác thì tỷ lệ lỗi trong cửa sổ trượt ổn định hơn. LiteLLM mặc định đưa deployment vào trạng thái nguội khi quá 50% request lỗi trong phút hiện tại [12].
- **Breaker theo đơn vị nào.** Theo `(provider, model)` là tối thiểu. Claude bị rate limit theo bucket họ model [2], nên một breaker cho cả bucket phản ánh đúng hơn khi lỗi là `429`.
- **Chia sẻ trạng thái.** Breaker trong bộ nhớ của từng process thì mỗi worker tự học lại. Với vài chục worker, chấp nhận được. Với hàng trăm, đặt trạng thái vào Redis hoặc dùng gateway có cooldown dùng chung.

**Health check chủ động** (gửi request định kỳ để kiểm model) tốn tiền và đo sai thứ: model trả lời được "ping" không có nghĩa trả lời được request 100.000 token có tool của bạn. Status page của provider cũng vậy: nó báo sự cố diện rộng, không báo bucket rate limit của tổ chức bạn. Tín hiệu đáng tin nhất là traffic thật, tức breaker thụ động như trên, cộng metric tỷ lệ lỗi theo model trên dashboard ([LLM Observability](../llm-observability/)).

## 3.4 Quản lý quota nội bộ theo tenant

Rate limit của provider áp cho cả tổ chức. Một tenant chạy batch lớn có thể làm cạn bucket và mọi tenant khác nhận `429`. Fallback không cứu được nếu model fallback cũng rút từ bucket đã cạn ([§1.2](#12-rate-limit-rpm-tpm-và-cách-provider-tính)).

Công cụ có sẵn ở phía provider:

- **Claude workspaces**: đặt giới hạn RPM, ITPM, OTPM và chi tiêu riêng cho từng workspace, thấp hơn giới hạn tổ chức. Giới hạn tổ chức luôn áp dụng, kể cả khi tổng giới hạn các workspace lớn hơn. Không đặt spend limit được cho workspace mặc định [2]. Mỗi nhóm tenant hoặc mỗi tính năng một workspace và một API key là cách cô lập rẻ nhất. Header `anthropic-workspace-id` cho biết request đã tính vào workspace nào [2].
- **OpenAI projects**: giới hạn ở cấp project; hard spend limit theo project trả `429` khi chạm, project khác vẫn chạy trừ khi giới hạn tổ chức cũng đã chạm [10][11].
- **Cloudflare dynamic routes**: budget và rate limit theo user, project, team ngay trong gateway [17].

Ở phía ứng dụng, thêm một token bucket theo tenant trước khi gọi provider, với dung lượng là phần quota bạn chia cho tenant đó. Ước lượng input token trước khi gửi (endpoint đếm token hoặc ước lượng theo số ký tự), trừ vào bucket, và hoàn lại phần chênh sau khi có `usage` thực. Tenant hết quota nhận lỗi của bạn, có thông báo rõ ràng, không làm hỏng traffic của người khác. LiteLLM có `max_parallel_requests`, nhưng giới hạn đó đếm trong từng process và trả `429` ngay khi hết slot, không xếp hàng [12]; với nhiều worker, tổng song song thực bằng giới hạn nhân số worker.

Chi phí theo tenant và cảnh báo vượt ngân sách có ở [LLM Observability §3.3](../llm-observability/#33-theo-dõi-chi-phí-theo-user-feature-tenant) và [Cost Optimization](../cost-optimization/).

## 3.5 Đo và eval khi route sang model khác

### Eval cả hệ thống, không chỉ router

Router có độ chính xác phân loại 90% vẫn có thể làm hỏng sản phẩm nếu 10% sai đều rơi vào ca khó nhất. Đo thứ người dùng nhận được: chất lượng câu trả lời cuối, chi phí, độ trễ. Cách làm:

1. Lấy mẫu prompt từ traffic thật, giữ phân phối theo tác vụ ([Evaluation §3.1](../evaluation/#31-xây-golden-dataset-từ-đâu)).
2. Chạy cùng tập qua ba cấu hình: router, luôn dùng model mạnh nhất, và bảng tĩnh theo loại tác vụ.
3. Chấm bằng grader hoặc LLM judge đã hiệu chỉnh ([Evaluation §2.2](../evaluation/#22-llm-as-judge)), so theo cặp.
4. So chất lượng và tổng chi phí. Router đáng giữ khi rẻ hơn rõ rệt mà chất lượng không giảm vượt ngưỡng bạn chấp nhận.

Microsoft đề xuất đúng quy trình này cho model router của Azure, thêm một điều: mỗi lần chỉ đổi một đòn bẩy và giữ nguyên dataset, baseline, cấu hình ứng dụng [19].

### Ba cái bẫy khi đánh giá router

- **Nhiễu lớn hơn hiệu ứng.** Trong paper của Lee, 5,37% cặp model-câu hỏi đổi kết quả khi chạy lại y hệt [26]. Router "thắng" baseline một hai điểm sau một lần chạy chưa chứng minh được gì. Chạy lặp và báo khoảng tin cậy ([Evaluation §1.4](../evaluation/#14-phương-sai-chạy-bao-nhiêu-lần-cần-bao-nhiêu-mẫu)).
- **Baseline chọn trên chính dữ liệu kiểm tra.** False Floors (Bhatti và Vaddina, arXiv 2610.01535, tháng 10/2026) chỉ ra rằng một benchmark routing lớn chọn "model đơn lẻ tốt nhất" để so sánh ngay trên dữ liệu đánh giá [27]. Trong điều kiện của chính benchmark thì vô hại, nhưng khi phân phối thay đổi thì không. Trên HELM Safety, sai lệch do cách chọn này là 0,003–0,030 harm với split ngẫu nhiên và 0,045–0,113 với category giữ lại, tương đương toàn bộ phần thua kém được quy cho routing; trên AgentDojo, sai lệch tăng 7–9 lần khi giữ lại suite. Tác giả kết luận: "Scored honestly under shift, routing buys little on these benchmarks", và đề xuất đánh giá dưới distribution shift với baseline chọn mà không nhìn nhãn kiểm tra [27]. Ứng dụng của bạn: chọn baseline trên tập dev, đo trên tập test tách riêng, và có một tập test đến từ thời điểm khác hoặc nhóm khách hàng khác.
- **Router huấn luyện trên phân phối khác.** RouteLLM chỉ học từ Arena thì ngang ngẫu nhiên trên MMLU [23]. Router mua sẵn hay học từ dữ liệu công khai cần được đo trên traffic của bạn trước khi tin.

### Eval nhánh fallback

Tỷ lệ request thành công không nói được fallback có ổn hay không. Đo riêng nhánh fallback:

- **Tỷ lệ request được model fallback phục vụ**, theo giờ và theo model. Đọc từ trường `model` của response và từ `log` của vòng fallback. Với fallback phía server của Claude, đếm mục `fallback_message` trong `usage.iterations` [3].
- **Chất lượng trên nhánh fallback**: chạy eval định kỳ trên model fallback. Model ít được gọi là model dễ bị quên khi đổi prompt.
- **Chi phí thực của nhánh fallback**, gồm cache write mới và các lần thử thất bại. Với Claude, cộng `usage.iterations` thay vì đọc usage cấp trên cùng [3].
- **Độ trễ p95 và p99** khi có retry. Trung bình che mất người dùng chờ hết các lần backoff.
- **Refusal là tín hiệu riêng.** Refusal là HTTP 200, nên dashboard dựng trên tỷ lệ lỗi hay 5xx không bao giờ thấy nó. Anthropic khuyên phát một event cho mỗi refusal và một event cho mỗi response do fallback phục vụ, rồi cảnh báo khi chênh lệch giữa hai số tăng [3].

### Log gì cho mỗi request

Model được yêu cầu, model thực trả lời, `stop_reason`, số lần thử và lý do mỗi lần (chính là `log` của `call_with_fallback`), `request-id` của Claude (có trong header và trong body lỗi, cần khi liên hệ hỗ trợ) [1], các header rate limit còn lại, và thời gian chờ backoff. Không log `fallback_credit_token` cùng nội dung nhạy cảm vào nơi nhiều người đọc được.

## 3.6 Failure modes thường gặp

| # | Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|---|
| 1 | Fallback sang Opus 5.5 hoặc Sonnet 5.5 nhận `400 tool_choice: type "tool" and "any" are not supported for this model` | Model đích không hỗ trợ forced tool use [1] | Lọc chuỗi theo capability; hoặc dùng `auto` cộng `strict: true` cho mọi model |
| 2 | Model fallback nhận `400` về `thinking` | `between_tools`, `disabled` hay `budget_tokens` không hợp lệ ở model đích [1] | Ghi đè `thinking` theo model; dùng `adaptive` khi có thể |
| 3 | Mọi request Claude `400` sau khi nâng SDK hoặc gateway | API version `2026-10-21` từ chối `temperature`, `top_p`, `top_k` [9] | Bỏ ba tham số khỏi request và khỏi cấu hình của thư viện trung gian |
| 4 | Vẫn `429` liên tục dù đã fallback sang "model khác" | Hai model chung một bucket rate limit, ví dụ Opus 4.8 và Opus 4.7 [2] | Fallback sang bucket khác: họ khác, nền tảng khác, provider khác |
| 5 | Retry kéo dài, mọi lần đều `429`, không có `retry-after` | Hết spend cap của tháng (`enforced_spend_limit_reached`) [2] | Xếp `FATAL`; báo động; không retry, không fallback |
| 6 | "Hết hạn mức" nhưng lỗi là `400`, không phải `429` | Spend limit tự đặt thấp hơn trần [2] | Nhận diện riêng theo message; báo động như trường hợp 5 |
| 7 | Một request làm 20–30 request lên provider | Retry ở SDK, gateway và ứng dụng nhân với nhau [11] | Một tầng retry; `max_retries=0` ở SDK; trần cho cả chuỗi |
| 8 | Người dùng chờ hai phút rồi mới nhận lỗi | SDK chờ đúng `retry-after` dài, timeout mặc định 10 phút áp cho từng lần thử [1][29] | Deadline cho cả chuỗi; không chờ khi `retry-after` vượt ngưỡng |
| 9 | Refusal không được fallback | Code chỉ bắt exception; refusal là HTTP 200 [3] | Kiểm `stop_reason` ở mọi response; bật `fallbacks` phía server cho model hỗ trợ |
| 10 | Đã bật `fallbacks` nhưng `529` vẫn đến tay người dùng | Fallback phía server của Claude chỉ xử lý refusal [3] | Thêm chuỗi fallback cho lỗi hạ tầng |
| 11 | Đổi `base_url` sang OpenRouter, fallback xảy ra cả khi bị rate limit | `fallbacks` của OpenRouter kích hoạt trên mọi lỗi [13] | Đọc lại ngữ nghĩa fallback mỗi khi đổi nền tảng |
| 12 | `400` do request sai vẫn bị gửi sang model khác | Portkey mặc định fallback với mọi non-2xx [15] | Đặt `on_status_codes` tường minh |
| 13 | Chi phí tăng mạnh trong giờ có sự cố | Cache tách theo model; fallback ghi lại toàn bộ prefix vào cache mới [4] | Chấp nhận và theo dõi; với refusal dùng fallback credit |
| 14 | Chi phí báo cáo thấp hơn hoá đơn | Đọc `usage` cấp trên cùng; bỏ sót lần thử bị từ chối trong `usage.iterations` [3] | Cộng `usage.iterations` |
| 15 | Output bị cắt sau fallback, `stop_reason: "max_tokens"` | Tokenizer khác; thinking tính vào `max_tokens` [6] | Đếm lại token với model đích; tăng `max_tokens` |
| 16 | Model fallback trả lời như "quên" các lượt trước | Thinking block của model cũ không đọc được ở model mới và bị bỏ lặng lẽ [7] | Ghim hội thoại vào một model; chấp nhận mất reasoning khi buộc phải chuyển |
| 17 | Mất reasoning khi chuyển sang key của tài khoản khác dù cùng model | Block của Sonnet 5.5 và Haiku 5.5 ràng buộc theo tài khoản [7] | Replay hội thoại qua đúng tài khoản đã tạo ra nó |
| 18 | `400 Invalid signature in thinking block` sau fallback | Đổi `system` hoặc `tools` theo model đích giữa hội thoại [1] | Lịch sử append-only; hoặc `prefix_mismatch_behavior: "drop_block"` |
| 19 | Router gửi gần hết traffic sang model mạnh | Ngưỡng quá thấp, hoặc chế độ Quality [19][23] | Đo phân bố trường `model`; chỉnh ngưỡng trên dữ liệu thật |
| 20 | Chất lượng giảm ở một nhóm khách hàng sau khi bật router | Router không thấy nhóm đó khi huấn luyện; Bedrock routing chỉ tối ưu cho tiếng Anh [20][23] | Eval theo nhóm; giữ baseline ngoài router |
| 21 | Model thay đổi giữa các lượt của cùng hội thoại | Router stateless chọn lại mỗi request [18] | Session affinity hoặc ghim model trong state hội thoại |
| 22 | Chuỗi fallback trỏ tới model đã retire, sự cố đầu tiên mới lộ | Model fallback ít được gọi, `404` không ai thấy | Test chuỗi trong CI; alert khi có `404` |
| 23 | Breaker mở vì người dùng gửi request sai | Đếm `400` là lỗi của model | Chỉ đếm lỗi `RETRY` |
| 24 | Một worker vẫn dồn request vào model đã sập | Breaker trong bộ nhớ từng process | Chấp nhận với ít worker; nhiều worker thì dùng trạng thái chung |
| 25 | Tool call hoặc đoạn text bị lặp sau khi stream lỗi giữa chừng | Retry hoặc fallback sau khi đã gửi token cho client | Chỉ fallback khi chưa gửi gì; nếu không thì báo lỗi hoặc đệm output |

## 3.7 Checklist

**Phân loại lỗi**

- [ ] Ba nhóm `RETRY`, `NEXT`, `FATAL`, phân theo status code **và** mã lỗi trong body.
- [ ] Hết spend cap và spend limit tự đặt được nhận diện riêng, có alert.
- [ ] `stop_reason` được kiểm ở mọi response: `refusal`, `max_tokens`, `model_context_window_exceeded`, `pause_turn` có nhánh riêng.

**Retry**

- [ ] Chỉ một tầng retry; SDK `max_retries=0` khi ứng dụng hoặc gateway retry.
- [ ] Backoff có jitter; tôn trọng `retry-after`; không chờ khi `retry-after` vượt ngưỡng hoặc vượt deadline.
- [ ] Trần số lần thử và deadline cho cả chuỗi; timeout từng lần thử lấy từ phần deadline còn lại.
- [ ] Test với mock lỗi đếm được tổng số request thực gửi đi.

**Chuỗi fallback**

- [ ] Model fallback nằm ở bucket rate limit khác model chính.
- [ ] Bảng capability theo model; tham số ghi đè theo model; request với mỗi model trong chuỗi được chạy trong CI.
- [ ] Không còn `temperature`, `top_p`, `top_k` trong request Claude trên API version `2026-10-21`, kể cả trong cấu hình gateway.
- [ ] Hội thoại đã fallback được ghim vào model fallback.
- [ ] Streaming chỉ fallback khi chưa gửi output cho client.
- [ ] Model Claude hỗ trợ thì bật `fallbacks` phía server cho refusal; biết nó không xử lý lỗi hạ tầng.

**Vận hành**

- [ ] Circuit breaker theo model hoặc theo bucket, chỉ đếm lỗi hạ tầng, cooldown không ngắn hơn `retry-after`.
- [ ] Quota theo tenant: workspace hoặc project riêng ở provider, token bucket ở ứng dụng.
- [ ] Log model được yêu cầu, model thực trả lời, `stop_reason`, các lần thử, `request-id`.
- [ ] Dashboard: tỷ lệ fallback theo model, refusal tách khỏi lỗi, chi phí cộng cả `usage.iterations`, p95 và p99 độ trễ.

**Router**

- [ ] Đã đo bảng tĩnh theo loại tác vụ, và một model mạnh với effort thấp, trước khi dựng router học được.
- [ ] Eval cả hệ thống trên traffic thật, chạy lặp, baseline chọn trên dev và đo trên test tách riêng.
- [ ] Theo dõi phân bố model theo ngày; eval lại khi provider hoặc gateway đổi pool model.

---

## Nguồn tham khảo

Truy cập tháng 11/2026.

| # | Nguồn | Nội dung dùng trong bài |
|---|---|---|
| [1] | [Claude API: Errors](https://platform.claude.com/docs/en/api/errors) | Mã lỗi, retry mặc định của SDK, `request-id`, các lỗi `400` theo model (thinking, forced tool use, prefill, thinking signature) |
| [2] | [Claude API: Rate limits](https://platform.claude.com/docs/en/api/rate-limits) | RPM/ITPM/OTPM, token bucket, bucket chung trong họ, spend cap, spend limit tự đặt, workspace limits, header |
| [3] | [Claude: Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback) | Refusal, tính tiền refusal, `fallbacks` phía server, middleware SDK, sticky routing, streaming, batch, các bẫy |
| [4] | [Claude: Fallback credit](https://platform.claude.com/docs/en/build-with-claude/fallback-credit) | "Prompt caches are per-model", credit token, trường phải giữ nguyên, model đích được phép |
| [5] | [Claude: Stop reasons and fallback](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons) | Các giá trị `stop_reason` |
| [6] | [Claude Haiku 5.5 migration guide](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide) | Tokenizer, thinking và `max_tokens`, prefill, computer use, Priority Tier, phong cách trả lời |
| [7] | [Claude: Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking) | Model nào đọc được thinking block của model nào, ràng buộc theo tài khoản |
| [8] | [Claude: Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) | Ngưỡng tối thiểu theo model, giá cache read và write |
| [9] | [Claude API: Migrating to API version 2026-10-21](https://platform.claude.com/docs/en/api/migration-2026-10-21) | Bỏ `temperature`, `top_p`, `top_k` |
| [10] | [OpenAI: Error codes](https://developers.openai.com/api/docs/guides/error-codes) | `503 server_is_overloaded`, các mã billing và spend limit, "won't restore API access" |
| [11] | [OpenAI: Rate limits](https://developers.openai.com/api/docs/guides/rate-limits) | Các chiều giới hạn, TPM theo `max_tokens`, `slow_down`, `Retry-After`, retry lồng nhau, Scale Tier và Reserved Tier |
| [12] | [LiteLLM: Router](https://docs.litellm.ai/docs/routing) | Chiến lược, `order`, fallback, cooldown mặc định, `num_retries`, `max_parallel_requests`, session affinity |
| [13] | [OpenRouter: Model fallbacks](https://openrouter.ai/docs/guides/routing/model-fallbacks) | `models`, lỗi kích hoạt, tính tiền theo model thực, `fallbacks` trên endpoint Messages |
| [14] | [OpenRouter: Auto Router](https://openrouter.ai/docs/guides/routing/routers/auto-router) | Phân loại tác vụ, xếp hạng theo chi tiêu 7 ngày, bộ lọc model |
| [15] | [Portkey: Fallbacks](https://portkey.ai/docs/product/ai-gateway/fallbacks) | `strategy.mode`, `on_status_codes`, mặc định non-2xx |
| [16] | [Cloudflare AI Gateway: Fallbacks](https://developers.cloudflare.com/ai-gateway/configuration/fallbacks/) | Universal endpoint, `cf-aig-step` |
| [17] | [Cloudflare AI Gateway: Dynamic routes](https://developers.cloudflare.com/ai-gateway/features/routing/dynamic-routing/) | Route có quota, ngân sách, A/B |
| [18] | [Microsoft Foundry: How to use model router](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/model-router) | Chế độ, tập model, session affinity, tham số reasoning, chi phí |
| [19] | [Microsoft Foundry: How model router works](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-router-how-it-works) | ML model nhẹ, ba chế độ, tối thiểu hai model cho failover, đổi một đòn bẩy mỗi lần |
| [20] | [Amazon Bedrock: Intelligent prompt routing](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html) | Hai model cùng họ, `responseQualityDifference`, chỉ tối ưu tiếng Anh |
| [21] | [Amazon Bedrock: Cross-Region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) | Profile theo vùng và global, khoảng 10% rẻ hơn, không hỗ trợ Provisioned Throughput, khuyến nghị fallback |
| [22] | [Gemini API: Troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting) | Lỗi nên retry, backoff có jitter, retry mặc định của Python SDK |
| [22b] | [Vertex AI: Locations](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations) | Global endpoint |
| [23] | [Ong et al., RouteLLM (arXiv 2406.18665, v4)](https://arxiv.org/html/2406.18665v4) | Dữ liệu Arena, kết quả MT-Bench, MMLU, GSM8K, phân tích chi phí |
| [24] | [Chen, Zaharia, Zou, FrugalGPT (arXiv 2305.05176)](https://arxiv.org/abs/2305.05176) | Cascade, "up to 98% cost reduction" |
| [25] | [Hu et al., RouterBench (arXiv 2403.12031)](https://arxiv.org/abs/2403.12031) | Bộ hơn 405.000 kết quả suy luận |
| [26] | [Lee, Most of the LLM routing gap is task type (arXiv 2608.23023)](https://arxiv.org/abs/2608.23023) | Bảng tĩnh theo loại tác vụ, nhiễu giữa hai lần chạy, không có holdout |
| [27] | [Bhatti và Vaddina, False Floors (arXiv 2610.01535)](https://arxiv.org/abs/2610.01535) | Baseline chọn trên dữ liệu kiểm tra, đánh giá dưới distribution shift |
| [28] | [LiteLLM issue #19985](https://github.com/BerriAI/litellm/issues/19985) | Fallback khởi động lại vòng retry; đóng ngày 2/2/2026 |
| [29] | [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) | Bản 2.6.0: `_base_client.py` (retry, `retry-after`), `lib/middleware/_fallbacks.py` |
| [30] | [Claude API: Service tiers](https://platform.claude.com/docs/en/api/service-tiers) | Priority Tier, `service_tier`, model không hỗ trợ, batch |
