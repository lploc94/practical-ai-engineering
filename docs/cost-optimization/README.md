# Tối ưu chi phí cho ứng dụng LLM

Tháng đầu, hoá đơn API vài trăm đô. Ba tháng sau, số người dùng gấp đôi còn hoá đơn gấp sáu. Không ai đổi model. Có ba thay đổi nhỏ: tính năng chat cho phép hội thoại dài hơn, mỗi lượt gửi lại toàn bộ lịch sử. System prompt có thêm dòng "Thời gian hiện tại: ...", nên cache không còn khớp. Một job tóm tắt ban đêm chạy qua API đồng bộ, dù không ai cần kết quả trong vài giây.

Chi phí LLM tăng theo số token, không theo số người dùng. Phần lớn chênh lệch nằm ở cách ứng dụng gửi token. Bài này đi theo thứ tự nên làm:

- **Nền tảng — Phần 1:** tiền đi vào đâu (input, output, cache, reasoning, tool), đo theo request, user và feature, vì sao output đắt hơn input, và đánh đổi giữa chi phí, chất lượng, độ trễ.
- **Đòn bẩy theo provider — Phần 2:** bảng giá và cách đếm token, prompt caching, Batch API và các tier rẻ hơn, model nhỏ và fine-tune, khi nào self-host rẻ hơn API.
- **Thiết kế và vận hành — Phần 3:** chọn model, rút ngắn context, giới hạn output và reasoning, cache kết quả, hạn mức theo user, dự báo chi phí, failure modes và checklist.

> Bản tháng 11/2026. Giá trong bài lấy từ trang giá chính thức của từng provider vào ngày 06/11/2026 và đổi thường xuyên; trước khi dùng để tính toán, mở lại trang giá. Code trong bài đã chạy với `anthropic` 2.6.0 (Python) bằng dữ liệu `usage` dựng tay, không gọi API thật. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

Tài liệu liên quan: [Multi-turn](../multi-turn-conversation/) (prompt caching chi tiết theo provider), [Model routing](../model-routing/) (chọn model theo request, chi phí của fallback), [Reasoning models](../reasoning-models/#35-chi-phí-và-độ-trễ-max_tokens-timeout-ước-lượng-ngân-sách) (thinking token và ngân sách), [Multimodal](../multimodal/#12-ảnh-tốn-bao-nhiêu-token) (token của ảnh), [LLM Observability §3.3](../llm-observability/#33-theo-dõi-chi-phí-theo-user-feature-tenant) (theo dõi chi phí theo tenant), [Agents §3.7](../agents/#37-chi-phí-và-độ-trễ), [Evaluation](../evaluation/).

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Chi phí đến từ đâu: input, output, cache, reasoning, tool](#11-chi-phí-đến-từ-đâu-input-output-cache-reasoning-tool)
  - [1.2 Đo trước khi tối ưu: chi phí theo request, user, feature](#12-đo-trước-khi-tối-ưu-chi-phí-theo-request-user-feature)
  - [1.3 Vì sao output token thường đắt hơn input token](#13-vì-sao-output-token-thường-đắt-hơn-input-token)
  - [1.4 Đánh đổi chi phí, chất lượng và độ trễ](#14-đánh-đổi-chi-phí-chất-lượng-và-độ-trễ)
- [Phần 2 — Đòn bẩy theo provider](#phần-2--đòn-bẩy-theo-provider)
  - [2.1 Bảng giá và cách tính token theo provider](#21-bảng-giá-và-cách-tính-token-theo-provider)
  - [2.2 Prompt caching](#22-prompt-caching)
  - [2.3 Batch API](#23-batch-api)
  - [2.4 Model nhỏ, distillation và fine-tune](#24-model-nhỏ-distillation-và-fine-tune)
  - [2.5 Self-host: khi nào rẻ hơn API](#25-self-host-khi-nào-rẻ-hơn-api)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Chọn model nhỏ nhất đạt yêu cầu](#31-chọn-model-nhỏ-nhất-đạt-yêu-cầu)
  - [3.2 Rút ngắn prompt và context](#32-rút-ngắn-prompt-và-context)
  - [3.3 Giới hạn output và reasoning](#33-giới-hạn-output-và-reasoning)
  - [3.4 Semantic cache và cache kết quả](#34-semantic-cache-và-cache-kết-quả)
  - [3.5 Giới hạn chi phí theo user và chống lạm dụng](#35-giới-hạn-chi-phí-theo-user-và-chống-lạm-dụng)
  - [3.6 Dự báo chi phí khi scale](#36-dự-báo-chi-phí-khi-scale)
  - [3.7 Failure modes thường gặp](#37-failure-modes-thường-gặp)
  - [3.8 Checklist](#38-checklist)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

### Đòn bẩy theo thứ tự nên thử

Nhóm đầu giảm chi phí mà không đổi chất lượng; nhóm sau đổi chi phí lấy chất lượng, cần eval trước khi áp dụng. Số liệu là đo của Anthropic trên workload của họ [2], hoặc mức giảm theo bảng giá; workload của bạn sẽ cho số khác.

| Đòn bẩy | Mức giảm đã công bố | Đánh đổi | Mục |
|---|---|---|---|
| Prompt caching | Chi phí giảm 4,8–7,2 lần trên một agent nghiên cứu, 83–86% trên một agent phân loại bug [2] | Không đổi chất lượng; phải giữ prefix ổn định | [§2.2](#22-prompt-caching) |
| Batch API | 50% ở Claude, OpenAI và Gemini; Claude và OpenAI giảm cả cache, Gemini tính cache hit theo giá cache thường [1][9][18] | Kết quả có trong tối đa 24 giờ | [§2.3](#23-batch-api) |
| Bỏ token thừa trong input (ảnh quá lớn, tool không dùng, bảng dữ liệu dán thẳng) | Thêm 21–26% trên agent phân loại bug; dữ liệu qua code execution thay vì dán vào prompt giảm 92% và trả lời đúng 25/25 thay vì 7/25 [2] | Model có thể phải gọi thêm tool để lấy thứ đã bị bỏ | [§3.2](#32-rút-ngắn-prompt-và-context) |
| Rà prompt cũ khi đổi model | 14% ở cả hai lần nâng model được đo [2] | Không | [§3.2](#32-rút-ngắn-prompt-và-context) |
| Yêu cầu câu trả lời ngắn hơn | Khoảng 13%, từ 39% ít output token hơn [2] | Không, nếu format mới vẫn đủ cho người đọc | [§3.3](#33-giới-hạn-output-và-reasoning) |
| Giảm effort / reasoning | Trên tác vụ code dài, `medium` rẻ hơn `high` khoảng 30–52%, `low` rẻ hơn khoảng hai phần ba [2] | Mất tới 34 điểm tuỳ tác vụ và model | [§3.3](#33-giới-hạn-output-và-reasoning) |
| Chạy rẻ trước, chạy lại phần lỗi ở mức cao | Khoảng 40% rẻ hơn chạy hết ở `high`, cùng tỉ lệ pass [2] | Cần tín hiệu lỗi (test, validator); phần lỗi chậm gấp đôi | [§3.1](#31-chọn-model-nhỏ-nhất-đạt-yêu-cầu) |
| Đổi model | Tuỳ hướng: nâng lên model mới cùng hạng thường rẻ hơn trên mỗi tác vụ hoàn thành [2]; hạ xuống model nhỏ rẻ hơn theo token nhưng có thể đắt hơn theo tác vụ | Chất lượng | [§3.1](#31-chọn-model-nhỏ-nhất-đạt-yêu-cầu) |
| Cache kết quả (exact hoặc semantic) | Tới 86% trên một chatbot trong benchmark của AWS [23] | Semantic cache trả lời sai khoảng 8–9% số lần hit ngay cả ở ngưỡng chặt nhất được đo [23] | [§3.4](#34-semantic-cache-và-cache-kết-quả) |

Đòn bẩy không có trong bảng: **fast mode** và các tier ưu tiên. Chúng mua tốc độ bằng giá cao hơn (Opus 5.5 fast mode gấp đôi giá thường [1]), không bao giờ là đòn bẩy giảm chi phí.

### Công thức chi phí một request

Với Claude, mỗi response có năm số token, mỗi số một đơn giá [1][2]:

```
chi phí = input_tokens            × giá input
        + cache write TTL 5 phút  × giá input × 1,25
        + cache write TTL 1 giờ   × giá input × 2
        + cache_read_input_tokens × giá đọc cache
        + output_tokens           × giá output      (gồm cả thinking token)
```

Batch API nhân tất cả với 0,5. Fast mode thay giá input, output bằng giá fast mode, các hệ số cache vẫn áp lên trên. Inference chỉ chạy ở Mỹ (`inference_geo: "us"`) nhân 1,1 trên Claude 4.6 trở lên [1]. Haiku 5.5 có hai bảng giá theo độ dài prompt ([§2.1](#21-bảng-giá-và-cách-tính-token-theo-provider)). OpenAI và Gemini có cấu trúc tương tự, khác hệ số.

### Chi phí một feature

```
chi phí / tháng = số tác vụ / tháng × chi phí trung bình mỗi tác vụ
chi phí mỗi tác vụ = Σ chi phí các request của tác vụ (mọi lượt, mọi tool call, mọi retry)
```

Đơn vị là **tác vụ hoàn thành**, không phải request hay token. Một model rẻ hơn mỗi token mà cần thêm lượt, thêm retry, hoặc làm hỏng việc thì không rẻ hơn [2].

---

# Phần 1 — Nền tảng

## 1.1 Chi phí đến từ đâu: input, output, cache, reasoning, tool

Một request LLM bị tính tiền theo nhiều dòng, và dòng lớn nhất thường không phải dòng bạn nghĩ.

**Input token** gồm mọi thứ gửi lên: system prompt, định nghĩa tool, lịch sử hội thoại, tài liệu, ảnh. API LLM không giữ trạng thái giữa các request, nên hội thoại 20 lượt gửi lượt đầu 20 lần. Anthropic viết: một tác vụ 40 lượt gửi lượt đầu 40 lần, nên chi phí tăng gần theo bình phương số lượt [2]. Đây là lý do hội thoại dài và agent loop đắt hơn nhiều so với dự tính khi nhìn một request đơn lẻ.

**Output token** là token model sinh ra, đắt gấp nhiều lần input: gấp 5 lần ở các model Claude hiện tại [2], gấp 5 lần ở GPT-6 Sol và GPT-6.1 Sol [9], gấp 5 lần ở Gemini 4 Argon và Gemini 3.8 Flash [16]. Trong agent loop, mỗi token model viết ra còn quay lại làm input ở mọi lượt sau [2].

**Reasoning token** (thinking) được tính như output token, kể cả khi bạn không thấy nội dung của chúng. OpenAI ghi rõ reasoning token "are billed as output tokens" [14]; bảng giá Gemini ghi giá output "including thinking tokens" [16]; với Claude, `output_tokens` là tổng chính thức dùng để tính tiền, gồm phần thinking [26]. Request dừng vì hết `max_output_tokens` có thể đã trả tiền cho input và reasoning mà không có chữ nào hiển thị [14]. Chi tiết ở [Reasoning models §1.2](../reasoning-models/#12-thinking-token-sinh-ra-tính-phí-nhưng-không-phải-lúc-nào-cũng-thấy).

**Cache** có hai dòng. Ghi cache đắt hơn input thường (Claude: 1,25 lần với TTL 5 phút, 2 lần với TTL 1 giờ; OpenAI GPT-5.6 trở lên: 1,25 lần với TTL 30 phút, 2,5 lần với TTL 12 giờ). Đọc cache rẻ hơn nhiều: 0,1 lần giá input ở phần lớn model, 0,05 lần ở Opus 5.5, Sonnet 5.5, GPT-6.1 Sol, GPT-6.1 Terra, và 0,025 lần ở Fable 5.1 [1][10]. Gemini thu thêm phí lưu trữ theo giờ cho cache tường minh [16]. Chi tiết ở [§2.2](#22-prompt-caching).

**Tool** tốn tiền theo ba cách [1]:

- Định nghĩa tool là input token trên mọi request. Claude còn thêm một system prompt ẩn khi request có tool: 286 token với Opus 5.5, Sonnet 5.5, Haiku 5.5 ở `tool_choice` `auto`; 406 token khi ép gọi tool trên Haiku 5.5. Toolset computer use thêm khoảng 4.500 token, browser use khoảng 6.600 token [1].
- Kết quả tool quay lại làm input, và ở lại trong lịch sử cho các lượt sau.
- Tool phía server có phí riêng: web search của Claude là 10 USD cho 1.000 lượt tìm; kết quả tìm được tính là input token ở lượt đó và các lượt sau [1]. Grounding với Google Search của Gemini miễn phí 5.000 request mỗi tháng, sau đó 14 USD cho 1.000 request [16].

**Phụ phí khác.** Fast mode (Claude Opus 5.5: 8 USD / 40 USD mỗi triệu token input/output thay cho 4/20 [1]; OpenAI Fast mode cho GPT-5.6 Sol gấp đôi giá thường [13]). Lưu trú dữ liệu: Claude nhân 1,1 khi chạy riêng ở Mỹ [1], OpenAI cộng 10% cho endpoint xử lý theo vùng với model ra mắt từ 05/03/2026 [9]. Refusal: một số refusal của Claude trước khi có output vẫn tính tiền ([Model routing §1.3](../model-routing/#13-retry-backoff-và-jitter)).

## 1.2 Đo trước khi tối ưu: chi phí theo request, user, feature

Không đo thì không biết đòn bẩy nào đáng làm. Một ứng dụng mà 80% chi phí là output của một tính năng sinh báo cáo sẽ không được lợi gì nhiều từ việc rút ngắn system prompt.

### Ba nguồn số liệu

1. **Trường `usage` trong mỗi response.** Nguồn chính xác nhất và miễn phí. Log nó cùng với user, tenant, feature và model cho mọi request. Với Claude, log đủ: `input_tokens`, `cache_creation_input_tokens` (và phần tách `cache_creation.ephemeral_5m_input_tokens`, `ephemeral_1h_input_tokens`), `cache_read_input_tokens`, `output_tokens`, `speed`, `service_tier` [26]. Khi response có `usage.iterations` (compaction, advisor tool, fallback phía server), một phần token chỉ nằm trong `iterations`, không nằm trong số tổng; phải cộng các mục trong `iterations` [2][26].
2. **API báo cáo của provider.** Claude có Usage and Cost API (cần Admin API key, key của workspace không dùng được); dữ liệu thường có sau khoảng 5 phút; polling liên tục nên ở mức mỗi phút một lần, nhanh hơn chỉ cho các đợt ngắn như tải dữ liệu nhiều trang [7]. Dùng để đối soát với log của bạn và để thấy traffic không qua ứng dụng (thử nghiệm, script).
3. **Hoá đơn.** Chậm nhất, gộp nhất. Dùng để kiểm tra hai nguồn trên có khớp không.

### Hàm tính chi phí từ `usage`

Hàm dưới đây tính chi phí token của một response Claude, gồm cache theo TTL, giá prompt dài của Haiku 5.5, fast mode, batch và `iterations`. Bảng giá chép từ trang giá Claude ngày 06/11/2026 [1]; dùng `Decimal` để cộng dồn hàng triệu request không lệch.

```python
from dataclasses import dataclass
from decimal import Decimal

M = Decimal(1_000_000)


@dataclass(frozen=True)
class Price:
    """Giá USD cho 1 triệu token. Hệ số cache theo bảng giá Claude, tháng 11/2026."""
    input: Decimal
    output: Decimal
    cache_read: Decimal                      # giá đọc cache, khác nhau theo model
    write_5m: Decimal = Decimal("1.25")      # hệ số so với input
    write_1h: Decimal = Decimal("2")
    long_threshold: int | None = None        # Haiku 5.5: prompt > 100.000 token
    long: "Price | None" = None              # bảng giá khi vượt ngưỡng


PRICES = {
    "claude-opus-5-5":   Price(Decimal("4"),   Decimal("20"),  Decimal("0.20")),
    "claude-sonnet-5-5": Price(Decimal("2"),   Decimal("10"),  Decimal("0.10")),
    "claude-fable-5-1":  Price(Decimal("10"),  Decimal("50"),  Decimal("0.25")),
    "claude-opus-4-8":   Price(Decimal("5"),   Decimal("25"),  Decimal("0.50")),
    "claude-haiku-5-5":  Price(Decimal("0.10"), Decimal("0.50"), Decimal("0.01"),
                               long_threshold=100_000,
                               long=Price(Decimal("0.50"), Decimal("2.50"), Decimal("0.05"))),
}
FAST = {"claude-opus-5-5": (Decimal("8"), Decimal("40")), "claude-opus-4-8": (Decimal("10"), Decimal("50"))}   # input, output khi speed="fast"


def _get(obj, name, default=0):
    v = obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    return default if v is None else v


def _entry_cost(u, model: str, *, batch: bool, fast: bool) -> Decimal:
    p = PRICES[model]
    inp = _get(u, "input_tokens")
    read = _get(u, "cache_read_input_tokens")
    cc = _get(u, "cache_creation", None)
    if cc is not None:
        w5, w1 = _get(cc, "ephemeral_5m_input_tokens"), _get(cc, "ephemeral_1h_input_tokens")
    else:                                    # không có breakdown: coi là TTL 5 phút
        w5, w1 = _get(u, "cache_creation_input_tokens"), 0
    if p.long_threshold and inp + read + w5 + w1 > p.long_threshold:
        p = p.long                           # cả request tính giá cao, kể cả output
    base_in, base_out = p.input, p.output
    if fast:
        base_in, base_out = FAST[model]      # hệ số cache áp lên giá fast mode
        read_price = base_in * p.cache_read / p.input
    else:
        read_price = p.cache_read
    cost = (inp * base_in + w5 * base_in * p.write_5m + w1 * base_in * p.write_1h
            + read * read_price + _get(u, "output_tokens") * base_out) / M
    return cost / 2 if batch else cost       # Batch API: giảm 50% mọi loại token


def request_cost(usage, model: str, *, batch: bool = False) -> Decimal:
    """Chi phí token của một response. Có `iterations` thì cộng từng mục: compaction,
    advisor và các lần thử bị refusal chỉ nằm ở đó, không nằm trong số tổng."""
    fast = _get(usage, "speed", None) == "fast"
    iterations = _get(usage, "iterations", None)
    if not iterations:
        return _entry_cost(usage, model, batch=batch, fast=fast)
    return sum((_entry_cost(it, _get(it, "model", None) or model, batch=batch, fast=fast)
                for it in iterations), Decimal(0))
```

Hàm nhận object `Usage` của SDK hoặc dict đọc từ log JSON. Test đã chạy (`anthropic` 2.6.0, `Usage` và `BetaUsage` thật của SDK, dữ liệu dựng tay):

1. Ví dụ trong trang giá Claude: 10.000 input, 40.000 cache read, 15.000 output trên Opus 5.5 ra đúng 0,348 USD [1].
2. Ghi 100.000 token cache TTL 1 giờ trên Opus 5.5 ra 0,80 USD; đọc 100.000 token ra 0,02 USD, khớp hai con số trong trang tối ưu chi phí của Anthropic [2].
3. Cache write không có breakdown được tính là TTL 5 phút (1,25 lần).
4. Haiku 5.5: prompt 110.000 token (gồm cache read) tính giá prompt dài cho cả input lẫn output; prompt đúng 100.000 token tính giá thường.
5. Batch ra một nửa.
6. Fast mode Opus 5.5: 8 USD input, 40 USD output, cache read 0,05 lần giá fast.
7. Response có `iterations` (một lần thử trên Fable 5.1, một `fallback_message` trên Opus 4.8): cộng từng mục theo model của mục đó.
8. Dict thô từ log chạy được như object SDK.

Hàm chỉ tính token. Phí web search, code execution theo giờ container, session runtime của Managed Agents là dòng riêng [1]. `usage.server_tool_use` chỉ đếm số lượt web search và web fetch; code execution tính theo thời gian container và chỉ hiện trong cost endpoint của Usage and Cost API, không có trong usage endpoint [1][7]. Model không có trong `PRICES` sẽ ném `KeyError`, có chủ ý: tính sai giá lặng lẽ tệ hơn báo lỗi.

### Gắn chi phí vào đúng chỗ

Tính chi phí theo request chưa đủ. Cần gộp được theo:

- **Feature** (endpoint, job, tính năng): biết tính năng nào đáng tối ưu.
- **User và tenant**: biết ai dùng nhiều, phát hiện lạm dụng, tính giá bán.
- **Tác vụ**: một tác vụ agent gồm nhiều request; chi phí tác vụ là tổng của chúng, kể cả retry và lần thử thất bại.
- **Model và phiên bản prompt**: biết một thay đổi làm chi phí tăng hay giảm.

Trace và attribute để làm việc này ở [LLM Observability §3.3](../llm-observability/#33-theo-dõi-chi-phí-theo-user-feature-tenant). Với Claude, tách API key hoặc workspace theo feature hay nhóm khách hàng giúp đọc được chi phí thật trong Usage and Cost API theo từng phần; một key dùng chung cho nhiều dự án làm số liệu không quy được cho dự án nào [7].

### Đọc profile

Khi đã có số, bốn câu hỏi quyết định đòn bẩy nào đáng làm:

- **Tỉ lệ input đọc từ cache là bao nhiêu?** Agent loop trên traffic thật của khách hàng Anthropic đọc trung vị 84% input từ cache, nhóm 10% tốt nhất từ 94% trở lên; dưới khoảng 80% thì có thứ gì đó đang làm vỡ cache [2].
- **Output chiếm bao nhiêu phần chi phí?** Nếu lớn, đòn bẩy là độ dài câu trả lời và effort, không phải prompt.
- **Có bao nhiêu traffic không ai chờ?** Đó là ứng viên cho batch.
- **Chi phí phân bố thế nào giữa các tác vụ?** Trên một lượt chạy 20 bài WideSearch, hai bài chiếm 43% chi phí [2]. Tối ưu đuôi phân phối thường được nhiều hơn tối ưu trường hợp điển hình.

## 1.3 Vì sao output token thường đắt hơn input token

Model xử lý input và output theo hai cách khác nhau:

- **Prefill** (xử lý input): mọi token của prompt được đưa qua model cùng lúc, trong một lần đọc trọng số. GPU làm nhiều phép tính trên mỗi byte đọc từ bộ nhớ, nên giới hạn là sức tính (compute-bound) [19].
- **Decode** (sinh output): model sinh từng token một, mỗi token cần một lượt đọc toàn bộ trọng số (và KV cache) từ bộ nhớ, chỉ để làm một ít phép tính. Giới hạn là băng thông bộ nhớ (memory-bound) [19].

Ví dụ tính trong bài của Hugging Face về TGI: model 7B, 16-bit, trên A100, batch 4, prompt 512 token. Phần tính toán của prefill mất khoảng 91,9 ms cho cả 2.048 token; mỗi bước decode chỉ cần 0,17 ms tính toán nhưng mất 9,3 ms để đọc trọng số [19]. Một token output tốn GPU-thời gian nhiều hơn hẳn một token input, và provider định giá theo đó.

Ba hệ quả thực tế:

1. **Cắt output có lợi hơn cắt input cùng số token.** Ở Claude hiện tại, 1 token output bằng 5 token input [2].
2. **Output tốn tiền nhiều lần trong agent loop.** Câu trả lời dài ở lượt 3 là input ở lượt 4, 5, 6... Nếu không có cache, trả tiền nhiều lần; có cache thì ít hơn, nhưng vẫn là cache write lần đầu.
3. **Thinking token đắt như output.** Bật reasoning cao cho tác vụ không cần là cách nhanh nhất để nhân đôi hoá đơn.

## 1.4 Đánh đổi chi phí, chất lượng và độ trễ

Đòn bẩy chia hai nhóm [2]:

- **Không đổi chất lượng:** prompt caching, bỏ token thừa, batch cho việc không ai chờ, rà lại prompt viết cho model cũ, hạn mức chi tiêu để chặn sự cố. Làm trước, áp dụng ngay.
- **Đổi chất lượng lấy chi phí:** chọn model, effort, giới hạn output và ngân sách, kiến trúc nhiều model. Làm sau, mỗi lần một thay đổi, đo bằng eval.

Độ trễ đi cùng chi phí ở phần lớn đòn bẩy: ít token hơn thì nhanh hơn, cache hit thì nhanh hơn, effort thấp thì nhanh hơn. Ngoại lệ:

- **Batch** rẻ hơn nhưng chậm hơn rất nhiều.
- **Chạy rẻ trước, chạy lại khi lỗi** rẻ hơn nhưng tác vụ lỗi mất gấp đôi thời gian [2].
- **Fast mode** nhanh hơn (Claude: tới 2,5 lần token output mỗi giây) với giá gấp đôi [2].
- **Cascade nhiều model** có thể rẻ hơn nhưng tác vụ khó đi qua nhiều tầng mới tới model mạnh ([Model routing §1.4](../model-routing/#14-routing-theo-độ-khó-theo-loại-tác-vụ-theo-chi-phí)).

Quy tắc ra quyết định: tính chi phí trên mỗi **tác vụ hoàn thành đạt chất lượng**, cộng chi phí của thất bại. Một model rẻ trả lời sai 10% số câu hỏi hỗ trợ khách hàng có thể tốn nhiều tiền hơn model đắt, nếu mỗi câu sai thành một ticket cho người xử lý.

---

# Phần 2 — Đòn bẩy theo provider

## 2.1 Bảng giá và cách tính token theo provider

### Giá niêm yết (USD / 1 triệu token, ngày 06/11/2026)

| Provider | Model | Input | Output | Đọc cache | Ghi chú |
|---|---|---|---|---|---|
| Claude [1] | Fable 5.1 | 10 | 50 | 0,25 | Model mạnh nhất |
| | Opus 5.5 | 4 | 20 | 0,20 | |
| | Sonnet 5.5 | 2 | 10 | 0,10 | |
| | Haiku 5.5 | 0,10 / 0,50 | 0,50 / 2,50 | 0,01 / 0,05 | Prompt ≤ 100.000 / > 100.000 token |
| OpenAI [9] | gpt-6-astra | 10 | 50 | 1,00 | |
| | gpt-6.1-sol | 2 | 10 | 0,10 | |
| | gpt-6.1-terra | 1 | 5 | 0,05 | |
| | gpt-6-luna | 0,10 | 0,50 | 0,01 | |
| Gemini [16] | Gemini 4 Argon | 2 → 4 | 10 → 20 | 0,10 → 0,20 | Giá tăng từ 01/12/2026; chỉ có implicit cache |
| | Gemini 3.8 Flash | 0,75 → 1,50 | 3,75 → 7,50 | 0,075 → 0,15 | Giá tăng từ 01/01/2027 |
| | Gemini 3.5 Flash-Lite | 0,30 | 2,50 | 0,03 | |

Bảng trên chỉ là một lát cắt. Những thứ cần kiểm thêm trong trang giá:

- **Giá theo độ dài prompt.** OpenAI tính giá "long context" khi input vượt 272.000 token, ví dụ gpt-6.1-sol tăng lên 4 USD input, 15 USD output [9]. Claude Haiku 5.5 tính giá gấp 5 lần cho mọi token của request khi prompt vượt 100.000 token, gồm cả cache read và cache write trong phần đếm prompt [1]. Các model Claude 4.6 trở lên khác (trừ Haiku 5.5) tính cùng một giá cho toàn bộ cửa sổ 1 triệu token [1].
- **Giá khuyến mãi có ngày hết hạn.** Gemini 4 Argon tăng gấp đôi từ 01/12/2026, Gemini 3.8 Flash từ 01/01/2027 [16]. Giá khuyến mãi của GPT-5.6 Sol được ghi là đến 30/11/2026 ở trang giá [9] và "at least through November 21, 2026" ở trang Fast mode [13]. Dự báo chi phí phải dùng giá sau khuyến mãi.
- **Tên gọi đổi.** OpenAI đổi tên Priority processing thành Fast mode ngày 30/07/2026 [9][13].

### Token không phải từ: tokenizer khác, ngôn ngữ khác

Cùng một văn bản, model khác nhau đếm ra số token khác nhau:

- **Đổi tokenizer giữa các thế hệ.** Claude 4.7 trở lên dùng tokenizer mới, sinh khoảng 30% token nhiều hơn cho cùng văn bản so với model trước đó [1]. So giá theo token giữa Opus 4.6 và Opus 5.5 mà không tính điều này sẽ sai. Anthropic khuyên so theo chi phí mỗi tác vụ hoàn thành [2].
- **Tiếng Việt tốn token hơn tiếng Anh.** Đo bằng `tiktoken` 0.14.0 [25] trên một đoạn chính sách hoàn tiền dịch song song (49 từ tiếng Việt, 39 từ tiếng Anh, cùng nội dung):

| Tokenizer | Tiếng Anh | Tiếng Việt (NFC) | Tiếng Việt (NFD) |
|---|---|---|---|
| `cl100k_base` | 44 | 93 | 199 |
| `o200k_base` | 44 | 56 | 162 |

Hai điều rút ra. Tokenizer mới thu hẹp khoảng cách nhiều (93 xuống 56 token), nhưng tiếng Việt vẫn tốn hơn khoảng một phần tư. Và **dạng Unicode quan trọng hơn chọn tokenizer**: cùng đoạn văn ở dạng NFD (dấu tách thành ký tự tổ hợp riêng, hay gặp khi copy từ macOS, PDF hoặc một số hệ thống cũ) tốn gần gấp ba token so với NFC. Chuẩn hoá `unicodedata.normalize("NFC", text)` mọi văn bản trước khi đưa vào prompt là đòn bẩy rẻ nhất trong bài này. Đây là số đo trên một đoạn văn với tokenizer của `tiktoken`, không phải tokenizer của mọi model; đo lại với endpoint đếm token của model bạn dùng.

### Đếm token trước khi gửi

- **Claude:** endpoint `count_tokens` miễn phí, có rate limit riêng theo tier; kết quả là ước lượng, có thể lệch nhỏ so với lúc gửi thật [5]. Đếm với đúng model ID sẽ dùng, vì tokenizer đổi giữa các thế hệ [5].
- **OpenAI:** endpoint đếm input token nhận cùng định dạng với Responses API, gồm ảnh, file, tool; `tiktoken` chỉ đếm được văn bản thuần và không tính token định dạng của request [15].
- **Gemini:** có `countTokens`; chi tiết ở [Multimodal](../multimodal/#12-ảnh-tốn-bao-nhiêu-token) cho ảnh, video, audio.

Ảnh là nguồn token dễ bị đánh giá thấp. Claude tính `⌈rộng/28⌉ × ⌈cao/28⌉` token mỗi ảnh; ảnh 1000×1000 là 1.296 token, ảnh 4K trên model 4.7 trở lên là 4.784 token. Trang vision của Claude tính ví dụ: ảnh 4K trên Opus 5.5 khoảng 19,14 USD mỗi nghìn ảnh [6].

## 2.2 Prompt caching

Prompt caching là đòn bẩy lớn nhất trong đo đạc của Anthropic, trên mọi model và benchmark họ đo [2]. Phần chi tiết theo provider (cơ chế, ngưỡng tối thiểu, TTL, cách đặt breakpoint, những gì làm vỡ cache) đã có trong [Multi-turn](../multi-turn-conversation/): [§1.4 kinh tế học của prompt caching](../multi-turn-conversation/#14-kinh-tế-học-của-prompt-caching), [§2.2 Claude](../multi-turn-conversation/#22-claude-anthropic), [§2.3 OpenAI](../multi-turn-conversation/#23-openai), [§2.6 Gemini](../multi-turn-conversation/#26-google-gemini), [§3.3 những thứ âm thầm làm vỡ cache](../multi-turn-conversation/#33-những-thứ-âm-thầm-làm-vỡ-cache). Phần này chỉ nói về chi phí; cơ chế gốc ở tài liệu prompt caching của Claude [4].

### Ví dụ: một hội thoại hỗ trợ khách hàng

Sonnet 5.5, system prompt và tool 6.000 token, 8 lượt, mỗi lượt người dùng viết 100 token và model trả lời 300 token. Tính bằng hàm `request_cost` ở [§1.2](#hàm-tính-chi-phí-từ-usage), giả sử breakpoint đặt ở cuối prompt, nên mỗi lượt đọc cache toàn bộ prompt của lượt trước và ghi phần mới (câu trả lời trước cộng câu hỏi mới). Sonnet 5.5 đọc cache với giá 0,05 lần input [1]:

| | 8 lượt | 40 lượt |
|---|---|---|
| Không cache | 0,1440 USD | 1,2320 USD |
| Có cache | 0,0514 USD | 0,2277 USD |
| Tỉ lệ | 2,8 lần | 5,4 lần |

Hội thoại càng dài, cache càng có lợi, vì phần gửi lại tăng theo bình phương số lượt còn phần ghi mới chỉ tăng tuyến tính. Ở 8 lượt, gần một nửa chi phí có cache (0,024 USD) là output; tới đây thì đòn bẩy tiếp theo là độ dài câu trả lời, không phải prompt.

### Hoà vốn và TTL

- **Claude, TTL 5 phút:** ghi 1,25 lần, đọc 0,1 lần giá input (0,05 lần ở Opus 5.5 và Sonnet 5.5, 0,025 lần ở Fable 5.1), nên có lợi ngay từ lần đọc đầu tiên. TTL 1 giờ ghi 2 lần, hoà vốn sau hai lần đọc [1].
- **OpenAI GPT-5.6 trở lên:** ghi 1,25 lần (TTL 30 phút) hoặc 2,5 lần (TTL 12 giờ, chỉ trên Responses API), đọc 0,1 lần (0,05 lần với GPT-6.1 Sol và Terra). Một lần ghi và một lần đọc tốn 1,35 lần, so với 2 lần khi không cache; mười request tốn 2,15 lần so với 10 lần [10]. Model trước GPT-5.6 không thu phí ghi cache [10].
- **Gemini:** implicit caching bật sẵn với Gemini 2.5 trở lên, không cần cấu hình; ngưỡng tối thiểu 4.096 token với Gemini 4 Argon và các model 3.x, 2.048 token với Gemini 2.5. Với Gemini 4 (Argon trở lên), trong hội thoại nhiều lượt, request mới chỉ dùng được cache của toàn bộ request trước khi phần thêm vào từ 2.050 token trở lên; nhỏ hơn thì 2.050 token cuối của request trước bị xử lý lại với giá input thường [17]. Cache tường minh (`cachedContents`) không có trên Gemini 4 Argon và Interactions API, và có phí lưu trữ theo giờ ở các model hỗ trợ [16][17].

Chọn TTL theo khoảng cách giữa các request, không theo trung bình. Anthropic đo trên một agent phân loại bug có chèn khoảng nghỉ mô phỏng người dùng: TTL 1 giờ đáng dùng khi khoảng 1 trong 20 lượt đến sau một khoảng nghỉ từ 5 phút đến 1 giờ, và ít khoảng nghỉ dài quá 1 giờ; điểm hoà vốn tính từ token là khoảng 1 trong 30 lượt trên Sonnet 5.5 và Opus 5.5 [2]. Với Fable 5.1, Opus 5.5 và Sonnet 5.5, giữ cache 5 phút ấm bằng request `max_tokens: 0` gửi mỗi 4 phút có thể rẻ hơn TTL 1 giờ khi khoảng nghỉ chỉ vài phút [2]. Request `max_tokens: 0` bị từ chối khi request có `thinking.type: "enabled"`, structured output, hoặc ép gọi tool [2].

### Những thay đổi nhỏ có giá lớn

Cache khớp theo prefix chính xác từng byte, theo thứ tự tools → system → messages; thay đổi ở đâu làm mất cache từ đó trở đi [2]. Trong đo của Anthropic trên một hội thoại khoảng 100.000 token đã cache:

- Sửa system prompt tốn khoảng 25–50 lần so với gửi cùng nội dung đó dưới dạng system message giữa hội thoại [2].
- Đổi `effort` ở cấp request tốn 20–40 lần so với đổi theo từng message [2].
- Một dòng trạng thái 25 token đặt ở đầu system prompt làm mọi request ghi lại toàn bộ hội thoại; chuyển nó xuống cuối lượt user mới nhất thì cache đọc lại bình thường [2].
- Trên Haiku 5.5, sửa system prompt tốn 12,5 lần một request không đổi [2].

Cache tách theo model, nên đổi model giữa hội thoại cũng là một lần ghi lại toàn bộ ([Model routing §1.5](../model-routing/#15-model-khác-nhau-trả-lời-khác-nhau-hệ-quả-khi-fallback)).

Kiểm tra cache bằng số trong `usage`, không bằng đọc code. Ở agent loop đã ấm, `cache_read_input_tokens` phải lớn hơn hẳn `input_tokens` cộng `cache_creation_input_tokens` [2].

## 2.3 Batch API

Request không ai chờ kết quả trong vài giây (eval, gán nhãn dữ liệu, tóm tắt hằng đêm, backfill, embedding hàng loạt) nên đi qua batch.

| | Claude Message Batches [3] | OpenAI Batch [11] | Gemini Batch [18] |
|---|---|---|---|
| Giảm giá | 50% mọi token, cộng dồn với cache | 50%, gồm cả cached input [9] | 50%; cache hit tính theo giá context caching thường |
| Thời hạn | Phần lớn xong dưới 1 giờ; hết hạn sau 24 giờ | Xong trong 24 giờ (`completion_window` chỉ nhận `24h`) | Mục tiêu 24 giờ, thường nhanh hơn |
| Giới hạn một batch | 100.000 request hoặc 256 MB | 50.000 request, file 200 MB | File input 2 GB; inline dưới 20 MB |
| Rate limit | Rate limit riêng cho batch | Pool rate limit riêng, cao hơn | Rate limit riêng |
| Kết quả lưu | 29 ngày | Qua output file | Qua file hoặc inline |

Những điểm hay bị bỏ qua:

- **Cache trong batch là best effort.** Request trong batch chạy đồng thời và không theo thứ tự, nên tỉ lệ cache hit của người dùng Claude dao động 30–98%. Để tăng khả năng hit: đặt `cache_control` giống hệt nhau ở mọi request, và cân nhắc TTL 1 giờ vì batch có thể chạy lâu hơn 5 phút [3].
- **Không phải mọi tham số đều dùng được.** Claude batch không nhận `stream`, `speed` (fast mode), `fallbacks` và `max_tokens: 0` [3]. Batch không dùng được với Managed Agents [2].
- **Batch hết hạn vẫn tính tiền phần đã xong.** OpenAI: request chưa xong bị huỷ và ghi vào error file, request đã xong vẫn tính tiền [11]. Claude: batch không xong trong 24 giờ thì hết hạn [3]. Job cần biết xử lý các item bị bỏ dở.
- **Batch là một lượt, không có tool loop giữa chừng.** Agent cần gọi tool nhiều lượt không đưa thẳng vào batch được; có thể nén thành một request bằng cách lấy trước dữ liệu tool cần, nhưng đó là thay đổi kiến trúc, làm model suy luận khác đi [2].

**Tier rẻ không phải batch.** OpenAI có Flex processing: request đồng bộ bình thường với `service_tier: "flex"`, tính giá bằng giá Batch, đổi lại phản hồi chậm hơn và đôi khi trả `429 Resource Unavailable`. Khi lỗi này xảy ra thì không bị tính tiền [12]. Hợp với eval, làm giàu dữ liệu, job không gấp mà code vẫn muốn gọi kiểu đồng bộ. OpenAI gợi ý hai cách xử lý `429`: retry có backoff, hoặc gửi lại ở tier thường nếu chấp nhận trả thêm để chắc chắn xong [12]. Timeout mặc định của SDK là 10 phút và có thể cần tăng [12].

## 2.4 Model nhỏ, distillation và fine-tune

Ba cách để dùng model nhỏ hơn cho một tác vụ, theo chi phí công sức tăng dần:

1. **Dùng model nhỏ với prompt tốt.** Model nhỏ hiện tại làm được nhiều việc mà model lớn của một hai năm trước mới làm được. Anthropic khuyên thử Haiku 5.5 trước cho tác vụ khối lượng lớn mà output kiểm tra được [2]. Trên GPQA Diamond, Haiku 5.5 đạt 85% so với 91% của Opus 5.5 và Sonnet 5.5, với chi phí mỗi câu đúng khoảng một phần hai mươi của Opus 5.5 và một phần mười của Sonnet 5.5 [2]. Hợp với phân loại, trích xuất, định tuyến; không hợp với agent loop dài [2].
2. **Distillation.** Dùng model lớn sinh output cho tập dữ liệu của bạn, rồi fine-tune model nhỏ trên đó. Chi phí là một lần sinh dữ liệu cộng huấn luyện; lợi ích là model nhỏ làm tốt hơn trên đúng tác vụ đó.
3. **Fine-tune trên dữ liệu có nhãn của bạn.**

Fine-tune có vài cái giá khó thấy:

- **Model fine-tune có thể đắt hơn model gốc khi chạy.** Ví dụ trong bảng giá OpenAI: gpt-4.1 fine-tune là 3 USD input, 12 USD output, so với 2 và 8 USD của gpt-4.1 gốc, chưa kể 25 USD mỗi triệu token huấn luyện [9].
- **Nền tảng có thể dừng.** OpenAI đang thu hẹp nền tảng fine-tuning tự phục vụ: người dùng mới không vào được, người dùng cũ còn tạo job huấn luyện được trong vài tháng tới, model đã fine-tune vẫn chạy được cho tới khi model gốc bị deprecate [9]. Một kế hoạch tiết kiệm dựa trên fine-tune cần tính cả việc phải làm lại trên nền tảng khác.
- **Model gốc cũ đi.** Model fine-tune gắn với một model gốc. Khi provider ra model mới rẻ hơn và tốt hơn, bạn phải fine-tune lại hoặc ở lại với model cũ.
- **Dữ liệu và eval là phần đắt nhất.** Phải có tập eval tốt để biết model fine-tune có thật sự đạt yêu cầu.

So sánh luôn với phương án không fine-tune: model nhỏ mới nhất với prompt viết lại và vài ví dụ trong prompt (được cache) thường đủ, và không có chi phí bảo trì. Chi tiết về khi nào fine-tune đáng làm ở [Fine-tuning](../fine-tuning/).

## 2.5 Self-host: khi nào rẻ hơn API

Self-host model open-weight đổi giá theo token lấy giá theo giờ GPU. Phép so sánh có dạng:

```
chi phí / triệu token (self-host) = (giá GPU mỗi giờ + chi phí vận hành mỗi giờ)
                                    / (triệu token xử lý được mỗi giờ × hệ số sử dụng)
```

Biến quyết định là **hệ số sử dụng**. GPU tính tiền theo giờ dù không có request; API tính tiền theo token. Ứng dụng có traffic dao động mạnh theo giờ trong ngày sẽ có GPU rảnh phần lớn thời gian, và số token mỗi giờ thực tế thấp hơn nhiều so với số đo benchmark ở tải đầy.

Những khoản thường bị quên khi tính:

- **Token mỗi giờ phụ thuộc tải.** Throughput cao nhất đạt được khi batch lớn; batch lớn làm mỗi request chậm hơn. Con số dùng để tính phải đo ở độ trễ bạn chấp nhận được, với độ dài prompt và output thật ([§1.3](#13-vì-sao-output-token-thường-đắt-hơn-input-token): output tốn GPU hơn input nhiều).
- **Dự phòng.** Cần ít nhất hai máy để không sập khi một máy lỗi, cần dư capacity cho giờ cao điểm.
- **Người vận hành.** Nâng cấp model, inference engine, driver, giám sát, trực sự cố.
- **Chất lượng.** Model open-weight nhỏ hơn có thể cần prompt dài hơn, nhiều lượt hơn, hoặc làm sai nhiều hơn. So theo chi phí mỗi tác vụ hoàn thành, như mọi so sánh khác trong bài.
- **API cũng có tier rẻ.** So với giá batch, flex, model nhỏ của API, không chỉ giá chuẩn của model lớn.

Self-host thường có lý khi traffic lớn và đều, khi có ràng buộc dữ liệu không được rời hạ tầng, hoặc khi cần model fine-tune mà API không cung cấp. Cách chọn GPU, inference engine và đo throughput ở [Self-hosting LLM](../self-hosting-llm/).

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Chọn model nhỏ nhất đạt yêu cầu

"Model nhỏ nhất" đúng hơn là "model rẻ nhất trên mỗi tác vụ hoàn thành". Hai điều trong đo đạc của Anthropic làm trực giác sai:

- **Model mạnh hơn có thể rẻ hơn mỗi tác vụ.** Trên một tập con SWE-bench Pro, Opus 5.5 ở effort `low` giải 87,4% số bài với 0,12 USD mỗi bài giải được, gần bằng Sonnet 5.5 ở mặc định (87,6%, 0,11 USD), dù mỗi đơn giá token của Opus 5.5 gấp đôi Sonnet 5.5 [2]. Opus 5.5 ở mặc định ngang Fable 5.1 trong phạm vi nhiễu, với khoảng một phần năm chi phí mỗi bài giải được [2].
- **Model mới thường rẻ hơn model cũ cùng hạng.** Từ Sonnet 5 lên Sonnet 5.5: thêm khoảng 10 điểm, chi phí mỗi bài giải được giảm 87% trên tập con SWE-bench Pro. Từ Haiku 4.5 lên Haiku 5.5: tỉ lệ pass từ 62% lên 93% trên một tập tác vụ Managed Agents, chi phí mỗi lần pass giảm khoảng 84% [2]. Nếu đang chạy model cũ một hai thế hệ, nâng model có thể là đòn bẩy rẻ nhất.

Quy trình gợi ý, dựa theo [2]:

1. Lấy vài chục tác vụ từ log production, phân bố giống traffic thật, viết cách kiểm tra kết quả cho từng tác vụ (test pass, ticket đóng, số dòng đúng). Ghi chi phí mỗi tác vụ cạnh điểm số.
2. Chạy model hiện tại ở hai ba mức effort, mỗi mức một phiên riêng (đổi effort giữa phiên làm vỡ cache và làm sai số đo).
3. Nếu effort thấp vẫn đạt, thử model hạng dưới ở effort mặc định của nó, rồi lại quét effort.
4. So trên **phần khó nhất** của tập (khoảng 10% tác vụ khó nhất), không chỉ trung vị. Hoá đơn được quyết định bởi các tác vụ model rẻ làm hỏng [2].
5. Chạy lặp. Sai khác một hai tác vụ hay vài cent trung bình nằm trong nhiễu một lần chạy ([Evaluation §1.4](../evaluation/#14-phương-sai-chạy-bao-nhiêu-lần-cần-bao-nhiêu-mẫu)).

**Chạy rẻ trước, chạy lại phần lỗi.** Khi kết quả kiểm tra được bằng code, chính sách rẻ nhất thường không phải một mức cố định. Trên tập con SWE-bench Pro với Opus 5.5: chạy mọi thứ ở `low` rồi chạy lại các bài lỗi ở `high` giải khoảng 97% với khoảng 0,17 USD mỗi bài, so với 95,3% và 0,29 USD khi chạy hết ở `high` (đã tính các lần thử rẻ thất bại) [2]. Điều kiện: cần tín hiệu lỗi đáng tin; checker cho qua bài sai thì lỗi lọt qua luôn [2]. Ý tưởng tương tự với nhiều model (cascade) ở [Decision models §3.7](../decision-models/#37-cascade-trích-xuất-dữ-liệu) và [Model routing §1.4](../model-routing/#14-routing-theo-độ-khó-theo-loại-tác-vụ-theo-chi-phí).

**Thêm model thứ hai là đòn bẩy cuối cùng.** Anthropic đo hai dạng: advisor (model rẻ chạy vòng lặp, hỏi model mạnh khi gặp quyết định khó) và orchestrator (model mạnh chia việc cho model rẻ). Trên benchmark code nội bộ, advisor cho thêm 1,7 điểm, ở rìa nhiễu, với khoảng 2,1 lần chi phí; ở hai benchmark khác, executor gần như không hỏi advisor và kết quả không hơn, có chỗ kém hơn [2]. Orchestrator chỉ thắng khi việc lớn hơn một context window hoặc là việc thường có đuôi chi phí dài [2]. Quét effort của một model trước; cặp model phải thắng toàn bộ đường cong điểm–chi phí của model đơn [2].

## 3.2 Rút ngắn prompt và context

### Rà prompt cũ khi đổi model

Prompt tích tụ câu chữ viết cho model cũ: "kiểm tra hai lần", "càng kỹ càng tốt", quy trình từng bước bắt buộc, scratchpad tự chế. Model mới làm theo đúng từng chữ, sinh thêm lượt tool và thêm chữ mà không chính xác hơn [2]. Trong đánh giá hỗ trợ khách hàng của Anthropic, khi chuyển từ Opus 4.8 sang Opus 5, prompt cũ làm mỗi ticket tốn hơn mà độ chính xác không đổi; rà lại prompt giảm 14% chi phí ở cả hai lần nâng model được đo, và ở một lần còn tăng độ chính xác [2]. Hai loại câu chữ cũ có giá khác nhau: câu dặn quá mức tốn tiền; cấu hình thinking đã bỏ, quy tắc mâu thuẫn và scratchpad xung đột với thinking của model thì tốn độ chính xác [2].

### Bỏ thứ không ảnh hưởng tới câu trả lời

- **Tài liệu tham khảo dài trong mọi request.** Nếu phần lớn request chỉ cần một phần, chuyển nó ra sau một tool để model lấy khi cần. Nếu mọi request đều cần gần hết, để nguyên và cache, vì tài liệu trong prefix đã cache rất rẻ.
- **Định nghĩa tool không dùng.** Trong đo của Anthropic, agent phân loại bug với tới 502 tool (gồm định nghĩa thật từ các MCP server công khai) có chi phí tăng từ 0,55 lên 1,02 USD mỗi lượt chạy khi nạp hết; dùng tool search với `defer_loading` giữ chi phí ở 0,56 USD với độ chính xác tương đương [2]. Giữ 3–5 tool hay dùng nhất luôn được nạp [2]. Với ít tool thì bước tìm kiếm chỉ thêm chi phí.
- **Bảng dữ liệu dán vào prompt.** Một file CSV 1.862 dòng dán thẳng vào prompt là khoảng 91.000 input token mỗi request; Sonnet 5 trả lời đúng 7/25 câu hỏi tổng hợp. Upload qua Files API và để model truy vấn bằng code execution: đúng 25/25, chi phí khoảng một phần mười hai [2].
- **Ảnh lớn hơn cần.** Ảnh tính token theo diện tích ([§2.1](#đếm-token-trước-khi-gửi)). Thu nhỏ trước khi gửi nếu tác vụ không cần chi tiết; cách chọn kích thước ở [Multimodal §3.1](../multimodal/#31-tiền-xử-lý-ảnh-độ-phân-giải-cắt-xoay).
- **Kết quả tool dài.** Trả về phần model cần, không phải toàn bộ payload. Tool nhận tham số `limit`, `fields` thay vì trả mọi thứ ([Tool use §3.3](../tool-use/#33-tool-result-là-prompt)).
- **Văn bản ở dạng Unicode NFD.** Chuẩn hoá về NFC ([§2.1](#token-không-phải-từ-tokenizer-khác-ngôn-ngữ-khác)).

Bỏ bớt context không luôn rẻ hơn. Model có thể mất thêm lượt để tìm lại thứ trước đây có sẵn trong prompt. Đo chi phí mỗi tác vụ trước và sau.

### Hội thoại và agent loop dài

Với phiên đủ dài, compaction (tóm tắt rồi tiếp tục) giảm chi phí; với phiên ngắn thì không. Trong đo của Anthropic trên một model cũ hơn, compaction giảm 32% và cắt bớt kết quả tool cũ giảm 39% trên lượt chạy dài; context editing làm lượt chạy ngắn tốn thêm 73% và không thay đổi gì trên lượt chạy dài [2]. Mỗi lần sửa lịch sử làm vỡ cache từ chỗ sửa trở đi. Với Claude 5.1 trở lên, sửa kết quả tool cũ ở phía client còn làm mọi thinking block sau đó mất hiệu lực, nên dùng compaction phía server thay vì tự cắt [2]. Haiku 5.5 có thêm một lý do để compact sớm: prompt vượt 100.000 token làm mọi token của request đắt gấp 5 lần. Nếu dùng compaction theo ngưỡng, đặt `trigger` thấp hơn 100.000 trừ đi mức tăng lớn nhất giữa hai request (mặc định là 150.000); trong thử nghiệm của Anthropic với lượt 10.000 token, trigger 90.000 rẻ hơn 70% mỗi request so với mặc định [2]. Chi tiết về trim, summarize, compact ở [Multi-turn §3.8](../multi-turn-conversation/#38-khi-context-quá-dài-trim-summarize-compact).

## 3.3 Giới hạn output và reasoning

### Yêu cầu câu trả lời ngắn

Output đắt nhất và có thể giảm mà không đổi chất lượng, nếu format ngắn vẫn đủ cho người đọc. Anthropic chạy agent phân loại bug với ba yêu cầu về câu trả lời cuối: một dòng, hai dòng (bản gốc), và một bản ghi nhớ sáu mục. Chi phí mỗi lượt chạy là 0,49 USD, 0,57 USD và 1,40 USD; độ chính xác của cả ba nằm trong nhiễu của nhau (78–85%) [2]. "Yêu cầu câu trả lời mà bạn sẽ đọc, không phải câu trả lời trông có vẻ kỹ" [2]. Cách hiệu quả nhất là mô tả chính xác format, kèm ví dụ.

### `max_tokens` là chốt chặn, không phải nút tiết kiệm

Model không thấy `max_tokens`. Hạ nó xuống không làm model viết gọn hơn; nó chỉ cắt câu trả lời giữa chừng với `stop_reason: "max_tokens"`, và phần đã sinh vẫn bị tính tiền [2]. Retry ở cùng mức thường lại bị cắt; retry ở mức cao hơn thì trả thêm cho lần thử bỏ đi [2]. Trên benchmark tác vụ repository nội bộ của Anthropic, Opus 5.5 với `max_tokens` 16.384 có 25% lần thử bị cắt và giải 57,6% số bài; ở 64.000 không lần nào bị cắt, giải 75,6%, với cùng chi phí mỗi bài giải được [2]. Khuyến nghị cho agent là 64.000, hoặc 128.000 khi một lần thử bị cắt rất đắt, kèm streaming cho response dài [2].

Dùng `max_tokens` để chặn sự cố (model lặp vô hạn, prompt injection bắt model viết dài), đặt đủ rộng để không chạm trong vận hành bình thường. Để giảm độ dài, dùng prompt; để giảm suy luận, dùng effort.

### Effort và reasoning

Effort điều khiển lượng thinking, số lượt tool và mức tự kiểm tra của model, và model thấy được nó [2]. Mặc định là `high` trên phần lớn model Claude, `medium` trên Opus 5.5 và Haiku 5.5 [2]. Mức tiết kiệm và cái giá khác nhau theo tác vụ:

| Benchmark | Model | `low` | `medium` | `high` |
|---|---|---|---|---|
| SWE-bench Pro (tập con) | Opus 5.5 | 87,4% · 0,11 USD | 92,8% · 0,21 USD | 95,3% · 0,29 USD |
| DeepResearch Bench II | Fable 5.1 | 60,5% · 3,47 USD | 60,5% · 4,44 USD | 63,0% · 5,64 USD |

Chi phí tính trên mỗi tác vụ [2]. Trên code dài, effort mua được độ chính xác thật. Trên nghiên cứu với Fable 5.1, `medium` không hơn `low`. Nhưng không thể tổng quát: Opus 5.5 ở `low` thấp hơn `medium` 13,3 điểm trên DRACO, và 34 điểm trên một benchmark tìm lỗi nội bộ [2]. Chỉ quét effort trên workload của bạn mới trả lời được.

Với Haiku 5.5, thinking bật mặc định và tính vào `max_tokens`, nên `max_tokens` nhỏ có thể làm response kết thúc trước khi có chữ nào; tăng `max_tokens` hoặc hạ effort [2]. OpenAI có hiện tượng tương tự: hết `max_output_tokens` trước khi có output hiển thị vẫn tính tiền input và reasoning [14]. Cách đặt ngân sách reasoning theo provider ở [Reasoning models §3.5](../reasoning-models/#35-chi-phí-và-độ-trễ-max_tokens-timeout-ước-lượng-ngân-sách).

### Ngân sách cho agent

Agent không có trần sẽ có vài lượt chạy rất đắt. Ba lớp chặn:

- **Ngân sách trong code**: số lượt, số token, thời gian, chi phí tối đa mỗi run ([Agents §3.2](../agents/#32-điều-kiện-dừng-giới-hạn-bước-và-ngân-sách)).
- **Ngân sách phía nền tảng**: Claude Managed Agents có session budget, một trần đô la cho mỗi session; chạm trần thì session dừng với `stop_reason: budget_reached` [2].
- **Spend limit của workspace hoặc project** là chốt chặn cuối cho toàn bộ workspace [2][8].

## 3.4 Semantic cache và cache kết quả

Prompt caching của provider tái dùng phần tính toán: model vẫn chạy và vẫn viết câu trả lời mới. Cache kết quả tái dùng chính câu trả lời: không gọi model. Hai loại cache kết quả có rủi ro rất khác nhau.

### Exact cache

Khoá là request đã chuẩn hoá (model, phiên bản prompt, tham số, nội dung); giá trị là response. Trùng khoá thì trả response cũ. Không có rủi ro trả lời sai cho câu hỏi khác, vì câu hỏi giống hệt.

Hợp với: tác vụ xác định trên dữ liệu không đổi (tóm tắt một tài liệu, phân loại một sản phẩm, trích xuất từ một file), endpoint được nhiều người gọi với cùng input (câu hỏi gợi ý sẵn trên giao diện), và eval chạy lại trên cùng dataset. Không hợp với hội thoại: lịch sử khác nhau thì khoá khác nhau.

Điều kiện để exact cache đúng:

- Khoá gồm mọi thứ ảnh hưởng tới output: model ID, phiên bản prompt template, tham số, tool, và dữ liệu được chèn vào prompt (hoặc hash của nó). Thiếu phiên bản prompt là lỗi hay gặp: đổi prompt mà cache vẫn trả câu trả lời của prompt cũ.
- Chuẩn hoá trước khi hash: Unicode NFC, khoảng trắng, thứ tự khoá JSON.
- TTL theo độ tươi của dữ liệu nguồn, và xoá theo nguồn khi dữ liệu nguồn đổi.
- Không cache chéo giữa tenant hoặc user khi prompt chứa dữ liệu riêng của họ.

### Semantic cache

Embed câu hỏi, tìm câu hỏi cũ gần nhất trong vector store; độ tương đồng vượt ngưỡng thì trả câu trả lời cũ. Rủi ro là hai câu hỏi gần nhau về embedding nhưng cần câu trả lời khác nhau ("hoàn tiền trong bao lâu" và "hoàn tiền được không").

Số liệu công bố cho thấy rủi ro đó không nhỏ:

- **AWS** đo semantic cache trên 63.796 câu hỏi chatbot thật và các bản diễn đạt lại của chúng (dataset SemBenchmarkLmArena), với Claude 3 Haiku và ElastiCache [23]:

| Ngưỡng | Tỉ lệ hit | Độ chính xác của câu trả lời từ cache | Giảm chi phí |
|---|---|---|---|
| 0,99 | 23,5% | 92,1% | 15,8% |
| 0,95 | 56,0% | 92,6% | 51,9% |
| 0,90 | 74,5% | 92,3% | 72,5% |
| 0,80 | 87,6% | 91,8% | 84,6% |
| 0,75 | 90,3% | 91,2% | 86,3% |
| 0,50 | 94,3% | 87,5% | 88,0% |

  Đáng chú ý: ngay ở ngưỡng 0,99, khoảng 8% câu trả lời lấy từ cache không đúng ý người hỏi. Ngưỡng chặt hơn giảm số hit nhiều hơn là giảm số hit sai.
- **Zhu và cộng sự** (arXiv 2402.01173) dựng một tập khó từ Natural Questions; embedding có sẵn chỉ đạt AUC 0,51 khi dự đoán câu trả lời cache có dùng được không, ngang đoán ngẫu nhiên. Fine-tune embedding cho việc này nâng lên 0,81 [21].
- **GPTCache** (Bang, NLP-OSS 2023), thư viện semantic cache mã nguồn mở phổ biến, tự ghi trong paper: tỉ lệ cache hit tốt nhất không vượt 90% với embedding hiện có, nghĩa là hit sai đáng kể khi dùng, và "would be unacceptable in real production scenarios" [22].
- **vCache** (arXiv 2502.03771, ICLR 2026) chỉ ra ngưỡng tĩnh dùng chung cho mọi request không cho đảm bảo nào về tỉ lệ lỗi; họ học ngưỡng riêng cho từng câu trong cache để giữ tỉ lệ lỗi dưới mức người dùng đặt, báo cáo tỉ lệ hit cao hơn tới 12,5 lần và tỉ lệ lỗi thấp hơn tới 26 lần so với ngưỡng tĩnh [20].

Khi nào dùng được semantic cache:

- Câu trả lời ổn định và giống nhau cho mọi người hỏi (FAQ, hướng dẫn sản phẩm), không phụ thuộc dữ liệu riêng của người hỏi hay thời điểm.
- Câu trả lời sai có hậu quả nhỏ, hoặc có bước kiểm tra sau.
- Lọc trước bằng metadata (sản phẩm, danh mục, ngôn ngữ) để chỉ so với câu hỏi cùng ngữ cảnh. AWS khuyên dùng filter theo ngữ cảnh ứng dụng để chọn câu hỏi nào được cache và để tăng độ liên quan của hit [23].
- Với hội thoại nhiều lượt, embed câu hỏi hiện tại cộng các thông tin chính cần để trả lời, không embed nguyên đoạn hội thoại [23].
- Đo độ chính xác của hit trên traffic thật (lấy mẫu hit, chấm bằng người hoặc LLM judge), không chỉ đo tỉ lệ hit.

Không nên dùng cho: câu hỏi về tài khoản, đơn hàng, số dư, trạng thái; câu trả lời pháp lý, y tế, tài chính; bất cứ thứ gì sai thì phải bồi thường.

## 3.5 Giới hạn chi phí theo user và chống lạm dụng

OWASP xếp "Unbounded Consumption" là LLM10 trong Top 10 cho ứng dụng LLM năm 2025: ứng dụng cho người dùng gọi inference quá mức, không kiểm soát, dẫn tới từ chối dịch vụ, thiệt hại tài chính, và trích xuất model [24]. Một kịch bản họ nêu: kẻ tấn công tạo lượng thao tác quá mức để khai thác mô hình trả tiền theo lượt dùng, gây chi phí không chịu nổi cho nhà cung cấp dịch vụ [24]. Biện pháp OWASP đề xuất gồm kiểm tra kích thước input, rate limit và quota theo người dùng, timeout, giám sát mức dùng bất thường, và thiết kế để suy giảm từ từ khi quá tải [24].

### Lớp phía provider

- **Spend limit** của workspace (Claude) hoặc project (OpenAI) chặn tổng chi phí của một nhóm. Claude: hạn mức tự đặt trả `400`, trần của bậc tài khoản trả `429` không có `retry-after`; cả hai nghĩa là mọi request của nhóm đó dừng [8]. Đây là chốt chặn cuối, không phải công cụ quản lý theo user: một user lạm dụng chạm hạn mức thì mọi user khác cùng workspace cũng dừng.
- **Rate limit theo workspace hoặc project** tách một tính năng hay một nhóm khách hàng khỏi phần còn lại ([Model routing §3.4](../model-routing/#34-quản-lý-quota-nội-bộ-theo-tenant)).

### Lớp phía ứng dụng: hạn mức đô la theo tenant

Rate limit theo số request không đủ: một request 200.000 token với `max_tokens` 64.000 đắt gấp hàng trăm lần một request phân loại. Hạn mức nên tính bằng tiền. Cách làm: trước khi gọi, giữ chỗ chi phí tối đa có thể của request; sau khi gọi, thay bằng chi phí thật. Giữ chỗ trước để nhiều request đồng thời của cùng tenant không cùng lúc vượt hạn mức.

```python
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal

from cost import M, PRICES


class BudgetExceeded(Exception):
    pass


class SpendLedger:
    """Hạn mức chi tiêu theo tenant theo ngày (UTC). Giữ chỗ chi phí tối đa trước khi gọi,
    trừ chi phí thật sau khi gọi. Bản trong bộ nhớ, một process; nhiều worker thì
    thay dict bằng Redis hoặc bảng DB với cập nhật nguyên tử."""

    def __init__(self, daily_limit_usd: dict[str, Decimal], clock=lambda: datetime.now(timezone.utc)):
        self.limits, self.clock = daily_limit_usd, clock
        self.spent: dict[tuple[str, str], Decimal] = {}
        self.reserved: dict[tuple[str, str], Decimal] = {}
        self.day: str | None = None
        self.lock = threading.Lock()

    @contextmanager
    def hold(self, tenant: str, max_cost: Decimal):
        with self.lock:
            day = self.clock().strftime("%Y-%m-%d")
            if day != self.day:                  # sang ngày mới: bỏ số của các ngày trước
                self.day = day
                self.spent = {k: v for k, v in self.spent.items() if k[1] >= day}
            key = (tenant, day)
            used = self.spent.get(key, Decimal(0)) + self.reserved.get(key, Decimal(0))
            if used + max_cost > self.limits[tenant]:
                raise BudgetExceeded(f"{tenant}: đã dùng {used}, cần {max_cost}, hạn mức {self.limits[tenant]}")
            self.reserved[key] = self.reserved.get(key, Decimal(0)) + max_cost
        actual = {"usd": max_cost}          # lỗi không rõ chi phí: tính như đã tiêu hết phần giữ chỗ
        try:
            yield actual
        finally:
            with self.lock:
                self.reserved[key] -= max_cost
                if not self.reserved[key]:
                    del self.reserved[key]
                if key[1] == self.day:           # request vắt qua nửa đêm: ngày cũ đã đóng, không ghi lại
                    self.spent[key] = self.spent.get(key, Decimal(0)) + actual["usd"]


def max_request_cost(model: str, input_tokens: int, max_tokens: int,
                     write_multiplier: Decimal = Decimal("1.25")) -> Decimal:
    """Trần chi phí của một request chưa gửi: mọi input token bị ghi cache (1.25 với TTL 5 phút,
    2 với TTL 1 giờ, 1 nếu không dùng cache), output dùng hết max_tokens. Không bao được
    request có server tool chạy nhiều vòng hoặc compaction: các vòng đó tính tiền riêng."""
    p = PRICES[model]
    if p.long_threshold and input_tokens > p.long_threshold:   # ngưỡng tính trên prompt
        p = p.long
    return (input_tokens * p.input * write_multiplier + max_tokens * p.output) / M
```

Cách dùng với `request_cost` ở [§1.2](#hàm-tính-chi-phí-từ-usage):

```python
import anthropic

ledger = SpendLedger({"acme": Decimal("20.00")})
n = client.messages.count_tokens(model=model, system=system, messages=messages).input_tokens
try:
    with ledger.hold("acme", max_request_cost(model, n, max_tokens)) as actual:
        try:
            resp = client.messages.create(model=model, max_tokens=max_tokens,
                                          system=system, messages=messages)
        except anthropic.APIStatusError as e:
            if e.status_code < 500:
                actual["usd"] = Decimal(0)   # 4xx: bị từ chối khi kiểm tra request, không có usage
            raise                            # 5xx: không biết chắc, giữ nguyên phần giữ chỗ
        actual["usd"] = request_cost(resp.usage, model)
except BudgetExceeded:
    ...  # báo người dùng đã hết hạn mức hôm nay, không gọi model
```

Test đã chạy:

1. Trần chi phí: Sonnet 5.5, 10.000 input ghi cache 5 phút, `max_tokens` 4.000 ra 0,065 USD; không cache ra 0,06 USD.
2. Haiku 5.5: prompt 101.000 token dùng giá prompt dài; 99.000 token dùng giá thường, dù cộng `max_tokens` thì vượt 100.000 (ngưỡng chỉ tính trên prompt [1]).
3. Sau khi gọi, phần giữ chỗ được thay bằng chi phí thật.
4. Một request đang giữ chỗ 0,80 USD chặn request khác cần 0,20 USD khi hạn mức là 1 USD, dù chưa tiêu đồng nào.
5. Exception trong lúc gọi mà không gán chi phí thì tính đủ phần giữ chỗ.
6. Vượt hạn mức ném `BudgetExceeded` với thông báo có số liệu.
7. Qua 00:00 UTC thì hạn mức làm mới, và số của ngày cũ bị xoá: chạy 1.000 ngày liên tiếp vẫn chỉ giữ một khoá. Request bắt đầu trước nửa đêm và xong sau nửa đêm không trừ vào hạn mức ngày mới, cũng không tạo lại khoá của ngày cũ.
8. 20 thread cùng giữ chỗ 0,10 USD với hạn mức 1 USD: đúng 10 thread qua, tổng chi tiêu đúng 1 USD.
9. Đoạn "cách dùng" ở trên, chạy với SDK `anthropic` 2.6.0 và response giả lập: `200` ghi đúng chi phí theo `usage`, `400` ghi 0, `500` giữ nguyên phần giữ chỗ.

Giới hạn cần biết:

- **Trong bộ nhớ một process.** Với nhiều worker, thay `dict` và `Lock` bằng Redis (`INCRBYFLOAT` trong Lua script để kiểm và cộng nguyên tử) hoặc một bảng DB với `UPDATE ... WHERE spent + reserved + :x <= limit`.
- **Trần không bao server tool loop và compaction**, vì các vòng đó sinh thêm token ngoài một lần gọi. Agent cần ngân sách theo run ([Agents §3.2](../agents/#32-điều-kiện-dừng-giới-hạn-bước-và-ngân-sách)), giữ chỗ theo từng lượt.
- **Lỗi giữa stream vẫn tính tiền phần đã sinh** ([Streaming §1.5](../streaming/#15-stream-không-thay-đổi-chi-phí-nhưng-thay-đổi-cách-tính-usage)). Với streaming, lấy `usage` từ event cuối nếu có; nếu stream đứt, để nguyên phần giữ chỗ.
- **Lỗi mạng sau khi request đã tới provider** (timeout phía client) và lỗi `5xx` có thể đã bị tính tiền; code trên để nguyên phần giữ chỗ cho các trường hợp này vì không biết chắc, rồi đối soát với Usage and Cost API [7].

### Những dấu hiệu lạm dụng

- Input dài bất thường, đặc biệt là dán văn bản lớn vào ô chat: đặt giới hạn độ dài input theo tính năng, đếm token trước khi gửi.
- Một user gọi liên tục với nội dung gần giống nhau: có thể là script.
- Prompt cố ép model viết dài ("viết 10.000 từ"), lặp, hoặc gọi tool liên tục.
- Tài khoản dùng thử dùng hết hạn mức ngay ngày đầu, nhiều tài khoản từ cùng một nguồn.

Cảnh báo theo chi phí mỗi user mỗi giờ, so với phân phối bình thường, bắt được phần lớn các trường hợp này sớm hơn hoá đơn ([LLM Observability §3.7](../llm-observability/#37-alert-ngưỡng-nào-đáng-đánh-thức-người-trực)).

## 3.6 Dự báo chi phí khi scale

### Từ đo đạc, không từ ước lượng

Ước lượng chi phí bằng cách nhân độ dài prompt trung bình với đơn giá thường sai vì ba lý do:

1. **Hội thoại dài lên.** Chi phí mỗi hội thoại tăng gần theo bình phương số lượt khi không cache ([§2.2](#ví-dụ-một-hội-thoại-hỗ-trợ-khách-hàng)). Tính năng mới cho phép hội thoại dài hơn làm chi phí mỗi người dùng tăng dù số người dùng không đổi.
2. **Phân phối có đuôi dài.** Trung bình bị kéo bởi vài tác vụ rất đắt [2]. Dùng phân vị (p50, p90, p99) chi phí mỗi tác vụ từ log, và mô phỏng với phân phối đó.
3. **Cache hit rate ở quy mô khác.** Traffic thưa (ít request mỗi phút trên một prefix) có hit rate thấp hơn vì cache hết hạn giữa các request; traffic dày hơn khi scale có thể làm hit rate tăng. Với OpenAI, cache nằm trên từng máy, và trên khoảng 15 request mỗi phút cho cùng prefix thì traffic có thể tràn sang máy khác, làm giảm hit [10].

### Một ví dụ tính

Hội thoại hỗ trợ khách hàng ở [§2.2](#ví-dụ-một-hội-thoại-hỗ-trợ-khách-hàng) (Sonnet 5.5, 8 lượt): 0,0514 USD mỗi hội thoại có cache, 0,1440 USD không cache. Ở 10.000 hội thoại mỗi ngày, 30 ngày:

- Có cache: khoảng 15.400 USD mỗi tháng.
- Không cache: khoảng 43.200 USD mỗi tháng.
- Nếu tính năng mới làm hội thoại trung bình dài 40 lượt: khoảng 68.300 USD mỗi tháng có cache, 369.600 USD không cache.

Con số thật phụ thuộc phân phối độ dài hội thoại, không phải một độ dài cố định. Mô phỏng bằng cách lấy mẫu độ dài từ log thật và tính chi phí từng hội thoại.

### Những thứ thay đổi theo thời gian

- **Giá đổi.** Giá khuyến mãi hết hạn ([§2.1](#21-bảng-giá-và-cách-tính-token-theo-provider)). Model mới thường rẻ hơn mỗi tác vụ, nhưng có thể dùng tokenizer khác.
- **Default đổi.** Effort mặc định khác nhau theo model [2]; nâng model mà không đặt effort tường minh có thể làm chi phí tăng hoặc giảm.
- **Bậc tài khoản.** Trần chi tiêu tháng theo bậc của Claude (500, 1.000, 200.000 USD ở Start, Build, Scale [8]) là trần cứng; dự báo vượt trần thì phải nâng bậc trước, không phải sau khi API trả `429`.

## 3.7 Failure modes thường gặp

| # | Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|---|
| 1 | Chi phí tăng nhanh hơn số người dùng | Hội thoại dài hơn; chi phí mỗi hội thoại tăng theo bình phương số lượt [2] | Đo chi phí theo độ dài hội thoại; cache; compaction cho phiên dài |
| 2 | Cache hit rate gần 0 dù đã bật cache | Nội dung động (thời gian, request ID) ở đầu prefix [2] | Chuyển nội dung động xuống cuối lượt user mới nhất |
| 3 | Cache vỡ giữa hội thoại sau một thay đổi nhỏ | Sửa system prompt, đổi tool, đổi `effort` hoặc thinking ở cấp request [2] | System message giữa hội thoại; đổi effort theo message; đổi ở ranh giới compaction |
| 4 | Batch rẻ hơn ít hơn dự tính | Cache hit trong batch là best effort, TTL 5 phút hết trước khi request sau chạy [3] | `cache_control` giống hệt nhau; TTL 1 giờ |
| 5 | Chi phí báo cáo thấp hơn hoá đơn | Đọc `usage` cấp trên cùng, bỏ sót compaction, advisor, lần thử bị refusal trong `usage.iterations` [2][26] | Cộng `usage.iterations` |
| 6 | Chi phí tăng sau khi nâng model dù đơn giá thấp hơn | Tokenizer mới sinh khoảng 30% token nhiều hơn [1]; effort mặc định khác; prompt cũ làm model mới làm thừa [2] | So theo chi phí mỗi tác vụ; rà prompt; đặt effort tường minh |
| 7 | Prompt tiếng Việt tốn gấp ba token dự tính | Văn bản ở dạng NFD | Chuẩn hoá NFC trước khi gửi |
| 8 | Response bị cắt, retry, lại bị cắt | `max_tokens` quá thấp; model không thấy nó nên không viết gọn hơn [2] | Tăng `max_tokens`; giảm độ dài bằng prompt và effort |
| 9 | Request Haiku 5.5 đột ngột đắt gấp 5 lần | Prompt (kể cả cache read và write) vượt 100.000 token [1] | Compact trước ngưỡng; đặt `trigger` dưới 100.000 [2] |
| 10 | Trả tiền nhưng không có output | Hết `max_tokens` / `max_output_tokens` trong lúc thinking [2][14] | Tăng ngân sách output hoặc giảm effort |
| 11 | Hoá đơn tăng sau khi thêm MCP server | Định nghĩa tool của server nằm trong mọi request [2] | Tool search với `defer_loading`; chỉ bật tool cần |
| 12 | Semantic cache trả câu trả lời của câu hỏi khác | Embedding gần nhau nhưng ý khác; ngưỡng tĩnh không bảo đảm tỉ lệ lỗi [20][23] | Chỉ dùng cho câu trả lời chung; lọc metadata; đo độ chính xác của hit |
| 13 | Exact cache trả câu trả lời của prompt cũ | Khoá không gồm phiên bản prompt hoặc model | Đưa phiên bản prompt, model, tham số vào khoá |
| 14 | Một user làm cả workspace dừng | Chỉ có spend limit cấp workspace; không có hạn mức theo user [8] | Hạn mức đô la theo user hoặc tenant trong ứng dụng |
| 15 | Chi phí tăng vọt sau khi bật fast mode cho một job | Fast mode giá gấp đôi [1][13] | Fast mode chỉ cho đường người dùng chờ; không dùng cho batch, ETL |
| 16 | Hoá đơn tăng đầu tháng dù traffic không đổi | Giá khuyến mãi hết hạn [9][16] | Dự báo bằng giá sau khuyến mãi; theo dõi ngày hết hạn |
| 17 | Model nhỏ rẻ hơn mỗi token nhưng tổng chi phí cao hơn | Thêm lượt, thêm retry, thêm lỗi cho người xử lý [2] | So theo chi phí mỗi tác vụ hoàn thành, gồm chi phí thất bại |
| 18 | Ảnh làm request tốn hàng nghìn token | Ảnh độ phân giải cao, tính token theo diện tích [6] | Thu nhỏ trước khi gửi khi không cần chi tiết |
| 19 | Flex request thất bại hàng loạt | `429 Resource Unavailable` khi thiếu capacity [12] | Retry có backoff, hoặc gửi lại ở tier thường |
| 20 | Context editing làm chi phí tăng | Mỗi lần xoá viết lại phần hội thoại đã cache [2] | Chỉ dùng khi cần chỗ trong context; xoá ít lần, mỗi lần nhiều |

## 3.8 Checklist

**Đo**

- [ ] Log đủ các trường `usage` (input, cache write theo TTL, cache read, output, `speed`, `service_tier`, `iterations`) cho mọi request, gắn user, tenant, feature, model, phiên bản prompt.
- [ ] Chi phí tính theo tác vụ, gồm retry và lần thử thất bại.
- [ ] Đối soát log với Usage and Cost API hoặc dashboard của provider.
- [ ] Biết tỉ lệ input đọc từ cache, tỉ lệ output trong chi phí, và phân vị chi phí mỗi tác vụ.

**Không đổi chất lượng**

- [ ] Prompt caching bật; nội dung động nằm sau breakpoint cuối; hit rate được theo dõi.
- [ ] TTL chọn theo phân phối khoảng cách giữa các request.
- [ ] Mọi việc không ai chờ đi qua batch hoặc flex.
- [ ] Văn bản chuẩn hoá NFC; ảnh thu nhỏ theo nhu cầu; tool ít dùng được nạp trễ.
- [ ] Prompt được rà lại mỗi khi đổi model.
- [ ] Format câu trả lời được mô tả rõ, ngắn nhất mà người đọc cần.

**Đổi chất lượng lấy chi phí**

- [ ] Effort được quét trên workload thật, mỗi mức một phiên riêng, chạy lặp.
- [ ] So model theo chi phí mỗi tác vụ hoàn thành, trên cả phần khó nhất.
- [ ] Đã thử model mới cùng hạng trước khi hạ xuống model nhỏ hơn.
- [ ] Mỗi thay đổi một lần, giữ hoặc bỏ theo eval.

**Chặn sự cố**

- [ ] `max_tokens` đủ rộng cho vận hành bình thường, đủ thấp để chặn chạy vô hạn.
- [ ] Hạn mức đô la theo user hoặc tenant trong ứng dụng; spend limit ở workspace hoặc project làm chốt cuối.
- [ ] Giới hạn độ dài input theo tính năng.
- [ ] Alert khi chi phí mỗi user hoặc mỗi feature lệch khỏi phân phối bình thường.
- [ ] Dự báo dùng giá sau khuyến mãi và trần chi tiêu của bậc tài khoản.

---

## Nguồn tham khảo

Truy cập ngày 06/11/2026.

| # | Nguồn | Nội dung dùng trong bài |
|---|---|---|
| [1] | [Claude: Pricing](https://platform.claude.com/docs/en/about-claude/pricing) | Giá model, hệ số cache, batch, fast mode, data residency, giá prompt dài của Haiku 5.5, token của tool, web search, tokenizer mới |
| [2] | [Claude: Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence) | Số đo của Anthropic về caching, TTL, token hygiene, batch, rà prompt, so model, effort, chạy lại phần lỗi, `max_tokens`, độ dài output, advisor, orchestrator |
| [3] | [Claude: Batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing) | Giới hạn batch, thời hạn, cache trong batch, tham số không hỗ trợ |
| [4] | [Claude: Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) | Cơ chế cache, pre-warming |
| [5] | [Claude: Token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting) | Đếm token miễn phí, là ước lượng, đếm theo model ID |
| [6] | [Claude: Vision](https://platform.claude.com/docs/en/build-with-claude/vision) | Công thức token của ảnh, ví dụ chi phí |
| [7] | [Claude: Usage and Cost API](https://platform.claude.com/docs/en/manage-claude/usage-cost-api) | Admin API key, độ trễ dữ liệu, tần suất polling, chi phí code execution chỉ có trong cost endpoint |
| [8] | [Claude API: Rate limits](https://platform.claude.com/docs/en/api/rate-limits) | Spend limit của workspace, trần chi tiêu theo bậc |
| [9] | [OpenAI: Pricing](https://developers.openai.com/api/docs/pricing) | Giá GPT-6, GPT-6.1, GPT-5.6, giá long context, batch, flex, fine-tuning, khuyến mãi, phụ phí xử lý theo vùng |
| [10] | [OpenAI: Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) | Hệ số ghi và đọc cache, TTL 30 phút và 12 giờ, phép tính hoà vốn, khác biệt theo thế hệ model, tràn sang máy khác trên khoảng 15 request/phút |
| [11] | [OpenAI: Batch API](https://developers.openai.com/api/docs/guides/batch) | Giảm 50%, 24 giờ, giới hạn batch, batch hết hạn |
| [12] | [OpenAI: Flex processing](https://developers.openai.com/api/docs/guides/flex-processing) | `service_tier: "flex"`, giá batch, `429 Resource Unavailable` không tính tiền |
| [13] | [OpenAI: Fast mode](https://developers.openai.com/api/docs/guides/priority-processing) | Đổi tên từ Priority processing, giá gấp đôi với GPT-5.6 Sol, ngày khuyến mãi |
| [14] | [OpenAI: Reasoning models](https://developers.openai.com/api/docs/guides/reasoning) | Reasoning token tính như output, `max_output_tokens`, response `incomplete` |
| [15] | [OpenAI: Counting tokens](https://developers.openai.com/api/docs/guides/token-counting) | Endpoint đếm token, giới hạn của `tiktoken` |
| [16] | [Gemini API: Pricing](https://ai.google.dev/gemini-api/docs/pricing) | Giá Gemini 4 Argon, 3.8 Flash, 3.5 Flash-Lite, ngày tăng giá, giá thinking, phí lưu cache, grounding |
| [17] | [Gemini API: Context caching](https://ai.google.dev/gemini-api/docs/caching) | Implicit caching, ngưỡng tối thiểu, quy tắc 2.050 token với Gemini 4 |
| [18] | [Gemini API: Batch API](https://ai.google.dev/gemini-api/docs/batch-api) | Giảm 50%, mục tiêu 24 giờ, giới hạn file, cache trong batch |
| [19] | [Hugging Face: LLM Inference at scale with TGI](https://huggingface.co/blog/martinigoyanes/llm-inference-at-scale-with-tgi) | Prefill compute-bound, decode memory-bound, ví dụ tính trên A100 |
| [20] | [Schroeder et al., vCache: Verified Semantic Prompt Caching (arXiv 2502.03771)](https://arxiv.org/abs/2502.03771) | Ngưỡng tĩnh không đảm bảo tỉ lệ lỗi; ngưỡng riêng từng mục |
| [21] | [Zhu et al., Efficient Prompt Caching via Embedding Similarity (arXiv 2402.01173)](https://arxiv.org/abs/2402.01173) | AUC 0,51 của embedding có sẵn, 0,81 sau fine-tune |
| [22] | [Bang, GPTCache (NLP-OSS 2023)](https://aclanthology.org/2023.nlposs-1.24/) | Giới hạn của semantic cache với embedding hiện có |
| [23] | [AWS Database Blog: ElastiCache as a semantic cache with Amazon Bedrock](https://aws.amazon.com/blogs/database/lower-cost-and-latency-for-ai-using-amazon-elasticache-as-a-semantic-cache-with-amazon-bedrock/) | Bảng ngưỡng, tỉ lệ hit, độ chính xác, chi phí trên 63.796 câu hỏi |
| [24] | [OWASP Top 10 for LLM Applications: LLM10:2025 Unbounded Consumption](https://genai.owasp.org/llmrisk/llm102025-unbounded-consumption/) | Rủi ro tiêu thụ không giới hạn, biện pháp |
| [25] | [openai/tiktoken](https://github.com/openai/tiktoken) | Bản 0.14.0, dùng để đo token tiếng Việt và tiếng Anh |
| [26] | [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) | Bản 2.6.0: các trường của `Usage` và `BetaUsage.iterations` |
