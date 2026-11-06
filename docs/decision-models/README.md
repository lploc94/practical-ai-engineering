# Decision models: khi AI chỉ cần chọn, không cần viết

Một shop online nhận vài chục nghìn tin nhắn hỗ trợ mỗi ngày. Với mỗi tin, hệ thống cần ba quyết định: chuyển cho đội nào, có khẩn cấp không, khách có đòi hoàn tiền không. Cách đầu tiên hầu hết đội đều thử là gọi LLM với structured output: mô tả ba field trong JSON Schema, nhận object về, rẽ nhánh trong code. Cách này chạy được, và sau vài tuần thì lộ ra ba vấn đề:

- Mỗi tin là một lượt sinh văn bản đầy đủ: model đọc prompt, sinh từng token của JSON, mất từ một đến vài giây. Nhân với vài chục nghìn tin, hoá đơn và độ trễ đều lớn hơn mức một quyết định "đội nào" đáng phải trả.
- Bạn thêm field `confidence` để biết khi nào nên chuyển cho người. Model ghi 0.9 cho tin rõ ràng và cũng ghi 0.9 cho tin mơ hồ, vì con số đó là văn bản model viết ra, không phải xác suất model tính.
- Gửi lại đúng một tin hai lần, thỉnh thoảng nhận về hai đội khác nhau. Khi khách khiếu nại, bạn không tái hiện được quyết định cũ.

**Decision model** (TypeSafe gọi là *System One model*) giải quyết loại việc này theo cách khác. Bạn gửi nội dung cần xét (gọi là *state*) cùng một danh sách câu hỏi có đáp án định sẵn: có hay không, chọn một trong các đội, xếp vào mức nào trên thang. Model không viết gì cả. Nó trả về đáp án kèm phân phối xác suất trên đúng các đáp án bạn đưa, và xác suất đó được huấn luyện để khớp với tần suất đúng thực tế. Code của bạn đọc con số, so với ngưỡng, rồi quyết định.

Nhóm model này mới xuất hiện từ tháng 9/2026: TypeSafe ra mắt Jev ngày 15/9, Cloudflare ra Clef ngày 1/10 với API tương thích Jev và weights mở, OpenAI có Decisions API, và vài chục model open-weight theo sau. Chúng không thay LLM. Chúng thay *những lượt gọi LLM chỉ để chọn*, và làm cho các quyết định đó rẻ, nhanh, lặp lại được và có độ bất định đọc được.

Bạn có thể đọc theo thứ tự nếu mới làm quen, hoặc mở thẳng phần cần:

- **Nền tảng — Phần 1:** decision model khác LLM ở đâu, bên trong hoạt động thế nào, bốn loại câu hỏi, xác suất và confidence, những việc model làm kém.
- **Tra cứu — Phần 2:** API, giới hạn, giá và khác biệt của Jev, Clef, OpenAI Decisions API, các model khác và cách chạy local.
- **Ứng dụng — Phần 3:** 14 bài toán thường gặp, mỗi bài có cách đặt câu hỏi, code định tuyến và lưu ý riêng: phân loại ticket, guardrail cho LLM, lọc passage RAG, kiểm tra trích dẫn, cascade trích xuất, chọn tool, routing model, phân loại theo taxonomy, ghép thực thể, chấm điểm đa tiêu chí, feature cho ML, đọc ảnh và tài liệu scan.
- **Thiết kế và vận hành — Phần 4:** viết state và câu hỏi, các pattern kết hợp, đặt ngưỡng, eval, chi phí, bảo mật. Khi cần tìm nguyên nhân một lỗi cụ thể, xem [bảng triệu chứng ở §4.9](#49-failure-modes-thường-gặp).

> Bản tháng 11/2026. Thuật ngữ kỹ thuật và tên field giữ nguyên tiếng Anh để tiện đối chiếu API. Đây là một hạng mục rất mới: tên model, giá và giới hạn thay đổi theo tuần. Các con số trong tài liệu ghi kèm nguồn và thời điểm; mục chưa xác minh được từ tài liệu chính thức được đánh dấu tại chỗ.

Tài liệu liên quan: [Structured Output](../structured-output/) (khi cần sinh object thay vì chọn đáp án), [Tool Use](../tool-use/) (khi model cần gọi hàm có tham số tự do), [Embeddings](../embeddings/) (retrieval trước khi rerank).

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 System One và System Two](#11-system-one-và-system-two)
  - [1.2 Hợp đồng đầu vào và đầu ra](#12-hợp-đồng-đầu-vào-và-đầu-ra)
  - [1.3 Bên trong: prefill rồi chấm điểm các lựa chọn](#13-bên-trong-prefill-rồi-chấm-điểm-các-lựa-chọn)
  - [1.4 Bốn loại câu hỏi](#14-bốn-loại-câu-hỏi)
  - [1.5 Xác suất, calibration và confidence](#15-xác-suất-calibration-và-confidence)
  - [1.6 Câu hỏi độc lập và chạy song song](#16-câu-hỏi-độc-lập-và-chạy-song-song)
  - [1.7 Những việc decision model làm kém](#17-những-việc-decision-model-làm-kém)
  - [1.8 So với classifier, embedding và LLM](#18-so-với-classifier-embedding-và-llm)
- [Phần 2 — Decision model theo nhà cung cấp](#phần-2--decision-model-theo-nhà-cung-cấp)
  - [2.1 Bức tranh chung](#21-bức-tranh-chung)
  - [2.2 TypeSafe Jev](#22-typesafe-jev)
  - [2.3 Cloudflare Clef](#23-cloudflare-clef)
  - [2.4 OpenAI Decisions API](#24-openai-decisions-api)
  - [2.5 Microsoft Decision-1 và Liquid AI d1](#25-microsoft-decision-1-và-liquid-ai-d1)
  - [2.6 Model open-weight](#26-model-open-weight)
  - [2.7 Chạy local](#27-chạy-local)
  - [2.8 Thư viện và tích hợp](#28-thư-viện-và-tích-hợp)
  - [2.9 Benchmark](#29-benchmark)
- [Phần 3 — Ứng dụng và hướng dẫn](#phần-3--ứng-dụng-và-hướng-dẫn)
  - [3.0 Cách dùng phần này](#30-cách-dùng-phần-này)
  - [3.1 Phân loại và định tuyến ticket hỗ trợ](#31-phân-loại-và-định-tuyến-ticket-hỗ-trợ)
  - [3.2 Guardrail cho input và output của LLM](#32-guardrail-cho-input-và-output-của-llm)
  - [3.3 Moderation nội dung thời gian thực](#33-moderation-nội-dung-thời-gian-thực)
  - [3.4 Lọc passage giữa retrieval và generation](#34-lọc-passage-giữa-retrieval-và-generation)
  - [3.5 Rerank kết quả tìm kiếm](#35-rerank-kết-quả-tìm-kiếm)
  - [3.6 Kiểm tra trích dẫn](#36-kiểm-tra-trích-dẫn)
  - [3.7 Cascade trích xuất dữ liệu](#37-cascade-trích-xuất-dữ-liệu)
  - [3.8 Trích xuất giá trị có miền đóng](#38-trích-xuất-giá-trị-có-miền-đóng)
  - [3.9 Chọn function và tham số](#39-chọn-function-và-tham-số)
  - [3.10 Chọn skill hoặc tool cho agent](#310-chọn-skill-hoặc-tool-cho-agent)
  - [3.11 Routing giữa các LLM](#311-routing-giữa-các-llm)
  - [3.12 Phân loại theo taxonomy lớn](#312-phân-loại-theo-taxonomy-lớn)
  - [3.13 Ghép thực thể và khử trùng lặp](#313-ghép-thực-thể-và-khử-trùng-lặp)
  - [3.14 Chấm điểm đa tiêu chí](#314-chấm-điểm-đa-tiêu-chí)
  - [3.15 Biến văn bản thành feature cho ML](#315-biến-văn-bản-thành-feature-cho-ml)
  - [3.16 Ảnh và tài liệu scan](#316-ảnh-và-tài-liệu-scan)
- [Phần 4 — Thiết kế và vận hành](#phần-4--thiết-kế-và-vận-hành)
  - [4.1 Viết state](#41-viết-state)
  - [4.2 Viết câu hỏi và criteria](#42-viết-câu-hỏi-và-criteria)
  - [4.3 Các pattern kết hợp](#43-các-pattern-kết-hợp)
  - [4.4 Đặt ngưỡng](#44-đặt-ngưỡng)
  - [4.5 Eval và so sánh model](#45-eval-và-so-sánh-model)
  - [4.6 Chi phí, độ trễ và rate limit](#46-chi-phí-độ-trễ-và-rate-limit)
  - [4.7 Version và chuyển provider](#47-version-và-chuyển-provider)
  - [4.8 Bảo mật](#48-bảo-mật)
  - [4.9 Failure modes thường gặp](#49-failure-modes-thường-gặp)
  - [4.10 Checklist trước khi lên production](#410-checklist-trước-khi-lên-production)
- [Nguồn tham khảo](#nguồn-tham-khảo)

---

## Tra cứu nhanh

### Những quyết định cần chốt trước khi viết code

1. **Đáp án có thuộc một tập đóng không?** Có/không, một trong N lựa chọn, một mức trên thang, một vùng trên ảnh: dùng decision model. Cần văn bản, giải thích, code hay một giá trị tự do như tên người: dùng LLM ([§1.8](#18-so-với-classifier-embedding-và-llm)).
2. **Mỗi câu hỏi là một phán đoán.** "Ticket này ưu tiên mức nào" giấu ba phán đoán: bug nặng đến đâu, khách bực đến đâu, báo cáo đủ chi tiết chưa. Hỏi ba câu, cộng có trọng số trong code ([§3.14](#314-chấm-điểm-đa-tiêu-chí)).
3. **Mọi câu hỏi về cùng một state đi chung một request**, kể cả câu chỉ cần cho một số trường hợp. Câu hỏi chạy song song và state chỉ tính tiền một lần ([§1.6](#16-câu-hỏi-độc-lập-và-chạy-song-song)).
4. **Ngưỡng nằm trong code, đặt từ dữ liệu có nhãn, theo rủi ro của từng hành động.** Vùng không chắc chắn phải có đường xử lý riêng, không làm tròn sang một phía ([§4.4](#44-đặt-ngưỡng)).
5. **Toán, đếm và so sánh ngày tháng để code làm.** Model đọc ngữ nghĩa tốt, tính toán kém ([§1.7](#17-những-việc-decision-model-làm-kém)).
6. **Câu hỏi chọn một (Choice) luôn chọn một đáp án**, kể cả khi không đáp án nào đúng. Kèm một câu yes/no hỏi "có đáp án nào phù hợp không" ([§4.3](#43-các-pattern-kết-hợp)).
7. **Pin version model** khi đã tune ngưỡng. Alias như `jev-latest` có thể trỏ sang model mới mà code không đổi ([§4.7](#47-version-và-chuyển-provider)).
8. **State là dữ liệu không tin cậy.** Nội dung trong state có thể cố tình kéo đáp án; decision model không phải ranh giới bảo mật ([§4.8](#48-bảo-mật)).

### Decision model hay LLM?

| Bạn cần | Dùng | Vì sao |
|---|---|---|
| Chọn đội, nhãn, intent, ngôn ngữ từ danh sách | Decision model (Choice) | Đáp án luôn thuộc danh sách, có xác suất cho từng lựa chọn |
| Biết một điều kiện đúng hay sai: có PII, có đòi hoàn tiền, có prompt injection | Decision model (yes/no) | Xác suất là tín hiệu để đặt ngưỡng |
| Xếp vào thang mức: mức nghiêm trọng, độ liên quan, độ bực bội | Decision model (Score) | Trả vị trí trên thang và phân phối giữa các mức |
| Trích tên, số tiền, địa chỉ từ văn bản | LLM hoặc regex tạo ứng viên, decision model chọn | Decision model không sinh giá trị mới ([§3.8](#38-trích-xuất-giá-trị-có-miền-đóng)) |
| Viết câu trả lời, tóm tắt, giải thích lý do | LLM | Decision model không sinh văn bản |
| Suy luận nhiều bước, lập kế hoạch, toán | Reasoning model | Decision model dành cho phán đoán nhanh, một bước |
| Gọi hàm có tham số số hoặc chuỗi tự do | LLM tool use | Decision model chỉ điền được tham số có miền đóng ([§3.9](#39-chọn-function-và-tham-số)) |

### So sánh nhanh (snapshot tháng 11/2026)

| | TypeSafe Jev | Cloudflare Clef | OpenAI Decisions API |
|---|---|---|---|
| Model | `jev-1.14.0`, `jev-1.13.0`; alias `jev-latest`, `jev-preview` | `clef`, `clef-flash`, `clef-omni` | `gpt-6-luna` (duy nhất) |
| Endpoint | `POST https://api.typesafe.ai/v1/systemone` | Workers AI `…/ai/run/@cf/cloudflare/clef` | `POST /v1/decisions` |
| Weights | Đóng | Mở, Apache 2.0 | Đóng |
| Giá input / 1M token | $0.042 | $0.24 (`clef`), $0.038 (`clef-flash`), $0.15 (`clef-omni`) | $0.10 (chưa cache) |
| Output | Miễn phí | Miễn phí | Không tính phí |
| Loại câu hỏi | Choice, Score, Noul, Box | Như Jev (tương thích API) | `predicate`, `choice`, `score` |
| Context | 64k / request, 32k / câu hỏi; 262k khi bật header long context | 64k (`clef`, `clef-omni`), 24k (`clef-flash`) | Không nêu riêng trong guide |
| Số câu hỏi / request | Không giới hạn số lượng, chỉ giới hạn context | Tối đa 64 | 1–400 |
| Input | Text, JSON, mảng; ảnh PNG/JPEG (thử nghiệm) | Text, ảnh; Clef-omni thêm audio, video | Text, ảnh, JSON, lịch sử hội thoại kèm tool call |
| Prompt caching | Không đề cập | Không đề cập | Có, prefix tối thiểu 1.024 token |
| Fine-tune | Không, chỉnh qua request | Có, qua đội FDE và nền tảng RL của Cloudflare | Không đề cập |

Chi tiết và nguồn ở Phần 2. Microsoft, Liquid và các model open-weight ở [§2.5](#25-microsoft-decision-1-và-liquid-ai-d1)–[§2.7](#27-chạy-local).

### Chọn loại câu hỏi

| Loại | Tên ở Jev / Clef | Tên ở OpenAI | Hỏi gì | Trả về |
|---|---|---|---|---|
| Yes/no | `noul` | `predicate` | Điều kiện này có đúng không? | Một xác suất từ 0 đến 1 |
| Chọn một | `choice` | `choice` | Đáp án nào trong danh sách? | Đáp án, xác suất từng đáp án, `confidence` |
| Thang mức | `score` | `score` | Nằm ở mức nào trên thang có thứ tự? | Vị trí (có thể lẻ), xác suất từng mức, `confidence` |
| Vùng trên ảnh | `box` | Không có | Vật X nằm đâu trong ảnh? | Bốn cạnh, mỗi cạnh là tỉ lệ 0–1 của chiều rộng/cao |

Nếu hai loại đều có vẻ hợp, chọn loại mà code dùng được trực tiếp: Choice giữa `refund`, `rebook`, `information` khớp với ba nhánh code; Score về mức bực bội khớp với một ngưỡng; yes/no khớp với một `if` [2].

---

# Phần 1 — Nền tảng

Chỉ cần đổi endpoint là chạy được ví dụ trong docs. Nhưng muốn biết vì sao một câu hỏi trả 0.5 thay vì 0.9, vì sao thêm option `other` lại làm kết quả tệ hơn, hay vì sao model chấm sai một phép so sánh ngày tháng đơn giản, cần hiểu model được thiết kế để làm gì và không làm gì.

## 1.1 System One và System Two

Tên "System One" lấy từ *Thinking, Fast and Slow* của Daniel Kahneman. System 1 là kiểu nghĩ nhanh, trực giác: nhìn một email và biết ngay đó là quảng cáo. System 2 chậm và có chủ đích: đọc một hợp đồng và lần ra điều khoản mâu thuẫn [1].

LLM hiện đại, nhất là reasoning model, được tối ưu cho System 2: suy luận nhiều bước, viết, lập kế hoạch. Decision model nhắm vào System 1: những phán đoán mà một người có kiến thức đưa ra trong một giây nếu có đúng ngữ cảnh. TypeSafe dùng ví dụ: "Tin nhắn này có khẩn cấp không?" là câu hỏi tốt; "Phân tích tin nhắn và quyết định hành động tốt nhất" thì không, vì nó cần suy luận chậm và là dấu hiệu nên tách thành nhiều câu nhỏ [2].

Hệ quả cho kiến trúc: decision model không thay agent. TypeSafe mô tả ba kiểu kiến trúc [3]:

- **Phần mềm truyền thống:** code là một cây quyết định ghép từ các primitive đơn giản và đáng tin.
- **LLM agent:** model đọc chỉ dẫn và tự chọn bước tiếp theo. Hợp khi có người theo dõi, nhưng mỗi vòng lặp là thêm một cơ hội đi chệch.
- **Phần mềm dùng AI làm primitive:** code giữ luồng điều khiển và mọi tác dụng phụ; model chỉ xuất hiện ở chỗ cần "common sense lập trình được" trên dữ liệu không cấu trúc, mỗi lần một phán đoán hẹp.

Decision model là công cụ cho kiểu thứ ba. Nó cũng dùng được bên trong kiểu thứ hai, làm các quyết định nhỏ cho agent: chọn skill, kiểm tra tool call, lọc context ([§3.10](#310-chọn-skill-hoặc-tool-cho-agent)).

## 1.2 Hợp đồng đầu vào và đầu ra

Mọi decision model hiện có đều theo cùng một hợp đồng, chỉ khác tên field:

```text
request:
  model      jev-1.14.0 / clef / gpt-6-luna
  state      nội dung cần xét: chuỗi, object JSON, mảng, ảnh, hoặc lịch sử hội thoại
  questions  danh sách câu hỏi, mỗi câu có: tên do bạn đặt, loại, instructions, đáp án cho phép

response:
  answers    mỗi câu hỏi một đáp án có kiểu: giá trị được chọn + xác suất + (confidence)
  usage      token input (và output, nếu provider báo)
```

Ví dụ với Jev, phân loại một ticket bằng ba câu hỏi trong một request [4]:

```json
{
  "model": "jev-1.14.0",
  "state": "Our API integration started returning 500 errors on every request about 20 minutes ago, and we can't process any customer orders until this is fixed.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payment or subscription issues",
        "technical": "Bugs or integration problems",
        "sales": "Pricing or account questions"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated does the customer appear?",
      "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]
    },
    "is_urgent": {
      "type": "noul",
      "instructions": "Does the message convey urgency or time-sensitivity?"
    }
  }
}
```

Response trả về một đáp án cho mỗi tên câu hỏi. Dạng của từng loại đáp án (ví dụ lấy từ docs Jev [5][6]):

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "returns",
      "confidence": 0.42,
      "probabilities": {"shipping": 0.04, "billing": 0.35, "returns": 0.61}
    },
    "bug_severity": {
      "type": "score",
      "score": 1.43,
      "confidence": 0.35,
      "legend": {"0": "Cosmetic", "1": "Broken but workaround exists", "2": "Blocking"},
      "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43}
    },
    "is_human_escalation": {"type": "noul", "noul": 0.99}
  },
  "usage": {"input_tokens": 589, "output_tokens": 212}
}
```

Ba điểm khác với structured output của LLM:

**Đáp án không bao giờ nằm ngoài tập bạn đưa.** Model trả phân phối trên các option, không trả chuỗi. Không có lỗi parse, không có `"anger": "khá cao"` khi bạn cần số, không có field sai tên. Với structured output qua constrained decoding, bạn cũng có đảm bảo về format, nhưng model vẫn chọn token theo cách sinh văn bản và không cho bạn phân phối xác suất có ý nghĩa ([Structured Output §1.6](../structured-output/#16-đúng-schema-chưa-phải-là-đúng)).

**Tên câu hỏi không được gửi cho model.** `department` hay `q1` chỉ để code đọc kết quả. Mọi thứ model cần biết phải nằm trong `instructions` và `criteria` [2]. Ngược lại, tên option (`billing`, `returns`) *có* được gửi, nên đặt tên option có nghĩa.

**Output gần như miễn phí.** Jev và OpenAI Decisions API đều không tính tiền output token [7][8]. Chi phí chủ yếu là input, nên đưa thêm câu hỏi vào request rẻ hơn nhiều so với gửi thêm request.

## 1.3 Bên trong: prefill rồi chấm điểm các lựa chọn

TypeSafe không công bố kiến trúc của Jev [9]. Cloudflare mô tả Clef khá chi tiết, và đó là hình dung tốt nhất hiện có về cách loại model này hoạt động [10]:

1. **Backbone là một LLM có sẵn.** Clef dùng Qwen3.8-27B, Clef-flash dùng Qwen3.5-9B. Blog mô tả phần huấn luyện thêm là routing head và LoRA bên trên backbone; model card trên Hugging Face chỉ ghi là "post-trained from" Qwen, không nói rõ backbone có bị đóng băng hoàn toàn không.
2. **Chỉ chạy prefill.** Model đọc state và câu hỏi một lần, tạo biểu diễn nội bộ, không sinh token nào. Đây là lý do output miễn phí và độ trễ thấp: phần đắt nhất của LLM là decode từng token, và decision model bỏ hẳn bước đó.
3. **Chấm điểm song song mọi lựa chọn hợp lệ.** Một "routing head" cho mỗi option trích phần ngữ cảnh liên quan đến nó, cho các field cross-attend với nhau và với payload gốc, rồi chấm điểm. Bước quyết định không autoregressive.
4. **Huấn luyện cho calibration.** Cloudflare huấn luyện routing head cùng LoRA rank 256, với cross-entropy có label smoothing cho output hợp lệ và Brier loss để tinh chỉnh calibration, trên dữ liệu tổng hợp hoán vị thứ tự field, prompt và schema. Thêm một mục tiêu RL gọi là RLCD (*Reinforcement Learning for Calibrated Decisions*), cho điểm một phần khi chọn mức liền kề trên thang có thứ tự.

Trước Clef, Cloudflare thử một hướng đơn giản hơn: lấy DiffusionGemma và đọc logprob mà LLM sinh ra cho từng lựa chọn [10]. Đây cũng là cách nhiều người tự làm "decision model nghèo" với LLM thường: hỏi một câu, đọc logprob của token đầu tiên. Khác biệt là logprob của một LLM đã qua RLHF không được tối ưu để khớp với tần suất đúng.

TypeSafe giải thích vì sao họ không dùng RLHF [9]. RLHF dạy model nói điều người thích nghe; nó có thể thưởng cho sự chiều lòng và cho những câu bịa nghe tự tin. RLHF cũng gây *mode dropping*: model dồn xác suất vào một kiểu output và bỏ các khả năng khác. Cả hai đều xấu khi bạn cần một xác suất phản ánh độ bất định thật. RLCD tối ưu cho một hợp đồng khác: model không sinh văn bản, chỉ trả quyết định và xác suất, và xác suất cao hơn phải tương ứng với khả năng đúng cao hơn.

Hệ quả thực tế:

- Độ trễ chủ yếu phụ thuộc độ dài state, gần như không phụ thuộc số câu hỏi.
- Thêm option vào một câu Choice tốn vài token mỗi option [5], không tốn thêm bước decode.
- Model không có "chỗ để suy nghĩ". Câu hỏi cần nhiều bước suy luận sẽ được trả lời bằng trực giác của một lần prefill ([§1.7](#17-những-việc-decision-model-làm-kém)).

## 1.4 Bốn loại câu hỏi

### Yes/no: `noul` / `predicate`

Trả một số từ 0 đến 1: xác suất câu trả lời là "có". Gần 1 là có chắc chắn, gần 0 là không chắc chắn, gần 0.5 là model không phân định được [6].

Docs Jev có một bảng minh hoạ rất rõ cho câu hỏi "Khách có đang muốn gặp người thật không?" trên `jev-1.13.0` [6]:

| Tin nhắn | `noul` |
|---|---|
| Thanks, that fixed it! | 0.02 |
| How do I reset my password? | 0.07 |
| I need this sorted today, whatever it takes. | 0.26 |
| Are you a bot? | 0.40 |
| Is there any way to speak to someone about my invoice? | 0.84 |
| I have asked three times now. Can I please just talk to a real person? | 0.99 |

"Are you a bot?" ám chỉ muốn gặp người mà không nói ra, và model chia gần đều. Đây đúng là loại tin cần ngưỡng trong code, không phải loại tin model nên tự quyết.

Một lỗi hay gặp: dùng yes/no để đo mức độ. "Ứng viên có giỏi Python không?" trả 0.81 cho người dùng Python hằng ngày hai năm và 0.92 cho người dùng tám năm. Con số đó là xác suất mệnh đề "giỏi" đúng, không phải thang đo kinh nghiệm; khoảng cách giữa hai ứng viên không có nghĩa gì. Muốn đo mức, dùng Score [6].

### Chọn một: `choice`

Trả option có xác suất cao nhất, phân phối trên mọi option (tổng bằng 1) và `confidence` [5]. Jev nhận tối đa 255 option mỗi câu [5]; cookbook của TypeSafe ghi Choice "hoạt động ổn định đến khoảng 240 option" [11]. OpenAI dùng mảng `choices` với `value` và `description` [8].

Choice **luôn chọn một option**, kể cả khi không option nào đúng. Hỏi tin nhắn về chiếc đèn bàn thuộc sản phẩm nào trong ba sản phẩm bàn, ghế, tay đỡ màn hình, model chọn "bàn" với xác suất 0.96, vì bàn gần đèn bàn nhất. Cách xử lý ở [§4.3](#43-các-pattern-kết-hợp).

### Thang mức: `score`

Bạn đưa một mảng các mức có thứ tự, từ thấp đến cao; Jev nhận tối đa 10 mức [12], OpenAI nhận 2–10 mức [8]. Mức được đánh số từ 0. Đáp án gồm:

- `probabilities`: xác suất của từng mức.
- `score`: trung bình có trọng số của số thứ tự mức. Với xác suất (0, 0.57, 0.43), score là 0×0 + 1×0.57 + 2×0.43 = 1.43.
- `confidence`: độ tập trung của phân phối.

Score là một vị trí, không phải đại lượng đo được. Score 1.43 không có nghĩa "43% khách không có cách khắc phục". Hai phân phối khác nhau có thể cùng score: 1.0 có thể là toàn bộ xác suất ở mức 1, hoặc một nửa ở mức 0 và một nửa ở mức 2. Luôn đọc `probabilities` và `confidence` cùng với `score` [12].

Mỗi mức được đánh giá riêng với state. Model không thấy số thứ tự hay mức bên cạnh, nên mô tả kiểu "nặng hơn mức trước" vô nghĩa với nó. Docs Jev cho thấy: cùng một báo cáo "nút export lệch vài pixel", với ba mức có mô tả tình huống cụ thể thì score là 0.0 với confidence 1.0; với ba mức chỉ ghi số thì model chia xác suất giữa mức 0 và 1 [12].

### Vùng trên ảnh: `box`

Chỉ Jev 1.14 có. Trả bốn cạnh `top`, `bottom`, `left`, `right`, mỗi cạnh là tỉ lệ 0–1 của chiều cao hoặc rộng ảnh, kèm `confidence` [2]. Giống Choice, Box luôn trả một vị trí, nên cần kèm một câu yes/no "trong ảnh có X không". Ảnh và Box đang ở trạng thái "dreamer-ready": TypeSafe ghi rõ chưa đủ chuẩn production, format và kết quả có thể đổi giữa các bản, và mọi request có ảnh phải gửi header `TypeSafe-Danger-Zone-Images: true` [13].

## 1.5 Xác suất, calibration và confidence

### Calibration nghĩa là gì

Một model được calibrate tốt thì trên nhiều dự đoán, những đáp án được gán xác suất 0.2 đúng khoảng 20% số lần, những đáp án 0.8 đúng khoảng 80% [9]. Đây là tính chất trên nhóm dự đoán, không phải đảm bảo cho từng đáp án. Một đáp án 0.95 vẫn có thể sai; điều calibration hứa là nếu bạn lấy 1.000 đáp án 0.95, khoảng 50 cái sẽ sai.

Đó là lý do xác suất của decision model dùng làm ngưỡng được, còn "confidence" do LLM tự viết thì không. Bạn có thể kiểm tra calibration trên dữ liệu của mình ([§4.5](#45-eval-và-so-sánh-model)), và nên làm, vì calibration trên benchmark của nhà cung cấp chưa chắc giữ nguyên trên miền của bạn.

### `confidence` khác `probability`

`confidence` là một số tóm tắt *hình dạng* của phân phối, không phải xác suất đáp án đúng. Jev công bố công thức [14]:

**Choice** với *n* option, *p*<sub>max</sub> là xác suất của option được chọn:

```text
confidence = (p_max − 1/n) / (1 − 1/n)
```

Phân phối đều cho 0, toàn bộ xác suất ở một option cho 1. Vì chỉ phụ thuộc *p*<sub>max</sub>, (0.6, 0.3, 0.1) và (0.6, 0.2, 0.2) có cùng confidence 0.4. Cùng confidence 0.5 ứng với *p*<sub>max</sub> khác nhau tuỳ số option:

| Số option | *p*<sub>max</sub> | `confidence` |
|---|---|---|
| 2 | 0.75 | 0.5 |
| 3 | 0.67 | 0.5 |
| 10 | 0.55 | 0.5 |

**Score** tính thêm khoảng cách giữa các mức: xác suất ở mức kề bên làm giảm confidence ít hơn xác suất ở mức xa. Với ba mức, (0, 0.5, 0.5) có confidence 0.25 vì model phân vân giữa hai mức kề nhau; (0.5, 0, 0.5) có confidence 0 vì model phân vân giữa hai đầu thang. Công thức đầy đủ:

```text
m       = mức có xác suất cao nhất
spread  = Σ p_i · |i − m|
MAD_unif = (1/n) · Σ |i − (n−1)/2|
confidence = max(0, 1 − spread / MAD_unif)
```

**Yes/no** không có `confidence` riêng; chính xác suất đã chứa độ bất định. Nếu muốn một số cùng thang với Choice, dùng `|2p − 1|` [14].

OpenAI Decisions API cũng trả `confidence` cho `choice` và `score` [8], nhưng guide không công bố công thức (chưa xác minh được là giống Jev).

### Ngưỡng nên đặt trên số nào

Theo khuyến nghị của TypeSafe [14]:

- **Yes/no:** đặt ngưỡng trực tiếp trên xác suất. Điểm khởi đầu: trên 0.7 là có, dưới 0.3 là không, khoảng giữa có đường xử lý riêng.
- **Choice:** đặt ngưỡng trên xác suất của option mà code sắp hành động theo. "Chỉ chuyển tiền khi P(approve_transfer) > 0.9" đọc được ngay là "chấp nhận dưới 10% rủi ro chọn sai". Với số option cố định, confidence chỉ là biến đổi của *p*<sub>max</sub>, nên kiểm tra cả hai không thêm thông tin. Muốn biết option thứ hai có sát không, dùng tỉ lệ *p*<sub>max</sub> / *p*<sub>second</sub>.
- **Score:** kiểm tra confidence trước để biết đáp án đã "ổn định" chưa, rồi mới rẽ nhánh theo score. Nếu quyết định phụ thuộc vào việc đáp án nằm bên nào của một ranh giới, cộng xác suất các mức ở phía đó thay vì dùng score, vì một đáp án chia đều giữa hai mức kề nhau vẫn có confidence khá cao ([§4.4](#44-đặt-ngưỡng)).

## 1.6 Câu hỏi độc lập và chạy song song

Các câu hỏi trong một request cùng thấy một state, nhưng được đánh giá độc lập: đáp án câu này không trở thành ngữ cảnh ẩn cho câu khác [2]. Hai hệ quả:

**Thêm hay bớt câu hỏi không làm thay đổi đáp án các câu khác.** TypeSafe kiểm tra điều này trên 13 câu hỏi về bài Wikipedia GDPR (khoảng 54.000 ký tự): gửi 13 câu trong một request và gửi 13 request mỗi request một câu, lặp 5 lần mỗi cách. Phần lớn đáp án giống hệt nhau qua cả 5 lần ở cả hai cách (độ lệch chuẩn đúng bằng 0); hai câu có dao động nhỏ, và dao động đó như nhau ở cả hai cách. Gộp vào một request rẻ hơn 12,2 lần và nhanh hơn 10 lần, vì bài viết chỉ bị gửi và tính tiền một lần [15].

**Hỏi cả những câu có thể không cần.** TypeSafe gọi đây là *speculative fan-out*: hỏi loại ticket, đồng thời hỏi luôn mức nghiêm trọng của bug và khách có đòi hoàn tiền không; nếu ticket không phải bug, code bỏ qua đáp án mức nghiêm trọng [16]. Tiết kiệm một round trip trong trường hợp cần, tốn vài token câu hỏi trong trường hợp không cần.

Chỉ tách thành request thứ hai khi code thật sự không thể dựng request đó nếu chưa có đáp án đầu: cần đáp án để lấy thêm dữ liệu vào state, để quyết định state gồm những gì, hoặc để chọn option cho câu tiếp theo [2]. Ví dụ ở [§3.10](#310-chọn-skill-hoặc-tool-cho-agent) (chọn 3 skill rồi đọc kỹ 3 skill đó) và [§3.12](#312-phân-loại-theo-taxonomy-lớn) (đi từng tầng của cây).

## 1.7 Những việc decision model làm kém

TypeSafe giữ một trang "model jaggedness" liệt kê các điểm yếu đã biết của Jev [17]. Phần lớn áp dụng cho mọi decision model vì đến từ cùng thiết kế: một lần prefill, không có bước suy luận.

| Điểm yếu | Biểu hiện | Cách xử lý |
|---|---|---|
| Đọc theo nghĩa đen | Trả lời đúng câu bạn viết, không phải câu bạn định hỏi; phủ định và điều kiện ngầm bị đọc nguyên văn | Viết điều kiện chính xác; đưa trường hợp biên vào criteria; khi phải giải thích "ý tôi là…", phần giải thích đó là nửa còn thiếu của câu hỏi |
| Toán và số | Không đếm được ký tự, số lần xuất hiện, số phần tử; không so được hai mã màu hex | Đếm và tính trong code; hỏi từng phần tử rồi cộng trong code; đổi số thành tên nhóm trước khi hỏi |
| Ngày tháng | Không so được ngày nào trước, cách nhau bao lâu, có nằm trong khoảng không | Cho model đọc từng thành phần ngày, code ráp và so ([§3.8](#38-trích-xuất-giá-trị-có-miền-đóng)) |
| Nhiều tầng gián tiếp | Phủ định kép, "thuộc tính của thuộc tính", câu cần nhiều bước suy luận | Viết trực tiếp; chỉ thẳng phần state liên quan bằng đường dẫn |
| State lớn nhiều chi tiết thừa | Độ chính xác giảm khi state chứa nhiều nội dung không liên quan | Lọc trong code trước; chỉ gửi field câu hỏi cần |
| Nội dung đối kháng | Chỉ dẫn chèn vào state, cách đóng khung gây hiểu lầm, văn bản tự biện hộ cho nhãn của nó có thể kéo đáp án | Criteria rõ ràng; test các ca đối kháng trước khi deploy ([§4.8](#48-bảo-mật)) |
| Instructions và criteria mâu thuẫn | Ví dụ câu yes/no mà `true` mô tả nghĩa "không" | Coi criteria là phần mở rộng của instructions, viết cùng chiều |
| Thứ tự option | Có trường hợp nghiêng về option đứng đầu | Đảo thứ tự option khi eval và kiểm tra đáp án có ổn định không |
| Sinh văn bản | Ép sinh bằng cách nối chuỗi các câu Choice vừa chậm vừa kém | Dùng LLM |

Không có tài liệu chính thức nào nói về chất lượng của các decision model trên tiếng Việt (chưa xác minh). Benchmark công bố đều bằng tiếng Anh. Nếu state hoặc câu hỏi của bạn bằng tiếng Việt, đo trên dữ liệu của bạn trước khi tin vào bất kỳ con số nào trong tài liệu này, và thử cả hai cách: câu hỏi tiếng Anh trên state tiếng Việt, và câu hỏi tiếng Việt.

## 1.8 So với classifier, embedding và LLM

**Classifier truyền thống** (fine-tune BERT, logistic regression trên embedding) cần dữ liệu gán nhãn cho từng nhãn và phải huấn luyện lại khi thêm nhãn. Decision model nhận nhãn mới qua request: thêm một dòng vào `criteria` là xong. Cloudflare nhấn mạnh đây là điểm khác cốt lõi: model đủ khả năng làm việc trên mọi tập input mà không phải huấn luyện lại khi có danh mục phân loại mới [10]. Đổi lại, classifier được fine-tune trên đúng miền của bạn, với đủ dữ liệu, vẫn có thể chính xác hơn và rẻ hơn ở quy mô rất lớn. Cloudflare cũng đi theo hướng đó: họ mở dịch vụ fine-tune Clef cho các tác vụ có nhiều năm dữ liệu đã gán nhãn [10].

**Embedding similarity** so sánh hai vector được tính riêng. Decision model đọc query và candidate *cùng lúc* trong một state, giống một cross-encoder, nên phân biệt được những đoạn có từ ngữ gần giống nhau nhưng nghĩa khác. Trong ví dụ RAG ở [§3.4](#34-lọc-passage-giữa-retrieval-và-generation), 12 passage có cosine similarity chỉ chênh nhau từ 0.455 đến 0.584, quá hẹp để tách đoạn trả lời đúng khỏi đoạn chứa prompt injection; decision model tách được [18]. Embedding vẫn là bước đầu không thay được khi cần lọc hàng triệu tài liệu, vì decision model phải gọi một lần cho mỗi cặp.

**LLM với structured output** linh hoạt hơn nhiều: sinh giá trị tự do, giải thích, suy luận. Đổi lại:

- Chậm hơn và đắt hơn với cùng một quyết định. Trong thử nghiệm self-consistency của TypeSafe, một request 8 câu Choice trên Jev 1.14 có độ trễ trung bình 103 ms; các LLM được thử (Claude Haiku 4.5, GPT-5.4 mini, GPT-5.5, Claude Opus 4.8) mất 1,9–13,5 giây cho cùng bài [19]. OpenAI nói Decisions API trả đáp án nhanh hơn Responses API khoảng 10 lần [8].
- Kém ổn định hơn khi hỏi lặp lại. Cùng thử nghiệm trên: với 40 bài đăng nằm ở ranh giới cần kiểm duyệt, mỗi bài hỏi 15 lần, quyết định kém ổn định nhất của Jev đổi trung bình 4,0% số lần, tối đa 6,7%; với mọi cấu hình LLM, có bài đổi quyết định hơn 15% số lần, và đặt temperature 0 chỉ thu hẹp chứ không xoá khoảng cách [19]. Lưu ý đây là benchmark do chính TypeSafe thực hiện.
- Confidence tự báo không dùng để đặt ngưỡng được.

Quy tắc chọn: nếu đáp án thuộc tập đóng và quyết định lặp lại nhiều lần, dùng decision model. Nếu cần sinh nội dung hoặc suy luận, dùng LLM. Nhiều hệ thống tốt nhất dùng cả hai: decision model quyết định nhanh và chuyển phần khó cho LLM ([§3.7](#37-cascade-trích-xuất-dữ-liệu), [§3.11](#311-routing-giữa-các-llm)).

---

# Phần 2 — Decision model theo nhà cung cấp

## 2.1 Bức tranh chung

Dòng thời gian (2026):

- **15/9:** TypeSafe AI ra mắt Jev, gọi là "System One model" đầu tiên [20].
- **1/10:** Cloudflare ra mắt Clef và Clef-flash trên Workers AI, weights mở trên Hugging Face theo Apache 2.0, API tương thích Jev [10].
- **28/9:** Ollama 0.35 hỗ trợ API kiểu Jev cho model local [58].
- **9/10:** Cloudflare thêm Clef-omni (thêm audio và video) [21].
- **14/10:** Jev 1.14 ra mắt, thêm loại câu hỏi Box [7].
- Trong khoảng tháng 9–11, OpenAI đưa Decisions API từ public beta lên general availability [8]; Microsoft, Liquid AI và nhiều nhóm open-source ra model tương tự ([§2.5](#25-microsoft-decision-1-và-liquid-ai-d1), [§2.6](#26-model-open-weight)).

Hiện có hai "phương ngữ" API:

| | Họ Jev (TypeSafe, Clef và các model tương thích) | OpenAI Decisions |
|---|---|---|
| Nội dung cần xét | `state` | `input` |
| Danh sách câu hỏi | `questions`: object, key là tên câu hỏi | `questions`: mảng, mỗi phần tử có `name` |
| Yes/no | `"type": "noul"`, `criteria` tuỳ chọn `{true, false}` | `"type": "predicate"` |
| Choice | `criteria`: object `{option: mô tả}` | `choices`: mảng `{value, description}` |
| Score | `criteria`: mảng mô tả theo thứ tự | `levels`: mảng `{label, description}` |
| Response | `answers`: object theo tên câu hỏi | `answers`: mảng, mỗi phần tử có `name` |
| Từ chối | Không đề cập | Có loại đáp án `refusal` |

Chuyển giữa hai họ cần một lớp adapter nhỏ; chuyển trong họ Jev chỉ cần đổi endpoint và model ID ([§4.7](#47-version-và-chuyển-provider)).

## 2.2 TypeSafe Jev

### Model và giá

| | Jev 1.14 | Jev 1.13 |
|---|---|---|
| Model ID | `jev-1.14.0` | `jev-1.13.0` |
| Giá | $0.042 / 1M token input ($42 / 1B token); output miễn phí | Như 1.14 |
| Context | 64k / request, 32k / câu hỏi | Như 1.14 |
| Input | Text, ảnh | Text, ảnh |
| Loại câu hỏi | Choice, Score, Noul, Box | Choice, Score, Noul |

Nguồn: trang Models của TypeSafe [7]. Alias `jev-latest` trỏ đến bản stable mới nhất (hiện là `jev-1.14.0`) và là mặc định của SDK; `jev-preview` trỏ đến bản mới nhất kể cả chưa chính thức. Mỗi response có field `model` ghi version thật đã trả lời, nên log field này.

Thông tin ban đầu lúc ra mắt (và trong bài so sánh của Cloudflare) nói Jev chỉ đọc text và context 32k [10]; docs hiện tại ghi Jev 1.13 và 1.14 đều đọc ảnh, context 64k mỗi request và 32k mỗi câu hỏi [7].

### Context hai tầng

Jev đọc state một lần và đánh giá mọi câu hỏi song song, nên một request phải vừa hai giới hạn [7]:

- **Per-request:** state cộng tất cả câu hỏi.
- **Per-question:** state cộng một câu hỏi bất kỳ.

Không giới hạn số câu hỏi, miễn vừa hai giới hạn này. Muốn nâng cả hai lên 262k token, gửi header `TypeSafe-Danger-Zone-Long-Context: true`. Cùng giá mỗi token, nhưng chậm hơn và kém chính xác hơn; chỉ bật cho request cần.

### Request

```bash
curl https://api.typesafe.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "jev-1.14.0", "state": "...", "questions": {...}}'
```

SDK chính thức: `pip install typesafe-sdk` (Python, có client sync và async) và `npm install @typesafe-ai/sdk` [4]. SDK có class `Choice`, `Score`, `Noul`, `NoulCriteria`, nhận cả dict thuần; tự retry `429` và `529` với backoff và tôn trọng header `retry-after` [7].

State nhận chuỗi, object JSON hoặc mảng [22]. Instructions và criteria cũng nhận object hoặc mảng, không chỉ chuỗi ([§4.2](#42-viết-câu-hỏi-và-criteria)).

### Giới hạn request

| Quy tắc | Giới hạn |
|---|---|
| Kích thước body | Tối đa 32 MiB (`413` nếu vượt) |
| Nén | `gzip` hoặc `zstd`, một lớp; zstd window tối đa 8 MiB |
| Độ sâu lồng | Tối đa 64 tầng trong mỗi `state`, `instructions`, `criteria` |
| Số | Chỉ số hữu hạn; gửi ID và số lớn dưới dạng chuỗi |
| Ảnh | PNG hoặc JPEG, mỗi cạnh 280–1568 px, tổng 78.400–2.458.624 pixel; cần header `TypeSafe-Danger-Zone-Images: true` |

Mã lỗi: `400` (vi phạm quy tắc hoặc `max_tokens_exceeded` khi vượt context), `401`, `413`, `415`, `422` (câu hỏi sai cấu trúc), `429` (rate limit, tính theo token/giây và request/giây cho mỗi organization), `529` (quá tải) [23][13].

### Tuỳ biến và dữ liệu

Jev không được fine-tune hay gắn LoRA theo dữ liệu khách hàng; mọi account dùng chung một bộ weights. Bạn đưa miền của mình vào qua state (dữ liệu, chính sách, tài liệu tham chiếu), qua instructions và criteria (quy tắc, trường hợp biên), và qua việc tách câu hỏi rồi kết hợp trong code. Jev không được huấn luyện trên request hay response của khách; khách enterprise có zero data retention [7].

## 2.3 Cloudflare Clef

Clef là họ decision model do Cloudflare huấn luyện, chạy trên Workers AI và có weights mở trên Hugging Face theo Apache 2.0 [10]:

- **Clef:** backbone Qwen3.8-27B, "precision model".
- **Clef-flash:** backbone Qwen3.5-9B, cho đường nóng cần độ trễ thấp.
- **Clef-omni:** thêm audio và video [21].

Theo Cloudflare, khác biệt so với Jev lúc ra mắt là Clef có vision encoder để đọc ảnh và context 64k so với 32k của Jev [10]. Như đã ghi ở [§2.2](#22-typesafe-jev), Jev hiện cũng đọc ảnh và có context 64k mỗi request.

Request tương thích hoàn toàn với API Jev, nên chuyển giữa hai bên chỉ cần đổi endpoint và model:

```bash
curl https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/ai/run/@cf/cloudflare/clef \
  -X POST \
  -H "Authorization: Bearer $CLOUDFLARE_AUTH_TOKEN" \
  -d '{
    "model": "clef",
    "state": "Checkout has been failing for every customer for the last hour.",
    "questions": {
      "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
          "billing": "Payments, invoices, and refunds",
          "technical": "Outages, errors, and configuration",
          "sales": "Plans and upgrades"
        }
      },
      "severity": {
        "type": "score",
        "instructions": "How severe is the customer impact?",
        "criteria": ["No impact", "Minor", "Major", "Critical"]
      }
    }
  }'
```

Ba model trên Workers AI [46]:

| | `clef` | `clef-flash` | `clef-omni` |
|---|---|---|---|
| Model ID Workers AI | `@cf/cloudflare/clef` | `@cf/cloudflare/clef-flash` | `@cf/cloudflare/clef-omni` |
| Context | 65.536 | 24.576 | 64.000 |
| Giá input / 1M token | $0.24 | $0.038 | $0.15 |
| Output | Miễn phí | Miễn phí | Miễn phí |
| Input ngoài text | Ảnh | Ảnh | Ảnh, audio (tối đa 4 clip), video (tối đa 2) |
| Weights | HF `Cloudflare/clef` | HF `Cloudflare/clef-flash` | HF `Cloudflare/clef-omni` (base Qwen3-Omni-30B-A3B) |

Giới hạn khác với Jev:

- Tối đa 64 câu hỏi mỗi request.
- Ảnh đi qua field mở rộng `images` (tối đa 4); không nhận URL từ xa.
- Clef-flash chỉ có context khoảng 24k, nhỏ hơn nhiều so với Clef.

Workers AI tính phí bằng Neuron: gói Free và Paid đều có 10.000 Neuron miễn phí mỗi ngày, phần vượt $0.011 / 1.000 Neuron. `clef` tốn 21.818 Neuron cho mỗi 1M token input, nên phần miễn phí tương đương khoảng 0,46M token `clef` mỗi ngày (tính từ hai con số trên).

Chưa xác minh: response của Workers AI có bọc trong `result` như các model Workers AI khác hay trả thẳng `answers`. Kiểm tra trên một request thật trước khi viết adapter.

Cloudflare cam kết không đọc, lưu hay huấn luyện trên request và response, trừ khi bạn dùng dịch vụ fine-tune [10]. Dịch vụ fine-tune bằng RL hiện làm qua đội forward-deployed engineer, sau đó sẽ thành nền tảng tự phục vụ: AI Gateway ghi lại traffic làm dataset, Workers AI sinh rollout, Containers làm sandbox chấm điểm, Trainer cập nhật weights, rồi deploy lại lên Workers AI.

Một ví dụ Cloudflare dùng nội bộ: phân loại domain cho đội Threat Intelligence. Clef cùng Browser Run mất 2,2 giây để tải, render và phân loại một website (ví dụ 95% thời trang, 85% thương mại điện tử, dưới 1% phishing); LLM nhanh nhất của họ, gpt-oss-120b, mất 4,7 giây cho cùng quy trình và chỉ trả hai nhãn [10].

## 2.4 OpenAI Decisions API

Decisions API đã general availability tính đến tháng 11/2026 [8]. Gửi `POST /v1/decisions` với model duy nhất hiện được hỗ trợ là `gpt-6-luna`.

### Request

```json
{
  "model": "gpt-6-luna",
  "input": "I was charged twice for my order.",
  "questions": [
    {
      "type": "choice",
      "name": "department",
      "instructions": "Which department should handle this complaint?",
      "choices": [
        {"value": "billing", "description": "Payments, invoices, and refunds."},
        {"value": "technical", "description": "Problems using the product."},
        {"value": "shipping", "description": "Delivery and tracking."},
        {"value": "other", "description": "Requests outside these categories."}
      ]
    },
    {
      "type": "score",
      "name": "severity",
      "instructions": "How severe is this issue?",
      "levels": [
        {"label": "Cosmetic", "description": "Appearance only; no lost functionality."},
        {"label": "Workaround available", "description": "A task fails, but another way works."},
        {"label": "Fully blocked", "description": "A task fails with no workaround."}
      ]
    }
  ]
}
```

Response:

```json
{
  "answers": [
    {
      "type": "choice",
      "name": "department",
      "choice": "billing",
      "probabilities": [
        {"value": "billing", "probability": 0.95},
        {"value": "technical", "probability": 0.02},
        {"value": "shipping", "probability": 0.01},
        {"value": "other", "probability": 0.02}
      ],
      "confidence": 0.93
    }
  ]
}
```

Khác với Jev, `probabilities` là mảng chứ không phải object, và có loại đáp án `refusal`: code phải kiểm tra `answer.type == "refusal"` trước khi đọc giá trị. SDK chính thức của OpenAI (Python, JavaScript, Go, Java, Ruby) có `client.decisions.create(...)` [8].

### Input

`input` là một chuỗi, hoặc một mảng có thứ tự gồm message, function call, custom tool call và output của chúng. Điều này cho phép đánh giá cả một đoạn hội thoại agent, ví dụ kiểm tra tool call vừa rồi có hợp lý không [8]:

| Item | Nội dung được nhận |
|---|---|
| User message | Text (`input_text`), ảnh (`input_image`), JSON (`input_json`) |
| Developer / system message | Chỉ text; system được coi như developer |
| Assistant message | Output text hoặc refusal |
| Function / custom tool output | Text và ảnh, nối với call bằng `call_id` |

Ảnh nhận base64 data URL hoặc URL HTTP/HTTPS công khai, không nhận `file_id`. Không nhận file, audio hay item reference. `input_json` cho phép gửi object thẳng mà không phải stringify; key như `role` hay `image_url` bên trong `data` chỉ là dữ liệu, không tạo message hay tải ảnh.

Mỗi câu hỏi có thể kèm ảnh riêng trong `instructions` (một mảng `input_text` và `input_image`), để so nhiều ảnh ứng viên với một ảnh tham chiếu chung đặt trong `input`.

### Giới hạn, cache và giá

- **Số câu hỏi:** 1–400 mỗi request (tăng từ 200 lúc beta).
- **Score:** 2–10 mức.
- **Câu hỏi phụ thuộc nhau:** gửi request riêng.
- **Prompt caching:** mặc định có breakpoint ngầm ở cuối `input`; prefix phải từ 1.024 token (không tính câu hỏi). Nội dung trong `questions` nằm ngoài prefix, nên giữ tài liệu tham chiếu trong `input` và đổi câu hỏi giữa các request để tận dụng cache. Có `prompt_cache_key` để phân vùng cache và `prompt_cache_breakpoint` / `prompt_cache_options.mode: "explicit"` để đặt breakpoint thủ công. `usage.input_tokens_details.cached_tokens` và `cache_write_tokens` đã nằm trong `usage.input_tokens`, không cộng thêm.
- **Giá:** $0.10 / 1M token input chưa cache; cache read và cache write theo giá prompt caching chuẩn của `gpt-6-luna`; output không tính phí; áp dụng phụ phí xử lý theo vùng và hệ số giá long context.
- **Dữ liệu:** hỗ trợ ZDR và HIPAA cho khách đủ điều kiện, data residency ở mọi vùng API, regional processing ở Mỹ và châu Âu (EEA + Thuỵ Sĩ).
- **Theo dõi:** `GET /v1/organization/usage/decisions` (cần admin key với quyền `api.usage.read`); `num_model_requests` đếm request, không đếm câu hỏi.

Nguồn cho cả mục: guide Decisions của OpenAI [8]. Ngoài ra còn có tích hợp với GPT-Live (voice) và Private Link.

Về tương quan với các tính năng khác, OpenAI khuyên: dùng Decisions khi cần một trong ba loại đáp án; dùng Structured Outputs với Responses API khi cần sinh object theo schema của bạn (field trích xuất, lời giải thích); dùng function calling khi cần model yêu cầu gọi tool với tham số [8].

## 2.5 Microsoft Decision-1 và Liquid AI d1

### Microsoft-Decision-1

Model post-train từ Qwen3.5-9B. Microsoft cho biết các bản sau sẽ chuyển sang nền MAI và model của OpenAI [47].

- **Giá:** $0.042 / 1M token input, output miễn phí, ngang Jev.
- **Context:** 32.768 theo OpenRouter.
- **API trên Azure AI Foundry:** `POST <resource>/providers/microsoft/v1/systemone`, request theo phương ngữ Jev, `model` là tên deployment. Xác thực bằng Entra ID bearer token hoặc header `api-key`. Deployment type là GlobalStandard hoặc DataZoneStandard [48].
- **OpenRouter:** `microsoft/microsoft-decision-1`, qua endpoint Decisions riêng của OpenRouter, không qua chat completions.
- **Weights:** không thấy công bố. Coi là chỉ có API; license chưa xác minh.

Hợp khi hạ tầng đã nằm trên Azure và cần data zone hoặc Entra ID.

### Liquid AI d1

Liquid có cả API hosted lẫn weights mở [49]:

- **API:** `https://api.liquid.ai/decisions/v1/systemone`, request theo phương ngữ Jev, dùng được client `typesafe_sdk` với base URL khác.
  - Model `d1.1` (mặc định) và `d1-ultra`. Bản `d1` cũ được OpenRouter ghi là ngừng ngày 20/11/2026.
  - Nhận text và ảnh: tối đa 8 ảnh, body tối đa 4,5 MB.
  - Chỉ tính phí input.
- **Giá trên OpenRouter:** `d1` và `d1.1` $0.02 / 1M input; `d1-ultra` $0.042 / 1M. Context của `d1` là 65.536 theo OpenRouter; docs của Liquid không công bố.
- **Weights mở:** `LiquidAI/d1-3B` trên Hugging Face.
  - Thông số: 3,12B tham số, nền LFM2.5-VL-3B, context 32.768, đọc text và ảnh.
  - Chạy bằng `transformers >= 5.14` với `trust_remote_code`, gọi `model.system_one(...)`.
  - License: model card không ghi. Trang license của Liquid áp dụng LFM Open License v1.0 cho các model công khai: miễn phí thương mại khi doanh thu năm dưới 10 triệu USD, trên mức đó cần license thương mại. Việc license này áp cho d1-3B là suy luận, chưa xác minh [50].

Một số bài viết thứ cấp nói d1 không có weights tải được. Điều đó sai với `d1-3B`.

## 2.6 Model open-weight

Các model dưới đây đều tự công bố trên Hugging Face. Số liệu chất lượng là do tác giả tự đo, trừ khi ghi khác.

| Model | Kích thước, nền | License | API tương thích Jev | Ghi chú |
|---|---|---|---|---|
| **Clef / Clef-flash / Clef-omni** | 27B / 9B / 35B tổng (MoE) | Apache 2.0 | Có | [§2.3](#23-cloudflare-clef) |
| **Liquid d1-3B** | 3,12B, LFM2.5-VL-3B | LFM Open License (suy luận) | Qua API hosted | [§2.5](#25-microsoft-decision-1-và-liquid-ai-d1) |
| **Kev-9B** [51] | LoRA r16 (45,4M) + pointer head trên Qwen3.5-9B-Base | Apache 2.0 | Có (`python -m kev.serve`) | Bản v2 ra 30/9/2026; khoảng 22 GB, chạy trên một L40S/H100, có bản MLX cho Apple Silicon; serve 65.536 context nhưng chỉ kiểm chứng đến 8.192. Collection còn có kev-27b, kev-4b, kev-0.8b (chưa xem) |
| **Strands Decider** [52] | LoRA trên Qwen3.5-2B-Base, khoảng 1,9B | Apache 2.0 | Có (`strands-decider serve`) | Tự báo 180/231 trên JevBench; có biến thể Gemma 4 (E2B, E4B, 12B, 26B-A4B); mirror ở `amazon/…` |
| **Bespoke Nimble 9B v2** [53] | LoRA khoảng 173 MB trên Qwen3.5-9B | Chưa xác minh (nguồn thứ cấp nói Apache 2.0) | Qua Ollama | Tối đa 255 option mỗi câu; temperature 2.179 lấy lại từ v1, chưa fit cho v2 |
| **Tev1-4B-experimental** (Together) [54] | SFT toàn bộ Qwen3.5-4B, khoảng 5B | Đang hoàn thiện | Không, trên API Together; có qua Ollama | Trả một chữ cái option qua chat completions; 2–24 option |
| **Laya** [55] | Encoder ModernBERT-large, 421M; bản multilingual mmBERT-base, 322M | Apache 2.0 | Có (`laya-serve`) | Context 512 (multilingual 1.024, tối đa 8.192); checkpoint base gần mức ngẫu nhiên khi zero-shot (0,362 trên typed-decisions) |
| **GLiNER2.5-Decide** (Fastino) [56] | Encoder DeBERTa-v3-large, khoảng 340M | Apache 2.0 | Không (`gliner2` `classify_text`) | Họ GLiNER, nhỏ, chạy CPU được |
| **djev** (DiffusionGemma Jev) [57] | Không huấn luyện; đọc có cấu trúc trên `google/diffusiongemma-26B-A4B-it` | Apache 2.0 | Có | Cần nhánh vLLM (PR vllm#57250, chưa xác nhận đã merge) |

Cách chọn:

- **Gần chất lượng API hosted:** Clef, rồi Clef-flash.
- **Một GPU tầm trung, cần API Jev:** Kev-9B hoặc Clef-flash.
- **Laptop hoặc edge:** Strands Decider 2B, d1-3B, Tev1 0.8B.
- **CPU, độ trễ vài ms, câu hỏi đơn giản:** Laya hoặc GLiNER2.5-Decide. Đổi lại, chất lượng thấp hơn hẳn ([§2.9](#29-benchmark)) và context rất ngắn.

Model càng nhỏ, càng cần đo calibration trên dữ liệu của bạn trước khi đặt ngưỡng ([§4.5](#45-eval-và-so-sánh-model)). Với tiếng Việt, bản multilingual của Laya là model duy nhất trong bảng nói rõ là đa ngôn ngữ. Các model nền Qwen có thể đọc được tiếng Việt, nhưng chưa ai công bố đo đạc.

## 2.7 Chạy local

**Ollama** hỗ trợ decision model kiểu Jev từ bản 0.35 (tag ngày 28/9/2026) [58]:

```bash
ollama pull nimble          # Bespoke Nimble 9B, khoảng 9,5 GB
ollama serve
curl http://localhost:11434/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"model": "nimble", "state": "...", "questions": {...}}'
```

- Model trong thư viện Ollama theo blog ra mắt: `nimble`, `tev1` (4B), `tev1:0.8b`.
- Theo blog đó, Clef, Kev, Laya và Strands Decider *không* có trong thư viện Ollama. Thư viện có thể đã đổi; kiểm tra lại trực tiếp.
- Client Python: dùng `typesafe-sdk` và đặt `TYPESAFE_BASE_URL` trỏ về localhost.

**Server riêng của từng model** đều mở `POST /v1/systemone`, nên helper `decide()` ở [§3.0](#30-cách-dùng-phần-này) dùng được chỉ bằng cách đổi `ENDPOINT`:

| Model | Lệnh |
|---|---|
| Kev-9B | `python -m kev.serve --run jaredpalmer/kev-9b` |
| Strands Decider | `pip install strands-decider` rồi `strands-decider serve` (bind `127.0.0.1`, không có auth) |
| Laya | `pip install laya` rồi `laya-serve` |
| djev | vLLM nhánh có PR #57250 |
| Clef | Tải weights từ Hugging Face; serving script đi kèm model card (chưa xem chi tiết) |

Lưu ý khi tự chạy:

- Server local thường không có auth. Không expose ra ngoài máy hoặc ngoài mạng nội bộ.
- Tự chạy không có rate limit, nhưng cũng không có retry, cache hay quan sát. Phần vận hành ở [§4.6](#46-chi-phí-độ-trễ-và-rate-limit) phải tự làm. Xem thêm [Self-hosting LLM](../self-hosting-llm/).
- Ghi lại model và revision weights vào log, giống pin version với API ([§4.7](#47-version-và-chuyển-provider)).

## 2.8 Thư viện và tích hợp

- **SDK:** `typesafe-sdk` (Python) và `@typesafe-ai/sdk` (JS) dùng được với mọi endpoint họ Jev khi đổi base URL: Liquid, Ollama, server local [4][49][58]. Với OpenAI, dùng SDK của OpenAI ([§2.4](#24-openai-decisions-api)).
- **Strands Agents:** repo `strands-labs/strands-decider` (Apache 2.0) có ví dụ dùng decision model làm cổng `before_tool_call`, tức kiểm tra tool call trước khi agent thực thi [52].
- **LangChain / LangGraph:** Laya có extras tích hợp [55].
- **Gateway:**
  - OpenRouter có endpoint Decisions cho `microsoft-decision-1`, `liquid/d1`, `d1.1`, `d1-ultra` (đã xác minh). Jev trên OpenRouter chỉ thấy trong danh sách thứ cấp.
  - Vercel AI Gateway được nhắc là có Liquid d1 và d1-ultra (nguồn thứ cấp).
  - Cloudflare AI Gateway đứng trước Workers AI ([§2.3](#23-cloudflare-clef)).
- **Agent skill:** TypeSafe có skill cho Claude Code và Codex để agent viết request đúng format [7].
- **Danh sách cộng đồng:** `AnotiaWang/awesome-decision-models`, `andyrewlee/awesome-system-one`, `agairola/awesome-decision-models`. Đây là nơi tốt để theo dõi model mới, vì danh sách ở [§2.6](#26-model-open-weight) sẽ cũ nhanh.

## 2.9 Benchmark

Cloudflare công bố so sánh trên các benchmark được chọn từ *Jev Decision Index* [10]. Trích một phần (số cao nhất mỗi hàng in đậm):

| Benchmark | Clef | Clef-flash | Jev | DiffusionGemma Jev | Kev 9B | Laya |
|---|---|---|---|---|---|---|
| BFCL · case exact | 98.47 | **98.76** | 95.75 | 96.52 | 94.51 | 38.13 |
| API-Bank · accuracy | 91.93 | **93.11** | 88.19 | 83.66 | 56.30 | 11.41 |
| When2Call · accuracy | 72.37 | 65.58 | **80.97** | 75.44 | 49.62 | 11.94 |
| BANKING77 · macro-F1 | **94.20** | 90.93 | 79.74 | 74.28 | 84.83 | 14.29 |
| CLINC150+OOS · macro-F1 | **97.43** | 66.77 | 89.27 | 83.49 | 79.03 | 3.19 |
| BRIGHT · nDCG@10 | 45.91 | 39.26 | **47.52** | 42.94 | 38.53 | 19.90 |
| PhishNChips · accuracy | 79.60 | 75.05 | 62.55 | **85.35** | 50.75 | 50.15 |
| Median latency (ms) | 209.3 | 38.8 | 524.1 | 84.4 | 51.4 | **5.8** |
| p95 latency (ms) | 238.6 | **122.4** | 536.0 | 211.2 | 187.9 | 222.5 |

Cách đọc:

- Đây là benchmark do Cloudflare chọn và chạy, công bố cùng lúc ra mắt sản phẩm của họ. Trên bộ workflow eval của chính TypeSafe, Cloudflare báo Clef thắng Jev 3/4 hạng mục với chênh lệch nhỏ (ví dụ customer service 76.3 so với 76.0) [10].
- Độ trễ của Jev ở đây (524 ms) đo từ hạ tầng của Cloudflare gọi sang API TypeSafe; TypeSafe nói phần lớn call hoàn tất trong khoảng 100 ms [3], và đo 103 ms trung bình trong thử nghiệm của họ [19]. Độ trễ phụ thuộc nhiều vào vị trí mạng và kích thước state.
- Không có model nào thắng mọi hàng. Clef-flash mạnh ở function calling (BFCL, API-Bank) nhưng tụt mạnh ở phân loại có option ngoài phạm vi (CLINC150+OOS: 66.77). Laya rất nhanh nhưng chất lượng thấp hơn hẳn.
- Kết luận thực tế: dùng benchmark để chọn ứng viên, rồi so trên dữ liệu của bạn ([§4.5](#45-eval-và-so-sánh-model)).

**Jev Decision Index** là leaderboard cộng đồng cho các model tái hiện Jev, chạy trên Hugging Face Spaces (`multimodalart/jev-decision-index`) [59]:

- **Điểm chính thức:** 60% từ bộ test private đo cùng các kỹ năng như bộ public, 40% từ tác vụ private ở miền mới. Bộ public chỉ hiển thị, không tính vào điểm.
- **Bộ public:** 37 benchmark trong năm nhóm. Mỗi benchmark được hiệu chỉnh theo mức ngẫu nhiên (0 là ngẫu nhiên, 100 là hoàn hảo). Request không trả lời được tính là sai.
- **Version:** model card trích các bản index khác nhau. Kev ghi "Decision Index 0.2", Clef và Liquid ghi 0.2.1. So hai con số chỉ có ý nghĩa khi cùng version.

---

# Phần 3 — Ứng dụng và hướng dẫn

## 3.0 Cách dùng phần này

Mỗi mục là một bài toán thường gặp, viết theo cùng khung: **bài toán**, **câu hỏi**, **định tuyến trong code**, **kết quả tham khảo** (nếu có số liệu công bố) và **lưu ý**. Phần lớn ví dụ dựa trên cookbook chính thức của TypeSafe; số liệu ghi kèm model và ngày đo của nguồn.

Bảng chọn nhanh:

| Bài toán | Loại câu hỏi chính | Pattern | Mục |
|---|---|---|---|
| Phân loại, định tuyến ticket | Choice + yes/no + Score | Fan-out, routing theo xác suất | [§3.1](#31-phân-loại-và-định-tuyến-ticket-hỗ-trợ) |
| Guardrail cho LLM | Nhiều yes/no + Score | Ngưỡng theo hazard | [§3.2](#32-guardrail-cho-input-và-output-của-llm) |
| Moderation chat | Choice + yes/no | Quy tắc thứ tự trong code | [§3.3](#33-moderation-nội-dung-thời-gian-thực) |
| Lọc passage RAG | Nhiều yes/no trên cặp query–passage | First match wins | [§3.4](#34-lọc-passage-giữa-retrieval-và-generation) |
| Rerank | Yes/no trên cặp query–candidate | Sắp xếp theo xác suất | [§3.5](#35-rerank-kết-quả-tìm-kiếm) |
| Kiểm tra trích dẫn | Choice | String match trước, model sau | [§3.6](#36-kiểm-tra-trích-dẫn) |
| Cascade trích xuất | Yes/no theo từng field | Escalate khi có cờ | [§3.7](#37-cascade-trích-xuất-dữ-liệu) |
| Trích ngày, giá trị | Choice trên ứng viên | Code ráp và kiểm tra | [§3.8](#38-trích-xuất-giá-trị-có-miền-đóng) |
| Function calling | Choice + yes/no | Tham số miền đóng | [§3.9](#39-chọn-function-và-tham-số) |
| Chọn skill cho agent | Choice + yes/no | Hai tầng: xếp hạng rồi kiểm lại | [§3.10](#310-chọn-skill-hoặc-tool-cho-agent) |
| Routing LLM | Choice + Score | Routing theo xác suất | [§3.11](#311-routing-giữa-các-llm) |
| Taxonomy lớn | Choice theo tầng | Beam search, lùi lên tầng cha | [§3.12](#312-phân-loại-theo-taxonomy-lớn) |
| Ghép thực thể | Score + yes/no | Làm tròn về mức | [§3.13](#313-ghép-thực-thể-và-khử-trùng-lặp) |
| Chấm điểm đa tiêu chí | Nhiều Score | Composite scoring | [§3.14](#314-chấm-điểm-đa-tiêu-chí) |
| Feature cho ML | Score + yes/no | Autoresearch loop | [§3.15](#315-biến-văn-bản-thành-feature-cho-ml) |
| Ảnh, tài liệu scan | Box + yes/no | Kiểm tra hai lần: trang và vùng cắt | [§3.16](#316-ảnh-và-tài-liệu-scan) |

Code trong phần này viết bằng Python, gọi HTTP trực tiếp theo phương ngữ Jev để chạy được với cả Jev và Clef. Helper dùng chung:

```python
import os
import requests

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"            # pin version, không dùng alias khi đã tune ngưỡng

def decide(state, questions, model=MODEL, timeout=10):
    """Gửi một state và nhiều câu hỏi; trả về dict answers theo tên câu hỏi."""
    resp = requests.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"},
        json={"model": model, "state": state, "questions": questions},
        timeout=timeout,
    )
    resp.raise_for_status()      # production: retry 429/529 với backoff, hoặc dùng SDK
    body = resp.json()
    log_decision(body["model"], state, questions, body["answers"])   # log version thật
    return body["answers"]

def noul(instructions, true=None, false=None):
    q = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {"true": true, "false": false}
    return q

def choice(instructions, options):
    return {"type": "choice", "instructions": instructions, "criteria": options}

def score(instructions, levels):
    return {"type": "score", "instructions": instructions, "criteria": levels}
```

Với OpenAI Decisions API, đổi tên field theo bảng ở [§2.1](#21-bức-tranh-chung) và kiểm tra đáp án `refusal`.

**Về các con số trong code.** Ngưỡng, trọng số, `timeout` và `K` trong các ví dụ của phần này là điểm khởi đầu để đọc code, không phải giá trị mặc định cho production. Chỗ nào lấy từ cookbook thì ghi nguồn bên cạnh; những chỗ khác là minh hoạ. Trước khi dùng thật, đặt lại từng con số từ dữ liệu có nhãn của bạn theo chi phí của false positive và false negative ([§4.4](#44-đặt-ngưỡng)).

## 3.1 Phân loại và định tuyến ticket hỗ trợ

**Bài toán.** Mỗi ticket cần được chuyển đúng đội, gắn cờ khẩn cấp, phát hiện yêu cầu hoàn tiền và mức bực bội, đồng thời lọc spam và lừa đảo. Đây là ứng dụng "chuẩn" của decision model.

**Câu hỏi.** Hỏi mọi thứ trong một request. Đưa vào state chỉ những gì câu hỏi cần: tin nhắn, người gửi, đơn hàng đang mở, chính sách liên quan. Ví dụ rút gọn từ hướng dẫn "How to build" của TypeSafe [3]:

```python
def triage(ticket, customer):
    if ticket["status"] == "closed":                   # quy tắc tất định: không gọi model
        return "no_action"

    state = {
        "ticket": {"message": ticket["message"], "sender": ticket["sender"]},
        "customer": {
            "plan": customer["plan"],
            "open_orders": [o for o in customer["orders"] if o["status"] != "delivered"],
        },
        "policy": {"sensitive_credentials": ["password", "security code", "API key"]},
    }

    questions = {
        "topic": choice(
            {"question": "Which team should handle `ticket.message`?",
             "focus": "Classify the customer's primary request."},
            {
                "billing": {"what": "Charges, invoices, refunds, or subscriptions",
                            "not_for": "Order tracking or account access",
                            "examples": ["I was charged twice", "Where is my refund?"]},
                "orders":  {"what": "Order status, delivery, cancellation, or returns",
                            "not_for": "Charges or account access",
                            "examples": ["Where is my order?", "Cancel my shipment"]},
                "account": {"what": "Login, profile, permissions, or security",
                            "not_for": "Charges or order tracking",
                            "examples": ["Reset my password", "I cannot sign in"]},
            },
        ),
        # tín hiệu spam: ba câu yes/no hẹp thay vì một câu "có phải spam không"
        "requests_credentials": noul(
            "Does `ticket.message` ask the recipient to disclose an item in `policy.sensitive_credentials`?",
            true="Asks to disclose a listed credential, e.g. 'Reply with your password'",
            false="No such request; a link to reset a credential does not count"),
        "sender_identity_mismatch": noul(
            "Does `ticket.sender.display_name` claim an organization unrelated to the domain of `ticket.sender.email`?"),
        "unexpected_reward": noul(
            "Does `ticket.message` announce an unsolicited prize, payment, or reward?"),
        # câu hỏi speculative: chỉ dùng cho một số nhánh
        "refund_requested": noul(
            "Does the customer explicitly ask for a refund or account credit?",
            false="A billing complaint without a requested remedy, e.g. 'Why was I charged twice?'"),
        "mentions_open_order": noul(
            "Does `ticket.message` refer to an order in `customer.open_orders` by id or identifying details?"),
        "frustration": score(
            "How frustrated does the customer appear in `ticket.message`? Judge expressed frustration, not issue severity.",
            ["Calm and matter-of-fact", "Frustrated but civil", "Very angry or threatening to leave"]),
    }

    a = decide(state, questions)

    spam_risk = (0.45 * a["requests_credentials"]["noul"]
                 + 0.30 * a["sender_identity_mismatch"]["noul"]
                 + 0.25 * a["unexpected_reward"]["noul"])
    topic = a["topic"]

    if 0.4 < spam_risk < 0.6 or topic["probabilities"][topic["choice"]] < 0.75:
        return route_to_human_review(ticket)
    if spam_risk >= 0.6:
        return quarantine_as_spam(ticket)
    if topic["choice"] == "billing":
        return route_to_billing(ticket, refund=a["refund_requested"]["noul"] >= 0.7)
    if topic["choice"] == "orders":
        return route_to_orders(ticket, has_order=a["mentions_open_order"]["noul"] >= 0.7)
    high = a["frustration"]["confidence"] >= 0.7 and a["frustration"]["score"] >= 1.5
    return route_to_account(ticket, priority="high" if high else "normal")
```

Bản gốc của TypeSafe đặt ngưỡng chủ đề trên `confidence < 0.75`; ở đây đổi sang xác suất của option được chọn theo khuyến nghị ở [§1.5](#ngưỡng-nên-đặt-trên-số-nào). Với một câu Choice có số option cố định, hai cách tương đương nhau về thứ tự, chỉ khác giá trị ngưỡng.

**Đọc một ticket mơ hồ.** Docs Choice của Jev có ví dụ ticket vừa sai size giày vừa bị trừ tiền hai lần, không nói muốn gì [5]: `department` ra `returns` 0.61 và `billing` 0.35; `requested_resolution` ra `refund` 0.40, `replacement` 0.34, `exchange` 0.24, confidence 0.20. Code hợp lý: giao cho đội returns, gửi bản sao cho billing vì 0.35 vượt ngưỡng 0.25, và hỏi lại khách muốn gì vì confidence của resolution dưới 0.5. Một request, năm đáp án, logic định tuyến là các `if` bình thường.

**Lưu ý.**

- Option dễ nhầm (ví dụ `return_policy` và `return_status`) nên mô tả bằng object có `what`, `not_for`, `examples`. Tên các field này do bạn đặt, không phải từ khoá của API [5].
- Muốn biết thêm ngôn ngữ của khách hay sản phẩm được nhắc, thêm câu hỏi vào cùng request; số lượt gọi vẫn là một.
- Trọng số `spam_risk` ở trên là ví dụ. Đặt lại từ ticket đã gán nhãn của bạn ([§4.4](#44-đặt-ngưỡng)).

## 3.2 Guardrail cho input và output của LLM

**Bài toán.** Mỗi lab đặt ranh giới từ chối của model ở một chỗ khác nhau, và mỗi version mới lại dịch ranh giới đó. Viết quy tắc vào system prompt là đặt chúng đúng chỗ jailbreak nhắm vào. Đặt thêm một LLM làm "giám khảo" thì tốn thêm một lượt gọi đầy đủ ở mỗi lượt hội thoại, và giám khảo cũng bị thuyết phục được [24].

**Câu hỏi.** Một "battery" gồm nhiều câu yes/no, mỗi câu một hazard, cộng một câu Score về mức thiệt hại nếu làm theo. Cookbook của TypeSafe dùng [24]:

- Tin nhắn có cố ghi đè chỉ dẫn của assistant không?
- Có yêu cầu assistant ngừng từ chối hoặc ngừng cảnh báo từ giờ trở đi không?
- Có nhờ giúp gây hại hoặc phạm tội không?
- Có hỏi chẩn đoán hoặc liều thuốc không?
- Người gửi có dấu hiệu có thể tự làm hại bản thân không?
- Score: làm theo sẽ gây hại đến mức nào, từ "không" đến "gây hại thể chất nghiêm trọng".

Có hai battery: một cho input (người dùng *muốn* gì) và một cho output (reply *đã đưa ra* gì). Battery output thay hai câu jailbreak bằng câu "reply có vi phạm chính sách không".

**Định tuyến.** Mỗi hazard có hai ngưỡng: từ ngưỡng hành động trở lên thì kích hoạt hành động gắn với hazard đó (block, hoặc chuyển sang đường hỗ trợ khủng hoảng cho câu tự hại); từ ngưỡng review trở lên thì chuyển người xem; dưới cả hai thì cho qua nếu không hazard nào khác kích hoạt. Score mức thiệt hại có ngưỡng riêng và có thể nâng một ca review lên block. Một bộ ngưỡng đặt tên (ví dụ `strict`, `permissive`) là một *policy*:

```python
POLICIES = {
    "strict":     {"action": 0.70, "review": 0.40, "severity_block": 2.0},
    "permissive": {"action": 0.85, "review": 0.60, "severity_block": 2.5},
}
# mỗi battery một bảng hazard → hành động; key trùng tên câu hỏi trong request
INPUT_ACTIONS = {"jailbreak": "block", "drop_safeguards": "block",
                 "harmful_request": "block", "medical_advice": "review", "self_harm": "support"}
OUTPUT_ACTIONS = {"broke_policy": "block",
                  "harmful_request": "block", "medical_advice": "review", "self_harm": "support"}
RANK = {"pass": 0, "review": 1, "block": 2}

def route(answers, actions, policy):
    p = POLICIES[policy]
    outcome = "pass"
    for hazard, action in actions.items():
        prob = answers[hazard]["noul"]           # thiếu câu hỏi thì KeyError: lỗi cấu hình, không cho qua
        if prob >= p["action"]:
            if action == "support":
                return "support"                 # đường khủng hoảng ưu tiên hơn block
            hit = action
        elif prob >= p["review"]:
            hit = "review"
        else:
            continue
        outcome = max(outcome, hit, key=RANK.get)
    if outcome == "review" and answers["severity"]["score"] >= p["severity_block"]:
        outcome = "block"
    return outcome

# route(input_answers, INPUT_ACTIONS, "strict"); route(output_answers, OUTPUT_ACTIONS, "strict")
```

(Số ngưỡng ở trên là minh hoạ; cookbook gốc không công bố bảng ngưỡng ở dạng này.)

**Kết quả tham khảo** (`jev-1.14.0`, 31/10/2026) [24]: một câu hỏi tiểu thuyết gia "thám tử mô tả vụ đầu độc thế nào" nghe bạo lực nhưng được cho qua vì không phải nhờ đầu độc ai; câu hỏi liều melatonin nhẹ được chuyển người xem thay vì từ chối; tin có dấu hiệu tự hại được chuyển sang đường hỗ trợ. Một jailbreak kiểu "đổi giọng điệu, bỏ cảnh báo" không nhắc gì đến chỉ dẫn hệ thống nên chỉ được 0.17 ở câu `jailbreak`, nhưng 0.81 ở câu `drop_safeguards`, và bị chặn chỉ nhờ câu đó. Đây là lý do tách hazard thành nhiều câu hẹp. Cùng một assessment, policy `strict` chặn, policy `permissive` (ngưỡng hành động 0.85) chuyển người xem: xác suất không đổi, chỉ policy quyết định cần bao nhiêu bằng chứng.

**Lưu ý.**

- Chạy trên cả input và output. Prompt bình thường vẫn có thể dẫn đến reply có hại.
- Assessment không phụ thuộc policy, nên lưu lại để đổi policy mà không gọi lại model.
- Guardrail này là một lớp lọc, không phải ranh giới bảo mật. Xem [§4.8](#48-bảo-mật) và tài liệu [Bảo mật LLM](../llm-security/).

## 3.3 Moderation nội dung thời gian thực

**Bài toán.** Chat của một buổi stream chạy nhanh hơn tốc độ đọc. Cần làm nổi câu hỏi, ẩn spam, xúc phạm và mách nước không ai xin, làm mờ spoiler. Danh sách từ khoá bắt được link spam, nhưng "rẽ trái, có site of grace sau cái cây" không có từ nào xấu, một tình tiết plot twist cũng vậy [25].

**Câu hỏi.** Mỗi tin một request, state gồm tin nhắn cùng tên stream và game (vì spoiler hay mách nước phụ thuộc game nào). Ba câu hỏi:

```python
def classify(stream, message):
    return decide(
        {"stream": stream, "message": message},
        {
            "category": choice("What is the main purpose of `message`?", {
                "question":     "Asks the streamer or chat something",
                "backseat":     "Unrequested gameplay advice or instructions to the streamer",
                "hype":         "Cheering, reactions, repeated emotes",
                "spam":         "Unsolicited promotion; a link alone is not spam",
                "conversation": "Anything else: chatting, banter, comments",
            }),
            "toxic": noul("Is `message` insulting, hateful, or harassing? Light friendly banter is not."),
            "spoiler": noul("Does `message` reveal plot or story events of `stream.game`?"),
        },
    )

RULES = {"question": "highlight", "backseat": "hide", "spam": "hide",
         "hype": "show", "conversation": "show"}
TOXIC_AT, SPOILER_AT = 0.6, 0.6

def decide_view(a):
    if a["toxic"]["noul"] >= TOXIC_AT:
        return "hide"
    if RULES[a["category"]["choice"]] == "hide":
        return "hide"
    if a["spoiler"]["noul"] >= SPOILER_AT:
        return "blur"
    return RULES[a["category"]["choice"]]
```

**Kết quả tham khảo** (`jev-1.14.0`, 29/10/2026, 29 tin viết cho một stream Elden Ring) [25]: ẩn 7, làm mờ 2, làm nổi 6. Một tin vừa giống trò chuyện vừa giống mách nước được `conversation` 0.67, `backseat` 0.32, nhưng câu spoiler bắt được. Một tin đọc như xúc phạm nhưng kết bằng "love you tho" có `toxic` thấp vì khớp mô tả "light banter". Trên mẫu này, mọi xác suất toxic hoặc ≥ 0.97 hoặc ≤ 0.38, nên mọi ngưỡng trong khoảng đó cho cùng kết quả; chat thật sẽ có nhiều giá trị ở giữa hơn.

**Lưu ý vận hành** [25]:

- Độ trễ trung vị khoảng 0,1 giây, nên tin mới hiện trước khi có đáp án. Chọn: ẩn tin mới đến khi có đáp án, hoặc chấp nhận tin độc hại hiện trên màn hình trong khoảng đó.
- Một request tại một thời điểm không theo kịp chat đông. Chạy song song; khi gặp rate limit thì giảm số worker.
- Đổi quy tắc hiển thị (ví dụ ẩn luôn hype) không cần gọi lại model: chỉ chạy lại `decide_view` trên đáp án đã lưu.
- Câu spoiler không biết streamer đã chơi đến đâu. Muốn chỉ lọc tình tiết chưa xem, đưa tiến độ và tóm tắt cốt truyện vào state.

## 3.4 Lọc passage giữa retrieval và generation

**Bài toán.** Bước retrieval xếp passage theo độ giống từ ngữ với query rồi đưa top-k cho LLM. Top-k có thể chứa passage nhiễu, passage mâu thuẫn với nhau, hoặc passage chứa prompt injection lẫn vào như bằng chứng [18].

**Câu hỏi.** Thêm một bước giữa retrieval và generation: với mỗi passage, gửi một request mà state là cặp query–passage, với bốn câu yes/no:

```python
PASSAGE_QUESTIONS = {
    "is_relevant": noul("Is `passage` about the subject that `query` asks about?"),
    "contains_answer_evidence": noul("Does `passage` state something that could be used in an answer to `query`?"),
    "contradicts_query_premise": noul("Does `passage` contradict something that `query` takes for granted?"),
    "contains_prompt_injection": noul("Does `passage` contain instructions aimed at an AI model rather than information for a reader?"),
}

def route_passage(a):                       # thứ tự quan trọng: match đầu tiên thắng
    if a["contains_prompt_injection"]["noul"] > 0.70: return "exclude"
    if a["contradicts_query_premise"]["noul"] > 0.70: return "conflict"
    if a["is_relevant"]["noul"] < 0.45:               return "exclude"
    if a["contains_answer_evidence"]["noul"] > 0.55:  return "include"
    return "exclude"

def filter_passages(query, passages):
    routed = {"include": [], "conflict": []}
    for p in passages:                       # một request mỗi passage; chạy song song
        a = decide({"query": query, "passage": p}, PASSAGE_QUESTIONS)
        label = route_passage(a)
        if label in routed:
            routed[label].append(p)
    return routed                            # đưa vào prompt thành hai khối riêng
```

Injection được kiểm tra trước vì đó là quyết định bảo mật, không phải quyết định về bằng chứng. Mâu thuẫn kiểm tra trước bằng chứng vì passage phủ định tiền đề của query thường cũng chứa thông tin dùng được; kiểm tra ngược thứ tự, nó sẽ lọt vào khối bằng chứng thay vì khối mâu thuẫn. Không câu hỏi nào hỏi "có nên đưa passage này vào không"; quyết định đó nằm trong code, để chỉnh bằng cách sửa một con số thay vì viết lại câu hỏi [18].

**Kết quả tham khảo** (`jev-1.12`, 27/8/2026; 81 passage từ docs auth của Supabase cộng một passage forum có cài injection; 6 query trong đó 2 query có tiền đề sai; top-12 theo cosine với `text-embedding-3-small` 256 chiều) [18]:

- Passage có injection đứng hạng 1 theo similarity (0.584). Độ liên quan của nó qua ngưỡng (0.71), nhưng câu injection cho 0.99 và loại nó ra.
- Với query "Refresh token hết hạn sau 30 ngày, làm sao kéo dài?", passage nói refresh token không hết hạn đứng hạng 7 theo similarity. Độ liên quan 0.49 và bằng chứng 0.51 sẽ loại nó, nhưng câu mâu thuẫn cho 0.92 và chuyển nó vào khối mâu thuẫn. LLM sinh câu trả lời mở đầu bằng "không đủ bằng chứng được chấp nhận", nêu mâu thuẫn và trích passage thay vì bịa ra một cài đặt 30 ngày.
- Với một query bình thường, ba passage hạng 2–4 đều nói về "lifetime của signing key", sai loại lifetime nhưng gần như đúng từ ngữ của query, và đều có độ liên quan ≤ 0.08. Ba trong bốn passage được chấp nhận nằm ở hạng 8, 9 và 11.
- Trên cả 72 passage, ít nhất hai phần ba mỗi query bị loại.

**Lưu ý.**

- Bốn ngưỡng ở trên được chọn cho corpus này, không phải mặc định. Lưu đáp án để chỉnh ngưỡng mà không gọi lại API.
- Chi phí tăng theo k, vì mỗi câu hỏi là về một cặp và không gộp nhiều passage vào một request được.
- Câu injection là một bộ lọc, không phải ranh giới bảo mật. Passage dưới ngưỡng vẫn vào prompt, nên prompt sinh câu trả lời vẫn phải coi mọi passage là văn bản không tin cậy.
- Xem thêm [Embeddings](../embeddings/) cho bước retrieval và [RAG nâng cao](../advanced-rag/) cho các kỹ thuật khác.

## 3.5 Rerank kết quả tìm kiếm

**Bài toán.** Fast search (BM25, embedding) chạy được trên toàn corpus nhưng hiếm khi đưa đúng tài liệu lên đầu. Rerank chấm lại từng ứng viên trong shortlist so với query rồi sắp xếp lại.

**Câu hỏi.** Một câu yes/no cho mỗi cặp query–candidate. Một câu trả lời có/không thuần tuý không đủ để xếp 30 ứng viên; xác suất 0–1 thì đủ, và đó chính là điểm để sort [26].

```python
RERANK_Q = {"match": noul(
    "Could `candidate` be the passage that `query` is citing?",
    true="The candidate is the source the query refers to",
    false="The candidate is about a different matter, even if it uses similar words")}

def rerank(query, shortlist):
    scored = [(decide({"query": query, "candidate": c}, RERANK_Q)["match"]["noul"], c)
              for c in shortlist]            # production: chạy song song
    return [c for _, c in sorted(scored, key=lambda x: x[0], reverse=True)]
```

**Kết quả tham khảo** (CLERC, 3.565 passage phán quyết toà án Mỹ, 40 query; BM25 lấy top-30, shortlist chứa đáp án đúng ở 100% query) [26]:

| | Top 1 | Top 5 | Top 10 |
|---|---|---|---|
| Chỉ BM25 | 5% | 15% | 38% |
| BM25 + rerank bằng Jev | 18% | 35% | 62% |

40 query × 30 ứng viên là 1.200 request, chạy song song.

**Lưu ý.**

- Rerank chỉ sắp xếp lại shortlist, không thêm được tài liệu mà fast search bỏ sót. Recall của bước đầu là trần.
- Trong thực tế nên hỏi thêm vài câu về cùng cặp trong một request (ví dụ "candidate có trả lời trực tiếp không", "có lỗi thời không") rồi kết hợp trong code.
- Kết quả trên là trên văn bản pháp lý tiếng Anh, một miền khó. Đo trên corpus của bạn.

## 3.6 Kiểm tra trích dẫn

**Bài toán.** LLM trả lời kèm trích dẫn: mỗi khẳng định có một đoạn nguồn và câu trích. Một số trích dẫn sai: câu trích không có trong tài liệu, hoặc có nguyên văn nhưng ngữ cảnh lại nói điều ngược lại [27].

**Thiết kế.** Hai bước, bước rẻ trước:

1. **String match.** Chuẩn hoá khoảng trắng và dấu nháy cong, tìm câu trích như một substring. Không thấy thì đánh dấu `fabricated`, không cần model. Nếu thấy, vị trí khớp cho biết câu trích thuộc đoạn nào. Trích dẫn chỉ nêu đoạn mà không có câu trích thì lấy luôn đoạn đó.
2. **Choice trên đoạn nguồn.** Một câu hỏi: đoạn này liên quan thế nào đến khẳng định?

```python
CITE_Q = {"relation": choice(
    "How does `section` relate to `claim`?",
    {"supports": "The section states or directly implies the claim",
     "contradicts": "The section states something incompatible with the claim",
     "unrelated": "The section says nothing that settles the claim"})}
VERDICT = {"supports": "verified", "contradicts": "contradicted", "unrelated": "unsupported"}
AUTO_ACCEPT = 0.8

def check_citation(source_sections, citation):
    section = find_section(source_sections, citation)      # bước 1
    if section is None:
        return "fabricated", 1.0, False
    a = decide({"claim": citation["claim"], "section": section}, CITE_Q)["relation"]
    p = a["probabilities"][a["choice"]]
    return VERDICT[a["choice"]], p, p < AUTO_ACCEPT        # cờ cuối: cần người xác nhận
```

**Kết quả tham khảo** (`jev-1.12`, 16/8/2026, 8 trích dẫn LLM viết về RFC 7519, 4 đúng và 4 bị sửa cho sai) [27]: cả 4 trích dẫn đúng ra `verified` với confidence từ 0.93; trích dẫn có câu trích không tồn tại bị string match bắt; trích dẫn khẳng định claim `exp` là bắt buộc, trích nguyên văn mục 4.1.4, ra `contradicted` 0.99 vì cùng mục ghi "Use of this claim is OPTIONAL"; hai trích dẫn `unsupported` ở 0.27 và 0.56 được chuyển người xem. Một trong hai có câu trích đúng nguyên văn nhưng đoạn nguồn không nói gì về khẳng định, cho thấy string match một mình không đủ.

**Lưu ý.** String match chính xác sau chuẩn hoá sẽ đánh `fabricated` cả câu trích bị cắt hoặc sửa nhẹ; production cần fuzzy match. Bắt đầu với ngưỡng cao rồi hạ dần khi thấy model làm tốt trên tài liệu của bạn.

## 3.7 Cascade trích xuất dữ liệu

**Bài toán.** Reasoning model trích dữ liệu có cấu trúc tốt nhưng chậm và đắt; model nhỏ rẻ nhưng hay sai, và cái sai thường hợp lệ với schema nên JSON Schema validation không bắt được. Cascade chạy model nhỏ trước, dùng decision model kiểm tra, chỉ gọi model lớn khi kiểm tra báo lỗi [28].

**Câu hỏi.** Với mỗi field, một nhóm câu yes/no hẹp, đặt sao cho `true` nghĩa là *có vấn đề*. Field có giá trị nhận đủ bộ; field rỗng chỉ nhận câu "để trống có sai không":

```python
FIELD_CHECKS = {
    "hallucinated":  "Is `extracted_field.value` absent from `source`?",
    "off_target":    "Was `extracted_field.value` taken from text in `source` that is about something other than `field_spec`?",
    "unreasonable":  "Is `extracted_field.value` an unreasonable value for `field_spec`?",
}
ABSENCE_CHECK = "Does `source` contain a value for `field_spec` even though `extracted_field.value` is empty?"
FIRE_T = 0.7

def verify(source, schema, record):
    qs = {}
    for name, spec in schema["properties"].items():    # duyệt schema, không duyệt record: field bị bỏ sót vẫn được hỏi
        value = record.get(name)
        base = {"field_spec": spec, "extracted_field": {"name": name, "value": value}}
        checks = FIELD_CHECKS if value not in (None, "", []) else {"absence_wrong": ABSENCE_CHECK}
        for head, text in checks.items():
            qs[f"{name}::{head}"] = noul({"main_question": text, **base})
    a = decide({"source": source}, qs)
    return {k: v["noul"] for k, v in a.items()}

def extract(source, schema):
    record = small_model_extract(source, schema)
    flags = verify(source, schema, record)
    if max(flags.values(), default=1.0) > FIRE_T:   # max, không phải trung bình; schema rỗng thì escalate
        record = reasoning_model_extract(source, schema)
    return record
```

Instructions ở đây là object: câu hỏi nằm trong `main_question`, còn `field_spec` và `extracted_field` thay đổi theo từng field. Đây là cách TypeSafe khuyên khi một phần câu hỏi đến từ code: đặt giá trị vào field riêng thay vì nối chuỗi template ([§4.2](#42-viết-câu-hỏi-và-criteria)).

**Kết quả tham khảo** [28]: trên một trang lịch sự kiện của trường đại học, schema hỏi `registration_open_date` và `description`. Trang không có ngày mở đăng ký. `gpt-5.4-mini` để trống ngày (đúng) nhưng chép luôn ví dụ trong description của schema ("Registration opens for the fall semester") vào `description`. Record hợp lệ với schema. Jev cho field `description` 0.98 ở `hallucinated`, 0.97 ở `off_target`, 0.86 ở `unreasonable`; field ngày để trống chỉ 0.05 ở `absence_wrong`. Cascade escalate lên `gpt-5.5` với `reasoning_effort="high"`, model này trả `description` rỗng. Trong một lần chạy nội bộ khác của TypeSafe trên 100 prompt ScrapeGraphAI-100k (verifier `jev-1.12`), cascade đạt điểm LLM-judge khoảng 0.79 với khoảng $0.02 mỗi lần trích, so với 0.82 và khoảng $0.045 khi chỉ dùng `gpt-5.5` reasoning.

**Lưu ý.**

- Kiểm tra theo `max`, không theo trung bình: một cờ đỏ chắc chắn là đủ, không để nó bị trung bình hoá thành im lặng.
- Một câu "record này có ổn không" cũng bắt được ví dụ trên (0.90), nhưng không nói field nào sai và vì sao. Tách theo field và kiểu lỗi để sửa và chỉnh ngưỡng từng câu.
- Verifier phải rẻ hơn rất nhiều so với tầng reasoning, nếu không cascade không tiết kiệm gì.
- Cookbook cố ý cho model trích xuất trả text thường thay vì structured output, vì lỗi cần bắt là lỗi hợp lệ với schema. Trong production, bạn vẫn có thể bật structured output cho tầng trích xuất; verifier vẫn cần thiết.

## 3.8 Trích xuất giá trị có miền đóng

Decision model không sinh giá trị, nhưng rất nhiều "trích xuất" thực ra là chọn trong một tập hữu hạn. Hai kỹ thuật:

### Tách giá trị thành thành phần có miền đóng

**Ngày tháng.** Thay vì hỏi "deadline là ngày nào", hỏi bảy câu Choice trong một request [29]:

- `mode`: `absolute` (có tháng), `relative` (so với hôm nay), `none` (văn bản không nêu ngày này).
- `month` (12 option), `day` (31 option), `year` (mỗi năm từ 1900 đến 2050, cộng `none` khi không nêu năm và `out_of_range` khi nêu năm ngoài danh sách).
- `day_anchor` (today, tomorrow, the day after, named weekday), `weekday`, `week_offset` (`current`, `next`).

Code chỉ đọc các thành phần mà `mode` cần, ráp thành `date`, tự điền năm khi văn bản không nêu, và tự giải "next Thursday" theo quy ước của bạn. Model chỉ đọc chữ, không làm phép tính lịch. Confidence của một ngày là confidence thấp nhất trong các thành phần đã dùng; dưới 0.60 hoặc không ráp được thì chuyển người xem. Trong cookbook, một biểu mẫu không hề nhắc đến ngày kickoff cho `mode = absolute` nhưng không có tháng; code phát hiện "absolute date incomplete", confidence 0.46, và gắn cờ [29].

### Tạo ứng viên bằng code, để model chọn

**Email, số điện thoại, số tiền.** Dùng regex tìm mọi ứng viên trong văn bản, đưa chúng làm option của một câu Choice ("đâu là số điện thoại của người gửi"), kèm option `none`. Model chọn đúng span; code chuẩn hoá span đó [30]. Giá trị đến tay code luôn là một chuỗi thật sự có trong văn bản.

```python
import re
# số di động VN: 0 hoặc +84, đầu số 3/5/7/8/9, thêm 8 chữ số; cho phép dấu cách, chấm, gạch
PHONE = re.compile(r"(?<![\d+])(?:\+84[\s.-]?|0)[35789](?:[\s.-]?\d){8}(?!\d)")

def pick_phone(text, role="the sender's contact number"):
    candidates = sorted(set(m.group().strip() for m in PHONE.finditer(text)))
    if not candidates:
        return None
    options = {f"c{i}": c for i, c in enumerate(candidates)}
    options["none"] = "None of the listed numbers is " + role
    a = decide({"text": text}, {"pick": choice(f"Which number in `text` is {role}?", options)})["pick"]
    if a["choice"] == "none" or a["probabilities"][a["choice"]] < 0.7:
        return None
    return normalize_phone(options[a["choice"]])
```

Regex trên chỉ nhận số di động Việt Nam 10 chữ số (dạng `0912 345 678`, `090-123-4567`, `+84 912 345 678`); số bàn, số quốc tế khác và số viết liền không có `+` (`84912…`) không được nhận. Muốn phủ rộng hơn, thay regex bằng thư viện như `phonenumbers` và chỉ giữ ứng viên mà thư viện xác nhận hợp lệ. Nguyên tắc vẫn vậy: code quyết định chuỗi nào *là* số điện thoại, model chỉ quyết định số nào đóng vai trò cần tìm.

Với danh sách năm dài, có thể dùng cùng kỹ thuật: lấy các số trông giống năm trong văn bản và chỉ đưa những số đó làm option [29].

## 3.9 Chọn function và tham số

**Bài toán.** Biến một câu lệnh tự nhiên thành lời gọi một trong N hàm có sẵn, khi tham số của các hàm là tập đóng (enum, `Literal`, `bool`).

**Thiết kế.** Cookbook của TypeSafe làm với 10 hàm của một trợ lý giao dịch chứng khoán [31]:

- Đọc type hint để phân loại tham số: **choice** (`Literal`, chọn một), **set** (`list[Literal]`, chọn nhiều) và **flag** (`bool`). Tham số `int`, chuỗi tự do hay ngày tháng không có câu hỏi; hàm dùng giá trị mặc định.
- Viết một spec: mô tả mỗi hàm, một câu hỏi cho mỗi tham số, một dòng cho mỗi option. `Literal` cho bạn chuỗi `"1mo"` và `"3mo"`; spec cho model biết người dùng gõ "quý này" nghĩa là cái thứ hai. Có thể nhờ LLM viết spec từ chữ ký hàm.
- Tham số set được hỏi một lần cho mỗi phần tử bằng câu yes/no: "Người dùng có muốn {} trong phép so sánh không?" thành một câu cho mỗi mã cổ phiếu.
- Mỗi tham số tuỳ chọn có thêm một câu yes/no `stated`: câu lệnh có nói gì về tham số này không? Nếu không, bỏ tham số để hàm dùng mặc định. Thiếu câu này, Choice sẽ chọn *một* giá trị nào đó, và chọn một cách tự tin.
- Một request chứa câu Choice chọn hàm cùng câu hỏi cho tham số của *mọi* hàm; code chỉ đọc đáp án của hàm được chọn (fan-out).

Confidence của cả lời gọi là confidence *thấp nhất* trong các phán đoán, không phải tích: một tham số sai là đủ làm hỏng lời gọi, và tích sẽ giảm dần khi hàm có nhiều tham số dù không phán đoán nào yếu [31].

**Kết quả tham khảo** [31]: "plot rolling correlation between nvda and spy for the past month" điền đúng bốn tham số từ một câu, kể cả hai tham số cùng lấy từ sáu mã cổ phiếu (`symbol` và `benchmark`), nhờ câu hỏi mô tả rõ vai trò: "mã được đo, nêu trước" và "mã thứ hai, làm chuẩn". "is amd tracking nvidia lately" đến đúng `rolling_correlation` dù spec không có chữ *tracking* hay *lately*; `window` và `resolution` bị bỏ vì "lately" không nói bao xa hay khung thời gian nào.

**Lưu ý.** Viết câu hỏi về *ý nghĩa*, không về từ người dùng có thể gõ, vì so khớp là theo nghĩa. Tránh đặt câu hỏi theo tên tham số: "Which resolution?" không cho câu lệnh gì để khớp. Với tham số tự do, dùng [tool use của LLM](../tool-use/).

## 3.10 Chọn skill hoặc tool cho agent

**Bài toán.** Agent có nhiều skill thường thấy chúng qua một index: mỗi skill một dòng, description bị cắt ngắn để không chiếm chỗ hội thoại. Hermes (agent harness của Nous Research) cắt ở 60 ký tự. Ở độ dài đó, skill *sửa* file `.pptx` và skill *tạo* slide tài chính đều đọc như "công cụ PowerPoint". Khi không skill nào phù hợp, danh sách tên vẫn kích thích model đoán [32].

**Thiết kế.** Hai request trước khi agent quyết định:

1. **Xếp hạng toàn bộ roster.** Một câu Choice trên cả 194 skill, criteria là description ngắn (đúng thứ agent thấy); xác suất là bảng xếp hạng. Cùng request có ba câu yes/no hỏi theo ba cách khác nhau xem lượt này có muốn *thực hiện một hành động* không (lấy trung bình, đảo chiều câu nào hỏi ngược), và một câu yes/no `named`: lượt này có nêu một công cụ, format, template hay phương pháp cụ thể không. Chỉ khi cả hai lý do đều thấp mới kết luận không cần skill.
2. **Kiểm tra lại top 3.** Một câu Choice trên ba ứng viên, lần này criteria là description đầy đủ cộng phần mở đầu `SKILL.md` của từng skill; cộng một câu yes/no `fits::{name}` cho mỗi ứng viên: skill này có làm đúng việc được yêu cầu không. Nếu mọi câu `fits` đều thấp, bỏ cả shortlist.

Kết quả là tối đa một tên skill, đưa vào system prompt như một gợi ý có thể bỏ qua. Gợi ý nằm trong một khối riêng sau roster để phần roster giống hệt nhau mọi lượt và giữ được prefix cache ([Multi-turn Conversation & Prompt Caching](../multi-turn-conversation/)).

**Kết quả tham khảo** (`jev-1.14.0` và `claude-haiku-4-5-20251001`, 452 request, 280 có đúng một skill phù hợp, 172 không skill nào phù hợp) [32]:

| | Load sai skill | Load skill khi không cần |
|---|---|---|
| Agent tự chọn từ roster | 15,7% | 14,0% |
| Agent có gợi ý từ Jev | 7,9% | 5,2% |
| Agent được cho sẵn đáp án đúng (mức sàn) | 3,2% | 4,1% |

Gợi ý sửa đúng 29 request và làm hỏng 7 request mà agent tự làm đúng: một gợi ý sai nhưng tự tin kéo agent khỏi đáp án đúng của nó.

**Lưu ý.**

- Câu hỏi "có cần hành động không" phải hỏi về hành động, không về chủ đề: "giải thích monad là gì" và một yêu cầu cần skill code đều về phần mềm.
- Lời gợi ý nên nói rõ là có thể bỏ qua; ép mạnh hơn sẽ khiến agent nghe theo cả gợi ý sai. Khi không có gì để gợi ý, vẫn gửi một câu nói vậy, để cân lại chỉ dẫn "khi nghi ngờ hãy load" có sẵn trong roster.
- Roster lớn hơn vài lần thì chia thành nhiều phần, xếp hạng từng phần rồi xếp hạng lại các ứng viên thắng.

## 3.11 Routing giữa các LLM

**Bài toán.** Gửi mọi request cho model mạnh nhất thì tốn; gửi mọi thứ cho model rẻ thì hỏng các ca khó. Router cần quyết định cho mỗi prompt: model nào, có cần reasoning không. TypeSafe liệt kê model routing trong các use case chính: phân loại intent và miền, ước lượng độ khó và rủi ro, escalate request cần model đắt hơn [33].

**Câu hỏi.** Một request với Choice về miền, Score về độ khó, và vài câu yes/no về các đặc điểm làm thay đổi lựa chọn model:

```python
ROUTER_Q = {
    "domain": choice("What is `prompt` mainly about?", {
        "coding": "Writing, reviewing, or debugging code",
        "math": "Calculations, proofs, quantitative reasoning",
        "writing": "Drafting, editing, translating prose",
        "lookup": "Factual questions answerable in a sentence or two",
        "chitchat": "Greetings, small talk",
    }),
    "difficulty": score("How much reasoning does answering `prompt` well require?", [
        "One step: recall or a direct rewrite",
        "A few steps a capable person does without notes",
        "Many dependent steps, careful checking, or planning",
    ]),
    "needs_tools": noul("Does answering `prompt` require current information or actions outside the conversation?"),
    "high_stakes": noul("Could a wrong answer to `prompt` cause financial, legal, medical, or safety harm?"),
}

def pick_model(prompt):
    a = decide({"prompt": prompt}, ROUTER_Q)
    d = a["difficulty"]
    p_hard = d["probabilities"]["2"]                # xác suất của mức khó nhất
    if a["high_stakes"]["noul"] > 0.5 or p_hard > 0.5:
        return "reasoning-model"
    if d["confidence"] < 0.5 or a["domain"]["choice"] == "coding":
        return "strong-model"
    return "cheap-model"
```

(Đây là ví dụ thiết kế; chưa có cookbook hay số liệu công bố cho router cụ thể này.)

**Lưu ý.**

- Ngưỡng cho `high_stakes` nên thấp hơn các ngưỡng khác: bỏ sót một ca rủi ro đắt hơn gửi nhầm một ca dễ cho model mạnh.
- Đo router bằng chất lượng và chi phí của *cả hệ thống*: chạy một tập prompt qua router và qua model mạnh nhất, so điểm của LLM-judge hoặc người chấm và tổng chi phí ([Evaluation](../evaluation/)).
- Router phải rẻ và nhanh hơn nhiều so với khoản tiết kiệm. Với decision model, thường là vậy; với một LLM làm router thì chưa chắc.
- Xem thêm [Model Routing](../model-routing/) cho fallback và rate limit.

## 3.12 Phân loại theo taxonomy lớn

### Taxonomy vừa một câu Choice

**Bài toán.** Phân loại báo cáo thường niên SEC vào 75 nhóm ngành SIC. Phần lớn dễ (ngân hàng khu vực là ngân hàng khu vực), một số khó (công ty vừa bán một trong hai mảng, startup mô tả mảng *dự định* làm) [34].

**Thiết kế.** Một câu Choice 75 option, mỗi option mô tả bằng danh sách các ngành bên trong nó (vì 42 trong 75 nhóm có tên chung chung, số còn lại không có tên). Confidence ≥ 0.9 thì báo cáo nhóm ngành; dưới 0.9 thì báo cáo *division* cha của nhóm đó. Division suy ra từ nhóm, không cần gọi thêm.

**Kết quả tham khảo** (`jev-1.12`, 12/8/2026, 60 báo cáo) [34]: ngưỡng 0.9 chia đôi tập. Nửa tự tin đúng 90%; nửa còn lại chỉ đúng 40% ở cấp nhóm, nhưng báo cáo ở cấp division thì đúng 70%. Ba báo cáo có confidence thấp nhất khó vì lý do đọc được ngay: hai công ty mô tả mảng định làm, một công ty vừa bán một mảng vài tuần trước khi nộp.

Ý tưởng chung: **khi không chắc, lùi lên một tầng thay vì bỏ hay chuyển người.** Mọi văn bản vẫn có nhãn dùng được. Nếu tầng cha quá thô để hành động, đó là chỗ chuyển cho người.

### Taxonomy sâu: đi từng tầng với beam search

**Bài toán.** Taxonomy hàng nghìn nút nhiều tầng (phân loại sáng chế CPC, taxonomy sản phẩm Shopify, MeSH y sinh) không vừa một câu Choice [35].

**Thiết kế.** Mỗi nút hỏi một câu Choice trên các con của nó; xác suất là trọng số cạnh. Greedy search đi theo con có xác suất cao nhất đến lá, nhưng một lựa chọn sai ở tầng đầu không sửa được. Beam search giữ K đường tốt nhất ở mỗi tầng; K câu hỏi của một tầng chạy song song, nên số vòng gọi vẫn bằng số tầng như greedy.

```python
import math

def path_score(edge_probs):
    # trung bình nhân, để lá sâu không bị phạt chỉ vì cần nhiều quyết định hơn
    return math.exp(sum(math.log(max(p, 1e-12)) for p in edge_probs) / len(edge_probs))

def beam_classify(doc, root, K=3):
    beams = [([root], [])]                     # (đường đi, xác suất các cạnh)
    while any(children(path[-1]) for path, _ in beams):
        frontier = []
        for path, probs in beams:              # production: gửi K request song song
            node = path[-1]
            if not children(node):
                frontier.append((path, probs)); continue
            a = decide({"document": doc}, {"next": choice(
                f"Which category under '{label(node)}' best fits `document`?",
                {c.id: c.description for c in children(node)})})["next"]
            for child_id, p in a["probabilities"].items():
                frontier.append((path + [child_id], probs + [p]))
        beams = sorted(frontier, key=lambda b: path_score(b[1]), reverse=True)[:K]
    return beams[0][0][-1]
```

Báo cáo thêm `separation = top_path_score / second_path_score`: gần 1× là hai đường gần hoà, càng xa 1× càng rõ người thắng.

**Kết quả tham khảo** (`jev-1.14.0`, 3/11/2026) [35]: trên ba ví dụ, beam K=3 đúng 3/3, greedy đúng 2/3. Ca greedy sai là sản phẩm Shopify "Cat Window Beds & Perches", greedy ra "Pet Chairs". Ba ví dụ là quá ít để kết luận về độ chính xác; giá trị của cookbook là cách làm.

Một câu hỏi cho mỗi nút cũng giúp vận hành: biết lỗi phân loại xảy ra nhiều nhất ở nút nào, đếm số lần mỗi cạnh được đi qua, và viết unit test khi cập nhật taxonomy.

## 3.13 Ghép thực thể và khử trùng lặp

**Bài toán.** Gộp dữ liệu từ hai nguồn, cùng một người, sản phẩm hay tác phẩm xuất hiện ở cả hai với mô tả khác nhau. Gộp nhầm thì trộn dữ kiện của hai thực thể, rất khó gỡ. Bỏ sót thì để lại bản trùng. Các cặp không chắc cần lựa chọn thứ ba: chuyển cho người phụ trách dữ liệu [36].

**Thiết kế.** Một bước rẻ (so tên gần giống) chọn cặp ứng viên trước. Với mỗi cặp, một request với state `{"entity_a": …, "entity_b": …}`:

- Một câu **Score** ba mức: để riêng, chuyển người xem, đề xuất gộp. Score giữ được thứ tự của ba kết quả và mô tả rõ mức giữa; Choice làm mất thứ tự; yes/no chỉ đến được mức giữa gián tiếp qua ngưỡng.
- Ba câu yes/no: cùng tên, cùng nghệ sĩ, cùng hãng, để người xem biết hai record lệch nhau ở đâu.

```python
LEVELS = [
    "Different entities: the records describe different works",
    "Needs a curator: a reissue, a compilation, a special or live edition, or a title that could refer to either",
    "Same entity: the records describe the same work",
]
def route_pair(a):
    s = a["decision"]["score"]
    return "unlinked" if s < 0.5 else ("review" if s < 1.5 else "assert_same_as")
```

Làm tròn score về mức gần nhất, với điểm cắt 0.5 và 1.5. Viết mô tả các mức *trước* khi nhìn score, để điểm cắt đến từ quy tắc làm tròn chứ không phải fit vào dữ liệu. So năm thì để code làm, có xử lý giá trị thiếu. Câu "cùng hãng" cần quy tắc: tên rút gọn hay chi nhánh vùng là cùng hãng (*Columbia Records* và *Columbia*); imprint và công ty mẹ là khác (*RCA Victor* và *BMG*) [36].

**Kết quả tham khảo** (`jev-1.14.0`, 16/10/2026, 450 cặp album giữa Wikidata và MusicBrainz) [36]: 57 đề xuất gộp, 79 chuyển người xem, 314 để riêng. 48 trong 57 đề xuất gộp được dữ liệu ghi là khớp; 311 trong 314 cặp để riêng được ghi là không khớp. Một số "sai" là lỗi của chính catalog, ví dụ MusicBrainz liệt kê cùng một album hai lần. Model cũng sai thật: cặp ghi nghệ sĩ *Justin Lo* ở một bên và *側田* (tên tiếng Trung của cùng ca sĩ) ở bên kia được 0.03 ở câu cùng nghệ sĩ và 0.44 tổng, ngay dưới điểm cắt.

**Lưu ý.** Nhãn trong dữ liệu không được kiểm tra thủ công, nên tỉ lệ khớp trên không phải độ chính xác đã xác minh. Ca tên viết bằng hai hệ chữ khác nhau là điểm yếu cần chú ý, nhất là với dữ liệu có tên tiếng Việt lúc có dấu lúc không.

## 3.14 Chấm điểm đa tiêu chí

**Bài toán.** Xếp hạng hồ sơ ứng viên, lead bán hàng, ticket theo độ ưu tiên: mỗi thứ phụ thuộc nhiều yếu tố độc lập.

**Thiết kế.** *Composite scoring*: một câu Score cho mỗi chiều, chuẩn hoá về 0–1 bằng cách chia cho mức cao nhất (`len(levels) - 1`), rồi cộng có trọng số trong code [37]. Cùng một bộ đáp án dùng được cho nhiều bộ trọng số:

```python
RESUME_Q = {
    "python_depth":    score("How much Python experience does `resume` show?", [...5 mức...]),
    "team_leadership": score("How much team leadership does `resume` show?", [...]),
    "system_design":   score("How much system design experience does `resume` show?", [...]),
    "generalist":      score("How broad is the range of work in `resume`?", [...]),
}
WEIGHTS = {
    "senior_ic": {"python_depth": .40, "team_leadership": .10, "system_design": .40, "generalist": .10},
    "eng_manager": {"python_depth": .15, "team_leadership": .40, "system_design": .20, "generalist": .25},
}

def rank(resumes, role):
    def total(a):
        return sum(w * a[k]["score"] / 4 for k, w in WEIGHTS[role].items())   # 5 mức: chia 4
    scored = [(total(decide({"resume": r}, RESUME_Q)), r) for r in resumes]
    return sorted(scored, key=lambda x: x[0], reverse=True)   # chỉ so điểm; hoà điểm giữ thứ tự đầu vào
```

Lợi ích chính không phải con số tổng mà là thấy được nó được tạo ra thế nào. Khi thứ hạng không khớp với đánh giá của đội, chỉnh trọng số trong code và chạy lại, không cần viết lại prompt.

**Viết mức cho tốt** [12]:

- Mô tả *tình huống*, không mô tả *mức độ*: "Hỏng một tính năng nhưng có cách khắc phục" thay vì "Nghiêm trọng vừa phải".
- Mỗi câu Score một chiều. "Đúng giờ, thông minh và có kinh nghiệm" là ba chiều; ứng viên cao chiều này thấp chiều kia không đặt vào đâu được, confidence giảm.
- Trường hợp cực đoan hiếm ở đầu thang cần hành động khác thì cho nó một mức riêng, ví dụ thêm "lăng mạ hoặc đe doạ" sau "rất tức giận".
- Khi model hay chấm lơ lửng giữa hai mức trên những ca bạn thấy rõ ràng, chuyển mỗi mức thành object có mô tả và vài ví dụ. Ví dụ chỉ giúp khi giống input thật: trên một báo cáo lỗi "chỉ hỏng trên Safari", thêm ví dụ "export hỏng trên một trình duyệt nhưng chạy trên trình duyệt khác" đưa confidence từ 0.35 lên 0.96; thêm ví dụ không liên quan đến trình duyệt thì không đổi gì. Confidence cao hơn không chứng minh đáp án đúng hơn; kiểm tra bằng ca đã biết đáp án.

**Lưu ý về công bằng.** Chấm hồ sơ tuyển dụng, tín dụng hay bảo hiểm bằng model là quyết định có hệ quả pháp lý ở nhiều nơi. Chỉ dùng tiêu chí liên quan đến công việc, không đưa vào state thông tin không được phép dùng, giữ người trong vòng quyết định, và kiểm tra kết quả theo nhóm.

## 3.15 Biến văn bản thành feature cho ML

**Bài toán.** Model ML cổ điển (gradient boosting, hồi quy) cần bảng số. Văn bản tự do (ghi chú bán hàng, review, mô tả sự cố) chứa tín hiệu nhưng không phải số.

**Thiết kế.** Mỗi câu hỏi là một cột: câu yes/no cho một cột xác suất; câu Score cho hai cột, vị trí trung bình và độ phân tán quanh nó [38]. Quy trình "autoresearch" của TypeSafe:

1. Một LLM đề xuất một loạt câu hỏi về văn bản.
2. Decision model trả lời mọi câu cho mọi dòng (mỗi dòng một request, mọi câu hỏi trong request đó).
3. Huấn luyện CatBoost trên các cột, đánh giá bằng k-fold trên tập dev.
4. Đưa cho LLM đề xuất báo cáo: câu nào CatBoost dùng nhiều, dòng nào vẫn dự đoán sai nhất. LLM đề xuất câu mới, sửa hoặc bỏ câu cũ.
5. Thêm câu mới thẳng vào; sửa hoặc bỏ thì thử trước bằng cách fit lại, chỉ giữ khi lỗi dev giảm. Fit lại không tốn API call.

**Kết quả tham khảo** (`jev-1.12` và `claude-sonnet-5`, 3/8/2026; 2.000 review rượu vang, dự đoán điểm của nhà phê bình thang 80–100; RMSE trên 800 dòng không dùng trong quá trình) [38]:

| Cách biến ghi chú thành điểm | RMSE |
|---|---|
| Dự đoán điểm trung bình | 3,09 |
| CatBoost trên đếm từ | 2,47 |
| Hỏi Jev thẳng điểm (một câu Score 10 mức, cộng một offset) | 2,15 |
| 18 câu hỏi từ một lần đề xuất, không lặp | 1,87 |
| 38 câu hỏi sau năm vòng | 1,77 |

Phần lớn cải thiện đến từ lần đề xuất đầu tiên; bốn vòng sau thêm 0,10 điểm, khoảng tin cậy 95% là [−0,147; −0,050].

**Lưu ý.**

- Số request tăng theo số dòng, không theo số câu hỏi: 100.000 dòng là 100.000 request mỗi vòng. Sửa một câu là một câu mới, tốn thêm một lượt qua mọi dòng.
- Không lọc câu hỏi trước khi trả lời: một câu chỉ áp dụng cho 1/10 số dòng có thể trông vô dụng trên mẫu nhỏ mà vẫn là cột hữu ích nhất.
- Giữ tập test không dùng cho cả việc khám phá feature lẫn chọn model. Dùng split theo thời gian nếu dự đoán tương lai.
- Score 10 mức là tối đa; 11 mức trả lỗi server [38].

## 3.16 Ảnh và tài liệu scan

> Ảnh và Box trên Jev đang ở trạng thái thử nghiệm ("dreamer-ready"): chưa đủ chuẩn production, format và kết quả có thể đổi giữa các bản, cần header `TypeSafe-Danger-Zone-Images: true` [13]. Clef và OpenAI Decisions API cũng nhận ảnh ([§2.3](#23-cloudflare-clef), [§2.4](#24-openai-decisions-api)).

**Bài toán.** Kiểm tra giá trị mà pipeline OCR cộng LLM đọc từ ảnh hoá đơn chụp bằng điện thoại. Phần lớn đúng, nhưng kiểm tra tay từng giá trị lâu ngang nhập lại [39].

**Thiết kế.** Ba request:

1. **Toàn trang.** Với mỗi field, một câu Box tìm vị trí (đưa giá trị đã trích làm gợi ý) và một câu yes/no "giá trị trích ra có phải là field này trên trang không". Xác suất của câu yes/no là *page check*.
2. **Phóng to.** Cắt một vùng quanh box đầu (tối thiểu 30% chiều rộng và 15% chiều cao trang để field không lọt ra ngoài nếu box đầu lệch), phóng to, hỏi lại câu Box. Ảnh của mỗi vùng nằm trong `instructions` của câu hỏi tương ứng, mọi vùng chung một request. Nới box mới ra mỗi cạnh một chút.
3. **Kiểm tra vùng cắt.** Cắt theo box đã nới, hỏi một câu yes/no mỗi field: vùng cắt này có hiện giá trị đã trích không. Đây là *crop check*.

Field có một trong hai check dưới 0.25 thì chuyển người xem.

**Kết quả tham khảo** (`jev-1.14.0`, 15/10/2026, hai hoá đơn nhà cung cấp, một bản carbon nhàu, một bản có bóng đổ giữa trang) [39]:

| Field | Đã trích | In trên hoá đơn | Page check | Crop check |
|---|---|---|---|---|
| Số hoá đơn | `AN400074` | `IN400074` | 0.88 | 0.18 |
| Số hoá đơn | `INV03070` | `INV03079` | 0.02 | 0.04 |
| Ngày | `16/09/2028` | `16/09/2026` | 0.17 | 0.66 |

Không check nào một mình bắt được cả ba lỗi: crop check bắt `AN400074` mà page check cho qua, page check bắt ngày sai mà crop check cho qua. Mọi giá trị đúng đều qua cả hai check (thấp nhất 0.35 và 0.31).

**Lưu ý.** Jev nhận ảnh 280–1568 px mỗi cạnh; ảnh lớn hơn được thu nhỏ, vùng cắt nhỏ được phóng to. Ảnh vẫn quá nhỏ thì đặt giữa nền trắng và giữ vị trí, để đổi box trên nền về toạ độ trên vùng cắt [39]. Cookbook khác của TypeSafe dùng cùng ý tưởng cho hồ sơ tiêm chủng scan (yes/no "có vaccine X không" cộng Box tìm vị trí, Choice đọc năm với option "không đọc được") và cho log camera chim [33].

---

# Phần 4 — Thiết kế và vận hành

## 4.1 Viết state

**Chỉ gửi những gì câu hỏi cần.** Độ chính xác giảm khi state chứa nhiều nội dung không liên quan, và state lớn khiến khó biết phần nào gây ra đáp án sai [17]. Lọc và retrieve trong code trước. Không dựa vào kiến thức trong weights khi thông tin hiện tại có thể lấy từ knowledge base của bạn [3].

**Dùng object hơn chuỗi.** Đặt tên cho từng phần để quan hệ giữa chúng rõ ràng [22]:

```json
{
  "ticket": {
    "subject": "Duplicate charge",
    "messages": [
      {"from": "customer", "text": "I was charged twice for order A-104. Please refund the duplicate."},
      {"from": "support", "text": "We are checking the charges."}
    ]
  },
  "order": {"id": "A-104", "charges": [{"amount_usd": 49, "status": "captured"}, {"amount_usd": 49, "status": "captured"}]},
  "refund_policy": "Duplicate charges are eligible for a refund."
}
```

**Trỏ câu hỏi vào đúng phần bằng đường dẫn có backtick**: "Does `refund_policy` support the refund requested in `ticket.messages[0].text`, given `order.charges`?" [2].

**Gộp những gì cần so sánh với nhau vào cùng một state.** Đơn hàng, hội thoại và chính sách là một state nếu quyết định cần đối chiếu chúng. Câu hỏi về một cặp (query–passage, record A–record B) thì đặt cả cặp vào state.

**Tách quy tắc tất định ra khỏi model.** `days_overdue > 30` là một phép so sánh, không phải phán đoán [3].

## 4.2 Viết câu hỏi và criteria

**Một câu hỏi một phán đoán, phán đoán quan sát được.** "Khách có giận *và* đòi hoàn tiền không?" là hai câu. TypeSafe gọi việc tách câu hỏi là "có lẽ là khái niệm quan trọng nhất" trong hướng dẫn của họ [3].

**Viết để giá trị cao nghĩa là "có".** "Tin có chứa dữ liệu cá nhân không?" rõ; "Tin có sạch dữ liệu cá nhân không?" đảo nghĩa, và code đọc sau sẽ hiểu ngược [6].

**Ranh giới rõ.** "Ứng viên có *bất kỳ* kinh nghiệm Python nào không?" tốt vì "bất kỳ" không để vùng giữa. Ranh giới tinh tế thì thêm criteria `true` / `false`. Thử có và không có criteria trên dữ liệu thật, giữ cách tốt hơn [6].

**Câu hỏi là một câu đầy đủ.** Tên câu hỏi không được gửi; option name có được gửi. Đặt option name có nghĩa (`desk_pro`, không phải `opt_1`), và mô tả để phân biệt các option với nhau.

**Dùng cấu trúc khi cần** [3]:

- Câu hỏi cần ngữ cảnh hoặc ví dụ dài: đặt vào field có tên cạnh câu hỏi, để code thêm hoặc đổi mà không phải viết lại câu hỏi.
- Một phần câu hỏi đến từ code (record từ database): đặt vào field riêng, không nối chuỗi template.
- Option dễ nhầm: mô tả bằng object có `what`, `not_for`, `examples`. Dùng cùng tên field cho mọi option để model so sánh được.

```json
{
  "type": "noul",
  "instructions": {
    "question": "Is the resume in `state` from the same person as `potential_duplicate`?",
    "potential_duplicate": {"id": 18, "name": "Nguyen Van An", "city": "Da Nang", "employer": "FPT Software"}
  }
}
```

**Nhớ model đọc theo nghĩa đen.** Khi nhìn một đáp án sai và thấy mình đang giải thích "ý tôi là…", phần giải thích đó là nửa còn thiếu của câu hỏi [17].

## 4.3 Các pattern kết hợp

Năm pattern chính thức của TypeSafe [40], áp dụng được cho mọi decision model:

| Pattern | Làm gì | Khi nào |
|---|---|---|
| **Speculative fan-out** | Mọi câu hỏi về cùng state trong một request, kể cả câu có thể không cần | Mặc định ([§1.6](#16-câu-hỏi-độc-lập-và-chạy-song-song)) |
| **Screen bằng yes/no trước** | Một câu yes/no rẻ trên state nhỏ; chỉ chạy bước đắt khi rõ là "có" | Bước sau cần state lớn hơn (phải fetch thêm), cần LLM, hoặc cần người |
| **Companion yes/no** | Kèm câu Choice một câu yes/no "có option nào phù hợp không" | Danh sách option có thể không phủ hết input |
| **Routing theo confidence** | Kiểm tra confidence của Score trước, rồi mới rẽ nhánh theo score | Score có thể bị phân tán giữa các mức xa nhau |
| **Routing theo xác suất** | Mỗi hành động một ngưỡng; hành động rủi ro hơn ngưỡng cao hơn | Các nhánh có hệ quả khác nhau |

### Companion yes/no thay cho option `other`

Trong ví dụ đèn bàn ở [§1.4](#14-bốn-loại-câu-hỏi), Choice chọn "bàn" 0.96, nhưng câu yes/no "tin có nói về một trong các sản phẩm của chúng ta không" cho 0.02 [41]:

| Tin nhắn | Choice | *p*<sub>max</sub> | Companion |
|---|---|---|---|
| Motor bàn đứng 140 cm kêu rè rè | `desk_pro` | 1.00 | 0.98 |
| Ghế đến rồi mà tay đỡ màn hình chưa thấy | `monitor_arm` | 1.00 | 0.99 |
| Đèn bàn hỏng sau hai tuần | `desk_pro` | 0.96 | 0.02 |
| Recruiter hỏi trưởng bộ phận vận hành có rảnh không | `desk_pro` | 0.81 | 0.02 |

Đọc companion trước: rõ là không thì bỏ qua Choice; rõ là có thì kiểm tra thêm xác suất của option được chọn; ở giữa thì chuyển người với Choice làm gợi ý. TypeSafe cho biết trong thử nghiệm của họ, cách này thường tốt hơn thêm option `other`, vì `other` tranh xác suất với các option thật: một tin nửa về sản phẩm nửa về chuyện khác chia xác suất giữa `other` và sản phẩm đó, và không con số nào cho code biết phải làm gì [41].

Biến thể: khi một input có thể thuộc nhiều option, hỏi mỗi option một câu yes/no riêng thay vì một câu Choice; trường hợp không option nào đúng tự nhiên xuất hiện khi mọi câu đều thấp.

### Screen trước

Hộp thư chung nhận đơn hàng, newsletter, biên lai, lời cảm ơn. Triage đầy đủ cần fetch tài khoản và lịch sử đơn. Một câu yes/no chỉ nhìn tin nhắn, "người gửi có đang chờ trả lời không", quyết định tin nào đáng triage [42]:

| Tin | `expects_reply` |
|---|---|
| Đặt bàn hai tuần rồi, tracking không cập nhật, ngày 30 tôi chuyển nhà | 0.98 |
| Báo để biết: thiếu một con ốc, tôi có ốc dự phòng nên lắp xong rồi | 0.44 |
| Cảm ơn, bàn đến hôm qua, rất ổn | 0.10 |
| Giảm 20% mọi bàn ghế đến chủ nhật. Unsubscribe | 0.02 |
| Thanh toán $129.00 thành công, không trả lời email này | 0.02 |

Dưới 0.3 lưu trữ; trên 0.7 fetch dữ liệu và triage; ở giữa chuyển người liếc qua. Nếu bước sau dùng cùng state thì đừng tách, dùng fan-out.

## 4.4 Đặt ngưỡng

**Ngưỡng là quyết định về rủi ro, không phải hằng số của model.** Cùng một hệ thống, hành động khác nhau dùng ngưỡng khác nhau. Ví dụ voice banking của TypeSafe [43]:

```python
a = decide({"command": transcript}, {"intent": choice("What does the user want to do?", {
    "check_balance": "View account balance",
    "approve_transfer": "Approve the pending withdrawal request",
    "other": "Anything else"})})["intent"]
p = a["probabilities"][a["choice"]]

if a["choice"] == "other" or p < 0.7:
    send_to_support_agent()
elif a["choice"] == "check_balance":
    show_balance()                              # rủi ro thấp: xem nhầm màn hình là sửa được
elif p > 0.9:
    show_transfer_details_and_require_confirmation()
else:
    ask_user_to_clarify()
```

Không ngưỡng nào tự chuyển tiền: lệnh chuyển rõ ràng mở màn hình xác nhận số tiền và người nhận. Bước xác nhận cần thiết vì một câu lệnh ngập ngừng vẫn có thể có ý định rõ: "Ờ, cái vụ chuyển tiền… ừ, làm đi" được 0.99 cho `approve_transfer`. Đáp án nói người dùng muốn hành động nào, không nói người dùng đã kiểm tra chi tiết chưa [43].

**Ba vùng.** Bắt đầu với: trên ngưỡng cao thì hành động tự động; giữa hai ngưỡng thì thận trọng (hỏi lại, gắn cờ, lấy thêm thông tin); dưới ngưỡng thấp thì không hành động, chuyển người hoặc hệ thống khác [14]. Với yes/no, điểm khởi đầu 0.7 và 0.3 [6].

**Nâng ngưỡng khi hành động sai đắt** (gọi người trực, hoàn tiền). **Hạ ngưỡng khi bỏ sót đắt** (không gắn cờ một vấn đề an toàn) [6].

**Đặt từ dữ liệu, không từ cảm giác.** Lấy vài trăm ví dụ có nhãn từ traffic thật, chạy model, vẽ độ chính xác theo xác suất hoặc confidence (ví dụ chia 10 khoảng, mỗi khoảng tính tỉ lệ đúng), rồi chọn ngưỡng theo chi phí của false positive và false negative. Đừng fit ngưỡng sát vào vài ca test [6]. Lưu đáp án: đổi ngưỡng chỉ cần chạy lại logic định tuyến, không cần gọi lại model.

**Score có ranh giới quan trọng: cộng xác suất, đừng dùng score.** Nếu "từ mức 3 trở lên thì gọi người trực", tính `sum(p for level, p in probs.items() if int(level) >= 3)`. Một đáp án chia đều giữa mức 2 và 3 có score 2.5 và confidence tương đối cao vì hai mức kề nhau, nhưng thực chất là 50/50 về quyết định của bạn [44].

## 4.5 Eval và so sánh model

Benchmark của nhà cung cấp cho biết model nào đáng thử. Dữ liệu của bạn cho biết model nào dùng được. Một bộ eval tối thiểu:

1. **Tập có nhãn.** Vài trăm request thật (đã ẩn danh), gán đáp án đúng cho từng câu hỏi. Giữ riêng một phần không nhìn vào khi viết câu hỏi và chỉnh ngưỡng.
2. **Độ chính xác** theo từng câu hỏi, và theo quyết định cuối của code (sau định tuyến).
3. **Calibration.** Chia đáp án theo khoảng xác suất, so tỉ lệ đúng thực tế với xác suất trung bình của khoảng. Model calibrate tốt thì hai số gần nhau. Đây là thứ cho phép bạn tin ngưỡng.
4. **Self-consistency.** Gửi lại cùng một request nhiều lần (thêm một field ngẫu nhiên vào state để tránh cache), đo xác suất dao động bao nhiêu và quyết định đổi bao nhiêu lần. Không cần nhãn [19].
5. **Độ nhạy thứ tự option.** Đảo thứ tự option của các câu Choice, kiểm tra đáp án có giữ nguyên không [17].
6. **Ca đối kháng.** Input cố tình kéo đáp án: chỉ dẫn chèn vào, văn bản tự biện hộ cho nhãn của nó ([§4.8](#48-bảo-mật)).
7. **Độ trễ p50, p95 và chi phí** mỗi quyết định ở tải thật.

Chạy cùng bộ eval khi: đổi câu hỏi hoặc criteria, đổi version model, đổi provider. Xem thêm [Evaluation](../evaluation/).

Để so decision model với LLM một cách công bằng, đưa LLM cùng câu hỏi và cùng option, yêu cầu trả xác suất cho từng option qua structured output, như thử nghiệm của TypeSafe [19]. So độ chính xác, calibration, độ ổn định, độ trễ và chi phí; không chỉ độ chính xác.

## 4.6 Chi phí, độ trễ và rate limit

**Chi phí gần như chỉ là input.** Với giá $0.042 / 1M token của Jev, một request 1.000 token input tốn khoảng $0.000042; một triệu quyết định như vậy khoảng $42. Với OpenAI ($0.10 / 1M), khoảng $100. Hai đòn bẩy chính:

- **Gộp câu hỏi** về cùng state vào một request: state chỉ tính một lần. Ví dụ GDPR ở [§1.6](#16-câu-hỏi-độc-lập-và-chạy-song-song): rẻ hơn 12,2 lần [15].
- **Rút gọn state**: đòn bẩy này giúp cả chi phí lẫn độ chính xác ([§4.1](#41-viết-state)).
- Với OpenAI, **prompt caching**: giữ tài liệu tham chiếu chung trong `input` (từ 1.024 token), đổi câu hỏi giữa các request [8].

**Chi phí tăng theo số cặp** trong các bài toán so sánh (rerank, lọc passage, ghép thực thể): một request cho mỗi cặp. Giữ shortlist nhỏ bằng bước rẻ phía trước.

**Độ trễ** chủ yếu phụ thuộc độ dài state và vị trí mạng. Các con số công bố: TypeSafe nói phần lớn call khoảng 100 ms [3]; Cloudflare đo median Clef-flash 38,8 ms, Clef 209 ms, Jev 524 ms từ hạ tầng của họ [10]. Đo từ vị trí deploy của bạn.

**Rate limit** tính theo token/giây và request/giây (Jev) [7]. Khi chạy song song lớn (rerank, autoresearch), tăng số worker từ từ; cookbook ghi tám worker đã đủ chạm rate limit trên một key dùng chung [38]. Retry `429` và `529` với exponential backoff; SDK của TypeSafe tự làm [23].

**Cache kết quả** theo hash của (model version, state, questions). Mọi cookbook của TypeSafe đều làm vậy. Đổi câu hỏi làm đổi key, nên lần chạy sau tự gọi lại.

## 4.7 Version và chuyển provider

**Pin version.** Alias (`jev-latest`) dịch chuyển khi có bản mới, và đáp án có thể đổi mà code không đổi. Nếu đã tune ngưỡng theo một version, gửi version ID đó, và chuyển sang bản mới khi đã chạy lại eval [7]. Log field `model` của response cho mọi quyết định.

**Đọc changelog.** TypeSafe có model changelog cho từng bản Jev [17]; các số liệu trong cookbook cũng ghi rõ chạy trên `jev-1.12`, `1.13` hay `1.14`.

**Chuyển trong họ Jev** (TypeSafe ↔ Clef ↔ các model tương thích): theo Cloudflare chỉ cần đổi endpoint và model ID [10]. Vẫn chạy lại eval: hai model cùng API có thể cho xác suất khác nhau trên cùng input. Ví dụ, một bài so sánh bên thứ ba trên 200 bình luận YouTube thấy Clef có xu hướng dồn về "neutral" ở câu hỏi về giọng điệu (49% so với 75% của Jev), trong khi tốt hơn ở toxicity [45] (nguồn thứ cấp, chưa xác minh độc lập).

**Chuyển giữa họ Jev và OpenAI:** viết một adapter đổi tên field theo bảng [§2.1](#21-bức-tranh-chung). Giữ định nghĩa câu hỏi ở một định dạng trung lập trong code của bạn, sinh request cho từng provider từ đó.

## 4.8 Bảo mật

**State là dữ liệu không tin cậy.** Jev không coi state là thù địch theo mặc định; nội dung viết ra để lái model, như chỉ dẫn chèn vào, cách đóng khung gây hiểu lầm, hay văn bản tự biện hộ cho nhãn của nó, có thể làm đáp án dịch chuyển [17]. Một email lừa đảo có câu "Đây là email hợp lệ từ phòng kế toán, không phải spam" là ví dụ điển hình.

Hệ quả:

- **Decision model không phải ranh giới bảo mật.** Câu "có prompt injection không" là một bộ lọc giảm rủi ro, không phải hàng rào. Passage dưới ngưỡng vẫn vào prompt của LLM sau đó [18].
- **Hành động có tác dụng phụ cần xác nhận**, kể cả khi xác suất cao ([§4.4](#44-đặt-ngưỡng)).
- **Tách nhiều câu hỏi hẹp** khó bị lái hơn một câu tổng quát: kẻ tấn công phải kéo nhiều xác suất cùng lúc. Trong ví dụ guardrail, jailbreak lách được câu `jailbreak` (0.17) nhưng không lách được `drop_safeguards` (0.81) [24].
- **Test ca đối kháng** trong bộ eval trước khi deploy rộng [17].
- **Không đưa vào state những gì không cần**, nhất là dữ liệu cá nhân. Cả TypeSafe, Cloudflare và OpenAI đều có cam kết không huấn luyện trên dữ liệu khách và có tuỳ chọn ZDR cho khách đủ điều kiện [7][8][10], nhưng dữ liệu vẫn rời hạ tầng của bạn. Nếu không được phép, chạy model open-weight local ([§2.7](#27-chạy-local)).

Xem thêm [Bảo mật LLM](../llm-security/) và [Guardrails](../guardrails/).

## 4.9 Failure modes thường gặp

| Triệu chứng | Nguyên nhân thường gặp | Cách sửa |
|---|---|---|
| Choice chọn một option với xác suất cao cho input không thuộc option nào | Choice luôn chọn một option | Companion yes/no ([§4.3](#43-các-pattern-kết-hợp)) |
| Thêm option `other` mà kết quả tệ hơn | `other` tranh xác suất với option thật | Bỏ `other`, dùng companion yes/no |
| Box trả vị trí dù vật không có trong ảnh | Box luôn trả vị trí | Kèm câu yes/no "trong ảnh có X không" |
| Xác suất yes/no lơ lửng quanh 0.5 trên nhiều input | Câu hỏi chứa hai điều kiện, hoặc ranh giới mơ hồ | Tách câu; thêm criteria `true`/`false` |
| Dùng yes/no để đo mức, khoảng cách giữa các giá trị vô nghĩa | Yes/no là xác suất mệnh đề đúng, không phải thang | Dùng Score với mức mô tả tình huống |
| Score lơ lửng giữa hai mức trên ca bạn thấy rõ | Mức mô tả bằng tính từ hoặc số, chồng lấn | Mô tả tình huống; thêm ví dụ giống input thật |
| Confidence của Score thấp | Mức chồng lấn, câu hỏi đo nhiều chiều, hoặc state thiếu thông tin | Tách chiều; viết lại mức; bổ sung state |
| Quyết định sai phía ranh giới dù confidence cao | Đáp án chia giữa hai mức kề nhau quanh ngưỡng | Cộng xác suất các mức ở phía quan trọng ([§4.4](#44-đặt-ngưỡng)) |
| Đếm, cộng, so sánh ngày sai | Model không tính toán | Làm trong code; hỏi từng phần tử hoặc thành phần |
| Độ chính xác giảm khi thêm ngữ cảnh | State có nhiều chi tiết thừa | Lọc trước; trỏ câu hỏi vào đúng field |
| Đáp án hiểu sai ý dù câu hỏi "rõ ràng" | Model đọc theo nghĩa đen | Viết điều kiện chính xác; đưa ca biên vào criteria |
| Đáp án đổi khi đảo thứ tự option | Thiên vị option đứng đầu | Đảo thứ tự khi eval; nếu vẫn lệch, ngẫu nhiên hoá thứ tự và lấy trung bình |
| Đáp án đổi sau một ngày mà code không đổi | Dùng alias, model đã lên version | Pin version ID; log `model` |
| `400 max_tokens_exceeded` | State cộng câu hỏi vượt context per-request hoặc per-question | Rút gọn state; chia câu hỏi sang nhiều request cùng state; long context nếu thật cần |
| `400` khi gửi ảnh lên Jev | Thiếu header `TypeSafe-Danger-Zone-Images: true`, hoặc ảnh dưới 280 px | Thêm header; resize hoặc pad |
| `422` | Câu hỏi sai cấu trúc, ví dụ Score hơn 10 mức | Đọc field lỗi trong body |
| Chi phí cao bất thường | Mỗi câu hỏi một request, state bị gửi lại nhiều lần | Gộp câu hỏi về cùng state (fan-out) |
| Gặp `429` khi chạy batch | Quá nhiều worker song song | Giảm worker, backoff, cache kết quả |
| Code đọc `answer["choice"]` bị lỗi với OpenAI | Gặp đáp án `refusal` | Kiểm tra `type` trước khi đọc giá trị |
| Input đối kháng kéo được đáp án | State không được coi là thù địch | Nhiều câu hẹp; xác nhận cho hành động; không coi là ranh giới bảo mật |

## 4.10 Checklist trước khi lên production

**Thiết kế**

- [ ] Mỗi câu hỏi là một phán đoán quan sát được, viết sao cho giá trị cao nghĩa là "có".
- [ ] Mọi câu hỏi về cùng một state nằm chung một request.
- [ ] State chỉ chứa thông tin câu hỏi cần, có cấu trúc, câu hỏi trỏ đúng field.
- [ ] Toán, đếm, ngày tháng, quy tắc tất định nằm trong code.
- [ ] Câu Choice có danh sách không phủ hết input có companion yes/no; câu Box có câu yes/no đi kèm.
- [ ] Mức của Score mô tả tình huống, mỗi câu một chiều.

**Ngưỡng và định tuyến**

- [ ] Ngưỡng đặt từ dữ liệu có nhãn, theo rủi ro của từng hành động.
- [ ] Vùng không chắc chắn có đường xử lý riêng (người, LLM, hỏi lại).
- [ ] Hành động có tác dụng phụ cần xác nhận, không chỉ dựa vào xác suất.
- [ ] Đáp án được lưu để chỉnh ngưỡng mà không gọi lại model.

**Eval**

- [ ] Tập có nhãn từ traffic thật, có phần giữ riêng.
- [ ] Đã đo độ chính xác, calibration, self-consistency, độ nhạy thứ tự option, ca đối kháng.
- [ ] Đã đo trên dữ liệu đúng ngôn ngữ của sản phẩm (tiếng Việt nếu có).
- [ ] Eval chạy lại khi đổi câu hỏi, version hay provider.

**Vận hành**

- [ ] Pin version model; log `model` của response cho mọi quyết định.
- [ ] Retry `429` / `529` với backoff; giới hạn số worker song song.
- [ ] Cache kết quả theo (version, state, questions).
- [ ] Đo p50, p95 và chi phí mỗi quyết định ở tải thật, từ vị trí deploy.
- [ ] Xử lý đáp án `refusal` nếu dùng OpenAI.
- [ ] Dữ liệu cá nhân được loại khỏi state hoặc đã có thoả thuận xử lý dữ liệu phù hợp.

---

## Nguồn tham khảo

Tài liệu chính thức truy cập tháng 11/2026. Các số liệu từ cookbook của TypeSafe là kết quả do chính TypeSafe đo, ghi kèm model và ngày đo.

1. TypeSafe — System One. https://docs.typesafe.ai/concepts/system-one
2. TypeSafe — Primitives (Questions). https://docs.typesafe.ai/primitives
3. TypeSafe — How to build with TypeSafe. https://docs.typesafe.ai/concepts/how-to-build-with-system-one
4. TypeSafe — Quick start. https://docs.typesafe.ai/introduction/quickstart
5. TypeSafe — Choice. https://docs.typesafe.ai/primitives/choice
6. TypeSafe — Noul. https://docs.typesafe.ai/primitives/noul
7. TypeSafe — Models. https://docs.typesafe.ai/models
8. OpenAI — Decisions API guide. https://developers.openai.com/api/docs/guides/decisions
9. TypeSafe — AI primer. https://docs.typesafe.ai/introduction/machine-learning-primer
10. Cloudflare — Introducing Clef: our open-source decision models, and new RL fine-tuning platform. https://blog.cloudflare.com/clef-decision-models/
11. TypeSafe — Cookbook: Classification using confidence (giới hạn ~240 option). https://docs.typesafe.ai/cookbooks/classification_using_confidence
12. TypeSafe — Score. https://docs.typesafe.ai/primitives/score
13. TypeSafe — Images. https://docs.typesafe.ai/images
14. TypeSafe — Confidence. https://docs.typesafe.ai/confidence
15. TypeSafe — Cookbook: Parallel questions. https://docs.typesafe.ai/cookbooks/parallel_questions
16. TypeSafe — Pattern: Speculative fan-out. https://docs.typesafe.ai/patterns/fan-out
17. TypeSafe — Model jaggedness. https://docs.typesafe.ai/model-jaggedness
18. TypeSafe — Cookbook: Classifying RAG passages. https://docs.typesafe.ai/cookbooks/classifying_rag_passages
19. TypeSafe — Cookbook: Self-consistency: choices. https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook
20. TypeSafe — Introducing System One models and Jev. https://typesafe.ai/blog/introducing-system-one-models-and-jev
21. Cloudflare — Clef: faster, cheaper, multimodal (bài ra mắt Clef-omni, được trích trên model card). https://blog.cloudflare.com/clef-faster-cheaper-multimodal
22. TypeSafe — State. https://docs.typesafe.ai/concepts/state
23. TypeSafe — Request rules and errors. https://docs.typesafe.ai/api/errors
24. TypeSafe — Cookbook: Guardrails for LLMs. https://docs.typesafe.ai/cookbooks/llm_guardrails
25. TypeSafe — Cookbook: Twitch chat filter. https://docs.typesafe.ai/cookbooks/twitch_chat_filter
26. TypeSafe — Cookbook: Re-ranking. https://docs.typesafe.ai/cookbooks/rerank_typesafe
27. TypeSafe — Cookbook: Double-checking citations. https://docs.typesafe.ai/cookbooks/citation_check
28. TypeSafe — Cookbook: SDE cascade. https://docs.typesafe.ai/cookbooks/sde_cascade
29. TypeSafe — Cookbook: Date extraction. https://docs.typesafe.ai/cookbooks/date_extraction_cookbook
30. TypeSafe — Cookbook: Pre-parsed value extraction. https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook
31. TypeSafe — Cookbook: Function calling. https://docs.typesafe.ai/cookbooks/function_calling
32. TypeSafe — Cookbook: Skill suggestion. https://docs.typesafe.ai/cookbooks/skill_suggestion
33. TypeSafe — Example use cases. https://docs.typesafe.ai/concepts/use-case-map
34. TypeSafe — Cookbook: Classification using confidence. https://docs.typesafe.ai/cookbooks/classification_using_confidence
35. TypeSafe — Cookbook: Hierarchical classification. https://docs.typesafe.ai/cookbooks/hierarchical_classification
36. TypeSafe — Cookbook: Knowledge graph entity alignment. https://docs.typesafe.ai/cookbooks/entity_alignment
37. TypeSafe — Pattern: Composite scoring. https://docs.typesafe.ai/patterns/composite-scoring
38. TypeSafe — Cookbook: Autoresearch feature discovery. https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery
39. TypeSafe — Cookbook: OCR confidence and bounding boxes. https://docs.typesafe.ai/cookbooks/ocr_confidence_bounding_boxes
40. TypeSafe — Patterns. https://docs.typesafe.ai/patterns
41. TypeSafe — Pattern: Companion Noul. https://docs.typesafe.ai/patterns/companion-noul
42. TypeSafe — Pattern: Screen with a Noul first. https://docs.typesafe.ai/patterns/noul-screen
43. TypeSafe — Pattern: Routing by probability. https://docs.typesafe.ai/patterns/probability-routing
44. TypeSafe — Pattern: Routing by confidence. https://docs.typesafe.ai/patterns/confidence-routing
45. Better Stack Community — Jev vs. Clef: A Practical Comparison of System One Decision Models (nguồn thứ cấp). https://betterstack.com/community/guides/ai/jev-vs-clef-decision-model/
46. Cloudflare Workers AI — Model pages `clef`, `clef-flash`, `clef-omni` và Pricing. https://developers.cloudflare.com/workers-ai/models/clef/ · https://developers.cloudflare.com/workers-ai/platform/pricing/
47. Microsoft — Microsoft-Decision-1 on Foundry. https://commandline.microsoft.com/microsoft-decision-1-model-foundry/
48. Microsoft Learn — Use Microsoft Decision models in Foundry. https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision
49. Liquid AI — d1. https://docs.liquid.ai/lfm/models/d1 · https://huggingface.co/LiquidAI/d1-3B
50. Liquid AI — Model license. https://docs.liquid.ai/lfm/help/model-license
51. Kev-9B. https://huggingface.co/jaredpalmer/kev-9b
52. Strands Decider. https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 · https://github.com/strands-labs/strands-decider
53. Bespoke Nimble 9B v2. https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2
54. Together — Tev1-4B-experimental. https://huggingface.co/togethercomputer/Tev1-4B-experimental
55. Laya. https://huggingface.co/convaiinnovations/laya
56. Fastino — GLiNER2.5-Decide. https://huggingface.co/fastino/GLiNER2.5-Decide
57. djev (DiffusionGemma Jev). https://github.com/mmastrac/djev
58. Ollama — Ollama now supports Jev-style decision models. https://ollama.com/blog/ollama-now-supports-jev-style-decision-models
59. Jev Decision Index. https://huggingface.co/spaces/multimodalart/jev-decision-index
