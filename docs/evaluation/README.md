# Evaluation: biết hệ thống LLM tốt lên hay tệ đi

Hai tình huống minh hoạ dưới đây là giả định, con số dùng để minh hoạ.

Một đội làm trợ lý chăm sóc khách hàng sửa system prompt để bot bớt dài dòng. Họ thử mười câu hỏi quen thuộc. Câu trả lời ngắn hơn, vẫn đúng, cả đội đồng ý merge. Hai tuần sau, tỉ lệ chuyển cho nhân viên tăng từ 12% lên 19%. Đọc lại log mới thấy: với câu hỏi về đổi trả, bot giờ bỏ mất bước "kiểm tra hoá đơn còn hạn không", vì phần hướng dẫn đó nằm trong đoạn prompt bị rút gọn. Mười câu thử không có câu nào về đổi trả.

Tháng sau, đội đổi sang model mới rẻ hơn. Lần này họ đã có 200 câu hỏi kèm đáp án mẫu. Model mới đạt 84%, model cũ 86%. Có người nói "chênh 2 điểm, không đáng kể", có người nói "tệ hơn rõ ràng". Không ai biết ai đúng, vì chạy lại cùng bộ đó trên model cũ lần nữa cho ra 83%.

Hai tình huống này là hai nửa của bài toán evaluation:

- **Đo cái gì:** bộ ca thử phải phủ những việc hệ thống thực sự phải làm, và tiêu chí "đúng" phải đủ rõ để hai người chấm cùng một câu trả lời ra cùng một kết luận.
- **Đọc số thế nào:** output của LLM không tất định. Một con số không kèm khoảng tin cậy không cho biết thay đổi có thật hay chỉ là nhiễu.

Unit test truyền thống so `output == expected`. Với văn bản tự do, có hàng trăm câu trả lời đúng mà không câu nào trùng chuỗi với đáp án mẫu, và cùng một input chạy hai lần có thể cho hai kết quả khác nhau. Eval cho LLM vì vậy là kết hợp của ba thứ: grader bằng code cho những gì kiểm được bằng code, LLM làm giám khảo (LLM-as-judge) cho phần cần phán đoán, và người cho phần còn lại cũng như để kiểm tra chính giám khảo.

- **Nền tảng — Phần 1:** vì sao thử vài câu không đủ, các loại eval, cách định nghĩa "tốt", phương sai và cỡ mẫu, benchmark công khai khác eval của bạn thế nào.
- **Phương pháp và công cụ — Phần 2:** grader bằng code, LLM-as-judge và các thiên lệch của nó, eval cho RAG, agent, structured output, và các công cụ phổ biến.
- **Thiết kế và vận hành — Phần 3:** xây golden dataset, gán nhãn, regression test, đưa eval vào CI, eval online, chi phí, và [bảng failure mode ở §3.7](#37-failure-modes-thường-gặp).

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và công cụ. Một số khuyến nghị đến từ blog của người làm thực tế chứ không phải nghiên cứu có kiểm soát; những chỗ đó được ghi rõ nguồn và mức độ chắc chắn.

Tài liệu liên quan: [Prompt Engineering §3.4](../prompt-engineering/#34-regression-test-khi-đổi-prompt) (regression test riêng cho prompt), [Tool Use §3.9](../tool-use/#39-eval-cho-tool-use) (eval cho tool use, pass^k), [Embeddings §3.6](../embeddings/#36-evaluation-methodology) (metric retrieval, MTEB, BEIR), [LLM Observability](../llm-observability/) (trace và đưa log production về eval).

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Vì sao "thử vài câu" không phải là eval](#11-vì-sao-thử-vài-câu-không-phải-là-eval)
  - [1.2 Eval offline, eval online và human review](#12-eval-offline-eval-online-và-human-review)
  - [1.3 Định nghĩa "tốt": tiêu chí, rubric và metric](#13-định-nghĩa-tốt-tiêu-chí-rubric-và-metric)
  - [1.4 Phương sai: chạy bao nhiêu lần, cần bao nhiêu mẫu](#14-phương-sai-chạy-bao-nhiêu-lần-cần-bao-nhiêu-mẫu)
  - [1.5 Benchmark công khai khác eval của bạn thế nào](#15-benchmark-công-khai-khác-eval-của-bạn-thế-nào)
- [Phần 2 — Phương pháp và công cụ](#phần-2--phương-pháp-và-công-cụ)
  - [2.1 Grader dựa trên code: exact match, regex, schema, unit test](#21-grader-dựa-trên-code-exact-match-regex-schema-unit-test)
  - [2.2 LLM-as-judge](#22-llm-as-judge)
  - [2.3 Eval cho RAG: retrieval và generation](#23-eval-cho-rag-retrieval-và-generation)
  - [2.4 Eval cho tool use và agent](#24-eval-cho-tool-use-và-agent)
  - [2.5 Eval cho structured output và extraction](#25-eval-cho-structured-output-và-extraction)
  - [2.6 Công cụ: OpenAI Evals, promptfoo, Braintrust, LangSmith, Langfuse, Inspect](#26-công-cụ-openai-evals-promptfoo-braintrust-langsmith-langfuse-inspect)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Xây golden dataset từ đâu](#31-xây-golden-dataset-từ-đâu)
  - [3.2 Gán nhãn và đồng thuận giữa người gán nhãn](#32-gán-nhãn-và-đồng-thuận-giữa-người-gán-nhãn)
  - [3.3 Regression test khi đổi prompt hoặc model](#33-regression-test-khi-đổi-prompt-hoặc-model)
  - [3.4 Đưa eval vào CI](#34-đưa-eval-vào-ci)
  - [3.5 Eval online: A/B test, feedback người dùng, sampling log](#35-eval-online-ab-test-feedback-người-dùng-sampling-log)
  - [3.6 Chi phí và thời gian chạy eval](#36-chi-phí-và-thời-gian-chạy-eval)
  - [3.7 Failure modes thường gặp](#37-failure-modes-thường-gặp)
  - [3.8 Checklist](#38-checklist)
- [Nguồn tham khảo](#nguồn-tham-khảo)

---

## Tra cứu nhanh

### Chọn grader theo loại tác vụ

| Tác vụ | Grader chính | Bổ sung |
|---|---|---|
| Phân loại, routing, trích một giá trị | Code: so khớp nhãn sau chuẩn hoá | Ma trận nhầm lẫn theo từng lớp |
| Trích xuất có cấu trúc (JSON) | Code: validate schema, rồi so từng field | Judge cho field văn bản tự do (mô tả, tóm tắt) |
| Sinh code | Code: chạy unit test, lint, type check | Judge cho tính dễ đọc nếu cần |
| Tool use, agent | Code: kiểm tra trạng thái cuối của môi trường, tool nào được gọi | Judge đọc transcript; pass^k cho độ ổn định |
| RAG | Metric retrieval bằng code (Recall@k) khi có nhãn tài liệu | Judge cho faithfulness và độ đúng của câu trả lời |
| Văn bản tự do (tóm tắt, trả lời, viết lại) | Judge nhị phân theo từng tiêu chí, đã hiệu chỉnh với nhãn người | Người duyệt mẫu định kỳ |
| Giọng điệu, an toàn, tuân thủ chính sách | Judge hoặc classifier | Người duyệt các ca judge không chắc |

### Cỡ mẫu tham khảo

| Mục đích | Bao nhiêu | Nguồn |
|---|---|---|
| Bắt đầu bộ eval đầu tiên | 20–50 ca lấy từ lỗi thật | Anthropic [1] |
| Đọc trace để tìm failure mode | Tự đọc ít nhất 30 trace; tổng cộng ít nhất khoảng 100, đọc tiếp khi vẫn còn thấy lỗi mới | Hamel Husain [4] |
| Hiệu chỉnh một LLM judge | Khoảng 100–200 nhãn người cho mỗi failure mode, mỗi tập dev/test có 30–50 Pass và 30–50 Fail nếu được | Hamel Husain [4] |
| Ước lượng pass rate ±8 điểm % (p ≈ 0,8) | 100 ca | Tính từ công thức SE ở [§1.4](#14-phương-sai-chạy-bao-nhiêu-lần-cần-bao-nhiêu-mẫu) |
| Phân biệt 80% với 85% (hai nhánh độc lập, power 80%) | Khoảng 900 ca mỗi nhánh; ít hơn nhiều nếu so cặp trên cùng bộ ca | Tính từ công thức chuẩn [6] |

### Eval đầu tiên trong một ngày

1. Bật trace cho toàn bộ request: input, output, tool call, tài liệu retrieve được.
2. Lấy khoảng 30 trace, ưu tiên những ca người dùng phàn nàn. Đọc từng cái, ghi chú tự do điều gì sai.
3. Gom ghi chú thành nhóm lỗi (failure mode), đếm số lần mỗi nhóm xuất hiện.
4. Viết 20–50 ca từ các nhóm lỗi phổ biến nhất, mỗi ca có tiêu chí pass/fail rõ ràng.
5. Viết grader bằng code cho những gì kiểm được. Chỉ dùng judge cho phần còn lại.
6. Chạy mỗi ca 3–5 lần để thấy mức dao động.
7. Đưa vào CI với ngưỡng không phải 100%.

---

# Phần 1 — Nền tảng

## 1.1 Vì sao "thử vài câu" không phải là eval

**Output không tất định, kể cả khi cấu hình "tất định".** Atil và cộng sự (2024) chạy 5 LLM với cấu hình được cho là tất định (temperature 0, seed cố định) trên 8 tác vụ, mỗi tác vụ 10 lần. Độ chính xác dao động tới 15% giữa các lần chạy, khoảng cách giữa lần tốt nhất và tệ nhất có thể tới 70%, và không model nào cho chuỗi output giống hệt nhau ở mọi lần chạy [5]. Một lần chạy đúng chỉ cho biết model *có thể* làm đúng.

**Thay đổi có tác dụng không cục bộ.** Sửa prompt để xử lý một loại câu hỏi có thể làm hỏng loại khác, như tình huống đổi trả ở đầu bài. Bộ thử chỉ gồm câu hỏi bạn vừa nghĩ tới sẽ chỉ kiểm tra đúng chỗ bạn vừa sửa.

**Mắt người dễ bị thuyết phục.** Câu trả lời trôi chảy, tự tin, đúng định dạng trông như câu trả lời đúng. OpenAI gọi kiểu đánh giá này là "vibe-based" và khuyên tránh nó, cùng với việc tránh chỉ dùng các metric học thuật chung chung [2].

Eval khác "thử vài câu" ở bốn điểm: bộ ca cố định và có version, tiêu chí pass/fail viết ra trước, chạy đủ số lần để thấy dao động, và kết quả được so với một baseline. Anthropic mô tả eval như một hạ tầng sống: bắt đầu nhỏ, thêm ca từ mỗi lỗi thật, và đọc transcript thường xuyên để biết grader còn chấm đúng không [1].

Một cách làm Anthropic gợi ý là *eval-driven development*: viết eval cho khả năng bạn muốn có trước khi hệ thống làm được. Bộ eval này ban đầu có pass rate thấp, và khi model mới ra, bạn biết ngay nó giải quyết được gì trong danh sách [1].

## 1.2 Eval offline, eval online và human review

| Loại | Câu hỏi trả lời | Dữ liệu | Khi nào chạy |
|---|---|---|---|
| **Offline** | Phiên bản B có tốt hơn A trên bộ ca có nhãn không? Có làm hỏng ca nào từng đúng không? | Dataset cố định, có version | Mỗi lần đổi prompt, model, tool, cấu hình retrieval |
| **Online** | Trên traffic thật, chất lượng có đổi không? Có loại lỗi nào bộ offline chưa phủ? | Trace production, lấy mẫu | Liên tục sau khi phát hành |
| **Human review** | Người có đồng ý với grader không? Grader còn đáng tin không? | Mẫu trace kèm rubric | Khi dựng judge, rồi định kỳ |

Ba loại này bổ sung cho nhau, không thay thế nhau. Eval offline cho câu trả lời nhanh và lặp lại được trước khi phát hành, nhưng chỉ phủ những gì bạn đã nghĩ tới. Eval online thấy được phân phối thật và các lỗi chưa ai nghĩ tới, nhưng thường không có đáp án đúng, nên phải dựa vào judge hoặc tín hiệu gián tiếp. Anthropic mô tả production monitoring là thứ phát hiện drift và các lỗi thực tế không lường trước, còn A/B test dùng để xác nhận thay đổi lớn khi đã đủ traffic [1]. LangSmith, Langfuse và Braintrust đều tổ chức sản phẩm theo đúng hai nhánh này: experiment offline trên dataset, và evaluator online chạy trên trace thật với tỉ lệ lấy mẫu [10][11][31].

Human review là thứ giữ cho hai loại còn lại trung thực. Một judge chưa từng được so với nhãn người là một con số không ai biết có ý nghĩa gì ([§2.2](#22-llm-as-judge)).

Bảng so sánh offline và online cho riêng tầng retrieval nằm ở [Embeddings §3.6](../embeddings/#36-evaluation-methodology).

## 1.3 Định nghĩa "tốt": tiêu chí, rubric và metric

Trước khi chấm, phải viết ra "tốt" nghĩa là gì. Docs của Anthropic khuyên tiêu chí phải cụ thể, đo được, khả thi và liên quan đến mục tiêu sản phẩm. Ví dụ của họ: thay vì "an toàn", viết "dưới 0,1% output trong 10.000 lần chạy bị content filter đánh dấu độc hại" [12]. Phần lớn sản phẩm cần nhiều tiêu chí cùng lúc: đúng sự kiện, đủ ý, đúng giọng, đúng định dạng, không lộ thông tin nội bộ.

### Pass/fail hay thang điểm

Có hai trường phái:

- **Thang điểm (Likert 1–5).** Ví dụ trong docs của Anthropic dùng thang 1–5 cho LLM grader đánh giá giọng điệu [12]. Điểm số cho cảm giác chi tiết hơn.
- **Nhị phân pass/fail.** Hamel Husain và Shreya Shankar khuyên dùng pass/fail [36]. Lý do của họ: khoảng cách giữa hai mức kề nhau (3 và 4) mang tính chủ quan và khác nhau giữa những người chấm; phát hiện chênh lệch có ý nghĩa thống kê trên thang điểm cần cỡ mẫu lớn hơn; người chấm hay chọn mức giữa để tránh phải quyết định. Muốn theo dõi tiến bộ từng chút, họ đề xuất tách thành nhiều kiểm tra nhị phân nhỏ, ví dụ thay vì chấm độ chính xác 1–5 thì đếm "4 trên 5 sự kiện cần có đã xuất hiện".

Chưa có nghiên cứu so sánh trực tiếp hai cách trong các nguồn đã kiểm tra. Bài này khuyên bắt đầu bằng pass/fail cho từng tiêu chí riêng: dễ gán nhãn nhất quán, dễ đo độ chính xác của judge bằng TPR/TNR ([§2.2](#22-llm-as-judge)), và một tiêu chí fail chỉ thẳng vào chỗ cần sửa. Nếu cần thang điểm, mỗi mức phải có mô tả và ví dụ cụ thể.

### Rubric

Rubric tốt có các đặc điểm sau:

- **Mỗi tiêu chí chấm riêng.** "Câu trả lời có đúng chính sách đổi trả không" và "câu trả lời có lịch sự không" là hai lần chấm, không gộp thành một điểm "chất lượng".
- **Có ví dụ biên.** Ví dụ một câu trả lời *vừa đủ* pass và một câu *vừa đủ* fail giúp người và judge hiểu ranh giới.
- **Có lối thoát.** Anthropic gợi ý cho judge một lựa chọn "Unknown" khi thiếu thông tin để quyết định, thay vì buộc nó đoán [1].
- **Viết sau khi đọc dữ liệu.** Shankar và cộng sự (EvalGen, 2024) quan sát thấy hiện tượng *criteria drift*: người chấm cần tiêu chí để chấm, nhưng chính việc chấm output lại làm họ đổi tiêu chí. Tiêu chí không thể định nghĩa xong hoàn toàn trước khi nhìn output [23]. Thực tế: đọc trace, ghi chú lỗi, rồi mới viết rubric, và chấp nhận sửa rubric trong vài vòng đầu.

## 1.4 Phương sai: chạy bao nhiêu lần, cần bao nhiêu mẫu

Đây là phần hay bị bỏ qua nhất, và là lý do đội ở đầu bài không biết 84% với 86% có khác nhau không.

### Khoảng tin cậy của một pass rate

Với n ca độc lập, mỗi ca chấm 0 hoặc 1, pass rate trung bình là p̂. Sai số chuẩn và khoảng tin cậy 95% xấp xỉ [6]:

```text
SE = sqrt( p̂ (1 − p̂) / n )
95% CI ≈ p̂ ± 1,96 × SE
```

| Pass rate | n = 50 | n = 100 | n = 200 | n = 400 |
|---|---|---|---|---|
| 0,80 | ±11,1 điểm % | ±7,8 | ±5,5 | ±3,9 |
| 0,95 | ±6,0 | ±4,3 | ±3,0 | ±2,1 |

Với 200 ca và pass rate 85%, khoảng tin cậy rộng khoảng ±5 điểm. Chênh 2 điểm giữa hai model, nếu so như hai số độc lập, nằm gọn trong nhiễu.

Khi n nhỏ hoặc pass rate gần 0 hay 1, công thức trên cho khoảng tin cậy sai (có thể vượt quá 1). Khoảng Wilson ổn định hơn. Đây là kiến thức thống kê chuẩn, không riêng gì LLM:

```python
import math

def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy 95% cho tỉ lệ pass; ổn định hơn p ± 1,96·SE khi n nhỏ hoặc p gần 0, 1."""
    if n == 0:
        return (0.0, 1.0)
    p = passes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

wilson_interval(9, 10)    # ≈ (0,60, 0,98): 9/10 không nói được nhiều
wilson_interval(0, 20)    # ≈ (0,00, 0,16): 0 lỗi trên 20 ca vẫn có thể là 16% lỗi
```

### Ca không độc lập: hiệu chỉnh theo cụm

Công thức trên giả định các ca độc lập. Khi nhiều câu hỏi cùng hỏi về một tài liệu, hoặc nhiều lượt cùng thuộc một hội thoại, các ca trong cùng nhóm thường cùng đúng hoặc cùng sai. Sai số chuẩn thật lớn hơn công thức ngây thơ. Miller (2024) đưa ra công thức sai số chuẩn theo cụm, và trong ví dụ của paper, hiệu chỉnh theo cụm làm sai số chuẩn lớn hơn hơn ba lần [6]. Nếu bộ eval của bạn có cấu trúc nhóm, resample theo nhóm (code ở dưới).

### So sánh hai phiên bản: so theo cặp

Khi so model A với model B trên *cùng* bộ ca, đừng so hai pass rate như hai số độc lập. Tính hiệu số trên từng ca, rồi tính khoảng tin cậy của trung bình hiệu số. Miller khuyên dùng cách so cặp ở mọi chỗ có thể: hai phiên bản thường đúng và sai trên cùng các ca khó, nên phần nhiễu chung triệt tiêu và khoảng tin cậy hẹp hơn đáng kể mà không tốn thêm lần gọi nào [6].

Bootstrap theo cặp là cách tính đơn giản, không cần giả định phân phối, và xử lý được cả cấu trúc cụm:

```python
from collections import defaultdict
import numpy as np

def paired_diff_ci(cand: dict, base: dict, clusters: dict | None = None,
                   n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """cand, base: {case_id: điểm 0..1, trung bình các trial} của hai phiên bản.
    clusters: {case_id: nhóm} khi nhiều ca chung một tài liệu hoặc hội thoại.
    Trả về (hiệu số trung bình, cận dưới, cận trên) của khoảng tin cậy 95%."""
    ids = sorted(cand.keys() & base.keys())
    groups = defaultdict(list)
    for i in ids:
        groups[clusters[i] if clusters else i].append(cand[i] - base[i])
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))   # resample theo nhóm
    boot = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(sums.sum() / sizes.sum()), float(lo), float(hi)
```

Đọc kết quả: nếu cả khoảng tin cậy nằm dưới 0, phiên bản mới tệ hơn với độ tin cậy 95%. Nếu khoảng tin cậy chứa 0, dữ liệu chưa đủ để kết luận theo hướng nào. "Chưa đủ bằng chứng là khác nhau" không có nghĩa là "bằng nhau".

### Cần bao nhiêu ca

Để phát hiện chênh lệch δ giữa hai tỉ lệ p₁ và p₂ với α = 0,05 (hai phía) và power 80%, khi hai nhánh độc lập, công thức chuẩn cho xấp xỉ:

```text
n mỗi nhánh ≈ 7,85 × (p₁(1 − p₁) + p₂(1 − p₂)) / δ²
```

| So sánh | n mỗi nhánh (độc lập) |
|---|---|
| 80% với 90% | khoảng 200 |
| 80% với 85% | khoảng 900 |
| 90% với 92% | khoảng 3.200 |

So cặp trên cùng bộ ca thường cần ít hơn nhiều, tuỳ mức tương quan giữa hai phiên bản. Miller đưa công thức power cho trường hợp tổng quát có tính cả số lần lấy mẫu mỗi câu; ví dụ trong paper cần khoảng 969 câu hỏi để phát hiện chênh lệch 3 điểm phần trăm [6]. Hệ quả thực tế: bộ eval vài trăm ca đủ để bắt regression lớn (5–10 điểm), không đủ để phân biệt hai model chênh nhau 1–2 điểm. Nếu quyết định của bạn phụ thuộc vào chênh lệch nhỏ như vậy, hoặc phải tăng cỡ mẫu, hoặc phải chấp nhận rằng hai phương án tương đương về chất lượng và chọn theo tiêu chí khác (giá, độ trễ).

### Chạy mỗi ca bao nhiêu lần

Chạy mỗi ca K lần rồi lấy trung bình làm giảm phần phương sai *trong từng câu* (cùng câu hỏi, khác lần chạy), nhưng không giảm phần phương sai *giữa các câu* (câu khó, câu dễ). Miller chỉ ra rằng khi phương sai giữa các câu chiếm phần lớn, tăng K thêm có ít tác dụng; lúc đó thêm câu hỏi mới hiệu quả hơn [6]. Thực tế, 3–5 lần mỗi ca là đủ để thấy ca nào chập chờn; muốn đo độ ổn định cho agent thì dùng pass^k ([§2.4](#24-eval-cho-tool-use-và-agent)).

**Đừng hạ temperature để "giảm nhiễu" trong eval.** Miller cảnh báo cách này có thể chuyển phương sai sang chỗ khác hoặc đưa vào bias, vì bạn không còn đo hệ thống ở cấu hình thật [6]. Và như Atil và cộng sự cho thấy, temperature 0 cũng không bảo đảm output lặp lại [5]. Eval ở đúng cấu hình production.

## 1.5 Benchmark công khai khác eval của bạn thế nào

Benchmark công khai (MMLU, GPQA, SWE-bench, τ-bench...) trả lời câu hỏi "model này làm được gì nói chung". Eval của bạn trả lời "hệ thống của tôi có làm đúng việc cho người dùng của tôi không". Hai câu hỏi này khác nhau ở ba chỗ:

**Contamination.** Benchmark công khai có thể đã lọt vào dữ liệu huấn luyện. Ngày 23/02/2026, OpenAI công bố ngừng báo cáo điểm SWE-bench Verified [15]. Họ audit 138 bài toán mà o3 không giải ổn định qua 64 lần chạy và thấy 59,4% số bài có vấn đề trong thiết kế test hoặc mô tả bài, khiến lời giải đúng vẫn bị chấm sai. Họ cũng thấy mọi model biên giới được thử đều tái tạo được lời giải gốc hoặc chi tiết nguyên văn của đề ở một số bài, tức đã thấy chúng khi huấn luyện. Kết luận của họ: tiến bộ trên SWE-bench Verified ngày càng phản ánh mức độ model đã tiếp xúc với benchmark hơn là năng lực thật, và họ khuyên dùng SWE-bench Pro thay thế [15]. Nghiên cứu về phát hiện leakage (Xu và cộng sự, 2024) đề xuất dùng perplexity và độ chính xác N-gram để đo mức độ model đã thấy benchmark [14].

**Bão hoà.** Anthropic định nghĩa eval bão hoà là khi agent đã giải được mọi task giải được, không còn chỗ để thấy cải thiện [1]. Bộ eval của bạn cũng bão hoà: pass 100% không có nghĩa là hệ thống hoàn hảo, mà là bộ eval không còn đủ khó để phân biệt phiên bản tốt với phiên bản rất tốt. Hamel Husain và Shreya Shankar viết rằng pass 100% thường có nghĩa là bạn chưa thử thách hệ thống đủ mạnh, và pass rate 70% có thể là dấu hiệu của một bộ eval có ý nghĩa hơn [4].

**Khác phân phối.** Benchmark không có câu hỏi về chính sách đổi trả của bạn, không có tài liệu nội bộ của bạn, không viết bằng tiếng Việt không dấu kiểu chat. Model đứng đầu bảng xếp hạng chung vẫn có thể thua trên dữ liệu của bạn. [Embeddings §3.6](../embeddings/#36-evaluation-methodology) có ví dụ với retrieval: model đứng đầu MTEB có thể giảm mạnh trên bộ cần suy luận như BRIGHT.

Benchmark công khai vẫn có ích để lọc danh sách model ứng viên. Quyết định cuối cùng thì dựa trên eval của bạn.

---

# Phần 2 — Phương pháp và công cụ

## 2.1 Grader dựa trên code: exact match, regex, schema, unit test

Grader bằng code nhanh, rẻ, tái lập được, và không có thiên lệch kiểu LLM. Nhược điểm là giòn: một câu trả lời đúng nhưng diễn đạt khác có thể bị chấm sai [1]. Quy tắc là dùng code cho mọi thứ kiểm được bằng code, và chỉ gọi judge cho phần còn lại. Docs của Anthropic còn khuyên ưu tiên tự động hoá: nhiều câu hỏi với grader tự động kém chính xác hơn một chút vẫn tốt hơn ít câu hỏi chấm tay chất lượng cao [12].

Những gì code chấm được:

- **Exact match sau chuẩn hoá.** Docs của Anthropic định nghĩa exact match là khớp với đáp án chuẩn, thường sau khi chuẩn hoá khoảng trắng và chữ hoa/thường [12]. Với tiếng Việt, thêm chuẩn hoá Unicode (NFC), vì cùng một chữ "ệ" có thể lưu ở dạng dựng sẵn hoặc tổ hợp.
- **Regex và chứa chuỗi.** Câu trả lời có nhắc mã đơn hàng không? Có chứa URL nội bộ không? Có nhắc tên đối thủ không?
- **Schema.** Output có parse được không, có đúng JSON Schema không, các trường bắt buộc có đủ không.
- **Unit test.** Code model sinh ra có chạy được và qua test không.
- **Trạng thái môi trường.** Với agent, kiểm tra database, file, hệ thống sau khi agent chạy, không đọc lời agent tự khai. Anthropic ghi nhận trường hợp agent báo đã làm xong nhưng trạng thái chưa hề thay đổi [1].
- **Các ràng buộc vận hành.** Số bước, số lần gọi tool, độ dài output, ngôn ngữ của câu trả lời, có gọi tool bị cấm không.

```python
import json, re, unicodedata
from urllib.parse import urlparse
from jsonschema import Draft202012Validator

# Chính sách của sản phẩm ví dụ; thay bằng danh sách và giới hạn của bạn.
INTERNAL_HOST_SUFFIXES = (".corp.example.com", ".internal.example.com")
URL_RE = re.compile(r"https?://[^\s<>()\[\]\"'`]+", re.IGNORECASE)
TRAILING_PUNCT = ".,;:!?…"            # dấu câu đứng sau URL trong văn bản, không thuộc URL

def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).lower().split())

def urls_in(text: str) -> list[str]:
    return [u.rstrip(TRAILING_PUNCT) for u in URL_RE.findall(text)]

def is_internal(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")   # "corp.example.com." là FQDN
    return any(host == s.lstrip(".") or host.endswith(s) for s in INTERNAL_HOST_SUFFIXES)

def grade_refund_answer(output: str, case: dict) -> dict[str, bool]:
    """Mỗi tiêu chí một kết quả pass/fail riêng; không gộp thành một điểm."""
    return {
        "mentions_order_id": case["order_id"] in output,
        "mentions_invoice_check": "hoá đơn" in norm(output) or "hóa đơn" in norm(output),
        "no_internal_url": not any(is_internal(u) for u in urls_in(output)),
        "within_length": len(output) <= case["max_chars"],   # giới hạn lấy từ đặc tả của ca
    }

def grade_extraction(raw: str, schema: dict, expected: dict) -> dict[str, bool]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"valid_json": False}
    if not isinstance(data, dict):                       # JSON hợp lệ nhưng là mảng, chuỗi, null...
        return {"valid_json": True, "schema_ok": False,
                **{f"field:{f}": False for f in expected}}
    result = {"valid_json": True,
              "schema_ok": not any(Draft202012Validator(schema).iter_errors(data))}
    for field, want in expected.items():                 # so từng field với nhãn
        result[f"field:{field}"] = data.get(field) == want
    return result
```

Danh sách host nội bộ và `max_chars` là chính sách của sản phẩm ví dụ, không phải giá trị chung. Kiểm tra URL bằng cách parse hostname rồi so với danh sách rõ ràng, thay vì regex trên chuỗi: regex dạng `internal\.` vừa bắt nhầm `notinternal.example.com`, vừa bỏ sót địa chỉ nội bộ không chứa chữ "internal". Bộ tách URL không phân biệt hoa thường ở scheme và bỏ dấu câu dính sau URL (`https://wiki.corp.example.com,`). Nó chỉ bắt URL có `http://` hoặc `https://`; nếu chính sách cấm cả hostname trần như `wiki.corp.example.com/trang`, cần thêm một bước dò hostname.

`"hoá đơn"` và `"hóa đơn"` là hai cách đặt dấu đều gặp trong thực tế. Grader bằng code cho tiếng Việt phải tính tới những biến thể như vậy, nếu không nó sẽ chấm sai câu trả lời đúng một cách có hệ thống.

## 2.2 LLM-as-judge

Khi tiêu chí cần phán đoán (câu trả lời có đúng ý không, có lịch sự không, có bám tài liệu không), một LLM có thể làm giám khảo. Cách này mở rộng được quy mô chấm, nhưng judge là một hệ thống LLM khác với lỗi riêng của nó. Anthropic viết rằng judge phải được hiệu chỉnh sát với chuyên gia là người [1]. Đây là điều kiện để dùng judge, không phải bước tuỳ chọn.

### Pointwise và pairwise

- **Pointwise:** judge chấm một output theo rubric. Dùng khi cần một con số tuyệt đối (pass rate) và khi theo dõi theo thời gian.
- **Pairwise:** judge chọn output tốt hơn giữa hai phương án. OpenAI nhận xét LLM phân biệt giữa các lựa chọn tốt hơn là đánh giá mở, và so sánh cặp hoặc phán quyết pass/fail đáng tin hơn chấm điểm [2]. Dùng khi chọn giữa hai prompt hoặc hai model. Nhược điểm: không cho biết cả hai có đủ tốt không, và nhạy với thứ tự (xem dưới).

### Các thiên lệch đã được đo

| Thiên lệch | Bằng chứng | Nguồn |
|---|---|---|
| **Vị trí** | Zheng và cộng sự (MT-Bench): khi đổi thứ tự hai câu trả lời, GPT-4 cho kết quả nhất quán ở 65% trường hợp và nghiêng về câu đứng đầu ở 30%; GPT-3.5 nhất quán 46,2% và nghiêng về câu đầu 50%. Wang và cộng sự: chỉ bằng cách đổi thứ tự, Vicuna-13B "thắng" ChatGPT ở 66/80 câu khi ChatGPT làm judge | [18][19] |
| **Độ dài** | Zheng và cộng sự dùng phép thử "danh sách lặp lại" (chèn nội dung lặp để câu trả lời dài hơn mà không thêm thông tin): Claude-v1 và GPT-3.5 bị lừa ở 91,3% trường hợp, GPT-4 ở 8,7%. Tác giả kết luận mọi LLM có thể có thiên lệch này, dù GPT-4 chống chịu tốt hơn nhiều | [18] |
| **Tự ưu ái** | Zheng và cộng sự: GPT-4 cho chính nó tỉ lệ thắng cao hơn khoảng 10%, Claude-v1 khoảng 25%, nhưng tác giả nói dữ liệu còn hạn chế. Panickssery và cộng sự (2024): khả năng model nhận ra output của chính mình tương quan tuyến tính với mức độ tự ưu ái | [18][20] |
| **Nhiều loại khác** | Ye và cộng sự (2024, CALM) liệt kê 12 loại thiên lệch và đo tự động; thiên lệch vẫn còn ở một số tác vụ cụ thể kể cả với model mạnh | [21] |

Các con số trên đo trên model của 2023–2024. Model 2026 có thể ít thiên lệch hơn, nhưng các nhóm thiên lệch vẫn là danh sách cần kiểm tra trên judge của bạn. Một chi tiết hay bị trích sai: abstract của Zheng và cộng sự ghi judge GPT-4 đồng thuận với người "trên 80%". Con số này là khi bỏ các ca hoà (85%); tính cả ca hoà thì là 66%, trong khi người với người đồng thuận 81% và 63% tương ứng [18].

### Giảm thiên lệch

- **Chấm hai lần đảo thứ tự** với pairwise. Nếu hai lần cho kết quả ngược nhau, coi là hoà [18][19].
- **Cho judge suy luận trước khi kết luận.** OpenAI khuyên yêu cầu judge giải thích trước khi chấm [2]. Trong thử nghiệm nhỏ của Zheng và cộng sự với câu hỏi toán (10 câu, 20 lần chấm), tỉ lệ judge chấm sai giảm từ 70% xuống 30% khi dùng chain-of-thought, và xuống 15% khi đưa đáp án tham chiếu cho judge [18].
- **Đưa đáp án tham chiếu** khi có. Judge so với đáp án mẫu chính xác hơn judge tự giải.
- **Kiểm soát độ dài:** thêm ca có câu trả lời ngắn mà đúng, và câu trả lời dài mà sai, vào bộ hiệu chỉnh judge [2].
- **Dùng judge khác họ với model được chấm.** Code tham chiếu của Anthropic ghi chú rằng nên chấm bằng một model khác với model sinh output [12], và kết quả của Panickssery và cộng sự cho thấy lý do [20].
- **Dùng model mạnh làm judge trước**, rồi mới thử judge rẻ hơn khi đã có số đo TPR/TNR để so [2][4].

Ví dụ judge pairwise có đảo thứ tự. Hàm `call_judge` là lời gọi tới bất kỳ provider nào, trả về `"A"`, `"B"` hoặc `"TIE"` bằng structured output:

```python
JUDGE_PROMPT = """Bạn chấm hai câu trả lời cho cùng một câu hỏi của khách hàng.
Tiêu chí duy nhất: câu trả lời nào áp dụng đúng chính sách đổi trả trong tài liệu.
Không xét độ dài hay văn phong. Suy luận ngắn gọn trước, rồi kết luận A, B hoặc TIE.

<policy>{policy}</policy>
<question>{question}</question>
<answer_A>{a}</answer_A>
<answer_B>{b}</answer_B>"""

def pairwise(call_judge, policy: str, question: str, x: str, y: str) -> str:
    """Trả về 'x', 'y' hoặc 'tie'. Chấm hai lần với thứ tự đảo ngược."""
    first = call_judge(JUDGE_PROMPT.format(policy=policy, question=question, a=x, b=y))
    second = call_judge(JUDGE_PROMPT.format(policy=policy, question=question, a=y, b=x))
    if first == "A" and second == "B":
        return "x"
    if first == "B" and second == "A":
        return "y"
    return "tie"                     # không nhất quán khi đổi thứ tự: coi là hoà
```

### Hiệu chỉnh judge với nhãn người

Đây là bước quan trọng nhất và hay bị bỏ qua nhất. Quy trình theo Hamel Husain [4]:

1. **Coi judge là một bộ phân loại nhị phân** cho một failure mode cụ thể: "câu trả lời có áp dụng sai chính sách đổi trả không". Mỗi failure mode một judge, không phải một judge chấm "chất lượng chung".
2. **Lấy nhãn từ chuyên gia domain**, người thực sự biết câu trả lời đúng là gì. Khoảng 100–200 nhãn cho mỗi failure mode, cố gắng có đủ cả ca Pass lẫn ca Fail.
3. **Chia ba tập:** train (10–20%, được dùng làm ví dụ few-shot trong prompt của judge), dev (40–45%, dùng để sửa prompt của judge), test (40–45%, chỉ dùng để đo lần cuối). Ca trong dev và test không bao giờ xuất hiện trong prompt của judge.
4. **Đo TPR và TNR, không đo agreement.** Agreement (tỉ lệ judge và người cùng kết luận) đánh lừa khi lỗi hiếm: nếu chỉ 5% câu trả lời có lỗi, judge luôn trả "Pass" vẫn đạt agreement 95% mà không bắt được lỗi nào [4]. Cần hai con số riêng: judge bắt được bao nhiêu phần trăm ca hỏng, và cho qua được bao nhiêu phần trăm ca tốt. Bài này lấy Pass làm lớp dương như thư viện `judgy` [32]: TPR là tỉ lệ ca tốt được judge cho Pass, TNR là tỉ lệ ca hỏng bị judge đánh Fail. FAQ của Hamel lấy "có lỗi" làm lớp dương, nên ở đó TPR và TNR đổi chỗ cho nhau [4]. Kiểm tra quy ước trước khi dùng công thức hiệu chỉnh dưới đây; nhầm hai con số sẽ cho kết quả sai mỗi khi chúng khác nhau.
5. **Lặp trên dev, đo một lần trên test.** Sửa prompt của judge theo các ca nó chấm sai trong dev. Khi hài lòng, đo trên test và ghi lại.

Khi đã biết TPR và TNR, có thể hiệu chỉnh pass rate mà judge báo về pass rate thật. Công thức Rogan–Gladen, vốn dùng trong dịch tễ học cho xét nghiệm không hoàn hảo, được thư viện `judgy` của nhóm khoá học AI Evals (Hamel Husain, Shreya Shankar) áp dụng cho LLM judge [32]:

```python
def judge_rates(human: list[int], judge: list[int]) -> tuple[float, float]:
    """1 = Pass, 0 = Fail, trên tập test. Trả về (TPR, TNR)."""
    if len(human) != len(judge):
        raise ValueError("số nhãn người và nhãn judge khác nhau")
    if not set(human) | set(judge) <= {0, 1}:
        raise ValueError("nhãn chỉ được là 0 hoặc 1")
    pos = sum(human)
    if pos == 0 or pos == len(human):
        raise ValueError("tập test cần có cả ca Pass và ca Fail")
    tp = sum(h == 1 and j == 1 for h, j in zip(human, judge))
    tn = sum(h == 0 and j == 0 for h, j in zip(human, judge))
    return tp / pos, tn / (len(human) - pos)

def corrected_pass_rate(p_obs: float, tpr: float, tnr: float) -> float:
    """Ước lượng pass rate thật từ pass rate judge báo (p_obs)."""
    if not all(0.0 <= v <= 1.0 for v in (p_obs, tpr, tnr)):   # cũng loại NaN
        raise ValueError("p_obs, tpr, tnr phải là xác suất trong [0, 1]")
    youden = tpr + tnr - 1
    if youden <= 0:
        raise ValueError("judge không tốt hơn đoán ngẫu nhiên")
    return min(1.0, max(0.0, (p_obs + tnr - 1) / youden))

corrected_pass_rate(0.85, tpr=0.92, tnr=0.90)   # ≈ 0,915
```

Ví dụ trên: judge báo 85% pass, nhưng judge này nghiêm hơn chuyên gia (TNR 90% nghĩa là 10% ca hỏng vẫn được cho Pass, còn TPR 92% nghĩa là 8% ca tốt bị đánh Fail), nên pass rate thật ước lượng khoảng 91,5%. Điều này quan trọng khi đổi judge: nếu đổi sang judge dễ tính hơn mà không hiệu chỉnh, "cải thiện" bạn thấy đến từ judge, không phải từ hệ thống. TPR và TNR chính là ước lượng từ vài chục ca, nên con số đã hiệu chỉnh cũng có khoảng tin cậy; `judgy` tính khoảng này bằng bootstrap [32].

### Ghi lại cấu hình judge

Judge là một phần của bộ đo. Mỗi lần chạy eval, ghi lại: model judge kèm phiên bản cụ thể, phiên bản prompt và rubric, TPR/TNR đo được với đúng tổ hợp đó. Đổi bất kỳ thứ nào thì đo lại trên tập test trước khi so số với các lần chạy cũ.

## 2.3 Eval cho RAG: retrieval và generation

RAG có ít nhất hai chỗ có thể sai: lấy sai tài liệu, hoặc lấy đúng tài liệu mà trả lời sai. Đo chung một con số "câu trả lời đúng" không cho biết lỗi nằm ở đâu. Đo theo tầng:

| Tầng | Metric | Cần nhãn gì |
|---|---|---|
| Retrieval | Recall@k, MRR, nDCG@k | Tài liệu/chunk nào liên quan với mỗi câu hỏi |
| Rerank và lắp context | Context precision, context recall | Tài liệu liên quan, hoặc đáp án tham chiếu |
| Generation | Faithfulness (bám context), độ đúng của câu trả lời | Đáp án tham chiếu hoặc judge |

Định nghĩa Recall@k, nDCG và cách xây bộ nhãn retrieval có ở [Embeddings §3.6](../embeddings/#36-evaluation-methodology). Phần dưới tập trung vào các metric dựa trên LLM của Ragas [24]:

- **Faithfulness:** một câu trả lời được coi là faithful nếu mọi nhận định trong nó đều được context truy xuất hỗ trợ. Ragas tách câu trả lời thành các nhận định (claim), kiểm tra từng claim có suy ra được từ context không. Điểm = số claim được hỗ trợ / tổng số claim [25].
- **Context precision@K:** trung bình có trọng số của precision tại các vị trí có chunk liên quan, đo xem chunk liên quan có được xếp lên đầu không. Ragas có bản dùng LLM để quyết định chunk nào liên quan, bản không dùng LLM (so chuỗi), và bản so theo ID tài liệu [26].
- **Context recall:** tỉ lệ thông tin cần thiết đã được truy xuất. Bản dùng LLM tách *đáp án tham chiếu* thành các claim và đếm bao nhiêu claim quy được về context truy xuất. Bản không dùng LLM và bản theo ID so trực tiếp với danh sách context tham chiếu [33].

Ba điểm cần nhớ:

- **Faithful không có nghĩa là đúng.** Câu trả lời bám sát một tài liệu đã lỗi thời vẫn faithful 100%. Cần cả đo độ đúng so với đáp án tham chiếu.
- **Metric bằng model vẫn là judge.** Faithfulness của Ragas dùng LLM để tách và kiểm claim. DeepEval cũng vậy ở chế độ mặc định `eval_mode="llm"`; ở chế độ `hybrid`, LLM tách claim còn Jev (model phân loại riêng của DeepEval) quyết định từng claim, còn ở `system_one` thì Jev chấm toàn bộ và không gọi LLM [37]. Dù chế độ nào, người chấm vẫn là một model. Chúng cũng cần được kiểm tra với nhãn người trên dữ liệu của bạn, đặc biệt với tiếng Việt, nơi bước tách claim có thể kém hơn tiếng Anh.
- **Ngưỡng là lựa chọn của bạn.** Trang best practices của OpenAI đưa ví dụ ngưỡng context recall ≥ 0,85 và context precision > 0,7 [2]. Đây là ví dụ, không phải chuẩn.

Các kỹ thuật RAG nâng cao và eval riêng cho chúng ở [RAG nâng cao §3.5](../advanced-rag/#35-eval-cho-rag-nâng-cao).

## 2.4 Eval cho tool use và agent

[Tool Use §3.9](../tool-use/#39-eval-cho-tool-use) đã trình bày eval theo tầng (chọn tool, tham số, quỹ đạo, kết quả) và benchmark công khai. Phần này bổ sung những điểm về phương pháp.

**pass@k và pass^k đo hai thứ khác nhau.** Anthropic định nghĩa pass@k là xác suất có ít nhất một lần thành công trong k lần thử, còn pass^k là xác suất *cả k lần* đều thành công [1]. τ-bench ước lượng pass^k bằng `E_task[ C(c, k) / C(n, k) ]`, với n lần chạy mỗi task và c lần thành công [27]. Với một agent thành công 90% mỗi lần một cách độc lập, pass@8 gần 100% nhưng pass^8 = 0,9⁸ ≈ 43%. Agent viết code cho lập trình viên tự review thì pass@k là thước đo hợp lý. Agent phục vụ khách hàng thì pass^k mới phản ánh trải nghiệm: mỗi khách chỉ có một lần.

**Chấm trạng thái, không chấm lời tự khai.** Grader đọc database, hệ thống file, hoặc trạng thái mock sau khi agent chạy [1].

**Chấm kết quả, không bắt buộc một đường đi.** Anthropic nhận thấy kiểm tra agent có gọi đúng chuỗi tool theo đúng thứ tự hay không làm test quá giòn, vì agent thường tìm ra cách hợp lệ mà người viết eval không nghĩ tới; tốt hơn là chấm thứ agent tạo ra. Với task nhiều phần, cho điểm từng phần: agent xác định đúng vấn đề và xác minh đúng khách nhưng không hoàn tiền vẫn tốt hơn agent hỏng ngay từ đầu [1].

**Môi trường cô lập cho từng lần chạy.** Trạng thái dùng chung giữa các lần chạy (file còn sót, cache, tài nguyên cạn) làm các lần chạy không còn độc lập, và có thể thổi phồng kết quả: trong một số eval nội bộ, Anthropic thấy Claude có lợi thế không công bằng nhờ đọc git history của các lần chạy trước [1]. Mỗi trial bắt đầu từ trạng thái sạch.

**Đọc transcript, kể cả ca pass.** Anthropic kể trường hợp Claude Opus 4.5 "fail" một bài đặt vé máy bay của τ²-bench vì tìm ra một kẽ hở trong chính sách, và thực ra đưa ra giải pháp tốt hơn cho người dùng [1]. Ngược lại, có ca pass vì grader quá lỏng. Chỉ đọc transcript mới thấy được cả hai.

**Task có pass rate 0% thường là task hỏng.** Với model biên giới, 0% qua nhiều lần chạy thường là dấu hiệu task hoặc grader có lỗi, không phải agent kém. Mỗi task nên có lời giải tham chiếu chứng minh nó giải được, và hai chuyên gia độc lập phải đi tới cùng kết luận pass/fail [1]. Ví dụ Anthropic nêu: Opus 4.5 ban đầu chỉ đạt 42% trên CORE-Bench; sau khi sửa grader quá cứng (chấm sai "96.12" vì đáp án là "96.124991…"), đề mơ hồ và task ngẫu nhiên không tái tạo được, cùng với scaffold ít ràng buộc hơn, điểm lên 95% [1].

**Cân bằng hai chiều.** Có ca *nên* gọi tool và ca *không nên* gọi. Bộ eval chỉ có ca nên tìm kiếm sẽ tối ưu ra một agent tìm kiếm mọi lúc [1].

Inspect hỗ trợ trực tiếp cả hai thước đo: chạy mỗi mẫu nhiều *epoch* và gộp bằng reducer `pass_at_{k}` (ít nhất một lần đúng) hoặc `pass_k_{k}` (cả k lần đúng, theo công thức của τ-bench) [30].

## 2.5 Eval cho structured output và extraction

Trích xuất có cấu trúc là chỗ grader bằng code làm tốt nhất, vì thường có đáp án đúng:

1. **Tầng cú pháp:** parse được JSON không, đúng schema không. Lỗi ở tầng này là lỗi của cấu hình (thiếu structured output, schema không được provider hỗ trợ) hơn là của model.
2. **Tầng giá trị:** so từng field với nhãn sau chuẩn hoá (khoảng trắng, định dạng ngày, số tiền, Unicode). Báo cáo độ chính xác *theo field*, không chỉ theo cả bản ghi: một hoá đơn sai 1/12 field là một bản ghi sai, nhưng con số theo field cho biết field nào cần sửa.
3. **Tầng ngữ nghĩa:** field văn bản tự do (mô tả sản phẩm, lý do khiếu nại) dùng judge hoặc so khớp mờ.

Ba loại lỗi cần đếm riêng, vì hậu quả khác nhau:

- **Đọc sai:** trả về giá trị khác với giá trị thật.
- **Bỏ sót:** trả `null` khi tài liệu có giá trị.
- **Bịa:** trả về giá trị khi tài liệu không có. Đây là lỗi nguy hiểm nhất cho pipeline tự động. Phải có ca cố ý thiếu field trong bộ eval để đo được nó.

Thiết kế schema và validation nhiều lớp ở [Structured Output §3.3](../structured-output/#33-validation-nhiều-lớp-và-retry); eval cho trích xuất từ ảnh và PDF ở [Multimodal §3.6](../multimodal/#36-eval-cho-tác-vụ-vision).

## 2.6 Công cụ: OpenAI Evals, promptfoo, Braintrust, LangSmith, Langfuse, Inspect

| Công cụ | Loại | Điểm đáng chú ý (đã kiểm tra trên docs 11/2026) |
|---|---|---|
| **OpenAI Evals platform** | Hosted | **Đang bị khai tử:** chuyển sang chỉ đọc từ 31/10/2026 và ngừng hoạt động ngày 30/11/2026 [3]. Không dùng cho dự án mới |
| **promptfoo** | CLI, chạy local và trong CI | Cấu hình YAML; exit code `100` khi có ca fail; `PROMPTFOO_PASS_RATE_THRESHOLD` đặt ngưỡng pass rate (mặc định 100%); `--repeat <n>` chạy lặp; cache kết quả gọi model bật mặc định, tắt bằng `--no-cache` [29] |
| **Inspect AI** (UK AI Security Institute) | Framework Python | Task = dataset + solver + scorer. Scorer có sẵn: `match`, `includes`, `pattern`, `exact`, `f1`, `model_graded_qa`, `model_graded_fact`... Có `epochs` với reducer `mean`, `mode`, `at_least_{n}`, `pass_at_{k}`, `pass_k_{k}`; báo cáo `stderr` và hỗ trợ sai số chuẩn theo cụm [30] |
| **Braintrust** | Hosted | Ba phần: data, task, evaluator. Playground để thử nhanh, experiment là bản ghi bất biến để so sánh, chạy trong CI, online scoring trên trace production [31] |
| **LangSmith** (LangChain) | Hosted | Dataset từ trace, tự viết hoặc sinh tổng hợp; evaluator bằng code, LLM-as-judge, pairwise, người; offline và online với sampling rate [10] |
| **Langfuse** | Mã nguồn mở, self-host hoặc cloud | Experiment offline trên dataset; evaluator online trên trace; LLM-as-judge, evaluator bằng code, annotation của người [11] |
| **DeepEval** (Confident AI) | Framework Python, Apache 2.0 | Hơn 50 metric dựng sẵn, phần lớn dùng LLM-as-judge (G-Eval, DAG); tích hợp pytest qua `deepeval test run`. **Ngưỡng mặc định của mọi metric là 0,5** [34] |
| **Ragas** | Thư viện Python | Metric cho RAG (faithfulness, context precision, context recall) và agent [24][25] |

Chọn theo chỗ dữ liệu đang nằm và cách đội làm việc:

- **Cần chạy trong CI ngay, ít phụ thuộc:** promptfoo, hoặc pytest với grader tự viết. Một file Python đọc dataset, gọi hệ thống, chấm, ghi kết quả JSON là đủ cho phần lớn đội nhỏ.
- **Eval agent có môi trường, nhiều lần chạy, cần thống kê đúng:** Inspect, hoặc harness tự viết theo mô hình task/trial/grader của Anthropic [1].
- **Đã có tracing ở một nền tảng:** dùng phần eval của chính nền tảng đó (Langfuse, LangSmith, Braintrust) để dataset đi thẳng từ trace production. Chọn nền tảng thì xem [LLM Observability](../llm-observability/).

Hai lưu ý chung. Thứ nhất, công cụ không làm thay việc hiệu chỉnh judge: metric dựng sẵn vẫn là judge chưa hiệu chỉnh với dữ liệu của bạn. Thứ hai, đọc kỹ giá trị mặc định: ngưỡng 0,5 của DeepEval nghĩa là một câu trả lời đạt 0,51 điểm faithfulness được tính là pass, và promptfoo mặc định đòi pass 100% nên CI sẽ đỏ vì nhiễu ([§3.4](#34-đưa-eval-vào-ci)).

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Xây golden dataset từ đâu

Golden dataset là bộ ca có nhãn, có version, dùng làm thước đo chung cho mọi thay đổi. Nguồn theo thứ tự ưu tiên:

1. **Trace production đã ẩn danh.** Phản ánh đúng phân phối câu hỏi thật, cách người dùng thật viết (không dấu, viết tắt, sai chính tả, hỏi nhiều ý một lúc). OpenAI khuyên ghi log ngay từ khi phát triển và khai thác log thành ca eval [2]. Quy trình đưa trace về dataset ở [LLM Observability §3.6](../llm-observability/#36-từ-log-production-về-eval-dataset).
2. **Lỗi đã biết.** Ticket hỗ trợ, khiếu nại, bug report. Mỗi sự cố thêm ít nhất một ca, để lỗi đã sửa không quay lại [1].
3. **Ca biên và ca đối kháng.** OpenAI khuyên bộ eval có cả ca điển hình, ca biên và ca đối kháng [2]: câu hỏi ngoài phạm vi, yêu cầu bị cấm, chỉ dẫn chèn trong tài liệu, câu hỏi mơ hồ cần hỏi lại.
4. **Dữ liệu tổng hợp (synthetic).** Dùng khi chưa có traffic, hoặc khi dữ liệu thật quá nhạy cảm. Dữ liệu do LLM sinh thường sạch hơn, đều hơn và "điển hình" hơn câu hỏi thật, nên không thay được dữ liệu thật. Khi có traffic, thay dần ca tổng hợp bằng ca thật.

### Bắt đầu từ error analysis

Cách xây bộ eval mà Hamel Husain và Shreya Shankar mô tả [4][8]:

1. Tự đọc ít nhất 30 trace, ghi chú tự do điều gì sai ở mỗi trace. Chưa phân loại, chưa viết rubric. Khi mới bắt đầu, ghi lỗi *đầu tiên* thấy trong trace, vì lỗi ở bước trước thường kéo theo lỗi ở bước sau.
2. Gom ghi chú thành các nhóm lỗi, đếm số lần xuất hiện.
3. Đọc tiếp cho đến khi trace mới không còn lộ ra nhóm lỗi mới hay làm đổi các nhóm đã có (*theoretical saturation*). Họ khuyên xem ít nhất khoảng 100 trace.
4. Biến mỗi nhóm lỗi phổ biến thành ca eval và grader. Nhóm nào kiểm được bằng code thì dùng code; nhóm nào cần phán đoán thì dựng judge riêng và hiệu chỉnh.

Hamel và Shankar viết rằng trong các dự án họ làm, 60–80% thời gian phát triển dành cho error analysis và eval, phần lớn là để hiểu lỗi chứ không phải để dựng kiểm tra tự động [4].

### Bao nhiêu ca

- **Bắt đầu với 20–50 ca** lấy từ lỗi thật [1]. Bộ nhỏ chạy nhanh và đủ để bắt các regression lớn.
- **Mở rộng theo nhu cầu đo.** Muốn phân biệt chênh lệch 5 điểm thì cần hàng trăm ca ([§1.4](#14-phương-sai-chạy-bao-nhiêu-lần-cần-bao-nhiêu-mẫu)).
- **Phủ theo nhóm, không chỉ theo tổng số.** 500 ca mà 480 ca là câu hỏi tra cứu đơn hàng thì nhóm đổi trả vẫn chỉ có 20 ca. Báo cáo pass rate theo từng nhóm.

### Versioning

- Mỗi lần thêm, sửa hoặc xoá ca là một version mới, có ghi lý do. Không sửa nhãn cũ một cách âm thầm.
- Mỗi lần chạy eval ghi lại version của dataset, rubric, prompt của judge và model judge. Kết quả của các lần chạy khác version không so trực tiếp được.
- Tách rõ ca dùng để phát triển (dev) và ca để đo (test). Ca test không được dùng làm ví dụ few-shot trong prompt của hệ thống hay của judge; nếu không, bạn đang đo khả năng nhớ đề.

Cách version prompt đi kèm ở [Prompt Engineering §3.2](../prompt-engineering/#32-versioning-và-review-thay-đổi-prompt).

## 3.2 Gán nhãn và đồng thuận giữa người gán nhãn

Nhãn là nền móng của mọi thứ khác: judge được hiệu chỉnh theo nhãn, pass rate được tính theo nhãn. Nhãn sai thì mọi con số phía sau đều sai.

- **Người gán nhãn phải hiểu domain.** Hamel nhấn mạnh nhãn nên đến từ chuyên gia domain đáng tin, không phải người gán nhãn thuê ngoài không biết chính sách [4]. Với trợ lý pháp lý, đó là luật sư; với trợ lý đổi trả, đó là trưởng nhóm chăm sóc khách hàng.
- **Ghi lý do kèm nhãn.** Một dòng giải thích vì sao Pass hay Fail. Lý do này là nguyên liệu cho rubric, cho ví dụ few-shot của judge, và để người sau hiểu nhãn.
- **Đo đồng thuận giữa người.** Cho hai người gán nhãn độc lập trên một phần mẫu (ví dụ 50 ca). Nếu hai người chỉ đồng ý 70% thời gian, đừng kỳ vọng judge đạt hơn thế; vấn đề nằm ở rubric. Cohen's kappa đo đồng thuận đã trừ phần ngẫu nhiên: `κ = (p_o − p_e) / (1 − p_e)`, với p_o là tỉ lệ đồng ý thực tế và p_e là tỉ lệ đồng ý nếu hai người gán ngẫu nhiên theo tỉ lệ nhãn của mỗi người. Kappa hữu ích vì tỉ lệ đồng ý thô bị thổi phồng khi một nhãn chiếm đa số, cùng lý do agreement của judge đánh lừa ở [§2.2](#22-llm-as-judge).
- **Họp gỡ bất đồng.** Mỗi ca hai người bất đồng là một chỗ rubric chưa rõ. Thảo luận, cập nhật rubric, gán lại. OpenAI khuyên dùng nhiều vòng review và bỏ phiếu đồng thuận cho human eval [2].
- **Chấp nhận rubric thay đổi.** Như EvalGen cho thấy, tiêu chí sẽ thay đổi trong những vòng gán nhãn đầu [23]. Khi tiêu chí đổi, gán lại các ca cũ theo tiêu chí mới, nếu không tập test sẽ trộn hai định nghĩa khác nhau.

## 3.3 Regression test khi đổi prompt hoặc model

Mọi thay đổi có thể làm đổi hành vi đều phải chạy lại eval: prompt, model, phiên bản model, tool description, cấu hình retrieval, chat template, engine inference, thư viện SDK. Danh sách các thay đổi cần chạy lại riêng cho tool use có ở [Tool Use §3.9](../tool-use/#39-eval-cho-tool-use); các loại ca riêng cho prompt (ca theo từng quy tắc, biến thể cách hỏi) ở [Prompt Engineering §3.4](../prompt-engineering/#34-regression-test-khi-đổi-prompt).

### Hai bộ eval với hai mục đích

Anthropic phân biệt [1]:

- **Capability eval:** những việc hệ thống chưa làm tốt. Pass rate thấp là bình thường. Dùng để đo tiến bộ.
- **Regression eval:** những việc hệ thống đã làm đúng và phải tiếp tục đúng. Pass rate gần 100%. Dùng để chặn thay đổi làm hỏng thứ đang chạy.

Khi một ca capability đạt pass ổn định, chuyển nó sang bộ regression.

### So sánh đúng cách

1. **Chạy baseline và candidate trên cùng bộ ca, cùng cấu hình** (temperature, max tokens, reasoning effort, tool). Chạy lại baseline trong cùng đợt, không dùng số của tuần trước: provider có thể đã cập nhật model, và chính hạ tầng eval cũng có thể đã đổi.
2. **Chạy mỗi ca vài lần** để giảm nhiễu trong từng ca.
3. **So theo cặp** và báo cáo khoảng tin cậy của hiệu số ([§1.4](#14-phương-sai-chạy-bao-nhiêu-lần-cần-bao-nhiêu-mẫu)), không so hai con số trung bình.
4. **Báo cáo theo nhóm.** Pass rate tổng không đổi vẫn có thể che một nhóm giảm 20 điểm và nhóm khác tăng nhẹ.
5. **Đọc transcript của mọi ca đổi từ pass sang fail** [1]. Một số là regression thật, một số là grader sai, một số là ca chập chờn. Chỉ đọc mới phân biệt được.

## 3.4 Đưa eval vào CI

### Ngưỡng

Ngưỡng 100% không hợp với hệ thống không tất định: một ca chập chờn là CI đỏ, và đội sẽ học cách bấm chạy lại cho đến khi xanh. promptfoo mặc định chính là 100%; đặt `PROMPTFOO_PASS_RATE_THRESHOLD` để hạ ngưỡng [29]:

```bash
# Chấp nhận tối đa 5% ca fail; exit code 1 thay cho mặc định 100 khi không đạt
export PROMPTFOO_PASS_RATE_THRESHOLD=95
export PROMPTFOO_FAILED_TEST_EXIT_CODE=1
npx promptfoo eval --repeat 3          # mỗi ca chạy 3 lần
```

Một ngưỡng tuyệt đối vẫn có vấn đề: nếu baseline đang ở 97%, ngưỡng 95% cho phép tụt 2 điểm mà không ai biết. Cổng chặn tốt hơn so candidate với baseline theo cặp, và chỉ chặn khi có bằng chứng thống kê là tệ đi. Ví dụ dưới đây dùng `paired_diff_ci` ở [§1.4](#14-phương-sai-chạy-bao-nhiêu-lần-cần-bao-nhiêu-mẫu); kết quả mỗi lần chạy là JSONL, mỗi dòng `{"case_id", "group", "pass"}`:

```python
# eval_gate.py: chặn merge khi candidate tệ hơn baseline có ý nghĩa thống kê
import json, sys
from collections import defaultdict
from eval_stats import paired_diff_ci       # hàm ở §1.4

MARGIN = 0.02        # chấp nhận tụt tối đa 2 điểm % trước khi chặn

def load(path: str) -> tuple[dict, dict]:
    runs, groups = defaultdict(list), {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            runs[r["case_id"]].append(1.0 if r["pass"] else 0.0)
            groups[r["case_id"]] = r["group"]
    return {k: sum(v) / len(v) for k, v in runs.items()}, groups

base, groups = load(sys.argv[1])
cand, _ = load(sys.argv[2])
missing, extra = base.keys() - cand.keys(), cand.keys() - base.keys()
if missing or extra:       # lần chạy dở dang hoặc dataset lệch: không so được, không được coi là pass
    print(f"case không khớp: thiếu {sorted(missing)[:10]}, thừa {sorted(extra)[:10]}")
    sys.exit(2)
failed = False
for g in sorted(set(groups.values())) + ["__all__"]:
    ids = {i for i, gi in groups.items() if g == "__all__" or gi == g}
    b = {i: base[i] for i in ids}
    c = {i: cand[i] for i in ids}
    diff, lo, hi = paired_diff_ci(c, b)
    print(f"{g:20s} n={len(b):4d} diff={diff:+.3f} CI95=[{lo:+.3f}, {hi:+.3f}]")
    if hi < -MARGIN:       # cả khoảng tin cậy nằm dưới -2 điểm: tệ đi chắc chắn
        failed = True
sys.exit(1 if failed else 0)
```

Nếu hai file không có cùng tập case (lần chạy bị ngắt, timeout, dataset lệch phiên bản), cổng thoát với mã 2 thay vì so phần giao nhau: kết quả thiếu ca thì không chứng minh được gì. Cổng này chỉ chặn khi *chắc chắn* có tụt lớn hơn biên cho phép. Với bộ eval nhỏ, nó sẽ để lọt các regression nhỏ; đó là cái giá của cỡ mẫu nhỏ, không phải lỗi của cổng. Nếu muốn chặt hơn, chặn khi `lo < -MARGIN` (không chứng minh được là *không* tụt quá biên), nhưng khi đó bộ eval phải đủ lớn, nếu không CI sẽ đỏ thường xuyên. Biên 2 điểm và ngưỡng 95% ở trên là ví dụ; chọn theo mức rủi ro của sản phẩm.

### Cache, thời gian và lịch chạy

- **Cache lời gọi model** cho các ca không đổi để CI không tốn tiền và thời gian cho thứ đã biết kết quả. promptfoo bật cache mặc định [29]. Với `--repeat N`, mỗi lần lặp có vùng cache riêng: lần chạy đầu sinh N output khác nhau, các lần chạy sau phát lại đúng N output đó [38]. Muốn đo độ ổn định bằng các lần sinh mới, chạy với `--no-cache`.
- **Chạy khi có thay đổi liên quan.** Path filter trong workflow: chạy khi file prompt, config model, định nghĩa tool hoặc dataset đổi.
- **Hai tầng.** Bộ nhỏ (vài chục đến vài trăm ca, chạy trong vài phút) cho mỗi pull request. Bộ đầy đủ, nhiều lần chạy, chạy theo lịch (hằng đêm) hoặc trước khi phát hành.
- **Lưu kết quả có cấu trúc** (JSON/JSONL) kèm version dataset, prompt, model, judge, để so lại về sau.

### CI xanh chưa phải là đủ

Trong postmortem tháng 09/2025, Anthropic viết rằng các eval họ chạy đã không bắt được sự suy giảm mà người dùng báo cáo [35]. Eval offline chỉ phủ những gì bạn đã nghĩ tới. Cần thêm tín hiệu từ production ([§3.5](#35-eval-online-ab-test-feedback-người-dùng-sampling-log)).

## 3.5 Eval online: A/B test, feedback người dùng, sampling log

### Judge trên trace production

Chạy judge đã hiệu chỉnh trên một phần trace production để theo dõi chất lượng theo thời gian. LangSmith, Langfuse và Braintrust đều hỗ trợ evaluator online với tỉ lệ lấy mẫu để kiểm soát chi phí [10][11][31]. Lưu ý: production không có đáp án tham chiếu, nên chỉ dùng được các tiêu chí không cần nhãn (bám tài liệu, đúng định dạng, có từ chối không, có lộ thông tin nội bộ không). DeepEval gọi đây là metric "referenceless" và lưu ý rằng chỉ loại metric này dùng được trong production [34].

### Lấy mẫu

- **Mẫu ngẫu nhiên đều đặn:** cho con số không thiên lệch về chất lượng chung.
- **Mẫu có chủ đích:** trace bị người dùng đánh giá thấp, trace có lỗi tool, trace quá dài, trace có từ khoá nhạy cảm, hội thoại kết thúc bằng chuyển cho nhân viên. Dùng để tìm lỗi, không dùng để tính pass rate chung.
- **Đưa trace thú vị về dataset:** mỗi lỗi mới phát hiện trong production trở thành ca offline.

### Feedback người dùng

Nút thích/không thích là tín hiệu thô: phần lớn người dùng không bấm, người bấm thường là người rất hài lòng hoặc rất bực, và nút không nói *vì sao*. Dùng nó để chỉ chỗ cần đọc, không dùng làm metric chất lượng chính. Tín hiệu gián tiếp thường đáng tin hơn: người dùng hỏi lại cùng ý, sửa lại output, sao chép câu trả lời, chuyển cho nhân viên, bỏ dở. Cách nối feedback với trace ở [LLM Observability §3.4](../llm-observability/#34-thu-feedback-người-dùng-và-nối-với-trace).

### A/B test

Với thay đổi lớn (đổi model, đổi kiến trúc retrieval), eval offline chỉ cho biết phiên bản mới không tệ hơn trên bộ ca đã biết. A/B test trên traffic thật đo tác động lên metric sản phẩm: tỉ lệ giải quyết, tỉ lệ chuyển cho nhân viên, thời gian xử lý, CSAT. Anthropic khuyên A/B test cho thay đổi đáng kể khi đã đủ traffic [1]. Quy tắc thống kê của [§1.4](#14-phương-sai-chạy-bao-nhiêu-lần-cần-bao-nhiêu-mẫu) áp dụng như cũ: quyết định cỡ mẫu và metric chính trước khi bắt đầu, không dừng sớm khi thấy kết quả đẹp.

### Judge cũng trôi

Phân phối câu hỏi thay đổi, sản phẩm thêm tính năng, provider cập nhật model judge. Định kỳ lấy mẫu trace mới, gán nhãn người, đo lại TPR/TNR của judge. Nếu tập test của judge được dùng đi dùng lại để sửa prompt judge, nó dần thành tập dev; bổ sung ca mới gán nhãn từ production.

## 3.6 Chi phí và thời gian chạy eval

Chi phí một lần chạy eval ≈ số ca × số lần chạy mỗi ca × (chi phí hệ thống + chi phí judge). Với agent nhiều bước, chi phí hệ thống mỗi ca có thể lớn gấp chục lần một lời gọi đơn.

Cách giảm mà không làm hỏng phép đo:

- **Grader code trước, judge sau.** Ca đã fail ở grader code (JSON hỏng, gọi tool cấm) không cần judge chấm thêm.
- **Cache** output của hệ thống cho ca không đổi khi chỉ sửa grader hoặc judge; cache kết quả judge khi chỉ thêm ca mới.
- **Judge nhỏ có kiểm chứng.** Thay judge lớn bằng judge rẻ hơn chỉ khi TPR/TNR của judge rẻ trên cùng tập test vẫn đạt mức bạn chấp nhận [4].
- **Batch API** cho bộ eval lớn không cần kết quả ngay. Các provider lớn giảm giá cho batch; chi tiết ở [Cost Optimization](../cost-optimization/).
- **Rút gọn transcript cho judge.** Với agent, judge chỉ cần các bước liên quan đến tiêu chí đang chấm, hoặc chỉ cần trạng thái cuối.
- **Chọn số lần chạy theo mục đích.** Bộ CI chạy mỗi ca 1–3 lần; đo pass^k cho agent mới cần k lớn.

Đừng tiết kiệm ở chỗ làm hỏng phép đo: hạ temperature, bỏ nhóm ca khó vì tốn kém, hay bỏ việc hiệu chỉnh judge.

## 3.7 Failure modes thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Pass rate tăng nhưng người dùng phàn nàn nhiều hơn | Bộ eval không phủ loại câu hỏi bị hỏng, hoặc đã bão hoà | Đọc trace production, thêm ca từ lỗi thật; báo cáo theo nhóm [1] |
| CI đỏ ngẫu nhiên, chạy lại thì xanh | Ngưỡng 100% với hệ thống không tất định | Ngưỡng theo khoảng tin cậy, so theo cặp với baseline, chạy nhiều lần [6][29] |
| Hai phiên bản chênh 2 điểm, không ai biết có thật không | Không có khoảng tin cậy; so như hai số độc lập | Bootstrap theo cặp; tính cỡ mẫu cần thiết trước khi so [6] |
| Judge báo gần như toàn Pass | Judge chưa hiệu chỉnh; rubric mơ hồ; lỗi hiếm khiến agreement đẹp giả | Đo TPR/TNR trên nhãn người; tách judge theo từng failure mode [4] |
| Kết quả pairwise đảo ngược khi đổi thứ tự | Thiên lệch vị trí | Chấm hai lần đảo thứ tự, không nhất quán thì coi là hoà [18][19] |
| Judge chấm câu trả lời dài cao hơn | Thiên lệch độ dài | Rubric ghi rõ không xét độ dài; thêm ca ngắn mà đúng vào bộ hiệu chỉnh [2][18] |
| Model chấm model cùng họ cao hơn | Tự ưu ái | Judge khác họ; kiểm tra bằng nhãn người [12][20] |
| "Cải thiện" xuất hiện ngay sau khi đổi judge | Judge mới dễ tính hơn | Đo TPR/TNR của judge mới; hiệu chỉnh pass rate trước khi so [32] |
| Sửa eval nhưng số không đổi | Cache trả kết quả cũ | Xoá cache hoặc `--no-cache` khi cần [29] |
| Agent báo thành công nhưng hệ thống không đổi | Grader đọc transcript thay vì trạng thái | Chấm trạng thái môi trường cuối [1] |
| Agent "gian lận" để qua bài kiểm tra | Môi trường dùng chung giữa các lần chạy, task có lỗ hổng | Cô lập từng trial; đọc transcript của ca pass [1] |
| Task 0% pass với mọi model | Task hoặc grader hỏng | Có lời giải tham chiếu; hai người độc lập xác nhận cách chấm [1] |
| Điểm eval cao bất thường | Ca test lọt vào prompt (few-shot) hoặc vào dữ liệu fine-tune | Tách dev/test; ca test không bao giờ vào prompt hay dữ liệu huấn luyện [4] |
| Benchmark công khai cao nhưng sản phẩm kém | Contamination; benchmark khác domain | Eval trên dữ liệu của bạn [14][15] |
| Faithfulness cao nhưng câu trả lời sai | Context truy xuất sai hoặc lỗi thời | Đo cả retrieval và độ đúng so với đáp án tham chiếu [25] |
| Metric dựng sẵn báo pass với điểm sát 0,5 | Ngưỡng mặc định thấp (DeepEval 0,5) | Đặt ngưỡng tường minh theo dữ liệu đã gán nhãn [34] |
| Điểm đổi sau khi đổi provider hoặc engine dù code không đổi | Khác chat template, parser, cấu hình mặc định | Chạy lại toàn bộ eval khi đổi hạ tầng ([Tool Use §3.9](../tool-use/#39-eval-cho-tool-use)) |
| Rubric bị sửa liên tục, số các tuần không so được | Criteria drift không được quản lý | Version rubric; gán lại ca cũ khi tiêu chí đổi [23] |
| Grader code chấm sai câu tiếng Việt đúng | Không chuẩn hoá Unicode, không tính biến thể đặt dấu | Chuẩn hoá NFC; chấp nhận các cách viết tương đương |

## 3.8 Checklist

**Dataset**

- [ ] Ca lấy từ trace production và lỗi thật, không chỉ từ trí tưởng tượng.
- [ ] Có ca biên, ca đối kháng, ca "không nên làm gì", ca cố ý thiếu thông tin.
- [ ] Dataset có version; mỗi lần chạy ghi version dataset, rubric, judge.
- [ ] Tách dev và test; ca test không bao giờ vào prompt.
- [ ] Báo cáo pass rate theo nhóm, không chỉ tổng.

**Grader**

- [ ] Mọi tiêu chí kiểm được bằng code thì dùng code.
- [ ] Mỗi judge chấm một tiêu chí, nhị phân, có lối thoát "Unknown".
- [ ] Judge được hiệu chỉnh với 100–200 nhãn chuyên gia mỗi failure mode; ghi TPR/TNR.
- [ ] Pairwise chấm hai lần đảo thứ tự.
- [ ] Judge khác họ với model được chấm.

**Thống kê**

- [ ] Mỗi con số đi kèm khoảng tin cậy.
- [ ] So hai phiên bản theo cặp trên cùng bộ ca, cùng đợt chạy.
- [ ] Biết bộ eval của mình phát hiện được chênh lệch tối thiểu bao nhiêu.
- [ ] Không hạ temperature để eval "ổn định hơn".

**Vận hành**

- [ ] Eval chạy trong CI khi prompt, model, tool, dataset đổi; ngưỡng không phải 100%.
- [ ] Bộ đầy đủ chạy theo lịch, nhiều lần mỗi ca.
- [ ] Đọc transcript của mọi ca đổi trạng thái pass/fail.
- [ ] Judge online trên mẫu trace production; trace lỗi được đưa về dataset.
- [ ] TPR/TNR của judge được đo lại định kỳ trên nhãn mới.

---

## Nguồn tham khảo

### Tài liệu chính thức và công cụ

| # | Tài liệu | URL |
|---|---|---|
| 1 | Anthropic: Demystifying evals for AI agents (01/2026) | [anthropic.com/engineering/demystifying-evals-for-ai-agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) |
| 2 | OpenAI: Evaluation best practices | [developers.openai.com/api/docs/guides/evaluation-best-practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices) |
| 3 | OpenAI: Evals guide (thông báo khai tử) | [developers.openai.com/api/docs/guides/evals](https://developers.openai.com/api/docs/guides/evals) |
| 10 | LangSmith: Evaluation | [docs.langchain.com/langsmith/evaluation](https://docs.langchain.com/langsmith/evaluation) |
| 11 | Langfuse: Evaluation overview | [langfuse.com/docs/evaluation/overview](https://langfuse.com/docs/evaluation/overview) |
| 12 | Claude Docs: Define success criteria and build evaluations | [platform.claude.com/docs/en/test-and-evaluate/develop-tests](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests) |
| 15 | OpenAI: Why SWE-bench Verified no longer measures frontier coding capabilities (23/02/2026) | [openai.com/index/why-we-no-longer-evaluate-swe-bench-verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) |
| 25 | Ragas: Faithfulness | [docs.ragas.io/.../faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) |
| 26 | Ragas: Context precision | [docs.ragas.io/.../context_precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) |
| 29 | promptfoo: Command line | [promptfoo.dev/docs/usage/command-line](https://www.promptfoo.dev/docs/usage/command-line/) |
| 30 | Inspect AI: Scorers, Scoring Metrics (epochs, reducer), Options | [inspect.aisi.org.uk/scorers.html](https://inspect.aisi.org.uk/scorers.html), [inspect.aisi.org.uk/metrics.html](https://inspect.aisi.org.uk/metrics.html), [inspect.aisi.org.uk/options.html](https://inspect.aisi.org.uk/options.html) |
| 31 | Braintrust: Evaluate | [braintrust.dev/docs/evaluate](https://www.braintrust.dev/docs/evaluate) |
| 32 | judgy (ai-evals-course): ước lượng pass rate có hiệu chỉnh sai số của judge | [github.com/ai-evals-course/judgy](https://github.com/ai-evals-course/judgy) |
| 33 | Ragas: Context recall | [docs.ragas.io/.../context_recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) |
| 34 | DeepEval: Introduction to LLM evaluation metrics | [deepeval.com/docs/metrics-introduction](https://deepeval.com/docs/metrics-introduction) |
| 37 | DeepEval: Faithfulness (các chế độ `llm`, `hybrid`, `system_one`) | [deepeval.com/docs/metrics-faithfulness](https://deepeval.com/docs/metrics-faithfulness) |
| 38 | promptfoo: Caching | [promptfoo.dev/docs/configuration/caching](https://www.promptfoo.dev/docs/configuration/caching/) |
| 35 | Anthropic: A postmortem of three recent issues (09/2025) | [anthropic.com/engineering/a-postmortem-of-three-recent-issues](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues) |

### Paper

| # | Paper | URL |
|---|---|---|
| 5 | Atil et al. (2024), *Non-Determinism of "Deterministic" LLM Settings*, arXiv:2408.04667 | [arxiv.org/abs/2408.04667](https://arxiv.org/abs/2408.04667) |
| 6 | Miller (2024), *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations*, arXiv:2411.00640 | [arxiv.org/abs/2411.00640](https://arxiv.org/abs/2411.00640) |
| 14 | Xu et al. (2024), *Benchmarking Benchmark Leakage in Large Language Models*, arXiv:2404.18824 | [arxiv.org/abs/2404.18824](https://arxiv.org/abs/2404.18824) |
| 18 | Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, arXiv:2306.05685 | [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685) |
| 19 | Wang et al. (2023), *Large Language Models are not Fair Evaluators*, arXiv:2305.17926 | [arxiv.org/abs/2305.17926](https://arxiv.org/abs/2305.17926) |
| 20 | Panickssery et al. (2024), *LLM Evaluators Recognize and Favor Their Own Generations*, arXiv:2404.13076 | [arxiv.org/abs/2404.13076](https://arxiv.org/abs/2404.13076) |
| 21 | Ye et al. (2024), *Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge*, arXiv:2410.02736 | [arxiv.org/abs/2410.02736](https://arxiv.org/abs/2410.02736) |
| 23 | Shankar et al. (2024), *Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences*, arXiv:2404.12272 | [arxiv.org/abs/2404.12272](https://arxiv.org/abs/2404.12272) |
| 24 | Es et al. (2023), *Ragas: Automated Evaluation of Retrieval Augmented Generation*, arXiv:2309.15217 | [arxiv.org/abs/2309.15217](https://arxiv.org/abs/2309.15217) |
| 27 | Yao et al. (2024), *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*, arXiv:2406.12045 | [arxiv.org/abs/2406.12045](https://arxiv.org/abs/2406.12045) |

### Bài viết của người làm thực tế

| # | Bài viết | URL |
|---|---|---|
| 4 | Hamel Husain, Shreya Shankar: AI Evals FAQ (số mẫu, hiệu chỉnh judge, TPR/TNR, error analysis) và *Validating an LLM judge* | [hamel.dev/blog/posts/evals-faq](https://hamel.dev/blog/posts/evals-faq/), [hamel.dev/blog/posts/evals-faq/how-many-examples-do-i-need-for-an-eval.html](https://hamel.dev/blog/posts/evals-faq/how-many-examples-do-i-need-for-an-eval.html), [hamel.dev/blog/posts/llm-judge](https://hamel.dev/blog/posts/llm-judge/index.html) |
| 8 | Hamel Husain, Shreya Shankar: Why is error analysis so important in LLM evals? | [hamel.dev/blog/posts/evals-faq/why-is-error-analysis-so-important-in-llm-evals-and-how-is-it-performed.html](https://hamel.dev/blog/posts/evals-faq/why-is-error-analysis-so-important-in-llm-evals-and-how-is-it-performed.html) |
| 36 | Hamel Husain, Shreya Shankar: Why do you recommend binary (pass/fail) evaluations instead of 1-5 ratings? | [hamel.dev/blog/posts/evals-faq/why-do-you-recommend-binary-passfail-evaluations-instead-of-1-5-ratings-likert-scales.html](https://hamel.dev/blog/posts/evals-faq/why-do-you-recommend-binary-passfail-evaluations-instead-of-1-5-ratings-likert-scales.html) |

---

Các điểm cần lưu ý: khuyến nghị của Hamel Husain và Shreya Shankar (pass/fail, số trace, 60–80% thời gian cho error analysis) là kinh nghiệm thực tế của họ, không phải nghiên cứu có kiểm soát. Các con số về thiên lệch của judge đo trên model 2023–2024. Công thức cỡ mẫu cho hai tỉ lệ và Cohen's kappa là kiến thức thống kê chuẩn, không lấy từ một nguồn riêng trong danh sách. Code thống kê trong bài đã chạy trên dữ liệu mô phỏng, chưa chạy trên một bộ eval thật.
