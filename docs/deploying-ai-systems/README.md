# Deploying AI Systems: đưa hệ thống AI vào production

Benchmark với 8 người dùng ảo trên một llama-server 8 slot cho kết quả 16 request trong 28,4 giây, tức 0,56 request mỗi giây. Lấy con số đó làm công suất của một replica là sai. Đo lại đúng server ấy, cùng bộ prompt, nhưng gửi request theo tốc độ cố định như người dùng thật: ở 0,5 request/giây, TTFT p95 là 9,0 giây. Ở 0,6 request/giây, TTFT p95 lên 29,5 giây, và sau khi ngừng gửi, server còn mất 36 giây để xử lý hết hàng đợi tồn. Throughput lớn nhất đo được nằm đúng ở điểm hàng đợi bắt đầu dài ra không dừng. Thí nghiệm ở [§3.1](#31-đo-công-suất-một-replica-bằng-tải-open-loop).

Bài này nói về phần nằm quanh model khi hệ thống chạy thật: kiến trúc các tầng, đo công suất, Kubernetes cho GPU, autoscaling, triển khai không làm rớt request, vector database, alert và xử lý sự cố, capacity planning. Phần chọn engine, tính bộ nhớ GPU và benchmark một server nằm ở [Self-hosting LLM](../self-hosting-llm/); trace, log và chi phí theo từng lượt gọi model nằm ở [LLM Observability](../llm-observability/).

- **Nền tảng — Phần 1:** các tầng của một hệ thống AI và tầng nào khó scale, hàng đợi và định luật Little, tải open-loop và coordinated omission, SLO và error budget.
- **Thành phần và công cụ — Phần 2:** batching và cache ở tầng serving, container và nạp model, GPU trên Kubernetes, probe và graceful shutdown, autoscaling, stack routing dành cho LLM, vector database.
- **Thiết kế và vận hành — Phần 3:** đo công suất một replica kèm code, capacity planning kèm code, triển khai thay đổi an toàn, phụ thuộc vào API của provider, alert theo SLO, xử lý sự cố, chi phí, failure modes và checklist.

> Bản tháng 11/2026. Phiên bản đối chiếu: Kubernetes 1.37.2, vLLM 0.33.0, KEDA 2.21.0, NVIDIA GPU Operator 26.7.2, pgvector 0.8.7, Qdrant 1.20.0, Weaviate 1.40.4. Số đo trong bài chạy trên Apple M4 (24 GB bộ nhớ hợp nhất) với llama.cpp và `Qwen2.5-3B-Instruct` Q4_K_M, giống cấu hình ở [Self-hosting LLM §3.2](../self-hosting-llm/#32-benchmark-throughput-và-độ-trễ-trên-tải-thật); số tuyệt đối trên GPU máy chủ khác nhiều, cách đo và hình dạng đường cong thì giống. Manifest Kubernetes trong bài đã kiểm schema bằng kubeconform 0.8.0 với schema Kubernetes 1.37.0 và CRD của KEDA, luật alert đã kiểm bằng promtool 3.16.0; manifest chưa chạy trên cluster thật. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu.

Tài liệu liên quan: [Self-hosting LLM](../self-hosting-llm/) (engine, bộ nhớ, benchmark), [Latency](../latency/) (TTFT, timeout, hedging), [Streaming §3.2](../streaming/#32-stream-qua-proxy-load-balancer-và-cdn) (stream qua proxy và load balancer), [Model routing](../model-routing/) (gateway, fallback, rate limit), [Cost optimization](../cost-optimization/) (Batch API, so chi phí tự host), [LLM Observability §3.7](../llm-observability/#37-alert-ngưỡng-nào-đáng-đánh-thức-người-trực) (alert phía ứng dụng), [Advanced RAG §3.2](../advanced-rag/#32-cập-nhật-index-khi-dữ-liệu-thay-đổi) (cập nhật index), [Evaluation](../evaluation/) (eval khi đổi model).

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Các tầng của một hệ thống AI trong production](#11-các-tầng-của-một-hệ-thống-ai-trong-production)
  - [1.2 Hàng đợi và định luật Little](#12-hàng-đợi-và-định-luật-little)
  - [1.3 Load test open-loop và coordinated omission](#13-load-test-open-loop-và-coordinated-omission)
  - [1.4 SLO và error budget cho dịch vụ LLM](#14-slo-và-error-budget-cho-dịch-vụ-llm)
- [Phần 2 — Thành phần và công cụ](#phần-2--thành-phần-và-công-cụ)
  - [2.1 Batching và cache ở tầng serving](#21-batching-và-cache-ở-tầng-serving)
  - [2.2 Container image và nạp model](#22-container-image-và-nạp-model)
  - [2.3 GPU trên Kubernetes](#23-gpu-trên-kubernetes)
  - [2.4 Probe, rolling update và graceful shutdown](#24-probe-rolling-update-và-graceful-shutdown)
  - [2.5 Autoscaling](#25-autoscaling)
  - [2.6 Routing dành cho LLM: InferencePool, llm-d, Dynamo, KServe](#26-routing-dành-cho-llm-inferencepool-llm-d-dynamo-kserve)
  - [2.7 Vector database trong production](#27-vector-database-trong-production)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Đo công suất một replica bằng tải open-loop](#31-đo-công-suất-một-replica-bằng-tải-open-loop)
  - [3.2 Capacity planning](#32-capacity-planning)
  - [3.3 Triển khai thay đổi an toàn](#33-triển-khai-thay-đổi-an-toàn)
  - [3.4 Phụ thuộc vào API của provider](#34-phụ-thuộc-vào-api-của-provider)
  - [3.5 Alert theo SLO cho tầng serving](#35-alert-theo-slo-cho-tầng-serving)
  - [3.6 Xử lý sự cố](#36-xử-lý-sự-cố)
  - [3.7 Chi phí vận hành](#37-chi-phí-vận-hành)
  - [3.8 Failure modes thường gặp](#38-failure-modes-thường-gặp)
  - [3.9 Checklist](#39-checklist)
- [Nguồn tham khảo](#nguồn-tham-khảo)

---

## Tra cứu nhanh

### Số cần có trước khi lên production

| Số | Đo hoặc lấy ở đâu | Dùng để |
|---|---|---|
| Tốc độ đến lớn nhất một replica chịu được mà vẫn đạt SLO | Tải open-loop trên phần cứng thật, prompt thật ([§3.1](#31-đo-công-suất-một-replica-bằng-tải-open-loop)) | Tính số replica |
| Tốc độ đến ở giờ đỉnh | Log traffic: số request mỗi ngày và tỉ lệ giờ đỉnh / trung bình | Tính số replica ([§3.2](#32-capacity-planning)) |
| Thời gian từ lúc thêm replica tới lúc nó nhận traffic | Đo: cấp node, kéo image, nạp weights, compile, probe | Headroom, `minReplicaCount`, startup probe |
| Thời gian một request ở lại hệ thống (p50, p99) | Metric e2e | Định luật Little cho số slot, connection pool; `--shutdown-timeout` |
| Quota GPU và rate limit của provider | Console cloud, trang rate limit của provider | Trần thật của hệ thống ([§3.4](#34-phụ-thuộc-vào-api-của-provider)) |

### Mặc định dễ gây sự cố

| Thành phần | Mặc định | Hệ quả |
|---|---|---|
| vLLM | `--shutdown-timeout 0` [3] | SIGTERM huỷ ngay mọi request đang chạy; mỗi lần rollout, scale down hay drain node là một loạt stream bị cắt |
| vLLM | `--max-num-queued-reqs` không giới hạn [3] | Quá tải thì hàng đợi dài mãi, TTFT tăng tới hàng phút thay vì trả 503 để client thử chỗ khác |
| Kubernetes | `terminationGracePeriodSeconds` 30 giây [12] | Request dài hơn 30 giây bị SIGKILL khi pod dừng, dù engine có drain |
| Kubernetes | Deployment rolling update 25% max unavailable, 25% max surge [13] | Pod mới cần GPU trống; cluster hết GPU thì rollout treo ở Pending |
| NVIDIA device plugin | Time-slicing không cô lập bộ nhớ và lỗi [8] | Một tiến trình hết bộ nhớ hay crash kéo theo mọi tiến trình chung GPU |
| KEDA Prometheus scaler | `ignoreNullValues: true` [19] | Prometheus mất target thì scaler không báo lỗi, autoscaler mất tín hiệu tải mà không ai biết |
| pgvector | `hnsw.ef_search` 40, lọc sau khi quét index [28] | Điều kiện lọc khớp 10% số dòng thì trung bình chỉ còn 4 kết quả |
| Qdrant | Payload index tạo sau khi đã nạp dữ liệu [32] | HNSW không được tối ưu cho truy vấn có filter |

### Chọn cách chạy model

| Tình huống | Cách chạy | Ghi chú |
|---|---|---|
| Tải nhỏ hoặc thất thường, cần model frontier | API của provider | Lo rate limit, quota, fallback ([§3.4](#34-phụ-thuộc-vào-api-của-provider)) |
| Tác vụ offline, không cần kết quả ngay | Batch API | Giảm 50% giá, xong trong 24 giờ ([Cost optimization §2.3](../cost-optimization/#23-batch-api)) |
| Tải lớn và đều, dữ liệu không được ra ngoài | Tự host trên Kubernetes, vài replica sau một Service | Manifest ở [§2.4](#24-probe-rolling-update-và-graceful-shutdown), autoscale ở [§2.5](#25-autoscaling) |
| Nhiều replica, prompt có prefix dài dùng chung, nhiều model hoặc LoRA | Thêm routing biết KV cache: InferencePool với EPP, llm-d, Dynamo | [§2.6](#26-routing-dành-cho-llm-inferencepool-llm-d-dynamo-kserve) |

---

# Phần 1 — Nền tảng

## 1.1 Các tầng của một hệ thống AI trong production

Một hệ thống AI trong production hiếm khi chỉ là "server gọi model". Hình dạng thường gặp:

```
client ──► app / API (stateless, CPU) ──► LLM gateway ──┬──► API của provider
                │                                       └──► pool model tự host (GPU, N replica)
                ├──► retrieval service ──► vector DB, DB nguồn
                └──► hàng đợi ──► worker (ingest, embedding, batch)

trace, metric, log đi ra từ mọi tầng
```

Các tầng khác nhau ở ba điểm quyết định cách vận hành:

- **Tốc độ scale.** Tầng app chạy trên CPU, image nhỏ, thêm pod mất vài giây. Pool GPU thêm replica mất vài phút: cấp node GPU nếu cluster hết chỗ, kéo image (image `vllm/vllm-openai:v0.33.0` cho amd64 là 9,7 GB nén [18]), nạp hàng chục GB weights, compile và capture CUDA graph ([Self-hosting LLM §2.1](../self-hosting-llm/#21-vllm)). Autoscaler của tầng GPU luôn phản ứng chậm hơn traffic, nên phải có headroom ([§3.2](#32-capacity-planning)).
- **Trạng thái.** App và gateway không giữ trạng thái, thay pod lúc nào cũng được. Replica model giữ KV cache của các request đang chạy và prefix cache của request trước; gỡ nó là cắt stream đang chạy và mất cache ([§2.4](#24-probe-rolling-update-và-graceful-shutdown)). Vector DB giữ dữ liệu thật, cần backup, snapshot, migration ([§2.7](#27-vector-database-trong-production)).
- **Giới hạn bên ngoài.** API của provider có rate limit theo tổ chức và project [37], quota GPU của cloud mặc định có thể là 0 [40]. Đây là trần mà autoscaler của bạn không vượt qua được.

Hệ quả thiết kế:

- Tách tầng theo cách scale. Code ứng dụng không chạy chung pod với model: deploy app không được làm restart model, và ngược lại.
- Việc không cần trả lời ngay (ingest tài liệu, embedding lại, đánh giá hàng loạt) đi qua hàng đợi và worker, hoặc Batch API của provider. Nó không được tranh slot GPU với request của người dùng.
- Request LLM kéo dài hàng chục giây và thường là stream. Mọi proxy trên đường đi cần timeout đủ dài và không buffer response ([Streaming §3.2](../streaming/#32-stream-qua-proxy-load-balancer-và-cdn)). Mọi chỗ giữ kết nối (connection pool, giới hạn concurrency của proxy, số slot của engine) phải đặt theo số request đồng thời thật, tính bằng định luật Little ([§1.2](#12-hàng-đợi-và-định-luật-little)).

## 1.2 Hàng đợi và định luật Little

Định luật Little: số khách trung bình trong một hệ thống ổn định (L) bằng tốc độ đến trung bình (λ) nhân thời gian trung bình mỗi khách ở trong hệ thống (W) [2]:

```
L = λ × W
```

Định luật không giả định gì về phân phối, nên dùng được cho bất kỳ tầng nào. Ví dụ: 2 request mỗi giây, mỗi request mất trung bình 12 giây (stream 600 token ở 50 token/giây) thì trung bình có 24 request đang chạy. Engine cần ít nhất 24 slot (`--max-num-seqs` của vLLM, `-np` của llama-server), connection pool của gateway cần ít nhất 24 kết nối tới pool model, cộng phần cho lúc tải dao động.

Điều định luật Little không nói: khi request đến nhanh hơn tốc độ hệ thống xử lý, hệ thống không còn ổn định. Hàng đợi dài thêm sau mỗi giây, W tăng theo, và TTFT của request đến sau tệ hơn request đến trước. Với LLM điều này xảy ra đột ngột: dưới ngưỡng, continuous batching gộp request mới vào batch đang chạy và TTFT gần như không đổi; vượt ngưỡng một chút, request phải chờ slot trống và TTFT nhảy từ vài giây lên hàng chục giây. Số đo ở [§3.1](#31-đo-công-suất-một-replica-bằng-tải-open-loop) cho thấy điểm gãy này trên một server thật.

Hai hệ quả:

- Công suất của một replica là một tốc độ đến (request mỗi giây) mà ở đó độ trễ còn trong SLO, không phải số token mỗi giây lớn nhất đo được. Throughput lớn nhất đạt được đúng ở vùng mà hàng đợi đã dài.
- Khi quá tải, trả lỗi nhanh tốt hơn xếp hàng vô hạn. vLLM có `--max-num-queued-reqs`: số request đang chạy cộng đang chờ vượt giới hạn thì request mới bị trả HTTP 503 ngay để client thử replica khác; mặc định không giới hạn [3]. `--max-num-queued-tokens` giới hạn theo tổng token prompt đang chờ prefill, và tài liệu trong source gợi ý đặt nó bằng TTFT mục tiêu × throughput prefill [3].

## 1.3 Load test open-loop và coordinated omission

Có hai cách tạo tải:

- **Closed-loop:** N người dùng ảo, mỗi người gửi request, chờ xong, rồi gửi request tiếp. Benchmark ở [Self-hosting LLM §3.2](../self-hosting-llm/#32-benchmark-throughput-và-độ-trễ-trên-tải-thật) đo theo cách này ở các mức đồng thời 1, 2, 4, 8.
- **Open-loop:** request đến theo một tốc độ định trước, không phụ thuộc request trước đã xong chưa. Người dùng thật hành xử như vậy: một nghìn người không chờ nhau.

Closed-loop hợp để so cấu hình (mức lượng tử hoá, số slot) ở cùng mức đồng thời. Nó không trả lời được câu hỏi capacity "chịu được bao nhiêu request mỗi giây", vì khi server chậm đi, các người dùng ảo cũng gửi chậm lại. Tác giả của wrk2 gọi hiện tượng này là coordinated omission: vì mỗi kết nối chỉ gửi request tiếp sau khi nhận response, các response chậm khiến công cụ đo "phối hợp" với server để tránh đo đúng lúc độ trễ cao, và phần lớn độ trễ cao bị bỏ qua [1].

Một load test open-loop đúng cần hai điều:

1. Thời điểm gửi tính trước theo lịch (ví dụ tiến trình Poisson với tốc độ λ), không phụ thuộc response.
2. Độ trễ tính từ thời điểm request *lẽ ra* được gửi theo lịch, không phải lúc nó thật sự rời client. Nếu client bị nghẽn (hết connection, event loop bận), thời gian chờ đó vẫn là độ trễ người dùng thấy.

Code ở [§3.1](#31-đo-công-suất-một-replica-bằng-tải-open-loop) làm đúng hai điều này.

## 1.4 SLO và error budget cho dịch vụ LLM

SLO (service level objective) là mục tiêu đo được cho một chỉ số người dùng cảm nhận được, ví dụ "99% request có TTFT dưới 2,5 giây trong 30 ngày". Phần còn lại (1%) là error budget: còn budget thì được deploy thay đổi, hết budget thì ưu tiên độ ổn định [6].

Với dịch vụ LLM, chọn SLI (chỉ số) cho từng loại lỗi người dùng gặp:

| Người dùng gặp | SLI | Đo ở đâu |
|---|---|---|
| Không có câu trả lời | Tỉ lệ request lỗi: 5xx, 429 do bạn trả, timeout, stream bị cắt giữa chừng | Gateway hoặc app, không phải engine: engine không thấy request bị proxy cắt |
| Chờ lâu mới thấy chữ đầu | Tỉ lệ request có TTFT trên ngưỡng | Gateway (gồm cả hàng đợi trước engine); histogram `vllm:time_to_first_token_seconds` [4] là góc nhìn từ engine |
| Chữ chạy chậm | Tỉ lệ request có thời gian giữa hai token trên ngưỡng | `vllm:inter_token_latency_seconds` [4] |
| Câu trả lời bị cụt | Tỉ lệ request dừng vì hết `max_tokens` | Counter `vllm:request_success_total` theo nhãn `finished_reason` [3] |

Vài điểm riêng của LLM:

- Đo TTFT theo phân vị trên toàn bộ request, không theo trung bình. Một replica quá tải làm p95 tăng mà trung bình gần như không đổi.
- Request có prompt 100.000 token và request có prompt 500 token không cùng TTFT kỳ vọng. Nếu sản phẩm có cả hai, đặt SLO riêng theo loại request hoặc theo khoảng độ dài prompt.
- Ngưỡng SLO nên trùng một biên bucket của histogram để tính được tỉ lệ chính xác. Bucket mặc định của `vllm:time_to_first_token_seconds` trong vLLM 0.33.0 có các biên 0,5; 0,75; 1; 2,5; 5; 7,5; 10 giây, không có 2 giây hay 3 giây [3].

Alert theo SLO dùng burn rate (tốc độ tiêu error budget) ở nhiều cửa sổ thời gian, chi tiết ở [§3.5](#35-alert-theo-slo-cho-tầng-serving).

---

# Phần 2 — Thành phần và công cụ

## 2.1 Batching và cache ở tầng serving

Ba cơ chế quyết định một replica phục vụ được bao nhiêu, và mỗi cơ chế có hệ quả cho cách deploy:

**Continuous batching.** Engine như vLLM, SGLang, llama-server gộp các request đang chạy vào cùng một bước decode, và thêm request mới vào batch mà không chờ batch cũ xong ([Self-hosting LLM §1.3](../self-hosting-llm/#13-prefill-decode-và-vì-sao-batching-quan-trọng)). Ở tầng deploy, việc của bạn là không phá nó: đừng đặt proxy giới hạn mỗi replica một request đồng thời, đừng gom request ở app rồi gửi tuần tự.

**Prefix cache.** vLLM bật prefix caching mặc định: request có chung phần đầu prompt (system prompt dài, tài liệu, lịch sử hội thoại) dùng lại KV cache đã tính. Cache nằm trong từng replica. Load balancer round-robin rải các lượt của cùng một hội thoại ra nhiều replica, và mỗi replica phải tính lại prefix từ đầu. Có hai cách giữ cache:

- Session affinity: gửi mọi request của một phiên tới cùng replica. vLLM production-stack có routing theo session ID [21]. Rẻ, nhưng phân tải không đều khi một số phiên dài hơn hẳn.
- Routing biết KV cache: router biết replica nào đang giữ prefix nào và chọn theo cả tải lẫn cache. llm-d, NVIDIA Dynamo và Endpoint Picker của Gateway API Inference Extension làm việc này ([§2.6](#26-routing-dành-cho-llm-inferencepool-llm-d-dynamo-kserve)). Mức cải thiện các dự án công bố (llm-d: throughput output gấp 3, TTFT nhanh gấp 2 so với round-robin [23]; Dynamo: TTFT nhanh gấp 2 [25]) là số của chính họ trên tải của họ; đo lại trên tải của bạn.

**Cache kết quả ở tầng app.** Câu hỏi trùng hoặc gần trùng có thể trả từ cache mà không chạm GPU. Cách làm và rủi ro ở [Cost optimization §3.4](../cost-optimization/#34-semantic-cache-và-cache-kết-quả).

## 2.2 Container image và nạp model

Thời gian một replica mới sẵn sàng là tổng của bốn bước, và mỗi bước có cách rút ngắn riêng:

| Bước | Thường mất | Cách rút ngắn |
|---|---|---|
| Kéo image | Image vLLM 9,7 GB nén cho amd64 [18] | Cache image trên node (pre-pull bằng DaemonSet, image streaming của cloud); dùng chung một image cho mọi model |
| Nạp weights | Phụ thuộc kích thước model và băng thông tới nơi lưu | Weights đặt sẵn trên volume gần node; Run:ai Model Streamer đọc thẳng từ S3 [16] |
| Compile, capture CUDA graph | Phụ thuộc model; lần đầu lâu hơn hẳn lần có cache | Giữ cache compile của vLLM (`VLLM_CACHE_ROOT`, mặc định `~/.cache/vllm`) trên volume dùng lại được [18] |
| Probe đầu tiên thành công | Theo `periodSeconds` của startup probe | Chu kỳ probe ngắn |

Đo từng bước trên cluster của bạn một lần, ghi lại, và đo lại khi đổi model hoặc image. Con số tổng quyết định startup probe ([§2.4](#24-probe-rolling-update-và-graceful-shutdown)), headroom và số replica tối thiểu ([§3.2](#32-capacity-planning)).

**Weights để ở đâu.** Bốn lựa chọn, từ tệ tới tốt cho production:

- Tải từ Hugging Face lúc pod khởi động. Mỗi replica mới phụ thuộc vào một dịch vụ bên ngoài, tốc độ mạng ra Internet và rate limit; Hugging Face chậm hoặc sập thì bạn không scale được. Chỉ dùng khi thử nghiệm.
- Đóng weights vào image. Image hàng chục GB, mỗi lần đổi code phải build và kéo lại cả weights.
- Volume dựng sẵn (PVC đọc chung, hoặc ổ local của node) chứa đúng một revision của model, mount read-only. Manifest ở [§2.4](#24-probe-rolling-update-và-graceful-shutdown) dùng cách này, kèm `HF_HUB_OFFLINE=1` để pod không gọi ra ngoài.
- Weights dưới dạng OCI artifact, mount bằng `image` volume source. Tính năng `ImageVolume` GA từ Kubernetes 1.36 [15]: weights được version, ký và phân phối như image, nhưng tách khỏi image của engine. Bản ghim theo digest (`ImageVolumeWithDigest`) còn ở alpha từ 1.35 [15].

Nếu weights nằm trên S3 hoặc object storage tương thích, vLLM đọc trực tiếp bằng `--load-format runai_streamer` sau khi cài `vllm[runai]` [16].

**Ghim phiên bản.** Image theo digest, model theo revision, engine theo phiên bản; chi tiết ở [Self-hosting LLM §3.4](../self-hosting-llm/#34-cập-nhật-model-và-rollback). Image vLLM chạy bằng root mặc định để tương thích ngược, và đã chuẩn bị sẵn user `vllm` (UID 2000, GID 0) để chạy không root [18].

## 2.3 GPU trên Kubernetes

**Cấp GPU cho pod.** Kubernetes hỗ trợ GPU AMD và NVIDIA qua device plugin, ở mức stable [7]. NVIDIA GPU Operator cài trọn bộ: driver, device plugin, container runtime, gán nhãn node tự động và giám sát bằng DCGM [9]. GPU chỉ khai trong `limits`; nếu khai thêm `requests` thì hai giá trị phải bằng nhau [7]. Mỗi đơn vị `nvidia.com/gpu` là một GPU nguyên, trừ khi bật một trong các cách chia sẻ dưới đây.

**Chia một GPU cho nhiều pod.** Ba cách, khác nhau ở mức cô lập:

| Cách | Cô lập bộ nhớ | Cô lập lỗi | Hợp khi |
|---|---|---|---|
| Time-slicing | Không: mọi workload thấy toàn bộ bộ nhớ GPU [8] | Không: một workload crash thì tất cả crash [8] | Dev, test, model nhỏ của cùng một nhóm |
| MPS | Có, giới hạn bộ nhớ và compute cho từng workload [8] | Không được nêu trong tài liệu | Nhiều model nhỏ cần chia chắc phần bộ nhớ; không dùng được trên GPU đã bật MIG [8] |
| MIG | Có, ở tầng phần cứng [9] | Có [9] | Production nhiều tenant, trên GPU có hỗ trợ MIG |

Time-slicing và MPS không bật cùng lúc được [8]. Với MIG, device plugin ở chiến lược `mixed` cấp tài nguyên dạng `nvidia.com/mig-<số slice>g.<bộ nhớ>gb` [8].

Với LLM trong production, cách an toàn là một model một hoặc nhiều GPU nguyên. vLLM mặc định chiếm 92% bộ nhớ GPU ([Self-hosting LLM §2.1](../self-hosting-llm/#21-vllm)); chạy chung GPU bằng time-slicing với một tiến trình khác gần như chắc chắn hết bộ nhớ.

**Dynamic Resource Allocation (DRA).** DRA là API mới để xin thiết bị theo thuộc tính (loại GPU, bộ nhớ, MIG profile) thay vì một con số. Feature gate `DynamicResourceAllocation` lên GA ở Kubernetes 1.34 theo source [10]; trang tài liệu ghi stable từ 1.35 [10]. Phía NVIDIA, DRA driver cho GPU (v0.5.1) ghi rằng phần cấp GPU "có thể thử nhưng chưa được hỗ trợ chính thức" [11], và GPU Operator còn liệt kê việc tích hợp DRA driver trong phần kế hoạch [9]. Tháng 11/2026, device plugin vẫn là đường chính cho GPU NVIDIA.

**Shared memory.** vLLM cần `/dev/shm` lớn hơn mặc định của container khi chạy tensor parallel; ví dụ Kubernetes trong tài liệu vLLM mount một `emptyDir` với `medium: Memory` vào `/dev/shm` [18]. Manifest ở §2.4 làm theo.

## 2.4 Probe, rolling update và graceful shutdown

Manifest dưới chạy `gpt-oss-20b` bằng vLLM 0.33.0, một GPU mỗi pod, kèm Service và PodDisruptionBudget; số replica do ScaledObject ở [§2.5](#25-autoscaling) quản lý. Đã kiểm bằng `kubeconform -strict` với schema Kubernetes 1.37.0; chưa chạy trên cluster thật. Các giá trị `--max-num-seqs`, `--max-model-len`, bộ nhớ là ví dụ: lấy số của bạn từ [Self-hosting LLM §3.1](../self-hosting-llm/#31-chọn-model-theo-tác-vụ-và-phần-cứng) và từ đo tải ([§3.1](#31-đo-công-suất-một-replica-bằng-tải-open-loop)).

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpt-oss-20b
  namespace: llm
spec:
  # không khai replicas: KEDA/HPA quản lý số replica; giữ trường này thì mỗi lần
  # kubectl apply kéo số pod về giá trị trong file
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # thêm pod mới trước, chỉ gỡ pod cũ khi pod mới Ready
      maxUnavailable: 0
  selector:
    matchLabels:
      app: gpt-oss-20b
  template:
    metadata:
      labels:
        app: gpt-oss-20b
    spec:
      # preStop 15 s + drain 90 s + dọn dẹp 15 s; phải lớn hơn --shutdown-timeout
      terminationGracePeriodSeconds: 120
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.33.0   # khi chạy thật, ghim theo digest
          command: ["vllm", "serve"]
          args:
            - /models/gpt-oss-20b            # weights tải sẵn vào volume, đúng một revision
            - --served-model-name=openai/gpt-oss-20b
            - --max-model-len=32768
            - --max-num-seqs=64
            - --max-num-batched-tokens=8192
            - --max-num-queued-reqs=96       # 64 đang chạy + 32 chờ; vượt thì trả 503 ngay
            - --shutdown-timeout=90          # mặc định 0: SIGTERM là huỷ ngay request đang chạy
          env:
            - name: HF_HUB_OFFLINE           # không gọi Hugging Face lúc khởi động
              value: "1"
          ports:
            - name: http
              containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: "1"            # GPU chỉ khai ở limits
              memory: 48Gi
            requests:
              cpu: "4"
              memory: 48Gi
          startupProbe:                      # tối đa 60 × 10 s = 10 phút để nạp model
            httpGet:
              path: /health
              port: http
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 60
          readinessProbe:
            httpGet:
              path: /health
              port: http
            periodSeconds: 5
            timeoutSeconds: 2
            failureThreshold: 3
          livenessProbe:                     # /health trả 503 khi engine chết
            httpGet:
              path: /health
              port: http
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 6
          lifecycle:
            preStop:                         # chờ load balancer gỡ pod rồi mới SIGTERM
              exec:
                command: ["sleep", "15"]
          volumeMounts:
            - name: models
              mountPath: /models
              readOnly: true
            - name: vllm-cache               # cache torch.compile; emptyDir chỉ giữ qua lần container restart
              mountPath: /root/.cache/vllm
            - name: shm
              mountPath: /dev/shm
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: models-gpt-oss-20b
            readOnly: true
        - name: vllm-cache
          emptyDir: {}
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 2Gi
---
apiVersion: v1
kind: Service
metadata:
  name: gpt-oss-20b
  namespace: llm
  labels:
    app: gpt-oss-20b
spec:
  selector:
    app: gpt-oss-20b
  ports:
    - name: http
      port: 80
      targetPort: http
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: gpt-oss-20b
  namespace: llm
spec:
  maxUnavailable: 1        # drain node hay nâng cấp cluster chỉ gỡ từng pod một
  selector:
    matchLabels:
      app: gpt-oss-20b
```

Từng phần và lý do:

**Startup probe.** Trong vLLM 0.33.0, server mở socket trước, nhưng chỉ bắt đầu trả lời HTTP sau khi engine dựng xong [3]. Trong lúc nạp model, mọi probe HTTP đều thất bại. Không có startup probe, liveness probe sẽ giết container trước khi model nạp xong, và pod restart mãi. Kubernetes không chạy liveness và readiness cho tới khi startup probe thành công [14]; `failureThreshold × periodSeconds` phải dài hơn thời gian khởi động tệ nhất đo được ở [§2.2](#22-container-image-và-nạp-model), ở đây là 10 phút. Hướng dẫn của llm-d khuyên dùng `/v1/models` cho readiness vì nó chỉ trả 200 khi model đã nạp [14]; với vLLM 0.33.0 chạy trực tiếp, `/health` cũng không trả lời trước lúc đó.

**Liveness probe.** `/health` của vLLM trả 503 khi engine chết (`EngineDeadError`) [3]. Liveness thất bại thì kubelet restart container [14]. Đặt `failureThreshold` đủ rộng để một lần GC hay một bước decode dài không làm restart nhầm.

**Readiness probe.** Pod không ready thì không nhận traffic từ Service [14]. Khi pod đang dừng, endpoint của nó luôn là `ready: false`, bất kể probe [12].

**Graceful shutdown.** Khi pod bị dừng (rollout, scale down, drain node), thứ tự là [12][14]:

1. Pod chuyển sang Terminating. Cùng lúc kubelet bắt đầu dừng pod, control plane đánh dấu endpoint của pod là đang dừng, `ready: false`, để load balancer không dùng nó cho traffic thường nữa [12]. Hai việc chạy song song, không việc nào chờ việc nào; kube-proxy, ingress controller, gateway mỗi thứ nhận thay đổi vào một lúc, nên pod có thể còn nhận request mới thêm một lúc.
2. kubelet chạy `preStop`. `sleep 15` giữ pod sống trong lúc mọi nơi cập nhật danh sách endpoint. SIGTERM chỉ được gửi sau khi `preStop` xong [12].
3. SIGTERM. Với `--shutdown-timeout=90`, vLLM ngừng nhận request mới và tiếp tục chạy request đang có trong tối đa 90 giây; hết hạn thì huỷ phần còn lại [3]. Mặc định là 0: huỷ ngay [3].
4. Hết `terminationGracePeriodSeconds` (tính từ lúc `preStop` bắt đầu) thì SIGKILL. Mặc định 30 giây [12]; ở đây 120 = 15 `preStop` + 90 drain + 15 dọn dẹp, theo đúng công thức trong hướng dẫn của llm-d [14].

Đặt `--shutdown-timeout` theo p99 thời gian một request ở lại hệ thống. Request dài hơn vẫn bị cắt, nên client phải xử lý được stream bị cắt giữa chừng ([Streaming §3.6](../streaming/#36-lỗi-giữa-chừng-đứt-kết-nối-lỗi-sau-khi-đã-gửi-một-phần-retry-và-resume)).

**Rolling update.** `maxSurge: 1, maxUnavailable: 0` thêm một pod mới, chờ nó ready rồi mới gỡ một pod cũ: công suất không giảm trong lúc rollout, nhưng cần một GPU trống cho pod thêm. Không có GPU trống thì pod mới nằm Pending và rollout treo. Cách ngược lại (`maxSurge: 0, maxUnavailable: 1`) không cần GPU thừa nhưng mất một replica công suất trong suốt rollout, mỗi bước kéo dài bằng thời gian khởi động. Mặc định của Deployment là 25% cho cả hai [13]. Chọn cách nào thì tính nó vào capacity ([§3.2](#32-capacity-planning)).

**PodDisruptionBudget.** PDB giới hạn số pod bị gỡ cùng lúc bởi các gián đoạn tự nguyện như drain node để nâng cấp [13]. Nó không áp lên rolling update của chính Deployment, và xoá pod hay xoá Deployment trực tiếp cũng bỏ qua PDB [13].

**Không khai `replicas`.** Khi HPA hoặc KEDA quản lý Deployment, tài liệu Kubernetes khuyên bỏ `spec.replicas` khỏi manifest: giữ lại thì mỗi lần `kubectl apply` (đổi image, đổi tham số) kéo số pod về đúng giá trị trong file, dù autoscaler đang giữ nhiều hơn [20]. Với Deployment đang chạy, bỏ trường này một cách ngây thơ cũng nguy hiểm: giá trị mặc định là 1, nên mọi pod trừ một bắt đầu dừng; tài liệu HPA hướng dẫn gỡ trường khỏi cấu hình đã apply trước (`kubectl apply edit-last-applied deployment/<tên>`) rồi mới gỡ khỏi file [20].

**Giới hạn hàng đợi.** `--max-num-queued-reqs=96` = 64 slot đang chạy + 32 chờ. Quá mức đó replica trả 503 ngay [3], gateway thử replica khác hoặc báo client thử lại, thay vì để request chờ một phút rồi timeout.

## 2.5 Autoscaling

**Chọn tín hiệu.** CPU và bộ nhớ của pod không nói gì về tải của một server LLM: engine cấp sẵn KV cache lúc khởi động, nên bộ nhớ GPU luôn gần đầy dù đang rảnh hay quá tải; CPU không phải tài nguyên nghẽn. Mức sử dụng GPU từ DCGM cho biết GPU bận, không cho biết người dùng đang phải chờ. Tín hiệu gần với trải nghiệm người dùng hơn:

| Tín hiệu | Metric vLLM [4] | Ưu, nhược |
|---|---|---|
| Số request trong hệ thống (đang chạy + đang chờ) | `vllm:num_requests_running` + `vllm:num_requests_waiting` | Tỉ lệ thuận với tổng tải (định luật Little); ổn định khi tải đều. Manifest dưới dùng tín hiệu này |
| Chỉ số request đang chờ | `vllm:num_requests_waiting` | Tăng ngay khi bắt đầu quá tải, vLLM production-stack dùng nó cho KEDA [21]; nhưng chỉ thấy phần tải tràn, không thấy tổng tải (xem dưới) |
| Mức dùng KV cache | `vllm:kv_cache_usage_perc` | Báo sớm trước khi có preemption; không phân biệt "đầy vì nhiều request" với "đầy vì vài request rất dài" |
| TTFT p95 | `vllm:time_to_first_token_seconds` | Đúng thứ người dùng thấy, nhưng tăng khi hàng đợi đã dài: scale theo nó là scale muộn |

llm-d đã đánh dấu Workload Variant Autoscaler của họ là deprecated và khuyên dùng KEDA với metric từ Endpoint Picker [23].

**KEDA.** KEDA đọc metric từ Prometheus và điều khiển HPA [19]. ScaledObject đi kèm Deployment ở §2.4:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: gpt-oss-20b
  namespace: llm
spec:
  scaleTargetRef:
    name: gpt-oss-20b
  minReplicaCount: 2
  maxReplicaCount: 8
  pollingInterval: 15
  fallback:                # Prometheus lỗi 3 lần liên tiếp: giữ số replica hiện tại, tối thiểu 4
    failureThreshold: 3
    replicas: 4
    behavior: currentReplicasIfHigher
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 0
          policies:
            - type: Pods
              value: 2
              periodSeconds: 60
        scaleDown:
          stabilizationWindowSeconds: 600   # pod mới mất vài phút mới Ready: đừng gỡ vội
          policies:
            - type: Pods
              value: 1
              periodSeconds: 120
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring.svc:9090
        # tổng request đang chạy + đang chờ trên toàn pool; metricType mặc định AverageValue
        # nên số replica mong muốn = tổng / threshold
        query: >-
          sum(vllm:num_requests_running{model_name="openai/gpt-oss-20b"})
          + sum(vllm:num_requests_waiting{model_name="openai/gpt-oss-20b"})
        # ví dụ: replica đạt SLO với 24 request đồng thời (đo ở §3.1), chừa 25% headroom:
        # 24 × 0,75 = 18
        threshold: "18"
        ignoreNullValues: "false"   # mặc định true: mất target thì không báo lỗi
```

- Với `metricType` mặc định `AverageValue`, số replica mong muốn = giá trị metric / `threshold`. Ví dụ trong tài liệu KEDA: hàng đợi 20, threshold 5 thì scale lên 4 replica, bất kể đang có bao nhiêu [19]. Ở đây tổng 126 request đang chạy và chờ trên toàn pool thì KEDA muốn 126 / 18 = 7 replica.
- `threshold` = số request đồng thời một replica gánh được mà vẫn đạt SLO × (1 − headroom). Phần đầu lấy từ đo tải ở [§3.1](#31-đo-công-suất-một-replica-bằng-tải-open-loop): cột "số request trong hệ thống" ở mức tải lớn nhất còn đạt SLO, hoặc `in_flight(tốc độ đạt SLO, thời gian ở lại)` ở [§3.2](#32-capacity-planning). Trên server thử nghiệm của bài, con số đó là 2,8, với headroom 25% thì threshold là 2,1; số 24 và 18 trong manifest chỉ là ví dụ cho một GPU máy chủ. Đặt threshold đúng bằng mức đạt SLO là bỏ hết headroom: autoscaler giữ mỗi replica ở mép SLO, cộng thêm tolerance 10% của HPA (xem dưới) thì tải vượt mép đó tới 10% vẫn chưa scale, và mọi đợt tải dồn hay khoảng chờ pod mới ready đều thành vi phạm SLO. Headroom ở đây cùng nghĩa với `headroom` của `replicas_needed`; phần dự phòng N+1 thì do `minReplicaCount` lo.
- Vì sao không scale chỉ theo `num_requests_waiting`: công thức chia đều cho threshold, nên khi pool đủ công suất, hàng đợi về 0 và KEDA muốn số replica tối thiểu. Hết cửa sổ scale-down, pool bị gỡ dần cho tới khi thiếu công suất, hàng đợi dài lại, pool scale lên, và vòng lặp lặp lại: số replica dao động, mỗi đợt thiếu là một đợt vi phạm SLO, mỗi lần gỡ pod là một lần drain. Tổng số request trong hệ thống thì ở tải đều bằng λ × W theo định luật Little, nên số replica mong muốn đứng yên. Nếu cần phản ứng nhanh hơn khi hàng đợi tăng, thêm `num_requests_waiting` làm trigger thứ hai: khi có nhiều metric, HPA lấy số replica lớn nhất trong các đề xuất [20].
- `query` phải trả về đúng một giá trị [19], nên dùng `sum(...)`. Nhãn `model_name` là tên model phục vụ (`--served-model-name`) [3].
- `scaleUp` không có stabilization window và cho thêm 2 pod mỗi phút: thêm sớm vì pod mới mất vài phút mới nhận traffic. `scaleDown` chờ 10 phút và bớt từng pod một: HPA mặc định chờ 5 phút trước khi giảm [20]; tải LLM dao động theo phút, và gỡ một pod là cắt request đang chạy trên nó.
- `ignoreNullValues: "false"` khiến scaler báo lỗi khi Prometheus mất target thay vì lặng lẽ không có số [19]. Khi scaler lỗi đủ `failureThreshold` lần liên tiếp, `fallback` giữ số replica hiện tại nếu nó cao hơn 4 [19].
- HPA bỏ qua thay đổi nhỏ hơn 10% quanh mục tiêu (tolerance mặc định) [20].

**Scale về 0.** KEDA tự xử lý chuyển 0 ↔ 1 replica, theo `activationThreshold` [19]. HPA của Kubernetes cũng cho `minReplicas: 0` với metric object hoặc external; feature gate `HPAScaleToZero` là beta và bật mặc định từ 1.37 [20]. Knative có scale-to-zero bật trong cấu hình mặc định [27]. Đánh đổi: request đầu tiên sau giờ rảnh chờ trọn thời gian khởi động ở [§2.2](#22-container-image-và-nạp-model), thường vài phút. Chỉ hợp với model nội bộ, batch, hoặc tác vụ chấp nhận chờ.

**Sleep mode của vLLM.** Sleep mode giải phóng phần lớn bộ nhớ GPU (weights và KV cache) mà không dừng server: level 1 chuyển weights sang RAM CPU và bỏ KV cache, gọi `POST /sleep?level=1` và `POST /wake_up` [17]. Hai endpoint này chỉ có khi server chạy với `--enable-sleep-mode` và biến môi trường `VLLM_SERVER_DEV_MODE=1`; tài liệu vLLM cảnh báo các endpoint dev không được mở cho người dùng [17], nên chặn chúng ở gateway hoặc bằng NetworkPolicy và chỉ gọi từ bên trong cluster. Nó hữu ích khi vài model thay nhau dùng một GPU, vì đánh thức nhanh hơn nạp lại từ đầu. Nó không giảm chi phí GPU trên Kubernetes: pod ngủ vẫn giữ GPU đã cấp.

**Autoscaler của node.** Nếu cluster phải thêm node GPU, thời gian chờ cloud cấp máy cộng vào thời gian scale, và quota GPU của tài khoản là trần cứng ([§3.4](#34-phụ-thuộc-vào-api-của-provider)).

## 2.6 Routing dành cho LLM: InferencePool, llm-d, Dynamo, KServe

Một Kubernetes Service phân request theo kết nối, không biết replica nào đang quá tải hay đang giữ prefix nào. Với hai, ba replica và prompt ít phần chung, điều đó chấp nhận được. Với nhiều replica, prompt có prefix dài dùng chung, nhiều model hoặc LoRA adapter, các dự án dưới đây thêm một tầng routing:

| Dự án | Bản mới nhất | Làm gì |
|---|---|---|
| Gateway API Inference Extension | v1.6.2 [22] | Định nghĩa `InferencePool` (`inference.networking.k8s.io/v1`): một nhóm pod model server dùng làm backend của Gateway API, thay cho Service [22]. Endpoint Picker (EPP) chọn replica cho từng request theo KV cache và chi phí request [22]. EPP và API `InferenceObjective` đã chuyển sang repo llm-d-router [24]; repo này giữ InferencePool, một EPP nhẹ và bộ conformance test [22] |
| llm-d | v1.0.0 [23] | Routing theo prefix cache và tải, offload KV cache sang CPU hoặc đĩa, tách prefill và decode, expert parallelism cho model rất lớn [23]; graceful shutdown có cả phía router: EPP trả 503 có thể retry cho request còn trong hàng đợi của nó [14] |
| NVIDIA Dynamo | v1.6.2 [25] | Tầng điều phối phía trên vLLM, SGLang, TensorRT-LLM: routing theo KV cache, tách prefill và decode thành hai pool scale riêng, cache KV nhiều tầng, Planner tự scale theo SLA [25] |
| vLLM production-stack | vllm-stack-0.1.14 [21] | Helm chart cho vLLM kèm router: round-robin, theo session ID; routing theo prefix còn đang làm, routing theo KV cache và tách prefill nằm trong roadmap [21] |
| KServe | v0.21.0 [26] | Nền tảng serving chung, hỗ trợ backend vLLM và llm-d, offload KV cache sang CPU hoặc đĩa; CRD `LLMInferenceService` (v1alpha1, v1alpha2) dành cho LLM [26] |

Chọn theo vấn đề đang có, không theo danh sách tính năng. Tách prefill và decode chỉ đáng khi model lớn, prompt dài và có đủ GPU để chia hai pool. Routing theo KV cache đáng khi cache hit rate của từng replica thấp hơn nhiều so với tỉ lệ prefix dùng chung trong traffic. Mỗi tầng thêm vào là một thành phần nữa phải nâng cấp, giám sát, và có thể hỏng: router chết thì cả pool mất traffic, nên router cũng cần nhiều replica và chế độ fail-open hoặc fail-close được chọn có chủ đích (`failureMode` của InferencePool [14]).

## 2.7 Vector database trong production

Phần chunking, embedding và chọn vector DB ở [Embeddings](../embeddings/); cập nhật index khi dữ liệu đổi ở [Advanced RAG §3.2](../advanced-rag/#32-cập-nhật-index-khi-dữ-liệu-thay-đổi). Phần này là vận hành.

**Ước lượng bộ nhớ.** Vector float32 chiếm `số vector × số chiều × 4` byte. Qdrant công bố công thức cho phần HNSW: `số vector × m × 2 × 4 byte × 1,2` (hệ số 1,2 cho các lớp dưới và dữ liệu quản lý) [33]. Với 1 triệu vector 1536 chiều và `m = 16`:

```
vector:  1.000.000 × 1536 × 4      ≈ 6,1 GB  (5,7 GiB)
HNSW:    1.000.000 × 16 × 2 × 4 × 1,2 ≈ 0,15 GB
```

Nhân với số bản sao (replication factor), cộng payload. Trong pgvector mỗi vector chiếm `4 × số chiều + 8` byte [28]; index không bắt buộc nằm hết trong RAM nhưng nằm được thì nhanh hơn [28].

**Giảm bộ nhớ bằng quantization.** Qdrant: scalar quantization (8 bit) giảm 4 lần, binary quantization giảm 32 lần, product quantization tới 64 lần [31]. Tài liệu Qdrant nói rõ một số embedding model không quantize tốt và phải kiểm recall so với bản không quantize [31]. pgvector có `halfvec` (tới 4.000 chiều) và binary quantization [28]. Lưu vector trên đĩa thì dùng SSD local từ 50.000 IOPS trở lên, và giữ index HNSW trong bộ nhớ vì duyệt graph là nhiều lần đọc ngẫu nhiên nhỏ [33].

**Dựng index.** HNSW của pgvector dựng nhanh hơn nhiều khi graph vừa `maintenance_work_mem`; không vừa thì Postgres in `NOTICE: hnsw graph no longer fits into maintenance_work_mem after ... tuples` và việc dựng chậm hẳn [28]. Đừng đặt `maintenance_work_mem` cao tới mức hết RAM của server [28]. Mặc định 2 worker song song; tăng bằng `max_parallel_maintenance_workers` [28]. Trên bảng đang phục vụ, dùng `CREATE INDEX CONCURRENTLY` để không chặn ghi [28].

**Filter.** Đây là chỗ hay sai nhất khi lên production:

- pgvector lọc *sau* khi quét index. Điều kiện khớp 10% số dòng, với `hnsw.ef_search` mặc định 40, trung bình chỉ còn 4 dòng [28]. Từ 0.8.0 có iterative index scan: `SET hnsw.iterative_scan = strict_order` (đúng thứ tự khoảng cách) hoặc `relaxed_order` (có thể lệch thứ tự một chút, recall tốt hơn) để quét thêm cho tới khi đủ kết quả [28].
- Qdrant cần payload index cho các trường dùng trong filter, và khuyên tạo chúng ngay sau khi tạo collection, trước khi nạp dữ liệu, để HNSW được tối ưu cho truy vấn có filter [32]. Chỉ index trường thật sự dùng để lọc, vì payload index tốn bộ nhớ và đĩa [32].

Test retrieval với filter thật của production (theo tenant, theo quyền, theo thời gian), không chỉ truy vấn không filter.

**Đổi model, đổi cách chunk: dựng song song rồi chuyển alias.** Vector từ hai embedding model khác nhau không được trộn trong một collection; Qdrant khuyên lưu model và phiên bản pipeline trong metadata của collection và embed lại toàn bộ khi một trong hai đổi [34]. Dựng collection mới, chạy eval, rồi chuyển alias: Qdrant đổi alias nguyên tử, request đang chạy không bị ảnh hưởng [29]; Weaviate có alias từ v1.32, cũng chuyển nguyên tử [35]. Quy trình đầy đủ ở [Embeddings §3.10](../embeddings/#310-migration--versioning--đổi-embedding-model-mà-không-làm-sập-hệ-thống).

**Backup và khôi phục.**

- Qdrant: snapshot là file tar chứa dữ liệu và cấu hình của một collection trên một node; cluster nhiều node thì phải snapshot từng node [30]. Snapshot toàn bộ storage chỉ dùng cho triển khai một node [30].
- Weaviate: backup lên S3, GCS hoặc Azure Storage bằng một lệnh, có backup tăng dần từ v1.37 [36]. Backend filesystem chỉ dành cho dev và test [36].
- pgvector: là Postgres, nên dùng backup và PITR của Postgres; index dựng lại được từ bảng.

Backup chưa từng khôi phục thử thì chưa phải backup. Đo thời gian khôi phục toàn bộ index, kể cả thời gian dựng lại HNSW nếu backup không chứa index; con số này là thời gian sập tối thiểu khi mất dữ liệu.

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Đo công suất một replica bằng tải open-loop

Code dưới gửi request tới endpoint OpenAI-compatible (vLLM, SGLang, llama-server) theo tiến trình Poisson với tốc độ cho trước, đo TTFT và thời gian tổng tính từ thời điểm request lẽ ra được gửi ([§1.3](#13-load-test-open-loop-và-coordinated-omission)). Cần Python 3.11+ và `httpx2`.

```python
"""Tải open-loop cho endpoint OpenAI-compatible: request đến theo tiến trình Poisson với tốc độ cố
định, không chờ request trước xong, giống người dùng thật. TTFT và thời gian tính từ lúc request
lẽ ra được gửi, nên hàng đợi ở bất kỳ đâu (client, proxy, server) đều bị tính vào."""
import asyncio
import json
import random
import time

import httpx2

TEXT_FIELDS = ("content", "reasoning", "reasoning_content")


async def _one(client, url, body, scheduled):
    """Một request. Lỗi (HTTP 503 khi server từ chối, timeout, stream không có token) được trả về
    như một kết quả, không raise: vùng quá tải là chỗ cần đo nhất, và tỉ lệ lỗi cũng là một SLI."""
    first = None
    try:
        async with client.stream("POST", url, json=body) as r:
            if r.status_code != 200:
                return {"error": f"http_{r.status_code}"}
            async for line in r.aiter_lines():
                if first is None and line.startswith("data: ") and line != "data: [DONE]":
                    choices = json.loads(line[6:]).get("choices", [])
                    if any(c["delta"].get(f) for c in choices for f in TEXT_FIELDS):
                        first = time.perf_counter()
    except httpx2.HTTPError as e:
        return {"error": type(e).__name__}
    if first is None:
        return {"error": "no_tokens"}
    return {"ttft": first - scheduled, "e2e": time.perf_counter() - scheduled}


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 2)


async def open_loop(url, model, prompts, rate, duration, *, max_tokens=128, extra=None, seed=0):
    """Gửi request trong `duration` giây, trung bình `rate` request mỗi giây, rồi chờ tất cả xong."""
    rng = random.Random(seed)
    limits = httpx2.Limits(max_connections=None, max_keepalive_connections=None)
    tasks, t = [], 0.0
    async with httpx2.AsyncClient(timeout=900, limits=limits) as client:
        start = time.perf_counter()
        while (t := t + rng.expovariate(rate)) < duration:
            await asyncio.sleep(max(0.0, start + t - time.perf_counter()))
            body = {"model": model, "messages": prompts[len(tasks) % len(prompts)],
                    "max_tokens": max_tokens, "temperature": 0, "stream": True, **(extra or {})}
            tasks.append(asyncio.create_task(_one(client, url, body, start + t)))
        res = await asyncio.gather(*tasks)
        span = time.perf_counter() - start
    ok = [r for r in res if "error" not in r]
    errors = {}
    for r in res:
        if "error" in r:
            errors[r["error"]] = errors.get(r["error"], 0) + 1
    ttft, e2e = [r["ttft"] for r in ok], [r["e2e"] for r in ok]
    # phân vị chỉ tính trên request thành công; đọc kèm error_rate
    return {"rate": rate, "requests": len(res),
            "error_rate": round(1 - len(ok) / len(res), 3) if res else 0.0, "errors": errors,
            "ttft_p50": pct(ttft, 50), "ttft_p95": pct(ttft, 95),
            "e2e_p50": pct(e2e, 50), "e2e_p95": pct(e2e, 95),
            # số request trung bình nằm trong hệ thống = tổng thời gian ở lại / thời gian đo
            "in_system_mean": round(sum(e2e) / span, 1),
            "drain_s": round(span - duration, 1)}
```

Những chỗ đáng chú ý:

- `scheduled` là thời điểm theo lịch, không phải lúc request thật sự đi. Nếu event loop bận hay hết kết nối, phần trễ đó vẫn được tính.
- Connection pool không giới hạn (`max_connections=None`). Giới hạn pool là một dạng closed-loop: request thứ N+1 chờ kết nối trống ở client và server trông nhanh hơn thật.
- Token đầu tính cả `reasoning` và `reasoning_content`, giống [Self-hosting LLM §3.2](../self-hosting-llm/#32-benchmark-throughput-và-độ-trễ-trên-tải-thật).
- Request lỗi (503 khi server chạm giới hạn hàng đợi, timeout, stream không có token) được đếm vào `errors` và `error_rate` thay vì làm dừng cả lượt đo; phân vị chỉ tính trên request thành công, nên luôn đọc kèm `error_rate`. Vùng quá tải là chỗ cần đo nhất, và server có giới hạn hàng đợi sẽ trả 503 đúng ở vùng đó.
- `drain_s`: thời gian từ lúc ngừng gửi tới lúc request cuối xong. Dưới công suất, nó gần bằng thời gian một request. Vượt công suất, nó tăng theo thời gian chạy test, vì hàng đợi còn tồn.

Kiểm bằng một server giả lập một slot, mỗi request chiếm slot 0,1 giây (`t_loadgen.py`): tải 2 request/giây cho TTFT p50 dưới 0,3 giây; tải 20 request/giây vào server chỉ làm được 10 cho TTFT p95 trên 1 giây và `drain_s` trên 1 giây; token reasoning cũng được tính là token đầu; server trả 503 cho một nửa số request thì lượt đo vẫn chạy hết, `errors` đếm đúng số 503 và phân vị tính trên nửa còn lại; mọi request đều 503 thì `error_rate` là 1 và không có phân vị.

**Kết quả trên một server thật.** llama-server 8 slot, `-c 16384`, `Qwen2.5-3B-Instruct` Q4_K_M trên Apple M4. Prompt lấy từ 46 câu hỏi thật, trung bình 565 token; mỗi request sinh đúng 128 token (`ignore_eos`), tắt cache prompt để mỗi request phải prefill. Mỗi mức tải chạy 240 giây; không request nào lỗi ở mọi mức:

| Tốc độ đến (request/giây) | Số request | TTFT p50 (giây) | TTFT p95 (giây) | E2E p50 (giây) | E2E p95 (giây) | Số request trong hệ thống, trung bình | Thời gian xả hàng đợi (giây) |
|---|---|---|---|---|---|---|---|
| 0,1 | 27 | 1,10 | 2,72 | 5,06 | 9,06 | 0,6 | 3,8 |
| 0,2 | 51 | 1,17 | 3,18 | 6,65 | 13,20 | 1,6 | 1,7 |
| 0,3 | 78 | 1,14 | 3,92 | 7,82 | 15,13 | 2,8 | 4,7 |
| 0,4 | 95 | 1,26 | 6,19 | 10,82 | 16,79 | 4,1 | 3,8 |
| 0,5 | 121 | 2,34 | 8,98 | 13,95 | 20,61 | 6,9 | 4,5 |
| 0,6 | 159 | 7,33 | 29,52 | 19,00 | 39,84 | 12,5 | 35,7 |
| 0,7 | 182 | 27,58 | 62,69 | 39,59 | 73,92 | 26,3 | 69,0 |

Đọc bảng:

- Tới 0,4 request/giây, TTFT p50 gần như không đổi (1,1–1,3 giây, gần bằng TTFT 1,09 giây khi server chỉ có một request ở benchmark closed-loop). p95 thì tăng dần từ 2,7 lên 6,2 giây: request đến theo Poisson nên có lúc vài request đến sát nhau và phải chờ nhau prefill. Trung bình che mất hiện tượng này, phân vị thì không.
- 0,5 request/giây: p50 gần gấp đôi (1,26 → 2,34 giây), nhưng hàng đợi vẫn xả hết trong vài giây sau khi ngừng gửi. Server còn theo kịp.
- 0,6 request/giây: p50 lên 7,3 giây, p95 lên 29,5 giây, hàng đợi tồn 36 giây. Số request trung bình trong hệ thống (12,5) vượt số slot (8): request phải chờ slot trống. Đây là điểm gãy ở [§1.2](#12-hàng-đợi-và-định-luật-little). Chạy test lâu hơn thì các con số ở mức này còn tệ hơn, vì hàng đợi dài thêm theo thời gian.
- 0,7 request/giây: TTFT p50 27,6 giây. Không người dùng nào chờ được.
- Benchmark closed-loop 8 đồng thời trên cùng server ([Self-hosting LLM §3.2](../self-hosting-llm/#32-benchmark-throughput-và-độ-trễ-trên-tải-thật)) cho 0,56 request/giây, nằm ngay giữa 0,5 và 0,6, tức đúng ở điểm gãy. Với SLO "TTFT p95 dưới 5 giây", công suất thật của replica là 0,3 request/giây, khoảng một nửa con số closed-loop.

Thêm hai lưu ý khi đo trên hệ thống của bạn:

- Đo với phân phối độ dài prompt và output thật. Request 128 token output đều nhau dễ hơn nhiều so với traffic thật, nơi một vài request dài chiếm slot hàng phút.
- Chạy mỗi mức tải đủ lâu để hàng đợi kịp lộ ra (vài phút), và chạy lại vài lần. Số đo trong bảng là một lần chạy; ở Self-hosting LLM, hai lần chạy cùng cấu hình trên máy này lệch 11–12%.

## 3.2 Capacity planning

Từ đường cong ở §3.1 và dự báo traffic, số replica cần cho giờ đỉnh:

```python
"""Capacity planning cho một dịch vụ LLM: từ số đo open-loop của một replica và dự báo traffic,
tính số replica cần cho giờ đỉnh."""
import math


def max_rate_within_slo(curve, slo_ttft_p95):
    """`curve`: các cặp (request/giây, TTFT p95 tính bằng giây) đo bằng tải open-loop trên MỘT
    replica. Trả về mức tải lớn nhất đã đo mà TTFT p95 còn trong SLO. Không nội suy và dừng ở
    điểm vượt SLO đầu tiên: sau điểm gãy, hàng đợi tăng theo thời gian và số đo không lặp lại."""
    best = None
    for rate, ttft_p95 in sorted(curve):
        if ttft_p95 > slo_ttft_p95:
            break
        best = rate
    if best is None:
        raise ValueError("mức tải thấp nhất đã đo cũng vượt SLO: đổi cấu hình hoặc đo ở mức thấp hơn")
    return best


def peak_rate(requests_per_day, peak_to_average):
    """Tốc độ đến ở giờ đỉnh. `peak_to_average` lấy từ traffic thật (giờ đỉnh / trung bình ngày)."""
    return requests_per_day / 86_400 * peak_to_average


def replicas_needed(peak, per_replica, *, headroom=0.25, spare=1):
    """`headroom`: phần công suất mỗi replica để trống cho dao động trong giờ đỉnh và cho khoảng
    thời gian autoscaler chưa kịp thêm máy. `spare`: replica dự phòng khi một máy chết hoặc đang
    deploy (N+1). Kết quả là số replica phải chạy ở giờ đỉnh."""
    if not 0 <= headroom < 1:
        raise ValueError("headroom phải trong [0, 1)")
    return math.ceil(peak / (per_replica * (1 - headroom))) + spare


def in_flight(rate, seconds_in_system):
    """Định luật Little: số request đang nằm trong hệ thống = tốc độ đến × thời gian mỗi request ở
    lại. Dùng để đặt số slot, `max_num_seqs`, connection pool và giới hạn concurrency của proxy."""
    return rate * seconds_in_system
```

`t_capacity.py` kiểm các trường hợp: chọn điểm đã đo lớn nhất còn trong SLO, không nội suy giữa hai điểm; dừng ở điểm vượt SLO đầu tiên kể cả khi một điểm đo nhiễu phía sau lại thấp; báo lỗi khi ngay điểm thấp nhất đã vượt; 200.000 request/ngày với giờ đỉnh gấp 3 trung bình cho 6,94 request/giây; 6,94 / (0,3 × 0,75) = 30,8 nên cần 31 replica, cộng 1 dự phòng là 32; Little với 0,4 request/giây và 10 giây mỗi request cho 4 request đang chạy.

Với đường cong đo ở §3.1 và SLO TTFT p95 dưới 5 giây, `max_rate_within_slo` trả 0,3 request/giây (0,4 đã vượt, 6,19 giây). Một sản phẩm 50.000 request/ngày, giờ đỉnh gấp 3 trung bình, có `peak_rate` 1,74 request/giây. `replicas_needed(1.74, 0.3)` = ⌈1,74 / (0,3 × 0,75)⌉ + 1 = ⌈7,7⌉ + 1 = 9 replica. Nếu lấy con số closed-loop 0,56 và không chừa headroom hay dự phòng, kết quả là 4 replica: thiếu hơn một nửa, và giờ đỉnh nào cũng vi phạm SLO. Ở giờ đỉnh, mỗi request ở lại tới 15 giây (E2E p95 tại 0,3 request/giây), nên `in_flight(1.74, 15.13)` cho khoảng 26 request đồng thời trên toàn pool: connection pool của gateway tới pool model cần ít nhất chừng đó.

**Headroom bao nhiêu.** `headroom` là phần công suất để trống cho hai thứ: dao động trong giờ đỉnh (traffic giờ đỉnh không đều từng phút), và khoảng thời gian từ lúc tải tăng tới lúc replica mới nhận traffic. Nếu traffic có thể tăng 30% trong 10 phút và replica mới mất 8 phút, headroom phải đủ cho mức tăng đó. 25% là điểm bắt đầu, không phải quy luật; chỉnh theo số đo thật.

**Dự phòng N+1.** `spare=1` cho phép một replica chết hoặc đang rollout mà vẫn đạt SLO. Nếu rollout dùng `maxUnavailable: 1` ([§2.4](#24-probe-rolling-update-và-graceful-shutdown)), một replica vắng mặt trong suốt rollout, nên một replica dự phòng là tối thiểu. Pool trải trên nhiều zone thì tính cho trường hợp mất cả một zone.

**Little cho các giới hạn khác.** `in_flight(rate, seconds_in_system)` cho số request đồng thời; dùng nó để đặt `--max-num-seqs`, connection pool của gateway, giới hạn concurrency của proxy và số kết nối tới vector DB. Lấy thời gian ở lại theo p95 hoặc p99, không theo trung bình, nếu không giới hạn sẽ chặn đúng lúc tải cao.

**Quota.** Trên AWS, quota mặc định cho "Running On-Demand G and VT instances" và "Running On-Demand P instances" là 0 vCPU; AWS khuyên xin tăng quota trước khi cần [40]. Xin quota cho số replica giờ đỉnh cộng một lần rollout cộng dự phòng, ở mọi region bạn định dùng, trước ngày ra mắt.

## 3.3 Triển khai thay đổi an toàn

Trong hệ thống AI, những thứ sau đều là một lần deploy, và đều cần cùng quy trình như deploy code:

- Code ứng dụng.
- Prompt, system prompt, định nghĩa tool.
- Model hoặc snapshot model của provider.
- Phiên bản engine, mức lượng tử hoá, cấu hình engine ([Self-hosting LLM §3.4](../self-hosting-llm/#34-cập-nhật-model-và-rollback)).
- Embedding model, cách chunk, index ([§2.7](#27-vector-database-trong-production)).

Mỗi thứ có version, được ghi vào trace của từng request ([LLM Observability](../llm-observability/)), và đổi được về bản trước trong vài phút.

**Thứ tự an toàn cho một thay đổi:**

1. Eval offline trên bộ regression ([Evaluation §3.3](../evaluation/#33-regression-test-khi-đổi-prompt-hoặc-model)).
2. Shadow (tuỳ chọn): bản mới nhận bản sao traffic thật, kết quả không trả cho người dùng, chỉ dùng để so độ trễ, lỗi, điểm judge. Tốn gấp đôi chi phí model cho phần traffic được sao; tắt các tool có tác dụng phụ (gửi email, ghi DB) ở bản shadow.
3. Canary: một phần nhỏ traffic (1%, rồi 5%, 25%) sang bản mới. So tỉ lệ lỗi, TTFT, tỉ lệ bị cắt, refusal, điểm judge online giữa hai nhóm ([Evaluation §3.5](../evaluation/#35-eval-online-ab-test-feedback-người-dùng-sampling-log)). Chia theo user hoặc phiên, không theo request, để một hội thoại không nhảy qua lại giữa hai bản.
4. Chuyển toàn bộ, giữ bản cũ chạy thêm một thời gian để rollback không phải chờ khởi động.

**Đổi model tự host.** Chạy hai Deployment song song (bản cũ và bản mới, mỗi bản một `--served-model-name` hoặc một nhãn), chia traffic ở gateway. Cần GPU cho cả hai trong suốt thời gian chuyển; nếu không có, chuyển từng replica và chấp nhận mất công suất, hoặc thuê thêm GPU tạm. Rollback nhanh chỉ khi bản cũ còn đang chạy hoặc khởi động lại nhanh (weights và image còn trên node).

**Error budget quyết định tốc độ.** Còn budget thì ra bản mới theo lịch; hết budget thì dừng thay đổi không khẩn cấp cho tới khi ổn định lại [6].

## 3.4 Phụ thuộc vào API của provider

Nếu model chạy ở provider, phần lớn việc "deploy" là quản lý một phụ thuộc bên ngoài.

**Rate limit và quota.** OpenAI tính rate limit theo tổ chức và theo project, đo bằng request và token mỗi phút, mỗi ngày; vượt thì trả 429 [37]. Header `x-ratelimit-remaining-requests`, `x-ratelimit-remaining-tokens` cho biết còn bao nhiêu [37]. Claude trả các header `anthropic-ratelimit-*` và `retry-after`; có `retry-after` thì chờ đúng số giây đó [38]. Đọc các header này trong gateway và đưa vào metric: chạm 80% hạn mức là cảnh báo capacity, không phải lỗi bất ngờ. Chi tiết cách provider tính và vòng retry ở [Model routing §1.2](../model-routing/#12-rate-limit-rpm-tpm-và-cách-provider-tính).

**Capacity đặt trước.** Anthropic đã ngừng bán Priority Tier: tổ chức đang có hợp đồng dùng tiếp tới hết hạn, ai cần capacity bảo đảm thì liên hệ bộ phận bán hàng [39]. Lên kế hoạch tải giờ đỉnh với provider từ sớm, như xin quota GPU.

**Sự cố của provider.** Theo dõi status page (status.openai.com, status.claude.com cho phép đăng ký nhận email khi có sự cố [43]) và quan trọng hơn là tỉ lệ lỗi theo provider trong metric của chính bạn: status page thường cập nhật sau khi người dùng đã thấy lỗi. Fallback sang model hoặc provider khác ở [Model routing](../model-routing/); trước khi cần fallback, kiểm rằng request của bạn hợp lệ ở model dự phòng ([Model routing §1.5](../model-routing/#15-model-khác-nhau-trả-lời-khác-nhau-hệ-quả-khi-fallback)).

**Tách traffic theo độ gấp.** Traffic người dùng chờ và traffic offline (đánh giá, ingest, tóm tắt hàng loạt) không nên dùng chung hạn mức. Đưa traffic offline qua Batch API: OpenAI và Anthropic đều giảm 50%, kết quả trong 24 giờ, và Batch API của OpenAI có hạn mức riêng ([Cost optimization §2.3](../cost-optimization/#23-batch-api)).

## 3.5 Alert theo SLO cho tầng serving

Alert phía ứng dụng (tỉ lệ lỗi theo provider, chi phí theo giờ, stream bị cắt) ở [LLM Observability §3.7](../llm-observability/#37-alert-ngưỡng-nào-đáng-đánh-thức-người-trực). Phần này là alert cho SLO và cho pool model tự host.

**Burn rate.** Burn rate là tốc độ tiêu error budget so với tốc độ cho phép; SLO 99,9% trong 30 ngày với tỉ lệ lỗi không đổi 0,1% là burn rate 1, vừa hết budget đúng cuối kỳ [5]. Google SRE Workbook khuyên alert ở nhiều cửa sổ và nhiều mức burn rate, với bộ giá trị khởi điểm [5]:

| Budget đã tiêu | Cửa sổ dài | Burn rate | Thông báo |
|---|---|---|---|
| 2% | 1 giờ | 14,4 | Page |
| 5% | 6 giờ | 6 | Page |
| 10% | 3 ngày | 1 | Ticket |

Mỗi alert dùng thêm một cửa sổ ngắn để tắt nhanh khi sự cố đã qua; ví dụ trong Workbook: page khi burn rate vượt 14,4 trong cả 1 giờ qua lẫn 5 phút qua [5].

Ví dụ luật Prometheus cho SLO "99% request có TTFT dưới 2,5 giây" (budget 1%), tính từ histogram của vLLM. Đã kiểm bằng `promtool check rules` và `promtool test rules` của Prometheus 3.16.0 với ba ca: 20% request chậm suốt một giờ thì page; 5% chậm (burn rate 5) thì không; một giờ xấu rồi mười phút đã hồi phục thì không page, vì cửa sổ 5 phút đã về dưới ngưỡng dù cửa sổ 1 giờ còn cao.

```yaml
groups:
  - name: llm-slo
    rules:
      - record: llm:ttft_slow:ratio_rate5m
        expr: |
          1 - sum(rate(vllm:time_to_first_token_seconds_bucket{le="2.5"}[5m]))
            / sum(rate(vllm:time_to_first_token_seconds_count[5m]))
      - record: llm:ttft_slow:ratio_rate1h
        expr: |
          1 - sum(rate(vllm:time_to_first_token_seconds_bucket{le="2.5"}[1h]))
            / sum(rate(vllm:time_to_first_token_seconds_count[1h]))
      - alert: TTFTBudgetBurnFast
        # SLO 99% request có TTFT dưới 2,5 giây: budget 1%, burn rate 14,4
        expr: llm:ttft_slow:ratio_rate1h > 14.4 * 0.01 and llm:ttft_slow:ratio_rate5m > 14.4 * 0.01
        labels:
          severity: page
```

Đổi `le="2.5"` theo ngưỡng của bạn và chọn ngưỡng trùng biên bucket ([§1.4](#14-slo-và-error-budget-cho-dịch-vụ-llm)). Luật cho cửa sổ 6 giờ và 3 ngày viết tương tự. Histogram của vLLM không thấy thời gian request chờ ở gateway hay bị 503 từ chối; SLO chính thức nên tính ở gateway, luật trên dùng làm tín hiệu từ phía engine.

**Tín hiệu của pool model** (ticket hoặc kênh chat, chỉ page khi kéo theo SLO):

| Tín hiệu | Metric | Thường nghĩa là |
|---|---|---|
| Hàng đợi tăng đều nhiều phút | `vllm:num_requests_waiting` [4] | Thiếu replica, autoscaler không theo kịp hoặc đã chạm `maxReplicaCount`, quota |
| KV cache gần đầy, preemption tăng | `vllm:kv_cache_usage_perc`, `rate(vllm:num_preemptions_total[5m])` [4] | Request dài hơn trước, `max_num_seqs` quá cao so với bộ nhớ |
| Request bị từ chối | `rate(vllm:admission_rejections_total[5m])` theo `reason` [3] | Chạm `--max-num-queued-reqs`; quá tải thật |
| Cache hit giảm | `rate(vllm:prefix_cache_hits_total[1h]) / rate(vllm:prefix_cache_queries_total[1h])` [4] | Prompt đổi prefix sau deploy, routing không còn giữ phiên |
| Số replica ready thấp hơn mong muốn | `kube_deployment_status_replicas_available` của kube-state-metrics [44] | Pod crash loop, Pending vì hết GPU, startup probe quá ngắn |
| Scaler lỗi | Trạng thái ScaledObject, sự kiện của KEDA | Prometheus mất target, query sai |

## 3.6 Xử lý sự cố

**Vai trò.** Khi sự cố lớn, một người làm incident commander: điều phối, phân việc, quyết định, và theo hướng dẫn của PagerDuty thì không tự sửa, không tự xem graph hay log [42]. Người khác lo xử lý kỹ thuật và liên lạc. Một người vừa debug vừa trả lời quản lý vừa cập nhật status page sẽ làm cả ba chậm.

**Việc đầu tiên theo loại sự cố:**

| Triệu chứng | Kiểm tra trước | Giảm thiệt hại ngay |
|---|---|---|
| Lỗi tăng ở một provider | Tỉ lệ lỗi theo provider và model, status page | Chuyển sang fallback; giảm traffic không gấp |
| TTFT tăng ở pool tự host | Hàng đợi, số replica ready, KV cache, deploy gần nhất | Tăng `minReplicaCount`, tạm dừng traffic offline, bật trả lỗi nhanh |
| Chất lượng giảm sau deploy | Version nào vừa đổi (prompt, model, engine, index) | Rollback version đó, không chờ hiểu hết nguyên nhân |
| Chi phí tăng vọt | Chi phí theo feature, user, tenant | Hạn mức theo tenant, tắt feature nghi ngờ ([Cost optimization §3.5](../cost-optimization/#35-giới-hạn-chi-phí-theo-user-và-chống-lạm-dụng)) |
| Retrieval chậm hoặc sai | Độ trễ vector DB, filter, alias đang trỏ đâu | Chuyển alias về collection cũ |
| Pod restart liên tục | Liveness probe, OOM, log engine | Nới probe, giảm `max_num_seqs` hoặc context |

Nguyên tắc chung: giảm thiệt hại trước, tìm nguyên nhân sau. Rollback một thay đổi vừa làm gần như luôn là bước đầu đúng.

**Postmortem.** Sau sự cố, viết postmortem không đổ lỗi: tìm các nguyên nhân góp phần mà không quy trách nhiệm cho cá nhân hay nhóm, giả định mọi người đã làm đúng với thông tin họ có lúc đó [41]. Mỗi postmortem kết thúc bằng việc cụ thể: alert còn thiếu, giới hạn còn thiếu, bước runbook cần thêm.

## 3.7 Chi phí vận hành

Chi phí của pool GPU tự host gần như là số replica × số giờ chạy, bất kể có traffic hay không. Ba thứ quyết định:

- **Mức sử dụng trung bình.** Pool tính cho giờ đỉnh cộng headroom cộng dự phòng sẽ rảnh phần lớn thời gian ngoài giờ đỉnh. Autoscaling theo tải thu hẹp khoảng cách, nhưng `minReplicaCount` (cần cho thời gian khởi động và N+1) là sàn.
- **Tải offline lấp chỗ trống.** Chạy embedding lại, eval, tóm tắt hàng loạt vào giờ thấp điểm trên chính pool đó, với độ ưu tiên thấp hơn traffic người dùng, hoặc gửi qua Batch API của provider.
- **So với API.** Công thức so chi phí tự host với API ở [Cost optimization §2.5](../cost-optimization/#25-self-host-khi-nào-rẻ-hơn-api) và [Self-hosting LLM §3.5](../self-hosting-llm/#35-so-sánh-tổng-chi-phí-với-api). Tính cả chi phí vận hành: người trực, giám sát, nâng cấp.

Gắn nhãn chi phí (team, feature, môi trường) cho node pool và namespace để chi phí GPU hiện ra theo feature, giống cách tính chi phí theo request ở [Cost optimization §1.2](../cost-optimization/#12-đo-trước-khi-tối-ưu-chi-phí-theo-request-user-feature).

## 3.8 Failure modes thường gặp

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| Mỗi lần deploy có một loạt stream bị cắt | `--shutdown-timeout` mặc định 0; `terminationGracePeriodSeconds` 30 giây [3][12] | `preStop` sleep, `--shutdown-timeout` theo p99, grace period lớn hơn tổng ([§2.4](#24-probe-rolling-update-và-graceful-shutdown)) |
| Pod restart mãi, không bao giờ ready | Liveness chạy trước khi model nạp xong | Startup probe với `failureThreshold × periodSeconds` dài hơn thời gian khởi động |
| Rollout treo, pod mới Pending | `maxSurge` cần GPU trống, cluster hết GPU | `maxSurge: 0, maxUnavailable: 1` hoặc giữ một GPU trống; tính vào capacity |
| Load test đạt, production chậm | Load test closed-loop; prompt và output ngắn, đều nhau | Tải open-loop với phân phối độ dài thật ([§3.1](#31-đo-công-suất-một-replica-bằng-tải-open-loop)) |
| TTFT tăng tới hàng phút khi tải cao, không có lỗi | Hàng đợi không giới hạn [3] | `--max-num-queued-reqs`; trả 503 và thử replica khác |
| Mỗi lần deploy, số pod rớt về một con số cố định rồi scale lại | Deployment còn khai `spec.replicas` trong khi KEDA/HPA quản lý [20] | Bỏ `spec.replicas` theo hướng dẫn chuyển đổi của HPA |
| Số replica dao động lên xuống dù tải đều | Scale chỉ theo `num_requests_waiting`: đủ công suất thì hàng đợi về 0 và pool bị gỡ | Scale theo tổng request trong hệ thống ([§2.5](#25-autoscaling)) |
| Autoscaler không thêm pod dù quá tải | Scale theo CPU hoặc bộ nhớ; scaler mất metric | Scale theo số request trong hệ thống (`num_requests_running` + `num_requests_waiting`); `ignoreNullValues: "false"` và `fallback` [19] |
| Scale lên đúng nhưng vẫn vi phạm SLO vài phút | Thời gian khởi động dài hơn thời gian tải tăng | Headroom, `minReplicaCount` cao hơn, rút ngắn khởi động ([§2.2](#22-container-image-và-nạp-model)) |
| Không scale được vì Hugging Face chậm hoặc sập | Pod tải weights lúc khởi động | Weights trên volume hoặc OCI artifact, `HF_HUB_OFFLINE=1` |
| Hai model chung GPU cùng chết | Time-slicing không cô lập bộ nhớ và lỗi [8] | MIG, hoặc một model một GPU |
| Retrieval trả ít kết quả hơn `k` khi có filter | pgvector lọc sau khi quét index với `ef_search` 40 [28] | Iterative scan; payload index trong Qdrant [32] |
| Recall giảm sau khi đổi embedding model | Trộn vector của hai model trong một collection [34] | Collection mới, eval, chuyển alias |
| Prefix cache hit thấp dù prompt có phần chung dài | Round-robin rải phiên ra nhiều replica | Session affinity hoặc routing biết KV cache ([§2.6](#26-routing-dành-cho-llm-inferencepool-llm-d-dynamo-kserve)) |
| Không thêm được máy GPU lúc cần | Quota GPU mặc định 0 hoặc thấp [40] | Xin quota trước, cho mọi region dùng tới |
| Backup có nhưng khôi phục mất nửa ngày | Chưa từng thử khôi phục; phải dựng lại index | Thử khôi phục định kỳ, đo thời gian |

## 3.9 Checklist

Trước khi ra mắt:

- [ ] Đã đo công suất một replica bằng tải open-loop với phân phối prompt và output thật; biết tốc độ đến lớn nhất còn đạt SLO.
- [ ] Đã tính số replica giờ đỉnh với headroom và N+1; quota GPU và rate limit provider đủ cho con số đó.
- [ ] SLO viết ra: chỉ số, ngưỡng, cửa sổ; ngưỡng trùng biên bucket của histogram.
- [ ] Đã đo thời gian khởi động một replica mới, từng bước.
- [ ] Weights, image, engine ghim phiên bản; pod không gọi ra Internet để khởi động.

Kubernetes:

- [ ] GPU khai trong `limits`; không dùng time-slicing cho production nhiều tenant.
- [ ] Startup probe dài hơn thời gian khởi động tệ nhất; liveness và readiness có ngưỡng hợp lý.
- [ ] `preStop`, `--shutdown-timeout`, `terminationGracePeriodSeconds` khớp nhau; đã thử dừng pod giữa lúc đang stream.
- [ ] Chiến lược rolling update tính đến GPU trống; có PodDisruptionBudget.
- [ ] Hàng đợi của engine có giới hạn; quá giới hạn thì trả 503.
- [ ] Autoscaling theo tổng số request trong hệ thống, threshold lấy từ số đo; scaler báo lỗi khi mất metric và có fallback.
- [ ] Deployment không khai `spec.replicas` khi đã có HPA hoặc KEDA.

Dữ liệu:

- [ ] Vector DB đã tính bộ nhớ cho số vector hiện tại và dự kiến một năm; đã kiểm recall nếu dùng quantization.
- [ ] Truy vấn có filter đã test với filter thật; index cho các trường lọc.
- [ ] Có alias để chuyển collection; đã thử khôi phục từ backup và biết mất bao lâu.

Vận hành:

- [ ] Alert burn rate cho SLO; alert cho hàng đợi, KV cache, replica ready, scaler lỗi.
- [ ] Prompt, model, engine, index đều có version, ghi trong trace, rollback được trong vài phút.
- [ ] Thay đổi đi qua eval offline rồi canary.
- [ ] Có runbook cho sự cố provider, quá tải pool tự host, deploy làm giảm chất lượng; có người trực biết runbook.
- [ ] Traffic offline tách khỏi traffic người dùng.

---

## Nguồn tham khảo

Truy cập tháng 11/2026.

| # | Nguồn | Nội dung dùng trong bài |
|---|---|---|
| [1] | [giltene/wrk2, README](https://github.com/giltene/wrk2) | Coordinated omission trong load test closed-loop |
| [2] | [Wikipedia: Little's law](https://en.wikipedia.org/wiki/Little%27s_law) | L = λW cho hệ thống ổn định |
| [3] | vLLM source tại tag `v0.33.0`: [`vllm/config/scheduler.py`](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/scheduler.py), [`vllm/config/vllm.py`](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/vllm.py), [`vllm/engine/arg_utils.py`](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py), [`vllm/entrypoints/launchers/`](https://github.com/vllm-project/vllm/tree/v0.33.0/vllm/entrypoints/launchers), [`vllm/entrypoints/serve/instrumentator/health.py`](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/serve/instrumentator/health.py), [`vllm/v1/metrics/loggers.py`](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/metrics/loggers.py), [`vllm/v1/metrics/buckets.py`](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/metrics/buckets.py) | `max_num_queued_reqs`, `max_num_queued_tokens`, `--shutdown-timeout` (0 = huỷ), thứ tự khởi động server, `/health` trả 503 khi engine chết, nhãn `model_name`, `vllm:request_success` theo `finished_reason`, `vllm:admission_rejections`, bucket của TTFT |
| [4] | [vLLM: Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/) | Tên metric `vllm:*` |
| [5] | [Google SRE Workbook: Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/) | Burn rate, bảng cửa sổ và mức burn rate, alert nhiều cửa sổ |
| [6] | [Google SRE Book: Embracing Risk](https://sre.google/sre-book/embracing-risk/) | Error budget quyết định khi nào được deploy |
| [7] | [Kubernetes: Schedule GPUs](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/) | Hỗ trợ GPU qua device plugin, GPU chỉ khai trong `limits` |
| [8] | [NVIDIA k8s-device-plugin v0.20.1, README](https://github.com/NVIDIA/k8s-device-plugin/blob/v0.20.1/README.md) | Time-slicing, MPS, chiến lược MIG, mức cô lập |
| [9] | [NVIDIA GPU Operator v26.7.2, README](https://github.com/NVIDIA/gpu-operator/blob/v26.7.2/README.md) và [GPU sharing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html) | Thành phần GPU Operator, DRA trong kế hoạch, MIG cô lập bộ nhớ và lỗi |
| [10] | [Kubernetes: Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/) và [`pkg/features/kube_features.go` tại v1.37.2](https://github.com/kubernetes/kubernetes/blob/v1.37.2/pkg/features/kube_features.go) | Trạng thái DRA; GA ở 1.34 theo source |
| [11] | [kubernetes-sigs/dra-driver-nvidia-gpu v0.5.1, README](https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu/blob/v0.5.1/README.md) | Phần cấp GPU chưa được hỗ trợ chính thức |
| [12] | [Kubernetes: Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/) và [Container Lifecycle Hooks](https://kubernetes.io/docs/concepts/containers/container-lifecycle-hooks/) | `terminationGracePeriodSeconds` mặc định 30 giây, `preStop` chạy xong mới gửi SIGTERM |
| [13] | [Kubernetes: Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) và [Disruptions](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/) | Mặc định 25% max unavailable, 25% max surge; PDB và giới hạn của nó |
| [14] | [Kubernetes: Liveness, Readiness, Startup Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/); llm-d v1.0.0: [Graceful Shutdown](https://github.com/llm-d/llm-d/blob/v1.0.0/docs/operations/lifecycle/graceful-shutdown.md), [Readiness Probes](https://github.com/llm-d/llm-d/blob/v1.0.0/docs/operations/lifecycle/readiness-probes.md) | Startup probe chặn liveness và readiness; thứ tự dừng pod; công thức grace period; readiness bằng `/v1/models`; EPP trả 503 khi drain; `failureMode` |
| [15] | [Kubernetes: Volumes, `image`](https://kubernetes.io/docs/concepts/storage/volumes/) và `kube_features.go` tại v1.37.2 | `ImageVolume` GA ở 1.36, `ImageVolumeWithDigest` alpha |
| [16] | [vLLM: Run:ai Model Streamer](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/models/extensions/runai_model_streamer.md) | `--load-format runai_streamer`, `vllm[runai]`, đọc từ S3 |
| [17] | [vLLM: Sleep Mode](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/features/sleep_mode.md) | Level 1, `/sleep`, `/wake_up`; cần `--enable-sleep-mode` và `VLLM_SERVER_DEV_MODE=1`; endpoint dev không được mở cho người dùng |
| [18] | [vLLM: Using Docker](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/deployment/docker.md), [Using Kubernetes](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/deployment/k8s.md), [Docker Hub `vllm/vllm-openai:v0.33.0`](https://hub.docker.com/r/vllm/vllm-openai/tags) | Cache compile, user không root, `/dev/shm`, kích thước image 9,7 GB (amd64, nén) |
| [19] | KEDA 2.21: [ScaledObject specification](https://keda.sh/docs/2.21/reference/scaledobject-spec/), [Prometheus scaler](https://keda.sh/docs/2.21/scalers/prometheus/), [Scaling Deployments](https://keda.sh/docs/2.21/concepts/scaling-deployments/) | `AverageValue`, `threshold`, `activationThreshold`, `ignoreNullValues`, `fallback`, scale 0 ↔ 1 |
| [20] | [Kubernetes: Horizontal Pod Autoscaling](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/) | Stabilization 5 phút khi giảm, tolerance 10%, nhiều metric thì lấy đề xuất lớn nhất, `HPAScaleToZero`, bỏ `spec.replicas` khi dùng HPA |
| [21] | [vLLM production-stack vllm-stack-0.1.14, README](https://github.com/vllm-project/production-stack/blob/vllm-stack-0.1.14/README.md) và [Autoscaling with KEDA](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html) | Các thuật toán routing, roadmap; scale theo `vllm:num_requests_waiting` |
| [22] | [Gateway API Inference Extension v1.6.2, README](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/v1.6.2/README.md) và [API overview](https://gateway-api-inference-extension.sigs.k8s.io/concepts/api-overview/) | `InferencePool`, EPP, việc chuyển EPP sang llm-d-router |
| [23] | [llm-d v1.0.0, README](https://github.com/llm-d/llm-d/blob/v1.0.0/README.md) và [Workload Variant Autoscaler (deprecated)](https://github.com/llm-d/llm-d/blob/v1.0.0/docs/architecture/autoscaling/wva.md) | Tính năng, số liệu tự công bố, KEDA với metric EPP |
| [24] | [llm-d/llm-d-router v1.0.0](https://github.com/llm-d/llm-d-router/tree/v1.0.0) | EPP, `InferenceObjective` (`llm-d.ai`) |
| [25] | [NVIDIA Dynamo v1.6.2, README](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/README.md) | Routing theo KV cache, tách prefill/decode, Planner; số liệu tự công bố |
| [26] | [KServe v0.21.0, README](https://github.com/kserve/kserve/blob/v0.21.0/README.md) | Backend vLLM, llm-d, offload KV cache, `LLMInferenceService` |
| [27] | [Knative Serving `config-autoscaler` tại knative-v1.24.0](https://github.com/knative/serving/blob/knative-v1.24.0/config/core/configmaps/autoscaler.yaml) | `enable-scale-to-zero` |
| [28] | [pgvector v0.8.7, README](https://github.com/pgvector/pgvector/blob/v0.8.7/README.md) | HNSW, `maintenance_work_mem`, worker song song, `CREATE INDEX CONCURRENTLY`, lọc sau khi quét, iterative scan, kích thước vector |
| [29] | [Qdrant: Collections, aliases](https://qdrant.tech/documentation/concepts/collections/) | Đổi alias nguyên tử |
| [30] | [Qdrant: Snapshots](https://qdrant.tech/documentation/snapshots/) | Snapshot theo node, snapshot toàn bộ storage chỉ cho một node |
| [31] | [Qdrant: Quantization](https://qdrant.tech/documentation/guides/quantization/) | Scalar 4 lần, binary 32 lần, product tới 64 lần; kiểm recall |
| [32] | [Qdrant: Indexing](https://qdrant.tech/documentation/concepts/indexing/) | Payload index cho trường lọc, tạo trước khi nạp dữ liệu |
| [33] | [Qdrant: Capacity Planning](https://qdrant.tech/documentation/capacity-planning/) | Công thức bộ nhớ vector và HNSW, SSD 50.000 IOPS, giữ HNSW trong bộ nhớ |
| [34] | [Qdrant: Incremental Embedding Updates](https://qdrant.tech/documentation/tutorials-operations/incremental-embedding-updates/) | Không trộn vector của nhiều model; lưu model và pipeline trong metadata |
| [35] | [Weaviate: Collection aliases](https://docs.weaviate.io/weaviate/manage-collections/collection-aliases) | Alias từ v1.32, chuyển nguyên tử |
| [36] | [Weaviate: Backups](https://docs.weaviate.io/weaviate/configuration/backups) | S3, GCS, Azure; backup tăng dần từ v1.37; filesystem chỉ cho dev |
| [37] | [OpenAI: Rate limits](https://developers.openai.com/api/docs/guides/rate-limits) | Theo tổ chức và project, 429, header `x-ratelimit-*` |
| [38] | [Claude API: Rate limits](https://docs.claude.com/en/api/rate-limits) | Header `anthropic-ratelimit-*`, `retry-after` |
| [39] | [Claude API: Service tiers](https://platform.claude.com/docs/en/api/service-tiers) | Priority Tier không còn bán |
| [40] | [AWS: EC2 instance type quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html) và [Service quotas](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html) | Quota mặc định 0 cho G, VT, P; xin trước |
| [41] | [Google SRE Book: Postmortem Culture](https://sre.google/sre-book/postmortem-culture/) | Postmortem không đổ lỗi |
| [42] | [PagerDuty Incident Response: Incident Commander](https://response.pagerduty.com/training/incident_commander/) | Vai trò incident commander |
| [43] | [Claude Status](https://status.claude.com/), [OpenAI Status](https://status.openai.com/) | Status page, đăng ký nhận thông báo sự cố |
| [44] | [kube-state-metrics: Deployment metrics](https://github.com/kubernetes/kube-state-metrics/blob/main/docs/metrics/workload/deployment-metrics.md) | `kube_deployment_status_replicas_available` |
