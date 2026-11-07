# Self-hosting LLM: chạy model open-weight

Một nhóm cài Ollama lên máy chủ GPU của công ty để làm chatbot nội bộ. Một người thử thì trả lời nhanh, ổn. Đến buổi demo có tám người hỏi cùng lúc, người cuối phải đợi gần nửa phút mới thấy chữ đầu tiên. GPU không quá tải, model cũng không chậm. Lý do là mặc định của Ollama chỉ xử lý một request mỗi lần cho mỗi model (`OLLAMA_NUM_PARALLEL=1`) [16], nên bảy request còn lại xếp hàng. Thí nghiệm ở [§3.2](#32-benchmark-throughput-và-độ-trễ-trên-tải-thật) tái hiện cảnh này: giữ tám request đồng thời trên một server một slot, mỗi request phải chờ trung vị 28,2 giây mới có token đầu.

Tự host model open-weight đổi giá theo token của API lấy ba thứ phải tự lo: bộ nhớ GPU (weights và KV cache), inference engine với các mặc định của nó, và vận hành (đo tải, cập nhật, giám sát). Bài này đi qua cả ba, với số đo thật khi có thể. Phần triển khai hệ thống quanh model (container, autoscaling, load balancing, capacity planning) nằm ở [Deploying AI Systems](../deploying-ai-systems/).

- **Nền tảng — Phần 1:** khi nào nên tự host, bộ nhớ cho weights và KV cache, prefill và decode, quantization, license.
- **Inference engine và phần cứng — Phần 2:** vLLM, SGLang, TensorRT-LLM, TGI, llama.cpp, Ollama; GPU; thuê GPU và dịch vụ serverless.
- **Thiết kế và vận hành — Phần 3:** chọn model theo phần cứng kèm code tính bộ nhớ, benchmark kèm code đo, tool calling và structured output, cập nhật và rollback, so chi phí với API, failure modes và checklist.

> Bản tháng 11/2026. Số đo trong bài chạy trên Apple M4 (10 nhân GPU, 24 GB bộ nhớ hợp nhất, băng thông 120 GB/s [26]) với llama.cpp 0.9.0 (build 12110, cài qua Homebrew) và `Qwen2.5-3B-Instruct` ở định dạng GGUF; số tuyệt đối trên GPU máy chủ sẽ khác nhiều, nhưng các hiện tượng (giới hạn băng thông, đánh đổi giữa batch và độ trễ, mất chất lượng khi lượng tử hoá quá sâu) là cùng một cơ chế. Phiên bản engine, giá GPU và license thay đổi thường xuyên; đối chiếu nguồn trước khi dùng. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

Tài liệu liên quan: [Latency](../latency/) (TTFT, prefill, decode, speculative decoding, chunked prefill), [Cost optimization §2.5](../cost-optimization/#25-self-host-khi-nào-rẻ-hơn-api) (công thức so chi phí), [Fine-tuning §3.5](../fine-tuning/#35-deploy-và-serving-lora-adapter) (serving LoRA adapter trên vLLM), [Streaming §2.6](../streaming/#26-self-hosted-vllm-sglang-ollama) (event stream của engine tự host), [Model routing](../model-routing/) (fallback giữa model tự host và API), [Deploying AI Systems](../deploying-ai-systems/).

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Khi nào nên tự host, khi nào không](#11-khi-nào-nên-tự-host-khi-nào-không)
  - [1.2 Bộ nhớ cần cho weights và KV cache](#12-bộ-nhớ-cần-cho-weights-và-kv-cache)
  - [1.3 Prefill, decode và vì sao batching quan trọng](#13-prefill-decode-và-vì-sao-batching-quan-trọng)
  - [1.4 Quantization: FP8, INT4, GGUF, AWQ, GPTQ](#14-quantization-fp8-int4-gguf-awq-gptq)
  - [1.5 License của model open-weight](#15-license-của-model-open-weight)
- [Phần 2 — Inference engine và phần cứng](#phần-2--inference-engine-và-phần-cứng)
  - [2.1 vLLM](#21-vllm)
  - [2.2 SGLang](#22-sglang)
  - [2.3 TensorRT-LLM và TGI](#23-tensorrt-llm-và-tgi)
  - [2.4 llama.cpp và Ollama](#24-llamacpp-và-ollama)
  - [2.5 Chọn GPU: NVIDIA, AMD, Apple Silicon](#25-chọn-gpu-nvidia-amd-apple-silicon)
  - [2.6 Cloud GPU và dịch vụ inference serverless](#26-cloud-gpu-và-dịch-vụ-inference-serverless)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Chọn model theo tác vụ và phần cứng](#31-chọn-model-theo-tác-vụ-và-phần-cứng)
  - [3.2 Benchmark throughput và độ trễ trên tải thật](#32-benchmark-throughput-và-độ-trễ-trên-tải-thật)
  - [3.3 Tool calling và structured output trên model tự host](#33-tool-calling-và-structured-output-trên-model-tự-host)
  - [3.4 Cập nhật model và rollback](#34-cập-nhật-model-và-rollback)
  - [3.5 So sánh tổng chi phí với API](#35-so-sánh-tổng-chi-phí-với-api)
  - [3.6 Failure modes thường gặp](#36-failure-modes-thường-gặp)
  - [3.7 Checklist](#37-checklist)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

### Bộ nhớ: hai công thức

```
weights (byte)          = số tham số × số bit mỗi tham số / 8
KV cache mỗi token (byte) = 2 (K và V) × số lớp × số KV head × head_dim × byte mỗi phần tử
```

Số lớp, số KV head và `head_dim` đọc trong `config.json` của model. Lớp sliding window chỉ giữ một số token cuối cố định ([§1.2](#12-bộ-nhớ-cần-cho-weights-và-kv-cache)).

| Model | Tham số [43] | Weights BF16 / FP8 / INT4 (GiB) | KV mỗi token, BF16 |
|---|---|---|---|
| Qwen2.5-3B-Instruct | 3,09 tỉ | 5,7 / 2,9 / 1,4 | 36 KiB |
| Llama 3.1 8B Instruct | 8,03 tỉ | 15,0 / 7,5 / 3,7 | 128 KiB |
| Qwen3-8B | 8,19 tỉ | 15,3 / 7,6 / 3,8 | 144 KiB |
| Qwen3-32B | 32,8 tỉ | 61,0 / 30,5 / 15,3 | 256 KiB |
| Llama 3.3 70B Instruct | 70,6 tỉ | 131,4 / 65,7 / 32,9 | 320 KiB |
| gpt-oss-20b | 20,9 tỉ | checkpoint MXFP4: 12,8 | 48 KiB (nửa số lớp sliding 128 token) |
| gpt-oss-120b | 116,8 tỉ | checkpoint MXFP4: 60,8 | 72 KiB (nửa số lớp sliding 128 token) |

Cột INT4 chưa tính scale và các lớp giữ độ chính xác cao hơn; file thật lớn hơn vài phần trăm tới vài chục phần trăm (Qwen2.5-3B Q4_K_M là 1,96 GiB). Một chuỗi 32.768 token của Llama 3.3 70B chiếm 10 GiB KV cache, khoảng 15% weights FP8 của nó.

### Inference engine, tháng 11/2026

| Engine | Bản mới nhất | Hợp khi | Lưu ý |
|---|---|---|---|
| vLLM | 0.33.0 [8] | Serving production trên GPU NVIDIA, AMD; nhiều model, nhiều tính năng | Mặc định chiếm 92% bộ nhớ GPU ([§2.1](#21-vllm)) |
| SGLang | 0.5.23 [11] | Tải có nhiều prefix chung (agent, chat nhiều lượt, few-shot); structured output | Metrics tắt mặc định |
| TensorRT-LLM | 1.2.1 stable, 1.4.0rc3 pre-release [12] | Tối ưu sâu trên GPU NVIDIA Hopper, Blackwell | Từ 1.3.0rc21 và cả dòng 1.4.0rc bỏ backend TensorRT và `trtllm-build`; 1.2.1 vẫn còn |
| TGI | 3.3.7 | Không chọn cho dự án mới | Đang ở chế độ bảo trì [13] |
| llama.cpp (`llama-server`) | b12180 [14] | Máy không có GPU NVIDIA, Apple Silicon, CPU, thiết bị biên | `-c` chia đều cho các slot ([§2.4](#24-llamacpp-và-ollama)) |
| Ollama | 0.40.15 [16] | Máy cá nhân, thử nghiệm, một người dùng | Một request mỗi lần cho mỗi model |

### Mặc định dễ gây sự cố

| Engine | Mặc định | Hệ quả |
|---|---|---|
| vLLM | `--gpu-memory-utilization 0.92` [2] | Chạy thêm tiến trình khác trên cùng GPU thì hết bộ nhớ khi khởi động |
| vLLM | `--max-num-seqs`, `--max-num-batched-tokens` đổi theo GPU: 1024 / 8192 trên H100, H200; 256 / 2048 trên L4, L40S, A100 [2] | Cùng một lệnh cho throughput và độ trễ khác nhau trên hai loại GPU |
| vLLM | `--max-model-len` lấy từ config của model [2] | Context dài nhất của model có thể không vừa KV cache; vLLM từ chối khởi động hoặc phải giảm |
| SGLang | `--enable-metrics` tắt [11] | Không có `/metrics` cho Prometheus |
| llama-server | `-c` là tổng cho mọi slot | `-c 2048 -np 2` nghĩa là mỗi request chỉ có 1.024 token, request dài hơn bị trả lỗi 400 |
| Ollama | `OLLAMA_NUM_PARALLEL=1` [16] | Request đồng thời xếp hàng |
| Ollama | Context 4k khi VRAM dưới 24 GiB [16] | Web search, agent, coding cần ít nhất 64.000 token theo chính tài liệu Ollama |
| Ollama | `keep_alive` 5 phút [16] | Request đầu sau 5 phút rảnh phải nạp lại model |

---

# Phần 1 — Nền tảng

## 1.1 Khi nào nên tự host, khi nào không

Lý do hợp lệ để tự host:

- **Dữ liệu không được rời hạ tầng**, do hợp đồng, luật, hoặc chính sách nội bộ, và không có API nào thoả (kể cả các bản triển khai trong cloud của bạn qua Bedrock, Vertex, Azure).
- **Traffic lớn và đều.** GPU tính tiền theo giờ dù rảnh; API tính theo token. Self-host chỉ rẻ hơn khi GPU bận phần lớn thời gian ([§3.5](#35-so-sánh-tổng-chi-phí-với-api)).
- **Model tuỳ biến**: model đã fine-tune, LoRA adapter riêng cho từng khách hàng ([Fine-tuning §3.5](../fine-tuning/#35-deploy-và-serving-lora-adapter)), hoặc model nhỏ chuyên một tác vụ.
- **Kiểm soát**: chạy offline, cố định đúng phiên bản model không bị nhà cung cấp đổi hay ngừng, đặt máy gần người dùng.

Lý do không đủ:

- **"API đắt."** Với model open-weight, các dịch vụ serverless bán theo token ở giá thấp ([§2.6](#26-cloud-gpu-và-dịch-vụ-inference-serverless)): gpt-oss-120b 0,15 USD mỗi triệu token input, 0,60 USD output trên Together và Fireworks [30][31]. Tự host phải rẻ hơn mức này, không phải rẻ hơn model frontier.
- **"Model open-weight tốt bằng."** Đo trên tác vụ của bạn ([Evaluation](../evaluation/)). Claude, GPT và Gemini không có bản open-weight; tự host nghĩa là đổi model.
- **Traffic thấp hoặc thất thường.** Một GPU H100 thuê theo giờ khoảng 4 USD [27] chạy cả tháng là gần 2.900 USD, bất kể có ai dùng.

Phần vận hành thường bị đánh giá thấp: nâng cấp engine và driver, theo dõi bộ nhớ và hàng đợi, xử lý sự cố lúc nửa đêm, dự phòng khi một máy chết. Một server không phải là một hệ thống; [Deploying AI Systems](../deploying-ai-systems/) đi qua phần còn lại.

## 1.2 Bộ nhớ cần cho weights và KV cache

Khi serving, bộ nhớ GPU chia cho ba phần: weights, KV cache, và phần còn lại (activation, buffer của engine, CUDA graph). Weights cố định; KV cache tăng theo số token đang xử lý và quyết định bao nhiêu request chạy được cùng lúc.

### Weights

Weights bằng số tham số nhân số byte mỗi tham số. Llama 3.3 70B có 70,55 tỉ tham số [43]: 131,4 GiB ở BF16, 65,7 GiB ở FP8. Không vừa một GPU 80 GB ở BF16; vừa ở FP8 nhưng gần như không còn chỗ cho KV cache ([§3.1](#31-chọn-model-theo-tác-vụ-và-phần-cứng)).

Với model MoE như gpt-oss, mọi expert đều phải nằm trong bộ nhớ dù mỗi token chỉ dùng vài expert. gpt-oss-120b có 128 expert mỗi lớp và dùng 4 cho mỗi token [43]; checkpoint MXFP4 của nó nặng 60,8 GiB. MoE giảm tính toán mỗi token, không giảm bộ nhớ weights.

### KV cache

Mỗi token đã xử lý để lại một vector key và một vector value ở mỗi lớp attention, cho từng KV head. Model hiện đại dùng grouped-query attention (GQA): nhiều query head dùng chung một KV head, nên số KV head nhỏ hơn số attention head nhiều. Qwen2.5-3B có 16 attention head nhưng chỉ 2 KV head:

```
2 × 36 lớp × 2 KV head × 128 (head_dim) × 2 byte (BF16) = 36.864 byte = 36 KiB mỗi token
```

llama.cpp in đúng con số này khi khởi động: `-c 4096` cấp 144,00 MiB KV cache (36 KiB × 4.096), K 72 MiB và V 72 MiB.

Hai điểm trong `config.json` làm công thức sai nếu đọc vội:

- **Sliding window.** gpt-oss xen kẽ lớp `sliding_attention` (chỉ nhìn 128 token gần nhất) với lớp `full_attention` [43]. Lớp sliding chỉ cần KV cho 128 token, nên KV cache của gpt-oss-120b ở 32.768 token chỉ khoảng 1,1 GiB, so với 10 GiB của Llama 3.3 70B.
- **Khai báo nhưng không dùng.** `config.json` của Qwen2.5-3B có `"sliding_window": 32768` nhưng `"use_sliding_window": false`. Mọi lớp là full attention.
- **Khai báo gián tiếp.** Gemma 3 không có danh sách lớp mà ghi `"sliding_window_pattern": 6`: cứ sáu lớp có một lớp full attention. Gemma 2 xen kẽ sliding và full nhưng `config.json` không ghi gì; quy tắc nằm trong code của model. Qwen3.5 có lớp `linear_attention` giữ state cố định, không có KV cache tăng theo token. Model đa phương thức (Gemma 3, Qwen3.5) để các tham số này trong `text_config`.

Hàm tính ở [§3.1](#31-chọn-model-theo-tác-vụ-và-phần-cứng) xử lý các trường hợp trên; khi config không đủ thông tin (Gemma 2, Mistral 7B v0.1), hàm coi mọi lớp là full attention, ước lượng dư chứ không thiếu.

**Lượng tử hoá KV cache.** Lưu KV ở 8 bit giảm gần một nửa: cùng cấu hình 4 slot × 4.096 token, llama.cpp cấp 576 MiB với KV F16 và 306 MiB với `-ctk q8_0 -ctv q8_0` (Q8_0 tốn 8,5 bit mỗi phần tử vì kèm scale). vLLM có `--kv-cache-dtype` với các giá trị `fp8`, `fp8_e4m3`, `fp8_e5m2`, mặc định `auto` theo dtype của model [2]. KV 8 bit có thể làm giảm chất lượng; đo trên eval của bạn trước khi bật.

**PagedAttention.** KV cache của mỗi request tăng dần và không biết trước độ dài. Cấp sẵn một vùng liên tục cho độ dài tối đa thì phí phần lớn. vLLM chia KV cache thành các block cố định như trang bộ nhớ ảo, cấp theo nhu cầu, gần như không phí, và cho phép các request dùng chung block khi có chung prefix; nhờ batch được nhiều hơn, throughput tăng 2–4 lần ở cùng độ trễ so với FasterTransformer và Orca [1]. Hệ quả thực tế: số request đồng thời phụ thuộc tổng token đang có, không phải `max_model_len` × số request.

## 1.3 Prefill, decode và vì sao batching quan trọng

Mỗi request qua hai giai đoạn [17]:

- **Prefill** xử lý toàn bộ prompt trong một lượt và sinh token đầu tiên. Nhiều token xử lý song song nên prefill dùng hết năng lực tính toán của GPU. Prefill quyết định TTFT.
- **Decode** sinh từng token một. Mỗi bước chỉ xử lý một token mỗi request nhưng phải đọc toàn bộ weights từ bộ nhớ. Decode bị giới hạn bởi băng thông bộ nhớ, không phải tính toán.

Từ đó có một giới hạn trên dễ tính cho tốc độ decode của một request đơn lẻ:

```
token/giây (một request) ≤ băng thông bộ nhớ / số byte weights đọc mỗi token
```

Đo trên M4 (120 GB/s [26]), Qwen2.5-3B, một request, 128 token output:

| File GGUF | Dung lượng | Giới hạn trên (120 GB/s ÷ dung lượng) | Đo được | Tỉ lệ |
|---|---|---|---|---|
| Q2_K | 1,38 GB | 87 token/giây | 48,9 | 56% |
| Q4_K_M | 2,10 GB | 57 token/giây | 44,0 | 77% |
| Q8_0 | 3,62 GB | 33 token/giây | 25,8 | 78% |

Ba hệ quả:

- **Weights nhỏ hơn thì decode nhanh hơn**, gần tỉ lệ thuận. Đây là lý do lượng tử hoá tăng tốc ngay cả khi GPU có đủ bộ nhớ.
- **GPU có băng thông cao decode nhanh hơn**, bất kể TFLOPS. H200 (4,8 TB/s) và H100 SXM (3,35 TB/s) cùng thế hệ tính toán, khác băng thông [24].
- **Một request không dùng hết GPU.** Trong một bước decode, đọc weights một lần phục vụ được nhiều request cùng lúc. Batching tăng tổng throughput cho tới khi chạm giới hạn tính toán hoặc hết KV cache; cái giá là mỗi request chậm đi.

Đo trên cùng máy, Qwen2.5-3B Q4_K_M, 16 request mỗi mức, prompt trung bình 565 token, 128 token output, tắt cache prompt:

| Request đồng thời | TTFT p50 | Token/giây mỗi request (p50) | Tổng token output/giây |
|---|---|---|---|
| 1 | 1,09 s | 40,3 | 30,6 |
| 2 | 2,32 s | 35,4 | 43,5 |
| 4 | 4,43 s | 28,1 | 56,7 |
| 8 | 8,42 s | 22,2 | 72,0 |

Tổng throughput tăng 2,4 lần từ 1 lên 8 request đồng thời, trong khi mỗi request chậm đi gần một nửa. TTFT tăng gần tỉ lệ với số request: prefill bị giới hạn bởi tính toán, và GPU 10 nhân của M4 phải prefill tám prompt 565 token trước khi request nào có token đầu. Trên GPU máy chủ, cả hai hiện tượng nhẹ hơn nhiều vì GPU có nhiều năng lực tính toán hơn hẳn, nhưng hình dạng đánh đổi vẫn vậy. Đây là đánh đổi trung tâm của serving: cấu hình theo SLO về TTFT và tốc độ mỗi request ở p95, không theo throughput tối đa.

Một lưu ý về độ tin cậy của chính các số này: chạy lại cùng cấu hình trong cùng buổi, cách nhau 20–45 phút, tốc độ decode lệch 11–12% (Q4_K_M một request: 45,3 rồi 40,3 token/giây; Q2_K: 55,3 rồi 48,9), vì máy đo đang dùng swap và chạy các việc khác. Hình dạng đánh đổi không đổi giữa hai lần. Khi so hai cấu hình, chạy xen kẽ nhiều lần trên máy sạch và so phân phối, đừng so hai con số đơn lẻ.

Chunked prefill (chia prompt dài thành khúc để xen với decode), speculative decoding và tách prefill/decode ra GPU khác nhau [18] là các kỹ thuật giảm xung đột giữa hai giai đoạn; [Latency §2.5](../latency/#25-speculative-decoding-và-các-kỹ-thuật-phía-serving) đi qua chúng. vLLM bật chunked prefill mặc định và ưu tiên decode khi xếp batch [3].

## 1.4 Quantization: FP8, INT4, GGUF, AWQ, GPTQ

Quantization lưu weights (và có thể cả activation, KV cache) ở ít bit hơn. Hai họ chính:

- **Chỉ lượng tử hoá weights** (W4A16, W8A16): weights 4 hoặc 8 bit, tính toán ở 16 bit. Giảm bộ nhớ và tăng tốc decode (đọc ít byte hơn). GPTQ lượng tử hoá model 175B xuống 3–4 bit trong khoảng 4 giờ GPU với mất mát không đáng kể [19]. AWQ chọn 1% weights quan trọng nhất dựa trên phân phối activation và bảo vệ chúng bằng cách scale theo kênh (không giữ riêng ở độ chính xác cao, để tránh mixed precision khó chạy nhanh trên phần cứng), không cần backprop, khái quát tốt sang model instruct [20].
- **Lượng tử hoá cả weights và activation** (W8A8, FP8, FP4): tính toán cũng ở số bit thấp, cần phần cứng hỗ trợ. SmoothQuant chuyển độ khó từ activation sang weights để chạy INT8 cả hai, tăng tốc tới 1,56 lần và giảm 2 lần bộ nhớ [21]. LLM.int8() tách các đặc trưng ngoại lai ra ma trận 16 bit để giữ nguyên chất lượng ở 175B [22].

Mất bao nhiêu chất lượng? Một nghiên cứu năm 2024 với hơn 500.000 lần eval trên họ Llama 3.1 (8B, 70B, 405B) kết luận: FP8 W8A8 gần như không mất gì ở mọi kích thước; INT8 W8A8 chỉnh kỹ mất 1–3%; INT4 chỉ weights cạnh tranh hơn dự đoán, gần bằng 8 bit. Về tốc độ trên vLLM, W4A16 rẻ nhất khi request được xử lý từng cái một, W8A8 thắng khi continuous batching dưới tải bất đồng bộ [23].

Kết luận đó là cho model 8B–405B và không xuống dưới 4 bit. Đo trên tác vụ trích xuất intent ở bài [Fine-tuning](../fine-tuning/#33-eval-trước-và-sau-fine-tune) (541 tin nhắn khác nhau, chỉ dùng system prompt), Qwen2.5-3B-Instruct chưa fine-tune, chạy qua llama-server:

| File GGUF | Dung lượng | Exact match | JSON hỏng (không có `json_schema`) |
|---|---|---|---|
| Q2_K | 1,28 GiB | 13,7% | 299 / 541 |
| Q4_K_M | 1,96 GiB | 95,4% | 0 / 541 |
| Q8_0 | 3,37 GiB | 96,3% | 0 / 541 |

Cột exact match lấy từ lần chạy đo tốc độ; cột JSON hỏng lấy từ lần chạy ở [§3.3](#33-tool-calling-và-structured-output-trên-model-tự-host), có đủ phân loại lỗi.

Q4_K_M gần như ngang Q8_0 với khoảng 58% dung lượng, khớp với kết luận ở trên. Q2_K thì sụp: không chỉ đoán sai nhiều hơn mà 299 câu trả lời không còn là JSON hợp lệ, ví dụ `{"intent": "warranty, "order_id": null}` (thiếu dấu nháy đóng). Từ Q4 xuống Q2, chất lượng không giảm dần mà gãy. Mức gãy khác nhau theo model và tác vụ, nên chọn mức lượng tử hoá bằng eval của bạn, trên đúng file sẽ chạy production.

### Định dạng và phần cứng

| Định dạng | Ở đâu | Ghi chú |
|---|---|---|
| FP8 (W8A8) | vLLM, SGLang, TensorRT-LLM | Trong vLLM, FP8 W8A8 chạy trên GPU NVIDIA Ada (SM 8.9), Hopper (SM 9.0) và GPU AMD; không chạy trên Ampere [7] |
| NVFP4, FP4 | TensorRT-LLM, SGLang | TensorRT-LLM: FP4 và FP8 trên Blackwell và Hopper thế hệ mới nhất [12]; SGLang `nvfp4_online` cần Blackwell SM100/SM103 [11] |
| MXFP4 | gpt-oss | Checkpoint gốc của gpt-oss phát hành ở MXFP4 [43]; AMD MI355X hỗ trợ MXFP4 trong phần cứng [25] |
| AWQ, GPTQ (W4A16) | vLLM, SGLang, TensorRT-LLM | Checkpoint lượng tử hoá sẵn có trên Hugging Face cho model phổ biến |
| GGUF (Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ…) | llama.cpp, Ollama | Định dạng một file, chạy trên CPU, Metal, CUDA, Vulkan; `llama-quantize` chuyển từ BF16 [15] |

Bảng hỗ trợ khác nhau giữa engine và đổi theo phiên bản; đọc bảng quantization của engine bạn dùng trước khi mua GPU.

## 1.5 License của model open-weight

"Open-weight" chỉ nói weights tải về được; quyền dùng nằm trong license. Model mà bài này dùng để đo là ví dụ: Qwen2.5-3B-Instruct phát hành theo **Qwen Research License**, chỉ cho phép dùng phi thương mại (nghiên cứu, đánh giá); dùng thương mại phải xin license riêng [38]. Cùng họ, Qwen3-8B dùng Apache 2.0 [37]. Không suy license từ tên họ model.

| Model | License | Điều cần chú ý |
|---|---|---|
| Llama 3.3, Llama 4 | Llama Community License [33][34] | Sản phẩm có trên 700 triệu người dùng hoạt động hàng tháng phải xin license từ Meta; phải ghi "Built with Llama"; kèm Acceptable Use Policy |
| Gemma 3 | Gemma Terms of Use [36] | Điều khoản riêng của Google, kèm chính sách cấm dùng |
| Gemma 4 | Apache 2.0 [35] | Google đổi sang Apache 2.0 từ Gemma 4 |
| Qwen3, Qwen3.5 | Apache 2.0 [37] | Một số model Qwen cũ hơn dùng license khác (ví dụ Qwen2.5-3B ở trên) |
| DeepSeek-V3 (bản gốc 12/2024) | DeepSeek License Agreement 1.0 [39] | License riêng cho weights; các bản sau DeepSeek-V3-0324, V3.1, V3.2 ghi MIT trên Hugging Face [39] |
| DeepSeek-R1 | MIT [40] | |
| gpt-oss | Apache 2.0 [41] | |
| Mistral Small 3.1 | Apache 2.0 [42] | Không phải mọi model Mistral đều Apache 2.0 |

Model fine-tune hoặc lượng tử hoá từ một model gốc vẫn chịu license của model gốc. Bản GGUF hay AWQ do cộng đồng đăng lại trên Hugging Face không đổi license; đọc license của repo gốc.

---

# Phần 2 — Inference engine và phần cứng

## 2.1 vLLM

vLLM là engine phổ biến nhất cho serving production, chạy trên GPU NVIDIA, AMD ROCm, Intel XPU, CPU, và Apple Silicon qua plugin vLLM-Metal [8]. Bản mới nhất là 0.33.0 (01/11/2026).

```bash
vllm serve openai/gpt-oss-20b \
    --revision <commit-hash> \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice --tool-call-parser openai
```

Các mặc định đáng biết, đọc từ source tại `vllm/config/` [2]:

- **`--gpu-memory-utilization`** mặc định 0,92: phần bộ nhớ GPU mà tiến trình vLLM này được dùng, gồm weights, KV cache và activation. Phần còn lại sau weights và activation thành KV cache, cấp sẵn lúc khởi động. Đây là giới hạn cho từng instance; hai instance trên một GPU thì mỗi cái đặt 0,5.
- **`--max-model-len`** mặc định lấy từ config của model. Model hỗ trợ 128k token nhưng KV cache không đủ cho một chuỗi 128k thì vLLM không khởi động được; đặt giá trị bằng context bạn thật sự cần.
- **`--max-num-seqs`** và **`--max-num-batched-tokens`**: trần số request trong một batch và số token mỗi bước. Với `vllm serve`, mặc định phụ thuộc GPU (`get_batch_defaults` trong `vllm/engine/arg_utils.py` [2]): GPU từ 160 GiB trở lên (B200, B300) là 1024 / 16384; từ 70 GiB trở lên trừ A100 (H100, H200) là 1024 / 8192; GPU còn lại, kể cả A100, là 256 / 2048. `--performance-mode throughput` nhân đôi cả hai. Theo tài liệu tối ưu của vLLM, `max_num_batched_tokens` nhỏ cho khoảng cách giữa token tốt hơn, lớn cho TTFT tốt hơn [3]. Hai máy khác loại GPU chạy cùng một lệnh sẽ có hành vi khác nhau; ghi rõ hai giá trị này trong cấu hình thay vì dựa vào mặc định.
- **Prefix caching** bật mặc định: request có chung prefix (system prompt dài, tài liệu, lịch sử hội thoại) dùng lại KV cache đã tính.
- **Chunked prefill** bật mặc định [3].
- **`--quantization`** mặc định đọc `quantization_config` trong config của model [2]: checkpoint AWQ, GPTQ, FP8 được nhận ra tự động.
- **`--tensor-parallel-size`** và **`--pipeline-parallel-size`** mặc định 1.

**Khi hết KV cache**, vLLM tạm dừng (preempt) một số request, giải phóng KV cache của chúng và tính lại khi có chỗ; mặc định là tính lại (`RECOMPUTE`). Dòng log thống kê định kỳ có thêm `Preemptions: N` và metric `vllm:num_preemptions` tăng [6]. Cách xử lý theo tài liệu: tăng `gpu_memory_utilization`, giảm `max_num_seqs` hoặc `max_num_batched_tokens`, hoặc thêm GPU qua tensor parallel [3].

**Metrics** ở `/metrics` theo chuẩn Prometheus, tiền tố `vllm:` [6]. Những metric đáng đặt alert: `vllm:num_requests_waiting` (hàng đợi), `vllm:kv_cache_usage_perc` (0–1), `vllm:num_preemptions`, `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:prefix_cache_hits` so với `vllm:prefix_cache_queries`.

**Structured output** có sẵn trong server OpenAI-compatible, backend mặc định `auto`: thử xgrammar trước, chuyển sang guidance khi request dùng tính năng xgrammar chưa hỗ trợ [4]. **Tool calling** cần `--enable-auto-tool-choice` và `--tool-call-parser` khớp với model: `hermes` cho Qwen2.5, `llama3_json` cho Llama 3.x, `openai` cho gpt-oss, `qwen3_xml` cho Qwen3-Coder [5] ([§3.3](#33-tool-calling-và-structured-output-trên-model-tự-host)).

**Khởi động chậm.** vLLM compile model bằng `torch.compile` và capture CUDA graph khi khởi động. Cache compile lưu ở `~/.cache/vllm` (`VLLM_CACHE_ROOT`), có thể copy sang máy khác hoặc đóng vào container image để lần khởi động sau nhanh hơn; `--enforce-eager` bỏ qua cả hai, khởi động nhanh nhất nhưng decode chậm hơn [3]. Điều này quan trọng khi autoscale.

## 2.2 SGLang

SGLang (0.5.23, 30/10/2026) [11] nổi bật ở hai điểm. **RadixAttention** lưu KV cache trong một cây radix theo prefix, nên các request chia sẻ prefix (agent gọi nhiều lần với cùng hướng dẫn, chat nhiều lượt, few-shot) dùng lại KV mà không cần cấu hình; bài báo SGLang đo throughput tới 6,4 lần so với các hệ thống khác trên agent, few-shot, JSON decoding, RAG và chat nhiều lượt [10]. **Structured output** dùng XGrammar làm mặc định, hỗ trợ JSON schema, regex và EBNF [11].

SGLang chạy trên GPU NVIDIA (A100, H100, H200, B200 tới GB300, cả dòng RTX), AMD Instinct MI300X tới MI355X, Google TPU, GPU và CPU Intel, Huawei Ascend, Apple Silicon [11]. Tham số tương ứng với vLLM:

- `--mem-fraction-static`: phần bộ nhớ cho weights và KV cache; không đặt thì tự tính `(bộ nhớ GPU − phần giữ lại) / bộ nhớ GPU`, mặc định 0,88 nếu không đọc được bộ nhớ GPU [11].
- `--max-running-requests`, `--chunked-prefill-size` (`-1` để tắt chunked prefill): mặc định do SGLang tự chọn [11].
- `--enable-metrics` **tắt mặc định** [11]; bật để có các metric `sglang:time_to_first_token_seconds`, `sglang:num_running_reqs`, `sglang:num_queue_reqs`, `sglang:token_usage`, `sglang:cache_hit_rate`.

Chọn giữa vLLM và SGLang bằng benchmark trên tải của bạn ([§3.2](#32-benchmark-throughput-và-độ-trễ-trên-tải-thật)). Cả hai cùng có server OpenAI-compatible nên chuyển qua lại dễ.

## 2.3 TensorRT-LLM và TGI

**TensorRT-LLM** là engine của NVIDIA. Bản stable trên PyPI là 1.2.1, vẫn có backend TensorRT và lệnh `trtllm-build` [12]. Từ pre-release 1.3.0rc21 và cả dòng 1.4.0rc (mới nhất 1.4.0rc3, 06/11/2026), backend TensorRT đã bị gỡ: PyTorch là backend duy nhất, checkpoint Hugging Face nạp trực tiếp, "không còn bước chuyển checkpoint hay build engine riêng" [12]. Dự án mới nên viết theo hướng PyTorch backend để khỏi viết lại khi dòng 1.4 thành stable; hướng dẫn dựa trên `trtllm-build` sẽ hết dùng được từ đó. Server OpenAI-compatible chạy bằng `trtllm-serve`. Điểm mạnh là hỗ trợ sớm các định dạng mới của NVIDIA: FP4, nhiều biến thể FP8, KV cache NVFP4, W4A8 AWQ/GPTQ [12]. Chỉ chạy trên GPU NVIDIA.

**TGI** (Text Generation Inference) của Hugging Face đã chuyển sang chế độ bảo trì: chỉ nhận sửa lỗi nhỏ, tài liệu và bảo trì nhẹ. README của dự án khuyên dùng vLLM, SGLang, hoặc llama.cpp và MLX cho chạy tại chỗ [13]. Hệ thống đang chạy TGI vẫn chạy được, nhưng nên lên kế hoạch chuyển.

## 2.4 llama.cpp và Ollama

**llama.cpp** chạy model GGUF trên CPU, Apple Silicon (Metal bật mặc định trên macOS), CUDA, Vulkan [15]. `llama-server` cung cấp API OpenAI-compatible, tool calling qua chat template Jinja (`--jinja`, bật mặc định), và ràng buộc output bằng grammar hoặc `json_schema` [14].

```bash
llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf -c 16384 -np 8 --port 8080
```

Hai tham số quyết định khả năng phục vụ đồng thời:

- **`-np` (số slot)**: số request xử lý song song; mặc định `-1` (tự chọn) [14].
- **`-c` là tổng context cho mọi slot**, chia đều. Đã kiểm trên build 12110: `-c 16384 -np 4` cho mỗi slot 4.096 token (`n_ctx_slot = 4096`); `-c 2048 -np 2` thì prompt 2.832 token bị từ chối với lỗi `exceed_context_size_error`, `"n_ctx": 1024`. Khi tăng `-np` phải tăng `-c` theo.

llama.cpp không có PagedAttention như vLLM; mỗi slot có phần KV riêng (trừ khi bật `--kv-unified`, mặc định bật khi số slot để tự động [14]). Với nhiều người dùng đồng thời trên GPU NVIDIA, vLLM hoặc SGLang thường hợp hơn; llama.cpp hợp với Apple Silicon, CPU, thiết bị biên, và khi cần một file chạy được ở mọi nơi.

**Ollama** (0.40.15) [16] đóng gói llama.cpp thành một dịch vụ dễ cài, có thư viện model và API OpenAI-compatible ở `http://localhost:11434/v1/`. Mặc định của Ollama hợp với một người dùng trên máy cá nhân, không hợp với server:

- **`OLLAMA_NUM_PARALLEL` mặc định 1** [16]: mỗi model xử lý một request mỗi lần. Bộ nhớ cần tăng theo `OLLAMA_NUM_PARALLEL × OLLAMA_CONTEXT_LENGTH`.
- **Context mặc định theo VRAM**: 4k token dưới 24 GiB, 32k từ 24 tới 48 GiB, 256k từ 48 GiB trở lên. Trang FAQ vẫn ghi mặc định 4.096 token [16]; kiểm giá trị thật bằng `ollama ps` (cột `CONTEXT`). Tài liệu Ollama khuyên ít nhất 64.000 token cho agent, web search và coding.
- **`keep_alive` 5 phút**: model bị gỡ khỏi bộ nhớ sau 5 phút rảnh [16]; request kế tiếp phải chờ nạp lại.
- **`OLLAMA_MAX_LOADED_MODELS`**: mặc định 3 × số GPU (hoặc 3 khi chạy CPU) [16].

## 2.5 Chọn GPU: NVIDIA, AMD, Apple Silicon

Hai thông số quan trọng nhất cho inference: **dung lượng bộ nhớ** (model và KV cache có vừa không) và **băng thông bộ nhớ** (decode nhanh tới đâu). TFLOPS quan trọng cho prefill và batch lớn.

| GPU | Bộ nhớ | Băng thông | Ghi chú |
|---|---|---|---|
| NVIDIA L4 | 24 GB | 300 GB/s | Rẻ, tiết kiệm điện; model nhỏ [24] |
| NVIDIA L40S | 48 GB GDDR6 | 864 GB/s | Ada, có FP8 [24] |
| NVIDIA A100 80GB | 80 GB HBM2e | 1.935 (PCIe) / 2.039 GB/s (SXM) | Ampere, không có FP8 W8A8 trong vLLM [7][24] |
| NVIDIA H100 SXM | 80 GB | 3,35 TB/s | Hopper, FP8 [24] |
| NVIDIA H200 | 141 GB | 4,8 TB/s | Hopper, nhiều bộ nhớ hơn H100 [24] |
| NVIDIA B200 | 180 GB (1.440 GB cho 8 GPU trong DGX B200) | 8 TB/s (64 TB/s cho 8 GPU) | Blackwell, FP4 [24] |
| NVIDIA RTX PRO 6000 Blackwell | 96 GB GDDR7 | 1.792 GB/s | Workstation, FP4 [24] |
| AMD Instinct MI300X | 192 GB HBM3 | 5,3 TB/s | [25] |
| AMD Instinct MI325X | 256 GB HBM3E | 6 TB/s | FP8 [25] |
| AMD Instinct MI355X | 288 GB HBM3E | 8 TB/s | MXFP4, MXFP6 [25] |
| Apple M4 | tới 32 GB hợp nhất | 120 GB/s | Máy đo trong bài [26] |
| Apple M4 Max | tới 128 GB | tới 546 GB/s | [26] |
| Apple M5 Ultra (Mac Studio) | tới 512 GB | 1,2 TB/s | [26] |

Số của B200 chia từ thông số DGX B200 cho 8 GPU; NVIDIA không ghi riêng từng GPU trên trang đó. Số "GB" của nhà sản xuất là đơn vị thập phân hoặc gần đúng; bộ nhớ thật mà engine thấy nhỏ hơn một chút.

Cách đọc bảng:

- **Model vừa một GPU thì đừng chia ra nhiều GPU.** Tensor parallel thêm chi phí đồng bộ giữa các GPU [3]. Nhưng nếu model chỉ vừa sát, GPU thứ hai cho nhiều KV cache hơn và nhiều request đồng thời hơn ([§3.1](#31-chọn-model-theo-tác-vụ-và-phần-cứng)).
- **AMD có nhiều bộ nhớ mỗi GPU** và được vLLM, SGLang hỗ trợ [8][11], nhưng bảng hỗ trợ quantization và kernel khác NVIDIA. Kiểm định dạng bạn cần trước.
- **Apple Silicon** có bộ nhớ hợp nhất lớn với giá thấp hơn GPU máy chủ cùng dung lượng, chạy được model lớn cho một hoặc vài người dùng. Băng thông thấp hơn GPU máy chủ nhiều lần, và không có hệ sinh thái serving như CUDA.
- **GPU tiêu dùng** (RTX 5090, 32 GB GDDR7 [24]) hợp cho model nhỏ và thử nghiệm. Kiểm điều khoản driver và trung tâm dữ liệu của nhà sản xuất nếu định đặt trong data center.

## 2.6 Cloud GPU và dịch vụ inference serverless

Giá thuê theo giờ mỗi GPU, on-demand, truy cập 07/11/2026:

| Nhà cung cấp | GPU | Giá mỗi GPU mỗi giờ |
|---|---|---|
| RunPod | H100 SXM 80 GB | 3,99 USD; H100 PCIe 2,89 USD [27] |
| Modal | H100 SXM5 | 0,001097 USD/giây ≈ 3,95 USD [28] |
| CoreWeave | HGX H100 8 GPU | 49,24 USD/node ≈ 6,16 USD [29] |
| Together (GPU cluster) | H100 / H200 / B200 | 3,99 / 5,99 / 8,19 USD [30] |
| DeepInfra | H100 / H200 / B200 | 2,90 / 3,49 / 4,89 USD [32] |
| Fireworks (on-demand deployment) | H100 / B200 | 8,00 / 13,00 USD [31] |

Giá AWS, Google Cloud và Azure cho H100 không đọc được từ trang giá tĩnh khi viết bài; tra trực tiếp trang giá của họ cho region bạn dùng. Giá reserved, spot và cam kết dài hạn thấp hơn on-demand đáng kể.

Dịch vụ serverless bán model open-weight theo token, không cần quản GPU:

| Model | Nhà cung cấp | Input / output, USD mỗi triệu token |
|---|---|---|
| gpt-oss-120b | Together | 0,15 / 0,60 [30] |
| gpt-oss-120b | Fireworks | 0,15 / 0,60 (input đã cache 0,015) [31] |
| Llama 3.3 70B Instruct Turbo | DeepInfra | 0,10 / 0,32 [32] |
| Qwen3-32B | DeepInfra | 0,08 / 0,28 [32] |

Đây là mốc phải so khi tính tự host ([§3.5](#35-so-sánh-tổng-chi-phí-với-api)). Dịch vụ serverless có thể chạy model ở độ chính xác số khác với checkpoint gốc, và tên như "Turbo" thường chỉ một biến thể đã tối ưu mà trang giá không nói rõ là gì; đọc trang mô tả model và chạy eval của bạn trên đúng endpoint đó nếu chất lượng quan trọng.

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Chọn model theo tác vụ và phần cứng

Thứ tự hợp lý:

1. **Chọn model bằng eval trên tác vụ của bạn**, ở vài kích thước, kể cả bản lượng tử hoá ([§1.4](#14-quantization-fp8-int4-gguf-awq-gptq)). Không có eval thì không biết model 8B có đủ không.
2. **Kiểm license** cho cách bạn định dùng ([§1.5](#15-license-của-model-open-weight)).
3. **Tính bộ nhớ** cho context và số request đồng thời bạn cần.
4. **Benchmark trên phần cứng thật** trước khi cam kết ([§3.2](#32-benchmark-throughput-và-độ-trễ-trên-tải-thật)).

`mem_calc.py` đọc `config.json` và tính bước 3:

```python
"""Ước lượng bộ nhớ GPU cho serving: weights + KV cache, đọc từ config.json của model."""
import json
import math

MiB, GiB = 2**20, 2**30


def text_config(cfg):
    """Model đa phương thức (Gemma 3, Qwen3.5) để tham số phần ngôn ngữ trong `text_config`."""
    return cfg.get("text_config", cfg)


def kv_bytes_per_layer(cfg, elem_bytes=2):
    """Byte KV cache cho một token ở một lớp attention: K và V, mỗi KV head một vector head_dim."""
    cfg = text_config(cfg)
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads") or cfg["num_attention_heads"]
    return 2 * kv_heads * head_dim * elem_bytes


def layer_kinds(cfg):
    """Loại của từng lớp. Ưu tiên `layer_types`; rồi `sliding_window_pattern` (Gemma 3: cứ N lớp
    có một lớp full); rồi `use_sliding_window: true`. Không đủ thông tin thì coi mọi lớp là full
    attention: ước lượng dư (Gemma 2, Mistral 7B v0.1 thật ra nhỏ hơn) còn hơn thiếu."""
    cfg = text_config(cfg)
    n = cfg["num_hidden_layers"]
    if cfg.get("layer_types"):
        return cfg["layer_types"]
    pattern = cfg.get("sliding_window_pattern")
    if pattern and cfg.get("sliding_window"):
        return ["full_attention" if (i + 1) % pattern == 0 else "sliding_attention" for i in range(n)]
    if cfg.get("use_sliding_window") is True and cfg.get("sliding_window"):
        return ["sliding_attention"] * n
    return ["full_attention"] * n


def kv_bytes(cfg, tokens, elem_bytes=2):
    """KV cache cho một chuỗi dài `tokens`. Lớp sliding window chỉ giữ `sliding_window` token cuối;
    lớp linear attention (Qwen3.5) giữ state cố định, không tăng theo token, nên không tính ở đây."""
    per_layer = kv_bytes_per_layer(cfg, elem_bytes)
    window = text_config(cfg).get("sliding_window") or tokens
    per_token = {"full_attention": tokens, "sliding_attention": min(tokens, window), "linear_attention": 0}
    kinds = layer_kinds(cfg)
    unknown = set(kinds) - set(per_token)
    if unknown:
        raise ValueError(f"chưa biết cách tính KV cho lớp {sorted(unknown)}")
    return sum(per_layer * per_token[k] for k in kinds)


def weights_gib(params_billion, bits):
    """Weights đã lượng tử hoá: số tham số × số bit. Thêm vài phần trăm cho scale và lớp giữ
    nguyên độ chính xác; với checkpoint có sẵn, lấy dung lượng file thật thay cho con số này."""
    return params_billion * 1e9 * bits / 8 / GiB


def plan(cfg, weights, gpu_gib, context, *, gpus=1, kv_elem_bytes=2, utilization=0.9, overhead_gib=2.0):
    """Bao nhiêu chuỗi dài `context` chạy đồng thời được trên `gpus` GPU, mỗi GPU `gpu_gib` GiB
    (dung lượng nvidia-smi báo, không phải số GB trên trang sản phẩm).
    `weights` tính bằng GiB. Tensor parallel chia cả weights lẫn KV cache cho các GPU, nên tính
    trên tổng. `utilization` giống --gpu-memory-utilization của vLLM: phần bộ nhớ engine được
    dùng, gồm cả weights. `overhead_gib` mỗi GPU (activation, CUDA graph, buffer) phải đo trên
    engine thật; 2 GiB chỉ là điểm khởi đầu."""
    budget = gpus * (gpu_gib * utilization - overhead_gib) - weights
    per_seq = kv_bytes(cfg, context, kv_elem_bytes) / GiB
    return {"kv_per_seq_gib": round(per_seq, 2),
            "kv_budget_gib": round(budget, 1),
            # tổng token mọi chuỗi cộng lại; model có lớp sliding window thực tế chứa được nhiều hơn
            "kv_tokens": max(0, math.floor(budget * GiB / kv_bytes(cfg, 1, kv_elem_bytes))),
            "max_seqs": max(0, math.floor(budget / per_seq))}


if __name__ == "__main__":
    import sys
    cfg = json.load(open(sys.argv[1]))
    w = weights_gib(float(sys.argv[2]), float(sys.argv[3]))
    print(plan(cfg, w, float(sys.argv[4]), int(sys.argv[5])))
```

Với các config thật, giả định `utilization` 0,9 và 2 GiB overhead mỗi GPU. Dung lượng GPU là số `nvidia-smi` báo, không phải số trên trang sản phẩm: L4 23.034 MiB (22,5 GiB) [44], H100 80GB HBM3 81.559 MiB (79,6 GiB) [44], H200 143.771 MiB (140,4 GiB) [44], RTX 5090 32.607 MiB (31,8 GiB) [44]. Với L4, khác biệt giữa "24 GB" và 22,5 GiB đủ đổi kết luận.

| Cấu hình | Weights | KV mỗi chuỗi 32k | Ngân sách KV | Số chuỗi 32k đồng thời (tổng token) |
|---|---|---|---|---|
| Llama 3.1 8B BF16, 1× L4 | 15,0 GiB | 4,0 GiB | 3,3 GiB | 0 (26.928 token) |
| Llama 3.1 8B BF16, 1× H100 | 15,0 GiB | 4,0 GiB | 54,7 GiB | 13 (448.308 token) |
| Qwen3-32B FP8, 1× H100 | 30,5 GiB | 8,0 GiB | 39,2 GiB | 4 (160.442 token) |
| Llama 3.3 70B FP8, 1× H100 | 65,7 GiB | 10,0 GiB | 4,0 GiB | 0 (13.023 token) |
| Llama 3.3 70B FP8, 2× H100 | 65,7 GiB | 10,0 GiB | 73,7 GiB | 7 (241.359 token) |
| Llama 3.3 70B BF16, 2× H200 | 131,4 GiB | 10,0 GiB | 117,3 GiB | 11 (384.388 token) |
| gpt-oss-120b MXFP4, 1× H100 | 60,8 GiB | 1,13 GiB | 8,9 GiB | 7 (129.363 token) |
| gpt-oss-20b MXFP4, 1× RTX 5090 | 12,8 GiB | 0,75 GiB | 13,9 GiB | 18 (302.743 token) |

Đọc bảng:

- **Llama 3.1 8B BF16 trên một L4 không chạy nổi một chuỗi 32k**: còn 3,3 GiB cho KV, khoảng 27.000 token. vLLM với `--max-model-len 32768` sẽ từ chối khởi động; phải giảm context, dùng FP8 (weights còn 7,5 GiB) hoặc lượng tử hoá KV.
- **Llama 3.3 70B FP8 trên một H100 chạy được nhưng chật**: còn khoảng 4 GiB cho KV cache, tức khoảng 13.000 token cho mọi request cộng lại. Không phục vụ nổi một chuỗi 32k. Thêm GPU thứ hai không chỉ gấp đôi mà tăng ngân sách KV từ 4,0 lên 73,7 GiB, vì weights đã nằm sẵn.
- **gpt-oss-120b vừa một H100** nhờ checkpoint MXFP4 và KV cache nhỏ (sliding window ở nửa số lớp).
- **Số chuỗi đồng thời là chặn dưới.** Bảng giả định mọi request dùng hết 32k token. Với PagedAttention, request ngắn chỉ chiếm phần nó dùng [1]; dùng cột `kv_tokens` của hàm (tổng token mọi request) với độ dài thật của traffic.
- **Overhead phải đo.** Con số 2 GiB là điểm khởi đầu. vLLM ghi vào log giá trị `--kv-cache-memory` ứng với KV cache vừa cấp; truyền lại giá trị đó ở lần khởi động sau để bỏ bước đo bộ nhớ, nhưng chỉ đúng trên cùng GPU với cùng lượng bộ nhớ trống [3]; llama.cpp in bảng chia bộ nhớ theo model, context và compute.

Test (`t_mem_calc.py`, chạy trên `config.json` thật): KV của Qwen2.5-3B khớp đúng số llama.cpp in ra (144 MiB cho 4.096 token, 576 MiB cho 4 slot); `use_sliding_window: false` được tôn trọng; Llama 3.1 8B ra 128 KiB mỗi token; phần tăng KV của gpt-oss-120b từ 16k lên 32k chỉ đến từ 18 lớp full attention; Gemma 3 27B đọc từ `text_config`, 10 lớp full và 52 lớp sliding 1.024 token, 2,91 GiB cho 32k; Qwen3.5-35B-A3B chỉ tính 10 lớp full attention, bỏ 30 lớp linear attention; Gemma 2 9B và Mistral 7B v0.1 (config không ghi lớp nào sliding) được tính như full attention; loại lớp lạ thì báo lỗi; KV 1 byte bằng nửa KV 2 byte; Llama 3.3 70B FP8 trên một H100 không chứa nổi chuỗi 32k nhưng vẫn còn 10.000–20.000 token; hai H200 chứa hơn 10 chuỗi; gpt-oss-120b chứa ít nhất 4 chuỗi 32k trên một H100.

## 3.2 Benchmark throughput và độ trễ trên tải thật

Số benchmark của người khác (và của bài này) chỉ cho biết thứ tự độ lớn. Con số dùng để quyết định phải đo với model, engine, GPU và **phân phối độ dài prompt và output của traffic thật**.

`bench.py` đo bất kỳ endpoint OpenAI-compatible nào (vLLM, SGLang, llama-server, Ollama):

```python
"""Đo TTFT, tốc độ sinh mỗi request và throughput tổng của một endpoint OpenAI-compatible
(vLLM, SGLang, llama-server, Ollama) ở nhiều mức đồng thời. Mỗi request stream và đếm token
từ `usage` của chunk cuối."""
import asyncio
import json
import statistics
import time

import httpx2


async def one(client, url, model, messages, max_tokens, extra):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, **extra}
    t0 = time.perf_counter()
    first = last = first_content = None
    usage = None
    async with client.stream("POST", url, json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for c in chunk.get("choices", []):
                d = c["delta"]
                # completion_tokens gồm cả token suy nghĩ, nên đo thời gian theo mọi loại chữ;
                # vLLM gọi trường đó là `reasoning`, nhiều engine khác gọi `reasoning_content`
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    last = time.perf_counter()
                    first = first or last
                    if d.get("content"):
                        first_content = first_content or last
    if usage is None or first is None:
        raise RuntimeError("server không trả usage hoặc không sinh chữ nào")
    n = usage["completion_tokens"]
    return {"ttft": first - t0, "ttft_content": first_content and first_content - t0,
            "e2e": last - t0, "out": n, "in": usage["prompt_tokens"],
            "decode_tps": (n - 1) / (last - first) if n > 1 and last > first else None}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


async def run(url, model, prompts, concurrency, max_tokens=128, extra=None):
    """Chạy hết `prompts`, tối đa `concurrency` request cùng lúc."""
    sem = asyncio.Semaphore(concurrency)
    async with httpx2.AsyncClient(timeout=600) as client:
        async def task(p):
            async with sem:
                return await one(client, url, model, p, max_tokens, extra or {})
        t0 = time.perf_counter()
        res = await asyncio.gather(*(task(p) for p in prompts))
        wall = time.perf_counter() - t0
    ttft = [r["ttft"] for r in res]
    tps = [r["decode_tps"] for r in res if r["decode_tps"]]
    return {"concurrency": concurrency, "requests": len(res),
            "prompt_tokens_mean": round(statistics.mean(r["in"] for r in res)),
            "ttft_p50": round(pct(ttft, 50), 3), "ttft_p95": round(pct(ttft, 95), 3),
            "decode_tps_p50": round(statistics.median(tps), 1),
            "output_tps_total": round(sum(r["out"] for r in res) / wall, 1),
            "wall_s": round(wall, 1)}
```

Những điều cần làm đúng, phần lớn học được khi viết chính benchmark này:

- **Tắt cache prompt, hoặc dùng prompt khác nhau mỗi lần.** Lần đo đầu của bài chạy lại cùng 16 prompt ở mỗi mức đồng thời; từ mức thứ hai, llama-server dùng lại KV của prompt đã thấy và TTFT giảm từ khoảng 1 giây xuống 0,04 giây. Số đó đo prefix cache, không đo prefill. Bảng ở [§1.3](#13-prefill-decode-và-vì-sao-batching-quan-trọng) chạy với `"cache_prompt": false`. Ngược lại, nếu traffic thật có prefix chung (system prompt dài), hãy đo với prefix chung đó, vì đó là lợi thế thật.
- **Cố định độ dài output khi so cấu hình.** Model dừng sớm ở prompt này, muộn ở prompt kia làm số liệu nhiễu. llama-server nhận `"ignore_eos": true`; vLLM có tham số tương tự. Khi đo cho capacity planning thì ngược lại: dùng độ dài output thật.
- **Đếm token từ `usage`**, không đếm chunk: một chunk có thể chứa nhiều token. `stream_options.include_usage` yêu cầu server gửi usage ở chunk cuối; server không gửi thì `bench.py` báo lỗi thay vì đoán.
- **Đo nhiều mức đồng thời** và báo p50, p95, không chỉ trung bình. Điểm cần tìm là mức đồng thời lớn nhất mà TTFT p95 và tốc độ mỗi request vẫn trong SLO.
- **Model có reasoning** (gpt-oss, Qwen3 bật thinking): `completion_tokens` gồm cả token suy nghĩ, nên `bench.py` đo thời gian theo mọi chữ stream về, cả trường `reasoning` (tên trong vLLM) lẫn `reasoning_content` (tên ở nhiều engine khác). Nếu chỉ đo theo `content`, tốc độ bị thổi lên nhiều lần vì tổng token bị chia cho một khoảng thời gian chỉ có phần trả lời. `ttft` là token đầu tiên của bất kỳ loại nào; `ttft_content` là lúc người dùng thấy chữ trả lời đầu tiên, thường mới là con số UX quan tâm.
- **Đo cả server một slot.** Tám request vào llama-server `-np 1`: TTFT p50 28,2 giây, tổng throughput 32,5 token/giây, xấp xỉ mức một request. Đó là cảnh ở đầu bài.

Test (`t_bench.py`, SSE giả lập, không cần server): chunk chỉ có `role` hoặc chỉ có `usage` không tính là chữ; số token lấy từ `usage`; thiếu `usage` thì báo lỗi; stream 100 token `reasoning` rồi 10 token `content`, mỗi token 10 ms, cho tốc độ gần 100 token/giây (không phải hơn 1.000) và `ttft` nhỏ hơn `ttft_content`; phân vị đúng.

Công cụ có sẵn: `vllm bench serve` với các dataset và `--max-concurrency` [9]; vLLM khuyên GuideLLM cho benchmark server production [9].

## 3.3 Tool calling và structured output trên model tự host

**Tool calling** trên model tự host phụ thuộc hai thứ: model được huấn luyện cho tool calling, và engine parse đúng định dạng model sinh ra. Trong vLLM, chọn parser theo họ model: chat template của Qwen2.5 đã hỗ trợ tool call kiểu Hermes nên dùng `--tool-call-parser hermes` [5]. Parser sai thì tool call xuất hiện như chữ thường trong `content` thay vì trong `tool_calls`. Test ít nhất: một câu cần tool, một câu không cần, một câu cần hai tool, kiểm `tool_calls` và `finish_reason`.

**Structured output** (constrained decoding) chặn ở mỗi bước những token làm output vi phạm schema. Trên model yếu, khác biệt rất lớn. Cùng Qwen2.5-3B, cùng 541 tin nhắn, cùng prompt:

Mỗi ô ghi: JSON hỏng / sai schema / đúng schema nhưng sai nội dung / đúng hoàn toàn, tổng 541.

| File | Không ràng buộc | `response_format` `json_schema` |
|---|---|---|
| Q2_K | 299 / 24 / 145 / 73 | 0 / 0 / 217 / 324 |
| Q4_K_M | 0 / 1 / 25 / 515 | 0 / 0 / 26 / 515 |
| Q8_0 | 0 / 3 / 17 / 521 | 0 / 0 / 20 / 521 |

Lần chạy này và lần đo ở [§1.4](#14-quantization-fp8-int4-gguf-awq-gptq) lệch nhau một câu ở Q2_K và Q4_K_M dù `temperature` 0: llama-server xử lý 8 request song song, và batch khác nhau cho kết quả số học khác nhau một chút.

Hai điều rút ra:

- **Ràng buộc bảo đảm cú pháp, không bảo đảm nội dung.** Với `json_schema`, không còn câu nào hỏng JSON hay sai schema ở cả ba file, nhưng số câu đúng của Q4_K_M và Q8_0 không đổi: các câu sai vẫn sai, chỉ là sai đúng định dạng. Vẫn cần eval nội dung.
- **Ràng buộc cứu được lỗi định dạng của model lượng tử hoá sâu, không cứu được hiểu sai.** Q2_K tăng từ 73 lên 324 câu đúng (59,9%), vẫn xa 515 của Q4_K_M. So các mức lượng tử hoá với đúng cấu hình structured output sẽ dùng ở production.

Schema dùng trong thí nghiệm giới hạn `intent` trong một enum và `order_id` theo pattern `^DH-[0-9]{6}$` hoặc `null`. Mỗi engine hỗ trợ một tập con JSON Schema khác nhau; kiểm các từ khoá bạn dùng (`pattern`, `anyOf`, `additionalProperties`) có được thực thi thật không, bằng một test cố tình vi phạm.

Trong vLLM, structured output có sẵn trong server OpenAI-compatible [4]; SGLang dùng XGrammar mặc định [11]; llama-server nhận `json_schema` và grammar [14]; Ollama nhận JSON schema ở trường `format` [16].

## 3.4 Cập nhật model và rollback

Model tự host không tự đổi, đó là ưu điểm, nhưng mọi thay đổi là việc của bạn:

- **Ghim mọi thứ.** Model theo commit hash (`--revision` trong vLLM nhận branch, tag hoặc commit [2]), engine theo phiên bản, image container theo digest, driver theo phiên bản. "Latest" ở bất kỳ tầng nào là một lần nâng cấp không kiểm soát.
- **Thay đổi engine cũng là thay đổi model.** Bản engine mới đổi kernel, mặc định, xử lý chat template; output có thể khác. Chạy eval khi nâng engine, như khi đổi model ([Evaluation §3.3](../evaluation/#33-regression-test-khi-đổi-prompt-hoặc-model)).
- **Đổi mức lượng tử hoá là đổi model** ([§1.4](#14-quantization-fp8-int4-gguf-awq-gptq)).
- **Triển khai song song rồi chuyển dần.** Chạy bản mới cạnh bản cũ, chuyển một phần traffic, so metric và eval online, giữ bản cũ đủ lâu để rollback được trong vài phút. Cách làm ở [Deploying AI Systems](../deploying-ai-systems/).
- **Tính thời gian khởi động.** Tải weights hàng chục GB, compile, capture CUDA graph mất vài phút. Lưu weights gần máy (ổ đĩa local hoặc cache trong cluster) và cache compile của vLLM [3] để rollback không phải chờ tải lại.

## 3.5 So sánh tổng chi phí với API

Công thức ở [Cost optimization §2.5](../cost-optimization/#25-self-host-khi-nào-rẻ-hơn-api). Một phép tính hoà vốn đơn giản cho gpt-oss-120b:

```
Một H100 on-demand:          ≈ 4 USD/giờ × 720 giờ = 2.880 USD/tháng        [27]
API serverless gpt-oss-120b:  0,15 USD input, 0,60 USD output mỗi triệu token [30]

Mỗi token output đi kèm r token input, nên API tính cho mỗi triệu token output:
                              0,60 + r × 0,15 USD
Hoà vốn (token output/tháng) = 2.880 USD / (0,60 + r × 0,15) USD mỗi triệu

r = 0 (chỉ output):  4,8 tỉ  ≈ 1.850 token output mỗi giây, liên tục 24/7
r = 5 (RAG, agent):  2,13 tỉ ≈ 820 token output mỗi giây, kèm ≈ 4.100 token input mỗi giây
```

Một GPU tự host phải đạt mức đó trung bình suốt cả tháng chỉ để bằng giá API. Traffic nhiều input hạ ngưỡng output, vì mỗi token input cũng là tiền API phải trả, nhưng GPU tự host phải prefill phần input đó; benchmark với đúng tỉ lệ input:output của bạn. Hai khoản chưa tính đều đẩy ngưỡng lên: dự phòng (ít nhất hai máy, tức gấp đôi chi phí GPU) và người vận hành. Con số throughput có thể đạt được trên một H100 phải lấy từ benchmark của bạn ở SLO của bạn ([§3.2](#32-benchmark-throughput-và-độ-trễ-trên-tải-thật)), không từ trang quảng cáo. Traffic thật dao động theo giờ trong ngày, nên mức trung bình nghĩa là giờ cao điểm cần nhiều hơn.

Phép tính đổi chiều khi: dữ liệu bắt buộc tự host; giá GPU reserved hoặc phần cứng sở hữu thấp hơn nhiều; model fine-tune riêng không có trên serverless; hoặc tải đủ lớn và đều để GPU luôn bận.

## 3.6 Failure modes thường gặp

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| Request đồng thời chờ rất lâu mới có token đầu | Ollama `OLLAMA_NUM_PARALLEL=1` [16]; llama-server ít slot | Tăng số slot hoặc chuyển sang vLLM, SGLang; đo lại ([§3.2](#32-benchmark-throughput-và-độ-trễ-trên-tải-thật)) |
| vLLM hết bộ nhớ khi khởi động | `--max-model-len` mặc định theo model quá lớn; tiến trình khác chiếm GPU trong khi mặc định dùng 92% [2] | Đặt `--max-model-len` theo nhu cầu; giảm `--gpu-memory-utilization` |
| Độ trễ tăng vọt lúc tải cao, log thống kê có `Preemptions: N` | Hết KV cache, vLLM tính lại request [3] | Theo dõi `vllm:num_preemptions`, `vllm:kv_cache_usage_perc`; thêm GPU, giảm `max_num_seqs`, lượng tử hoá KV |
| llama-server trả 400 `exceed_context_size_error` | `-c` chia cho các slot; mỗi slot nhỏ hơn bạn nghĩ | `-c` = context mỗi request × `-np` |
| Model trả lời sai hoặc bị cắt với prompt dài trên Ollama | Context mặc định 4k dưới 24 GiB VRAM [16] | Đặt `OLLAMA_CONTEXT_LENGTH`; kiểm bằng `ollama ps` |
| Output rác, JSON hỏng sau khi đổi sang file GGUF nhỏ hơn | Lượng tử hoá quá sâu ([§1.4](#14-quantization-fp8-int4-gguf-awq-gptq)) | Eval mọi mức lượng tử hoá; dùng structured output |
| Tool call nằm trong `content` thay vì `tool_calls` | Sai hoặc thiếu `--tool-call-parser` [5] | Chọn parser theo họ model; test tool call |
| Kết quả benchmark đẹp, production chậm | Đo với prompt lặp lại (dính prefix cache), output cố định ngắn, một mức đồng thời | Prompt thật, độ dài thật, nhiều mức đồng thời, p95 |
| FP8 không chạy hoặc chậm trên GPU cũ | GPU không hỗ trợ (FP8 W8A8 cần Ada, Hopper trong vLLM) [7] | Dùng W4A16 (AWQ, GPTQ) hoặc INT8 |
| Request đầu sau giờ rảnh rất chậm | Ollama gỡ model sau 5 phút [16]; autoscale về 0 | Tăng `keep_alive`; giữ tối thiểu một instance |
| Model không được phép dùng thương mại | Đọc tên họ model, không đọc license [38] | Kiểm license từng model, từng phiên bản |
| Output khác sau khi nâng engine | Kernel, mặc định, chat template đổi | Ghim phiên bản; chạy eval khi nâng |
| Hệ thống dựa trên TGI không có bản mới | TGI ở chế độ bảo trì [13] | Lên kế hoạch chuyển sang vLLM hoặc SGLang |

## 3.7 Checklist

Trước khi quyết định:

- [ ] Có lý do tự host cụ thể (dữ liệu, tải lớn và đều, model tuỳ biến, kiểm soát).
- [ ] Đã so với giá serverless của cùng model, không chỉ với API model frontier.
- [ ] Đã chọn model bằng eval trên tác vụ của mình, kể cả mức lượng tử hoá định dùng.
- [ ] Đã đọc license của đúng model và phiên bản.

Hạ tầng:

- [ ] Đã tính weights và KV cache cho context và số request đồng thời cần; còn dư cho overhead đo thật.
- [ ] GPU hỗ trợ định dạng lượng tử hoá đã chọn trên engine đã chọn.
- [ ] Đã benchmark trên phần cứng thật với prompt và độ dài output thật, nhiều mức đồng thời, có p95.
- [ ] Cấu hình engine viết rõ: `--max-model-len`, bộ nhớ, số request đồng thời, parser cho tool calling.

Vận hành:

- [ ] Model, engine, image, driver đều ghim phiên bản.
- [ ] Alert trên hàng đợi, KV cache, preemption, TTFT p95, lỗi.
- [ ] Ít nhất hai instance; biết thời gian khởi động một instance mới.
- [ ] Eval chạy lại khi đổi model, mức lượng tử hoá hoặc engine.
- [ ] Có đường rollback về bản cũ trong vài phút.
- [ ] Có fallback sang API khi hệ thống tự host quá tải hoặc sập ([Model routing](../model-routing/)).

---

## Nguồn tham khảo

Truy cập tháng 11/2026.

| # | Nguồn | Nội dung dùng trong bài |
|---|---|---|
| [1] | [Kwon và cộng sự, PagedAttention / vLLM (arXiv 2309.06180)](https://arxiv.org/abs/2309.06180) | KV cache theo block, chia sẻ giữa request, throughput 2–4 lần |
| [2] | vLLM source tại tag `v0.33.0`: [`vllm/config/`](https://github.com/vllm-project/vllm/tree/v0.33.0/vllm/config) (`cache.py`, `scheduler.py`, `model.py`, `parallel.py`) và [`vllm/engine/arg_utils.py`](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py) (`get_batch_defaults`, `--performance-mode`) | Mặc định `gpu_memory_utilization`, `max_num_seqs` và `max_num_batched_tokens` theo GPU, `enable_prefix_caching`, `enable_chunked_prefill`, `cache_dtype`, `quantization`, `max_model_len`, `revision`, tensor/pipeline parallel |
| [3] | [vLLM: Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) | Preemption, chunked prefill, `max_num_batched_tokens`, cache compile, `--kv-cache-memory`, `--enforce-eager` |
| [4] | [vLLM: Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) và [`vllm/sampling_params.py`](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/sampling_params.py) tại `v0.33.0` | Hỗ trợ mặc định; backend `auto` thử xgrammar rồi chuyển sang guidance |
| [5] | [vLLM: Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling/) | `--enable-auto-tool-choice`, các parser, Qwen2.5 dùng `hermes` |
| [6] | [vLLM: Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/) và [`vllm/v1/metrics/loggers.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py) | Endpoint `/metrics`, tên metric, dòng log `Preemptions: N` |
| [7] | [vLLM: Quantization](https://docs.vllm.ai/en/latest/features/quantization/) | Bảng hỗ trợ theo GPU, FP8 W8A8 trên Ada/Hopper |
| [8] | [vLLM: Installation](https://docs.vllm.ai/en/latest/getting_started/installation/) và [PyPI](https://pypi.org/project/vllm/) | Phần cứng hỗ trợ; bản 0.33.0 |
| [9] | [vLLM: Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/) | `vllm bench serve`, khuyến nghị GuideLLM |
| [10] | [Zheng và cộng sự, SGLang (arXiv 2312.07104)](https://arxiv.org/abs/2312.07104) | RadixAttention, throughput tới 6,4 lần |
| [11] | [SGLang docs: Server Arguments, Structured Outputs, Quantization](https://docs.sglang.ai/) và [README](https://github.com/sgl-project/sglang) | Bản 0.5.23, `--mem-fraction-static`, `--enable-metrics`, XGrammar mặc định, phần cứng, NVFP4 |
| [12] | [TensorRT-LLM: TensorRT backend removal](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/tensorrt-backend-removal.md), [Quantization](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md), [Quick start](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/quick-start-guide.md) | Backend PyTorch duy nhất từ 1.3.0rc21 (`setup.py` của `v1.3.0rc20` còn `trtllm-build`, của `v1.3.0rc21` thì không; trang removal có ở tag `v1.4.0rc3`, không có ở `v1.2.1`; `setup.py` của `v1.2.1` vẫn khai báo `trtllm-build`), `trtllm-serve`, FP4/FP8; PyPI: 1.2.1 stable, 1.4.0rc3 |
| [13] | [Hugging Face TGI README](https://github.com/huggingface/text-generation-inference) | Chế độ bảo trì, khuyến nghị thay thế |
| [14] | [llama.cpp: server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) và [releases](https://github.com/ggml-org/llama.cpp/releases) | `-np`, `-c`, `--kv-unified`, `--jinja`, grammar, `json_schema`; build b12180 |
| [15] | [llama.cpp: quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) và [build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) | Kiểu GGUF, Metal mặc định trên macOS |
| [16] | [Ollama: FAQ](https://docs.ollama.com/faq), [Context length](https://docs.ollama.com/context-length), [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility), [Structured outputs](https://docs.ollama.com/capabilities/structured-outputs), [releases](https://github.com/ollama/ollama/releases) | `OLLAMA_NUM_PARALLEL`, context mặc định, `keep_alive`, `OLLAMA_MAX_LOADED_MODELS`; bản 0.40.15 |
| [17] | [Agrawal và cộng sự, Sarathi-Serve (arXiv 2403.02310)](https://arxiv.org/abs/2403.02310) | Prefill và decode, chunked prefill |
| [18] | [Zhong và cộng sự, DistServe (arXiv 2401.09670)](https://arxiv.org/abs/2401.09670) | Tách prefill và decode ra GPU khác nhau |
| [19] | [Frantar và cộng sự, GPTQ (arXiv 2210.17323)](https://arxiv.org/abs/2210.17323) | 3–4 bit, 175B trong khoảng 4 giờ GPU |
| [20] | [Lin và cộng sự, AWQ (arXiv 2306.00978)](https://arxiv.org/abs/2306.00978) | Bảo vệ 1% weights quan trọng theo activation |
| [21] | [Xiao và cộng sự, SmoothQuant (arXiv 2211.10438)](https://arxiv.org/abs/2211.10438) | W8A8, 1,56 lần nhanh hơn, 2 lần ít bộ nhớ |
| [22] | [Dettmers và cộng sự, LLM.int8() (arXiv 2208.07339)](https://arxiv.org/abs/2208.07339) | INT8 với phân tách đặc trưng ngoại lai |
| [23] | [Kurtic và cộng sự, "Give Me BF16 or Give Me Death"? (arXiv 2411.02355)](https://arxiv.org/abs/2411.02355) | FP8 gần như không mất, INT8 1–3%, INT4 W4A16 cạnh tranh; W4A16 và W8A8 theo kiểu tải |
| [24] | NVIDIA: [L4](https://www.nvidia.com/en-us/data-center/l4/), [L40S](https://www.nvidia.com/en-us/data-center/l40s/), [A100](https://www.nvidia.com/en-us/data-center/a100/), [H100](https://www.nvidia.com/en-us/data-center/h100/), [H200](https://www.nvidia.com/en-us/data-center/h200/), [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/), [RTX PRO 6000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/), [RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) | Bộ nhớ, băng thông |
| [25] | AMD Instinct: [MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html), [MI325X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html), [MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) | Bộ nhớ, băng thông, FP8, MXFP4 |
| [26] | Apple: [M4 Pro và M4 Max](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/), [Mac Studio specs](https://www.apple.com/mac-studio/specs/) | M4 120 GB/s, M4 Max 546 GB/s và 128 GB, M5 Ultra 512 GB và 1,2 TB/s |
| [27] | [RunPod pricing](https://www.runpod.io/pricing) | H100 theo giờ |
| [28] | [Modal pricing](https://modal.com/pricing) | H100 theo giây |
| [29] | [CoreWeave pricing](https://www.coreweave.com/pricing) | HGX H100 8 GPU theo giờ |
| [30] | [Together AI pricing](https://www.together.ai/pricing) | GPU cluster theo giờ; gpt-oss-120b theo token |
| [31] | [Fireworks pricing](https://fireworks.ai/pricing) | GPU on-demand theo giờ; gpt-oss-120b theo token |
| [32] | [DeepInfra pricing](https://deepinfra.com/pricing) | GPU theo giờ; Llama 3.3 70B, Qwen3-32B theo token |
| [33] | [Llama 3.3 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE) | 700 triệu MAU, "Built with Llama", Acceptable Use Policy |
| [34] | [Llama 4 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE) | Cùng điều khoản 700 triệu MAU |
| [35] | [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license) | Apache 2.0 |
| [36] | [Gemma 3 model card](https://huggingface.co/google/gemma-3-27b-it) | `license: gemma` |
| [37] | [Qwen3-8B LICENSE](https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE) và [Qwen3.5 model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | Apache 2.0 |
| [38] | [Qwen2.5-3B-Instruct-GGUF LICENSE](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/blob/main/LICENSE) | Qwen Research License, chỉ phi thương mại |
| [39] | [DeepSeek-V3 LICENSE-MODEL](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL); model card [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324), [V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1), [V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) | DeepSeek License Agreement 1.0 cho bản gốc; `license: mit` cho các bản sau |
| [40] | [DeepSeek-R1 LICENSE](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/LICENSE) | MIT |
| [41] | [gpt-oss-120b LICENSE](https://huggingface.co/openai/gpt-oss-120b/blob/main/LICENSE) | Apache 2.0 |
| [42] | [Mistral Small 3.1 model card](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) | `license: apache-2.0` |
| [43] | `config.json` và Hugging Face API (`safetensors.total`) của [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B), [Llama-3.1-8B-Instruct](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct) và [Llama-3.3-70B-Instruct](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct) (bản mirror, repo gốc của Meta yêu cầu đăng nhập), [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); test thêm với [Gemma 3 27B](https://huggingface.co/unsloth/gemma-3-27b-it), [Gemma 2 9B](https://huggingface.co/unsloth/gemma-2-9b-it), [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) | Số lớp, KV head, head_dim, sliding window, số expert, số tham số, dung lượng checkpoint |
| [44] | Dung lượng `nvidia-smi` báo: L4 trong [Docker docs: GPU access](https://docs.docker.com/engine/containers/gpu/); H100 80GB HBM3 trong [NVIDIA DGX BasePOD deployment guide](https://docs.nvidia.com/dgx-basepod/deployment-guide-dgx-basepod/latest/container.html); H200 trong [Ori: Installing NVIDIA Drivers (SXM)](https://docs.ori.co/super-computers/nvidia-drivers-sxm/); RTX 5090 trong [NVIDIA open-gpu-kernel-modules issue #862](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/862) | 23.034, 81.559, 143.771, 32.607 MiB |
