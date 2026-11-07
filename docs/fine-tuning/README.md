# Fine-tuning: khi nào và làm thế nào

Chatbot của một cửa hàng điện máy trả lời sai chính sách đổi trả. Nhóm phát triển gom 2.000 câu hỏi và câu trả lời đúng từ lịch sử chat, fine-tune một model, và trên tập test model trả lời đúng chính sách hơn hẳn. Ba tháng sau cửa hàng đổi thời hạn đổi trả từ 7 lên 15 ngày. Model vẫn nói 7 ngày, tự tin như cũ, và muốn sửa thì phải gom dữ liệu rồi huấn luyện lại. Cùng lúc đó, nhà cung cấp thông báo model gốc sẽ ngừng chạy, kéo theo model đã fine-tune.

Kịch bản này không hiếm, và cả hai vấn đề đều đoán trước được. Fine-tune dạy model *cách* làm việc tốt hơn là dạy nó *biết* thêm điều gì. Kiến thức thay đổi thì nên nằm ngoài weights, trong prompt hoặc trong hệ thống retrieval. Model fine-tune cũng gắn chặt với một model gốc cụ thể và một nền tảng cụ thể. Bài này trả lời hai câu hỏi: khi nào fine-tune đáng làm, và nếu làm thì làm sao cho model tốt hơn thật, không chỉ tốt hơn trên tập test.

- **Nền tảng — Phần 1:** fine-tune thay đổi gì trong model, so với prompt và RAG, các phương pháp (SFT, DPO, RFT), LoRA và QLoRA, distillation.
- **Phương pháp và nền tảng — Phần 2:** fine-tune qua API (tình hình tháng 11/2026 khác nhiều so với một năm trước), fine-tune model open-weight bằng TRL, Unsloth, Axolotl, chọn base model, phần cứng và chi phí.
- **Thiết kế và vận hành — Phần 3:** chuẩn bị và chia dữ liệu kèm code đã chạy, dữ liệu tổng hợp, eval trước và sau với một lần huấn luyện thật, overfitting và catastrophic forgetting, serving LoRA adapter, bảo trì khi model gốc ngừng, failure modes và checklist.

> Bản tháng 11/2026. Code huấn luyện trong bài đã chạy thật với TRL 1.17.1, transformers 5.22.2, PEFT 0.21.2 và PyTorch 2.15 trên Apple M4 24 GB (backend MPS), model `Qwen/Qwen2.5-0.5B-Instruct`. Tên model, giá và lịch ngừng hỗ trợ thay đổi thường xuyên; đối chiếu trang của nhà cung cấp trước khi dùng. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API.

Tài liệu liên quan: [Evaluation](../evaluation/) (eval phải có trước khi fine-tune), [Prompt engineering](../prompt-engineering/) (phương án phải thử trước), [RAG nâng cao](../advanced-rag/) (cho bài toán kiến thức), [Cost optimization §2.4](../cost-optimization/#24-model-nhỏ-distillation-và-fine-tune) (fine-tune như một đòn bẩy chi phí), [Self-hosting LLM](../self-hosting-llm/) (serving model open-weight), [Guardrails](../guardrails/) (fine-tune có thể làm yếu safety của model gốc).

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Fine-tune thay đổi gì trong model và không thay đổi gì](#11-fine-tune-thay-đổi-gì-trong-model-và-không-thay-đổi-gì)
  - [1.2 Fine-tune, RAG và prompt: giải quyết bài toán khác nhau](#12-fine-tune-rag-và-prompt-giải-quyết-bài-toán-khác-nhau)
  - [1.3 SFT, preference tuning (DPO), reinforcement fine-tuning](#13-sft-preference-tuning-dpo-reinforcement-fine-tuning)
  - [1.4 Full fine-tune, LoRA và QLoRA](#14-full-fine-tune-lora-và-qlora)
  - [1.5 Distillation: dạy model nhỏ bắt chước model lớn](#15-distillation-dạy-model-nhỏ-bắt-chước-model-lớn)
- [Phần 2 — Phương pháp và nền tảng](#phần-2--phương-pháp-và-nền-tảng)
  - [2.1 Fine-tune qua API: OpenAI, Gemini/Vertex, Bedrock](#21-fine-tune-qua-api-openai-geminivertex-bedrock)
  - [2.2 Fine-tune open-weight: Hugging Face TRL, Unsloth, Axolotl](#22-fine-tune-open-weight-hugging-face-trl-unsloth-axolotl)
  - [2.3 Chọn base model](#23-chọn-base-model)
  - [2.4 Phần cứng và chi phí huấn luyện](#24-phần-cứng-và-chi-phí-huấn-luyện)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Chuẩn bị dữ liệu: nguồn, làm sạch, format](#31-chuẩn-bị-dữ-liệu-nguồn-làm-sạch-format)
  - [3.2 Sinh dữ liệu tổng hợp từ model lớn](#32-sinh-dữ-liệu-tổng-hợp-từ-model-lớn)
  - [3.3 Eval trước và sau fine-tune](#33-eval-trước-và-sau-fine-tune)
  - [3.4 Overfitting và catastrophic forgetting](#34-overfitting-và-catastrophic-forgetting)
  - [3.5 Deploy và serving LoRA adapter](#35-deploy-và-serving-lora-adapter)
  - [3.6 Bảo trì: khi base model ra phiên bản mới](#36-bảo-trì-khi-base-model-ra-phiên-bản-mới)
  - [3.7 Failure modes thường gặp](#37-failure-modes-thường-gặp)
  - [3.8 Checklist](#38-checklist)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

### Có nên fine-tune không

Đi lần lượt, dừng ở bước đầu tiên giải quyết được vấn đề:

1. **Chưa có eval thì chưa fine-tune.** Không có tập test và cách chấm, bạn không biết model fine-tune tốt hơn hay chỉ khác đi. OpenAI ghi thẳng trong hướng dẫn: chỉ đầu tư vào fine-tune sau khi đã có eval [1]. Cách xây eval ở [Evaluation](../evaluation/).
2. **Model thiếu kiến thức, hoặc kiến thức hay thay đổi (giá, chính sách, tồn kho, tài liệu nội bộ): dùng RAG hoặc đưa vào prompt.** Fine-tune nạp kiến thức mới kém: RAG thắng fine-tune không giám sát một cách nhất quán, cả với kiến thức model đã gặp lẫn kiến thức mới [23], và ví dụ chứa kiến thức mới làm tăng xu hướng bịa của model khi nó học được chúng [24].
3. **Prompt tốt hơn, vài ví dụ trong prompt, model mạnh hơn.** Thử với model mới nhất trước khi kết luận prompt không đủ ([Prompt engineering](../prompt-engineering/)).
4. **Cân nhắc fine-tune khi:** output phải theo một format hoặc giọng văn ổn định mà prompt không giữ được; prompt đã dài vì chứa nhiều ví dụ và bạn muốn bỏ chúng đi; cần một model nhỏ, rẻ, nhanh làm tốt đúng một tác vụ hẹp; hoặc phải chạy model trên hạ tầng của mình. Hai trường hợp đầu là những gì Google liệt kê: format và hành vi khó mô tả bằng prompt, prompt cho kết quả không ổn định, và bỏ ví dụ few-shot để rút ngắn context [6].
5. **Có cách chấm đúng sai bằng code nhưng không có sẵn câu trả lời mẫu:** reinforcement fine-tuning ([§1.3](#13-sft-preference-tuning-dpo-reinforcement-fine-tuning)).

### Nền tảng fine-tune, tháng 11/2026

| Nền tảng | Tình trạng | Ghi chú |
|---|---|---|
| OpenAI | Đang đóng | Không nhận người dùng mới. Khách hàng cũ đủ điều kiện tạo job tới 06/01/2027. Model fine-tune chạy tới khi model gốc ngừng: `gpt-4.1-nano` đã ngừng 23/10/2026, `gpt-4.1` và `gpt-4.1-mini` ngừng 14/01/2027 [1][2] |
| Claude API | Không có | "The Claude API does not currently offer fine-tuning" [3]. Model Claude duy nhất từng fine-tune được trên Bedrock là Claude 3 Haiku, đã hết vòng đời 10/09/2026 [10] |
| Gemini API, AI Studio | Không có | Không còn model nào hỗ trợ từ khi Gemini 1.5 Flash-001 ngừng (05/2025) [5] |
| Vertex AI (Gemini Enterprise Agent Platform) | Có | SFT cho Gemini 3.5 Flash, 3.5 Flash-Lite, 3.1 Flash-Lite, 2.5 Pro, 2.5 Flash, 2.5 Flash-Lite (LoRA, adapter size 1–16); RL fine-tuning cho 3.5 Flash; preference tuning cho 2.5 Flash, 2.5 Flash-Lite; SFT cho một số model open-weight như Gemma 3, Llama 3.x, Llama 4 Scout, Qwen 3 [6][11] |
| Amazon Bedrock | Có | SFT cho Nova, Llama 3.x; RFT cho Nova 2 Lite, gpt-oss-20B, Qwen3 32B [8][9] |
| Together, Fireworks | Có | Model open-weight; tính theo token huấn luyện [12][13] |
| Tự huấn luyện | Có | TRL, Unsloth, Axolotl trên GPU của bạn hoặc thuê ([§2.2](#22-fine-tune-open-weight-hugging-face-trl-unsloth-axolotl)) |

Nếu đang dùng Claude hoặc model mới của OpenAI, fine-tune nghĩa là đổi sang model khác: Gemini trên Vertex, hoặc một model open-weight. Phần lớn lý do "fine-tune để tốt hơn" nên được so với phương án ở lại model mạnh và cải thiện prompt.

### Số ví dụ để bắt đầu

| Nguồn | Khuyến nghị |
|---|---|
| OpenAI SFT | Tối thiểu 10; thường thấy cải thiện với 50–100; bắt đầu với 50 ví dụ tốt [1] |
| Vertex AI, Gemini SFT | 100–500 ví dụ để có kết quả tốt [7] |
| Bedrock, Claude 3 Haiku (đã ngừng) | 50–100 dòng chất lượng cao để bắt đầu [10] |
| Bedrock RFT | 100–200 ví dụ, kiểm hàm reward trước rồi mới tăng [9] |
| LIMA (nghiên cứu) | 1.000 cặp prompt–response chọn kỹ đủ để model 65B theo được format [21] |

Nếu 50 ví dụ tốt không cải thiện gì, OpenAI khuyên xem lại tác vụ hoặc prompt trước khi thêm dữ liệu [1].

### Mặc định của TRL `SFTConfig` cần biết (bản 1.17.1)

| Tham số | Mặc định | Hệ quả nếu để nguyên |
|---|---|---|
| `max_length` | `1024` | Ví dụ dài hơn bị cắt ([§3.1](#31-chuẩn-bị-dữ-liệu-nguồn-làm-sạch-format)) |
| `truncation_mode` | `keep_start` | Cắt phần cuối, tức là cắt câu trả lời và token kết thúc |
| `assistant_only_loss` | `False` | Với dữ liệu hội thoại, loss tính trên cả system và user |
| `learning_rate` | `2e-5` | Hợp với full fine-tune; LoRA thường cần khoảng `1e-4` [15][16] |
| `num_train_epochs` | `3` | Dữ liệu nhỏ dễ overfit ([§3.4](#34-overfitting-và-catastrophic-forgetting)) |

Giá trị đọc từ `SFTConfig(output_dir=...)` của TRL 1.17.1 đã cài [15].

---

# Phần 1 — Nền tảng

## 1.1 Fine-tune thay đổi gì trong model và không thay đổi gì

Fine-tune là huấn luyện tiếp một model đã huấn luyện, trên dữ liệu của bạn, để model bắt chước các mẫu trong dữ liệu đó [3]. Kết quả là một bộ weights mới (hoặc một adapter gắn lên weights cũ). Mọi thứ khác của model giữ nguyên: context window, tokenizer, kiến trúc, tốc độ sinh mỗi token.

Fine-tune làm tốt:

- **Format và cấu trúc output**: luôn trả JSON đúng schema, luôn theo một mẫu tóm tắt, thay tên người bằng `#Person1` như ví dụ của Google [6].
- **Hành vi ổn định**: trả lời ngắn hay dài, khi nào từ chối, giữ giọng của một thương hiệu.
- **Tác vụ hẹp**: phân loại, trích xuất, định tuyến, nơi model nhỏ đã fine-tune có thể thay model lớn.
- **Rút ngắn prompt**: bỏ được phần lớn ví dụ few-shot, giảm token input mỗi request [6].

Fine-tune làm kém hoặc gây hại:

- **Nạp kiến thức mới.** Ovadia và cộng sự so fine-tune không giám sát với RAG trên nhiều tác vụ cần kiến thức: RAG thắng nhất quán, và model khó học fact mới qua fine-tune [23]. Gekhman và cộng sự đi xa hơn: ví dụ chứa kiến thức mới được học chậm hơn hẳn ví dụ khớp với kiến thức sẵn có, và khi cuối cùng được học, chúng làm xu hướng bịa của model tăng tuyến tính [24]. Kết luận của họ: model lấy kiến thức chủ yếu từ pretraining, còn fine-tune dạy nó dùng kiến thức đó hiệu quả hơn.
- **Giữ nguyên năng lực ngoài tác vụ.** Fine-tune liên tiếp làm model quên kiến thức, khả năng suy luận và đọc hiểu, ở các model từ 1B tới 7B [26] ([§3.4](#34-overfitting-và-catastrophic-forgetting)).
- **Giữ nguyên safety.** Qi và cộng sự phá được safety của GPT-3.5 Turbo bằng 10 ví dụ thiết kế đối kháng, tốn dưới 0,20 USD qua API fine-tune. Đáng lo hơn: fine-tune trên dữ liệu lành tính, phổ biến cũng làm safety giảm, ở mức nhẹ hơn [27]. Model fine-tune cần được test lại phần từ chối và guardrail, không thừa hưởng kết quả test của model gốc.

## 1.2 Fine-tune, RAG và prompt: giải quyết bài toán khác nhau

| Vấn đề | Công cụ hợp | Vì sao |
|---|---|---|
| Model không biết chính sách, giá, tài liệu của bạn | RAG, hoặc đưa vào prompt | Fine-tune nạp kiến thức mới kém hơn RAG và làm tăng xu hướng bịa [23][24]; kiến thức nằm ngoài weights thì sửa được ngay |
| Model biết nhưng trả lời sai format, dài dòng, sai giọng | Prompt trước, fine-tune nếu prompt không giữ được | Fine-tune giỏi dạy hành vi [6] |
| Prompt dài vì 30 ví dụ few-shot, tốn tiền và chậm | Prompt caching trước; fine-tune nếu vẫn đắt | Cache giảm giá phần lặp lại mà không cần huấn luyện ([Cost optimization §2.2](../cost-optimization/#22-prompt-caching)) |
| Model lớn đủ tốt nhưng quá đắt hoặc chậm cho tác vụ hẹp | Fine-tune model nhỏ, hoặc distillation | Model nhỏ chuyên một việc có thể bằng model lớn ở việc đó ([§1.5](#15-distillation-dạy-model-nhỏ-bắt-chước-model-lớn)) |
| Dữ liệu không được rời hạ tầng | Model open-weight, fine-tune và chạy tại chỗ | Không có API nào thoả |
| Tác vụ có đáp án kiểm được bằng code, không có lời giải mẫu | Reinforcement fine-tuning (Bedrock, Vertex) | Chỉ cần hàm chấm, không cần câu trả lời mẫu [9] |

Ba cách không loại trừ nhau. Một hệ thống thường dùng model đã fine-tune để theo format, RAG để có kiến thức đúng, và prompt để nói cho model biết tình huống hiện tại.

## 1.3 SFT, preference tuning (DPO), reinforcement fine-tuning

**Supervised fine-tuning (SFT).** Dữ liệu là cặp đầu vào và câu trả lời mong muốn. Model được huấn luyện để tăng xác suất sinh đúng câu trả lời đó. Đây là cách phổ biến nhất và là cách duy nhất có trên mọi nền tảng ở bảng trên. SFT hợp với phân loại, dịch có sắc thái, sinh nội dung theo format, sửa lỗi không theo chỉ dẫn [1].

**Preference tuning (DPO).** Dữ liệu là bộ ba: đầu vào, câu trả lời được chọn, câu trả lời bị loại. RLHF truyền thống huấn luyện một reward model rồi dùng reinforcement learning; DPO viết lại bài toán để giải bằng một loss phân loại đơn giản, không cần sinh mẫu từ model trong lúc huấn luyện, và đạt bằng hoặc hơn RLHF dùng PPO trong thí nghiệm của tác giả [22]. DPO hợp khi bạn dễ nói "câu A tốt hơn câu B" hơn là viết ra câu trả lời hoàn hảo. Together tính phí DPO khoảng 2,5 lần SFT cho cùng model [12].

**Reinforcement fine-tuning (RFT).** Dữ liệu chỉ là đầu vào; bạn cung cấp một hàm chấm điểm (reward). Model sinh nhiều câu trả lời cho mỗi đầu vào, hàm chấm cho điểm, và model được cập nhật để tăng điểm. Bedrock dùng thuật toán GRPO và cho viết hàm chấm bằng AWS Lambda hoặc dùng model làm giám khảo [9]. Hướng dẫn của Bedrock có hai mốc dễ áp dụng: nếu điểm reward của model gốc luôn bằng 0, làm SFT trước để model có năng lực cơ bản; nếu trên 95%, RFT có thể không cần [9]. Vertex có RL fine-tuning cho Gemini 3.5 Flash [11], và Google hướng người dùng sang cách này khi muốn cải thiện suy luận nhiều bước, vì SFT dạy model bỏ qua phần suy nghĩ [6].

| | SFT | DPO | RFT |
|---|---|---|---|
| Dữ liệu | Đầu vào + câu trả lời mẫu | Đầu vào + câu tốt + câu kém | Đầu vào + hàm chấm |
| Hợp khi | Biết câu trả lời đúng trông thế nào | Dễ so sánh hơn dễ viết | Kiểm được đúng sai bằng code hoặc judge |
| Rủi ro chính | Model chép cả lỗi trong câu trả lời mẫu | Cặp so sánh nhiễu | Model tìm cách ăn điểm mà không làm đúng (reward hacking) |
| TRL trainer | `SFTTrainer` | `DPOTrainer` | `GRPOTrainer`, `RLOOTrainer` |

## 1.4 Full fine-tune, LoRA và QLoRA

**Full fine-tune** cập nhật mọi tham số. Với Adam và mixed precision, mỗi tham số cần 16 byte cho riêng trạng thái huấn luyện: 2 byte weights fp16, 2 byte gradient, 12 byte cho bản fp32 của weights cùng hai moment của Adam. Bài báo ZeRO tính ví dụ: GPT-2 1,5 tỉ tham số cần ít nhất 24 GB, trong khi weights fp16 chỉ 3 GB [28]. Chưa tính activation. Một model 8B theo công thức này cần khoảng 128 GB chỉ cho trạng thái huấn luyện.

**LoRA** giữ nguyên weights gốc và thêm vào mỗi lớp một cặp ma trận hạng thấp; chỉ cặp ma trận này được huấn luyện. Với GPT-3 175B, LoRA giảm số tham số huấn luyện 10.000 lần và bộ nhớ GPU 3 lần so với fine-tune bằng Adam, chất lượng ngang hoặc hơn trên các model thử, và không thêm độ trễ khi inference vì adapter gộp được vào weights [17]. Trong thí nghiệm ở [§3.3](#33-eval-trước-và-sau-fine-tune), LoRA rank 16 trên mọi lớp linear của Qwen2.5-0.5B chỉ huấn luyện 8,8 triệu trên 502,8 triệu tham số (1,75%).

**QLoRA** lượng tử hoá model gốc xuống 4-bit (kiểu NF4) và huấn luyện LoRA trên đó, cộng double quantization và paged optimizer để giảm đỉnh bộ nhớ. Kết quả: fine-tune model 65B trên một GPU 48 GB mà vẫn giữ chất lượng fine-tune 16-bit [18].

LoRA có làm tốt bằng full fine-tune không? Hai nghiên cứu cho câu trả lời có điều kiện:

- Biderman và cộng sự so trên code và toán, với khoảng 100 nghìn cặp instruction và với 20 tỉ token continued pretraining: ở cấu hình rank thấp thông thường, LoRA kém full fine-tune đáng kể, nhưng giữ năng lực ngoài miền tốt hơn. Full fine-tune học các thay đổi có rank lớn hơn cấu hình LoRA thường dùng 10–100 lần [25]. Tên bài báo tóm tắt kết quả: LoRA học ít hơn và quên ít hơn.
- Thinking Machines (Schulman và cộng sự, 2025) cho thấy LoRA đạt hiệu quả mẫu và chất lượng cuối như full fine-tune với dữ liệu ở quy mô post-training, nếu cấu hình đúng: áp LoRA lên mọi lớp, nhất là lớp MLP (LoRA chỉ trên attention kém hơn kể cả khi tăng rank cho bằng số tham số); learning rate tối ưu khoảng 10 lần learning rate của full fine-tune; batch size không quá lớn, vì LoRA chịu batch lớn kém hơn. Dữ liệu vượt dung lượng của adapter thì LoRA thua [16].

Trong thực tế, LoRA là mặc định hợp lý cho SFT trên dữ liệu vài trăm tới vài chục nghìn ví dụ: rẻ, ít quên, adapter nhỏ dễ lưu và dễ đổi. Full fine-tune đáng cân nhắc khi dữ liệu rất lớn hoặc khi đang dạy một miền mới hẳn.

## 1.5 Distillation: dạy model nhỏ bắt chước model lớn

Distillation dùng model lớn (teacher) sinh câu trả lời cho tập đầu vào của bạn, rồi fine-tune model nhỏ (student) trên các câu trả lời đó. Bedrock tự động hoá quy trình: bạn chọn teacher và student, đưa prompt, Bedrock sinh response từ teacher rồi fine-tune student [8]. Lợi ích là chi phí và độ trễ của model nhỏ trên đúng tác vụ của bạn. Cái giá là student chỉ tốt bằng teacher trên phân phối đầu vào bạn đưa, và học cả lỗi của teacher.

Trước khi dùng output của một model thương mại để huấn luyện model khác, đọc điều khoản:

- **OpenAI**: không dùng output để phát triển model AI cạnh tranh với sản phẩm của OpenAI, trừ hai ngoại lệ: model phân loại, sắp xếp dữ liệu (embedding, classifier) không phân phối hay bán cho bên thứ ba; và fine-tune model trên chính nền tảng của OpenAI [29].
- **Anthropic**: không truy cập dịch vụ để xây sản phẩm cạnh tranh, kể cả huấn luyện model AI cạnh tranh, trừ khi Anthropic đồng ý [30].
- **Google Gemini API**: không dùng dịch vụ để phát triển model cạnh tranh với dịch vụ [31].

"Cạnh tranh" là khái niệm pháp lý, không phải kỹ thuật. Một student tổng quát dùng làm chatbot có rủi ro khác một classifier nội bộ. Hỏi bộ phận pháp lý trước khi đưa model distill ra sản phẩm. Teacher là model open-weight thì kiểm tra license của model đó.

---

# Phần 2 — Phương pháp và nền tảng

## 2.1 Fine-tune qua API: OpenAI, Gemini/Vertex, Bedrock

### OpenAI: đang đóng nền tảng

OpenAI thông báo ngày 07/05/2026 rằng nền tảng fine-tune tự phục vụ sẽ thu hẹp dần [2]:

| Ngày | Thay đổi |
|---|---|
| 07/05/2026 | Tổ chức chưa từng fine-tune không tạo được job |
| 02/07/2026 | Tổ chức không chạy inference trên model fine-tune trong 60 ngày qua không tạo được job |
| 23/10/2026 | `gpt-4.1-nano` (và bản fine-tune của nó), `o4-mini` ngừng |
| 06/01/2027 | Khách hàng còn hoạt động không tạo được job mới |
| 14/01/2027 | `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini` và các bản fine-tune ngừng |

Model fine-tune chạy được tới khi model gốc ngừng; bản fine-tune theo đúng lịch của model gốc [2]. Nếu bạn đang có model fine-tune trên OpenAI, đây là việc cần lên kế hoạch ngay: chọn đích chuyển sang, giữ dữ liệu và eval để huấn luyện lại ([§3.6](#36-bảo-trì-khi-base-model-ra-phiên-bản-mới)). Giá hiện hành: huấn luyện `gpt-4.1` 25 USD mỗi triệu token, inference 3 USD input và 12 USD output, so với 2 và 8 USD của model gốc [4]. Model fine-tune đắt hơn model gốc khi chạy.

Hướng dẫn của OpenAI vẫn đáng đọc vì áp dụng cho mọi nền tảng: tối thiểu 10 ví dụ, bắt đầu với 50 ví dụ tốt, giữ lại một phần dữ liệu làm tập đối chứng có cùng độ đa dạng với tập train [1].

### Claude: không có fine-tune qua API

Claude API không có fine-tune; tài liệu khuyên liên hệ Anthropic nếu cần [3]. Trên Bedrock, Claude 3 Haiku là model Claude duy nhất trong bảng model fine-tune được, chỉ ở `us-west-2` [8], và AWS ghi model này đã hết vòng đời ngày 10/09/2026 [10]. Với Claude, các đòn bẩy thay fine-tune là prompt, ví dụ trong prompt cùng prompt caching, structured output, và chọn đúng model.

### Gemini: chỉ trên Vertex AI

Gemini API và AI Studio không còn model nào fine-tune được, và Google chưa có kế hoạch đưa lại [5]. Fine-tune Gemini làm trên Vertex AI, nay mang tên Gemini Enterprise Agent Platform [6]:

| Thông số | Giá trị [6][7] |
|---|---|
| Model | Gemini 3.5 Flash, 3.5 Flash-Lite, 3.1 Flash-Lite, 2.5 Pro, 2.5 Flash, 2.5 Flash-Lite |
| Phương pháp | SFT bằng LoRA, adapter size 1, 2, 4, 8, 16 (2.5 Pro: tới 8); RL fine-tuning (3.5 Flash) và preference tuning (2.5 Flash, 2.5 Flash-Lite) có tài liệu và bảng giá riêng [11] |
| Dữ liệu | JSONL trên Cloud Storage, tối đa 1 GB; tối đa 131.072 token input + output mỗi ví dụ |
| Format | `systemInstruction` (tuỳ chọn) và `contents` gồm các lượt `role` / `parts` |
| Tập validation | Tối đa 5.000 ví dụ, hoặc 30% số ví dụ train nếu có hơn 1.000 ví dụ validation |
| Region | Gemini 3.x chỉ tune ở multi-region `us` và `eu` |
| SLA | Supervised tuning không thuộc dịch vụ có SLA |

Ba điểm dễ bỏ sót:

- **Thinking.** Google khuyên đặt thinking budget `0` (Gemini 2.5) hoặc thinking level `MINIMAL` (Gemini 3) khi dùng model đã tune, vì SFT dạy model trả lời mà bỏ qua phần suy nghĩ [6].
- **Structured output.** Bật controlled generation khi gọi model đã tune có thể làm giảm chất lượng, vì lúc tune không có controlled generation. Google khuyên không bật, để model tự sinh đúng format đã học [6].
- **Giá inference.** Từ Gemini 3, endpoint của model đã tune có giá bằng 1,5 lần model gốc ở region không phải `global`, và bằng model gốc ở `global` [11]. Region `global` lại không hỗ trợ CMEK và data residency, và adapter của model `global` được lưu ở Mỹ [6]. Chọn region là chọn giữa giá và yêu cầu dữ liệu.

Giá huấn luyện tính theo token huấn luyện, bằng số token của tập train nhân số epoch: Gemini 3.5 Flash 0,01 USD mỗi 1.000 token (10 USD mỗi triệu), 3.5 Flash-Lite 0,003 USD, 2.5 Pro 0,025 USD mỗi 1.000 token [11].

### Amazon Bedrock

Bedrock có ba cách tuỳ biến: supervised fine-tuning, reinforcement fine-tuning và distillation [8]. Model SFT gồm Amazon Nova (Micro, Lite, Pro, Nova 2 Lite), Llama 3.1, 3.2, 3.3 và Claude 3 Haiku đã hết vòng đời; hầu hết chỉ ở `us-east-1` hoặc `us-west-2` [8]. RFT hỗ trợ Nova 2 Lite, `openai.gpt-oss-20b` và `qwen.qwen3-32b` [9]. Phí huấn luyện tính theo số token của tập dữ liệu nhân số epoch, cộng phí lưu model theo tháng [8].

### Together, Fireworks: model open-weight không cần tự quản GPU

Together tính theo triệu token, bằng token của tập train nhân số epoch cộng token của tập eval nhân số lần eval, mỗi job có mức phí tối thiểu. Ví dụ ở bảng giá LoRA: Llama 3.1 8B và Qwen3.5 9B 0,34 USD SFT, 0,84 USD DPO, tối thiểu 4 USD mỗi job; Llama 3.3 70B 2,03 USD SFT [12]. Fireworks chỉ phục vụ model đã huấn luyện trên deployment on-demand (dedicated) [13], nghĩa là bạn trả tiền GPU theo giờ khi chạy model, không trả theo token như model serverless.

## 2.2 Fine-tune open-weight: Hugging Face TRL, Unsloth, Axolotl

| Công cụ | Là gì | Hợp khi |
|---|---|---|
| TRL | Thư viện của Hugging Face: `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, `KTOTrainer`, `RLOOTrainer`, tích hợp PEFT, vLLM, DeepSpeed [14] | Viết code Python, muốn kiểm soát từng bước |
| Unsloth | Thư viện tối ưu kernel cho fine-tune; tự công bố nhanh gấp 2 lần, ít hơn 70% VRAM, không giảm độ chính xác; chạy trên NVIDIA, AMD, Intel, CPU [19] | Một GPU, muốn chạy model lớn nhất có thể trên GPU đó |
| Axolotl | Huấn luyện theo file cấu hình YAML; hỗ trợ FSDP2, DeepSpeed, expert parallelism cho MoE, QLoRA [20] | Nhiều GPU, nhiều thí nghiệm, muốn cấu hình thay code |
| torchtune | Thư viện của PyTorch | Không còn được phát triển tích cực từ 2025 [32]; không chọn cho dự án mới |

Số liệu tốc độ của Unsloth là do Unsloth công bố; đo lại trên model và GPU của bạn trước khi dùng nó để ước tính chi phí.

Code trong bài dùng TRL vì nó ít trừu tượng nhất: thấy được dataset sau khi xử lý, token nào được tính loss, checkpoint nào được giữ. Unsloth và Axolotl đều dựa trên cùng các khái niệm, nên những gì kiểm được ở đây (chat template, mask, độ dài) áp dụng cho cả hai.

**Chat template và token kết thúc.** Model instruct học khi nào dừng nhờ token kết thúc lượt trong chat template, ví dụ `<|im_end|>` của Qwen. Nếu token đó không nằm trong phần được tính loss, model không học được cách dừng và sinh tiếp tới `max_tokens`. TRL xử lý việc này cho các họ model nó nhận ra: với `assistant_only_loss=True`, TRL thay chat template bằng bản có đánh dấu `{% generation %}` bao quanh phần model phải sinh, gồm cả token kết thúc [15][33]. Danh sách họ model được nhận ra gồm Qwen2.5, Qwen3, Llama 3, Gemma, Phi-3, GPT-OSS, DeepSeek-V3 và nhiều họ khác; model ngoài danh sách thì bạn phải tự thêm các đánh dấu đó [33]. Khi fine-tune một base model (chưa instruct), TRL cần được chỉ rõ `eos_token` khớp với chat template, ví dụ `<|im_end|>` cho `Qwen/Qwen2.5-1.5B` [15]. Test ở [§3.1](#31-chuẩn-bị-dữ-liệu-nguồn-làm-sạch-format) kiểm điều này trên token thật.

## 2.3 Chọn base model

- **Đích serving quyết định trước.** Model sẽ chạy ở đâu (Vertex, Bedrock, vLLM trên GPU của bạn, thiết bị của người dùng) giới hạn kích thước và kiến trúc. vLLM hỗ trợ LoRA cho nhiều kiến trúc nhưng có trần rank mặc định ([§3.5](#35-deploy-và-serving-lora-adapter)).
- **Instruct hay base.** Bắt đầu từ bản instruct khi tác vụ là hội thoại hoặc theo chỉ dẫn: model đã biết chat template và biết dừng. Base model cần nhiều dữ liệu hơn và phải cấu hình `eos_token` [15].
- **Nhỏ nhất đạt yêu cầu.** Fine-tune vài kích thước (ví dụ 0,5B, 3B, 8B) trên cùng dữ liệu và so trên cùng tập test. Model nhỏ rẻ hơn ở mọi lần gọi, và chênh lệch chất lượng sau fine-tune trên tác vụ hẹp thường nhỏ hơn trước fine-tune. Điều này phải đo, không giả định.
- **License.** Đọc license của model trước khi bỏ công: một số license giới hạn dùng thương mại, số người dùng, hoặc bắt dẫn tên model gốc.
- **Tokenizer với tiếng Việt.** Tokenizer huấn luyện chủ yếu trên tiếng Anh tốn nhiều token hơn cho tiếng Việt. Đo bằng tokenizer Qwen2.5, câu "Máy giặt mua ở đơn DH-914332 kêu rất to khi vắt, mình muốn hỏi còn bảo hành không và mang tới đâu để sửa ạ?" có 107 ký tự, 41 token (2,6 ký tự mỗi token); bản tiếng Anh cùng nội dung "The washing machine from order DH-914332 is very loud during the spin cycle. Is it still under warranty, and where should I take it for repair?" có 143 ký tự, 36 token (4,0 ký tự mỗi token). Câu tiếng Việt ngắn hơn về ký tự nhưng tốn nhiều token hơn. Một câu không đại diện cho cả ngôn ngữ, nhưng đủ để thấy nên đo số token trên dữ liệu của bạn khi so model: nhiều token hơn nghĩa là huấn luyện đắt hơn, ví dụ dễ chạm `max_length` hơn, và inference chậm hơn.
- **Đúng phiên bản.** Adapter LoRA chỉ dùng được với đúng model gốc nó được huấn luyện. Ghi lại tên và revision (commit hash) của model gốc cùng adapter.

## 2.4 Phần cứng và chi phí huấn luyện

### Bộ nhớ

| Cách | Trạng thái huấn luyện, chưa tính activation | Nguồn |
|---|---|---|
| Full fine-tune, Adam, mixed precision | 16 byte mỗi tham số: model 8B khoảng 128 GB | [28] |
| LoRA trên model bf16 | 2 byte mỗi tham số cho weights gốc, cộng trạng thái của adapter (nhỏ) | [17] |
| QLoRA | Weights gốc 4-bit: model 65B vừa một GPU 48 GB | [18] |

Activation tăng theo độ dài chuỗi và batch size. Gradient checkpointing (TRL bật mặc định [15]) đổi thời gian lấy bộ nhớ bằng cách tính lại activation khi backward.

Số đo thật: LoRA rank 16 cho Qwen2.5-0.5B-Instruct ở bf16, batch 8, chuỗi dài tối đa 122 token, gradient checkpointing tắt, trên Apple M4: 361 ví dụ × 2 epoch mất khoảng 150 giây (145–152 giây qua các lần chạy), bộ nhớ MPS driver cấp phát đo sau khi huấn luyện là 6,2 GiB ([§3.3](#33-eval-trước-và-sau-fine-tune)). Laptop đủ để thử pipeline với model nhỏ. Model vài tỉ tham số trở lên cần GPU rời hoặc thuê.

### Chi phí: ước lượng trước khi chạy

Với dịch vụ managed, chi phí huấn luyện bằng:

```
token huấn luyện = tổng token của tập train × số epoch (+ token eval nếu có tính)
chi phí = token huấn luyện × giá mỗi token, và không thấp hơn phí tối thiểu của job
```

Đây là cách tính của Vertex [11], Together [12] và Bedrock [8]. Ví dụ 2.000 ví dụ, trung bình 500 token, 3 epoch: 3 triệu token huấn luyện. Trên Together với Llama 3.1 8B LoRA: 3 × 0,34 = 1,02 USD, dưới mức tối thiểu nên trả 4 USD [12]. Trên Vertex với Gemini 3.5 Flash: 3 × 10 = 30 USD [11].

Phí huấn luyện thường là phần nhỏ. Các phần lớn hơn:

- **Mỗi lần thử là một lần huấn luyện.** Đổi dữ liệu, learning rate, số epoch; một dự án bình thường chạy hàng chục job.
- **Chi phí inference thay đổi.** Model fine-tune có thể đắt hơn model gốc: OpenAI 3/12 USD so với 2/8 USD [4], Vertex 1,5 lần ở region không phải `global` [11]. Fireworks bắt chạy trên deployment riêng [13]. Tự host thì trả GPU kể cả lúc không có request.
- **Dữ liệu và eval**: gán nhãn, review, xây tập test. Đây thường là phần đắt nhất (xem [Cost optimization §2.4](../cost-optimization/#24-model-nhỏ-distillation-và-fine-tune)).
- **Huấn luyện lại** khi model gốc ngừng hoặc dữ liệu đổi ([§3.6](#36-bảo-trì-khi-base-model-ra-phiên-bản-mới)).

So sánh luôn với phương án không fine-tune trên tổng chi phí trong một năm, không chỉ phí một job.

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Chuẩn bị dữ liệu: nguồn, làm sạch, format

### Nguồn

Dữ liệu tốt nhất là đầu vào thật từ production cùng câu trả lời đã được người có chuyên môn duyệt. OpenAI liệt kê dữ liệu lịch sử, dữ liệu chuyên gia, dữ liệu log [1]. Ví dụ phải giống những gì model sẽ gặp: cùng độ dài, cùng kiểu viết (viết tắt, không dấu, sai chính tả), cùng tỉ lệ các loại câu hỏi. Câu trả lời mẫu dạy model mọi thứ trong nó, cả lỗi: một câu trả lời sai chính sách trong tập train là một lần dạy model nói sai.

### Format

Mỗi nền tảng có format riêng nhưng cùng ý: một dòng JSON mỗi ví dụ, một danh sách lượt hội thoại, lượt cuối là câu trả lời mong muốn. OpenAI dùng format chat completions với `messages` [1]; Vertex dùng `systemInstruction` và `contents` với `parts` [7]; TRL nhận `messages` theo format hội thoại và tự áp chat template của model [15]. Giữ một format gốc trong kho của bạn và viết hàm chuyển sang format của từng nền tảng, để chuyển nền tảng không phải làm lại dữ liệu.

### Code: kiểm, loại trùng lặp và chia tập

`ft_data.py` nhận dữ liệu dạng `messages`, loại ví dụ hỏng, trùng hoặc quá dài, và chia train/val/test theo nhóm:

```python
import hashlib
import json
import re
import unicodedata

ROLES = {"system", "user", "assistant"}


def load_jsonl(path):
    """Đọc JSONL, trả (records, errors). Lỗi được gom theo số dòng thay vì dừng ở lỗi đầu tiên."""
    records, errors = [], []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append((n, json.loads(line)))
            except json.JSONDecodeError as e:
                errors.append((n, f"JSON hỏng: {e.msg}"))
    return records, errors


def problems(rec) -> list[str]:
    """Các lỗi cấu trúc của một ví dụ dạng {"messages": [...]}; rỗng nghĩa là hợp lệ."""
    msgs = rec.get("messages") if isinstance(rec, dict) else None
    if not isinstance(msgs, list) or not msgs:
        return ["thiếu messages"]
    out = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            out.append(f"messages[{i}]: role không hợp lệ")
            continue
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            out.append(f"messages[{i}]: content rỗng")
        if m["role"] == "system" and i != 0:
            out.append(f"messages[{i}]: system không ở đầu")
    turns = [m.get("role") for m in msgs if isinstance(m, dict) and m.get("role") != "system"]
    if not turns or turns[0] != "user":
        out.append("lượt đầu không phải user")
    if not turns or turns[-1] != "assistant":
        out.append("lượt cuối không phải assistant")
    if any(a == b for a, b in zip(turns, turns[1:])):
        out.append("hai lượt liên tiếp cùng role")
    return out


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def prompt_key(rec) -> str:
    """Dấu vân tay của mọi lượt trước câu trả lời cuối, kể cả lượt assistant ở giữa (ngữ cảnh
    khác thì là ví dụ khác). Dùng để tìm trùng và rò rỉ giữa các tập."""
    parts = [m["role"] + ":" + norm(m["content"]) for m in rec["messages"][:-1]]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def split_of(group: str, test=0.1, val=0.1) -> str:
    """Chia theo hash của nhóm: cùng nhóm luôn vào cùng tập, thêm dữ liệu mới không xáo lại tập cũ."""
    h = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


def prepare(records, group_of, count_tokens=None, max_tokens=None, test=0.1, val=0.1):
    """records: list (số dòng, ví dụ). group_of(ví dụ) -> khoá nhóm (mã khách, mã ticket...).
    Trả (splits, report). Ví dụ lỗi, trùng hoặc quá dài bị loại và ghi vào report."""
    splits = {"train": [], "val": [], "test": []}
    report = {"invalid": [], "duplicate": [], "too_long": [], "conflict": []}
    seen = {}                                   # prompt_key -> (số dòng, câu trả lời)
    for n, rec in records:
        errs = problems(rec)
        if errs:
            report["invalid"].append((n, errs))
            continue
        if count_tokens and max_tokens and count_tokens(rec) > max_tokens:
            report["too_long"].append(n)        # trainer sẽ cắt mất phần cuối, gồm cả token kết thúc
            continue
        key, answer = prompt_key(rec), rec["messages"][-1]["content"]
        if key in seen:                         # giữ bản đầu; bản sau có thể rơi vào tập khác và gây rò rỉ
            first, first_answer = seen[key]
            same = unicodedata.normalize("NFC", answer) == unicodedata.normalize("NFC", first_answer)
            kind = "duplicate" if same else "conflict"    # đáp án so đúng từng ký tự: "DH-1" khác "dh-1"
            report[kind].append((first, n))     # conflict: hai đáp án cho cùng đầu vào, người phải chọn
            continue
        seen[key] = (n, answer)
        part = split_of(str(group_of(rec)), test, val)
        splits[part].append(rec)
    return splits, report
```

Các lựa chọn trong code:

- **Chia theo nhóm, không chia ngẫu nhiên từng dòng.** Nhóm là thứ mà các ví dụ trong đó giống nhau hơn mức bình thường: cùng khách hàng, cùng ticket, cùng tài liệu nguồn, cùng mẫu câu nếu dữ liệu sinh theo mẫu. Chia ngẫu nhiên từng dòng đưa ví dụ gần như giống hệt vào cả train lẫn test, và điểm test đo khả năng nhớ thay vì khả năng tổng quát. Đây là cùng lỗi rò rỉ dữ liệu ở [Evaluation §3.1](../evaluation/#31-xây-golden-dataset-từ-đâu).
- **Chia bằng hash của nhóm.** Một nhóm luôn rơi vào cùng một tập, và thêm dữ liệu mới không làm ví dụ cũ đổi tập. Tập test vì thế ổn định qua các lần huấn luyện, nên so được kết quả giữa các phiên bản.
- **Dấu vân tay của đầu vào chuẩn hoá Unicode NFC, chữ thường, khoảng trắng.** Văn bản tiếng Việt từ nhiều nguồn trộn NFC và NFD; hai chuỗi trông như nhau có thể khác byte. Dấu vân tay gồm mọi lượt trước câu trả lời cuối, kể cả lượt assistant ở giữa: hai hội thoại có cùng câu hỏi nhưng ngữ cảnh khác là hai ví dụ khác nhau. Chỉ bắt được trùng sau chuẩn hoá, không bắt được câu diễn đạt lại.
- **Cùng đầu vào, khác đáp án là `conflict`, không phải trùng.** Đáp án được so đúng từng ký tự, không chuẩn hoá chữ thường: `DH-914332` và `dh-914332` là hai nhãn khác nhau. Hai người gán hai nhãn khác nhau cho cùng một câu là dấu hiệu hướng dẫn gán nhãn chưa rõ. Code giữ bản đầu và báo lại; người phải quyết bản nào đúng. Vì đầu vào được so không phân biệt hoa thường, một số `conflict` là hai ví dụ đều đúng (đầu vào viết `dh-…`, đáp án chép theo); người xem báo cáo sẽ thấy ngay.
- **Lọc theo số token trước khi tới trainer.** Lý do ở test 4 bên dưới.

Test đã chạy (`t_ft_data.py`, không cần GPU):

1. Ví dụ hợp lệ không có lỗi; thiếu `messages`, lượt cuối không phải assistant, lượt đầu không phải user, content rỗng, role lạ, hai lượt liên tiếp cùng role, system không ở đầu: mỗi lỗi được báo đúng.
2. JSONL có dòng hỏng và dòng trống: lỗi được gom theo số dòng, các dòng khác vẫn đọc được.
3. Cùng câu viết NFD, chữ hoa, thừa khoảng trắng: cùng dấu vân tay với bản NFC chữ thường.
4. Chia theo hash ổn định; trên 10.000 nhóm, tỉ lệ test và val nằm trong khoảng 8–12%.
5. 65 ví dụ: bắt đúng một cặp trùng (khác nhóm), một cặp mâu thuẫn, một ví dụ hỏng, một ví dụ quá dài; cả ba tập đều có dữ liệu; mỗi khách hàng chỉ nằm trong một tập; không có dấu vân tay nào chung giữa train và test hay val.
6. Hai đáp án chỉ khác hoa thường (`DH-9` và `dh-9`) là `conflict`, không phải trùng. Hai hội thoại nhiều lượt cùng câu hỏi nhưng lượt assistant ở giữa khác nhau được giữ cả hai.
7. Thêm 30 ví dụ mới: mọi ví dụ test cũ vẫn ở tập test.

### Token nào được tính loss

`t_masking.py` chạy `SFTTrainer` của TRL 1.17.1 với tokenizer thật của Qwen2.5-0.5B-Instruct, chỉ xử lý dữ liệu, không huấn luyện, rồi đọc `labels` để biết token nào được tính loss:

1. Với `assistant_only_loss=True`, phần được tính loss đúng bằng `'\n{"intent":"cancel"}<|im_end|>\n'`: câu trả lời và token kết thúc lượt, không có system hay user.
2. Mặc định (không bật), loss tính trên cả system prompt và câu của user. Model bị dạy cả việc sinh câu hỏi của khách.
3. Hội thoại nhiều lượt: mọi lượt assistant đều được tính loss, không chỉ lượt cuối. Nếu các lượt trước là câu trả lời cũ chất lượng kém, chúng cũng được học.
4. **Ví dụ dài hơn `max_length` bị cắt từ cuối.** Một ví dụ dài với `max_length=128`: ở chế độ mặc định, ví dụ còn đúng 128 token và không còn chữ nào của câu trả lời, nghĩa là model được huấn luyện trên một ví dụ không có đáp án. Bật `assistant_only_loss`, ví dụ đó bị bỏ hẳn khỏi dataset; TRL chỉ hiện thanh tiến trình "Dropping fully masked examples", không có cảnh báo hay số lượng.
5. `prepare()` với bộ đếm token loại đúng ví dụ đó trước khi tới trainer.

Viết `t_masking.py` còn bắt được một lỗi trong chính code của bài: ở transformers 5, `apply_chat_template(..., tokenize=True)` trả về một dict, nên `len()` của kết quả là số khoá (2), không phải số token. Bộ đếm token đầu tiên vì thế không bao giờ thấy ví dụ nào quá dài. Code trong bài đã sửa bằng `return_dict=True` và đọc `["input_ids"]`.

### Kiểm nhãn khớp với văn bản

Kiểm cấu trúc không bắt được nhãn sai. Mỗi tác vụ cần thêm kiểm tra riêng. Với tác vụ trích xuất, kiểm giá trị trích ra có xuất hiện trong văn bản đầu vào. Dữ liệu đồ chơi ở [§3.3](#33-eval-trước-và-sau-fine-tune) có một lỗi như vậy ở phiên bản đầu: bộ sinh viết thường cả câu sau khi đã chèn mã đơn, nên văn bản có `dh-914332` trong khi nhãn ghi `DH-914332`. Lỗi lộ ra khi đọc từng câu model đoán sai, không phải khi nhìn điểm tổng. Một dòng kiểm "giá trị trong nhãn phải nằm trong văn bản" bắt được ngay.

## 3.2 Sinh dữ liệu tổng hợp từ model lớn

Khi thiếu dữ liệu thật, có thể dùng model lớn sinh đầu vào, câu trả lời, hoặc cả hai. Self-Instruct là ví dụ sớm: model sinh chỉ dẫn, đầu vào và câu trả lời, lọc bỏ mẫu hỏng hoặc quá giống nhau, rồi dùng để fine-tune chính model đó; GPT-3 tăng 33 điểm phần trăm tuyệt đối trên Super-NaturalInstructions [34].

Rủi ro và cách giảm:

- **Điều khoản.** Dữ liệu sinh từ model thương mại chịu điều khoản ở [§1.5](#15-distillation-dạy-model-nhỏ-bắt-chước-model-lớn).
- **Đuôi phân phối biến mất.** Huấn luyện lặp lại trên dữ liệu do model sinh làm mất dần phần hiếm của phân phối gốc (model collapse) [35]. Nghiên cứu sau đó cho thấy hiện tượng này xảy ra khi dữ liệu tổng hợp *thay thế* dữ liệu thật, và tránh được khi dữ liệu tổng hợp được *cộng thêm* vào dữ liệu thật [36]. Với fine-tune: giữ dữ liệu thật trong tập train, và không lấy output của model đã fine-tune để huấn luyện vòng sau mà không lọc.
- **Đa dạng giả.** Model lớn sinh câu hỏi sạch sẽ, đúng chính tả, cùng một giọng. Khách thật viết tắt, không dấu, hỏi hai việc trong một câu. Đưa vào prompt sinh dữ liệu các ví dụ thật, yêu cầu đúng các kiểu lỗi đó, và đếm tỉ lệ các kiểu sau khi sinh.
- **Tập test phải là dữ liệu thật.** Test trên dữ liệu sinh từ cùng một prompt với tập train là đo model bắt chước bộ sinh giỏi tới đâu. Thí nghiệm ở [§3.3](#33-eval-trước-và-sau-fine-tune) chia theo mẫu câu chính vì lý do này, và vẫn chỉ là dữ liệu đồ chơi.
- **Người duyệt một mẫu ngẫu nhiên** trước khi huấn luyện, và ghi lại tỉ lệ lỗi tìm được.

## 3.3 Eval trước và sau fine-tune

### Ba so sánh cần có

1. **Model fine-tune so với phương án mạnh nhất không fine-tune**: model gốc tốt nhất bạn dùng được với prompt tốt nhất. So với model gốc dùng prompt sơ sài là thổi phồng lợi ích.
2. **Trên tác vụ mục tiêu**, bằng tập test tách theo nhóm ([§3.1](#31-chuẩn-bị-dữ-liệu-nguồn-làm-sạch-format)).
3. **Ngoài tác vụ mục tiêu**: một bộ câu hỏi chung và bộ câu hỏi safety (yêu cầu cần từ chối, prompt injection) chạy trên cả model gốc lẫn model fine-tune. Fine-tune có thể làm model quên [26] và làm yếu safety [27]; chỉ đo tác vụ mục tiêu thì không thấy.

Cách chấm, khoảng tin cậy và so theo cặp ở [Evaluation §1.4](../evaluation/#14-phương-sai-chạy-bao-nhiêu-lần-cần-bao-nhiêu-mẫu). Model fine-tune được chấm trên đúng artifact sẽ chạy production: adapter, hay model đã gộp, ở đúng độ chính xác số (bf16, 4-bit).

### Một lần huấn luyện thật

Tác vụ đồ chơi: đọc tin nhắn chăm sóc khách hàng tiếng Việt, trả một dòng JSON `{"intent", "order_id"}` với sáu intent (`track`, `cancel`, `return`, `warranty`, `payment`, `other`). Dữ liệu sinh từ 48 mẫu câu, tám mẫu mỗi intent, chèn mã đơn ngẫu nhiên, thêm đuôi câu và đổi hoa thường ngẫu nhiên. Chia theo mẫu câu bằng `prepare(test=0.2, val=0.1)`, nên **mọi câu trong tập test là cách diễn đạt model chưa thấy khi train**. Bộ sinh tạo 576 dòng; loại 54 dòng trùng còn 522: 361 train, 101 validation, 60 test.

`train_lora.py`:

```python
import json
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from ft_data import load_jsonl, prepare

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def count_tokens(tok):
    # transformers 5 trả dict khi tokenize: len() của dict là số khoá, không phải số token
    return lambda rec: len(tok.apply_chat_template(rec["messages"], tokenize=True, return_dict=True)["input_ids"])


def evaluate(model, tok, rows, max_new_tokens=64):
    """Exact match trên JSON đã parse, và tỉ lệ câu trả lời dừng bằng token kết thúc."""
    model.eval()
    hit = stopped = 0
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"][:-1], add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**prompt, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[0, prompt["input_ids"].shape[1]:]
        stopped += int(new[-1].item() in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")})
        text = tok.decode(new, skip_special_tokens=True).strip()
        try:
            hit += json.loads(text) == json.loads(r["messages"][-1]["content"])
        except json.JSONDecodeError:
            pass
    return hit / len(rows), stopped / len(rows)


def main(path="toy.jsonl", out="out_lora"):
    tok = AutoTokenizer.from_pretrained(BASE)
    records, errors = load_jsonl(path)
    splits, report = prepare(records, group_of=lambda r: r["template"],
                             count_tokens=count_tokens(tok), max_tokens=MAX_LEN, test=0.2, val=0.1)
    print("lỗi JSON", len(errors), {k: len(v) for k, v in report.items()},
          {k: len(v) for k, v in splits.items()})
    ds = {k: Dataset.from_list([{"messages": r["messages"]} for r in v]) for k, v in splits.items()}

    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEVICE)
    print("base     exact, stopped:", evaluate(base, tok, splits["test"]))

    args = SFTConfig(
        output_dir=out,
        assistant_only_loss=True,       # loss chỉ trên câu trả lời, gồm cả <|im_end|>
        max_length=MAX_LEN,             # khớp với bước lọc ở prepare(): không ví dụ nào bị cắt
        num_train_epochs=2,
        per_device_train_batch_size=8,
        learning_rate=1e-4,             # LoRA: cao hơn full fine-tune khoảng 10 lần
        lr_scheduler_type="constant",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,    # giữ checkpoint có eval_loss thấp nhất, không phải checkpoint cuối
        metric_for_best_model="eval_loss",
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=False,
        report_to=[],
        seed=0,
    )
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=base, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
                         processing_class=tok, peft_config=peft)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"tham số train: {trainable:,} / {total:,} ({trainable / total:.2%})")
    t0 = time.time()
    trainer.train()
    print(f"train {time.time() - t0:.0f}s")
    if DEVICE == "mps":
        print(f"MPS driver {torch.mps.driver_allocated_memory() / 2**30:.1f} GiB")
    for h in trainer.state.log_history:
        if "eval_loss" in h:
            print("epoch", h["epoch"], "eval_loss", round(h["eval_loss"], 4))
    trainer.save_model(out)              # chỉ lưu adapter

    tuned = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEVICE), out)
    print("lora     exact, stopped:", evaluate(tuned, tok, splits["test"]))
    merged = tuned.merge_and_unload()    # gộp adapter vào trọng số: phục vụ như model thường
    print("merged   exact, stopped:", evaluate(merged, tok, splits["test"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
```

Kết quả (Apple M4 24 GB, MPS):

| | Exact match trên tập test | Dừng bằng token kết thúc |
|---|---|---|
| Qwen2.5-0.5B-Instruct, chỉ system prompt | 48,3% (29/60) | 100% |
| + LoRA adapter | 86,7% (52/60) | 100% |
| + LoRA đã gộp vào weights (bf16) | 88,3% (53/60) | 100% |

Huấn luyện: 8,8 triệu tham số (1,75%), khoảng 150 giây, `eval_loss` sau epoch 1 là 0,037, sau epoch 2 là 0,092.

Đọc kết quả:

- **`eval_loss` tăng ở epoch 2** trong khi loss train gần bằng 0: model bắt đầu thuộc tập train. `load_best_model_at_end=True` giữ checkpoint epoch 1; mặc định của `Trainer` là giữ checkpoint cuối.
- **Mọi lỗi còn lại của model fine-tune đến từ một mẫu câu**: "Quạt đơn … hỏng motor, có được sửa miễn phí", nhãn `warranty`, bị đoán sai. 60 câu test thực chất là 5 mẫu câu, nên số câu độc lập nhỏ hơn nhiều so với 60, và khoảng tin cậy thật rộng hơn khoảng tính theo 60 câu. Đây là hiệu ứng cụm ở [Evaluation §1.4](../evaluation/#ca-không-độc-lập-hiệu-chỉnh-theo-cụm). Tập test cũng không có câu `track` hay `other` nào vì hash không chọn mẫu câu nào của hai intent đó. Với dữ liệu thật, kiểm tra số lượng mỗi lớp trong tập test sau khi chia.
- **Model gộp và adapter cho kết quả khác nhau** trên 1 câu: gộp ở bf16 làm tròn weights khác với cộng adapter lúc chạy. Chênh lệch nhỏ, nhưng đủ để chấm lại artifact thực sự đem đi serve.
- **Model gốc nhầm intent ở hai mẫu câu và bỏ tiền tố mã đơn ở mẫu thứ ba.** 12/12 câu "mình bị trừ tiền nhưng đơn … không tạo được" (`payment`) bị đoán thành `return`, 12/12 câu bảo hành quạt sai intent, và 7/12 câu "Mình muốn huỷ đơn DH-…" trả `order_id` là `"680804"` thay vì `"DH-680804"`. Lỗi của model gốc cũng dồn theo mẫu câu. Một system prompt có ví dụ cụ thể có thể sửa phần lớn các lỗi này mà không cần fine-tune. Bảng trên so model fine-tune với model gốc dùng prompt sơ sài, đúng kiểu so sánh mục 1 ở trên cảnh báo. Với dữ liệu thật, phải so với model mạnh nhất dùng prompt tốt nhất.

Đây là dữ liệu đồ chơi để kiểm pipeline, không phải bằng chứng model 0,5B làm tốt tác vụ này trên tin nhắn thật.

## 3.4 Overfitting và catastrophic forgetting

**Overfitting**: model nhớ tập train thay vì học quy luật. Dấu hiệu là loss train tiếp tục giảm trong khi loss validation tăng, như ở epoch 2 trong thí nghiệm trên. Bedrock dùng cùng dấu hiệu cho RFT: reward train tăng trong khi reward validation giảm [9]. Cách xử lý:

- **Lưu checkpoint mỗi epoch và chọn theo validation**, không lấy checkpoint cuối. OpenAI tạo checkpoint ở cuối mỗi epoch và giữ ba epoch cuối để bạn dùng thay model cuối [1].
- **Ít epoch hơn với dữ liệu nhỏ.** Mặc định 3 epoch của TRL [15] là nhiều với vài trăm ví dụ dễ.
- **Thêm dữ liệu đa dạng** thay vì lặp lại dữ liệu cũ nhiều epoch.

**Catastrophic forgetting**: model mất năng lực đã có trước khi fine-tune. Luo và cộng sự quan sát hiện tượng này ở model 1B–7B khi instruction-tune liên tiếp, trên kiến thức miền, suy luận và đọc hiểu; trong khoảng kích thước đó, model càng lớn quên càng nặng [26]. Cách giảm:

- **LoRA thay full fine-tune.** LoRA giữ năng lực ngoài miền tốt hơn full fine-tune, và giảm quên tốt hơn weight decay hay dropout [25].
- **Bắt đầu từ model đã instruction-tune chung.** Luo và cộng sự kết luận instruction-tune chung trước đó giúp giảm quên ở lần fine-tune sau [26]. Đây thêm một lý do để bắt đầu từ bản instruct ([§2.3](#23-chọn-base-model)).
- **Đo.** Bộ câu hỏi ngoài tác vụ ở [§3.3](#33-eval-trước-và-sau-fine-tune) là cách duy nhất biết model đã quên gì.

## 3.5 Deploy và serving LoRA adapter

### Gộp hay giữ adapter riêng

| | Gộp vào weights (`merge_and_unload`) | Giữ adapter riêng |
|---|---|---|
| Serving | Như model thường, mọi engine chạy được | Engine phải hỗ trợ LoRA |
| Nhiều tác vụ trên một model gốc | Mỗi tác vụ một bản model đầy đủ | Một model gốc, nhiều adapter nhỏ |
| Độ trễ | Không thêm | Có chi phí tính adapter |
| Kết quả | Có thể lệch nhẹ do làm tròn khi gộp ([§3.3](#33-eval-trước-và-sau-fine-tune)) | Đúng như lúc eval adapter |

LoRA gốc nhấn mạnh việc không thêm độ trễ inference chính vì adapter gộp được vào weights [17].

### Nhiều adapter trên một model gốc

S-LoRA và Punica giải bài toán phục vụ nhiều adapter cùng lúc trên một bản model gốc, gộp các request dùng adapter khác nhau vào cùng một batch. S-LoRA phục vụ hàng nghìn adapter trên một GPU, throughput cao hơn tới 4 lần so với vLLM dùng LoRA kiểu đơn giản thời điểm đó [37]. Punica báo throughput cao hơn 12 lần, thêm 2 ms mỗi token [38]. Các ý tưởng này nay có trong vLLM.

### vLLM

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enable-lora \
    --lora-modules intent=/models/out_lora \
    --max-lora-rank 16 \
    --max-loras 4
```

Request gọi adapter bằng tên trong `--lora-modules` ở trường `model` [39]. Ba tham số có mặc định dễ gây lỗi [40]:

- `--max-lora-rank` mặc định 16. Adapter rank 32 không chạy với mặc định. Đặt bằng rank lớn nhất bạn dùng, không cao hơn, vì giá trị cao tốn bộ nhớ và làm chậm [39].
- `--max-loras` mặc định 1: một batch chỉ chứa request của một adapter. Phục vụ nhiều adapter cùng lúc thì tăng giá trị này.
- `--max-cpu-loras`: số adapter giữ trong RAM để đổi vào GPU nhanh.

vLLM có endpoint `/v1/load_lora_adapter` để nạp adapter lúc đang chạy, bật bằng biến môi trường `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. Tài liệu cảnh báo tính năng này có rủi ro bảo mật và không nên dùng ở production trừ môi trường tách biệt, hoàn toàn tin cậy [39]: ai gọi được endpoint là nạp được weights bất kỳ vào server.

### Nền tảng managed

Vertex trả về endpoint cho model đã tune, giá theo [§2.1](#gemini-chỉ-trên-vertex-ai). Fireworks bắt chạy trên deployment riêng [13]. Bedrock cần cấu hình cách chạy model tuỳ biến riêng; xem tài liệu Bedrock cho model và region bạn dùng [8]. Cách chọn GPU và engine khi tự host ở [Self-hosting LLM](../self-hosting-llm/).

## 3.6 Bảo trì: khi base model ra phiên bản mới

Model fine-tune gắn với một model gốc cụ thể. Ba sự kiện buộc phải làm lại:

1. **Model gốc ngừng.** Trên OpenAI, bản fine-tune ngừng cùng ngày với model gốc [2]. Model open-weight không bị ngừng, nhưng thư viện serving và phần cứng thay đổi.
2. **Model gốc mới tốt hơn.** Model mới có thể làm bằng model fine-tune chỉ với prompt. Mỗi khi có model mới đáng kể, chạy lại eval với model mới dùng prompt tốt nhất trước khi huấn luyện lại.
3. **Dữ liệu hoặc yêu cầu thay đổi**: intent mới, format mới, chính sách mới mà model đã học sai.

Để làm lại rẻ:

- **Lưu dữ liệu ở format gốc của bạn**, kèm phiên bản, và hàm chuyển sang format từng nền tảng ([§3.1](#31-chuẩn-bị-dữ-liệu-nguồn-làm-sạch-format)).
- **Huấn luyện bằng script, không bằng giao diện.** Cấu hình (model gốc và revision, rank, learning rate, epoch, seed) nằm trong code và được version.
- **Tập test cố định qua các phiên bản.** Chia theo hash của nhóm giữ tập test ổn định khi thêm dữ liệu.
- **Ngưỡng chấp nhận viết trước**: model mới phải bằng hoặc hơn model đang chạy trên tác vụ, và không kém hơn quá một mức đã định trên bộ ngoài tác vụ và safety. Đưa vào CI như [Evaluation §3.4](../evaluation/#34-đưa-eval-vào-ci).

## 3.7 Failure modes thường gặp

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| Model fine-tune trả lời sai kiến thức mới, tự tin | Dùng fine-tune để nạp kiến thức hay thay đổi [23][24] | Đưa kiến thức vào RAG hoặc prompt |
| Model sinh mãi không dừng, chạm `max_tokens` | Token kết thúc lượt không được tính loss; base model thiếu `eos_token` | `assistant_only_loss` với template có `{% generation %}`; đặt `eos_token` [15] |
| Model trả lời cụt, mất phần cuối | Ví dụ dài bị cắt ở `max_length`, câu trả lời không còn trong ví dụ | Đếm token, loại hoặc rút gọn ví dụ trước khi huấn luyện ([§3.1](#31-chuẩn-bị-dữ-liệu-nguồn-làm-sạch-format)) |
| Dataset ít hơn số dòng đã đưa, không có cảnh báo | `assistant_only_loss` bỏ ví dụ bị cắt hết câu trả lời | Như trên; đếm số dòng sau khi trainer xử lý |
| Model sinh lại câu hỏi của user hoặc system prompt | Loss tính trên cả user và system | `assistant_only_loss=True` |
| Điểm test rất cao, production kém | Rò rỉ: ví dụ gần giống nhau ở cả train và test; test sinh từ cùng bộ sinh với train | Chia theo nhóm; test bằng dữ liệu thật |
| `eval_loss` tăng, loss train gần 0 | Overfitting | Ít epoch hơn; chọn checkpoint theo validation |
| Model kém ở việc nó từng làm tốt | Catastrophic forgetting [26] | LoRA; bắt đầu từ bản instruct; đo ngoài tác vụ |
| Model chịu làm theo yêu cầu có hại mà model gốc từ chối | Fine-tune làm yếu safety, kể cả với dữ liệu lành tính [27] | Chạy lại bộ test safety và guardrail trên model fine-tune |
| Kết quả production khác kết quả eval | Eval adapter, serve model đã gộp hoặc lượng tử hoá | Eval đúng artifact đem đi serve |
| vLLM không nạp được adapter | Rank lớn hơn `--max-lora-rank` (mặc định 16) [40] | Đặt bằng rank lớn nhất đang dùng |
| Chi phí inference tăng sau fine-tune | Model fine-tune có giá cao hơn model gốc [4][11]; phải chạy deployment riêng [13] | Tính chi phí inference cả năm trước khi quyết |
| Model fine-tune sắp ngừng chạy | Model gốc bị deprecate [2] | Giữ dữ liệu, script, eval để huấn luyện lại trên nền tảng khác |
| Tuned Gemini kém khi bật structured output | Controlled generation không có lúc tune [6] | Không bật khi gọi model đã tune |
| Nhãn không khớp văn bản | Lỗi ở bước sinh hoặc gán nhãn | Kiểm riêng theo tác vụ; đọc từng câu sai |

## 3.8 Checklist

Trước khi quyết định:

- [ ] Có tập test tách theo nhóm và cách chấm tự động.
- [ ] Đã thử model mạnh nhất dùng được với prompt tốt nhất, và ghi điểm làm baseline.
- [ ] Vấn đề là hành vi hoặc format, không phải kiến thức hay thay đổi.
- [ ] Đã đọc điều khoản nếu dữ liệu sinh từ model thương mại.
- [ ] Đã tính chi phí cả năm: huấn luyện nhiều lần, inference sau fine-tune, huấn luyện lại.
- [ ] Biết model sẽ chạy ở đâu và model gốc đó còn được hỗ trợ bao lâu.

Dữ liệu:

- [ ] Kiểm cấu trúc, loại trùng lặp sau chuẩn hoá, báo mâu thuẫn.
- [ ] Đếm token bằng tokenizer của model; không ví dụ nào vượt `max_length`.
- [ ] Chia theo nhóm; tập test có đủ các lớp; không có dấu vân tay chung giữa các tập.
- [ ] Có kiểm tra riêng theo tác vụ (nhãn khớp văn bản, giá trị hợp lệ).
- [ ] Người đã đọc một mẫu ngẫu nhiên.

Huấn luyện:

- [ ] Kiểm token nào được tính loss trên vài ví dụ thật; token kết thúc lượt nằm trong đó.
- [ ] LoRA trên mọi lớp linear; learning rate cho LoRA, không phải mặc định full fine-tune.
- [ ] Checkpoint mỗi epoch, chọn theo validation.
- [ ] Cấu hình, model gốc và revision, seed nằm trong code đã version.

Sau huấn luyện:

- [ ] So với baseline không fine-tune trên cùng tập test, kèm khoảng tin cậy có tính cụm.
- [ ] Đo bộ ngoài tác vụ và bộ safety trên cả model gốc và model fine-tune.
- [ ] Đọc từng câu sai, không chỉ nhìn điểm tổng.
- [ ] Eval đúng artifact đem đi serve (adapter hay đã gộp, độ chính xác số).
- [ ] Serving: `--max-lora-rank`, `--max-loras` đúng; endpoint nạp adapter lúc chạy không mở ra ngoài.
- [ ] Ghi lại kế hoạch huấn luyện lại khi model gốc ngừng.

---

## Nguồn tham khảo

Truy cập tháng 11/2026.

| # | Nguồn | Nội dung dùng trong bài |
|---|---|---|
| [1] | [OpenAI: Supervised fine-tuning](https://developers.openai.com/api/docs/guides/supervised-fine-tuning) | Thông báo đóng nền tảng, eval trước, số ví dụ (tối thiểu 10, bắt đầu 50), format, tập đối chứng, checkpoint ba epoch cuối |
| [2] | [OpenAI: Deprecations](https://developers.openai.com/api/docs/deprecations) | Lịch thu hẹp fine-tune (07/05/2026, 02/07/2026, 06/01/2027); ngày ngừng `gpt-4.1-nano`, `o4-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini`; bản fine-tune theo lịch model gốc |
| [3] | [Claude: Glossary](https://platform.claude.com/docs/en/about-claude/glossary) | Định nghĩa fine-tune; Claude API không có fine-tune |
| [4] | [OpenAI: Pricing](https://developers.openai.com/api/docs/pricing) | Giá huấn luyện và inference của model fine-tune |
| [5] | [Google: Fine-tuning with the Gemini API](https://ai.google.dev/gemini-api/docs/model-tuning) | Gemini API và AI Studio không còn model fine-tune được |
| [6] | [Google Cloud: About supervised fine-tuning for Gemini](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning) | Model hỗ trợ, giới hạn dataset, adapter size, region, thinking, known issue với controlled generation, use case, region `global` và CMEK |
| [7] | [Google Cloud: Tune Gemini models by using supervised fine-tuning](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-use-supervised-tuning) và [Prepare data](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning-prepare) | 100–500 ví dụ, format `systemInstruction` / `contents`, epoch, learning rate multiplier |
| [8] | [AWS: Customize your model](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html) và [Fine-tuning supported models](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-model-fine-tuning.html) | Ba phương pháp, distillation, bảng model SFT và region, cách tính phí |
| [9] | [AWS: Reinforcement fine-tuning in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/reinforcement-fine-tuning.html) | Model RFT, GRPO, reward bằng Lambda hoặc judge, 100–200 ví dụ, mốc 0% và 95%, dấu hiệu overfit |
| [10] | [AWS: Fine-tune Anthropic's Claude 3 Haiku in Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/fine-tune-anthropics-claude-3-haiku-in-amazon-bedrock-to-boost-model-accuracy-and-quality/) | Claude 3 Haiku hết vòng đời 10/09/2026; 50–100 dòng để bắt đầu |
| [11] | [Google Cloud: Generative AI pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing) | Giá huấn luyện theo token × epoch; loại tuning theo model (SFT, RL fine-tuning, preference tuning; model open-weight như Gemma 3, Llama, Qwen 3); giá inference model đã tune từ Gemini 3 |
| [12] | [Together AI: Pricing](https://www.together.ai/pricing) | Giá SFT, DPO theo triệu token, cách tính, phí tối thiểu |
| [13] | [Fireworks: Supervised fine-tuning, text](https://docs.fireworks.ai/fine-tuning/fine-tuning-models) | Model đã huấn luyện chỉ chạy trên deployment on-demand |
| [14] | [Hugging Face: TRL](https://huggingface.co/docs/trl/index) | Các trainer và tích hợp |
| [15] | [Hugging Face: TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) | `assistant_only_loss`, `completion_only_loss`, `eos_token` cho base model, learning rate cho adapter, tham số `SFTConfig` (đối chiếu với TRL 1.17.1 đã cài) |
| [16] | [Thinking Machines: LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) và [bản tóm tắt của TRL](https://huggingface.co/docs/trl/lora_without_regret) | LoRA ngang full fine-tune khi cấu hình đúng; mọi lớp; learning rate gấp 10; batch size |
| [17] | [Hu và cộng sự, LoRA (arXiv 2106.09685)](https://arxiv.org/abs/2106.09685) | Giảm 10.000 lần tham số, 3 lần bộ nhớ; không thêm độ trễ inference |
| [18] | [Dettmers và cộng sự, QLoRA (arXiv 2305.14314)](https://arxiv.org/abs/2305.14314) | 65B trên một GPU 48 GB; NF4, double quantization, paged optimizer |
| [19] | [Unsloth README](https://github.com/unslothai/unsloth) | Số liệu tốc độ và VRAM tự công bố; nền tảng hỗ trợ |
| [20] | [Axolotl README](https://github.com/axolotl-ai-cloud/axolotl) | Cấu hình YAML, FSDP2, DeepSpeed, expert parallelism |
| [21] | [Zhou và cộng sự, LIMA (arXiv 2305.11206)](https://arxiv.org/abs/2305.11206) | 1.000 ví dụ chọn kỹ cho model 65B |
| [22] | [Rafailov và cộng sự, DPO (arXiv 2305.18290)](https://arxiv.org/abs/2305.18290) | DPO thay RLHF bằng loss phân loại |
| [23] | [Ovadia và cộng sự, Fine-Tuning or Retrieval? (arXiv 2312.05934)](https://arxiv.org/abs/2312.05934) | RAG thắng fine-tune không giám sát khi nạp kiến thức |
| [24] | [Gekhman và cộng sự, Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? (arXiv 2405.05904)](https://arxiv.org/abs/2405.05904) | Kiến thức mới học chậm, làm tăng xu hướng bịa |
| [25] | [Biderman và cộng sự, LoRA Learns Less and Forgets Less (arXiv 2405.09673)](https://arxiv.org/abs/2405.09673) | LoRA kém full fine-tune trên code và toán nhưng quên ít hơn; rank của full fine-tune |
| [26] | [Luo và cộng sự, Catastrophic Forgetting during Continual Fine-tuning (arXiv 2308.08747)](https://arxiv.org/abs/2308.08747) | Quên ở model 1B–7B; instruction-tune chung giảm quên |
| [27] | [Qi và cộng sự, Fine-tuning Aligned Language Models Compromises Safety (arXiv 2310.03693)](https://arxiv.org/abs/2310.03693) | 10 ví dụ, dưới 0,20 USD; dữ liệu lành tính cũng làm giảm safety |
| [28] | [Rajbhandari và cộng sự, ZeRO (arXiv 1910.02054)](https://arxiv.org/abs/1910.02054) | 16 byte mỗi tham số với Adam mixed precision; GPT-2 1,5B cần 24 GB |
| [29] | [OpenAI Services Agreement](https://openai.com/policies/services-agreement/) | Mục 3.3(e) và định nghĩa Permitted Exception |
| [30] | [Anthropic Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms) | Mục D.4 Use Restrictions |
| [31] | [Gemini API Additional Terms of Service](https://ai.google.dev/gemini-api/terms) | Không phát triển model cạnh tranh |
| [32] | [torchtune README](https://github.com/pytorch/torchtune) | Không còn được phát triển tích cực |
| [33] | [Hugging Face: TRL Chat Templates](https://huggingface.co/docs/trl/chat_templates) | Template huấn luyện có `{% generation %}`, danh sách họ model được nhận ra |
| [34] | [Wang và cộng sự, Self-Instruct (arXiv 2212.10560)](https://arxiv.org/abs/2212.10560) | Sinh và lọc dữ liệu chỉ dẫn; +33 điểm trên Super-NaturalInstructions |
| [35] | [Shumailov và cộng sự, The Curse of Recursion (arXiv 2305.17493)](https://arxiv.org/abs/2305.17493) | Model collapse khi huấn luyện trên dữ liệu do model sinh |
| [36] | [Gerstgrasser và cộng sự, Is Model Collapse Inevitable? (arXiv 2404.01413)](https://arxiv.org/abs/2404.01413) | Cộng dồn dữ liệu thật và tổng hợp tránh được collapse |
| [37] | [Sheng và cộng sự, S-LoRA (arXiv 2311.03285)](https://arxiv.org/abs/2311.03285) | Hàng nghìn adapter trên một GPU, throughput tới 4 lần |
| [38] | [Chen và cộng sự, Punica (arXiv 2310.18547)](https://arxiv.org/abs/2310.18547) | Throughput 12 lần, thêm 2 ms mỗi token |
| [39] | [vLLM: LoRA Adapters](https://docs.vllm.ai/en/latest/features/lora/) | `--enable-lora`, `--lora-modules`, nạp adapter lúc chạy và cảnh báo bảo mật, cách đặt `--max-lora-rank` |
| [40] | [vLLM: Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/) | Mặc định `--max-loras` 1, `--max-lora-rank` 16, `--max-cpu-loras` |
