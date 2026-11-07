# RAG nâng cao: khi retrieval cơ bản không đủ

Một pipeline RAG cơ bản (chia tài liệu thành chunk, embed, lấy top-k theo cosine, nhét vào prompt) thường chạy tốt trong demo. Sau đó người dùng thật bắt đầu hỏi. "Hợp đồng gói Pro có cho hoàn tiền không?": retriever trả về đúng file hợp đồng nhưng lại là đoạn nói về gói Basic, vì chunk chứa điều khoản hoàn tiền không có chữ "Pro" nào. "So sánh chính sách bảo hành năm 2025 và 2026": câu trả lời cần hai tài liệu, top-5 chỉ chứa tài liệu năm 2026. "Khách hàng phàn nàn nhiều nhất về điều gì?": câu hỏi này không có đoạn văn nào trả lời trực tiếp, vì đáp án phải tổng hợp từ hàng nghìn ticket.

Tài liệu này dành cho những trường hợp như vậy. Nó giả định bạn đã nắm phần RAG trong [Embeddings](../embeddings/): bi-encoder và cross-encoder, dense/sparse/hybrid, pipeline ba giai đoạn ([§2.2](../embeddings/#22-rag--rerank-pipeline-3-giai-đoạn)), chunking ([§3.3](../embeddings/#33-chunking-strategies)), hybrid search ([§3.5](../embeddings/#35-hybrid-search-sparse--dense)). Những phần đó không nhắc lại ở đây.

- **Phần 1** giải thích pipeline hỏng ở đâu, long context thay được RAG tới mức nào, và khi nào nên để model tự quyết việc tìm kiếm.
- **Phần 2** đi qua từng kỹ thuật: biến đổi query, rerank, contextual retrieval, chunk phân cấp, GraphRAG, dữ liệu có cấu trúc, tài liệu nhiều bảng và ảnh, trích dẫn. Mỗi kỹ thuật kèm số liệu từ paper hoặc tài liệu chính thức, chi phí thêm và cách nó hỏng.
- **Phần 3** là vận hành: chọn kỹ thuật theo triệu chứng, cập nhật index, phân quyền, chi phí từng tầng, eval.

> Bản tháng 11/2026. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh để tiện đối chiếu tài liệu và API. Giá và giới hạn của provider thay đổi thường xuyên; kiểm tra lại trang chính thức trước khi dùng con số trong dự toán.

---

## Mục lục

- [Tra cứu nhanh](#tra-cứu-nhanh)
- [Phần 1 — Nền tảng](#phần-1--nền-tảng)
  - [1.1 Pipeline RAG có những điểm hỏng nào](#11-pipeline-rag-có-những-điểm-hỏng-nào)
  - [1.2 Retrieval, reranking và generation: tối ưu từng tầng](#12-retrieval-reranking-và-generation-tối-ưu-từng-tầng)
  - [1.3 Long context có thay thế được RAG không](#13-long-context-có-thay-thế-được-rag-không)
  - [1.4 Agentic RAG: model tự quyết khi nào và tìm gì](#14-agentic-rag-model-tự-quyết-khi-nào-và-tìm-gì)
- [Phần 2 — Kỹ thuật](#phần-2--kỹ-thuật)
  - [2.1 Query rewriting, HyDE, multi-query, decomposition](#21-query-rewriting-hyde-multi-query-decomposition)
  - [2.2 Reranking: cross-encoder và reranker API](#22-reranking-cross-encoder-và-reranker-api)
  - [2.3 Contextual retrieval và chunk có ngữ cảnh](#23-contextual-retrieval-và-chunk-có-ngữ-cảnh)
  - [2.4 Parent–child, sentence window và late chunking](#24-parentchild-sentence-window-và-late-chunking)
  - [2.5 GraphRAG và knowledge graph](#25-graphrag-và-knowledge-graph)
  - [2.6 Metadata filtering và structured retrieval (text-to-SQL)](#26-metadata-filtering-và-structured-retrieval-text-to-sql)
  - [2.7 RAG đa phương thức: bảng, ảnh, PDF phức tạp](#27-rag-đa-phương-thức-bảng-ảnh-pdf-phức-tạp)
  - [2.8 Trích dẫn nguồn và grounding](#28-trích-dẫn-nguồn-và-grounding)
- [Phần 3 — Thiết kế và vận hành](#phần-3--thiết-kế-và-vận-hành)
  - [3.1 Chọn kỹ thuật theo triệu chứng](#31-chọn-kỹ-thuật-theo-triệu-chứng)
  - [3.2 Cập nhật index khi dữ liệu thay đổi](#32-cập-nhật-index-khi-dữ-liệu-thay-đổi)
  - [3.3 Phân quyền tài liệu trong retrieval](#33-phân-quyền-tài-liệu-trong-retrieval)
  - [3.4 Độ trễ và chi phí của từng tầng](#34-độ-trễ-và-chi-phí-của-từng-tầng)
  - [3.5 Eval cho RAG nâng cao](#35-eval-cho-rag-nâng-cao)
  - [3.6 Failure modes thường gặp](#36-failure-modes-thường-gặp)
  - [3.7 Checklist](#37-checklist)
- [Nguồn tham khảo](#nguồn-tham-khảo)

## Tra cứu nhanh

**Triệu chứng → kỹ thuật nên thử trước**

| Triệu chứng | Đo gì để xác nhận | Thử trước | Chi phí thêm mỗi query | Chi tiết |
|---|---|---|---|---|
| Đáp án có trong corpus nhưng không vào top-k | Recall@50 thấp | Hybrid (BM25 + dense) với RRF, tăng số candidate | Một truy vấn BM25 | [§2.1](#21-query-rewriting-hyde-multi-query-decomposition), [Embeddings §3.5](../embeddings/#35-hybrid-search-sparse--dense) |
| Đáp án nằm trong top-50 nhưng rơi khỏi top-5 | Recall@50 cao, Recall@5 thấp | Rerank top-N candidate, N chọn theo Recall@N đo được | Một lần gọi reranker | [§2.2](#22-reranking-cross-encoder-và-reranker-api) |
| Lấy đúng tài liệu, sai đoạn; chunk thiếu chủ ngữ ("gói này", "điều khoản trên") | Đọc chunk được trả về | Contextual retrieval, hoặc parent–child | Không thêm lúc query; tốn lúc index | [§2.3](#23-contextual-retrieval-và-chunk-có-ngữ-cảnh), [§2.4](#24-parentchild-sentence-window-và-late-chunking) |
| Câu hỏi so sánh hoặc nhiều phần, chỉ trả lời được một phần | Recall theo từng phần của câu hỏi | Decomposition, multi-query | Một lần gọi LLM + N retrieval | [§2.1](#21-query-rewriting-hyde-multi-query-decomposition) |
| Câu hỏi nhiều bước, bước sau phụ thuộc kết quả bước trước | Tỉ lệ đúng trên nhóm multi-hop | Iterative retrieval, agentic RAG có giới hạn vòng | Nhiều lần gọi LLM và retrieval | [§1.4](#14-agentic-rag-model-tự-quyết-khi-nào-và-tìm-gì) |
| Câu hỏi tổng hợp toàn corpus ("vấn đề lớn nhất là gì") | Không có chunk nào chứa đáp án | GraphRAG global, LazyGraphRAG, hoặc map-reduce trên tóm tắt | Lớn lúc index (GraphRAG) | [§2.5](#25-graphrag-và-knowledge-graph) |
| Câu hỏi về con số, đếm, lọc theo ngày | Đáp án nằm trong bảng/DB, không trong văn bản | Metadata filter, text-to-SQL | Một lần gọi LLM sinh SQL | [§2.6](#26-metadata-filtering-và-structured-retrieval-text-to-sql) |
| PDF nhiều bảng và biểu đồ, OCR sai | Đọc text đã parse của trang lỗi | Chunk bảng giữ header, visual retrieval, gửi trang PDF cho model | Lưu trữ lớn hơn (visual) | [§2.7](#27-rag-đa-phương-thức-bảng-ảnh-pdf-phức-tạp) |
| Model trả lời khi không có tài liệu nào liên quan | Nhóm câu hỏi không có đáp án | Prompt cho phép nói "không có trong tài liệu", ngưỡng điểm | Gần như không | [§1.1](#11-pipeline-rag-có-những-điểm-hỏng-nào) |
| Trích dẫn trỏ sai đoạn hoặc đoạn không tồn tại | Kiểm tra quote trong nguồn | Citations API, hoặc quote matching | Gần như không | [§2.8](#28-trích-dẫn-nguồn-và-grounding) |
| Người dùng thấy tài liệu không được phép xem | Test với user ít quyền | Filter quyền trong truy vấn, không trong prompt | Thấp | [§3.3](#33-phân-quyền-tài-liệu-trong-retrieval) |

**Thứ tự nâng cấp pipeline**, từ rẻ đến đắt:

1. Đo trước. Tách Recall@N của retrieval khỏi chất lượng câu trả lời. Không có số này thì mọi kỹ thuật bên dưới đều là đoán.
2. Hybrid search và RRF nếu chưa có.
3. Rerank trên top-N candidate, với N chọn bằng đo Recall@N ([§2.2](#22-reranking-cross-encoder-và-reranker-api)).
4. Sửa chunk: contextual retrieval hoặc parent–child.
5. Biến đổi query (multi-query, decomposition), chỉ bật cho nhóm câu hỏi cần, qua routing.
6. Agentic RAG cho câu hỏi nhiều bước, có giới hạn vòng và ngân sách.
7. GraphRAG cho câu hỏi tổng hợp, khi eval của bạn cho thấy nó thắng.

---

# Phần 1 — Nền tảng

## 1.1 Pipeline RAG có những điểm hỏng nào

Barnett và cộng sự (2024) rút ra bảy điểm hỏng từ ba hệ thống RAG thật: một công cụ đọc tài liệu nghiên cứu, một trợ giảng cho sinh viên, và một hệ hỏi đáp y sinh trên BioASQ [1]. Danh sách này hữu ích vì mỗi điểm hỏng ứng với một tầng của pipeline, nên biết nó là biết phải sửa ở đâu.

| # | Tên trong paper | Hỏng ở tầng | Biểu hiện |
|---|---|---|---|
| FP1 | Missing Content | Corpus | Câu hỏi không có đáp án trong tài liệu, hệ thống vẫn trả lời thay vì nói không biết |
| FP2 | Missed the Top Ranked Documents | Retrieval | Đáp án có trong corpus nhưng không lọt vào top-k |
| FP3 | Not in Context | Ghép context | Tài liệu đã được lấy về nhưng bị bỏ khi ghép prompt (giới hạn token, cắt sai thứ tự) |
| FP4 | Not Extracted | Generation | Đáp án nằm trong context nhưng model không lấy ra, thường do nhiễu hoặc các đoạn mâu thuẫn |
| FP5 | Wrong Format | Generation | Đúng nội dung, sai định dạng được yêu cầu |
| FP6 | Incorrect Specificity | Generation | Trả lời quá chung hoặc quá chi tiết so với câu hỏi |
| FP7 | Incomplete | Retrieval và generation | Câu hỏi nhiều phần, chỉ trả lời được một phần |

Paper không đưa tỉ lệ cho từng điểm hỏng. Kết luận chính của nó là kiểm chứng một hệ RAG chỉ làm được khi hệ đã vận hành, vì không đoán trước được người dùng sẽ hỏi gì [1]. Vài bài học cụ thể trong paper: thêm tên file và số thứ tự chunk vào metadata giúp model đọc context tốt hơn; context dài hơn cho kết quả tốt hơn trong case study trợ giảng.

Bảng trên dẫn tới một quy tắc đo: **mỗi điểm hỏng cần một phép đo riêng**. FP2 đo bằng Recall@N trên tập câu hỏi có nhãn tài liệu đúng. FP4 đo bằng tỉ lệ câu trả lời sai trên những câu mà tài liệu đúng đã nằm trong context. FP1 cần một nhóm câu hỏi cố ý không có đáp án trong corpus. Nếu bạn chỉ có một con số "accuracy của câu trả lời", bạn không biết sửa tầng nào. Cách xây các tập này ở [§3.5](#35-eval-cho-rag-nâng-cao).

Ngoài bảy điểm của paper, mấy lỗi sau hay gặp trong hệ thống doanh nghiệp:

- **Chunk cắt ngang câu trả lời.** Điều kiện ở chunk này, ngoại lệ ở chunk sau.
- **Khác ngôn ngữ hoặc khác thuật ngữ.** Tài liệu tiếng Anh, câu hỏi tiếng Việt; tài liệu gọi là "SLA", người dùng hỏi "cam kết thời gian phản hồi".
- **Phiên bản cũ.** Corpus chứa cả chính sách 2025 lẫn 2026, không có metadata ngày hiệu lực, nên hai bản cạnh tranh nhau trong top-k.
- **Index lỗi thời.** Tài liệu nguồn đã sửa hoặc xoá, index vẫn giữ bản cũ ([§3.2](#32-cập-nhật-index-khi-dữ-liệu-thay-đổi)).
- **Nội dung độc hại trong corpus.** Một đoạn văn chứa chỉ thị cho model, được retrieve vì giống câu hỏi. Chi tiết ở [Embeddings §3.8.2](../embeddings/#382-corpus-poisoning-và-indirect-prompt-injection).

## 1.2 Retrieval, reranking và generation: tối ưu từng tầng

Pipeline có ba tầng, mỗi tầng có một việc và một metric:

| Tầng | Việc | Metric | Đầu ra điển hình |
|---|---|---|---|
| Retrieval | Không bỏ sót | Recall@N, N = số candidate đưa vào rerank | N candidate (thí nghiệm của Anthropic dùng 150 [21]) |
| Rerank | Đưa cái đúng lên đầu | nDCG@k, Recall@k, k = số đoạn đưa vào prompt | k đoạn (thí nghiệm của Anthropic dùng 20 [21]) |
| Generation | Dùng đúng và chỉ dùng context | Faithfulness, độ đúng của câu trả lời | Câu trả lời có trích dẫn |

Đặt hai con số cạnh nhau là biết vấn đề nằm ở đâu. Recall@50 bằng 0,85 nhưng Recall@10 sau rerank chỉ 0,70: retrieval ổn, rerank hoặc số đoạn đưa vào prompt là chỗ cần sửa. Recall@50 bằng 0,50: sửa reranker không giúp được gì, vì một nửa số đáp án chưa bao giờ tới được reranker; phải sửa chunk, embedding, hoặc thêm BM25.

Một hệ quả ít được chú ý: **các tầng đánh đổi lẫn nhau**. Lấy 150 candidate thay vì 50 tăng recall nhưng làm reranker chậm hơn và đắt hơn. Đưa 20 đoạn vào prompt thay vì 5 tăng khả năng có đáp án trong context, nhưng mỗi đoạn thêm là thêm nhiễu cho model. Bài contextual retrieval của Anthropic thử 5, 10 và 20 chunk và thấy 20 cho kết quả tốt nhất trên bộ dữ liệu của họ, kèm lưu ý rằng nhiều thông tin hơn có thể làm model phân tâm và nên tự thử trên use case của mình [21]. Đó là số của một bộ dữ liệu, không phải hằng số.

## 1.3 Long context có thay thế được RAG không

Nhiều model hiện có cửa sổ context 1 triệu token [5]. Một kho tài liệu vài trăm trang vừa trong đó. Câu hỏi hợp lý: sao không bỏ hết retrieval, đưa cả kho vào prompt?

**Bằng chứng ủng hộ long context.** Li và cộng sự (2024) so sánh trên 9 dataset. Khi đủ tài nguyên, long context (LC) cho điểm trung bình cao hơn RAG rõ rệt [3]:

| Model | Long context | RAG | Self-Route |
|---|---|---|---|
| Gemini-1.5-Pro | 49,70 | 37,33 | 46,41 |
| GPT-4o | 48,67 | 32,60 | 48,89 |
| GPT-3.5-Turbo | 32,07 | 30,33 | 35,32 |

Self-Route là cách lai của paper: thử RAG trước, để model tự đánh giá context có đủ trả lời không; nếu không, chuyển sang long context. Theo bảng của paper, phần lớn câu hỏi dừng ở bước RAG (76,78% với Gemini-1.5-Pro, 57,36% với GPT-4o), và Self-Route dùng khoảng 38% số token của long context với Gemini-1.5-Pro, khoảng 61% với GPT-4o [3]. Phần văn bản của paper ghi vài con số không khớp với bảng của chính nó (ví dụ mức giảm chi phí 65% cho Gemini); bảng trên lấy từ bảng.

**Bằng chứng chống lại.** Model không dùng đều mọi vị trí trong context:

- *Lost in the middle* (Liu và cộng sự): model trả lời tốt nhất khi thông tin cần nằm ở đầu hoặc cuối context, kém hẳn khi nó nằm ở giữa. Với GPT-3.5-Turbo, độ chính xác khi trả lời không có tài liệu nào là 56,1%; khi có 20 hoặc 30 tài liệu và đáp án nằm ở vị trí xấu nhất, độ chính xác xuống dưới mức đó [2]. Nghĩa là thêm tài liệu đặt sai chỗ còn tệ hơn không đưa gì.
- *RULER* (Hsieh và cộng sự): trong 17 model được thử, tác giả viết rằng chỉ khoảng một nửa giữ được hiệu năng thỏa đáng ở độ dài 32K [4]. Độ dài hiệu dụng của GPT-4 là 64K dù cửa sổ 128K; Llama 3.1 70B cũng 64K trên 128K [4].
- Tài liệu context window của Claude ghi thẳng rằng khi số token tăng, độ chính xác và khả năng nhớ lại giảm [5].

**Chi phí.** Long context trả tiền cho cả kho tài liệu ở mỗi lượt gọi. Prompt caching giảm phần lớn chi phí đó khi prefix ổn định (tính toán ở [Multi-turn §1.4](../multi-turn-conversation/#14-kinh-tế-học-của-prompt-caching)), nhưng cache chỉ khớp theo prefix: tài liệu thay đổi thì cache vỡ.

**Quy tắc chọn:**

| Tình huống | Chọn |
|---|---|
| Corpus nhỏ (Anthropic gợi ý mốc khoảng 200.000 token, tức khoảng 500 trang [21]), ít thay đổi, nhiều câu hỏi cần đọc toàn bộ | Long context với prompt caching |
| Corpus lớn hơn cửa sổ, hoặc thay đổi hằng ngày | RAG |
| Một người dùng hỏi sâu về một vài tài liệu cụ thể (hợp đồng, báo cáo) | Retrieve tài liệu, rồi đưa nguyên tài liệu (không phải chunk) vào context |
| Câu hỏi khó trả lời từ vài đoạn, tỉ lệ nhỏ | Self-Route: RAG trước, model báo không đủ thì mở rộng |
| Cần biết câu trả lời dựa trên đoạn nào | RAG có trích dẫn, hoặc long context kèm Citations API ([§2.8](#28-trích-dẫn-nguồn-và-grounding)) |

Long context và RAG không loại trừ nhau. Mẫu hay dùng nhất trong thực tế là dòng thứ ba: retrieval chọn tài liệu, long context đọc tài liệu.

## 1.4 Agentic RAG: model tự quyết khi nào và tìm gì

Pipeline RAG cố định retrieve đúng một lần cho mỗi câu hỏi, với query là chính câu hỏi. Agentic RAG giao việc tìm kiếm cho model: model quyết định có cần tìm không, tìm gì, đọc kết quả, rồi tìm tiếp hay trả lời. Ba hướng nghiên cứu đã đặt nền cho ý tưởng này.

**Self-RAG** (Asai và cộng sự, 2023) huấn luyện model sinh token đặc biệt để tự quyết khi nào cần retrieve và tự chấm đoạn văn lấy về có liên quan không, câu trả lời có được đoạn văn đó ủng hộ không [6]. Self-RAG 7B và 13B vượt ChatGPT trên PopQA, PubHealth và Bio, nhưng thua trên ARC và ASQA [6]. Cách này cần fine-tune model nên ít dùng trực tiếp với API, nhưng các bước tự kiểm tra của nó là khuôn mẫu cho prompt của agent.

**CRAG** (Corrective RAG, Yan và cộng sự, 2024) thêm một evaluator nhỏ (T5-large đã fine-tune) chấm độ liên quan của tài liệu lấy về, rồi rẽ ba nhánh: *Correct* thì lọc bớt câu thừa trong tài liệu, *Incorrect* thì bỏ tài liệu và tìm trên web, *Ambiguous* thì dùng cả hai [7]. Ngưỡng rẽ nhánh được chỉnh riêng cho từng dataset (ví dụ 0,59 và −0,99 cho PopQA, 0,95 và −0,91 cho Biography) [7]. Trên generator SelfRAG-LLaMA2-7b:

| Hệ thống | PopQA (acc) | Bio (FactScore) | PubHealth (acc) | ARC (acc) |
|---|---|---|---|---|
| RAG | 52,8 | 59,2 | 39,0 | 53,2 |
| CRAG | 59,8 | 74,1 | 75,6 | 68,6 |
| Self-RAG | 54,9 | 81,2 | 72,4 | 67,3 |
| Self-CRAG | 61,8 | 86,2 | 74,8 | 67,2 |

Ngưỡng khác nhau theo dataset là chi tiết đáng nhớ: một evaluator "relevant/irrelevant" không có ngưỡng chung, bạn phải chỉnh trên dữ liệu của mình.

**Adaptive-RAG** (Jeong và cộng sự, 2024) đặt một classifier nhỏ trước pipeline, phân câu hỏi thành ba mức: không cần retrieve, retrieve một lần, retrieve nhiều bước [8]. Trên FLAN-T5-XL (3B), trung bình 6 dataset:

| Chiến lược | F1 | Số bước | Thời gian (so với retrieve một lần) |
|---|---|---|---|
| Không retrieve | 21,12 | 0 | 0,11× |
| Retrieve một lần | 44,31 | 1,00 | 1,00× |
| Adaptive-RAG | 46,94 | 2,17 | 3,60× |
| Luôn nhiều bước | 48,85 | 4,69 | 8,81× |

Routing không thắng về chất lượng: luôn chạy nhiều bước vẫn cho F1 cao nhất. Nó thắng về chi phí, đạt phần lớn mức tăng với chưa tới một nửa thời gian. Đó là đánh đổi bạn sẽ gặp trong mọi hệ agentic.

**Chi phí trong thực tế.** Anthropic đo trong hệ research của họ: một agent dùng khoảng 4 lần số token của một lượt chat, hệ nhiều agent khoảng 15 lần [9]. Trên BrowseComp, lượng token dùng giải thích phần lớn chênh lệch điểm giữa các lần chạy [9]. Tìm nhiều hơn thì tốt hơn, và đắt hơn tương ứng.

**Khi nào dùng agentic RAG:**

- Câu hỏi nhiều bước, bước sau cần kết quả bước trước ("ai là trưởng nhóm của người đã duyệt PR này").
- Query đầu tiên thường tệ và cần sửa sau khi thấy kết quả.
- Nguồn dữ liệu nhiều loại (vector store, SQL, API nội bộ), model phải chọn nguồn.

**Khi nào không:** chatbot cần trả lời dưới vài giây cho mọi câu hỏi, câu hỏi phần lớn là tra cứu một sự kiện. Với những hệ này, dùng routing như Adaptive-RAG: phân loại rẻ trước, chỉ bật vòng agentic cho nhóm câu khó.

Khi đã bật, vòng agentic cần đủ những thứ mà [Agents §3.2](../agents/#32-điều-kiện-dừng-giới-hạn-bước-và-ngân-sách) mô tả: giới hạn số vòng, ngân sách token, điều kiện dừng, và log từng lượt tìm. Retrieval đưa vào agent dưới dạng một tool; cách viết tool search và trả kết quả có trích dẫn ở [§2.8](#28-trích-dẫn-nguồn-và-grounding).

---

# Phần 2 — Kỹ thuật

## 2.1 Query rewriting, HyDE, multi-query, decomposition

Câu hỏi của người dùng thường là query tệ: ngắn, mơ hồ, dùng từ khác tài liệu, gộp nhiều ý. Biến đổi query là dùng một lần gọi LLM để viết lại nó trước khi tìm.

**HyDE** (Gao và cộng sự, 2022): LLM viết một đoạn văn giả định trả lời câu hỏi, rồi embed đoạn văn đó thay cho câu hỏi [10]. Lý do: đoạn văn trả lời giống tài liệu hơn là câu hỏi giống tài liệu. Paper báo cáo HyDE vượt hẳn retriever không giám sát Contriever và ngang retriever đã fine-tune trên nhiều tác vụ [10]. Rủi ro: đoạn văn giả có thể chứa tên, số, thực thể bịa, kéo retrieval về sai hướng. Giữ nhánh query gốc song song và trộn kết quả.

**Multi-query / RAG-Fusion** (Rackauckas, 2024): LLM sinh vài cách diễn đạt khác của câu hỏi, retrieve cho từng cách, trộn các danh sách bằng RRF [11]. Bằng chứng trong paper là đánh giá thủ công trên câu hỏi về sản phẩm của Infineon, không có benchmark công khai; tác giả ghi nhận một số câu trả lời lạc đề khi các query sinh ra không đủ sát câu hỏi gốc [11].

**Step-back prompting** (Zheng và cộng sự, 2023): sinh một câu hỏi trừu tượng hơn trước ("nguyên lý vật lý nào liên quan?"), tìm theo câu đó, rồi mới trả lời câu cụ thể [12]. Số liệu của paper đo trên tác vụ suy luận (MMLU, TimeQA, MuSiQue), không phải đo retrieval riêng [12].

**Decomposition và iterative retrieval.** Câu "so sánh chính sách bảo hành 2025 và 2026" tách thành hai query, mỗi query một lần retrieve. Với câu nhiều bước, IRCoT (Trivedi và cộng sự, 2022) xen kẽ: mỗi bước suy luận sinh ra một query mới, retrieve, suy luận tiếp [13]. Trên HotpotQA, 2WikiMultihopQA, MuSiQue và IIRC, IRCoT với GPT-3 tăng tới 21 điểm recall và tới 15 điểm QA so với retrieve một lần [13]. Bước đầu sai thì các bước sau sai theo, nên log từng query trung gian để debug.

| Kỹ thuật | Bằng chứng | Gọi LLM thêm | Retrieval thêm | Hỏng thế nào |
|---|---|---|---|---|
| HyDE | Benchmark retrieval zero-shot [10] | 1 | 0 | Đoạn văn giả bịa thực thể |
| Multi-query + RRF | Đánh giá thủ công, một công ty [11] | 1 | N | Query sinh ra lạc đề |
| Step-back | Benchmark suy luận [12] | 1 | 0–1 | Mất chi tiết cụ thể |
| Decomposition | Nằm trong Adaptive-RAG, IRCoT | 1 | Số câu con | Tách sai, mất quan hệ giữa các phần |
| IRCoT | Benchmark QA nhiều bước [13] | 1 mỗi bước | 1 mỗi bước | Lỗi tích lũy, độ trễ tăng theo số bước |

Tất cả các kỹ thuật này nằm trên đường găng: người dùng chờ thêm một lần gọi LLM trước khi retrieval bắt đầu. Dùng model nhỏ và nhanh cho bước viết lại, và chỉ bật cho nhóm câu hỏi cần (routing ở [§1.4](#14-agentic-rag-model-tự-quyết-khi-nào-và-tìm-gì)).

**RRF: trộn nhiều danh sách xếp hạng.** Reciprocal Rank Fusion cộng điểm 1/(k + rank) của mỗi tài liệu trên mọi danh sách, không cần chuẩn hoá điểm giữa BM25 và cosine [14]. Cormack và cộng sự cố định k = 60 sau một thử nghiệm sơ bộ và viết rằng giá trị này gần tối ưu, nhưng lựa chọn không quan trọng lắm [14]. Nhiều search engine và dịch vụ managed có RRF dựng sẵn; ví dụ vector store của OpenAI cho chỉnh trọng số RRF giữa phần embedding và phần text qua `ranking_options.hybrid_search` [36]. Khi tự trộn:

```python
def rrf(rankings: dict[str, list[str]], k: int = 60,
        weights: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """rankings: tên retriever -> danh sách doc id, tốt nhất đứng đầu."""
    scores: dict[str, float] = {}
    for name, ids in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for rank, doc_id in enumerate(dict.fromkeys(ids), start=1):   # bỏ id trùng, giữ thứ hạng đầu
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

fused = rrf({"bm25": bm25_ids, "dense": dense_ids, "hyde": hyde_ids},
            weights={"hyde": 0.5})        # nhánh HyDE nhẹ ký hơn vì có thể bịa
```

`dict.fromkeys` loại id trùng trong cùng một danh sách. Chuyện này xảy ra khi nhiều chunk của cùng tài liệu được map về doc id: không loại thì một tài liệu được cộng điểm nhiều lần chỉ vì nó dài.

## 2.2 Reranking: cross-encoder và reranker API

Reranker là cross-encoder: đọc câu hỏi và đoạn văn cùng lúc, cho một điểm liên quan. Chính xác hơn bi-encoder nhiều nhưng không index trước được, nên chỉ chạy trên vài chục tới vài trăm candidate mà retrieval đã lấy về. Nguyên lý đã có ở [Embeddings §2.2](../embeddings/#22-rag--rerank-pipeline-3-giai-đoạn). Phần này là bảng chọn và chi phí.

Bài contextual retrieval của Anthropic cho một ví dụ về tác động: thêm rerank (lấy 150 candidate, giữ 20) làm tỉ lệ retrieval thất bại giảm từ 2,9% xuống 1,9% trên nền đã có contextual embedding và BM25 [21].

**Reranker qua API (tháng 11/2026)**

| Provider | Model | Context mỗi cặp | Cách tính phí | Ghi chú |
|---|---|---|---|---|
| Cohere | `rerank-v5.0-pro`, `rerank-v5.0-fast` (và các bản v4, v3.5) | Xem docs | Theo "search": 1 query với tối đa 100 tài liệu; tài liệu trên 500 token bị tách và mỗi phần tính là một tài liệu [15] | Trang giá không niêm yết giá Rerank 5 tại thời điểm viết. Trên Bedrock, Cohere Rerank 3.5 là $2,00 / 1.000 query [39] |
| Voyage | `rerank-3`, `rerank-3-lite` | 32.000 token (query + một tài liệu); tối đa 1.000 tài liệu mỗi request [16] | $0,05 / triệu token (`rerank-3`), $0,02 (`lite`), với số token = token query × số tài liệu + tổng token tài liệu [17] | Query tối đa 8.000 token; tổng mỗi request tối đa 600K token [16] |
| Jina | `jina-reranker-v3.5`, `v3`, `m0` (tài liệu dạng ảnh) | 131K token (v3.5) [18] | Theo token, khoảng $0,05 / triệu token ở gói nạp 1 tỷ token [18] | Trọng số mở theo giấy phép CC-BY-NC: tự host cho sản phẩm thương mại cần giấy phép riêng |

Công thức tính token của Voyage có một hệ quả đáng chú ý: query dài bị nhân với số tài liệu. Rerank 100 tài liệu, mỗi tài liệu 300 token, query 50 token: 100 × 50 + 100 × 300 = 35.000 token, khoảng $0,00175 với `rerank-3`. Query 2.000 token (ví dụ cả lịch sử hội thoại) với cùng 100 tài liệu thành 230.000 token, đắt gấp gần bảy lần. Đưa câu hỏi đã viết lại cho reranker, không đưa cả hội thoại.

**Reranker tự host**

| Model | Kích thước | Giấy phép | Ghi chú |
|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` | 0,6B | Apache-2.0 | Đa ngôn ngữ, phổ biến làm baseline [19] |
| `Qwen/Qwen3-Reranker` | 0,6B, 4B, 8B | Apache-2.0 | Context 32K; model card công bố điểm theo MTEB-R, CMTEB-R, MMTEB-R, MLDR, MTEB-Code [20] |

Model card của Qwen3-Reranker có một chi tiết nên kiểm tra trước khi chọn bản lớn nhất: trên MTEB-R, bản 8B (69,02) thấp hơn bản 4B (69,76) [20]. Model lớn hơn không phải lúc nào cũng tốt hơn trên mọi tác vụ.

**LLM làm reranker.** Một model chat chấm từng đoạn ("đoạn này có trả lời câu hỏi không, có/không") là reranker theo điểm. Cách này chậm và đắt hơn cross-encoder nhiều, nhưng hiểu được tiêu chí phức tạp ("chỉ tính tài liệu áp dụng cho khách hàng doanh nghiệp"). Cách thiết kế bộ câu hỏi có/không và tính điểm ở [Decision models §3.4](../decision-models/#34-lọc-passage-giữa-retrieval-và-generation).

**Độ trễ.** Không có benchmark độ trễ nào so sánh có kiểm soát các reranker trên cùng phần cứng và cùng số candidate. Con số trên model card đo trên phần cứng khác nhau với batch size không ghi rõ. Tự đo trên phần cứng và số candidate thật của bạn, và ghi p95 chứ không chỉ trung bình. Độ trễ rerank tăng gần tuyến tính theo số candidate, nên số candidate là núm vặn chính. Không có con số đúng chung: đo Recall@N và p95 độ trễ ở vài giá trị N trên tập eval của bạn (ví dụ 50, 100, 150; thí nghiệm của Anthropic dùng 150 [21]), rồi chọn N nhỏ nhất mà Recall@N không còn tăng đáng kể.

## 2.3 Contextual retrieval và chunk có ngữ cảnh

Chunk "Điều khoản này không áp dụng cho đơn hàng dưới 500.000 đồng" không nói "điều khoản này" là gì. Embedding của nó không chứa tên chính sách, BM25 không có từ khoá nào để khớp. Contextual retrieval của Anthropic sửa bằng cách cho LLM viết một đoạn ngữ cảnh ngắn cho từng chunk, dựa trên toàn bộ tài liệu, rồi ghép đoạn đó vào đầu chunk trước khi embed và trước khi đưa vào index BM25 [21].

**Kết quả** (1 − recall@20, trung bình nhiều domain, embedding Gemini Text 004) [21]:

| Cấu hình | Tỉ lệ retrieval thất bại | Giảm so với baseline |
|---|---|---|
| Embedding + BM25 | 5,7% | — |
| Contextual embedding | 3,7% | 35% |
| Contextual embedding + contextual BM25 | 2,9% | 49% |
| Thêm rerank | 1,9% | 67% |

**Cách làm.** Prompt trong bài gốc, dùng với Claude 3 Haiku lúc đó [21]:

```text
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
```

Đoạn ngữ cảnh sinh ra thường dài 50–100 token [21]. Vì mọi chunk của một tài liệu dùng chung phần `<document>`, đặt phần đó ở đầu prompt và cache nó:

```python
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait

import anthropic

client = anthropic.Anthropic()          # SDK tự retry lỗi 429 và 5xx (mặc định 2 lần)

CHUNK_PROMPT = """Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document \
for the purposes of improving search retrieval of the chunk. \
Answer only with the succinct context and nothing else."""

def _situate(doc_text: str, chunk: str, model: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": f"<document>\n{doc_text}\n</document>",
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": CHUNK_PROMPT.format(chunk=chunk)},
        ]}],
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"context bị cắt hoặc bị từ chối: {resp.stop_reason}")
    context = "".join(b.text for b in resp.content if b.type == "text").strip()
    return f"{context}\n\n{chunk}"               # bản này dùng cho cả embedding lẫn BM25

def contextualize(doc_text: str, chunks: list[str], *, workers: int,
                  model: str = "claude-haiku-5-5") -> list[str]:
    """workers: số request song song, đặt theo rate limit của tài khoản."""
    if workers < 1:
        raise ValueError("workers phải >= 1")
    if not chunks:
        return []
    first = _situate(doc_text, chunks[0], model)   # chờ request đầu ghi cache xong
    pool = ThreadPoolExecutor(max_workers=workers)  # các chunk sau đọc cache, chạy song song
    try:
        futures = [pool.submit(_situate, doc_text, c, model) for c in chunks[1:]]
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        for f in done:
            if f.exception() is not None:
                raise f.exception()
        return [first] + [f.result() for f in futures]
    finally:
        pool.shutdown(cancel_futures=True)          # một chunk lỗi: huỷ các chunk chưa chạy
```

Vài điểm khi đưa vào production:

- **Cache phải được ghi trước khi đọc.** Nếu gửi song song mọi chunk của một tài liệu ngay từ đầu, các request đến trước khi cache được ghi sẽ đều trả giá ghi cache. Hàm trên gửi chunk đầu, chờ xong, rồi mới chạy song song các chunk còn lại. Một chunk lỗi (sau khi SDK đã retry) làm `contextualize` ném exception cho cả tài liệu và huỷ các chunk còn trong hàng đợi; các request đang chạy dở vẫn chạy hết và kết quả của chúng bị bỏ. Đơn vị để đánh dấu đã xong và chạy lại ở đây là tài liệu, kết hợp với hash ở [§3.2](#32-cập-nhật-index-khi-dữ-liệu-thay-đổi). Tài liệu có hàng trăm chunk thì lưu kết quả của từng chunk theo hash (nội dung chunk, phiên bản prompt) ngay khi có, để lần chạy lại chỉ gọi model cho các chunk còn thiếu. Corpus lớn thì chạy song song thêm ở mức tài liệu, và tổng số request đồng thời vẫn phải nằm trong rate limit. Tài liệu ngắn hơn ngưỡng tối thiểu của model thì không được cache; ngưỡng và giá ở [Multi-turn §2.2](../multi-turn-conversation/#22-claude-anthropic). Kiểm tra `usage.cache_read_input_tokens` của các request sau để chắc cache có khớp.
- **Chi phí.** Anthropic ước tính $1,02 cho mỗi triệu token tài liệu, chi phí một lần, với chunk 800 token, tài liệu 8K token, có prompt caching [21]. Con số tính theo giá Claude 3 Haiku năm 2024; tính lại theo model và giá bạn dùng.
- **Tài liệu dài hơn cửa sổ của model rẻ** thì không đưa nguyên được. Đưa phần mục lục, phần mở đầu và section chứa chunk thay vì cả tài liệu.
- **Ngữ cảnh do LLM viết có thể sai.** Kiểm tra ngẫu nhiên vài chục chunk trước khi chạy cho cả corpus. Lỗi hay gặp: gán chunk cho sai section khi tài liệu có nhiều phần cùng cấu trúc.
- **Đổi prompt là index lại toàn bộ.** Hash của chunk nên gồm cả phiên bản prompt ([§3.2](#32-cập-nhật-index-khi-dữ-liệu-thay-đổi)).
- **Khi trả lời,** có thể đưa cho model cả ngữ cảnh lẫn chunk gốc, nhưng đánh dấu rõ phần nào là ngữ cảnh do máy viết, phần nào là nguyên văn tài liệu. Trích dẫn chỉ nên trỏ vào nguyên văn.

**Hướng khác: contextualized chunk embedding.** Một số embedding model nhận cả tài liệu và trả vector cho từng chunk đã mang ngữ cảnh, không cần bước LLM riêng (ví dụ `voyage-context-4`, xem [Embeddings §3.3](../embeddings/#33-chunking-strategies)). Late chunking ở [§2.4](#24-parentchild-sentence-window-và-late-chunking) cùng ý tưởng. Hai hướng này rẻ hơn nhiều lúc index, nhưng không cải thiện BM25, vì chúng không thêm chữ nào vào chunk.

## 2.4 Parent–child, sentence window và late chunking

Có một mâu thuẫn trong việc chọn kích thước chunk: chunk nhỏ khớp chính xác hơn khi tìm, chunk lớn cho model đủ ngữ cảnh khi đọc. Các kỹ thuật ở đây tách hai việc đó ra.

**Parent–child (small-to-big).** Index chunk nhỏ, khi một chunk nhỏ khớp thì trả về chunk cha lớn hơn cho model. LlamaIndex dựng sẵn dạng phân cấp: `HierarchicalNodeParser` mặc định chia ba tầng 2048 → 512 → 128 token, và `AutoMergingRetriever` gộp các chunk con về chunk cha khi tỉ lệ con được lấy về vượt `simple_ratio_thresh`, mặc định 0,5 [24]. Nghĩa là nếu hơn một nửa số chunk con của một đoạn cha cùng khớp, model nhận cả đoạn cha thay vì các mảnh rời.

**Sentence window.** Index từng câu, khi một câu khớp thì trả về câu đó kèm vài câu trước và sau. `SentenceWindowNodeParser` của LlamaIndex mặc định lấy 3 câu mỗi phía (`window_size=3`), lưu cửa sổ trong metadata key `window` [23]; sau retrieval dùng `MetadataReplacementPostProcessor(target_metadata_key="window")` để thay câu bằng cửa sổ. Hợp với văn bản dày đặc như hợp đồng, nơi một câu là một điều khoản.

**Late chunking** (Günther và cộng sự, 2024): đưa cả tài liệu qua một embedding model có context dài, rồi mới chia thành chunk ở tầng token và lấy trung bình cho từng chunk [22]. Mỗi vector chunk vì thế đã "nhìn thấy" phần còn lại của tài liệu. Không cần huấn luyện thêm. Trên 4 dataset BEIR (SciFact, NFCorpus, FiQA, TREC-COVID), trung bình qua 3 model, nDCG@10 tăng khoảng 1,5–1,9 điểm tuỳ cách chia chunk [22]. Mức tăng không đều: lớn nhất ở NFCorpus với model jina-v2-small, gần như không có với model nomic trên SciFact và NFCorpus [22]. Điều kiện: model embedding phải có context đủ dài cho tài liệu, và bạn phải truy cập được embedding ở tầng token, nên với API thường phải dùng API có hỗ trợ sẵn.

| Tình huống | Chọn |
|---|---|
| Chunk thiếu chủ ngữ, tài liệu có cấu trúc rõ | Contextual retrieval ([§2.3](#23-contextual-retrieval-và-chunk-có-ngữ-cảnh)) hoặc late chunking |
| Cần model đọc cả đoạn xung quanh | Parent–child hoặc sentence window |
| Tài liệu có heading rõ (docs kỹ thuật, wiki) | Chia theo heading, gắn đường dẫn heading vào chunk ([Embeddings §3.3](../embeddings/#33-chunking-strategies)) |
| Văn bản pháp lý, mỗi câu là một ý | Sentence window |

Lưu ý khi dùng parent–child cùng rerank: rerank trên chunk con (nhỏ, chính xác) rồi mới mở rộng ra chunk cha. Rerank trên chunk cha thì mất lợi thế chunk nhỏ, và tốn token reranker hơn.

## 2.5 GraphRAG và knowledge graph

Khái niệm GraphRAG, HippoRAG, LightRAG đã có ở [Embeddings §2.13](../embeddings/#213-graphrag--kết-hợp-knowledge-graph-và-embedding). Phần này trả lời câu hỏi thực tế hơn: khi nào graph đáng tiền.

**GraphRAG của Microsoft** (Edge và cộng sự, 2024) dùng LLM trích entity và quan hệ từ toàn bộ corpus, gom thành các cộng đồng (community), và viết báo cáo tóm tắt cho từng cộng đồng [25]. Câu hỏi tổng hợp ("chủ đề chính trong bộ dữ liệu là gì") được trả lời bằng cách map-reduce trên các báo cáo này. Paper báo cáo GraphRAG thắng RAG thường về độ toàn diện và độ đa dạng trên loại câu hỏi đó, với corpus cỡ 1 triệu token [25].

**Chi phí index.** Toàn bộ corpus phải qua LLM ít nhất một lần, cộng thêm bước viết báo cáo. Tài liệu của Microsoft ước tính bước trích graph chiếm khoảng 75% chi phí index [26]. Index lại khi dữ liệu đổi cũng tốn tương ứng.

**LazyGraphRAG** (Microsoft Research, 11/2024) hoãn phần đắt sang lúc truy vấn: index chỉ dùng NLP trích cụm danh từ, không dùng LLM tóm tắt. Microsoft báo cáo chi phí index bằng vector RAG và bằng 0,1% GraphRAG đầy đủ, chất lượng câu hỏi tổng hợp tương đương GraphRAG global search với chi phí truy vấn thấp hơn hơn 700 lần [27]. Bằng chứng giới hạn: một bộ dữ liệu (5.590 bài báo AP), 100 câu hỏi, chấm bằng LLM so cặp [27].

**Bằng chứng khi graph không thắng.** Ba nghiên cứu độc lập đáng đọc trước khi quyết định:

- *Han và cộng sự (2025)*, generator Llama-3.1-8B-Instruct: RAG thường thắng ở câu hỏi một bước và câu hỏi về chi tiết; GraphRAG thắng ở câu hỏi nhiều bước. Trên NQ, RAG đạt F1 64,78, phương pháp graph tốt nhất 63,01. Trên MultiHop-RAG, HippoRAG2 đạt 70,27 so với RAG 67,02. GraphRAG global search (community) yếu trên tác vụ hỏi đáp. Kết hợp cả hai thắng từng cái riêng ở phần lớn trường hợp [28].
- *GraphRAG-Bench (Xiang và cộng sự, 2025)*, GPT-4o-mini, độ chính xác trên bộ Novel: tra cứu sự kiện, RAG có rerank 60,92 so với HippoRAG2 60,14 (ngang nhau); suy luận phức tạp, HippoRAG2 53,38 so với RAG 42,93; tóm tắt theo ngữ cảnh, MS-GraphRAG local 64,40 so với RAG 51,30 [29]. Chi phí token mỗi câu hỏi theo hình 8 của paper: vector RAG khoảng 879, HippoRAG2 khoảng 1.008, MS-GraphRAG local khoảng 38.707, MS-GraphRAG global khoảng 331.375 [29]. Phần văn bản của paper có một chỗ ghi prompt global search "tới 4×10⁴ token", không khớp với hình; dù theo số nào, global search cũng đắt hơn vector RAG nhiều bậc.
- *Zeng và cộng sự (2025)* chỉ ra rằng các đánh giá GraphRAG trước đó có câu hỏi không liên quan và thiên lệch khi dùng LLM chấm. Sau khi khử thiên lệch, tỉ lệ thắng của LightRAG trước NaiveRAG trên bộ Agriculture giảm từ 66,70% (theo paper LightRAG) xuống 39,06%, tức NaiveRAG thắng [30].

**Tiêu chí chọn:**

| Loại câu hỏi | Graph có đáng không |
|---|---|
| Tra cứu một sự kiện, một điều khoản | Không. RAG ngang hoặc hơn, graph chỉ thêm chi phí |
| Nhiều bước, nối thông tin qua nhiều tài liệu | Có thể. HippoRAG2 thắng trên MultiHop-RAG và nhóm suy luận phức tạp, với chi phí token gần vector RAG |
| Tổng hợp toàn corpus ("vấn đề lặp lại nhiều nhất") | Có, đây là chỗ GraphRAG được thiết kế cho. Cân nhắc LazyGraphRAG hoặc map-reduce trên tóm tắt trước |
| Dữ liệu thay đổi hằng ngày | Nghiêng về vector RAG; graph phải cập nhật tăng dần hoặc index lại |

Quy tắc an toàn: phân loại câu hỏi thật của người dùng trước (lấy từ log), đo tỉ lệ từng loại. Nếu câu hỏi tổng hợp chỉ chiếm 3% traffic, một pipeline map-reduce trên tóm tắt tài liệu, chạy riêng cho 3% đó qua routing, thường rẻ hơn dựng và duy trì một graph cho cả corpus. Khi so sánh graph với vector RAG trên eval của bạn, dùng câu hỏi có đáp án kiểm tra được, đừng chỉ dùng LLM chấm so cặp, vì đó chính là cách đánh giá mà Zeng và cộng sự chỉ ra là thiên lệch [30].

## 2.6 Metadata filtering và structured retrieval (text-to-SQL)

Nhiều câu hỏi không phải tìm kiếm ngữ nghĩa. "Hợp đồng nào hết hạn trong tháng 12?", "Có bao nhiêu ticket về lỗi thanh toán tuần này?", "Chính sách hoàn tiền áp dụng cho Việt Nam". Embedding không biết so sánh ngày, không đếm được, và "Việt Nam" với "Thái Lan" nằm khá gần nhau trong không gian vector.

**Metadata filter.** Gắn thuộc tính có cấu trúc cho mỗi tài liệu khi index (ngày hiệu lực, quốc gia, sản phẩm, phiên bản, loại tài liệu, nhóm được phép đọc) và lọc cứng trong truy vấn vector. OpenAI vector store gọi là `attributes` trên file, lọc bằng `filters` với so sánh (`eq`, `gt`, `in`…) và kết hợp `and`/`or` [36]. Gemini File Search có `custom_metadata` trên tài liệu và `metadata_filter` khi truy vấn [38]. Vector DB tự host đều có cơ chế tương đương.

Lấy giá trị filter từ câu hỏi có ba cách, theo mức tin cậy giảm dần:

1. **Từ ngữ cảnh ứng dụng.** Người dùng đang ở trang sản phẩm X, tài khoản thuộc quốc gia Y. Không cần model đoán.
2. **Từ giao diện.** Bộ lọc người dùng tự chọn.
3. **Model trích từ câu hỏi.** Dùng structured output với schema có enum cho các giá trị hợp lệ (danh sách sản phẩm, quốc gia), xem [Structured output](../structured-output/). Khi model không chắc, để trống thay vì đoán: filter sai loại mất tài liệu đúng, còn thiếu filter chỉ làm kết quả nhiễu hơn.

Filter trả về 0 kết quả thì phân biệt hai loại. Filter là điều kiện người dùng hoặc ứng dụng đặt ra ("hết hạn trong tháng 12", quốc gia của tài khoản, bộ lọc trên giao diện) thì giữ nguyên và trả lời "không tìm thấy" kèm điều kiện đã dùng; nới nó là trả lời một câu hỏi khác, và với filter theo quyền còn là lộ dữ liệu. Chỉ nới filter do hệ thống tự thêm hoặc model tự đoán khi câu hỏi không nói (ví dụ mặc định lấy phiên bản mới nhất), theo thứ tự đã định, và cho người dùng thấy điều kiện nào đã bị bỏ.

**Text-to-SQL.** Khi đáp án nằm trong bảng (đơn hàng, ticket, doanh thu), cho model viết SQL rồi chạy, thay vì cố embed từng dòng. Bedrock Knowledge Bases có tính năng sinh SQL dựng sẵn cho dữ liệu có cấu trúc [39]. Tự làm thì tối thiểu cần:

- **Quyền đọc hạn chế ở tầng database**, không chỉ trong prompt: một role chỉ có `SELECT` trên các view được duyệt. Row-level security nếu người dùng chỉ được thấy dữ liệu của họ.
- **Schema đưa cho model là schema đã chọn lọc**: các view có tên cột dễ hiểu, kèm mô tả cột và vài dòng mẫu. Với DB nhiều bảng, thêm một bước chọn bảng trước bước sinh SQL.
- **Kiểm tra câu SQL trước khi chạy**: parse bằng thư viện như `sqlglot`, chỉ chấp nhận một câu `SELECT`, chỉ trên các bảng/view được phép.
- **Giới hạn ở tầng chạy**: timeout cho câu lệnh, giới hạn số dòng trả về.
- **Trả cho người dùng cả câu SQL** (hoặc diễn giải của nó), để họ thấy con số được tính thế nào.

Eval cho text-to-SQL so kết quả chạy, không so chuỗi SQL: hai câu SQL khác nhau có thể cùng đúng.

## 2.7 RAG đa phương thức: bảng, ảnh, PDF phức tạp

Ba cách đưa bảng, biểu đồ và ảnh vào retrieval (chuyển về text, embedding đa phương thức, retrieval trên ảnh trang) đã so sánh ở [Multimodal §3.4](../multimodal/#34-rag-đa-phương-thức). Phần này đi vào chi tiết vận hành: chia bảng, chi phí visual retrieval, và gửi trang PDF cho model.

**Chia bảng dài.** Bảng 200 dòng chia theo số token sẽ cho những chunk chỉ có số, không có header. Chunk "| 12 | 4,5 | 18% |" vô nghĩa với cả embedding lẫn model. Chia theo dòng, lặp lại header và caption trong mọi chunk:

```python
import re

def split_table(md_table: str, max_rows: int, caption: str = "") -> list[str]:
    """Chia bảng Markdown dài thành nhiều chunk, chunk nào cũng có caption và dòng header."""
    if max_rows < 1:
        raise ValueError("max_rows phải >= 1")
    lines = [l for l in md_table.strip().splitlines() if l.strip()]
    if len(lines) < 2 or not re.fullmatch(r"\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?", lines[1].strip()):
        raise ValueError("không phải bảng Markdown: thiếu dòng header hoặc dòng phân cách")
    header, sep, rows = lines[0], lines[1], lines[2:]
    chunks = []
    for i in range(0, max(len(rows), 1), max_rows):
        part = rows[i:i + max_rows]
        note = f"{caption} (dòng {i + 1}–{i + len(part)} / {len(rows)})" if caption else ""
        chunks.append("\n".join(([note] if note else []) + [header, sep] + part))
    return chunks
```

Thêm vào metadata của mỗi chunk: tài liệu nguồn, số trang, id của bảng. Khi một chunk bảng được retrieve, cân nhắc đưa cả bảng (hoặc ảnh trang chứa bảng) vào context, vì câu hỏi về bảng thường cần so sánh giữa các dòng không nằm cùng chunk.

**Visual retrieval (ColPali).** Embed ảnh trang trực tiếp, bỏ OCR. Trên benchmark ViDoRe, ColPali đạt nDCG@5 trung bình 81,3, so với 65,5–66,1 của pipeline OCR rồi BM25 hoặc BGE-M3 [31]. Chi phí chính là lưu trữ: mỗi trang khoảng 1.030 vector 128 chiều, tức 257,5 KB ở float16, so với 8,6 KB của BGE-M3 cho cùng trang [31]. Một triệu trang là khoảng 257 GB vector, chưa tính index. Thời gian index mỗi trang trên GPU L4 là 0,39 giây với ColPali, so với 7,22 giây của pipeline parse layout bằng Unstructured [31]. Repo gốc `illuin-tech/colpali` hiện ghi engine đã ngừng phát triển; kiểm tra model kế nhiệm trên ViDoRe leaderboard trước khi chọn.

**Gửi thẳng trang PDF cho model.** Claude, OpenAI và Gemini đều nhận PDF và đọc cả ảnh trang lẫn text layer. Giới hạn, số token mỗi trang và cách gửi của từng provider ở [Multimodal §1.3](../multimodal/#13-pdf-ảnh-trang-text-layer-hay-cả-hai) và [§2.1–2.3](../multimodal/#21-openai). Mẫu dùng hiệu quả: retrieval (text hoặc visual) chọn vài trang, rồi gửi các trang đó dạng PDF hoặc ảnh cho model đọc. Model thấy bảng đúng như người đọc thấy, không qua bước parse có thể sai.

| Loại tài liệu | Cách làm |
|---|---|
| Văn bản thuần, ít bảng | Pipeline text thông thường |
| Có bảng, bảng parse được sạch | Chia bảng giữ header; đưa cả bảng vào context khi cần |
| Bảng phức tạp (ô gộp, nhiều tầng header), biểu đồ | Retrieval chọn trang, gửi ảnh trang hoặc PDF cho model |
| Hàng triệu trang scan, OCR kém | Visual retrieval, chấp nhận chi phí lưu trữ |

## 2.8 Trích dẫn nguồn và grounding

Trích dẫn có hai công dụng: người dùng kiểm tra được câu trả lời, và bạn đo được model có bịa không. Cả hai chỉ có ý nghĩa nếu trích dẫn đúng. Cách tệ nhất là để model tự viết "[Nguồn: tài liệu 3]" trong văn bản tự do: model có thể đánh số sai, trích tài liệu không có trong context, hoặc trích đúng tài liệu nhưng câu được trích không có trong đó.

**Claude: Citations API.** Gửi tài liệu dưới dạng content block `document` với `"citations": {"enabled": true}`. Claude chia tài liệu thành câu (với text và PDF) và trả về các text block, mỗi block kèm danh sách citation trỏ vào vị trí chính xác: khoảng ký tự với text, khoảng trang với PDF, khoảng block với tài liệu dạng custom content [32]. Trường `cited_text` chứa nguyên văn đoạn được trích, không tính vào output token [32]. Muốn tự kiểm soát độ hạt của trích dẫn (mỗi đoạn, mỗi dòng bảng), dùng custom content: Claude chỉ trích nguyên block, không chia nhỏ thêm [32].

Với hệ RAG đã có retrieval riêng, block `search_result` hợp hơn. Tool search của bạn trả về kết quả dạng này trong `tool_result`, và Claude trích dẫn theo đúng nguồn bạn gán [33]:

```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_01...",
  "content": [
    {
      "type": "search_result",
      "source": "kb://policy/refund-2026#sec-3",
      "title": "Chính sách hoàn tiền 2026, mục 3",
      "content": [
        {"type": "text", "text": "Khách hàng gói Pro được hoàn tiền trong 14 ngày kể từ ngày thanh toán."},
        {"type": "text", "text": "Không áp dụng cho đơn hàng dưới 500.000 đồng."}
      ],
      "citations": {"enabled": true}
    }
  ]
}
```

Citation trả về có loại `search_result_location`, gồm `source`, `title`, `search_result_index` và khoảng block (`start_block_index`, `end_block_index`) [33]. Mấy ràng buộc cần biết [33]:

- Mọi `search_result` trong một request phải cùng bật hoặc cùng tắt citations.
- Nếu một `tool_result` chứa `search_result`, mọi block trong `tool_result` đó phải là `search_result`; trộn loại khác trả lỗi validation.
- Chỉ có text, không có ảnh trong `content`.
- Khi không tìm thấy gì, trả một text block "No results found." thay vì báo lỗi.

Đặt `source` là id ổn định do hệ thống của bạn cấp (URL hoặc `kb://...`), để trích dẫn map ngược về tài liệu gốc mà không cần model chép lại tên.

**Không dùng cùng structured output.** Bật citations cùng `output_config.format` thì API trả 400 [34] (ghi nhận cả ở [Structured output](../structured-output/)). Tài liệu citations gợi ý cách thay thế: yêu cầu model trả mỗi trường trên một dòng dạng `tên: giá trị`, rồi tự parse và gắn citation của từng text block vào dòng tương ứng [32]. Prompt đòi "chỉ trả JSON" thì request vẫn thành công nhưng có thể có rất ít hoặc không có citation nào [32].

**OpenAI: file_search.** Câu trả lời có annotation loại `file_citation` với `file_id`, `filename` và vị trí trong văn bản trả lời [35]. Mặc định response không chứa các đoạn đã tìm được; thêm `include=["file_search_call.results"]` để lấy chúng, cần cho việc kiểm tra trích dẫn và cho eval [35].

**Gemini: File Search.** Annotation loại `file_citation` trong nội dung trả về; nếu tài liệu có `custom_metadata`, metadata đó đi kèm citation [38]. Từ Gemini 3, File Search dùng được cùng structured output [38].

**Khi không có API trích dẫn: quote matching.** Yêu cầu model trả kèm mỗi khẳng định một cặp `(source_id, quote)` với quote là nguyên văn từ nguồn. Source id do bạn gán khi ghép context. Rồi kiểm tra bằng code:

```python
import re
import unicodedata

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).casefold()
    s = re.sub(r"[‘’]", "'", re.sub(r"[“”]", '"', s))
    return re.sub(r"\s+", " ", s).strip()

def check_citations(citations: list[dict], sources: dict[str, str]) -> list[dict]:
    """citations: [{"source_id": ..., "quote": ...}] do model trả về.
    sources: source_id -> văn bản đã đưa vào prompt. Trả về các trích dẫn không đạt."""
    bad = []
    for c in citations:
        sid, raw = c.get("source_id"), c.get("quote")
        if not isinstance(sid, str) or not isinstance(raw, str):
            bad.append({**c, "why": "source_id hoặc quote không phải chuỗi"})
            continue
        text = sources.get(sid)
        quote = _norm(raw)
        if text is None:
            bad.append({**c, "why": "source_id không có trong context"})
        elif not quote:
            bad.append({**c, "why": "quote rỗng"})
        elif quote not in _norm(text):
            bad.append({**c, "why": "quote không có trong nguồn"})
    return bad
```

`unicodedata.normalize("NFC")` quan trọng với tiếng Việt: cùng một chữ "ộ" có thể được lưu ở dạng dựng sẵn hoặc dạng tổ hợp dấu, nhìn giống hệt nhưng so chuỗi khác nhau. Kiểm tra này bắt được trích dẫn bịa, không bắt được trích dẫn có thật nhưng không ủng hộ khẳng định. Quote ngắn như "14 ngày" có thật trong nguồn thì qua kiểm tra; nhưng chuỗi càng ngắn càng dễ khớp tình cờ ở một chỗ không liên quan, nên khi hiển thị, highlight vị trí khớp để người đọc tự đối chiếu, và để phần "có ủng hộ khẳng định không" cho bước kiểm tiếp theo. Phần đó cần LLM judge hoặc người đọc (faithfulness ở [§3.5](#35-eval-cho-rag-nâng-cao)).

Trích dẫn không đạt thì làm gì tuỳ sản phẩm: bỏ khẳng định đó, hiện cảnh báo, hoặc sinh lại. Ghi tỉ lệ không đạt thành metric; nó tăng là dấu hiệu sớm của prompt hoặc model mới bịa nhiều hơn.

---

# Phần 3 — Thiết kế và vận hành

## 3.1 Chọn kỹ thuật theo triệu chứng

Bảng ở [Tra cứu nhanh](#tra-cứu-nhanh) liệt kê triệu chứng và kỹ thuật. Quy trình để dùng nó:

1. **Lấy 50–100 câu trả lời sai từ log hoặc eval.** Với mỗi câu, xác định điểm hỏng theo bảng ở [§1.1](#11-pipeline-rag-có-những-điểm-hỏng-nào): tài liệu đúng có trong corpus không, có trong top-50 không, có trong context không, model có dùng không.
2. **Đếm.** Nếu 60% lỗi là FP2 (không vào top-k), sửa retrieval. Nếu phần lớn là FP4 (có trong context mà không dùng), retrieval tốt hơn không giúp gì; giảm số đoạn đưa vào, sắp xếp lại, hoặc đổi prompt.
3. **Chọn kỹ thuật rẻ nhất cho nhóm lỗi lớn nhất**, theo thứ tự ở Tra cứu nhanh.
4. **Đo lại trên cùng tập**, cộng với một tập câu hỏi đang trả lời đúng, để chắc kỹ thuật mới không làm hỏng chỗ đang chạy tốt.

Mỗi kỹ thuật thêm vào là thêm một thành phần phải vận hành, thêm độ trễ, thêm chỗ hỏng. Pipeline có HyDE, multi-query, rerank, contextual retrieval và GraphRAG cùng lúc rất khó debug khi một câu trả lời sai. Chỉ giữ kỹ thuật nào eval chứng minh được giá trị trên dữ liệu của bạn.

## 3.2 Cập nhật index khi dữ liệu thay đổi

Index là bản sao của dữ liệu nguồn, và bản sao luôn có nguy cơ lệch. Ba việc phải làm được: thêm và sửa tài liệu, xoá tài liệu, index lại toàn bộ khi pipeline đổi.

**Phát hiện thay đổi bằng hash.** Lưu hash nội dung của mỗi tài liệu đã index. Lần đồng bộ sau, so hash với nguồn: khác thì index lại tài liệu đó, không còn ở nguồn thì xoá. Đưa phiên bản pipeline vào hash, để đổi cách chia chunk, prompt contextual hay embedding model sẽ tự kích hoạt index lại toàn bộ:

```python
import hashlib

def plan_sync(source_docs: dict[str, str], indexed: dict[str, str], pipeline_version: str):
    """source_docs: doc_id -> nội dung hiện tại. indexed: doc_id -> hash đã index.
    Hash gồm cả phiên bản pipeline: đổi chunking, prompt contextual hay embedding model thì index lại hết."""
    def h(text):
        return hashlib.sha256(f"{pipeline_version}\x00{text}".encode()).hexdigest()
    current = {d: h(t) for d, t in source_docs.items()}
    upsert = [d for d, hv in current.items() if indexed.get(d) != hv]
    delete = [d for d in indexed if d not in current]
    return upsert, delete, current
```

Index lại một tài liệu nghĩa là xoá mọi chunk cũ của nó rồi thêm chunk mới. Số chunk có thể đổi sau khi sửa, nên đừng ghi đè theo vị trí chunk; gắn `doc_id` vào mọi chunk và xoá theo `doc_id`.

**Xoá.** Xoá là chỗ hay sót nhất, vì nguồn thường không báo "tài liệu này đã bị xoá"; nó chỉ không còn trả về tài liệu đó nữa. Cách hàm trên phát hiện xoá (có trong index, không có ở nguồn) chỉ đúng khi bạn liệt kê được toàn bộ nguồn. Với nguồn quá lớn để liệt kê mỗi lần, dùng cơ chế của nguồn: change feed, soft-delete flag, webhook. Một số dịch vụ managed không xoá tức thì: OpenAI ghi rằng xoá file khỏi vector store là eventually consistent, nội dung đã xoá có thể còn xuất hiện trong kết quả tìm kiếm một thời gian ngắn [36]. Nếu tài liệu bị xoá vì lý do pháp lý hoặc quyền riêng tư, lọc thêm theo danh sách đã xoá ở tầng ứng dụng cho tới khi chắc index đã cập nhật.

**Index lại toàn bộ không gián đoạn.** Khi đổi embedding model hoặc cách chia chunk, dựng index mới song song, chạy eval trên index mới, rồi chuyển truy vấn sang bằng alias, giữ index cũ để quay lại nếu cần. Quy trình chi tiết ở [Embeddings §3.10](../embeddings/#310-migration--versioning--đổi-embedding-model-mà-không-làm-sập-hệ-thống). Trong thời gian dựng, các thay đổi ở nguồn phải ghi vào cả hai index.

**Chi phí của các kỹ thuật ở Phần 2 khi dữ liệu đổi:**

| Kỹ thuật | Khi một tài liệu thay đổi |
|---|---|
| Embedding thường | Embed lại chunk của tài liệu đó |
| Contextual retrieval | Gọi LLM cho mọi chunk của tài liệu đó (ngữ cảnh phụ thuộc cả tài liệu) |
| GraphRAG | Trích lại entity của tài liệu, và các community bị ảnh hưởng cần viết lại báo cáo |
| Managed (OpenAI, Gemini, Bedrock) | Upload lại file; dịch vụ tự chunk và embed |

## 3.3 Phân quyền tài liệu trong retrieval

Ba cách làm (post-filter, pre-filter, index riêng theo tenant) và các điểm dễ sót đã có ở [Embeddings §3.8.3](../embeddings/#383-kiểm-soát-truy-cập-trong-retrieval). Phần này là luồng xử lý một request và những chỗ các kỹ thuật ở Phần 2 làm rò rỉ.

**Filter quyền nằm trong truy vấn, không nằm trong prompt.** "Không được trả lời từ tài liệu mật" trong system prompt không phải kiểm soát truy cập. Model đã đọc tài liệu thì nội dung có thể lộ qua tóm tắt, qua diễn giải, hoặc qua prompt injection. Tài liệu người dùng không có quyền không được đi vào context.

Azure AI Search mô tả mẫu cơ bản: một trường kiểu `Collection(Edm.String)` chứa id các nhóm được đọc, đánh dấu `filterable` và không `retrievable`, truy vấn kèm filter `group_ids/any(g: search.in(g, 'group_id1, group_id2'))` [40]. Tài liệu đó cũng nói rõ rằng đây không phải cơ chế xác thực: id nhóm chỉ là chuỗi, và ứng dụng phải gắn filter vào mọi truy vấn [40].

**Luồng một request:**

1. Lấy danh sách nhóm của người dùng từ hệ thống định danh, lúc request, không lấy từ input của client.
2. Truy vấn vector store với filter nhóm (pre-filter).
3. Kiểm tra lại quyền với hệ thống nguồn cho các tài liệu trả về, nếu ACL trong index có thể đã cũ.
4. Lấy dư rồi lọc, khi bước 3 loại nhiều.
5. Chỉ tài liệu qua cả hai lớp mới đi vào reranker, cache và context.

```python
def retrieve_for_user(search, can_read, query: str, user, *, k: int,
                      fetch: int, max_fetch: int) -> list:
    """search(query, top_k, filter) -> list hit có .doc_id, đã lọc theo ACL lưu trong index.
    can_read(user, doc_ids) -> set doc_id user vẫn đọc được, hỏi hệ thống nguồn lúc truy vấn."""
    if not (k >= 1 and 1 <= fetch <= max_fetch):
        raise ValueError("cần k >= 1 và 1 <= fetch <= max_fetch")
    acl_filter = {"allowed_groups": {"$in": sorted(user.groups)}}
    while True:
        hits = search(query, top_k=fetch, filter=acl_filter)
        allowed = can_read(user, {h.doc_id for h in hits})
        kept = [h for h in hits if h.doc_id in allowed]
        if len(kept) >= k or len(hits) < fetch or fetch >= max_fetch:
            return kept[:k]
        fetch = min(fetch * 2, max_fetch)    # bị lọc nhiều: lấy dư rồi lọc lại
```

`max_fetch` giới hạn số lần nhân đôi để một người dùng gần như không có quyền gì không kéo theo truy vấn lớn dần. Khi hàm trả ít hơn `k` kết quả, đó là dữ kiện thật về những gì người dùng được đọc, không phải lỗi. Cú pháp filter (`$in`) tuỳ vector DB.

**Những chỗ Phần 2 làm rò rỉ:**

- **Contextual retrieval.** Đoạn ngữ cảnh được viết từ toàn bộ tài liệu. Nếu tài liệu có phần mật và phần công khai với quyền khác nhau, ngữ cảnh của chunk công khai có thể chứa thông tin từ phần mật. Tài liệu có quyền theo từng phần thì contextualize trong phạm vi từng phần.
- **GraphRAG.** Báo cáo community tổng hợp từ nhiều tài liệu với quyền khác nhau. Một báo cáo được viết từ tài liệu mật và tài liệu công khai thì người chỉ có quyền đọc tài liệu công khai không được đọc báo cáo đó. Với dữ liệu nhiều mức quyền, dựng graph riêng theo mức quyền, hoặc không dùng báo cáo tổng hợp.
- **Semantic cache.** Câu trả lời cache cho người dùng A không được trả cho người dùng B có quyền khác. Khoá cache phải gồm tập quyền.
- **Query rewriting và agentic RAG.** Query do model viết lại vẫn phải đi qua cùng filter; tool search của agent nhận danh tính người dùng từ ứng dụng, không từ tham số do model điền.
- **Trích dẫn.** Hiển thị tiêu đề hoặc link tài liệu trong trích dẫn là hiển thị sự tồn tại của tài liệu. Chỉ trích dẫn tài liệu đã qua filter.

**Test riêng cho quyền.** Tạo một người dùng có ít quyền, hỏi những câu mà đáp án chỉ có trong tài liệu họ không được đọc. Câu trả lời phải là "không tìm thấy", không phải đáp án đúng. Chạy bộ test này trong CI.

## 3.4 Độ trễ và chi phí của từng tầng

| Tầng | Chi phí | Độ trễ | Nguồn số liệu |
|---|---|---|---|
| Embed query | Rất nhỏ; một lần gọi embedding | Vài chục ms qua API | Tự đo |
| Vector search + BM25 | Hạ tầng | Vài ms tới vài chục ms với index vừa phải | Tự đo |
| Biến đổi query (HyDE, multi-query, step-back) | Một lần gọi LLM | Thêm một lần gọi LLM trên đường găng | [§2.1](#21-query-rewriting-hyde-multi-query-decomposition) |
| Rerank qua API | Voyage: ~$0,0018 cho 100 tài liệu 300 token [17]; Cohere trên Bedrock $2,00 / 1.000 query [39] | Tăng theo số candidate | Trang giá |
| Rerank tự host | GPU | Tăng theo số candidate và độ dài tài liệu | Tự đo |
| Agentic RAG | Khoảng 4× token so với chat (một agent), khoảng 15× (nhiều agent) [9] | Nhiều vòng, cộng dồn | Anthropic |
| Contextual retrieval | Một lần lúc index, $1,02 / triệu token tài liệu theo giá 2024 [21] | Không thêm lúc query | Anthropic |
| GraphRAG | Index rất đắt, ~75% ở bước trích graph [26]; global search hàng trăm nghìn token mỗi câu hỏi [29] | Global search chậm | Microsoft, GraphRAG-Bench |
| Generation | Token context + output; phần lớn tổng chi phí | Phần lớn tổng độ trễ | [Multi-turn §1.4](../multi-turn-conversation/#14-kinh-tế-học-của-prompt-caching) |

**Dịch vụ RAG managed (tháng 11/2026)**

| Dịch vụ | Lưu trữ | Truy vấn | Index | Giới hạn đáng chú ý |
|---|---|---|---|---|
| OpenAI vector store + `file_search` | 1 GB đầu miễn phí, sau đó $0,10 / GB / ngày, tính trên chunk đã parse và embedding [37] | $2,50 / 1.000 lần gọi tool (Responses API) [37], cộng token kết quả đưa vào model | Miễn phí | Chunk mặc định 800 token, overlap 400; chỉnh được trong khoảng 100–4.096 [36]. Search trả tối đa 10 kết quả mặc định, tối đa 50 qua `max_num_results` [36]. 300 request/phút cho việc thêm file vào mỗi store [36] |
| Gemini File Search | Miễn phí | Embedding query miễn phí; token tài liệu tìm được tính như context thường [38] | Tính theo giá embedding lúc index [38] | 100 MB mỗi tài liệu; tổng dung lượng store theo tier 1 GB (Free) tới 1 TB (Tier 3), tính trên input cộng embedding, thường khoảng 3 lần input; khuyến nghị mỗi store dưới 20 GB [38]. Không dùng chung với Google Search grounding [38] |
| Bedrock Managed Knowledge Base | $5,00 / GB dữ liệu thô / tháng [39] | $1,00 / 1.000 lần Retrieve; agentic retrieval $4,00 / 1.000 lần cộng các lần Retrieve bên dưới [39] | Parse và rerank managed miễn phí [39] | Connector tới S3, SharePoint, Confluence [39] |

Dịch vụ managed rẻ để bắt đầu và ít việc vận hành. Đổi lại, bạn ít quyền kiểm soát hơn với chunking, rerank, cách xoá, và cách phân quyền. Kiểm tra từng điểm ở [§3.2](#32-cập-nhật-index-khi-dữ-liệu-thay-đổi) và [§3.3](#33-phân-quyền-tài-liệu-trong-retrieval) với dịch vụ bạn chọn trước khi đưa dữ liệu có phân quyền lên.

**Đo theo span.** Mỗi tầng một span trong trace (embed, search, rerank, generate), để biết độ trễ tăng ở đâu khi nó tăng. Mẫu trace cho RAG ở [LLM Observability §3.2](../llm-observability/#32-trace-cho-rag-tool-use-và-agent-nhiều-bước) và [Embeddings §3.7](../embeddings/#37-observability-cho-embedding--rag-systems).

## 3.5 Eval cho RAG nâng cao

Metric cơ bản (Recall@N, nDCG, faithfulness) và cách chấm bằng LLM đã có ở [Evaluation §2.3](../evaluation/#23-eval-cho-rag-retrieval-và-generation) và [Embeddings §3.6](../embeddings/#36-evaluation-methodology). Phần này là những gì riêng cho các kỹ thuật ở Phần 2.

**Tập câu hỏi theo loại.** Một tập trộn lẫn che mất chỗ kỹ thuật thắng hoặc thua. GraphRAG thắng ở câu tổng hợp và thua ở câu tra cứu ([§2.5](#25-graphrag-và-knowledge-graph)); trung bình hai nhóm có thể ra con số không đổi. Chia tập eval ít nhất thành:

| Nhóm | Ví dụ | Đo |
|---|---|---|
| Tra cứu một sự kiện | "Gói Pro hoàn tiền trong bao nhiêu ngày?" | Recall@N, độ đúng |
| Nhiều phần / so sánh | "So sánh bảo hành 2025 và 2026" | Recall theo từng phần, độ đầy đủ |
| Nhiều bước | "Ai duyệt thay đổi gây ra sự cố tuần trước?" | Độ đúng, số vòng, chi phí |
| Tổng hợp | "Ba lý do khách huỷ nhiều nhất?" | Rubric, so với đáp án do người viết |
| Không có đáp án | Câu hỏi về thứ không có trong corpus | Tỉ lệ từ chối đúng |
| Quyền | Đáp án chỉ có trong tài liệu người hỏi không được đọc | Tỉ lệ không rò rỉ (phải là 100%) |

**RAGAS.** Thư viện RAGAS có các metric context precision, context recall, faithfulness, response relevancy và nhiều metric khác [41]. Tên metric đã đổi qua các phiên bản (ví dụ `answer_relevancy` cũ nay là Response Relevancy); kiểm tra tên trong phiên bản bạn cài, và đọc định nghĩa từng metric trên trang riêng của nó trước khi đặt ngưỡng.

**Cẩn thận với LLM chấm so cặp.** Phần lớn các tuyên bố "GraphRAG tốt hơn" đến từ việc cho LLM so hai câu trả lời. Zeng và cộng sự cho thấy cách này thiên lệch đủ để đảo kết quả [30]. Khi so hai pipeline, ưu tiên câu hỏi có đáp án kiểm tra được; với câu tổng hợp, chấm theo rubric từng tiêu chí, đổi vị trí hai câu trả lời và chấm hai lần, và cho người kiểm tra một mẫu. Các bẫy của LLM-as-judge ở [Evaluation §2.2](../evaluation/#22-llm-as-judge).

**Thứ tự đo:**

1. Retrieval: Recall@N trên câu hỏi có nhãn tài liệu đúng. Không cần LLM.
2. Rerank: nDCG@k, Recall@k trên cùng tập.
3. Context: tài liệu đúng có thật sự nằm trong prompt cuối cùng không (bắt FP3).
4. Câu trả lời: độ đúng, faithfulness, trích dẫn qua quote matching ([§2.8](#28-trích-dẫn-nguồn-và-grounding)).
5. Nhóm không có đáp án và nhóm quyền.
6. Chi phí và độ trễ p95 cho từng nhóm.

Bước 1 và 2 rẻ, nhanh, xác định; chạy trong CI cho mọi thay đổi chunking, embedding hay retrieval. Bước 4 đắt hơn, chạy khi đổi prompt hoặc model.

## 3.6 Failure modes thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Trả lời khi không có tài liệu liên quan | Prompt không cho phép từ chối; không có nhóm eval câu không có đáp án | Prompt cho phép "không có trong tài liệu"; đo tỉ lệ từ chối đúng |
| Đáp án có trong corpus nhưng không được lấy về | Chunk cắt ngang; khác thuật ngữ; chỉ có dense | Hybrid + RRF; contextual retrieval; tăng số candidate |
| Lấy về rồi nhưng model không dùng | Đáp án ở giữa context dài; quá nhiều đoạn nhiễu | Rerank, giảm số đoạn, đặt đoạn tốt nhất ở đầu |
| Chính sách cũ cạnh tranh với chính sách mới | Thiếu metadata ngày hiệu lực | Gắn ngày hiệu lực, filter hoặc ưu tiên bản mới |
| HyDE kéo retrieval về sai thực thể | Đoạn văn giả bịa tên, số | Giữ nhánh query gốc, trộn bằng RRF với trọng số thấp cho HyDE |
| Rerank chậm hoặc đắt bất thường | Đưa cả lịch sử hội thoại làm query; quá nhiều candidate | Rerank bằng câu hỏi đã viết lại; giảm candidate |
| Chi phí tăng vọt sau khi bật agentic RAG | Không giới hạn vòng, không có ngân sách | Giới hạn vòng và token; routing để chỉ câu khó vào vòng agentic |
| GraphRAG tốn kém mà chất lượng không tăng | Dùng graph cho câu hỏi tra cứu | Routing theo loại câu hỏi; đo riêng từng nhóm |
| Trích dẫn trỏ vào đoạn không tồn tại | Model tự đánh số nguồn | Citations API, `search_result`, hoặc quote matching |
| Bật citations cùng JSON output bị lỗi 400 | Claude không cho dùng chung | Parse dòng `tên: giá trị` kèm citation, hoặc field `source_quote` trong schema |
| Ít hơn k kết quả sau khi lọc quyền | Post-filter trên top-k | Pre-filter trong truy vấn, lấy dư rồi lọc |
| Người dùng đọc được nội dung tài liệu không có quyền | Filter trong prompt; cache dùng chung; báo cáo GraphRAG trộn quyền | Filter trong truy vấn; khoá cache gồm tập quyền; graph theo mức quyền |
| Tài liệu đã xoá vẫn xuất hiện | Không phát hiện xoá; xoá eventually consistent | So danh sách nguồn với index; lọc danh sách đã xoá ở tầng ứng dụng |
| Chất lượng giảm sau khi đổi prompt contextual hoặc embedding model | Index trộn chunk cũ và mới | Phiên bản pipeline trong hash; index lại toàn bộ; dựng index song song |
| Chunk bảng không có nghĩa | Chia bảng theo token, mất header | Chia theo dòng, lặp lại header và caption |
| SQL do model sinh đọc nhầm bảng hoặc sửa dữ liệu | Quyền database quá rộng | Role chỉ `SELECT` trên view được duyệt; parse và kiểm tra SQL trước khi chạy |

## 3.7 Checklist

**Đo lường**

- [ ] Recall@N của retrieval được đo tách khỏi chất lượng câu trả lời.
- [ ] Tập eval chia theo loại câu hỏi, có nhóm không có đáp án và nhóm quyền.
- [ ] Mỗi kỹ thuật trong pipeline có số liệu eval cho thấy nó giúp trên dữ liệu của bạn.
- [ ] Độ trễ p95 và chi phí đo theo từng tầng, có span trong trace.

**Retrieval**

- [ ] Hybrid search với RRF; tham số k và trọng số được ghi lại.
- [ ] Rerank trên số candidate đã đo, query đưa vào reranker là câu hỏi đã viết lại, không phải cả hội thoại.
- [ ] Chunk có đủ ngữ cảnh (contextual retrieval, parent–child, hoặc chia theo heading); bảng giữ header.
- [ ] Biến đổi query và agentic RAG chỉ bật qua routing, có giới hạn vòng và ngân sách.

**Dữ liệu và quyền**

- [ ] Filter quyền nằm trong truy vấn; tập quyền lấy từ hệ thống định danh lúc request.
- [ ] Contextual retrieval, GraphRAG và cache không trộn nội dung giữa các mức quyền.
- [ ] Đồng bộ phát hiện được sửa và xoá; hash gồm phiên bản pipeline.
- [ ] Có quy trình index lại toàn bộ song song và chuyển bằng alias.

**Câu trả lời**

- [ ] Trích dẫn dùng id do hệ thống cấp, kiểm tra được bằng Citations API hoặc quote matching.
- [ ] Không bật citations cùng structured output trên Claude.
- [ ] Model được phép nói "không có trong tài liệu".
- [ ] Giá, giới hạn và tên model đã kiểm tra lại trên trang chính thức trước khi đưa vào dự toán.

---

## Nguồn tham khảo

| # | Nguồn |
|---|---|
| [1] | Barnett, S. và cộng sự (2024). [Seven Failure Points When Engineering a Retrieval Augmented Generation System](https://arxiv.org/abs/2401.05856) |
| [2] | Liu, N. F. và cộng sự (2023). [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) |
| [3] | Li, Z. và cộng sự (2024). [Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach](https://arxiv.org/abs/2407.16833) |
| [4] | Hsieh, C.-P. và cộng sự (2024). [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654) |
| [5] | Anthropic. [Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows) |
| [6] | Asai, A. và cộng sự (2023). [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) |
| [7] | Yan, S.-Q. và cộng sự (2024). [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) |
| [8] | Jeong, S. và cộng sự (2024). [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403) |
| [9] | Anthropic (2025). [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) |
| [10] | Gao, L. và cộng sự (2022). [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)](https://arxiv.org/abs/2212.10496) |
| [11] | Rackauckas, Z. (2024). [RAG-Fusion: a New Take on Retrieval-Augmented Generation](https://arxiv.org/abs/2402.03367) |
| [12] | Zheng, H. S. và cộng sự (2023). [Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models](https://arxiv.org/abs/2310.06117) |
| [13] | Trivedi, H. và cộng sự (2022). [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://arxiv.org/abs/2212.10509) |
| [14] | Cormack, G. V., Clarke, C. L. A., Büttcher, S. (2009). [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) |
| [15] | Cohere. [Pricing](https://cohere.com/pricing) (định nghĩa search unit của Rerank) |
| [16] | Voyage AI. [Rerankers](https://docs.voyageai.com/docs/reranker) |
| [17] | Voyage AI. [Pricing](https://docs.voyageai.com/docs/pricing) |
| [18] | Jina AI. [Reranker](https://jina.ai/reranker/) |
| [19] | BAAI. [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| [20] | Qwen. [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) |
| [21] | Anthropic (2024). [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) |
| [22] | Günther, M. và cộng sự (2024). [Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models](https://arxiv.org/abs/2409.04701) |
| [23] | LlamaIndex. [`sentence_window.py`](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/text/sentence_window.py) (mặc định `window_size`) |
| [24] | LlamaIndex. [`hierarchical.py`](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/relational/hierarchical.py) và [`auto_merging_retriever.py`](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/auto_merging_retriever.py) |
| [25] | Edge, D. và cộng sự (2024). [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) |
| [26] | Microsoft. [GraphRAG: Indexing methods](https://microsoft.github.io/graphrag/index/methods/) |
| [27] | Microsoft Research (25/11/2024). [LazyGraphRAG: Setting a new standard for quality and cost](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) |
| [28] | Han, H. và cộng sự (2025). [RAG vs. GraphRAG: A Systematic Evaluation and Key Insights](https://arxiv.org/abs/2502.11371) |
| [29] | Xiang, Z. và cộng sự (2025). [When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation (GraphRAG-Bench)](https://arxiv.org/abs/2506.05690) |
| [30] | Zeng, Q. và cộng sự (2025). [How Significant Are the Real Performance Gains? An Unbiased Evaluation Framework for GraphRAG](https://arxiv.org/abs/2506.06331) |
| [31] | Faysse, M. và cộng sự (2024). [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449) |
| [32] | Anthropic. [Citations](https://platform.claude.com/docs/en/build-with-claude/citations) |
| [33] | Anthropic. [Search results](https://platform.claude.com/docs/en/build-with-claude/search-results) |
| [34] | Anthropic. [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) (không tương thích với citations) |
| [35] | OpenAI. [File search](https://developers.openai.com/api/docs/guides/tools-file-search) |
| [36] | OpenAI. [Retrieval](https://developers.openai.com/api/docs/guides/retrieval) (vector store, chunking, filter, ranking options) |
| [37] | OpenAI. [Pricing: built-in tools](https://developers.openai.com/api/docs/pricing#built-in-tools) |
| [38] | Google. [Gemini API: File Search](https://aistudio.google.com/docs/file-search) |
| [39] | AWS. [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) (Knowledge Bases, Rerank) |
| [40] | Microsoft. [Security filters for trimming results in Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-security-trimming-for-azure-search) |
| [41] | Ragas. [Available metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) |
