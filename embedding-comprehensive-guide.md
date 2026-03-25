# Tài liệu Toàn diện về Embedding

> **Phiên bản**: March 2026 | **Ngôn ngữ**: Việt-Anh (thuật ngữ kỹ thuật giữ nguyên tiếng Anh)
> **Nguồn**: 65+ citations từ official docs, papers, benchmarks
> **Cấu trúc**: 3-Layer Architecture — Foundations → Systems → Operations

---

## Mục lục

- [Layer 1 — Foundations (Nền tảng)](#layer-1--foundations-nền-tảng)
  - [1.1 Embedding là gì?](#11-embedding-là-gì)
  - [1.2 Lịch sử phát triển](#12-lịch-sử-phát-triển)
  - [1.3 Các thuật toán tính Similarity](#13-các-thuật-toán-tính-similarity)
- [Layer 2 — Systems & Applications (Hệ thống & Ứng dụng)](#layer-2--systems--applications-hệ-thống--ứng-dụng)
  - [2.1 Semantic Search / Information Retrieval](#21-semantic-search--information-retrieval)
  - [2.2 RAG + Rerank (Pipeline 2 giai đoạn)](#22-rag--rerank-pipeline-2-giai-đoạn)
  - [2.3 Clustering (Phân cụm)](#23-clustering-phân-cụm)
  - [2.4 Recommendation Systems](#24-recommendation-systems)
  - [2.5 Classification & Sentiment Analysis](#25-classification--sentiment-analysis)
  - [2.6 Anomaly Detection](#26-anomaly-detection)
  - [2.7 Deduplication / Near-duplicate Detection](#27-deduplication--near-duplicate-detection)
  - [2.8 Multimodal Embedding](#28-multimodal-embedding)
- [Layer 3 — Operations (Vận hành & Tối ưu)](#layer-3--operations-vận-hành--tối-ưu)
  - [3.1 Embedding Models Comparison](#31-embedding-models-comparison)
  - [3.2 Vector Databases](#32-vector-databases-ma-trận-chọn-nhanh)
  - [3.3 Chunking Strategies](#33-chunking-strategies)
  - [3.4 Dimension Reduction & Quantization](#34-dimension-reduction--quantization)
  - [3.5 Hybrid Search (Sparse + Dense)](#35-hybrid-search-sparse--dense)
  - [3.6 Evaluation Methodology](#36-evaluation-methodology)
- [Tổng hợp Sources](#-tổng-hợp-sources)

---

# Layer 1 — Foundations (Nền tảng)

## 1.1 Embedding là gì?

### Concept

**Embedding** là phép ánh xạ dữ liệu rời rạc (discrete) — chẳng hạn từ ngữ, câu, hình ảnh, âm thanh — sang **vector liên tục** (continuous) trong không gian nhiều chiều (high-dimensional space). Ký hiệu toán học:

$$f: X \rightarrow \mathbb{R}^n$$

Trong đó:
- **X** là tập dữ liệu đầu vào (ví dụ: tập từ vựng, tập hình ảnh)
- **ℝⁿ** là không gian vector n chiều (thường n = 384, 768, 1024, 1536, 3072)
- **f** là hàm embedding (có thể là neural network)

### Intuition

> *"Words that appear in similar contexts have similar meanings."*
> — Distributional Hypothesis ([Harris, 1954](https://www.tandfonline.com/doi/abs/10.1080/00437956.1954.11659520))

Ý tưởng cốt lõi: nếu hai từ thường xuất hiện trong ngữ cảnh giống nhau, chúng có ý nghĩa tương tự → vectors của chúng sẽ **gần nhau** trong không gian embedding. Ví dụ:
- "king" và "queen" gần nhau vì cùng ngữ cảnh hoàng gia
- "cat" và "dog" gần nhau vì cùng ngữ cảnh thú cưng
- Nổi tiếng: `king - man + woman ≈ queen` (word analogy)

```
                    Embedding Space (2D projection)
    ┌──────────────────────────────────────────────┐
    │                                              │
    │           ★ king                             │
    │                  ★ queen                     │
    │       ★ prince                               │
    │              ★ princess                      │
    │                                              │
    │                                              │
    │  ● cat                                       │
    │     ● dog                                    │
    │        ● hamster                             │
    │                                              │
    │                          ▲ car               │
    │                             ▲ truck          │
    │                          ▲ bicycle           │
    │                                              │
    └──────────────────────────────────────────────┘
    Hình 1: Embedding space — các khái niệm liên quan
    nằm gần nhau (★ hoàng gia, ● thú cưng, ▲ phương tiện)
```

### Code Example: Generate Embeddings

```python
"""
Generate embeddings với OpenAI, Cohere, và Google SDKs.
Task mẫu: embed cùng một câu để so sánh.
"""

sample_texts = [
    "Embedding ánh xạ dữ liệu sang vector trong không gian nhiều chiều.",
    "Machine learning là nhánh của trí tuệ nhân tạo.",
    "Mèo và chó đều là thú cưng phổ biến.",
]

# ── OpenAI ──────────────────────────────────────────────────
from openai import OpenAI

client = OpenAI()  # OPENAI_API_KEY từ env

response = client.embeddings.create(
    model="text-embedding-3-large",  # 3072 dims, MRL-supported
    input=sample_texts,
    dimensions=1536,                 # MRL: giảm chiều native (không cần PCA)
)

openai_embeddings = [item.embedding for item in response.data]
print(f"OpenAI dims: {len(openai_embeddings[0])}")  # 1536


# ── Cohere ──────────────────────────────────────────────────
import cohere

co = cohere.ClientV2()  # COHERE_API_KEY từ env

response = co.embed(
    texts=sample_texts,
    model="embed-v4.0",              # 256-1536 dims, 128k context, multimodal
    input_type="search_document",    # "search_query" cho query side
    embedding_types=["float"],
)

cohere_embeddings = response.embeddings.float_
print(f"Cohere dims: {len(cohere_embeddings[0])}")


# ── Google (Gemini Embedding 2) ─────────────────────────────
from google import genai

google_client = genai.Client()  # GOOGLE_API_KEY từ env

result = google_client.models.embed_content(
    model="gemini-embedding-2-preview",  # Gemini Embedding 2 (preview, March 2026)
    contents=sample_texts,
    config={
        "output_dimensionality": 768,    # MRL: 3072 → 768
    },
)

google_embeddings = result.embeddings
print(f"Google dims: {len(google_embeddings[0].values)}")  # 768
```

> **Lưu ý**: Gemini Embedding 2 đang ở giai đoạn **preview** (March 2026) — specs và pricing có thể thay đổi.
> Sources: [OpenAI Embedding docs](https://openai.com/index/new-embedding-models-and-api-updates/), [Cohere Embed v4](https://docs.cohere.com/docs/cohere-embed), [Gemini Embedding 2 Blog](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/)

---

## 1.2 Lịch sử phát triển

| Năm | Model | Đặc điểm | Paper |
|-----|-------|----------|-------|
| 2013 | **Word2Vec** | CBOW & Skip-gram, static embeddings — mỗi từ có đúng 1 vector bất kể ngữ cảnh | [Mikolov et al., 2013](https://arxiv.org/abs/1301.3781) |
| 2014 | **GloVe** | Ma trận co-occurrence + global statistics, kết hợp count-based và prediction-based | [Pennington et al., 2014](https://aclanthology.org/D14-1162.pdf) |
| 2016 | **FastText** | Subword n-grams → xử lý OOV (out-of-vocabulary) words tốt hơn | [Bojanowski et al., 2016](https://arxiv.org/abs/1607.04606) |
| 2017 | **Transformer** | Self-attention mechanism, "Attention Is All You Need" — nền tảng cho tất cả modern models | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) |
| 2019 | **Sentence-BERT** | Siamese/triplet BERT networks cho sentence-level embeddings — nhanh hơn cross-encoder 1000x cho search | [Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084) |
| 2022 | **Matryoshka Representation Learning** | Cho phép truncate embedding ở bất kỳ dimension nào mà vẫn giữ chất lượng | [Kusupati et al., 2022](https://arxiv.org/abs/2205.13147) |
| 2024-2026 | **Modern Embedding Models** | text-embedding-3, embed-v4, Gemini Embedding 2 — multimodal, MRL, 128k+ context | Xem [Section 3.1](#31-embedding-models-comparison) |

### Bước nhảy lớn nhất: Static → Contextual

```
    Timeline: Embedding Evolution
    ═══════════════════════════════════════════════════════════════

    2013        2014        2016        2017        2019        2022        2024-2026
     │           │           │           │           │           │           │
     ▼           ▼           ▼           ▼           ▼           ▼           ▼
    Word2Vec    GloVe      FastText  Transformer  SBERT       MRL      Gemini Emb2
    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
    ├── Static Embeddings ──┤     ├── Contextual Embeddings ────────────┤
                                  │
                             ★ BIGGEST LEAP ★
                             Từ "bank" có 1 vector        Từ "bank" có vector
                             dù là ngân hàng hay          khác nhau tùy ngữ cảnh:
                             bờ sông                      "river bank" ≠ "bank account"
```

**Static embeddings** (Word2Vec, GloVe, FastText): mỗi từ → 1 vector cố định, không phân biệt ngữ cảnh. Từ "bank" (ngân hàng) và "bank" (bờ sông) cùng chung 1 vector.

**Contextual embeddings** (Transformer-based): cùng 1 từ nhưng vector thay đổi theo ngữ cảnh. "The bank of the river" → vector khác hoàn toàn so với "I went to the bank to deposit money."

---

## 1.3 Các thuật toán tính Similarity

| Metric | Công thức | Ưu điểm | Nhược điểm | Khi nào dùng |
|--------|----------|---------|------------|-------------|
| **Cosine Similarity** | `cos(θ) = (A·B) / (‖A‖ × ‖B‖)` | Không phụ thuộc magnitude; range [-1, 1] | Cần normalize nếu magnitude quan trọng | **Mặc định** cho semantic similarity |
| **Dot Product** | `A·B = Σ(aᵢ × bᵢ)` | Nhanh nhất; giữ tín hiệu popularity/norm | Thiên về vectors có norm lớn | Recommendation (khi popularity matters) |
| **Euclidean (L2)** | `√Σ(aᵢ - bᵢ)²` | Trực giác hình học, khoảng cách thực | Nhạy cảm với scale | Clustering (K-means dùng L2) |
| **Manhattan (L1)** | `Σ|aᵢ - bᵢ|` | Robust với outliers hơn L2 | Ít phổ biến trong NLP | High-dimensional sparse data |
| **Jaccard** | `|A∩B| / |A∪B|` | Tốt cho set/boolean data | Không dùng cho continuous vectors | Token-level overlap, set similarity |

### Key Insights

> **Với L2-normalized vectors**: cosine similarity ≡ dot product ≡ linear kernel.
> Khi `‖A‖ = ‖B‖ = 1`, thì `cos(θ) = A·B` và `‖A-B‖² = 2(1 - A·B)`.
> → Chọn metric nào cũng cho cùng ranking.
>
> Sources: [FAISS docs — MetricType and distances](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances), [scikit-learn cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

> **Dot product bias**: Dot product thiên về item phổ biến (norm lớn) vì `A·B = ‖A‖‖B‖cos(θ)`. Item popular → trained nhiều → norm lớn → dot product cao dù θ không nhỏ. Nếu mục tiêu là **pure semantic similarity** → normalize + cosine. Nếu muốn **kết hợp relevance + popularity** → dot product.
>
> Source: [Google ML Recommendation — Candidate Generation](https://developers.google.com/machine-learning/recommendation/overview/candidate-generation)

### Code Example: Tính Similarity với NumPy/SciPy

```python
"""
Tính 5 loại similarity/distance giữa 2 vectors.
"""
import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock

# Giả sử 2 embedding vectors (đã generate từ model)
a = np.array([0.12, -0.34, 0.56, 0.78, -0.91])
b = np.array([0.11, -0.30, 0.52, 0.80, -0.88])

# ── Cosine Similarity ──
cosine_sim = 1 - cosine(a, b)  # scipy trả distance, nên 1 - distance = similarity
print(f"Cosine Similarity:  {cosine_sim:.6f}")  # ~0.999+

# ── Dot Product ──
dot_prod = np.dot(a, b)
print(f"Dot Product:        {dot_prod:.6f}")

# ── Euclidean Distance (L2) ──
l2_dist = euclidean(a, b)
print(f"Euclidean (L2):     {l2_dist:.6f}")

# ── Manhattan Distance (L1) ──
l1_dist = cityblock(a, b)
print(f"Manhattan (L1):     {l1_dist:.6f}")

# ── Jaccard (cho binary/set data) ──
set_a = {0, 1, 2, 3, 5}
set_b = {0, 2, 3, 4, 5}
jaccard_sim = len(set_a & set_b) / len(set_a | set_b)
print(f"Jaccard Similarity: {jaccard_sim:.4f}")  # 4/6 = 0.6667


# ── Demo: Normalized vectors → cosine ≡ dot product ──
a_norm = a / np.linalg.norm(a)
b_norm = b / np.linalg.norm(b)

print(f"\nAfter L2-normalization:")
print(f"  Cosine sim: {1 - cosine(a_norm, b_norm):.8f}")
print(f"  Dot product: {np.dot(a_norm, b_norm):.8f}")
# → Hai giá trị bằng nhau!
```

---

# Layer 2 — Systems & Applications (Hệ thống & Ứng dụng)

## 2.1 Semantic Search / Information Retrieval

**Bi-encoder architecture**: encode query và document **độc lập** → lưu document embeddings vào vector database → khi có query, encode query → ANN (Approximate Nearest Neighbor) lookup → trả về top-K kết quả gần nhất.

```
    Semantic Search Pipeline (Bi-encoder)
    ═══════════════════════════════════════

    Indexing Phase (offline):
    ┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
    │ Documents    │────▶│  Bi-encoder  │────▶│  Vector Database │
    │ (corpus)     │     │  (encode)    │     │  (HNSW index)    │
    └─────────────┘     └──────────────┘     └──────────────────┘

    Query Phase (online):
    ┌─────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────┐
    │  Query  │────▶│  Bi-encoder  │────▶│  ANN Search      │────▶│  Top-K   │
    │         │     │  (encode)    │     │  (cosine sim)    │     │ results  │
    └─────────┘     └──────────────┘     └──────────────────┘     └──────────┘
```

### Asymmetric Search

Quan trọng: query và document có **vai trò khác nhau** (asymmetric). Query thường ngắn ("best practices for embedding"), document thường dài (đoạn văn, trang web). Nhiều model yêu cầu chỉ định `input_type`:
- `search_query` — cho câu hỏi/query
- `search_document` — cho document/passage cần index

Dùng sai input_type sẽ giảm chất lượng retrieval đáng kể.

> Source: [SBERT — Semantic Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)

---

## 2.2 RAG + Rerank (Pipeline 2 giai đoạn)

**Retrieval-Augmented Generation (RAG)** kết hợp retrieval với LLM generation để trả lời câu hỏi dựa trên knowledge base.

```
    RAG + Rerank Pipeline (3 Stages)
    ════════════════════════════════════════════════════════════════════

                         Stage 1: HYBRID RETRIEVAL
    ┌───────┐     ┌─────────────────────────────────┐     ┌──────────┐
    │       │     │  ┌─────────┐    ┌────────────┐  │     │          │
    │ Query │────▶│  │  BM25   │───▶│            │  │     │  Top-N   │
    │       │     │  │ (sparse)│    │    RRF     │  │────▶│candidates│
    │       │────▶│  ├─────────┤    │  (merge)   │  │     │ (N~100)  │
    │       │     │  │  Dense  │───▶│            │  │     │          │
    │       │     │  │embedding│    └────────────┘  │     │          │
    │       │     │  └─────────┘                    │     │          │
    └───────┘     └─────────────────────────────────┘     └────┬─────┘
                   70-80% dense + 20-30% sparse                │
                                                               ▼
                         Stage 2: RERANKING
                  ┌──────────────────────────────┐     ┌──────────┐
                  │   Cross-encoder Reranker      │     │  Top-K   │
                  │   (query, doc) → relevance    │────▶│ refined  │
                  │   score (pair-wise)           │     │ (K~5-10) │
                  │                               │     │          │
                  │ Models: Cohere Rerank v3.5,   │     │ Precision│
                  │ BGE Reranker, ColBERT         │     │ +10-30%  │
                  └──────────────────────────────┘     └────┬─────┘
                                                            │
                         Stage 3: GENERATION                ▼
                  ┌──────────────────────────────┐     ┌──────────┐
                  │   LLM (GPT-4, Claude, etc.)  │     │  Answer  │
                  │   + top-K context             │────▶│  with    │
                  │   + system prompt             │     │ citations│
                  └──────────────────────────────┘     └──────────┘
```

### Stage 1: Hybrid Retrieval

Kết hợp **BM25** (keyword/sparse) và **dense embedding** để tận dụng cả hai:
- **BM25**: mạnh với exact keyword match, tên riêng, mã số
- **Dense**: mạnh với semantic similarity, paraphrases, multilingual

**Reciprocal Rank Fusion (RRF)** để merge 2 ranked lists:

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
               i
```

Trong đó `k` thường = 60 (constant), `rank_i(d)` là thứ hạng của document `d` trong ranked list thứ `i`.

Tỷ lệ khuyến nghị: **70-80% dense + 20-30% sparse** (heuristic phổ biến trong community, có thể tune theo domain — không có chuẩn cố định)

### Stage 2: Reranking

**Cross-encoder** khác với bi-encoder:
- **Bi-encoder**: encode query và document **riêng rẽ** → so sánh vectors → nhanh nhưng kém chính xác
- **Cross-encoder**: nhận **cặp** (query, document) → xử lý attention giữa cả hai → chính xác hơn nhưng chậm

→ Dùng bi-encoder cho Stage 1 (xử lý triệu documents), cross-encoder cho Stage 2 (chỉ re-score ~100 candidates).

**Reranker models nổi bật**:
- Cohere Rerank v3.5 — [docs](https://docs.cohere.com/docs/rerank-overview)
- BGE Reranker — open-source
- ColBERT — late interaction, cân bằng giữa bi-encoder và cross-encoder

Cải thiện precision: **10-30%** so với chỉ dùng retrieval (tùy dataset và domain; estimate từ community experience, xem [Cohere Rerank](https://docs.cohere.com/docs/rerank-overview))

### Stage 3: LLM Generation

LLM nhận top-K reranked documents làm context → generate câu trả lời có citations.

### Evaluation Metrics

- **Trước rerank**: Recall@N (coverage — lấy được bao nhiêu relevant docs trong N candidates)
- **Sau rerank**: nDCG@K (quality of ranking — docs relevant ở vị trí cao hơn?)

> Sources: [RAG paper — Lewis et al., 2020](https://arxiv.org/abs/2005.11401), [Pinecone Rerankers Guide](https://docs.pinecone.io/guides/search/rerank-results), [Cohere Rerank Overview](https://docs.cohere.com/docs/rerank-overview)

### Code Example: Retrieve → Rerank → Generate Pipeline

```python
"""
RAG + Rerank pipeline minh họa.
Yêu cầu: pip install openai cohere pinecone
"""
from openai import OpenAI
import cohere

openai_client = OpenAI()
cohere_client = cohere.ClientV2()

# ── Stage 1: Retrieve (giả sử đã có vector DB với documents) ──
query = "Matryoshka Representation Learning là gì?"

# Embed query
query_embedding = openai_client.embeddings.create(
    model="text-embedding-3-large",
    input=query,
    dimensions=1536,
).data[0].embedding

# Giả sử search trả về top-N candidates
# (Trong thực tế: gọi vector DB + BM25 → RRF merge)
candidates = [
    "MRL cho phép truncate embedding vectors ở bất kỳ dimension nào.",
    "Transformer sử dụng self-attention mechanism.",
    "Matryoshka learning tạo embeddings giống búp bê Nga - lớp ngoài chứa lớp trong.",
    "Word2Vec là mô hình embedding tĩnh ra đời năm 2013.",
    "MRL paper của Kusupati et al. 2022 đề xuất training loss ở nhiều granularities.",
]

# ── Stage 2: Rerank với Cohere ──
rerank_response = cohere_client.rerank(
    model="rerank-v3.5",
    query=query,
    documents=candidates,
    top_n=3,  # chỉ lấy top 3
)

reranked_docs = [candidates[r.index] for r in rerank_response.results]
print("Reranked top-3:")
for i, doc in enumerate(reranked_docs):
    score = rerank_response.results[i].relevance_score
    print(f"  {i+1}. [{score:.4f}] {doc}")

# ── Stage 3: Generate với LLM ──
context = "\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(reranked_docs))

response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Trả lời dựa trên context. Trích dẫn [số] cho mỗi claim."},
        {"role": "user", "content": f"Context:\n{context}\n\nCâu hỏi: {query}"},
    ],
)

print(f"\nAnswer: {response.choices[0].message.content}")
```

---

## 2.3 Clustering (Phân cụm)

Embedding vectors có thể được dùng để phân cụm dữ liệu — gom các items có semantic meaning gần nhau vào cùng nhóm.

### K-means

- **Mục tiêu**: tối ưu inertia (tổng bình phương khoảng cách từ mỗi điểm đến centroid gần nhất)
- **Distance metric**: L2 (Euclidean)
- **Ưu điểm**: đơn giản, nhanh, scale tốt
- **Nhược điểm**: phải chỉ định trước K, giả sử clusters có dạng spherical
- Source: [scikit-learn K-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

### HDBSCAN

- **Mục tiêu**: phát hiện clusters có varying density
- **Ưu điểm**: tự xác định số cụm, phát hiện outlier (noise points), xử lý clusters không đều
- **Nhược điểm**: chậm hơn K-means ở scale lớn, cần tune `min_cluster_size`
- Source: [scikit-learn HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)

### Code Example: Clustering + UMAP Visualization

```python
"""
Clustering embeddings + UMAP visualization.
Yêu cầu: pip install scikit-learn umap-learn matplotlib openai
"""
import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from umap import UMAP
import matplotlib.pyplot as plt

# Giả sử embeddings đã generate từ model (N texts × D dims)
# Trong thực tế: embeddings = [openai_client.embeddings.create(...).data[0].embedding for text in texts]
embeddings = np.random.randn(200, 1536)  # placeholder

# ── K-means Clustering ──
kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
kmeans_labels = kmeans.fit_predict(embeddings)

# ── HDBSCAN Clustering ──
hdbscan = HDBSCAN(min_cluster_size=10, min_samples=5)
hdbscan_labels = hdbscan.fit_predict(embeddings)
n_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
n_noise = list(hdbscan_labels).count(-1)
print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

# ── UMAP Dimension Reduction cho Visualization ──
reducer = UMAP(n_components=2, random_state=42, metric="cosine")
coords_2d = reducer.fit_transform(embeddings)

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(coords_2d[:, 0], coords_2d[:, 1], c=kmeans_labels, cmap="tab10", s=10)
axes[0].set_title("K-means (K=5)")

axes[1].scatter(coords_2d[:, 0], coords_2d[:, 1], c=hdbscan_labels, cmap="tab10", s=10)
axes[1].set_title(f"HDBSCAN ({n_clusters} clusters)")

plt.tight_layout()
plt.savefig("embedding_clusters.png", dpi=150)
plt.show()
```

---

## 2.4 Recommendation Systems

Embeddings cho **user** và **item** vào cùng một không gian vector → dùng **dot product** để ranking:

```
score(user, item) = user_embedding · item_embedding
```

### Tại sao dùng Dot Product thay vì Cosine?

Trong recommendation, **popularity là tín hiệu quan trọng**:
- Item phổ biến → trained nhiều → embedding norm lớn
- Dot product giữ tín hiệu norm → item phổ biến có score cao hơn tự nhiên
- Cosine normalize mất tín hiệu này

→ Dot product = **relevance × popularity**. Phù hợp khi muốn recommend items vừa liên quan vừa phổ biến.

Nếu muốn **pure semantic relevance** (không ưu tiên popularity) → normalize + cosine.

> Source: [Google ML — Recommendation: Candidate Generation](https://developers.google.com/machine-learning/recommendation/overview/candidate-generation)

---

## 2.5 Classification & Sentiment Analysis

Embeddings biến text thành **fixed-size feature vectors** → input cho classifier:

```
    Text ──▶ Embedding Model ──▶ Vector (ℝⁿ) ──▶ Classifier ──▶ Label
                                                   │
                                                   ├── Logistic Regression
                                                   ├── SVM
                                                   ├── Random Forest
                                                   └── Neural Network (MLP)
```

**Ưu điểm so với fine-tuning full model**:
- Nhanh: chỉ train classifier nhỏ (seconds thay vì hours)
- Ít data: embedding đã capture semantic → ít training samples hơn
- Flexible: thay embedding model dễ dàng

**Ví dụ use cases**: sentiment analysis, spam detection, topic classification, intent recognition.

> Source: [OpenAI Cookbook — Classification using Embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Classification_using_embeddings.ipynb)

---

## 2.6 Anomaly Detection

Embedding features kết hợp với **Isolation Forest** để phát hiện anomaly:

- **Isolation Forest**: xây random trees, cô lập (isolate) data points. Anomaly cần ít splits hơn để bị cô lập → **path length ngắn = anomaly**.
- **Pipeline**: raw data → embedding → Isolation Forest → anomaly score

```python
from sklearn.ensemble import IsolationForest

# embeddings: numpy array shape (n_samples, n_dims)
iso_forest = IsolationForest(contamination=0.05, random_state=42)  # 5% expected anomalies
anomaly_labels = iso_forest.fit_predict(embeddings)
# -1 = anomaly, 1 = normal

anomalies = [text for text, label in zip(texts, anomaly_labels) if label == -1]
print(f"Detected {len(anomalies)} anomalies out of {len(texts)} texts")
```

> Sources: [scikit-learn Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html), [OpenAI — Text and Code Embeddings use cases](https://openai.com/index/introducing-text-and-code-embeddings/)

---

## 2.7 Deduplication / Near-duplicate Detection

Dùng embedding similarity để phát hiện duplicates hoặc near-duplicates (paraphrases):

1. **Simple approach**: tính cosine similarity giữa mọi cặp → threshold (ví dụ > 0.95 = duplicate)
   - Complexity: O(n²) → chỉ phù hợp với dataset nhỏ (<10k)

2. **Scalable approach**: paraphrase mining với chunk-based processing
   - Chia corpus thành chunks → tìm top-k similar pairs trong mỗi chunk → merge
   - Giảm complexity đáng kể

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = ["...", "..."]  # list of texts to deduplicate

# Paraphrase mining — tìm cặp tương tự nhất
# Trả về: [(score, idx_i, idx_j), ...]
duplicates = paraphrase_mining(model, texts, show_progress_bar=True)

# Lọc với threshold
threshold = 0.90
near_dupes = [(score, i, j) for score, i, j in duplicates if score > threshold]

for score, i, j in near_dupes[:10]:
    print(f"Score: {score:.4f}")
    print(f"  Text 1: {texts[i][:100]}...")
    print(f"  Text 2: {texts[j][:100]}...")
```

> Source: [SBERT — Paraphrase Mining](https://www.sbert.net/examples/sentence_transformer/applications/paraphrase-mining/README.html)

---

## 2.8 Multimodal Embedding

### 2.8.1 CLIP (OpenAI)

**CLIP** (Contrastive Language-Image Pre-training) — model tiên phong cho multimodal embedding:

- **Training**: Contrastive learning trên **400M cặp image-text** từ internet
- **Ý tưởng**: encode image và text vào **cùng** embedding space → so sánh trực tiếp
- **Zero-shot transfer**: classify image bằng cách so text descriptions, không cần fine-tune
- **Use cases**: cross-modal retrieval (search ảnh bằng text, search text bằng ảnh), image classification

```
    CLIP Architecture
    ═════════════════

    ┌──────────┐          ┌──────────┐
    │  Image   │          │  Text    │
    │  "a cat  │          │ "a photo │
    │  sitting │          │  of a    │
    │  on mat" │          │  cat"    │
    └────┬─────┘          └────┬─────┘
         │                     │
         ▼                     ▼
    ┌──────────┐          ┌──────────┐
    │  Image   │          │  Text    │
    │  Encoder │          │  Encoder │
    │  (ViT)   │          │(Transf.) │
    └────┬─────┘          └────┬─────┘
         │                     │
         ▼                     ▼
    ┌──────────┐          ┌──────────┐
    │  Image   │◄────────▶│  Text    │
    │ Embedding│  cosine  │ Embedding│
    │  vector  │ similarity│  vector  │
    └──────────┘          └──────────┘

    Training: maximize similarity cho matching pairs,
              minimize cho non-matching pairs (contrastive)
```

> Source: [CLIP paper — Radford et al., 2021](https://arxiv.org/abs/2103.00020)

### 2.8.2 Vertex AI multimodalembedding@001 (Legacy)

- **Modalities**: Text + Image + Video → **1408-dim** vector space
- **Dimension reduction**: hỗ trợ giảm chiều 128 / 256 / 512
- **Giới hạn**: text input ngắn trong mode text+image (phù hợp captions, không phải long docs)
- **Use cases**: e-commerce image search, video retrieval, visual Q&A

> Source: [Vertex AI — Multimodal Embeddings](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings)

### 2.8.3 Gemini Embedding 2 (March 2026) ⭐

> ⚠️ **Preview Notice**: Model đang ở giai đoạn preview. Specs, pricing, và model name có thể thay đổi.

**Gemini Embedding 2** là **first natively multimodal embedding model** built trên kiến trúc Gemini — không phải ghép nối 2 encoder riêng rẽ mà embed tất cả modalities trong cùng architecture.

**5 Modalities được hỗ trợ**:

| Modality | Limit | Ghi chú |
|----------|-------|---------|
| Text | 8192 tokens | Long-context cho documents |
| Image | Max 6 images/request | PNG, JPEG; hỗ trợ interleaved text+image |
| Video | Max 128 giây | MP4, MOV; codec H264/H265/AV1/VP9 |
| Audio | Max 80 giây | MP3, WAV |
| PDF | Max 6 trang | Native PDF understanding |

**Specs**:
- Dimensions: default **3072**, recommended **1536 / 768** (MRL-based → truncate native)
- **Interleaved input**: gửi text + images trong cùng 1 request → single embedding

**Pricing** (snapshot March 2026, preview):
- Text: **$0.20 / 1M tokens**
- Image: **$0.00012 / image**

```
    Gemini Embedding 2 — Cross-modal Retrieval
    ════════════════════════════════════════════════════════════════

    Index Phase:
    ┌──────────┐     ┌───────────────────┐     ┌──────────────────┐
    │ Mixed    │     │  Gemini Embedding │     │                  │
    │ Content: │────▶│  2 (5 modalities) │────▶│  Vector Database │
    │ • docs   │     │                   │     │  (unified space) │
    │ • images │     │  → ℝ¹⁵³⁶ vector   │     │                  │
    │ • videos │     │  (MRL truncated)  │     │                  │
    │ • audio  │     │                   │     │                  │
    │ • PDFs   │     └───────────────────┘     └──────────────────┘
    └──────────┘

    Query Phase:
    ┌──────────┐     ┌───────────────────┐     ┌──────────────────┐
    │ Query:   │     │  Gemini Embedding │     │   Results:       │
    │ "show me │────▶│  2                │────▶│   🖼️ image_3.jpg │
    │ sunset   │     │  → ℝ¹⁵³⁶ query    │     │   📄 doc_17.pdf  │
    │ photos"  │     │                   │     │   🎥 video_5.mp4 │
    └──────────┘     └───────────────────┘     └──────────────────┘

    Cross-modal: text query → tìm images, videos, docs cùng lúc!
```

```python
"""
Gemini Embedding 2 — Multimodal embedding example.
⚠️ Preview API — model name và behavior có thể thay đổi.
"""
from google import genai
from google.genai import types

client = genai.Client()

# ── Text-only embedding ──
text_result = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents="Sunset over the ocean with orange and purple sky",
    config={"output_dimensionality": 768},
)
print(f"Text embedding dims: {len(text_result.embeddings[0].values)}")

# ── Interleaved text + image embedding ──
image_part = types.Part.from_uri(
    file_uri="gs://my-bucket/sunset.jpg",  # hoặc base64, hoặc upload via Files API
    mime_type="image/jpeg",
)

multimodal_result = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents=[
        types.Content(parts=[
            types.Part(text="A beautiful sunset photograph"),
            image_part,
        ])
    ],
    config={"output_dimensionality": 768},
)
print(f"Multimodal embedding dims: {len(multimodal_result.embeddings[0].values)}")
```

> Sources: [Gemini Embedding 2 Blog](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/), [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)

---

# Layer 3 — Operations (Vận hành & Tối ưu)

## 3.1 Embedding Models Comparison

### Bảng 1 — Specs

| Model | Provider | Dims | Context | Multimodal | MRL | Languages |
|-------|----------|------|---------|------------|-----|-----------|
| **text-embedding-3-large** | OpenAI | 3072 (shortable) | 8191 tokens | ❌ | ✅ | Multi |
| **text-embedding-3-small** | OpenAI | 1536 (shortable) | 8191 tokens | ❌ | ✅ | Multi |
| **embed-v4.0** | Cohere | 256-1536 | 128k tokens | ✅ (text+image+PDF) | ✅ | 100+ |
| **embed-v3.0** | Cohere | 384/1024 | 512 tokens | ❌ | ❌ | Multi |
| **gemini-embedding-001** | Google | max 3072 | 2048 tokens | ❌ | ✅ | Multi |
| **Gemini Embedding 2** ⭐ | Google | 3072 (rec. 1536/768) | 8192 tokens | ✅ (5 modalities) | ✅ | Multi |
| **multimodalembedding@001** | Google (Legacy) | 1408 | Short | ✅ (text+img+video) | ❌ | Multi |
| **jina-embeddings-v3** | Jina AI | 1024 (MRL→32) | 8192 tokens | ❌ | ✅ | Multi |
| **all-MiniLM-L6-v2** | Sentence-Transformers | 384 | 256 tokens | ❌ | ❌ | English |
| **all-mpnet-base-v2** | Sentence-Transformers | 768 | 384 tokens | ❌ | ❌ | English |
| **voyage-3** | Voyage AI | 1024 | 32k tokens | ❌ | ❌ | Multi |
| **voyage-3-lite** | Voyage AI | 512 | 32k tokens | ❌ | ❌ | Multi |

> Sources: [OpenAI](https://openai.com/index/new-embedding-models-and-api-updates/), [Cohere embed-v4](https://docs.cohere.com/changelog/embed-multimodal-v4), [Cohere embed-v3](https://docs.cohere.com/docs/cohere-embed), [Gemini Embedding 2](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/), [gemini-embedding-001 docs](https://ai.google.dev/gemini-api/docs/embeddings), [Jina v3 paper](https://arxiv.org/pdf/2409.10173), [SBERT models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html), [Voyage AI models](https://docs.voyageai.com/docs/embeddings)

### Bảng 2 — Quality (MTEB Benchmarks)

| Model | MTEB Avg | Source Type | Date | Source Link |
|-------|----------|------------|------|-------------|
| text-embedding-3-large | 64.6 | vendor-reported | Jan 2024 | [OpenAI blog](https://openai.com/index/new-embedding-models-and-api-updates/) |
| voyage-large-2-instruct | 68.28 | vendor-reported | May 2024 | [Voyage AI blog](https://blog.voyageai.com/2024/05/05/voyage-large-2-instruct/) |
| jina-embeddings-v3 | 65.52 (English) | paper-reported | Sep 2024 | [Jina v3 paper](https://arxiv.org/pdf/2409.10173) |
| Cohere embed-v3 | N/A (xem note) | vendor-reported | 2023 | [Cohere blog](https://cohere.com/blog/introducing-embed-v3) |

> **Note (Cohere embed-v3)**: Cohere không công bố single MTEB average number; họ report individual tasks trên MTEB và BEIR. Xem link nguồn để xem breakdown chi tiết.

> ⚠️ **Benchmark Comparability Warning**:
>
> Vendor-reported benchmarks **không** "apple-to-apple":
> - Khác dataset subset, evaluation protocol, thời điểm chạy
> - Self-reported → có thể cherry-pick kết quả tốt nhất
> - MTEB leaderboard thay đổi thường xuyên
>
> **Best practice**: Luôn ghi rõ **source type** (vendor-reported / independent / paper-reported) + **snapshot date**. Cần **benchmark riêng** cho ngôn ngữ mục tiêu (ví dụ: tiếng Việt không có trên MTEB tiêu chuẩn).
>
> Sources: [MTEB paper — Muennighoff et al., 2022](https://arxiv.org/abs/2210.07316), [Jina v3 paper](https://arxiv.org/pdf/2409.10173), [HuggingFace MTEB Blog](https://huggingface.co/blog/mteb)

### Bảng 3 — Pricing (Snapshot: March 2026)

| Model | Native Unit | Native Price | ~USD/1M tokens | Ghi chú |
|-------|-------------|-------------|----------------|---------|
| text-embedding-3-large | tokens | $0.13/1M tokens | **$0.13** | |
| text-embedding-3-small | tokens | $0.02/1M tokens | **$0.02** | Rẻ nhất |
| Gemini Embedding 2 (text) | tokens | $0.20/1M tokens | **$0.20** | Preview pricing |
| Gemini Embedding 2 (image) | per image | $0.00012/image | N/A | Tính theo ảnh |
| embed-v4.0 | tokens | TBD | TBD | Giá chưa công bố chính thức |
| embed-v3.0 | tokens | Xem [Cohere pricing](https://cohere.com/pricing) | N/A | Không dùng estimate |
| voyage-3 | tokens | Xem [Voyage pricing](https://www.voyageai.com/pricing) | N/A | Không dùng estimate |

> **Conversion note**: Ước tính 1 character ≈ 0.25 token (trung bình cho tiếng Anh). Tiếng Việt có thể cao hơn do tokenization. Luôn test thực tế với `tiktoken` hoặc API response `usage.total_tokens`.
>
> Sources: [OpenAI Pricing](https://openai.com/index/new-embedding-models-and-api-updates/), [Gemini Pricing](https://ai.google.dev/gemini-api/docs/pricing), [Vertex Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)

---

## 3.2 Vector Databases (Ma trận chọn nhanh)

| Category | Database | Best for | Key Feature | Source |
|----------|----------|----------|-------------|--------|
| **Managed** | **Pinecone** | Dễ dùng, production-ready | Serverless, hybrid search, integrated rerank | [docs](https://docs.pinecone.io/guides/get-started/overview) |
| **Open-source (cloud-native)** | **Weaviate** | Hybrid search + GraphQL API | Semantic + keyword search, generative modules | [docs](https://docs.weaviate.io/weaviate/introduction) |
| | **Qdrant** | Performance + advanced filtering | Rust-based, payload filtering, quantization | [docs](https://qdrant.tech/documentation/overview/) |
| | **Milvus** | Massive scale (tỷ vectors) | GPU acceleration, HNSW/IVF/DiskANN | [docs](https://milvus.io/docs/overview.md) |
| **Extension** | **pgvector** | PostgreSQL integration | HNSW/IVFFlat, cosine/L2/inner product | [repo](https://github.com/pgvector/pgvector) |
| **Local/Prototyping** | **Chroma** | Rapid prototyping, local dev | AI-native, simple API, local-first | [docs](https://docs.trychroma.com/docs/overview/introduction) |

### Khi nào chọn cái nào?

```
    Decision Tree: Chọn Vector Database
    ═════════════════════════════════════

    Bạn cần gì?
    │
    ├── "Tôi muốn prototype nhanh, local"
    │   └──▶ Chroma
    │
    ├── "Tôi đã có PostgreSQL, không muốn thêm infra"
    │   └──▶ pgvector
    │
    ├── "Tôi cần managed, không muốn quản lý infra"
    │   └──▶ Pinecone
    │
    ├── "Tôi cần hybrid search (keyword + semantic)"
    │   └──▶ Weaviate hoặc Pinecone
    │
    ├── "Tôi cần high performance + complex filtering"
    │   └──▶ Qdrant
    │
    └── "Tôi có hàng tỷ vectors, cần GPU acceleration"
        └──▶ Milvus
```

---

## 3.3 Chunking Strategies

| Strategy | Mô tả | Khi nào dùng | Ưu/Nhược |
|----------|--------|-------------|----------|
| **Fixed-size** | ~512 tokens + 20-25% overlap | Baseline, simple setup | ✅ Đơn giản; ❌ Cắt giữa câu/ý |
| **Semantic** | Chunk theo ranh giới ngữ nghĩa (topic shift) | Cải thiện retrieval quality | ✅ Recall thường tốt hơn fixed-size (mức cải thiện tùy domain); ❌ Phức tạp hơn |
| **Recursive / Hierarchical** | Theo cấu trúc document (heading → section → paragraph) | Tài liệu có heading/section rõ ràng | ✅ Giữ cấu trúc; ❌ Phụ thuộc format |
| **Sentence-boundary** | Chunk tại câu hoàn chỉnh (không cắt giữa câu) | Tránh mất ngữ nghĩa ở biên | ✅ Tự nhiên; ❌ Chunk size không đều |

### Best Practices

1. **Overlap**: 20-25% giữa các chunks để tránh mất context ở biên
2. **Chunk size**: 256-512 tokens là sweet spot cho hầu hết use cases
3. **Metadata**: giữ metadata (source, page, section heading) cùng mỗi chunk
4. **Test**: benchmark chunk strategy trên **domain-specific** data — không có "one size fits all"

> Sources: [Pinecone — Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/), [Azure — Chunk Documents for Vector Search](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents)

---

## 3.4 Dimension Reduction & Quantization

### Matryoshka-first Optimization ⭐

**Nếu model hỗ trợ MRL** (Matryoshka Representation Learning):

→ **Luôn dùng tham số native giảm chiều của từng provider** thay vì post-processing (PCA/truncation):
- OpenAI: `dimensions` parameter
- Cohere embed-v4: `output_dimension` parameter
- Google Gemini: `output_dimensionality` trong config

| Model | MRL Support | Cách dùng |
|-------|-------------|-----------|
| text-embedding-3-large | ✅ | `dimensions=1536` trong API call |
| text-embedding-3-small | ✅ | `dimensions=512` trong API call |
| embed-v4.0 | ✅ | `output_dimension` parameter ([Cohere API ref](https://docs.cohere.com/reference/embed)) |
| Gemini Embedding 2 | ✅ | `output_dimensionality` trong config ([Google docs](https://ai.google.dev/gemini-api/docs/embeddings)) |
| jina-embeddings-v3 | ✅ | Truncate từ 1024 xuống 32 |

**Tại sao MRL tốt hơn PCA truncation?**
- MRL training: model được train để đảm bảo prefix dimensions đã capture đủ thông tin
- PCA truncation: áp dụng **sau** training → mất thông tin không thể recover
- MRL giữ **chất lượng cao ở 50% dimensions** (ví dụ 3072→1536; xem MTEB scores bảng trên trong [Google docs](https://ai.google.dev/gemini-api/docs/embeddings)), PCA thường mất nhiều hơn

> Source: [Matryoshka Representation Learning — Kusupati et al., 2022](https://arxiv.org/abs/2205.13147)

### PCA/UMAP (Vẫn cần cho một số trường hợp)

| Method | Khi nào dùng | Ghi chú |
|--------|-------------|---------|
| **UMAP** | Visualization (2D/3D), manifold learning | Non-linear → giữ cấu trúc local tốt |
| **PCA** | Exploratory analysis, baseline reduction | Linear → nhanh, deterministic |
| **t-SNE** | Visualization (chỉ 2D/3D) | Chậm hơn UMAP ở scale lớn |

```python
# ── MRL: native dimension reduction (recommended) ──
# Prerequisites: pip install openai; OPENAI_API_KEY set in env
from openai import OpenAI
client = OpenAI()
texts = ["sample text 1", "sample text 2"]  # replace with your texts

response = client.embeddings.create(
    model="text-embedding-3-large",
    input=texts,
    dimensions=768,  # Giảm từ 3072 → 768 native
)

# ── PCA: post-processing reduction (khi model không hỗ trợ MRL) ──
from sklearn.decomposition import PCA
import numpy as np

# embeddings_3072: numpy array shape (n_samples, 3072) — replace with your actual embeddings
embeddings_3072 = np.random.randn(100, 3072)  # placeholder

pca = PCA(n_components=768)
reduced = pca.fit_transform(embeddings_3072)
print(f"Variance retained: {pca.explained_variance_ratio_.sum():.2%}")

# ── UMAP: visualization ──
from umap import UMAP

# embeddings: numpy array shape (n_samples, n_dims)
embeddings = embeddings_3072  # or use `reduced` from PCA above

reducer = UMAP(n_components=2, metric="cosine")
vis_2d = reducer.fit_transform(embeddings)
```

> Sources: [PCA — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), [UMAP documentation](https://umap-learn.readthedocs.io/en/latest/)

### Quantization (Scale lớn: >10M vectors)

Khi vector database chứa hàng triệu/tỷ vectors, memory là bottleneck:

| Method | Memory Reduction | Quality Impact | Khi nào dùng |
|--------|-----------------|-------------|-------------|
| **Scalar Quantization (SQ8)** | 4x (32-bit→8-bit) | Nhỏ — thường chấp nhận được | Scale vừa (10M-100M vectors) |
| **Product Quantization (PQ)** | 16-64x | Lớn hơn SQ8 — cần test | Scale lớn (>100M vectors) |
| **Binary Quantization** | 32x | Đáng kể — dùng cho coarse pass | Coarse filtering, first-pass retrieval |

> Sources: [FAISS — Index types and Quantization](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

---

## 3.5 Hybrid Search (Sparse + Dense)

### Tại sao cần Hybrid?

| Method | Mạnh | Yếu |
|--------|------|------|
| **BM25 (Sparse)** | Exact match, tên riêng, mã số, keywords | Không hiểu synonyms, paraphrases |
| **Dense Embedding** | Semantic similarity, multilingual, paraphrases | Yếu exact match, rare keywords |
| **Hybrid** | Cả hai! | Phức tạp hơn, cần tune weights |

### Reciprocal Rank Fusion (RRF)

```
RRF_score(d) = 1/(k + rank_sparse(d)) + 1/(k + rank_dense(d))
```

- `k = 60` (constant, mặc định trong hầu hết implementations)
- Tỷ lệ khuyến nghị: **70-80% dense + 20-30% sparse** (heuristic, tune theo domain)
- Tune theo domain: legal/medical cần sparse nhiều hơn (exact terminology), casual Q&A cần dense nhiều hơn

### SPLADE: Learned Sparse Model

SPLADE là learned sparse model — output vẫn là sparse vectors (như BM25) nhưng được train bằng neural network → tốt hơn BM25 truyền thống vì expand terms và learn term weights.

```python
"""
Hybrid Search example với Elasticsearch.
Prerequisites: pip install elasticsearch openai; OPENAI_API_KEY và ES endpoint configured.
"""
from openai import OpenAI
client = OpenAI()
# query_embedding: generate từ embed model
query_text = "Matryoshka embedding dimension reduction"
query_embedding = client.embeddings.create(
    model="text-embedding-3-large", input=query_text
).data[0].embedding

# Elasticsearch 8.x hỗ trợ native hybrid search
hybrid_query = {
    "retriever": {
        "rrf": {
            "retrievers": [
                {
                    "standard": {
                        "query": {
                            "match": {
                                "content": "Matryoshka embedding dimension reduction"
                            }
                        }
                    }
                },
                {
                    "knn": {
                        "field": "embedding",
                        "query_vector": query_embedding,
                        "k": 100,
                        "num_candidates": 200
                    }
                }
            ],
            "rank_window_size": 100,
            "rank_constant": 60
        }
    }
}
```

> Source: [Elasticsearch — Semantic text hybrid search](https://www.elastic.co/guide/en/elasticsearch/reference/current/semantic-text-hybrid-search.html)

---

## 3.6 Evaluation Methodology

### Benchmark Frameworks

| Framework | Focus | Datasets | Metrics | Source |
|-----------|-------|----------|---------|--------|
| **MTEB** | General embedding quality | 58+ datasets, 8 tasks, 112 languages | Nhiều (task-dependent) | [paper](https://arxiv.org/abs/2210.07316), [leaderboard](https://huggingface.co/spaces/mteb/leaderboard) |
| **BEIR** | Information Retrieval | 18 datasets, diverse domains | nDCG@10 focus | [repo](https://github.com/beir-cellar/beir) |

**MTEB 8 Tasks**:
1. **Classification** — dùng embedding làm features
2. **Clustering** — phân cụm semantic
3. **Retrieval** — tìm relevant documents
4. **Reranking** — sắp xếp lại kết quả
5. **STS** (Semantic Textual Similarity) — đo similarity score
6. **Pair Classification** — phân loại cặp câu (entailment, paraphrase)
7. **Bitext Mining** — tìm cặp dịch song ngữ
8. **Summarization** — đánh giá embedding cho summarization

### Metrics Set

**Retrieval Metrics:**

| Metric | Ý nghĩa | Formula (simplified) |
|--------|---------|---------------------|
| **nDCG@k** | Chất lượng ranking top-k (penalize relevant docs ở vị trí thấp) | Normalized DCG, higher = better |
| **MAP@k** | Average precision over relevant docs | Mean of AP across queries |
| **Recall@k** | Bao nhiêu relevant docs nằm trong top-k? | \|relevant ∩ top-k\| / \|relevant\| |
| **Precision@k** | Bao nhiêu % top-k là relevant? | \|relevant ∩ top-k\| / k |
| **MRR** | Rank của relevant result đầu tiên | 1 / rank_of_first_relevant |

**RAG-specific Metrics:**

| Metric | Ý nghĩa | Cách đo |
|--------|---------|---------|
| **Faithfulness** | Answer có grounded trong context không? | LLM-as-judge hoặc NLI model |
| **Answer Relevance** | Answer có trả lời đúng câu hỏi không? | LLM-as-judge |
| **Context Precision** | Context được retrieve có relevant không? | So với ground truth |

**Operational Metrics:**

| Metric | Ý nghĩa | Target |
|--------|---------|--------|
| **P95 Latency** | 95th percentile response time | <100ms cho retrieval, <500ms cho rerank |
| **Index RAM/GB** | Bộ nhớ cần cho vector index | Scale linearly với n_vectors × dims × bytes |
| **QPS** | Queries per second | Tùy SLA, thường >100 QPS production |

### Best Practices cho Evaluation

1. **Không dựa hoàn toàn vào MTEB leaderboard** — benchmark trên domain data của bạn
2. **Tách eval set** — không dùng data đã index để eval
3. **Multilingual**: MTEB chưa cover tiếng Việt tốt → cần build eval set riêng
4. **A/B test** trong production — offline metrics không đảm bảo online performance

> Sources: [MTEB paper — Muennighoff et al., 2022](https://arxiv.org/abs/2210.07316), [BEIR benchmark](https://github.com/beir-cellar/beir), [HuggingFace MTEB Blog](https://huggingface.co/blog/mteb)

---

# 📚 Tổng hợp Sources

| # | URL | Description |
|---|-----|-------------|
| 1 | [arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781) | Word2Vec paper — Mikolov et al., 2013 |
| 2 | [aclanthology.org/D14-1162.pdf](https://aclanthology.org/D14-1162.pdf) | GloVe paper — Pennington et al., 2014 |
| 3 | [arxiv.org/abs/1607.04606](https://arxiv.org/abs/1607.04606) | FastText paper — Bojanowski et al., 2016 |
| 4 | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) | Transformer paper — Vaswani et al., 2017 |
| 5 | [arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084) | Sentence-BERT paper — Reimers & Gurevych, 2019 |
| 6 | [arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147) | Matryoshka Representation Learning — Kusupati et al., 2022 |
| 7 | [arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316) | MTEB benchmark paper — Muennighoff et al., 2022 |
| 8 | [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020) | CLIP paper — Radford et al., 2021 |
| 9 | [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) | RAG paper — Lewis et al., 2020 |
| 10 | [arxiv.org/pdf/2409.10173](https://arxiv.org/pdf/2409.10173) | Jina Embeddings v3 paper, 2024 |
| 11 | [blog.google/.../gemini-embedding-2/](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/) | Gemini Embedding 2 announcement |
| 12 | [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing) | Gemini API pricing |
| 13 | [cloud.google.com/vertex-ai/generative-ai/pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) | Vertex AI pricing |
| 14 | [cloud.google.com/...get-multimodal-embeddings](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings) | Vertex Multimodal Embeddings docs |
| 15 | [openai.com/.../new-embedding-models-and-api-updates/](https://openai.com/index/new-embedding-models-and-api-updates/) | OpenAI embedding-3 announcement |
| 16 | [docs.cohere.com/docs/cohere-embed](https://docs.cohere.com/docs/cohere-embed) | Cohere Embed models documentation |
| 17 | [docs.cohere.com/changelog/embed-multimodal-v4](https://docs.cohere.com/changelog/embed-multimodal-v4) | Cohere Embed v4 changelog |
| 18 | [docs.pinecone.io/guides/search/rerank-results](https://docs.pinecone.io/guides/search/rerank-results) | Pinecone rerankers guide |
| 19 | [docs.cohere.com/docs/rerank-overview](https://docs.cohere.com/docs/rerank-overview) | Cohere Rerank overview |
| 20 | [sbert.net/.../semantic-search/](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) | SBERT semantic search |
| 21 | [sbert.net/.../paraphrase-mining/](https://www.sbert.net/examples/sentence_transformer/applications/paraphrase-mining/README.html) | SBERT paraphrase mining / deduplication |
| 22 | [github.com/facebookresearch/faiss/wiki/](https://github.com/facebookresearch/faiss/wiki/) | FAISS — metrics, indexes, quantization |
| 23 | [developers.google.com/.../candidate-generation](https://developers.google.com/machine-learning/recommendation/overview/candidate-generation) | Google ML — Recommendation candidate generation |
| 24 | [scikit-learn.org/.../KMeans.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) | scikit-learn K-means |
| 25 | [scikit-learn.org/.../HDBSCAN.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) | scikit-learn HDBSCAN |
| 26 | [scikit-learn.org/.../IsolationForest.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) | scikit-learn Isolation Forest |
| 27 | [scikit-learn.org/.../PCA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) | scikit-learn PCA |
| 28 | [umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/en/latest/) | UMAP documentation |
| 29 | [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir) | BEIR benchmark framework |
| 30 | [huggingface.co/blog/mteb](https://huggingface.co/blog/mteb) | HuggingFace MTEB overview |
| 31 | [docs.pinecone.io/guides/get-started/overview](https://docs.pinecone.io/guides/get-started/overview) | Pinecone documentation |
| 32 | [docs.weaviate.io/weaviate/introduction](https://docs.weaviate.io/weaviate/introduction) | Weaviate documentation |
| 33 | [qdrant.tech/documentation/overview/](https://qdrant.tech/documentation/overview/) | Qdrant documentation |
| 34 | [milvus.io/docs/overview.md](https://milvus.io/docs/overview.md) | Milvus documentation |
| 35 | [docs.trychroma.com/docs/overview/introduction](https://docs.trychroma.com/docs/overview/introduction) | Chroma documentation |
| 36 | [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector) | pgvector — PostgreSQL vector extension |
| 37 | [pinecone.io/learn/chunking-strategies/](https://www.pinecone.io/learn/chunking-strategies/) | Pinecone chunking strategies guide |
| 38 | [learn.microsoft.com/.../vector-search-how-to-chunk-documents](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents) | Azure — Chunking for vector search |
| 39 | [elastic.co/.../semantic-text-hybrid-search.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/semantic-text-hybrid-search.html) | Elasticsearch hybrid search |
| 40 | [scikit-learn.org/.../cosine_similarity.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) | scikit-learn cosine similarity |
| 41 | [github.com/openai/openai-cookbook/.../Classification_using_embeddings.ipynb](https://github.com/openai/openai-cookbook/blob/main/examples/Classification_using_embeddings.ipynb) | OpenAI Cookbook — Classification |
| 42 | [openai.com/index/introducing-text-and-code-embeddings/](https://openai.com/index/introducing-text-and-code-embeddings/) | OpenAI — Text and Code Embeddings |
| 43 | [sbert.net/docs/.../pretrained_models.html](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) | SBERT pretrained models list |

---

> **Disclaimer**: Pricing data là snapshot tại **March 2026**. Benchmark data có nguồn khác nhau (vendor-reported, paper-reported, independent). Gemini Embedding 2 đang ở **preview** — specs và pricing có thể thay đổi. Luôn kiểm tra official docs cho thông tin mới nhất.
