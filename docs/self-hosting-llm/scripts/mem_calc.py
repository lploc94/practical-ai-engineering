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
