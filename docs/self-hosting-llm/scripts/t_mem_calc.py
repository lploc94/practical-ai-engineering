import json
from mem_calc import GiB, MiB, kv_bytes, plan, weights_gib

cfg = lambda n: json.load(open(f"configs/{n}.json"))
H100, H200 = 81559 / 1024, 143771 / 1024          # GiB mà nvidia-smi báo cho H100 80GB HBM3, H200
q25 = cfg("qwen2.5-3b")

# 1. khớp số llama.cpp in ra cho Qwen2.5-3B (f16): 4096 token = 144 MiB; 4 slot x 4096 = 576 MiB
assert kv_bytes(q25, 4096) == 144 * MiB and 4 * kv_bytes(q25, 4096) == 576 * MiB
# 2. Qwen2.5 khai báo sliding_window nhưng use_sliding_window=false: mọi lớp là full attention
assert q25["sliding_window"] == 32768 and q25["use_sliding_window"] is False
assert kv_bytes(q25, 32768) == 32 * kv_bytes(q25, 1024)
# 3. GQA: Llama 3.1 8B 8 KV head x 128 x 32 lớp x 2 (K,V) x 2 byte = 128 KiB mỗi token
assert kv_bytes(cfg("unsloth_Llama-3.1-8B-Instruct"), 1) == 128 * 1024
# 4. gpt-oss: nửa số lớp sliding 128 token, nên KV gần như chỉ tăng theo nửa còn lại
oss = cfg("openai_gpt-oss-120b")
assert kv_bytes(oss, 128) == 36 * 2 * 8 * 64 * 2 * 128
assert kv_bytes(oss, 32768) - kv_bytes(oss, 16384) == 18 * 2 * 8 * 64 * 2 * 16384
# 5. KV 1 byte (FP8) bằng nửa KV 2 byte
assert kv_bytes(q25, 1000, elem_bytes=1) * 2 == kv_bytes(q25, 1000)
# 6. plan: Llama 3.3 70B FP8 trên một H100: weights 65 GiB, không còn chỗ cho chuỗi 32k nào
llama70 = cfg("unsloth_Llama-3.3-70B-Instruct")
w = weights_gib(70.55, 8)
assert round(w, 1) == 65.7
p = plan(llama70, w, H100, 32768)
assert p["max_seqs"] == 0 and 10_000 < p["kv_tokens"] < 20_000, p      # vẫn chạy được chuỗi ngắn hơn
# 7. cùng model trên 2 GPU H200 141 GB (tensor parallel): KV chia đều cho 2 GPU
p = plan(llama70, w, H200, 32768, gpus=2)
assert p["max_seqs"] > 10, p
# 8. gpt-oss-120b: checkpoint MXFP4 60,8 GiB vừa một GPU 80 GB, KV nhỏ nhờ sliding window
p = plan(cfg("openai_gpt-oss-120b"), 60.8, H100, 32768)
assert p["max_seqs"] >= 4, p
# 9. Gemma 3 27B: tham số nằm trong text_config; sliding_window_pattern 6 nghĩa là lớp 6, 12, ..., 60
#    là full attention (10 lớp), 52 lớp còn lại sliding 1024 token
g3 = cfg("unsloth_gemma-3-27b-it")
per_layer = 2 * 16 * 128 * 2
assert kv_bytes(g3, 32768) == per_layer * (10 * 32768 + 52 * 1024)
assert round(kv_bytes(g3, 32768) / GiB, 2) == 2.91
# 10. Qwen3.5: lớp linear_attention không có KV theo token, chỉ 10 lớp full attention tính
q35 = cfg("Qwen_Qwen3.5-35B-A3B")
assert kv_bytes(q35, 32768) == 10 * 2 * 2 * 256 * 2 * 32768
# 11. không đủ thông tin để biết lớp nào sliding (Gemma 2, Mistral 7B v0.1): tính như full attention,
#     ước lượng dư chứ không thiếu
g2, m7 = cfg("unsloth_gemma-2-9b-it"), cfg("mistralai_Mistral-7B-Instruct-v0.1")
assert kv_bytes(g2, 32768) == 42 * 2 * 8 * 256 * 2 * 32768
assert kv_bytes(m7, 32768) == 32 * 2 * 8 * 128 * 2 * 32768
# 12. loại lớp chưa biết thì báo lỗi thay vì đoán
try:
    kv_bytes({**q25, "layer_types": ["mamba"] * 36}, 10)
    raise AssertionError("phải báo lỗi")
except ValueError:
    pass
print("MEM_CALC OK")
