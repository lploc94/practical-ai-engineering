"""Sinh lora-update.svg: SVG tĩnh, LoRA giữ nguyên W và chỉ huấn luyện cặp B·A (§1.4).

Kích thước lấy từ config.json của Qwen/Qwen2.5-0.5B-Instruct (model dùng ở §3.3):
hidden_size 896, intermediate_size 4864, 24 lớp, 14 attention head, 2 KV head
(head_dim 64, nên k_proj và v_proj ra 128), tie_word_embeddings true, q/k/v có bias.
Ví dụ một ma trận: gate_proj của một lớp MLP, weight PyTorch dạng (out, in) = (4864, 896).
LoRA rank 16 trên mọi lớp linear (target_modules="all-linear", như train_lora.py):
B có dạng (out, r), A có dạng (r, in). Tổng số tham số tính lại từ config phải khớp
con số §1.4 và §3.3: 8,8 triệu trên 502,8 triệu (1,75%).
Hình tĩnh vì đây là cấu trúc, không có diễn biến theo thời gian.

Chạy: python3 make_lora_update.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

HID, INTER, LAYERS, HEADS, KV, VOCAB, R = 896, 4864, 24, 14, 2, 151936, 16
KVD = HID // HEADS * KV                                   # 128
# (in, out) của 7 ma trận linear trong mỗi lớp decoder
LINEARS = {"q": (HID, HID), "k": (HID, KVD), "v": (HID, KVD), "o": (HID, HID),
           "gate": (HID, INTER), "up": (HID, INTER), "down": (INTER, HID)}
BIAS = HID + KVD + KVD                                    # q, k, v có bias
per_layer = sum(i * o for i, o in LINEARS.values()) + BIAS + 2 * HID   # + 2 RMSNorm
base = VOCAB * HID + LAYERS * per_layer + HID              # embedding dùng chung với lm_head
lora = LAYERS * sum(R * (i + o) for i, o in LINEARS.values())
total = base + lora
assert base == 494_032_768                                 # số tham số công bố của Qwen2.5-0.5B
assert round(lora / 1e6, 1) == 8.8 and round(total / 1e6, 1) == 502.8
assert f"{lora / total:.2%}" == "1.75%"

w_gate = HID * INTER
b_gate, a_gate = INTER * R, R * HID
assert (w_gate, b_gate + a_gate) == (4_358_144, 92_160)
ratio = (b_gate + a_gate) / w_gate


def vn(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def pct(v: float, nd: int) -> str:
    return f"{v * 100:.{nd}f}".replace(".", ",") + "%"


W, H = 860, 440
body: list[str] = []
body.append('<text class="title" x="20" y="38">LoRA: giữ nguyên W, chỉ huấn luyện cặp ma trận hạng thấp</text>')
body.append(svgkit.legend(W - 330, 66, [("blue", "đóng băng"), ("orange", "được huấn luyện")]))
body.append('<text class="cap2" x="20" y="66">Ví dụ: gate_proj của một lớp MLP trong Qwen2.5-0.5B, rank r = 16.</text>')

TOP = 110
# W: hình chữ nhật cao (out 4864) x rộng (in 896); không theo đúng tỉ lệ
WX, WW, WH = 60, 150, 220
body.append(f'<rect class="blue" x="{WX}" y="{TOP}" width="{WW}" height="{WH}" rx="7"/>')
body.append(f'<text class="row mid" x="{WX + WW / 2}" y="{TOP + WH / 2 - 4}">W</text>')
body.append(f'<text class="sub mid" x="{WX + WW / 2}" y="{TOP + WH / 2 + 16}">4864 × 896</text>')
body.append(f'<text class="cap mid" x="{WX + WW / 2}" y="{TOP + WH + 26}">{vn(w_gate)} tham số</text>')
body.append(f'<text class="sub mid" x="{WX + WW / 2}" y="{TOP + WH + 46}">không đổi khi huấn luyện</text>')

body.append(f'<text class="title mid" x="{WX + WW + 45}" y="{TOP + WH / 2 + 7}">+</text>')

# B: cao 4864, rộng r; A: cao r, rộng 896
BX, BW_ = WX + WW + 90, 22
body.append(f'<rect class="orange" x="{BX}" y="{TOP}" width="{BW_}" height="{WH}" rx="4"/>')
body.append(f'<text class="row mid" x="{BX + BW_ / 2}" y="{TOP - 10}">B</text>')
body.append(f'<text class="sub" x="{BX - 4}" y="{TOP + WH + 26}">4864 × 16</text>')
AX = BX + BW_ + 20
body.append(f'<rect class="orange" x="{AX}" y="{TOP}" width="{WW}" height="{BW_}" rx="4"/>')
body.append(f'<text class="row mid" x="{AX + WW / 2}" y="{TOP - 10}">A</text>')
body.append(f'<text class="sub end" x="{AX + WW}" y="{TOP + BW_ + 18}">16 × 896</text>')
body.append(f'<text class="sub" x="{AX}" y="{TOP + BW_ + 56}">B·A cùng kích thước với W,</text>')
body.append(f'<text class="sub" x="{AX}" y="{TOP + BW_ + 74}">nhưng hạng tối đa 16</text>')
body.append(f'<text class="cap" x="{BX}" y="{TOP + WH + 50}" style="font-weight:600">'
            f'{vn(b_gate)} + {vn(a_gate)} = {vn(b_gate + a_gate)} tham số ({pct(ratio, 1)} của W)</text>')

# Cột phải: cả model và lúc serve
RX = 610
body.append(f'<line class="grid" x1="{RX - 24}" y1="{TOP - 20}" x2="{RX - 24}" y2="{TOP + WH + 10}"/>')
body.append(f'<text class="row" x="{RX}" y="{TOP + 10}">Cả model, mọi lớp linear</text>')
body.append(f'<text class="sub" x="{RX}" y="{TOP + 30}">7 ma trận × 24 lớp</text>')
body.append(f'<text class="cap" x="{RX}" y="{TOP + 62}">{vn(lora)} tham số train</text>')
body.append(f'<text class="cap" x="{RX}" y="{TOP + 84}">trên {vn(total)} ({pct(lora / total, 2)})</text>')
body.append(f'<text class="row" x="{RX}" y="{TOP + 136}">Khi serve</text>')
body.append(f'<text class="cap" x="{RX}" y="{TOP + 162}">gộp W + (α/r)·B·A thành một</text>')
body.append(f'<text class="cap" x="{RX}" y="{TOP + 182}">ma trận: không thêm độ trễ,</text>')
body.append(f'<text class="cap" x="{RX}" y="{TOP + 202}">hoặc giữ adapter riêng (§3.5)</text>')

body.append(f'<text class="cap2" x="20" y="{H - 14}">Kích thước tính từ config của Qwen2.5-0.5B; hình không theo '
            'đúng tỉ lệ (W thật cao gấp 5,4 lần bề rộng).</text>')

svgkit.write(__file__, "lora-update.svg", svgkit.svg(
    W, H, f"LoRA giữ nguyên ma trận W 4864×896 ({vn(w_gate)} tham số) và chỉ huấn luyện B 4864×16 cộng A 16×896, "
    f"{vn(b_gate + a_gate)} tham số, {pct(ratio, 1)} của W. Trên cả Qwen2.5-0.5B, rank 16 mọi lớp linear: "
    f"{vn(lora)} trên {vn(total)} tham số ({pct(lora / total, 2)}). Khi serve có thể gộp B·A vào W", body))
