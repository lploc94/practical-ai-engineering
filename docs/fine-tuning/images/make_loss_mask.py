"""Sinh loss-mask.svg: SVG tĩnh, token nào được tính loss trong SFTTrainer (§3.1).

Hai ví dụ lấy đúng từ scripts/t_masking.py, kết quả theo các assert trong đó:
- Ví dụ ngắn `one` (system "Phân loại.", user "Huỷ đơn DH-1", đáp án {"intent":"cancel"}):
  mặc định loss tính trên cả system và user (test 2); với assistant_only_loss=True
  phần được tính đúng bằng '\\n{"intent":"cancel"}<|im_end|>\\n' (test 1).
- Ví dụ dài `long` ("máy giặt kêu to khi vắt. " × 40) với max_length=128: mặc định còn
  128 token và không còn đáp án; với assistant_only_loss ví dụ bị bỏ (test 4).
Số token đếm bằng chat template của Qwen2.5-0.5B-Instruct (transformers 5.22.2,
apply_chat_template). Nếu import được transformers, script đếm lại và assert các hằng số
dưới đây; nếu không, script dùng hằng số đã ghi (in ra cảnh báo). Số token tính theo
lượt, gồm cả các thẻ <|im_start|>, tên vai, xuống dòng, <|im_end|>:
- `one`: 30 token; hết system 9, hết user 20; phần được tính loss 8 token
  (xuống dòng sau "assistant", 5 token JSON, <|im_end|>, xuống dòng).
- `long`: 440 token; chat template tự thêm lượt system mặc định của Qwen (21 token),
  lượt user 408 token, "<|im_start|>assistant" 2 token, phần đáp án 9 token.
Ví dụ ngắn vẽ không theo tỉ lệ token để đọc được chữ; ví dụ dài vẽ theo tỉ lệ.
Hình tĩnh: so bốn trạng thái cạnh nhau, không có diễn biến theo thời gian.

Chạy: python3 make_loss_mask.py (có transformers và tokenizer Qwen trong cache thì đếm lại số token)
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "assets"))
import svgkit  # noqa: E402

# Ghim vào chuỗi đã assert trong t_masking.py
t_masking = (HERE.parent / "scripts" / "t_masking.py").read_text(encoding="utf-8")
LABELLED = '\n{"intent":"cancel"}<|im_end|>\n'
assert repr(LABELLED) in t_masking
assert '"Phân loại."' in t_masking and '"Huỷ đơn DH-1"' in t_masking
assert "max_length=128" in t_masking and '"máy giặt kêu to khi vắt. " * 40' in t_masking
assert re.search(r"len\(row\[\"input_ids\"\]\) == 128", t_masking)

ONE_TOTAL, ONE_SYS, ONE_USER, ONE_LOSS = 30, 9, 11, 8
assert ONE_SYS + ONE_USER + 2 + ONE_LOSS == ONE_TOTAL          # + "<|im_start|>assistant"
LONG_SYS, LONG_USER, LONG_HEAD, LONG_ANS, MAXLEN = 21, 408, 2, 9, 128
LONG_TOTAL = LONG_SYS + LONG_USER + LONG_HEAD + LONG_ANS
assert LONG_TOTAL == 440 and LONG_SYS < MAXLEN < LONG_SYS + LONG_USER   # cắt giữa lượt user

ONE = [{"role": "system", "content": "Phân loại."}, {"role": "user", "content": "Huỷ đơn DH-1"},
       {"role": "assistant", "content": '{"intent":"cancel"}'}]
LONG = [{"role": "user", "content": "máy giặt kêu to khi vắt. " * 40},
        {"role": "assistant", "content": '{"intent":"warranty"}'}]
try:
    from transformers import AutoTokenizer
except ImportError:
    print("không có transformers: dùng số token đã ghi trong script, không đếm lại")
else:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    def n(msgs) -> int:
        return len(tok.apply_chat_template(msgs, tokenize=True, return_dict=True)["input_ids"])

    assert [n(ONE[:k]) for k in (1, 2, 3)] == [ONE_SYS, ONE_SYS + ONE_USER, ONE_TOTAL]
    assert len(tok(LABELLED, add_special_tokens=False)["input_ids"]) == ONE_LOSS
    assert tok.apply_chat_template(ONE, tokenize=False).endswith("<|im_start|>assistant" + LABELLED)
    assert [n(LONG[:1]), n(LONG)] == [LONG_SYS + LONG_USER, LONG_TOTAL]
    ids = tok.apply_chat_template(LONG, tokenize=True, return_dict=True)["input_ids"]
    sys_turn = tok.decode(ids[:LONG_SYS])                 # lượt system mặc định do template tự thêm
    assert sys_turn.startswith("<|im_start|>system\nYou are Qwen") and sys_turn.endswith("<|im_end|>\n")
    assert tok.decode(ids[-(LONG_HEAD + LONG_ANS):]) == "<|im_start|>assistant\n" + '{"intent":"warranty"}<|im_end|>\n'
    print("đã đếm lại số token bằng tokenizer: khớp")

W, H = 900, 490
X0, SW, BH = 250, 600, 40
body: list[str] = []
body.append('<text class="title" x="20" y="38">Token nào được tính loss khi SFT bằng TRL</text>')
body.append(svgkit.legend(W - 360, 38, [("orange", "tính loss"), ("tok", "không tính"), ("slot", "bị cắt")]))
body.append('<text class="cap2" x="20" y="62">Phần được tính loss theo các test trong t_masking.py (TRL 1.17.1); '
            'số token đếm bằng chat template của Qwen2.5-0.5B-Instruct.</text>')


def seg(x: float, y: float, w: float, cls: str, text: str = "", mono: bool = False) -> str:
    out = f'<rect class="{cls}" x="{x:.1f}" y="{y}" width="{w:.1f}" height="{BH}" rx="5"/>'
    if text:
        style = ' style="font-size:12px"' if mono else ""
        k = "lbl" if mono else "cap mid"
        out += f'<text class="{k}"{style} x="{x + w / 2:.1f}" y="{y + BH / 2 + 5}">{text}</text>'
    return out


def label(y: float, name: str, sub: str) -> None:
    body.append(f'<text class="row" x="20" y="{y + 16}">{name}</text>')
    body.append(f'<text class="sub" x="20" y="{y + 34}">{sub}</text>')


# ---- Ví dụ ngắn, không theo tỉ lệ ----
body.append(f'<text class="row" x="{X0}" y="104">Ví dụ ngắn: {ONE_TOTAL} token</text>')
widths = [140, 140, 76, 150, 86]
texts = [("system: Phân loại.", False), ("user: Huỷ đơn DH-1", False), ("assistant", False),
         ('{"intent":"cancel"}', True), ("&lt;|im_end|&gt;", True)]
assert sum(widths) + 2 * (len(widths) - 1) == SW


def short_row(y: float, classes: list[str]) -> None:
    x = X0
    for w, (t, mono), c in zip(widths, texts, classes):
        body.append(seg(x, y, w, c, t, mono))
        x += w + 2


RA, RB = 118, 172
label(RA, "Mặc định", "cả system và user")
short_row(RA, ["orange"] * 5)
label(RB, "assistant_only_loss", f"{ONE_LOSS}/{ONE_TOTAL} token: đáp án và token dừng")
short_row(RB, ["tok", "tok", "tok", "orange", "orange"])

# ---- Ví dụ dài, theo tỉ lệ ----
PX = SW / LONG_TOTAL


def tx(n: float) -> float:
    return X0 + n * PX


body.append(f'<text class="row" x="{X0}" y="262">Ví dụ dài: {LONG_TOTAL} token, max_length = {MAXLEN}</text>')
# Chú thích các đoạn, đặt phía trên dải
LY = 292
for a, b, t, anchor in ((0, LONG_SYS, "system", "start"), (LONG_SYS, LONG_SYS + LONG_USER, f"lượt user: {LONG_USER} token", "mid"),
                        (LONG_TOTAL - LONG_ANS, LONG_TOTAL, "đáp án", "end")):
    x1, x2 = tx(a), tx(b)
    body.append(f'<path class="lmuted" d="M{x1 + 1:.1f},{LY + 6}V{LY}H{x2 - 1:.1f}V{LY + 6}"/>')
    if anchor == "mid":
        body.append(f'<text class="sub mid" x="{(x1 + x2) / 2:.1f}" y="{LY - 6}">{t}</text>')
    elif anchor == "end":
        body.append(f'<text class="sub end" x="{x2:.1f}" y="{LY - 6}">{t}</text>')
    else:
        body.append(f'<text class="sub" x="{x1:.1f}" y="{LY - 6}">{t}</text>')

RC, RD = 308, 362
cut = tx(MAXLEN)
label(RC, "Mặc định", f"còn {MAXLEN} token, mất đáp án")
body.append(seg(X0, RC, cut - X0 - 1, "orange"))
body.append(seg(cut + 1, RC, X0 + SW - cut - 1, "slot", "bị cắt khỏi ví dụ"))
label(RD, "assistant_only_loss", "che hết, TRL bỏ ví dụ")
body.append(seg(X0, RD, cut - X0 - 1, "tok"))
body.append(seg(cut + 1, RD, X0 + SW - cut - 1, "slot", "bị cắt khỏi ví dụ"))
body.append(f'<line class="lfg dash" x1="{cut:.1f}" y1="{RC - 10}" x2="{cut:.1f}" y2="{RD + BH + 10}"/>')
body.append(f'<text class="sub mid" x="{cut:.1f}" y="{RD + BH + 26}">token {MAXLEN}</text>')

body.append(f'<text class="cap2" x="20" y="{H - 30}">TRL cắt từ cuối (keep_start): cắt mất đáp án và token dừng. '
            'Lọc ví dụ quá dài trước khi tới trainer.</text>')
body.append(f'<text class="cap2" x="20" y="{H - 12}">Ví dụ ngắn vẽ không theo tỉ lệ token; ví dụ dài vẽ theo tỉ lệ. '
            'Chat template tự thêm system mặc định.</text>')

svgkit.write(__file__, "loss-mask.svg", svgkit.svg(
    W, H, f"Mặc định TRL tính loss trên mọi token, cả system và user; với assistant_only_loss chỉ {ONE_LOSS}/"
    f"{ONE_TOTAL} token của đáp án và token dừng được tính. Ví dụ {LONG_TOTAL} token với max_length {MAXLEN} bị "
    "cắt giữa lượt user: mặc định model học trên ví dụ không có đáp án, còn với assistant_only_loss ví dụ bị "
    "bỏ mà không có cảnh báo", body))
