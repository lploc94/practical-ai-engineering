"""Sinh stage-recall.svg: SVG tĩnh minh hoạ cách đọc Recall của từng tầng (§1.2).

Dùng đúng hai ví dụ trong đoạn văn của §1.2 (số liệu ví dụ, không phải đo thật):
- Trường hợp A: Recall@50 của retrieval = 0,85, Recall@10 sau rerank = 0,70.
  Top-10 sau rerank là tập con của 50 candidate, nên 0,15 rơi ở rerank và
  0,15 chưa bao giờ vào top-50.
- Trường hợp B: Recall@50 = 0,50. Bài không cho Recall@10, nên phần 0,50 đã tới
  reranker vẽ là "chưa đo"; reranker tốt mấy cũng không vượt mức 0,50 này.
100% = mọi câu hỏi có tài liệu đúng trong corpus.

Chạy: python3 make_stage_recall.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

A_R50, A_R10 = 0.85, 0.70
B_R50 = 0.50
assert A_R10 <= A_R50, "top-10 sau rerank là tập con của top-50"
A_RERANK_LOSS = round(A_R50 - A_R10, 2)
A_RETR_LOSS = round(1 - A_R50, 2)
B_RETR_LOSS = round(1 - B_R50, 2)
assert (A_RERANK_LOSS, A_RETR_LOSS, B_RETR_LOSS) == (0.15, 0.15, 0.50)

W, H = 740, 420
X0, BARW, BH = 200, 450, 44
RA, RB = 172, 310


def pct(v: float) -> str:
    return f"{round(v * 100)}%"


def vn(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def seg(x0: float, x1: float, y: float, cls: str, label: str = "") -> str:
    x, w = X0 + x0 * BARW, (x1 - x0) * BARW
    out = f'<rect class="{cls}" x="{x + 1}" y="{y}" width="{w - 2}" height="{BH}" rx="5"/>'
    if label:
        out += f'<text class="cap mid halo" x="{x + w / 2}" y="{y + BH / 2 + 5}">{label}</text>'
    return out


def marker(v: float, y: float, text: str, anchor: str) -> str:
    x = X0 + v * BARW
    cls = "sub end" if anchor == "end" else "sub"
    dx = -4 if anchor == "end" else 4
    return (f'<line class="lfg" x1="{x}" y1="{y - 16}" x2="{x}" y2="{y + BH + 4}"/>'
            f'<text class="{cls}" x="{x + dx}" y="{y - 6}">{text}</text>')


body: list[str] = []
body.append('<text class="title" x="20" y="36">Recall của từng tầng chỉ ra tầng cần sửa</text>')
body.append('<text class="cap2" x="20" y="58">Số liệu ví dụ của §1.2. Cả thanh = mọi câu hỏi có tài liệu đúng trong corpus.</text>')
body.append(svgkit.legend(20, 90, [("blue", "vào top-10 sau rerank"), ("tok", "trong top-50, rơi ở rerank")]))
body.append(svgkit.legend(20, 114, [("slot", "đã tới reranker, chưa đo"), ("hatch", "không có trong top-50")]))

# Trường hợp A
body.append(f'<text class="row" x="20" y="{RA + 28}">Trường hợp A</text>')
body.append(seg(0, A_R10, RA, "blue", f"{pct(A_R10)} vào top-10"))
body.append(seg(A_R10, A_R50, RA, "tok", pct(A_RERANK_LOSS)))
body.append(seg(A_R50, 1, RA, "hatch", pct(A_RETR_LOSS)))
body.append(marker(A_R10, RA, f"Recall@10 = {vn(A_R10)}", "end"))
body.append(marker(A_R50, RA, f"Recall@50 = {vn(A_R50)}", "start"))
body.append(f'<text class="cap" x="{X0}" y="{RA + BH + 26}">Retrieval ổn. Sửa rerank hoặc số đoạn đưa vào prompt.</text>')

# Trường hợp B
body.append(f'<text class="row" x="20" y="{RB + 28}">Trường hợp B</text>')
body.append(seg(0, B_R50, RB, "slot", f"{pct(B_R50)} tới reranker"))
body.append(seg(B_R50, 1, RB, "hatch", f"{pct(B_RETR_LOSS)} không tới"))
body.append(marker(B_R50, RB, f"Recall@50 = {vn(B_R50)}", "start"))
body.append(f'<text class="cap" x="{X0}" y="{RB + BH + 26}">Reranker tốt mấy cũng không vượt {pct(B_R50)}.</text>')
body.append(f'<text class="cap" x="{X0}" y="{RB + BH + 46}">Sửa chunk, embedding, hoặc thêm BM25.</text>')

svgkit.write(__file__, "stage-recall.svg", svgkit.svg(
    W, H, f"Trường hợp A: Recall@50 bằng {vn(A_R50)}, Recall@10 sau rerank bằng {vn(A_R10)}, "
    f"{pct(A_RERANK_LOSS)} đáp án rơi ở rerank nên sửa rerank hoặc số đoạn. Trường hợp B: Recall@50 bằng "
    f"{vn(B_R50)}, một nửa đáp án không tới reranker nên phải sửa chunk, embedding hoặc thêm BM25", body))
