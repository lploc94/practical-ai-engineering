"""Sinh stream-buffer.svg: SVG động so "stream ngay, kiểm sau" với "giữ tới hết câu, kiểm rồi phát" (§3.2).

Ba câu lấy từ test 2 của scripts/t_stream_guard.py; câu thứ ba hứa hoàn tiền. Script chạy
thật `guarded_text` của scripts/stream_guard.py (event giả lập, `check` chặn khi gặp chữ
"hoàn tiền", min_chars=20 như trong test) và assert kết quả: hai câu đầu được phát,
câu ba bị chặn và không bao giờ được phát.

Làn trên là hành vi mặc định của NeMo khi stream (`stream_first: True`): client nhận
chunk trước khi rail chạy trên chunk đó, nên khi rail chặn thì phần đã gửi vẫn nằm
trên màn hình [16]. Hình vẽ theo từng câu cho dễ so; NeMo thật gom theo `chunk_size`
token.

Thời gian giả định, chỉ để thấy thứ tự: model bắt đầu sinh câu i ở giây 1, 5, 9 và sinh
xong sau 3 giây (giây 4, 8, 12); mỗi lần `check` mất 2 giây; phát cho khách thêm 1 giây.
Ở làn trên, ô của câu hiện ngay lúc model bắt đầu sinh câu đó, vì token tới khách ngay
khi sinh; rail chạy sau khi có đủ chunk. Ở làn dưới, ô "đang giữ" hiện từ lúc bắt đầu
sinh (chữ dồn vào bộ đệm), `check` chỉ chạy khi gặp ranh giới câu, nên chữ tới màn hình
muộn hơn cả thời gian sinh hết câu cộng một lần `check`, đúng như §3.1 và §3.2 ghi. Chuyển động cần thiết vì ý
chính là thứ tự: ở làn trên chữ tới khách trước khi được kiểm, ở làn dưới chữ chỉ tới
khách sau khi kiểm. Khung tĩnh (prefers-reduced-motion) hiện trạng thái cuối.

Chạy: python3 make_stream_buffer.py
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "assets"))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import svgkit  # noqa: E402
from stream_guard import guarded_text  # noqa: E402

README = (HERE.parents[1] / "README.md").read_text(encoding="utf-8")
TEST = (HERE.parents[1] / "scripts" / "t_stream_guard.py").read_text(encoding="utf-8")
S = ["Đơn hàng của anh đang được giao. ", "Dự kiến tới nơi vào thứ Sáu. ",
     "Shop cam kết hoàn tiền 100% nếu trễ. "]
for i, s in enumerate(S, 1):
    assert f's{i} = "{s}"' in TEST
assert "`stream_first` | `True` | `True`: client nhận token trước khi rail chạy trên chunk đó" in README
assert "phần đã gửi vẫn nằm trên màn hình [16]" in README
assert "**TTFT tăng** thêm thời gian sinh đoạn đầu (ít nhất `min_chars` ký tự, tới ranh giới câu) cộng một lần `check`" in README
assert "đổi lại mỗi đoạn chỉ hiện sau khi model sinh tới hết câu và `check` chạy xong" in README


async def _run():
    async def events():
        for t in S + ["Câu sau không được đọc tới. "]:
            yield NS(type="content_block_delta", delta=NS(type="text_delta", text=t))

    async def check(window):
        return "cam_ket_hoan_tien" if "hoàn tiền" in window else None

    return [o async for o in guarded_text(events(), check, min_chars=20)]


OUT = asyncio.run(_run())
assert [o.kind for o in OUT] == ["text", "text", "blocked"]
assert [o.text for o in OUT[:2]] == S[:2]
LABELS = ["câu 1: đang được giao", "câu 2: tới vào thứ Sáu", "câu 3: hứa hoàn tiền 100%"]
assert "đang được giao" in S[0] and "thứ Sáu" in S[1] and "hoàn tiền 100%" in S[2]

# Mốc thời gian (giây), giả định
START = [1, 5, 9]        # model bắt đầu sinh câu i
GEN_DUR = 3              # sinh một câu mất 3 giây
GEN = [t + GEN_DUR for t in START]   # model sinh xong câu i
CHECK = 2                # một lần check
SEND = 1                 # đưa đoạn đã duyệt xuống màn hình
DUR, FADE = 24, 95
FIN = DUR * FADE / 100   # bắt đầu mờ dần để lặp lại
TF = 0.8                 # thời gian hiện dần của một phần tử

a = svgkit.Anim(dur=DUR, fade_out=FADE)


def at(t: float) -> str:
    p = t / DUR * 100
    return a([(p, 0), ((t + TF) / DUR * 100, 1)])


W, H = 980, 430
LX, X0, CW, GAP, BH = 20, 190, 240, 15, 36
assert X0 + 3 * CW + 2 * GAP <= W - 30


def cx(i: int) -> float:
    return X0 + i * (CW + GAP)


body: list[str] = []
body.append('<text class="title" x="20" y="38">Câu vi phạm có tới màn hình khách không</text>')
body.append(svgkit.legend(20, 66, [("blue", "đã hiện cho khách"), ("tok", "đang giữ, chưa hiện"),
                                    ("hatch", "vi phạm")]))

steps: list[float] = []


def screen_box(y: float, i: int, t: float) -> None:
    steps.append(t)
    body.append(svgkit.box(cx(i), y, CW, BH, "blue", "", at(t)).replace(
        "</g>", f'<text class="cap mid" x="{cx(i) + CW / 2}" y="{y + 23}">{LABELS[i]}</text></g>'))


def mark(y: float, i: int, t: float, ok: bool, extra: str = "") -> None:
    steps.append(t)
    cls, txt = ("tblue", "✓ qua") if ok else ("tred", "✗ chặn")
    body.append(f'<g {at(t)}><text class="cap {cls}" x="{cx(i) + 8}" y="{y}">{txt}</text>'
                + (f'<text class="sub" x="{cx(i) + 62}" y="{y}">{extra}</text>' if extra else "") + "</g>")


def slots(y: float) -> None:
    for i in range(3):
        body.append(f'<rect class="slot" x="{cx(i)}" y="{y}" width="{CW}" height="{BH}" rx="7"/>')


# ---------- Làn trên: stream ngay, kiểm sau ----------
Y1 = 104
body.append(f'<text class="row" x="{LX}" y="{Y1}">Stream ngay, kiểm sau</text>'
            f'<text class="sub" x="{LX + 190}" y="{Y1}">như NeMo mặc định, stream_first: True</text>')
SC1 = Y1 + 14
body.append(f'<text class="cap" x="{LX}" y="{SC1 + 23}">màn hình</text>')
slots(SC1)
for i, g in enumerate(GEN):
    screen_box(SC1, i, START[i])        # token tới khách ngay khi sinh
    if i < 2:
        mark(SC1 + BH + 20, i, g + CHECK, True)
# câu 3: đã hiện, rồi mới bị chặn
t_bad = GEN[2] + CHECK
steps.append(t_bad)
body.append(svgkit.box(cx(2), SC1, CW, BH, "hatch", "", at(t_bad)).replace(
    "</g>", f'<text class="cap mid halo" x="{cx(2) + CW / 2}" y="{SC1 + 23}">{LABELS[2]}</text></g>'))
mark(SC1 + BH + 20, 2, t_bad, False, "nhưng khách đã đọc")

# ---------- Làn dưới: guarded_text ----------
Y2 = 222
body.append(f'<line class="grid" x1="20" y1="{Y2 - 26}" x2="{W - 30}" y2="{Y2 - 26}"/>')
body.append(f'<text class="row" x="{LX}" y="{Y2}">Giữ tới hết câu, kiểm rồi mới phát</text>'
            f'<text class="sub" x="{LX + 290}" y="{Y2}">guarded_text ở §3.2</text>')
HOLD, SC2 = Y2 + 14, Y2 + 102
body.append(f'<text class="cap" x="{LX}" y="{HOLD + 23}">đang giữ</text>')
body.append(f'<text class="cap" x="{LX}" y="{SC2 + 23}">màn hình</text>')
slots(HOLD)
slots(SC2)
for i, g in enumerate(GEN):
    steps.append(START[i])
    body.append(svgkit.box(cx(i), HOLD, CW, BH, "tok", "", at(START[i])).replace(
        "</g>", f'<text class="cap mid" x="{cx(i) + CW / 2}" y="{HOLD + 23}">{LABELS[i]}</text></g>'))
    if i < 2:
        mark(HOLD + BH + 26, i, g + CHECK, True)
        screen_box(SC2, i, g + CHECK + SEND)
steps.append(t_bad)
body.append(svgkit.box(cx(2), HOLD, CW, BH, "hatch", "", at(t_bad)).replace(
    "</g>", f'<text class="cap mid halo" x="{cx(2) + CW / 2}" y="{HOLD + 23}">{LABELS[2]}</text></g>'))
mark(HOLD + BH + 26, 2, t_bad, False)
t_never = t_bad + SEND
steps.append(t_never)
body.append(f'<g {at(t_never)}><text class="sub mid" x="{cx(2) + CW / 2}" y="{SC2 + 23}">'
            'không bao giờ hiện</text></g>')

# Chú thích cuối
CAP1 = "Trên: câu 3 tới màn hình trước, bị chặn sau, khách đã đọc."
CAP2 = "Dưới: câu 3 bị chặn khi còn giữ; mỗi câu chờ sinh hết và check xong mới hiện."
t_c1, t_c2 = t_never, t_never + 1
steps += [t_c1, t_c2]
body.append(f'<g {at(t_c1)}><text class="cap" x="20" y="{H - 40}">{CAP1}</text></g>')
body.append(f'<g {at(t_c2)}><text class="cap" x="20" y="{H - 18}">{CAP2}</text></g>')
body.append(f'<text class="sub end" x="{W - 30}" y="66">thời gian là giả định</text>')

# Nhịp: các mốc khác nhau cách nhau >= 1 giây (mốc trùng nhau là việc xảy ra cùng lúc ở hai làn),
# trạng thái cuối giữ >= 3 giây, chú thích <= 15 ký tự/giây
ts = sorted(set(steps))
assert all(b - a_ >= 1 for a_, b in zip(ts, ts[1:])), ts
last = max(ts) + TF
assert FIN - last >= 3, (FIN, last)
for cap, t in ((CAP1, t_c1), (CAP2, t_c2)):
    assert len(cap) / (FIN - t) <= 15, (cap, len(cap) / (FIN - t))

svgkit.write(__file__, "stream-buffer.svg", svgkit.svg(
    W, H, "Ba câu, câu thứ ba hứa hoàn tiền. Stream ngay rồi kiểm sau như mặc định stream_first: True của "
    "NeMo: câu thứ ba hiện trên màn hình trước khi bị chặn, khách đã đọc. Giữ tới hết câu, kiểm rồi mới "
    "phát như guarded_text: hai câu đầu hiện sau khi được duyệt, câu thứ ba bị chặn khi còn đang giữ và "
    "không bao giờ hiện; cái giá là mỗi câu chờ model sinh hết câu và check chạy xong mới hiện", body, a))
