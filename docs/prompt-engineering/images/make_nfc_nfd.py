"""Sinh nfc-nfd.svg: SVG tĩnh minh hoạ chữ "ệ" ở dạng NFC và NFD (§3.6).

Hai dạng hiển thị giống hệt nhau nhưng khác code point và khác byte UTF-8:
- NFC: 1 code point U+1EC7, 3 byte E1 BB 87.
- NFD: 3 code point U+0065 (e), U+0323 (dấu nặng), U+0302 (dấu mũ), 5 byte
  65 CC A3 CC 82. Thứ tự dấu theo combining class chuẩn (220 trước 230).

Mọi giá trị được tính bằng unicodedata và kiểm tra bằng assert.

Chạy: python3 make_nfc_nfd.py
"""

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

CH = "ệ"  # ệ
NFC = unicodedata.normalize("NFC", CH)
NFD = unicodedata.normalize("NFD", CH)
NAMES = {0x65: "e", 0x323: "dấu nặng", 0x302: "dấu mũ"}

assert [ord(c) for c in NFC] == [0x1EC7]
assert [ord(c) for c in NFD] == [0x65, 0x323, 0x302]
assert NFC.encode() == bytes.fromhex("e1bb87") and NFD.encode() == bytes.fromhex("65cca3cc82")
assert NFC != NFD and unicodedata.normalize("NFC", NFD) == NFC


def hexbytes(c: str) -> str:
    return " ".join(f"{b:02X}" for b in c.encode())


W, H, BH = 860, 344, 52
R1, R2 = 128, 226
XG, XC, XB = 150, 250, 610   # cột chữ hiển thị, cột code point, cột byte
CPW, CPG = 110, 8            # độ rộng ô code point và khe

body: list[str] = []
body.append('<text class="title" x="20" y="38">Cùng một chữ "ệ", hai chuỗi byte khác nhau</text>')
body.append('<text class="cap2" x="20" y="62">Trên màn hình giống hệt; với tokenizer, prompt cache và phép so chuỗi '
            'thì đây là hai chuỗi khác nhau.</text>')
body.append(svgkit.legend(20, 92, [("blue", "NFC: dạng nên chuẩn hoá về"),
                                   ("hatch", "NFD: khác byte, làm vỡ cache và so khớp")]))

for x, t in ((XG, "hiển thị"), (XC, "code point"), (XB, "byte UTF-8")):
    body.append(f'<text class="sub" x="{x}" y="{R1 - 10}">{t}</text>')


def glyph(y: float) -> str:
    return (f'<rect class="tok" x="{XG}" y="{y}" width="{BH + 20}" height="{BH}" rx="7"/>'
            f'<text class="mid" style="font-size:32px" x="{XG + (BH + 20) / 2}" y="{y + 37}">{CH}</text>')


def cp_box(x: float, y: float, w: float, cls: str, c: str) -> str:
    code = f"U+{ord(c):04X}"
    name = NAMES.get(ord(c), "ệ")
    return (f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{BH}" rx="7"/>'
            f'<text class="lbl halo" x="{x + w / 2}" y="{y + 22}">{code}</text>'
            f'<text class="sub mid halo" x="{x + w / 2}" y="{y + 41}">{name}</text>')


def byte_box(y: float, cls: str, s: str) -> str:
    n = len(s.encode())
    w = 30 + 28 * n
    return (f'<rect class="{cls}" x="{XB}" y="{y}" width="{w}" height="{BH}" rx="7"/>'
            f'<text class="lbl halo" x="{XB + w / 2}" y="{y + 31}">{hexbytes(s)}</text>'
            f'<text class="row" x="{XB + w + 14}" y="{y + 31}">{n} byte</text>')


# Hàng NFC
body.append(f'<text class="row" x="20" y="{R1 + 23}">NFC</text>')
body.append(f'<text class="sub" x="20" y="{R1 + 41}">dựng sẵn</text>')
body.append(glyph(R1))
body.append(cp_box(XC, R1, CPW, "blue", NFC))
body.append(byte_box(R1, "blue", NFC))

# Hàng NFD
body.append(f'<text class="row" x="20" y="{R2 + 23}">NFD</text>')
body.append(f'<text class="sub" x="20" y="{R2 + 41}">tách dấu</text>')
body.append(glyph(R2))
for i, c in enumerate(NFD):
    body.append(cp_box(XC + i * (CPW + CPG), R2, CPW, "hatch", c))
assert XC + len(NFD) * (CPW + CPG) < XB
body.append(byte_box(R2, "hatch", NFD))

body.append(f'<text class="cap" x="20" y="{H - 30}">Chuẩn hoá mọi văn bản về NFC ở ranh giới hệ thống: '
            'unicodedata.normalize("NFC", s).</text>')

svgkit.write(__file__, "nfc-nfd.svg", svgkit.svg(
    W, H, "Chữ ệ ở dạng NFC là 1 code point, 3 byte; ở dạng NFD là e cộng dấu nặng cộng dấu mũ, 3 code point, "
    "5 byte. Hai dạng hiển thị giống hệt nhưng khác byte nên làm vỡ prompt cache và so khớp chuỗi",
    body))
