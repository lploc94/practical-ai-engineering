"""Thư viện nhỏ dùng chung cho các script sinh hình SVG trong docs/.

Gom bảng màu, kiểu chữ và cơ chế animation theo docs/ILLUSTRATIONS.md để mọi
hình trông giống nhau. Script sinh hình import module này:

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
    import svgkit
"""

from __future__ import annotations

from pathlib import Path

FONT = '-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif'
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

# Bảng màu: xem bảng "Bảng màu" trong docs/ILLUSTRATIONS.md
CSS = f"""
svg{{--bg:#ffffff;--fg:#1f2328;--muted:#59636e;--slot:#d1d9e0;--tok:#f6f8fa;
--blue:#0969da;--blue-bg:#ddf4ff;--orange:#bc4c00;--orange-bg:#fff1e5;
--red:#cf222e;--red-bg:#ffebe9;--green:#1a7f37;--green-bg:#dafbe1}}
@media (prefers-color-scheme: dark){{svg{{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;
--slot:#3d444d;--tok:#151b23;--blue:#4493f8;--blue-bg:#0c2d6b;--orange:#db6d28;--orange-bg:#3d1e00;
--red:#f85149;--red-bg:#490202;--green:#3fb950;--green-bg:#033a16}}}}
.bg{{fill:var(--bg)}}
text{{font-family:{FONT};fill:var(--fg)}}
.title{{font-size:20px;font-weight:600}}
.row{{font-size:15px;font-weight:600}}
.sub{{font-size:12px;fill:var(--muted)}}
.cap{{font-size:14px}}
.cap2{{font-size:13px;fill:var(--muted)}}
.lbl{{font-family:{MONO};font-size:14px;text-anchor:middle}}
.mid{{text-anchor:middle}}.end{{text-anchor:end}}
.slot{{fill:none;stroke:var(--slot);stroke-width:1.5;stroke-dasharray:4 4}}
.tok{{fill:var(--tok);stroke:var(--slot);stroke-width:1.5}}
.blue{{fill:var(--blue-bg);stroke:var(--blue);stroke-width:2}}
.orange{{fill:var(--orange-bg);stroke:var(--orange);stroke-width:2}}
.red{{fill:var(--red-bg);stroke:var(--red);stroke-width:2}}
.hatch{{fill:url(#hatch);stroke:var(--red);stroke-width:2}}
.fblue{{fill:var(--blue)}}.forange{{fill:var(--orange)}}.fred{{fill:var(--red)}}.ffg{{fill:var(--fg)}}
.tblue{{fill:var(--blue)}}.torange{{fill:var(--orange)}}.tred{{fill:var(--red)}}
.lblue{{stroke:var(--blue);stroke-width:2;fill:none}}
.lorange{{stroke:var(--orange);stroke-width:2;fill:none}}
.lred{{stroke:var(--red);stroke-width:2;fill:none}}
.lmuted{{stroke:var(--slot);stroke-width:1.5;fill:none}}
.lfg{{stroke:var(--fg);stroke-width:1.5;fill:none}}
.halo{{paint-order:stroke;stroke:var(--bg);stroke-width:4px;stroke-linejoin:round}}
.dash{{stroke-dasharray:6 4}}
.grid{{stroke:var(--slot);stroke-width:1;opacity:.6}}
.axis{{stroke:var(--muted);stroke-width:1}}
.static-only{{opacity:0}}
@media (prefers-reduced-motion: reduce){{
.a-show,.a-hide{{animation:none!important}}.a-show{{opacity:1}}.a-hide{{opacity:0}}.static-only{{opacity:1}}}}
"""

DEFS = (
    "<defs>"
    + "".join(
        f'<marker id="a{c}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path class="f{c}" d="M0,0L10,5L0,10z"/></marker>'
        for c in ("blue", "orange", "red", "fg")
    )
    # Sọc chéo: mã hoá phụ cho trạng thái "phải tính lại", để không chỉ dựa vào màu
    + '<pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" '
    'patternTransform="rotate(45)"><rect width="7" height="7" style="fill:var(--red-bg)"/>'
    '<line x1="0" y1="0" x2="0" y2="7" style="stroke:var(--red);stroke-width:2;opacity:.55"/></pattern>'
    "</defs>"
)


class Anim:
    """Gom @keyframes opacity cho một hình có chu kỳ `dur` giây.

    Mỗi phần tử có keyframes trải hết chu kỳ (animation-delay chỉ áp dụng ở lần
    chạy đầu nên không dùng được cho vòng lặp). Từ `fade_out`% đến 100% mọi
    phần tử mờ dần để vòng sau bắt đầu sạch.

    Class a-show / a-hide quyết định phần tử hiện hay ẩn trong khung tĩnh khi
    người dùng bật prefers-reduced-motion. Mặc định lấy theo opacity cuối.
    """

    def __init__(self, dur: float, fade_out: float = 95):
        self.dur = dur
        self.fade_out = fade_out
        self.frames: list[str] = []
        self._n = 0

    def __call__(self, points: list[tuple[float, float]], static: bool | None = None) -> str:
        self._n += 1
        name = f"k{self._n}"
        pts = [(0.0, points[0][1])] + points + [(self.fade_out, points[-1][1]), (100.0, 0.0)]
        merged: dict[float, float] = {}
        for p, v in pts:
            merged[round(p, 2)] = v
        body = " ".join(f"{p}%{{opacity:{v}}}" for p, v in sorted(merged.items()))
        self.frames.append(f"@keyframes {name}{{{body}}}")
        shown = bool(points[-1][1]) if static is None else static
        cls = "a-show" if shown else "a-hide"
        return f'class="{cls}" style="animation:{name} {self.dur}s linear infinite"'

    def css(self) -> str:
        return "".join(self.frames)


def svg(width: int, height: int, label: str, body: list[str], anim: Anim | None = None) -> str:
    extra = anim.css() if anim else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{label}">'
        f"<style>{CSS}{extra}</style>{DEFS}"
        f'<rect class="bg" width="{width}" height="{height}" rx="12"/>'
        + "".join(body)
        + "</svg>\n"
    )


def box(x: float, y: float, w: float, h: float, cls: str, label: str = "", attrs: str = "") -> str:
    text = (
        f'<text class="lbl" x="{x + w / 2}" y="{y + h / 2 + 5}">{label}</text>' if label else ""
    )
    return f'<g {attrs}><rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="7"/>{text}</g>'


def legend(x: float, y: float, items: list[tuple[str, str]]) -> str:
    """Chú giải: danh sách (class của ô màu, nhãn). Trả về chuỗi SVG."""
    out = []
    for cls, text in items:
        out.append(f'<rect class="{cls}" x="{x}" y="{y - 11}" width="14" height="14" rx="3"/>'
                   f'<text class="sub" x="{x + 20}" y="{y}">{text}</text>')
        x += 20 + 7 * len(text) + 24
    return "".join(out)


def write(script_file: str, name: str, content: str) -> None:
    out = Path(script_file).with_name(name)
    out.write_text(content, encoding="utf-8")
    print(f"wrote {out} ({len(content)} bytes)")
