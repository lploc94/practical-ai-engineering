"""Kiểm tra link nội bộ (file + anchor) trong docs/*/README.md theo quy tắc slug của GitHub."""
import re, sys, pathlib, unicodedata

ROOT = pathlib.Path(sys.argv[1]).resolve()
targets = [pathlib.Path(p).resolve() for p in sys.argv[2:]] or sorted(ROOT.glob("docs/*/README.md"))

def slug(h):
    h = h.strip().lower()
    h = re.sub(r"<[^>]+>", "", h)
    h = h.replace("`", "")
    out = []
    for ch in h:
        cat = unicodedata.category(ch)
        if ch in "-_ " or cat[0] in "LN" or cat == "Mn":
            out.append(ch)
    return "".join(out).replace(" ", "-")

cache = {}
def anchors(path):
    if path not in cache:
        seen, res = {}, set()
        in_code = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("```"):
                in_code = not in_code
            if in_code:
                continue
            m = re.match(r"^(#{1,6})\s+(.*)$", line)
            if m:
                s = slug(m.group(2))
                n = seen.get(s, 0)
                res.add(s if n == 0 else f"{s}-{n}")
                seen[s] = n + 1
        cache[path] = res
    return cache[path]

bad = 0
for f in targets:
    text = f.read_text(encoding="utf-8")
    text_nocode = re.sub(r"```.*?```", "", text, flags=re.S)
    for m in re.finditer(r"\]\(([^)\s]+)\)", text_nocode):
        link = m.group(1)
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        path_part, _, frag = link.partition("#")
        target = (f.parent / path_part).resolve() if path_part else f
        if target.is_dir():
            target = target / "README.md"
        if not target.exists():
            print(f"{f.relative_to(ROOT)}: missing file {link}"); bad += 1; continue
        if frag and frag not in anchors(target):
            print(f"{f.relative_to(ROOT)}: missing anchor {link}"); bad += 1
print("broken:", bad)
