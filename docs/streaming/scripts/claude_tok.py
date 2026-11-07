import math
def claude_image_tokens(w: int, h: int, high_res: bool) -> int:
    """Ước lượng visual token của Claude theo công thức patch 28px (docs 11/2026)."""
    max_edge, max_tokens = (2576, 4784) if high_res else (1568, 1568)
    def tokens(s):
        return math.ceil(math.floor(w * s) / 28) * math.ceil(math.floor(h * s) / 28)
    s = min(1.0, max_edge / max(w, h))
    if tokens(s) > max_tokens:
        lo, hi = 0.0, s
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if tokens(mid) <= max_tokens else (lo, mid)
        s = lo
    return tokens(s)
for (w,h,hr,exp) in [(1000,1000,False,1296),(1000,1000,True,1296),(1920,1080,False,1560),(1920,1080,True,2691),(2000,1500,False,1564),(2000,1500,True,3888),(3840,2160,False,1560),(3840,2160,True,4784)]:
    print(w,h,hr,claude_image_tokens(w,h,hr),exp)
