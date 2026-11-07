from dataclasses import dataclass
from decimal import Decimal

M = Decimal(1_000_000)


@dataclass(frozen=True)
class Price:
    """Giá USD cho 1 triệu token. Hệ số cache theo bảng giá Claude, tháng 11/2026."""
    input: Decimal
    output: Decimal
    cache_read: Decimal                      # giá đọc cache, khác nhau theo model
    write_5m: Decimal = Decimal("1.25")      # hệ số so với input
    write_1h: Decimal = Decimal("2")
    long_threshold: int | None = None        # Haiku 5.5: prompt > 100.000 token
    long: "Price | None" = None              # bảng giá khi vượt ngưỡng


PRICES = {
    "claude-opus-5-5":   Price(Decimal("4"),   Decimal("20"),  Decimal("0.20")),
    "claude-sonnet-5-5": Price(Decimal("2"),   Decimal("10"),  Decimal("0.10")),
    "claude-fable-5-1":  Price(Decimal("10"),  Decimal("50"),  Decimal("0.25")),
    "claude-opus-4-8":   Price(Decimal("5"),   Decimal("25"),  Decimal("0.50")),
    "claude-haiku-5-5":  Price(Decimal("0.10"), Decimal("0.50"), Decimal("0.01"),
                               long_threshold=100_000,
                               long=Price(Decimal("0.50"), Decimal("2.50"), Decimal("0.05"))),
}
FAST = {"claude-opus-5-5": (Decimal("8"), Decimal("40")), "claude-opus-4-8": (Decimal("10"), Decimal("50"))}   # input, output khi speed="fast"


def _get(obj, name, default=0):
    v = obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    return default if v is None else v


def _entry_cost(u, model: str, *, batch: bool, fast: bool) -> Decimal:
    p = PRICES[model]
    inp = _get(u, "input_tokens")
    read = _get(u, "cache_read_input_tokens")
    cc = _get(u, "cache_creation", None)
    if cc is not None:
        w5, w1 = _get(cc, "ephemeral_5m_input_tokens"), _get(cc, "ephemeral_1h_input_tokens")
    else:                                    # không có breakdown: coi là TTL 5 phút
        w5, w1 = _get(u, "cache_creation_input_tokens"), 0
    if p.long_threshold and inp + read + w5 + w1 > p.long_threshold:
        p = p.long                           # cả request tính giá cao, kể cả output
    base_in, base_out = p.input, p.output
    if fast:
        base_in, base_out = FAST[model]      # hệ số cache áp lên giá fast mode
        read_price = base_in * p.cache_read / p.input
    else:
        read_price = p.cache_read
    cost = (inp * base_in + w5 * base_in * p.write_5m + w1 * base_in * p.write_1h
            + read * read_price + _get(u, "output_tokens") * base_out) / M
    return cost / 2 if batch else cost       # Batch API: giảm 50% mọi loại token


def request_cost(usage, model: str, *, batch: bool = False) -> Decimal:
    """Chi phí token của một response. Có `iterations` thì cộng từng mục: compaction,
    advisor và các lần thử bị refusal chỉ nằm ở đó, không nằm trong số tổng."""
    fast = _get(usage, "speed", None) == "fast"
    iterations = _get(usage, "iterations", None)
    if not iterations:
        return _entry_cost(usage, model, batch=batch, fast=fast)
    return sum((_entry_cost(it, _get(it, "model", None) or model, batch=batch, fast=fast)
                for it in iterations), Decimal(0))
