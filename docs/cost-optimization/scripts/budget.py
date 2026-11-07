import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal

from cost import M, PRICES


class BudgetExceeded(Exception):
    pass


class SpendLedger:
    """Hạn mức chi tiêu theo tenant theo ngày (UTC). Giữ chỗ chi phí tối đa trước khi gọi,
    trừ chi phí thật sau khi gọi. Bản trong bộ nhớ, một process; nhiều worker thì
    thay dict bằng Redis hoặc bảng DB với cập nhật nguyên tử."""

    def __init__(self, daily_limit_usd: dict[str, Decimal], clock=lambda: datetime.now(timezone.utc)):
        self.limits, self.clock = daily_limit_usd, clock
        self.spent: dict[tuple[str, str], Decimal] = {}
        self.reserved: dict[tuple[str, str], Decimal] = {}
        self.day: str | None = None
        self.lock = threading.Lock()

    @contextmanager
    def hold(self, tenant: str, max_cost: Decimal):
        with self.lock:
            day = self.clock().strftime("%Y-%m-%d")
            if day != self.day:                  # sang ngày mới: bỏ số của các ngày trước
                self.day = day
                self.spent = {k: v for k, v in self.spent.items() if k[1] >= day}
            key = (tenant, day)
            used = self.spent.get(key, Decimal(0)) + self.reserved.get(key, Decimal(0))
            if used + max_cost > self.limits[tenant]:
                raise BudgetExceeded(f"{tenant}: đã dùng {used}, cần {max_cost}, hạn mức {self.limits[tenant]}")
            self.reserved[key] = self.reserved.get(key, Decimal(0)) + max_cost
        actual = {"usd": max_cost}          # lỗi không rõ chi phí: tính như đã tiêu hết phần giữ chỗ
        try:
            yield actual
        finally:
            with self.lock:
                self.reserved[key] -= max_cost
                if not self.reserved[key]:
                    del self.reserved[key]
                if key[1] == self.day:           # request vắt qua nửa đêm: ngày cũ đã đóng, không ghi lại
                    self.spent[key] = self.spent.get(key, Decimal(0)) + actual["usd"]


def max_request_cost(model: str, input_tokens: int, max_tokens: int,
                     write_multiplier: Decimal = Decimal("1.25")) -> Decimal:
    """Trần chi phí của một request chưa gửi: mọi input token bị ghi cache (1.25 với TTL 5 phút,
    2 với TTL 1 giờ, 1 nếu không dùng cache), output dùng hết max_tokens. Không bao được
    request có server tool chạy nhiều vòng hoặc compaction: các vòng đó tính tiền riêng."""
    p = PRICES[model]
    if p.long_threshold and input_tokens > p.long_threshold:   # ngưỡng tính trên prompt
        p = p.long
    return (input_tokens * p.input * write_multiplier + max_tokens * p.output) / M
