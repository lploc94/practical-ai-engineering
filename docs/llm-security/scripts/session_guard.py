import inspect
import json
import logging
import re
from dataclasses import dataclass
from enum import Flag, auto

log = logging.getLogger(__name__)


class Cap(Flag):
    """Ba thuộc tính trong Agents Rule of Two của Meta."""
    UNTRUSTED = auto()   # A: đưa nội dung không tin cậy vào context (web, email, file người ngoài gửi)
    PRIVATE = auto()     # B: đọc dữ liệu riêng tư hoặc hệ thống nhạy cảm
    OUTWARD = auto()     # C: đổi trạng thái hoặc gửi dữ liệu ra ngoài


@dataclass(frozen=True)
class Tool:
    name: str
    caps: Cap
    fn: object           # fn(ctx, **args): ctx do ứng dụng cấp, model không điền được


class Denied(Exception):
    pass


# Ký tự không hiện ra nhưng model vẫn đọc: Unicode tag, variation selector, zero-width,
# điều khiển hướng chữ (bidi).
HIDDEN = re.compile("[\U000E0000-\U000E007F\U000E0100-\U000E01EF\uFE00-\uFE0F"
                    "\u200B-\u200D\u2060\u202A-\u202E\u2066-\u2069]")


def strip_hidden(text: str) -> str:
    """Xoá ký tự ẩn khỏi nội dung trước khi đưa cho model. Emoji ghép (trái tim đỏ, gia đình)
    mất variation selector và zero-width joiner nên bị tách: dùng cho input của model, không cho
    văn bản hiển thị cho người dùng."""
    return HIDDEN.sub("", text)


def visible(text: str) -> str:
    """Hiện ký tự ẩn và ký tự điều khiển dạng \\u{...}; chữ có dấu giữ nguyên."""
    return "".join(f"\\u{{{ord(c):x}}}" if HIDDEN.match(c) or not c.isprintable() else c for c in text)


class Session:
    """Theo dõi những gì đã vào context của một phiên agent. Lời gọi có OUTWARD trong một phiên
    đã (hoặc sẽ, ngay trong lời gọi này) có cả UNTRUSTED và PRIVATE phải được người duyệt,
    từng lần một. `seen` là những gì đã có sẵn trong context khi mở phiên (system prompt,
    tin nhắn của người dùng, tài liệu đính kèm). Thuộc tính không bao giờ được gỡ khỏi phiên:
    muốn sạch thì mở phiên mới."""

    def __init__(self, tools: list[Tool], ctx, approve, seen: Cap = Cap(0)):
        self.tools = {t.name: t for t in tools}
        self.ctx, self.approve = ctx, approve
        self.seen = seen

    def needs_approval(self, tool: Tool) -> bool:
        after = self.seen | tool.caps
        return Cap.OUTWARD in tool.caps and Cap.UNTRUSTED in after and Cap.PRIVATE in after

    def call(self, name: str, args: dict):
        tool = self.tools.get(name)
        if tool is None:
            raise Denied(f"tool không có trong danh sách: {name!r}")
        try:
            inspect.signature(tool.fn).bind(self.ctx, **args)    # model điền thừa tham số (kể cả "ctx") là lỗi
        except TypeError as e:
            raise Denied(f"tham số không hợp lệ cho {name}: {e}") from None
        if self.needs_approval(tool):
            # người duyệt phải thấy đúng thứ sẽ chạy, kể cả ký tự ẩn
            shown = visible(json.dumps(args, ensure_ascii=False, sort_keys=True))
            if not self.approve(name, shown):
                raise Denied(f"người dùng không duyệt {name}")
        self.seen |= tool.caps          # trước khi chạy: kết quả lỗi cũng có thể mang nội dung vào context
        return tool.fn(self.ctx, **args)


def run_tool_uses(session: Session, blocks) -> list[dict]:
    """Chạy các block `tool_use` trong một response của Claude, trả các block `tool_result`.
    Thuộc tính của mọi tool trong lượt được tính trước khi chạy tool đầu tiên, nên thứ tự
    các block trong lượt không làm lọt bước duyệt."""
    for b in blocks:
        if b.name in session.tools:
            session.seen |= session.tools[b.name].caps & ~Cap.OUTWARD
    results = []
    for b in blocks:
        try:
            out = strip_hidden(json.dumps(session.call(b.name, dict(b.input)), ensure_ascii=False))
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        except Denied as e:
            results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True, "content": str(e)})
        except Exception as e:
            # lỗi của chính tool (backend, timeout...): block nào cũng phải có tool_result. Thông điệp
            # lỗi có thể chứa secret, nên không vào context và log chỉ ghi loại lỗi.
            # Tool có thể đã có tác dụng trước khi lỗi.
            log.error("tool %s lỗi: %s", b.name, type(e).__name__)
            err = {"code": "TOOL_FAILED", "retryable": False,
                   "message": f"{b.name} lỗi khi chạy, chưa rõ đã có tác dụng hay chưa. "
                              "Kiểm tra trạng thái trước khi gọi lại."}
            results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                            "content": json.dumps({"error": err}, ensure_ascii=False)})
    return results
