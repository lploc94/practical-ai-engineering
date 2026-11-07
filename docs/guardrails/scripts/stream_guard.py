import re
from dataclasses import dataclass

# ranh giới câu: dấu kết câu (có thể kèm nháy hoặc ngoặc đóng) rồi khoảng trắng, hoặc xuống dòng.
# Đòi khoảng trắng phía sau nên "3.5" không bị coi là hết câu; "TP. HCM" thì có, chỉ tốn thêm một lần check.
BOUNDARY = re.compile(r"[.!?…][\"'”’)\]]*\s+|\n\s*")


@dataclass
class Out:
    kind: str                # "text": phát cho người dùng; "blocked": guardrail chặn; "refusal": model từ chối
    text: str = ""
    reason: str | None = None


async def guarded_text(events, check, *, min_chars=80, max_chars=600, context_chars=300):
    """Đọc raw event của `client.messages.create(stream=True)` (AsyncAnthropic). Chữ chỉ được
    phát sau khi `check` đã duyệt, theo từng đoạn kết thúc ở ranh giới câu (ít nhất `min_chars`
    ký tự). Không đoạn nào dài quá `max_chars`: không có ranh giới câu thì cắt đúng ở đó.
    `check(window)` là coroutine trả lý do chặn hoặc None; `window` gồm `context_chars` ký tự
    cuối đã phát cộng đoạn đang giữ, để classifier thấy câu trước. Bị chặn thì dừng, đoạn bị chặn không được phát.
    Model từ chối (`stop_reason: "refusal"`) thì phát Out("refusal"): mọi chữ đã phát trong
    lượt này phải được gỡ khỏi màn hình, vì output dở dang của một refusal phải bỏ đi."""
    if any(type(v) is not int for v in (min_chars, max_chars, context_chars)):
        raise TypeError("min_chars, max_chars, context_chars phải là int")
    if not 0 <= min_chars <= max_chars or max_chars < 1 or context_chars < 0:
        raise ValueError("cần 0 <= min_chars <= max_chars, max_chars >= 1, context_chars >= 0")
    released, pending = "", ""

    async def flush(n):
        nonlocal released, pending
        chunk = pending[:n]
        reason = await check(released[len(released) - context_chars:] + chunk)
        if reason:
            return Out("blocked", reason=reason)
        released, pending = released + chunk, pending[n:]
        return Out("text", chunk)

    def cut():
        """Vị trí cắt: ranh giới câu cuối cùng sau ít nhất min_chars ký tự, nhưng không quá max_chars."""
        if len(pending) >= max_chars:
            ends = [m.end() for m in BOUNDARY.finditer(pending, 0, max_chars) if m.end() >= min_chars]
            return ends[-1] if ends else max_chars
        ends = [m.end() for m in BOUNDARY.finditer(pending) if m.end() >= min_chars]
        return ends[-1] if ends else None

    async for ev in events:
        if ev.type == "content_block_delta" and ev.delta.type == "text_delta":
            pending += ev.delta.text
            while (n := cut()) is not None:
                out = await flush(n)
                yield out
                if out.kind == "blocked":
                    return
        elif ev.type == "message_delta" and ev.delta.stop_reason == "refusal":
            details = ev.delta.stop_details
            yield Out("refusal", reason=details.category if details else None)
            return
    if pending:
        yield await flush(len(pending))
