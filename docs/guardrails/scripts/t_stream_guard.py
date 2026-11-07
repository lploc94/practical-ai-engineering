import asyncio
import json

import anthropic
import httpx2

from stream_guard import guarded_text

MSG = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5", "content": [],
       "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}


def events(texts, stop="end_turn", details=None):
    ev = [{"type": "message_start", "message": MSG},
          {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}]
    ev += [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}} for t in texts]
    ev += [{"type": "content_block_stop", "index": 0},
           {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None, "stop_details": details},
            "usage": {"output_tokens": len(texts)}},
           {"type": "message_stop"}]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in ev).encode()


def client(body):
    def handler(request):
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    return anthropic.AsyncAnthropic(api_key="k", max_retries=0,
                                    http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))


async def run(body, check, **kw):
    c = client(body)
    stream = await c.messages.create(model="claude-haiku-5-5", max_tokens=100,
                                     messages=[{"role": "user", "content": "hi"}], stream=True)
    async with stream:
        return [o async for o in guarded_text(stream, check, **kw)]


def checker(bad_word, seen):
    async def check(window):
        seen.append(window)
        return "cam_ket_hoan_tien" if bad_word in window else None
    return check


async def main():
    s1 = "Đơn hàng của anh đang được giao. "
    s2 = "Dự kiến tới nơi vào thứ Sáu. "
    s3 = "Shop cam kết hoàn tiền 100% nếu trễ. "

    # 1. câu sạch: phát theo từng câu đủ dài, không phát token lẻ
    seen = []
    out = await run(events([s1[:10], s1[10:], s2]), checker("hoàn tiền", seen), min_chars=20)
    assert [o.kind for o in out] == ["text", "text"] and "".join(o.text for o in out) == s1 + s2, out
    assert len(seen) == 2

    # 2. câu thứ ba vi phạm: hai câu đầu đã phát, câu ba không phát, dừng với lý do
    seen = []
    out = await run(events([s1, s2, s3, "Câu sau không được đọc tới. "]), checker("hoàn tiền", seen), min_chars=20)
    assert [o.kind for o in out] == ["text", "text", "blocked"] and out[-1].reason == "cam_ket_hoan_tien", out
    assert all("hoàn tiền" not in o.text for o in out) and len(seen) == 3

    # 3. vi phạm bị chia đôi qua hai delta ("hoàn" | " tiền"): vẫn bị bắt vì check chạy trên cả câu
    out = await run(events(["Shop cam kết hoàn", " tiền nếu trễ. "]), checker("hoàn tiền", []), min_chars=20)
    assert [o.kind for o in out] == ["blocked"], out

    # 4. window có ngữ cảnh: câu trước đã phát nằm trong window của câu sau
    seen = []
    await run(events([s1, s2]), checker("zzz", seen), min_chars=20, context_chars=300)
    assert seen[1] == s1 + s2, seen

    # 5. đoạn dài không có dấu câu (code, bảng): bị cắt đúng ở max_chars, không giữ mãi
    seen = []
    out = await run(events(["x" * 50] * 5), checker("zzz", seen), min_chars=20, max_chars=120)
    assert [len(o.text) for o in out] == [120, 120, 10] and len(seen) == 3, [len(o.text) for o in out]
    #    một delta lớn hơn max_chars cũng bị chia, không đoạn nào vượt trần
    out = await run(events(["y" * 700]), checker("zzz", []), min_chars=20, max_chars=300)
    assert [len(o.text) for o in out] == [300, 300, 100], [len(o.text) for o in out]
    #    có ranh giới câu trong 120 ký tự đầu thì cắt ở đó thay vì giữa chữ
    out = await run(events(["Câu ngắn đầu tiên. " + "z" * 200]), checker("zzz", []), min_chars=10, max_chars=120)
    assert out[0].text == "Câu ngắn đầu tiên. " and all(len(o.text) <= 120 for o in out), [o.text for o in out]

    # 6. phần cuối không có dấu câu vẫn được kiểm rồi mới phát
    seen = []
    out = await run(events(["Cảm ơn anh"]), checker("zzz", seen), min_chars=20)
    assert [o.text for o in out] == ["Cảm ơn anh"] and seen == ["Cảm ơn anh"]

    # 7. model từ chối giữa stream: phát "refusal" kèm category để UI gỡ phần đã hiện
    details = {"type": "refusal", "category": "cyber", "explanation": "declined"}
    out = await run(events([s1, "phần dở"], stop="refusal", details=details), checker("zzz", []), min_chars=20)
    assert [o.kind for o in out] == ["text", "refusal"] and out[-1].reason == "cyber", out

    # 8. refusal trước mọi output, category null: vẫn phát "refusal", reason None
    details = {"type": "refusal", "category": None, "explanation": None}
    out = await run(events([], stop="refusal", details=details), checker("zzz", []), min_chars=20)
    assert [(o.kind, o.reason) for o in out] == [("refusal", None)], out

    # 9. server-side fallback giữa stream: block `fallback` đánh dấu chỗ chuyển model, model sau viết tiếp
    #    trên cùng stream, không có refusal; chữ của cả hai model đều được kiểm rồi phát
    ev = [{"type": "message_start", "message": MSG},
          {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
          {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": s1}},
          {"type": "content_block_stop", "index": 0},
          {"type": "content_block_start", "index": 1, "content_block": {
              "type": "fallback", "from": {"model": "claude-haiku-5-5"}, "to": {"model": "claude-opus-5-5"},
              "trigger": {"type": "refusal", "category": "cyber"}}},
          {"type": "content_block_stop", "index": 1},
          {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
          {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": s2}},
          {"type": "content_block_stop", "index": 2},
          {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None, "stop_details": None},
           "usage": {"output_tokens": 2}},
          {"type": "message_stop"}]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in ev).encode()
    out = await run(body, checker("zzz", []), min_chars=20)
    assert [o.kind for o in out] == ["text", "text"] and "".join(o.text for o in out) == s1 + s2, out
    # 10. ranh giới câu nằm giữa hai delta ("giao." | " Dự kiến"): token thường mang khoảng trắng ở đầu
    out = await run(events(["Đơn hàng của anh đang được giao.", " Dự kiến tới nơi", " vào thứ Sáu."]),
                    checker("zzz", []), min_chars=20)
    assert [o.text for o in out] == ["Đơn hàng của anh đang được giao. ", "Dự kiến tới nơi vào thứ Sáu."], out

    # 11. không phải hết câu: dấu hai chấm, chấm phẩy, số thập phân
    for text in ["Điều kiện đổi trả: còn tem; còn hoá đơn", "Phiên bản 3.5 có bảo hành 24 tháng"]:
        seen = []
        out = await run(events([text[:15], text[15:]]), checker("zzz", seen), min_chars=10)
        assert [o.text for o in out] == [text] and len(seen) == 1, (text, out)

    # 12. dấu kết câu kèm nháy hoặc ngoặc đóng vẫn là ranh giới
    out = await run(events(['Anh ấy nói "Đồng ý." ', "Câu sau."]), checker("zzz", []), min_chars=10)
    assert [o.text for o in out] == ['Anh ấy nói "Đồng ý." ', "Câu sau."], out

    # 13. tham số sai bị từ chối ngay, không treo; context_chars=0 nghĩa là không kèm chữ đã phát
    for kw in [{"max_chars": 0}, {"min_chars": 50, "max_chars": 40}, {"context_chars": -1},
               {"min_chars": 1, "max_chars": 1.5}, {"min_chars": 1.2, "max_chars": 2}, {"context_chars": 0.5},
               {"min_chars": True, "max_chars": 2}, {"max_chars": float("inf")}]:
        try:
            await run(events(["Xin chào."]), checker("zzz", []), **kw)
        except (ValueError, TypeError):
            continue
        raise AssertionError(kw)
    seen = []
    await run(events(["Câu một đủ dài. ", "Câu hai."]), checker("zzz", seen), min_chars=10, context_chars=0)
    assert seen == ["Câu một đủ dài. ", "Câu hai."], seen
    print("STREAM GUARD OK")


asyncio.run(main())
