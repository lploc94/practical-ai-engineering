import json, os, tempfile, unicodedata
from ft_data import load_jsonl, prepare, problems, prompt_key, split_of


def ex(user, answer, system=None, **extra):
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    return {"messages": msgs, **extra}


# 1. ví dụ hợp lệ
assert problems(ex("Huỷ đơn DH-1", '{"intent":"cancel"}', system="Phân loại")) == []
# 2. các lỗi cấu trúc thường gặp
assert problems({}) == ["thiếu messages"]
assert "lượt cuối không phải assistant" in problems({"messages": [{"role": "user", "content": "a"}]})
assert "lượt đầu không phải user" in problems({"messages": [{"role": "system", "content": "s"}]})
assert "messages[1]: content rỗng" in problems({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "  "}]})
assert "messages[0]: role không hợp lệ" in problems({"messages": [{"role": "bot", "content": "a"}, {"role": "assistant", "content": "b"}]})
assert "hai lượt liên tiếp cùng role" in problems({"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}]})
assert "messages[2]: system không ở đầu" in problems({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "system", "content": "s"}, {"role": "user", "content": "d"}, {"role": "assistant", "content": "e"}]})

# 3. JSONL hỏng được gom theo dòng, dòng trống bỏ qua
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
    f.write(json.dumps(ex("a", "b")) + "\n\n{hỏng\n" + json.dumps(ex("c", "d")) + "\n")
recs, errs = load_jsonl(f.name); os.unlink(f.name)
assert [n for n, _ in recs] == [1, 4] and errs[0][0] == 3, (recs, errs)

# 4. NFC/NFD, hoa thường, khoảng trắng: cùng dấu vân tay
assert prompt_key(ex(unicodedata.normalize("NFD", "HUỶ   đơn"), "x")) == prompt_key(ex("huỷ đơn", "y"))

# 5. chia theo nhóm: ổn định, cùng nhóm cùng tập, tỉ lệ gần đúng
assert split_of("khach-42") == split_of("khach-42")
parts = [split_of(f"g{i}") for i in range(10000)]
assert 800 < parts.count("test") < 1200 and 800 < parts.count("val") < 1200, (parts.count("test"), parts.count("val"))

# 6. prepare: trùng, mâu thuẫn, lỗi, quá dài; nhóm không bị tách
rows = [
    (1, ex("Huỷ đơn DH-1", '{"intent":"cancel"}', customer="c1")),
    (2, ex("huỷ đơn  dh-1", '{"intent":"cancel"}', customer="c9")),     # trùng sau chuẩn hoá, nhóm khác
    (3, ex("Huỷ đơn DH-1", '{"intent":"refund"}', customer="c2")),      # cùng đầu vào, đáp án khác
    (4, {"messages": []}),
    (5, ex("rất dài " * 50, '{"intent":"other"}', customer="c3")),
] + [(100 + i, ex(f"Đơn DH-{i} giao chưa?", '{"intent":"track"}', customer=f"c{i % 20}")) for i in range(60)]
splits, rep = prepare(rows, group_of=lambda r: r["customer"],
                      count_tokens=lambda r: sum(len(m["content"].split()) for m in r["messages"]), max_tokens=40)
assert rep["duplicate"] == [(1, 2)] and rep["conflict"] == [(1, 3)], rep
assert rep["invalid"][0][0] == 4 and rep["too_long"] == [5], rep
assert sum(len(v) for v in splits.values()) == 61 and all(splits.values()), {k: len(v) for k, v in splits.items()}
where = {}
for name, recs in splits.items():
    for r in recs:
        where.setdefault(r["customer"], set()).add(name)
assert all(len(s) == 1 for s in where.values()), where             # mỗi khách chỉ ở một tập
keys = {name: {prompt_key(r) for r in recs} for name, recs in splits.items()}
assert not (keys["train"] & keys["test"]) and not (keys["train"] & keys["val"])

# 6b. đáp án chỉ khác hoa thường là mâu thuẫn, không phải trùng; ngữ cảnh nhiều lượt khác nhau là hai ví dụ
_, rep = prepare([(1, ex("Huỷ đơn DH-9", '{"order_id":"DH-9"}')), (2, ex("Huỷ đơn DH-9", '{"order_id":"dh-9"}'))],
                 group_of=lambda r: "g")
assert rep["conflict"] == [(1, 2)] and rep["duplicate"] == [], rep
def turns(mid):
    return {"messages": [{"role": "user", "content": "Đơn DH-1?"}, {"role": "assistant", "content": mid},
                         {"role": "user", "content": "Vậy giờ sao?"}, {"role": "assistant", "content": "Tuỳ: " + mid}]}
s2, rep = prepare([(1, turns("Đang giao.")), (2, turns("Đã giao hôm qua."))], group_of=lambda r: "g")
assert rep["conflict"] == [] and rep["duplicate"] == [] and sum(len(v) for v in s2.values()) == 2, rep

# 7. thêm dữ liệu mới không làm ví dụ cũ đổi tập
more = rows + [(500 + i, ex(f"Bảo hành máy {i}", '{"intent":"warranty"}', customer=f"new{i}")) for i in range(30)]
splits2, _ = prepare(more, group_of=lambda r: r["customer"])
old_test = {prompt_key(r) for r in splits["test"]}
assert old_test <= {prompt_key(r) for r in splits2["test"]}
print("FT_DATA OK")
