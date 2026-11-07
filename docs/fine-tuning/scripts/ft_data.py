import hashlib
import json
import re
import unicodedata

ROLES = {"system", "user", "assistant"}


def load_jsonl(path):
    """Đọc JSONL, trả (records, errors). Lỗi được gom theo số dòng thay vì dừng ở lỗi đầu tiên."""
    records, errors = [], []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append((n, json.loads(line)))
            except json.JSONDecodeError as e:
                errors.append((n, f"JSON hỏng: {e.msg}"))
    return records, errors


def problems(rec) -> list[str]:
    """Các lỗi cấu trúc của một ví dụ dạng {"messages": [...]}; rỗng nghĩa là hợp lệ."""
    msgs = rec.get("messages") if isinstance(rec, dict) else None
    if not isinstance(msgs, list) or not msgs:
        return ["thiếu messages"]
    out = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            out.append(f"messages[{i}]: role không hợp lệ")
            continue
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            out.append(f"messages[{i}]: content rỗng")
        if m["role"] == "system" and i != 0:
            out.append(f"messages[{i}]: system không ở đầu")
    turns = [m.get("role") for m in msgs if isinstance(m, dict) and m.get("role") != "system"]
    if not turns or turns[0] != "user":
        out.append("lượt đầu không phải user")
    if not turns or turns[-1] != "assistant":
        out.append("lượt cuối không phải assistant")
    if any(a == b for a, b in zip(turns, turns[1:])):
        out.append("hai lượt liên tiếp cùng role")
    return out


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def prompt_key(rec) -> str:
    """Dấu vân tay của mọi lượt trước câu trả lời cuối, kể cả lượt assistant ở giữa (ngữ cảnh
    khác thì là ví dụ khác). Dùng để tìm trùng và rò rỉ giữa các tập."""
    parts = [m["role"] + ":" + norm(m["content"]) for m in rec["messages"][:-1]]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def split_of(group: str, test=0.1, val=0.1) -> str:
    """Chia theo hash của nhóm: cùng nhóm luôn vào cùng tập, thêm dữ liệu mới không xáo lại tập cũ."""
    h = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


def prepare(records, group_of, count_tokens=None, max_tokens=None, test=0.1, val=0.1):
    """records: list (số dòng, ví dụ). group_of(ví dụ) -> khoá nhóm (mã khách, mã ticket...).
    Trả (splits, report). Ví dụ lỗi, trùng hoặc quá dài bị loại và ghi vào report."""
    splits = {"train": [], "val": [], "test": []}
    report = {"invalid": [], "duplicate": [], "too_long": [], "conflict": []}
    seen = {}                                   # prompt_key -> (số dòng, câu trả lời)
    for n, rec in records:
        errs = problems(rec)
        if errs:
            report["invalid"].append((n, errs))
            continue
        if count_tokens and max_tokens and count_tokens(rec) > max_tokens:
            report["too_long"].append(n)        # trainer sẽ cắt mất phần cuối, gồm cả token kết thúc
            continue
        key, answer = prompt_key(rec), rec["messages"][-1]["content"]
        if key in seen:                         # giữ bản đầu; bản sau có thể rơi vào tập khác và gây rò rỉ
            first, first_answer = seen[key]
            same = unicodedata.normalize("NFC", answer) == unicodedata.normalize("NFC", first_answer)
            kind = "duplicate" if same else "conflict"    # đáp án so đúng từng ký tự: "DH-1" khác "dh-1"
            report[kind].append((first, n))     # conflict: hai đáp án cho cùng đầu vào, người phải chọn
            continue
        seen[key] = (n, answer)
        part = split_of(str(group_of(rec)), test, val)
        splits[part].append(rec)
    return splits, report
