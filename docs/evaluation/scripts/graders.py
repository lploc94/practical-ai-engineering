import json, re, unicodedata
from urllib.parse import urlparse
from jsonschema import Draft202012Validator

# Chính sách của sản phẩm ví dụ; thay bằng danh sách và giới hạn của bạn.
INTERNAL_HOST_SUFFIXES = (".corp.example.com", ".internal.example.com")
URL_RE = re.compile(r"https?://[^\s<>()\[\]\"'`]+", re.IGNORECASE)
TRAILING_PUNCT = ".,;:!?…"            # dấu câu đứng sau URL trong văn bản, không thuộc URL

def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).lower().split())

def urls_in(text: str) -> list[str]:
    return [u.rstrip(TRAILING_PUNCT) for u in URL_RE.findall(text)]

def is_internal(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")   # "corp.example.com." là FQDN
    return any(host == s.lstrip(".") or host.endswith(s) for s in INTERNAL_HOST_SUFFIXES)

def grade_refund_answer(output: str, case: dict) -> dict[str, bool]:
    """Mỗi tiêu chí một kết quả pass/fail riêng; không gộp thành một điểm."""
    return {
        "mentions_order_id": case["order_id"] in output,
        "mentions_invoice_check": "hoá đơn" in norm(output) or "hóa đơn" in norm(output),
        "no_internal_url": not any(is_internal(u) for u in urls_in(output)),
        "within_length": len(output) <= case["max_chars"],   # giới hạn lấy từ đặc tả của ca
    }

def grade_extraction(raw: str, schema: dict, expected: dict) -> dict[str, bool]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"valid_json": False}
    if not isinstance(data, dict):                       # JSON hợp lệ nhưng là mảng, chuỗi, null...
        return {"valid_json": True, "schema_ok": False,
                **{f"field:{f}": False for f in expected}}
    result = {"valid_json": True,
              "schema_ok": not any(Draft202012Validator(schema).iter_errors(data))}
    for field, want in expected.items():                 # so từng field với nhãn
        result[f"field:{field}"] = data.get(field) == want
    return result
