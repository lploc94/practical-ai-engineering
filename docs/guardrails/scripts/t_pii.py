from presidio_analyzer.predefined_recognizers import PhoneRecognizer
from vn_pii import redact

cases = {
    "Chị Lan, SĐT 0901234567, email lan.nguyen@gmail.com": "Chị Lan, SĐT <PHONE_NUMBER>, email <EMAIL_ADDRESS>",
    "Gọi 090 123 4567 hoặc +84 912 345 678 nhé": "Gọi <PHONE_NUMBER> hoặc <PHONE_NUMBER> nhé",
    "Số 0701234567 của MobiFone": "Số <PHONE_NUMBER> của MobiFone",
    "Tổng đài 1900 1234": "Tổng đài <PHONE_NUMBER>",
    "Số CCCD của tôi là 001190012345": "Số CCCD của tôi là <VN_PERSONAL_ID>",
    "Thẻ 4111 1111 1111 1111 hết hạn 12/28": "Thẻ <CREDIT_CARD> hết hạn 12/28",
    "Đơn giá 150000000 đồng": "Đơn giá 150000000 đồng",
}
for text, want in cases.items():
    got, _ = redact(text)
    assert got == want, (text, got)
# chặn nhầm đã biết: mọi dãy 12 chữ số đều bị coi là số định danh
assert redact("Mã vận đơn 123456789012")[0] == "Mã vận đơn <VN_PERSONAL_ID>"
# tên người không được che: không có model NER tiếng Việt
assert "Chị Lan" in redact("Chị Lan, SĐT 0901234567")[0]
# vùng mặc định của PhoneRecognizer bỏ sót đầu 070 và 1900, dù bắt được 090 (khớp định dạng số Pháp)
d = PhoneRecognizer()
hits = lambda t: bool(d.analyze(t, ["PHONE_NUMBER"]))
assert hits("SĐT 0901234567") and not hits("SĐT 0701234567") and not hits("Tổng đài 1900 1234")
print("PII OK")
