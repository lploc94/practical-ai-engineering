"""Sinh dữ liệu đồ chơi: tin nhắn chăm sóc khách hàng -> JSON {"intent", "order_id"}.
Mỗi ví dụ mang `template` để chia tập theo mẫu câu: tập test chỉ có cách diễn đạt chưa thấy khi train."""
import json
import random

SYSTEM = ('Trích xuất ý định của khách. Trả đúng một dòng JSON {"intent": ..., "order_id": ...}. '
          'intent là một trong: track, cancel, return, warranty, payment, other. '
          'order_id dạng DH-xxxxxx, hoặc null nếu tin nhắn không có.')

TEMPLATES = {
    "track": ["Đơn {o} của mình giao tới đâu rồi?", "cho hỏi đơn {o} bao giờ tới", "{o} sao chưa thấy giao vậy shop",
              "kiểm tra giúp em tình trạng đơn {o}", "đơn {o} đang ở kho nào thế ạ", "shipper có lấy đơn {o} chưa",
              "mã đơn {o}, khi nào mình nhận được hàng?", "hàng của đơn {o} đi tới đâu rồi ad"],
    "cancel": ["Mình muốn huỷ đơn {o}", "huỷ giúp em đơn {o} với, đặt nhầm", "không mua nữa, cancel đơn {o} nha",
               "cho mình huỷ {o} được không", "đơn {o} em đổi ý rồi, huỷ giùm", "làm sao để huỷ đơn {o} vậy",
               "đơn {o} chưa giao thì huỷ luôn giúp mình", "em xin huỷ đơn hàng {o}"],
    "return": ["Đơn {o} giao sai màu, mình muốn trả hàng", "nồi cơm đơn {o} bị móp, cho em đổi trả",
               "trả lại hàng đơn {o} thế nào ạ", "đơn {o} không đúng mẫu, em muốn hoàn hàng",
               "nhận {o} rồi nhưng không ưng, trả được không", "hàng trong đơn {o} bị vỡ, đổi giúp mình",
               "muốn gửi trả sản phẩm của đơn {o}", "đơn {o} thiếu phụ kiện, em trả lại nhé"],
    "warranty": ["Máy giặt mua ở đơn {o} kêu to, bảo hành thế nào", "tủ lạnh đơn {o} không lạnh nữa, còn bảo hành không",
                 "điều hoà mua tháng trước ({o}) chảy nước, gọi bảo hành giúp", "bảo hành tivi của đơn {o} ở đâu",
                 "quạt đơn {o} hỏng motor, có được sửa miễn phí", "lò vi sóng đơn {o} không nóng, bảo hành giúp em",
                 "đơn {o} máy lọc nước báo lỗi E2, xin bảo hành", "máy sấy của đơn {o} chập điện, mang bảo hành ở đâu"],
    "payment": ["Đơn {o} trừ tiền 2 lần", "chuyển khoản đơn {o} rồi mà vẫn báo chưa thanh toán",
                "đơn {o} thanh toán thẻ bị lỗi", "mình bị trừ tiền nhưng đơn {o} không tạo được",
                "hoàn tiền đơn {o} chưa về tài khoản", "đơn {o} trả góp bị tính sai lãi",
                "ví điện tử trừ tiền đơn {o} hai lần", "đơn {o} đã trả COD sao còn nhắc thanh toán"],
    "other": ["Shop có bán máy rửa bát không", "cửa hàng mở cửa mấy giờ", "tư vấn giúp mình nồi chiên không dầu",
              "chi nhánh gần quận 7 ở đâu", "có chương trình giảm giá cuối tuần không", "mình muốn xuất hoá đơn công ty",
              "tivi 55 inch loại nào tốt", "shop có giao hàng ra Phú Quốc không"],
}


def make(seed=0, per_template=12):
    rng = random.Random(seed)
    rows = []
    for intent, templates in TEMPLATES.items():
        for t_id, tpl in enumerate(templates):
            for _ in range(per_template):
                order = f"DH-{rng.randrange(10**6):06d}" if "{o}" in tpl else None
                text = tpl + rng.choice(["", " ạ", " nhé", " shop ơi", "??", " gấp"])
                if rng.random() < 0.3:          # đổi hoa thường trước khi chèn mã đơn: nhãn phải khớp văn bản
                    text = text.capitalize() if text[0].islower() else text.lower()
                text = text.replace("{o}", order) if order else text
                answer = json.dumps({"intent": intent, "order_id": order}, ensure_ascii=False)
                rows.append({"template": f"{intent}/{t_id}", "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": answer}]})
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    with open("toy.jsonl", "w", encoding="utf-8") as f:
        for r in make():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
