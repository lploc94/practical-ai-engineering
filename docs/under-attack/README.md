# Runbook: bị DDoS khi server vật lý đứng sau Cloudflare

Runbook này dùng khi bạn **nghi ngờ đang bị tấn công**: site chậm, Cloudflare trả 52x, CPU hay băng thông của origin tăng vọt. Các bước được xếp theo thứ tự **việc nào giảm thiệt hại nhanh nhất thì làm trước**. Đọc từ trên xuống và dừng ở bước nào thấy hệ thống đã ổn.

Áp dụng cho: origin là server vật lý (bare metal / on-prem), website hoặc API được proxy qua Cloudflare (orange cloud), plan Free đến Business. Chỗ nào chỉ có ở plan cao hơn đều được ghi rõ.

> Cập nhật 11/2026, đối chiếu với tài liệu Cloudflare tại thời điểm viết. Tên menu trong dashboard Cloudflare thay đổi khá thường xuyên. Nếu không thấy đúng tên, tìm theo tên tính năng.

---

## Mục lục

- [Phút 0–2: Traffic tấn công đi qua đâu?](#phút-02-traffic-tấn-công-đi-qua-đâu)
- [Nhánh A — Tấn công đi qua Cloudflare (L7)](#nhánh-a--tấn-công-đi-qua-cloudflare-l7)
  - [A1. Phút 2: Bật Under Attack Mode](#a1-phút-2-bật-under-attack-mode)
  - [A2. Phút 3–8: Tìm signature của traffic tấn công](#a2-phút-38-tìm-signature-của-traffic-tấn-công)
  - [A3. Phút 8–15: Chặn đúng signature bằng custom rule](#a3-phút-815-chặn-đúng-signature-bằng-custom-rule)
  - [A4. Phút 15–25: Rate limiting](#a4-phút-1525-rate-limiting)
  - [A5. Phút 25+: Giảm tải origin](#a5-phút-25-giảm-tải-origin)
  - [A6. Theo dõi và lặp lại](#a6-theo-dõi-và-lặp-lại)
- [Nhánh B — Tấn công đánh thẳng vào origin](#nhánh-b--tấn-công-đánh-thẳng-vào-origin)
  - [B1. Phút 2: Chặn mọi thứ không đến từ Cloudflare](#b1-phút-2-chặn-mọi-thứ-không-đến-từ-cloudflare)
  - [B2. Nếu đường truyền đã nghẽn: gọi nhà cung cấp](#b2-nếu-đường-truyền-đã-nghẽn-gọi-nhà-cung-cấp)
  - [B3. Bịt chỗ lộ IP, rồi mới đổi IP](#b3-bịt-chỗ-lộ-ip-rồi-mới-đổi-ip)
- [Hạ nhiệt và sau sự cố](#hạ-nhiệt-và-sau-sự-cố)
- [Phụ lục 1 — Security rules và rate limiting](#phụ-lục-1--security-rules-và-rate-limiting)
- [Phụ lục 2 — Chuẩn bị trước để runbook chạy nhanh](#phụ-lục-2--chuẩn-bị-trước-để-runbook-chạy-nhanh)
- [Phụ lục 3 — Lỗi hay gặp khi xử lý](#phụ-lục-3--lỗi-hay-gặp-khi-xử-lý)
- [Nguồn](#nguồn)

---

## Phút 0–2: Traffic tấn công đi qua đâu?

Trước khi bấm bất cứ thứ gì, cần trả lời một câu: **traffic tấn công có đi qua Cloudflare không?** Mọi rule trên Cloudflare chỉ có tác dụng với traffic đi qua Cloudflare. Nếu kẻ tấn công đã biết IP origin và đánh thẳng vào đó, bật Under Attack Mode chẳng thay đổi gì.

**Trên Cloudflare** — mở zone → **Security → Analytics** (dashboard cũ: **Security → Overview / Analytics**). Số request có tăng vọt cùng lúc với sự cố không?

**Trên origin** — xem ai đang kết nối vào:

```bash
# Top IP đang giữ kết nối vào 80/443
ss -Htn state established '( sport = :80 or sport = :443 )' \
  | awk '{print $4}' | sed 's/:[0-9]*$//' | sort | uniq -c | sort -rn | head -20

# Số kết nối đang half-open (dấu hiệu SYN flood)
ss -Htn state syn-recv | wc -l

# Băng thông trên interface, cập nhật mỗi giây
sar -n DEV 1        # gói sysstat; hoặc: nload, iftop
```

Nếu nginx đã cấu hình realip ([Phụ lục 2](#nginx-lấy-ip-thật-và-không-lộ-certificate)), access log có `$realip_remote_addr` cũng cho biết request đến từ Cloudflare hay từ ngoài.

Đối chiếu các IP đầu bảng với [dải IP Cloudflare](https://www.cloudflare.com/ips/). Nếu dùng Cloudflare Tunnel, kết nối web đến từ `cloudflared` trên chính máy (localhost), không có kết nối web nào từ ngoài vào.

| Bạn thấy | Kết luận | Làm tiếp |
|---|---|---|
| Analytics Cloudflare tăng vọt, các kết nối vào origin đều là IP Cloudflare | Tấn công L7 qua Cloudflare | [Nhánh A](#nhánh-a--tấn-công-đi-qua-cloudflare-l7) |
| Analytics Cloudflare bình thường, origin có nhiều kết nối từ IP lạ, nhiều `syn-recv`, hoặc băng thông đầy | Origin IP đã lộ, bị đánh trực tiếp | [Nhánh B](#nhánh-b--tấn-công-đánh-thẳng-vào-origin) |
| Cả hai | Làm **B1 trước** (mất 1 phút), rồi sang Nhánh A | B1 → A1 |
| Analytics Cloudflare tăng nhưng origin vẫn khoẻ | Cloudflare đang chặn tốt | Theo dõi; làm A2 để chắc chắn, không cần vội |

Đừng nhầm mã lỗi 52x là do origin trả về. Các trang 520–524 do **Cloudflare** sinh ra khi không nói chuyện được với origin. Chúng cho biết origin đang quá tải hoặc bị chặn, không cho biết traffic tấn công đi đường nào.

| Mã | Ý nghĩa |
|---|---|
| 520 | Origin trả response rỗng hoặc lỗi định dạng |
| 521 | Origin từ chối kết nối (web server chết, hoặc firewall chặn IP Cloudflare) |
| 522 | Hết thời gian chờ kết nối tới origin — thường gặp nhất khi origin quá tải |
| 523 | Không định tuyến được tới origin |
| 524 | Đã kết nối nhưng origin không trả lời trong 125 giây |

---

## Nhánh A — Tấn công đi qua Cloudflare (L7)

Lớp L3/L4 (SYN flood, UDP flood) nhắm vào zone đã proxy được Cloudflare tự hấp thụ trên mọi plan. Cái lọt qua là **HTTP flood**: request trông hợp lệ, được chuyển tiếp đến origin. Bộ rule HTTP DDoS của Cloudflare luôn bật và tự chặn phần lớn các đợt lớn. Các bước dưới đây xử lý phần còn lại.

### A1. Phút 2: Bật Under Attack Mode

**Hiệu quả:** cao nhất trong thời gian ngắn nhất với site phục vụ trình duyệt. Mọi visitor phải qua một trang challenge trước khi vào. Đa số script flood không chạy JavaScript nên bị dừng ở đây.

**Cách bật** — một trong hai:

- Dashboard: zone → trang **Overview** → khung **Quick Actions** → bật **Under Attack Mode**.
- API (nhanh hơn nếu đã chuẩn bị token, xem [Phụ lục 2](#phụ-lục-2--chuẩn-bị-trước-để-runbook-chạy-nhanh)):

```bash
curl -X PATCH "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/settings/security_level" \
  -H "Authorization: Bearer $CF_API_TOKEN" -H "Content-Type: application/json" \
  --data '{"value":"under_attack"}'
```

**Visitor sẽ thấy gì:** trang "đang kiểm tra trình duyệt" dạng Managed Challenge, thường xong trong vài giây. Qua rồi thì trình duyệt nhận cookie `cf_clearance` và không bị hỏi lại trong một khoảng thời gian.

**Đánh đổi phải biết trước khi bật:**

- **API, app mobile, webhook, health check sẽ hỏng.** Challenge trả về trang HTML, client không phải trình duyệt không vượt qua được. Response bị challenge có header `cf-mitigated: challenge`, có thể dùng để phát hiện trong log client.
- Nếu zone vừa có web vừa có API, có hai cách để chừa API ra:
  - Bật Under Attack toàn zone, rồi tạo **Configuration Rule** (zone → **Rules → Configuration Rules**, mọi plan) với expression `starts_with(http.request.uri.path, "/api/")` và setting **I'm Under Attack** = **Off**.
  - Hoặc không bật toàn zone mà tạo custom rule **Managed Challenge** chỉ cho phần web (rule [R1 ở Phụ lục 1](#thư-viện-rule-dùng-ngay)).

  Ngược lại, muốn bật Under Attack cho riêng một phần site (ví dụ `/admin`), dùng Configuration Rule với setting **I'm Under Attack** = **On**.
- Security Level Low/Medium/High **không còn tác dụng** (Cloudflare bỏ từ 03/2025, `cf.threat_score` nay luôn bằng 0). Đừng mất thời gian chỉnh nó. "I'm Under Attack" là giá trị duy nhất còn ý nghĩa.

**Kiểm tra đã có tác dụng:** sau 1–2 phút, CPU và số kết nối của origin giảm; trong Security Analytics, phần lớn request chuyển sang trạng thái challenged.

Nếu origin hồi phục: sang A2 để tìm signature, rồi thay Under Attack toàn zone bằng rule nhắm đúng mục tiêu. Under Attack để lâu làm khó chịu người dùng thật.

### A2. Phút 3–8: Tìm signature của traffic tấn công

Mục tiêu: tìm **một đặc điểm chung** mà traffic tấn công có và người dùng thật không có. Có nó, bạn chặn được chính xác thay vì challenge tất cả.

Mở zone → **Security → Analytics**. Chọn khoảng thời gian bao trọn đợt tấn công, rồi xem các bảng **top** theo từng chiều:

| Chiều | Signature thường gặp |
|---|---|
| Path | Một path bị đánh dồn dập: `/`, `/login`, `/search`, một endpoint API nặng |
| Query string | Query ngẫu nhiên (`/?a8f3k2=...`) để né cache — cache busting |
| Method | `POST` vào path vốn chỉ nhận `GET` |
| User-Agent | Một UA lặp lại, UA rỗng, UA của thư viện HTTP (`python-requests`, `Go-http-client`, `curl`) |
| ASN | Tập trung vào vài ASN hosting / cloud (nơi đặt botnet thuê) thay vì ISP dân dụng |
| Country | Tập trung vào quốc gia không phải thị trường của bạn |
| Host | Một subdomain cụ thể |

**Security Events** (tab **Events** trong Analytics) chỉ chứa request đã bị Cloudflare chặn hay challenge, hữu ích để xem rule nào đang ăn. Plan Free chỉ thấy log lấy mẫu.

Ghi lại 1–3 signature rõ nhất. Khi mỗi IP chỉ gửi ít request nhưng có hàng chục nghìn IP, rate limiting theo IP sẽ không ăn thua. Khi đó signature (path, UA, ASN) là vũ khí chính.

### A3. Phút 8–15: Chặn đúng signature bằng custom rule

Zone → **Security → Security rules** → **Create rule → Custom rule** (dashboard cũ: **Security → WAF → Custom rules**). Dán expression vào **Expression Editor**.

Thứ tự ưu tiên khi chọn action:

1. **Block** — khi signature chắc chắn chỉ khớp traffic tấn công (một UA lạ, một ASN không có khách thật).
2. **Managed Challenge** — khi signature có thể khớp cả người thật (một quốc gia, một path phổ biến). Người thật qua được, bot thì không. Đây là action Cloudflare khuyến nghị cho đa số trường hợp.
3. Không dùng **Interactive Challenge** trừ khi bất đắc dĩ — người thật phải bấm, tỉ lệ bỏ trang cao.

Ví dụ nhanh (danh sách đầy đủ ở [Phụ lục 1](#thư-viện-rule-dùng-ngay)):

```text
# Path bị flood + UA của thư viện HTTP → Block
http.request.uri.path eq "/search" and http.user_agent contains "python-requests"

# Các ASN hosting đang bắn → Managed Challenge, chừa bot hợp lệ
ip.src.asnum in {14061 16509 24940} and not cf.client.bot

# Cache busting vào trang chủ: trang chủ không dùng query → Block
http.request.uri.path eq "/" and http.request.uri.query ne ""
```

Sau khi bật rule, quay lại Security Analytics. Trong 1–2 phút, request khớp rule phải chuyển sang blocked / challenged. Nếu không, xem [Phụ lục 3](#phụ-lục-3--lỗi-hay-gặp-khi-xử-lý).

Rule đã chặn đúng và origin đã ổn → tắt Under Attack Mode toàn zone (A1).

### A4. Phút 15–25: Rate limiting

Rate limiting hữu ích khi **từng IP gửi nhiều request** (login brute-force, scraping, flood từ ít nguồn). Nó vô dụng khi botnet trải rộng, mỗi IP chỉ vài request.

Zone → **Security → Security rules** → **Create rule → Rate limiting rule**.

Giới hạn theo plan quyết định bạn làm được gì:

| | Free | Pro | Business | Enterprise |
|---|---|---|---|---|
| Số rule | 1 | 2 | 5 | 100 |
| Chu kỳ đếm | 10 giây | tới 1 phút | tới 10 phút | tới ~18 giờ |
| Thời gian chặn | 10 giây | tới 1 giờ | tới 1 ngày | tới 1 ngày |
| Đếm theo | IP | IP | IP, IP có hỗ trợ NAT | Nhiều characteristic (header, cookie, ASN, JA4...) |
| Field trong expression | Path, Verified Bot | + Host, URI, Query | + Method, Source IP, User-Agent | + Header, body |

Hệ quả thực tế:

- **Free:** chỉ có một rule, chỉ lọc theo path. Đặt nó cho path đang bị đánh, ví dụ `/login`: `10 request / 10 giây / IP → Block 10 giây`. Không có field Method nên không phân biệt `GET` với `POST`.
- **Pro / Business:** rule cho login, rule cho API ghi dữ liệu. Ví dụ ở [Phụ lục 1](#rate-limiting-thiết-kế-và-ví-dụ).
- Trên Free, Pro, Business, nếu chọn action **challenge** thì chỉ chạy được ở chế độ throttle (chỉ áp lên phần request vượt ngưỡng), không chặn theo thời lượng.

Chọn ngưỡng: lấy mức cao nhất một người dùng thật có thể tạo trong một chu kỳ, nhân 3–5. Nếu không có số liệu, bắt đầu cao rồi hạ dần; một ngưỡng quá thấp sẽ chặn người thật đứng chung NAT (văn phòng, mạng di động).

Counter của rate limiting được đếm **riêng tại từng data center** Cloudflare, không đếm gộp toàn mạng. Botnet trải khắp thế giới có thể vượt "ngưỡng" tổng mà không bị chặn ở đâu.

### A5. Phút 25+: Giảm tải origin

Khi đã chặn được phần lớn, những việc sau giảm phần còn lọt tới origin:

1. **Cache thêm.** Mặc định Cloudflare không cache HTML. Với trang không phụ thuộc người dùng (trang chủ, danh mục, trang sản phẩm khi chưa đăng nhập):
   - Cho origin trả header `Cache-Control: public, s-maxage=60` trên các trang đó (Cloudflare tôn trọng `s-maxage` của origin trên mọi plan).
   - Zone → **Caching → Cache Rules** (mọi plan): tạo rule **Eligible for cache** cho các path đó, Edge TTL để **dùng header Cache-Control của origin**.
   - Không dùng Edge TTL override để đặt TTL ngắn trên Free / Pro: Edge Cache TTL tối thiểu là **2 giờ (Free)**, **1 giờ (Pro)**, 1 giây từ Business.
   - Chỉ áp cho request không có cookie session, nếu không sẽ cache nhầm trang của người đã đăng nhập. Ví dụ expression: `http.request.uri.path in {"/" "/products"} and not http.cookie contains "session_id"` (thay `session_id` bằng tên cookie của bạn).
2. **Chặn cache busting.** Bỏ query string khỏi cache key chỉ có ở Enterprise. Ở plan thấp hơn, chặn bằng custom rule: path nào không dùng query thì block request có query (ví dụ ở A3).
3. **Tắt tạm tính năng nặng.** Search không cache, export báo cáo, endpoint gọi DB nặng: trả `503` ngay ở nginx, hoặc chặn bằng custom rule trong thời gian bị đánh.
4. **Waiting Room** (Business trở lên): xếp hàng visitor vào các trang quan trọng thay vì để origin sập.
5. **Rate limit ở nginx** làm lớp thứ hai — chỉ đúng khi nginx đã lấy được IP thật của visitor ([Phụ lục 2](#nginx-lấy-ip-thật-và-không-lộ-certificate)). Nếu chưa cấu hình, **đừng bật**: nginx sẽ thấy mọi request đến từ vài IP Cloudflare (hoặc `127.0.0.1` nếu dùng Tunnel) và chặn luôn người dùng thật.

### A6. Theo dõi và lặp lại

- Xem Security Analytics mỗi 5–10 phút. Kẻ tấn công đổi pattern khi thấy bị chặn: đổi UA, đổi path, chuyển ASN.
- Mỗi pattern mới → quay lại A2–A3. Sửa rule cũ thay vì tạo thêm, vì Free chỉ có 5 custom rule.
- Thông báo cho người dùng và team support: "Hệ thống đang được bảo vệ khỏi một đợt tấn công, bạn có thể thấy trang kiểm tra trình duyệt." Ghi lại mốc thời gian mọi thay đổi để làm post-mortem.

---

## Nhánh B — Tấn công đánh thẳng vào origin

Khi IP origin đã lộ, traffic tấn công đi thẳng vào server, không qua Cloudflare. Zone Cloudflare thông thường không giúp được gì ở đây. Bảo vệ L3/L4 cho chính dải IP của bạn chỉ có qua **Magic Transit** (Enterprise). **Spectrum** đưa các dịch vụ TCP/UDP khác (SSH, game, cổng tuỳ ý) ra sau IP của Cloudflare để không phải lộ IP origin, nhưng không bảo vệ IP đã lộ; cổng TCP/UDP tuỳ ý cần Enterprise, Pro / Business chỉ có SSH và Minecraft (add-on trả phí).

### B1. Phút 2: Chặn mọi thứ không đến từ Cloudflare

**Hiệu quả:** loại bỏ ngay HTTP flood trực tiếp vào origin. **Không** cứu được nếu đường truyền đã bị lấp đầy (xem B2).

Nếu dùng **Cloudflare Tunnel**: origin không cần nhận kết nối vào cổng web từ ngoài. Đóng hẳn 80/443:

```bash
# Thay toàn bộ bảng cf_guard (kể cả khi trước đó đã chạy script allowlist bên dưới)
nft -f - <<'EOF'
table inet cf_guard
delete table inet cf_guard
table inet cf_guard {
  chain input {
    type filter hook input priority -10; policy accept;
    tcp dport { 80, 443 } drop
  }
}
EOF
```

Nếu **không** dùng Tunnel: chỉ cho dải IP Cloudflare vào 80/443. Script dưới thay thế toàn bộ bảng `cf_guard` trong một transaction, chạy lại bao nhiêu lần cũng được:

```bash
#!/usr/bin/env bash
# /usr/local/sbin/cf-only.sh — chỉ cho Cloudflare vào 80/443. Chạy bằng root.
set -euo pipefail
v4=$(curl -fsS https://www.cloudflare.com/ips-v4 | paste -sd, -)
v6=$(curl -fsS https://www.cloudflare.com/ips-v6 | paste -sd, -)
[ -n "$v4" ] && [ -n "$v6" ] || { echo "Không tải được danh sách IP Cloudflare" >&2; exit 1; }

nft -f - <<EOF
table inet cf_guard
delete table inet cf_guard
table inet cf_guard {
  set cf4 { type ipv4_addr; flags interval; elements = { $v4 } }
  set cf6 { type ipv6_addr; flags interval; elements = { $v6 } }
  chain input {
    type filter hook input priority -10; policy accept;
    tcp dport { 80, 443 } ip  saddr @cf4 accept
    tcp dport { 80, 443 } ip6 saddr @cf6 accept
    tcp dport { 80, 443 } drop
  }
}
EOF
echo "OK: chỉ Cloudflare được vào 80/443"
```

Chú ý:

- Script chỉ đụng tới cổng 80/443. **Kiểm tra SSH và các cổng quản trị khác** — nếu đang mở cho cả internet, chúng cũng là mục tiêu. Giới hạn SSH theo IP quản trị, hoặc chuyển sang Cloudflare Tunnel + Access.
- Bảng có priority `-10` nên chạy trước ufw / firewalld. Lệnh `drop` có hiệu lực ngay cả khi firewall khác cho phép.
- Allowlist IP Cloudflare chỉ chứng minh request **đi qua mạng Cloudflare**, không chứng minh đi qua **zone của bạn**. Kẻ tấn công có thể dùng tài khoản Cloudflare riêng trỏ vào IP của bạn. Để chặn trường hợp này, cần Tunnel, hoặc Authenticated Origin Pulls với **certificate của riêng bạn** (zone-level hoặc per-hostname). Xem [Phụ lục 2](#bảo-vệ-origin-chọn-cách-nào).

### B2. Nếu đường truyền đã nghẽn: gọi nhà cung cấp

Nếu băng thông vào interface đã chạm trần (thấy ở `sar -n DEV 1`), firewall trên máy vô dụng: gói tin đã chiếm hết đường truyền trước khi tới máy. Chỉ phía trên mới chặn được:

1. **Gọi nhà cung cấp hosting / datacenter / ISP.** Hỏi họ có lọc DDoS upstream không, có đặt được ACL chỉ cho dải IP Cloudflare vào 80/443 không.
2. Nhiều nhà cung cấp sẽ **null-route** (blackhole) IP bị tấn công để bảo vệ mạng của họ. Server mất kết nối hoàn toàn trên IP đó. Hãy chuẩn bị tinh thần và chuyển sang B3.

### B3. Bịt chỗ lộ IP, rồi mới đổi IP

Đổi IP mà chưa tìm ra chỗ lộ thì IP mới sẽ lộ tiếp sau vài giờ. Kiểm tra theo thứ tự hay gặp:

| Chỗ lộ | Kiểm tra |
|---|---|
| Record DNS không proxy (grey cloud) cùng IP: `mail`, `ftp`, `direct`, `cpanel`, subdomain cũ | Zone → **DNS → Records**, lọc record DNS-only |
| Record SPF chứa `ip4:<IP origin>`, record MX trỏ về cùng máy | Xem các record TXT, MX |
| Email gửi đi từ chính server (header `Received:` chứa IP), email bounce khi gửi tới địa chỉ không tồn tại | Gửi thử một email từ hệ thống, xem header |
| Server tự đi fetch URL theo yêu cầu người dùng (webhook, preview link, upload từ URL) — request đi ra mang IP origin | Tìm tính năng gọi HTTP ra ngoài |
| Origin trả certificate chứa domain khi truy cập thẳng IP — các máy quét như Censys, Shodan đã ghi lại | `curl -vk https://<IP-origin>/` từ máy ngoài, xem certificate trả về |
| Lịch sử DNS trước khi dùng Cloudflare | Các dịch vụ DNS history công khai lưu IP cũ |
| Trang lỗi, header, debug endpoint in ra IP | Rà header `X-Backend`, trang lỗi framework |

Sau khi bịt:

1. **Chuyển sang Cloudflare Tunnel** nếu có thể (xem [Phụ lục 2](#cloudflare-tunnel)). Tunnel đi ra ngoài, không cần mở cổng web, nên dù IP lộ cũng không có gì để đánh vào cổng web.
2. **Xin IP mới** từ nhà cung cấp, tốt nhất ở dải khác (không cùng /24 với IP cũ). Không gắn IP mới vào bất cứ record DNS-only nào.
3. Cập nhật IP mới trong record DNS **đã proxy** (hoặc không cần nếu dùng Tunnel). Chạy lại B1 trên IP mới.

---

## Hạ nhiệt và sau sự cố

Gỡ biện pháp khẩn cấp theo thứ tự ngược, mỗi bước chờ 15–30 phút xem có tái phát không:

1. Tắt Under Attack Mode toàn zone (nếu chưa tắt ở A3).
2. Chuyển các rule Block rộng (chặn cả quốc gia, cả ASN) sang Managed Challenge, rồi tắt hẳn (disable, chưa xoá).
3. Bỏ các cache rule TTL ngắn tạm thời nếu chúng gây dữ liệu cũ.
4. Bật lại các tính năng đã tắt tạm.

Trong 1–3 ngày sau:

- **Post-mortem:** mốc thời gian, signature, rule nào hiệu quả, rule nào chặn nhầm, mất bao lâu từ lúc phát hiện tới lúc ổn.
- **Giữ lại** những rule có giá trị lâu dài (UA rỗng vào API, rate limit login) dưới dạng **tắt sẵn** để lần sau bật bằng một cú bấm.
- Nếu IP đã lộ: đổi IP **ngay cả khi tấn công đã dừng**. IP đó đã nằm trong danh sách của kẻ tấn công.
- Hoàn thiện [Phụ lục 2](#phụ-lục-2--chuẩn-bị-trước-để-runbook-chạy-nhanh) cho những phần còn thiếu.
- Nếu tấn công lặp lại với quy mô lớn: cân nhắc nâng plan (Business cho rate limiting đủ dùng và Waiting Room; Enterprise cho Advanced Rate Limiting, Bot Management, Magic Transit).

---

## Phụ lục 1 — Security rules và rate limiting

### Thứ tự Cloudflare xử lý một request

Một request HTTP đi qua các lớp theo thứ tự:

```text
HTTP DDoS ruleset (luôn bật)
  → Custom rules
  → Rate limiting rules
  → Managed rules (WAF)
  → Super Bot Fight Mode (Pro+)
```

Ngoài chuỗi trên, một số tính năng chạy độc lập: IP Access Rules, Security Level / Under Attack, Browser Integrity Check, User Agent Blocking, Zone Lockdown, và **Bot Fight Mode** (bản Free).

Hệ quả:

- Một action kết thúc (Block, Challenge) ở custom rule sẽ dừng các lớp sau. Request đã bị block ở custom rule thì không được rate limiting đếm.
- **Skip** trong custom rule bỏ qua được: các custom rule còn lại, rate limiting, managed rules, Super Bot Fight Mode, và các product như Security Level, Browser Integrity Check.
- **Bot Fight Mode (Free) không skip được** bằng custom rule. Nếu nó chặn API hay app mobile của bạn, cách duy nhất để chừa ra là **IP Access Rule** với action Allow cho IP đó, hoặc tắt Bot Fight Mode.
- IP Access Rule với action **Allow** bỏ qua custom rules, rate limiting và managed rules. Dùng cẩn thận.

### Action nào cho tình huống nào

| Action | Dùng khi | Ghi chú |
|---|---|---|
| Block | Signature chắc chắn chỉ khớp tấn công | Có thể tuỳ chỉnh response ở rate limiting (mặc định 429) |
| Managed Challenge | Signature có thể khớp người thật | Cloudflare khuyến nghị; tự chọn mức challenge |
| JS Challenge | Cần challenge không tương tác | Visitor chờ dưới 5 giây |
| Interactive Challenge | Hầu như không | Người thật phải bấm, tỉ lệ bỏ trang cao |
| Skip | Chừa traffic hợp lệ (health check, IP văn phòng, webhook đối tác) | Đặt rule Skip **trước** các rule chặn |
| Log | Thử rule trước khi chặn thật | **Chỉ Enterprise.** Plan thấp hơn: thử bằng Managed Challenge trước rồi chuyển sang Block |

Challenge (mọi loại) làm hỏng client không phải trình duyệt. Đừng đặt challenge lên path API mà app mobile hay đối tác gọi. Với API, dùng Block theo signature hoặc rate limiting.

### Viết expression: những lỗi hay gặp

| Viết sai | Viết đúng |
|---|---|
| `http.request.uri.path starts_with "/api/"` | `starts_with(http.request.uri.path, "/api/")` — `starts_with` là **hàm** |
| `http.request.method in {"POST", "PUT"}` | `http.request.method in {"POST" "PUT"}` — phần tử trong set cách nhau bằng **dấu cách** |
| `ip.geoip.asnum`, `ip.geoip.country` | `ip.src.asnum`, `ip.src.country` — tên field hiện hành |
| `cf.threat_score gt 10` | Không dùng — luôn bằng 0 từ 03/2025 |
| `http.request.uri.path contains "/login"` khi có `/api/login-history` | `http.request.uri.path eq "/login"` — `contains` khớp quá rộng |
| `http.request.uri.path matches "^/api/"` trên Free / Pro | `matches` (regex) chỉ có ở Business+; dùng `starts_with()` |

So sánh path phân biệt hoa thường. Nếu ứng dụng chấp nhận `/Login`, kẻ tấn công cũng dùng được — thêm `lower(http.request.uri.path) eq "/login"`.

`cf.client.bot` là `true` với các bot đã được Cloudflare xác minh (Googlebot, Bingbot...). Thêm `and not cf.client.bot` vào các rule chặn rộng để không làm rớt SEO.

### Thư viện rule dùng ngay

Tạo sẵn các rule này ở trạng thái **tắt** trước khi có sự cố. Free có 5 custom rule; chọn những rule phù hợp nhất với hệ thống của bạn.

**R1 — Challenge phần web, chừa API** (thay cho Under Attack toàn zone khi có API)

```text
not starts_with(http.request.uri.path, "/api/") and not cf.client.bot
```
Action: Managed Challenge.

**R2 — Chặn client tự động vào trang HTML**

```text
not starts_with(http.request.uri.path, "/api/")
and (http.user_agent eq ""
  or http.user_agent contains "python-requests"
  or http.user_agent contains "Go-http-client"
  or http.user_agent contains "curl/")
```
Action: Block. Đừng áp lên API nếu đối tác của bạn gọi bằng các thư viện này.

**R3 — Cache busting vào trang không dùng query**

```text
http.request.uri.path in {"/" "/index.html"} and http.request.uri.query ne ""
```
Action: Block. Thêm các path khác nếu chúng cũng bị đánh.

**R4 — Method lạ vào path chỉ đọc**

```text
http.request.uri.path eq "/search" and http.request.method ne "GET"
```
Action: Block.

**R5 — Challenge theo ASN hoặc quốc gia đang bắn** (tạm thời)

```text
(ip.src.asnum in {14061 16509 24940} or ip.src.country in {"XX" "YY"})
and not cf.client.bot
```
Action: Managed Challenge. Thay số ASN và mã quốc gia theo số liệu ở A2. Chặn theo quốc gia bằng IP Access Rules chỉ có ở Enterprise; ở plan khác dùng custom rule như trên.

**R0 — Skip cho traffic tin cậy** (đặt đầu tiên)

```text
ip.src in {203.0.113.10 198.51.100.0/24}
or (http.request.uri.path eq "/healthz" and http.request.headers["x-health-token"][0] eq "<bí-mật>")
```
Action: Skip → chọn các custom rule còn lại, rate limiting, Super Bot Fight Mode. Không chừa được Bot Fight Mode (Free).

### Rate limiting: thiết kế và ví dụ

Mỗi rule gồm: **expression** (request nào bị xét), **characteristic** (đếm theo gì — thường là IP), **period** (chu kỳ đếm), **ngưỡng**, **action** và **mitigation timeout** (chặn bao lâu).

| Mục tiêu | Expression | Ngưỡng gợi ý | Plan |
|---|---|---|---|
| Flood một path (Free) | `http.request.uri.path eq "/login"` | 10 req / 10 giây / IP → Block 10 giây | Free+ |
| Login brute-force | `http.request.uri.path eq "/login" and http.request.method eq "POST"` | 10 req / 1 phút / IP → Block 10 phút | Business+ (cần field Method) |
| API ghi dữ liệu | `starts_with(http.request.uri.path, "/api/") and http.request.method in {"POST" "PUT" "PATCH" "DELETE"}` | 60 req / 1 phút / IP → Block 5 phút | Business+ |
| API đọc | `starts_with(http.request.uri.path, "/api/")` | Theo baseline × 3–5 | Pro+ |

Điểm cần biết:

- **Counting expression** (Business+): đếm theo một điều kiện khác với điều kiện áp dụng. Ví dụ chỉ đếm response `401` (`http.response.code eq 401`) để chặn IP đoán mật khẩu sai nhiều lần mà không chặn người đăng nhập đúng.
- **IP có hỗ trợ NAT** (Business+) dùng cookie `_cfuvid` để tách người dùng sau cùng một IP. Client không giữ cookie sẽ bị gộp chung một bucket.
- Rate limiting chạy **sau** custom rules. Request đã bị custom rule block hay challenge thì không được rate limiting đếm.
- **Request rate analysis** để gợi ý ngưỡng chỉ có ở Enterprise. Ở plan khác, tự ước lượng từ Security Analytics lúc bình thường.

### Thêm rule qua API

Dashboard có thể chậm hoặc không vào được giữa sự cố. Chuẩn bị sẵn API token với quyền **Zone → WAF → Edit** và **Zone → Zone Settings → Edit** (cho Under Attack).

```bash
export CF_API_TOKEN=...   # token đã tạo sẵn
export ZONE_ID=...        # Zone ID ở trang Overview

# Lấy ID của ruleset custom rules (entrypoint của phase)
curl -s "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/rulesets/phases/http_request_firewall_custom/entrypoint" \
  -H "Authorization: Bearer $CF_API_TOKEN" | jq -r '.result.id'
```

Thêm một custom rule (không ảnh hưởng các rule đang có):

```bash
curl -X POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/rulesets/$CUSTOM_RULESET_ID/rules" \
  -H "Authorization: Bearer $CF_API_TOKEN" -H "Content-Type: application/json" \
  --data '{
    "description": "incident: cache busting trang chủ",
    "expression": "http.request.uri.path eq \"/\" and http.request.uri.query ne \"\"",
    "action": "block"
  }'
```

Thêm một rate limiting rule (lấy ID entrypoint của phase `http_ratelimit` như trên). Qua API phải khai báo `cf.colo.id` trong `characteristics`:

```bash
curl -X POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/rulesets/$RATELIMIT_RULESET_ID/rules" \
  -H "Authorization: Bearer $CF_API_TOKEN" -H "Content-Type: application/json" \
  --data '{
    "description": "incident: login flood",
    "expression": "http.request.uri.path eq \"/login\"",
    "action": "block",
    "ratelimit": {
      "characteristics": ["cf.colo.id", "ip.src"],
      "period": 10,
      "requests_per_period": 10,
      "mitigation_timeout": 10
    }
  }'
```

Rule Skip qua API dùng `action_parameters`:

```json
{
  "description": "skip cho IP văn phòng",
  "expression": "ip.src in {203.0.113.10}",
  "action": "skip",
  "action_parameters": { "phases": ["http_ratelimit"] }
}
```

Muốn bỏ qua các custom rule còn lại thay vì một phase, dùng `"action_parameters": { "ruleset": "current" }`.

Nếu entrypoint chưa tồn tại (zone chưa từng có rule ở phase đó), lệnh GET trả lỗi 404. Tạo rule đầu tiên trên dashboard, hoặc dùng `PUT .../phases/<phase>/entrypoint` — lệnh này **thay toàn bộ** rule của phase, không dùng khi đã có rule.

---

## Phụ lục 2 — Chuẩn bị trước để runbook chạy nhanh

Mỗi mục dưới đây cắt được vài phút đến vài giờ khi sự cố xảy ra.

### Danh sách ngắn

- [ ] Mọi record web (A, AAAA, CNAME) đã proxy (orange cloud). Không có record DNS-only nào trỏ cùng IP với origin.
- [ ] Origin dùng Cloudflare Tunnel, **hoặc** firewall chỉ cho IP Cloudflare vào 80/443 ([B1](#b1-phút-2-chặn-mọi-thứ-không-đến-từ-cloudflare)) + Authenticated Origin Pulls với certificate riêng.
- [ ] SSH và cổng quản trị không mở cho cả internet.
- [ ] Đã đổi IP origin sau khi chuyển sang Cloudflare (IP cũ nằm trong lịch sử DNS).
- [ ] nginx lấy IP thật của visitor và không trả certificate khi truy cập bằng IP.
- [ ] API token, Zone ID, ID các ruleset đã lưu sẵn ở nơi đội trực truy cập được, kể cả khi dashboard không vào được.
- [ ] Các rule ở [Phụ lục 1](#thư-viện-rule-dùng-ngay) đã tạo sẵn ở trạng thái tắt.
- [ ] Đã bật alert **HTTP DDoS Attack Alert** (mọi plan) trong **Notifications**.
- [ ] Đã ghi lại baseline lúc bình thường: request/giây, top path, top ASN — để biết thế nào là bất thường.
- [ ] Đã có số điện thoại / kênh hỗ trợ khẩn của nhà cung cấp hosting.
- [ ] Đã diễn tập: bật Under Attack, bật một rule, chạy script B1 — và biết mỗi bước tốn bao lâu.

### Bảo vệ origin: chọn cách nào

| Cách | Chặn được | Không chặn được | Ghi chú |
|---|---|---|---|
| **Cloudflare Tunnel** | Mọi truy cập trực tiếp vào cổng web | — | Khuyến nghị cho server vật lý. Không cần mở cổng vào. Không dùng chung với Authenticated Origin Pulls |
| **Authenticated Origin Pulls**, certificate riêng (zone-level / per-hostname) | Truy cập không qua zone của bạn | Tấn công băng thông vào IP | Cần SSL mode Full / Full (strict). Mọi plan |
| Authenticated Origin Pulls, certificate chung của Cloudflare (global) | Truy cập không qua mạng Cloudflare | Truy cập qua tài khoản Cloudflare khác | Certificate dùng chung cho mọi khách hàng Cloudflare |
| Allowlist IP Cloudflare | Truy cập không qua mạng Cloudflare | Truy cập qua tài khoản Cloudflare khác; tấn công băng thông | Mức tối thiểu, nên kết hợp với cách khác |

### Cloudflare Tunnel

Tạo tunnel quản lý từ dashboard (đơn giản nhất):

1. Dashboard → **Networking → Tunnels → Create a tunnel**, đặt tên.
2. Chọn hệ điều hành, **copy lệnh cài đặt dashboard đưa ra** và chạy trên origin. Lệnh có dạng `sudo cloudflared service install <TOKEN>`, cài `cloudflared` thành service chạy bằng token của tunnel. Giữ token như mật khẩu; nếu lộ, rotate token trên dashboard.
3. Tab **Routes → Add route → Published application**: chọn hostname, **Service URL** là địa chỉ nội bộ, ví dụ `http://localhost:8080`.
4. Đóng 80/443 trên firewall ([B1](#b1-phút-2-chặn-mọi-thứ-không-đến-từ-cloudflare)).

Cài `cloudflared` từ repo apt của Cloudflare (Debian / Ubuntu) để cập nhật dễ:

```bash
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" \
  | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update && sudo apt-get install cloudflared
```

Lưu ý khi dùng Tunnel:

- Nếu bật **Super Bot Fight Mode**, để nhóm **Definitely automated** ở **Allow**, nếu không WebSocket qua Tunnel có thể lỗi `websocket: bad handshake`.
- Chạy thêm một replica `cloudflared` (máy khác hoặc cùng máy) để không mất kết nối khi một tiến trình chết hay khi cập nhật.

### nginx: lấy IP thật và không lộ certificate

Sau Cloudflare, địa chỉ nginx thấy là IP của Cloudflare (hoặc `127.0.0.1` khi dùng Tunnel). Mọi rate limit, log, ban IP ở origin đều sai cho đến khi khôi phục IP thật:

```nginx
# /etc/nginx/conf.d/cloudflare-realip.conf
# Dải IP lấy từ https://www.cloudflare.com/ips — kiểm tra lại trước khi dùng
set_real_ip_from 173.245.48.0/20;
set_real_ip_from 103.21.244.0/22;
set_real_ip_from 103.22.200.0/22;
set_real_ip_from 103.31.4.0/22;
set_real_ip_from 141.101.64.0/18;
set_real_ip_from 108.162.192.0/18;
set_real_ip_from 190.93.240.0/20;
set_real_ip_from 188.114.96.0/20;
set_real_ip_from 197.234.240.0/22;
set_real_ip_from 198.41.128.0/17;
set_real_ip_from 162.158.0.0/15;
set_real_ip_from 104.16.0.0/13;
set_real_ip_from 104.24.0.0/14;
set_real_ip_from 172.64.0.0/13;
set_real_ip_from 131.0.72.0/22;
set_real_ip_from 2400:cb00::/32;
set_real_ip_from 2606:4700::/32;
set_real_ip_from 2803:f800::/32;
set_real_ip_from 2405:b500::/32;
set_real_ip_from 2405:8100::/32;
set_real_ip_from 2a06:98c0::/29;
set_real_ip_from 2c0f:f248::/32;
# Khi dùng Tunnel với Service URL là localhost: cloudflared kết nối từ 127.0.0.1 / ::1
# (cloudflared chạy máy khác thì thay bằng IP của máy đó)
set_real_ip_from 127.0.0.1;
set_real_ip_from ::1;
real_ip_header CF-Connecting-IP;
```

Kiểm tra: gửi một request qua domain, xem access log có ra IP của bạn không. Nên log thêm `$realip_remote_addr` (địa chỉ kết nối thực tế) để biết request đến từ Cloudflare hay từ ngoài.

Chỉ tin `CF-Connecting-IP` khi request thực sự đến từ Cloudflare. Nếu origin vẫn nhận kết nối từ internet, kẻ tấn công tự đặt header này để giả IP. Đó là lý do `set_real_ip_from` chỉ liệt kê IP Cloudflare và cần firewall ở B1.

Không trả certificate cho request vào bằng IP hoặc hostname lạ — máy quét Internet sẽ không gắn được IP với domain của bạn:

```nginx
server {
    listen 443 ssl default_server;
    listen [::]:443 ssl default_server;
    ssl_reject_handshake on;      # nginx >= 1.19.4
}
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    return 444;                   # đóng kết nối, không trả gì
}
```

Rate limit và timeout ở origin (lớp thứ hai, sau khi đã có IP thật):

```nginx
# Trong khối http {}
limit_req_zone  $binary_remote_addr zone=perip:10m rate=20r/s;
limit_req_zone  $binary_remote_addr zone=login:10m rate=5r/m;
limit_conn_zone $binary_remote_addr zone=conn:10m;
limit_req_status  429;   # mặc định là 503, dễ nhầm với lỗi origin
limit_conn_status 429;

# Thêm vào server {} hiện có của site, không tạo server mới
server {
    client_header_timeout 10s;
    client_body_timeout   10s;
    keepalive_timeout     30s;
    limit_conn conn 50;

    location / {
        limit_req zone=perip burst=40 nodelay;
    }
    location = /login {
        limit_req zone=login burst=5 nodelay;
    }
}
```

### Cấu hình Cloudflare nền

- SSL mode **Full (strict)**.
- Bot Fight Mode (Free) hoặc Super Bot Fight Mode (Pro+). Bot Fight Mode có thể challenge API và app mobile, và không chừa ra được bằng custom rule — kiểm tra kỹ trước khi bật nếu bạn có API công khai.
- WAF Managed Rules bật (Free có Free Managed Ruleset; Cloudflare Managed Ruleset từ Pro).
- HTTP DDoS ruleset luôn bật và không tắt được. Free / Pro / Business có một override để chỉnh sensitivity và action cho toàn zone; chỉ hạ sensitivity khi bị chặn nhầm.
- Cache Rules cho tài nguyên tĩnh với TTL dài.

---

## Phụ lục 3 — Lỗi hay gặp khi xử lý

| Triệu chứng | Nguyên nhân khả dĩ | Xử lý |
|---|---|---|
| Đã bật Under Attack nhưng origin vẫn quá tải | Traffic đánh thẳng vào IP origin | [Nhánh B](#nhánh-b--tấn-công-đánh-thẳng-vào-origin) |
| App mobile, webhook, health check hỏng sau khi bật Under Attack | Challenge không vượt được với client không phải trình duyệt | Configuration Rule tắt I'm Under Attack cho path API ([A1](#a1-phút-2-bật-under-attack-mode)), hoặc thay bằng rule R1 |
| Custom rule không khớp request nào | Sai cú pháp (`starts_with`, set có dấu phẩy), path phân biệt hoa thường, sai hostname | Xem bảng lỗi expression ở Phụ lục 1; thử expression trên dashboard |
| Request khớp rule nhưng không bị chặn | Có rule Skip hoặc IP Access Rule Allow đứng trước | Xem Security Events: rule nào đã xử lý request |
| Rate limiting không chặn | Botnet trải rộng, mỗi IP dưới ngưỡng; counter tính riêng từng data center | Chuyển sang chặn theo signature (A3) |
| Người dùng thật bị rate limit hàng loạt ở Cloudflare | Ngưỡng thấp, nhiều người dùng sau cùng một NAT | Tăng ngưỡng; Business+ dùng IP có hỗ trợ NAT |
| Người dùng thật bị rate limit hàng loạt ở nginx | nginx đếm theo IP Cloudflare hoặc `127.0.0.1` | Cấu hình realip (Phụ lục 2) hoặc tắt rate limit ở nginx |
| Origin trả 521 ngay sau khi chạy script firewall | Đang dùng Tunnel nhưng chạy script allowlist, hoặc dải IP tải thiếu | Kiểm tra `nft list table inet cf_guard`; với Tunnel, dùng lệnh đóng 80/443 |
| Googlebot bị chặn, traffic SEO giảm | Rule chặn rộng thiếu `not cf.client.bot` | Thêm điều kiện, rà lại các rule tạm |
| WebSocket qua Tunnel lỗi `bad handshake` | Super Bot Fight Mode chặn "Definitely automated" | Để nhóm đó ở Allow |
| Không thêm được rule mới | Hết quota rule của plan (Free: 5 custom, 1 rate limit) | Sửa rule cũ; gộp điều kiện bằng `or` |
| Tấn công quay lại ngay sau khi đổi IP | Chưa bịt chỗ lộ | [B3](#b3-bịt-chỗ-lộ-ip-rồi-mới-đổi-ip) |

---

## Nguồn

Tài liệu chính thức của Cloudflare (developers.cloudflare.com) và nginx, đối chiếu 11/2026.

| Chủ đề | URL |
|---|---|
| Under Attack Mode | [developers.cloudflare.com/fundamentals/reference/under-attack-mode/](https://developers.cloudflare.com/fundamentals/reference/under-attack-mode/) |
| Security Level (deprecation 03/2025) | [developers.cloudflare.com/waf/tools/security-level/](https://developers.cloudflare.com/waf/tools/security-level/) |
| Threat score luôn bằng 0 | [developers.cloudflare.com/ruleset-engine/rules-language/fields/reference/cf.threat_score/](https://developers.cloudflare.com/ruleset-engine/rules-language/fields/reference/cf.threat_score/) |
| Security Analytics | [developers.cloudflare.com/waf/analytics/security-analytics/](https://developers.cloudflare.com/waf/analytics/security-analytics/) |
| Security Events | [developers.cloudflare.com/waf/analytics/security-events/](https://developers.cloudflare.com/waf/analytics/security-events/) |
| HTTP DDoS managed ruleset | [developers.cloudflare.com/ddos-protection/managed-rulesets/http/](https://developers.cloudflare.com/ddos-protection/managed-rulesets/http/) |
| Network-layer DDoS | [developers.cloudflare.com/ddos-protection/managed-rulesets/network/](https://developers.cloudflare.com/ddos-protection/managed-rulesets/network/) |
| DDoS alerts | [developers.cloudflare.com/ddos-protection/reference/alerts/](https://developers.cloudflare.com/ddos-protection/reference/alerts/) |
| Lỗi 520–524 | [developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/) |
| Thứ tự xử lý các tính năng bảo mật | [developers.cloudflare.com/waf/feature-interoperability/](https://developers.cloudflare.com/waf/feature-interoperability/) |
| Custom rules | [developers.cloudflare.com/waf/custom-rules/](https://developers.cloudflare.com/waf/custom-rules/) |
| Skip options | [developers.cloudflare.com/waf/custom-rules/skip/options/](https://developers.cloudflare.com/waf/custom-rules/skip/options/) |
| Rate limiting rules | [developers.cloudflare.com/waf/rate-limiting-rules/](https://developers.cloudflare.com/waf/rate-limiting-rules/) |
| Rate limiting parameters | [developers.cloudflare.com/waf/rate-limiting-rules/parameters/](https://developers.cloudflare.com/waf/rate-limiting-rules/parameters/) |
| Cách rate limiting đếm request | [developers.cloudflare.com/waf/rate-limiting-rules/request-rate/](https://developers.cloudflare.com/waf/rate-limiting-rules/request-rate/) |
| Tạo rate limiting rule qua API | [developers.cloudflare.com/waf/rate-limiting-rules/create-api/](https://developers.cloudflare.com/waf/rate-limiting-rules/create-api/) |
| Thêm rule vào phase ruleset | [developers.cloudflare.com/ruleset-engine/basic-operations/add-rule-phase-rulesets/](https://developers.cloudflare.com/ruleset-engine/basic-operations/add-rule-phase-rulesets/) |
| Rules language: operators, functions, fields | [developers.cloudflare.com/ruleset-engine/rules-language/](https://developers.cloudflare.com/ruleset-engine/rules-language/) |
| `cf.client.bot` | [developers.cloudflare.com/ruleset-engine/rules-language/fields/reference/cf.client.bot/](https://developers.cloudflare.com/ruleset-engine/rules-language/fields/reference/cf.client.bot/) |
| IP Access Rules | [developers.cloudflare.com/waf/tools/ip-access-rules/](https://developers.cloudflare.com/waf/tools/ip-access-rules/) |
| Phát hiện response bị challenge (`cf-mitigated`) | [developers.cloudflare.com/cloudflare-challenges/challenge-types/challenge-pages/detect-response/](https://developers.cloudflare.com/cloudflare-challenges/challenge-types/challenge-pages/detect-response/) |
| Bot Fight Mode | [developers.cloudflare.com/bots/get-started/bot-fight-mode/](https://developers.cloudflare.com/bots/get-started/bot-fight-mode/) |
| Super Bot Fight Mode | [developers.cloudflare.com/bots/get-started/super-bot-fight-mode/](https://developers.cloudflare.com/bots/get-started/super-bot-fight-mode/) |
| Configuration Rules (I'm Under Attack theo path) | [developers.cloudflare.com/rules/configuration-rules/settings/](https://developers.cloudflare.com/rules/configuration-rules/settings/) |
| Cache Rules | [developers.cloudflare.com/cache/how-to/cache-rules/](https://developers.cloudflare.com/cache/how-to/cache-rules/) |
| Edge Cache TTL tối thiểu theo plan | [developers.cloudflare.com/cache/how-to/edge-browser-cache-ttl/](https://developers.cloudflare.com/cache/how-to/edge-browser-cache-ttl/) |
| Origin Cache-Control (`s-maxage`) | [developers.cloudflare.com/cache/concepts/cache-control/](https://developers.cloudflare.com/cache/concepts/cache-control/) |
| Spectrum theo plan | [developers.cloudflare.com/spectrum/protocols-per-plan/](https://developers.cloudflare.com/spectrum/protocols-per-plan/) |
| Inline list và header trong expression | [developers.cloudflare.com/ruleset-engine/rules-language/values/](https://developers.cloudflare.com/ruleset-engine/rules-language/values/) |
| Waiting Room plans | [developers.cloudflare.com/waiting-room/plans/](https://developers.cloudflare.com/waiting-room/plans/) |
| Protect your origin server | [developers.cloudflare.com/fundamentals/security/protect-your-origin-server/](https://developers.cloudflare.com/fundamentals/security/protect-your-origin-server/) |
| Authenticated Origin Pulls | [developers.cloudflare.com/ssl/origin-configuration/authenticated-origin-pull/](https://developers.cloudflare.com/ssl/origin-configuration/authenticated-origin-pull/) |
| Cloudflare Tunnel (remote) | [developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel/](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel/) |
| Cloudflare Tunnel (cài `cloudflared`, cấu hình local) | [developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/create-local-tunnel/](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/create-local-tunnel/) |
| Tunnel token (`cloudflared service install`) | [developers.cloudflare.com/tunnel/reference/tunnel-tokens/](https://developers.cloudflare.com/tunnel/reference/tunnel-tokens/) |
| Dải IP Cloudflare | [cloudflare.com/ips](https://www.cloudflare.com/ips/) |
| nginx `return 444` | [nginx.org/en/docs/http/ngx_http_rewrite_module.html](https://nginx.org/en/docs/http/ngx_http_rewrite_module.html) |
| nginx realip | [nginx.org/en/docs/http/ngx_http_realip_module.html](https://nginx.org/en/docs/http/ngx_http_realip_module.html) |
| nginx `ssl_reject_handshake` | [nginx.org/en/docs/http/ngx_http_ssl_module.html](https://nginx.org/en/docs/http/ngx_http_ssl_module.html) |
| nginx `limit_req`, `limit_conn` | [nginx.org/en/docs/http/ngx_http_limit_req_module.html](https://nginx.org/en/docs/http/ngx_http_limit_req_module.html) |
