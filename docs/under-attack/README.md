# Xử lý tấn công DDoS & Flood với Cloudflare trước Server Vật lý (Origin)

Tài liệu playbook thực chiến dành cho trường hợp **server vật lý (bare metal / on-prem)** đặt sau Cloudflare. Cloudflare cung cấp lớp bảo vệ DDoS rất mạnh (L3/L4/L7 tự động), nhưng rủi ro lớn nhất là **origin IP bị lộ** hoặc traffic bypass Cloudflare → server vật lý trực tiếp nhận toàn bộ lượng tấn công.

Mục tiêu: Giảm thiểu thiệt hại nhanh nhất, khôi phục dịch vụ, và hạn chế lặp lại.

> **Phiên bản**: June 2026 | Ngôn ngữ: Việt (thuật ngữ kỹ thuật giữ nguyên tiếng Anh)

---

## TL;DR — Khi bị tấn công (5 phút đầu)

1. **Bật ngay "I'm Under Attack" mode** (global hoặc theo path).
2. **Kiểm tra Security > Events / Analytics** → xác định pattern (path, UA, ASN, country, method).
3. **Tạo Custom Rule hoặc Rate Limiting** dựa trên signature thực tế.
4. **Trên origin**: siết firewall chỉ cho phép Cloudflare IPs (nếu chưa).
5. **Nếu nghi ngờ IP origin bị lộ**: rotate origin IP ngay + cập nhật Cloudflare origin + rebuild tunnel.
6. Bật caching mạnh (nếu có thể) và theo dõi origin resource (CPU, connection, bandwidth).

**Mục tiêu**: Chặn tại Cloudflare. Origin chỉ nhận traffic sạch.

---

## 1. Chuẩn bị trước khi bị tấn công (Bắt buộc)

### 1.1 Ẩn origin IP hoàn toàn (quan trọng nhất)

| Phương pháp | Độ an toàn | Khuyến nghị | Ghi chú |
|-------------|------------|-------------|---------|
| **Cloudflare Tunnel** (`cloudflared`) | Rất cao | **Khuyến nghị mạnh** cho server vật lý | Không mở port công khai nào, outbound-only |
| **Authenticated Origin Pulls** (mTLS) | Rất cao | Rất tốt | Bắt buộc Full / Full (strict) SSL mode |
| **Proxy toàn bộ DNS** (orange cloud) + Firewall allow CF IPs | Trung bình | Tối thiểu bắt buộc | Dễ bị lộ nếu có DNS-only record hoặc historical data |
| HTTP Header / JWT validation | Trung bình | Bổ sung | Không thay thế được các biện pháp trên |

**Hành động bắt buộc:**

- Tất cả record web (A/AAAA/CNAME cho website, API) phải **proxied**.
- Xem lại mọi DNS record (SPF, mail, v.v.) để không leak IP.
- Sau khi onboard Cloudflare → **rotate origin IP** ngay (nếu có thể).
- Trên server vật lý: **chỉ cho phép traffic từ Cloudflare IP ranges** trên port 80/443 (và các port khác nếu dùng).

Lấy danh sách IP Cloudflare:

```bash
# IPv4
curl -s https://www.cloudflare.com/ips-v4

# IPv6
curl -s https://www.cloudflare.com/ips-v6
```

Ví dụ ufw (khuyến nghị):

```bash
# Cho phép toàn bộ dải Cloudflare
for ip in $(curl -s https://www.cloudflare.com/ips-v4); do
  sudo ufw allow from "$ip" to any port 80,443 proto tcp
done

# (Tương tự cho IPv6)
```

Hoặc dùng iptables / firewalld / nftables tương tự. Cập nhật định kỳ (script cron).

**Khuyến nghị cao nhất**: Dùng **Cloudflare Tunnel** → origin server không cần expose port nào ra internet.

#### Quick setup Cloudflare Tunnel (khuyến nghị cho server vật lý)

```bash
# 1. Cài cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb

# 2. Tạo tunnel (chạy 1 lần)
cloudflared tunnel login
cloudflared tunnel create my-origin

# 3. Config ~/.cloudflared/config.yml
# tunnel: <tunnel-id>
# credentials-file: /root/.cloudflared/<tunnel-id>.json
#
# ingress:
#   - hostname: api.example.com
#     service: http://localhost:8080
#   - service: http_status:404

# 4. Run as service
sudo cloudflared service install
```

Sau đó chỉ cần trỏ DNS record **proxied** đến tunnel (không cần public IP/port cho origin). Đây là cách an toàn nhất chống lộ IP.

### 1.2 Cấu hình Cloudflare cơ bản (luôn bật)

- **WAF Managed Rules** → đặc biệt **L7 HTTP DDoS** ruleset (bật mặc định).
- **Bot Fight Mode** (hoặc Super Bot Fight Mode nếu có).
- **Caching**:
  - Cache static assets lâu.
  - Cân nhắc Cache Everything + Edge Cache TTL cho một số endpoint (nếu idempotent).
- **Security Level**: Medium hoặc cao hơn (không để Low).
- **Rate Limiting Rules** cho các endpoint nhạy cảm:
  - `/login`, `/api/auth/*`, `/admin/*`, API write endpoints.
  - Dùng đặc tính đếm theo IP (hoặc phức tạp hơn nếu có plan cao).

### 1.3 Giám sát & Alert

- Bật **Origin Error Rate Alerts** (5xx cao).
- **Passive Origin Monitoring**.
- Setup **Health Checks** + Load Balancing nếu có nhiều origin.
- Log / metrics: Cloudflare + origin (nginx/access log, hệ thống).

---

## 2. Phát hiện tấn công

Dấu hiệu điển hình:

- Traffic tăng đột biến không giải thích được.
- Origin CPU / connection / bandwidth cao bất thường.
- Nhiều 5xx từ origin (521/522/523/524 nếu origin chết).
- Trong Cloudflare:
  - Security > Events có lượng request lớn với pattern giống nhau.
  - Analytics → top paths, user agents, countries, ASNs bất thường.

**Nơi kiểm tra nhanh:**

1. Cloudflare Dashboard → **Security → Events**
2. **Analytics → Security** (hoặc HTTP requests)
3. Origin monitoring (htop, `ss -s`, `netstat`, logs).

---

## 3. Playbook xử lý khi bị tấn công (Incident Response)

### Bước 1: Kích hoạt bảo vệ ngay

**Cách 1 (nhanh nhất)**: Bật **Under Attack Mode** toàn zone
- Account home → zone → Quick Actions → bật **Under Attack Mode**.

**Cách 2 (tốt hơn)**: Dùng **Configuration Rule** (Security Level = I'm Under Attack) cho path cụ thể.
- Ít ảnh hưởng toàn site hơn.

> Under Attack Mode sẽ show JS challenge ~5 giây cho visitor. Hiệu quả rất cao với hầu hết L7 flood script.

**Lưu ý**: Có thể chặn một số verified bot (Googlebot, healthcheck...). Dùng rule để skip cho `cf.client.bot == true` hoặc path đặc biệt.

### Bước 2: Phân tích signature tấn công

Vào Security Events, lọc và ghi nhận:

- URI path / method bị nhắm nhiều
- User-Agent đặc trưng
- ASN / Country tập trung
- Header bất thường
- Rate (req/s hoặc req/min)

### Bước 3: Tạo quy tắc chặn/challenge có mục tiêu

**Custom Rules** (WAF) — ví dụ thực tế (copy vào Expression Editor):

```text
# Chặn / challenge flood login (bỏ qua verified bot)
(http.request.uri.path eq "/login" or http.request.uri.path eq "/api/auth/login") and not cf.client.bot

# Chặn user-agent đặc trưng
http.user_agent contains "AttackBot" or http.user_agent contains "Go-http-client"

# Tạm chặn ASN (thay số)
ip.geoip.asnum in {12345 67890}

# Challenge theo path admin khi đang attack
starts_with(http.request.uri.path, "/admin") and not cf.client.bot
```

**Rate Limiting Rules**:
- Path `/login` hoặc `/api/*` → ví dụ: 20 req / 1 phút / IP → Block hoặc Managed Challenge 10 phút.
- Dùng "IP" làm characteristic cho DDoS cơ bản.

**IP Access Rules** (nếu cần):
- Block toàn bộ ASN hoặc country (dùng tạm thời, cẩn thận ảnh hưởng user hợp lệ).

**Mẹo**: Luôn thêm `and not cf.client.bot` khi tạo rule challenge/block để tránh ảnh hưởng Googlebot, Bing, health checks... (verified bots).

### Bước 4: Cứng hóa origin ngay lập tức

- Siết firewall chỉ cho Cloudflare IPs (nếu chưa).
- Restart/reload web server nếu cần (nginx -s reload).
- Tăng connection limits / worker processes tạm thời nếu server còn chịu được.
- Nếu dùng nginx: bật rate limit ở tầng origin như lớp sau (double protection).
- Tắt các endpoint không cần thiết tạm thời.

**Ví dụ rate limit nginx (lớp bảo vệ thứ hai):**

```nginx
limit_req_zone $binary_remote_addr zone=general:10m rate=50r/s;
limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;

server {
    # ...
    location / {
        limit_req zone=general burst=100 nodelay;
    }
    location /login {
        limit_req zone=login burst=3 nodelay;
    }
}
```

### Bước 5: Xử lý khi origin IP đã bị lộ

Dấu hiệu: Attack vẫn tiếp tục dù Cloudflare chặn tốt, hoặc log origin thấy nhiều IP không phải Cloudflare.

Hành động:
1. **Thay đổi origin IP** ngay (nếu provider hỗ trợ nhanh).
2. Cập nhật origin trong Cloudflare (hoặc recreate Tunnel).
3. Nếu dùng Tunnel: chỉ cần restart cloudflared với config mới (thường không lộ IP cũ).
4. Xóa cache DNS cũ nếu cần (Cloudflare purge + chờ TTL).

### Bước 6: Giảm tải origin

- Tăng caching mạnh (Edge + Browser).
- Bật **Waiting Room** cho các trang nhạy cảm (nếu có).
- Tắt một số tính năng nặng (search không cache, báo cáo realtime...).
- Scale origin tạm thời (nếu có nhiều máy) hoặc chuyển bớt traffic.

### Bước 7: Theo dõi & điều chỉnh liên tục

- Theo dõi Security Events mỗi 5-10 phút.
- Thêm rule khi thấy pattern mới.
- Theo dõi origin health real-time.
- Giao tiếp với team/user: "Chúng tôi đang xử lý, một số truy cập có thể chậm do challenge".

---

## 4. Cấu hình Cloudflare khuyến nghị (Production)

### Security Level & Under Attack

- Dùng **Configuration Rules** thay vì chỉ global setting khi có thể.
- Exempt verified bots và internal health checks.

### Rate Limiting (ví dụ thực tế)

Ví dụ cho login + API chung (Pro/Business trở lên):

- Rule 1: `http.request.uri.path contains "/login"` → 10 req / 1 phút → Block 5 phút
- Rule 2: `http.request.uri.path starts_with "/api/" and http.request.method in {"POST" "PUT" "DELETE"}` → 60 req / 1 phút → Managed Challenge

### WAF

- Giữ **Cloudflare Managed Ruleset** và **L7 DDoS** luôn ON.
- Tạo Custom Rules theo sự kiện thực tế thay vì đoán trước.

### Cache

- Static: Cache > 1 ngày hoặc lâu hơn.
- Dynamic nhưng idempotent: Edge Cache TTL + Cache Everything (cẩn thận với cookie/auth).

### Khác

- Bật **Always Use HTTPS**.
- **Full (strict)** SSL mode + Authenticated Origin Pulls (nếu không dùng Tunnel).
- Xem xét **Cloudflare Load Balancing** + nhiều origin cho redundancy.

---

## 5. Cứng hóa Server Vật lý (Origin Hardening)

### Firewall (ưu tiên chỉ Cloudflare)

Ví dụ script cập nhật CF IPs (chạy định kỳ):

```bash
#!/bin/bash
# /usr/local/bin/update-cf-ips.sh
IPS4=$(curl -s https://www.cloudflare.com/ips-v4)
IPS6=$(curl -s https://www.cloudflare.com/ips-v6)

# Ví dụ nftables hoặc iptables - flush và rebuild rules cho port 80/443
# (viết script phù hợp với distro của bạn)
```

### Web server (nginx ví dụ)

```nginx
# Rate limit cơ bản ở origin (lớp 2)
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

server {
    listen 80;
    server_name _;

    # Chỉ nhận từ Cloudflare (hoặc từ tunnel)
    # ...

    location /api/ {
        limit_req zone=api burst=20 nodelay;
        # ...
    }
}
```

### Cloudflare Tunnel (khuyến nghị)

- Cài `cloudflared`.
- Tạo tunnel → expose service nội bộ (ví dụ `http://localhost:8080`).
- Không mở port 80/443 ra ngoài.
- Dùng service systemd + auto-restart.

### Monitoring trên origin

- `htop`, `nload`, `ss -s`, `journalctl`.
- Export metric (node_exporter, nginx prometheus...) nếu có observability stack.

---

## 6. Các loại tấn công phổ biến

| Loại | Lớp | Cloudflare xử lý | Origin cần làm gì |
|------|-----|------------------|-------------------|
| Volumetric (UDP, SYN, ICMP) | L3/L4 | Tự động hấp thụ rất tốt | Giữ firewall chặt, không lộ IP |
| TCP flood | L4 | Tốt (có Spectrum cho bypass) | Giới hạn connection |
| HTTP GET/POST flood | L7 | Rất tốt nếu có WAF + rate limit | Under Attack + custom rule |
| Slowloris / Slow read | L7 | Tốt | Timeout ngắn, connection limit |
| Botnet với UA + IP thay đổi | L7 | Bot Fight + rate limit + custom rule | Phân tích Events để tạo rule |
| Cache busting (random query) | L7 | WAF + cache config | Cache rule chặt chẽ hơn |

**Lưu ý đặc biệt với server vật lý**: L3/L4 volumetric chỉ an toàn nếu IP không bị lộ. Nếu lộ → nhà cung cấp hosting / ISP thường phải null-route.

---

## 7. Sau tấn công (Post-Incident)

- Tắt Under Attack mode khi traffic ổn định.
- Review tất cả rule tạm thời → giữ lại những cái có giá trị, xóa rule quá rộng.
- Rotate origin IP nếu đã từng bị lộ (ngay cả sau khi hết attack).
- Cập nhật playbook từ sự kiện thực tế.
- Ghi nhận timeline, quyết định, tác động (post-mortem).
- Cân nhắc nâng cấp plan (Business/Enterprise) hoặc thêm tính năng (Bot Management, Advanced Rate Limiting, Magic Transit...) nếu attack lớn lặp lại.
- Kiểm tra lại toàn bộ origin exposure (DNS history, mail server, upload-from-URL, v.v.).

---

## Checklist nhanh

### Trước khi có attack
- [ ] Tất cả web record đã proxied
- [ ] Origin firewall chỉ cho Cloudflare IPs (hoặc dùng Tunnel)
- [ ] Đã rotate IP sau onboarding
- [ ] Có ít nhất 1-2 Rate Limiting rule quan trọng
- [ ] WAF Managed + Bot Fight bật
- [ ] Alerts cho origin error
- [ ] Có script cập nhật CF IPs

### Trong lúc bị tấn công
- [ ] Bật Under Attack mode
- [ ] Phân tích Events → ghi signature
- [ ] Tạo Custom Rule / Rate Limit phù hợp
- [ ] Siết firewall origin
- [ ] Kiểm tra origin IP leak → rotate nếu cần
- [ ] Tăng cache / giảm tải origin
- [ ] Theo dõi liên tục (5-15 phút/lần)

### Sau attack
- [ ] Tắt Under Attack mode
- [ ] Dọn rule tạm
- [ ] Post-mortem + cập nhật playbook
- [ ] Xem xét cải tiến dài hạn

---

## Tài liệu tham khảo chính thức

- [Protect your origin server](https://developers.cloudflare.com/fundamentals/security/protect-your-origin-server/)
- [Under Attack mode](https://developers.cloudflare.com/fundamentals/reference/under-attack-mode/)
- [Rate limiting rules](https://developers.cloudflare.com/waf/rate-limiting-rules/)
- Cloudflare IP ranges: https://www.cloudflare.com/ips/
- [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/)

---

**Lưu ý cuối**: Không có cấu hình nào là "set and forget" 100%. DDoS là cuộc chạy đua. Playbook này tập trung vào **hành động nhanh + quan sát thực tế** hơn là chỉ dựa vào mặc định.

Cập nhật tài liệu này mỗi khi có incident thực tế.
