# Hướng dẫn làm hình minh họa

Quy tắc cho ảnh tĩnh và ảnh động trong các tài liệu của repo. Mục tiêu là hình giúp người đọc hiểu nhanh hơn, không phải để trang trí.

## Nghiên cứu nói gì

Các quy tắc bên dưới rút ra từ nghiên cứu về học qua hình ảnh (multimedia learning):

- Animation có giúp ích hơn ảnh tĩnh, nhưng hiệu ứng nhỏ: meta-analysis 61 nghiên cứu (N = 7036) cho Hedges's g = 0.23. Hiệu ứng lớn hơn khi animation tự chạy theo nhịp cố định (g = 0.31), có lời thuyết minh (g = 0.34), hoặc khi bài học không kèm văn bản nào (g = 0.88) ([Berney & Bétrancourt, 2016](https://doi.org/10.1016/j.compedu.2016.06.005)).
- Animation có ích khi chính sự thay đổi là thứ cần hiểu. Animation chỉ để trang trí gần như không có tác dụng ([Höffler & Leutner, 2007](https://doi.org/10.1016/j.learninstruc.2007.09.013)).
- Nhiều thí nghiệm thấy animation hơn ảnh tĩnh thực ra là so sánh không công bằng: bản animation chứa nhiều thông tin hơn. Animation chỉ hiệu quả khi chuyển động khớp với cấu trúc của khái niệm (*congruence*) và đủ chậm, đủ đơn giản để người xem kịp nắm (*apprehension*) ([Tversky, Morrison & Bétrancourt, 2002](https://doi.org/10.1006/ijhc.2002.1017)).
- Tín hiệu dẫn mắt như màu, mũi tên, highlight giúp người học nhìn đúng chỗ ([Alpizar, Adesope & Wong, 2020](https://doi.org/10.1007/s11423-020-09748-7)).
- Chữ đặt ngay cạnh phần hình nó mô tả và hiện cùng lúc với phần đó thì dễ hiểu hơn ([Ginns, 2006](https://doi.org/10.1016/j.learninstruc.2006.10.001); [Schroeder & Cenkci, 2018](https://doi.org/10.1007/s10648-018-9435-9)).
- Chia nội dung thành từng đoạn có điểm dừng giúp tránh quá tải ([Rey et al., 2019](https://doi.org/10.1007/s10648-018-9456-4)). Thông tin hiện ra rồi biến mất buộc người xem phải tự nhớ ([Castro-Alonso et al., 2021](https://doi.org/10.1007/s10648-021-09606-9)).
- Nội dung chuyển động tự chạy, kéo dài hơn 5 giây và hiện song song với nội dung khác phải có cách tạm dừng ([WCAG 2.2, tiêu chí 2.2.2](https://www.w3.org/WAI/WCAG22/Understanding/pause-stop-hide.html)). Nội dung không được nhấp nháy quá 3 lần mỗi giây ([tiêu chí 2.3.1](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html)).

## Chọn loại hình

| Nội dung | Dùng |
|---|---|
| Cấu trúc, quan hệ, luồng không cần nhấn mạnh thứ tự thời gian | Mermaid hoặc SVG tĩnh |
| Diễn biến theo thời gian, nhân quả, tuần tự so với song song | SVG động |
| Kết quả chạy lệnh trong terminal | GIF quay bằng `vhs` hoặc `asciinema` + `agg`, dưới vài MB |
| Bảng số liệu, so sánh | Bảng Markdown, không vẽ thành hình |

Chỉ làm ảnh động khi bỏ chuyển động đi thì người đọc mất một ý quan trọng.

## Quy tắc thiết kế

1. **Congruence.** Việc gì xảy ra đồng thời thì hiện đồng thời, việc gì tuần tự thì hiện từng bước. Chuyển động phải khớp với cơ chế thật.
2. **Mỗi bước chỉ có một thay đổi.** Giữa các bước có khoảng dừng. Trạng thái cuối giữ ít nhất 2–3 giây trước khi lặp lại, để người xem quan sát toàn cảnh. Chú thích đổi theo từng bước phải hiện đủ lâu để đọc hết: không quá khoảng 15 ký tự mỗi giây, mức thường dùng cho phụ đề.
3. **Chỉ vẽ thứ phục vụ ý chính.** Không icon trang trí, gradient, bóng đổ hay easing cầu kỳ. Khi nhiều đường nối làm hình rối, gom chúng thành một ký hiệu như khung bao hoặc ngoặc.
4. **Màu có nghĩa và nhất quán giữa các tài liệu** (bảng màu bên dưới). Không dùng màu chỉ để làm đẹp.
5. **Chữ ngắn, đặt cạnh phần hình nó mô tả.** Mỗi pha có một tiêu đề và tối đa hai dòng chú thích. Phần giải thích dài để trong đoạn văn, không chép vào hình.
6. **Hình là phần bổ sung.** Đoạn văn phải đủ ý khi hình không tải được. Alt text nêu ý chính của hình, không mô tả hình thức.
7. **Tôn trọng `prefers-reduced-motion`.** Khi người dùng bật giảm chuyển động, ảnh động hiển thị một khung tĩnh tóm tắt cả quá trình. GitHub hiển thị SVG như ảnh nên không thêm được nút tạm dừng, vì vậy đây là cách gần nhất để đáp ứng WCAG 2.2.2.
8. **Không nhấp nháy.** Mỗi phần tử đổi trạng thái tối đa một lần mỗi giây.

## Bảng màu

Dùng chung cho mọi hình. Màu lấy theo bảng màu của GitHub nên hợp với cả giao diện sáng và tối.

| Ý nghĩa | Sáng (viền / nền) | Tối (viền / nền) |
|---|---|---|
| Đầu vào, dữ liệu có sẵn, cache hit | `#0969da` / `#ddf4ff` | `#4493f8` / `#0c2d6b` |
| Đang sinh ra, đang xử lý, phần mới | `#bc4c00` / `#fff1e5` | `#db6d28` / `#3d1e00` |
| Phải tính lại, lỗi, cache miss | `#cf222e` / `#ffebe9` | `#f85149` / `#490202` |
| Thành công, kết quả cuối | `#1a7f37` / `#dafbe1` | `#3fb950` / `#033a16` |
| Trung tính, chưa có dữ liệu | `#d1d9e0` (viền nét đứt) | `#3d444d` (viền nét đứt) |

Bảng màu đã được kiểm tra bằng validator của skill `dataviz` (độ sáng, độ tách màu cho người mù màu và người nhìn bình thường, tương phản với nền). Kết quả quy định cách ghép màu:

- **Xanh dương đi với cam hoặc đỏ:** đạt mọi tiêu chí, dùng làm cặp chính.
- **Không đặt cam cạnh đỏ** trong cùng một hình: ΔE chỉ khoảng 8 với người nhìn bình thường và dưới 2 với người mù màu đỏ–lục.
- **Không đặt đỏ cạnh xanh lá** trong cùng một hình nếu không có mã hoá phụ: người mù màu đỏ–lục không phân biệt được.
- **Trạng thái "phải tính lại" luôn có sọc chéo** (pattern `hatch` trong `svgkit`) và nhãn, để không chỉ dựa vào màu.
- Mỗi hình dùng tối đa hai màu có nghĩa, cộng với màu trung tính.

## Kỹ thuật cho SVG trên GitHub

GitHub hiển thị SVG như một ảnh, nên:

- Chỉ dùng CSS `@keyframes` hoặc SMIL bên trong file. JavaScript không chạy, `:hover` và click không có tác dụng.
- Không tải font hay ảnh từ ngoài. Dùng font hệ thống: `-apple-system, "Segoe UI", Helvetica, Arial, sans-serif` cho chữ và `ui-monospace, Menlo, Consolas, monospace` cho code.
- Không dùng `<foreignObject>`.
- Đổi màu theo giao diện bằng biến CSS và `@media (prefers-color-scheme: dark)` trong file SVG.
- Thêm `role="img"` và `aria-label` cho thẻ `<svg>`.
- Nhúng bằng đường dẫn tương đối: `![alt text](images/ten-hinh.svg)`.
- Animation lặp vô hạn nên dùng keyframes trải hết chu kỳ, vì `animation-delay` chỉ có tác dụng ở lần chạy đầu.

## Biểu đồ

- Chọn dạng biểu đồ theo việc cần thể hiện: độ lớn theo thời gian hoặc theo thứ tự dùng cột, so hai chiến lược dùng hai biểu đồ nhỏ cùng trục tung.
- Một trục tung duy nhất. Hai đại lượng khác thang đo thì tách thành hai biểu đồ.
- Có chú giải khi có từ hai chuỗi trở lên, kèm nhãn trực tiếp cho vài giá trị đáng chú ý (đầu, cuối, tổng). Không ghi số lên mọi cột.
- Chữ dùng màu chữ, không dùng màu của chuỗi dữ liệu.
- Lưới và trục mờ. Khe 2px giữa các đoạn cột chồng.
- Ghi rõ khi số liệu là giả định, và đặt giả định trong docstring của script để kiểm tra lại được.
- GitHub hiển thị SVG như ảnh nên không có hover hay tooltip. Thông tin cần đọc chính xác phải nằm trong tiêu đề, nhãn hoặc đoạn văn.

## Tổ chức file

- Hình đặt trong `docs/<chủ-đề>/images/`, tên file `kebab-case`.
- Hình được sinh bằng script thì để script cạnh hình (ví dụ `make_prefill_decode.py`), để sau này sửa script rồi chạy lại.
- Script dùng chung module [`docs/assets/svgkit.py`](./assets/svgkit.py): bảng màu, kiểu chữ, đầu mũi tên, pattern sọc chéo, class `Anim` tạo keyframes kèm khung tĩnh cho `prefers-reduced-motion`. Sửa màu ở đó thì chạy lại mọi script `make_*.py`.
- Trước khi commit, kiểm tra hình ở cả giao diện sáng, tối và chế độ giảm chuyển động. Có thể chụp từng thời điểm bằng Chrome headless: nhúng SVG vào một file HTML, thêm CSS `*{animation-play-state:paused!important;animation-delay:-<giây>s!important}`, rồi chạy `--screenshot`.
