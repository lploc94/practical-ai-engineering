from capacity import in_flight, max_rate_within_slo, peak_rate, replicas_needed

curve = [(0.1, 1.2), (0.2, 1.4), (0.3, 2.0), (0.4, 3.5), (0.5, 9.0), (0.6, 40.0)]
# 1. lấy điểm đã đo lớn nhất còn trong SLO, không nội suy
assert max_rate_within_slo(curve, 2.0) == 0.3
assert max_rate_within_slo(curve, 3.0) == 0.3
# 2. dừng ở điểm vượt đầu tiên, kể cả khi một điểm sau đó (đo nhiễu) lại thấp
assert max_rate_within_slo(curve + [(0.7, 1.0)], 2.0) == 0.3
# 3. ngay điểm thấp nhất đã vượt: báo lỗi
try:
    max_rate_within_slo(curve, 1.0)
    raise AssertionError("phải báo lỗi")
except ValueError:
    pass
# 4. 200.000 request/ngày, giờ đỉnh gấp 3 trung bình: khoảng 6,9 request/giây
assert round(peak_rate(200_000, 3), 2) == 6.94
# 5. replica: 6,94 / (0,3 × 0,75) = 30,8 -> 31, cộng 1 dự phòng
assert replicas_needed(6.94, 0.3) == 32
assert replicas_needed(6.94, 0.3, headroom=0, spare=0) == 24
try:
    replicas_needed(1, 1, headroom=1)
    raise AssertionError("phải báo lỗi")
except ValueError:
    pass
# 6. Little: 0,4 request/giây, mỗi request ở 10 giây -> 4 request đang chạy
assert in_flight(0.4, 10) == 4
print("CAPACITY OK")
