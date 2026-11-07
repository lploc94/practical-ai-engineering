"""Capacity planning cho một dịch vụ LLM: từ số đo open-loop của một replica và dự báo traffic,
tính số replica cần cho giờ đỉnh."""
import math


def max_rate_within_slo(curve, slo_ttft_p95):
    """`curve`: các cặp (request/giây, TTFT p95 tính bằng giây) đo bằng tải open-loop trên MỘT
    replica. Trả về mức tải lớn nhất đã đo mà TTFT p95 còn trong SLO. Không nội suy và dừng ở
    điểm vượt SLO đầu tiên: sau điểm gãy, hàng đợi tăng theo thời gian và số đo không lặp lại."""
    best = None
    for rate, ttft_p95 in sorted(curve):
        if ttft_p95 > slo_ttft_p95:
            break
        best = rate
    if best is None:
        raise ValueError("mức tải thấp nhất đã đo cũng vượt SLO: đổi cấu hình hoặc đo ở mức thấp hơn")
    return best


def peak_rate(requests_per_day, peak_to_average):
    """Tốc độ đến ở giờ đỉnh. `peak_to_average` lấy từ traffic thật (giờ đỉnh / trung bình ngày)."""
    return requests_per_day / 86_400 * peak_to_average


def replicas_needed(peak, per_replica, *, headroom=0.25, spare=1):
    """`headroom`: phần công suất mỗi replica để trống cho dao động trong giờ đỉnh và cho khoảng
    thời gian autoscaler chưa kịp thêm máy. `spare`: replica dự phòng khi một máy chết hoặc đang
    deploy (N+1). Kết quả là số replica phải chạy ở giờ đỉnh."""
    if not 0 <= headroom < 1:
        raise ValueError("headroom phải trong [0, 1)")
    return math.ceil(peak / (per_replica * (1 - headroom))) + spare


def in_flight(rate, seconds_in_system):
    """Định luật Little: số request đang nằm trong hệ thống = tốc độ đến × thời gian mỗi request ở
    lại. Dùng để đặt số slot, `max_num_seqs`, connection pool và giới hạn concurrency của proxy."""
    return rate * seconds_in_system
