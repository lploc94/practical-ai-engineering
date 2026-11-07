from thresholds import pick_threshold, _wilson_upper

# 1. tách được hoàn toàn: ngưỡng thấp nhất không chặn nhầm mẫu an toàn nào
p = pick_threshold([0.9, 0.8, 0.7, 0.2, 0.1, 0.05], [True, True, True, False, False, False], max_fpr=0.0)
assert p.threshold == 0.7 and p.recall == 1.0 and p.fpr == 0.0, p
# 2. chồng lấn: max_fpr 0 thì phải bỏ một mẫu vi phạm; max_fpr 1/3 thì bắt hết
s, y = [0.9, 0.6, 0.4, 0.7, 0.3, 0.1], [True, True, True, False, False, False]
p = pick_threshold(s, y, max_fpr=0.0)
assert p.threshold == 0.9 and abs(p.recall - 1 / 3) < 1e-9, p
p = pick_threshold(s, y, max_fpr=1 / 3)
assert p.threshold == 0.4 and p.recall == 1.0 and abs(p.fpr - 1 / 3) < 1e-9, p
# 3. không ngưỡng nào đạt
assert pick_threshold([0.9, 0.95], [True, False], max_fpr=0.0) is None
# 4. cận trên: 0 lỗi trên 100 mẫu an toàn vẫn để fpr thật tới khoảng 3,7%
assert 0.036 < _wilson_upper(0, 100) < 0.038, _wilson_upper(0, 100)
assert _wilson_upper(0, 1000) < 0.004
# 5. thiếu một trong hai lớp: lỗi
try:
    pick_threshold([0.5], [True], max_fpr=0.1); raise AssertionError
except ValueError:
    pass
print("THRESHOLDS OK")
