"""Kiểm tra token nào được tính loss trong TRL SFTTrainer (không train, chỉ xử lý dữ liệu)."""
import logging
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer
from ft_data import prepare
from train_lora import BASE, count_tokens

logging.disable(logging.WARNING)
tok = AutoTokenizer.from_pretrained(BASE)


def processed(rows, **kw):
    args = SFTConfig(output_dir="out_t", max_steps=1, report_to=[], bf16=False, gradient_checkpointing=False, **kw)
    return SFTTrainer(model=BASE, args=args, train_dataset=Dataset.from_list(rows), processing_class=tok).train_dataset


def labelled(row):
    return tok.decode([i for i in row["labels"] if i != -100])


one = {"messages": [{"role": "system", "content": "Phân loại."}, {"role": "user", "content": "Huỷ đơn DH-1"},
                    {"role": "assistant", "content": '{"intent":"cancel"}'}]}

# 1. assistant_only_loss: chỉ câu trả lời và token kết thúc lượt được tính loss
assert labelled(processed([one], assistant_only_loss=True)[0]) == '\n{"intent":"cancel"}<|im_end|>\n'

# 2. mặc định với dữ liệu hội thoại: loss trên mọi token, gồm system và user
lab = labelled(processed([one])[0])
assert "Phân loại." in lab and "Huỷ đơn DH-1" in lab

# 3. nhiều lượt: mọi lượt assistant đều được tính, không chỉ lượt cuối
multi = {"messages": [{"role": "user", "content": "Đơn DH-1?"}, {"role": "assistant", "content": "Đang giao."},
                      {"role": "user", "content": "Bao giờ tới?"}, {"role": "assistant", "content": "Mai ạ."}]}
lab = labelled(processed([multi], assistant_only_loss=True)[0])
assert "Đang giao." in lab and "Mai ạ." in lab and "Bao giờ tới?" not in lab, lab

# 4. ví dụ dài hơn max_length bị cắt từ cuối (truncation_mode mặc định keep_start)
long = {"messages": [{"role": "user", "content": "máy giặt kêu to khi vắt. " * 40},
                     {"role": "assistant", "content": '{"intent":"warranty"}'}]}
row = processed([long], max_length=128)[0]
assert len(row["input_ids"]) == 128 and "warranty" not in labelled(row)        # học trên ví dụ không có đáp án
assert len(processed([long, one], max_length=128, assistant_only_loss=True)) == 1  # bị bỏ, không cảnh báo

# 5. prepare() với bộ đếm token loại ví dụ đó trước khi tới trainer
splits, rep = prepare([(1, long), (2, one)], group_of=lambda r: "g", count_tokens=count_tokens(tok), max_tokens=128)
assert rep["too_long"] == [1] and sum(len(v) for v in splits.values()) == 1
print("MASKING OK")
