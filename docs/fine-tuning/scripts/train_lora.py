import json
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from ft_data import load_jsonl, prepare

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def count_tokens(tok):
    # transformers 5 trả dict khi tokenize: len() của dict là số khoá, không phải số token
    return lambda rec: len(tok.apply_chat_template(rec["messages"], tokenize=True, return_dict=True)["input_ids"])


def evaluate(model, tok, rows, max_new_tokens=64):
    """Exact match trên JSON đã parse, và tỉ lệ câu trả lời dừng bằng token kết thúc."""
    model.eval()
    hit = stopped = 0
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"][:-1], add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**prompt, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[0, prompt["input_ids"].shape[1]:]
        stopped += int(new[-1].item() in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")})
        text = tok.decode(new, skip_special_tokens=True).strip()
        try:
            hit += json.loads(text) == json.loads(r["messages"][-1]["content"])
        except json.JSONDecodeError:
            pass
    return hit / len(rows), stopped / len(rows)


def main(path="toy.jsonl", out="out_lora"):
    tok = AutoTokenizer.from_pretrained(BASE)
    records, errors = load_jsonl(path)
    splits, report = prepare(records, group_of=lambda r: r["template"],
                             count_tokens=count_tokens(tok), max_tokens=MAX_LEN, test=0.2, val=0.1)
    print("lỗi JSON", len(errors), {k: len(v) for k, v in report.items()},
          {k: len(v) for k, v in splits.items()})
    ds = {k: Dataset.from_list([{"messages": r["messages"]} for r in v]) for k, v in splits.items()}

    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEVICE)
    print("base     exact, stopped:", evaluate(base, tok, splits["test"]))

    args = SFTConfig(
        output_dir=out,
        assistant_only_loss=True,       # loss chỉ trên câu trả lời, gồm cả <|im_end|>
        max_length=MAX_LEN,             # khớp với bước lọc ở prepare(): không ví dụ nào bị cắt
        num_train_epochs=2,
        per_device_train_batch_size=8,
        learning_rate=1e-4,             # LoRA: cao hơn full fine-tune khoảng 10 lần
        lr_scheduler_type="constant",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,    # giữ checkpoint có eval_loss thấp nhất, không phải checkpoint cuối
        metric_for_best_model="eval_loss",
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=False,
        report_to=[],
        seed=0,
    )
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=base, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
                         processing_class=tok, peft_config=peft)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"tham số train: {trainable:,} / {total:,} ({trainable / total:.2%})")
    t0 = time.time()
    trainer.train()
    print(f"train {time.time() - t0:.0f}s")
    if DEVICE == "mps":
        print(f"MPS driver {torch.mps.driver_allocated_memory() / 2**30:.1f} GiB")
    for h in trainer.state.log_history:
        if "eval_loss" in h:
            print("epoch", h["epoch"], "eval_loss", round(h["eval_loss"], 4))
    trainer.save_model(out)              # chỉ lưu adapter

    tuned = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEVICE), out)
    print("lora     exact, stopped:", evaluate(tuned, tok, splits["test"]))
    merged = tuned.merge_and_unload()    # gộp adapter vào trọng số: phục vụ như model thường
    print("merged   exact, stopped:", evaluate(merged, tok, splits["test"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
