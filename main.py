import math
from pathlib import Path

import requests
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# -------------------------
# 1. Download and clean data
# -------------------------

url = "https://www.gutenberg.org/cache/epub/77278/pg77278.txt"
raw_text = requests.get(url).text

start_token = "*** START OF THE PROJECT GUTENBERG EBOOK"
end_token = "*** END OF THE PROJECT GUTENBERG EBOOK"
start_idx = raw_text.find(start_token)
end_idx = raw_text.find(end_token)

if start_idx != -1 and end_idx != -1:
    text = raw_text[start_idx:end_idx]
else:
    raise RuntimeError("Couldn't find Gutenberg START/END markers in text.")

text = text.replace("\r\n", "\n").strip()

print("Word count (cleaned):", len(text.split()))

# Train/val split by characters (90/10)
split_idx = int(len(text) * 0.9)
train_text = text[:split_idx]
val_text = text[split_idx:]

print("Train chars:", len(train_text))
print("Val chars:", len(val_text))

# -------------------------
# 2. Tokenizer and model
# -------------------------

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

def encode_text(txt: str):
    return tokenizer(
        txt,
        return_tensors=None,
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]

train_ids = encode_text(train_text)
val_ids = encode_text(val_text)

print("Train tokens:", len(train_ids))
print("Val tokens:", len(val_ids))

# -------------------------
# 3. Dataset definition
# -------------------------

class BookDataset(Dataset):
    def __init__(self, token_ids, block_size: int):
        self.block_size = block_size
        total_len = (len(token_ids) // block_size) * block_size
        token_ids = token_ids[:total_len]
        self.data = torch.tensor(token_ids).view(-1, block_size)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        return {"input_ids": x, "labels": x}

block_size = 512 

train_dataset = BookDataset(train_ids, block_size)
val_dataset = BookDataset(val_ids, block_size)

print("Train examples:", len(train_dataset))
print("Val examples:", len(val_dataset))

# -------------------------
# 4. Training setup
# -------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Base run directory (relative)
base_dir = Path("gpt2_motives_run")
checkpoints_dir = base_dir / "checkpoints"
final_model_dir = base_dir / "final_model"

# Create dirs (no error if they exist)
checkpoints_dir.mkdir(parents=True, exist_ok=True)
final_model_dir.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(checkpoints_dir),
    overwrite_output_dir=True,
    num_train_epochs=3,              
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="epoch",    
    save_strategy="no",              
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Baseline evaluation before training
baseline_results = trainer.evaluate()
baseline_loss = baseline_results["eval_loss"]
baseline_ppl = math.exp(baseline_loss)
print("Baseline val loss:", baseline_loss)
print("Baseline val perplexity:", baseline_ppl)

# -------------------------
# 5. Train and evaluate
# -------------------------

trainer.train()

eval_results = trainer.evaluate()
val_loss = eval_results["eval_loss"]
val_ppl = math.exp(val_loss)
print("Fine-tuned val loss:", val_loss)
print("Fine-tuned val perplexity:", val_ppl)

# -------------------------
# 6. Save final model + tokenizer for inference
# -------------------------

model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print("Final model saved to:", final_model_dir.resolve())
