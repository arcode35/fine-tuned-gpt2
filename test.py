import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "gpt2_motives_run/final_model"

tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

seed_text = "The motives of men are often"
inputs = tokenizer(seed_text, return_tensors="pt").to(device)

model.eval()
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_length=150,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Seed:", seed_text)
print("Generated:\n", generated)
