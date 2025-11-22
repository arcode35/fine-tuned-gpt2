# Fine-Tuned GPT-2 on *The Motives of Men*

This repo contains code to fine-tune a GPT-2 language model on a single Project Gutenberg book,  
**“The Motives of Men” by George A. Coe** (Gutenberg ID: 77278), and to generate text from the
fine-tuned model.

The assignment requirements:

- Use a transformer model (GPT-2) on an NLP task.
- Use exactly one English book from Project Gutenberg as the dataset.
- For this project, the task is **text generation**.
- Evaluate using **perplexity** on a held-out validation split.

---

## 1. Repository Structure

    .
    ├── main.py                # Training script: download data, fine-tune GPT-2, save final model
    ├── test.py                # Inference script: load final model and generate text
    ├── requirements.txt       # Python dependencies
    └── (generated at runtime)
        gpt2_motives_run/
            checkpoints/       # Trainer output; not required for inference
            final_model/       # Saved model + tokenizer used by test.py

The `gpt2_motives_run/` folder is created when you run `main.py`.

---

## 2. Environment Setup

Recommended:

- Python 3.10+ (3.11/3.12 are also fine)
- Virtual environment (venv or conda)

### 2.1. Create and activate a virtual environment

From the repo root:

    python -m venv venv

On Windows:

    venv\Scripts\activate

On Linux / macOS:

    source venv/bin/activate

### 2.2. Install dependencies

    pip install --upgrade pip
    pip install -r requirements.txt

`requirements.txt` includes:

- `torch`
- `transformers`
- `accelerate`
- `requests`
- `protobuf`
- `safetensors`

A GPU (e.g., RTX 3080) is optional. If CUDA is available, training will automatically use FP16;
otherwise it runs on CPU.

---

## 3. Training (`main.py`)

`main.py` does the following:

1. Downloads the text of *The Motives of Men* from Project Gutenberg via URL  
   `https://www.gutenberg.org/cache/epub/77278/pg77278.txt`.
2. Strips Gutenberg boilerplate using the standard markers  
   `*** START OF THE PROJECT GUTENBERG EBOOK` and  
   `*** END OF THE PROJECT GUTENBERG EBOOK`.
3. Normalizes newlines and trims whitespace.
4. Splits the cleaned text into:
   - 90% training characters  
   - 10% validation characters
5. Loads the GPT-2 tokenizer and model from HuggingFace.
6. Tokenizes the train/validation text and builds a custom `BookDataset`:
   - Token sequences are chopped into fixed-length blocks (default: 512 tokens).
7. Evaluates the **baseline** (pretrained GPT-2) on the validation split:
   - Baseline validation loss ≈ **3.8042**
   - Baseline validation perplexity ≈ **44.89**
8. Fine-tunes GPT-2 for **3 epochs** using HuggingFace `Trainer`:
   - Learning rate: `5e-5`
   - Batch size: 2
   - Block size: 512
   - Weight decay: `0.01`
   - Evaluation once per epoch
   - No intermediate checkpoint saving (`save_strategy="no"`).
9. Evaluates the fine-tuned model:
   - Fine-tuned validation loss ≈ **3.3628**
   - Fine-tuned validation perplexity ≈ **28.87**
10. Saves the final model and tokenizer to:
    - `gpt2_motives_run/final_model/`

### 3.1. Run training

From the repo root (with venv activated):

    python main.py

On first run, this will:

- Download GPT-2 weights and tokenizer.
- Download the book text.
- Train and evaluate.
- Create `gpt2_motives_run/final_model/`.

---

## 4. Inference (`test.py`)

`test.py` loads the saved model from `gpt2_motives_run/final_model/` and generates text from a seed prompt.

Default seed:

    The motives of men are often

Generation settings in `test.py`:

- `max_length = 150`
- `do_sample = True`
- `top_k = 50`
- `top_p = 0.95`
- `num_return_sequences = 1`
- `pad_token_id = tokenizer.eos_token_id`

### 4.1. Run inference

After `main.py` has been run at least once:

    python test.py

You should see output similar to:

    Seed: The motives of men are often
    Generated:
    The motives of men are often quite clear, and so we see that there are many examples in
    literature, in science, in history, and in the arts of both sexes.

    Let us pause, then, to reflect upon how we have arrived at this conclusion. Why does it
    seem that it is good to use these words?

    What is not good to use them?

    What is not the matter?

    Some say that we make use of words that are not of this nature; others that we simply
    add something to the general meaning, and use it as though we had a word of it. I do
    not hold these claims to be correct. One thing is clear, however; men have the advantage
    of knowing how to

---

## 5. Reproducibility Notes

- The code does **not** rely on any local data paths; the book is always downloaded from
  Project Gutenberg via URL.
- All model and tokenizer files used for inference are stored under
  `gpt2_motives_run/final_model/`.
- Baseline and fine-tuned metrics are computed with the same validation split and pipeline:
  - Baseline perplexity ≈ 44.89
  - Fine-tuned perplexity ≈ 28.87
- To re-run from scratch:
  1. (Optional) Delete `gpt2_motives_run/`.
  2. Create a fresh virtual environment.
  3. `pip install -r requirements.txt`
  4. `python main.py`
  5. `python test.py`

---

