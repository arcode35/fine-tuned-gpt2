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

```text
.
├── main.py                # Training script: download data, fine-tune GPT-2, save final model
├── test.py                # Inference script: load final model and generate text
├── requirements.txt       # Python dependencies
└── (generated at runtime)
    gpt2_motives_run/
        checkpoints/       # Trainer output; not required for inference
        final_model/       # Saved model + tokenizer used by test.py
