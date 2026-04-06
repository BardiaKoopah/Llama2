# Intro
#### Here is my from scratch implementation of Llama 2 based on the paper [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288). Tokenizer was trained as a SentencePiece BPE model on [Wikitext-103](https://huggingface.co/datasets/wikitext), which was also used for both training and validation. I was able to get ~2.49 valid loss. Check out model params below for more details and also notes so that your computer doesn't get fried.

# Model Params
####
- **Layers (Transformer Blocks):** 8
- **d_model:** 512
- **num_heads (Attention Heads):** 8
- **KV Heads (Grouped Query Attention):** 4
- **Head Dimension (`d_model / num_heads`):** 64
- **Vocabulary Size:** 32,000
- **seq_len (Context Length):** 512
- **Batch Size:** 4 (effective 64 via gradient accumulation)
- **Optimizer:** AdamW (betas 0.9/0.95, eps 1e-5)
- **Learning Rate Schedule:**
  - Linear warmup over first 2000 updates
  - Cosine decay to 10% of max LR (3e-4)
- **Weight Decay:** 0.1
- **Gradient Clipping:** max norm 1.0
- **Mixed Precision:** float16 with GradScaler
- **Early Stopping:** patience 5, min delta 0.005
- **Inference:** Top-p sampling (p=0.9, temp=0.8)

# Notes
#### Same deal as always — the official Llama 2 paper has way bigger model configs, but this was trained on MPS (Apple Silicon) so I had to scale down. Batch size of 4 with gradient accumulation steps of 16 to simulate an effective batch size of 64, context length of 512. Even then, keep an eye on your memory — there's a commented out one-shot test block in train.py you can use to sanity check your setup before committing to a full run. If you can afford bigger, check out Table 2 in the Llama 2 paper for the other model sizes they used (up to 70B params). Checkpoints save every 2 epochs and the script auto-resumes from the latest one, so you won't lose progress if something crashes.
