# save as app.py

import streamlit as st
import torch
from tokenizers import ByteLevelBPETokenizer
import os

# -----------------------------
# CONFIG
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
MODEL_PATH = "gpt_shakespeare.pt"
TOKENIZER_DIR = "tokenizer"  # folder containing vocab.json & merges.txt

# -----------------------------
# LOAD TOKENIZER
# -----------------------------
tokenizer = ByteLevelBPETokenizer(
    os.path.join(TOKENIZER_DIR, "vocab.json"),
    os.path.join(TOKENIZER_DIR, "merges.txt")
)
vocab_size = tokenizer.get_vocab_size()

# -----------------------------
# DEFINE MODEL
# -----------------------------
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.token_emb(x)
        logits = self.lm_head(x)
        return logits


# -----------------------------
# LOAD MODEL
# -----------------------------
model = MiniGPT(vocab_size, embed_dim=256)
# model = MiniGPTWithAttention(vocab_size).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# TEXT GENERATION FUNCTION
# -----------------------------
def generate(prompt, max_new_tokens=200, temperature=0.7):
    model.eval()
    ids = [i if i < vocab_size else tokenizer.token_to_id("<unk>") for i in tokenizer.encode(prompt).ids]
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        logits = model(x)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        x = torch.cat((x, next_id), dim=1)

    return tokenizer.decode(x[0].tolist())


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Shakespeare GPT", page_icon="🖋️", layout="centered")

st.title("🖋️ Shakespeare GPT")
st.write("Enter a prompt and let the model generate Shakespearean text!")

prompt = st.text_area("Enter your prompt:", "To be, or not to be", height=100)
max_tokens = st.slider("Max tokens to generate:", min_value=50, max_value=500, value=200, step=10)
temperature = st.slider("Temperature (creativity):", min_value=0.1, max_value=1.2, value=0.7, step=0.05)

if st.button("Generate"):
    with st.spinner("Generating..."):
        try:
            output = generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
            st.subheader("Generated Text:")
            st.write(output)
        except Exception as e:
            st.error(f"Error: {e}")

