import streamlit as st
import torch
from model import ShakespeareGPT
from generate import generate
from config import CHECKPOINT_PATH, device

# PAGE CONFIG
st.set_page_config(
    page_title="Shakespeare GPT",
    page_icon="🖋️",
    layout="centered"
)

# HEADER (TITLE LEFT, LOGO RIGHT)
col_title, col_logo = st.columns([5, 2])

with col_title:
    st.markdown("## 🖋️ Shakespeare GPT")
    st.caption("Because sanity is overrated.")

with col_logo:
    # slightly bigger logo
    st.image("imgs/LOGO.png", width=170)


# LOAD CHECKPOINT
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

model = ShakespeareGPT(
    vocab_size=checkpoint["vocab_size"],
    **checkpoint["config"]
).to(device)

model.load_state_dict(checkpoint["model_state"])
model.eval()

# USER INPUT
prompt = st.text_area("Prompt", "To be or not be\n", height=120)
tokens = st.slider("Max tokens", 100, 800, 400)
temp = st.slider("Temperature", 0.3, 1.2, 0.8)

# GENERATION
if st.button("Shake the Pear"):
    st.text(generate(
        model,
        checkpoint["stoi"],
        checkpoint["itos"],
        checkpoint["config"]["block_size"],
        prompt,
        tokens,
        temp
    ))

# FOOTER (TEXT ONLY)
st.markdown("---")
st.markdown(
    """
    **Built by:** Danish Talpur, Aun Naqvi    
    **License:** MIT  
    **Email:** danishshuja11@gmail.com
    """
)
