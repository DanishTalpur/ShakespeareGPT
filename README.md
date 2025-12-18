# 🖋️ Shakespeare GPT

A mini GPT model that generates Shakespearean text from a user-provided prompt. Built in **PyTorch**, with a **Streamlit GUI** for easy interaction.  

This project demonstrates **language modeling**, **BPE tokenization**, and a **lightweight transformer-style architecture** for creative text generation.  

---

## 🔹 Features

- Generates Shakespeare-like paragraphs and short dialogues  
- Supports **custom prompts**, **max token length**, and **temperature** (creativity)  
- Lightweight model suitable for CPU or GPU  
- Streamlit GUI for **instant web-based interaction**  

---

## 📁 Project Structure

.
│ gpt_shakespeare.pt # Trained model weights
│ gui.py # Streamlit GUI
│ Shakespeare.ipynb # Training and experimentation notebook
│ shakespeare.txt # Raw text data for training
└───tokenizer
merges.txt # BPE tokenizer merges
vocab.json # BPE tokenizer vocab


---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/shakespeare-gpt.git
cd shakespeare-gpt
'''


### 2. Install dependencies

pip install -r requirements.txt

### 3. Run locally
'''bash
streamlit run gui.py
Opened automatically in your browser.
'''

Enter a prompt, adjust max tokens and temperature, and click Generate.