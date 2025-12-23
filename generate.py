import torch
import torch.nn.functional as F
from config import device

# Autoregressive text generation
@torch.no_grad()
def generate(model, stoi, itos, block_size, prompt, max_new_tokens, temperature):
    # Encode prompt
    idx = torch.tensor([[stoi[c] for c in prompt if c in stoi]], device=device)

    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)

    # Decode to text
    return "".join([itos[i] for i in idx[0].tolist()])
