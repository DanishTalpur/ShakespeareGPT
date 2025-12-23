import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "checkpoint/gpt_shakespeare.pth"
