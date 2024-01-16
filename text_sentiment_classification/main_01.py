from pathlib import Path

import torch
from fastai.learner import load

# https://huggingface.co/bert-base-uncased/tree/main
model_path = "bert-base-uncased.bin"

# Load the learner
learner = torch.load(Path(__file__).parent / model_path)
"breakpoint"
