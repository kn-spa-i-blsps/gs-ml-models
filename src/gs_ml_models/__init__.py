import torch

TORCH_DEVICE = "cuda"
if not torch.cuda.is_available():
    TORCH_DEVICE = "cpu"
    print("CUDA is unavailable, torch will run on CPU")
