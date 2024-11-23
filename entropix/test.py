import jax

try:
    device = jax.devices("gpu")[0]
    print("GPU found:", device)
except RuntimeError:
    print("GPU not found. Using CPU instead.")


import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
