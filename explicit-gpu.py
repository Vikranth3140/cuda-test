import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a random tensor and move it to GPU
x = torch.rand(5, 3).to(device)
print(f"Tensor on device: {x.device}")
