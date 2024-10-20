import torch
print(torch.__version__)              # Print PyTorch version
print(torch.version.cuda)             # Check CUDA version PyTorch was built with
print(torch.backends.cudnn.enabled)   # Check if cuDNN (CUDA for deep learning) is enabled
