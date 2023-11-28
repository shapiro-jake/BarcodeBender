import torch
use_cuda = torch.cuda.is_available()
print(use_cuda)

mps_device = torch.device("mps")
print(mps_device)