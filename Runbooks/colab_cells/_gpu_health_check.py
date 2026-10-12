import torch
x = torch.randn(1000, 1000, device="cuda")
print("GPU healthy:", (x @ x).sum().item())
print("free/total:", torch.cuda.mem_get_info())
