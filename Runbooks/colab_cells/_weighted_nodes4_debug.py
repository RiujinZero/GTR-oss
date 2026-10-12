import sys

sys.path.insert(0, "/mnt/c/Users/user/Documents/projects/GTR")

import torch

from gator.weighted_nodes4 import weighted_nodes4

torch.manual_seed(0)
B, C, T, N, D = 2, 4, 16, 4, 32
q = torch.randn(B, C, T, N, device="cuda", dtype=torch.bfloat16)
destination = torch.randn(B, C, T, N, D, device="cuda", dtype=torch.bfloat16)
h = torch.randn(B, C, T, D, device="cuda", dtype=torch.bfloat16)

USE_H = sys.argv[1] == "with_h" if len(sys.argv) > 1 else False
print("USE_H =", USE_H)


def reference(q, destination, h=None):
    mixed = (q.unsqueeze(-1) * destination).sum(-2)
    return mixed + h if h is not None else mixed


h_arg = h if USE_H else None
ref = reference(q, destination, h_arg)
my = weighted_nodes4(q, destination, h_arg)
diff = (ref.float() - my.float()).abs()
print("forward matches:", torch.allclose(ref, my, atol=1e-2, rtol=1e-2))
print("forward max abs diff:", diff.max().item(), "at", diff.argmax().item())
print("ref sample:", ref.flatten()[:5])
print("my  sample:", my.flatten()[:5])

q_ref = q.detach().clone().requires_grad_(True)
dest_ref = destination.detach().clone().requires_grad_(True)
h_ref = h.detach().clone().requires_grad_(True) if USE_H else None
ref2 = reference(q_ref, dest_ref, h_ref)
ref2.sum().backward()

q_my = q.detach().clone().requires_grad_(True)
dest_my = destination.detach().clone().requires_grad_(True)
h_my = h.detach().clone().requires_grad_(True) if USE_H else None
my2 = weighted_nodes4(q_my, dest_my, h_my)
my2.sum().backward()

dq_diff = (q_ref.grad.float() - q_my.grad.float()).abs()
print("dq matches:", torch.allclose(q_ref.grad, q_my.grad, atol=1e-1, rtol=5e-2))
print("dq max abs diff:", dq_diff.max().item())
print("dq ref sample:", q_ref.grad.flatten()[:5])
print("dq my  sample:", q_my.grad.flatten()[:5])
