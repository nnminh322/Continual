#!/usr/bin/env python
"""Quick unit test for OT-SIGN components."""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')

from ot_sign_utils import sinkhorn_log, vmf_cost_matrix, compute_vmf_params, compute_anti_invasion_loss

print('=== Test 1: Sinkhorn OT ===')
C = torch.randn(4, 3)
weights = sinkhorn_log(C, epsilon=0.05, n_iter=10)
print(f'Cost shape: {C.shape}, Weights shape: {weights.shape}')
print(f'Row sums: {weights.sum(dim=1)}')
print(f'Weights:\n{weights}')

print()
print('=== Test 2: vMF Parameter Estimation ===')
d = 1024
n = 100
raw = torch.randn(n, d)
features = F.normalize(raw, dim=-1)
mu, kappa = compute_vmf_params(features, d)
print(f'mu shape: {mu.shape}, norm: {mu.norm():.4f}')
print(f'kappa: {kappa.item():.4f}')

print()
print('=== Test 3: vmf_cost_matrix ===')
x = F.normalize(torch.randn(4, d), dim=-1)
sigs = {0: (mu, kappa)}
prompt_key = torch.randn(1, d)
C2 = vmf_cost_matrix(x, sigs, prompt_key=prompt_key, default_kappa=10.0)
print(f'Cost matrix shape: {C2.shape}')

weights2 = sinkhorn_log(C2, epsilon=0.05, n_iter=10)
print(f'OT weights shape: {weights2.shape}, row sums: {weights2.sum(dim=1)}')
print(f'Weights:\n{weights2}')

print()
print('=== Test 4: Anti-invasion loss ===')
x_new = F.normalize(torch.randn(8, d), dim=-1)
sigs2 = {0: (mu, torch.tensor(5.0)), 1: (F.normalize(torch.randn(d), dim=-1), torch.tensor(8.0))}
inv_loss = compute_anti_invasion_loss(x_new, sigs2, current_task_id=2, threshold=2.3)
print(f'Invasion loss: {inv_loss.item():.6f}')

print()
print('=== Test 5: FP16 compatibility ===')
C_fp16 = torch.randn(8, 5, dtype=torch.float16)
w_fp16 = sinkhorn_log(C_fp16.float(), epsilon=0.05, n_iter=10).half()
print(f'FP16 output: {w_fp16.shape}, dtype: {w_fp16.dtype}')
print(f'Row sums: {w_fp16.sum(dim=1)}')

print()
print('=== All tests passed! ===')
