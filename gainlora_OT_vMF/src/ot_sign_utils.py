"""
OT-SIGN: Statistical Signatures + Optimal Transport Routing for GainLoRA
=========================================================================
Utility functions for:
  1. vMF (von Mises-Fisher) knowledge signatures
  2. Sinkhorn OT routing with vMF log-likelihood cost
  3. Anti-drift and anti-invasion losses

Author: OT-SIGN implementation for GainLoRA continual learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 1. Sinkhorn Optimal Transport (log-domain for numerical stability)
# ============================================================================

def sinkhorn_log(C, epsilon=0.05, n_iter=10):
    """
    Log-domain Sinkhorn algorithm for entropic OT.
    
    Args:
        C: Cost matrix of shape (B, N) — cost of assigning sample b to expert n
        epsilon: Entropic regularization strength (lower = sharper assignment)
        n_iter: Number of Sinkhorn iterations
        
    Returns:
        Transport plan of shape (B, N), rows sum to ~1
    """
    B, N = C.shape
    
    # Log kernel: K = exp(-C / epsilon)
    log_K = -C / epsilon  # (B, N)
    
    # Initialize dual variables
    u = torch.zeros(B, device=C.device, dtype=C.dtype)
    v = torch.zeros(N, device=C.device, dtype=C.dtype)
    
    for _ in range(n_iter):
        # Update u (row normalization in log domain)
        u = -torch.logsumexp(log_K + v.unsqueeze(0), dim=1)  # (B,)
        # Update v (column normalization in log domain)
        v = -torch.logsumexp(log_K + u.unsqueeze(1), dim=0)  # (N,)
    
    # Compute transport plan
    log_pi = log_K + u.unsqueeze(1) + v.unsqueeze(0)  # (B, N)
    pi = log_pi.exp()
    
    # Normalize rows to sum to 1 (for routing weights)
    pi = pi / (pi.sum(dim=1, keepdim=True) + 1e-8)
    
    return pi  # (B, N)


# ============================================================================
# 2. vMF Parameter Estimation (MLE)
# ============================================================================

def compute_vmf_params(features, d_model, kappa_min=0.1, kappa_max=50.0):
    """
    Compute vMF MLE parameters from features on unit sphere.
    
    Args:
        features: Tensor of shape (num_samples, d) — L2-normalized features
        d_model: Feature dimension
        kappa_min: Minimum allowed kappa (prevents numerical issues)
        kappa_max: Maximum allowed kappa (prevents overflow)
        
    Returns:
        mu: Mean direction (d,) — unit vector
        kappa: Concentration parameter — scalar tensor
    """
    # Mean resultant vector
    x_bar = features.mean(dim=0)  # (d,)
    r_bar = x_bar.norm()  # scalar — mean resultant length
    
    # Mean direction
    mu = F.normalize(x_bar, dim=0)  # (d,)
    
    # Concentration parameter (Banerjee et al. 2005 approximation)
    d = d_model
    r_bar_clamped = r_bar.clamp(min=1e-6, max=1.0 - 1e-6)
    kappa = r_bar_clamped * (d - r_bar_clamped ** 2) / (1 - r_bar_clamped ** 2)
    
    # Clamp kappa for numerical stability
    kappa = kappa.clamp(min=kappa_min, max=kappa_max)
    
    return mu.detach(), kappa.detach()


# ============================================================================
# 3. Cost Matrix Construction
# ============================================================================

def vmf_cost_matrix(x, vmf_signatures, prompt_key=None, default_kappa=10.0):
    """
    Build OT cost matrix from vMF signatures.
    
    ORDERING: Current task FIRST, then previous tasks sorted by task_id.
    This matches GainLoRA's agg_lora_states ordering:
      concat_q = cat([cur_lora, prev_lora_0, prev_lora_1, ...])
    
    For current task: C[b, 0] = -default_kappa * (prompt_key · x_b)
    For previous tasks: C[b, t+1] = -kappa_t * (mu_t · x_b)
    
    Args:
        x: Normalized input features (B, d)
        vmf_signatures: Dict {task_id: (mu, kappa)} for previous tasks
        prompt_key: Current task's prompt key (1, d) or None
        default_kappa: Kappa value used for the current task's proxy
        
    Returns:
        C: Cost matrix (B, N_total) where N_total = 1 + len(signatures)
           Position 0 = current task, positions 1..N-1 = prev tasks in order
    """
    device = x.device
    dtype = x.dtype
    costs = []
    
    # Position 0: Cost for current task (using prompt_key as proxy direction)
    if prompt_key is not None:
        pk = prompt_key.squeeze()  # (d,)
        pk = F.normalize(pk, dim=-1)
        dot_product = (x * pk.unsqueeze(0)).sum(dim=-1)  # (B,)
        default_kappa_t = torch.tensor(default_kappa, device=device, dtype=dtype)
        cost_cur = -default_kappa_t * dot_product  # (B,)
        costs.append(cost_cur)

    # Positions 1..N: Cost from vMF signatures (previous tasks) — sorted by task_id
    for task_id in sorted(vmf_signatures.keys()):
        mu_t, kappa_t = vmf_signatures[task_id]
        mu_t = mu_t.to(device=device, dtype=dtype)
        kappa_t_val = kappa_t.to(device=device, dtype=dtype) if isinstance(kappa_t, torch.Tensor) else torch.tensor(kappa_t, device=device, dtype=dtype)
        
        # vMF negative log-likelihood (ignoring constant normalization)
        dot_product = (x * mu_t.unsqueeze(0)).sum(dim=-1)  # (B,)
        cost_t = -kappa_t_val * dot_product  # (B,) — lower cost = higher likelihood
        costs.append(cost_t)
    
    if len(costs) == 0:
        raise ValueError("No signatures or prompt_key provided for cost matrix")
    
    C = torch.stack(costs, dim=1)  # (B, N_total)
    return C


# ============================================================================
# 4. Anti-Drift Loss
# ============================================================================

def compute_anti_drift_loss(model_encoder, ref_trans_input_state, input_ids, attention_mask):
    """
    MSE between current and reference trans_input outputs on replay data.
    Protects the shared MLP from drifting away from old task representations.
    
    Args:
        model_encoder: The model's encoder (has embed_tokens, trans_input)
        ref_trans_input_state: Frozen state dict of trans_input from previous task
        input_ids: Replay input token IDs (B, seq_len)
        attention_mask: Replay attention mask (B, seq_len)
        
    Returns:
        drift_loss: Scalar MSE loss
    """
    # Get current trans_input output
    with torch.no_grad():
        embeddings = model_encoder.embed_tokens(input_ids)
    avg_emb = (attention_mask.unsqueeze(-1) * embeddings).mean(dim=1, keepdim=True)  # (B, 1, d)
    
    # Current trans_input output
    medium_curr = model_encoder.trans_input[1](model_encoder.trans_input[0](avg_emb))
    x_curr = model_encoder.trans_input[3](model_encoder.trans_input[2](medium_curr))
    x_curr = F.normalize(x_curr, dim=-1)  # (B, 1, d)
    
    # Reference trans_input output (frozen copy)
    ref_trans_input = nn.Sequential(
        nn.Linear(model_encoder.trans_input[0].in_features, model_encoder.trans_input[0].out_features, bias=False),
        nn.SiLU(),
        nn.Linear(model_encoder.trans_input[2].in_features, model_encoder.trans_input[2].out_features, bias=False),
        nn.SiLU(),
    ).to(device=avg_emb.device, dtype=avg_emb.dtype)
    ref_trans_input.load_state_dict(ref_trans_input_state)
    for p in ref_trans_input.parameters():
        p.requires_grad = False
    
    with torch.no_grad():
        medium_ref = ref_trans_input[1](ref_trans_input[0](avg_emb))
        x_ref = ref_trans_input[3](ref_trans_input[2](medium_ref))
        x_ref = F.normalize(x_ref, dim=-1)  # (B, 1, d)
    
    drift_loss = F.mse_loss(x_curr, x_ref)
    return drift_loss


class CachedRefTransInput(nn.Module):
    """
    Cached frozen copy of trans_input for efficient anti-drift computation.
    Avoids recreating nn.Sequential every step.
    """
    def __init__(self, trans_input_module):
        super().__init__()
        import copy
        self.ref = copy.deepcopy(trans_input_module)
        for p in self.ref.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        with torch.no_grad():
            medium = self.ref[1](self.ref[0](x))
            out = self.ref[3](self.ref[2](medium))
        return F.normalize(out, dim=-1)


def compute_anti_drift_loss_cached(model_encoder, ref_trans_input_cached, input_ids, attention_mask):
    """
    Efficient anti-drift loss using a pre-cached frozen trans_input.
    
    Args:
        model_encoder: Model encoder with embed_tokens and trans_input
        ref_trans_input_cached: CachedRefTransInput instance
        input_ids: Replay input IDs (B, seq_len)
        attention_mask: Replay attention mask (B, seq_len)
        
    Returns:
        Scalar MSE loss
    """
    with torch.no_grad():
        embeddings = model_encoder.embed_tokens(input_ids)
    avg_emb = (attention_mask.unsqueeze(-1) * embeddings).mean(dim=1, keepdim=True)
    
    # Current trans_input output
    medium_curr = model_encoder.trans_input[1](model_encoder.trans_input[0](avg_emb))
    x_curr = model_encoder.trans_input[3](model_encoder.trans_input[2](medium_curr))
    x_curr = F.normalize(x_curr, dim=-1)
    
    # Reference output (frozen)
    x_ref = ref_trans_input_cached(avg_emb)
    
    return F.mse_loss(x_curr, x_ref)


# ============================================================================
# 5. Anti-Invasion Loss
# ============================================================================

def compute_anti_invasion_loss(x_normalized, vmf_signatures, current_task_id, threshold=2.3):
    """
    Hinge loss penalizing current task features that invade old task regions.
    
    Args:
        x_normalized: Current task's normalized features (B, d) or (B, 1, d)
        vmf_signatures: Dict {task_id: (mu, kappa)}
        current_task_id: Current task index
        threshold: vMF log-likelihood threshold (default -log(0.1) ≈ 2.3)
        
    Returns:
        invasion_loss: Scalar hinge loss
    """
    if x_normalized.dim() == 3:
        x_normalized = x_normalized.squeeze(1)  # (B, d)
    
    device = x_normalized.device
    dtype = x_normalized.dtype
    invasion_loss = torch.tensor(0.0, device=device, dtype=dtype)
    n_old_tasks = 0
    
    for task_id, (mu_s, kappa_s) in vmf_signatures.items():
        if task_id >= current_task_id:
            continue
        
        mu_s = mu_s.to(device=device, dtype=dtype)
        kappa_val = kappa_s.item() if isinstance(kappa_s, torch.Tensor) else kappa_s
        
        # Log-likelihood of x_new under vMF(mu_s, kappa_s): kappa_s * (mu_s · x_new)
        log_lik = kappa_val * (x_normalized * mu_s.unsqueeze(0)).sum(dim=-1)  # (B,)
        
        # Hinge: penalize when log_lik > threshold (feature too close to old task)
        violation = F.relu(log_lik - threshold)  # (B,)
        invasion_loss = invasion_loss + violation.mean()
        n_old_tasks += 1
    
    if n_old_tasks > 0:
        invasion_loss = invasion_loss / n_old_tasks
    
    return invasion_loss
