# SRT + SGWI + Dual Fisher Algorithm Summary

## TASK 0 (First task)
```
1. Train LoRA_0 normally on D_0 with CE loss
2. Save θ_0* = {lora_A, lora_B} after training
3. Extract μ_0, Σ_0 from frozen backbone embeddings of D_0
4. srt_router.add_task(0, μ_0, Σ_0)
```

## TASK t > 0 (Continual loop)

### Phase 1: SRT Profiling
```
Input: D_t (current task data)
μ_t = mean(B(x)) over x∈D_t      # B = frozen backbone
Σ_t = cov(B(x)) over x∈D_t
srt_router.add_task(t, μ_t, Σ_t)  # includes shrink previous tasks
```

### Phase 2: SGWI Initialization
```
Input: {μ_s, Σ_s}_{s<t}, {θ_s* = (B_s, A_s)}_{s<t}, D_t

# Step 1: Compute SRT distances to all past tasks
for s in past_tasks:
    d_s = (μ_t - μ_s)^T · Σ_pool^{-1} · (μ_t - μ_s)
    w_s = softmax(-d_s / τ)   # τ = median of all d_s

# Step 2: Weighted LoRA fusion
ΔW_init = Σ_{s<t} w_s · (B_s · A_s)   # [d_out, d_in]

# Step 3: SVD → initialize new adapter
U, S, Vt = svd(ΔW_init)
lora_A_t = sqrt(S[:r]) · Vt[:r, :]     # [r, d_in]
lora_B_t = U[:, :r] · diag(sqrt(S[:r]))  # [d_out, r]
```

### Phase 3: Dual Fisher Training
```
for each batch (x, y) in D_t:
    # CE loss
    L_CE = CrossEntropyLoss(model(x), y)
    
    # Dual Fisher penalty (SRT-weighted)
    L_F = 0
    for s in past_tasks:
        L_F += w_s · ||θ_t - θ_s*||²_F_emb(s)
    
    # Total loss
    L_total = L_CE + λ · L_F
    
    backward(L_total)
    optimizer.step()
```

### Phase 4: SRT Inference (at eval/test time)
```
Input: test input x
h = B(x)                           # frozen backbone embedding
h_whitened = ZCA_transform(h)     # whitened embedding
t* = argmin_s ||h_whitened - μ_s_whitened||²
output = model(x, adapter=t*)      # hard one-hot, NOT soft blend
```

## Key Differences from ROOT

| Aspect | ROOT | SRT+SGWI |
|--------|------|----------|
| Router | Learned MLP (trans_input) | ZCA+L2 (non-parametric) |
| Init | Random (kaiming) | SRT-weighted fusion |
| Gradient protection | InfLoRA (null-space) | Dual Fisher (soft) |
| Inference | Soft blend | Hard one-hot |
