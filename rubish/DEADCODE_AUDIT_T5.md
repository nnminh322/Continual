# Dead Code Audit — T5 (order 3/4)

**Status: REPORT ONLY — do NOT modify while training is running**

## 1. `cal_attention()` in t5_gainlora.py — ⚠️ DEAD CODE (routing)

The `cal_attention` method (sigmoid-based soft weights) was **removed from the routing path** in the pure SRT hard one-hot commit. Forward routing now uses:
- Training: `w = [1, 0, 0, ...]` (current adapter only)
- Inference: SRT hard one-hot (per-sample routing)

**BUT**: `cal_attention` is still **defined** in the class. No code path calls it.
**Is it truly dead?** YES — only callers were `forward()` (removed) and `memory_replay()` (also dead).

## 2. `memory_replay()` in t5_gainlora.py — ⚠️ DEAD CODE

Was called by `cl_trainer_gainlora.py` (old trainer). The current trainer is `SGWI_DualFisher_Trainer` (from sgwi_trainer.py) which never calls `memory_replay()`.

## 3. `all_attn_weights` accumulation — ⚠️ PROBLEMATIC (caused task 9 crash)

In `run_t5.py` lines 1067-1074:
```python
attn_w = np.concatenate(trainer.model.encoder.all_attn_weights)
```
With hard one-hot routing, `all_attn_weights` may be empty → `np.concatenate([])` crashes → predict metrics never saved. **This is the root cause of task 9-sst2 missing results.**

## 4. GPM feature collection (`get_matrix3`, `get_chunk`, `get_trans_feature`) — ✅ STILL ACTIVE

These ARE still used by `GainLoRATrainer.get_repsentation()` and `_inner_training_loop()` for GPM gradient projection. **NOT dead code.**

## 5. `l2_normalize()` helper — ⚠️ CHECK

Need to verify if any remaining code path calls it. If only `cal_attention` used it → dead.

## Recommendation (after training completes)

1. Remove `cal_attention()` and `memory_replay()` from both t5_gainlora.py and llama_gainlora.py
2. Fix `all_attn_weights` crash: guard with `if len(all_attn_weights) > 0` or remove entirely
3. Keep GPM code (get_matrix3, get_chunk, etc.) — still active for regularization
