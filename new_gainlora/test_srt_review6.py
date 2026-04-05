#!/usr/bin/env python3
"""
Numerical verification tests for SRT Router — Review Round 6.
Tests: pinv_ridge, PSR, full pipeline, string task_ids, SRM propagation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from srt_router import (
    metric_l2, metric_mahalanobis, metric_psr, pinv_ridge,
    participation_ratio, ledoit_wolf_shrinkage, update_pooled_cov,
    pooled_shrinkage_target, srm_metric_selection,
    TaskSignature, SRTRouter
)

np.random.seed(42)
PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 1: pinv_ridge
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ TEST 1: pinv_ridge ═══")
d = 50
A = np.random.randn(100, d)
Sigma = np.cov(A, rowvar=False, ddof=1)
Sinv = pinv_ridge(Sigma)
eye_approx = Sigma @ Sinv
check("pinv_ridge ≈ identity", np.allclose(eye_approx, np.eye(d), atol=1e-4),
      f"max error = {np.max(np.abs(eye_approx - np.eye(d))):.2e}")

# Ill-conditioned case
Sigma_ill = np.diag(np.concatenate([np.ones(5), np.full(d-5, 1e-10)]))
Sinv_ill = pinv_ridge(Sigma_ill)
check("pinv_ridge ill-conditioned: no NaN/Inf",
      np.all(np.isfinite(Sinv_ill)),
      f"has nan={np.any(np.isnan(Sinv_ill))}, inf={np.any(np.isinf(Sinv_ill))}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 2: PSR non-negativity
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ TEST 2: PSR metric ═══")
d = 64
mu = np.zeros(d)
eigvecs = np.eye(d)[:, :10]  # top-10
eigvals = np.arange(10, 0, -1, dtype=float)

h_test = np.random.randn(100, d) * 0.5
psr_vals = metric_psr(h_test, mu, eigvecs, eigvals, d)
check("PSR non-negative", np.all(psr_vals >= -1e-12),
      f"min={psr_vals.min():.6e}")
check("PSR shape correct", psr_vals.shape == (100,),
      f"shape={psr_vals.shape}")

# Single input
psr_single = metric_psr(h_test[0], mu, eigvecs, eigvals, d)
check("PSR single input shape", psr_single.shape == (1,))


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 3: Full pipeline with INTEGER task IDs
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ TEST 3: Full pipeline (int task_ids) ═══")
d = 32
router = SRTRouter(use_srm=True)

# Create 3 well-separated clusters
centers = {0: np.ones(d)*2.0, 1: np.ones(d)*(-2.0), 2: np.zeros(d)}
for tid, c in centers.items():
    h = c + np.random.randn(200, d) * 0.3
    router.add_task(task_id=tid, h_train=h, use_shrink=True, shrink_factor=0.1)

check("3 tasks registered", len(router.signatures) == 3)
check("Pool updated", router._n_pool > 0)

# Route: points near center 0 should route to task 0
test_h = centers[0] + np.random.randn(50, d) * 0.1
pred, dists = router.route(test_h)
accuracy_0 = np.mean(pred == 0)
check(f"Route accuracy task 0: {accuracy_0:.0%}", accuracy_0 > 0.9,
      f"acc={accuracy_0:.2%}")

# Points near center 1
test_h_1 = centers[1] + np.random.randn(50, d) * 0.1
pred_1, _ = router.route(test_h_1)
accuracy_1 = np.mean(pred_1 == 1)
check(f"Route accuracy task 1: {accuracy_1:.0%}", accuracy_1 > 0.9,
      f"acc={accuracy_1:.2%}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 4: STRING task IDs — THE CRITICAL BUG
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ TEST 4: String task IDs (BUG check) ═══")
router_str = SRTRouter(use_srm=False)
str_task_ids = ["yelp", "amazon", "mnli"]

try:
    for i, tid in enumerate(str_task_ids):
        c = np.ones(d) * (i * 3.0)
        h = c + np.random.randn(100, d) * 0.3
        router_str.add_task(task_id=tid, h_train=h, use_shrink=True)
    check("add_task with string IDs", True)
except Exception as e:
    check("add_task with string IDs", False, str(e))

try:
    test_h = np.ones(d) * 0.0 + np.random.randn(10, d) * 0.1
    pred, dists = router_str.route(test_h)
    check("route() with string IDs", True)
    check("route() returns string IDs", all(isinstance(p, str) for p in pred),
          f"types={[type(p).__name__ for p in pred[:3]]}")
except Exception as e:
    check("route() with string IDs — CRASHES", False,
          f"{type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 5: SRM metric propagation
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ TEST 5: SRM metric propagation ═══")
router_srm = SRTRouter(use_srm=True)

# Add 3 tasks (SRM triggers when ≥2 existing)
for tid in range(3):
    c = np.ones(d) * (tid * 5.0)
    h = c + np.random.randn(200, d) * 0.5
    router_srm.add_task(task_id=tid, h_train=h)

# Check: SRM results should be propagated to TaskSignature._metric
srm_dict = router_srm._srm_metrics
sig_metrics = {tid: router_srm.signatures[tid]._metric for tid in router_srm.signatures}

# SRM ran for task IDs 0,1 when task 2 was being added
# Check if SRM results are in _srm_metrics
check("SRM ran (has results)", len(srm_dict) >= 2,
      f"srm_dict keys={list(srm_dict.keys())}")

# Check if SRM results propagated to existing signatures
propagated = True
for tid in srm_dict:
    if tid in router_srm.signatures:
        if router_srm.signatures[tid]._metric != srm_dict[tid]:
            propagated = False
            print(f"    Task {tid}: sig._metric={router_srm.signatures[tid]._metric}, srm={srm_dict[tid]}")

check("SRM results propagated to TaskSignature._metric", propagated,
      f"sig_metrics={sig_metrics}, srm_dict={dict(srm_dict)}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 6: Welford-Hart pooled covariance
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ TEST 6: Pooled covariance ═══")
d = 16
h1 = np.random.randn(100, d)
h2 = np.random.randn(150, d) + 1.0

mu1, cov1 = h1.mean(0), np.cov(h1, rowvar=False, ddof=1)
mu2, cov2 = h2.mean(0), np.cov(h2, rowvar=False, ddof=1)

mu_p, cov_p, n_p = update_pooled_cov(mu1, cov1, 100, mu2, cov2, 150)

# Verify against brute-force pooled stats
h_all = np.vstack([h1, h2])
mu_bf = h_all.mean(0)
cov_bf = np.cov(h_all, rowvar=False, ddof=1)

check("Pooled mean correct", np.allclose(mu_p, mu_bf, atol=1e-10),
      f"max_error={np.max(np.abs(mu_p - mu_bf)):.2e}")
check("Pooled cov correct", np.allclose(cov_p, cov_bf, atol=1e-10),
      f"max_error={np.max(np.abs(cov_p - cov_bf)):.2e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 7: Re-shrink uses Sigma_raw (not compounding)
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ TEST 7: Re-shrink from Sigma_raw ═══")
Sigma_raw = np.diag(np.arange(1, d+1, dtype=float))
Sigma_pool = np.eye(d) * 5.0

sig = TaskSignature(task_id=0, mu=np.zeros(d), Sigma=Sigma_raw, n=100)
check("Sigma_raw == Sigma initially", np.allclose(sig.Sigma_raw, Sigma_raw))

# First re-shrink
sig.reshrink(Sigma_pool, 300, alpha=0.5)
expected1 = 0.5 * Sigma_raw + 0.5 * Sigma_pool
check("After 1st reshrink: correct", np.allclose(sig.Sigma, expected1, atol=1e-10))

# Second re-shrink with new pool — should still use Sigma_raw, NOT the shrunk Sigma
Sigma_pool_2 = np.eye(d) * 10.0
sig.reshrink(Sigma_pool_2, 500, alpha=0.3)
expected2 = 0.7 * Sigma_raw + 0.3 * Sigma_pool_2
check("After 2nd reshrink: uses Sigma_raw (no compounding)",
      np.allclose(sig.Sigma, expected2, atol=1e-10),
      f"max_error={np.max(np.abs(sig.Sigma - expected2)):.2e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 8: Save / Load round-trip
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ TEST 8: Save/Load round-trip ═══")
import tempfile
router_save = SRTRouter(use_srm=False)
for tid in range(3):
    c = np.ones(32) * tid
    router_save.add_task(task_id=tid, h_train=c + np.random.randn(50, 32)*0.2)

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "srt_test.npz")
    router_save.save(path)
    
    router_load = SRTRouter()
    router_load.load(path)
    
    check("Load: same # tasks", len(router_load.signatures) == len(router_save.signatures))
    for tid in router_save.signatures:
        s1 = router_save.signatures[tid]
        s2 = router_load.signatures[tid]
        check(f"Load task {tid}: mu match", np.allclose(s1.mu, s2.mu, atol=1e-10))
        check(f"Load task {tid}: Sigma match", np.allclose(s1.Sigma, s2.Sigma, atol=1e-10))
        check(f"Load task {tid}: Sigma_raw match", np.allclose(s1.Sigma_raw, s2.Sigma_raw, atol=1e-10))


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  TOTAL: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(0 if FAIL == 0 else 1)
