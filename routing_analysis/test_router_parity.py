#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def pooled_within_covariance(blocks: list[np.ndarray]) -> tuple[np.ndarray, int]:
    dof = sum(max(block.shape[0] - 1, 0) for block in blocks)
    scatter = np.zeros((blocks[0].shape[1], blocks[0].shape[1]), dtype=np.float64)
    for block in blocks:
        if block.shape[0] <= 1:
            continue
        centered = block - block.mean(axis=0)
        scatter += centered.T @ centered
    if dof <= 0:
        return scatter, 0
    return scatter / dof, dof


def main():
    root = Path(__file__).resolve().parents[1]
    runtime_mod = load_module(
        "runtime_router",
        root / "new_llama_gainlora" / "src" / "srt_router_v2.py",
    )
    legacy_mod = load_module(
        "legacy_router",
        root / "new_gainlora" / "src" / "srt_router.py",
    )
    offline_mod = load_module(
        "offline_router",
        root / "routing_analysis" / "routing_class_v2.py",
    )

    rng = np.random.default_rng(7)
    d = 24
    task_names = ["task_a", "task_b", "task_c"]
    task_blocks = []
    for idx, shift in enumerate((0.0, 1.2, -1.4)):
        base = rng.normal(loc=shift, scale=0.35, size=(64, d))
        low_rank = rng.normal(scale=0.12, size=(64, 1)) * np.linspace(0.2, 1.0, d)
        task_blocks.append(base + low_rank + idx * 0.05)

    runtime_router = runtime_mod.PooledMahalanobisRouter(shrinkage="ridge", device="cpu")
    legacy_router = legacy_mod.PooledMahalanobisRouter(shrinkage="ridge", device="cpu")
    offline_router = offline_mod.PooledMahalanobisRouter(shrinkage="ridge", device="cpu")

    seen_blocks: list[np.ndarray] = []
    for task_name, block in zip(task_names, task_blocks):
        seen_blocks.append(block)
        runtime_router.add_task(task_name, block)
        legacy_router.add_task(task_name, block)
        offline_router.add_task(block, task_name=task_name)

        expected_cov, expected_dof = pooled_within_covariance(seen_blocks)
        np.testing.assert_allclose(
            runtime_router._Sigma_pool_t.cpu().numpy(),
            expected_cov,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            legacy_router._Sigma_pool_t.cpu().numpy(),
            expected_cov,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            offline_router.Sigma_pool_t.cpu().numpy(),
            expected_cov,
            rtol=1e-5,
            atol=1e-6,
        )
        assert runtime_router._cov_dof == expected_dof
        assert legacy_router._cov_dof == expected_dof
        assert offline_router.cov_dof == expected_dof

    queries = np.vstack([block[:10] for block in task_blocks])
    runtime_pred, runtime_dists = runtime_router.route(queries)
    legacy_pred, legacy_dists = legacy_router.route(queries)
    offline_pred = offline_router.route(queries)
    task_to_idx = {task_name: idx for idx, task_name in enumerate(task_names)}
    runtime_idx = np.array([task_to_idx[task_name] for task_name in runtime_pred])
    legacy_idx = np.array([task_to_idx[task_name] for task_name in legacy_pred])

    np.testing.assert_array_equal(runtime_idx, offline_pred)
    np.testing.assert_array_equal(legacy_idx, offline_pred)
    np.testing.assert_allclose(runtime_dists, legacy_dists, rtol=1e-5, atol=1e-6)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "router_state.npz"
        runtime_router.save(str(save_path))
        payload = np.load(save_path, allow_pickle=True)
        assert str(payload["covariance_mode"][0]) == runtime_mod.COVARIANCE_MODE
        assert int(payload["cov_dof"][0]) == runtime_router._cov_dof

        restored = runtime_mod.PooledMahalanobisRouter(shrinkage="ridge", device="cpu")
        restored.load(str(save_path))
        restored_pred, restored_dists = restored.route(queries)
        np.testing.assert_array_equal(restored_pred, runtime_pred)
        np.testing.assert_allclose(restored_dists, runtime_dists, rtol=1e-5, atol=1e-6)

    print("router parity ok")


if __name__ == "__main__":
    main()