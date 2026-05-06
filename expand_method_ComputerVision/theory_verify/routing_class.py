from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from common import THIS_DIR, log

try:
    import torch
except ImportError:
    torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline continual routing evaluation on frozen CV embeddings.")
    parser.add_argument("--emb_dir", required=True,
                        help="Directory produced by theory_verify/extract_embeddings.py")
    parser.add_argument("--out_dir", default=str(THIS_DIR / "results"),
                        help="Directory for the JSON routing report")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Only evaluate the first N tasks from metadata.json")
    parser.add_argument("--routers", default="all",
                        help="Comma-separated subset. Choices: nearest, cosine, online_zca, maha_ridge, maha_oas")
    parser.add_argument("--device", default="auto",
                        help="Routing compute device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--embed_dtype", default="float32", choices=["float32", "float64"],
                        help="Embedding dtype used during routing")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing JSON report")
    return parser.parse_args()


def load_metadata(emb_dir: Path) -> dict:
    with open(emb_dir / "metadata.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split(emb_dir: Path, task_name: str, split: str, dtype=np.float32) -> np.ndarray:
    path = emb_dir / task_name / f"{split}.npz"
    with np.load(path, allow_pickle=True) as data:
        return data["embeddings"].astype(dtype, copy=False)


def resolve_compute_device(device_arg: str) -> str:
    if torch is None:
        return "cpu"

    raw = (device_arg or "auto").strip().lower()
    if raw == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if raw.startswith("cuda"):
        return raw if torch.cuda.is_available() else "cpu"
    if raw.isdigit():
        return f"cuda:{raw}" if torch.cuda.is_available() else "cpu"
    return "cpu"


def resolve_numpy_dtype(dtype_name: str):
    return np.float64 if dtype_name == "float64" else np.float32


def welford_pooled_update(mu_old, cov_old, n_old, mu_new, cov_new, n_new):
    total = n_old + n_new
    if total <= 1:
        return mu_old.copy(), cov_old.copy(), n_old

    mu_pool = (n_old * mu_old + n_new * mu_new) / total
    delta = mu_new - mu_old
    cross = (n_old * n_new / total) * np.outer(delta, delta)
    cov_pool = ((n_old - 1) * cov_old + (n_new - 1) * cov_new + cross) / max(total - 1, 1)
    return mu_pool, cov_pool, total


def safe_cov(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] <= 1:
        return np.eye(embeddings.shape[1], dtype=np.float64)
    cov = np.cov(embeddings, rowvar=False, ddof=1)
    if np.ndim(cov) == 0:
        cov = np.eye(embeddings.shape[1], dtype=np.float64)
    if not np.all(np.isfinite(cov)):
        cov = np.eye(embeddings.shape[1], dtype=np.float64)
    return cov.astype(np.float64)


def l2_argmin(query: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    q_sq = np.sum(query ** 2, axis=1, keepdims=True)
    c_sq = np.sum(centroids ** 2, axis=1)
    dists = q_sq + c_sq[None, :] - 2.0 * (query @ centroids.T)
    return np.argmin(dists, axis=1)


def cosine_argmax(query: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    sims = q_norm @ c_norm.T
    return np.argmax(sims, axis=1)


def l2_argmin_torch(query_t, centroids_t):
    q_sq = torch.sum(query_t * query_t, dim=1, keepdim=True)
    c_sq = torch.sum(centroids_t * centroids_t, dim=1)
    dists = q_sq + c_sq.unsqueeze(0) - 2.0 * (query_t @ centroids_t.T)
    return torch.argmin(dists, dim=1)


def cosine_argmax_torch(query_t, centroids_t):
    q_norm = query_t / (torch.linalg.norm(query_t, dim=1, keepdim=True) + 1e-12)
    c_norm = centroids_t / (torch.linalg.norm(centroids_t, dim=1, keepdim=True) + 1e-12)
    sims = q_norm @ c_norm.T
    return torch.argmax(sims, dim=1)


class NearestCentroidRouter:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.centroids = []
        self._centroids_t = None

    def add_task(self, embeddings: np.ndarray):
        self.centroids.append(embeddings.mean(axis=0).astype(np.float64))
        self._centroids_t = None

    def _centroids_torch(self):
        if self._centroids_t is None:
            stacked = np.stack(self.centroids).astype(np.float32, copy=False)
            self._centroids_t = torch.from_numpy(stacked).to(self.device)
        return self._centroids_t

    def route(self, query: np.ndarray) -> np.ndarray:
        centroids = np.stack(self.centroids)
        if self.device != "cpu" and torch is not None:
            query_t = torch.from_numpy(query.astype(np.float32, copy=False)).to(self.device, non_blocking=True)
            preds_t = l2_argmin_torch(query_t, self._centroids_torch())
            return preds_t.cpu().numpy().astype(np.int64, copy=False)
        return l2_argmin(query, centroids)


class CosineNearestCentroidRouter:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.centroids = []
        self._centroids_t = None

    def add_task(self, embeddings: np.ndarray):
        self.centroids.append(embeddings.mean(axis=0).astype(np.float64))
        self._centroids_t = None

    def _centroids_torch(self):
        if self._centroids_t is None:
            stacked = np.stack(self.centroids).astype(np.float32, copy=False)
            self._centroids_t = torch.from_numpy(stacked).to(self.device)
        return self._centroids_t

    def route(self, query: np.ndarray) -> np.ndarray:
        centroids = np.stack(self.centroids)
        if self.device != "cpu" and torch is not None:
            query_t = torch.from_numpy(query.astype(np.float32, copy=False)).to(self.device, non_blocking=True)
            preds_t = cosine_argmax_torch(query_t, self._centroids_torch())
            return preds_t.cpu().numpy().astype(np.int64, copy=False)
        return cosine_argmax(query, centroids)


class OnlineZCAL2Router:
    def __init__(self, eps: float = 1e-8, device: str = "cpu"):
        self.eps = eps
        self.device = device
        self.raw_centroids = []
        self.mu_pool = None
        self.cov_pool = None
        self.n_pool = 0
        self.mu_global = None
        self.w_zca = None
        self.whitened_centroids = []
        self._mu_global_t = None
        self._w_zca_t = None
        self._whitened_centroids_t = None

    def _recompute_zca(self):
        eigvals, eigvecs = np.linalg.eigh(self.cov_pool)
        scale = max(float(np.abs(eigvals).max()), self.eps)
        eigvals = np.maximum(eigvals, self.eps * scale)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        self.w_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        self.mu_global = np.mean(np.stack(self.raw_centroids), axis=0)
        self.whitened_centroids = [
            ((mu - self.mu_global) @ self.w_zca.T).astype(np.float64)
            for mu in self.raw_centroids
        ]
        self._mu_global_t = None
        self._w_zca_t = None
        self._whitened_centroids_t = None

    def _prepare_torch_cache(self):
        if self._whitened_centroids_t is None:
            self._mu_global_t = torch.from_numpy(self.mu_global.astype(np.float32, copy=False)).to(self.device)
            self._w_zca_t = torch.from_numpy(self.w_zca.astype(np.float32, copy=False)).to(self.device)
            wc = np.stack(self.whitened_centroids).astype(np.float32, copy=False)
            self._whitened_centroids_t = torch.from_numpy(wc).to(self.device)

    def add_task(self, embeddings: np.ndarray):
        mu_t = embeddings.mean(axis=0).astype(np.float64)
        cov_t = safe_cov(embeddings)
        n_t = int(embeddings.shape[0])
        self.raw_centroids.append(mu_t)

        if self.n_pool == 0:
            self.mu_pool = mu_t.copy()
            self.cov_pool = cov_t.copy()
            self.n_pool = n_t
        else:
            self.mu_pool, self.cov_pool, self.n_pool = welford_pooled_update(
                self.mu_pool, self.cov_pool, self.n_pool, mu_t, cov_t, n_t)

        self._recompute_zca()

    def route(self, query: np.ndarray) -> np.ndarray:
        if self.device != "cpu" and torch is not None:
            self._prepare_torch_cache()
            query_t = torch.from_numpy(query.astype(np.float32, copy=False)).to(self.device, non_blocking=True)
            query_w_t = (query_t - self._mu_global_t) @ self._w_zca_t.T
            preds_t = l2_argmin_torch(query_w_t, self._whitened_centroids_t)
            return preds_t.cpu().numpy().astype(np.int64, copy=False)

        query_w = (query - self.mu_global) @ self.w_zca.T
        centroids = np.stack(self.whitened_centroids)
        return l2_argmin(query_w, centroids)


def shrink_cov(cov: np.ndarray, n_pool: int, method: str) -> tuple[np.ndarray, float]:
    d = cov.shape[0]
    trace = float(np.trace(cov))
    target = (trace / max(d, 1)) * np.eye(d, dtype=np.float64)

    if method == "ridge":
        delta = d / (n_pool + d + 1e-12)
        return (1.0 - delta) * cov + delta * target, float(delta)

    if method == "oas":
        tr_s2 = float((cov ** 2).sum())
        rho_hat = tr_s2 / (trace ** 2 + 1e-12)
        delta = rho_hat / (n_pool + 1.0 - 2.0 / max(d, 1))
        delta = float(min(1.0, max(0.0, delta)))
        return (1.0 - delta) * cov + delta * target, delta

    if method == "none":
        return cov, 0.0

    raise ValueError(f"Unknown shrinkage method: {method}")


def batch_mahalanobis_argmin(query: np.ndarray, centroids: np.ndarray, sinv: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    preds = []
    for start in range(0, query.shape[0], batch_size):
        batch = query[start:start + batch_size]
        diff = batch[:, None, :] - centroids[None, :, :]
        proj = np.einsum("btd,dd->btd", diff, sinv, optimize=True)
        dists = np.einsum("btd,btd->bt", proj, diff, optimize=True)
        preds.append(np.argmin(dists, axis=1))
    return np.concatenate(preds, axis=0)


def batch_mahalanobis_argmin_torch(
    query: np.ndarray,
    centroids_t,
    sinv_t,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    preds = []
    for start in range(0, query.shape[0], batch_size):
        batch = query[start:start + batch_size].astype(np.float32, copy=False)
        batch_t = torch.from_numpy(batch).to(device, non_blocking=True)
        diff = batch_t[:, None, :] - centroids_t[None, :, :]
        proj = torch.matmul(diff, sinv_t)
        dists = torch.sum(proj * diff, dim=2)
        preds.append(torch.argmin(dists, dim=1).cpu().numpy().astype(np.int64, copy=False))
    return np.concatenate(preds, axis=0)


class PooledMahalanobisRouter:
    def __init__(self, shrinkage: str, device: str = "cpu"):
        self.shrinkage = shrinkage
        self.device = device
        self.centroids = []
        self.mu_pool = None
        self.cov_pool = None
        self.n_pool = 0
        self.sinv = None
        self.delta = 0.0
        self._centroids_t = None
        self._sinv_t = None

    def add_task(self, embeddings: np.ndarray):
        mu_t = embeddings.mean(axis=0).astype(np.float64)
        cov_t = safe_cov(embeddings)
        n_t = int(embeddings.shape[0])
        self.centroids.append(mu_t)

        if self.n_pool == 0:
            self.mu_pool = mu_t.copy()
            self.cov_pool = cov_t.copy()
            self.n_pool = n_t
        else:
            self.mu_pool, self.cov_pool, self.n_pool = welford_pooled_update(
                self.mu_pool, self.cov_pool, self.n_pool, mu_t, cov_t, n_t)

        shrunk_cov, self.delta = shrink_cov(self.cov_pool, self.n_pool, self.shrinkage)
        eigvals, eigvecs = np.linalg.eigh(shrunk_cov)
        scale = max(float(np.abs(eigvals).max()), 1e-8)
        eigvals = np.maximum(eigvals, 1e-8 * scale)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        self.sinv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        self._centroids_t = None
        self._sinv_t = None

    def _prepare_torch_cache(self):
        if self._centroids_t is None:
            centroids = np.stack(self.centroids).astype(np.float32, copy=False)
            self._centroids_t = torch.from_numpy(centroids).to(self.device)
        if self._sinv_t is None:
            sinv = self.sinv.astype(np.float32, copy=False)
            self._sinv_t = torch.from_numpy(sinv).to(self.device)

    def route(self, query: np.ndarray) -> np.ndarray:
        if self.device != "cpu" and torch is not None:
            self._prepare_torch_cache()
            return batch_mahalanobis_argmin_torch(query, self._centroids_t, self._sinv_t, self.device)
        return batch_mahalanobis_argmin(query, np.stack(self.centroids), self.sinv)


def build_routers(selection: str, device: str) -> OrderedDict[str, object]:
    registry = OrderedDict([
        ("nearest", ("NearestCentroid", NearestCentroidRouter(device=device))),
        ("cosine", ("CosineNearestCentroid", CosineNearestCentroidRouter(device=device))),
        ("online_zca", ("OnlineZCAL2", OnlineZCAL2Router(device=device))),
        ("maha_ridge", ("PooledMahalanobis_RIDGE", PooledMahalanobisRouter("ridge", device=device))),
        ("maha_oas", ("PooledMahalanobis_OAS", PooledMahalanobisRouter("oas", device=device))),
    ])

    if selection == "all":
        items = registry.values()
    else:
        requested = [item.strip() for item in selection.split(",") if item.strip()]
        missing = [item for item in requested if item not in registry]
        if missing:
            raise ValueError(f"Unknown routers: {missing}")
        items = [registry[item] for item in requested]

    return OrderedDict(items)


def run_incremental_eval(
    emb_dir: Path,
    task_specs: list[dict],
    routers: OrderedDict[str, object],
    embed_dtype,
) -> dict:
    results = {name: [] for name in routers}
    step_durations: list[float] = []

    for step_idx, task_spec in enumerate(tqdm(task_specs, desc="Routing steps", unit="task", dynamic_ncols=True), start=0):
        step_start = time.time()
        train_embs = load_split(emb_dir, task_spec["task_name"], "train", dtype=embed_dtype)
        log(f"[step {step_idx + 1}/{len(task_specs)}] add {task_spec['task_name']} train={train_embs.shape}")

        for router in routers.values():
            router.add_task(train_embs)

        seen_specs = task_specs[:step_idx + 1]

        # Preload all test splits for the seen specs once to avoid repeated disk reads
        test_cache: dict[str, np.ndarray] = {}
        for seen_spec in seen_specs:
            test_cache[seen_spec["task_name"]] = load_split(
                emb_dir,
                seen_spec["task_name"],
                "test",
                dtype=embed_dtype,
            )

        for router_name, router in routers.items():
            per_task = OrderedDict()
            total_correct = 0
            total_count = 0

            for true_task_idx, seen_spec in enumerate(seen_specs):
                test_embs = test_cache[seen_spec["task_name"]]
                preds = router.route(test_embs)
                correct = int((preds == true_task_idx).sum())
                count = int(test_embs.shape[0])
                acc = correct / max(count, 1)
                per_task[seen_spec["task_name"]] = float(acc)
                total_correct += correct
                total_count += count

            macro = float(np.mean(list(per_task.values()))) if per_task else 0.0
            micro = float(total_correct / max(total_count, 1))
            results[router_name].append({
                "step": step_idx + 1,
                "seen_tasks": [spec["task_name"] for spec in seen_specs],
                "macro_accuracy": macro,
                "micro_accuracy": micro,
                "per_task": per_task,
            })

            per_task_str = " | ".join(f"{name}:{acc * 100:.1f}%" for name, acc in per_task.items())
            print(f"    {router_name:24s} macro={macro * 100:6.2f}% micro={micro * 100:6.2f}% [{per_task_str}]")

        # step timing summary
        step_time = time.time() - step_start
        step_durations.append(step_time)
        avg = sum(step_durations) / len(step_durations)
        remaining = avg * (len(task_specs) - (step_idx + 1))
        log(f"[timing] finished step {step_idx + 1}/{len(task_specs)} in {step_time:.1f}s; avg {avg:.1f}s; est remaining {remaining/60:.1f}min")

    return results


def run_routing(args: argparse.Namespace) -> Path:
    emb_dir = Path(args.emb_dir).resolve()
    compute_device = resolve_compute_device(getattr(args, "device", "auto"))
    embed_dtype_name = str(getattr(args, "embed_dtype", "float32"))
    embed_dtype = resolve_numpy_dtype(embed_dtype_name)
    metadata = load_metadata(emb_dir)
    task_specs = metadata["task_specs"]
    if args.max_tasks is not None:
        task_specs = task_specs[:args.max_tasks]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"routing__{emb_dir.name}.json"
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} already exists. Use --force to overwrite.")

    routers = build_routers(args.routers, compute_device)
    print(f"=== Routing Evaluation: {emb_dir.name} ===")
    print(f"dataset={metadata['dataset']} descriptor={metadata['descriptor']} tasks={len(task_specs)}")
    print(f"routers={list(routers.keys())}")
    print(f"routing_device={compute_device} embed_dtype={embed_dtype_name}")

    start_time = time.time()
    results = run_incremental_eval(emb_dir, task_specs, routers, embed_dtype=embed_dtype)
    elapsed = time.time() - start_time

    summary = {}
    print("\n=== Final Summary ===")
    for router_name, steps in results.items():
        final = steps[-1]
        avg_macro = float(np.mean([step["macro_accuracy"] for step in steps]))
        summary[router_name] = {
            "final_macro_accuracy": final["macro_accuracy"],
            "final_micro_accuracy": final["micro_accuracy"],
            "avg_macro_accuracy": avg_macro,
            "step_macro_accuracies": [float(step["macro_accuracy"]) for step in steps],
        }
        print(
            f"{router_name:24s} final_macro={final['macro_accuracy'] * 100:6.2f}% "
            f"final_micro={final['micro_accuracy'] * 100:6.2f}% avg_macro={avg_macro * 100:6.2f}%")

    report = {
        "embedding_dir": str(emb_dir),
        "dataset": metadata["dataset"],
        "descriptor": metadata["descriptor"],
        "routing_device": compute_device,
        "embed_dtype": embed_dtype_name,
        "elapsed_seconds": elapsed,
        "task_specs": task_specs,
        "results": results,
        "summary": summary,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[done] wrote report to {out_path}")
    return out_path


def main() -> None:
    run_routing(parse_args())


if __name__ == "__main__":
    main()
