# Theory Verify

This folder contains an offline, zero-rehearsal routing sanity check for the CV SRT setting.

Protocol:
- Reuse the original InfLoRA task order from `DataManager`.
- For each task, extract frozen vision embeddings from the current task train split and current task test split.
- Re-run continual routing offline: at step `t`, fit each router only on train embeddings from tasks `0..t`, then evaluate on all test tasks seen so far.
- The current benchmark of interest is DomainNet, because it provides a natural domain-incremental task geometry instead of artificial class blocks.

Hypothesis check:
- Keep the same InfLoRA repo and the same frozen SRT extraction path.
- Evaluate whether frozen SRT routing remains coherent when the benchmark is a natural domain-incremental one.
- The main target is `domainnet_natural`.

Main one-command hypothesis wrapper:

```bash
python expand_method_ComputerVision/theory_verify/verify_benchmark_hypothesis.py \
  --experiment domainnet_natural \
  --descriptor cls \
  --reuse_embeddings
```

This writes the DomainNet routing report and a hypothesis summary under `expand_method_ComputerVision/theory_verify/results/`.
If `--data_path` is omitted, the code will also try common Kaggle and local mount locations automatically.
If auto-detection still fails, pass the directory that contains `data/DomainNet/...`.

If you explicitly want the old artificial control comparison, you can still run:

```bash
python expand_method_ComputerVision/theory_verify/verify_benchmark_hypothesis.py \
  --experiment both \
  --descriptor cls \
  --reuse_embeddings
```

Extractor:

```bash
python expand_method_ComputerVision/theory_verify/extract_embeddings.py \
  --config expand_method_ComputerVision/InfLoRA/configs/domainnet_srt_inflora.json \
  --descriptor cls
```

Use `--data_path` whenever DomainNet is mounted outside the path stored in the config, or rely on the automatic fallback to common Kaggle/local roots. The value should be the parent directory of `data/DomainNet`, not the image folder itself.

Useful descriptor ablations:
- `cls`: matches the current runtime path.
- `mean_patch`: average of patch tokens only.
- `cls_mean_concat`: concatenation of `cls` and `mean_patch`.

Routing evaluation:

```bash
python expand_method_ComputerVision/theory_verify/routing_class.py \
  --emb_dir expand_method_ComputerVision/theory_verify/embeddings/domainnet__init69__inc69__cls
```

Routers currently included:
- `NearestCentroid`
- `CosineNearestCentroid`
- `OnlineZCAL2`
- `PooledMahalanobis_RIDGE`
- `PooledMahalanobis_OAS`

Debug options:
- `--max_tasks N`: only use the first `N` tasks.
- `--limit_per_split N`: only save the first `N` samples per split during extraction.
- `--routers nearest,online_zca,maha_ridge`: run a subset of routers.

Current intended workflow:
- Extract DomainNet frozen embeddings with `domainnet_srt_inflora.json`.
- Run offline routing on the extracted DomainNet directory.
- Use `verify_benchmark_hypothesis.py --experiment domainnet_natural` as the default entrypoint for this branch.
