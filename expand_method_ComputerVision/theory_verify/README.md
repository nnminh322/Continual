# Theory Verify

This folder contains an offline, zero-rehearsal routing sanity check for the CV SRT setting.

Protocol:
- Reuse the original InfLoRA class-incremental task order from `DataManager`.
- For each task, extract frozen vision embeddings from the current task train split and current task test split.
- Re-run continual routing offline: at step `t`, fit each router only on train embeddings from tasks `0..t`, then evaluate on all test tasks seen so far.

Extractor:

```bash
python expand_method_ComputerVision/theory_verify/extract_embeddings.py \
  --config expand_method_ComputerVision/InfLoRA/configs/domainnet_srt_inflora.json \
  --descriptor cls
```

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
