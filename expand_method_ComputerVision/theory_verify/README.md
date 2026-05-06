# Theory Verify

This folder contains offline, zero-rehearsal routing sanity checks for the CV SRT setting.

Protocol:
- Reuse the original InfLoRA task order from `DataManager`.
- For each task, extract frozen vision embeddings from the current task train split and current task test split.
- Re-run continual routing offline: at step `t`, fit each router only on train embeddings from tasks `0..t`, then evaluate on all test tasks seen so far.
- Two extraction backends now exist:
  - `InfLoRA`: the original ViT-based path already used for DomainNet.
  - `S-Prompts CLIP`: a new frozen CLIP ViT-B/16 path for true domain-incremental Office-Home.

Hypothesis check:
- Keep the same offline SRT routing logic.
- Swap only the benchmark and frozen backbone when needed.
- Primary target for the new CLIP track is `officehome_sprompt_clip`.
- The older InfLoRA track remains available as `domainnet_natural`.

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

Office-Home with the frozen CLIP backbone copied from S-Prompts:

```bash
python expand_method_ComputerVision/theory_verify/verify_benchmark_hypothesis.py \
  --experiment officehome_sprompt_clip \
  --descriptor cls \
  --task_order Art,Clipart,Product,Real_World \
  --reuse_embeddings
```

If `--data_path` is omitted, the code probes common local and Kaggle-style locations for Office-Home roots such as `OfficeHome`, `Office-Home`, or `OfficeHomeDataset_10072016`.
The path can point either to the dataset root itself or to a parent directory containing it.
Official download page: https://www.hemanthdv.org/officeHomeDataset.html

Kaggle note:
- If you attach the dataset as a Kaggle input, the common layout is `/kaggle/input/<dataset-slug>/OfficeHomeDataset_10072016/...`.
- The extractor now probes one level below `/kaggle/input`, so omission of `--data_path` should often work.
- If it still misses the mount, pass `--data_path /kaggle/input/<dataset-slug>` explicitly.

Example with an explicit Kaggle path:

```bash
python expand_method_ComputerVision/theory_verify/verify_benchmark_hypothesis.py \
  --experiment officehome_sprompt_clip \
  --descriptor cls \
  --task_order Art,Clipart,Product,Real_World \
  --data_path /kaggle/input/<dataset-slug> \
  --reuse_embeddings
```

Extractor:

```bash
python expand_method_ComputerVision/theory_verify/extract_embeddings.py \
  --config expand_method_ComputerVision/InfLoRA/configs/domainnet_srt_inflora.json \
  --descriptor cls
```

Standalone Office-Home CLIP extractor:

```bash
python expand_method_ComputerVision/theory_verify/extract_embeddings_sprompt_clip.py \
  --descriptor cls \
  --task_order Art,Clipart,Product,Real_World
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
- Extract DomainNet frozen embeddings with `domainnet_srt_inflora.json` when you want the original InfLoRA comparison.
- Extract Office-Home frozen CLIP embeddings with `extract_embeddings_sprompt_clip.py` when you want the new CLIP/DIL sanity check.
- Run offline routing on the extracted directory with `routing_class.py`.
- Use `verify_benchmark_hypothesis.py --experiment officehome_sprompt_clip` for the new CLIP-first benchmark path.
