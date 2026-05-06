# Theory Verify

This folder contains offline, zero-rehearsal routing sanity checks for the CV SRT setting.

Protocol:
- Reuse the original continual task order from the selected backend.
- For each task, extract frozen vision embeddings from the current task train split and current task test split.
- Re-run continual routing offline: at step `t`, fit each router only on train embeddings from tasks `0..t`, then evaluate on all test tasks seen so far.
- Two extraction backends now exist:
  - `S-Prompts CLIP`: a frozen CLIP ViT-B/16 path for true domain-incremental Office-Home.
  - `SOYO CLIP`: a frozen CLIP path that reuses SOYO's continual `DataManager` for `CORe50` and `CDDB` while bypassing SOYO's selector.

Hypothesis check:
- Keep the same offline SRT routing logic.
- Swap only the benchmark and frozen backbone when needed.
- Primary target for the new CLIP track is `soyo_clip_pair`.

Main one-command hypothesis wrapper:

```bash
python expand_method_ComputerVision/theory_verify/verify_benchmark_hypothesis.py \
  --experiment soyo_clip_pair \
  --descriptor cls \
  --data_path /path/to/parent-containing-CORe50-and-CDDB \
  --reuse_embeddings
```

This writes the routing reports and a combined summary under `expand_method_ComputerVision/theory_verify/results/`.

Important:
- `CORe50` and `CDDB` are not shipped inside the repo.
- The SOYO repo only contains code and configs.
- You need to download or mount the datasets separately, then point `--data_path` either to the dataset root itself or to a parent folder that contains `CORe50` and/or `CDDB`.
- In the current workspace, no local `CORe50` or `CDDB` folder was found automatically.
- Prefer an absolute `--data_path` when possible. If you pass a relative path such as `expand_method_ComputerVision/both_data`, it is resolved from the workspace root.

Quick dataset sources:
- `CDDB`: use the Hugging Face mirror [nebula/CDDB](https://huggingface.co/datasets/nebula/CDDB). It currently hosts a `CDDB.tar` archive. A scripted download is:

```bash
python -m pip install -U "huggingface_hub[cli]"
mkdir -p /tmp/cddb_hf
huggingface-cli download nebula/CDDB CDDB.tar --repo-type dataset --local-dir /tmp/cddb_hf
mkdir -p /path/to/parent-containing-CORe50-and-CDDB
tar -xf /tmp/cddb_hf/CDDB.tar -C /path/to/parent-containing-CORe50-and-CDDB
```

If `tar` prints `Cannot open: No such file or directory`, the usual cause is that the directory passed to `-C` does not exist yet. Create it first with `mkdir -p ...` and rerun only the `tar -xf ...` line.

If `/tmp/cddb_hf/CDDB.tar` is missing when you retry, download it again or use a non-temporary location instead of `/tmp`.

- `CORe50`: use the official page [CORe50 Project](https://vlomonaco.github.io/core50/index.html#dataset). For this repo, the minimum required files are `core50_imgs.npz`, `paths.pkl`, `labels.pkl`, and `LUP.pkl`. A direct download is:

```bash
mkdir -p /path/to/parent-containing-CORe50-and-CDDB/CORe50
cd /path/to/parent-containing-CORe50-and-CDDB/CORe50
curl -L -O http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz
curl -L -O https://vlomonaco.github.io/core50/data/paths.pkl
curl -L -O https://vlomonaco.github.io/core50/data/labels.pkl
curl -L -O https://vlomonaco.github.io/core50/data/LUP.pkl
```

Office-Home with the frozen CLIP backbone copied from S-Prompts:

```bash
python expand_method_ComputerVision/theory_verify/verify_benchmark_hypothesis.py \
  --experiment officehome_sprompt_clip \
  --descriptor cls \
  --task_order Art,Clipart,Product,Real_World \
  --reuse_embeddings
```

SOYO-CLIP with the continual streams from `CORe50` and `CDDB`:

```bash
python expand_method_ComputerVision/theory_verify/verify_benchmark_hypothesis.py \
  --experiment soyo_clip_pair \
  --descriptor cls \
  --data_path /path/to/parent-containing-CORe50-and-CDDB \
  --reuse_embeddings
```

Here `/path/to/parent-containing-CORe50-and-CDDB` means a directory shaped like this:

```text
datasets/
├── CORe50/
│   ├── core50_imgs.npz
│   ├── labels.pkl
│   ├── LUP.pkl
│   └── paths.pkl
└── CDDB/
    ├── biggan/
    ├── gaugan/
    ├── san/
    ├── whichfaceisreal/
    └── wild/
```

You can also run each SOYO benchmark separately:

```bash
python expand_method_ComputerVision/theory_verify/verify_benchmark_hypothesis.py \
  --experiment core50_soyo_clip \
  --descriptor cls \
  --data_path /path/to/CORe50-or-its-parent \
  --reuse_embeddings
```

```bash
python expand_method_ComputerVision/theory_verify/verify_benchmark_hypothesis.py \
  --experiment cddb_soyo_clip \
  --descriptor cls \
  --data_path /path/to/CDDB-or-its-parent \
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

Standalone Office-Home CLIP extractor:

```bash
python expand_method_ComputerVision/theory_verify/extract_embeddings_sprompt_clip.py \
  --descriptor cls \
  --task_order Art,Clipart,Product,Real_World
```

Standalone SOYO-CLIP extractor:

```bash
python expand_method_ComputerVision/theory_verify/extract_embeddings_soyo_clip.py \
  --config expand_method_ComputerVision/SOYO/configs/core50_soyo_clip.yaml \
  --descriptor cls \
  --data_path /path/to/CORe50-or-its-parent
```

```bash
python expand_method_ComputerVision/theory_verify/extract_embeddings_soyo_clip.py \
  --config expand_method_ComputerVision/SOYO/configs/cddb_soyo_clip.yaml \
  --descriptor cls \
  --data_path /path/to/CDDB-or-its-parent
```

For SOYO-based extraction, `--data_path` can point either to the dataset root itself (`CORe50` or `CDDB`) or to a parent directory that contains one of those folders.
The SOYO extractor currently supports `--descriptor cls` only, because SOYO exposes pooled CLIP image embeddings rather than token grids.

Useful descriptor ablations:
- `cls`: matches the current runtime path.
- `mean_patch`: average of patch tokens only.
- `cls_mean_concat`: concatenation of `cls` and `mean_patch`.

Routing evaluation:

```bash
python expand_method_ComputerVision/theory_verify/routing_class.py \
  --emb_dir expand_method_ComputerVision/theory_verify/embeddings/core50__soyoclip__init50__inc50__cls
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
- Extract Office-Home frozen CLIP embeddings with `extract_embeddings_sprompt_clip.py` when you want the new CLIP/DIL sanity check.
- Extract `CORe50` or `CDDB` frozen embeddings with `extract_embeddings_soyo_clip.py` when you want the new SOYO-based continual stream without the original selector.
- Run offline routing on the extracted directory with `routing_class.py`.
- Use `verify_benchmark_hypothesis.py --experiment soyo_clip_pair` as the default path for `CORe50 + CDDB`, or `--experiment officehome_sprompt_clip` if you specifically want Office-Home.
