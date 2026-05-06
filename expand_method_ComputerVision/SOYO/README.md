# SOYO

This is the official implementation of our CVPR 2025 paper:
_Boosting Domain Incremental Learning: **S**electing the **O**ptimal Parameters is All **Yo**u Need_


## Environment

```bash
conda create -n soyo python=3.8
conda activate soyo
pip install -r requirements.txt
```

## Datasets

### DomainNet
Please refer to [DomainNet Project](http://ai.bu.edu/M3SDA/) to download the dataset or run:
```bash
cd dil_dataset
bash download_domainnet.sh
```
Then unzip the downloaded files, and confirm the file directory as shown below:
```
DomainNet
├── clipart
│   ├── aircraft_carrier
│   ├── airplane
│   ... ...
├── clipart_test.txt
├── clipart_train.txt
├── infograph
│   ├── aircraft_carrier
│   ├── airplane
│   ... ...
├── infograph_test.txt
├── infograph_train.txt
├── painting
│   ├── aircraft_carrier
│   ├── airplane
... ...
```
### CDDB
Please refer to [CDDB Project](https://github.com/Coral79/CDDB). If you want a reproducible scripted download, the easiest current source is the Hugging Face mirror at [nebula/CDDB](https://huggingface.co/datasets/nebula/CDDB), which hosts a `CDDB.tar` archive.

Quick download with Hugging Face CLI:

```bash
python -m pip install -U "huggingface_hub[cli]"
mkdir -p dil_dataset/_hf_cddb
huggingface-cli download nebula/CDDB CDDB.tar --repo-type dataset --local-dir dil_dataset/_hf_cddb
mkdir -p dil_dataset
tar -xf dil_dataset/_hf_cddb/CDDB.tar -C dil_dataset
```

If you download to a temporary location like `/tmp`, make sure the extraction target already exists before using `tar -C ...`.

Browser-based alternative:
- Open [nebula/CDDB](https://huggingface.co/datasets/nebula/CDDB).
- Go to the `Files` tab and download `CDDB.tar`.
- Create `dil_dataset/` first if needed, then extract it there.

The original project page also links the older Google Drive release: [CDDB Dataset](https://drive.google.com/file/d/1NgB8ytBMFBFwyXJQvdVT_yek1EaaEHrg/view?usp=sharing).

After extraction, make sure the final folder name is exactly `CDDB` so the shipped configs can use `data_path: dil_dataset/CDDB`.

Then unzip the downloaded files, and confirm the file directory as shown below:
```
CDDB
├── biggan
│   ├── train
│   └── val
├── gaugan
│   ├── train
│   └── val
├── san
│   ├── train
│   └── val
├── whichfaceisreal
│   ├── train
│   └── val
├── wild
│   ├── train
│   └── val
... ...
```


### CORe50
Please refer to the official [CORe50 Project](https://vlomonaco.github.io/core50/index.html#dataset). For SOYO, you do not need the full image tree; the shipped loader only needs the four files below for the NI scenario.

Quick download:

```bash
mkdir -p dil_dataset/CORe50
cd dil_dataset/CORe50
curl -L -O http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz
curl -L -O https://vlomonaco.github.io/core50/data/paths.pkl
curl -L -O https://vlomonaco.github.io/core50/data/labels.pkl
curl -L -O https://vlomonaco.github.io/core50/data/LUP.pkl
```

If you prefer downloading through the browser, the same page exposes direct links for `core50_imgs.npz`, `paths.pkl`, `labels.pkl`, and `LUP.pkl`.

After the download, confirm the folder layout below and keep `data_path: dil_dataset/CORe50` in the config.

Please refer to [CORe50 Project](https://vlomonaco.github.io/core50/index.html#dataset) and download the file shown below:
```
CORe50
├── core50_imgs.npz
├── labels.pkl
├── LUP.pkl
└── paths.pkl
```



## Training and Inference
Please confirm the path of your datasets in the config files.
### DomainNet
```
python main.py --config configs/domainnet_soyo_vit.yaml --device 0
python main.py --config configs/domainnet_soyo_clip.yaml --device 0
```

### CDDB
```
python main.py --config configs/cddb_soyo_vit.yaml --device 0
python main.py --config configs/cddb_soyo_clip.yaml --device 0
```

### CORe50
```
python main.py --config configs/core50_soyo_vit.yaml --device 0
python main.py --config configs/core50_soyo_clip.yaml --device 0
```

### Domain Incremental Object Detection
Please refer to [LDB](https://github.com/SONGX1997/LDB) project.

### Domain Incremental Speech Enhancement
Please refer to [LNA](https://ieeexplore.ieee.org/abstract/document/10720156) paper.

## Acknowledgement
We thank [PyCIL](https://github.com/G-U-N/PyCIL) and [S-Prompts](https://github.com/iamwangyabin/S-Prompts) for their wonderful framework and codes!
We also thank [CLIP](https://github.com/openai/CLIP) and [CoOp](https://github.com/KaiyangZhou/CoOp) for their helpful components.


## Citation
If any part of our paper and code is helpful to your research, please consider citing the following bib entry:
```
@inproceedings{wang2025boosting,
  title={Boosting Domain Incremental Learning: Selecting the Optimal Parameters is All You Need},
  author={Wang, Qiang and Song, Xiang and He, Yuhang and Han, Jizhou and Ding, Chenhao and Gao, Xinyuan and Gong, Yihong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={4839--4849},
  year={2025}
}

@inproceedings{wang2024non,
  title={Non-exemplar domain incremental learning via cross-domain concept integration},
  author={Wang, Qiang and He, Yuhang and Dong, Songlin and Gao, Xinyuan and Wang, Shaokun and Gong, Yihong},
  booktitle={European Conference on Computer Vision},
  pages={144--162},
  year={2024},
  organization={Springer}
}
```
