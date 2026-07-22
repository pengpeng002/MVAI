# Towards Generalized Deepfake Detection by Leveraging Multiple Views

Official PyTorch implementation of **MVAF**, the detector proposed in
**Towards Generalized Deepfake Detection by Leveraging Multiple Views**.

<img src="./MVA_VIT.png" width="100%" alt="MVAF overall pipeline">

## Overview

This repository provides:

- training and inference code for MVAF;
- dataset organization instructions for the train, validation, and test sets;
- released model weights for direct reproduction;
- scripts that report Accuracy, Average Precision, F1, and ROC-AUC for each test subset.

The fastest way to reproduce the reported results is to download the released test data and checkpoint, then run `run.py`.

## Released Files

| File | Link | Notes |
|:--|:--|:--|
| Training data | [Baidu Disk](https://pan.baidu.com/s/1TuF46uaLjyF7XPpTL-OGWQ?pwd=m89y) | Images used for training. |
| Test data | [Baidu Disk](https://pan.baidu.com/s/1feNJO6r3w_QVpH6YYQNm9w?pwd=9v64) | Test subsets used for cross-dataset evaluation. |
| Model weight | [Baidu Disk](https://pan.baidu.com/s/14DXN4JRuG5lhrD_5e6TlKA?pwd=s72k) | Checkpoint for direct inference. |

Please download and extract the files before running the code. For archival review, we recommend keeping the extracted folders under this repository:

```text
MVAF/
|-- dataset/
|   |-- train/
|   |-- val/
|   `-- test/
|-- weights/
|   `-- best.pth
|-- run.py
|-- train_vit.py
`-- test_vit.py
```

## Environment

The code is tested with Python 3.12 and CUDA-enabled PyTorch. A GPU is strongly recommended because the model uses CLIP ViT-L/14 and Stable Diffusion VAE components.

```bash
conda create -n mvaf python=3.12 -y
conda activate mvaf
pip install -r requirements.txt
```

The first run may download pretrained CLIP and Stable Diffusion components. If the download is slow, configure a reachable Hugging Face mirror or cache these models in advance.

## Dataset Preparation

MVAF uses binary image folders where real images are stored in `0_real` and fake images are stored in `1_fake`.

The expected dataset layout is:

```text
dataset/
|-- train/
|   |-- AttGAN/
|   |   |-- 0_real/
|   |   `-- 1_fake/
|   |-- Palette/
|   |   |-- 0_real/
|   |   `-- 1_fake/
|   |-- ProGAN/
|   |   |-- 0_real/
|   |   `-- 1_fake/
|   |-- SD_v15/
|   |   |-- 0_real/
|   |   `-- 1_fake/
|   |-- StyleGAN2_FFHQ/
|   |   |-- 0_real/
|   |   `-- 1_fake/
|   `-- latent_diffusion_FFHQ/
|       |-- 0_real/
|       `-- 1_fake/
|-- val/
|   |-- dalle2/
|   |   |-- 0_real/
|   |   `-- 1_fake/
|   |-- midjourney/
|   |   |-- 0_real/
|   |   `-- 1_fake/
|   `-- stylegan3/
|       |-- 0_real/
|       `-- 1_fake/
`-- test/
    |-- MMD_GAN/
    |   |-- 0_real/
    |   `-- 1_fake/
    |-- MSG_STYLE_GAN/
    |   |-- 0_real/
    |   `-- 1_fake/
    |-- STARGAN/
    |   |-- 0_real/
    |   `-- 1_fake/
    `-- ...
```

The training and in-domain evaluation data are based on AI-Face-FairnessBench:
[https://github.com/Purdue-M2/AI-Face-FairnessBench](https://github.com/Purdue-M2/AI-Face-FairnessBench).

Additional datasets used in the paper can be obtained from their official sources:

- Diffusion Face: [https://github.com/Rapisurazurite/DiffFace](https://github.com/Rapisurazurite/DiffFace)
- DFFD: [http://cvlab.cse.msu.edu/dffd-dataset.html](http://cvlab.cse.msu.edu/dffd-dataset.html)
- DiFF: [https://github.com/iLearn-Lab/MM24-DiFF](https://github.com/iLearn-Lab/MM24-DiFF)

## Quick Reproduction

After downloading the released checkpoint and test data, run:

```bash
python test.py \
  --model_path ./weights/best.pth \
  --test_path ./dataset/test \
  --batch_size 64 \
  --save_file
```

The script evaluates every subset under `./dataset/test`. Each subset must contain `0_real/` and `1_fake/` folders.

Example output:

```text
Testing on dataset...
( 0 MMD_GAN     ) acc: xx.xx; ap: xx.xx f1: xx.xx; auc_roc: xx.xx
...
(Mean    ) mAcc: xx.xx, mAP: xx.xx, mF1: xx.xx, mAuc_Roc: xx.xx
```

When `--save_file` is enabled, detailed metrics are written to:

```text
test_results_<timestamp>/
|-- results_cross_dataset.json
```

Important arguments:

| Argument | Description |
|:--|:--|
| `--model_path` | Path to the released or trained checkpoint. |
| `--test_path` | Path to the directory containing test subsets. |
| `--batch_size` | Batch size for inference. Reduce this if GPU memory is insufficient. |
| `--save_file` | Save JSON and PKL result files. |

## Training

To train MVAF from scratch, organize the training and validation data as described above and run:

```bash
python train_vit.py \
  --name mvaf_train \
  --dataroot ./dataset \
  --train_split train \
  --val_split val \
  --batch_size 64 \
  --loss_freq 400 \
  --lr 0.00001 \
  --niter 50
```

Training logs and checkpoints are saved under:

```text
checkpoints/<experiment_name_with_timestamp>/
```

The training script keeps top validation checkpoints after epoch 3 and also uses early stopping. The saved checkpoint can be evaluated with `run.py` or `test_vit.py`.
