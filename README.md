# [CVPR2026] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-green?style=for-the-badge&logo=googlechrome&logoColor=white)](https://yayuanli.github.io/MATT/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.20525-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.20525)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-ffcc00?style=for-the-badge)](https://huggingface.co/datasets/yayuanli/MATT-Bench)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

**[Yayuan Li](https://www.linkedin.com/in/yayuan-li-148659272/)<sup>1</sup> · [Aadit Jain](https://www.linkedin.com/in/jain-aadit/)<sup>1</sup> · [Filippos Bellos](https://www.linkedin.com/in/filippos-bellos-168595156/)<sup>1</sup> · [Jason J. Corso](https://www.linkedin.com/in/jason-corso/)<sup>1,2</sup>**

<sup>1</sup>University of Michigan | <sup>2</sup>Voxel51

![Mistake Attribution Teaser](assets/teaser.png)
</div>

## 📑 0. Open-Source Plan

We plan to release all components of our project according to the following schedule:

- ✅ Paper release
- ✅ Project page setup
- ✅ datasets release
- ✅ MisFormer inference code & model weights
- ✅ MisFormer training scripts
- ✅ MisEngine data construction pipeline
- ⬜ Baseline implementations

## 📝 1. Abstract

We introduce **Mistake Attribution (MATT)**, a new task for fine-grained understanding of human mistakes in egocentric videos. Beyond detecting whether a mistake occurs, MATT attributes the mistake to **what** semantic role is violated, **when** the deviation becomes irreversible (the Point-of-No-Return), and **where** it appears in the frame. We contribute **MisEngine**, a scalable data engine that yields Ego4D-M (257K) and EPIC-KITCHENS-M (221K), and **MisFormer**, a unified model that outperforms task-specific SOTA methods across all attribution subtasks.

## 🛠️ 2. Environment Setup

> **Tested Environment**: CUDA 11.7, Ubuntu, Python 3.9

### 2.1. MisFormer (`misformer`)

```bash
# Clone Repository
git clone https://github.com/yayuanli/MATT.git
cd MATT/misformer

# Create Environment (Python 3.9 recommended)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The MisFormer visual backbone requires a pre-trained [LaViLa](https://github.com/facebookresearch/LaViLa) checkpoint. Download the **LAVILA TSF-L Epoch 3** dual encoder (`clip_openai_timesformer_large.narrator_rephraser.ep_0003`) from the [LaViLa Model Zoo](https://github.com/facebookresearch/LaViLa/blob/main/docs/MODEL_ZOO.md#zero-shot) and place it at `misformer/model/checkpoint_best.pt` (or pass a custom path via `--LaViLa_ckpt`).

### 2.2. MisEngine (`misengine`)

> **Note:** MisEngine requires **Python 3.9** due to AllenNLP's dependency stack (fails on 3.11+).

```bash
cd MATT/misengine

# Create Environment (requires Python 3.9)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2.3. Ego4D
> You don't need to go throught this section if you just want to learn the MisEngine process.
Follow [MATT-Bench @ Hugging Face](https://huggingface.co/datasets/mistakeattribution/MATT-Bench) to download dataset and annotations to `misengine/ego4d/dat/`. It'd be ~1TB so consider downloading them to a different storage and make link (`ln`).

Expect folder structure
```
misengine/ego4d/dat/
├── clips
│   ├── <clip1_uid>.mp4 
│   ├── <clip2_uid>.mp4 
│   ├── ... 
│   ├── manifest.csv 
│   ├── manifest.ver 
├── parquet.xlsx
├── test.xlsx
├── train.xlsx
├── valid.xlsx
```

**Frame Extraction:** The dataloader expects extracted frames (~6TB) in the following structure.

```
misengine/ego4d/dat/
├── clips_frames
│   ├── <clip1_uid>_frames
│   │   ├── 00001.png
│   │   ├── ...
│   ├── <clip2_uid>_frames
│   │   ├── 00001.png
│   │   ├── ... 
│   ├── manifest.csv 
│   ├── manifest.ver 
├── parquet.xlsx
├── test.xlsx
├── train.xlsx
├── valid.xlsx
```

Use the provided script to extract frames from all clips:
> **Note:** Consider modifing the `output_directory` path in the script to a large storage and make link to `clips_frames` in current folder. Or, one could conceptually split clips into subsets and put each subset in different large storage. The dataloader accepts up to three root paths via `--root1`, `--root2`, and `--root3` and searches across all of them, so there could be upto 3 storage subsets.  

```bash
cd misengine/ego4d

python extract_frames.py
```

This scans `clips_dir` for all `.mp4` files and extracts frames at 640×360 resolution using `ffmpeg`.


### 2.4. EPIC-Kitchens
> You don't need to go throught this section if you just want to learn the MisEngine process.
Follow [MATT-Bench @ Hugging Face](https://huggingface.co/datasets/mistakeattribution/MATT-Bench) to download dataset and annotations to `misengine/epickitchens/dat/`. It'd be ~1TB so consider downloading them to a different storage and make link (`ln`).

**Frame Extraction:** The dataloader expects the standard EPIC-Kitchens frame layout:

```
misengine/epickitchens/dat/frames/<participant_id>/rgb_frames/<video_id>/frame_0000000001.jpg
misengine/epickitchens/dat/frames/<participant_id>/rgb_frames/<video_id>/frame_0000000002.jpg
...
```

### 2.5. HoloAssist 
> You don't need to go throught this section if you just want to learn the MisEngine process.
Although not reported in the paper, we also support the HoloAssist dataset.

Download the following from the [HoloAssist project page](https://holoassist.github.io/):

| Resource | Link | Size |
|----------|------|------|
| Videos (pitch-shifted) | [video_pitch_shifted.tar](https://hl2data.z5.web.core.windows.net/holoassist-data-release/video_pitch_shifted.tar) | 184.20 GB |
| Labels | [data-annotation-trainval-v1_1.json](https://hl2data.z5.web.core.windows.net/holoassist-data-release/data-annotation-trainval-v1_1.json) | 111 MB |
| Dataset splits | [data-splits-v1_2.zip](https://holoassist.github.io/label_files/data-splits-v1_2.zip) | — |

**Frame Extraction:**

```bash
cd misengine/holoassist

python extract_frames.py \
  --input df_fg_output.xlsx \
  --video_base_path /path/to/HoloAssist/video_pitch_shifted
```

Frames are written to `<video_dir>/Export_py/video_frames/frame_00001.jpg` (5-digit zero-padded) within each video's directory.

## 🔧 3. MisEngine Data Construction Pipeline
> You don't need to go throught this section if you just want to inference/train MisFormer model since you have downloaded the constrcuted data in the Environment Setup section.
In this repo, we apply MisEngine to the Ego4D, EPIC-Kitchens, and HoloAssist datasets by programmatically generating semantic-role misalignment samples from existing egocentric video annotations. 

**The methodology can be applied to any other dataset that follows the required structure (i.e., most action recognition, video captioning dataset). For datasets that do not originally have semantic roles labels, we recommand [AllenNLP SRL](https://docs.allennlp.org/models/main/models/structured_prediction/predictors/srl/).**

All augmentation scripts produce samples with four label classes:

| Label | Meaning                           |
|-------|-----------------------------------|
| 0     | Aligned (no misalignment)         |
| 1     | Verb misaligned                   |
| 2     | Argument misaligned               |
| 3     | Both verb and argument misaligned |

### 3.1. Ego4D

The Ego4D pipeline maps video-level annotations to clip-level frame coordinates and then generates misalignment samples.

> **Note on ambiguous samples:** The original Ego4D annotations contain video segments, defined by `(video_uid, start_frame, end_frame)`, that are associated with multiple distinct verb or argument labels. These accounted for a small percentage of the data used in our training/inference splits. The provided parquet and split files have been **pre-filtered** to remove these ambiguous video segments. Note that the provided splits have considerable overlap with, but are not identical to, the splits used to train our released checkpoints.

**Step 1 — Map to clip coordinates:**

```bash
cd misengine/ego4d

python clips.py \
  --metadata /path/to/ego4d.json \
  --input parquet.xlsx \
  --output clips.xlsx
```

Reads the Ego4D metadata JSON and the cleaned parquet export. Outputs `clips.xlsx` with clip UIDs and clip-local frame ranges (`clip1_uid`, `clip1_start_frame`, `clip1_end_frame`, and `clip2_*` fields for segments that span two clips).

**Step 2 — Generate misalignment samples:**

```bash
python augment.py \
  --input clips.xlsx \
  --clips_dir /path/to/ego4d/clips \
  --output all_clips_samples.xlsx
```

Filters for clips that exist on disk, groups samples by `(V, ARG1)`, and creates balanced misalignment samples. Outputs `all_clips_samples.xlsx`.

**Step 3 — Create train/valid/test splits:**

```bash
python create_splits.py \
  --input all_clips_samples.xlsx \
  --output_dir /path/to/output \
  --seed 42
```

Randomly shuffles and splits into 80/10/10 train/valid/test. Outputs `train.xlsx`, `valid.xlsx`, `test.xlsx`.

### 3.2. EPIC-Kitchens

EPIC-Kitchens uses a single augmentation script that reads directly from the [official EPIC-Kitchens-100 annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations).

```bash
cd misengine/epickitchens

python augment.py \
  --split train \
  --annotations_dir /path/to/epic-kitchens-100-annotations \
  --output train.xlsx
```

Reads `EPIC_100_{split}.csv`, renames columns to match the MisFormer schema (`verb` → `V`, `noun` → `ARG1`, `stop_frame` → `end_frame`), groups by `(V, ARG1)`, and generates balanced misalignment samples. Repeat for `validation` and `test` splits.

### 3.3. HoloAssist

The HoloAssist pipeline extracts fine-grained action annotations from the official JSON, splits by video ID, and generates misalignment samples.

> **Note on ambiguous samples:** Like Ego4D, some HoloAssist segments have multiple distinct verb or argument labels for the same `(video_id, start_frame, end_frame)` tuple. `df_fg.py` **automatically filters** these ambiguous segments and reports how many were removed. The provided split files have already been filtered. Our released checkpoints were trained on splits that still included these ambiguous samples, but they represented a negligible fraction of the data; the provided files have considerable overlap with the splits we trained on.

**Step 1 — Parse fine-grained annotations:**

```bash
cd misengine/holoassist

python df_fg.py \
  --json_path /path/to/data-annotation-trainval-v1_1.json \
  --output df_fg_output.xlsx
```

Extracts all "Fine grained action" events from the HoloAssist annotation JSON, computes frame-level boundaries from timestamps and per-video FPS, and filters ambiguous segments. Outputs `df_fg_output.xlsx`.

**Step 2 — Split by video ID:**

```bash
python split.py \
  --input df_fg_output.xlsx \
  --train_ids /path/to/train.txt \
  --val_ids /path/to/val.txt \
  --test_ids /path/to/test.txt \
  --output_dir .
```

Partitions annotations into `train_base.xlsx`, `validation_base.xlsx`, and `test_base.xlsx` using the official video ID lists from the dataset splits download.

**Step 3 — Generate misalignment samples:**

```bash
python augment.py --split train
python augment.py --split validation
python augment.py --split test
```

For each split, reads `{split}_base.xlsx`, groups by `(V, ARG1)`, and generates balanced misalignment samples. Outputs `train.xlsx`, `validation.xlsx`, `test.xlsx`.


## 🚀 4. Evaluation & Inference

Evaluation is run via `eval_model.py`, which **automatically downloads** the appropriate model checkpoints from Hugging Face (`mistakeattribution/<dataset>`). See `python eval_model.py -h` for the full list of options.

```bash
cd misformer
export PYTHONPATH=$(pwd):$PYTHONPATH

python eval_model.py \
  --dataset <dataset_name> \
  --root1 /path/to/frames \
  --test_dataset_path /path/to/test.xlsx \ \
  --clip_length <num_frames>
```

Key notes:

- Set `--dataset` to `ego4d`, `epic-kitchens`, or `holoassist`.
- Use `--clip_length 30` for Ego4D / EPIC-Kitchens and `--clip_length 8` for HoloAssist.
- For Ego4D, `--root2` and `--root3` can specify additional frame directories if frames span multiple drives.
- `--test_dataset_path` is in downloaded dataset annotations (e.g., `misengine/ego4d/dat/test.xlsx`)

## 📊 5. Model Training

Training uses Distributed Data Parallel (DDP) and requires **at least 2 GPUs**. Experiment logging is handled by [Weights & Biases](https://wandb.ai/).

### 5.1. Weights & Biases Setup

```bash
pip install wandb   # included in requirements.txt
wandb login         # paste your API key when prompted
```

### 5.2. Running Training

See `python training.py -h` for the full list of options.

```bash
cd misformer
export PYTHONPATH=$(pwd):$PYTHONPATH
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=12355

CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py \
  --dataset <dataset_name> \
  --root1 /path/to/frames \
  --train_dataset_path /path/to/train.xlsx \
  --valid_dataset_path /path/to/valid.xlsx \
  --output_dir /path/to/checkpoints \
  --recording_epochs /path/to/log.txt \
  --pretrained_ckpt None \
  --wandb_project <project_name>
```

Key notes:

- Set `--dataset` to `ego4d`, `epic-kitchens`, or `holoassist`.
- Use `--clip_length 8` for HoloAssist (default 30 works for all others).
- For Ego4D, use `--root2` / `--root3` if frames span multiple drives.


## 📜 Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{li2026mistakeattribution,
  title     = {Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos},
  author    = {Li, Yayuan and Jain, Aadit and Bellos, Filippos and Corso, Jason J.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}
```
