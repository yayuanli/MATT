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
- ✅ Ego4D-M & EPIC-KITCHENS-M datasets
- ⬜ MisFormer inference code & model weights
- ⬜ MisFormer training scripts
- ⬜ MisEngine data construction pipeline
- ⬜ Baseline implementations

## 📝 1. Abstract

We introduce **Mistake Attribution (MATT)**, a new task for fine-grained understanding of human mistakes in egocentric videos. Beyond detecting whether a mistake occurs, MATT attributes the mistake to **what** semantic role is violated, **when** the deviation becomes irreversible (the Point-of-No-Return), and **where** it appears in the frame. We contribute **MisEngine**, a scalable data engine that yields Ego4D-M (257K) and EPIC-KITCHENS-M (221K), and **MisFormer**, a unified model that outperforms task-specific SOTA methods across all attribution subtasks.


## 🛠️ 2. Environment Setup

> **Tested Environment**: CUDA 11.7, Ubuntu, Python 3.9

> **Note**: Python 3.9 is recommended

```bash
# Clone Repository
git clone https://github.com/yayuanli/MATT.git
cd MATT/semantic_attr

# Create Environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

## 🚀 3. Evaluation & Inference

EgoPER:



Ego4D, EpicKitchen, HoloAssist:




## 📊 4. Model Training

Training uses [Weights & Biases](https://wandb.ai/) for logging. Make sure you have run `wandb login` during environment setup.


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
