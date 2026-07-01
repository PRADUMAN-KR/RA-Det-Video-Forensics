# RA-Det Video Forensics

This repository adapts the **RA-Det** (Robustness Asymmetry Detection) framework for video forensics, extending its capabilities to detect AI-generated and manipulated multimodal content.

## 🏆 Acknowledgments and Credits

This project builds heavily upon the foundational work presented in the ICML 2026 paper **"RA-Det: Towards Universal Detection of AI-Generated Images via Robustness Asymmetry."** Full credit for the core RA-Det architecture and methodology belongs to the original authors:
* **Authors:** Xinchang Wang, Yunhao Chen, Yuechen Zhang, Congcong Bian, Zihao Guo, Xingjun Ma, and Hui Li.
* **Original Paper:** [RA-Det: Towards Universal Detection of AI-Generated Images via Robustness Asymmetry](https://arxiv.org/abs/2603.01544)
* **Original Repository:** [dongdongunique/RA-Det](https://github.com/dongdongunique/RA-Det)

If you build upon the code in this repository, please ensure you cite the original RA-Det paper alongside this extension.

---

## 📖 Overview

While the original RA-Det framework focuses on image-based generative artifacts by probing robustness asymmetry (where synthetic images exhibit larger feature drift under small perturbations than natural images), this project extends those principles to temporal media. 

This repository expands the framework to include robust temporal modeling and audio-visual alignment, aiming to identify artifacts specific to video generation and manipulation, such as deepfake lip-sync anomalies and cross-modal inconsistencies.

## ⚙️ Installation

```bash
# Clone this repository
git clone [https://github.com/PRADUMAN-KR/RA-Det-Video-Forensics.git](https://github.com/PRADUMAN-KR/RA-Det-Video-Forensics.git)
cd RA-Det-Video-Forensics

# Install dependencies
pip install -r requirements.txt
