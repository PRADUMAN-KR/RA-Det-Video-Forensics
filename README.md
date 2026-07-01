# RA-Det Video Forensics
Research implementation extending the RA-Det framework from image-level representation analysis to temporal video representation analysis for detecting modern AI-generated videos, including neural rendering, diffusion-based generation, and next-generation deepfakes.


## 📖 Overview

While the original RA-Det framework focuses on image-based generative artifacts by probing robustness asymmetry (where synthetic images exhibit larger feature drift under small perturbations than natural images), this project extends those principles to temporal media. 

RA-Det-Video-Forensics is a research project that investigates whether the core ideas introduced in RA-Det can be extended from static images to videos.

Unlike conventional deepfake detectors that primarily rely on low-level visual artifacts, this project explores representation-aware temporal analysis, where the behavior of semantic feature representations is analyzed across time to detect synthetic videos.

The long-term goal is to improve the detection of modern AI-generated videos produced by:

Neural Rendering
Diffusion-based Video Models
Talking-Head Generation Models
Face Reenactment Models
Future Video Foundation Models


## 🏆 Acknowledgments and Credits

This project builds heavily upon the foundational work presented in the ICML 2026 paper **"RA-Det: Towards Universal Detection of AI-Generated Images via Robustness Asymmetry."** Full credit for the core RA-Det architecture and methodology belongs to the original authors:
* **Authors:** Xinchang Wang, Yunhao Chen, Yuechen Zhang, Congcong Bian, Zihao Guo, Xingjun Ma, and Hui Li.
* **Original Paper:** [RA-Det: Towards Universal Detection of AI-Generated Images via Robustness Asymmetry](https://arxiv.org/abs/2603.01544)
* **Original Repository:** [dongdongunique/RA-Det](https://github.com/dongdongunique/RA-Det)

If you build upon the code in this repository, please ensure you cite the original RA-Det paper alongside this extension.

---



## ⚙️ Installation

```bash
# Clone this repository
git clone [https://github.com/PRADUMAN-KR/RA-Det-Video-Forensics.git](https://github.com/PRADUMAN-KR/RA-Det-Video-Forensics.git)
cd RA-Det-Video-Forensics

# Install dependencies
pip install -r requirements.txt



