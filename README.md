# Next-Frame Prediction using Pix2Pix GAN on Moving MNIST

This project implements a **Pix2Pix GAN** for predicting the next frame in a sequence from the **Moving MNIST** dataset. It combines adversarial training and L1 reconstruction loss to generate future frames that preserve digit structure and motion.

---

## 🧾 Files

- `pix2pix.ipynb`: Main notebook containing training, testing, and visualization logic

---

## 📁 Dataset

- **Dataset Used:** Moving MNIST (grayscale, 64x64)
- **Source:** [https://www.cs.toronto.edu/~nitish/unsupervised_video/](https://www.cs.toronto.edu/~nitish/unsupervised_video/)
- **Required File:** `mnist_test_seq.npy`

### 🔻 How to Set Up the Dataset

1. Download `mnist_test_seq.npy` from [this link](https://www.cs.toronto.edu/~nitish/unsupervised_video/).
2. Place it in the root directory of this repository (same location as the notebook).

---

## 🧠 Model Overview

- **Generator:** U-Net-based architecture
- **Discriminator:** PatchGAN (penalizes structure at the patch level)
- **Losses Used:**
  - Binary Cross Entropy (adversarial loss)
  - L1 Loss (reconstruction)
  - Total Generator Loss:\
    `Gen_Loss = BCE_Loss + λ * L1_Loss` (with λ = 10)

---

## 🚀 How to Run

Make sure you have the following installed:

```bash
pip install torch torchvision matplotlib tqdm numpy
