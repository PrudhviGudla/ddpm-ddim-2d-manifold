# Denoising Diffusion Probabilistic Models (DDPM) & DDIM from Scratch

A from scratch implementation of Denoising Diffusion Probabilistic Models (Ho et al., 2020) and Denoising Diffusion Implicit Models (Song et al., 2020), built entirely from scratch in PyTorch without using libraries like `diffusers`.

This project trains a diffusion model to learn and generate a 2D manifold distribution (the "Dino" shape), demonstrating the core mathematical foundations of generative AI.

<img width="1464" height="528" alt="Result" src="https://github.com/user-attachments/assets/b65a6ee1-bffd-4fd5-84b0-b6a5e815d2e6" />

## Key Features
* **Zero "Black Box" Dependencies:** All logic (Forward Diffusion process, Reverse Denoising Step, Sampling loops) is implemented manually.
* **Custom Scheduler:** `LinearNoiseScheduler` class handling variance schedules ($\beta_t$, $\alpha_t$, $\bar{\alpha}_t$) and tensor broadcasting.
* **Sinusoidal Time Embeddings:** Transformer-style positional embeddings to condition the MLP on diffusion timesteps.
* **Dual Sampling Algorithms:**
    * **DDPM:** Stochastic sampling (Markovian) requiring 1000 steps.
    * **DDIM:** Deterministic sampling (Non-Markovian) achieving high quality in just 50-100 steps.
* **Evaluation:** Quantitative assessment using Chamfer Distance.

## Architecture
The model is a Residual MLP designed for 2D coordinate data:
* **Input:** 2D Point $(x, y)$ + Time Step $t$.
* **Embedding:** Sinusoidal Time Embeddings concatenated with projected input features and passed into fusion MLP 
* **Backbone:** 4-layer Residual MLP blocks with `ReLU` activations.
* **Objective:** Simple MSE Loss ($||\epsilon - \epsilon_\theta(x_t, t)||^2$) predicting the noise added at step $t$.

## Getting Started

### Prerequisites
* Python 3.8+
* PyTorch
* Matplotlib, NumPy, Pandas, Seaborn

### Dataset
This model relies on a 2D point cloud dataset (CSV format with `x` and `y` columns).
* **Note:** The specific dinosaur dataset used in the visualization is part of the "Datasaurus Dozen". You can download a compatible version [here](https://www.autodesk.com/research/publications/same-stats-different-graphs) or use your own 2D shape CSV.
* Place your `.csv` file in the root directory and update the `dataset_path` variable in the notebook.

### Usage
1.  Clone the repository:
    ```bash
    git clone https://github.com/PrudhviGudla/ddpm-ddim-2d-manifold
    ```
2.  Open the notebook `Diffusion_Scratch_2D_Dino_Dataset.ipynb` in Jupyter or Google Colab.
3.  Run all cells to train the model and generate the overlay plots.

## Results
Comparison of sampling methods after training for 50k epochs:

| Method | Steps | Characteristics | Chamfer Distance |
| :--- | :--- | :--- | :--- |
| **DDPM** | 1000 | Stochastic, adds noise at every step | `0.1354` |
| **DDIM** | 50 | Deterministic, ODE-like trajectory | `0.1426` |

## References
1.  Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
2.  Song, J., Meng, C., & Ermon, S. (2020). [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502).
