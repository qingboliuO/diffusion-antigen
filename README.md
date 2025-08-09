# DiffAD: Diffusion Model for Antigen Distance Prediction

## Overview
DiffAD is a novel diffusion-based model designed for predicting antigenic distances in influenza viruses. Our approach leverages receptor binding site (RBS) constraints and learnable attention mechanisms to achieve superior performance in antigenic distance prediction tasks.

## Model Architecture
![Model Architecture](/model.png)
*Figure 1: Architecture of the DiffAD model showing the diffusion process and attention mechanisms*

## Key Features
- 🧬 Specialized for influenza HA protein antigenic distance prediction
- 🔄 Novel diffusion-based approach with RBS constraints
- 🎯 Learnable attention mechanism for critical antigenic sites
- 📊 Superior performance with only 73.92M parameters
- 🚀 Outperforms large pre-trained models (ESM2, ProGen2, BioGPT, etc.)

## Installation

git clone https://github.com/qingboliuO/diffusion-antigen.git
cd diffusion-antigen
pip install -r requirements.txt

## Environment Information
Our model was developed and tested with the following environment:
Python: 3.8.19
PyTorch: 1.12.0
CUDA: 11.3
NumPy: 1.24.3
Pandas: 2.0.3
Matplotlib: 3.7.2
Scikit-learn: 1.3.0

