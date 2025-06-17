Project Overview

This repository contains a machine learning pipeline for classifying fermentation time of Lactococcus lactis into two categories based on genomic data. The model leverages three types of data matrices—GENE, SNP, and IGR—to build a robust binary classification model. The approach integrates feature selection, semi-supervised learning, and dual diffusion models for data augmentation to enhance classification performance. This method is particularly suited for handling high-dimensional biological datasets with limited labeled samples.

Prerequisites

To run the code in this repository, install the following Python packages:
numpy: For numerical computations.
pandas: For data manipulation and loading CSV files
torch: For building and training diffusion models.
scikit-learn: For SVM classification, feature selection, and cross-validation.
joblib: For parallel processing in cross-validation.
Ensure you have a compatible GPU and CUDA setup if you intend to use GPU acceleration with PyTorch. The code automatically detects and utilizes GPU if available.

Reproduction of the optimal result.py: Reproduces the optimal classification results for Lactococcus lactis fermentation time using pre-tuned hyperparameters, a cosine noise schedule for diffusion models, and an SVM classifier.

FermentationTimeClassification.py: Implements a full pipeline for Lactococcus lactis fermentation time classification, optimizing diffusion model and SVM parameters using the Dung Beetle Optimizer with a sigmoid noise schedule.

DBO.py: Provides the Dung Beetle Optimizer algorithm, a bio-inspired optimization method used in FermentationTimeClassification.py to tune hyperparameters for diffusion models and SVM.
