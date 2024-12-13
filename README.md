# Parkinson's MRI Classification Using Deep Learning

This project uses transfer learning techniques to classify patients with Parkinson's disease based on 2D slices of MRI images. By leveraging pre-trained convolutional neural networks (CNNs), the project aims to deliver a high-performance solution for early and accurate diagnosis of Parkinson's disease.

## Key Features
- MRI Preprocessing Pipeline: Ensures clean and standardized input for model training using FSL.
- Transfer Learning Approach: Fine-tunes pre-trained 2D CNN architectures (e.g., ResNet, EfficientNet, or VGG) for Parkinson’s MRI slice classification.
- Deep Learning Architecture: Custom or fine-tuned CNN models optimized for MRI data.
- Comprehensive Evaluation: Metrics including accuracy, sensitivity, specificity, and ROC-AUC.
- Reproducible Research: Fully documented code and configuration files for easy replication.

## Tools and Technologies
- Python (PyTorch)
- nibabel for MRI preprocessing
- Visualization libraries: Matplotlib, Seaborn
- DICOM/NIfTI handling
