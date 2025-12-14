# Comparative Analysis of CNN and Transformer-Based Models for Bird Species Classification

## Overview

This project presents a comparison between **pre-trained transfer learning models** and **custom architectures trained from scratch** for bird species classification across 200 different species. The study evaluates the effectiveness of Vision Transformers (ViT), EfficientNet, BEiT, and Swin against custom CNN and ResNet implementations, with a focus on understanding the trade-offs between model complexity, training efficiency, and classification accuracy.

## Key Results

| Approach | Best Model | Accuracy | vs Random |
|----------|-----------|----------|-----------|
| **Pre-trained** | BEIT (20 epochs) | **89.23%** | **178x** |
| **From Scratch** | ResNet18 Skinny | **31.40%** | **63x** |

With 200 bird species classes, the random baseline is 0.5%, making these results highly significant.

## Project Structure

```
├── BEiT_pretrained.ipynb              # BEiT fine-tuning
├── ResNet_skinny.ipynb                # ResNet18 Skinny training
├── experiments/
│   ├── architectures_scratch.py       # Custom CNN & ResNet implementations
│   └── transfer_learning.py           # Other transfer learning models
├── multimodal_approach/
│   ├── embedding-space.ipynb          # Embedding space training
│   ├── multimodal-skinny18.ipynb      # Multimodal ResNet18 Skinny
│   └── embeddings/
│       ├── test_embeddings.npy
│       ├── train_embeddings.npy
│       └── val_embeddings.npy
├── submissions/
│   ├── submission_beit.csv            # BEiT predictions
│   ├── submission_multimodal.csv      # Multimodal predictions
│   └── submission_skinny.csv          # ResNet18 Skinny predictions
└── weights/
    ├── attribute_net.pth              # Embedding space net weights
    ├── multimodal_resnet18_skinny.pth
    ├── multimodal_resnet18_skinny_best.pth
    └── resnet18Skinny.pth
```

## Experimental Approaches

### Pre-trained Models (Transfer Learning)
Fine-tuning of models pre-trained on ImageNet with varying unfrozen layers:

- **ViT**: 68.00%
- **EfficientNet**: 63.25%
- **Swin**: 85.17%
- **BEiT (classifier head only)**:
  - 10 epochs: 88.62%
  - 20 epochs: 89.23% 
- **BEiT (two encoder layers unfrozen)**:
  - 10 epochs: 87.30%
  - 20 epochs: 86.15%

### From Scratch Models
Custom architectures trained from initialization:

- **Basic CNN**: 8.12%
- **ResNet8 Multimodal**: 10.12%
- **ResNet34**: 15.62%
- **ResNet18 Multimodal**: 22.06%
- **ResNet18 Enhanced**: 24.28%
- **ResNet18 Skinny**: 31.40%

### Multimodal Approach
Integration of bird attribute features (plumage, size, habitat) alongside image features to improve classification, particularly beneficial for under-represented species.

## Technologies & Dependencies

- **Deep Learning**: PyTorch
- **Vision Models**: HuggingFace transformers
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Computer Vision**: OpenCV, PIL
- **Training**: CUDA, Kaggle environment, GPU P100

## Key Insights

1. **Transfer Learning Dominance**: Pre-trained models significantly outperform from-scratch architectures, with BEiT achieving nearly 3x the accuracy of the best from-scratch model.

2. **Frozen vs Unfrozen Layers**: Fine-tuning with only the classifier head unfrozen yields better results than unfrozen encoder layers, suggesting the pre-trained features are highly effective.

3. **Model Efficiency**: ResNet18 Skinny provides the best accuracy-to-complexity ratio among from-scratch models, making it a practical model.

4. **Multimodal Learning**: Incorporating bird attributes would improve generalization on challenging species with limited training samples.

5. **Transformer Advantage**: Vision Transformers (BEiT, Swin, ViT) outperform CNNs when pre-trained, indicating better feature learning for this classification task.

## Files Description

- **BEiT_pretrained.ipynb**: Complete pipeline for fine-tuning BEiT with hyperparameter exploration
- **ResNet_skinny.ipynb**: Custom ResNet18 Skinny architecture design and training
- **transfer_learning.py**: Utilities for loading pre-trained models
- **architectures_scratch.py**: Custom CNN and ResNet implementations
- **embedding-space.ipynb**: Training of embedded representations via CNNs

## Authors

- Esteban Gatein
- Asal Mehrabi
- Tejaswi Madduri

---

**Last Updated**: December 2025