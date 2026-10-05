# SmartEye — Project Structure

Scaffold reorganized from `SmartEye_original.ipynb`.

## Current project concept

The original notebook contains:
- Image classification with 3 classes: `benign`, `malignant`, `normal`
- Grayscale images resized to 224x224
- ResNet50 pretrained on ImageNet
- Grayscale -> RGB conversion before ResNet50
- Image-only and multimodal/hybrid model experiments
- Metadata features `tipo A` (rayos X) and `tipo B` (ultrasonido)
- Optional CLAHE preprocessing
- Fine-tuning experiments
- Evaluation/plots/confusion matrix
- Ant Lion Optimizer / MEALPY-related experimentation

## Intended flow

data -> preprocessing -> generators -> model -> training -> evaluation -> results

## Important

This is a scaffold, not a replacement of the original notebook.
`SmartEye_original.ipynb` is included at the project root for reference.

Paths and environment-specific Colab code have intentionally not been made authoritative yet.
