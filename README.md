# Applying VanillaNet to the Ships in Satellite Imagery Dataset

![image](https://github.com/user-attachments/assets/e135666b-fbc6-43df-b0fd-0e9dc21a89a9)


## Introduction
This document outlines the application of **VanillaNet** to the [Ships in Satellite Imagery](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery) dataset available on Kaggle. The implementation is based on the **VanillaNet** model architecture from Huawei Noah’s research team, as described in their [official repository](https://github.com/huawei-noah/VanillaNet?tab=readme-ov-file) and the corresponding research paper: [VanillaNet: the Power of Minimalism in Deep Learning](https://arxiv.org/abs/2305.12972).

## Dataset Overview
The **Ships in Satellite Imagery** dataset contains satellite images labeled for ship detection. The goal is to develop a **deep learning model** capable of classifying images as containing ships or not.

### Dataset Structure:
- **train**: Images for training the model.
- **test**: Images for evaluation.
- **annotations.csv**: Contains labels for the images.

## Model Selection: VanillaNet
VanillaNet is a lightweight convolutional neural network designed for efficiency without relying on complex components such as self-attention or depth-wise convolutions. It provides:
- **High efficiency** with minimal computational overhead.
- **Competitive accuracy** compared to more complex CNN architectures.
- **Simplicity** in design, making it easy to adapt and deploy.

## Implementation Details
1. **Pretrained Model**: We use a **pretrained VanillaNet model** from Huawei’s official implementation.
2. **Model Location**: The model implementation is found in `models/vanillanet.py`.
3. **Training Strategy**:
   - **Data Augmentation**: Techniques such as rotation, flipping, and normalization are applied to improve generalization.
   - **Loss Function**: Cross-entropy loss is used for classification.
   - **Optimizer**: Adam optimizer with a learning rate scheduler.
   - **Evaluation Metrics**: Accuracy and F1-score are used to assess performance.

## Results and Future Work
After training the VanillaNet model on the dataset, we evaluate its performance using accuracy and F1-score. Further improvements can include:
- **Fine-tuning hyperparameters** such as learning rate and batch size.
- **Experimenting with different VanillaNet variants** to optimize for ship detection.
- **Applying post-processing techniques** to refine predictions.

## Conclusion
VanillaNet demonstrates a strong balance between **efficiency and performance** for ship detection in satellite imagery. The lightweight nature of the model makes it suitable for real-time applications in remote sensing and maritime surveillance.

---

For more details, refer to the [VanillaNet repository](https://github.com/huawei-noah/VanillaNet?tab=readme-ov-file) and the original [research paper](https://arxiv.org/abs/2305.12972).
