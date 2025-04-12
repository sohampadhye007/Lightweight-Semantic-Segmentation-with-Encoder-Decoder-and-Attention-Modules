# Lightweight-Semantic-Segmentation-with-Encoder-Decoder-and-Attention-Modules
Objective- To train a segmentation model using a MobileNet pre-trained on the ImageNet dataset as an encoder and design a decoder that predicts segmented masks.

# Skin Lesion Segmentation Using Attention and Atrous Convolution

This repository contains an implementation of a semantic segmentation model for skin lesion detection using deep learning techniques. It leverages a pre-trained MobileNet encoder and explores multiple custom decoder architectures, including UNet-style, SegNet with attention gates, and SegNet with attention + atrous convolution.

## 📌 Introduction

The objective is to develop an effective semantic segmentation model for skin lesion identification using encoder-decoder architectures. The encoder is based on MobileNet pre-trained on ImageNet, while the decoder is custom-built and tested across three variants.

## 🧠 Methodology

Three architectures were explored:
- **UNet-style Decoder** with skip connections.
- **SegNet + Attention** for enhancing focus on key regions.
- **SegNet + Attention + Atrous Convolution** to capture multi-scale context.

Fine-tuning was also performed to improve performance using a small learning rate on the encoder.

## 📊 Dataset

- **Dataset**: [ISIC 2016 Skin Lesion Dataset](https://challenge.isic-archive.com/data/)
- 900 training images + masks
- 379 test images + masks
- Augmented dataset: 900 additional samples with transforms like random crop, flip, rotation, jitter, and Gaussian noise

## ⚙️ Setup & Training

- **Environment**: Local machine with NVIDIA GPU
- **Training Time**: ~4 hours for 100 epochs
- **Loss Function**: Binary Cross Entropy with Logits
- **Optimizations**: Fine-tuning, data augmentation, and architecture experiments

## 🚀 Performance

| Architecture                       | Mean IoU | Dice Score |
|------------------------------------|----------|------------|
| UNet                               | 0.77     | 0.858      |
| SegNet + Attention                 | 0.70     | 0.79       |
| SegNet + Attention + Atrous Conv   | 0.81     | 0.88       |

Fine-tuned models showed improved segmentation accuracy, capturing lesion boundaries more precisely and reducing false positives.

## 📁 Code Links
  - `sementation_code.py`

## 📈 Results & Observations

- Loss decreased consistently for both training and validation.
- Fine-tuned models outperformed pre-trained-only models in both IoU and Dice metrics.
- Atrous convolutions enabled better multi-scale feature extraction but increased model size.

## 🔧 Future Improvements

- Experiment with more lightweight backbones for edge deployment.
- Explore contrastive learning for feature extraction.
- Test on more diverse dermatological datasets.

## 📚 References

- MobileNetV2: Inverted Residuals and Linear Bottlenecks
- SegNet: A Deep Convolutional Encoder-Decoder Architecture
- DeepLab: Encoder-Decoder with Atrous Convolutions
- Binary Cross Entropy with Logits Loss

