# Scratch or Finetune on Small Datasets 🌸

Welcome to the **Scratch or Finetune on Small Datasets** repository! This project presents a comparative study of the benefits of transfer learning versus building a custom CNN architecture for very small datasets. We focus on the Oxford Flower Dataset to demonstrate the effectiveness of different approaches in classification tasks.

[![Download Releases](https://img.shields.io/badge/Download%20Releases-Click%20Here-blue)](https://github.com/nateattoh/scratch-or-finetune-on-small-datasets/releases)

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

In the realm of computer vision, choosing the right approach for training models on small datasets is crucial. This project explores two main strategies: building a custom Convolutional Neural Network (CNN) from scratch and utilizing transfer learning with pre-trained models. We aim to uncover which method yields better performance in classifying images of flowers.

## Project Overview

The repository contains code, models, and results from our experiments. We employ PyTorch for our implementations, making it accessible and easy to modify. The focus is on the Oxford Flower Dataset, which consists of 102 flower categories, each with 40 to 258 images.

### Topics Covered

- Classification
- CNN
- CNN Classification
- Computer Vision
- Transfer Learning
- Custom Models
- Oxford Flower Dataset

## Dataset

The Oxford Flower Dataset is a popular benchmark for image classification tasks. It includes:

- 102 flower categories
- 8,189 images in total
- Varied image sizes and backgrounds

This dataset allows us to evaluate the performance of different models effectively. For more details, visit the official [Oxford Flower Dataset page](http://www.robots.ox.ac.uk/~vgg/data/flowers/).

## Methodology

### Data Preprocessing

We perform several preprocessing steps to prepare the data for training:

1. **Resizing**: Images are resized to a consistent size (e.g., 224x224 pixels).
2. **Normalization**: Pixel values are normalized to a range suitable for model training.
3. **Augmentation**: We apply techniques like rotation, flipping, and color jitter to increase dataset diversity.

### Model Architectures

We compare two main approaches:

1. **Custom CNN**: A model built from scratch, specifically designed for the dataset.
2. **Transfer Learning**: Utilizing pre-trained models such as ResNet, VGG, and MobileNet, fine-tuned on our dataset.

### Training and Evaluation

We split the dataset into training, validation, and test sets. We then train both models and evaluate their performance based on accuracy and loss metrics.

## Results

The results of our experiments reveal insights into the effectiveness of each approach. 

### Performance Metrics

- **Custom CNN**: 
  - Accuracy: 75%
  - Loss: 0.45

- **Transfer Learning (ResNet)**:
  - Accuracy: 85%
  - Loss: 0.30

The transfer learning approach outperformed the custom CNN, demonstrating its advantages in scenarios with limited data.

## Installation

To get started, clone the repository:

```bash
git clone https://github.com/nateattoh/scratch-or-finetune-on-small-datasets.git
cd scratch-or-finetune-on-small-datasets
```

### Dependencies

Ensure you have the following installed:

- Python 3.6 or higher
- PyTorch
- torchvision
- matplotlib
- numpy

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

After installation, you can run the training scripts. 

### Training the Custom CNN

To train the custom CNN model, use:

```bash
python train_custom_cnn.py
```

### Training with Transfer Learning

To train using transfer learning, execute:

```bash
python train_transfer_learning.py
```

### Evaluation

To evaluate the models, run:

```bash
python evaluate.py
```

The evaluation script will provide metrics such as accuracy and loss for both models.

## Contributing

We welcome contributions! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

### Steps to Contribute

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Make your changes and commit them.
4. Push your branch and create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please reach out to:

- **Nate Attoh**: [nateattoh@example.com](mailto:nateattoh@example.com)

Explore the project further and check out the latest updates in the [Releases section](https://github.com/nateattoh/scratch-or-finetune-on-small-datasets/releases).

Thank you for visiting the Scratch or Finetune on Small Datasets repository! We hope you find this project insightful and useful for your own research and projects.