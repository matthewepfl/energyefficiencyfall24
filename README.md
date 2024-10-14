
# Energy Efficiency Prediction

This repository contains a multi-modal model that predicts energy efficiency from property listings using a combination of **images**, **text**, and **tabular data**. The architecture is designed to combine data from multiple modalities using specialized encoders for each input type. The goal is to use information from property images, descriptive text, and numerical/tabular data to improve the accuracy of the energy efficiency prediction.

![Model Architecture](https://private-user-images.githubusercontent.com/151882909/376159541-efb25917-3589-4289-8ad1-3d29a9095ea6.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mjg4OTQ2ODYsIm5iZiI6MTcyODg5NDM4NiwicGF0aCI6Ii8xNTE4ODI5MDkvMzc2MTU5NTQxLWVmYjI1OTE3LTM1ODktNDI4OS04YWQxLTNkMjlhOTA5NWVhNi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDE0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAxNFQwODI2MjZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yZDc3NTUwMDYyZTQ0Mzk5ZDU2MTA4NDRmNjkyMDc3MTE4MzFhOWNkOTVjMzQ4MWU5YzFhNGM0NGU1NGQzZmVhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.74B4HZKp9953qwfsDRIqJ6O3UawRmPs1VVyrzFaWynM)

## Model Overview

The architecture is composed of the following components:

### 1. **Tabular Data**
- **Models**: [TabNet](https://arxiv.org/abs/1908.07442), [LightGBM](https://lightgbm.readthedocs.io/)
- The tabular data (numerical features such as area, number of rooms, etc.) is processed using TabNet and LightGBM to create embeddings that represent this data in the multi-modal model.

### 2. **Text Data**
- **Models**: [BERT](https://huggingface.co/transformers/model_doc/bert.html), [KeyBERT](https://github.com/MaartenGr/KeyBERT)
- The descriptive text (such as property descriptions) is processed using a textual encoder, leveraging BERT for contextual embeddings. KeyBERT is used for keyword extraction, which helps to emphasize the most relevant features of the text.

### 3. **Image Data**
- **Models**: [ResNet](https://arxiv.org/abs/1512.03385), [DenseNet](https://arxiv.org/abs/1608.06993), [ViT](https://arxiv.org/abs/2010.11929)
- Property images (such as kitchen, bathroom, balconies, etc.) are processed using a visual encoder. The backbone of the visual encoder is based on ResNet, DenseNet, or Vision Transformer (ViT). These encoders transform image data into embeddings for the multi-modal model.

### 4. **Fusion Layer**
- The embeddings from the tabular, text, and image data are concatenated into a single representation. This is then passed through a regressor to predict the energy efficiency score.

### 5. **Backpropagation**
- The model is trained end-to-end using backpropagation, ensuring that all encoders are fine-tuned together to optimize prediction accuracy.

## Requirements

To install the necessary dependencies, execute the following command:

```bash
pip install -r requirements.txt
```

### Main Dependencies:
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [TabNet](https://github.com/dreamquark-ai/tabnet)

## Data Structure

The input data consists of three main types:

- **Tabular**: Numerical features like the area, number of rooms, heating type, etc.
- **Text**: Descriptions of the property such as location, nearby amenities, and other key features.
- **Images**: Photos of different sections of the property (kitchen, bathroom, balconies, living areas, etc.).

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/EnergyEfficiencyPrediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd EnergyEfficiencyPrediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Directory Structure

```
├── classification/
├── ensemble/
├── model/
├── multi_img/
├── tabular/
├── text/
├── requirements.txt
├── README.md
```
