'''
Models for Multi-Modal Diagnosis from Radiology Images and Tabular metadata.
Vision encoders: ResNet50, DenseNet121, ViT
Tabular encoder: Fully-connected network
Joint encoder: Vision + Tabular encoders
'''

from data import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import DenseNet121_Weights, ResNet50_Weights
from transformers import ViTForImageClassification

IMAGE_EMBEDDING_DIM = 512   # Vision encoders produce 512-dimensional embeddings

class RegressionHead(nn.Module):
    '''
    Regression Head for multi-class multi-label prediction from embedding.
    Single layer -- with batch normalization, dropout and sigmoid activation.

    Args:
        dim_input (int): Input dimension
        num_classes (int): Number of classes
        num_labels (int): Number of labels for each class
    '''
    def __init__(self, dim_input):
        super(RegressionHead, self).__init__()
        self.dim_input = dim_input
        self.hidden = nn.Linear(self.dim_input, 1024)
        self.hidden2 = nn.Linear(1024, 512)
        self.regression = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = self.regression(x)

        return x
    
class DualVisionEncoder(nn.Module):
    '''
    Dual vision encoders with six inputs (PA and lateral images, for example).
    Uses one vision encoder for each image, then concatenates the features.

    Args:
        vision (str): Type of vision encoder (resnet50, densenet121, or vit)
    '''

    def __init__(self, vision: str):
        super().__init__()
        self.vision = vision
        self.num_models = 6
        self.models = []

        # Load the appropriate model based on the vision type
        if vision == 'resnet50':
            for _ in range(self.num_models):
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                embed_size = model.fc.in_features
                model.fc = nn.Linear(embed_size, IMAGE_EMBEDDING_DIM)
                self.models.append(model)

        elif vision == 'densenet121':
            for _ in range(self.num_models):
                model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
                embed_size = model.classifier.in_features
                model.classifier = nn.Linear(embed_size, IMAGE_EMBEDDING_DIM)
                self.models.append(model)

        elif vision == 'vit':
            for _ in range(self.num_models):
                model = ViTForImageClassification.from_pretrained(
                    'google/vit-large-patch32-384',
                    image_size=384,
                    patch_size=32,
                    ignore_mismatched_sizes=True
                )
                embed_size = model.classifier.in_features
                model.classifier = nn.Linear(embed_size, IMAGE_EMBEDDING_DIM)
                self.models.append(model)
        else:
            raise ValueError(f'Vision encoder type {vision} not supported.')

    def forward(self, *inputs):
        if len(inputs) != self.num_models:
            raise ValueError(f"Expected {self.num_models} inputs, but got {len(inputs)}.")

        features = []
        for model, input in zip(self.models, inputs):
            if self.vision == 'vit':
                output = model(input).logits
            else:
                output = model(input)
            features.append(output)

        combined_features = torch.cat(features, dim=1)
        return combined_features

class JointEncoder(nn.Module):
    '''
    Joint Encoder: Encodes image data separately, 
    concatenates embeddings, and passes through fully connected classifier network.

    Args:
        vision (str): Type of vision encoder 'densenet121', 'resnet50' or 'vit'. Default: None --> No vision encoder
    '''
    def __init__(self, 
                 vision = None,
                 ):
        super(JointEncoder, self).__init__()

        self.vision = vision
        self.dim_input = 0
        if not vision: 
            raise ValueError('Must specify the vision encoder.')
        
        if vision and vision not in ['resnet50', 'densenet121', 'vit']:
            raise ValueError(f'Vision encoder type {vision} not supported.')

        num_params = 0
        if vision:
            print(f'\tVision encoder: {vision}')
            self.vision_encoder = DualVisionEncoder(vision)
            self.dim_input += IMAGE_EMBEDDING_DIM * 6
            num_params += sum(p.numel() for p in self.vision_encoder.parameters())
        
        self.regression = RegressionHead(self.dim_input)
        num_params += sum(p.numel() for p in self.regression.parameters())
        print('Total number of parameters:', num_params)

    def forward(self, x_0=None, x_1=None, x_2=None, x_3=None, x_4=None, x_5=None, labels=None):
        # Generate embeddings (image and/or tabular)
        if self.vision:
            if x_0 is None or x_1 is None or x_2 is None or x_3 is None or x_4 is None or x_5 is None:
                raise ValueError('Vision encoder is specified but no images are provided.')
            vision_embedding = self.vision_encoder(x_0, x_1, x_2, x_3, x_4, x_5)

        # Embeddings
        embedding = vision_embedding
        efficiency = self.regression(embedding)

        # Return prediction, logits (and loss if labels are provided)
        outputs = {'prediction': efficiency}
        if labels is not None:
            loss_fct = nn.MSELoss()
            outputs['loss'] = loss_fct(efficiency, labels)
        return outputs

