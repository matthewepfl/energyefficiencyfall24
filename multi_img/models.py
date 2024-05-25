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
        self.regression = nn.Linear(self.dim_input, 1)

    def forward(self, x):
        x = self.regression(x)
        # x = x.view(-1)
        return x
    
class DualVisionEncoder(nn.Module):
    '''
    Dual vision encoders with dual input (PA and lateral images).
    Uses one vision encoder for each image, then concatenates the features.

    Args:
        vision (str): Type of vision encoder (resnet50, densenet121 or vit)
    '''
    def __init__(self, vision : str):
        super().__init__()

        self.vision = vision
        # Load two pre-trained visual encoders
        if vision == 'resnet50':
            self.model_a = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model_b = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.embed_size = self.model_a.fc.in_features # 2048
            self.model_a.fc = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_b.fc = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)

        elif vision == 'densenet121':
            self.model_a = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.model_b = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.embed_size = self.model_a.classifier.in_features # 1024
            self.model_a.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_b.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)

        elif vision == 'vit': 
            self.model_a = ViTForImageClassification.from_pretrained(
                'google/vit-large-patch32-384', image_size=384, patch_size=32, ignore_mismatched_sizes=True)
            self.model_b = ViTForImageClassification.from_pretrained(
                'google/vit-large-patch32-384', image_size=384, patch_size=32, ignore_mismatched_sizes=True)
            self.embed_size = self.model_a.classifier.in_features # 1024
            self.model_a.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_b.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
        else: 
            raise ValueError(f'Vision encoder type {vision} not supported.')

    def forward(self, x_a, x_b):
        if self.vision in ['resnet50', 'densenet121']:
            features_a = self.model_a(x_a)
            features_b = self.model_b(x_b)
        elif self.vision == 'vit':
            features_a = self.model_a(x_a).logits
            features_b = self.model_b(x_b).logits
        combined_features = torch.cat((features_a, features_b), dim=1)
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
        print('Model initialization')
        num_params = 0
        if vision:
            print(f'\tVision encoder: {vision}')
            self.vision_encoder = DualVisionEncoder(vision)
            self.dim_input += IMAGE_EMBEDDING_DIM * 2
            num_params += sum(p.numel() for p in self.vision_encoder.parameters())
        
        self.regression = RegressionHead(self.dim_input)
        num_params += sum(p.numel() for p in self.regression.parameters())
        print('Total number of parameters:', num_params)

    def forward(self, x_a=None, x_b=None, labels=None):
        '''
        Args:
            x_a (tensor): PA image
            x_b (tensor): Lateral image
        '''
        # Generate embeddings (image and/or tabular)
        if self.vision:
            if x_a is None or x_b is None:
                raise ValueError('Vision encoder is specified but no images are provided.')
            vision_embedding = self.vision_encoder(x_a, x_b)

        # Embeddings
        embedding = vision_embedding
        demand = self.regression(embedding)

        # Return prediction, logits (and loss if labels are provided)
        outputs = {'prediction': demand}
        if labels is not None:
            loss_fct = nn.MSELoss()
            outputs['loss'] = loss_fct(demand, labels)
        return outputs

