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
            self.model_0 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model_1 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model_2 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model_3 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model_4 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model_5 = models.resnet50(weights=ResNet50_Weights.DEFAULT)

            self.embed_size = self.model_0.fc.in_features # 2048
            self.model_0.fc = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_1.fc = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_2.fc = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_3.fc = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_4.fc = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_5.fc = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)

        elif vision == 'densenet121':
            self.model_0 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.model_1 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.model_2 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.model_3 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.model_4 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.model_5 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

            self.embed_size = self.model_0.classifier.in_features # 1024
            self.model_0.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_1.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_2.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_3.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_4.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_5.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)

        elif vision == 'vit': 
            self.model_0 = ViTForImageClassification.from_pretrained('google/vit-large-patch32-384', image_size=384, patch_size=32, ignore_mismatched_sizes=True)
            self.model_1 = ViTForImageClassification.from_pretrained('google/vit-large-patch32-384', image_size=384, patch_size=32, ignore_mismatched_sizes=True)
            self.model_2 = ViTForImageClassification.from_pretrained('google/vit-large-patch32-384', image_size=384, patch_size=32, ignore_mismatched_sizes=True)
            self.model_3 = ViTForImageClassification.from_pretrained('google/vit-large-patch32-384', image_size=384, patch_size=32, ignore_mismatched_sizes=True)
            self.model_4 = ViTForImageClassification.from_pretrained('google/vit-large-patch32-384', image_size=384, patch_size=32, ignore_mismatched_sizes=True)
            self.model_5 = ViTForImageClassification.from_pretrained('google/vit-large-patch32-384', image_size=384, patch_size=32, ignore_mismatched_sizes=True)

            self.embed_size = self.model_0.classifier.in_features # 1024
            self.model_0.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_1.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_2.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_3.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_4.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_5.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
        else: 
            raise ValueError(f'Vision encoder type {vision} not supported.')

    def forward(self, x_0, x_1, x_2, x_3, x_4, x_5):
        if self.vision in ['resnet50', 'densenet121']:
            features_0 = self.model_0(x_0)
            features_1 = self.model_1(x_1)
            features_2 = self.model_2(x_2)
            features_3 = self.model_3(x_3)
            features_4 = self.model_4(x_4)
            features_5 = self.model_5(x_5)

        elif self.vision == 'vit':
            features_0 = self.model_0(x_0).logits
            features_1 = self.model_1(x_1).logits
            features_2 = self.model_2(x_2).logits
            features_3 = self.model_3(x_3).logits
            features_4 = self.model_4(x_4).logits
            features_5 = self.model_5(x_5).logits

        combined_features = torch.cat((features_0, features_1, features_2, features_3, features_4, features_5), dim=1)
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

