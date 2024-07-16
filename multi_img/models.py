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
    def __init__(self, dim_input, hidden_dim=[512, 128], dropout_prob=0.0, batch_norm=False):
        super(RegressionHead, self).__init__()
        self.dim_input = dim_input
        self.hidden = []

        if isinstance(hidden_dim, str):
            hidden_dim = [int(x) for x in hidden_dim.split('-')]

        for dimension in hidden_dim:
            self.hidden.append(nn.Linear(self.dim_input, dimension))
            self.hidden.append(nn.ReLU())
            if dropout_prob > 0:
                self.hidden.append(nn.Dropout(p=dropout_prob))
            self.dim_input = dimension

        self.hidden.append(nn.Linear(self.dim_input, 1))
        self.hidden = nn.Sequential(*self.hidden)
        
        self._initialize_weights()

    def forward(self, x):
        return self.hidden(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class SixVisionEncoder(nn.Module):
    def __init__(self, vision: str, mask_branch=[]):
        super().__init__()

        self.vision = vision
        self.mask_branch = mask_branch

        if vision == 'resnet50':
            self.model_0 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model_1 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model_2 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model_3 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model_4 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model_5 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            self.embed_size = self.model_0.fc.in_features  # 2048
            self.model_0.fc = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_1.fc = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_2.fc = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_3.fc = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_4.fc = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_5.fc = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)

        elif vision == 'densenet121':
            self.model_0 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.model_1 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.model_2 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.model_3 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.model_4 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.model_5 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

            self.embed_size = self.model_0.classifier.in_features  # 1024
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

            self.embed_size = self.model_0.classifier.in_features  # 1024
            self.model_0.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_1.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_2.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_3.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_4.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_5.classifier = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)

        elif vision == 'efficientnet_b0':
            self.model_0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.model_1 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.model_2 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.model_3 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.model_4 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.model_5 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

            self.embed_size = self.model_0.classifier[1].in_features  # 1280
            self.model_0.classifier[1] = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_1.classifier[1] = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_2.classifier[1] = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_3.classifier[1] = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_4.classifier[1] = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)
            self.model_5.classifier[1] = nn.Linear(self.embed_size, IMAGE_EMBEDDING_DIM)

        else:
            raise ValueError(f'Vision encoder type {vision} not supported.')

        # Define additional layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.norm = nn.LayerNorm(IMAGE_EMBEDDING_DIM)
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(IMAGE_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM)

    def forward(self, x_0, x_1, x_2, x_3, x_4, x_5):
        def process_input(model, x):
            features = model(x)
            print("The shape of features is: ", features.shape)
            if self.vision == 'vit':
                features = features.logits
            features = self.avg_pool(features)
            print("The shape of features after avg_pool is: ", features.shape)
            features = self.flatten(features)
            print("The shape of features after flatten is: ", features.shape)
            features = self.norm(features)
            print("The shape of features after norm is: ", features.shape)
            features = self.dropout(features)
            print("The shape of features after dropout is: ", features.shape)
            features = self.dense(features)
            print("The shape of features after dense is: ", features.shape)
            return features
        
        print("The shape of x_0 is: ", x_0.shape)

        features_0 = process_input(self.model_0, x_0)
        features_1 = process_input(self.model_1, x_1)
        features_2 = process_input(self.model_2, x_2)
        features_3 = process_input(self.model_3, x_3)
        features_4 = process_input(self.model_4, x_4)
        features_5 = process_input(self.model_5, x_5)



        LIST_FEATURES = [features_0, features_1, features_2, features_3, features_4, features_5]
        if 0 in self.mask_branch:
            LIST_FEATURES.remove(features_0)
        if 1 in self.mask_branch:
            LIST_FEATURES.remove(features_1)
        if 2 in self.mask_branch:
            LIST_FEATURES.remove(features_2)
        if 3 in self.mask_branch:
            LIST_FEATURES.remove(features_3)
        if 4 in self.mask_branch:
            LIST_FEATURES.remove(features_4)
        if 5 in self.mask_branch:
            LIST_FEATURES.remove(features_5)

        combined_features = torch.cat(LIST_FEATURES, dim=1)
        print("The shape of combined_features is: ", combined_features.shape)
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
                 hidden_dims='512-256',
                 dropout_prob = 0.0, 
                 batch_norm = False,
                 mask_branch = [],
                 ):
        super(JointEncoder, self).__init__()

        self.vision = vision
        self.dim_input = 0
        self.mask_branch = mask_branch
        if not vision: 
            raise ValueError('Must specify the vision encoder.')
        
        if vision and vision not in ['resnet50', 'densenet121', 'vit', 'efficientnet_b0']:
            raise ValueError(f'Vision encoder type {vision} not supported.')

        num_params = 0
        if vision:
            print(f'\tVision encoder: {vision}')
            self.vision_encoder = SixVisionEncoder(vision, mask_branch)
            self.dim_input += IMAGE_EMBEDDING_DIM * (6 - len(self.mask_branch))
            num_params += sum(p.numel() for p in self.vision_encoder.parameters())
        
        self.regression = RegressionHead(self.dim_input, hidden_dim=hidden_dims, dropout_prob=dropout_prob, batch_norm=batch_norm)
        num_params += sum(p.numel() for p in self.regression.parameters())

    def forward(self, x_0=None, x_1=None, x_2=None, x_3=None, x_4=None, x_5=None, labels=None):
        vision_embedding = self.vision_encoder(x_0, x_1, x_2, x_3, x_4, x_5)
        efficiency = self.regression(vision_embedding)

        outputs = {'prediction': efficiency}
        if labels is not None:
            loss_fct = nn.MSELoss()
            outputs['loss'] = loss_fct(efficiency, labels)
        return outputs
