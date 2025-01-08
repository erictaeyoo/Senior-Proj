import logging
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

# For modern SOTA backbones, we can use the `timm` library 
import timm

import cv2
import matplotlib.pyplot as plt

import timm
import torch
import torch.nn as nn


class DeepfakeDetector(nn.Module):
    """
    A deepfake detection model leveraging EfficientNet as the backbone.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_name = config.get('model_name', 'efficientnet_b0')  # Default to EfficientNet-B0
        pretrained = config.get('pretrained', True)
        num_classes = config.get('num_classes', 2)

        # Initialize EfficientNet backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )

        # Extract the feature dimension dynamically
        self.feature_dim = getattr(self.backbone, 'num_features', 1280)  # EfficientNet-B0 outputs 1280-dim features

        # Add a classification head
        self.classifier_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # Loss function
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, data_dict):
        images = data_dict['image']
        features = self.backbone(images)

        logits = self.classifier_head(features)
        return {'logits': logits}

    def get_losses(self, data_dict, pred_dict):
        labels = data_dict['label']
        logits = pred_dict['logits']
        overall_loss = self.loss_func(logits, labels)
        return {'overall': overall_loss}
    
    def get_train_metrics(self, data_dict: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # Convert the ground truth labels and predicted logits to numpy arrays
        labels = data_dict['label'].cpu().numpy()
        logits = pred_dict['logits'].detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)  # Get the predicted class indices

        # Accuracy
        acc = np.mean(preds == labels)

        # AUC (Area Under the ROC Curve)
        try:
            auc = metrics.roc_auc_score(labels, logits[:, 1])
        except ValueError:
            auc = float('nan')  # In case AUC cannot be computed

        # Precision, Recall, F1-score
        precision = metrics.precision_score(labels, preds, zero_division=0)
        recall = metrics.recall_score(labels, preds, zero_division=0)
        f1 = metrics.f1_score(labels, preds, zero_division=0)

        return {
            'accuracy': acc,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def get_test_metrics(self, data_dict: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # The test metrics are the same as the training metrics
        return self.get_train_metrics(data_dict, pred_dict)