"""
Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. extract_features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Notes:
This model combines CNN-based feature extraction with wavelet transforms and Sobel filters to enhance deepfake detection.
"""

import logging
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

from torchvision import models
import pywt
import cv2
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class DeepfakeDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.classifier_head = self.build_classifier_head()

    def build_backbone(self, config):
        # Prepare the backbone
        backbone_name = config.get('backbone_name', 'resnet18')
        pretrained = config.get('pretrained', True)
        input_channels = config.get('input_channels', 3)
        
        # Load a pre-trained model
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Modify the first convolutional layer if input channels are different
        if input_channels != 3:
            original_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias
            )
            # Initialize the new conv1 layer weights
            nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        # Remove the last fully connected layer
        backbone.fc = nn.Identity()
        logger.info(f"Backbone {backbone_name} built with input channels {input_channels}")
        return backbone


    def build_loss(self, config):
        # Prepare the loss function
        loss_name = config.get('loss_func', 'CrossEntropyLoss')
        if loss_name == 'CrossEntropyLoss':
            loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
        logger.info(f"Loss function {loss_name} initialized")
        return loss_func

    def build_classifier_head(self):
        # Build the classifier head
        num_features = self.backbone.fc.in_features if hasattr(self.backbone.fc, 'in_features') else 512
        classifier_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # binary output classes: deepfake or not deepfake
        )
        logger.info("Classifier head built")
        return classifier_head

    def extract_features(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        images = data_dict['image']  # Shape: (batch_size, channels, height, width)

        # Apply wavelet transform
        wavelet_features = self.apply_wavelet_transform(images)

        # Apply Sobel filters
        sobel_features = self.apply_sobel_filters(images)

        # Resize wavelet and Sobel features to match the original image size (224x224)
        wavelet_features = F.interpolate(wavelet_features, size=(224, 224), mode='bilinear', align_corners=False)
        sobel_features = F.interpolate(sobel_features, size=(224, 224), mode='bilinear', align_corners=False)

        # Concatenate features along the channel dimension
        combined_features = torch.cat((images, wavelet_features, sobel_features), dim=1)

        # Extract features using the backbone
        features = self.backbone(combined_features)
        return features

    def apply_wavelet_transform(self, images: torch.Tensor) -> torch.Tensor:
        # Apply wavelet transform to each image in the batch
        wavelet_features = []
        for img in images:
            # Convert tensor to numpy array
            img_np = img.cpu().numpy()
            # Apply wavelet transform to each channel
            coeffs = [pywt.dwt2(img_np[c], 'db1') for c in range(img_np.shape[0])]
            # Concatenate approximation and details coefficients
            cA, (cH, cV, cD) = zip(*coeffs)
            # Stack the coefficients back into a tensor
            features = np.stack(cA + cH + cV + cD, axis=0)
            wavelet_features.append(torch.tensor(features))
        wavelet_features = torch.stack(wavelet_features).to(images.device)
        return wavelet_features

    def apply_sobel_filters(self, images: torch.Tensor) -> torch.Tensor:
        # Apply Sobel filters to each image in the batch
        sobel_features = []
        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32, device=images.device).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                       [ 0,  0,  0],
                                       [ 1,  2,  1]], dtype=torch.float32, device=images.device).unsqueeze(0).unsqueeze(0)
        for img in images:
            # Convert to grayscale
            img_gray = img.mean(dim=0, keepdim=True).unsqueeze(0)  # Shape: (1, 1, H, W)
            # Apply Sobel filters
            grad_x = F.conv2d(img_gray, sobel_kernel_x, padding=1)
            grad_y = F.conv2d(img_gray, sobel_kernel_y, padding=1)
            # Compute gradient magnitude
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            sobel_features.append(grad_mag.squeeze(0))
        sobel_features = torch.stack(sobel_features)
        return sobel_features

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        # Classification
        logits = self.classifier_head(features)
        return logits

    def get_losses(self, data_dict: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Compute the loss
        labels = data_dict['label']
        logits = pred_dict['logits']
        loss = self.loss_func(logits, labels)
        return {'overall': loss}

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
            auc = float('nan')  # In case AUC cannot be computed due to missing positive/negative samples

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
        # The test metrics are the same as the training metrics, just applied to the test data
        return self.get_train_metrics(data_dict, pred_dict)

    def forward(self, data_dict: Dict[str, torch.Tensor], inference=False) -> Dict[str, torch.Tensor]:
        # Forward propagation
        features = self.extract_features(data_dict)
        logits = self.classifier(features)
        pred_dict = {'logits': logits}
        return pred_dict

    def show_image(title, image):
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    # Helper function to convert tensor to a displayable image
    def tensor_to_image(tensor):
        # Move tensor to CPU, convert to numpy, and scale to range [0, 255]
        img_np = tensor.cpu().numpy()
        img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np)) * 255.0
        return img_np.astype(np.uint8)

    # Load a deepfake image (example: replace with actual image loading logic)
    def load_image(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = img.transpose(2, 0, 1)  # Convert to (channels, height, width)
        img = torch.tensor(img, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        return img.unsqueeze(0)  # Add batch dimension (1, channels, height, width)