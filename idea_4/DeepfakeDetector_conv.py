import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from sklearn import metrics
from typing import Dict
import torchvision.transforms.functional as TF

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing to reduce overconfidence."""
    def __init__(self, epsilon=0.15):  # Slightly increased
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds, targets):
        num_classes = preds.size(1)
        log_preds = F.log_softmax(preds, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(preds)
            true_dist.fill_(self.epsilon / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=1))

def edge_extraction(image: torch.Tensor, sigma1=1.0, sigma2=2.0) -> torch.Tensor:
    """
    Combines Difference of Gaussians (DOG) + Sobel Edge Detection.
    Returns: (B,1,H,W) edge maps.
    """
    blurred1 = TF.gaussian_blur(image, kernel_size=7, sigma=sigma1)
    blurred2 = TF.gaussian_blur(image, kernel_size=7, sigma=sigma2)
    dog_edges = blurred1 - blurred2  # Difference of Gaussians (DOG)
    
    # Sobel Edge Detection
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_x, sobel_y = sobel_x.to(image.device), sobel_y.to(image.device)
    
    sobel_edges_x = F.conv2d(image, sobel_x, padding=1)
    sobel_edges_y = F.conv2d(image, sobel_y, padding=1)
    sobel_edges = torch.sqrt(sobel_edges_x**2 + sobel_edges_y**2)

    # Combine DOG and Sobel (weighted sum)
    return 0.6 * dog_edges + 0.4 * sobel_edges

class DeepfakeDetector(nn.Module):
    """
    Improved deepfake detector with:
      - **Backbone**: Partially frozen ResNet18 for global RGB features.
      - **Edge Branch**: Hybrid DOG + Sobel Edge CNN.
      - **Fusion**: RGB + Edge features MLP classifier.
      - **Stronger Regularization**: Dropout + LayerNorm.
    """

    def __init__(self, config):
        """
        Config keys:
          - 'model_name': e.g. 'resnet18'
          - 'pretrained': bool
          - 'num_classes': int
          - 'freeze_ratio': float portion of the backbone to freeze (0.9 default)
        """
        super().__init__()
        self.config = config
        model_name = config.get('model_name', 'resnet18')
        pretrained = config.get('pretrained', True)
        self.num_classes = config.get('num_classes', 2)
        self.freeze_ratio = config.get('freeze_ratio', 0.9)  # Increased freeze ratio

        # 1) **Backbone (RGB Feature Extractor)**
        self.backbone_rgb = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        self.feature_dim_rgb = getattr(self.backbone_rgb, 'num_features', 512)
        self._freeze_backbone_partially(self.backbone_rgb, self.freeze_ratio)

        # 2) **Edge Feature Extractor**
        self.edge_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),  # Increased Dropout

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)  # Increased Dropout
        )
        self.feature_dim_edge = 128

        # 3) **Fusion + Classifier**
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim_rgb + self.feature_dim_edge),  # Normalization before MLP
            nn.Linear(self.feature_dim_rgb + self.feature_dim_edge, 128),  # Smaller FC layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)
        )

        # **Loss Function**
        self.loss_func = LabelSmoothingCrossEntropy(epsilon=0.15)

    def _freeze_backbone_partially(self, backbone, ratio):
        """Freeze `ratio` proportion of backbone parameters."""
        all_params = list(backbone.parameters())
        freeze_count = int(len(all_params) * ratio)
        for p in all_params[:freeze_count]:
            p.requires_grad = False
        print(f"[INFO] Freed {freeze_count}/{len(all_params)} backbone params (ratio={ratio:.2f}).")

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        data_dict['image']: shape (B,3,H,W)
        - RGB � backbone � (B, feature_dim_rgb)
        - Edge Map � edge_cnn � (B, 128)
        - Fusion � classifier
        """
        images = data_dict['image']
        B, C, H, W = images.shape

        # (1) **RGB Features**
        rgb_feats = self.backbone_rgb(images)

        # (2) **Edge Features**
        gray = images.mean(dim=1, keepdim=True)  # Convert to grayscale
        edge_map = edge_extraction(gray)  # Hybrid DOG + Sobel
        edge_feats = self.edge_cnn(edge_map)

        # (3) **Feature Fusion**
        fused = torch.cat([rgb_feats, edge_feats], dim=1)
        logits = self.classifier_head(fused)
        return {'logits': logits}

    def get_losses(self, data_dict, pred_dict):
        """Compute classification loss."""
        labels = data_dict['label']
        logits = pred_dict['logits']
        return {'overall': self.loss_func(logits, labels)}

    def get_train_metrics(self, data_dict, pred_dict):
        """Compute accuracy, AUC, precision, recall, and F1-score."""
        labels = data_dict['label'].cpu().numpy()
        logits = pred_dict['logits'].detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)

        acc = np.mean(preds == labels)
        auc_val = metrics.roc_auc_score(labels, logits[:, 1]) if len(np.unique(labels)) > 1 else float('nan')
        prec = metrics.precision_score(labels, preds, zero_division=0)
        rec = metrics.recall_score(labels, preds, zero_division=0)
        f1 = metrics.f1_score(labels, preds, zero_division=0)

        return {'accuracy': acc, 'auc': auc_val, 'precision': prec, 'recall': rec, 'f1_score': f1}

    def get_test_metrics(self, data_dict, pred_dict):
        return self.get_train_metrics(data_dict, pred_dict)
