import os
import sys
import pickle
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from sklearn import metrics
import matplotlib.pyplot as plt

# Import the DeepfakeDetector
from DeepfakeDetector import DeepfakeDetector

# --------------------------- CONFIGURATION --------------------------------
config = {
    'model_name': 'efficientnet_b0',
    'pretrained': True,
    'num_classes': 2
}

# Directories for fake and real frames
fake_directory = sys.argv[1]
real_directory = sys.argv[2]

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

# --------------------------- DATASET HANDLING -----------------------------
class DeepfakeDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except UnidentifiedImageError:
            print(f"Skipping corrupted image: {img_path}")
            return self.__getitem__((idx + 1) % len(self.data))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.data))


def create_or_load_splits(fake_dir, real_dir, split_file="training_dataset_splits.pkl"):
    if os.path.exists(split_file):
        print(f"Loading dataset splits from {split_file}")
        with open(split_file, 'rb') as f:
            data_splits = pickle.load(f)
        return data_splits['train'], data_splits['val']
    else:
        print("Creating new dataset splits...")
        train_data = [(os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir) if f.startswith("train_fake_")]
        train_data += [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir) if f.startswith("train_real_")]
        val_data = [(os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir) if f.startswith("valid_fake_")]
        val_data += [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir) if f.startswith("valid_real_")]

        with open(split_file, 'wb') as f:
            pickle.dump({'train': train_data, 'val': val_data}, f)
        print(f"Saved dataset splits to {split_file}")
        return train_data, val_data

# --------------------------- TRAINING UTILITIES ---------------------------
def evaluate(model, dataloader, device):
    model.eval()
    total_loss, metrics_accum, total_batches = 0, {
        'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc': 0.0
    }, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            pred_dict = model({'image': images})
            logits = pred_dict['logits']
            loss = model.get_losses({'image': images, 'label': labels}, pred_dict)['overall']
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            y_true, y_pred = labels.cpu().numpy(), preds.cpu().numpy()
            y_proba = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]

            metrics_accum['accuracy'] += metrics.accuracy_score(y_true, y_pred)
            metrics_accum['precision'] += metrics.precision_score(y_true, y_pred, zero_division=0)
            metrics_accum['recall'] += metrics.recall_score(y_true, y_pred, zero_division=0)
            metrics_accum['f1_score'] += metrics.f1_score(y_true, y_pred, zero_division=0)
            if len(np.unique(y_true)) > 1:
                metrics_accum['auc'] += metrics.roc_auc_score(y_true, y_proba)
            total_batches += 1

    for key in metrics_accum:
        metrics_accum[key] /= total_batches
    return total_loss / total_batches, metrics_accum


def train(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=10, patience=5):
    best_val_loss, epochs_no_improve = float('inf'), 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            pred_dict = model({'image': images})
            loss = model.get_losses({'image': images, 'label': labels}, pred_dict)['overall']
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {running_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break


# --------------------------- MAIN EXECUTION ------------------------------
if __name__ == "__main__":
    train_data, val_data = create_or_load_splits(fake_directory, real_directory)
    train_dataset = DeepfakeDataset(train_data, transform=transform)
    val_dataset = DeepfakeDataset(val_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeDetector(config).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=10)
