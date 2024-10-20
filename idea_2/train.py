import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from DeepfakeDetector import DeepfakeDetector
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image, UnidentifiedImageError

# Define the configuration
config = {
    'backbone_name': 'resnet18',
    'pretrained': True,
    'input_channels': 16,  # Original 3 channels + wavelet features + Sobel features
    'loss_func': 'CrossEntropyLoss',
}

# Initialize the detector
model = DeepfakeDetector(config)

# Define the directories for fake and real frames
fake_directory = sys.argv[1]
real_directory = sys.argv[2]

# Define a separate directory for testing data
test_directory = sys.argv[3]

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
])

# Dataset for training and validation
class DeepfakeDataset(torch.utils.data.Dataset):
    def __init__(self, fake_dir, real_dir, subset='train', transform=None):
        self.fake_dir = fake_dir
        self.real_dir = real_dir
        self.transform = transform
        self.subset = subset  # 'train' or 'val'

        # Load filenames based on subset (train or validation)
        if self.subset == 'train':
            self.fake_images = [(os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir) if f.startswith("train_fake_") and f.endswith(".png")]
            self.real_images = [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir) if f.startswith("train_real_") and f.endswith(".png")]
        elif self.subset == 'val':
            self.fake_images = [(os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir) if f.startswith("valid_fake_") and f.endswith(".png")]
            self.real_images = [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir) if f.startswith("valid_real_") and f.endswith(".png")]

        # Combine both lists
        self.data = self.fake_images + self.real_images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            print(f"Skipping corrupted image: {img_path}")
            return self.__getitem__((idx + 1) % len(self.data))  # Skip to the next image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.data))  # Skip to the next image

        if self.transform:
            img = self.transform(img)
        return img, label

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform

        # Load both deepfake and real frames and assign labels (1 for deepfake, 0 for real)
        self.deepfake_images = [(os.path.join(test_dir, f), 1) for f in os.listdir(test_dir) if "deepfake" in f and f.endswith(".jpg")]
        self.real_images = [(os.path.join(test_dir, f), 0) for f in os.listdir(test_dir) if "real" in f and f.endswith(".jpg")]

        # Combine both lists
        self.test_images = self.deepfake_images + self.real_images

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        img_path, label = self.test_images[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# Create the dataset for training
train_dataset = DeepfakeDataset(fake_directory, real_directory, subset='train', transform=transform)

# Create the dataset for validation
val_dataset = DeepfakeDataset(fake_directory, real_directory, subset='val', transform=transform)

# Create the dataset for testing (different deepfake frames from a new video)
test_dataset = TestDataset(test_directory, transform=transform)

# Create DataLoader for training, validation, and testing sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define an optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Helper function to save the image instead of showing
def save_image(title, image, batch_idx, save_dir='output_images'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    file_name = os.path.join(save_dir, f"{title}_batch_{batch_idx}.png")
    plt.savefig(file_name)
    plt.close()

# Helper function to convert tensor to a displayable image
def tensor_to_image(tensor):
    # Move tensor to CPU, convert to numpy, and scale to range [0, 255]
    img_np = tensor.cpu().numpy()
    img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np)) * 255.0
    return img_np.astype(np.uint8)

# Function to evaluate on validation set with additional metrics
def validate(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0

    # Metrics for precision, recall, f1, and accuracy
    metrics_accum = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'auc': 0.0
    }
    total_batches = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Prepare the data dict
            data_dict = {'image': images, 'label': labels}

            # Forward pass
            pred_dict = model(data_dict)

            # Compute loss
            losses = model.get_losses(data_dict, pred_dict)
            loss = losses['overall']
            running_loss += loss.item()

            # Compute metrics
            metrics_dict = model.get_train_metrics(data_dict, pred_dict)
            for key in metrics_accum:
                metrics_accum[key] += metrics_dict[key]
            total_batches += 1

    # Average the metrics over all batches
    val_loss = running_loss / len(dataloader)
    for key in metrics_accum:
        metrics_accum[key] /= total_batches

    # Print validation metrics
    print(f'Validation Loss: {val_loss:.4f}, '
          f'Accuracy: {metrics_accum["accuracy"]:.4f}, '
          f'Precision: {metrics_accum["precision"]:.4f}, '
          f'Recall: {metrics_accum["recall"]:.4f}, '
          f'F1-Score: {metrics_accum["f1_score"]:.4f}, '
          f'AUC: {metrics_accum["auc"]:.4f}')

    return val_loss, metrics_accum  # Return the validation loss and metrics


# Training loop with additional metrics
def train(model, train_loader, val_loader, optimizer, num_epochs=50, patience=5):
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Metrics for precision, recall, f1, and accuracy
        metrics_accum = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'auc': 0.0
        }
        total_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Prepare the data dict
            data_dict = {'image': images, 'label': labels}

            # Forward pass
            pred_dict = model(data_dict)

            # Compute loss
            losses = model.get_losses(data_dict, pred_dict)
            loss = losses['overall']

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate running loss and compute metrics
            running_loss += loss.item()
            metrics_dict = model.get_train_metrics(data_dict, pred_dict)
            for key in metrics_accum:
                metrics_accum[key] += metrics_dict[key]
            total_batches += 1

        # Average the metrics over all batches
        epoch_loss = running_loss / len(train_loader)
        for key in metrics_accum:
            metrics_accum[key] /= total_batches

        print(f'Epoch [{epoch + 1}/{num_epochs}] finished. '
              f'Training Loss: {epoch_loss:.4f}, '
              f'Accuracy: {metrics_accum["accuracy"]:.4f}, '
              f'Precision: {metrics_accum["precision"]:.4f}, '
              f'Recall: {metrics_accum["recall"]:.4f}, '
              f'F1-Score: {metrics_accum["f1_score"]:.4f}, '
              f'AUC: {metrics_accum["auc"]:.4f}')

        # Validate the model on the validation dataset
        val_loss, val_metrics = validate(model, val_loader)

        # Early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # Save the model checkpoint when validation loss improves
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Validation loss decreased. Saving model from epoch {epoch + 1}')
        else:
            epochs_no_improve += 1
            print(f'No improvement in validation loss for {epochs_no_improve} epochs.')

            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break  # Exit the training loop

    print(f'Training complete. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}')


# Testing function with additional metrics
def test(model, dataloader):
    model.eval()
    running_loss = 0.0

    # Metrics for precision, recall, f1, and accuracy
    metrics_accum = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'auc': 0.0
    }
    total_batches = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Prepare the data dict
            data_dict = {'image': images, 'label': labels}

            # Forward pass
            pred_dict = model(data_dict)

            # Compute loss
            losses = model.get_losses(data_dict, pred_dict)
            loss = losses['overall']
            running_loss += loss.item()

            # Compute metrics
            metrics_dict = model.get_test_metrics(data_dict, pred_dict)
            for key in metrics_accum:
                metrics_accum[key] += metrics_dict[key]
            total_batches += 1

    # Average the metrics over all batches
    test_loss = running_loss / len(dataloader)
    for key in metrics_accum:
        metrics_accum[key] /= total_batches

    print(f'Test Loss: {test_loss:.4f}, '
          f'Accuracy: {metrics_accum["accuracy"]:.4f}, '
          f'Precision: {metrics_accum["precision"]:.4f}, '
          f'Recall: {metrics_accum["recall"]:.4f}, '
          f'F1-Score: {metrics_accum["f1_score"]:.4f}, '
          f'AUC: {metrics_accum["auc"]:.4f}')

    return test_loss, metrics_accum  # Return the test loss and metrics

# Start the training process
train(model, train_loader, val_loader, optimizer, num_epochs=50, patience=5)

# Load the best model before testing
model.load_state_dict(torch.load('best_model.pth'))

# Test the model on the separate test dataset (from the new directory)
test_loss, test_metrics = test(model, test_loader)