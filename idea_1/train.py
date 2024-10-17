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

# Define the configuration
config = {
    'backbone_name': 'resnet18',
    'pretrained': True,
    'input_channels': 16,  # Original 3 channels + wavelet features + Sobel features
    'loss_func': 'CrossEntropyLoss',
}

# Initialize the detector
model = DeepfakeDetector(config)

# Define the directories for deepfake and real frames (training data)
deepfake_directory = sys.argv[1]
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
    def __init__(self, deepfake_dir, real_dir, transform=None):
        self.deepfake_dir = deepfake_dir
        self.real_dir = real_dir
        self.transform = transform

        # Load all filenames and labels
        self.deepfake_images = [(os.path.join(deepfake_dir, f), 1) for f in os.listdir(deepfake_dir) if f.endswith(".jpg")]
        self.real_images = [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir) if f.endswith(".jpg")]

        # Combine both lists
        self.data = self.deepfake_images + self.real_images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
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

# Create the dataset for training and validation
full_dataset = DeepfakeDataset(deepfake_directory, real_directory, transform=transform)

# Split the dataset into training and validation sets (e.g., 80% train, 20% val)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

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

# Function to evaluate on validation set
def validate(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    correct_preds = 0
    total_preds = 0
    running_loss = 0.0

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

            # Compute accuracy
            _, preds = torch.max(pred_dict['logits'], 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)

    val_loss = running_loss / len(dataloader)
    val_acc = correct_preds / total_preds
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

    return val_loss, val_acc  # Return the validation loss and accuracy

# Trying early stopping
def train(model, train_loader, val_loader, optimizer, num_epochs=50, patience=5):
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

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

            # Compute metrics
            _, preds = torch.max(pred_dict['logits'], 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)
            running_loss += loss.item()

            # # Save wavelet and Sobel filtered images for the first image in the batch every 10 batches
            # if batch_idx % 10 == 0:
            #     print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], '
            #           f'Loss: {loss.item():.4f}, Accuracy: {correct_preds / total_preds:.4f}')

            #     # Apply wavelet and Sobel transforms
            #     wavelet_features = model.apply_wavelet_transform(images[:1])  # Only visualize the first image in the batch
            #     sobel_features = model.apply_sobel_filters(images[:1])

            #     wavelet_img = tensor_to_image(wavelet_features[0][0])  # First channel of wavelet transform
            #     sobel_img = tensor_to_image(sobel_features[0][0])  # First channel of Sobel-filtered image

            #     # Save images
            #     save_image("Wavelet Transform", wavelet_img, batch_idx)
            #     save_image("Sobel Filter Output", sobel_img, batch_idx)

        # Log epoch loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds
        print(f'Epoch [{epoch + 1}/{num_epochs}] finished. Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}')

        # Validate the model on the validation dataset
        val_loss, val_acc = validate(model, val_loader)
        # At the end of each epoch in the train function
        # After validation
        print(f'Epoch [{epoch + 1}/{num_epochs}] completed.')
        print(f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')


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

# Testing function
def test(model, dataloader):
    model.eval() 
    correct_preds = 0
    total_preds = 0
    running_loss = 0.0

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

            # Compute accuracy
            _, preds = torch.max(pred_dict['logits'], 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)

    test_loss = running_loss / len(dataloader)
    test_acc = correct_preds / total_preds
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

    return test_loss, test_acc  # Return the test loss and accuracy

# Start the training process with the updated training function
train(model, train_loader, val_loader, optimizer, num_epochs=1, patience=5)  # Adjusted patience parameter

# Load the best model before testing
model.load_state_dict(torch.load('best_model.pth'))

# Test the model on the separate test dataset (from the new directory)
test_loss, test_acc = test(model, test_loader)
