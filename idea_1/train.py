import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from DeepfakeDetector import DeepfakeDetector
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the configuration
config = {
    'backbone_name': 'resnet18',
    'pretrained': True,
    'input_channels': 16,  # Original 3 channels + wavelet features + Sobel features
    'loss_func': 'CrossEntropyLoss',
}

# Initialize the detector
model = DeepfakeDetector(config)

# Define the directories for deepfake and real frames
#deepfake_dir = ""
#real_dir = ""

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
])


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

# Create the dataset
dataset = DeepfakeDataset(deepfake_directory, real_directory, transform=transform)

# Split dataset into training and testing sets
train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42, stratify=[label for _, label in dataset.data])

# Create DataLoader for training and testing sets
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

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

# Training function with added saving of wavelet and Sobel images
def train(model, dataloader, optimizer, num_epochs=10):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
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

            # Compute metrics (for monitoring during training)
            _, preds = torch.max(pred_dict['logits'], 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)
            running_loss += loss.item()

            # Save wavelet and Sobel filtered images for the first image in the batch every 10 batches
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {correct_preds / total_preds:.4f}')
                
                # Apply wavelet and Sobel transforms
                wavelet_features = model.apply_wavelet_transform(images[:1])  # Only visualize the first image in the batch
                sobel_features = model.apply_sobel_filters(images[:1])

                wavelet_img = tensor_to_image(wavelet_features[0][0])  # First channel of wavelet transform
                sobel_img = tensor_to_image(sobel_features[0][0])  # First channel of Sobel-filtered image

                # Save images
                save_image("Wavelet Transform", wavelet_img, batch_idx)
                save_image("Sobel Filter Output", sobel_img, batch_idx)

        # Log epoch loss and accuracy
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct_preds / total_preds
        print(f'Epoch [{epoch + 1}/{num_epochs}] finished. Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # Optionally save the model checkpoint after each epoch
        torch.save(model.state_dict(), f'deepfake_detector_epoch_{epoch+1}.pth')

# Testing function
def test(model, dataloader):
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

    test_loss = running_loss / len(dataloader)
    test_acc = correct_preds / total_preds
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# Start the training process
train(model, train_loader, optimizer, num_epochs=1)

# Test the model on the test dataset
test(model, test_loader)
