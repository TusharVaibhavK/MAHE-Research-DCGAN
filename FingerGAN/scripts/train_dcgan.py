import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image  # Add Pillow for loading BMP images

# Hyperparameters
image_size = 64
latent_dim = 100
hidden_dim = 64
num_epochs = 5  # Small for demo; increase for better results
batch_size = 64
learning_rate = 0.0002
beta1 = 0.5  # For Adam optimizer
train_ratio = 0.8  # 80% train, 20% test

# Device configuration
# Check if GPU is available and set device accordingly

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Custom dataset for SOCOFing


class FingerprintDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        # Recursively collect all BMP files from Real and Altered subfolders
        for root, _, files in sorted(os.walk(root_dir)):
            for file in files:
                if file.endswith('.BMP'):
                    self.images.append(os.path.join(root, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        # Use Pillow to load BMP image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image


# Transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),  # Converts PIL Image to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Load and split dataset
dataset = FingerprintDataset(root_dir='data/SOCOFing/', transform=transform)
total_size = len(dataset)
train_size = int(train_ratio * total_size)
test_size = total_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Generator model for 1-channel images


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim *
                               8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim *
                               4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim *
                               2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 1, 4, 2, 1, bias=False),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

# Discriminator model for 1-channel images


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, x):
        output = self.model(x)
        # Reshape to [batch_size, 1] to match label shape
        return output.view(-1, 1)


# Initialize models and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)

g_optimizer = torch.optim.Adam(
    generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
d_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Loss function
adversarial_loss = nn.BCELoss()

# Training loop
for epoch in range(num_epochs):
    for i, real_images in enumerate(train_loader):
        real_images = real_images.to(device)
        # Added this line
        print(f"Batch {i}: real_images device: {real_images.device}")

        batch_size = real_images.size(0)

        # Labels
        real_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        d_optimizer.zero_grad()
        real_output = discriminator(real_images)
        d_real_loss = adversarial_loss(real_output, real_label)

        z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_images = generator(z)
        fake_output = discriminator(fake_images.detach())
        d_fake_loss = adversarial_loss(fake_output, fake_label)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        fake_output = discriminator(fake_images)
        g_loss = adversarial_loss(fake_output, real_label)
        g_loss.backward()
        g_optimizer.step()

        # Print progress
        if i % 10 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}], Batch [{i}/{len(train_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Save generated images
with torch.no_grad():
    z = torch.randn(16, latent_dim, 1, 1).to(device)
    fake_images = generator(z).cpu()
    fake_images = (fake_images + 1) / 2  # Rescale to [0, 1]

    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(fake_images[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.savefig('generated_images/generated_fingerprints.png')
    plt.close()

# Save models
torch.save(generator.state_dict(), 'models/generator.pth')
torch.save(discriminator.state_dict(), 'models/discriminator.pth')
