import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# --- settings ---
data_dir = "data"
batch_size = 32
epochs = 10
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- transforms ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

# --- datasets ---
train_data = datasets.ImageFolder(root=f"{data_dir}/real", transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = datasets.ImageFolder(root=f"{data_dir}/synthetic", transform=transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# --- model ---
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- training ---
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# --- evaluation ---
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy on Synthetic Data: {100*correct/total:.2f}%")
