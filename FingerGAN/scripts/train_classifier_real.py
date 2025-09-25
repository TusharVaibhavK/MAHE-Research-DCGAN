import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# --- settings ---
batch_size = 32
epochs = 10
num_classes = 2  # Africa vs India
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- transforms ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # convert to 3-channel
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --- Custom dataset wrapper ---
class FingerprintDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        # Africa -> SOCOFing
        africa_path = os.path.join(root_dir, "SOCOFing", "Real")
        for root, _, files in os.walk(africa_path):
            for f in files:
                if f.lower().endswith((".bmp", ".png")):
                    self.samples.append((os.path.join(root, f), 0))  # label 0 = Africa

        # India -> Family_Fingerprint
        family_path = os.path.join(root_dir, "FAMILY_FINGERPRINT_DATASET")
        for fam in os.listdir(family_path):
            fam_dir = os.path.join(family_path, fam)
            if os.path.isdir(fam_dir):
                for person in os.listdir(fam_dir):
                    person_dir = os.path.join(fam_dir, person)
                    if os.path.isdir(person_dir):
                        for f in os.listdir(person_dir):
                            if f.lower().endswith((".bmp", ".png")):
                                self.samples.append((os.path.join(person_dir, f), 1))  # label 1 = India

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label

# --- load datasets ---
train_data = FingerprintDataset(root_dir="data/real", transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

print(f"Loaded {len(train_data)} samples: {sum(l==0 for _,l in train_data)} Africa, {sum(l==1 for _,l in train_data)} India")

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
    correct, total = 0, 0
    
    # Create progress bar for current epoch
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                       unit="batch", leave=True)
    
    for imgs, labels in progress_bar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar with current metrics
        current_acc = 100 * correct / total
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg_Loss': f'{total_loss/(progress_bar.n+1):.4f}',
            'Acc': f'{current_acc:.2f}%'
        })

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} Complete - Loss: {total_loss/len(train_loader):.4f}, Accuracy: {acc:.2f}%")

torch.save(model.state_dict(), "checkpoints/regional_classifier.pth")
print("Classifier saved!")
