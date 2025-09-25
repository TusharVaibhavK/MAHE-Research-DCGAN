import os
import torch
from torchvision import transforms, models
from PIL import Image

# --- settings ---
synthetic_dir = "generated_samples/individual"
checkpoint = "checkpoints/regional_classifier.pth"
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- transforms ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --- load classifier ---
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model = model.to(device)
model.eval()

# --- evaluate synthetic samples ---
class_names = {0: "Africa", 1: "India"}

for file in os.listdir(synthetic_dir):
    if file.lower().endswith((".png", ".bmp", ".jpg")):
        img_path = os.path.join(synthetic_dir, file)
        img = Image.open(img_path).convert("L")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).item()

        print(f"{file} → Predicted as {class_names[pred]}")
