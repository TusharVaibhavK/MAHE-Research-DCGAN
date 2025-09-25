import sys
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Add parent directory to path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import models after path is set
try:
    from models.cond_generator import CondGenerator
    from models.cond_discriminator import CondDiscriminator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the FingerGAN directory")
    sys.exit(1)


# --------------------------
# Dataset
# --------------------------


class FingerprintDataset(Dataset):
    def __init__(self, csv_file, img_size=64):
        self.df = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # Map region to integer label
        self.label_map = {region: idx for idx,
                          region in enumerate(self.df['region'].unique())}
        self.df['label'] = self.df['region'].map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['file']).convert("L")
        img = self.transform(img)
        label = int(row['label'])
        return img, label

# --------------------------
# Training
# --------------------------


def train(
    csv_path="data/combined_metadata.csv",
    img_size=64,
    nz=100,
    n_labels=2,
    batch_size=64,
    epochs=50,
    lr=0.0002,
    out_dir="outputs"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FingerprintDataset(csv_path, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models
    G = CondGenerator(nz=nz, n_labels=n_labels, nc=1).to(device)
    D = CondDiscriminator(n_labels=n_labels, nc=1).to(device)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_aux = nn.CrossEntropyLoss()

    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    os.makedirs(f"{out_dir}/checkpoints", exist_ok=True)

    for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"):
        epoch_pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}/{epochs}", 
                         total=len(dataloader), leave=False, unit="batch")
        
        for i, (imgs, labels) in epoch_pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            b_size = imgs.size(0)

            # Real labels = 1, fake labels = 0
            real_label = torch.ones(b_size, 1, device=device)
            fake_label = torch.zeros(b_size, 1, device=device)

            # ----------------
            # Train D
            # ----------------
            D.zero_grad()
            real_out, real_aux = D(imgs)
            loss_D_real = criterion_gan(real_out, real_label)
            loss_D_aux = criterion_aux(real_aux, labels)

            z = torch.randn(b_size, nz, device=device)
            fake_imgs = G(z, labels)
            fake_out, fake_aux = D(fake_imgs.detach())
            loss_D_fake = criterion_gan(fake_out, fake_label)

            loss_D = loss_D_real + loss_D_fake + loss_D_aux
            loss_D.backward()
            optimizerD.step()

            # ----------------
            # Train G
            # ----------------
            G.zero_grad()
            z = torch.randn(b_size, nz, device=device)
            gen_labels = torch.randint(0, n_labels, (b_size,), device=device)
            gen_imgs = G(z, gen_labels)
            out, aux = D(gen_imgs)
            loss_G_gan = criterion_gan(out, real_label)
            loss_G_aux = criterion_aux(aux, gen_labels)
            loss_G = loss_G_gan + loss_G_aux
            loss_G.backward()
            optimizerG.step()
            
            # Update progress bar with current losses
            epoch_pbar.set_postfix({
                'D_loss': f'{loss_D.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}'
            })

        print(
            f"Epoch [{epoch+1}/{epochs}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")

        # Save samples
        z = torch.randn(n_labels*8, nz, device=device)
        labels_fixed = torch.tensor(
            [i % n_labels for i in range(n_labels*8)], device=device)
        samples = G(z, labels_fixed).detach().cpu()
        vutils.save_image(
            samples, f"{out_dir}/samples/epoch_{epoch+1}.png", nrow=n_labels, normalize=True)

        # Save checkpoint
        torch.save(G.state_dict(),
                   f"{out_dir}/checkpoints/G_epoch{epoch+1}.pth")
        torch.save(D.state_dict(),
                   f"{out_dir}/checkpoints/D_epoch{epoch+1}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Conditional GAN')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--nz', type=int, default=100,
                        help='Size of latent vector')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    parser.add_argument('--n_labels', type=int, default=2,
                        help='Number of labels/regions')

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        nz=args.nz,
        img_size=args.img_size,
        n_labels=args.n_labels
    )
