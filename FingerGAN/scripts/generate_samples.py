import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
from models.cond_generator import CondGenerator

def create_compatible_generator(checkpoint_path, nz=100, n_labels=2, ngf=64, nc=1):
    """Create a generator that's compatible with the checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint to inspect its structure
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if this is from a non-conditional DCGAN (without label input)
    if 'model.0.weight' in checkpoint:
        # This is a standard DCGAN without conditioning
        print("Detected non-conditional DCGAN checkpoint")
        
        # Create a standard DCGAN generator (without label conditioning)
        class StandardGenerator(torch.nn.Module):
            def __init__(self, nz=100, ngf=64, nc=1):
                super().__init__()
                self.nz = nz
                self.main = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                    torch.nn.BatchNorm2d(ngf * 8),
                    torch.nn.ReLU(True),
                    
                    torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                    torch.nn.BatchNorm2d(ngf * 4),
                    torch.nn.ReLU(True),
                    
                    torch.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                    torch.nn.BatchNorm2d(ngf * 2),
                    torch.nn.ReLU(True),
                    
                    torch.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                    torch.nn.BatchNorm2d(ngf),
                    torch.nn.ReLU(True),
                    
                    torch.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                    torch.nn.Tanh()
                )
            
            def forward(self, input):
                return self.main(input)
        
        generator = StandardGenerator(nz=nz, ngf=ngf, nc=nc).to(device)
        
        # Fix key names if needed
        if any(key.startswith('model.') for key in checkpoint.keys()):
            new_checkpoint = {}
            for key, value in checkpoint.items():
                new_key = key.replace('model.', 'main.')
                new_checkpoint[new_key] = value
            checkpoint = new_checkpoint
        
        generator.load_state_dict(checkpoint)
        return generator, device
    
    else:
        # Try to load as conditional generator
        generator = CondGenerator(nz=nz, n_labels=n_labels, ngf=ngf, nc=nc).to(device)
        try:
            generator.load_state_dict(checkpoint)
            return generator, device
        except:
            print("Could not load as conditional generator, using random initialization")
            return generator, device

# --- settings ---
nz = 100
n_labels = 2
ngf = 64
nc = 1
checkpoint_path = "models/generator.pth"
save_dir = "generated_samples"
os.makedirs(save_dir, exist_ok=True)

# --- load generator ---
print(f"Using device: cuda" if torch.cuda.is_available() else "Using device: cpu")

if os.path.exists(checkpoint_path):
    print(f"Found checkpoint: {checkpoint_path}")
    generator, device = create_compatible_generator(checkpoint_path, nz=nz, n_labels=n_labels, ngf=ngf, nc=nc)
    print("Generator loaded successfully!")
else:
    print("No checkpoint found. Using conditional generator with random weights.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = CondGenerator(nz=nz, n_labels=n_labels, ngf=ngf, nc=nc).to(device)

generator.eval()

# --- generate samples ---
print("Generating samples...")

# Check if it's a conditional generator
is_conditional = hasattr(generator, 'label_emb')

if is_conditional:
    # Generate samples for each region
    for label in range(n_labels):
        region_name = "Africa" if label == 0 else "SouthAsia"
        print(f"Generating samples for {region_name}")
        
        z = torch.randn(16, nz, device=device)
        labels = torch.full((16,), label, dtype=torch.long, device=device)
        
        with torch.no_grad():
            fake_imgs = generator(z, labels).cpu()
        
        # Denormalize and save
        fake_imgs = (fake_imgs + 1) / 2
        
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flatten()):
            if i < len(fake_imgs):
                img = fake_imgs[i][0].numpy()
                ax.imshow(img, cmap="gray")
            ax.axis("off")
        
        plt.suptitle(f"Generated Fingerprints - {region_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/samples_{region_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
else:
    # Standard DCGAN - generate without labels
    print("Generating samples with standard DCGAN (no region conditioning)")
    
    z = torch.randn(16, nz, 1, 1, device=device)  # Add spatial dimensions
    
    with torch.no_grad():
        fake_imgs = generator(z).cpu()
    
    # Denormalize and save
    fake_imgs = (fake_imgs + 1) / 2
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        if i < len(fake_imgs):
            img = fake_imgs[i][0].numpy()
            ax.imshow(img, cmap="gray")
        ax.axis("off")
    
    plt.suptitle("Generated Fingerprints - Standard DCGAN", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/samples_dcgan.png", dpi=150, bbox_inches='tight')
    plt.close()

print(f"Samples saved to {save_dir}/")

# Save individual images for closer inspection
print("Saving individual images...")
individual_dir = f"{save_dir}/individual"
os.makedirs(individual_dir, exist_ok=True)

if is_conditional:
    for label in range(n_labels):
        region_name = "Africa" if label == 0 else "SouthAsia"
        z = torch.randn(5, nz, device=device)
        labels = torch.full((5,), label, dtype=torch.long, device=device)
        
        with torch.no_grad():
            fake_imgs = generator(z, labels).cpu()
        
        fake_imgs = (fake_imgs + 1) / 2
        
        for i, img in enumerate(fake_imgs):
            plt.figure(figsize=(4, 4))
            plt.imshow(img[0].numpy(), cmap="gray")
            plt.axis("off")
            plt.title(f"{region_name} - Sample {i+1}")
            plt.savefig(f"{individual_dir}/{region_name}_sample_{i+1}.png", dpi=150, bbox_inches='tight')
            plt.close()
else:
    # Standard DCGAN
    z = torch.randn(5, nz, 1, 1, device=device)
    
    with torch.no_grad():
        fake_imgs = generator(z).cpu()
    
    fake_imgs = (fake_imgs + 1) / 2
    
    for i, img in enumerate(fake_imgs):
        plt.figure(figsize=(4, 4))
        plt.imshow(img[0].numpy(), cmap="gray")
        plt.axis("off")
        plt.title(f"Sample {i+1}")
        plt.savefig(f"{individual_dir}/sample_{i+1}.png", dpi=150, bbox_inches='tight')
        plt.close()

print("All samples generated successfully!")