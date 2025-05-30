# MAHE Research DCGAN Project 🧠🖼️

This repository contains implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) for generating synthetic images.

## 📋 Overview

DCGANs are a class of Convolutional Neural Networks where:
- A Generator network learns to create images that look real
- A Discriminator network learns to distinguish between real and fake images
- Both networks compete and improve through adversarial training

## 🛠️ Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm

Install dependencies with:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MAHE-Research-DCGAN
   ```

2. **Prepare your dataset**
   - Place your images in the `data/` directory, or
   - Modify the data loading code to use your custom dataset

3. **Train the model**
   ```bash
   python train.py --epochs 100 --batch_size 64 --lr 0.0002
   ```

4. **Generate images**
   ```bash
   python generate.py --model_path checkpoints/generator.pth --num_images 10
   ```

## 🖼️ Sample Generated Images

![Generated Fingerprint Images](samples/fingerprints_grid.png)

*The above image shows a grid of synthetically generated fingerprint patterns created by our DCGAN model. The model has learned to reproduce realistic ridge patterns, loops, and whorls characteristic of human fingerprints while maintaining diversity across samples.*

## 📊 Model Architecture

### Generator
- Takes random noise vector as input
- Uses transposed convolutions to generate images
- Batch normalization and ReLU activations
- Outputs images with tanh activation

### Discriminator
- Convolutional network with stride=2
- Leaky ReLU activations
- Outputs probability of image being real

## 🔒 Ethical and Security Framework

Our research addresses the ethical and security challenges associated with generating synthetic biometric data using DCGANs. Below is a framework visualization:

```mermaid
graph LR
    subgraph "DCGAN Generation Process"
        A[Random Noise] --> B[Generator Network]
        B --> C[Synthetic Biometric Data]
        D[Real Biometric Data] --> E[Discriminator Network]
        C --> E
        E --> F{Real or Fake?}
        F --> |Feedback| B
    end
    
    subgraph "Ethical & Security Challenges"
        C --> G[Privacy Concerns]
        C --> H[Identity Theft]
        C --> I[Biometric System Spoofing]
        C --> J[Consent Issues]
        C --> K[Data Ownership]
    end
    
    subgraph "Proposed Framework"
        L[Detection Mechanisms] --> M[Synthetic Data Watermarking]
        L --> N[Authenticity Verification]
        O[Governance] --> P[Ethical Guidelines]
        O --> Q[Usage Policies]
        R[Technical Safeguards] --> S[Access Control]
        R --> T[Encryption]
    end
    
    C --> L
    C --> O
    C --> R
    
    style A fill:#f9f,stroke:#333,stroke-width:2px,color:#333
    style B fill:#f9f,stroke:#333,stroke-width:1px,color:#333
    style C fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style D fill:#bbf,stroke:#333,stroke-width:1px,color:#333
    style E fill:#bbf,stroke:#333,stroke-width:1px,color:#333
    style F fill:#bbf,stroke:#333,stroke-width:1px,color:#333
    style G fill:#fbb,stroke:#333,stroke-width:2px,color:#600
    style H fill:#fbb,stroke:#333,stroke-width:2px,color:#600
    style I fill:#fbb,stroke:#333,stroke-width:2px,color:#600
    style J fill:#fbb,stroke:#333,stroke-width:2px,color:#600
    style K fill:#fbb,stroke:#333,stroke-width:2px,color:#600
    style L fill:#bfb,stroke:#333,stroke-width:2px,color:#060
    style M fill:#bfb,stroke:#333,stroke-width:1px,color:#060
    style N fill:#bfb,stroke:#333,stroke-width:1px,color:#060
    style O fill:#bfb,stroke:#333,stroke-width:2px,color:#060
    style P fill:#bfb,stroke:#333,stroke-width:1px,color:#060
    style Q fill:#bfb,stroke:#333,stroke-width:1px,color:#060
    style R fill:#bfb,stroke:#333,stroke-width:2px,color:#060
    style S fill:#bfb,stroke:#333,stroke-width:1px,color:#060
    style T fill:#bfb,stroke:#333,stroke-width:1px,color:#060
```

This framework addresses how our DCGAN implementation for generating biometric data intersects with privacy, security, and ethical concerns, while proposing technical and governance solutions to mitigate potential risks.

## 📝 Citation

If you use this code for your research, please cite:

```
@misc{mahe-dcgan,
  author = {MAHE Research Team},
  title = {MAHE-Research-DCGAN},
  year = {2025},
  url = {https://github.com/username/MAHE-Research-DCGAN}
}
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- MAHE Research Team

## 🙏 Acknowledgements

- [Referenced DCGAN Paper](https://arxiv.org/abs/1511.06434)
