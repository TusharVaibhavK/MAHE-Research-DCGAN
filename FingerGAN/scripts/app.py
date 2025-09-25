import streamlit as st
import os
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------- SETTINGS ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_classifier = "checkpoints/regional_classifier.pth"
num_classes = 2
class_names = {0: "Africa", 1: "India"}

# ---------------- LOAD CLASSIFIER ---------------- #
@st.cache_resource
def load_classifier():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_classifier, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_classifier()

# ---------------- TRANSFORMS ---------------- #
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ---------------- APP LAYOUT ---------------- #
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏋️ Training", "🎨 Generation", "🧾 Classification", "📊 Analysis"])

# ---------------- TRAINING PAGE ---------------- #
if page == "🏋️ Training":
    st.title("Train Regional Classifier")
    st.write("Set hyperparameters and start training (backend integration needed).")
    epochs = st.slider("Epochs", 1, 20, 10)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
    st.button("Start Training (Coming Soon)")

# ---------------- GENERATION PAGE ---------------- #
elif page == "🎨 Generation":
    st.title("Generate Synthetic Fingerprints")
    st.write("Select region and generate synthetic samples (backend integration needed).")
    region = st.radio("Region", ["Africa", "India"])
    num_samples = st.slider("Number of Samples", 1, 16, 4)
    st.button("Generate (Coming Soon)")

# ---------------- CLASSIFICATION PAGE ---------------- #
elif page == "🧾 Classification":
    st.title("Fingerprint Classification")
    st.write("Upload a fingerprint image to predict its region (Africa/India).")

    uploaded = st.file_uploader("Upload Fingerprint", type=["png", "jpg", "bmp"])
    if uploaded:
        img = Image.open(uploaded).convert("L")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Transform and predict
        inp = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            pred = out.argmax(dim=1).item()
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]

        st.success(f"Predicted Region: **{class_names[pred]}**")
        st.write(f"Confidence: Africa {probs[0]:.2f}, India {probs[1]:.2f}")

        # Plot confidence
        fig, ax = plt.subplots()
        sns.barplot(x=list(class_names.values()), y=probs, ax=ax)
        ax.set_ylim(0, 1)
        st.pyplot(fig)

# ---------------- ANALYSIS PAGE ---------------- #
elif page == "📊 Analysis":
    st.title("Classification Analysis")
    st.write("Show performance metrics (confusion matrix, accuracy, etc.).")

    # Example: load test images from generated_samples and evaluate
    test_dir = "generated_samples/individual"
    preds, labels = [], []

    if os.path.exists(test_dir):
        for f in os.listdir(test_dir):
            if f.endswith((".png", ".bmp")):
                img = Image.open(os.path.join(test_dir, f)).convert("L")
                inp = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(inp)
                    pred = out.argmax(dim=1).item()
                preds.append(pred)
                # Assign dummy ground-truth (if unknown, just use 0)
                labels.append(0)

        if preds:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            cm = confusion_matrix(labels, preds, labels=[0,1])
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Africa","India"])
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            st.pyplot(fig)
        else:
            st.info("No samples found in synthetic test directory.")
