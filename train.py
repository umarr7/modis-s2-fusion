import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import FusionModel
from dataset import FusionDataset
import os

# -----------------------------------------------------
# 1. DIRECTORIES
# -----------------------------------------------------
# Adjusted paths based on project structure
MODIS_DIR = "modis and s2 datasets/modis/"
S2_DIR = "modis and s2 datasets/s2/"
IMAGE_COUNT = 20   # we are using 20 pairs

def train():
    # -----------------------------------------------------
    # 2. LOAD DATA
    # -----------------------------------------------------
    if not os.path.exists(MODIS_DIR):
        print(f"Error: MODIS directory not found at {os.path.abspath(MODIS_DIR)}")
        return
        
    dataset = FusionDataset(MODIS_DIR, S2_DIR, count=IMAGE_COUNT)
    if len(dataset) == 0:
        print("Dataset is empty. Check your directories.")
        return

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # -----------------------------------------------------
    # 3. TRAINING SETUP
    # -----------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = FusionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -----------------------------------------------------
    # 4. TRAIN THE MODEL
    # -----------------------------------------------------
    EPOCHS = 50

    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for modis, s2 in loader:
            modis = modis.to(device)
            s2 = s2.to(device)

            optimizer.zero_grad()
            output = model(modis)
            loss = criterion(output, s2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
             print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    # -----------------------------------------------------
    # 5. SAVE MODEL
    # -----------------------------------------------------
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
        
    save_path = "fusion_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel Saved Successfully as {save_path}!")

if __name__ == "__main__":
    train()
