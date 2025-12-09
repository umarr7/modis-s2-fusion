import os
import numpy as np
import rasterio
import cv2
import torch
from torch.utils.data import Dataset

class FusionDataset(Dataset):
    def __init__(self, modis_dir, s2_dir, count=20):
        self.modis_dir = modis_dir
        self.s2_dir = s2_dir
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # Handle simple indexing or if user requests specific index (though count usually implies 0..N-1)
        # The notebook used f"MODIS_sample_{idx}.tif". Note that some datasets might be 0-indexed or 1-indexed.
        # Looking at notebook, user didn't specify 0 or 1 base, but loop usually goes 0..count-1.
        # Wait, the notebook outputs show "Saved fused Sentinel-2 image → Predicted_S2_sample_10.tif".
        # Let's assume 0-indexed for now based on standard loop range(EPOCHS) but the filenames might tricky.
        # Let's verify filenames later if needed, but for now stick to notebook logic: f"{name}_{idx}.tif"
        
        modis_path = os.path.join(self.modis_dir, f"MODIS_sample_{idx}.tif")
        s2_path = os.path.join(self.s2_dir, f"S2_sample_{idx}.tif")
        
        # Check if files exist to avoid hard crash, or perhaps just let it crash so we know
        if not os.path.exists(modis_path) or not os.path.exists(s2_path):
             # Try 1-based indexing if 0 fails? Or just maybe the files are named differently. 
             # I will stick to exact notebook logic for now.
             pass

        # -------------------------
        # Load MODIS (2 bands)
        # -------------------------
        with rasterio.open(modis_path) as src:
            modis_raw = src.read([1, 2])    # b01=RED, b02=NIR

        # -------------------------
        # Load Sentinel-2 (2 bands)
        # -------------------------
        with rasterio.open(s2_path) as src:
            s2_raw = src.read([1, 2])       # B4=RED, B8=NIR (since you exported only these)

        # -------------------------
        # Resize MODIS to 32×32
        # -------------------------
        modis_b1 = cv2.resize(modis_raw[0], (32, 32))
        modis_b2 = cv2.resize(modis_raw[1], (32, 32))
        modis_resized = np.stack([modis_b1, modis_b2], axis=0)

        # -------------------------
        # Resize Sentinel-2 to 128×128
        # -------------------------
        s2_b4 = cv2.resize(s2_raw[0], (128, 128))
        s2_b8 = cv2.resize(s2_raw[1], (128, 128))
        s2_resized = np.stack([s2_b4, s2_b8], axis=0)

        # -------------------------
        # Normalize 0–1
        # -------------------------
        # Avoid division by zero if max is 0
        m_max = np.max(modis_resized)
        if m_max > 0:
            modis_resized = modis_resized / m_max
            
        s_max = np.max(s2_resized)
        if s_max > 0:
            s2_resized = s2_resized / s_max

        # Convert to float tensors
        modis_tensor = torch.tensor(modis_resized, dtype=torch.float32)
        s2_tensor = torch.tensor(s2_resized, dtype=torch.float32)

        return modis_tensor, s2_tensor
