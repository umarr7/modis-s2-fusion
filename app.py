import streamlit as st
import torch
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
import os
from model import FusionModel

# Page configuration
st.set_page_config(
    page_title="MODIS-Sentinel Fusion",
    page_icon="🌍",
    layout="wide"
)

# Constants
# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fusion_model.pth")
MODIS_DIR = os.path.join(BASE_DIR, "modis and s2 datasets", "modis")
S2_DIR = os.path.join(BASE_DIR, "modis and s2 datasets", "s2")

@st.cache_resource
def load_model():
    """Load the trained FusionModel."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FusionModel()
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:

        st.warning(f"Model file '{MODEL_PATH}' not found.")
        st.info(f"Current Directory: {os.getcwd()}")
        st.info(f"Files in {BASE_DIR}: {os.listdir(BASE_DIR)}")
        return None

def normalize(arr):
    """Normalize array to 0-1 range for visualization."""
    if arr.max() > 0:
        return arr / arr.max()
    return arr

def predict(model, modis_path):
    """Run prediction on a MODIS image."""
    # Load and Preprocess
    with rasterio.open(modis_path) as src:
        modis_raw = src.read([1, 2])    # b01=RED, b02=NIR

    modis_b1 = cv2.resize(modis_raw[0], (32, 32))
    modis_b2 = cv2.resize(modis_raw[1], (32, 32))
    modis_resized = np.stack([modis_b1, modis_b2], axis=0)

    # Normalize
    m_max = np.max(modis_resized)
    if m_max > 0:
        modis_resized = modis_resized / m_max
    
    inp = torch.tensor(modis_resized, dtype=torch.float32).unsqueeze(0) # Add batch dim

    # Predict
    with torch.no_grad():
        pred = model(inp)

    pred_np = pred.squeeze().numpy()
    return pred_np, modis_resized

def calculate_ndvi(red, nir):
    """Calculate NDVI from Red and NIR bands."""
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

def main():
    st.title("🌍 MODIS to Sentinel-2 Image Fusion")
    st.markdown("""
    This application demonstrates a Deep Learning model that enhances low-resolution **MODIS** satellite imagery 
    to match the high resolution of **Sentinel-2** imagery.
    """)

    model = load_model()

    if not model:
        st.stop()
        
    st.sidebar.header("Options")
    
    # Selection mode
    mode = st.sidebar.radio("Input Source", ["Select Sample File", "Upload MODIS Image"])
    
    # Get list of sample files
    sample_files = []
    if os.path.exists(MODIS_DIR):
        sample_files = [f for f in os.listdir(MODIS_DIR) if f.endswith(".tif")]
        sample_files = sorted(sample_files, key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x and x.split('_')[-1].split('.')[0].isdigit() else x)

    if mode == "Select Sample File":
        if not sample_files:
            st.error("No sample files found in the MODIS directory.")
            return

        selected_file = st.sidebar.selectbox("Choose a sample", sample_files)
        modis_path = os.path.join(MODIS_DIR, selected_file)
        
        # Try to find corresponding S2 file
        # ID extraction logic assuming format MODIS_sample_{id}.tif
        try:
            file_id = selected_file.split("_")[-1] # e.g. "10.tif"
            s2_filename = f"S2_sample_{file_id}" 
            s2_path = os.path.join(S2_DIR, s2_filename)
        except:
            s2_path = None
            
        if st.sidebar.button("Run Prediction"):
            run_analysis(model, modis_path, s2_path)
            
    elif mode == "Upload MODIS Image":
        uploaded_file = st.sidebar.file_uploader("Upload a MODIS .tif file", type=["tif", "tiff"])
        
        if uploaded_file:
            # Save temporarily
            with open("temp_modis.tif", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.sidebar.button("Run Prediction"):
                run_analysis(model, "temp_modis.tif", None)

def run_analysis(model, modis_path, s2_path=None):
    with st.spinner("Processing..."):
        pred_sentinel, modis_input = predict(model, modis_path)
        
        # modis_input is [2, 32, 32] (Red, NIR)
        # pred_sentinel is [2, 128, 128] (Red, NIR)
        
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.subheader("Input MODIS (Low Res)")
        #     # fig, ax = plt.subplots()
        #     # ax.imshow(modis_input[0], cmap='gray')
        #     # ax.set_title("MODIS Red Band (32x32)")
        #     # ax.axis('off')
        #     # st.pyplot(fig)
            
        with col2:
            st.subheader("Predicted Sentinel-2 (High Res)")
            fig, ax = plt.subplots()
            ax.set_title("Predicted Red Band (128x128)")
            ax.axis('off')
            st.pyplot(fig)
            
        st.divider()
        
        if s2_path and os.path.exists(s2_path):
            st.success("Ground Truth Available")
            
            with rasterio.open(s2_path) as src:
                 s2_raw = src.read([1, 2])
                 s2_b4 = cv2.resize(s2_raw[0], (128, 128)) # Ensure size matches for visual consistency if needed
                 # Actually raw is already 128x128? Notebook said "Resize Sentinel-2 to 128x128"
                 # Let's assume raw might be bigger or same. Safe to resize for consistency.
            
            s2_disp = normalize(s2_b4)
            
            col3, col4 = st.columns(2)
            with col3:
                 st.subheader("Ground Truth Sentinel-2")
                 fig, ax = plt.subplots()
                 ax.set_title("Actual Red Band")
                 ax.axis('off')
                 st.pyplot(fig)
            
            with col4:
                 # Difference/Error map
                 diff = np.abs(s2_disp - normalize(pred_sentinel[0]))
                 st.subheader("Difference Map")
                 fig, ax = plt.subplots()
                 im = ax.imshow(diff, cmap='hot')
                 plt.colorbar(im, ax=ax)
                 ax.set_title("|Actual - Predicted|")
                 ax.axis('off')
                 st.pyplot(fig)

        st.divider()
        st.subheader("NDVI Analysis")
        
        # Calculate NDVIs
        modis_ndvi = calculate_ndvi(modis_input[0], modis_input[1])
        pred_ndvi = calculate_ndvi(pred_sentinel[0], pred_sentinel[1])
        
        c1, c2 = st.columns(2)
        with c1:
            # st.image(normalize(modis_ndvi), clamp=True, caption="MODIS NDVI", width=300) 
            # imshow might be better for heatmap
            fig, ax = plt.subplots()
            im = ax.imshow(modis_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_title("MODIS NDVI")
            ax.axis('off')
            st.pyplot(fig)
            st.metric("Mean NDVI", f"{modis_ndvi.mean():.4f}")

        with c2:
            fig, ax = plt.subplots()
            im = ax.imshow(pred_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_title("Predicted S2 NDVI")
            ax.axis('off')
            st.pyplot(fig)
            st.metric("Mean NDVI", f"{pred_ndvi.mean():.4f}")

if __name__ == "__main__":
    main()
