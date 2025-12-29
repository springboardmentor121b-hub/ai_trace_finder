import streamlit as st
from PIL import Image

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Upload Scanned Document")

# ----------------- PAGE TITLE -----------------
st.title("📁 Upload Scanned Document")
st.subheader("Upload a scanned image for analysis")

st.write("""
**Instructions:**
- Allowed file types: `png`, `jpg`, `jpeg`, `tiff`  
- Make sure the scanned document is **clear and high-resolution**  
- Supported documents: certificates, official letters, PDFs (as images), IDs, etc.
""")

# ----------------- FILE UPLOADER -----------------
uploaded = st.file_uploader(
    "Choose a file to upload",
    type=["png", "jpg", "jpeg", "tiff"]
)

# ----------------- DISPLAY UPLOADED FILE -----------------
if uploaded:
    st.success("✅ File uploaded successfully!")
    
    # Display the image
    try:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    except:
        st.warning("⚠ Unable to open this file as an image. Please upload a valid image file.")

    st.markdown("---")
    st.write("You can now go to the **Live Prediction** page to select the model and predict.")
else:
    st.info("👈 Please upload a scanned image file from your computer.")