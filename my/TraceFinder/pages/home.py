import streamlit as st

# ---------------------- PAGE HEADER ----------------------

st.title("🔍 TraceFinder")
st.subheader("Forensic Scanner Identification System")

st.write("Welcome! This tool helps identify the scanner or device used to create forensic images.")
st.write("---")


# ---------------------- FEATURES SECTION ----------------------

st.header("✨ Features")
st.write("Here’s what TraceFinder can do:")

# 3 feature boxes in 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    st.info("📤 **Upload Image**\nUpload scanned forensic images for analysis.")

with col2:
    st.success("🔬 **Scanner Detection**\nAI model predicts which scanner/device captured the image.")

with col3:
    st.warning("📊 **Results Summary**\nGet probability scores and analysis details.")

st.write("---")


# ---------------------- NAVIGATION SECTION ----------------------

st.header("🚀 Navigation")

colA, colB = st.columns(2)

with colA:
    if st.button("Go to Upload Page"):
        st.switch_page("pages/Upload_file.py")

with colB:
    if st.button("About Project"):
        st.switch_page("pages/About_project.py")

