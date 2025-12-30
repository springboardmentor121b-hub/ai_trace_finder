import streamlit as st

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="TRACEFINDER | Forensic Scanner Identification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("🔍 TRACEFINDER")
    st.caption("Forensic Scanner Identification System")

    st.markdown("---")

    st.markdown("### 📂 Navigation")
    st.page_link("pages/home.py", label="🏠 Home")
    st.page_link("pages/upload_file.py", label="📤 Upload Document")
    st.page_link("pages/live_prediction.py", label="🔮 Live Prediction")
    st.page_link("pages/about_project.py", label="ℹ️ About Project")

    st.markdown("---")
    st.info(
        "⚙️ Models Used:\n"
        "- CNN\n"
        "- Random Forest\n"
        "- SVM"
    )

# -------------------- MAIN CONTENT --------------------
st.markdown(
    "<h1 style='text-align: center;'>🔍 TRACEFINDER</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align: center; color: gray;'>"
    "AI-Based Forensic Scanner & Document Source Identification System"
    "</h4>",
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------- HERO SECTION --------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.success("📤 **Upload Scanned Documents**\n\nSupports JPG, PNG, JPEG formats")

with col2:
    st.info("🤖 **Multiple AI Models**\n\nCNN, Random Forest, SVM")

with col3:
    st.warning("📊 **Detailed Predictions**\n\nModel name, result & confidence score")

st.markdown("---")

# -------------------- WHY TRACEFINDER --------------------
st.subheader("🚀 Why TRACEFINDER?")

st.markdown("""
- Detects **forged or tampered documents**
- Identifies **scanner/device source**
- Uses **machine learning & deep learning**
- Designed for **forensic & academic use**
""")

# -------------------- CALL TO ACTION --------------------
st.markdown("---")

cta_col1, cta_col2 = st.columns(2)

with cta_col1:
    if st.button("📤 Upload Document", use_container_width=True):
        st.switch_page("pages/upload_file.py")

with cta_col2:
    if st.button("🔮 Run Live Prediction", use_container_width=True):
        st.switch_page("pages/live_prediction.py")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("© 2025 TRACEFINDER | AI & ML Forensic Project")