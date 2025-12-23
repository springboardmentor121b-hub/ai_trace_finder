import streamlit as st

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(
    page_title="TraceFinder – Forensic Image Source Identification",
    page_icon="🕵️",
    layout="wide"
)

# -------------------- CUSTOM CSS FOR DARK UI -------------------- #
st.markdown("""
    <style>
        .title {
            font-size: 48px;
            font-weight: 800;
            color: #00e6e6;
            text-align: center;
            margin-top: 20px;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #d9d9d9;
            margin-bottom: 40px;
        }
        .feature-box {
            padding: 20px;
            background-color: #1b1b1b;
            border-radius: 10px;
            color: #e6e6e6;
            text-align: center;
            border: 1px solid #333;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            font-size: 14px;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- TITLE SECTION -------------------- #
st.markdown('<div class="title">🕵️ TraceFinder</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Forensic Image Source Identification Dashboard</div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------- PROJECT INTRO -------------------- #
st.markdown("""
### 🔍 What is TraceFinder?

TraceFinder is a **digital forensic system** built to identify **which scanner produced a scanned image**.

It uses:

- Metadata Feature Extraction  
- Noise Residual & Statistical Analysis  
- Machine Learning (Random Forest / SVM)  
- Streamlit-based Forensic Dashboard  

This dashboard allows you to **visualize the system, inspect features, make predictions, and explore applications**.
""")

# -------------------- FEATURE CARDS -------------------- #
st.markdown("### 🚀 Key Dashboard Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="feature-box">
            <h3>📘 System Flowchart</h3>
            Complete TraceFinder pipeline explained.
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-box">
            <h3>⚙️ Internal Mechanism</h3>
            Understand how metadata & noise reveal scanner identity.
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-box">
            <h3>🧪 Feature Extraction</h3>
            Explore metadata extracted from scanned images.
        </div>
    """, unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
        <div class="feature-box">
            <h3>🖼️ Predict Scanner</h3>
            Upload an image & identify its originating scanner.
        </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
        <div class="feature-box">
            <h3>📌 Applications</h3>
            Real-world forensic and legal use cases.
        </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
        <div class="feature-box">
            <h3>🎥 About Project</h3>
            Overview + demonstration video.
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -------------------- FOOTER -------------------- #
st.markdown("""
<div class="footer">
    © 2025 TraceFinder – Developed for Forensic Image Analysis  
</div>
""", unsafe_allow_html=True)
