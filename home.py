import streamlit as st
st.set_page_config(
    page_title="Trace Finder",
    layout="wide"
)

st.markdown("""
<style>

body {
    background-color: #F7FAFC; 
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827;
    padding: 25px 20px;
}

.sidebar-title {
    color: #38BDF8;
    font-size: 32px;
    font-weight: 700;
    margin-bottom: -5px;
}

.sidebar-subtitle {
    color: #CBD5E1;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.menu-btn {
    background: #1F2937;
    padding: 12px 16px;
    border-radius: 8px;
    color: #E5E7EB;
    font-size: 15px;
    margin-bottom: 8px;
}

.menu-btn:hover {
    background: #374151;
}

/* System status box */
.status-box {
    background: #0F172A;
    padding: 14px;
    border-radius: 10px;
    margin-top: 20px;
    color: #E2E8F0;
}

.status-label {
    color: #94A3B8;
    font-size: 12px;
    letter-spacing: 1px;
}

/* Hero section */
.hero-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #475569 100%);
    padding: 55px;
    border-radius: 28px;
    text-align: center;
    color: white;
    margin: 30px auto 20px auto;
    width: 95%;
    box-shadow: 0px 20px 45px rgba(0,0,0,0.35);
}

.hero-small {
    font-size: 12px;
    letter-spacing: 2px;
    color: #7DD3FC;
}

.hero-title {
    font-size: 48px;
    font-weight: 700;
    margin-bottom: 8px;
}

.hero-desc {
    color: #CBD5E1;
    font-size: 17px;
}

/* Feature blocks */
.feature-box {
    text-align: center;
    margin-top: 40px;
}

.feature-title {
    font-size: 22px;
    margin-top: 8px;
    font-weight: 600;
}

.feature-desc {
    font-size: 13px;
    color: #6B7280;
}

</style>
""", unsafe_allow_html=True)

with st.sidebar:

    st.markdown("<div class='sidebar-title'>Trace Finder</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-subtitle'>FORENSIC SCANNER ID</div>", unsafe_allow_html=True)
    st.markdown("<div class='menu-btn'>🏠 Dashboard Home</div>", unsafe_allow_html=True)
    st.markdown("<div class='menu-btn'>📄 Project Overview</div>", unsafe_allow_html=True)
    # st.markdown("<div class='menu-btn'>⚙️ System Architecture</div>", unsafe_allow_html=True)

home , project = st.tabs(["Home" , "Project"])

with home:
    st.markdown("""
    <div class='hero-box'>
        <div class='hero-title'>Trace Finder</div>
        <div class='hero-small'>FORENSIC SCANNER IDENTIFICATION</div>
        <div class='hero-desc'>
            Identify the source scanner device used to scan a document or image by analyzing the unique patterns or artifacts left behind during the scanning process.
        </div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
        st.image("https://www.shutterstock.com/image-vector/fingerprint-scan-icon-search-vector-600nw-2231491229.jpg",
                 width=85)
        st.markdown("<div class='feature-title'>Digital Forensic</div>", unsafe_allow_html=True)
        st.markdown("<div class='feature-desc'>Duplicates legal documents</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
        st.image(
            "https://media.istockphoto.com/id/1288288160/vector/tick-check-mark-icon-with-document-list-with-tick-check-marks-with.jpg?s=612x612&w=0&k=20&c=w9SxGrLykc374m6hCkvt7LfJqSEnzQvrfYvOgH_0j1o=",
            width=85)
        st.markdown("<div class='feature-title'>Document Authentication</div>", unsafe_allow_html=True)
        st.markdown("<div class='feature-desc'>Identifies source of scanned images</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
        st.image("https://img.freepik.com/premium-vector/verification-logo-design-simple-icon_1012565-29.jpg", width=85)
        st.markdown("<div class='feature-title'>Legal Evidence Verification</div>", unsafe_allow_html=True)
        st.markdown("<div class='feature-desc'>Ensures the scanned copies</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with project:
    st.title("📘 Project Overview")

    st.write("""
    The **Trace Finder** project is designed to identify the **source scanning device** 
    used to generate a digital or scanned document using noise-based forensic analysis.
    """)

    st.subheader("🔍 Core Objectives")
    st.markdown("""
    - Extract device noise patterns (PRNU)
    - Analyze frequency domain signals (FFT, Wavelets)
    - Classify scanner models using CNN / SVM
    """)

    st.subheader("📁 Project Workflow")
    st.markdown("""
    1. Upload scanned document  
    2. Extract noise fingerprint  
    3. Preprocess using filters  
    4. Compare noise signatures  
    5. Predict scanner device  
    """)

    st.success("This page will later include diagrams, tables and dataset details.")
