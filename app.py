import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="AI TraceFinder | Forensic Scanner ID",
    page_icon="🔍",
    layout="wide",
)

# 2. Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .feature-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        height: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HERO SECTION ---
st.title("🔍 AI TraceFinder")
st.subheader("Forensic Machine Learning for Scanner Device Identification")

col_h1, col_h2 = st.columns([2, 1])

with col_h1:
    st.markdown("""
    **Every scanner leaves a digital fingerprint.** AI TraceFinder is a specialized forensic platform that identifies the specific source scanner (brand and model) used to digitize documents. 
    By analyzing microscopic noise patterns, texture variations, and compression artifacts, we provide high-confidence device authentication for legal and security professionals.
    """)
    # st.button("Request Early Access")

with col_h2:
    # Placeholder for a technical graphic
    st.info("💡 **Did you know?** Scanners introduce unique 'PRNU' (Photo-Response Non-Uniformity) patterns that are nearly impossible to forge.")

st.divider()

# --- ADVANTAGES SECTION ---
st.header("🛡️ Why AI TraceFinder?")
col_a1, col_a2, col_a3 = st.columns(3)

with col_a1:
    st.markdown("### 🔬 Unrivaled Precision")
    st.write("Our ML models are trained on a large datset to detect artifacts invisible to the human eye.")

with col_a2:
    st.markdown("### ⚡ Instant Validation")
    st.write("Get a forensic source report in seconds, not weeks. Streamline your document verification workflow.")

with col_a3:
    st.markdown("### ⚖️ Legal-Grade Logic")
    st.write("Built with forensic standards in mind, providing explainable evidence for authentication and fraud detection.")

st.divider()

# --- USE CASES SECTION ---
st.header("🌍 Real-World Use Cases")

# Using tabs for an organized look
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Digital Forensics", "Document Auth", "Legal Verification", "Corporate Security", "Supply Chain"
])

with tab1:
    st.subheader("Digital Forensics")
    st.write("**Scenario:** Determine which scanner was used to forge or duplicate legal documents.")
    st.info("Example: Detect whether a fake degree or certificate was created using a specific high-end consumer scanner.")

with tab2:
    st.subheader("Document Authentication")
    st.write("**Scenario:** Identify the source of printed/scanned images to detect tampering.")
    st.info("Example: Differentiate between legitimate scans from a government office vs. unauthorized external devices.")

with tab3:
    st.subheader("Legal Evidence Verification")
    st.write("**Scenario:** Ensure scanned copies submitted in court come from approved devices.")
    st.info("Example: Verify that a signed contract originated from the company’s official secure scanner.")

with tab4:
    st.subheader("Corporate IP Protection")
    st.write("**Scenario:** Trace the source of leaked internal documents.")
    st.info("Example: Identify the specific office department where a leaked internal memo was scanned.")

with tab5:
    st.subheader("Supply Chain Integrity")
    st.write("**Scenario:** Prevent the entry of fraudulent shipping labels or invoices.")
    st.info("Example: Automated systems can flag invoices that don't match the known 'scanner fingerprint' of a trusted supplier.")

st.divider()

# --- DEMO PLACEHOLDER ---
with st.container(border=True):
    st.header("🧪 Interactive Sandbox (Coming Soon)")
    st.write("Upload a scanned image below to see how our fingerprinting engine analyzes noise patterns.")
    uploaded_file = st.file_uploader("Choose a scan...", type=["jpg", "png", "pdf"], disabled=True)
    st.warning("The forensic backend is currently being integrated. Please check back soon!")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2025 AI TraceFinder | Powered by Forensic Machine Learning")