import streamlit as st

st.set_page_config(
    page_title="TraceFinder",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GLOBAL UI STYLE (applies to ALL pages)
st.markdown("""
<style>
.stApp {
    background-color: #f3f4f6;
}

section[data-testid="stSidebar"] {
    background-color: #1f2937;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

h1, h2, h3 {
    color: #1f2937;
}

.card {
    background-color: white;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 24px;
}

.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 8px 20px;
    border: none;
}

.stButton > button:hover {
    background-color: #1e40af;
}

.stFileUploader {
    background-color: #f9fafb;
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# HOME PAGE CONTENT
st.title("TraceFinder")

st.write(
    "A machine learning based system for identifying the source scanner "
    "of a scanned image using CNN, SVM, and Random Forest models."
)

st.info("Use the sidebar to navigate through different sections of the project.")
