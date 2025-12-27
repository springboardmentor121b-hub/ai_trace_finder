import streamlit as st

st.set_page_config(
    page_title="TraceFinder | Hybrid CNN Scanner Identification",
    layout="wide"
)

st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #F1F3FA, #E8ECFF);
}
.hero {
    background-color: #EEF1FF;
    padding: 3rem;
    border-radius: 22px;
    box-shadow: 0px 15px 35px rgba(90,100,180,0.15);
}
.badge {
    background-color: #6C7BFF;
    color: white;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🔍 TraceFinder</h1>
<span class="badge">Hybrid CNN – Primary Model</span>
<p style="font-size:18px; margin-top:10px;">
An intelligent <b>scanner source identification system</b> using
Hybrid Convolutional Neural Networks and forensic feature analysis.
</p>
<ul>
<li>Digital Image Forensics</li>
<li>Document Authentication</li>
<li>Legal Evidence Verification</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.info("Use the sidebar to explore the project, test predictions, and view model performance.")
