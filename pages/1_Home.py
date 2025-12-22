import streamlit as st

st.title("TraceFinder")

st.subheader("Project Overview")

st.write("""
TraceFinder is a deep learning–based image source classification system designed 
to identify the origin of digital images using convolutional neural networks. 
The project focuses on distinguishing images obtained from official sources 
and online repositories by learning intrinsic patterns present in image data.

This system demonstrates the practical application of deep learning in digital 
forensics and multimedia analysis. The model is trained using a custom dataset 
and evaluated using standard performance metrics.
""")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.button("Go to CNN Prediction")

with col2:
    st.button("View Model Details")
