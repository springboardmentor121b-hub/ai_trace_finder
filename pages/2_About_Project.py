import streamlit as st

st.title("About the Project")

st.write("""
With the rapid growth of digital media, identifying the source of images has 
become increasingly important in areas such as digital forensics, copyright 
protection, and misinformation detection. Traditional image analysis methods 
often fail to capture subtle statistical differences between image sources.

TraceFinder addresses this challenge by leveraging a convolutional neural 
network trained on images collected from multiple sources. The model learns 
discriminative features such as texture patterns, resolution artifacts, and 
pixel-level variations that are difficult to detect manually.

The project demonstrates an end-to-end pipeline including dataset preparation, 
model training, evaluation, and deployment through an interactive web interface.
""")
