import streamlit as st

st.title("Model and Dataset Details")

st.subheader("CNN Architecture")
st.write("""
The proposed convolutional neural network consists of three convolutional layers 
followed by max pooling operations. Fully connected layers are used for final 
classification. Dropout regularization is applied to reduce overfitting.
""")

st.subheader("Dataset")
st.write("""
The dataset is composed of images collected from official sources and online 
repositories. Images are resized to 128×128 resolution and normalized before 
training. The dataset is split into training, validation, and testing sets.
""")

st.subheader("Training Configuration")
st.write("""
Optimizer: Adam  
Loss Function: Cross Entropy Loss  
Batch Size: 32  
Image Size: 128×128  
Framework: PyTorch
""")
