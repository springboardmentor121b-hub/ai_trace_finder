import streamlit as st
from pathlib import Path

st.title("📘 System Flowchart – TraceFinder")

flowchart_path = Path("assets/flowchart.png")

if flowchart_path.exists():
    st.image(str(flowchart_path), caption="TraceFinder Full System Flowchart", use_column_width=True)
else:
    st.error("flowchart.png not found. Please place it in the main folder.")

st.markdown("""
### 🔍 Flow Explanation  
This flowchart represents the complete TraceFinder pipeline:

1. **Preprocess Official Dataset**  
2. **Preprocess Wikipedia Dataset**  
3. **Extract Metadata Features**  
4. **Perform Robust Feature Selection**  
5. **Baseline Model Training**  
6. **Fine-tuned Model Training**  
7. **Model Evaluation**  
8. **Feedback Loop until Accuracy Is High**
""")
