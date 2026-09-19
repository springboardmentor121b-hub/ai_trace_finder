import streamlit as st
from pathlib import Path
import sys

# Ensure configs module can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from configs.config import FLOWCHART_PATH
except ImportError:
    FLOWCHART_PATH = Path(__file__).resolve().parents[1] / "doc" / "assets" / "flowchart.png"

st.title("📘 System Flowchart – TraceFinder")

if FLOWCHART_PATH.exists():
    st.image(str(FLOWCHART_PATH), caption="TraceFinder Full System Flowchart", use_column_width=True)
else:
    st.error("flowchart.png not found. Please place it in doc/assets/.")

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
