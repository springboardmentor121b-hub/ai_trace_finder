import streamlit as st


# Sets the title of the page in the browser tab
st.set_page_config(page_title="About TraceFinder")


st.title("🔍 About TraceFinder – Forensic Scanner Identification")

# Short description about the project
st.write("""
TraceFinder is an intelligent forensic tool designed to **identify the source scanner or device**
from a scanned document.  
It helps forensic analysts detect **document forgery, tampering, and identity fraud** by studying
hidden patterns left behind by scanning devices.
""")

# HOW TRACEFINDER WORKS
st.header("📌 How TraceFinder Works")

st.write("""
TraceFinder follows a systematic forensic pipeline that extracts scanner-specific features 
from scanned documents.

Here is the complete workflow:
""")

# SHOW FLOWCHART IMAGE
st.subheader("📈 System Workflow Flowchart")

# Display the flowchart image from the 'pages' folder
st.image("pages/about_flowchart.png", caption="TraceFinder System Workflow", width=600)
# DETAILED PROCESS EXPLANATION
st.header("📘 Step-by-Step Workflow Explanation")

st.write("""
### **1️⃣ Upload Document**
The user uploads a scanned document. This could be:
- Certificates  
- Official documents  
- PDFs or scanned images  

### **2️⃣ Preprocessing**
The uploaded image is cleaned:
- Noise removal  
- Resizing  
- Contrast enhancement  

Improving quality helps extract scanner signatures accurately.

### **3️⃣ Feature Extraction**
TraceFinder extracts unique scanner fingerprints such as:
- Noise patterns  
- Sensor imperfections  
- Edge artefacts  

These are unique to each scanning device.

### **4️⃣ Classification / Matching**
Machine learning or pattern analysis identifies:
- Which scanner produced the document  
- Whether the document is forged or modified  

### **5️⃣ Final Report**
The system generates:
- Scanner identity  
- Confidence score  
""")

# BACK BUTTON
if st.button("⬅ Back to Home"):
    st.switch_page("pages/home.py")
