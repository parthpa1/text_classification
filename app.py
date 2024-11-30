import streamlit as st
from inference_pipeline import InferencePipeline

# Initialize the inference pipeline
model_path = 'best_text_classifier_model.joblib'
label_encoder_path = 'label_encoder.joblib'
pipeline = InferencePipeline(model_path, label_encoder_path)

# Streamlit app
st.title("PDF Classification Inference")

# Input for PDF URL
pdf_url = st.text_input("Enter the PDF URL:")

# Predict button
if st.button("Predict"):
    if pdf_url:
        label, probabilities = pipeline.predict(pdf_url)
        if isinstance(label, str) and label.startswith("Failed"):
            st.error(label)
        else:
            st.success(f"Predicted Label: {label}")
            st.write(f"Class Probabilities: {probabilities}")
    else:
        st.error("Please enter a PDF URL.")
