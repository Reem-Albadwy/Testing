import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ===== Page Config =====
st.set_page_config(
    page_title="🦷 AI Teeth Diagnosis",
    page_icon="🦷",
    layout="centered"
)

# ===== Background Image CSS =====
st.markdown("""
    <style>
        body {
            background-image: url('https://images.unsplash.com/photo-1588776814546-ec7d2b3896f7?auto=format&fit=crop&w=1350&q=80');
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }
        .main, .block-container {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 2px 2px 20px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ===== Class Names and Links =====
disease_links = {
    'CaS': "https://www.webmd.com/oral-health/guide/canker-sores",
    'CoS': "https://www.webmd.com/oral-health/what-is-cold-sore",
    'Gum': "https://www.webmd.com/oral-health/guide/gum-disease",
    'MC': "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7151223/",
    'OC': "https://www.webmd.com/oral-health/guide/oral-thrush",
    'OLP': "https://www.mayoclinic.org/diseases-conditions/oral-lichen-planus",
    'OT': "https://www.cdc.gov/cancer/oral/basic_info/index.htm"
}
class_names = list(disease_links.keys())

# ===== Title =====
st.markdown('<h1 style="text-align:center; color:#003366;">🦷 AI Teeth Disease Diagnosis</h1>', unsafe_allow_html=True)
st.write("Upload your teeth image to receive a quick AI-powered diagnosis.")

# ===== Patient Name =====
patient_name = st.text_input("👤 Patient Name:")
if patient_name:
    st.success(f"👋 Welcome, {patient_name}!")

# ===== Load Model =====
@st.cache_resource
def load_model():
    model_path = "SavedModel_format"
    if not os.path.exists(model_path):
        return None
    return tf.saved_model.load(model_path)

model = load_model()

# ===== Generate PDF Report =====
def generate_pdf(name, disease, confidence):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 760, "🦷 AI Teeth Diagnosis Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 720, f"Patient Name: {name}")
    c.drawString(100, 700, f"Predicted Disease: {disease}")
    c.drawString(100, 680, f"Confidence: {confidence:.2f}%")
    c.drawString(100, 640, f"More Info: {disease_links.get(disease, 'N/A')}")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ===== Upload Image =====
uploaded_file = st.file_uploader("📤 Upload a teeth image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and patient_name:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼 Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    if model is None:
        st.error("❌ Model not loaded!")
    else:
        infer = model.signatures["serving_default"]
        input_tensor = tf.convert_to_tensor(img_array)
        predictions = infer(input_tensor)
        key = list(predictions.keys())[0]
        preds = predictions[key].numpy()[0]

        pred_index = int(np.argmax(preds))
        pred_class = class_names[pred_index]
        confidence = float(np.max(preds)) * 100

        st.markdown(f"### ✅ Prediction for {patient_name}: **{pred_class}**")
        st.markdown(f"**📊 Confidence:** {confidence:.2f}%")

        pdf = generate_pdf(patient_name, pred_class, confidence)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf,
            file_name=f"{patient_name}_teeth_report.pdf",
            mime="application/pdf"
        )

        st.markdown(f"[🌐 Learn more about {pred_class}]({disease_links[pred_class]})")

        st.markdown("### 🔎 Other possible conditions:")
        for i, prob in enumerate(preds):
            if i != pred_index:
                percent = prob * 100
                st.markdown(f"• {class_names[i]} — {percent:.2f}%")

elif uploaded_file is not None and not patient_name:
    st.warning("⚠️ Please enter your name before uploading an image.")

# ===== Footer =====
st.markdown('<div style="text-align:center; margin-top: 40px; color: #aaa;">© 2025 Dental AI Assistant | Made with ❤️ by Reem</div>', unsafe_allow_html=True)
