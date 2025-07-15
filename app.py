import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ===== Page Configuration =====
st.set_page_config(
    page_title="🦷 AI Teeth Disease Diagnosis",
    page_icon="🦷",
    layout="centered"
)

# ===== Custom CSS =====
st.markdown("""
    <style>
        body {
            background-image: url("https://img.freepik.com/free-vector/abstract-soft-pastel-background_23-2148923270.jpg");
            background-size: cover;
            background-attachment: fixed;
        }
        .main-title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #1450A3;
            margin-bottom: 10px;
        }
        .predict-card {
            background-color: #E8F0FEcc;
            padding: 25px;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 1px 1px 10px rgba(0,0,0,0.1);
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            font-size: 13px;
            color: #888;
        }
        .other-classes {
            font-size: 16px;
            line-height: 1.6;
            color: #333;
            padding-left: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# ===== Disease info links =====
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
st.markdown('<div class="main-title">🦷 AI Teeth Disease Diagnosis</div>', unsafe_allow_html=True)
st.write("Upload your teeth image and get an instant AI diagnosis — fast, simple, and reliable.")

# ===== Patient Name Input =====
st.markdown("### 🙋‍♀️ Enter your name to personalize your report:")
patient_name = st.text_input("Patient Name:", max_chars=30)

if patient_name:
    st.success(f"👋 Welcome, {patient_name}!")

# ===== Load model =====
@st.cache_resource
def load_model():
    model_path = "SavedModel_format"
    if not os.path.exists(model_path):
        return None
    return tf.saved_model.load(model_path)

model = load_model()

# ===== PDF Report Generator =====
def generate_pdf(name, disease, confidence):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 760, "🦷 AI Teeth Disease Diagnosis Report")
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
uploaded_file = st.file_uploader("📤 Upload a teeth image (jpg/png)", type=["jpg", "jpeg", "png"])

# ===== Prediction =====
if uploaded_file is not None and patient_name:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    if model is None:
        st.error("❌ Model not loaded. Please check your model directory.")
    else:
        infer = model.signatures["serving_default"]
        input_tensor = tf.convert_to_tensor(img_array)
        predictions = infer(input_tensor)
        key = list(predictions.keys())[0]
        preds = predictions[key].numpy()[0]

        pred_index = int(np.argmax(preds))
        pred_class = class_names[pred_index]
        confidence = float(np.max(preds)) * 100

        st.markdown(f"""
            <div class="predict-card">
                <h3>🩺 <b>Diagnosis for {patient_name}:</b> <span style="color:#1450A3;">{pred_class}</span></h3>
                <p>📊 <b>Confidence:</b> {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

        # PDF Button
        pdf = generate_pdf(patient_name, pred_class, confidence)
        st.download_button(
            label="📄 Download Diagnosis Report (PDF)",
            data=pdf,
            file_name=f"{patient_name}_Teeth_Report.pdf",
            mime="application/pdf"
        )

        # Link to learn more
        link = disease_links.get(pred_class)
        if link:
            st.markdown(f"[🌐 Learn more about {pred_class}]({link})", unsafe_allow_html=True)

        # Other predictions
        st.markdown("### 🤔 Other possible conditions:")
        alt_predictions = []
        for i, prob in enumerate(preds):
            if i == pred_index:
                continue
            percent = prob * 100
            if percent > 10:
                desc = f"▪ {class_names[i]} — <span style='color:#cc0000;'>maybe ({percent:.1f}%)</span>"
            elif percent > 3:
                desc = f"▪ {class_names[i]} — <span style='color:#999;'>unlikely ({percent:.1f}%)</span>"
            else:
                desc = f"▪ {class_names[i]} — very rare ({percent:.1f}%)"
            alt_predictions.append(desc)
        st.markdown("<div class='other-classes'>" + "<br>".join(alt_predictions) + "</div>", unsafe_allow_html=True)

elif uploaded_file is not None and not patient_name:
    st.warning("⚠️ Please enter your name before uploading an image.")

# ===== Footer =====
st.markdown('<div class="footer">© 2025 AI Dental Assistant — Built with ❤️ by Reem</div>', unsafe_allow_html=True)
