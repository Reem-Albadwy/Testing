import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# ========= إعداد صفحة Streamlit ========= #
st.set_page_config(
    page_title="Teeth Classifier",
    page_icon="🩇",
    layout="centered"
)

# ========= تعيين خلفية مخصصة ========= #
def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .report-container {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

set_background("background.png")

# ========= تحميل الموديل ========= #
@st.cache_resource
def load_model():
    model_path = "SavedModel_format"
    return tf.saved_model.load(model_path)

model = load_model()

class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# ========= إدخال اسم المستخدم ========= #
st.markdown("""
    <h2 style='text-align: center; color: #003366;'>🙌 Welcome to the Smart Teeth Classifier</h2>
    <p style='text-align: center; color: #222;'>Please upload a teeth image below.</p>
""", unsafe_allow_html=True)

name = st.text_input("👤 Patient Name:")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# ========= التنبؤ وعرض النتائج ========= #
if uploaded_file is not None and model is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    infer = model.signatures["serving_default"]
    input_tensor = tf.convert_to_tensor(img_array)
    predictions = infer(input_tensor)
    key = list(predictions.keys())[0]
    preds = predictions[key].numpy().flatten()

    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds)) * 100

    # ===== تقرير النتائج ===== #
    with st.container():
        st.markdown("<div class='report-container'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#0066cc;'>🚑 Prediction: {pred_class}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#333333; font-size:18px;'>📊 Confidence: <strong>{confidence:.2f}%</strong></p>", unsafe_allow_html=True)

        with st.expander("👉 Show probabilities for all classes"):
            for i, cls in enumerate(class_names):
                st.markdown(f"<span style='color:#444;'>{cls}: <strong>{preds[i]*100:.2f}%</strong></span>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ===== روابط إضافية حسب التشخيص ===== #
    st.markdown("---")
    st.markdown("### 🌐 Read more about the disease:")
    disease_links = {
        "CaS": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6422532/",
        "CoS": "https://www.ncbi.nlm.nih.gov/books/NBK470190/",
        "Gum": "https://www.webmd.com/oral-health/guide/gum-disease",
        "MC": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3783794/",
        "OC": "https://www.cdc.gov/cancer/oral/basic_info/index.htm",
        "OLP": "https://www.mayoclinic.org/diseases-conditions/oral-lichen-planus/symptoms-causes/syc-20353232",
        "OT": "https://en.wikipedia.org/wiki/Oral_thrush"
    }
    st.markdown(f"[Click here to read more about {pred_class}](" + disease_links[pred_class] + ")")

    # ===== زر تحميل تقرير PDF ===== #
    st.markdown("---")
    if st.button("📄 Download Report as PDF"):
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(100, 750, "Teeth Classification Report")
        c.setFont("Helvetica", 12)
        c.drawString(100, 720, f"Patient Name: {name if name else 'N/A'}")
        c.drawString(100, 700, f"Prediction: {pred_class}")
        c.drawString(100, 680, f"Confidence: {confidence:.2f}%")
        c.drawString(100, 650, "Probabilities:")
        for i, cls in enumerate(class_names):
            c.drawString(120, 630 - (i * 15), f"{cls}: {preds[i]*100:.2f}%")
        c.save()
        st.download_button(
            label="📄 Download PDF",
            data=pdf_buffer.getvalue(),
            file_name=f"{name}_teeth_report.pdf",
            mime='application/pdf'
        )
