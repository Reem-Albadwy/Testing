import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from datetime import datetime

# ✅ إعدادات الصفحة (لازم تكون أول حاجة)
st.set_page_config(
    page_title="Teeth Classifier",
    page_icon="🦷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ✅ خلفية مخصصة
def set_background(image_path):
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: #ffffff;
    }}
    .report-container {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 15px;
        color: #000000;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("background.png")

# ✅ تحميل الموديل
@st.cache_resource
def load_model():
    model_path = "SavedModel_format"
    return tf.saved_model.load(model_path)

model = load_model()
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# ✅ واجهة ترحيب
st.markdown("<h2 style='text-align: center;'>🦷 Welcome to the Teeth Disease Classifier</h2>", unsafe_allow_html=True)
patient_name = st.text_input("Enter Patient's Name:", "")

# ✅ رفع صورة
uploaded_file = st.file_uploader("Upload a teeth image", type=["jpg", "jpeg", "png"])

# ✅ لما تترفع صورة
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # تجهيز الصورة
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # توقع
    infer = model.signatures["serving_default"]
    input_tensor = tf.convert_to_tensor(img_array)
    predictions = infer(input_tensor)
    key = list(predictions.keys())[0]
    preds = predictions[key].numpy().flatten()

    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds)) * 100

    # ✅ تقرير التوقع
    with st.container():
        st.markdown("<div class='report-container'>", unsafe_allow_html=True)
        st.subheader(f"🩺 Predicted Disease: `{pred_class}`")
        st.write(f"📊 Confidence: **{confidence:.2f}%**")

        # ✅ نسب باقي الأمراض
        st.markdown("### Full Class Probabilities:")
        for i, cls in enumerate(class_names):
            st.write(f"{cls}: **{preds[i]*100:.2f}%**")
        st.markdown("</div>", unsafe_allow_html=True)

    # ✅ رابط خارجي للمعلومة
    st.markdown(f"""
        <a href="https://www.google.com/search?q={pred_class}+oral+disease" target="_blank">
            🔎 Learn more about {pred_class}
        </a>
    """, unsafe_allow_html=True)

    # ✅ زر تنزيل التقرير
    def generate_pdf_report(name, prediction, confidence):
        filename = "report.pdf"
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(width / 2.0, height - 80, "🦷 Teeth Disease Diagnosis Report")

        c.setFont("Helvetica", 14)
        c.drawString(50, height - 130, f"👤 Patient Name: {name}")
        c.drawString(50, height - 160, f"🩺 Predicted Disease: {prediction}")
        c.drawString(50, height - 190, f"📊 Confidence: {confidence:.2f}%")

        c.drawString(50, height - 230, "📅 Date: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        c.save()
        return filename

    if st.button("⬇️ Download PDF Report"):
        report_file = generate_pdf_report(patient_name or "Unknown", pred_class, confidence)
        with open(report_file, "rb") as f:
            st.download_button(
                label="📄 Click to Download",
                data=f,
                file_name="teeth_diagnosis_report.pdf",
                mime="application/pdf"
            )
