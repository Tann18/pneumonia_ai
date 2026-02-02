# =============================
# IMPORTS
# =============================
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import json
import uuid
import tempfile
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# =============================
# CONSTANTS
# =============================
MODEL_PATH = "best_pneumonia_model.keras"
CASES_FILE = "cases/cases.json"
AI_REPORTS_DIR = "cases/ai_reports"
FINAL_REPORTS_DIR = "cases/final_reports"
IMAGES_DIR = "cases/images"

CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
IMG_SIZE = 224
DISPLAY_WIDTH = 520

# =============================
# FILE SETUP
# =============================
os.makedirs("cases", exist_ok=True)
os.makedirs(AI_REPORTS_DIR, exist_ok=True)
os.makedirs(FINAL_REPORTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

if not os.path.exists(CASES_FILE):
    with open(CASES_FILE, "w") as f:
        json.dump([], f)

def load_cases_safe():
    try:
        with open(CASES_FILE, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except:
        return []

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="PneumoVision",
    page_icon="🩺",
    layout="centered"
)

import base64

def set_bg(img_file):
    with open(img_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

if os.path.exists("bg.jpg"):
    set_bg("bg.jpg")

# =============================
# STYLING
# =============================
st.markdown("""
<style>
.main { background-color: #0e1117; }
.card {
    background-color: #111827;
    padding: 1.6rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
}
.result { font-size: 2rem; font-weight: 800; }
.confidence { color: #9ca3af; }
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown("<h1 style='text-align:center;'>🩺 PneumoVision</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#9ca3af;'>AI-powered Chest X-ray Analysis with Doctor Review</p>",
    unsafe_allow_html=True
)

# =============================
# LOAD MODEL  ✅ FIXED HERE ONLY
# =============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )

model = load_model()

# =============================
# GRAD-CAM
# =============================
def make_gradcam_heatmap(img_array, model, layer_name, class_index):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]

    heatmap = conv_out @ pooled[..., None]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    return heatmap.numpy()

# =============================
# AI PDF
# =============================
def generate_ai_pdf(case, out_path):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    e = []

    e.append(Paragraph("AI Pneumonia Analysis Report", styles["Title"]))
    e.append(Spacer(1, 12))
    e.append(Paragraph(case["timestamp"], styles["Normal"]))
    e.append(Spacer(1, 12))

    e.append(Paragraph(f"AI Diagnosis: {case['ai_prediction']}", styles["Heading2"]))
    e.append(Paragraph(f"Confidence: {case['confidence']}%", styles["Normal"]))
    e.append(Spacer(1, 10))

    for cls, p in case["class_probabilities"].items():
        e.append(Paragraph(f"{cls}: {p}%", styles["Normal"]))

    e.append(Spacer(1, 16))
    e.append(RLImage(case["xray_image_path"], 4*inch, 4*inch))
    e.append(Spacer(1, 12))
    e.append(RLImage(case["gradcam_image_path"], 4*inch, 4*inch))

    doc.build(e)

# =============================
# FINAL PDF
# =============================
def generate_final_pdf(case, out_path):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    e = []

    e.append(Paragraph("Final Pneumonia Diagnosis Report", styles["Title"]))
    e.append(Spacer(1, 12))
    e.append(Paragraph(case["timestamp"], styles["Normal"]))
    e.append(Spacer(1, 16))

    e.append(Paragraph("AI Analysis", styles["Heading2"]))
    e.append(Paragraph(f"Prediction: {case['ai_prediction']}", styles["Normal"]))
    e.append(Paragraph(f"Confidence: {case['confidence']}%", styles["Normal"]))
    e.append(Spacer(1, 10))

    for cls, p in case["class_probabilities"].items():
        e.append(Paragraph(f"{cls}: {p}%", styles["Normal"]))

    e.append(Spacer(1, 16))
    e.append(RLImage(case["xray_image_path"], 4*inch, 4*inch))
    e.append(Spacer(1, 12))
    e.append(RLImage(case["gradcam_image_path"], 4*inch, 4*inch))

    e.append(Spacer(1, 16))
    e.append(Paragraph("Doctor Review", styles["Heading2"]))
    e.append(Paragraph(f"Final Diagnosis: {case['doctor_final_diagnosis']}", styles["Normal"]))
    e.append(Spacer(1, 8))
    e.append(Paragraph(f"Doctor Notes: {case['doctor_notes']}", styles["Normal"]))

    doc.build(e)

# =============================
# UPLOAD & ANALYZE
# =============================
uploaded = st.file_uploader("Upload Chest X-ray", ["jpg", "png", "jpeg"])

if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        st.image(img, width=DISPLAY_WIDTH)

    if st.button("🔍 Analyze X-ray"):
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        inp = np.expand_dims(resized, 0)

        probs_np = model.predict(inp, verbose=0)[0]
        probs = [float(p) for p in probs_np]
        idx = int(np.argmax(probs))

        st.session_state.analysis = {
            "img": img,
            "probs": probs,
            "idx": idx,
            "inp": inp
        }

# =============================
# RESULTS
# =============================
if "analysis" in st.session_state:
    a = st.session_state.analysis

    prediction = CLASS_NAMES[a["idx"]]
    confidence = a["probs"][a["idx"]] * 100

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<p class='result'>{prediction}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='confidence'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
    st.progress(int(confidence))
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("📊 Class Probabilities")
    for cls, p in zip(CLASS_NAMES, a["probs"]):
        pct = p * 100
        st.write(f"**{cls}: {pct:.2f}%**")
        st.progress(int(pct))

    heatmap = make_gradcam_heatmap(a["inp"], model, "conv5_block3_out", a["idx"])
    heatmap = cv2.resize(heatmap, (a["img"].shape[1], a["img"].shape[0]))

    overlay = cv2.addWeighted(
        a["img"], 0.6,
        cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET),
        0.4, 0
    )

    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        st.image(overlay, width=DISPLAY_WIDTH)

    if st.button("📤 Send to Doctor"):
        case_id = str(uuid.uuid4())[:8]

        xray_path = os.path.join(IMAGES_DIR, f"{case_id}_xray.png")
        gradcam_path = os.path.join(IMAGES_DIR, f"{case_id}_gradcam.png")

        cv2.imwrite(xray_path, cv2.cvtColor(a["img"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        case = {
            "case_id": case_id,
            "timestamp": datetime.now().strftime("%d %B %Y, %H:%M"),
            "ai_prediction": prediction,
            "confidence": float(confidence),
            "class_probabilities": {
                CLASS_NAMES[i]: float(a["probs"][i] * 100) for i in range(3)
            },
            "xray_image_path": xray_path,
            "gradcam_image_path": gradcam_path,
            "doctor_status": "pending",
            "doctor_final_diagnosis": "",
            "doctor_notes": ""
        }

        ai_pdf_path = os.path.join(AI_REPORTS_DIR, f"{case_id}_ai.pdf")
        generate_ai_pdf(case, ai_pdf_path)
        case["ai_report_path"] = ai_pdf_path

        cases = load_cases_safe()
        cases.append(case)

        with open(CASES_FILE, "w") as f:
            json.dump(cases, f, indent=4)

        st.success(f"✅ Sent to doctor. Case ID: {case_id}")
        del st.session_state.analysis

# =============================
# PATIENT CHECK
# =============================
st.markdown("---")
st.subheader("🔍 Check Doctor Review")

cid = st.text_input("Enter Case ID")

if cid:
    cases = load_cases_safe()
    case = next((c for c in cases if c["case_id"] == cid), None)

    if not case:
        st.error("❌ Case not found")
    elif case["doctor_status"] == "pending":
        st.warning("⏳ Doctor review pending")
    else:
        st.success("✅ Reviewed")
        st.write("**Final Diagnosis:**", case["doctor_final_diagnosis"])
        st.write("**Doctor Notes:**", case["doctor_notes"])

        final_path = os.path.join(FINAL_REPORTS_DIR, f"{cid}_final.pdf")
        if not os.path.exists(final_path):
            generate_final_pdf(case, final_path)

        with open(final_path, "rb") as f:
            st.download_button(
                "📄 Download Final Report (PDF)",
                f,
                file_name=f"{cid}_final_report.pdf",
                mime="application/pdf"
            )