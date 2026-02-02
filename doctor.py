# =============================
# IMPORTS
# =============================
import streamlit as st
import json
import os
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# =============================
# PATHS
# =============================
CASES_FILE = "cases/cases.json"
FINAL_REPORTS_DIR = "cases/final_reports"

os.makedirs("cases", exist_ok=True)
os.makedirs(FINAL_REPORTS_DIR, exist_ok=True)

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Doctor Review Panel",
    #page_icon="👨‍⚕️",
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

set_bg("bg1.jpg")

st.title("Doctor Review Dashboard")

# =============================
# HELPERS
# =============================
def load_cases_safe():
    try:
        with open(CASES_FILE, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def generate_final_pdf(case, out_path):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    e = []

    # ---- HEADER ----
    e.append(Paragraph("Final Pneumonia Diagnosis Report", styles["Title"]))
    e.append(Spacer(1, 12))
    e.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        styles["Normal"]
    ))
    e.append(Spacer(1, 16))

    # ---- AI SECTION ----
    e.append(Paragraph("AI Analysis", styles["Heading2"]))
    e.append(Paragraph(f"AI Prediction: {case['ai_prediction']}", styles["Normal"]))
    e.append(Paragraph(f"Confidence: {case['confidence']}%", styles["Normal"]))
    e.append(Spacer(1, 12))

    e.append(Paragraph("Class Probabilities", styles["Heading3"]))
    for cls, p in case["class_probabilities"].items():
        e.append(Paragraph(f"{cls}: {p}%", styles["Normal"]))

    e.append(Spacer(1, 20))

    # ---- IMAGES ----
    if "xray_image_path" in case and os.path.exists(case["xray_image_path"]):
        e.append(Paragraph("Chest X-ray", styles["Heading3"]))
        e.append(Spacer(1, 8))
        e.append(RLImage(case["xray_image_path"], 4*inch, 4*inch))
        e.append(Spacer(1, 16))

    if "gradcam_image_path" in case and os.path.exists(case["gradcam_image_path"]):
        e.append(Paragraph("Grad-CAM Heatmap", styles["Heading3"]))
        e.append(Spacer(1, 8))
        e.append(RLImage(case["gradcam_image_path"], 4*inch, 4*inch))
        e.append(Spacer(1, 20))

    # ---- DOCTOR SECTION ----
    e.append(Paragraph("Doctor Review", styles["Heading2"]))
    e.append(Paragraph(
        f"Final Diagnosis: {case['doctor_final_diagnosis']}",
        styles["Normal"]
    ))
    e.append(Spacer(1, 8))
    e.append(Paragraph(
        f"Doctor Notes: {case['doctor_notes']}",
        styles["Normal"]
    ))

    doc.build(e)

# =============================
# LOAD CASES
# =============================
cases = load_cases_safe()
pending = [c for c in cases if c.get("doctor_status") == "pending"]

st.subheader(f" Pending Cases: {len(pending)}")

if not pending:
    st.success("No pending cases ")
    st.stop()

# =============================
# CASE SELECTION
# =============================
case_ids = [c["case_id"] for c in pending]
selected = st.selectbox("Select Case ID", case_ids)

case = next(c for c in pending if c["case_id"] == selected)

# =============================
# AI SUMMARY
# =============================
st.markdown("###  AI Summary")
st.write("**Prediction:**", case["ai_prediction"])
st.write("**Confidence:**", f"{case['confidence']}%")
st.write("**Class Probabilities:**")
st.json(case["class_probabilities"])

# =============================
# IMAGES (THIS WAS MISSING)
# =============================
st.markdown("### AI Visual Evidence")

if "xray_image_path" in case and os.path.exists(case["xray_image_path"]):
    st.image(case["xray_image_path"], caption="Chest X-ray", use_container_width=True)

if "gradcam_image_path" in case and os.path.exists(case["gradcam_image_path"]):
    st.image(case["gradcam_image_path"], caption="Grad-CAM Heatmap", use_container_width=True)

# =============================
# DOCTOR INPUT
# =============================
st.markdown("### Doctor Review")

final_dx = st.text_input(
    "Final Diagnosis",
    placeholder="e.g. Bacterial Pneumonia"
)

notes = st.text_area(
    "Doctor Notes",
    placeholder="Clinical observations, follow-up advice, medication notes..."
)

# =============================
# SUBMIT REVIEW
# =============================
if st.button("✅ Submit Review"):
    if not final_dx.strip():
        st.error("Final diagnosis cannot be empty")
        st.stop()

    for c in cases:
        if c["case_id"] == selected:
            c["doctor_final_diagnosis"] = final_dx
            c["doctor_notes"] = notes
            c["doctor_status"] = "reviewed"

            final_pdf_path = os.path.join(
                FINAL_REPORTS_DIR,
                f"{selected}_final.pdf"
            )

            generate_final_pdf(c, final_pdf_path)
            c["final_report_path"] = final_pdf_path

    with open(CASES_FILE, "w") as f:
        json.dump(cases, f, indent=4)

    st.success("✅ Review saved & final PDF generated")
    st.rerun()