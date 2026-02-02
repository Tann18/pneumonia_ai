# PneumoVision

PneumoVision is a web-based application for automated analysis of chest X-ray images to assist in pneumonia detection.  
The system uses a deep learning convolutional neural network (CNN) to classify X-rays and provides visual explanations using Grad-CAM, along with structured PDF reports.  
This project is intended for educational and research purposes.

---
## Author

**Tanmay Das**  
Computer Science Undergraduate  
Project developed as part of academic and research work in applied machine learning and medical imaging.

## Overview

The application allows users to upload chest X-ray images and receive:
- A predicted diagnosis category
- Confidence scores for each class
- Visual heatmaps highlighting important regions of the X-ray
- Automatically generated AI reports
- A workflow for doctor review and final diagnosis

The application is implemented using Streamlit for the user interface and TensorFlow/Keras for model inference.

---

## Features

- Upload chest X-ray images (JPG, PNG, JPEG)
- Classification into three categories:
  - Bacterial Pneumonia
  - Viral Pneumonia
  - Normal
- Confidence scores for each prediction
- Grad-CAM based visual explanations
- Automatic PDF report generation
- Doctor review and final diagnosis workflow
- Web-based interface accessible through a browser

---

## Model Details

- Architecture: Convolutional Neural Network (ResNet-based)
- Framework: TensorFlow / Keras
- Input size: 224 × 224 RGB images
- Number of output classes: 3
- Model format: `.keras`

The model was migrated from a legacy `.h5` format to the modern `.keras` format to ensure compatibility across different systems and TensorFlow versions.

---

## Technology Stack

- Frontend / UI: Streamlit
- Machine Learning: TensorFlow, Keras
- Image Processing: OpenCV, NumPy
- Model Explainability: Grad-CAM
- Report Generation: ReportLab (PDF)
- Deployment: Local system, GitHub, Streamlit Cloud

---

## Project Structure
pneumovision/
├── app.py
├── best_pneumonia_model.keras
├── requirements.txt
├── bg.jpg
├── README.md
├── cases/
│ ├── images/
│ ├── ai_reports/
│ ├── final_reports/
│ └── cases.json



---


## Installation and Setup (Local)


### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/pneumovision.git
cd pneumovision
2. Create a virtual environment
python -m venv venv

Activate the virtual environment:

Windows:

venv\Scripts\activate

macOS / Linux:

source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Run the application
streamlit run app.py

The application will be available at:

http://localhost:8501
Online Deployment

The application can be deployed online using Streamlit Cloud.

Steps:

Push this repository to GitHub

Visit https://share.streamlit.io

Create a new app and select this repository

Set app.py as the entry point

Deploy

This provides a public URL that can be accessed from any device with a web browser.

Disclaimer

This project is intended strictly for educational and research purposes.
It is not a certified medical device and must not be used for clinical diagnosis or treatment decisions.

Limitations

CPU-based inference only

Performance depends on image quality

Not clinically validated

License

This project is released for academic and educational use.
If used in research or demonstrations, appropriate credit is expected.

Acknowledgements

TensorFlow and Keras

Streamlit

Open-source medical imaging datasets

Academic research in AI-assisted medical imaging
