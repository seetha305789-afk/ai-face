# AI Face Detection & Recognition Studio

This is a comprehensive, single-file Streamlit web application that demonstrates four distinct computer vision techniques for face detection and recognition.

## Features
- **Template Matching**: Localize objects using OpenCV correlation.
- **Viola-Jones**: Fast, traditional face detection using Haar Cascades.
- **DeepFace**: Advanced face attribute analysis (age, gender, emotion).
- **FaceNet**: Deep learning-based face verification using embeddings.

## Technologies Used
- Python 3.10/3.11
- Streamlit (UI)
- OpenCV (Image processing & Viola-Jones)
- DeepFace (Analysis)
- Keras-FaceNet (Embeddings)

## Installation & Running
1. Open terminal in the project folder.
2. `python -m venv venv`
3. `venv\Scripts\activate`
4. `pip install -r requirements.txt`
5. `streamlit run app.py`

## Troubleshooting
- If DeepFace fails, ensure you have an active internet connection on the first run, as it automatically downloads pre-trained models.
- If you see "No Face Detected", ensure your uploaded images are high quality and faces are clearly visible.
