import streamlit as st
import numpy as np
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Face Studio",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

h1 {
    color: #222222;
}

h2, h3 {
    color: #333333;
}

.stButton > button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
}

.analysis-box {
    background-color: white;
    color: #222222;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dddddd;
    margin-top: 15px;
}

.analysis-box h3 {
    color: #222222 !important;
}

.analysis-item {
    color: #222222 !important;
    font-size: 16px;
    margin: 7px 0;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def image_to_array(image):
    return np.array(image)


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def load_haar_cascade():
    try:
        import cv2

        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                "OpenCV installation problem: "
                "CascadeClassifier is not available."
            )

        cascade_path = cv2.data.haarcascades + \
            "haarcascade_frontalface_default.xml"

        cascade = cv2.CascadeClassifier(cascade_path)

        if cascade.empty():
            raise RuntimeError(
                "Haar Cascade file could not be loaded."
            )

        return cv2, cascade

    except Exception as e:
        raise RuntimeError(str(e))


def load_facenet():
    from keras_facenet import FaceNet

    return FaceNet()


def make_json_safe(obj):

    if isinstance(obj, dict):
        return {
            str(key): make_json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            make_json_safe(value)
            for value in obj
        ]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    return obj


# =========================================================
# HEADER
# =========================================================

st.title("🤖 AI Face Detection & Recognition Studio")

st.write(
    "Template Matching • Viola-Jones • DeepFace • FaceNet"
)

st.divider()

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "🎯 Template Matching",
    "👁️ Viola-Jones",
    "🧠 DeepFace",
    "🔍 FaceNet"
])


# =========================================================
# TAB 1 - OVERVIEW
# =========================================================

with tab1:

    st.header("AI Face Detection & Recognition")

    st.write(
        """
        This application demonstrates different Image and Video
        Analytics techniques for face detection and recognition.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎯 Template Matching")

        st.write(
            """
            Template Matching compares a template image with
            another image to find similar regions.
            """
        )

    with col2:

        st.subheader("👁️ Viola-Jones")

        st.write(
            """
            Viola-Jones is a classical object detection algorithm
            commonly used for face detection.
            """
        )

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("🧠 DeepFace")

        st.write(
            """
            DeepFace can analyze faces for age, gender and emotion.
            """
        )

    with col4:

        st.subheader("🔍 FaceNet")

        st.write(
            """
            FaceNet generates face embeddings and can be used
            for face similarity and recognition.
            """
        )


# =========================================================
# TAB 2 - TEMPLATE MATCHING
# =========================================================

with tab2:

    st.header("🎯 Template Matching")

    st.write(
        "Upload a main image and a template image."
    )

    image_file = st.file_uploader(
        "Upload Main Image",
        type=["jpg", "jpeg", "png"],
        key="template_main"
    )

    template_file = st.file_uploader(
        "Upload Template Image",
        type=["jpg", "jpeg", "png"],
        key="template"
    )

    if image_file and template_file:

        main_image = Image.open(image_file)
        template_image = Image.open(template_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                main_image,
                caption="Main Image",
                width="stretch"
            )

        with col2:
            st.image(
                template_image,
                caption="Template Image",
                width="stretch"
            )

        if st.button(
            "Run Template Matching",
            key="template_button"
        ):

            try:

                import cv2

                main_array = image_to_array(
                    main_image.convert("RGB")
                )

                template_array = image_to_array(
                    template_image.convert("RGB")
                )

                main_gray = cv2.cvtColor(
                    main_array,
                    cv2.COLOR_RGB2GRAY
                )

                template_gray = cv2.cvtColor(
                    template_array,
                    cv2.COLOR_RGB2GRAY
                )

                result = cv2.matchTemplate(
                    main_gray,
                    template_gray,
                    cv2.TM_CCOEFF_NORMED
                )

                _, max_val, _, max_loc = cv2.minMaxLoc(
                    result
                )

                st.success(
                    f"Template matching completed!"
                )

                st.metric(
                    "Similarity Score",
                    f"{max_val * 100:.2f}%"
                )

                st.write(
                    f"Best Match Location: {max_loc}"
                )

            except Exception as e:

                st.error(
                    f"Template Matching failed: {e}"
                )


# =========================================================
# TAB 3 - VIOLA-JONES
# =========================================================

with tab3:

    st.header("👁️ Viola-Jones Face Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="viola_image"
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width="stretch"
        )

        if st.button(
            "Detect Faces",
            key="viola_button"
        ):

            try:

                cv2, face_cascade = load_haar_cascade()

                image_array = np.array(image)

                gray = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2GRAY
                )

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                result_image = image_array.copy()

                for (x, y, w, h) in faces:

                    cv2.rectangle(
                        result_image,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                if len(faces) > 0:

                    st.success(
                        f"{len(faces)} face(s) detected!"
                    )

                else:

                    st.warning(
                        "No face detected."
                    )

                st.image(
                    result_image,
                    caption="Viola-Jones Result",
                    width="stretch"
                )

            except Exception as e:

                st.error(
                    f"Viola-Jones could not start: {e}"
                )


# =========================================================
# TAB 4 - DEEPFACE
# =========================================================

with tab4:

    st.header("🧠 DeepFace Analysis")

    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"],
        key="deepface_image"
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Face",
            width="stretch"
        )

        if st.button(
            "Analyze Face",
            key="deepface_button"
        ):

            try:

                from deepface import DeepFace
                import cv2

                rgb_image = np.array(image)

                bgr_image = cv2.cvtColor(
                    rgb_image,
                    cv2.COLOR_RGB2BGR
                )

                analysis = DeepFace.analyze(
                    img_path=bgr_image,
                    actions=[
                        "age",
                        "gender",
                        "emotion"
                    ],
                    enforce_detection=False
                )

                if isinstance(analysis, list):

                    analysis = analysis[0]

                analysis = make_json_safe(analysis)

                # -----------------------------------------
                # MAIN RESULTS
                # -----------------------------------------

                st.subheader("📊 Analysis Result")

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Age",
                        analysis.get("age", "N/A")
                    )

                with col2:

                    st.metric(
                        "Gender",
                        analysis.get(
                            "dominant_gender",
                            "N/A"
                        )
                    )

                with col3:

                    st.metric(
                        "Emotion",
                        analysis.get(
                            "dominant_emotion",
                            "N/A"
                        ).capitalize()
                    )

                # -----------------------------------------
                # COMPLETE ANALYSIS
                # -----------------------------------------

                if st.button(
                    "View Complete Analysis",
                    key="complete_analysis"
                ):

                    clean_analysis = make_json_safe(
                        analysis
                    )

                    st.markdown("""
                    <style>

                    .analysis-box {
                        background-color: white;
                        color: #222222;
                        padding: 20px;
                        border-radius: 12px;
                        border: 1px solid #dddddd;
                        margin-top: 15px;
                    }

                    .analysis-box h3 {
                        color: #222222 !important;
                    }

                    .analysis-item {
                        color: #222222 !important;
                        font-size: 16px;
                        margin: 7px 0;
                    }

                    </style>
                    """, unsafe_allow_html=True)

                    st.markdown(
                        '<div class="analysis-box">',
                        unsafe_allow_html=True
                    )

                    # -------------------------------------
                    # FACE DETAILS
                    # -------------------------------------

                    st.markdown(
                        "### 👤 Face Details"
                    )

                    st.markdown(
                        f"""
                        <div class="analysis-item">
                        <b>Age:</b>
                        {clean_analysis.get("age", "N/A")}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    confidence = clean_analysis.get(
                        "face_confidence",
                        0
                    )

                    st.markdown(
                        f"""
                        <div class="analysis-item">
                        <b>Face Confidence:</b>
                        {confidence * 100:.2f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class="analysis-item">
                        <b>Dominant Gender:</b>
                        {clean_analysis.get(
                            "dominant_gender",
                            "N/A"
                        )}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class="analysis-item">
                        <b>Dominant Emotion:</b>
                        {clean_analysis.get(
                            "dominant_emotion",
                            "N/A"
                        ).capitalize()}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # -------------------------------------
                    # GENDER SCORES
                    # -------------------------------------

                    gender = clean_analysis.get(
                        "gender",
                        {}
                    )

                    st.markdown(
                        "### 🚻 Gender Scores"
                    )

                    for name, score in gender.items():

                        st.markdown(
                            f"""
                            <div class="analysis-item">
                            <b>{name}:</b>
                            {score:.2f}%
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # -------------------------------------
                    # EMOTION SCORES
                    # -------------------------------------

                    emotion = clean_analysis.get(
                        "emotion",
                        {}
                    )

                    st.markdown(
                        "### 😊 Emotion Scores"
                    )

                    for name, score in emotion.items():

                        st.markdown(
                            f"""
                            <div class="analysis-item">
                            <b>{name.capitalize()}:</b>
                            {score:.2f}%
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # -------------------------------------
                    # FACE REGION
                    # -------------------------------------

                    region = clean_analysis.get(
                        "region",
                        {}
                    )

                    st.markdown(
                        "### 📍 Face Region"
                    )

                    st.markdown(
                        f"""
                        <div class="analysis-item">
                        <b>X:</b>
                        {region.get("x", "N/A")}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class="analysis-item">
                        <b>Y:</b>
                        {region.get("y", "N/A")}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class="analysis-item">
                        <b>Width:</b>
                        {region.get("w", "N/A")}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class="analysis-item">
                        <b>Height:</b>
                        {region.get("h", "N/A")}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

            except Exception as e:

                st.error(
                    f"DeepFace analysis failed: {e}"
                )


# =========================================================
# TAB 5 - FACENET
# =========================================================

with tab5:

    st.header("🔍 FaceNet Face Similarity")

    col1, col2 = st.columns(2)

    with col1:

        image1_file = st.file_uploader(
            "Upload First Face",
            type=["jpg", "jpeg", "png"],
            key="facenet_image1"
        )

    with col2:

        image2_file = st.file_uploader(
            "Upload Second Face",
            type=["jpg", "jpeg", "png"],
            key="facenet_image2"
        )

    if image1_file and image2_file:

        image1 = Image.open(
            image1_file
        ).convert("RGB")

        image2 = Image.open(
            image2_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image1,
                caption="First Face",
                width="stretch"
            )

        with col2:

            st.image(
                image2,
                caption="Second Face",
                width="stretch"
            )

        if st.button(
            "Compare Faces",
            key="facenet_button"
        ):

            try:

                embedder = load_facenet()

                img1_array = np.array(image1)
                img2_array = np.array(image2)

                embedding1 = embedder.embeddings(
                    [img1_array]
                )[0]

                embedding2 = embedder.embeddings(
                    [img2_array]
                )[0]

                similarity = cosine_similarity(
                    embedding1,
                    embedding2
                )

                st.metric(
                    "Cosine Similarity",
                    f"{similarity:.4f}"
                )

                if similarity >= 0.70:

                    st.success(
                        "High similarity - Faces are likely the same."
                    )

                else:

                    st.warning(
                        "Low similarity - Faces are likely different."
                    )

            except Exception as e:

                st.error(
                    f"FaceNet comparison failed: {e}"
                )