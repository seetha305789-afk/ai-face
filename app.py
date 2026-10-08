import streamlit as st
import numpy as np
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Face Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LIGHT PROFESSIONAL THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f5f8fc;
    color: #172033;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.stApp p,
.stApp span,
.stApp label,
.stApp li,
.stApp small {
    color: #172033 !important;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #102a43 !important;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #123b68 !important;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #486581 !important;
    margin-bottom: 30px;
}

.section-title {
    font-size: 28px;
    font-weight: 750;
    color: #123b68 !important;
    margin-top: 15px;
    margin-bottom: 15px;
}

.tech-card {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-radius: 16px;
    padding: 22px;
    min-height: 180px;
    box-shadow: 0 5px 18px rgba(31, 45, 61, 0.08);
    transition: 0.2s ease;
}

.tech-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 25px rgba(31, 45, 61, 0.14);
}

.tech-icon {
    font-size: 35px;
}

.tech-title {
    font-size: 20px;
    font-weight: 700;
    color: #123b68 !important;
    margin-top: 8px;
    margin-bottom: 8px;
}

.tech-text {
    font-size: 14px;
    color: #486581 !important;
    line-height: 1.6;
}

.result-card {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-radius: 14px;
    padding: 18px;
    margin-top: 12px;
    margin-bottom: 12px;
    box-shadow: 0 4px 15px rgba(31, 45, 61, 0.07);
}

.info-box {
    background: #eaf4ff;
    border-left: 5px solid #1976d2;
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
    color: #17324d !important;
}

.info-box * {
    color: #17324d !important;
}

.success-box {
    background: #eafaf1;
    border-left: 5px solid #16a34a;
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
}

.success-box * {
    color: #14532d !important;
}

.stButton > button {
    width: 100%;
    border-radius: 9px;
    border: none;
    background: #1976d2;
    color: white !important;
    font-weight: 700;
    padding: 10px;
    box-shadow: 0 4px 10px rgba(25, 118, 210, 0.18);
}

.stButton > button:hover {
    background: #125ea8;
    color: white !important;
}

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #d9e2ec;
}

section[data-testid="stSidebar"] * {
    color: #172033 !important;
}

.sidebar-title {
    font-size: 24px;
    font-weight: 800;
    color: #123b68 !important;
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stSelectbox"] {
    background: #ffffff;
    border-radius: 10px;
}

[data-baseweb="select"] * {
    color: #172033 !important;
}

button[data-baseweb="tab"] {
    color: #486581 !important;
    font-weight: 700;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #1976d2 !important;
}

[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-radius: 12px;
}

[data-testid="stExpander"] * {
    color: #172033 !important;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stMetricLabel"] * {
    color: #486581 !important;
}

[data-testid="stMetricValue"] * {
    color: #123b68 !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
    color: #627d98 !important;
}

hr {
    border-color: #d9e2ec;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def image_to_array(image_file):
    image = Image.open(image_file).convert("RGB")
    return np.array(image)


def cosine_similarity(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# ============================================================
# VIOLA-JONES MODEL
# ============================================================

@st.cache_resource
def load_haar_cascade():

    # IMPORTANT:
    # OpenCV is imported only when Viola-Jones is used.
    import cv2

    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            "OpenCV installation problem: "
            "CascadeClassifier is not available."
        )

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        raise RuntimeError(
            "Could not load Viola-Jones Haar Cascade."
        )

    return cascade


# ============================================================
# FACENET MODEL
# ============================================================

@st.cache_resource
def load_facenet():

    from keras_facenet import FaceNet

    return FaceNet()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI Face Detection & Recognition Studio</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Template Matching • Viola-Jones • DeepFace • FaceNet'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">⚙️ Control Panel</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.info(
        """
        **IVA Assignment**

        This application demonstrates four
        important Image & Video Analytics
        techniques:

        • Template Matching  
        • Viola-Jones  
        • DeepFace  
        • FaceNet
        """
    )

    st.markdown("---")

    st.write("### 📌 Quick Navigation")

    st.write("🏠 Overview")
    st.write("🎯 Template Matching")
    st.write("👁️ Viola-Jones")
    st.write("🧠 DeepFace")
    st.write("🔐 FaceNet")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "🎯 Template Matching",
    "👁️ Viola-Jones",
    "🧠 DeepFace",
    "🔐 FaceNet"
])


# ============================================================
# TAB 1 - OVERVIEW
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">📚 Image & Video Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <b>Image and Video Analytics (IVA)</b> uses computer vision
        and artificial intelligence techniques to understand
        images and videos automatically.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🔬 Technologies Used")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-icon">🎯</div>
                <div class="tech-title">Template Matching</div>
                <div class="tech-text">
                    Finds a small template image inside a larger image
                    using OpenCV template matching.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-icon">👁️</div>
                <div class="tech-title">Viola-Jones</div>
                <div class="tech-text">
                    A classical object detection algorithm that uses
                    Haar-like features and a cascade classifier.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-icon">🧠</div>
                <div class="tech-title">DeepFace</div>
                <div class="tech-text">
                    A deep-learning based facial analysis framework
                    for face recognition and attribute analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-icon">🔐</div>
                <div class="tech-title">FaceNet</div>
                <div class="tech-text">
                    Converts faces into numerical embeddings that
                    can be compared for face similarity.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### 📊 Application Areas")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Technology", "Computer Vision")
    c2.metric("Domain", "IVA")
    c3.metric("Recognition", "Face AI")
    c4.metric("Platform", "Python")


# ============================================================
# TAB 2 - TEMPLATE MATCHING
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">🎯 Template Matching</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <b>Template Matching</b> searches for a smaller image
        (template) inside a larger image.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🖼️ Upload Main Image")

        main_file = st.file_uploader(
            "Choose main image",
            type=["jpg", "jpeg", "png"],
            key="template_main"
        )

    with col2:

        st.markdown("### 🔎 Upload Template")

        template_file = st.file_uploader(
            "Choose template image",
            type=["jpg", "jpeg", "png"],
            key="template_file"
        )

    if main_file and template_file:

        main_image = image_to_array(main_file)
        template_image = image_to_array(template_file)

        st.markdown("### 📷 Input Images")

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                main_image,
                caption="Main Image",
                width="stretch"
            )

        with c2:
            st.image(
                template_image,
                caption="Template Image",
                width="stretch"
            )

        if st.button(
            "🔍 Find Template",
            key="template_button"
        ):

            try:

                import cv2

                main_gray = cv2.cvtColor(
                    main_image,
                    cv2.COLOR_RGB2GRAY
                )

                template_gray = cv2.cvtColor(
                    template_image,
                    cv2.COLOR_RGB2GRAY
                )

                th, tw = template_gray.shape[:2]

                if (
                    th > main_gray.shape[0]
                    or tw > main_gray.shape[1]
                ):
                    st.error(
                        "Template image must be smaller than the main image."
                    )

                else:

                    result = cv2.matchTemplate(
                        main_gray,
                        template_gray,
                        cv2.TM_CCOEFF_NORMED
                    )

                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    top_left = max_loc

                    bottom_right = (
                        top_left[0] + tw,
                        top_left[1] + th
                    )

                    output = main_image.copy()

                    cv2.rectangle(
                        output,
                        top_left,
                        bottom_right,
                        (255, 0, 0),
                        3
                    )

                    st.markdown(
                        '<div class="success-box">'
                        '<b>Template Found Successfully!</b>'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.image(
                        output,
                        caption="Template Matching Result",
                        width="stretch"
                    )

                    st.metric(
                        "Matching Score",
                        f"{max_val:.2%}"
                    )

            except Exception as e:

                st.error(
                    f"Template Matching failed: {e}"
                )


# ============================================================
# TAB 3 - VIOLA-JONES
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">👁️ Viola-Jones Face Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <b>Viola-Jones</b> is a classical real-time object detection
        algorithm commonly used for face detection.
        </div>
        """,
        unsafe_allow_html=True
    )

    image_file = st.file_uploader(
        "📤 Upload an image",
        type=["jpg", "jpeg", "png"],
        key="viola_image"
    )

    if image_file:

        image = image_to_array(image_file)

        st.image(
            image,
            caption="Original Image",
            width="stretch"
        )

        if st.button(
            "👁️ Detect Faces",
            key="viola_button"
        ):

            try:

                import cv2

                cascade = load_haar_cascade()

                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2GRAY
                )

                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                result = image.copy()

                for (x, y, w, h) in faces:

                    cv2.rectangle(
                        result,
                        (x, y),
                        (x + w, y + h),
                        (0, 200, 0),
                        3
                    )

                    cv2.putText(
                        result,
                        "Face",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 200, 0),
                        2
                    )

                if len(faces) > 0:

                    st.markdown(
                        f"""
                        <div class="success-box">
                        <b>{len(faces)} face(s) detected!</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.warning("No face detected.")

                st.image(
                    result,
                    caption="Viola-Jones Detection Result",
                    width="stretch"
                )

            except Exception as e:

                st.error(
                    f"Viola-Jones could not start: {e}"
                )


# ============================================================
# TAB 4 - DEEPFACE
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">🧠 DeepFace</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <b>DeepFace</b> is a deep-learning framework for
        facial analysis. It can perform face recognition,
        age, gender and emotion analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    deepface_file = st.file_uploader(
        "📤 Upload a face image",
        type=["jpg", "jpeg", "png"],
        key="deepface_image"
    )

    if deepface_file:

        image = image_to_array(deepface_file)

        st.image(
            image,
            caption="Uploaded Face",
            width="stretch"
        )

        if st.button(
            "🧠 Analyze Face",
            key="deepface_button"
        ):

            try:

                from deepface import DeepFace
                import cv2

                image_bytes = deepface_file.getvalue()

                img_array = np.frombuffer(
                    image_bytes,
                    np.uint8
                )

                bgr_image = cv2.imdecode(
                    img_array,
                    cv2.IMREAD_COLOR
                )

                with st.spinner(
                    "DeepFace is analyzing the image..."
                ):

                    result = DeepFace.analyze(
                        img_path=bgr_image,
                        actions=[
                            "age",
                            "gender",
                            "emotion"
                        ],
                        enforce_detection=False
                    )

                if isinstance(result, list):
                    analysis = result[0]
                else:
                    analysis = result

                st.markdown("### 📊 Analysis Result")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Age",
                        str(analysis.get("age", "N/A"))
                    )

                with c2:
                    st.metric(
                        "Gender",
                        str(
                            analysis.get(
                                "dominant_gender",
                                "N/A"
                            )
                        )
                    )

                with c3:
                    st.metric(
                        "Emotion",
                        str(
                            analysis.get(
                                "dominant_emotion",
                                "N/A"
                            )
                        )
                    )

                with st.expander(
                    "📋 View Complete Analysis"
                ):

                    st.json(analysis)

            except Exception as e:

                st.error(
                    f"DeepFace analysis failed: {e}"
                )


# ============================================================
# TAB 5 - FACENET
# ============================================================

with tab5:

    st.markdown(
        '<div class="section-title">🔐 FaceNet</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <b>FaceNet</b> converts a face into a numerical vector
        called an embedding. Two embeddings can be compared
        using similarity.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        face1 = st.file_uploader(
            "📷 Upload Face 1",
            type=["jpg", "jpeg", "png"],
            key="facenet_face1"
        )

    with col2:

        face2 = st.file_uploader(
            "📷 Upload Face 2",
            type=["jpg", "jpeg", "png"],
            key="facenet_face2"
        )

    if face1 and face2:

        image1 = image_to_array(face1)
        image2 = image_to_array(face2)

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                image1,
                caption="Face 1",
                width="stretch"
            )

        with c2:
            st.image(
                image2,
                caption="Face 2",
                width="stretch"
            )

        if st.button(
            "🔐 Compare Faces",
            key="facenet_button"
        ):

            try:

                with st.spinner(
                    "FaceNet is creating embeddings..."
                ):

                    embedder = load_facenet()

                    img1 = image1.astype(np.uint8)
                    img2 = image2.astype(np.uint8)

                    emb1 = embedder.embeddings(
                        np.expand_dims(img1, axis=0)
                    )[0]

                    emb2 = embedder.embeddings(
                        np.expand_dims(img2, axis=0)
                    )[0]

                    similarity = cosine_similarity(
                        emb1,
                        emb2
                    )

                st.markdown(
                    '<div class="success-box">'
                    '<b>Face comparison completed!</b>'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.metric(
                    "Cosine Similarity",
                    f"{similarity:.4f}"
                )

                if similarity >= 0.70:

                    st.success(
                        "😊 The faces are relatively similar."
                    )

                else:

                    st.warning(
                        "🙂 The faces appear less similar."
                    )

            except Exception as e:

                st.error(
                    f"FaceNet comparison failed: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#627d98;
        font-size:14px;
        padding:15px;
    ">
        🤖 AI Face Detection & Recognition Studio
        <br>
        Image & Video Analytics Assignment
    </div>
    """,
    unsafe_allow_html=True
)