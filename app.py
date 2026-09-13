from io import BytesIO

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from models.mnist_cnn_model import MnistCnnModel
from preprocessing.image_preprocessor import ImagePreprocessor
from services.prediction_service import PredictionService
from utils.evaluation import build_probability_figure


# Configure the Streamlit web page.
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            max-width: 1180px;
        }
        .title {
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 1.5rem;
        }
        .result-card {
            border: 1px solid rgba(128,128,128,.35);
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            margin-top: 8px;
        }
        .prediction-number {
            font-size: 76px;
            font-weight: 700;
            line-height: 1;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cache the trained model so it is not recreated on every page refresh.
@st.cache_resource(show_spinner=False)
# Load the saved model, or train it the first time the app runs.
def get_model():
    model = MnistCnnModel()
    model.load_or_train()
    return model


model = get_model()
preprocessor = ImagePreprocessor()
prediction_service = PredictionService(model)

st.markdown(
    '<h1 class="title">✍️ Handwritten Digit Recognition</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">MNIST + CNN model for handwritten digits 0 to 9</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Input")
    # Let the user choose between drawing a digit and uploading an image.
    input_method = st.radio(
        "Choose input method",
        ["Draw on Canvas", "Upload Image"],
        index=0,
    )
    st.divider()
    st.write("Model: **MNIST CNN**")
    st.caption("Python 3.11 recommended")
    st.caption("The first run downloads MNIST and trains the model once.")

tab_predict, tab_about = st.tabs(["Predict", "About"])

with tab_predict:
    left, right = st.columns([1, 1], gap="large")

    image_to_predict = None

    with left:
        st.subheader("1. Draw or upload a digit")

        if input_method == "Draw on Canvas":
            st.write("Draw one large white digit in the center.")

            # Create the drawing canvas for handwritten digit input.
            canvas_result = st_canvas(
                fill_color="rgba(255,255,255,0)",
                stroke_width=16,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="mnist_canvas",
            )

            if canvas_result.image_data is not None:
                image_to_predict = Image.fromarray(
                    canvas_result.image_data.astype("uint8"),
                    mode="RGBA",
                ).convert("RGB")

        else:
            # Allow the user to upload a handwritten digit image.
            uploaded_file = st.file_uploader(
                "Upload PNG, JPG, or JPEG",
                type=["png", "jpg", "jpeg"],
            )

            if uploaded_file is not None:
                image_to_predict = Image.open(
                    BytesIO(uploaded_file.getvalue())
                ).convert("RGB")
                st.image(image_to_predict, caption="Uploaded image", width=280)

    with right:
        st.subheader("2. Prediction")

        if st.button("Predict Digit", type="primary", use_container_width=True):
            if image_to_predict is None:
                st.warning("Please draw or upload a digit first.")
            else:
                # Convert the input image into the same format used by the ML model.
                processed = preprocessor.preprocess(image_to_predict)

                if processed is None:
                    st.warning("No digit was detected. Please draw a clearer digit.")
                else:
                    # Send the processed image to the prediction service.
                    result = prediction_service.predict(processed.tensor)

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div>Prediction</div>
                            <div class="prediction-number">{result.digit}</div>
                            <div>Confidence: <b>{result.confidence * 100:.2f}%</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    preview_col, chart_col = st.columns([0.8, 1.2])

                    with preview_col:
                        st.write("Processed 28×28 image")
                        fig, ax = plt.subplots(figsize=(2.7, 2.7))
                        ax.imshow(
                            processed.image_28x28,
                            cmap="gray",
                            vmin=0,
                            vmax=1,
                        )
                        ax.axis("off")
                        st.pyplot(fig, clear_figure=True)

                    with chart_col:
                        st.write("Prediction probability")
                        # Create a chart showing the probability for each digit.
                        fig = build_probability_figure(result.probabilities)
                        st.pyplot(fig, clear_figure=True)

with tab_about:
    st.subheader("About This Project")
    st.write(
        """
        This application recognizes handwritten digits from 0 to 9.

        The user draws a digit on a Streamlit canvas or uploads an image.
        The image is converted to MNIST format: 28×28 grayscale, bright digit
        on a black background, resized and centered. A CNN model trained on the
        MNIST dataset predicts the digit and returns a confidence score.
        """
    )

    st.markdown(
        """
        **Technology**
        - Python
        - Streamlit
        - TensorFlow / Keras
        - MNIST
        - NumPy
        - Pillow
        - Matplotlib
        """
    )
