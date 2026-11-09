from io import BytesIO
from pathlib import Path
import string

from services.image_service import save_original_image
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from models.emnist_model import EmnistCnnModel
from preprocessing.image_preprocessor import ImagePreprocessor
from services.prediction_service import PredictionService
from utils.evaluation import build_probability_figure
import pandas as pd
from services.retraining_service import retrain_model
from tensorflow import keras

from services.database_service import (
    create_table,
    create_feedback_table,
    insert_prediction,
    save_feedback,
    delete_predictions,
    search_predictions,
    get_feedback_statistics
)

# Configure the Streamlit web page.
st.set_page_config(
    page_title="Handwritten Digit/Character Recognition",
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
    model_path = (
        "artifacts/emnist_model_finetuned.keras"
        if Path("artifacts/emnist_model_finetuned.keras").exists()
        else "artifacts/emnist_model.keras"
)
    model = EmnistCnnModel()
    model.load_or_train()
    return model

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "processed_image" not in st.session_state:
    st.session_state.processed_image = None

if "current_prediction_id" not in st.session_state:
    st.session_state.current_prediction_id = None

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

model = get_model()
preprocessor = ImagePreprocessor()
prediction_service = PredictionService(model)

# Initialize database
create_table()
create_feedback_table()

st.markdown(
    '<h1 class="title">✍️ Handwritten Digit/Character Recognition</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">EMNIST + CNN model for handwritten digit and character 0-9,A-Z,a-z</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.write("Model: **EMNIST CNN**")
    st.caption("Python 3.12 recommended")
    st.caption("The first run downloads EMNIST and trains the model once.")

tab_predict,tab_history, tab_about = st.tabs(["Predict", "History", "About"])

with tab_predict:
    left, right = st.columns([1, 1], gap="large")

    image_to_predict = None

    with left:
        st.subheader("1. Draw or upload a digit/char")
        input_method = st.radio(
            "Choose input method",
            ["Draw on Canvas", "Upload Image"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

        if input_method == "Draw on Canvas":
            st.write("Draw a digit or a character at the center.")

            # Create the drawing canvas for handwritten digit input.
            canvas_result = st_canvas(
                fill_color="rgba(255,255,255,0)",
                stroke_width=16,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                update_streamlit=True,
                key=f"canvas_{st.session_state.canvas_key}",
            )

            if canvas_result.image_data is not None:
                has_drawing = bool(canvas_result.image_data[:, :, :3].any())
                if st.session_state.get("canvas_has_drawing", False) and not has_drawing:
                    st.session_state.prediction_result = None
                    st.session_state.processed_image = None
                    st.session_state.current_prediction_id = None
                    st.session_state.pop("current_character", None)
                    for key in list(st.session_state):
                        if key.startswith(("feedback_choice_", "actual_character_")):
                            del st.session_state[key]
                st.session_state.canvas_has_drawing = has_drawing
                if has_drawing:
                    image_to_predict = Image.fromarray(
                        canvas_result.image_data.astype("uint8")
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

        if st.button("Predict Character", type="primary", width="stretch"):
            if image_to_predict is None:
                st.warning("Please draw or upload a digit or a char first.")
            else:
                # Convert the input image into the same format used by the ML model.
                processed = preprocessor.preprocess(image_to_predict)

                if processed is None:
                    st.warning("No digit/char was detected. Please draw a clearer character.")
                else:
                    # Send the processed image to the prediction service.
                    result = prediction_service.predict(processed.tensor)

                    st.session_state.prediction_result = result
                    st.session_state.processed_image = processed.image_28x28

                    # Save original image
                    image_path = save_original_image(image_to_predict)

                    # Save prediction information
                    predict_id = insert_prediction(
                        image_path=str(image_path),
                        character=result.character,
                        confidence=result.confidence
                    )

                    st.session_state.current_prediction_id = predict_id
                    st.session_state.current_character = result.character
        
        # Display prediction after Streamlit reruns
        if st.session_state.prediction_result is not None:

            result = st.session_state.prediction_result

            prediction_column, image_column = st.columns(2, gap="medium")
            with prediction_column:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div>Prediction</div>
                        <div class="prediction-number">{result.character}</div>
                        <div>Confidence: <b>{result.confidence * 100:.2f}%</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with image_column:
                st.markdown("**Processed image**")
                st.image(
                    st.session_state.processed_image,
                    width=200,
                    clamp=True,
                )

    if st.session_state.get("current_prediction_id") is not None:

        st.divider()
        st.subheader("Prediction Feedback")

        prediction_id = st.session_state.current_prediction_id
        predicted_character = st.session_state.current_character

        feedback_left, feedback_right = st.columns([1, 2], gap="large")

        with feedback_left:
            st.write(f"Predicted character: **{predicted_character}**")

            feedback = st.radio(
                "Was the prediction correct?",
                ["Yes", "No"],
                index=None,
                key=f"feedback_choice_{prediction_id}"
            )

        actual_character = None

        if feedback == "Yes":
            actual_character = predicted_character

        elif feedback == "No":
            with feedback_right:
                probabilities = st.session_state.prediction_result.probabilities
                ranked_indices = sorted(
                    (
                        index for index, probability in enumerate(probabilities)
                        if round(float(probability) * 100) > 0
                    ),
                    key=lambda index: probabilities[index],
                    reverse=True,
                )
                feedback_candidates = [
                    PredictionService.EMNIST_CLASSES[index]
                    for index in ranked_indices
                ]
                candidate_probabilities = {
                    PredictionService.EMNIST_CLASSES[index]: float(probabilities[index])
                    for index in ranked_indices
                }
                st.write("What character did you actually write?")
                selection_key = f"actual_character_{prediction_id}"
                actual_character = st.session_state.get(selection_key)
                if actual_character not in feedback_candidates:
                    actual_character = None

                for row_start in range(0, len(feedback_candidates), 4):
                    columns = st.columns(4)
                    for column, character in zip(
                        columns, feedback_candidates[row_start:row_start + 4]
                    ):
                        column.button(
                            f"{character} ({candidate_probabilities[character]:.2%})",
                            key=f"candidate_{prediction_id}_{character}",
                            type="primary" if actual_character == character else "secondary",
                            width="stretch",
                            on_click=st.session_state.update,
                            kwargs={selection_key: character},
                        )

        if st.button(
            "Submit Feedback",
            disabled=actual_character is None,
            key=f"submit_feedback_{prediction_id}"
        ):
            save_feedback(
                prediction_id=prediction_id,
                actual_character=actual_character
            )

            st.success("Feedback saved successfully!")   

    # Display probability chart below feedback.
    if st.session_state.prediction_result is not None:
        st.subheader("Prediction Probability", text_alignment="center")
        fig = build_probability_figure(
            st.session_state.prediction_result.probabilities
        )
        st.pyplot(fig, clear_figure=True)

with tab_history:
    st.subheader("Prediction History")

    stats = get_feedback_statistics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Reviewed", stats["total_reviewed"])

    with col2:
        st.metric("Correct", stats["correct"])

    with col3:
        st.metric("Incorrect", stats["incorrect"])

    with col4:
        accuracy = stats["accuracy"]

        st.metric(
            "Feedback Accuracy",
            f"{accuracy:.2f}%" if accuracy is not None else "N/A"
        )

    characters = (
        ["All"]
        + list(string.digits)
        + list(string.ascii_uppercase)
        + list(string.ascii_lowercase)
    )

    with st.form("history_search"):
        col1, col2, col3 = st.columns(3)

        with col1:
            start_date = st.date_input(
                "From Date", value=None
            )
            end_date = st.date_input(
                "To Date", value=None
            )

        with col2:
            min_confidence = st.slider(
                "Minimum Confidence (%)", 0, 100, 0
            )
            max_confidence = st.slider(
                "Maximum Confidence (%)", 0, 100, 100
            )

        with col3:
            character = st.selectbox(
                "Character", characters
            )
            character_type = st.selectbox(
                "Character Type",
                ["All", "Digit", "Uppercase", "Lowercase"]
            )

        sort_order = st.selectbox(
            "Sort By",
            [
                "Newest first",
                "Oldest first",
                "Highest confidence",
                "Lowest confidence"
            ]
        )

        search_clicked = st.form_submit_button(
            "Search", type="primary"
        )

    if "history_filters" not in st.session_state:
        st.session_state.history_filters = {}

    if search_clicked:
        if (start_date and end_date and
                start_date > end_date):
            st.error("Invalid date range.")
        elif min_confidence > max_confidence:
            st.error("Invalid confidence range.")
        else:
            st.session_state.history_filters = {
                "start_date": start_date,
                "end_date": end_date,
                "min_confidence": min_confidence,
                "max_confidence": max_confidence,
                "character": character,
                "character_type": character_type,
                "sort_order": sort_order
            }

    filters = st.session_state.history_filters
    results = search_predictions(**filters)

    st.write(f"**{len(results)} matching records**")

    if results:
        df = pd.DataFrame(results)
        df.insert(0, "Select", False)
        df.insert(1, "Preview", False)

        if "history_version" not in st.session_state:
            st.session_state.history_version = 0

        if search_clicked:
            st.session_state.history_version += 1

        edited_df = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            disabled=[
                "id",
                "character",
                "confidence_percent",
                "timestamp",
                "actual_character",
                "feedback_status"
            ],
            column_order=[
                "Select",
                "Preview",
                "id",
                "character",
                "confidence_percent",
                "timestamp",
                "actual_character",
                "feedback_status"
            ],
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    default=False
                ),
                "character": st.column_config.TextColumn(
                    "Predicted"
                ),
                "confidence_percent": st.column_config.NumberColumn(
                    "Confidence (%)",
                    format="%.2f%%"
                ),
                "actual_character": st.column_config.TextColumn(
                    "Actual"
                ),
                "Preview": st.column_config.CheckboxColumn(
                    "Preview",
                    default=False
                ),
                "feedback_status": st.column_config.TextColumn(
                    "Feedback"
                )
            },
            key=f"history_{st.session_state.history_version}"
        )

        # Find selected records for preview
        preview_rows = edited_df[edited_df["Preview"]]

        if not preview_rows.empty:

            # Preview the first checked record
            record = preview_rows.iloc[0]

            st.divider()
            st.subheader("Prediction Image Preview")

            image_path = record["image_path"]

            if image_path and Path(image_path).is_file():
                st.image(
                    str(image_path),
                    caption=f"Original handwritten image — Record #{record['id']}",
                    width=280
                )
            else:
                st.warning("Original image file not found.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted", record["character"])

            with col2:
                st.metric(
                    "Confidence",
                    f"{record['confidence_percent']:.2f}%"
                )

            with col3:
                st.metric("Feedback", record["feedback_status"])

            if record["actual_character"] is not None:
                st.write(
                    f"Actual character: **{record['actual_character']}**"
                )
                
        selected_ids = edited_df.loc[
            edited_df["Select"], "id"
        ].tolist()

        st.write(f"{len(selected_ids)} selected")

        if st.button(
            "Delete Selected",
            disabled=not selected_ids
        ):
            delete_predictions(selected_ids)
            st.session_state.history_version += 1
            st.rerun()
    else:
        st.info("No matching predictions.")

        # -----------------------------------------
    # Model Retraining
    # -----------------------------------------

    st.divider()
    st.subheader("Model Retraining")

    st.write(
        "Fine-tune the CNN using reviewed handwriting "
        "and original EMNIST training samples."
    )

    st.write(
        f"Reviewed samples available: "
        f"{stats['total_reviewed']}"
    )

    epochs = st.number_input(
        "Retraining Epochs",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    if st.button(
        "Retrain Model",
        disabled=stats["total_reviewed"] == 0
    ):

        with st.spinner("Retraining CNN... Please wait."):

            try:
                # Fine-tune and save separate model
                history, new_path = retrain_model(
                    epochs=int(epochs)
                )

                # Load the same test data
                model = EmnistCnnModel()
                X_test, y_test = model.load_test_samples()

                original = keras.models.load_model(
                    model.model_path
                )

                finetuned = keras.models.load_model(
                    new_path
                )

                original_accuracy = original.evaluate(
                    X_test, y_test, verbose=0
                )[1]

                finetuned_accuracy = finetuned.evaluate(
                    X_test, y_test, verbose=0
                )[1]

                st.session_state.retraining_results = {
                    "original_accuracy": original_accuracy,
                    "finetuned_accuracy": finetuned_accuracy,
                    "model_path": str(new_path)
                }

                st.success("Retraining completed!")

            except Exception as e:
                st.error(f"Retraining failed: {e}")

    # Keep results visible after Streamlit reruns
    if "retraining_results" in st.session_state:

        result = st.session_state.retraining_results

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Original Accuracy",
                f"{result['original_accuracy']:.2%}"
            )

        with col2:
            st.metric(
                "Fine-tuned Accuracy",
                f"{result['finetuned_accuracy']:.2%}",
                delta=(
                    f"{(result['finetuned_accuracy'] - result['original_accuracy']) * 100:+.2f} pp"
                )
            )

        st.write(
            f"Fine-tuned model saved to: "
            f"`{result['model_path']}`"
        )

with tab_about:
    st.subheader("About This Project")
    st.write(
        """
        This application recognizes handwritten single digits and characters 0-9, A-Z, a-z.

        The user draws a digit on a Streamlit canvas or uploads an image.
        The image is converted to EMNIST format: 28×28 grayscale, bright digit
        on a black background, resized and centered. A CNN model trained on the
        EMNIST dataset predicts the digit/char and returns a confidence score.
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
