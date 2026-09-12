# Handwritten Digit Recognition System

A simple machine learning application that recognizes handwritten digits from 0 to 9.

This project is developed as a Python group final project.

## Features

- Draw or upload a handwritten digit
- Predict digits from 0 to 9
- Show prediction confidence
- Show model accuracy
- Compare different machine learning models
- Simple web interface using Streamlit

## Technologies

- Python
- Streamlit
- Scikit-learn
- NumPy
- Matplotlib
- TensorFlow / Keras (optional)

## Dataset

We use the handwritten digits dataset from Scikit-learn.

The dataset contains images of handwritten digits from 0 to 9.

## System Architecture

```mermaid
flowchart LR
    A[User] --> B[Streamlit UI]
    B --> C[Image Preprocessing]
    C --> D[Prediction Service]
    D --> E[Machine Learning Model]
    E --> F[Prediction Result]

    G[Scikit-learn Dataset] --> H[Training Service]
    H --> E
