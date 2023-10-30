import streamlit as st
import librosa
import numpy as np
from tensorflow import keras
import pickle

# Load the trained model
model = keras.models.load_model('mfcc_cnn_model_2.h5')  # Load the saved model

# Function to extract MFCC features from an audio signal
def extract_mfcc(y, sr, num_mfcc=13, n_fft=2048, hop_length=512):
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Transpose the MFCC matrix to have shape (number_of_mfcc_coefficients, number_of_frames)
    mfccs = mfccs.T

    return mfccs

# Load labels from the 'labels.pkl' file
with open('label_encoder.pkl', 'rb') as file:
    labels = pickle.load(file)

# Function to predict raag from MFCC features
def predict_raag(audio, model, labels):
    # Use librosa to load the audio
    y, sr = librosa.load(audio, sr=None)

    # Extract MFCC features from the audio signal
    mfccs = extract_mfcc(y, sr)

    # Reshape and normalize the MFCC data
    desired_shape = (216, 13)  # Shape expected by the model
    if mfccs.shape[0] < desired_shape[0]:
        pad_width = desired_shape[0] - mfccs.shape[0]
        mfccs = np.pad(mfccs, pad_width=((0, pad_width), (0, 0)), mode='constant', constant_values=0)
    else:
        mfccs = mfccs[:desired_shape[0], :desired_shape[1]]  # Trim if longer

    mfccs = np.expand_dims(mfccs, axis=-1)  # Add a channel dimension
    mfccs = np.expand_dims(mfccs, axis=0)  # Add a batch dimension

    # Make predictions using the loaded model
    predictions = model.predict(mfccs)

    # Get the predicted label index
    predicted_index = np.argmax(predictions, axis=1)

    # Get the predicted label from the labels list
    predicted_label = labels[predicted_index[0]]  # Assuming 'labels' is a list of labels corresponding to indices

    return predicted_label

# Streamlit web app
st.title("Raag Prediction from Audio")
st.markdown("---")
st.markdown("Upload your audio file or try one of our samples below.")

audio_files = {
    "Sample 1": "sample_assets/sample1(alhaiyabilawal).wav",
    "Sample 2": "sample_assets/sample2(yaman).wav",
    "Sample 3": "sample_assets/sample3(malkans).wav"
}

# Radio buttons to select a sample audio file
selected_file = st.radio("Select a sample audio file", list(audio_files.keys()))

if st.button("Use selected sample"):
    audio_data = open(audio_files[selected_file], "rb").read()
    predicted_raag = predict_raag(audio_files[selected_file], model, labels)
    st.audio(audio_data, format='audio/wav')
    st.write("Predicted Raag:", predicted_raag)
    st.markdown("---")

uploaded_file = st.file_uploader("Or upload your own audio file", type=["mp3", "wav", "ogg", "m4a", "flac"])

if uploaded_file is not None:
    uploaded_file_data = uploaded_file.read()
    predicted_raag = predict_raag(uploaded_file, model, labels)
    st.audio(uploaded_file_data, format='audio/wav')
    st.write("Predicted Raag:", predicted_raag)
    st.markdown("---")
