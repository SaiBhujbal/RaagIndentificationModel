import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from io import BytesIO
from scipy import signal
from tensorflow.keras.models import load_model
import joblib

# Load the trained model
model = load_model('mfcc_cnn_model_2.h5')  # Load the saved model

# Load the label encoder from the pickled file
label_encoder = joblib.load('label_encoder.pkl')  # Replace 'label_encoder.pkl' with your label encoder filename

# Function to extract MFCC features from an audio file
def extract_mfcc(audio, sample_rate, num_mfcc=13, n_fft=2048, hop_length=512):
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Transpose the MFCC matrix to have shape (number_of_mfcc_coefficients, number_of_frames)
    mfccs = mfccs.T

    return mfccs

# Function to detect the most dominant frequency
def detect_dominant_frequency(audio):
    sample_rate = audio.frame_rate
    y = np.array(audio.get_array_of_samples())
    f, t, S = signal.spectrogram(y, sample_rate)

    max_freq_indices = np.argmax(S, axis=0)
    dominant_freqs = f[max_freq_indices]

    return np.mean(dominant_freqs)

# Function to preprocess and predict raag from MFCC features
def predict_raag(audio_file, model, label_encoder):
    audio = AudioSegment.from_file(audio_file, format=audio_file.name.split('.')[-1])

    # Convert the audio to MP3 format in memory
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format="mp3")
    audio_bytes.seek(0)
    converted_audio = AudioSegment.from_file(audio_bytes, format="mp3")

    # Extract the audio as an array
    audio_array = np.array(converted_audio.get_array_of_samples())
    sample_rate = converted_audio.frame_rate

    # Extract MFCC features from the converted audio
    mfccs = extract_mfcc(audio_array, sample_rate)

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

    # Get the predicted label
    predicted_label = np.argmax(predictions, axis=1)
    
    # Map the predicted label to raag names
    predicted_raag = label_encoder.inverse_transform(predicted_label)

    return predicted_raag

# Streamlit web app
st.title("Raag Prediction from Audio")

audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg", "m4a", "flac"])

if audio_file is not None:
    predicted_raag = predict_raag(audio_file, model, label_encoder)
    st.write("Predicted Raag:", predicted_raag)
