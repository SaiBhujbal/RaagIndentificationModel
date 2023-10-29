import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from io import BytesIO
from scipy import signal
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('mfcc_cnn_model.h5')  # Load the saved model

# Function to extract MFCC features from an audio file
def extract_mfcc(audio, sample_rate, num_mfcc=13, n_fft=2048, hop_length=512):
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Transpose the MFCC matrix to have shape (number_of_mfcc_coefficients, number_of_frames)
    mfccs = mfccs.T

    return mfccs

def detect_and_convert_to_C_sharp(file_path):
    audio = AudioSegment.from_file(file_path)
    sample_rate = audio.frame_rate
    y = np.array(audio.get_array_of_samples())
    f, t, S = signal.spectrogram(y, sample_rate)

    # Detect the note from the audio
    max_freq_indices = np.argmax(S, axis=0)
    dominant_freqs = f[max_freq_indices]

    # Find the most dominant frequency
    note_freq = np.mean(dominant_freqs)

    # Convert the note to C#
    C_sharp_freq = 277.18  # Frequency of C# (assuming it's in Hz)
    freq_ratio = C_sharp_freq / note_freq

    audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * freq_ratio)
    })

    buffer = BytesIO()
    audio.export(buffer, format="mp3")
    buffer.seek(0)
    return buffer

# Function to predict raag from MFCC features
def predict_raag(file_path, model):
    buffer = detect_and_convert_to_C_sharp(file_path)
    audio, sample_rate = librosa.load(buffer, sr=None)

# Function to predict raag from MFCC features
def predict_raag(audio, model, sample_rate):
    # Extract MFCC features from the audio
    mfccs = extract_mfcc(audio, sample_rate)

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

    return predicted_label

# Streamlit web app
st.title("Raag Prediction from Audio")

audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg", "m4a", "flac"])

if audio_file is not None:
    audio = AudioSegment.from_file(audio_file, format=audio_file.name.split('.')[-1])
    audio_bytes = audio.export(format='wav').read()  # Export the audio as WAV and read the bytes

    audio, sample_rate = librosa.load(BytesIO(audio_bytes), sr=None)  # Load audio from bytes

    predicted_raag = predict_raag(audio, model, sample_rate)
    st.write("Predicted Raag:", predicted_raag)
