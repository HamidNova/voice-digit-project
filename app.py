import os
import numpy as np
import librosa
import streamlit as st
import tensorflow as tf
from audio_processor import extract_audio_features, SAMPLING_RATE, FEATURE_TIMESTEPS

MODEL_FILE = "speech_digit_classifier.h5"

st.set_page_config(
    page_title="AI Speech Digit Recognizer",
    layout="centered"
)

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 700;
    color: #4A90E2;
    margin-bottom: 0.5rem;
}
.sub-title {
    text-align: center;
    font-size: 1rem;
    color: #A0AAB8;
    margin-bottom: 2rem;
}
.result-card {
    background: linear-gradient(135deg, #1E2640 0%, #0F1423 100%);
    border: 1px solid #2E3C61;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    margin-top: 15px;
}
.digit-banner {
    font-size: 5rem;
    font-weight: 900;
    color: #00E676;
    margin: 10px 0;
    text-shadow: 0 0 20px rgba(0, 230, 118, 0.4);
}
.confidence-text {
    font-size: 1.1rem;
    color: #E0E6ED;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_FILE):
        return tf.keras.models.load_model(MODEL_FILE)
    return None


classifier = load_trained_model()

st.markdown('<div class="main-title">AI Speech Digit Recognizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">System for Persian Single-Digit Audio Classification</div>', unsafe_allow_html=True)

if classifier is None:
    st.error(f"Model file '{MODEL_FILE}' not found. Please run 'model_trainer.py' first.")
    st.stop()


def predict_audio_signal(audio_data_buffer):
    try:
        signal, sr = librosa.load(audio_data_buffer, sr=SAMPLING_RATE)
        features = extract_audio_features(signal, sr=SAMPLING_RATE, max_len=FEATURE_TIMESTEPS)

        if features is not None:
            input_tensor = np.expand_dims(features, axis=0)
            predictions = classifier.predict(input_tensor, verbose=0)[0]
            predicted_digit = int(np.argmax(predictions))
            confidence = float(predictions[predicted_digit] * 100)
            return predicted_digit, confidence, predictions
        else:
            return None, None, None
    except Exception as err:
        st.error(f"Audio processing error: {err}")
        return None, None, None


tab_mic, tab_file = st.tabs(["Live Microphone", "Upload Audio File"])

audio_source = None

with tab_mic:
    st.write("Record your voice using the microphone below:")
    mic_audio = st.audio_input("Record Digit")
    if mic_audio is not None:
        audio_source = mic_audio

with tab_file:
    st.write("Upload a WAV or MP3 audio file containing a single digit:")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])
    if uploaded_file is not None:
        audio_source = uploaded_file

if audio_source is not None:
    st.audio(audio_source)

    if st.button("Analyze and Predict Digit", use_container_width=True):
        with st.spinner("Processing audio features..."):
            digit, confidence, probs = predict_audio_signal(audio_source)

        if digit is not None:
            st.markdown(f"""
                <div class="result-card">
                    <div style="color: #A0AAB8; font-size: 0.9rem;">DETECTED DIGIT</div>
                    <div class="digit-banner">{digit}</div>
                    <div class="confidence-text">Confidence Score: <b>{confidence:.2f}%</b></div>
                </div>
            """, unsafe_allow_html=True)

            st.write("---")
            st.write("Class Probabilities Distribution:")
            st.bar_chart(probs)

            if confidence < 50.0:
                st.warning("Low confidence detected. Try speaking more clearly or reduce background noise.")
        else:
            st.error("Could not process audio. Ensure the recording is clear and contains audio.")
