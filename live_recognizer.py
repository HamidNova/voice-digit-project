import os
import numpy as np
import sounddevice as sd
import tensorflow as tf
from audio_processor import extract_audio_features, SAMPLING_RATE, FEATURE_TIMESTEPS

MODEL_FILE = "speech_digit_classifier.h5"
RECORD_DURATION = 2.5


def run_live_inference():
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file '{MODEL_FILE}' not found.")
        print("Please run 'model_trainer.py' first.")
        return

    print("Loading model...")
    try:
        classifier = tf.keras.models.load_model(MODEL_FILE)
        print("Model loaded successfully.")
    except Exception as err:
        print(f"Failed to load model: {err}")
        return

    print("==========================================")
    print("   Real-Time Speech Digit Recognition     ")
    print("==========================================")

    while True:
        try:
            user_input = input("\nPress Enter to record audio (or type 'q' to quit): ")
            if user_input.strip().lower() == 'q':
                print("Exiting application.")
                break

            print(f"Recording for {RECORD_DURATION} seconds... (Speak a digit)")

            recorded_audio = sd.rec(
                int(RECORD_DURATION * SAMPLING_RATE),
                samplerate=SAMPLING_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            print("Recording completed. Processing...")

            audio_buffer = recorded_audio.flatten()

            features = extract_audio_features(audio_buffer, sr=SAMPLING_RATE, max_len=FEATURE_TIMESTEPS)

            if features is not None:
                model_input = np.expand_dims(features, axis=0)

                predictions = classifier.predict(model_input, verbose=0)[0]
                predicted_digit = np.argmax(predictions)
                confidence = predictions[predicted_digit] * 100

                print("----------------------------------")
                print(f" Predicted Digit: >>> [{predicted_digit}] <<<")
                print(f" Confidence:      {confidence:.2f}%")
                print("----------------------------------")

                if confidence < 50.0:
                    print("Warning: Low confidence prediction. Environment may be noisy.")
            else:
                print("Error: Invalid audio signal or complete silence detected.")

        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
            break
        except Exception as err:
            print(f"Error during live inference: {err}")


if __name__ == "__main__":
    run_live_inference()
