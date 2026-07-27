# Persian Speech Digit Recognizer

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/HamidNova/voice-digit-project?style=social)](https://github.com/HamidNova/voice-digit-project)

A deep learning-based Persian speech digit recognition system that combines advanced audio preprocessing with a hybrid CNN + Bidirectional GRU architecture for accurate spoken digit classification through both a command-line interface and an interactive Streamlit web application.

---

## What's New

- **Hybrid CNN + BiGRU Model** – combines convolutional feature extraction with bidirectional sequence learning
- **Advanced Audio Preprocessing** – silence removal, noise reduction, MFCC and Delta-MFCC extraction
- **Real-Time Speech Recognition** – live microphone inference from the terminal
- **Interactive Streamlit Interface** – microphone recording, audio upload, confidence scores, and prediction visualization
- **Robust Training Pipeline** – Batch Normalization, Dropout, and EarlyStopping for improved generalization
- **Multiple Audio Format Support** – WAV, MP3, FLAC, and OGG files
- **Probability Distribution Visualization** – confidence chart for every prediction

---

## Features

- Persian spoken digit recognition (digits **0–9**)
- Automatic silence trimming using **Librosa**
- Background noise reduction with **Noisereduce**
- MFCC and Delta-MFCC feature extraction
- Fixed-length sequence generation and feature normalization
- Hybrid CNN + Bidirectional GRU neural network
- Batch Normalization, Dropout, and EarlyStopping
- Real-time microphone prediction (CLI)
- Interactive Streamlit web application
- Audio file upload and live recording
- Confidence score and class probability visualization

---

## Model Architecture

```text
Input (50 × 26)
        │
        ▼
Conv1D (64 Filters)
        │
Batch Normalization
        │
MaxPooling1D
        │
Dropout (0.3)
        │
BiGRU (64 Units)
        │
BiGRU (32 Units)
        │
Dropout (0.3)
        │
Dense (64)
        │
Dense (10, Softmax)
        │
Prediction (Digits 0–9)
```

---

## Project Structure

```text
persian-speech-digit-recognizer/
├── dataset/                      # Persian speech dataset
│
├── audio_processor.py            # Audio preprocessing & feature extraction
├── model_trainer.py              # Training and evaluation pipeline
├── live_recognizer.py            # CLI microphone recognizer
├── app.py                        # Streamlit web application
│
├── speech_digit_classifier.h5    # Trained CNN-BiGRU model
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation & Usage

### Local

```bash
git clone https://github.com/HamidNova/voice-digit-project.git
cd voice-digit-project

pip install -r requirements.txt

python model_trainer.py          # Train the model

python live_recognizer.py        # CLI speech recognition

streamlit run app.py             # Launch Streamlit UI
```

The Streamlit application will be available at:

```
http://localhost:8501
```

---

## Evaluation Results

*Sample performance on the test dataset*

| Metric | Value |
|---------|------:|
| Test Accuracy | **86.8%** |
| Input Shape | (50, 26) |
| Number of Classes | 10 |
| Features | MFCC + Delta MFCC |
| Training Strategy | EarlyStopping |
| Audio Processing | Noise Reduction + Silence Removal |

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python 3.10+ |
| Deep Learning | TensorFlow, Keras |
| Audio Processing | Librosa, Noisereduce, SoundDevice, SciPy |
| Machine Learning | NumPy, Scikit-learn |
| Visualization | Matplotlib |
| Web Framework | Streamlit |

---

## License & Contact

MIT License.

**Author:** Hamidreza Mirzaei

**Email:** hamidrezamirzaei8363@gmail.com

**GitHub:** [github.com/HamidNova](https://github.com/HamidNova)