
# 🎙️ Persian Speech Digit Recognizer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Librosa-Audio%20Processing-green" alt="Librosa">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Project Status">
</p>

<p align="center">
  <b>A Deep Learning System for Recognizing Persian Spoken Digits (0–9)</b>
</p>

A Persian speech digit recognition system built with **Python, TensorFlow, and Librosa**. It combines audio preprocessing, MFCC-based feature extraction, and a hybrid **Convolutional Neural Network + Bidirectional GRU (CNN-BiGRU)** architecture to classify spoken digits from 0 to 9.

The project provides two ways to recognize speech:

- **Live CLI Recognition:** Record audio through a microphone and predict the spoken digit directly from the terminal.
- **Interactive Web Application:** Record audio or upload an audio file through a Streamlit interface, then view the predicted digit, confidence score, and class probability distribution.

---

## ✨ Key Features

- **Persian Speech Digit Classification** – Recognizes spoken digits from 0 to 9.
- **Advanced Audio Preprocessing** – Silence trimming and background noise reduction.
- **MFCC Feature Extraction** – Combines 20 MFCC coefficients with 20 Delta-MFCC coefficients.
- **Hybrid CNN-BiGRU Architecture** – Learns local audio patterns and temporal dependencies.
- **Fixed-Length Feature Sequences** – Converts audio signals into normalized sequences of 50 time steps.
- **Regularized Model Training** – Uses Batch Normalization, Dropout, EarlyStopping, and learning-rate reduction.
- **Live Microphone Recognition** – Performs inference on microphone recordings through the command-line interface.
- **Interactive Streamlit Application** – Supports microphone recording and audio file uploads.
- **Prediction Confidence** – Displays the model's confidence score for each prediction.
- **Probability Visualization** – Shows the predicted probability distribution across all ten classes.

---

## 🧠 Model Architecture

The recognition model uses a hybrid CNN-BiGRU architecture.

The CNN layers extract local patterns from the audio feature sequence, while Bidirectional GRU layers learn temporal relationships in both forward and backward directions.

### Architecture Overview

```text
Input Audio Features
      │
      ▼
Input (50 × 40)
      │
      ▼
Conv1D (64 Filters, Kernel Size = 3)
      │
      ▼
Batch Normalization
      │
      ▼
MaxPooling1D (Pool Size = 2)
      │
      ▼
Dropout (0.25)
      │
      ▼
Bidirectional GRU (64 Units)
      │
      ▼
Dropout (0.30)
      │
      ▼
Bidirectional GRU (32 Units)
      │
      ▼
Dense (64 Units, ReLU)
      │
      ▼
Dropout (0.30)
      │
      ▼
Dense (10 Units, Softmax)
      │
      ▼
Predicted Digit (0–9)
```

### Model Configuration

| Component | Configuration |
|---|---|
| Input Shape | `(50, 40)` |
| Convolutional Layer | Conv1D, 64 filters |
| Normalization | Batch Normalization |
| Pooling | MaxPooling1D |
| Recurrent Layers | Bidirectional GRU |
| GRU Units | 64 and 32 |
| Dense Layer | 64 units, ReLU |
| Output Layer | 10 units, Softmax |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |
| Evaluation Metric | Accuracy |

---

## 🎧 Audio Processing Pipeline

Audio preprocessing is an essential part of the recognition system.

Each audio sample passes through the following steps:

```text
Raw Audio
    │
    ▼
Audio Loading (Librosa)
    │
    ▼
Silence Trimming
    │
    ▼
Noise Reduction
    │
    ▼
MFCC Extraction (20 Coefficients)
    │
    ▼
Delta-MFCC Extraction (20 Coefficients)
    │
    ▼
Feature Concatenation (40 Features)
    │
    ▼
Padding / Truncation
    │
    ▼
Normalization
    │
    ▼
Feature Sequence (50 × 40)
    │
    ▼
CNN-BiGRU Classifier
```

### Feature Extraction Details

| Parameter | Value |
|---|---:|
| Sampling Rate | 22,050 Hz |
| Number of MFCC Coefficients | 20 |
| Number of Delta-MFCC Coefficients | 20 |
| Total Features per Time Step | 40 |
| Sequence Length | 50 |
| Final Feature Shape | `(50, 40)` |

The same feature extraction pipeline is shared by the training process, terminal recognizer, and Streamlit application to maintain consistent model inputs.

---

## 📁 Project Structure

```text
voice-digit-project/
│
├── dataset/                       # Local training dataset (not tracked by Git)
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── ...
│   └── 9/
│
├── audio_processor.py             # Audio preprocessing and feature extraction
├── model_trainer.py               # Dataset preparation, model creation and training
├── live_recognizer.py             # Live microphone-based CLI recognition
├── app.py                         # Streamlit web application
│
├── speech_digit_classifier.h5     # Generated trained model (not tracked by Git)
│
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git exclusion rules
└── README.md                      # Project documentation
```

**Note:** The dataset and trained model are not included in this repository. The model is generated locally by running the training script.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/HamidNova/voice-digit-project.git

cd voice-digit-project
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Install Streamlit if it is not already included in `requirements.txt`:

```bash
pip install streamlit
```

---

## 📦 Dataset Preparation

Before training, place the audio dataset inside the `dataset/` directory.

The expected structure is:

```text
dataset/
├── 0/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
│
├── 1/
│   ├── audio_001.wav
│   └── ...
│
├── 2/
│   └── ...
│
...
│
└── 9/
    └── ...
```

Each subdirectory represents one digit class.

- Folder names must be numeric labels from `0` to `9`.
- Training audio files must use the `.wav` format.
- The dataset directory must be named `dataset`.

The dataset is not distributed with this repository. You must provide the audio files before training.

---

## 🚀 Usage

All commands below should be executed from the project root directory.

### 1. Train the Model

```bash
python model_trainer.py
```

The training pipeline:

1. Loads audio files from the dataset.
2. Extracts audio features using the shared preprocessing module.
3. Creates stratified training and testing subsets.
4. Builds the CNN-BiGRU classifier.
5. Trains the model with validation monitoring.
6. Evaluates the classifier.
7. Saves the trained model as:

```text
speech_digit_classifier.h5
```

### 2. Run Live Recognition (CLI)

```bash
python live_recognizer.py
```

The terminal application loads the trained model and allows you to record spoken digits through your microphone.

- Press `Enter` to record a sample.
- Speak a single digit.
- View the predicted digit and confidence score.
- Type `q` to exit.

### 3. Launch the Streamlit Application

```bash
streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

The web application provides two input methods:

**Live Microphone**
- Record a spoken digit directly in the browser.
- Analyze the recording.
- View the prediction and confidence score.

**Upload Audio File**
- Upload a supported audio file.
- Analyze the audio.
- View the predicted digit and probability distribution.

Supported upload formats:

`WAV`, `MP3`, `OGG`, `FLAC`

---

## 📊 Evaluation

The model training script evaluates the classifier using a held-out dataset split and reports:

- Classification accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The final numerical results should be reported from a reproducible run using the current version of the preprocessing pipeline and model architecture.

> Evaluation metrics are intentionally not hardcoded here until they have been verified against the current implementation.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Audio Processing | Librosa, Noisereduce |
| Feature Extraction | MFCC, Delta-MFCC |
| Numerical Computing | NumPy, SciPy |
| Machine Learning Utilities | Scikit-learn |
| Microphone Recording | SoundDevice |
| Web Application | Streamlit |
| Testing | Python unittest |

---

## 🔍 Design Highlights

### Shared Audio Processing

The project uses a single audio feature extraction module across training and inference. This helps ensure that the model receives consistently prepared input data.

### Sequence-Based Learning

The CNN-BiGRU architecture combines convolutional feature extraction with bidirectional recurrent layers to process temporal audio information.

### Multiple Recognition Interfaces

The project separates model training, terminal inference, and web-based inference into dedicated modules, making the system easier to use and maintain.

### Input Validation and Error Handling

The processing and inference modules include checks for invalid audio, incompatible feature shapes, missing model files, and loading errors.

---

## 🧪 Testing

The project includes unit tests for the main processing, training, and inference components.

Run the test suite from the project root:

```bash
python -m unittest test.py
```

The tests use mocked dependencies where appropriate, so they do not require actual microphone recording or a full model training run.

---

## 📌 Limitations

- The system is designed for single-digit audio classification, not continuous speech transcription.
- The classifier recognizes ten predefined classes (0–9).
- A trained model and the corresponding dataset must be available locally.
- Prediction confidence is a model score and should not be interpreted as a guarantee of correctness.
- Recognition performance can vary with recording quality, pronunciation, background noise, and dataset characteristics.

---

## 🔮 Future Improvements

- Expand the dataset with more speakers and recording conditions.
- Improve robustness against background noise and pronunciation variation.
- Evaluate additional model architectures.
- Add a dedicated test-set evaluation protocol.
- Explore real-time streaming inference.
- Improve visualization and prediction analysis.

---

## 👨‍💻 Author

**Hamidreza Mirzaei**

AI & Machine Learning Enthusiast

- **GitHub:** [HamidNova](https://github.com/HamidNova)
- **Email:** [hamidrezamirzaei8363@gmail.com](mailto:hamidrezamirzaei8363@gmail.com)

---

## 📄 License

This project is distributed under the license specified in the repository's `LICENSE` file.
