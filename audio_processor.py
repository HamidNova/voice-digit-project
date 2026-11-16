import os
import numpy as np
import librosa
import noisereduce as nr

SAMPLING_RATE = 22050
FEATURE_TIMESTEPS = 50
NUM_MFCC = 20


def extract_audio_features(audio_input, sr=SAMPLING_RATE, max_len=FEATURE_TIMESTEPS):
    try:
        if not isinstance(sr, (int, np.integer)) or isinstance(sr, bool) or sr <= 0:
            return None

        if not isinstance(max_len, (int, np.integer)) or isinstance(max_len, bool) or max_len <= 0:
            return None

        if isinstance(audio_input, str):
            if not os.path.isfile(audio_input):
                return None

            signal, current_sr = librosa.load(audio_input, sr=sr)

        elif isinstance(audio_input, np.ndarray):
            signal = audio_input.astype(np.float32)
            current_sr = sr

        else:
            return None

        if signal.ndim != 1 or signal.size == 0:
            return None

        if not np.all(np.isfinite(signal)) or np.all(signal == 0):
            return None

        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val

        trimmed_signal, _ = librosa.effects.trim(signal, top_db=22)

        if trimmed_signal.size < (current_sr * 0.1):
            trimmed_signal = signal

        cleaned_signal = nr.reduce_noise(
            y=trimmed_signal,
            sr=current_sr,
            stationary=True
        )

        mfcc = librosa.feature.mfcc(
            y=cleaned_signal,
            sr=current_sr,
            n_mfcc=NUM_MFCC
        )

        delta_mfcc = librosa.feature.delta(mfcc)

        combined_features = np.vstack([mfcc, delta_mfcc])

        total_frames = combined_features.shape[1]

        if total_frames < max_len:
            pad_size = max_len - total_frames
            padded_features = np.pad(
                combined_features,
                ((0, 0), (0, pad_size)),
                mode="constant"
            )
        else:
            padded_features = combined_features[:, :max_len]

        mean = np.mean(padded_features)
        std = np.std(padded_features) + 1e-8

        normalized_features = (padded_features - mean) / std

        return normalized_features.T

    except Exception as err:
        print(f"Preprocessing error: {err}")
        return None
