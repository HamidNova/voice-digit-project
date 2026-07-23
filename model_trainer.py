import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from audio_processor import extract_audio_features, FEATURE_TIMESTEPS

DATASET_DIRECTORY = "dataset"
MODEL_SAVE_PATH = "speech_digit_classifier.h5"

def build_dataset(dataset_path):
    features_list = []
    labels_list = []

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory '{dataset_path}' not found.")

    digit_folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
    print(f"Detected classes: {digit_folders}")

    for folder_name in digit_folders:
        folder_path = os.path.join(dataset_path, folder_name)
        file_names = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        print(f"Processing class [{folder_name}] - Files: {len(file_names)}...")

        success_count = 0
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            extracted_feat = extract_audio_features(file_path)

            if extracted_feat is not None:
                features_list.append(extracted_feat)
                labels_list.append(int(folder_name))
                success_count += 1

        print(f"Loaded {success_count} files from class {folder_name}.")

    return np.array(features_list), np.array(labels_list)

def create_classifier(input_shape, num_classes=10):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.GRU(32))(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Speech_Digit_Model")
    return model

if __name__ == "__main__":
    print("Loading dataset...")
    X, y = build_dataset(DATASET_DIRECTORY)

    if len(X) == 0:
        print("Error: No data loaded.")
        exit()

    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    input_dim = (X.shape[1], X.shape[2])
    model = create_classifier(input_dim)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]

    print("Starting training...")
    model.fit(
        X_train, y_train,
        epochs=60,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=cb_list,
        verbose=1
    )

    print("Evaluating model...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to '{MODEL_SAVE_PATH}'.")
