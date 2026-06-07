import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models

DATASET_PATH = "."
CLASSES = ["Cargo", "Passengership", "Tanker", "Tug"]
SAMPLE_RATE = 16000
SEGMENT_SECONDS = 3
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS
IMG_SIZE = (128, 128)


def audio_to_cqt(y, sr):
    cqt = librosa.cqt(y, sr=sr)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    cqt_db = tf.image.resize(cqt_db[..., np.newaxis], IMG_SIZE).numpy()
    return cqt_db


def load_dataset():
    X = []
    y = []

    for label, class_name in enumerate(CLASSES):
        class_path = os.path.join(DATASET_PATH, class_name)

        for file_name in os.listdir(class_path):
            if not file_name.endswith(".wav"):
                continue

            file_path = os.path.join(class_path, file_name)
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

            for start in range(0, len(audio) - SEGMENT_SAMPLES, SEGMENT_SAMPLES):
                segment = audio[start:start + SEGMENT_SAMPLES]
                feature = audio_to_cqt(segment, SAMPLE_RATE)

                X.append(feature)
                y.append(label)

    return np.array(X), np.array(y)


def build_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(CLASSES), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


X, y = load_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = build_cnn_model()

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32
)

loss, acc = model.evaluate(X_test, y_test)
print("測試準確率：", acc)

y_pred = model.predict(X_test)
y_pred_label = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_label, target_names=CLASSES))
print(confusion_matrix(y_test, y_pred_label))
