
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.model import build_model


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "note_sequences.npy"
MODEL_PATH = PROJECT_ROOT / "models" / "music_model.keras"

# Training settings
SEQUENCE_LENGTH = 50
NUMBER_OF_NOTES = 128
BATCH_SIZE = 64
EPOCHS = 10


def train_model():
    """Train the LSTM model using the preprocessed MIDI dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run python src/data_preprocessing.py first."
        )

    print("Loading preprocessed music dataset...")

    sequences = np.load(DATASET_PATH)

    print(f"Dataset shape: {sequences.shape}")

    # Split inputs and target notes
    X = sequences[:, :SEQUENCE_LENGTH].astype(np.int32)
    y = sequences[:, SEQUENCE_LENGTH].astype(np.int32)

    # Convert note pitches into one-hot encoded vectors
    print("Preparing training data...")

    X = tf.one_hot(X, depth=NUMBER_OF_NOTES)
    y = tf.one_hot(y, depth=NUMBER_OF_NOTES)

    # Build the LSTM model
    print("Building LSTM model...")

    model = build_model(
        sequence_length=SEQUENCE_LENGTH,
        number_of_notes=NUMBER_OF_NOTES
    )

    model.summary()

    # Save the best model during training
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    # Train the model
    print("\nStarting LSTM training...")

    model.fit(
        X,
        y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save the final trained model
    model.save(str(MODEL_PATH))

    print("\nTraining completed successfully!")
    print(f"Trained model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
