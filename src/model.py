
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(number_of_notes=128, sequence_length=50):
    """Build an LSTM model for music generation."""

    model = models.Sequential([
        layers.Input(shape=(sequence_length, number_of_notes)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(128),
        layers.Dropout(0.2),
        layers.Dense(number_of_notes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy"
    )

    return model


def prepare_training_data(sequences):
    """Convert note sequences into model inputs and targets."""

    inputs = sequences[:, :-1].astype(np.int32)
    targets = sequences[:, -1].astype(np.int32)

    inputs = tf.one_hot(inputs, depth=128)
    targets = tf.one_hot(targets, depth=128)

    return inputs, targets
