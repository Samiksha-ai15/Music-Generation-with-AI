
from pathlib import Path

import numpy as np
import pretty_midi
import tensorflow as tf


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "note_sequences.npy"
MODEL_PATH = PROJECT_ROOT / "models" / "music_model.keras"
OUTPUT_FOLDER = PROJECT_ROOT / "generated_music"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def generate_music(number_of_notes=200):
    """Generate a new MIDI melody using the trained LSTM model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Note dataset not found: {DATASET_PATH}"
        )

    print("Loading trained music model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading note sequences...")
    sequences = np.load(DATASET_PATH, mmap_mode="r")

    # Select a starting sequence from the dataset
    random_index = np.random.randint(len(sequences))
    seed = sequences[random_index, :50].astype(np.int32)

    generated_notes = seed.tolist()

    print("Generating new musical notes...")

    for _ in range(number_of_notes):
        input_sequence = np.array(
            generated_notes[-50:],
            dtype=np.int32
        )

        input_sequence = tf.one_hot(
            input_sequence,
            depth=128
        )

        input_sequence = tf.expand_dims(
            input_sequence,
            axis=0
        )

        predictions = model.predict(
            input_sequence,
            verbose=0
        )[0]

        # Sample from the model's predicted probabilities
        predictions = np.asarray(predictions, dtype=np.float64)
        predictions = np.clip(predictions, 1e-8, 1.0)
        predictions = predictions / predictions.sum()

        next_note = np.random.choice(
            128,
            p=predictions
        )

        generated_notes.append(int(next_note))

    # Convert generated pitches into a MIDI file
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    )

    start_time = 0.0
    note_duration = 0.35
    gap = 0.05

    for pitch in generated_notes[-number_of_notes:]:
        note = pretty_midi.Note(
            velocity=90,
            pitch=int(pitch),
            start=start_time,
            end=start_time + note_duration
        )

        instrument.notes.append(note)
        start_time += note_duration + gap

    midi.instruments.append(instrument)

    output_file = OUTPUT_FOLDER / "ai_generated_music.mid"
    midi.write(str(output_file))

    print("\nMusic generated successfully!")
    print(f"Output file: {output_file}")
    print(f"Generated notes: {number_of_notes}")
    print("Instrument: Acoustic Grand Piano")


if __name__ == "__main__":
    generate_music()
