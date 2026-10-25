from pathlib import Path

import numpy as np
import pretty_midi


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIDI_FOLDER = PROJECT_ROOT / "data" / "midi" / "maestro-v3.0.0"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def extract_notes(midi_file):
    """Extract musical note pitches from a MIDI file."""
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_file))
        notes = []

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                notes.append(note.pitch)

        return notes

    except Exception as error:
        print(f"Skipping {midi_file.name}: {error}")
        return []


def prepare_dataset():
    """Convert MIDI files into note sequences for model training."""

    if not MIDI_FOLDER.exists():
        raise FileNotFoundError(
            f"MIDI dataset folder not found: {MIDI_FOLDER}"
        )

    midi_files = [
    file
    for file in MIDI_FOLDER.rglob("*")
    if file.is_file() and file.suffix.lower() in [".mid", ".midi"]
]

    print(f"Found {len(midi_files)} MIDI files.")

    sequences = []
    sequence_length = 50

    for index, midi_file in enumerate(midi_files, start=1):
        notes = extract_notes(midi_file)

        if len(notes) > sequence_length:
            for start in range(0, len(notes) - sequence_length, 10):
                sequence = notes[start:start + sequence_length + 1]
                sequences.append(sequence)

        if index % 100 == 0:
            print(f"Processed {index}/{len(midi_files)} files.")

    if not sequences:
        raise ValueError(
            "No usable note sequences found. "
            "Please check the MIDI dataset folder."
        )

    sequences = np.asarray(sequences, dtype=np.int16)

    output_file = OUTPUT_FOLDER / "note_sequences.npy"
    np.save(output_file, sequences)

    print("\nPreprocessing completed successfully!")
    print(f"Total sequences: {len(sequences)}")
    print(f"Dataset shape: {sequences.shape}")
    print(f"Saved dataset to: {output_file}")


if __name__ == "__main__":
    prepare_dataset()