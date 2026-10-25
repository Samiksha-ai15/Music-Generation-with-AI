
from pathlib import Path
from datetime import datetime
import random

import numpy as np
import pretty_midi
import streamlit as st
import tensorflow as tf


# ---------------- PROJECT SETTINGS ----------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "music_model.keras"
DATASET_PATH = ROOT / "data" / "processed" / "note_sequences.npy"
OUTPUT_FOLDER = ROOT / "generated_music"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="MelodyMind AI",
    page_icon="🎹",
    layout="wide"
)


# ---------------- CUSTOM DESIGN ----------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1020;
        color: #f1f5f9;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #182848, #4b2c83);
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 42px;
    }

    .hero p {
        color: #e2e8f0;
        font-size: 17px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def get_starting_sequences():
    data = np.load(DATASET_PATH, mmap_mode="r")
    return data


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("🎹 MelodyMind AI")
    st.caption("AI-POWERED MUSIC GENERATION")

    st.markdown("---")
    st.subheader("🎛️ Music Controls")

    instrument_name = st.selectbox(
        "Choose an instrument",
        [
            "Acoustic Grand Piano",
            "Violin",
            "Acoustic Guitar (nylon)",
            "Flute",
            "Cello",
            "Trumpet",
            "Electric Piano 1"
        ]
    )

    number_of_notes = st.slider(
        "Number of generated notes",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )

    note_duration = st.slider(
        "Note duration (seconds)",
        min_value=0.15,
        max_value=1.0,
        value=0.35,
        step=0.05
    )

    gap = st.slider(
        "Gap between notes (seconds)",
        min_value=0.0,
        max_value=0.3,
        value=0.05,
        step=0.05
    )

    st.markdown("---")
    st.caption("Built with Python, TensorFlow and Streamlit")


# ---------------- MAIN HEADER ----------------

st.markdown(
    """
    <div class="hero">
        <h1>🎹 MelodyMind AI</h1>
        <p>
            Create original note sequences using a trained
            LSTM neural network.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- PROJECT INFORMATION ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Training MIDI Files", "1,276")

with col2:
    st.metric("Training Sequences", "698,209")

with col3:
    st.metric("Model", "LSTM Neural Network")


st.markdown("---")

st.subheader("🎼 Generate Your Music")

st.write(
    "Choose your instrument and settings in the sidebar, "
    "then click the button to create a new MIDI file."
)


# ---------------- MUSIC GENERATION ----------------

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. "
        "Please train the model before generating music."
    )

elif not DATASET_PATH.exists():
    st.error(
        "Processed dataset not found. "
        "Please run the preprocessing script first."
    )

else:
    if st.button("✨ Generate AI Music", type="primary"):

        try:
            progress = st.progress(0)
            status = st.empty()

            status.write("Loading the trained AI model...")
            model = load_model()
            progress.progress(15)

            status.write("Preparing a musical starting sequence...")
            sequences = get_starting_sequences()

            seed_index = random.randrange(len(sequences))
            seed = np.asarray(
                sequences[seed_index, :50],
                dtype=np.int32
            )

            generated_notes = seed.tolist()
            progress.progress(25)

            status.write("Generating musical notes...")

            for index in range(number_of_notes):
                input_notes = np.asarray(
                    generated_notes[-50:],
                    dtype=np.int32
                )

                input_data = tf.one_hot(
                    input_notes,
                    depth=128
                )

                input_data = tf.expand_dims(input_data, axis=0)

                probabilities = model(
                    input_data,
                    training=False
                ).numpy()[0]

                probabilities = np.clip(
                    probabilities,
                    1e-8,
                    None
                )

                probabilities /= probabilities.sum()

                next_note = int(
                    np.random.choice(
                        128,
                        p=probabilities
                    )
                )

                generated_notes.append(next_note)

                if index % 10 == 0:
                    percent = 25 + int(
                        60 * (index + 1) / number_of_notes
                    )
                    progress.progress(min(percent, 85))

            status.write("Creating the MIDI file...")
            progress.progress(90)

            midi = pretty_midi.PrettyMIDI()

            program_number = (
                pretty_midi.instrument_name_to_program(
                    instrument_name
                )
            )

            instrument = pretty_midi.Instrument(
                program=program_number
            )

            start_time = 0.0

            # Use the newly generated notes, not the seed notes.
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

            filename = f"ai_music_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid"
            output_path = OUTPUT_FOLDER / filename

            midi.write(str(output_path))

            with open(output_path, "rb") as midi_file:
                midi_bytes = midi_file.read()

            st.session_state["generated_midi"] = midi_bytes
            st.session_state["generated_filename"] = filename = (
                "ai_generated_music.mid"
            )
            st.session_state["generated_instrument"] = (
                instrument_name
            )
            st.session_state["generated_note_count"] = (
                number_of_notes
            )

            progress.progress(100)
            status.success("Your AI-generated music is ready!")

        except Exception as error:
            st.error(f"Music generation failed: {error}")


# ---------------- RESULTS AND DOWNLOAD ----------------

if "generated_midi" in st.session_state:

    st.markdown("---")
    st.subheader("🎧 Your Generated Music")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.write(
            f"**Instrument:** "
            f"{st.session_state['generated_instrument']}"
        )

        st.write(
            f"**Generated notes:** "
            f"{st.session_state['generated_note_count']}"
        )

        st.write(
            "**Format:** MIDI (.mid)"
        )

    with result_col2:
        st.download_button(
            label="📥 Download Your Music",
            data=st.session_state["generated_midi"],
            file_name=st.session_state["generated_filename"],
            mime="audio/midi",
            type="primary"
        )

    st.info(
        "The downloaded file is a MIDI file. "
        "Open it with a MIDI-compatible music player to listen."
    )


# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "MelodyMind AI | AI Music Generation Project | "
    "Powered by TensorFlow, LSTM and Streamlit"
)
