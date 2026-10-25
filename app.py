
from pathlib import Path
from datetime import datetime
import random
import io
import shutil
import subprocess
import tempfile

import numpy as np
import pretty_midi
import streamlit as st
import tensorflow as tf


# =====================================================
# PROJECT SETTINGS
# =====================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "music_model.keras"
DATASET_PATH = ROOT / "data" / "processed" / "note_sequences.npy"
OUTPUT_FOLDER = ROOT / "generated_music"
HISTORY_FILE = OUTPUT_FOLDER / "history.json"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MelodyMind AI Studio",
    page_icon="🎹",
    layout="wide"
)


# =====================================================
# PROFESSIONAL DESIGN
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #f1f5f9;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    padding: 32px;
    border-radius: 20px;
    background: linear-gradient(120deg, #182848, #4b2c83);
    margin-bottom: 25px;
    border: 1px solid #554080;
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

[data-testid="stMetric"] {
    background: #151e32;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #293650;
}

.track-card {
    background: #151e32;
    border: 1px solid #293650;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 12px;
}

.small-note {
    color: #a5b4cc;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# MODEL AND DATASET LOADING
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


@st.cache_data
def get_starting_sequences():
    return np.load(
        DATASET_PATH,
        mmap_mode="r"
    )


# =====================================================
# SESSION STATE
# =====================================================

if "music_history" not in st.session_state:
    st.session_state["music_history"] = []

if "generated_midi" not in st.session_state:
    st.session_state["generated_midi"] = None

if "generated_wav" not in st.session_state:
    st.session_state["generated_wav"] = None

if "generated_filename" not in st.session_state:
    st.session_state["generated_filename"] = None

if "generated_instrument" not in st.session_state:
    st.session_state["generated_instrument"] = None

if "generated_note_count" not in st.session_state:
    st.session_state["generated_note_count"] = None

if "generated_mood" not in st.session_state:
    st.session_state["generated_mood"] = None


# =====================================================
# HELPER: MIDI TO WAV
# =====================================================

def convert_midi_to_wav(midi_path):
    """
    Convert MIDI to WAV if FluidSynth and a sound font
    are available. Otherwise return None.
    """

    fluidsynth = shutil.which("fluidsynth")

    if not fluidsynth:
        return None

    soundfont_candidates = [
        ROOT / "soundfonts" / "default.sf2",
        ROOT / "soundfonts" / "soundfont.sf2",
    ]

    soundfont = next(
        (path for path in soundfont_candidates if path.exists()),
        None
    )

    if soundfont is None:
        return None

    wav_path = midi_path.with_suffix(".wav")

    try:
        result = subprocess.run(
            [
                fluidsynth,
                "-ni",
                str(soundfont),
                str(midi_path),
                "-F",
                str(wav_path),
                "-r",
                "22050"
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False
        )

        if result.returncode == 0 and wav_path.exists():
            return wav_path.read_bytes()

    except (subprocess.SubprocessError, OSError):
        pass

    return None


# =====================================================
# SIDEBAR: MUSIC CONTROLS
# =====================================================

with st.sidebar:

    st.title("🎹 MelodyMind AI")
    st.caption("AI-POWERED MUSIC STUDIO")

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

    mood = st.selectbox(
        "Choose a music mood",
        [
            "Calm",
            "Happy",
            "Suspenseful",
            "Energetic"
        ]
    )

    number_of_notes = st.slider(
        "Number of generated notes",
        min_value=50,
        max_value=500,
        value=100,
        step=50
    )

    # Mood settings influence timing, not the trained
    # model's learned emotional understanding.
    mood_settings = {
        "Calm": {
            "duration": 0.65,
            "gap": 0.15
        },
        "Happy": {
            "duration": 0.25,
            "gap": 0.03
        },
        "Suspenseful": {
            "duration": 0.45,
            "gap": 0.20
        },
        "Energetic": {
            "duration": 0.18,
            "gap": 0.02
        }
    }

    note_duration = st.slider(
        "Note duration (seconds)",
        min_value=0.15,
        max_value=1.0,
        value=float(mood_settings[mood]["duration"]),
        step=0.05
    )

    gap = st.slider(
        "Gap between notes (seconds)",
        min_value=0.0,
        max_value=0.3,
        value=float(mood_settings[mood]["gap"]),
        step=0.05
    )

    st.markdown("---")

    st.caption("Built with Python, TensorFlow and Streamlit")


# =====================================================
# MAIN HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <h1>🎹 MelodyMind AI Studio</h1>
    <p>
        Generate original melodies using your trained
        LSTM neural network.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# DASHBOARD METRICS
# =====================================================

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Training MIDI Files", "1,276")

with metric2:
    st.metric("Training Sequences", "698,209")

with metric3:
    st.metric("Model", "LSTM Neural Network")


st.markdown("---")


# =====================================================
# NAVIGATION
# =====================================================

page = st.radio(
    "Studio Navigation",
    [
        "🎼 Create Music",
        "📚 Music Library",
        "ℹ️ About"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")


# =====================================================
# PAGE 1: CREATE MUSIC
# =====================================================

if page == "🎼 Create Music":

    st.subheader("🎼 Create Your Music")

    st.write(
        "Choose your instrument and mood, then generate "
        "a new musical sequence."
    )

    if not MODEL_PATH.exists():

        st.error(
            "Trained model not found. Check models/music_model.keras."
        )

    elif not DATASET_PATH.exists():

        st.error(
            "Processed dataset not found. Run preprocessing first."
        )

    else:

        if st.button(
            "✨ Generate AI Music",
            type="primary"
        ):

            progress = st.progress(0)
            status = st.empty()

            try:

                status.write("Loading the trained AI model...")
                model = load_model()
                progress.progress(10)

                status.write("Preparing a starting sequence...")
                sequences = get_starting_sequences()

                seed_index = random.randrange(len(sequences))

                seed = np.asarray(
                    sequences[seed_index, :50],
                    dtype=np.int32
                )

                generated_notes = seed.tolist()

                progress.progress(20)

                status.write(
                    f"Generating {number_of_notes} musical notes..."
                )

                for index in range(number_of_notes):

                    input_notes = np.asarray(
                        generated_notes[-50:],
                        dtype=np.int32
                    )

                    input_data = tf.one_hot(
                        input_notes,
                        depth=128
                    )

                    input_data = tf.expand_dims(
                        input_data,
                        axis=0
                    )

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

                        percent = 20 + int(
                            60 * (index + 1) / number_of_notes
                        )

                        progress.progress(
                            min(percent, 80)
                        )

                status.write("Creating the MIDI file...")
                progress.progress(85)

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

                # Unique filename avoids overwriting old tracks.
                filename = (
                    "melody_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    + ".mid"
                )

                output_path = OUTPUT_FOLDER / filename

                midi.write(str(output_path))

                midi_bytes = output_path.read_bytes()

                status.write("Preparing optional audio playback...")
                progress.progress(92)

                wav_bytes = convert_midi_to_wav(output_path)

                st.session_state["generated_midi"] = midi_bytes
                st.session_state["generated_wav"] = wav_bytes
                st.session_state["generated_filename"] = filename
                st.session_state["generated_instrument"] = instrument_name
                st.session_state["generated_note_count"] = number_of_notes
                st.session_state["generated_mood"] = mood

                # Add track to current-session history.
                st.session_state["music_history"].append({
                    "filename": filename,
                    "instrument": instrument_name,
                    "notes": number_of_notes,
                    "mood": mood,
                    "path": str(output_path),
                    "created": datetime.now().strftime(
                        "%d %b %Y, %I:%M:%S %p"
                    )
                })

                progress.progress(100)
                status.success("Your AI-generated music is ready!")

            except Exception as error:

                st.error(
                    f"Music generation failed: {error}"
                )

    # Display the latest generated track.
    if st.session_state["generated_midi"] is not None:

        st.markdown("---")

        st.subheader("🎧 Your Generated Track")

        info1, info2, info3 = st.columns(3)

        with info1:
            st.metric(
                "Instrument",
                st.session_state["generated_instrument"]
            )

        with info2:
            st.metric(
                "Notes",
                st.session_state["generated_note_count"]
            )

        with info3:
            st.metric(
                "Mood Setting",
                st.session_state["generated_mood"]
            )

        # In-app audio playback if conversion worked.
        wav_bytes = st.session_state["generated_wav"]

        if wav_bytes:

            st.write("### ▶️ Listen to your music")

            st.audio(
                wav_bytes,
                format="audio/wav"
            )

        else:

            st.info(
                "Direct audio playback is not configured yet. "
                "You can still download the MIDI file below. "
                "To enable playback, install FluidSynth and place "
                "a compatible .sf2 sound font in the soundfonts folder."
            )

        st.download_button(
            label="📥 Download MIDI",
            data=st.session_state["generated_midi"],
            file_name=st.session_state["generated_filename"],
            mime="audio/midi",
            type="primary",
            key="download_latest_midi"
        )

        if wav_bytes:

            st.download_button(
                label="🔊 Download WAV Audio",
                data=wav_bytes,
                file_name=(
                    Path(st.session_state["generated_filename"]).stem
                    + ".wav"
                ),
                mime="audio/wav",
                key="download_latest_wav"
            )


# =====================================================
# PAGE 2: MUSIC LIBRARY
# =====================================================

elif page == "📚 Music Library":

    st.subheader("📚 Your Music Library")

    history = st.session_state["music_history"]

    if not history:

        st.info(
            "No tracks generated in this session yet. "
            "Create a track to see it here."
        )

    else:

        st.write(
            f"You have generated {len(history)} track(s) "
            "in this session."
        )

        if st.button("🗑️ Clear Session History"):

            st.session_state["music_history"] = []
            st.rerun()

        for index, track in enumerate(reversed(history)):

            with st.container(border=True):

                st.write(f"### 🎵 {track['filename']}")

                st.caption(
                    f"Created: {track['created']}"
                )

                st.write(
                    f"**Instrument:** {track['instrument']}  \n"
                    f"**Notes:** {track['notes']}  \n"
                    f"**Mood setting:** {track['mood']}"
                )

                track_path = Path(track["path"])

                if track_path.exists():

                    midi_data = track_path.read_bytes()

                    st.download_button(
                        label="📥 Download MIDI",
                        data=midi_data,
                        file_name=track["filename"],
                        mime="audio/midi",
                        key=f"history_download_{index}_{track['filename']}"
                    )

                else:

                    st.warning(
                        "This MIDI file could not be found on disk."
                    )


# =====================================================
# PAGE 3: ABOUT
# =====================================================

elif page == "ℹ️ About":

    st.subheader("ℹ️ About MelodyMind AI")

    st.write("""
    MelodyMind AI is an AI-powered music generation application.
    It uses a trained Long Short-Term Memory (LSTM) neural network
    to predict sequences of musical pitches.
    """)

    st.markdown("### 🧠 Technologies")

    st.markdown("""
    - **Python** — core programming language
    - **TensorFlow / Keras** — LSTM model
    - **NumPy** — numerical operations
    - **PrettyMIDI** — MIDI file creation
    - **Streamlit** — interactive web interface
    """)

    st.markdown("### 🎼 Current capabilities")

    st.markdown("""
    - Generate MIDI music
    - Select an instrument
    - Adjust note count, duration and spacing
    - Apply mood-inspired timing settings
    - Download generated MIDI files
    - Maintain a session-based music library
    - Play WAV audio when audio conversion is configured
    """)

    st.warning(
        "Mood selection currently changes timing settings only. "
        "The model has not been trained to understand mood labels."
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "MelodyMind AI Studio | AI Music Generation "
    "| TensorFlow • LSTM • Streamlit"
)
