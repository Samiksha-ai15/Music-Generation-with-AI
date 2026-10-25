# 🎵 MelodyMind AI — AI Music Generation Using LSTM

MelodyMind AI is an AI-powered music generation application that uses a Long Short-Term Memory (LSTM) neural network to generate musical note sequences and create MIDI music. It provides an interactive Streamlit interface where users can generate music and explore different musical moods and instruments.

## ✨ Features

* 🎼 **AI Music Generation:** Generates musical sequences using a trained LSTM model.
* 🎭 **Mood Selection:** Offers Calm, Happy, Suspenseful, and Energetic settings.
* 🎹 **Instrument Selection:** Allows users to choose an instrument for generated music.
* 🎧 **Music Playback:** Supports audio playback when the required audio tools and soundfont are configured.
* 📥 **Download Generated Music:** Allows users to download generated MIDI files and WAV audio when available.
* 📚 **Music Library:** Provides access to generated tracks during the current app session.
* 🖥️ **Interactive Dashboard:** Built with Streamlit.

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* LSTM Neural Networks
* NumPy
* PrettyMIDI
* Streamlit

## 📂 Project Structure

```text
Music-Generation-with-AI/
│
├── app.py
├── app_backup.py
├── requirements.txt
│
├── models/
│   └── music_model.keras
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── generate.py
│   ├── midi_utils.py
│   ├── model.py
│   └── train.py
│
├── data/
│   ├── midi/
│   └── processed/
│
└── generated_music/
```

*Note: The dataset and generated files may be excluded from the GitHub repository.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Samiksha-ai15/Music-Generation-with-AI.git
```

### 2. Navigate to the Project Folder

```bash
cd Music-Generation-with-AI
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the Application

```bash
python -m streamlit run app.py
```

The application should open in your browser.

## 🧠 How It Works

1. MIDI files provide musical note data for training.
2. The preprocessing module converts the musical data into note sequences.
3. An LSTM model learns patterns from the training sequences.
4. The trained model predicts subsequent musical notes.
5. The generated notes are converted into MIDI files.
6. The Streamlit interface allows users to generate, listen to, and download music.

## 🎭 Available Mood Settings

* 🌿 **Calm**
* 😊 **Happy**
* 👻 **Suspenseful**
* ⚡ **Energetic**

Mood settings influence the generation parameters configured in the application. They do not necessarily represent a model trained specifically to understand emotions.

## 📋 Requirements

* Python 3.11
* Dependencies listed in `requirements.txt`
* A compatible trained model at `models/music_model.keras`
* MIDI dataset files if you want to preprocess the data or train the model again
* Audio synthesis software and a compatible SoundFont for WAV playback, if required by your setup

## 🚀 Future Improvements

* Develop stronger mood-conditioned music generation.
* Improve musical structure and melody quality.
* Add persistent music-generation history.
* Support additional instruments and musical styles.
* Deploy the application online.

## 👩‍💻 Author

**Samiksha M S**

GitHub: [Samiksha-ai15](https://github.com/Samiksha-ai15)

## 📄 License

Choose an appropriate open-source license before redistributing this project or its trained model.
