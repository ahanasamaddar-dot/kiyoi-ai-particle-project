# Elsa Magic AR Project

This is an Augmented Reality (AR) application that uses your webcam to create Elsa-themed magical effects. 

## Features
- **Ice World Toggle**: Perform a **Peace Sign** with your hand or press the **'B'** key to toggle the virtual background and "Let It Go" music.
- **Ice Crown**: Automatically appears on your forehead.
- **Snow Beam**: Open your palm towards the camera to blast snow!
- **Magical Pixels**: Perform an **OK Sign** for crystalline blue magic.
- **Frozen Sun**: Give a **Thumbs Up** for a bright white burst.
- **Hail Storm**: Make a **Fist** to drop frozen hail.
- **Hi Olaf!**: Say "Hi Olaf" or "Hey Olaf" to summon Olaf to your side!

## Setup Instructions

### 1. Install Python
Ensure you have Python 3.9 or higher installed on your laptop. You can download it from [python.org](https://python.org).

### 2. Download the Files
Copy this folder to your laptop. Make sure it contains:
- `elsa3.py` (The main application)
- `ice_background.jpg`
- `let_it_go.mp3`
- `olaf_gif2.gif`
- `requirements.txt`

### 3. Install Dependencies
Open your terminal (Command Prompt on Windows, Terminal on Mac) and navigate to this folder. Run the following command:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> If you are on a Mac and face issues installing `PyAudio`, you may need to install `portaudio` first using Homebrew: `brew install portaudio`.

### 4. Run the App
Run the following command in your terminal:

```bash
python elsa3.py
```

On the first run, the app will automatically download the necessary AI models (about 50MB).

## Controls
- **'B' or Peace Sign**: Toggle Background/Music.
- **'Q' or Close Window**: Quit the application.

## Troubleshooting
- **Webcam Access**: Ensure no other apps (like Zoom or Teams) are using your camera.
- **Voice Recognition**: Make sure your microphone is working. The app listens for "Hi Olaf".
- **Performance**: This app uses AI for hand and face tracking. It works best on laptops with decent processors.

## How to Share
To share this with others, the simplest way is to:
1. **ZIP the Folder**: Right-click the folder containing these files and choose "Compress" or "Send to Compressed (zipped) folder".
2. **Send the ZIP**: Send the ZIP file via email, Google Drive, or any other file-sharing service.

## Creating a Standalone Executable (Optional)
If you want others to run the app without installing Python, you can create an executable using `PyInstaller`. 

1. Install PyInstaller: `pip install pyinstaller`
2. Run the build command:
```bash
pyinstaller --onefile --windowed --add-data "ice_background.jpg:." --add-data "let_it_go.mp3:." --add-data "olaf_gif2.gif:." elsa3.py
```
> [!WARNING]
> MediaPipe apps can be tricky to package into a single file due to their internal dependencies. If the executable doesn't work, sharing the folder with the `README.md` instructions is the most reliable method.
