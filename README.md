# Hand Gesture Recognition Application

This application uses **Python 3.11.6**  
*(3.11.0 may also work, but if you run into issues, use 3.11.6+).*

---

## Installation

Use your package manager of choice to install the following libraries:

```bash
# Core libs
pip install opencv-python mediapipe numpy pandas

# Utilities / UI
pip install pillow keyboard pyautogui

# Windows-specific (provides win32api / win32con)
pip install pywin32

```
## Run Application

python HandDetecterClass.py

## Supported Gestures

✌️ Index + Middle up: Open Google in the system’s default browser

🤙 Only Pinky up: Take a screenshot

🖖 Index + Pinky up: Close selected program (Alt+F4)

✋ Index + Middle + Ring up: Close current browser tab (Ctrl+W)

🖕 Middle finger: [CENSORED]

🖐️ All five fingers up: Activate mouse tracking mode
