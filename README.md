This application uses python 3.11.6<br>

use your package manager of choice to install the following libraries:<br>

cv2        <br>
mediapipe  <br>
numpy      <br>
pandas    <br>
time      <br>
webbrowser<br>
PIL.ImageGrab<br>
keyboard<br>
pyautogui<br>
win32con<br>
win32api<br><br>

When that is complete run the "HandDetecterClass.py" class<br>
<br>

This application supports the following gestures:<br>

Index and middle finger up: Open google in the system's default browser<br>
Thumb and pinky up : Take a screenshot<br>
Index finger and pinky up : Close selected program (Alt + F4)<br>
Index, middle, and ring finger up : Close the current browser tab<br>
Middle finger: [CENSORED]<br><br>

All five fingers out : Activate mouse tracking<br>
<br>
  While in mouse mode:<br>
  Putting the index finger down in this mode will act as a click.<br>
  Fist (all fingers down) : Exit mouse tracking mode.<br>
<br>

The program can be stopped by entering "Ctrl + C" on the console or "Esc" on the camera vision window.<br>
<br>
The mouse tracking tracks "Index finger mcp" as depicted in "THE HAND.png"<br>
<br>
a finger is considered down if the corresponding "dip" is closer to the wrist than the "pip" as described by THE HAND<br>
<br>
<br>
Failures:<br>
Not many to speak of, we experimented with a couple of mouse options, but found one we liked in win32<br>
<br>
Successes:<br>
Using numpy and some quick math to easily and quickly determine whether a finger is "up" or not<br>
<br>
Improvements:<br>
We would like to refine the precision and ease of use with mouse mode, and incorporate swiping gestures.<br>
<br>
Future work:<br>
Integrate linear classifier and cleaner gestures<br>
