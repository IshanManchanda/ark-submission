# ARK Task 2: Face Detection

### Project Structure
main.py is the entry-point into the project
which imports the Ball class from ball.py.
The files required for face detection are present in the models directory.

The initial version of the project which used OpenCV's
haarcascade face detector is retained in the legacy folder,
and it follows the same overall structure.

Note that the DirectShow backend is used for the webcam feed
and the frame size is manually set to 1280x720,
the maximum camera resolution on the development machine.
These parameters may need to be tweaked depending on the
platform and hardware of the testing machine.

