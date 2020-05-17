# OpenCV-Web-Stream-With-Aruco-Tracking

## Starting the camera server
1. Be sure to have `opencv-python`, `opencv-contrib-python`, `imutils`, and `Flask` install by using pip.
2. Calibrate your camera using `calibration/calibration_ChAruco.py`
    - Please visit [this repository](https://github.com/ddelago/Aruco-Marker-Calibration-and-Pose-Estimation) for more information on how to properly calibrate your camera.
3. `python webstreaming.py`
4. The camera server can now be viewed at [http://localhost:6006](http://localhost:6006).
    - Ensure that a webcam is connected and that the correct camera is being sourced (line 20).
    - You can now embed this camera stream into other web pages such as a dashboard.
