# Sleepy-driver-detection-with-python-and-opencv
This code is an implementation of a drowsiness detection system using facial landmarks and eye aspect ratio (EAR) calculation. The purpose of this code is to detect if a person's eyes are closed or showing signs of drowsiness, indicating that they may be falling asleep.

# How it works?
Code uses computer vision techniques to detect drowsiness in real-time video streams. It utilizes the dlib library for facial landmark detection, calculates the eye aspect ratio (EAR) to determine eye openness, and triggers an alarm if the eyes are closed for a certain duration. The code demonstrates the usage of various computer vision and image processing techniques such as face detection, landmark extraction, contour drawing, and real-time video stream processing.<br/>
<br/>
1. Video stream is opened in real time with smaller size and grayscale for faster processing.<br/>
2. Face is detected using HaarCaascade Classifiers<br/>
3. Facial landmarks are extracted using Dlib and are converted into numpy arrays.<br/>
4. EAR is calculated for both eyes. if the ear falls below threashold for 50 consecutive frames, alarm goes off.<br/><br/>
![Alt text](https://b2633864.smushcdn.com/2633864/wp-content/uploads/2017/04/blink_detection_plot.jpg?lossy=1&strip=1&webp=1)

# Working Demo: 
https://www.youtube.com/watch?v=MmNHO5OBMmA
