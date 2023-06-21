# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:53:46 2020

@author: Hamza
"""

from scipy.spatial import distance as dist #for calculating distance between facial landmarks.
from imutils.video import VideoStream #for video stream handling
from imutils import face_utils #for various face utility functions.
import playsound #for playing an alarm sound.
import imutils #for image and video processing utilities.
import time # for time-related functions.
import dlib #for facial landmark detection.
import cv2 #for computer vision tasks.

def sound_alarm(path): #This function plays a sound using the playsound library.
	playsound.playsound(path)

def EAR(eye): #This function calculates the eye aspect ratio (EAR) based on the given eye landmarks.
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

eye_threshold=0.3 #The threshold value for determining drowsiness based on the eye aspect ratio.
frames_limit=50 #The number of consecutive frames the eye aspect ratio should be below the threshold to trigger an alarm.
count=0

print("[INFO] loading facial landmark predictor...")

#Loading the facial landmark predictor and initializing the left and right eye indices.
detector = cv2.CascadeClassifier(r"C:\Users\Hamza\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor(r"D:\Projects\Project Resources\shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")

vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450) #Resizing the frame for faster processing.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, #Detecting faces in the grayscale frame using the Haar cascade classifier.
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), #Converting the dlib rectangle to a bounding box
			int(y + h))
        shape= predictor(gray, rect) #Extracting the facial landmarks using the shape predictor.
	    
	#Converting the landmarks to NumPy arrays.
        shape= face_utils.shape_to_np(shape) 
        leftEye= shape[lStart:lEnd]
        rightEye= shape[rStart:rEnd]

	#Calculating the EAR for both the left and right eyes.
        leftEAR= EAR(leftEye)
        rightEAR= EAR(rightEye)

	#Computing the average EAR.
        ear=(leftEAR+rightEAR)/2.0

	#Drawing the eye contours on the frame.
        leftEyeHull= cv2.convexHull(leftEye)
        rightEyeHull= cv2.convexHull(rightEye)
        
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear<eye_threshold:
            count+=1
            
            if count>=frames_limit:
                
                sound_alarm(r"D:\Projects\Project Resources\alarm.wav")
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            count=0
            
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
