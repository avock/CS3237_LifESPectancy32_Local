import torch
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import cv2
from enum import Enum

from ML.utils import segment_hand

class GestureType(Enum):
    THUMBS_UP = "Thumbs Up"
    CHEESE = "Cheese"
    PALM = "Palm"
    FIST = "Fist"
    HAND_DETECTED = "Hand Detected, No Gesture"
    NO_HAND_DETECTED = "No Hand Detected"

# for one image classification ('blank'-0, 'ok'-1, 'thumbsup'-2, 'thumbsdown'-3, 'fist'-4, 'five'-5)
def classify_hand_gesture(model, fg_img):
# def classify_hand_gesture(model, bg_img, fg_img):
    gray_img = segment_hand(fg_img)
    img_tensor_x = torch.Tensor(gray_img)
    
    with torch.no_grad():
        output = model(img_tensor_x)
        _, pred = torch.max(output, 1)
        
    return pred.item()

def detect_anomaly(model, X, Y):

    with torch.no_grad():
        accuracy = model.test_model(X, Y)
        
    return accuracy

def classify_gesture(img):
    """
    Initialize the webcam to capture video
    The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
    cap = cv2.VideoCapture(0)

    Initialize the HandDetector class with the given parameters

    Continuously get frames from the webcam
    """
    
    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    try:
        hands, img = detector.findHands(img, draw=False, flipType=True)
    except Exception as e:
        return f'Image file error: {str(e)}'
    if hands:
        hand = hands[0]

        try:
            # would return a tuple with 5 values, corresponding to each finger on the left hand      
            fingers_up = detector.fingersUp(hand)
        except Exception as e:
            return f'cvzone detector error: {str(e)}'
        
        gesture_mapping = {
            (1, 0, 0, 0, 0): GestureType.THUMBS_UP,
            (0, 1, 1, 0, 0): GestureType.CHEESE,
            (1, 1, 1, 1, 1): GestureType.PALM,
            (0, 0, 0, 0, 0): GestureType.FIST,
        }

        gesture = gesture_mapping.get(tuple(fingers_up), GestureType.HAND_DETECTED)
        return gesture
    
    else:
        gesture = GestureType.NO_HAND_DETECTED
        return gesture