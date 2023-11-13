import torch
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import cv2

from ML.utils import segment_hand

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
    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        # lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
        # bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
        # center1 = hand1['center']  # Center coordinates of the first hand
        # handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

        # Count the number of fingers up for the first hand
        try:
            fingers1 = detector.fingersUp(hand1)
        except Exception as e:
            return f'cvzone detector error: {str(e)}'
        
        if fingers1 == [1,0,0,0,0]:
            # thumb_gesture_count += 1
            # if thumb_gesture_count == 10:
                # thumb_gesture_count = 0
            print(f'Thumbs Up', end=" ")
            return "Thumbs Up"

        # else:
        #     thumb_gesture_count = 0
        if fingers1 == [0,1,1,0,0]:
            # peace_gesture_count += 1
            # if peace_gesture_count == 10:
            #     peace_gesture_count = 0
            print(f'Cheese', end=" ")
            return "Cheese"
        # else:
        #     peace_gesture_count = 0

        if fingers1 == [1,1,1,1,1]:
            # palm_gesture_count += 1
            # if palm_gesture_count == 10:
            #     palm_gesture_count = 0
            print(f'Palm', end=" ")
            return "Palm"     
        # else:
        #     palm_gesture_count = 0

        if fingers1 == [0,0,0,0,0]:
            # fist_gesture_count += 1
            # if fist_gesture_count == 10:
            #     fist_gesture_count = 0
            print(f'Fist', end=" ")
            return "Fist"
        # else:
        #     fist_gesture_count = 0
        return "Hand Detected, no gesture"
    
    else:
        print("No hand detected")
        return "No Hand"

        
    #     print(f'H1 = {fingers1.count(1)}', end=" ")  # Print the count of fingers that are up

    #     # Calculate distance between specific landmarks on the first hand and draw it on the image
    #     length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255),
    #                                               scale=10)

    #     # Check if a second hand is detected
    #     if len(hands) == 2:
    #         # Information for the second hand
    #         hand2 = hands[1]
    #         lmList2 = hand2["lmList"]
    #         bbox2 = hand2["bbox"]
    #         center2 = hand2['center']
    #         handType2 = hand2["type"]

    #         # Count the number of fingers up for the second hand
    #         fingers2 = detector.fingersUp(hand2)
    #         print(f'H2 = {fingers2.count(1)}', end=" ")

    #         # Calculate distance between the index fingers of both hands and draw it on the image
    #         length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0),
    #                                                   scale=10)


    #     print(" ")  # New line for better readability of the printed output

    # # Display the image in a window
    # cv2.imshow("Image", img)

    # # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    # cv2.waitKey(1)
