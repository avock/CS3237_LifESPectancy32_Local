import cv2
import numpy as np

def segment_hand(target_img, threshold=45):
# def segment_hand(bg, target_img, threshold=45):
    # def segment(image):
    #     # find the absolute difference between background and current frame
    #     diff = cv2.absdiff(bg, image)

    #     # threshold the diff image so that we get the foreground
    #     thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    #     thresholded = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)

    #     # get the contours in the thresholded image
    #     (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     # return None, if no contours detected
    #     if len(cnts) == 0:
    #         return
    #     else:
    #         # based on contour area, get the maximum contour which is the hand
    #         segmented = max(cnts, key=cv2.contourArea)
    #         return (thresholded, segmented)

    # thresholded, segmented = segment(target_img)
    # gray_img = cv2.resize(thresholded,(100, 120))
    # gray_img = gray_img.reshape((1,1,120,100))
    ycc = cv2.cvtColor(target_img , cv2.COLOR_BGR2YCR_CB)

    min_ycc = np.array([0,133,85], np.uint8)
    max_ycc = np.array([255,170,125], np.uint8 )
    skin  = cv2.inRange(ycc, min_ycc, max_ycc)

    opening = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=3)
    thresholded = cv2.dilate(opening,np.ones((3,3),np.uint8), iterations=2)

    gray_img = cv2.resize(thresholded,(100, 120))
    gray_img = gray_img.reshape((1,1,120,100))

    return gray_img