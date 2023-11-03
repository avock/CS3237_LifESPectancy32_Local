import cv2
import numpy as np

def segment_hand(bg, target_img, threshold=45):
    def segment(image):
        # find the absolute difference between background and current frame
        diff = cv2.absdiff(bg, image)

        # threshold the diff image so that we get the foreground
        thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)

        # get the contours in the thresholded image
        (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # return None, if no contours detected
        if len(cnts) == 0:
            return
        else:
            # based on contour area, get the maximum contour which is the hand
            segmented = max(cnts, key=cv2.contourArea)
            return (thresholded, segmented)

    thresholded, segmented = segment(target_img)
    gray_img = cv2.resize(thresholded,(100, 120))
    gray_img = gray_img.reshape((1,1,120,100))

    return gray_img
    
# def compare_image(image1, image2):
#     gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#     def mse(img1, img2):
#         h, w = img1.shape
#         diff = cv2.subtract(img1, img2)
#         err = np.sum(diff**2)
#         mse = err/(float(h*w))
#         return mse
    
#     mse_results = mse(gray_image1, gray_image2)
#     return mse_results