from enum import Enum

class GestureType(Enum):
    THUMBS_UP = "Thumbs Up"
    CHEESE = "Cheese"
    PALM = "Palm"
    FIST = "Fist"
    HAND_DETECTED = "Hand Detected, No Gesture"
    NO_HAND_DETECTED = "No Hand Detected"
    GESTURE_ERROR = "Error In Gesture Recognition"