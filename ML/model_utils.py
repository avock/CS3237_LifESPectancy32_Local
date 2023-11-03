import torch

from ML.utils import segment_hand

# for one image classification ('blank'-0, 'ok'-1, 'thumbsup'-2, 'thumbsdown'-3, 'fist'-4, 'five'-5)
def classify_hand_gesture(model, bg_img, fg_img):
    gray_img = segment_hand(bg_img, fg_img)
    img_tensor_x = torch.Tensor(gray_img)
    
    with torch.no_grad():
        output = model(img_tensor_x)
        _, pred = torch.max(output, 1)
        
    return pred.item()