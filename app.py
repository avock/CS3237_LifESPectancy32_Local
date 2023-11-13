#app.py
from flask import Flask, request, jsonify
import os
import torch
 
import cv2
from ML.models import CNNModel, AE
from ML.model_utils import classify_gesture

from mqtt_server import MQTTServer
 
app = Flask(__name__, static_folder='../static')
 
mqtt_server = MQTTServer()
mqtt_server.start()

# cnn_model = CNNModel()
# cnn_model.load_state_dict(torch.load("./ML/model/new_handclassifier.pt", map_location = torch.device('cpu')))
# # cnn_model.load_state_dict(torch.load("./ML/model/handclassifier.pt", map_location = torch.device('cpu')))

# features = ['temperature','relative_humidity','light_switch', 'ultrasonic','pir', 'pressure'] 
# ae_model = AE(60*24, 10, len(features))
# ae_model = torch.load('./ML/model/autoencoder.pt')

SERVER_URL = 'http://128.199.83.151:5000'
 
UPLOAD_FOLDER = os.path.join('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
FOREGROUND_FILENAME = 'esp32_cam_fg.jpg'
BACKGROUND_FILENAME = 'esp32_cam_bg.jpg'
 
def allowed_filetype(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/')
def main():
    return 'Homepage'
 
@app.route('/image/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'imageFile' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
 
    file = request.files['imageFile']
    errors = {}
    success = False
     
    # Checks for validity of image filetype
    if allowed_filetype(file.filename):
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], FOREGROUND_FILENAME))
        success = True
    else:
        errors[file.filename] = 'File type is not allowed'
    
    # File saved succesfully, proceed with opencv gesture prediction
    if success and not errors:
        gestures  = []

        fg_image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], FOREGROUND_FILENAME))        
        gesture = classify_gesture(fg_image)
        print(gesture)
        
        if (not gesture):
            resp = jsonify({
                'message': 'Somehting Went Wrong with the Gesture Recognition'
            })
            resp.status_code = 401
            return resp
        gestures.append(gesture)
        if (gestures[0] == gestures[0] == gestures[0]):
            gestures = ['', '']
            print(f'Classified gesture is: {gesture}')

            # sending the classified result to cloud server
            # try:
            #     response = requests.post(SERVER_URL + '/gestures', json={'gesture': gesture})
            #     if response.status_code == 200:
            #         print(f'gesture received: {gesture}')
            #         print('Successfully sent gesture data to the remote server.')
            #     else:
            #         print('Failed to send gesture data to the remote server. Status code:', response.status_code)
                
            # except requests.exceptions.RequestException as e:
            #     print('Failed to connect to the remote server:', e)

        else:
            gestures.pop(0)
            
        resp = jsonify({
            'gesture': f'{gesture}',
            'message' : 'Files successfully uploaded'})
        resp.status_code = 201
        return resp
    
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
 
@app.route('/image/background', methods=['POST'])
def upload_background():
    # check if the post request has the file part
    if 'imageFile' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
 
    file = request.files['imageFile']
    errors = {}
    success = False
     
    if allowed_filetype(file.filename):
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], BACKGROUND_FILENAME))
        success = True
    else:
        errors[file.filename] = 'File type is not allowed'
 
    if success and not errors:
        
        resp = jsonify({
            'message' : 'Background Image Uploaded'})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
 
# @app.route('/anomaly_check', methods=['POST'])
# def anomaly_check():
    
 
@app.route('/<path:path>')
def fallback(path):
    return "Wrong Endpoint!"
 
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)