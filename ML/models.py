import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import Sequential, Model

from ML.utils import segment_hand

class CNNModel(nn.Module):
    def __init__(self): # class constructor function

        super(CNNModel, self).__init__() # initialize an instance of the parent class

        # first convolutional and maxpool layer
        # input 1x120x100, output 16x116x96
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size=(5, 5))
        # input 16x116x96, output 16x58x48
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # second convolutional and maxpool layer
        # input 16x58x48, output 32x54x44
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5))
        # input 32x54x44, output 32x27x22
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # linear layer
        self.fc1 = nn.Linear(in_features=19008, out_features=120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,3)


    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        # Layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        # Output Fully Connected Layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = x

        # return the output predictions
        return output
    
class AE(nn.Module):
  #initialise the model, ran before saving the model
    def __init__(self, n_past, n_future, n_features):
        super(AE, self).__init__()
        # the autoencoder LSTM itself
        self.encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
        self.encoder_l1 = tf.keras.layers.LSTM(100,return_sequences = True, return_state=True)
        self.encoder_outputs1 = self.encoder_l1(self.encoder_inputs)
        self.encoder_states1 = self.encoder_outputs1[1:]

        self.encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
        self.encoder_outputs2 = self.encoder_l2(self.encoder_outputs1[0])
        self.encoder_states2 = self.encoder_outputs2[1:]

        self.decoder_inputs = tf.keras.layers.RepeatVector(n_future)(self.encoder_outputs2[0])

        self.decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(self.decoder_inputs,initial_state = self.encoder_states1)
        self.decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(self.decoder_l1,initial_state = self.encoder_states2)
        self.decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(self.decoder_l2)
        self.model = tf.keras.models.Model(self.encoder_inputs,self.decoder_outputs2)

  #train the model, run ONCE, takes in X_train, Y_train, X_val, Y_val, Adam optimizer
  # for dimensions, see data preprocessing
    def train_model(self, X_train, Y_train, X_val, Y_val, optimizer):
        reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['accuracy'])
        self.model.fit(X_train,Y_train,epochs=25,validation_data=(X_val,Y_val),batch_size=128,verbose=2,callbacks=[reduce_lr])

    # tests the model, returns test accuracy
    def test_model(self, X_test, Y_test):
        scores = self.model.evaluate(X_test, Y_test)
        test_accuracy = scores[1]*100
        return test_accuracy

    def test_anomaly(self, X_test, Y_test, threshold):
        scores = self.model.evaluate(X_test, Y_test)
        test_accuracy = scores*100

        print(f'Test Accuracy: {test_accuracy}')
        return 1 if test_accuracy > threshold else 0