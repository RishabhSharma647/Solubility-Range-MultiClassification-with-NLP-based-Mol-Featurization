""" MLP Network for Multi-Class Classification in TF 2.0 """
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dropout


""" MLP Network Class """
class MLP_Classifier:
    def __init__(self, hidden_layers_dict, dropout):
        self.hidden_layers_dict = hidden_layers_dict
        self.dropout = dropout
        
        # Define model
        self.model = tf.keras.Sequential()
        for i in range(1, len(self.hidden_layers_dict.keys())):
            self.model.add(layers.Dense(self.hidden_layers_dict['h' + str(i)][0], activation = self.hidden_layers_dict['h' + str(i)][1]))
            self.model.add(Dropout(self.dropout))
        # Define output layer
        self.model.add(layers.Dense(self.hidden_layers_dict['output'][0], activation = self.hidden_layers_dict['output'][1]))
        
        
    def compile_model(self, loss, optimizer, metric):
        """ Compile model and set optimizer
        
        Args:
            loss (string): type of loss to optimize
            optimizer (keras arg): optimizer for loss
            metric (keras arg): metric to track during training
            
        """
        self.model.compile(loss = loss, optimizer = optimizer, metrics=[metric])
        
    def fit_model(self, checkpoint, batch_size, num_epochs, validation_split, X, y):
        """ Fit model
        
        Args:
            batch_size (int): batch size
            num_epochs (int): number of epochs
            validation_split (float): validation set size as fraction
            X (numpy.ndarray): input data
            y (numpy.ndarray): output data
            checkpoint (tensorflow.python.keras.callbacks.ModelCheckpoint): Model checkpoint information for saving
            
        """
        
        self.model.fit(X, y, 
          epochs = num_epochs,
          batch_size = batch_size,
          callbacks = checkpoint,
          validation_split = .2, 
          shuffle = True)
        
        
    def predict(self, x):
        """ Predict output from new network input 
        
        Args: 
            x (numpy.ndarray): input data
        
        Returns:
            (numpy.ndarray): output vector (probability distributon for multi-class classification)
            
        """
        return self.model.predict(x)
    
    
        
        
        
