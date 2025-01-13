import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class NeuralNetworkModel:
    def __init__(self, input_shape, optimizer, loss, activation, layers_vector, output_layer, batch_size = 5):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.loss = loss
        self.activation = activation
        self.layers_vector = layers_vector
        self.output_layer = output_layer
        self.model = self._build_model()
        self.batch_size = batch_size

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        
        for units in self.layers_vector:
            model.add(tf.keras.layers.Dense(units=units, activation=self.activation))
        
        model.add(tf.keras.layers.Dense(units=self.output_layer))
        return model

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def train_model(self, train_x, train_y, validation_split=0.2, epochs=100, verbose= 1 ,patience=5, checkpoint_path='model_checkpoint.keras'):
        self.compile_model()
        
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
        
        self.losses = self.model.fit(train_x, train_y, validation_split=validation_split,
                                     batch_size=self.batch_size, epochs=epochs,
                                     verbose = verbose,
                                     callbacks=[early_stopping, model_checkpoint])
        return self.losses

    def last_val_loss_value(self):
        return self.losses.history['val_loss'][-1]

    def plot_losses(self):
        loss_df = pd.DataFrame(self.losses.history)
        fig = loss_df.loc[:, ['loss', 'val_loss']].plot()
        plt.show()

    def predict(self, test_x,verbose = 1):
        results = self.model.predict(test_x,
                                     verbose = verbose)
        df = pd.DataFrame(results)
        df_mean = np.mean(df, axis=1)
        return df_mean

    def summary(self):
        return self.model.summary()
    
    def load_model(self, checkpoint_path='model_checkpoint.keras'):

        try:
            self.model = tf.keras.models.load_model(checkpoint_path)
            # print(f"Model loaded from {checkpoint_path}.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
        return self.model
        

    def save_model(self,case, subset, args):

        try:
            self.model.save(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\Regression\\Case_{case}\\GridSearch\\Neural Network\\{subset}_{args}.keras")
            # print(f"Model successfully saved to {file_path}.")
        except Exception as e:
            print(f"Error saving model: {str(e)}")