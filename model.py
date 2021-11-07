import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model,Model
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split

save_dir = './checkpoints'
logdir = './logs'

class Model:
    def __init__(self):
        input = Input(shape=(200,200,3))
        conv1 = Conv2D(140,(3,3),activation="relu")(input)
        conv2 = Conv2D(130,(3,3),activation="relu")(conv1)
        batch1 = BatchNormalization()(conv2)
        pool3 = MaxPool2D((2,2))(batch1)
        conv3 = Conv2D(120,(3,3),activation="relu")(pool3)
        batch2 = BatchNormalization()(conv3)
        pool4 = MaxPool2D((2,2))(batch2)
        flt = Flatten()(pool4)

        #age
        age_l = Dense(128,activation="relu")(flt)
        age_l = Dense(64,activation="relu")(age_l)
        age_l = Dense(32,activation="relu")(age_l)
        age_l = Dense(1,activation="relu")(age_l)

        #gender
        gender_l = Dense(128,activation="relu")(flt)
        gender_l = Dense(80,activation="relu")(gender_l)
        gender_l = Dense(64,activation="relu")(gender_l)
        gender_l = Dense(32,activation="relu")(gender_l)
        gender_l = Dropout(0.5)(gender_l)
        gender_l = Dense(2,activation="softmax")(gender_l)

        self.model = Model(inputs=input,outputs=[age_l,gender_l])
        self.model.compile(optimizer="adam",loss=["mse","sparse_categorical_crossentropy"],metrics=['mae','accuracy'])

    def train(self, x_train, y_train, y_train_2, x_test, y_test, y_test_2, epochs = 50):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(save_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

        save = self.model.fit(
                        x_train,[y_train,y_train_2],
                        validation_data=(x_test,[y_test,y_test_2]),
                        callbacks=[tf.keras.callbacks.checkpoint_callback , tensorboard_callback, early_stopping_callback],
                        epochs=epochs)
        return save

if __name__ == '__main__':
    print('test')