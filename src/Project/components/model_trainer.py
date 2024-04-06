import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.Project.exception import CustomException
from src.Project.logger import logging


import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import Sequential
from keras import layers, models
from keras.layers import Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D

# Pre Trained model Import
from keras.applications import EfficientNetV2L

import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact','model.keras')

class modeltrainer:
    def __init__(self):
       self.model_trainer_config = ModelTrainerConfig()


    def get_train_val_batches(self, train_dir, valid_dir, batch_size):
        try:
           train_data = tf.keras.preprocessing.image_dataset_from_directory(
                        train_dir,
                        batch_size = batch_size,
                        image_size = (64,64),
                        shuffle = True
                        )
           print(f'Total {len(train_data.class_names)} nummber of classes is detected')

           val_data = tf.keras.preprocessing.image_dataset_from_directory(
                valid_dir,
                batch_size = batch_size,
                image_size = (64,64),
                shuffle = True
                )
           self.class_labels = train_data.class_names
           
           return train_data, val_data, self.class_labels

        except Exception as e:
           raise CustomException(e,sys)
        
    def initiatemodel(self, train_dir, valid_dir, batch_size, epoch):
        try:
            #load the data
            train_data, val_data, class_labels = self.get_train_val_batches(train_dir, valid_dir, batch_size)
           
            base_model = EfficientNetV2L(
                          input_shape=(64,64,3),
                          include_top=False,
                          weights='imagenet'
                          )
            # Freeze the pre-trained layers
            for layer in base_model.layers:
                layer.trainable = False

            # Create a Sequential model

            model = Sequential()

            model.add(base_model)
            model.add(GlobalAveragePooling2D())
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(256,activation="relu"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(len(class_labels), activation='softmax'))

            opt  = keras.optimizers.Adam(learning_rate=0.0001)

            model.summary()

            model.compile(optimizer=opt,
                            loss='sparse_categorical_crossentropy',
                            metrics= ["accuracy"]
                        )
            

            my_callbacks = [
                            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, restore_best_weights=True),
                            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.001, verbose=1)
                            ]

            model.fit(
                train_data,
                epochs = epoch,
                validation_data = val_data,
                callbacks = [my_callbacks]
            )

            #model.save(self.model_trainer_config.trained_model_file_path)

            logging.info("Model Training has completed")

        except Exception as e:
            raise CustomException(e,sys)

    def get_classlabels(self):
        try:
            if self.class_labels is not None:
                return self.class_labels
            else:
                raise CustomException("Class labels are not available. Please run get_train_val_batches method first.", sys)
        except Exception as e:
            raise CustomException(e,sys)
                