import os
import sys
import zipfile
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.Project.exception import CustomException
from src.Project.logger import logging
from src.Project.entity.config_entity import ModelTrainerConfig
from src.Project.entity.artifact_entity import ModelTrainerArtifact


import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import Sequential
from keras import layers, models
from keras.layers import Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D

# Pre Trained model Import
from tensorflow.keras.applications.inception_v3 import InceptionV3
#from keras.applications import EfficientNetV2L

import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:

        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")

            with zipfile.ZipFile("data.zip", 'r') as zip_ref:
                zip_ref.extractall()
                print("\n Unzipping data successfull \n")
            os.system("rm data.zip")


            train_dir = "train"
            valid_dir = "valid"
            batch_size=self.model_trainer_config.batch_size
            no_epochs = self.model_trainer_config.no_epochs

            self.model_training(train_dir, valid_dir, batch_size, no_epochs)


            os.system("rm -rf train")
            os.system("rm -rf valid")
            os.system("rm -rf test")

        except Exception as e:
            raise CustomException(e,sys)

    def data_preprocessing(self,train_dir,valid_dir,batch_size):
        try:
            train_data = tf.keras.preprocessing.image_dataset_from_directory(
                        train_dir,
                        batch_size = batch_size,
                        image_size = (228,228),
                        shuffle = True
                        )
            print(f'Total {len(train_data.class_names)} nummber of classes is detected')

            val_data = tf.keras.preprocessing.image_dataset_from_directory(
                valid_dir,
                batch_size = batch_size,
                image_size = (228,228),
                shuffle = True
                )
            self.class_labels = train_data.class_names
           
            return train_data, val_data, self.class_labels
        except Exception as e:
            raise CustomException(e,sys)


    def model_training(self,train_dir, valid_dir, batch_size, epoch):
        try:
            #load the data
            train_data, val_data, class_labels = self.data_preprocessing(train_dir, valid_dir, batch_size)
           
            base_model = InceptionV3(
                          input_shape=(228,228,3),
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
                            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.001, verbose=1),
                            keras.callbacks.TensorBoard(log_dir='logs',histogram_freq=1, write_graph=True, write_images=True)
                            ]

            model.fit(
                train_data,
                epochs = epoch,
                validation_data = val_data,
                callbacks = [my_callbacks]
            )

            model.save(os.path.join('artifacts',"model.keras"))

            logging.info("Model Training has completed")

        except Exception as e:
            raise CustomException(e,sys)

    def get_classlabels(self):
        try:
            if self.class_labels is not None:
                return self.class_labels
            else:
                raise CustomException("Class labels are not available. Please run data_preprocessing method first.", sys)
        except Exception as e:
            raise CustomException(e,sys)
                