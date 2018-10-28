# utf-8
# Author: ilikewind

import os
import sys
import parser

import cv2
from PIL import Image
import matplotlib.pyplot as plt

from keras.models import model_from_json
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

def get_data_generator(val_path):
    val_datagen = ImageDataGenerator(rescale=1./255,
                                     horizontal_flip=False)

    val_generator = val_datagen.flow_from_directory(val_path,
                                                    target_size=(256, 256),
                                                    class_mode=None,
                                                    shuffle=False)

    return val_generator

if __name__ == '__main__':
    test_dir = 'G:/Data/wsi_data_dir/Postprocess/patch/predict01'
    validate_generator = get_data_generator(test_dir)

    model = model_from_json(open(
        'G:/Data/wsi_data_dir/Keras_finetuning/saved_model/inception_resnet_v2/inception_resnet_v2_finetuning.json').read())

    model.load_weights('G:/Data\wsi_data_dir/Keras_finetuning/saved_model/inception_resnet_v2/inception_resnet_v2_best.h5')
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.predict_generator(validate_generator, steps=None, max_queue_size=10, workers=6,
                                      use_multiprocessing=False, verbose=1)
    print(history)
