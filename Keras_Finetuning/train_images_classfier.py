# utf-8
# Author: ilikewind
'''
fine-tuning for binary classifier
'''

import os
import argparse
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns

from keras.applications import xception
from keras.applications import inception_resnet_v2
from keras.applications import inception_v3
from keras.applications import resnet50
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.models import model_from_json
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from util_defined import config, hp


parser = argparse.ArgumentParser(description='training the base model')
parser.add_argument('--model_name', default='inception_resnet_v2', metavar='MODEL NAME', type=str,
                    help='Select the train model. ')
parser.add_argument('--trainset_path', default=config.TRAIN_PATCHES, metavar='TRAINSET PATH',
                    type=str, help='Trainset path. ')
parser.add_argument('--valset_path', default=config.VAL_PATCHES, metavar='VALSET PATH',
                    type=str, help='Validationset path. ')
parser.add_argument('--savedmodel_path', default=config.TRAIN_SAVED_MODEL, metavar='SAVED MODEL PATH',
                    type=str, help='Saved model path. ')
parser.add_argument('--batch_size', default=hp.BATCH_SIZE, metavar='BATCH SIZE',
                    type=int, help='batch size (inter). ')

# select model module.
def select_model_moduel(model_name):
    if model_name == 'inception_resnet_v2':
        model_used = inception_resnet_v2.InceptionResNetV2
    elif model_name == 'resnet50':
        model_used = resnet50.ResNet50
    elif model_name == 'inception_v3':
        model_used = inception_v3.InceptionV3
    elif model_name == 'xception':
        model_used = xception.Xception

    return model_used
# sort file
# if the best-5 models are made, when retrain, sort the model
# and select the last one.
def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        dir_list = sorted(dir_list, key=lambda x:
                          os.path.getmtime(os.path.join(file_path, x)))
    return dir_list

'''
create data generator 
'''
def get_data_generator(train_path, val_path):

    train_datagen = ImageDataGenerator(featurewise_center=False,
                                       featurewise_std_normalization=False,
                                       rescale=1./255,
                                       rotation_range=20,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(train_path,
                                                        target_size=(256,256),
                                                        batch_size=args.batch_size,
                                                        class_mode='categorical',
                                                        classes={'normal': 0, 'tumor': 1})

    val_datagen = ImageDataGenerator(rescale=1./255,
                                     horizontal_flip=False)

    val_generator = val_datagen.flow_from_directory(val_path,
                                                    target_size=(256,256),
                                                    batch_size=args.batch_size,
                                                    class_mode='categorical',
                                                    classes={'normal': 0, 'tumor': 1})

    return train_generator, val_generator

def create_model(name):
    model_used = select_model_moduel(name)
    base_model = model_used(weights='imagenet',
                            include_top=False,
                            pooling='avg')
    x = base_model.output
    x = Dense(500, activation='relu')(x)
    predictons = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictons)
    # no frozen
    for layer in base_model.layers:
        layer.trainable = True

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def reload_model(json_path, weights_dir, model_name):
    model = model_from_json(open(json_path).read())
    filepath = model_name + '_best.h5'
    weights_path = os.path.join(weights_dir, filepath)
    model.load_weights(weights_path)
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def callback_function(saved_model_dir, model_name):
    filepath = model_name + '_best.h5'
    saved_path = os.path.join(saved_model_dir, filepath)
    csv_path = os.path.join(saved_model_dir, 'result.csv')

    model_chekpoint = ModelCheckpoint(filepath=saved_path, monitor='acc', verbose=1,
                                      save_best_only=True, save_weights_only=True, period=1)

    early_stoping = EarlyStopping(monitor='acc', verbose=1, min_delta=0,
                                  patience=6, mode='auto')

    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.1,
                                  patience=1, min_lr=0.00001)

    csv_logger = CSVLogger(filename=csv_path, separator=',', append=True)
    return [model_chekpoint, early_stoping, reduce_lr, csv_logger]

if __name__ == '__main__':

    args = parser.parse_args()

    # define data and output path
    train_path = args.trainset_path
    val_path = args.valset_path
    saved_model_dir = args.savedmodel_path # save model dir
    model_name = args.model_name # select model name
    # the saved model path
    train_save_weight_path = os.path.join(saved_model_dir, model_name)
    if not os.path.exists(train_save_weight_path):
        os.mkdir(train_save_weight_path)

    # get train_generator and validate_generator
    train_generator, validate_generator = get_data_generator(train_path, val_path)

    # json file name
    json_name = model_name + '_finetuning.json'
    json_path = os.path.join(train_save_weight_path, json_name)

    if not os.path.exists(json_path):
        # create train model
        model = create_model(model_name, model_name)
        print('create model')
        # save net to json
        json_string = model.to_json()
        open(json_path, 'w').write(json_string)
    else:
        model = reload_model(json_path, train_save_weight_path, model_name)
        print("reload model")

    # get callback function
    callbacks = callback_function(train_save_weight_path, model_name)


    history = model.fit_generator(generator=train_generator, epochs=hp.EPOCH,
                                  verbose=1, callbacks=callbacks, validation_data=validate_generator,
                                  class_weight=None,
                                  workers=6, use_multiprocessing=False, shuffle=True, initial_epoch=0)

# do not forget make roc curve