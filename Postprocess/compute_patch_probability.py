# utf-8
# Author: ilikewind

'''
input: patches path which patch's name is row_**
output: file that contain patch name and its probability (0-1)
'''
import os
import glob
import numpy as np
import pandas as pd
import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

from util_defined import config, hp

parser = argparse.ArgumentParser(description='Computer the patch"s probability of tumor. ')
parser.add_argument('--patch_path', default=config.PATCH_FOR_HEATMAP_test_DIR, type=str,
                    metavar='PATCH PATH', help='The input test patch"s dir path. ')
parser.add_argument('--model_name', default='inception_resnet_v2', type=str,
                    metavar='MODEL NAME', help='The computer model name, like (xception, inception_v3 ..)')
parser.add_argument('--model_dir', default=config.TRAIN_SAVED_MODEL, type=str,
                    metavar='MODEL DIR', help='The computer model, like (xception, inception_v3 ..)')

parser.add_argument('--result_txt', default=config.PATCH_FOR_HEATMAP_PROBABILITY, type=str,
                    metavar='RESULT TXT', help='The result probability. ')
parser.add_argument('--start', default=0, type=int,
                    metavar='SLIDE_START_INDEX', help='The slide start index. ')
parser.add_argument('--end', default=2, type=int,
                    metavar='SLIDE END INDEX', help='The slide end index. ')

# data flow
def get_data_generator(val_path):
    val_datagen = ImageDataGenerator(rescale=1./255)

    val_generator = val_datagen.flow_from_directory(val_path,
                                                    target_size=(256, 256),
                                                    class_mode=None,
                                                    shuffle=False)

    return val_generator

# generator the model for predict.
def get_model_for_predict(name):
    model_json, model_weight = choose_model_according_to_model_name(name)
    model = model_from_json(open(model_json).read())
    model.load_weights(model_weight)
    return model


# choose the model for predict
def choose_model_according_to_model_name(name):
    model_dir = os.path.join(args.model_dir, name)
    return os.path.join(model_dir, name+'_finetuning.json'), os.path.join(model_dir, name+'.best.h5')



if __name__ == "__main__":

    args = parser.parse_args()

    patch_dirs = args.patch_path # the test patch dir
    model_name = args.model_name # the model name
    # select patch to computer
    patch_dir_all = os.listdir(patch_dirs)
    patch_dir_select = patch_dir_all[args.start:args.end]

    # select model to computer
    model = get_model_for_predict(model_name)
    for patch_dir in patch_dir_select:
        if patch_dir+'_result.txt' not in os.listdir(config.PATCH_FOR_HEATMAP_PROBABILITY_DIR):
            patch_generator = get_data_generator(os.path.join(patch_dirs, patch_dir))
            history = model.predict_generator(patch_generator, steps=None, max_queue_size=10,
                                              workers=6, use_multiprocessing=True, verbose=1)

            probability_list = history[:, 1:len(history)].flatten().tolist()
            filename_list = os.listdir(os.path.join(patch_dirs, os.path.join(patch_dir, patch_dir)))
            file_probability = {'file': filename_list,
                                'probability': probability_list}
            data = pd.DataFrame(file_probability)

            result_path = args.result_txt.replace('WSI_NAME', patch_dir) # result saved path
            data.to_csv(result_path)
            print(' %s result has saved ! ' % patch_dir)









