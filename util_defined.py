# utf-8
# Author: ilikewind

import getpass
import os

users = ['qianslab', 'Administrator']
user = getpass.getuser()
print('user: %s' % user)

# 数据总路径
WSI_DATA_DIR_LIST = {"Administrator": "G:/Data/wsi_data_dir/",
                     "qianslab": "/media/qianslab/Data/ACH1/wsi_data_dir/"}
WSI_DATA_DIR = WSI_DATA_DIR_LIST[user]

class config():
    RAW_DATA_DIR = os.path.join(WSI_DATA_DIR, 'wsi')
    RAW_NORMAL_DATA_DIR = os.path.join(RAW_DATA_DIR, 'normal')
    RAW_TUMOR_DATA_DIR = os.path.join(RAW_DATA_DIR, 'tumor')

    TRAIN_PREPROCESS = os.path.join(WSI_DATA_DIR, 'Preprocess')
    SLIDE_MAP_AND_MASK = os.path.join(TRAIN_PREPROCESS, 'map_mask')

    TRAIN_PATCHES = os.path.join(TRAIN_PREPROCESS, 'patch/train')
    NORMAL_PATCHES = os.path.join(TRAIN_PATCHES, 'normal')
    TUMOR_PATCHES = os.path.join(TRAIN_PATCHES, 'tumor')

    VAL_PATCHES = os.path.join(TRAIN_PREPROCESS, 'patch/validation')
    VAL_NORMAL_PATCHES = os.path.join(TRAIN_PATCHES, 'normal')
    VAL_TUMOR_PATCHES = os.path.join(TRAIN_PATCHES, 'tumor')

    # TRAINING
    TRAIN_SAVED = os.path.join(WSI_DATA_DIR, 'Keras_finetuning')
    TRAIN_SAVED_MODEL = os.path.join(TRAIN_SAVED, 'saved_model')
    TRAIN_SAVED_MODEL_INCEPTIONRESNET_V2 = os.path.join(TRAIN_SAVED_MODEL, 'inceptionresnet_v2')

    # Postprocess
    TEST_SLIDE_DIR = os.path.join(RAW_DATA_DIR, 'test')
    POSTPROCESS_DIR = os.path.join(WSI_DATA_DIR, 'Postprocess')
    TISSUE_MASK_DIR = os.path.join(POSTPROCESS_DIR, 'tissue_mask')
    PATCH_FOR_HEATMAP_tumor = os.path.join(POSTPROCESS_DIR, 'patch/WSI_NAME/tumor')
    PATCH_FOR_HEATMAP_normal = os.path.join(POSTPROCESS_DIR, 'patch/WSI_NAME/normal')
    PATCH_FOR_HEATMAP_test = os.path.join(POSTPROCESS_DIR, 'test_patch/WSI_NAME/WSI_NAME')
    PATCH_FOR_HEATMAP_test_DIR = os.path.join(POSTPROCESS_DIR, 'test_patch')
    PATCH_FOR_HEATMAP_PROBABILITY = os.path.join(POSTPROCESS_DIR, 'probability/WSI_NAME_result.txt')
    PATCH_FOR_HEATMAP_PROBABILITY_DIR = os.path.join(POSTPROCESS_DIR, 'probability')
    HEATMAP_PATH = os.path.join(POSTPROCESS_DIR, 'heatmap')



'''
Preprocess parameters
'''
class hp():
    level = 2

    EXTRACT_SAMPLES_EVERY_NORMAL_SLIDE = 2000
    EXTRACT_SAMPLES_EVERY_TUMOR_SLIDE = 3000
    PATCH_SIZE = 256

    # training
    EPOCH = 100
    BATCH_SIZE = 32


