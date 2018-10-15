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

    TRAIN_PATCHES = os.path.join(TRAIN_PREPROCESS, 'patch')
    NORMAL_PATCHES = os.path.join(TRAIN_PATCHES, 'normal')
    TUMOR_PATCHES = os.path.join(TRAIN_PATCHES, 'tumor')


'''
Preprocess parameters
'''
class hp():
    level = 2

    EXTRACT_SAMPLES_EVERY_NORMAL_SLIDE = 1000
    EXTRACT_SAMPLES_EVERY_TUMOR_SLIDE = 2000
    PATCH_SIZE = 256


