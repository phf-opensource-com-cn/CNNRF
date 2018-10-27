# utf-8
# Author: ilikewind
'''
extract patches from trainset to validateset.
'''
import os
import shutil
from tqdm import tqdm
from util_defined import config, hp

trainset_path = [config.NORMAL_PATCHES, config.TUMOR_PATCHES]
valset_path = [config.VAL_NORMAL_PATCHES, config.VAL_TUMOR_PATCHES]

val_wsi_name = ['normal5', 'normal7', 'normal18', 'normal19', 'normal22', 'normal31',
                'tumor4', 'tumor8', 'tumor25', 'tumor31', 'tumor46', 'tumor58', 'tumor72']

for i, path in enumerate(trainset_path):
    for file in tqdm(os.listdir(path)):
        if file.split('_')[0] in (c for c in val_wsi_name):
            print(file)
            shutil.move(os.path.join(trainset_path[i], file), valset_path[i])
