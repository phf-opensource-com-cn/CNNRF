# utf-8
# Author: ilikewind

import os
import shutil
import glob
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import cv2
import openslide
from xml.etree.ElementTree import parse
from PIL import Image

from util_defined import config
from util_defined import hp
import ops

def get_normal_patches_from_normal_slide(path, level, maskdir):
    normal_wsi_paths = ops.get_normal_wsi_path(path)
    for slide_path in normal_wsi_paths:
        ops.extract_patches_from_slide_and_mask(slide_path, maskdir, '_tissue_mask.png',
                                                level, is_for_tumor_patch=False, tumor_patch=False)

def get_normal_patches_from_tumor_slide(path, level, maskdir):
    tumor_slide_paths, _ = ops.get_tumor_wsi_path(path)
    for slide_path in tumor_slide_paths:
        ops.extract_patches_from_slide_and_mask(slide_path, maskdir, '_normal_mask.png',
                                                level, is_for_tumor_patch=False, tumor_patch=False)

def get_tumor_patches_from_tumor_slide(path, level, maskdir):
    tumor_slide_paths, _ = ops.get_tumor_wsi_path(path)
    for slide_path in tumor_slide_paths:
        # extract patches from slide
        ops.extract_patches_from_slide_and_mask(slide_path, maskdir, '_tumor_mask.png',
                                                level, is_for_tumor_patch=True, tumor_patch=True)
########################################################################################################
def get_normal_patches_from_few_normal_mask(path, level, maskdir):                                     #
    tumor_slide_paths, _ = ops.get_tumor_wsi_path(path)                                                #                                                                        #
    for slide_path in tumor_slide_paths:                                                               #
        ops.extract_patches_from_slide_and_mask(slide_path, maskdir, '_mining_few_normal_mask.png',    #
                                                level, is_for_tumor_patch=False, tumor_patch=False)    #
                                                                                                       #
def get_normal_patches_from_most_normal_mask(path, level, maskdir):                                    #
    tumor_slide_paths, _ = ops.get_tumor_wsi_path(path)                                                #
    for slide_path in tumor_slide_paths:                                                               #
        ops.extract_patches_from_slide_and_mask(slide_path, maskdir, '_mining_most_normal_mask.png',   #
                                                level, is_for_tumor_patch=True, tumor_patch=False)     #
                                                                                                       #
                                                                                                       #
def get_normal_mining_patch_from_tumor_mask(tumor_wsi_paths, level, maskdir):                          #
    # long distance patch of normal to tumor area                                                      #
    get_normal_patches_from_few_normal_mask(tumor_wsi_paths, level, maskdir)                           #
                                                                                                       #
    # short distance patch of normal to tumor area                                                     #
    get_normal_patches_from_most_normal_mask(tumor_wsi_paths, level, maskdir)                          #
                                                                                                       #
########################################################################################################


if __name__ == '__main__':
    # paths of data and defined utils
    normal_wsi_paths = config.RAW_NORMAL_DATA_DIR
    tumor_wsi_paths = config.RAW_TUMOR_DATA_DIR
    level = hp.level
    maskdir = config.SLIDE_MAP_AND_MASK

    # get normal patches from normal slide
    get_normal_patches_from_normal_slide(normal_wsi_paths, level, maskdir)

    # get tumor patches from tumor slide
    get_tumor_patches_from_tumor_slide(tumor_wsi_paths, level, maskdir)

    # get normal patches from tumor slide
    # get_normal_patches_from_tumor_slide(tumor_wsi_paths, level, maskdir)

    '''
    normal patches data mining using the around area or tumor mask;
    make sure do not using the few/most_mask function and
    get_normal_patches_from_tumor_slide at one time
    '''
    # get normal patch data mining from tumor slide
    get_normal_mining_patch_from_tumor_mask(tumor_wsi_paths, level, maskdir)
