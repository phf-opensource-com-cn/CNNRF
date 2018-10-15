# utf-8
# Author: ilikewind
'''
get mask for extract patches from whole slide images.
if tumor wsi
input: wsi, xml
output: tumor mask, tissue mask, non-tumor mask, adjoining tumor mask

if normal wsi
input: wsi
output: normal mask
'''

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




def get_normal_mask(path, level, mask_dir):
    '''
    :param path: normal slide paths
    :param level: level
    :return: get tissue mask and slide map
    '''
    normal_wsi_paths = ops.get_normal_wsi_path(path)
    for normal_wsi_path in normal_wsi_paths:
        print('Get mask from: %s' % normal_wsi_path)
        normal_slide = openslide.OpenSlide(normal_wsi_path) #get wsi picture
        # get slide map
        # print(normal_slide.level_dimensions[level])
        normal_slide_map = np.array(normal_slide.get_thumbnail(normal_slide.level_dimensions[level]))

        #　save normal slide map
        normal_map_name, normal_map_exist = ops.name_and_exist(normal_wsi_path, mask_dir, '_map.png')
        if normal_map_exist == False:
            cv2.imwrite(os.path.join(mask_dir, normal_map_name), normal_slide_map) # add noraml slide map

        # save tissue mask / draw tissue mask
        tissue_mask_name, tissue_mask_exit = ops.name_and_exist(normal_wsi_path, mask_dir, "_tissue_mask.png")
        if tissue_mask_exit == False:
            ops.saved_tissue_mask(normal_slide, mask_dir, tissue_mask_name, level)

def get_tumor_mask(path, level, maskdir):
    tumor_wsi_paths, tumor_xml_paths = ops.get_tumor_wsi_path(path)
    for tumor_wsi_path, tumor_xml_path in zip(tumor_wsi_paths, tumor_xml_paths):
        print('Get mask from: %s - %s' % (tumor_wsi_path, tumor_xml_path))
        tumor_slide = openslide.OpenSlide(tumor_wsi_path)

        tumor_slide_map = np.array(tumor_slide.get_thumbnail(tumor_slide.level_dimensions[level]))
        level_downsample = tumor_slide.level_downsamples[level]
        coors_list = ops.read_xml(tumor_xml_path, level_downsample)

        # draw boundary of tumor in map
        tumor_map_name, tumor_map_exist = ops.name_and_exist(tumor_wsi_path, maskdir, '_map.png')
        if tumor_map_exist == False:
            for coors in coors_list:
                cv2.drawContours(tumor_slide_map, np.array([coors]), -1, 255, 1)
            cv2.imwrite(os.path.join(maskdir, tumor_map_name), tumor_slide_map)

        # check tumor mask / draw tumor mask
        tumor_mask_name, tumor_mask_exist = ops.name_and_exist(tumor_wsi_path, maskdir, '_tumor_mask.png')
        if tumor_mask_exist == False:
            tumor_mask = np.zeros(tumor_slide.level_dimensions[level][::-1])
            for coors in coors_list:
                cv2.drawContours(tumor_mask, np.array([coors]), -1, 255, -1)
                cv2.imwrite(os.path.join(maskdir, tumor_mask_name), tumor_mask)

        # check tissue mask / draw tissue mask
        tumor_tissue_mask_name, tissue_mask_exist = ops.name_and_exist(tumor_wsi_path, maskdir, '_tissue_mask.png')
        if tissue_mask_exist == False:
            ops.saved_tissue_mask(tumor_slide, maskdir, tumor_tissue_mask_name, level)

        # check tumor_normal mask / draw normal mask
        normal_mask_name, normal_mask_exist = ops.name_and_exist(tumor_wsi_path, maskdir, '_normal_mask.png')
        if normal_mask_exist == False:
            ops.saved_normal_tumor_mask(maskdir, tumor_tissue_mask_name, tumor_mask_name, normal_mask_name)

# in order to add normal patches from the area around tumor mask, maybe add function for this
def get_mining_data_mask(path, maskdir):
    tumor_wsi_paths, _ = ops.get_tumor_wsi_path(path)
    for slide_path in tumor_wsi_paths:
        print('mining data from %s' % slide_path)
        tumor_mask_name, tumor_mask_exist = ops.name_and_exist(slide_path, maskdir, '_tumor_mask.png')
        tumor_tissue_mask_name, tissue_mask_exist = ops.name_and_exist(slide_path, maskdir, '_tissue_mask.png')

        tumor_mask = cv2.imread(os.path.join(maskdir, tumor_mask_name), 0)
        tissue_mask = cv2.imread(os.path.join(maskdir, tumor_tissue_mask_name), 0)

        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(50, 50))
        dilate_tumor_mask = cv2.dilate(tumor_mask, kernel)

        dilate_tumor_mask_inv = cv2.bitwise_not(dilate_tumor_mask)
        tumor_mask_inv = cv2.bitwise_not(tumor_mask)

        mining_normal_mask_name, mining_normal_mask_exist = ops.name_and_exist(slide_path, maskdir, '_mining_few_normal_mask.png')
        if mining_normal_mask_exist == False:
            mining_Tumor_normal_mask = cv2.bitwise_and(dilate_tumor_mask_inv, tissue_mask)
            cv2.imwrite(os.path.join(maskdir, mining_normal_mask_name), mining_Tumor_normal_mask)

        mining_most_normal_mask_name, mining_most_normal_mask_exist = ops.name_and_exist(slide_path, maskdir,'_mining_most_normal_mask.png')
        if mining_most_normal_mask_exist == False:
            mining_most_normal_mask = cv2.bitwise_and(tumor_mask_inv, dilate_tumor_mask)
            cv2.imwrite(os.path.join(maskdir, mining_most_normal_mask_name), mining_most_normal_mask)


if __name__ == "__main__":
    # paths of data and defined utils
    normal_wsi_paths = config.RAW_NORMAL_DATA_DIR
    tumor_wsi_paths = config.RAW_TUMOR_DATA_DIR
    level = hp.level
    maskdir = config.SLIDE_MAP_AND_MASK

    get_normal_mask(normal_wsi_paths, level, maskdir)
    get_tumor_mask(tumor_wsi_paths, level, maskdir)
    get_mining_data_mask(tumor_wsi_paths, maskdir)
