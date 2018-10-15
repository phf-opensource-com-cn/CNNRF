# utf-8
# Author：ilikewind
'''
this ops is writed for get patches on whole slide images(WSI).
'''

import os
import numpy
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import openslide
import numpy as np
from xml.etree.ElementTree import parse
import glob
from util_defined import hp, config
'''
############ ===========                    =========== ############
############ =========== get mask functions =========== ############
############ ===========                    =========== ############
'''
def get_normal_wsi_path(path):
    '''
    get sorted normal wsi paths for get mask
    :param path: normal wsi path
    :return: sort wsi paths
    '''
    normal_wsi_paths = glob.glob(os.path.join(path, "*.tif"))
    normal_wsi_paths.sort()
    return normal_wsi_paths

def get_tumor_wsi_path(path):
    tumor_wsi_path = glob.glob(os.path.join(path, "*.tif"))
    tumor_wsi_path.sort()
    tumor_xml_path = glob.glob(os.path.join(path, "*.xml"))
    tumor_xml_path.sort()
    return tumor_wsi_path, tumor_xml_path

# return a new file name and check if it in dir
def name_and_exist(path, dir, last_filename):
    normal_map_name = os.path.split(path)[-1].split('.')[0] + last_filename
    normal_map_exist = chk_file(dir, normal_map_name)
    return normal_map_name, normal_map_exist

# check file is/or saved
def chk_file(dir_path, filename):
    exit = False
    for file_name in os.listdir(dir_path):
        if file_name == filename:
            exit = True

    return exit

# get labeled coors_list from xml file
def read_xml(path,  level_downsam):

    xml = parse(path).getroot()
    coors_list = []
    coors = []
    for areas in xml.iter('Coordinates'):
        for area in areas:
            coors.append([round(float(area.get('X')) / (level_downsam)),
                          round(float(area.get('Y')) / (level_downsam))])
        coors_list.append(coors)
        coors = []
    return np.array(coors_list)


# save tissue mask
def saved_tissue_mask(slide, mask_dir, tissue_mask_name, level):
    normal_slide_lv = slide.read_region((0, 0), level, slide.level_dimensions[level])
    R, G, B, _ = normal_slide_lv.split()
    normal_slide_lv_rgb = np.array(Image.merge('RGB', (R, G, B)))
    normal_slide_lv_rgb = cv2.cvtColor(normal_slide_lv_rgb, cv2.COLOR_BGR2HSV)
    normal_slide_lv_rgb = normal_slide_lv_rgb[:, :, 1]
    _, normal_tissue_mask = cv2.threshold(normal_slide_lv_rgb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(mask_dir, tissue_mask_name), np.array(normal_tissue_mask))

# save normal mask from tumor slide
def saved_normal_tumor_mask(maskdir, tissue_mask_name, tumor_mask_name, normal_mask_name):
    tissue_mask = cv2.imread(os.path.join(maskdir, tissue_mask_name), 0)  # read gracyscale tissue
    tumor_mask = cv2.imread(os.path.join(maskdir, tumor_mask_name), 0)
    tumor_mask_inv = cv2.bitwise_not(tumor_mask)  # inv tumor mask

    tumor_mask_inv_resized = cv2.resize(tumor_mask_inv, (tissue_mask.shape[1], tissue_mask.shape[0]))
    tumor_normal_mask = cv2.bitwise_and(tumor_mask_inv_resized, tissue_mask)
    cv2.imwrite(os.path.join(maskdir, normal_mask_name), tumor_normal_mask)

'''
############ ===========                       =========== ############
############ =========== get patches functions =========== ############
############ ===========                       =========== ############
'''
#
def get_bbox_from_mask_image(mask_image):
    image, contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    return bounding_boxes

# random sampling worked poorly,
# therefore generating samples at stride 10
def get_samples_of_patch_starting_points_with_stride(mask_image, stride):

    bounding_boxes = get_bbox_from_mask_image(mask_image)
    list_starting_points = []
    for x, y, w, h in bounding_boxes:
        #print x, y, w, h
        #print " x, y, w, h : {0}, {1}, {2}, {3}".format(x, y, w, h)
        X = range(x, x+w, stride) #np.arange(x, x+w)
        Y = range(y, y+h, stride) #np.arange(y, y+h)

        for row_starting_point in X:
            for col_starting_point in Y:
                # append in the list
                list_starting_points.append((row_starting_point, col_starting_point))
    return list_starting_points

# extract patches from slide and mask.
def extract_patches_from_slide_and_mask(slide_path, maskdir, mask_lastname, level,
                                        is_for_tumor_patch=False, tumor_patch=True):

    mask_name, mask_exist = name_and_exist(slide_path, maskdir, mask_lastname)

    slide = openslide.OpenSlide(slide_path)
    mask = cv2.imread(os.path.join(maskdir, mask_name), 0)
    if is_for_tumor_patch == True:
        patches_start_points = get_samples_of_patch_starting_points_with_stride(mask, stride=2)
    else:
        patches_start_points = get_samples_of_patch_starting_points_with_stride(mask, stride=10)

    down_samples = round(slide.level_downsamples[level])

    # save patches
    count = 0
    sample_accepted = 0
    sample_rejected = 0
    # number of extract patch at most
    if is_for_tumor_patch == True:
        sample_number = hp.EXTRACT_SAMPLES_EVERY_TUMOR_SLIDE
    else:
        sample_number = hp.EXTRACT_SAMPLES_EVERY_NORMAL_SLIDE
    quotient = round(len(patches_start_points) / sample_number) + 1
    for x, y in patches_start_points:
        if sample_accepted > sample_number:
            break

        count += 1
        if count % quotient != 0:
            continue
        if (mask[y, x] != 0):
            patch_read_from_wsi_at_zero_level = slide.read_region((x * down_samples, y * down_samples),
                                                                        0,
                                                                        (hp.PATCH_SIZE, hp.PATCH_SIZE))

            r, g, b, _ = patch_read_from_wsi_at_zero_level.split()
            normal_patch_rgb = Image.merge("RGB", (r, g, b))
            if tumor_patch == True:
                dir_for_pathes = config.TUMOR_PATCHES
            else:
                dir_for_pathes = config.NORMAL_PATCHES

            normal_name = mask_name.split('_')[0] + '_' + str(sample_accepted) + '_' + \
                          str(count) + '_' + str(sample_rejected) + '.png'
            normal_patch_rgb.save(os.path.join(dir_for_pathes, normal_name))
            sample_accepted += 1
        else:
            sample_rejected += 1
    slide_name = os.path.split(slide_path)[-1]
    slide.close()
    print('File: %s; Accept: %d; Reject: %d' % (slide_name, sample_accepted, sample_rejected))