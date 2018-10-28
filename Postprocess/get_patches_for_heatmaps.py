# utf-8
# Author: ilikewind

import os
import numpy as np
import cv2
from PIL import Image
import openslide

from util_defined import config, hp
from Preprocess import ops

import argparse

parser = argparse.ArgumentParser(description='get patches for heatmaps')
parser.add_argument('--mask_path', default=config.TISSUE_MASK_DIR, type=str,
                    metavar='MASK PATH', help='Save the mask path. ')
parser.add_argument('--test_wsis', default=config.TEST_SLIDE_DIR, type=str,
                    metavar='TEST SLIDE PATH', help='Test slide path. ')
parser.add_argument('--level', default=hp.level, type=int,
                    metavar='LEVEL', help='The level. ')
parser.add_argument('--wsis_list_start', default=0, type=int,
                    metavar='WSIS LIST START', help='Select the wsis to get patches for getting heatmaps. ')
parser.add_argument('--wsis_list_end', default=1, type=int,
                    metavar='WSIS LIST END', help='Select the wsis to get patches for getting heatmaps. ')
parser.add_argument('--stride', default=1000, type=int, metavar='STRIDE',
                    help='Get patches stride. ')



# extract patches from slide and mask.
def extract__consecutive_patches_use_slide_and_mask(slide_path, maskdir, mask_lastname, level):

    mask_name, mask_exist = ops.name_and_exist(slide_path, maskdir, mask_lastname)

    slide = openslide.OpenSlide(slide_path)
    mask = cv2.imread(os.path.join(maskdir, mask_name), 0)

    patches_start_points = ops.get_samples_of_patch_starting_points_with_stride(mask, stride=args.stride)

    down_samples = round(slide.level_downsamples[level])

    # save patches
    sample_accepted = 0
    sample_rejected = 0
    for x, y in patches_start_points:
        if (mask[y, x] != 0):
            patch_read_from_wsi_at_zero_level = slide.read_region((x * down_samples, y * down_samples),
                                                                        0,
                                                                        (hp.PATCH_SIZE, hp.PATCH_SIZE))


            r, g, b, _ = patch_read_from_wsi_at_zero_level.split()
            normal_patch_rgb = Image.merge("RGB", [r, g, b])

            wsi_name = os.path.split(slide_path)[-1].split('.')[0]
            dir_for_pathes = config.PATCH_FOR_HEATMAP.replace('WSI_NAME', wsi_name)
            if not os.path.exists(dir_for_pathes):
                os.mkdir(dir_for_pathes)

            patch_name = str(x) + '_' + str(y) + '.png'
            normal_patch_rgb.save(os.path.join(dir_for_pathes, patch_name))
            sample_accepted += 1
        else:
            sample_rejected += 1
    slide_name = os.path.split(slide_path)[-1]
    slide.close()
    print('File: %s; Accept: %d; Reject: %d' % (slide_name, sample_accepted, sample_rejected))

def get_test_tissue_mask(path, level, maskdir):
    test_wsi_paths = ops.get_normal_wsi_path(path)
    select_test_wsi_path = test_wsi_paths[args.wsis_list_start:args.wsis_list_end]
    print(select_test_wsi_path)
    for test_wsi_path in select_test_wsi_path:
        print('Get mask for heatmap: %s' % test_wsi_path)
        slide = openslide.OpenSlide(test_wsi_path)

        slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[level]))

        #　save  slide map
        slide_map_name, slide_map_exist = ops.name_and_exist(test_wsi_path, maskdir, '_map.png')
        if slide_map_exist == False:
            cv2.imwrite(os.path.join(maskdir, slide_map_name), slide_map) # add slide map

        # save tissue mask / draw tissue mask
        tissue_mask_name, tissue_mask_exit = ops.name_and_exist(test_wsi_path, maskdir, "_tissue_mask.png")
        if tissue_mask_exit == False:
            ops.saved_tissue_mask(slide, maskdir, tissue_mask_name, level)

def get_consecutive_patch(path, level, maskdir):
    test_wsi_paths = ops.get_normal_wsi_path(path)
    select_test_wsi_path = test_wsi_paths[args.wsis_list_start:args.wsis_list_end] # change the file for get patches
    print(select_test_wsi_path)
    for test_wsi_path in select_test_wsi_path:
        extract__consecutive_patches_use_slide_and_mask(test_wsi_path, maskdir, '_tissue_mask.png',level)


if __name__ == '__main__':

    args = parser.parse_args()

    test_wsi_paths = args.test_wsis
    maskdir = args.mask_path
    level = args.level

    # get tissue mask from test slide
    get_test_tissue_mask(test_wsi_paths, level, maskdir)

    # get consecutive patch
    get_consecutive_patch(test_wsi_paths, level, maskdir)