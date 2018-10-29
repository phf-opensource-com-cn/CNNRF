# utf-8
# Author: ilikewind

import openslide
import cv2
import numpy as np

import os
from util_defined import config, hp
from Preprocess import ops

import argparse

parser = argparse.ArgumentParser(description='get patches for heatmaps')
parser.add_argument('--mask_path', default=config.TISSUE_MASK_DIR, type=str,
                    metavar='MASK PATH', help='Save the mask path. ')
parser.add_argument('--test_wsis', default=config.TEST_SLIDE_DIR, type=str,
                    metavar='TEST SLIDE PATH', help='Test slide path. ')

parser.add_argument('--test_patch_path', default=config.PATCH_FOR_HEATMAP_test, type=str,
                    metavar='test PATCH PATH', help='test patch path. ')
parser.add_argument('--test_patch_dir', default=config.PATCH_FOR_HEATMAP_test_DIR, type=str,
                    metavar='test PATCH DIR', help='test patch path DIR. ')

parser.add_argument('--level', default=4, type=int,
                    metavar='LEVEL', help='The level. ')
parser.add_argument('--wsis_list_start', default=0, type=int,
                    metavar='WSIS LIST START', help='Select the wsis to get patches for getting heatmaps. ')
parser.add_argument('--wsis_list_end', default=2, type=int,
                    metavar='WSIS LIST END', help='Select the wsis to get patches for getting heatmaps. ')
parser.add_argument('--stride', default=1, type=int, metavar='STRIDE',
                    help='Get patches stride. ')



# for test slide
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
    select_test_wsi_path = test_wsi_paths[args.wsis_list_start:args.wsis_list_end]  # change the file for get patches

    print('Total get patches for heatmaps in\n%s' % (select_test_wsi_path))
    for normal_wsi_path in select_test_wsi_path:
        if os.path.split(normal_wsi_path)[-1].split('.')[0] not in os.listdir(args.test_patch_dir):
            ops.extract_patches_from_slide_and_mask_for_heatmap(normal_wsi_path,
                                                                maskdir,
                                                                '_tissue_mask.png',
                                                                level,
                                                                stride=args.stride,
                                                                tumor_patch=False,
                                                                normal_patch_path=args.test_patch_path)

if __name__ == '__main__':

    args = parser.parse_args()

    level = args.level

    #for testset, as we know. testset dont have tumor mask.
    #slide path
    test_wsi_paths = args.test_wsis
    maskdir = args.mask_path

    #get tissue mask from test slide
    get_test_tissue_mask(test_wsi_paths, level, maskdir)

    # get consecutive patch
    get_consecutive_patch(test_wsi_paths, level, maskdir)