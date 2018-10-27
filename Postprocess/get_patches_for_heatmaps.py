# utf-8
# Author: ilikewind

import os
import numpy as np
import cv2
from PIL import Image
import openslide

from util_defined import config, hp
from Preprocess import ops

# extract patches from slide and mask.
def extract__consecutive_patches_use_slide_and_mask(slide_path, maskdir, mask_lastname, level):

    mask_name, mask_exist = ops.name_and_exist(slide_path, maskdir, mask_lastname)

    slide = openslide.OpenSlide(slide_path)
    mask = cv2.imread(os.path.join(maskdir, mask_name), 0)

    patches_start_points = ops.get_samples_of_patch_starting_points_with_stride(mask, stride=1000)

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
    for test_wsi_path in test_wsi_paths:
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
    for test_wsi_path in test_wsi_paths:
        extract__consecutive_patches_use_slide_and_mask(test_wsi_path, maskdir, '_tissue_mask.png',level)


if __name__ == '__main__':

    test_wsi_paths = config.TEST_SLIDE_DIR
    maskdir = config.TISSUE_MASK_DIR
    level = hp.level

    # get tissue mask from test slide
    get_test_tissue_mask(test_wsi_paths, level, maskdir)

    # get consecutive patch
    get_consecutive_patch(test_wsi_paths, level, maskdir)