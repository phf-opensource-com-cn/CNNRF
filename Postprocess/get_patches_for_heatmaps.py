# utf-8
# Author: ilikewind



from util_defined import config, hp
from Preprocess import ops

import argparse

parser = argparse.ArgumentParser(description='get patches for heatmaps')
parser.add_argument('--mask_path', default=config.TISSUE_MASK_DIR, type=str,
                    metavar='MASK PATH', help='Save the mask path. ')
parser.add_argument('--test_wsis', default=config.TEST_SLIDE_DIR, type=str,
                    metavar='TEST SLIDE PATH', help='Test slide path. ')

parser.add_argument('--normal_path', default=config.RAW_NORMAL_DATA_DIR, type=str,
                    metavar='NORMAL PATH', help='normal slide path. ')
parser.add_argument('--tumor_path', default=config.RAW_TUMOR_DATA_DIR, type=str,
                    metavar='TUMOR PATH', help='tumor slide path. ')
parser.add_argument('--trainset_mask', default=config.SLIDE_MAP_AND_MASK, type=str,
                    metavar='TRAINSET MASK PATH', help='Train set mask path. ')

parser.add_argument('--normal_patch_path', default=config.PATCH_FOR_HEATMAP_normal, type=str,
                    metavar='NORMAL PATCH PATH', help='normal patch path. ')
parser.add_argument('--tumor_patch_path', default=config.PATCH_FOR_HEATMAP_tumor, type=str,
                    metavar='NORMAL PATCH PATH', help='normal patch path. ')

parser.add_argument('--level', default=hp.level, type=int,
                    metavar='LEVEL', help='The level. ')
parser.add_argument('--wsis_list_start', default=0, type=int,
                    metavar='WSIS LIST START', help='Select the wsis to get patches for getting heatmaps. ')
parser.add_argument('--wsis_list_end', default=1, type=int,
                    metavar='WSIS LIST END', help='Select the wsis to get patches for getting heatmaps. ')
parser.add_argument('--stride', default=10000, type=int, metavar='STRIDE',
                    help='Get patches stride. ')


'''
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
'''

def get_consecutive_patch(normal_path, tumor_path, level, maskdir):
    # test_wsi_paths = ops.get_normal_wsi_path(path)
    # select_test_wsi_path = test_wsi_paths[args.wsis_list_start:args.wsis_list_end]  # change the file for get patches
    ######### select slide to get the heatmaps ##########
    normal_wsi_paths = ops.get_normal_wsi_path(normal_path)
    tumor_wsi_paths, _ = ops.get_tumor_wsi_path(tumor_path)
    select_normal_wsi_path = normal_wsi_paths[args.wsis_list_start:args.wsis_list_end]
    select_tumor_wsi_path = tumor_wsi_paths[args.wsis_list_start:args.wsis_list_end]

    '''
    extract patches from normal slide
    '''
    print('Total get patches for heatmaps in\n%s' % (select_normal_wsi_path))
    for normal_wsi_path in select_normal_wsi_path:
            ops.extract_patches_from_slide_and_mask_for_heatmap(normal_wsi_path,
                                                                maskdir,
                                                                '_tissue_mask.png',
                                                                level,
                                                                stride=args.stride,
                                                                tumor_patch=False,
                                                                normal_patch_path=args.normal_patch_path,
                                                                tumor_patch_path=args.tumor_patch_path)
    '''
    extract patches from tumor slide
    '''
    print('Total get patches for heatmaps in \n%s' % select_tumor_wsi_path)
    for tumor_wsi_path in select_tumor_wsi_path:
        # extract normal patches from tumor slide
        ops.extract_patches_from_slide_and_mask_for_heatmap(tumor_wsi_path, maskdir,
                                                            '_normal_mask.png', level,
                                                            stride=args.stride,tumor_patch=False,
                                                            normal_patch_path=args.normal_patch_path,
                                                            tumor_patch_path=args.tumor_patch_path)
        # extract tumor patches from tumor slide
        ops.extract_patches_from_slide_and_mask_for_heatmap(tumor_wsi_path, maskdir,
                                                            '_tumor_mask.png', level,
                                                            stride=args.stride,tumor_patch=True,
                                                            normal_patch_path=args.normal_patch_path,
                                                            tumor_patch_path=args.tumor_patch_path)


if __name__ == '__main__':

    args = parser.parse_args()

    level = args.level
    '''
    #####################################################
    for testset, as we know. testset dont have tumor mask.
    # slide path
    # test_wsi_paths = args.test_wsis
    # maskdir = args.mask_path
    # get tissue mask from test slide
    get_test_tissue_mask(test_wsi_paths, level, maskdir)
    #####################################################
    '''
    # trainset slide path
    normal_wsi_path = args.normal_path
    tumor_wsi_path = args.tumor_path
    trainset_mask = args.trainset_mask


    # get consecutive patch
    get_consecutive_patch(normal_wsi_path, tumor_wsi_path, level, trainset_mask)