# utf-8
# Author: ilikewind

import os
import argparse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from util_defined import config, hp

parser = argparse.ArgumentParser(description='Generator heatmap. ')
parser.add_argument('--tissue_mask_dir', default=config.TISSUE_MASK_DIR, type=str,
                    metavar='TISSUE MASK DIR', help='Tissue mask dir for the heatmap shape')
parser.add_argument('--probability_result', default=config.PATCH_FOR_HEATMAP_PROBABILITY_DIR, type=str,
                    metavar='PROBABILITY RESULT', help='Probability result. ')
parser.add_argument('--heatmap_path', default=config.HEATMAP_PATH, type=str,
                    metavar='HEATMAP PATH', help='Heatmap path. ')

if __name__ == '__main__':
    args = parser.parse_args()

    probability_results = os.listdir(args.probability_result) # get all probability results files
    for probability_result in probability_results:
        slide_name = probability_result.split('_')[0] # in order to choose
        tissue_mask_name = slide_name + "_tissue_mask.png" #
        tissue_mask = plt.imread(os.path.join(args.tissue_mask_dir, tissue_mask_name))
        heatmap_zeros = np.zeros(shape=tissue_mask.shape) #

        probability_csv = pd.read_csv(os.path.join(args.probability_result, probability_result))
        for i in range(len(probability_csv)):
            [h, w] = [int(probability_csv['file'][i].split('_')[0]),
                      int(probability_csv['file'][i].split('_')[-1].split('.')[0])]

            #heatmap_zeros[w][h] = round(probability_csv['probability'][i], 2)
            heatmap_zeros[w][h] = 1 if round(probability_csv['probability'][i], 2)>0.9 else 0

        heatmap_name = os.path.join(args.heatmap_path, slide_name+'_heatmap.png')
        plt.imsave(heatmap_name, heatmap_zeros)

        corlor = sns.heatmap(heatmap_zeros, cmap='Reds')
        corlor_name = os.path.join(args.heatmap_path, slide_name+'_corlor.jpg')
        plt.show()
        f = corlor.get_figure()
        # corlor.save(corlor_name)
        f.savefig(corlor_name)

