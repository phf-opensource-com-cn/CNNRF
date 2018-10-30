# utf-8
# Author: ilikewind

'''
in order to get the filename -- probability correspond. I have
read the keras code in github and copy the function for my aim.
the link:
https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image.py
https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
'''

import os
import multiprocessing.pool
import warnings

def get_class_fnames(directory, classes=None,
                     white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'},
                     follow_links=False):
    '''
    get the one by one order, for its correspond. such as predict1 -- file1
    :param directory:the base directory such as tumor0, and tumor0 contains tumor, normal subdirectory
    :param classes:if None, it will be [subdir1, subdir0, subdir2.....], else it will be classes=[tumor,normal]
    :param white_list_formats:the file format need feed to the iter
    :param follow_links:always False.
    :return:classes, filenames  ======  [0,1,1,0,1,0,0,1], [normal.png, tumor.png, tumor.png, ..]
    '''
    if not classes:
        classes = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                classes.append(subdir)
    num_classes = len(classes)
    class_indices = dict(zip(classes, range(len(classes))))

    pool = multiprocessing.pool.ThreadPool()

    results = []
    filenames = []
    i = 0
    for dirpath in (os.path.join(directory, subdir) for subdir in classes):
        results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                        (dirpath, white_list_formats,
                                         class_indices, follow_links)))
    classes_list = []
    filenames_list = []
    for res in results:
        classes, filenames = res.get()
        classes_list += classes
        filenames_list += filenames

    return classes_list, filenames_list


def _list_valid_filenames_in_directory(directory, white_list_formats,
                                       class_indices,
                                       follow_links):
    dirname = os.path.basename(directory)
    valid_files = _iter_valide_files(directory,
                                     white_list_formats, follow_links)

    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)
    return classes, filenames


def _iter_valide_files(directory, white_list_formats, follow_links):
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            for extension in white_list_formats:
                if fname.lower().endswith('.tiff'):
                    warnings.warn('Using \'.tiff\' files with multiple bands '
                                  'will cause distortion. '
                                  'Please verify your output.')
                if fname.lower().endswith('.' + extension):
                    yield root, fname

