# utf-8
# Author: ilikewind

import os
import multiprocessing.pool
import warnings


def get_class_fnames(directory, classes=None, ):
    if not classes:
        classes = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                classes.append(subdir)
    num_classes = len(classes)
    class_indices = dict(zip(classes, range(len(classes))))

    pool = multiprocessing.pool.ThreadPool()
    '''
    设置默认的功能
    '''
    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'}
    follow_links = False

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

