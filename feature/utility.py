__author__ = 'fucus'
import os
import skimage.io
import logging
from feature.hog import get_hog
from config import Project
import pickle


cache_path = "%s/cache" % Project.project_path
if not os.path.exists(cache_path):
    os.makedirs(cache_path)
hog_feature_cache_file_path = "%s/%s" % (cache_path, "hog_feature_cache.pickle")


def load_cache():
    # load cache
    hog_feature_cache = {}
    if os.path.exists(hog_feature_cache_file_path):
        hog_feature_file = open(hog_feature_cache_file_path, "rb")
        hog_feature_cache = pickle.load(hog_feature_file)
        hog_feature_file.close()

    return hog_feature_cache

def save_cache(hog_feature_cache):
    hog_feature_file = open(hog_feature_cache_file_path, "wb")
    pickle.dump(hog_feature_cache, hog_feature_file)
    hog_feature_file.close()

def load_test_feature(img_data_path, hog_feature_cache):
    return load_feature_from_range(img_data_path, hog_feature_cache, range(8, 11))


def load_train_feature(img_data_path, hog_feature_cache):
    return load_feature_from_range(img_data_path, hog_feature_cache, range(1, 8))


def load_feature_from_range(img_data_path, hog_feature_cache, num_range):
    x_feature = []
    y = []
    relevant_image_path_list = []
    type_dir_list = ["s%s" % x for x in range(1, 41)]

    logging.info("start to check the train image")
    for dir in type_dir_list:
        images = ["%s.pgm" % x for x in num_range]
        for img in images:
            img_path = "%s/%s" % (dir, img)
            if not os.path.exists("%s/%s" % (img_data_path, img_path)):
                logging.error("img %s do not exist" % img_path)
            else:
                relevant_image_path_list.append(img_path)
                y.append(dir)

    logging.info("check the train image end")

    logging.info("start to load feature from train image")
    for relevant_image_path in relevant_image_path_list:
        full_path = "%s/%s" % (img_data_path, relevant_image_path)
        x_feature.append(extract_feature(full_path, hog_feature_cache))
    logging.info("load feature from train image end")

    return relevant_image_path_list, x_feature, y

def extract_feature(img_path, hog_feature_cache):
    img_name = "/".join(img_path.split("/")[-2:])
    img = skimage.io.imread(img_path)
    feature = []
    if img_name in hog_feature_cache:
        hog_feature = hog_feature_cache[img_name]
    else:
        hog_feature = get_hog(img)
        hog_feature_cache[img_name] = hog_feature
    feature += hog_feature
    return feature