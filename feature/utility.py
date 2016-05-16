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


def load_train_feature(img_data_path, hog_feature_cache, limit=-1):
    img_num_every_person = 10
    x_feature = []
    y = []
    relevant_image_path_list = []
    type_dir_list = ["s%s" % x for x in range(1, 41)]

    logging.info("start to check the train image")
    for dir in type_dir_list:
        images = sorted([x for x in os.listdir("%s/%s" % (img_data_path, dir)) if x.endswith(".pgm")])
        if len(images) != img_num_every_person:
            logging.warning("the type of %s train images number:%d is not equal to %d, it's incorrect"
                            % (dir, len(images), img_num_every_person))
        else:
            logging.info("the type of %s train images number:%d is equal to %d, it's correct"
                            % (dir, len(images), img_num_every_person))
        for img in images:
            img_path = "%s/%s" % (dir, img)
            relevant_image_path_list.append(img_path)
            y.append(dir)

    logging.info("check the train image end")

    logging.info("start to load feature from train image")
    count = 0
    for relevant_image_path in relevant_image_path_list:
        if count >= limit > 0:
            break
        if count % 1000 == 0:
            logging.info("extract %s th image feature now" % count)
        count += 1
        full_path = "%s/%s" % (img_data_path, relevant_image_path)
        x_feature.append(extract_feature(full_path, hog_feature_cache))
    logging.info("load feature from train image end")

    return relevant_image_path_list[:count], x_feature[:count], y[:count]

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