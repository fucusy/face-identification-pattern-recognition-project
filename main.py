__author__ = 'fucus'


import logging
import sys
import datetime
from config import Project
from feature.utility import load_train_feature
from feature.utility import load_cache
from feature.utility import save_cache


hog_feature_cache = {}

if __name__ == '__main__':
    LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}
    level = logging.INFO
    if len(sys.argv) >= 2:
        level_name = sys.argv[1]
        level = LEVELS.get(level_name, logging.INFO)
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    train_num = -1

    start_time = datetime.datetime.now()
    logging.info('start program---------------------')
    logging.info("loading feature cache now")
    hog_feature_cache = load_cache()
    logging.info("load feature cache end")

    logging.info("load train data feature now")
    train_img_relevant_paths, train_x_feature, train_y = load_train_feature(Project.train_img_folder_path, hog_feature_cache, train_num)
    logging.info("extract train data feature done")

    logging.info("start to train the model")
    Project.predict_model.fit(x_train=train_x_feature, y_train=train_y)
    logging.info("train the model done")

    if Project.save_cache:
        logging.info("saving feature cache now")
        save_cache(hog_feature_cache)
        logging.info("save feature cache end")
    else:
        logging.info("skip saving the feature cache")

    del hog_feature_cache

    end_time = datetime.datetime.now()
    logging.info('total running time: %.2f second' % (end_time - start_time).seconds)
    logging.info('end program---------------------')