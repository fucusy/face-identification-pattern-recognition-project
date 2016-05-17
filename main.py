__author__ = 'fucus'


import logging
import sys
import datetime
from config import Project
from feature.utility import *
from feature.utility import load_cache
from feature.utility import save_cache
from sklearn.metrics import classification_report

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


    start_time = datetime.datetime.now()
    logging.info('start program---------------------')
    logging.info("loading feature cache now")
    hog_feature_cache = load_cache()
    logging.info("load feature cache end")

    logging.info("load train data feature now")
    train_img_relevant_paths, train_x_feature, train_y = load_train_feature(Project.train_img_folder_path,
                                                                            hog_feature_cache)
    logging.info("extract train data feature done")

    logging.info("start to train the model")
    Project.predict_model.fit(x_train=train_x_feature, y_train=train_y)
    logging.info("train the model done")

    test_img_relevant_paths, test_x_feature, test_y = load_test_feature(Project.train_img_folder_path,
                                                                            hog_feature_cache)

    logging.info("start to predict")
    predict = Project.predict_model.predict(test_x_feature)
    logging.info("predict end")

    report = classification_report(test_y, predict)
    logging.info("the cross validation report:\n %s" % report)

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