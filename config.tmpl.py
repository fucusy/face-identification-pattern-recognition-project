__author__ = 'fucus'
import model.model as model


class Project:

    # required, this path contain the train image in sub-folder, the there are ten sub-folders, s1, s2, s3, s4 .... s40
    train_img_folder_path = "/Users/fucus/Documents/buaa/PR/face_identification_homework/data/att_faces"

    # required, your project's absolute path, in other way, it's the absolute path for this file
    project_path = "/Users/fucus/Documents/buaa/PR/face_identification_homework/code"

    # not required, a img path for you exercise program
    test_img_example_path = "/Users/fucus/Documents/buaa/PR/face_identification_homework/data/att_faces/s1/1.pgm"

    # required, predict model
    predict_model = model.RandomForestClassification()

    # required, result output path
    result_output_path = "./result/"

    # required, save cache or not
    save_cache = True
