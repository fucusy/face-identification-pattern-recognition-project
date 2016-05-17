# re-idenficate human face
A project built for pattern recoginition course to re-identificate human face

# 环境配置

## environment

* python 3.5
* python library:
    in the requirements.md
    to install pip packages use : `pip3.5 install -r pip_packages.txt`
* Linux or unix System, may be windows

## data


the data is almost 4.7M, contains 40 persons face image, 10 images for every person, totally 400 images. 

you can download it from [here](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)

or contact fucus@qq.com to get one
    

    
## how to run
1. move `config.tmpl.py` to `config.py`, you can use *unix command line `mv config.tmpl.py config.py`, and update the variable in this file 
2. run the `main.py` by `python3.5 main.py`


## evaluation

training set: 70%
test set: 30%

recall and precision

#Idea



#实验结果记录

| submit date | name      | off f1-score    |   compare   |feature                  | model   | other trick                                   | comments                |
| ----------  |--------   |----             | ------------|-------------------------|---------|-----------------------------------------------|----------               |
| 2016-05-16  | chenqiang |   0.95          |   0         |   hog                   | forest  | best param, {'n_estimators': 200}             |                         |
