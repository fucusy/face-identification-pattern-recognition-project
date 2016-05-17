#encoding=utf8

from sklearn.ensemble import RandomForestClassifier
from model.base_model import Model
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import classification_report
import logging



class RandomForestClassification(Model):

    def __init__(self):
        Model.__init__(self)
        self.model = RandomForestClassifier(n_jobs=-1, random_state=2016, verbose=1)

    def fit(self, x_train, y_train, need_transform_label=False):
        param_grid = {'n_estimators': [100, 200]}
        self.model = self.grid_search_fit_(self.model, param_grid, x_train, y_train)

    def predict(self, x_test, need_transform_label=False):
        return self.model.predict(x_test)