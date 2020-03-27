import sys 
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class Uncertainty_Sampling ():

    def __init__(self, df_train, df_test, df_pool, label, aq_num, classifier):
    
        self.df_train = df_train
        self.df_test = df_test
        self.df_pool = df_pool
        self.label = label
        self.aq_num = aq_num
        self.scores = []
        # classifier is only randomforest
        self.classifier = classifier

    def calc_loss(self):

        self.read_data()

        for i in range(self.aq_num):
        
            self.select_pool()

        return self.scores

    def read_data(self):

        df_test = self.df_test
        self.X_test = df_test.loc[:,df_test.columns[df_test.columns != self.label]].values
        self.y_test = df_test.loc[:, self.label].values
        df_train = self.df_train
        self.X_train= df_train.loc[:,df_train.columns[df_train.columns != self.label]].values
        self.y_train= df_train.loc[:, self.label].values
        df_pool = self.df_pool
        self.X_pool = df_pool.loc[:,df_pool.columns[df_pool.columns != self.label]].values
        self.y_pool = df_pool.loc[:, self.label].values

        rfc = self.classifier.fit(self.X_train, self.y_train)

        score = rfc.score(self.X_test, self.y_test)
        self.scores.append(score)

    def select_pool(self):

        rfc = self.classifier.fit(self.X_train, self.y_train) 
        prob_cl = (rfc.predict_proba(self.X_pool)).max(axis=1)
        x_prime = np.argmin(prob_cl)
        idx_rm = np.ones(len(self.X_pool), dtype=bool)
        idx_rm[x_prime] = False
        in_p, self.X_pool = self.X_pool[x_prime], self.X_pool[idx_rm]
        in_p_y, self.y_pool = self.y_pool[x_prime], self.y_pool[idx_rm]
        self.X_train= np.vstack((self.X_train, in_p))
        self.y_train= np.append(self.y_train, in_p_y)

        rfc = self.classifier.fit(self.X_train, self.y_train)

        score = rfc.score(self.X_test, self.y_test)
        self.scores.append(score)