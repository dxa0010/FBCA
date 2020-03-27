import sys 
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import itertools

class Oracle_Batch ():

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

        self.select_pool()

        return self.scores, self.oracle_index

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
        validation_score = []
        seq = [x for x in range(len(self.X_pool))]
        index_combi = list(itertools.combinations(seq,self.aq_num))
        for i,x_comb in enumerate(index_combi):
            print(i/len(index_combi) * 100, " % : complete\n") 
            x_combi = list(x_comb)
            in_p = self.X_pool[x_combi]
            in_p_y = self.y_pool[x_combi]
            X_train= np.vstack((self.X_train, in_p))
            y_train= np.append(self.y_train, in_p_y)
            rfc = self.classifier.fit(X_train, y_train) 
            validation_score.append(rfc.score(self.X_test, self.y_test))

        prime_index = np.argmax(np.array(validation_score))
        prime_combi = list(index_combi[prime_index])

        score = np.max(np.array(validation_score))
        self.scores.append(score)
        self.oracle_index = prime_combi