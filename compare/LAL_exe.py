import sys 
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from LAL_datagen import LAL_Data_Gen
from sklearn.externals import joblib
from sklearn.decomposition import PCA

class LAL_Data_Selecter ():

    def __init__(self, df_train, df_test, df_pool, label, aq_num, classifier):
    
        self.df_train = df_train
        self.df_test = df_test
        self.df_pool = df_pool
        self.label = label
        self.aq_num = aq_num
        self.scores = []
        # classifier is only randomforest
        self.classifier = classifier
        self.classifier_g = joblib.load(os.path.join("/study","RAL",'model','lal_gain.pkl'))

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

        pred_gain = self.classifier_g.predict(self.gen_parameta().fillna(0))
        x_prime = np.argmax(-pred_gain)
        idx_rm = np.ones(len(self.X_pool), dtype=bool)
        idx_rm[x_prime] = False
        in_p, self.X_pool = self.X_pool[x_prime], self.X_pool[idx_rm]
        in_p_y, self.y_pool = self.y_pool[x_prime], self.y_pool[idx_rm]
        self.X_train= np.vstack((self.X_train, in_p))
        self.y_train= np.append(self.y_train, in_p_y)

        rfc = self.classifier.fit(self.X_train, self.y_train)

        score = rfc.score(self.X_test, self.y_test)
        self.scores.append(score)
    
    def gen_parameta(self):
        
        X_train = self.X_train
        y_train = self.y_train 
        rfc = self.classifier.fit(X_train, y_train) 
        for i in range(rfc.n_estimators):
            pred_dectree =  rfc.estimators_[i].predict(self.X_pool)
            if i == 0: 
                cl_var = pd.get_dummies(pred_dectree)
            else:
                cl_var = cl_var + pd.get_dummies(pred_dectree)
        cl_var = (cl_var/rfc.n_estimators).max(axis=1)

        prob_pool = (rfc.predict_proba(self.X_pool)).max(axis=1)
        voting_pool = cl_var
        oob_score = rfc.oob_score_
        pca_ft = self.pca_feature(self.X_train)
        rf_ft = self.random_forest_feature(rfc)
        obs = np.hstack((oob_score, pca_ft, rf_ft)) 
        # print(obs.shape)
        gen_data_X = pd.DataFrame(
            {'prob_dp':prob_pool,'voting_dp':voting_pool,'oob_cl':obs[0],
            'pca_1_cl':obs[1],'pca_2_cl':obs[2],'pca_3_cl':obs[3],'pca_4_cl':obs[4],'pca_5_cl':obs[5],'pca_6_cl':obs[6],
            'node_all_cl':obs[7],'node_rg_cl':obs[8],'node_lf_cl':obs[8]})

        return gen_data_X

    def pca_feature(self, data, n_components = 6):
        pca = PCA(n_components= n_components)
        pca_fit = pca.fit(data)
        pca_ratio = pca_fit.explained_variance_ratio_

        return pca_ratio

    def random_forest_feature(self, rf):
        num = rf.n_estimators
        full_node_n = np.array([])
        left_node_n = np.array([])
        right_node_n = np.array([])
        for i in range(num):
            clf = rf.estimators_[i].tree_
            full_node_n = np.append(full_node_n, clf.node_count)
            left_node_n = np.append(left_node_n, sum(clf.children_left>0))
            right_node_n = np.append(right_node_n, sum(clf.children_right>0))

        node_info = np.array([np.mean(full_node_n), np.mean(left_node_n), np.mean(right_node_n)])

        return node_info