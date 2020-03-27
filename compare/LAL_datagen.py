import sys 
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.decomposition import PCA



class LAL_Data_Gen ():

    def __init__(self, tau, aq_size):
        self.tau = tau
        self.aq_size = aq_size
        self.dataset_name_X = os.path.join("data","X_10_class_8_feature_10000_n_sample.npy")
        self.dataset_name_y = os.path.join("data","y_10_class_8_feature_10000_n_sample.npy")
        self.rfc_gains = [0]

    def gen_g(self):

        for i in range(self.tau):

            self.data_gen_tau(i)
            self.data_gen_q(i)
            rfc_gain = RandomForestRegressor(
                n_estimators=500, max_depth=10, oob_score= True, random_state=0
                ).fit(self.gain_data_X, self.gains) 
            self.save_data(rfc_gain)

        return rfc_gain

    def save_data(self, rfc_gain):

        self.gain_data_X.to_csv(os.path.join("/study","RAL","data","lal_gain_x.csv"), index=False)
        gain_y = pd.DataFrame({'gain':self.gains})
        gain_y.to_csv(os.path.join("/study","RAL","data","lal_gain_y.csv"), index=False)
        joblib.dump(rfc_gain, os.path.join("/study","RAL",'model','lal_gain.pkl'))


    def data_gen_tau(self, tau):

        df = np.load(self.dataset_name_X)
        df = pd.DataFrame(df)
        test_num = 1000
        train_num = 100
        df['Y'] = np.load(self.dataset_name_y)
        df = df.sample(frac=1).reset_index(drop=True)
        df_test = df[:test_num]
        self.X_test = df_test.iloc[:, :-1].values
        self.y_test = df_test.loc[:, 'Y'].values
        df_train = df[test_num:(test_num + train_num)]
        self.X_train= df_train.iloc[:, :-1].values
        self.y_train= df_train.loc[:, 'Y'].values
        df_pool = df[(test_num + train_num):]
        self.X_pool = df_pool.iloc[:, :-1].values
        self.y_pool = df_pool.loc[:, 'Y'].values

        for i in range((tau+1)):

            if i==0:
                break

            pred_gain = self.rfc_gains[i].predict(self.X_pool)
            x_prime = np.argsort(-pred_gain)[:10]
            idx_rm[x_prime] = False
            in_p, self.X_pool = self.X_pool[x_prime], self.X_pool[idx_rm]
            in_p_y, self.y_pool = self.y_pool[x_prime], self.y_pool[idx_rm]
            self.X_train= np.vstack((self.X_train, in_p))
            self.y_train= np.append(self.y_train, in_p_y)

    def data_gen_q(self, i):

        x_random = np.random.randint(0, (len(self.X_pool)-1), self.aq_size)
        prob_pool, voting_pool, obs, scores = self.gen_parameta(x_random)
        _score = []

        for idx in x_random:
            X_train = np.vstack((self.X_train, self.X_pool[idx]))
            y_train = np.append(self.y_train, self.y_pool[idx])
            rfc = RandomForestClassifier(
                    n_estimators=500, max_depth=10, oob_score= True, random_state=0
                    ).fit(X_train, y_train) 
            _score.append(rfc.score(self.X_test, self.y_test))
        
        print(i)
        gain = np.array(_score) - scores
        gen_data_X = pd.DataFrame(
            {'prob_dp':prob_pool,'voting_dp':voting_pool,'oob_cl':obs[0],
            'pca_1_cl':obs[1],'pca_2_cl':obs[2],'pca_3_cl':obs[3],'pca_4_cl':obs[4],'pca_5_cl':obs[5],'pca_6_cl':obs[6],
            'node_all_cl':obs[7],'node_rg_cl':obs[8],'node_lf_cl':obs[8]})

        if i == 0:
            self.gain_data_X = gen_data_X
            self.gains = gain
        else:
            self.gain_data_X = pd.concat([self.gain_data_X, gen_data_X])
            self.gains = np.append(self.gains, gain)

        print(gen_data_X.shape, gain.shape)

        rfc_gain = RandomForestRegressor(
                    n_estimators=500, max_depth=10, oob_score= True, random_state=0
                    ).fit(gen_data_X, gain) 

        self.rfc_gains.append(rfc_gain)

    def gen_parameta(self, x_random):
        
        X_train = self.X_train
        y_train = self.y_train 
        rfc = RandomForestClassifier(n_estimators=500, max_depth=10, 
        oob_score= True, random_state=0).fit(X_train, y_train) 
        for i in range(rfc.n_estimators):
            pred_dectree =  rfc.estimators_[i].predict(self.X_pool[x_random])
            if i == 0: 
                cl_var = pd.get_dummies(pred_dectree)
            else:
                cl_var = cl_var + pd.get_dummies(pred_dectree)
        cl_var = (cl_var/rfc.n_estimators).max(axis=1)

        prob_pool = (rfc.predict_proba(self.X_pool[x_random])).max(axis=1)
        voting_pool = cl_var
        scores = rfc.score(self.X_test, self.y_test)
        oob_score = rfc.oob_score_
        pca_ft = self.pca_feature(self.X_train)
        rf_ft = self.random_forest_feature(rfc)
        obs = np.hstack((oob_score, pca_ft, rf_ft)) 
        # print(obs.shape)

        return prob_pool, voting_pool, obs, scores

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


if __name__ == '__main__':
    lal_gen = LAL_Data_Gen(90, 100)
    lal_g = lal_gen.gen_g() 