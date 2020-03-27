import sys
import os
import gym
from gym import spaces
from gym.spaces import Discrete, Box
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process.kernels import RBF



class RAL_Multi_Class_Env(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        # print("2________________________________________________")
        np.random.seed(seed=123)
        # action_space, observation_space, reward_range を設定する
        self.action_space = Box(low=np.array([0.0,0.0]), high=np.array([1.0, 1.0]) , dtype=np.float32) 
        # oob_score, pca_1~6, rf.node, rf.left.node, rf.right.node
        self.observation_space = Box(low=np.array([0,0,0,0,0,0,0,0,0,0]),
            high=np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,15.0,15.0,15.0]), dtype=np.float32) 
        self.reward_range = [-10., 100.]
        self.weight_num = ""
        self.aq_num = 20
        self.epsode_num = 0
        self.seed_df = 2
        self.class_num = 10
        self.name = "ral_multi_model_log"
        self.reset()


    def step(self, action): # actionを実行し、結果を返す
        observation, reward = self.get_observe(action)
        done = self.is_done()
        if done:
            result_log_name =  os.path.join("result","{}_white_episode_{ep}_weight_{wg}.csv").format(self.name, ep=self.epsode_num, wg = self.seed_df)
            np.savetxt(result_log_name, self.mse_test_list, fmt='%s', delimiter=',')

        return observation, reward, done, {}

    def reset(self): # 状態を初期化し、初期の観測値を返す
        self.epsode_num = self.epsode_num + 1
        asset = os.path.join("/study","RAL","data")
        df_name =  "_ral_multi_class_{ds}_{ep}.csv".format(ds = "white", ep = self.seed_df)
        data_dir_train =  pd.read_csv(os.path.join(asset, "df_train" + df_name))
        data_dir_test = pd.read_csv(os.path.join(asset, "df_test" + df_name))
        data_dir_pool = pd.read_csv(os.path.join(asset, "df_pool" + df_name))
        label = 'quality'
        df_test = data_dir_test
        self.X_test = df_test.iloc[:, :-1].values
        self.y_test = df_test.loc[:, label].values
        df_train = data_dir_train
        self.X_train= df_train.iloc[:, :-1].values
        self.y_train= df_train.loc[:, label].values
        df_pool = data_dir_pool
        self.X_pool = df_pool.iloc[:, :-1].values
        self.y_pool = df_pool.loc[:, label].values
        self.get_num = 0
        
        self.mse_test_list = np.array([])
        # df_name = "_ral_multi_class_white_{}.csv".format(self.epsode_num)
        # df_train_name = os.path.join("data","df_train" + df_name)
        # df_train.to_csv(df_train_name, index=False)
        # df_test_name = os.path.join("data","df_test" + df_name)
        # df_test.to_csv(df_test_name, index=False)
        # df_pool_name = os.path.join("data","df_pool"+df_name)
        # df_pool.to_csv(df_pool_name, index=False)

        if self.epsode_num > 2:
            self.X_test, self.y_test, self.X_train, self.y_train, self.X_pool, self.y_pool = self._X_test, self._y_test, self._X_train, self._y_train, self._X_pool, self._y_pool

        self.prob_pool, self.voting_pool, self.score, self.feature = self.make_model()
        observation = self.feature

        return observation

    def render(self, mode='human', close=False): # 環境を可視化する
        outfile = sys.stdout
        outfile.write("\r{0}\n".format(self.score))
        self.mse_test_list = np.append(self.mse_test_list, self.score)
        return outfile
    
    def close(self): # 環境を閉じて後処理をする
        pass

    def seed(self, seed=None): # ランダムシードを固定する
        pass
    
    def get_observe(self, action):
        pre_score = self.score
        # print(self.prob_pool.shape)
        # print(self.voting_pool.shape)
        pool_info = np.stack([self.prob_pool, self.voting_pool],axis=1)
        x_prime = np.argmin(np.power(pool_info-action,2).sum(axis=1))
        pool_size = len(self.X_pool)
        idx_rm = np.ones(pool_size, dtype=bool)
        idx_rm[x_prime] = False
   
        in_p, self.X_pool = self.X_pool[x_prime], self.X_pool[idx_rm]
        in_p_y, self.y_pool = self.y_pool[x_prime], self.y_pool[idx_rm]

        self.X_train= np.vstack((self.X_train, in_p))
        self.y_train= np.append(self.y_train, in_p_y)
        
        self.prob_pool, self.voting_pool, self.score, self.feature = self.make_model()
        
        gain = (self.score - pre_score)
        observation = self.feature
        self.get_num = self.get_num + 1
        # print(observation)

        # assert self.observation_space.contains(observation)

        return observation, gain

    def is_done(self):
        done = (self.get_num>self.aq_num) 
        self._X_test, self._y_test, self._X_train, self._y_train, self._X_pool, self._y_pool = self.X_test, self.y_test, self.X_train, self.y_train, self.X_pool, self.y_pool
        if self.epsode_num ==6:
            self.epsode_num = 1
        return done

    def make_model(self):
        
        X_train = self.X_train
        y_train = self.y_train 
        rfc = RandomForestClassifier(n_estimators=500, max_depth=6, 
        oob_score= True, random_state=0).fit(X_train, y_train) 
        for i in range(rfc.n_estimators):
            pred_dectree =  rfc.estimators_[i].predict(self.X_pool)
            if i == 0: 
                cl_var = pd.get_dummies(pred_dectree)
            else:
                cl_var = cl_var + pd.get_dummies(pred_dectree)
        cl_var = (cl_var/rfc.n_estimators).max(axis=1)

        prob_pool = (rfc.predict_proba(self.X_pool)).max(axis=1)
        voting_pool = cl_var
        score = rfc.score(self.X_test, self.y_test)
        oob_score = rfc.oob_score_
        pca_ft = self.pca_feature(self.X_train)
        rf_ft = self.random_forest_feature(rfc)
        obs = np.hstack((oob_score, pca_ft, rf_ft)) 
        # print(obs.shape)

        return prob_pool, voting_pool, score, obs

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

