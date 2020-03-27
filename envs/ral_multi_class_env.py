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
        np.random.seed(seed=123)
        # action_space, observation_space, reward_range を設定する
        self.action_space = Box(low=np.array([0.0,0.0]), high=np.array([1.0, 1.0]) , dtype=np.float32) 
        # oob_score, pca_1~6, rf.node, rf.left.node, rf.right.node
        self.observation_space = Box(low=np.array([0,0,0,0,0,0,0,0,0,0]),
            high=np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,15.0,15.0,15.0]), dtype=np.float32) 
        self.reward_range = [-10., 100.]
        self.epsode_num = 0
        self.class_num = 10
        self.reset()


    def step(self, action): # actionを実行し、結果を返す
        observation, reward = self.get_observe(action)
        done = self.is_done()
        if done:
            result_log_name =  os.path.join("result","AL_log_Cl_" + str(self.epsode_num) + ".csv")
            # np.savetxt(result_log_name, self.mse_test_list, fmt='%s', delimiter=',')

        return observation, reward, done, {}

    def reset(self): # 状態を初期化し、初期の観測値を返す
        self.epsode_num = self.epsode_num + 1
        df = np.load(os.path.join("data","X_10_class_30_feature.npy"))
        df = pd.DataFrame(df)
        test_num = 1000
        all_num = len(df)
        train_num = np.random.randint(10, (all_num-test_num-100))
        df['Y'] = np.load(os.path.join("data","y_10_class_30_feature.npy"))
        self.df = df.sample(frac=1).reset_index(drop=True)
        df_test = df[:test_num]
        self.X_test = df_test.iloc[:, :-1].values
        self.y_test = df_test.loc[:, 'Y'].values
        df_train = df[test_num:(test_num + train_num)]
        self.X_train= df_train.iloc[:, :-1].values
        self.y_train= df_train.loc[:, 'Y'].values
        df_pool = df[(test_num + train_num):]
        self.X_pool = df_pool.iloc[:, :-1].values
        self.y_pool = df_pool.loc[:, 'Y'].values
        self.get_num = 0
        
        self.mse_test_list = np.array([])
        df_name = os.path.join("data","df_ral_multi_class" + str(self.epsode_num)+ ".csv")
        # self.df.to_csv(df_name, index=False)

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
        print(self.prob_pool.shape)
        print(self.voting_pool.shape)
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
        done = (self.get_num>=100) 
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

