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



class LalClEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        # action_space, observation_space, reward_range を設定する
        self.action_space = Discrete(10) 
        # oob_score, 
        self.observation_space = Box(low=np.array([0,]), high=np.array([1.0,]), dtype=np.float32) 
        self.reward_range = [-10., 100.]
        self.epsode_num = 0
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
        test_num = 1000
        train_num = 100
        df = np.load(os.path.join("data","X_auto_comp.npy"))
        df = pd.DataFrame(df)
        df['Y'] = np.load(os.path.join("data","y_auto_comp_C.npy"))
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
        

        self.mse_test_list = np.array([])
        df_name = os.path.join("data","df_test_" + str(self.epsode_num)+ ".csv")
        self.df.to_csv(df_name, index=False)

        self.prob_pool, self.variance_pool, self.score, self.oob_score = self.make_model()
        observation = self.oob_score

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
        pool_size = len(self.prob_pool)
        pool_unit = pool_size // (5)
        idx = np.argsort(np.amin(abs(self.prob_pool), axis=1))[::-1]
        idx_rm = np.ones(pool_size, dtype=bool)

        idx_var = np.argsort(self.variance_pool)

        if action < 5:
            x_prime = idx[action*pool_unit]
            idx_rm[(idx[action*pool_unit])] = False
        else:
            x_prime = idx_var[(action-5)*pool_unit]
            idx_rm[(idx_var[(action-5)*pool_unit])] = False

        in_p, self.X_pool = self.X_pool[x_prime], self.X_pool[idx_rm]
        in_p_y, self.y_pool = self.y_pool[x_prime], self.y_pool[idx_rm]

        self.X_train= np.vstack((self.X_train, in_p))
        self.y_train= np.append(self.y_train, in_p_y)
        
        self.prob_pool, self.variance_pool, self.score, self.oob_score = self.make_model()
        
        gain = (self.score - pre_score)
        observation = self.oob_score
        # print(observation)

        # assert self.observation_space.contains(observation)

        return observation, gain

    def is_done(self):
        done = (len(self.X_train)>=200) 
        return done

    def make_model(self):
        
        X_train = self.X_train
        y_train = self.y_train 
        rfc = RandomForestClassifier(n_estimators=100, max_depth=2, 
        oob_score= True, random_state=0).fit(X_train, y_train) 
        cl_var = [0] * len(self.X_pool)
        for i in range(rfc.n_estimators):
            cl_var = cl_var + rfc.estimators_[i].predict(self.X_pool)
        
        prob_pool = rfc.predict_proba(self.X_pool)
        variance_pool = np.abs(cl_var/rfc.n_estimators - 0.5)
        score = rfc.score(self.X_test, self.y_test)
        oob_score = rfc.oob_score_

        return prob_pool, variance_pool, score, oob_score 