import sys
import os
import gym
import gym.spaces
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score



class LalEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.MultiBinary(1) 
        self.observation_space = gym.spaces.MultiBinary(1) 
        self.reward_range = [-10., 100.]
        self.epsode_num = 0
        self.reset()


    def step(self, action): # actionを実行し、結果を返す
        observation, reward = self.get_observe(action)
        done = self.is_done()
        if done:
            result_log_name =  "result/AL_log_" + str(self.epsode_num) + ".csv"
            np.savetxt(result_log_name, self.mse_test_list, fmt='%s', delimiter=',')

        return observation, reward, done, {}

    def reset(self): # 状態を初期化し、初期の観測値を返す
        self.epsode_num = self.epsode_num + 1
        test_num = 1000
        train_num = 100
        df = np.load("data/X_auto_comp.npy")
        df = pd.DataFrame(df)
        df['Y'] = np.load("data/y_auto_comp.npy")
        self.df = df.sample(frac=1).reset_index(drop=True)
        df_test = df[:test_num]
        self.X_test = df_test.iloc[:, :-1].values
        self.y_test = df_test.loc[:, 'Y'].values
        self.df_train = np.arange(test_num, (test_num + train_num))
        df_pool = df[(test_num + train_num):]
        X_pool = df_pool.iloc[:, :-1].values
        pca = PCA(n_components=1)
        X_pool_pca = pca.fit(X_pool.T)
        med_1 = np.median(X_pool_pca.components_[0])
        self.X_pool_0 = np.where(X_pool_pca.components_[0] > med_1)[0] + (test_num + train_num)
        self.X_pool_1 = np.where(X_pool_pca.components_[0] <= med_1)[0] + (test_num + train_num)
        
        self.mse_test_list = np.array([])
        df_name = "data/df_test_" + str(self.epsode_num)+ ".csv"
        self.df.to_csv(df_name, index=False)

        self.mse_train, self.mse_test, self.r2_train, self.r2_test = self.make_model()

        observation = np.array([int(self.mse_train > 1.5)])
        return observation

    def render(self, mode='human', close=False): # 環境を可視化する
        outfile = sys.stdout
        outfile.write("\r{0},{1}\n".format(self.mse_test, self.r2_test))
        self.mse_test_list = np.append(self.mse_test_list, self.mse_test)
        return outfile
    
    def close(self): # 環境を閉じて後処理をする
        pass

    def seed(self, seed=None): # ランダムシードを固定する
        pass
    
    def get_observe(self, action):
        pre_mse_test = self.mse_test
        
        if action == 0: 
            in_p, self.X_pool_0 = self.X_pool_0[0], self.X_pool_0[1:]
            self.df_train = np.append(self.df_train, in_p)
            self.mse_train, self.mse_test, self.r2_train, self.r2_test = self.make_model()
        else: 
            in_p, self.X_pool_1 = self.X_pool_1[0], self.X_pool_1[1:]            
            self.df_train = np.append(self.df_train, in_p)
            self.mse_train, self.mse_test, self.r2_train, self.r2_test = self.make_model()
        gain = (pre_mse_test - self.mse_test)
        return np.array([int(self.mse_train > 1.5)]), gain

    def is_done(self):
        done = (len(self.df_train)>=200) 
        return done

    def make_model(self):
        df = self.df.iloc[self.df_train]
        X_train = df.iloc[:, :-1].values
        y_train = df.loc[:, 'Y'].values    
        mod = xgb.XGBRegressor()
        mod.fit(X_train, y_train)   
        y_train_pred = mod.predict(X_train)
        y_test_pred = mod.predict(self.X_test)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(self.y_test, y_test_pred)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(self.y_test, y_test_pred)

        return mse_train, mse_test, r2_train, r2_test