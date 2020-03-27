import sys
import gym
import gym.spaces
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class Random_AL:

    def __init__(self, df):
        self.df = df
        test_num = 1000
        train_num = 100
        self.get_num =100
        df_test = df[:test_num]
        self.X_test = df_test.iloc[:, :-1].values
        self.y_test = df_test.loc[:, 'Y'].values
        self.df_train = np.arange(test_num, (test_num + train_num))
        self.X_pool = np.arange((test_num + train_num), len(df))
        self.mse_train, self.mse_test, self.r2_train, self.r2_test = self.make_model()

    def execute(self):
        result_log = np.array([])
        for i in range(self.get_num):
            result_log =  np.append(result_log, self.step())
        return result_log

    def step(self):
        in_p, self.X_pool = self.X_pool[0], self.X_pool[1:]
        self.df_train = np.append(self.df_train, in_p)
        self.mse_train, self.mse_test, self.r2_train, self.r2_test = self.make_model()

        return self.mse_test

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

if __name__ == '__main__':
    df_names = ["../data/df_test_1.csv", "../data/df_test_2.csv", "../data/df_test_3.csv", "../data/df_test_4.csv", "../data/df_test_5.csv"]
    results = pd.DataFrame([])
    for i, dn in enumerate(df_names):
        df = pd.read_csv(dn)
        rnal = Radom_AL(df)
        results[str(i)] = rnal.execute()
    results.to_csv("../result/random_al_comp.csv", index=False)