import os
import numpy as np
import pandas as pd
from oracle import Oracle_Batch
from random_ral import Random
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(
        n_estimators=500, max_depth=6, oob_score= True, random_state=0
        )
data_sets = ["trip"]

episode = [2,3,4,5,6]
labels = ['Category 1']

result_dir = os.path.join("/study","RAL","result")
oracle_scores=[]
np.random.seed(seed=12)

for i, data_set in enumerate(data_sets):
        ds_oracle_score = []
        ds_random_score = []

        for ep in episode:
                asset = os.path.join("/study","RAL","data")
                df_name =  "_ral_compare_oracle_trip_{}.csv".format(ep)
                data_dir_train =  pd.read_csv(os.path.join(asset, "df_train" + df_name))
                data_dir_test = pd.read_csv(os.path.join(asset, "df_test" + df_name))
                data_dir_pool = pd.read_csv(os.path.join(asset, "df_pool" + df_name))
                label = labels[i]
                
                # score_oracle, index_oracle = Oracle_Batch(data_dir_train, data_dir_test, data_dir_pool, label, 5, rfc).calc_loss()
                # np.savetxt(os.path.join(result_dir,"oracle_{ds}_episode_{ep}_index.csv".format(ds=data_set, ep = ep)) , np.array(index_oracle), delimiter=",")
                # ds_oracle_score.append(score_oracle)

                score_random, index_random = Random(data_dir_train, data_dir_test, data_dir_pool, label, 5, rfc).calc_loss()
                ds_random_score.append(score_random)
                np.savetxt(os.path.join(result_dir,"oracle_random_{ds}_episode_{ep}_index.csv".format(ds=data_set, ep = ep)) , np.array(index_random), delimiter=",")

        # oracle_scores.append(np.mean(np.array(ds_oracle_score),axis=0))

        # np.savetxt(os.path.join(result_dir,"oracle_{}_score.csv".format(data_set)), np.mean(np.asarray(ds_oracle_score),axis=0), delimiter=",")
        np.savetxt(os.path.join(result_dir,"oracle_random_{}_score.csv".format(data_set)), np.mean(np.asarray(ds_random_score),axis=0), delimiter=",")  
        np.savetxt(os.path.join(result_dir,"oracle_random_{}_std.csv".format(data_set)), np.std(np.asarray(ds_random_score),axis=0), delimiter=",")  
