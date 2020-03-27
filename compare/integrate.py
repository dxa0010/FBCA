import os
import numpy as np
import pandas as pd
from LAL_exe import LAL_Data_Selecter
from uncertainty_sampling import Uncertainty_Sampling
from query_by_committee import Query_By_Committee
from random_ral import Random
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(
        n_estimators=500, max_depth=6, oob_score= True, random_state=0
        )
# data_sets = ["car", "white", "red", "google", "trip"]
data_sets = ["adalt", "car", "white", "red", "google", "trip"]

episode = [2,3,4,5,6]
# labels = ['6', 'quality', 'quality', 'Category 1', 'Category 1']
labels = ['14', '6', 'quality', 'quality', 'Category 1', 'Category 1']

result_dir = os.path.join("/study","RAL","result")
us_scores=[]
qbc_scores=[]
lal_scores=[]
random_scores=[]

for i, data_set in enumerate(data_sets):
        ds_us_scores = []
        ds_qbc_scores = []
        ds_lal_scores = []
        ds_random_scores = []

        for ep in episode:
                asset = os.path.join("/study","RAL","data")
                df_name =  "_ral_multi_class_{ds}_{ep}.csv".format(ds = data_set, ep = ep)
                data_dir_train =  pd.read_csv(os.path.join(asset, "df_train" + df_name))
                data_dir_test = pd.read_csv(os.path.join(asset, "df_test" + df_name))
                data_dir_pool = pd.read_csv(os.path.join(asset, "df_pool" + df_name))
                label = labels[i]
                
                score_us = Uncertainty_Sampling(data_dir_train, data_dir_test, data_dir_pool, label, 100, rfc).calc_loss()
                score_qbc = Query_By_Committee(data_dir_train, data_dir_test, data_dir_pool, label, 100, rfc).calc_loss()
                score_lal = LAL_Data_Selecter(data_dir_train, data_dir_test, data_dir_pool, label, 100, rfc).calc_loss()
                score_random = Random(data_dir_train, data_dir_test, data_dir_pool, label, 100, rfc).calc_loss()

                ds_us_scores.append(score_us)
                ds_qbc_scores.append(score_qbc)
                ds_lal_scores.append(score_lal)
                ds_random_scores.append(score_random)

        us_scores.append(np.mean(np.asarray(ds_us_scores),axis=0))
        qbc_scores.append(np.mean(np.asarray(ds_qbc_scores),axis=0))
        lal_scores.append(np.mean(np.asarray(ds_lal_scores),axis=0))
        random_scores.append(np.mean(np.asarray(ds_random_scores),axis=0))

        np.savetxt(os.path.join(result_dir,"us_{}_include.csv".format(data_set)), np.mean(np.asarray(ds_us_scores),axis=0), delimiter=",")
        np.savetxt(os.path.join(result_dir,"qbc_{}_include.csv".format(data_set)), np.mean(np.asarray(ds_qbc_scores),axis=0), delimiter=",")
        np.savetxt(os.path.join(result_dir,"lal_{}_include.csv".format(data_set)), np.mean(np.asarray(ds_lal_scores),axis=0), delimiter=",")
        np.savetxt(os.path.join(result_dir,"random_{}_include.csv".format(data_set)), np.mean(np.asarray(ds_random_scores),axis=0), delimiter=",")
