import numpy as np
import os
from sklearn.datasets import make_classification

class Make_Dataset():
    def __init__(self, n_sample, n_feature, n_classes):
        self.X_train, self.y_train = make_classification(n_samples = n_sample,
                            n_features=n_feature, 
                            n_redundant=0, 
                            n_informative=n_feature,
                            n_clusters_per_class=1,
                            n_classes=n_classes,
                            random_state=123)
    def save(self):
        name = str(n_classes) + "_class_" + str(n_feature) + "_feature_" + str(n_sample) + "_n_sample"
        np.save(os.path.join("data","X_"+name), self.X_train)
        np.save(os.path.join("data","y_"+name), self.y_train)

if __name__ == '__main__':
    Make_Dataset(10000, 6, 2).save()
    Make_Dataset(10000, 8, 10).save()
    # Make_Dataset(10000, 15, 20).save()
    Make_Dataset(10000, 7, 5).save()
    Make_Dataset(10000, 8, 10).save()
    Make_Dataset(10000, 10, 15).save()
    Make_Dataset(10000, 17, 25).save()
