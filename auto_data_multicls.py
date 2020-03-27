import numpy as np
from sklearn.datasets import make_classification

X_train, Y_train = make_classification(n_samples = 10000,
                            random_state=12,
                            n_features=30, 
                            n_redundant=2, 
                            n_informative=6,
                            n_clusters_per_class=2,
                            n_classes=10)

X_test, Y_test = make_classification(n_samples = 10000,
                            random_state=12,
                            n_features=50, 
                            n_redundant=2, 
                            n_informative=7,
                            n_clusters_per_class=2,
                            n_classes=20)

np.save("data/X_10_class_30_feature", X_train)
np.save("data/X_20_class_50_feature", X_test)

np.save("data/y_10_class_30_feature", Y_train)
np.save("data/y_20_class_50_feature", Y_test)
