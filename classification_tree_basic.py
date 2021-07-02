##############################################################################
# IMPORT REQUIRED PACKAGES
##############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



##############################################################################
# IMPORT SAMPLE DATA
##############################################################################

my_df = pd.read_csv("data/sample_data_classification.csv")



##############################################################################
# SPLIT INPUT VARIABLES & OUTPUT VARIABLES
##############################################################################

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]



##############################################################################
# SPLIT OUT TRAINING & TEST SETS
##############################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)



##############################################################################
# INSTANTIATE MODEL
##############################################################################

clf = DecisionTreeClassifier(random_state = 42, min_samples_leaf = 7)



##############################################################################
# MODEL TRAINING
##############################################################################

clf.fit(X_train, y_train)



##############################################################################
# MODEL ASSESSMENT
##############################################################################

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)



##############################################################################
# A DEMONSTRATION OF OVERFITTING
##############################################################################

y_pred_training = clf.predict(X_train)
accuracy_score(y_train, y_pred_training)


# Plot Decision Tree
plt.figure(figsize=(25,15))
tree = plot_tree(clf,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 24)