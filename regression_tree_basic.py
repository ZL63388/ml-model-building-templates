##############################################################################
# IMPORT REQUIRED PACKAGES
##############################################################################

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



##############################################################################
# IMPORT SAMPLE DATA
##############################################################################

my_df = pd.read_csv("sample_data_regression.csv")



##############################################################################
# SPLIT INPUT VARIABLES & OUTPUT VARIABLES
##############################################################################

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]



##############################################################################
# SPLIT OUT TRAINING & TEST SETS
##############################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



##############################################################################
# INSTANTIATE OUR MODEL
##############################################################################

regressor = DecisionTreeRegressor(min_samples_leaf = 7)



##############################################################################
# MODEL TRAINING
##############################################################################

regressor.fit(X_train, y_train)



##############################################################################
# MODEL ASSESSMENT
##############################################################################

y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)



##############################################################################
# A DEMONSTRATION OF OVERFITTING
##############################################################################

y_pred_training = regressor.predict(X_train)
r2_score(y_train, y_pred_training)


# pot decision tree
plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 24)