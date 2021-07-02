##############################################################################
# IMPORT REQUIRED PACKAGES
##############################################################################

import pandas as pd

from sklearn.linear_model import LinearRegression
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

regressor = LinearRegression()



##############################################################################
# MODEL TRAINING
##############################################################################

regressor.fit(X_train, y_train)


##############################################################################
# MODEL ASSESSMENT
##############################################################################

y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)