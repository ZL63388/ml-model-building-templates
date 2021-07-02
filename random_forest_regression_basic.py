##############################################################################
# IMPORT REQUIRED PACKAGES
##############################################################################

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
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

regressor = RandomForestRegressor(random_state = 42, n_estimators = 1000)



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
# FEATURE IMPORTANCE
##############################################################################

regressor.feature_importances_

feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable", "feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)


plt.barh(feature_importance_summary["input_variable"], feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()