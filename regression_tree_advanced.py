##############################################################################
# IMPORT REQUIRED PACKAGES
##############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder



##############################################################################
# IMPORT SAMPLE DATA
##############################################################################

# import
data_for_model = pickle.load(open("data/abc_regression_modelling.p", "rb"))


# drop necessary columns
data_for_model.drop("customer_id", axis = 1, inplace = True)


# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)



##############################################################################
# DEAL WITH MISSING VALUES
##############################################################################

data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)



##############################################################################
# SPLIT INPUT VARIABLES & OUTPUT VARIABLES
##############################################################################

X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]



##############################################################################
# SPLIT OUT TRAINING & TEST SETS
##############################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



##############################################################################
# DEAL WITH CATEGORICAL VALUES
##############################################################################

categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)



##############################################################################
# MODEL TRAINING
##############################################################################

regressor = DecisionTreeRegressor(random_state = 42, max_depth = 4)
regressor.fit(X_train, y_train)



##############################################################################
# MODEL ASSESSMENT
##############################################################################

# predict on the test set
y_pred = regressor.predict(X_test)


# calculate r-squared
r_squared = r2_score(y_test, y_pred)
print(r_squared)


# cross validation (CV)
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()


# calculate adjusted r-squared
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)



##############################################################################
# A DEMONSTRATION OF OVERFITTING
##############################################################################

y_pred_training = regressor.predict(X_train)
r2_score(y_train, y_pred_training)



##############################################################################
# FINDING THE BEST MAX DEPTH
##############################################################################

max_depth_list = list(range(1,9))
accuracy_scores = []

for depth in max_depth_list:
    
    regressor = DecisionTreeRegressor(max_depth = depth, random_state = 42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
opitmal_depth = max_depth_list[max_accuracy_idx]


# plot of max depths
plt.plot(max_depth_list, accuracy_scores)
plt.scatter(opitmal_depth, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy by Max Depth \n Optimal Tree Depth: {opitmal_depth} (Accuracy: {round(max_accuracy, 4)}")
plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()


# plot model

plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 16)