##############################################################################
# IMPORT REQUIRED PACKAGES
##############################################################################

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



##############################################################################
# IMPORT SAMPLE DATA
##############################################################################

# import
data_for_model = pd.read_csv("data/sample_data_pca.csv")

# drop uneccessary columns
data_for_model.drop("user_id", axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

# class balance
data_for_model["purchased_album"].value_counts(normalize = True)



##############################################################################
# DEAL WITH MISSING VALUES
##############################################################################

data_for_model.isna().sum().sum()
data_for_model.dropna(how = "any", inplace = True)



##############################################################################
# SPLIT INPUT VARIABLES & OUTPUT VARIABLES
##############################################################################

X = data_for_model.drop(["purchased_album"], axis = 1)
y = data_for_model["purchased_album"]



##############################################################################
# SPLIT OUT TRAINING & TEST SETS
##############################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)



##############################################################################
# FEATURE SCALING
##############################################################################

scale_standard = StandardScaler()

X_train = scale_standard.fit_transform(X_train)
X_test = scale_standard.transform(X_test)



##############################################################################
# APPLY PCA
##############################################################################

# instantiate & fit
pca = PCA(n_components = None, random_state = 42)
pca.fit(X_train)


# extract the explained variance across components
explained_variance = pca.explained_variance_ratio_
explained_variance_cumulative = pca.explained_variance_ratio_.cumsum()



##############################################################################
# PLOT THE EXPLAINED VARIANCES ACROSS COMPONENTS
##############################################################################

# create a list for number of components
num_vars_list = list(range(1,101))
plt.figure(figsize=(15,10))


# plot the variance explained by each component
plt.subplot(2,1,1)
plt.bar(num_vars_list, explained_variance)
plt.title("Variance across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("% Variance")
plt.tight_layout()


# plot the cumulative variance
plt.subplot(2,1,2)
plt.plot(num_vars_list, explained_variance_cumulative)
plt.title("cumulative Variance across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative % Variance")
plt.tight_layout()
plt.show()



##############################################################################
# APPLY PCA WITH SELECTED NUMBER OF COMPONENTS
##############################################################################

pca = PCA(n_components = 0.75, random_state = 42)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

pca.n_components_



##############################################################################
# APPLY PCA WITH SELECTED NUMBER OF COMPONENTS
##############################################################################

clf = RandomForestClassifier(random_state = 42)
clf.fit(X_train, y_train)



##############################################################################
# ASSESS MODEL ACCURACY
##############################################################################

y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)