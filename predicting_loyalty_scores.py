##############################################################################
# IMPORT REQUIRED PACKAGES
##############################################################################

import pandas as pd
import pickle



##############################################################################
# IMPORT CUSTOMERS FOR SCORING
##############################################################################

to_be_scored = pickle.load(open("data/abc_regression_scoring.p", "rb"))



##############################################################################
# IMPORT MODEL AND MODEL OBJECTS
##############################################################################

regressor = pickle.load(open("data/random_forest_regression_model.p", "rb"))
one_hot_encoder = pickle.load(open("data/random_forest_regression_ohe.p", "rb"))



##############################################################################
# DROP UNUSED COLUMNS
##############################################################################

to_be_scored.drop(["customer_id"], axis = 1, inplace = True)



##############################################################################
# DROP MISSING VALUES
##############################################################################

to_be_scored.dropna(how = "any", inplace = True)



##############################################################################
# APPLY ONE HOT ENCODING
##############################################################################

categorical_vars = ["gender"]

encoder_vars_array = one_hot_encoder.transform(to_be_scored[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names(categorical_vars)

encoder_vars_df = pd.DataFrame(encoder_vars_array, columns = encoder_feature_names)

to_be_scored = pd.concat([to_be_scored.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis = 1)

to_be_scored.drop(categorical_vars, axis = 1, inplace = True)



##############################################################################
# MAKE OUR PREDICTIONS
##############################################################################

loyalty_predictions = regressor.predict(to_be_scored)