##############################################################################
# IMPORT REQUIRED PACKAGES
##############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


##############################################################################
# IMPORT SAMPLE DATA
##############################################################################

loyalty_scores = pd.read_excel("grocery_database.xlsx", sheet_name = "loyalty_scores")
customer_details = pd.read_excel("grocery_database.xlsx", sheet_name = "customer_details")
transactions = pd.read_excel("grocery_database.xlsx", sheet_name = "transactions")


# create the customer level dataset
data_for_regression = pd.merge(customer_details, loyalty_scores, how = "left", on = "customer_id")


sales_summary = transactions.groupby("customer_id").agg({"sales_cost" : "sum",
                                                         "num_items" : "sum",
                                                         "transaction_id" : "count",
                                                         "product_area_id" : "nunique"}).reset_index()

sales_summary.columns = ["customer_id", "total_sales", "total_items", 
                         "transaction_count", "product_area_count"]

sales_summary["average_basket_value"] = sales_summary["total_sales"] / sales_summary["transaction_count"]

data_for_regression = pd.merge(data_for_regression, sales_summary, how = "inner", on = "customer_id")

regression_modelling = data_for_regression.loc[data_for_regression["customer_loyalty_score"].notna()]
regression_scoring = data_for_regression.loc[data_for_regression["customer_loyalty_score"].isna()]
regression_scoring.drop(["customer_loyalty_score"], axis = 1, inplace = True)


# save files
pickle.dump(regression_modelling, open("data/abc_regression_modelling.p", "wb"))
pickle.dump(regression_scoring, open("data/abc_regression_scoring.p", "wb"))