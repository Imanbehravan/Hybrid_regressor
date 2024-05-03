import numpy as np
from PSO_SVR_Regressor import PSO_SVR_regressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

datapath = "/home/iman/projects/kara/Projects/House_price_prediction/APSO_Clustering_PSO_SVR/data/Dataset.xlsx"
dataset = pd.read_excel(datapath)
dataset = dataset.dropna()
dataset = dataset.reset_index()
dataset = dataset.drop(columns=['index'])
dataset = dataset.drop(columns=['PricePerMeter', 'URL'])
target = dataset["TotalPrice"]
dataset = dataset.drop(columns = ["TotalPrice"])
x_train, x_test, y_train, y_test = train_test_split(dataset, target, test_size=0.3, random_state=42)

x_train = x_train.reset_index()
x_train = x_train.drop(columns = ['index'])

y_train = y_train.reset_index()
y_train = y_train.drop(columns = ['index'])

x_test = x_test.reset_index()
x_test = x_test.drop(columns = ['index'])

y_test = y_test.reset_index()
y_test = y_test.drop(columns = ['index'])


popSize = 5
maxIt = 15
regressor = PSO_SVR_regressor(x_train, y_train, popSize, maxIt)
regressor.fit(True)

print('best cost: ', regressor.gbestCost)
print('best solution: ', regressor.gbestPos)

y_pred = regressor.predict(x_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print('r2 score: ', r2, ' mean squared error: ', mse)

