import numpy as np
from Multi_cluster_PSO_SVR import *
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


dataPath = "/home/karaubuntu/Iman_projects/data/NewDataset.xlsx"
result_path = "/home/karaubuntu/Iman_projects/"
dataset = pd.read_excel(dataPath)
dataset = dataset.drop(columns = ['Unnamed: 0'])
dataset = dataset.dropna()
dataset = dataset.reset_index()
dataset = dataset.drop(columns=['index'])

targetName  = 'TotalPrice'
target = dataset[targetName]
dataset = dataset.drop(columns=[targetName])
dataset = dataset.drop(columns = ['PricePerMeter'])
x_train, x_test, y_train, y_test = train_test_split(dataset, target, test_size=0.3, random_state=42)



labels_json = result_path + 'Result/labels.json'
outCentroidsPath_json = result_path + 'Result/centroids.json'

base_regressor = 'XGBoost'

regressor = Multi_Cluster_PSO_SVR_Regressor(result_path, base_regressor, False, False, labels_json, outCentroidsPath_json)

x_train = x_train.reset_index()
x_train = x_train.drop(columns = ['index'])

y_train = y_train.reset_index()
y_train = y_train.drop(columns = ['index'])

regressor.clustering(x_train)
regressor.fit(x_train, y_train)

x_test = x_test.reset_index()
x_test = x_test.drop(columns = ['index'])
pred = []

for i in range(len(x_test)):
    pred.append(regressor.predict(x_test.loc[i]))

r2 = r2_score(pred, y_test)
print('r2 score of hybrid model: ', r2)

mse = mean_squared_error(pred, y_test)
print("mean squared error hybrid model: ", mse)

model_filename = 'Hybrid_Model.pkl'
model_path = os.path.join(result_path, model_filename)
with open(model_path, 'wb') as f:
    pickle.dump(regressor, f)
     

# with open(model_path, 'rb') as f:
#         model = pickle.load(f)
        
print()