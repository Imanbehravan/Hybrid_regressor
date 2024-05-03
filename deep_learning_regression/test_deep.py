from deep_regression import *
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score


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

x_train = x_train.reset_index()
x_train = x_train.drop(columns = ['index'])

y_train = y_train.reset_index()
y_train = y_train.drop(columns = ['index'])

regressor  = deep_regressor()

regressor.fit(x_train, y_train)

x_test = x_test.reset_index()
x_test = x_test.drop(columns = ['index'])

y_test = y_test.reset_index()
y_test = y_test.drop(columns = ['index'])

# y_pred = []
# for i in range(len(x_test)):
#     y_pred.append(regressor.predict(x_test.loc[i]))

y_pred = np.array([regressor.predict(x_test.loc[i])[0] for i in range(len(x_test))])

# y_pred = regressor.predict(x_test)
r2 = r2_score(y_pred, y_test)
print("r2: ", r2)
print("prediction: ", y_pred)