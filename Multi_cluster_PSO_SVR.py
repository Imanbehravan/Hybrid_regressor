import numpy as np
import subprocess
import os
import json
import pandas as pd
from PSO_SVR.PSO_SVR_Regressor import *
import pickle
from APSO_Clustering import *
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV




class Multi_Cluster_PSO_SVR_Regressor():
    def __init__(self, mainPath, modelName, isLoad = False, feature_selection = False, labels_Address = None, centroids_Address = None) -> None:
        # self.data = dataset
        self.outputPath = mainPath+ 'Result/result.txt'
        self.outLabelPath = mainPath + 'Result/labels.txt'
        self.outLabelPath_json = mainPath + 'Result/labels.json'
        self.outCentroidsPath_json = mainPath + 'Result/centroids.json'
        self.save_model_folder = mainPath + 'saved_models'
        self.bestModels = []
        self.bestModelsParam = []
        self.centroids = []
        self.labels = []
        self.unique_labels = []
        self.feature_selection = feature_selection
        self.model_name = modelName
        
        
        if isLoad:
            self.load(labels_Address, centroids_Address)
       
    
    def clustering(self, dataset, rand_Index = 'False'):
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_script_path = os.path.join(script_dir, 'main.py')
        Processing_style = "serial" # serial or parallel
       
        dataset_to_json = dataset.to_json()
        subprocess.run(['python3', main_script_path, self.outputPath, self.outLabelPath, self.outLabelPath_json, self.outCentroidsPath_json, rand_Index, Processing_style], input=dataset_to_json.encode())
        # APSO_Cluster(self.outputPath, self.outLabelPath, self.outLabelPath_json, self.outCentroidsPath_json, rand_Index, dataset)
        with open(self.outLabelPath_json, 'r') as f:
            self.labels = json.load(f)
            self.unique_labels = list(set(self.labels['labels']))

        with open(self.outCentroidsPath_json, 'r') as f:
            centers = json.load(f)
            self.centroids = centers['centroids']
        


    def fit(self, X_train, Y_train):
        if os.path.exists(self.outLabelPath_json) and os.path.exists(self.outCentroidsPath_json):
            print("JSON files including labels and centroids found.")            
                
            for i in range(len(self.unique_labels)):
                class_label = self.unique_labels[i]
                sample_indexes = np.where(np.array(self.labels['labels']) == class_label)
                # dataset = self.data.loc[sample_indexes]
                x_train = X_train.loc[sample_indexes]
                x_train = x_train.reset_index()
                x_train = x_train.drop(columns = ['index'])
                
                y_train = Y_train.loc[sample_indexes]
                y_train = y_train.reset_index()
                y_train = y_train.drop(columns = ['index'])
                
                
                print()
                # print('Training regression model on cluster ', class_label )
                print('Training ', self.model_name, 'on cluster: ', class_label)
                if self.feature_selection:
                    regressor = PSO_SVR_regressor(x_train, y_train, 10, 20)
                    regressor.fit()
                    self.bestModels.append(regressor.bestModel)
                    self.bestModelsParam.append(regressor.gbestPos)
                else:
                    if self.model_name == 'SVR':
                        regressor = SVR()
                    if self.model_name == "XGBoost":
                        # regressor = xgb.XGBRegressor(n_estimators = 500, max_depth = 5, learning_rate = 0.1) # simple regressor with parameters hard coded
                        
                        model =  xgb.XGBRegressor()
                        # Define hyperparameter grid
                        param_grid = {
                            'learning_rate': [0.001, 0.01, 0.1, 0.3, 0.4],
                            'n_estimators': [50, 100, 300, 500, 700, 1000],
                            'max_depth': [3, 5, 7, 9, 11],
                            # Add more hyperparameters to tune
                        }
                        regressor = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2')
                        # #grid_search.fit(X_train, y_train)

                    if self.model_name == "DecisionTree":
                        regressor = DecisionTreeRegressor(random_state= 42)
                    if self.model_name == 'RandomForest':
                        regressor = RandomForestRegressor()
                    if self.model_name == 'LinearRegression':
                        regressor = LinearRegression()
                    if self.model_name == 'Lasso':
                        regressor = Lasso()
                    if self.model_name == 'KNN':
                        regressor = KNeighborsRegressor(n_neighbors=3)
                    if self.model_name == 'ElasticNet':
                        regressor = ElasticNet(alpha=0.1, l1_ratio = 0.5)
                        
                    regressor.fit(x_train, y_train)
                    self.bestModels.append(regressor)
             
                
                
                # model_filename = 'PSO_SVR' + str(class_label) + '.pkl'
                # model_path = os.path.join(self.save_model_folder, model_filename)
                # with open(model_path, 'wb') as f:
                #     pickle.dump(regressor.bestModel, f)
                # print(regressor.bestModel)
                    
            print('training finished succesfully')
            
        else:
            print("JSON files do not exist.")
    
    def predict(self, testSample):
        # print('prediction')
        if self.feature_selection:
            centers = np.array(self.centroids)
            distances = np.linalg.norm(centers - np.array(testSample), axis=1)
            min_distance_index = np.argmin(distances[1:]) + 1
            cluster_model = self.bestModels[min_distance_index]
            cluster_model_params = self.bestModelsParam[min_distance_index]
            
            features = testSample.index.tolist()
            selected_features = []
            for i in range(len(cluster_model_params)):
                if cluster_model_params[i] == 1:
                    selected_features.append(features[i])
            test_sample = testSample.loc[selected_features]
            # test_sample = [np.array(test_sample)]
            pred = cluster_model.predict([test_sample])
        else:
            centers = np.array(self.centroids)
            distances = np.linalg.norm(centers - np.array(testSample), axis=1)
            min_distance_index = np.argmin(distances[1:]) + 1
            cluster_model = self.bestModels[min_distance_index]
            pred = cluster_model.predict([testSample])
            if (self.model_name == "KNN") | (self.model_name == "LinearRegression"):
                pred = pred[0]
        return pred


        
        
        
    def load(self, labels_adderess, centroids_address):
        
        with open(labels_adderess, 'r') as f:
            self.labels = json.load(f)
            self.unique_labels = list(set(self.labels['labels']))
            
        with open(centroids_address, 'r') as f:
            centroids_dict = json.load(f)
            self.centroids = centroids_dict['centroids']
            