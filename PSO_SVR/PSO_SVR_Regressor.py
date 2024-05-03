import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class PSO_SVR_regressor():
    
    def __init__(self, data, target, popSize, maxIt) -> None:
        
        self.popSize = popSize
        self.maxIt = maxIt
        self.x_data = data
        self.y_data = target
        self.features = self.x_data.columns.tolist()
        self.dim = self.x_data.shape[1]
        self.testSize = 0.3
        self.bestModel = None
        pbestPos =[]
        pbestCost = []
        self.gbestPos = None
        cost = []
        
        self.gbestCost = float("inf")
        
    def fit(self, isPlot = False):
        c1 = 2
        c2 = 2
        w = 1
        wDamp = 0.99
        population, velocity, cost = self.initialization()
        pbestPos = population
        pbestCost = cost
        gbestCost_it = []
        gbestAverageCost_it = []
        gbestCost_it.append(self.gbestCost)
        gbestAverageCost_it.append(np.mean(cost))
        
        ##### main loop
        for it in range(self.maxIt):
            for p in range(len(population)):
                particle = population[p]
                velocity[p] = w * velocity[p] + c1 * np.random.rand()*(pbestPos[p] - particle) + c2 * np.random.rand() * (self.gbestPos - particle)
                particle = particle + velocity[p]
                particle[particle >= 0.5] = 1
                particle[particle < 0.5] = 0
                if np.all(particle == 0):
                    random_index = np.random.randint(0, len(particle))
                    particle[random_index] = 1
                population[p] = particle
                cost[p], trainedModel = self.costEval(particle)
                
                if cost[p] < self.gbestCost:
                    self.gbestCost = cost[p]
                    self.gbestpos = population[p]
                    self.bestModel = trainedModel
                
                if cost[p] < pbestCost[p]:
                    pbestCost[p] = cost[p]
                    pbestPos[p] = population[p] 
                    
           
            w = wDamp * w
            gbestCost_it.append(self.gbestCost)
            gbestAverageCost_it.append(np.average(cost))
            print('iteration: ', it, ' best cost: ', self.gbestCost, ' average cost: ', gbestAverageCost_it[it])
        
        if isPlot:
            iterations = [iter for iter in range(1, self.maxIt + 2)]
            plt.plot(iterations, gbestCost_it, label="best cost")
            plt.plot(iterations, gbestAverageCost_it, label="average cost")
            plt.show()
        
        
        
    def initialization(self):
        population = []
        velocity = []
        cost = []
        trainedModels = []
        for i in range(self.popSize):
            random_particle = np.array([random.randint(0, 1) for _ in range(self.dim)])
            if np.all(random_particle == 0):
                random_index = np.random.randint(0, len(random_particle))
                random_particle[random_index] = 1
            
            population.append(random_particle)
            velocity.append(np.array([random.randint(0, 0) for _ in range(self.dim)]))
            costval, regModel = self.costEval(population[i])
            cost.append(costval)
            if cost[i] < self.gbestCost:
                self.gbestPos = population[i]
                self.gbestCost = cost[i]
                self.bestModel = regModel
        return population, velocity, cost
        
    def costEval(self, particle):
        zero_indices = np.where(particle == 0)[0]
        redundant_Features = []
        dataset = self.x_data
        for i in zero_indices:
            dataset = dataset.drop(columns = [self.features[i]])
                    
        target = self.y_data
        x_train, x_test, y_train, y_test = train_test_split(dataset, target, test_size=self.testSize, random_state=42)
        model = SVR()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        costVal = mean_squared_error(y_test, y_pred)
        return costVal, model
    
    def predict(self, x_test):
        zero_indices = np.where(self.gbestPos == 0)[0]
        dataset = x_test
        for i in zero_indices:
            dataset = dataset.drop(columns = [self.features[i]])
            
        y_pred = self.bestModel.predict(dataset)
       
            
        return y_pred
         
        
        
