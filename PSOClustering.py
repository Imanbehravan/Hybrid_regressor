import numpy as np
import math
from sklearn import metrics
import matplotlib.pyplot as plt


def costFunction(data, centers):
    labels = assignLabels(data, centers)
    unique_labels = np.unique(labels)
    # if (len(unique_labels) == 1):
    #     check = True
    #     while check:
    #         new_label = np.random.randint(0, len(centers), 1)
    #         if new_label[0] != unique_labels[0]:
    #             labels[0] = new_label[0]
    #             check = False
    if (len(unique_labels) == 1):
        cost = float('inf')
    else:
        metric_value = metrics.calinski_harabasz_score(data, labels)
        cost = 1/metric_value
    return cost


def assignLabels(data, centers):
    labels = []
    for i in range(len(data)):
        dataPoint = data.loc[i]
        distance_Mat = []
        for j in range(len(centers)):
            center = centers[j]
            distance_Mat.append(math.dist(dataPoint, center))
        minimum_dist = min(distance_Mat)
        labels.append(distance_Mat.index(minimum_dist))
    return labels

def chooseNearestSample(particle, data):
    nearestSample = particle
    data = data.reset_index()
    data = data.drop(columns=["index"])
    for i in range(len(particle)):
        sampleDist = []
        centroid = particle[i]
        for j in range(len(data)):
            dataPoint = data.loc[j]
            distance = math.dist(centroid, dataPoint)
            sampleDist.append(distance)
        minimum_dist = min(sampleDist)
        minimum_distindex = sampleDist.index(minimum_dist)
        nearestSample[i] = data.loc[minimum_distindex]
        if i > 0:
            for h in range(i-1):
                if (all(nearestSample[i]) == all(nearestSample[h])):
                    nearestSample[i] = data.sample()

    return nearestSample




def PSO_Clustering(dataset, clusterNum, popNum, MaxIt, isPlot, numRepSamples, stage, sequenceNum, stepNum):
    # if stage == 1:
    #     #print("stage: ", stage,  "------------- sequence: ", i+1, "-------------- step: ", j+1)
    dimension = clusterNum * len(dataset.axes[1])
    pop = []
    cost = []
    pbestCost =[]
    w = 1
    c1 = 2
    c2 = 2
    wdamp = 0.99
    globalCost = float('inf')
    globaCostit = []
    averageCostit =[]
    pbestPos = []
    particle_velocity = []
    for it in range(MaxIt):
        if (it == 0):
            for i in range(popNum):
                centroids = np.array(dataset.sample(clusterNum))
                centroids.reshape(1, dimension)
                pop.append(centroids)
                particle_velocity.append(np.zeros(dimension))
                costValue = costFunction(dataset, centroids)
                cost.append(costValue)
                pbestPos.append(centroids)
                pbestCost.append(costValue)
                if (costValue < globalCost):
                    globalCost = costValue
                    globalPos = centroids
            globaCostit.append(globalCost)
            averageCostit.append(np.mean(cost))
            iman_breakpoint = 7

        else:
            globalCost = globaCostit[-1]
            for i in range(popNum):
                particle = pop[i]
                particle = particle.reshape(1, dimension)
                particle_velocity[i] = w*particle_velocity[i] + c1 * np.random.rand()*(pbestPos[i].reshape(1,dimension) - particle) + c2*np.random.rand()*(globalPos.reshape(1,dimension) - particle)
                particle = particle + particle_velocity[i]
                pop[i] = chooseNearestSample(particle.reshape(clusterNum, len(dataset.axes[1])), dataset.sample(n=int(numRepSamples*len(dataset))))
                cost[i] = costFunction(dataset, pop[i])
                if (cost[i]<pbestCost[i]):
                    pbestCost[i] = cost[i]
                    pbestPos[i] = pop[i]

                if (cost[i]<globalCost):
                    globalCost = cost[i]
                    globalPos = pop[i]

            globaCostit.append(globalCost)
            averageCostit.append(np.mean(cost))
            w = wdamp * w
            
            if stage == 2:
                print("---------- stage: ", stage, "---------- iteration: ", it+1 )
            else:
                print("step number: ", stepNum, " iteration: ", it)
            #print([" iteration: ", it, "bestcost: ", globalCost])

    if (isPlot):
        plt.plot(globaCostit, label="Best cost")
        plt.plot(averageCostit, label="Average cost")
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title("Convergence graph")
        plt.legend()
        plt.show()

    return globalCost, globalPos






