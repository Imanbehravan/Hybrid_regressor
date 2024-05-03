from PSOClustering import PSO_Clustering
import pandas as pd
import numpy as np
import math
from sklearn.metrics import rand_score
from PSOClustering import assignLabels
import joblib
import json
import sys


def runSequence(data, popNum, MaxIt, sequenceSteps, numRepSamples, i):
    print("----------------------sequence number ", i+1, " is started ----------------------")
    print(" ")
    for j in range(sequenceSteps):
        bestK_list = []
        bestCost_list = []
        if (j == 0):
            # k = np.random.randint(2, math.sqrt(len(data)))
            k = np.random.randint(2, 30)
            bestKSofar = k
            bestK_list.append(bestKSofar)
            cost, pos = PSO_Clustering(data, k, popNum, MaxIt, False, numRepSamples, 1, i, j)
            bestCostSofar = cost
            bestCost_list.append(bestCostSofar)
        else:
            k = k + np.random.randint(-3, 3)
            if (k < 2):
                k = 2
            cost, pos = PSO_Clustering(data, k, popNum, MaxIt, False, numRepSamples, 1, i, j)
            if (cost < bestCostSofar):
                bestCostSofar = cost
                bestKSofar = k
            bestK_list.append(bestKSofar)
            bestCost_list.append(bestCostSofar)
    bestcost_index = bestCost_list.index(min(bestCost_list))
    print("----------------------sequence number ", i+1, " is finished ----------------------")
    print(" ")
    return bestK_list[bestcost_index], min(bestCost_list)


if __name__ == '__main__':
    sequenceNum = 3
    sequenceSteps = 15
    sequence_PSOPop = 5
    sequence_MaxIt = 75
    secondStage_PSOpop = 5
    secondStage_MaxIt = 150
    numRepSamples = 0.05
    # rand_index_flag = False
    centroids = []
    # outputfile = "/home/iman/projects/kara/Projects/House_price_prediction/APSO_Clustering_PSO_SVR/Result/result.txt"
    # outLabelPath = "/home/iman/projects/kara/Projects/House_price_prediction/APSO_Clustering_PSO_SVR/Result/labels.txt"
    # outLabelPath_json = "/home/iman/projects/kara/Projects/House_price_prediction/APSO_Clustering_PSO_SVR/Result/labels.json"
    # outCentroidsPath_json = "/home/iman/projects/kara/Projects/House_price_prediction/APSO_Clustering_PSO_SVR/Result/centroids.json"
    # datapath = "/home/iman/projects/kara/Projects/House_price_prediction/APSO_Clustering_PSO_SVR/data/Dataset.xlsx"
    if len(sys.argv) != 7:
        print("wrong inputs")
        sys.exit(1)
        
    outputfile = sys.argv[1]
    outLabelPath = sys.argv[2]
    outLabelPath_json = sys.argv[3]
    outCentroidsPath_json = sys.argv[4]
    
    rand_index_flag = sys.argv[5]
    processing_style = sys.argv[6]
    
    dataset_json = sys.stdin.read()
    trainDataset = pd.read_json(dataset_json)
    
    #dataset = pd.read_excel(datapath)
    if (rand_index_flag == 'True') | (rand_index_flag == 'true'):
        target = trainDataset["targets"]
    #trainDataset = dataset.drop(columns=["targets"])

    if (processing_style == 'serial'):
        bestCostList = []
        result = []
        for i in range(sequenceNum):
            result.append(runSequence(trainDataset, sequence_PSOPop, sequence_MaxIt, sequenceSteps, numRepSamples, i))
            bestCostList.append(result[i][1])
   
    if (processing_style == 'parallel'): 
        result = joblib.Parallel(n_jobs=sequenceNum)(joblib.delayed(runSequence)(trainDataset, sequence_PSOPop, sequence_MaxIt, sequenceSteps, numRepSamples,i) for i in range(sequenceNum))
        bestCostList = []
        for j in range(sequenceNum):
            bestCostList.append(result[j][1])


    minimumCost = min(bestCostList)
    bestK_index = bestCostList.index(minimumCost)
    bestK = result[bestK_index][0]
    finalBestCost, finalCentroids = PSO_Clustering(trainDataset, bestK, secondStage_PSOpop, secondStage_MaxIt, True, numRepSamples, 2, 0, 0)
    predLabels = assignLabels(trainDataset, finalCentroids)
    index = 1
    for i in range(bestK):
        if predLabels.count(i) > 0:
            centroids.append(finalCentroids[i])
            print("number of elements in cluster ", index, "is: ", predLabels.count(i))
            index = index + 1
    print("best number of clusters: ", len(centroids))
    print("best centroids: ", centroids)
    if (rand_index_flag == "true") | (rand_index_flag == 'True'):
        rand_index = rand_score(target, predLabels)
        print("rand index: ", rand_index)

    ###################################
    f = open(outputfile, "w")
    f.write("number of centroids: ")
    f.write(str(len(centroids)))
    f.write('\n')
    f.write('\n')
    f.write("centroids: ")
    f.write('\n')
    f.write('\n')
    np.savetxt(f, centroids)
    #f.write(finalCentroids)
    f.write('\n')
    f.write("cost: ")
    #np.savetxt(f,finalBestCost)
    f.write(str(finalBestCost))
    f.write('\n')
    f.write("Rand index: ")
    if (rand_index_flag == "true") | (rand_index_flag == 'True'):
        f.write(str(rand_index))

    else:
        f.write("not calculated")

    f.close()

    #################################
    f = open(outLabelPath, "w")
    f.write("sample labels: ")
    f.write('\n')
    f.write('\n')
    np.savetxt(f, predLabels)
    
    labels_result_dic = {"labels": predLabels}
    json_object = json.dumps(labels_result_dic, indent=4)
    with open(outLabelPath_json, "w") as outfile:
        outfile.write(json_object)
        
    centroids_result_dic = {"centroids": finalCentroids.tolist()}
    json_object = json.dumps(centroids_result_dic, indent=4)
    with open(outCentroidsPath_json, "w") as outfile:
        outfile.write(json_object)

    print()

