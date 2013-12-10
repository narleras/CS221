from time import time
import numpy as np
import pylab as pl
import os, collections, copy, loadDataSingleClass
import extractFeatures


from sklearn import metrics
from sklearn import svm, datasets
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale



testingDataPath = '/Users/naroazurutuza/Documents/ATUMN2013/Josquin/JRP_data-20131020/dataTesting/'
trainingDataPath = '/Users/naroazurutuza/Documents/ATUMN2013/Josquin/JRP_data-20131020/dataTraining/'
labels = {'Joa':'Authentic Josquin', \
    'Ock':'Ockeghem', 'Ort':'de Orto', 'Rue':'La Rue'}
composersNames = {'Joa':'Authentic Josquin', 'Job':'Suspect Josquin', \
    'Ock':'Ockeghem', 'Ort':'de Orto', 'Rue':'La Rue'}
sample_size = 300

#def convertDictToMatrix(dictionary, labelNames):   
#    nameList = list()
#    for name in labelNames:
#        nameList.append(name)
#    print 'begin convertion...'
#    data = np.zeros((len(dictionary), len(labelNames)))
#    labels = []
#    for author in range(len(dictionary)):
#        labels.append(labelNames[author])
#        for feature in dictionary[author]:
#            data[author][nameList.index(feature)] = dictionary[author][feature]
#    print '... convertion done'
#    return data,labels 
##end convertDictToMatrix

def bench_svm(trainingData, testingData, labels):
    clf = svm.SVC()
    clf.fit(trainingData, labels) 
    #C = 1.0  # SVM regularization parameter
    #svc = svm.SVC(C, cache_size=200, class_weight=None, coef0=0.0, degree=3,
    #    gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
    #    shrinking=True, tol=0.001, verbose=False).fit(data, labels)
    predictions = clf.predict(testingData)
    return predictions
#end bench_k_means

def printStatistics(predictions, composerNames, numComposers):
    #statistics = np.zeros((5, 5))
    correct = np.zeros(numComposers)
    incorrect = np.zeros(numComposers)
    total = np.zeros(numComposers)
    for index in range(len(predictions)):
        #statistics[predictions[index]][composerNames[index]] += 1
        total[composerNames[index]] += 1
        if composerNames[index] == predictions[index]:
            correct[composerNames[index]] += 1
        else:
            incorrect[composerNames[index]] += 1
    
    for index in range(numComposers):
        print 'Correct', index, ': ', correct[index], 'out of ', total[index]
        print 'Error', index, ': ', incorrect[index]/total[index]
    #m, n = statistics.shape()
    #for row in range(5):
    #    print 'statistics for cluster', row
    #    for column in range(5):
    #        print column, statistics[row][column]
    #for column in range(5):
    #    sum = 0
    #    for row in range(5):
    #        sum += statistics[row][column]
    #    print 'total',column,':', sum

#end printStatistics

def features1():

    trainingSet, trainingComposers, numTrainingComposers, numTrainingPieces = loadData.loadTrainingFiles()
    trainingFeatureVectors, trainingFeatureNames, trainingComposerNames = extractFeatures.initFeatureVectors(4, trainingSet, len(labels)+1)

    testingSet, testingComposers, numTestingComposers, numTestingPieces = loadData.loadTestingFiles()
    testingFeatureVectors, testingFeatureNames, testingComposerNames = extractFeatures.initFeatureVectors(4, testingSet, len(labels)+1)

    nameList = list()
    for name in trainingFeatureNames:
        nameList.append(name)

    for name in testingFeatureNames:
        nameList.append(name)

    print 'begin convertion'
    trainingData = np.zeros((len(trainingFeatureVectors), len(nameList)))
    testingData = np.zeros((len(testingFeatureVectors), len(nameList)))

    for author in range(len(trainingFeatureVectors)):
        for feature in trainingFeatureVectors[author]:
            trainingData[author][nameList.index(feature)] = trainingFeatureVectors[author][feature]
    for author in range(len(testingFeatureVectors)):
        for feature in testingFeatureVectors[author]:
            testingData[author][nameList.index(feature)] = testingFeatureVectors[author][feature]
    print 'done convertion'
    #data = np.array([[featureVectors[author][feature] for feature in sorted(featureVectors[author])] for author in range(len(featureVectors))])

    predictions = bench_svm(trainingData, testingData, trainingComposerNames)

    printStatistics(predictions, testingComposerNames)

def features2():
    trainingSet, trainingComposers, numTrainingComposers, numTrainingPieces = loadDataSingleClass.loadTrainingFiles()
    trainingData, trainingComposerNames = extractFeatures.initFeatureVectors(trainingSet, numTrainingComposers, numTrainingPieces)

    #permute data
    trainingSet, testingSet, trainingComposerNames, testingComposerNames = train_test_split(data, trainingComposerNames, test_size=0.20)

    #testingSet, testingComposers, numTestingComposers, numTestingPieces = loadDataSingleClass.loadTestingFiles()
    #testingFeatureVectors, testingComposerNames = extractFeatures.initFeatureVectors(testingSet, numTestingComposers, numTestingPieces)

    #trainingComposerNames = [0, 1, 2, 3, 4]
    #testingComposerNames = [0, 1, 2, 3, 4]
    predictions = bench_svm(trainingData, testingFeatureVectors, trainingComposerNames)

    printStatistics(predictions, testingComposerNames, numTrainingComposers)

#Main body of code
features2()
