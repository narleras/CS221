from time import time
import numpy as np
import pylab as pl
import os, collections, copy, loadData
import extractFeatures


from sklearn import metrics
from sklearn import svm, datasets
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split



testingDataPath = '/Users/naroazurutuza/Documents/ATUMN2013/Josquin/JRP_data-20131020/dataTesting/'
trainingDataPath = '/Users/naroazurutuza/Documents/ATUMN2013/Josquin/JRP_data-20131020_2/'
labels = {'Joa':'Authentic Josquin', \
    'Ock':'Ockeghem', 'Ort':'de Orto', 'Rue':'La Rue'}
globalComposerNames = {'Joa':'Authentic Josquin', 'Job':'Suspect Josquin', \
    'Ock':'Ockeghem', 'Ort':'de Orto', 'Rue':'La Rue'}
sample_size = 300



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

def printStatistics(predictions, composerNames):
    #statistics = np.zeros((5, 5))
    correct = np.zeros(5)
    incorrect = np.zeros(5)
    total = np.zeros(5)
    for index in range(len(predictions)):
        print predictions[index], composerNames[index]
        #statistics[predictions[index]][composerNames[index]] += 1
        total[composerNames[index]] += 1
        if composerNames[index] == predictions[index]:
            correct[composerNames[index]] += 1
        else:
            incorrect[composerNames[index]] += 1
    
    for index in range(5):
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

def dimReduction(newDim, trainingSet, testingSet, trainingComposers, testingComposers):
    #n_components = 150

    #print("Extracting the top %d eigenfaces from %d faces"
     #   % (n_components, X_train.shape[0]))
    #t0 = time()
    pca = PCA(n_components=newDim)
    reducedTrainingSet = pca.fit(trainingSet).transform(trainingSet)
    reducedTestingSet = pca.fit(testingSet).transform(testingSet)
    
    #lda = LDA(n_components=newDim)
    #reducedTrainingSet = lda.fit(trainingSet, trainingComposers).transform(trainingSet)
    #reducedTestingSet = lda.fit(testingSet, testingComposers).transform(testingSet)
    return reducedTrainingSet, reducedTestingSet
#end dimReduction

def plotData(x, y, composerValues):
    #print len(x), len(x[0])
    pl.figure()
    for c, i, compName in zip("rgbmy", composerValues, globalComposerNames):
        pl.scatter(x[y == i, 0], x[y == i, 1], c=c, label=compName)
    pl.legend()
    pl.title('SVM classifier')
    
    pl.show()
    
def plotFeatureData(x, y, composerValues):
    print 'length', len(x)
    pl.figure()
    for c, i, compName in zip("rgbmy", composerValues, globalComposerNames):
        pl.scatter(x[y == i], y[y == i], c=c, label=compName)
    pl.legend()
    pl.title('SVM classifier')
    
    pl.show()

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
    #load data
    dataSet, dataComposers, numTrainingComposers, numDataPieces = loadData.loadTrainingFiles()
    data, composerNames = extractFeatures.initFeatureVectors(dataSet, numTrainingComposers, numDataPieces)
    
    # split into a training and testing set
    trainingSet, testingSet, trainingComposerNames, testingComposerNames = train_test_split(data, composerNames, test_size=0.20)
    
    print 'training', len(trainingSet), len(trainingComposerNames), 'testing', len(testingSet), len(testingComposerNames)
    #make predictions
    predictions = bench_svm(trainingSet, testingSet, trainingComposerNames)
    
    reducedTrainingSet, reducedTestingSet = dimReduction(2, trainingSet, testingSet, trainingComposerNames, testingComposerNames)

    printStatistics(predictions, testingComposerNames)
    plotData(reducedTrainingSet, trainingComposerNames, [0, 1, 2, 3, 4])
    #plotFeatureData(trainingSet[:, 1], trainingComposerNames, [0, 1, 2, 3, 4])

#Main body of code
features2()
