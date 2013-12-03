from time import time
import numpy as np
import pylab as pl
import os, collections, copy


from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


testingDataPath = '/Users/naroazurutuza/Documents/ATUMN2013/Josquin/JRP_data-20131020/dataTesting/'
trainingDataPath = '/Users/naroazurutuza/Documents/ATUMN2013/Josquin/JRP_data-20131020/dataTraining/'
labels = {'Joa':'Authentic Josquin', \
    'Ock':'Ockeghem', 'Ort':'de Orto', 'Rue':'La Rue'}
composerNames = {'Joa':'Authentic Josquin', 'Job':'Suspect Josquin', \
    'Ock':'Ockeghem', 'Ort':'de Orto', 'Rue':'La Rue'}
sample_size = 300


def loadSongFiles(dataPath):
    files = []
    composers  = []
    numPieces = 0
    
    folders = os.listdir(dataPath)
    #print 'folders', folders
    numComposers = len(folders)
    for composer in range(len(folders)):
        if folders[composer] == '.DS_Store':
            numComposers -= 1 
            continue
        composers.append((composer - 1, folders[composer]))
        #print 'composer', composersIndexed[composer-1], 'folder', folders[composer]
        #to remove a given composer from data path, insert code here
        songPaths = os.listdir(dataPath + folders[composer])
        for song in range(len(songPaths)):
            #print 'song', songPaths[song], 'composer', composersIndexed[composer-1]
            if songPaths[song] == '.DS_Store': continue
            numPieces += 1
            file = open(dataPath + folders[composer] + '/' + songPaths[song], "r")
            data = []
            for line in file.readlines():   
                if line[0] != "%":
                    data.append(line.rstrip("\n").split("\t"))
            piece = np.array(data, dtype=float)
            row, col = piece.shape
            files.append((composer - 1, piece, (col-5)/4))
            #print 'composer', composer-1
    return files, composers, numComposers, numPieces
#end loadSongFiles

def trimFeatures(features, numComposers):
    print ''
    print 'Trimming weight vectors'
   # removed = [0] * numComposers
    trimmedFeatures = copy.deepcopy(features)
    for feature in features:
        # while len(trimmedWeights[i]) > 50:
        #     val = min(trimmedWeights[i], key=trimmedWeights[i].get)
        #     del trimmedWeights[i][val]
        #     removed[i] += 1
        #for feature in features[i]:
        if abs(features[feature]) <= 0:
            del trimmedFeatures[feature]
            #removed[i] += 1
    print ''
    print 'Finished trimming.... Displaying trimming results'
    print 'Number of features removed:'
    #print removed
    print 'Length of weight vectors before trimming:'
    print [len(features)]
    print 'Length of weight vectors after trimming:'
    print [len(trimmedFeatures)]
    print ''
    return trimmedFeatures
#end trimWeights


def extractSequenceFeatures(piece, K, numVoices):
    sequenceFeatures = collections.defaultdict(int)
    row, col = piece.shape
    for voiceIndex in range(numVoices):
        voice = 5 + 4 * voiceIndex
        for note in range(1,row-K+1):
            if piece[note][voice] != 0:
                sequence = list()
                for i in range(K):
                    sequence.append(str(int(piece[note+i][voice])))
                stringSeq = ''.join(sequence)
                sequenceFeatures[stringSeq] = sequenceFeatures[stringSeq] + 1
    return sequenceFeatures
#end extractSequenceFeatures

def extractIntervalFeatures(piece, numVoices):
    intervalFeatures = collections.defaultdict(int)
    row, col = piece.shape
    for voiceIndex in range(numVoices):
        voice = 5 + 4 * voiceIndex
        for note in range(1,row):
            if piece[note][voice] != 0:
                nextNote = piece[note][voice+3]
                if nextNote != 0 and nextNote != -1:
                    interval = str(int(piece[nextNote][voice] - piece[note][voice]))
                    intervalFeatures[interval] = intervalFeatures[interval] + 1
    return intervalFeatures
#end extractIntervalFeatures
     
def extractFeatures(piece, K, numVoices):    
    features = {}
    sequenceFeatures = extractSequenceFeatures(piece, K, numVoices)
    intervalFeatures = extractIntervalFeatures(piece, numVoices)
    features.update(sequenceFeatures)
    features.update(intervalFeatures)
    return features
#end extractFeatures 
# function initFeatures
#   @param K - sequence lengths for extraction
#   @param trainingSongs - list of songs in training set
#   @param numComposers - number of composers in training set  
#
# @return list of (composer, feature vector, num Voices) tuples
#
# Usage: should be called from train to store feature vectors in memory
def initFeatureVectors(K, trainingSongs, numComposers):
    featureVectors = []
    featureNames = collections.defaultdict(int)
    composerNames = []
    for composer, piece, numVoices in trainingSongs:
        features = extractFeatures(piece, K, numVoices)
        trimmedFeatures = trimFeatures(features, numComposers)
        composerNames.append(composer)
        for key in trimmedFeatures:
            featureNames[key] += 1
        featureVectors.append(trimmedFeatures)
    return featureVectors, featureNames, composerNames
#end initFeatureVectors

def convertDictToMatrix(dictionary, labelNames):   
    nameList = list()
    for name in labelNames:
        nameList.append(name)
    print 'begin convertion...'
    data = np.zeros((len(dictionary), len(labelNames)))
    labels = []
    for author in range(len(dictionary)):
        labels.append(labelNames[author])
        for feature in dictionary[author]:
            data[author][nameList.index(feature)] = dictionary[author][feature]
    print '... convertion done'
    return data,labels 
#end convertDictToMatrix

def bench_k_means(estimator, name, data):
    #t0 = time()
    estimator.fit(data)
    predictions = estimator.predict(data)
    #print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
    #      % (name, (time() - t0), estimator.inertia_,
    #         metrics.homogeneity_score(labels, estimator.labels_),
    #         metrics.completeness_score(labels, estimator.labels_),
    #         metrics.v_measure_score(labels, estimator.labels_),
    #         metrics.adjusted_rand_score(labels, estimator.labels_),
    #         metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
    #         metrics.silhouette_score(data, estimator.labels_,
    #                                  metric='euclidean',
    #                                  sample_size=sample_size)))
    return predictions
#end bench_k_means

def printStatistics(predictions, composerNames):
    statistics = np.zeros([4, 5])
    for index in range(len(predictions)):
        statistics[predictions[index]][composerNames[index]] += 1
    
    #m, n = statistics.shape()
    for row in range(4):
        print 'statistics for cluster', row
        for column in range(5):
            print column, statistics[row][column]
    for column in range(5):
        sum = 0
        for row in range(4):
            sum += statistics[row][column]
        print 'total',column,':', sum

#end printStatistics


trainingSet, trainingComposers, numTrainingComposers, numTrainingPieces = loadSongFiles(trainingDataPath)
featureVectors, featureNames, composerNames = initFeatureVectors(4, trainingSet, len(labels)+1)
data, labels = convertDictToMatrix(featureVectors, featureNames)

predictions = bench_k_means(KMeans(init='k-means++', n_clusters=len(labels), n_init=10),
              name="k-means++", data=data)
printStatistics(predictions, labels)

