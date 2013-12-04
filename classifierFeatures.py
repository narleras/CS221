#Feature extraction for multiClassifier features
import numpy, collections, os, loadData, copy, classifierFeatures

def extractSequenceFeatures(piece, K, numVoices):
    sequenceFeatures = collections.defaultdict(int)
    row, col = piece.shape
    for voiceIndex in range(numVoices):
        voice = 5 + 4 * voiceIndex
        for k in K:
            for note in range(1,row-k+1):
                if piece[note][voice] != 0:
                    sequence = list()
                    for i in range(k):
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
    def printExtractionResults():
        print ''
        print 'Printing feature extraction sum'
        print sum(features[i] for i in features)
        print ''
    #printExtractionResults()
    return features
#end extractFeatures 

# function initFeatureVectors
#   @param K - sequence lengths for extraction
#   @param trainingSongs - list of songs in training set
#   @param numComposers - number of composers in training set  
#
# @return list of (composer, feature vector, num Voices) tuples
#
# Usage: should be called from train to store feature vectors in memory
def initFeatureVectors(K, trainingSongs, numComposers):
    print ''
    print 'Extracting features...' 
    featureVectors = []
    for composer, piece, numVoices in trainingSongs:
        features = extractFeatures(piece, K, numVoices)
        featureVectors.append((composer, features, numVoices))
    print 'Finished feature extraction'
    print ''
    return featureVectors
