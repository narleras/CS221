# This file is meant to return the desired feature vectors. 

# function extractPitch
# @ return: 
#    Pitch features
#    Number of notes
import numpy as np
import math, collections

def extractPitchAndNoteCount(piece, numVoices):
    features = [0.0] * 12
    numRows, numCols = piece.shape
    numNotes = 0
    for voiceIndex in range(numVoices):
        voice = 5 + 4 * voiceIndex
        for r in range(numRows):
            note = (int)(piece[r][voice])
            if note <= 0: continue
            pitch = note % 12
            numNotes += 1
            features[pitch] += 1
    for i in range(len(features)):
        features[i] /= numNotes
    return features, numNotes, 13
#end extractPitch

def extractOctaveFeatures(piece, numVoices):
    features = collections.defaultdict(int)
    numRows, numCols = piece.shape
    for voiceIndex in range(numVoices):
        voice = 5 + 4 * voiceIndex
        for r in range(numRows):
            note = (int)(piece[r][voice])
            if note <= 0: continue
            octave = math.floor(note / 12)
            features[octave] += 1
    keys = list(features.viewkeys())
    #keys = [0]*len(features)
    
    print 'keys', keys
    numFeatures = len(keys)
    if numFeatures > 12: print 'ERROR'
    keys.sort()
    sortedFeatures = [0] * 12
    for elem in keys:
        sortedFeatures[int(elem) - 7] = features[elem]
    return sortedFeatures, 12
#end extractOctaveFeatures

def extractNoteDuration():
    return []
#end extractNoteDuration

def extractPitchGradient():
    return []
#end extractPitchGradient

def extractFeatures(piece, numVoices):   
    features = []
    numFeatures = 0
    pitchFeatures, numNotes, pitchCount = extractPitchAndNoteCount(piece, numVoices)
    numFeatures += pitchCount
    for elem in pitchFeatures:
        features.append(elem)
    features.append(numNotes)
    octaveFeatures, octaveCount = extractOctaveFeatures(piece, numVoices)
    numFeatures += octaveCount
    for elem in octaveFeatures:
        print elem
        features.append(elem)
    #print 'features', features
    return features, numFeatures
#end extractFeatures 

# function initFeatureVectors
#   @param K - sequence lengths for extraction
#   @param trainingSongs - list of songs in training set
#   @param numComposers - number of composers in training set  
#
# @return list of (composer, feature vector, num Voices) tuples
#
# Usage: should be called from train to store feature vectors in memory
def initFeatureVectors(trainingSongs, numComposers, numTrainingPieces):
    print ''
    print 'Extracting features...' 
    #numFeatures = 13
    featureVector = []
    composers = np.zeros(numTrainingPieces)
    currPiece = 0
    for composer, piece, numVoices in trainingSongs:
        composers[currPiece] = (int)(composer)
        features, numFeatures = extractFeatures(piece, numVoices)
        if not len(featureVector):
            featureVector = np.zeros((numTrainingPieces, numFeatures))
        for i in range(numFeatures):
            featureVector[currPiece][i] = features[i]
        currPiece += 1
    print 'Finished feature extraction'
    print ''
    return featureVector, composers
