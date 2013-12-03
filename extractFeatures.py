# This file is meant to return the desired feature vectors. 

# function extractPitch
# @ return: 
#    Pitch features
#    Number of notes
import numpy as np
import math

def extractPitchAndNoteCount(piece, numVoices):
    features = [0.0] * 12
    numRows, numCols = piece.shape
    numNotes = 0
    for voiceIndex in range(numVoices):
        voice = 5 + 4 * voiceIndex
        for r in range(numRows):
            note = (int)piece[r][voice]
            if note <= 0: continue
            pitch = note % 12
            numNotes += 1
            features[pitch] += 1
    for i in range(len(features)):
        features[i] /= numNotes
    return features, numNotes, 13
#end extractPitch

def extractOctaveFeatures():
    features = collections.defaultdict(int)
    numRows, numCols = piece.shape
    for voiceIndex in range(numVoices):
        voice = 5 + 4 * voiceIndex
        for r in range(numRows):
            note = (int)piece[r][voice]
            if note <= 0: continue
            octave = math.floor(note / 12)
            features[octave] += 1
    keys = features.viewkeys()
    numFeatures = len(keys)
    keys.sort()
    sortedFeatures = [0] * numFeatures
    for i in range(numFeatures):
        sortedFeatures[i] = features[keys[i]]
    return sortedFeatures, numFeatures
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
        features.append(elem)
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
    numFeatures = 13
    featureVector = np.zeros((numTrainingPieces, numFeatures))
    composers = np.zeros(numTrainingPieces)
    currPiece = 0
    for composer, piece, numVoices in trainingSongs:
        composers[currPiece] = (int)(composer)
        features = extractFeatures(piece, numVoices)
        for i in range(numFeatures):
            featureVector[currPiece][i] = (int)(features[i])
        currPiece += 1
    print 'Finished feature extraction'
    print ''
    return featureVector, composers
