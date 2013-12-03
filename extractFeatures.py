# This file is meant to return the desired feature vectors. 

# function extractPitch
# @ return: 
#    Pitch features
#    Number of notes
import numpy as np

def extractPitch(piece, numVoices):
    features = [0.0] * 12
    numRows, numCols = piece.shape
    numNotes = 0
    for voiceIndex in range(numVoices):
        voice = 5 + 4 * voiceIndex
        for r in range(numRows):
            note = piece[r][voice]
            if note <= 0: continue
            pitch = (int)(note % 12)
            numNotes += 1
            features[pitch] += 1
    for i in range(len(features)):
        features[i] /= numNotes
    return features, numNotes
#end extractPitch

def extractOctaveFeatures():
    return []
#end extractOctaveFeatures

def extractNoteCount():
    return []
#end extractNoteCount

def extractNoteDuration():
    return []
#end extractNoteDuration

def extractPitchGradient():
    return []
#end extractPitchGradient

def extractFeatures(piece, numVoices):   
    features = []
    pitchFeatures, numNotes = extractPitch(piece, numVoices)
    for elem in pitchFeatures:
        features.append(elem)
    features.append(numNotes)
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
