# This file is meant to return the desired feature vectors. 

# function extractPitch
# @ return: 
#    Pitch features
#    Number of notes
import numpy as np
import math, collections

def extractPitchAndNoteCount(piece, numVoices):
    features = [0.0] * 40
    numRows, numCols = piece.shape
    numNotes = 0
    for voiceIndex in range(numVoices):
        voice = 5 + 4 * voiceIndex
        for r in range(numRows):
            note = (int)(piece[r][voice])
            if note <= 0: continue
            pitch = note % 40
            numNotes += 1
            features[pitch] += 1
    for i in range(len(features)):
        features[i] /= numNotes
    return features, numNotes, 41
#end extractPitch

def extractOctaveFeatures(piece, numVoices):
    features = collections.defaultdict(int)
    numRows, numCols = piece.shape
    for voiceIndex in range(numVoices):
        voice = 5 + 4 * voiceIndex
        for r in range(numRows):
            note = (int)(piece[r][voice])
            if note <= 0: continue
            octave = math.floor(note / 40)
            features[octave] += 1
    keys = list(features.viewkeys())
    numFeatures = len(keys)
    keys.sort()
    sortedFeatures = [0] * 4
    for elem in keys:
        sortedFeatures[int(elem) - 2] = features[elem]
    return sortedFeatures, 4
#end extractOctaveFeatures

def extractNoteDuration(piece, numVoices):
    features = []
    vals = []
    variance = []
    numRows, numCols = piece.shape
    prevNote = 0
    numNotes = 0
    duration = 0
    total = 0
    maxLen = 0
    for voiceIndex in range(numVoices):
        voice = 5 + 4 * voiceIndex
        for r in range(numRows):
            note = int(piece[r][voice])
            if note == 0: 
                duration = 0
                prevNote = 0
                continue
            if note > 0:
                if prevNote < 0:
                    total += duration
                    if duration > maxLen:
                        maxLen = duration
                    vals.append(duration)
                numNotes += 1
                duration = 1
                prevNote = -1
            if note < 0:
                duration += 1
                prevNote = -1
    mean = float(total)/numNotes
    features.append(mean)
    for val in vals:
        variance.append((val - mean)**2)
    var = sum(variance) / len(variance)
    features.append(var)
    print features
    return features, 2
#end extractNoteDuration

def extractPitchGradient(piece, numVoices):
    features = []
    vals = []
    variance = []
    numRows, numCols = piece.shape
    validPrevNote = False
    prevNotePitch = 0
    for voiceIndex in range(numVoices):
        voice = 5 + 4 * voiceIndex
        for r in range(numRows):
            note = int(piece[r][voice])
            pitch = note % 40
            if note < 0:
                #do something else
                continue
            elif note == 0:
                validPrevNote = False
            elif validPrevNote == False:
                prevNotePitch = pitch
                validPrevNote = True
            else:
                grad = pitch - prevNotePitch
                if grad > 0:
                    grad = 1
                elif grad < 0:
                    grad = -1
                vals.append(grad)
    mean = float(sum(vals)) / len(vals)
    for elem in vals:
        variance.append((elem - mean)**2)
    var = sum(variance) / len(variance)
    features.append(mean)
    features.append(var)
    print features
    return features, 2
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
    lenFeatures, num = extractNoteDuration(piece, numVoices)
    numFeatures += num
    for elem in lenFeatures:
        features.append(elem)
    gradFeatures, numGrad = extractPitchGradient(piece, numVoices)
    numFeatures += numGrad
    for elem in gradFeatures:
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
