# This file is meant to return the desired feature vectors. 

# function extractPitch
# @ return: 
#    Pitch features
#    Number of notes
def extractPitch(piece, numVoices):
    features = [0.0] * 12
    numRows, numCols = piece.shape
    numNotes = 0
    for voiceIndex in range(numVoices):
        voice = 5 + 4 * voiceIndex
        for r in range(numRows + 1):
            note = (int)piece[r][voice]
            if note <= 0: continue
                pitch = note % 12
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

def extractFeatures(piece, K, numVoices):   
    features = []
    pitchFeatures, numNotes = extractPitch(piece, numVoices)
    features.update(sequenceFeatures)
    features.update(intervalFeatures)
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
def initFeatureVectors(trainingSongs, numComposers):
    print ''
    print 'Extracting features...' 
    featureVectors = []
    for composer, piece, numVoices in trainingSongs:
        features = extractFeatures(piece, K, numVoices)
        featureVectors.append((composer, features, numVoices))
    print 'Finished feature extraction'
    print ''
    return featureVectors
