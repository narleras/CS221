import numpy, collections, os, loadDataSingleClass, copy, classifierFeatures

#Static + hard coded variables
composerNames = {'Joa':'Authentic Josquin', 'Job':'Suspect Josquin', \
    'Ock':'Ockeghem', 'Ort':'de Orto', 'Rue':'La Rue'}
composersIndexed = {0:'Authentic Josquin', 1: 'Non-Josquin'}
K =[4]
numIterations = 100

def learnWeights(numIt, K, featureVectors, currentComposer, numTrainingPieces):
    print ''
    print 'Learning weight vector for ' + composersIndexed[currentComposer]
    print ''
    w = collections.defaultdict(int)
    for iteration in range(numIt):
        arr = numpy.arange(numTrainingPieces - 1)
        numpy.random.shuffle(arr)
        for i in arr:
            composer, features, numVoices = featureVectors[i]
            #for composer, features, numVoices in featureVectors:
            prediction = 0.0
            for key in features:
                prediction += features[key]*w[key]
            #print prediction
            if composer == currentComposer and prediction <= 0:
                for key in features:
                    w[key] += features[key]
                #delta = 1
            elif composer != currentComposer and prediction >= 0:
                for key in features:
                    w[key] -= features[key]  
                #delta = -1
                         
    return w
#end learnWeights
            
def train(numIt, K, numComposers, numTrainingPieces, featureVectors):
    print ''
    print "TRAINING FOR " + str(numIt) + " iterations..."
    print ''    
    weights = []   
    for currentComposer in range(numComposers):
        weights.append(learnWeights(numIt, K, featureVectors, currentComposer, numTrainingPieces)) 
    print ''
    print 'FINISHED TRAINING'
    print '' 
    temp = [sum(weights[i][j] for j in weights[i]) for i in range(numComposers)]
    print 'Sum of weights in training vectors'
    print temp
    print ''                      
    return weights
#end train

def trimWeights(weights, numComposers):
    print ''
    print 'Trimming weight vectors'
    removed = [0] * numComposers
    trimmedWeights = copy.deepcopy(weights)
    for i in range(len(weights)):
        # while len(trimmedWeights[i]) > 50:
        #     val = min(trimmedWeights[i], key=trimmedWeights[i].get)
        #     del trimmedWeights[i][val]
        #     removed[i] += 1
        for feature in weights[i]:
            if abs(weights[i][feature]) <= 100:
                del trimmedWeights[i][feature]
                removed[i] += 1
    print ''
    print 'Finished trimming.... Displaying trimming results'
    print 'Number of features removed:'
    print removed
    print 'Length of weight vectors before trimming:'
    print [len(weights[i]) for i in range(len(weights))]
    print 'Length of weight vectors after trimming:'
    print [len(trimmedWeights[i]) for i in range(len(trimmedWeights))]
    print ''
    return trimmedWeights
#end trimWeights

def normalizeWeights(trimmedWeights, numComposers):
    for i in range(len(trimmedWeights)):
        n = sum(abs(trimmedWeights[i][elem]) for elem in trimmedWeights[i])
        for elem in trimmedWeights[i]:
            trimmedWeights[i][elem] /= n
    print '' 
    temp = [sum(trimmedWeights[i][j] for j in trimmedWeights[i]) for i in range(numComposers)]
    print 'Sum of weights in training vectors after normalization'
    print temp
    print ''
#end normalizeWeights

def testSongs(K, weights, testingSongs, numComposers):
    print ''
    print 'Testing...'
    results = [(0, 0)] * numComposers
    totalCorrect = 0
    temp = [0] * numComposers
    for composer, features, numVoices in testingSongs:
         
        predictions = [0.0] * numComposers
        for i in range(numComposers):                
            for key in features:
                old = predictions[i]
                predictions[i] += features[key]*weights[i][key]
                new = predictions[i]
                if features[key] > 0 and weights[i][key] > 0:
                    if old == new:
                        print 'ERROR'
                #print 'features ' + str(features[key])
                #print 'weights ' + str(weights[i][key])
        #print predictions
        prediction = predictions.index(max(predictions)) 
        temp[prediction] +=1
        x, y = results[composer]
        if prediction == composer:
            x += 1
            totalCorrect += 1
        y += 1
        results[composer] = (x, y)
    print ''
    print 'Number of predictions per composer'
    print temp
    print ''
    print 'Finished testing'
    print ''
    return results, totalCorrect
#end test
   
def printStatistics(results, totalCorrect, numTestingPieces, numTrainingPieces, numTestingComposers, composersIndexed, K, numIterations):
    print ''
    print 'TESTING RESULTS'
    print ''
    for i in range(numTestingComposers):
        correct, total = results[i]
        print "Accuracy on " + composersIndexed[i] + " pieces: " + str(float(correct)/total)
        print str(correct) + " correct, " + str(total) + " total."
        print ''
    print "Number of pieces in training set: " + str(numTrainingPieces)
    print "Number of pieces in testing set: " + str(numTestingPieces)
    print "Overall accuracy on testing set: " + str(float(totalCorrect)/numTestingPieces)
    print "Number of iterations: " + str(numIterations)
    print "Range of sequence lengths: "
    print K 
    print ''
#end printStatistics
   
#Main body of code         
trainingSet, trainingComposers, numTrainingComposers, numTrainingPieces = loadDataSingleClass.loadTrainingFiles()
testingSet, testingComposers, numTestingComposers, numTestingPieces = loadDataSingleClass.loadTestingFiles()
trainingFeatureVectors = classifierFeatures.initFeatureVectors(K, trainingSet, numTrainingComposers)
weights = train(numIterations, K, numTrainingComposers, numTrainingPieces, trainingFeatureVectors)
#trimmedWeights = trimWeights(weights, numTrainingComposers)
trimmedWeights = weights
#normalizeWeights(trimmedWeights, numTrainingComposers)
testingFeatureVectors = classifierFeatures.initFeatureVectors(K, testingSet, numTestingComposers)
results, totalCorrect = testSongs(K, trimmedWeights, testingFeatureVectors, numTestingComposers)
printStatistics(results, totalCorrect, numTestingPieces, numTrainingPieces, numTestingComposers, composersIndexed, K, numIterations)
