import numpy, collections, os

#Static + hard coded variables
testingDataPath = '/Users/kevinlaube/documents/classes/josquinProject/dataTesting_small/'
trainingDataPath = '/Users/kevinlaube/documents/classes/josquinProject/dataTraining_small/'

def loadSongFiles(dataPath):
    files = []
    composers  = []
    numPieces = 0
    c = collections.Counter({i: 0 for i in range(4)})
    folders = os.listdir(dataPath)
    numComposers = len(folders)
    for composer in range(len(folders)):
        if folders[composer] == '.DS_Store':
            numComposers -= 1 
            continue
        composers.append((composer - 1, folders[composer]))
        #to remove a given composer from data path, insert code here
        songPaths = os.listdir(dataPath + folders[composer])
        for song in range(len(songPaths)):
            if songPaths[song] == '.DS_Store': continue
            numPieces += 1
            file = open(dataPath + folders[composer] + '/' + songPaths[song], "r")
            data = []
            for line in file.readlines():   
                if line[0] != "%":
                    data.append(line.rstrip("\n").split("\t"))
            piece = numpy.array(data, dtype=float)
            row, col = piece.shape
            files.append((composer - 1, piece, (col-5)/4))
            c[composer - 1] += 1
    print ''
    print 'Input file statistics: '
    print 'Following line shows number of songs per composer'
    temp = [c[i] for i in range(numComposers)]
    print temp
    print ''
    return files, composers, numComposers, numPieces
#end loadSongFiles

def loadTrainingFiles(dataPath=trainingDataPath):
    return loadSongFiles(dataPath)

def loadTestingFiles(dataPath=testingDataPath):
    return loadSongFiles(dataPath)    
