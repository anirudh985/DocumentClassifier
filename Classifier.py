import sys
import time
from itertools import chain

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import neighbors
from sklearn import tree

from Preprocessor import Preprocessor
from ModelAccuracyEvaluator import ModelAccuracyEvaluator

class Classifier:

    def __init__(self):
        self.dataSplit = float(sys.argv[2])
        self.classifierType = sys.argv[1]
        self.numberOfNeighbours = int(sys.argv[3])
        self.isPredict = sys.argv[4]
        # self.dataSplit = float(0.2)
        # self.classifierType = 'K'
        # self.numberOfNeighbours = 3
        # self.isPredict = 'N'
        if self.classifierType == 'N':
            self.classifier = OneVsRestClassifier(MultinomialNB(alpha=1))
        elif self.classifierType == 'K':
            self.classifier = neighbors.KNeighborsClassifier(n_neighbors = self.numberOfNeighbours, weights='distance')
        elif self.classifierType == 'D':
            self.classifier = tree.DecisionTreeClassifier(criterion='gini')
        else:
            print "Invalid Classifier Input"
            sys.exit(1)

        self.preprocessor = Preprocessor('N')
        self.modelAccuracyEvaluator = ModelAccuracyEvaluator()
        generateTfIdf = False
        self.preprocessor.construct_feature_vetor_matrix_for_naive_bayes(generateTfIdf)



    def prepareDataForClassifier(self):
        self.Y = MultiLabelBinarizer().fit_transform(self.preprocessor.Y_Labels)
        self.listOfTopics = sorted(set(chain.from_iterable(self.preprocessor.Y_Labels)))
        self.modelAccuracyEvaluator.SetListOfTopics(self.listOfTopics)
        self.X = self.preprocessor.X
        random_state = np.random.RandomState(0)
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(self.X, self.Y, test_size=self.dataSplit, random_state=random_state)
        if(self.classifierType == 'N'):
            self.train_X = np.array(self.train_X)
            self.train_Y = np.array(self.train_Y)
            self.test_X = np.array(self.test_X)
            self.test_Y = np.array(self.test_Y)

    def fittingModelAndMakingPredictions(self):
        time1 = time.time()
        self.classifier.fit(self.train_X, self.train_Y)
        time2 = time.time()
        print("Time after fitting--- %s seconds ---" % (time.time() - time1))
        if(self.isPredict == 'Y'):
            predictions = self.classifier.predict(self.test_X)
            time3 = time.time() - time2
            print("Time after Prediction--- %s seconds ---" % (time3))

            file_predicted_original_labels = open(self.GetClassifierOutputFilename(), 'w+')
            file_predicted_original_labels.write(self.GetClassifierType() + '\n')
            file_predicted_original_labels.write("Data Split:"+str(self.dataSplit)+'\n'+'\n')
            for i in range(0, len(predictions)):
                self.modelAccuracyEvaluator.EvaluateDocument(predictions[i],self.test_Y[i])
                self.writePredictedAndOriginalLabelsToFile(file_predicted_original_labels,predictions[i],self.test_Y[i])
            file_predicted_original_labels.close()
            print self.modelAccuracyEvaluator.GetAccuracy()
            print "\n"
        else:
            print self.classifier.score(self.test_X,self.test_Y)
            time3 = time.time() - time2
            print("Time after score--- %s seconds ---" % (time3))

    def writePredictedAndOriginalLabelsToFile(self, file_predicted_original_labels,outputPrediction, originalBinarizedLabels):
        actualLabels = self.listOfTopics
        predictedLabelString = self.GetLabelsForDocuments(outputPrediction,actualLabels,"P: ")
        originalLabelString = self.GetLabelsForDocuments(originalBinarizedLabels,actualLabels,"O: ")
        file_predicted_original_labels.write(predictedLabelString+originalLabelString + '\n')


    def GetLabelsForDocuments(self,binarizedLabels,actualLabels, initChar):
        string_to_write = initChar
        for i in range(0, len(binarizedLabels)):
            if binarizedLabels[i] == 1:
                string_to_write = string_to_write + actualLabels[i] + ','
        string_to_write = string_to_write[:-1]
        string_to_write += '\n'
        return  string_to_write

    def GetClassifierType(self):
        if self.classifierType == 'N':
            return "Naive Bayes Classifier"
        elif self.classifierType == 'K':
            return "K-Nearest Neighbour Classifier," + "Number Of neighbours:"+ str(self.numberOfNeighbours)
        else:
            return "Decision Tree Classifier"

    def GetClassifierOutputFilename(self):
        if self.classifierType == 'N':
            return "./NBC-"+str(self.dataSplit)+".txt"
        elif self.classifierType == 'K':
            return "./KNN-"+str(self.numberOfNeighbours)+"neighbours-"+str(self.dataSplit)+".txt"
        else:
            return "./DT-"+str(self.dataSplit)+".txt"

start_time = time.time()
classifier = Classifier()
new_time = time.time()
print("Initial time to build feature vector--- %s seconds ---" % (new_time - start_time))
print(len(classifier.preprocessor.X))
classifier.prepareDataForClassifier()
print("Prepare data for classifier--- %s seconds ---" % (time.time() - new_time))
classifier.fittingModelAndMakingPredictions()
print("--- %s seconds ---" % (time.time() - start_time))


