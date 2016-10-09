import sys
import time
from itertools import chain

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import neighbors

from Preprocessor import Preprocessor
from ModelAccuracyEvaluator import ModelAccuracyEvaluator

class NaiveBayesClassifier:

    def __init__(self):
        classifierType = sys.argv[3]
        if classifierType == 'N':
            self.classifier = OneVsRestClassifier(MultinomialNB())
        else:
            self.classifier = neighbors.KNeighborsClassifier(n_neighbors = 3)
        # self.classifier = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors = 3))
        self.preprocessor = Preprocessor('N')
        self.modelAccuracyEvaluator = ModelAccuracyEvaluator()
        # generateTfIdf = (sys.argv[2] == 'Y') or (sys.argv[2] == 'y')
        generateTfIdf = False
        self.preprocessor.construct_feature_vetor_matrix_for_naive_bayes(generateTfIdf)



    def prepareDataForClassifier(self):
        self.Y = MultiLabelBinarizer().fit_transform(self.preprocessor.Y_Labels)
        self.listOfTopics = sorted(set(chain.from_iterable(self.preprocessor.Y_Labels)))
        self.modelAccuracyEvaluator.SetListOfTopics(self.listOfTopics)
        self.X = self.preprocessor.X
        random_state = np.random.RandomState(0)
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(self.X, self.Y, test_size=0.33, random_state=random_state)

    def fittingModelAndMakingPredictions(self):
        # k = 61
        self.classifier.fit(self.train_X, self.train_Y)
        # self.predictions = self.classifier.predict(self.test_X)
        # resY = self.classifier.predict(np.array(self.test_X[k]).reshape(1,-1))
        predictions = self.classifier.predict(np.array(self.test_X))
        # self.printCorrespodingTopics(resY[0])
        for i in range(0, len(predictions)):
            self.modelAccuracyEvaluator.EvaluateDocument(predictions[i],self.test_Y[i])
            # self.printCorrespodingTopics(predictions[i])
            # self.printCorrespodingTopics(self.test_Y[i])
            # print "\n"
        print self.modelAccuracyEvaluator.GetAccuracy()
        print "\n"
        print self.classifier.score(self.test_X,self.test_Y)

    # def evaluatingPredictions(self):

    def printCorrespodingTopics(self, outputPrediction):
        print map(lambda (x, y): y if x == 1 else '', zip(outputPrediction, self.listOfTopics))


nbClassifier = NaiveBayesClassifier()
start_time = time.time()
nbClassifier.prepareDataForClassifier()
nbClassifier.fittingModelAndMakingPredictions()
print("--- %s seconds ---" % (time.time() - start_time))


