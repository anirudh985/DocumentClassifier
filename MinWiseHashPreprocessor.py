#!/usr/bin/python

from __future__ import division
from bs4 import BeautifulSoup
import re
import urllib2
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import sys
import time
import random
import math
import matplotlib.pyplot as plt
from itertools import izip


class MinWiseHashPreprocessor:
    baseFileName = 'http://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-'
    baseFileExtension = '.sgm'
    listOfDocuments = []

    def __init__(self, kGramValue):
        self.kGramValue = kGramValue
        self.numOfHashFunctions = 256
        self.listOfNumberOfHashFunctions = [16,32,64,128,256]
        self.SEValues = [0,0,0,0,0]
        self.TotalDocs = 0


    def init_file_parsing(self):
        for a in range(0, 2):
            file_name = self.baseFileName + str(a).zfill(3) + self.baseFileExtension
            print file_name
            file_handle = urllib2.urlopen(file_name)
            file_handle.readline()
            xml_content = '<a>'
            xml_content += file_handle.read()
            xml_content += '</a>'
            soup = BeautifulSoup(xml_content, "xml")
            for article in soup.find_all('REUTERS'):
                articleId = article['NEWID']
                title = ''
                body = ''
                if article.TITLE:
                    title = self.removeTags(article.TITLE.string)
                # else:
                # 	title = ''
                if article.BODY:
                    body = self.removeTags(article.BODY.string)
                # else:
                #     body = ''

                if title == '' and body == '' and article.TEXT:
                    body = self.removeTags(article.TEXT.string)
                # else:
                # 	body = ''
                if (title == '' and body == ''):
                    continue

                # if article.TOPICS:
                #     topicsClassLabels = ""
                #     for topic in article.TOPICS.find_all('D'):
                #         topicsClassLabels += topic.string + ","
                #     topicsClassLabels = topicsClassLabels[:-1]
                #     Preprocessor.articleIdTopicsClassLabelMap[articleId] = topicsClassLabels
                #
                # if article.PLACES:
                #     placeClassLabels = ""
                #     for place in article.PLACES.find_all('D'):
                #         placeClassLabels += place.string + ","
                #     placeClassLabels = placeClassLabels[:-1]
                #     Preprocessor.articleIdPlacesClassLabelMap[articleId] = placeClassLabels

                self.TotalDocs += 1
                MinWiseHashPreprocessor.listOfDocuments.append(title + " " + body)

    def removeTags(self, body):
        return re.sub('<[^<>]+>', '', body)


    def creatingKgramFeatureVector(self, k):
        # max_features=feature_count
        vectorizer = CountVectorizer(ngram_range=(k, k), stop_words='english', analyzer=u'word', binary=True, dtype=np.bool)
        self.kgramFeatureVector = vectorizer.fit_transform(MinWiseHashPreprocessor.listOfDocuments)
        self.maxFeatures = self.kgramFeatureVector.get_shape()[1]
        # self.calculateJaccardSimilarity()

    def calculateJaccardSimilarity(self):
        for i in range(0, self.TotalDocs):
            jaccardSimilarityOfDoc = ""
            for j in range(i+1, self.TotalDocs):
                jack = self.calculateJaccardSimilarityForAPairOfDocs(i,j)
                jaccardSimilarityOfDoc += str(jack)
            self.fileJaccardSimilarity.write(jaccardSimilarityOfDoc)


    def generateRandomCoeffs(self):
        randomCoeffList = random.sample(xrange(0, self.maxFeatures), self.numOfHashFunctions)
        return randomCoeffList

    def createMinWiseSignatureMatrix(self):
        coeffA = self.generateRandomCoeffs()
        coeffB = self.generateRandomCoeffs()
        primeNumber = 72613
        self.signatureMatrix = []
        for i in xrange(self.kgramFeatureVector.get_shape()[0]):
            self.signatureMatrix.append(self.generateSingleSignatureMatrixRow(i, coeffA, coeffB, primeNumber))

    def generateSingleSignatureMatrixRow(self, docNumber, coeffA, coeffB, primeNumber):
        return map(self.generateMinwiseHash, izip(coeffA, coeffB, [primeNumber]*self.numOfHashFunctions, [docNumber]*self.numOfHashFunctions))

    def generateMinwiseHash(self, arg):
        minValue = sys.maxint
        for i in self.kgramFeatureVector[arg[3]].nonzero()[1]:
            hashValue = ((arg[0]*i+ arg[1]) % arg[2]) % self.maxFeatures
            if hashValue < minValue:
                minValue = hashValue
        return minValue

    def generateSimilaritiesandReturnSE(self):
        # squaredError = 0
        for i in xrange(0,self.TotalDocs):
            minHashForADoc = ""
            for j in xrange(i,self.TotalDocs):
                # minHashForADoc += str(self.generateMinSimilaritiesForAPairOfDocs(i,j)) + "\t"
                squaredErrorMap  = self.generateSimilaritiesAndSEForAPairOfDocs(i,j)
                self.SEValues = map(lambda (x, y): (x + y),izip(squaredErrorMap, self.SEValues))
            # self.fileMinHashingSimilarity.write(minHashForADoc)
        # return squaredError

    def generateSimilaritiesAndSEForAPairOfDocs(self,firstDocId, secondDocId):
        jaccardSimilarity = self.calculateJaccardSimilarityForAPairOfDocs(firstDocId,secondDocId)
        minHashSimilarity = self.generateMinSimilaritiesForAPairOfDocs(firstDocId,secondDocId)
        return map(lambda (x, y): (x-y) * (x-y), izip([jaccardSimilarity]* len(self.listOfNumberOfHashFunctions), minHashSimilarity))

    def generateMinSimilaritiesForAPairOfDocs(self,firstDocId, secondDocId):
        similarityCount = [0,0,0,0,0]
        for i in xrange(0,self.numOfHashFunctions):
            if self.signatureMatrix[firstDocId][i] == self.signatureMatrix[secondDocId][i]:
                for (k,l) in izip(range(len(self.listOfNumberOfHashFunctions)),self.listOfNumberOfHashFunctions):
                    if i < l:
                        similarityCount[k] += 1
        return map(lambda(x,y): x/float(y), izip(similarityCount,self.listOfNumberOfHashFunctions))

    def calculateJaccardSimilarityForAPairOfDocs(self,firstDocId,secondDocId):
        num = np.array(self.kgramFeatureVector[firstDocId] * self.kgramFeatureVector[secondDocId].transpose().sum(1)).flatten()[0]
        # num = (self.kgramFeatureVector[i] * self.kgramFeatureVector[j].transpose()).sum()
        denom = self.kgramFeatureVector[firstDocId].count_nonzero() + self.kgramFeatureVector[secondDocId].count_nonzero() - num
        jack = num / float(denom)
        return jack

    def generateSimilaritiesAndReturnRMSE(self):
        self.generateSimilaritiesandReturnSE()
        totalUniquePairs = self.TotalDocs * (self.TotalDocs - 1) / 2
        RMSEList = map(lambda x : math.sqrt(x/totalUniquePairs) ,self.SEValues)
        return RMSEList

            # math.sqrt(squaredError) / totalUniquePairs
        # return RMSE
    def PlotKvsRMSE(self):
        RMSEList = self.generateSimilaritiesAndReturnRMSE()
        plot = izip(self.listOfNumberOfHashFunctions,RMSEList)
        plt.xlabel('number of hash functions')
        plt.ylabel('Square errors')
        plt.plot(self.listOfNumberOfHashFunctions,RMSEList,linewidth=2.0, marker='o')
        plt.show()
        # plt.setp(lines, color='r', linewidth=2.0)
        print plot


#minWiseHashPreproc = MinWiseHashPreprocessor(int(sys.argv[1]))
minWiseHashPreproc = MinWiseHashPreprocessor(3)
start_time = time.time()
minWiseHashPreproc.init_file_parsing()
fileParsing_time = time.time()
print("Time taken for file parsing -- %s" % (fileParsing_time - start_time))
minWiseHashPreproc.creatingKgramFeatureVector(minWiseHashPreproc.kGramValue)
kgramFeatureVectorTime = time.time()
print("Creating k gram Vector done -- %s" % (kgramFeatureVectorTime -fileParsing_time ))
minWiseHashPreproc.createMinWiseSignatureMatrix()
sigMatrixTime = time.time()
print("Min wise signature matrix created -- %s" % (sigMatrixTime - kgramFeatureVectorTime))
minWiseHashPreproc.PlotKvsRMSE()
print("Time taken for creating k gram feature vector -- %s " % (time.time() - sigMatrixTime))