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
from multiprocessing.dummy import Pool
import itertools
from contextlib import closing


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
        for a in range(0, 1):
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

                self.TotalDocs += 1
                MinWiseHashPreprocessor.listOfDocuments.append(title + " " + body)

    def removeTags(self, body):
        return re.sub('<[^<>]+>', '', body)


    def creatingKgramFeatureVector(self, k):
        vectorizer = CountVectorizer(ngram_range=(k, k), stop_words='english', analyzer=u'word', binary=True, dtype=np.bool)
        self.kgramFeatureVector = vectorizer.fit_transform(MinWiseHashPreprocessor.listOfDocuments)
        self.maxFeatures = self.kgramFeatureVector.get_shape()[1]

    # Not being used
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
        primeNumber = 72613 #137183 (for 2000 docs)
        self.signatureMatrix = []
        for i in xrange(self.kgramFeatureVector.get_shape()[0]):
            self.signatureMatrix.append(self.generateSingleSignatureMatrixRow(i, coeffA, coeffB, primeNumber))
    #    creating numpy array from the above generated list of list
    #     self.signatureMatrixusingNumpy = np.asarray(self.signatureMatrix)

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
        for i in xrange(0,self.TotalDocs):
            for j in xrange(i,self.TotalDocs):
                squaredErrorMap  = self.generateSimilaritiesAndSEForAPairOfDocs((i,j))
                self.SEValues = map(lambda (x, y): (x + y),izip(squaredErrorMap, self.SEValues))

    def generateSimilaritiesandReturnSEusingMap(self):
        squaredErrorMap = itertools.imap(self.generateSimilaritiesAndSEForAPairOfDocs, itertools.combinations(xrange(self.TotalDocs), 2))
        self.SEValues = reduce(lambda x, y: map(lambda (a, b): a + b, izip(x, y)), squaredErrorMap, self.SEValues)

    def generateSimilaritiesandReturnSEparallel(self):
        chunkSize = int(self.TotalDocs * (self.TotalDocs - 1) / 8)
        with closing(Pool(processes=4)) as pool:
            squaredErrorMap = pool.imap_unordered(self.generateSimilaritiesAndSEForAPairOfDocs, itertools.combinations(range(self.TotalDocs), 2), chunksize=chunkSize)
            self.SEValues = reduce(lambda x, y: map(lambda (a,b) : a + b, izip(x,y)), squaredErrorMap, self.SEValues)
            pool.terminate()

    def generateSimilaritiesAndSEForAPairOfDocs(self,(firstDocId, secondDocId)):
        jaccardSimilarity = self.calculateJaccardSimilarityForAPairOfDocs(firstDocId,secondDocId)
        minHashSimilarity = self.generateMinSimilaritiesForAPairOfDocs(firstDocId,secondDocId)
        # minHashSimilarity = self.generateMinSimilaritiesForAPairOfDocsusingNumpy(firstDocId, secondDocId)
        return map(lambda (x, y): (x-y) * (x-y), izip([jaccardSimilarity]* len(self.listOfNumberOfHashFunctions), minHashSimilarity))

    def generateMinSimilaritiesForAPairOfDocs(self,firstDocId, secondDocId):
        similarityCount = [0,0,0,0,0]
        for i in xrange(0,self.numOfHashFunctions):
            if self.signatureMatrix[firstDocId][i] == self.signatureMatrix[secondDocId][i]:
                for (k,l) in izip(range(len(self.listOfNumberOfHashFunctions)),self.listOfNumberOfHashFunctions):
                    if i < l:
                        similarityCount[k] += 1
        return map(lambda(x,y): x/float(y), izip(similarityCount,self.listOfNumberOfHashFunctions))

    def generateMinSimilaritiesForAPairOfDocsusingNumpy(self, firstDocId, secondDocId):
        return map(lambda k: (np.count_nonzero(self.signatureMatrixusingNumpy[firstDocId][:k] == self.signatureMatrixusingNumpy[secondDocId][:k]))/float(k), self.listOfNumberOfHashFunctions)

    def calculateJaccardSimilarityForAPairOfDocs(self,firstDocId,secondDocId):
        # num = np.array(self.kgramFeatureVector[firstDocId] * self.kgramFeatureVector[secondDocId].transpose().sum(1)).flatten()[0]
        # num = (self.kgramFeatureVector[i] * self.kgramFeatureVector[j].transpose()).sum()
        # denom = self.kgramFeatureVector[firstDocId].count_nonzero() + self.kgramFeatureVector[secondDocId].count_nonzero() - num
        array1 = np.array(self.kgramFeatureVector[firstDocId].nonzero()[1])
        array2 = np.array(self.kgramFeatureVector[secondDocId].nonzero()[1])
        arrayIntersection = np.intersect1d(array1, array2, True)
        num = arrayIntersection.size
        denom = array1.size + array2.size - num
        if denom == 0:
            jack = 0
        else:
            jack = num / float(denom)
        return jack

    def generateSimilaritiesAndReturnRMSE(self):
        self.generateSimilaritiesandReturnSEusingMap()
        # self.generateSimilaritiesandReturnSE()
        # self.generateSimilaritiesandReturnSEparallel()
        totalUniquePairs = self.TotalDocs * (self.TotalDocs - 1) / 2
        RMSEList = map(lambda x : math.sqrt(x/totalUniquePairs) ,self.SEValues)
        return RMSEList

    def PlotKvsRMSE(self):
        RMSEList = self.generateSimilaritiesAndReturnRMSE()
        plot = zip(self.listOfNumberOfHashFunctions,RMSEList)
        plt.xlabel('number of hash functions (k)')
        plt.ylabel('RMSE')
        plt.plot(self.listOfNumberOfHashFunctions,RMSEList,linewidth=2.0, marker='o')
        plt.show()
        print plot


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
print("Time taken for calculating RMSE value -- %s " % (time.time() - sigMatrixTime))