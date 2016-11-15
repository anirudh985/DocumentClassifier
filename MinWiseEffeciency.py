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
        # self.p = Pool(4)


    def init_file_parsing(self):
        for a in range(0, 22):
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
        print self.maxFeatures
        # jaccard_time_start = time.time()
        # self.calculateJaccardSimilarity()
        # jaccard_time_end = time.time()
        # print "Jaccard Time:",jaccard_time_end-jaccard_time_start

    def calculateJaccardSimilarity(self):
        for i in range(0, self.TotalDocs):
            # jaccardSimilarityOfDoc = ""
            for j in range(i+1, self.TotalDocs):
                jack = self.calculateJaccardSimilarityForAPairOfDocs(i,j)
                # jaccardSimilarityOfDoc += str(jack)
            # self.fileJaccardSimilarity.write(jaccardSimilarityOfDoc)

    def generateRandomCoeffs(self,lengthOfSignatureMatrix):
        randomCoeffList = random.sample(xrange(0, self.maxFeatures), lengthOfSignatureMatrix)
        return randomCoeffList

    def createMinWiseSignatureMatrix(self,lengthOfSignatureMatrix):
        coeffA = self.generateRandomCoeffs(lengthOfSignatureMatrix)
        coeffB = self.generateRandomCoeffs(lengthOfSignatureMatrix)
        primeNumber = 1284583
            # 323507
        # 72613
        self.signatureMatrix = []
        for i in xrange(self.kgramFeatureVector.get_shape()[0]):
            self.signatureMatrix.append(self.generateSingleSignatureMatrixRow(i, coeffA, coeffB, primeNumber,lengthOfSignatureMatrix))
    #    creating numpy array from the above generated list of list
        self.signatureMatrixNP = np.asarray(self.signatureMatrix)

    def generateSingleSignatureMatrixRow(self, docNumber, coeffA, coeffB, primeNumber,lengthOfSignatureMatrix):
        return map(self.generateMinwiseHash, izip(coeffA, coeffB, [primeNumber]*lengthOfSignatureMatrix, [docNumber]*lengthOfSignatureMatrix))

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

    def mapperGenerateSimilaritiesandReturnSE(self):
        squaredErrorMap = itertools.imap(self.generateSimilaritiesAndSEForAPairOfDocs, itertools.combinations(xrange(self.TotalDocs), 2))
        self.SEValues = reduce(lambda x, y: map(lambda (a, b): a + b, izip(x, y)), squaredErrorMap, self.SEValues)

    def threadedGenerateSimilaritiesandReturnSE(self):
        chunkSize = int(self.TotalDocs * (self.TotalDocs - 1) / 8)
        with closing(Pool()) as pool:
            squaredErrorMap = pool.imap_unordered(self.generateSimilaritiesAndSEForAPairOfDocs, itertools.combinations(range(self.TotalDocs), 2), chunksize=chunkSize)
            self.SEValues = reduce(lambda x, y: map(lambda (a,b) : a + b, izip(x,y)), squaredErrorMap, self.SEValues)
            pool.terminate()

    def generateMinhashSimlaritiesForGivenSignatureLength(self,lengthOfSignatureMatrix):
        for i in xrange(0,self.TotalDocs):
            for j in xrange(i,self.TotalDocs):
                minHashSimilarity = self.generateMinSimilaritiesForAPairOfDocs(i,j,lengthOfSignatureMatrix)


    def generateSimilaritiesAndSEForAPairOfDocs(self,(firstDocId, secondDocId)):
        # TODO: Need to change the minwise similarity function
        jaccardSimilarity = self.calculateJaccardSimilarityForAPairOfDocs(firstDocId,secondDocId)
        # minHashSimilarity = self.generateMinSimilaritiesForAPairOfDocs(firstDocId,secondDocId)
        minHashSimilarity = self.generateMinSimilaritiesForAPairOfDocsNP(firstDocId, secondDocId)
        return map(lambda (x, y): (x-y) * (x-y), izip([jaccardSimilarity]* len(self.listOfNumberOfHashFunctions), minHashSimilarity))

    def generateMinSimilaritiesForAPairOfDocs(self,firstDocId, secondDocId,lengthOfSignatureMatrix):
        similarityCount = 0
        for i in xrange(0,lengthOfSignatureMatrix):
            if self.signatureMatrix[firstDocId][i] == self.signatureMatrix[secondDocId][i]:
                similarityCount+=1
        return similarityCount/lengthOfSignatureMatrix

    def generateMinSimilaritiesForAPairOfDocsNP(self, firstDocId, secondDocId):
        return map(lambda k: (np.count_nonzero(self.signatureMatrixNP[firstDocId][:k] == self.signatureMatrixNP[secondDocId][:k])/float(k)), self.listOfNumberOfHashFunctions)

    def calculateJaccardSimilarityForAPairOfDocs(self,firstDocId,secondDocId):
        array1 = np.array(self.kgramFeatureVector[firstDocId].nonzero()[1])
        array2 = np.array(self.kgramFeatureVector[secondDocId].nonzero()[1])
        arrayIntersection = np.intersect1d(array1,array2,True)
        num = arrayIntersection.size
        denom = array1.size + array2.size - num
        # num = np.array(self.kgramFeatureVector[firstDocId] * self.kgramFeatureVector[secondDocId].transpose().sum(1)).flatten()[0]
        # num = (self.kgramFeatureVector[i] * self.kgramFeatureVector[j].transpose()).sum()
        # denom = self.kgramFeatureVector[firstDocId].count_nonzero() + self.kgramFeatureVector[secondDocId].count_nonzero()
        if denom == 0:
            jack = 0
        else:
            jack = num / float(denom)
        return jack

    def generateSimilaritiesAndReturnRMSE(self):
        # self.mapperGenerateSimilaritiesandReturnSE()
        # self.generateSimilaritiesandReturnSE()
        self.threadedGenerateSimilaritiesandReturnSE()
        totalUniquePairs = self.TotalDocs * (self.TotalDocs - 1) / 2
        RMSEList = map(lambda x : math.sqrt(x/totalUniquePairs) ,self.SEValues)
        return RMSEList

            # math.sqrt(squaredError) / totalUniquePairs
        # return RMSE
    def generateMinHashSimilaritiesForDifferentLengths(self):
        self.minHashtimes = []
        time_sig_start = time.time()
        print ":SigStarted"
        self.createMinWiseSignatureMatrix(self.numOfHashFunctions)
        print "SigEnd,Elapsed:",time.time()-time_sig_start
        for i in self.listOfNumberOfHashFunctions:
            start_time = time.time()
            self.generateMinhashSimlaritiesForGivenSignatureLength(i)
            end_time = time.time()
            elapsed_time = end_time-start_time
            print i,":",elapsed_time
            self.minHashtimes.append(elapsed_time)

    def PlotKvsTime(self):
        plt.xlabel('number of hash functions')
        plt.ylabel('Time')
        plt.plot(self.listOfNumberOfHashFunctions, self.minHashtimes, linewidth=2.0, marker='o')
        plt.show()

    def PlotKvsRMSE(self):
        RMSEList = self.generateSimilaritiesAndReturnRMSE()
        plot = izip(self.listOfNumberOfHashFunctions,RMSEList)
        print RMSEList
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
minWiseHashPreproc.generateMinHashSimilaritiesForDifferentLengths()
kgramFeatureVectorTime = time.time()
print("Creating k gram Vector done -- %s" % (kgramFeatureVectorTime -fileParsing_time ))
# minWiseHashPreproc.PlotKvsTime()
# minWiseHashPreproc.createMinWiseSignatureMatrix()
sigMatrixTime = time.time()
print("Min wise signature matrix created -- %s" % (sigMatrixTime - kgramFeatureVectorTime))
# minWiseHashPreproc.PlotKvsRMSE()
# print("Time taken for creating k gram feature vector -- %s " % (time.time() - sigMatrixTime))