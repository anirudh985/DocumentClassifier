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
        for a in range(0, 10):
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
                if article.BODY:
                    body = self.removeTags(article.BODY.string)
                if title == '' and body == '' and article.TEXT:
                    body = self.removeTags(article.TEXT.string)
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
        print self.maxFeatures

    def calculateJaccardSimilarity(self):
        jaccard_time_start = time.time()
        for i in range(0, self.TotalDocs):
            for j in range(i+1, self.TotalDocs):
                jack = self.calculateJaccardSimilarityForAPairOfDocs(i,j)
        jaccard_time_end = time.time()
        print "Jaccard Time:", jaccard_time_end - jaccard_time_start

    def generateRandomCoeffs(self,lengthOfSignatureMatrix):
        randomCoeffList = random.sample(xrange(0, self.maxFeatures), lengthOfSignatureMatrix)
        return randomCoeffList

    def createMinWiseSignatureMatrix(self,lengthOfSignatureMatrix):
        coeffA = self.generateRandomCoeffs(lengthOfSignatureMatrix)
        coeffB = self.generateRandomCoeffs(lengthOfSignatureMatrix)
        primeNumber = 633079 #for 10000 Docs, 1284583 for 22000 docs 323507 for 5000 docs ; 72613 for 1000 docs
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

    def generateMinhashSimlaritiesForGivenSignatureLength(self,lengthOfSignatureMatrix):
        for i in xrange(0,self.TotalDocs):
            for j in xrange(i,self.TotalDocs):
                minHashSimilarity = self.generateMinSimilaritiesForAPairOfDocs(i,j,lengthOfSignatureMatrix)

    def generateMinSimilaritiesForAPairOfDocs(self,firstDocId, secondDocId,lengthOfSignatureMatrix):
        similarityCount = 0
        for i in xrange(0,lengthOfSignatureMatrix):
            if self.signatureMatrix[firstDocId][i] == self.signatureMatrix[secondDocId][i]:
                similarityCount+=1
        return similarityCount/lengthOfSignatureMatrix

    def calculateJaccardSimilarityForAPairOfDocs(self,firstDocId,secondDocId):
        array1 = np.array(self.kgramFeatureVector[firstDocId].nonzero()[1])
        array2 = np.array(self.kgramFeatureVector[secondDocId].nonzero()[1])
        arrayIntersection = np.intersect1d(array1,array2,True)
        num = arrayIntersection.size
        denom = array1.size + array2.size - num
        if denom == 0:
            jack = 0
        else:
            jack = num / float(denom)
        return jack

    def generateMinHashSimilaritiesForDifferentLengths(self):
        self.minHashtimes = []
        time_sig_start = time.time()
        self.createMinWiseSignatureMatrix(self.numOfHashFunctions)
        print "Signature Matrix Created in time:",time.time()-time_sig_start
        for i in self.listOfNumberOfHashFunctions:
            start_time = time.time()
            self.generateMinhashSimlaritiesForGivenSignatureLength(i)
            end_time = time.time()
            elapsed_time = end_time-start_time
            print i,":",elapsed_time
            self.minHashtimes.append(elapsed_time)

    def PlotKvsTime(self):
        plt.xlabel('number of hash functions (k)')
        plt.ylabel('Time in seconds')
        plt.plot(self.listOfNumberOfHashFunctions, self.minHashtimes, linewidth=2.0, marker='o')
        plt.show()


minWiseHashPreproc = MinWiseHashPreprocessor(3)
start_time = time.time()
minWiseHashPreproc.init_file_parsing()
fileParsing_time = time.time()
print("Time taken for file parsing -- %s" % (fileParsing_time - start_time))
minWiseHashPreproc.creatingKgramFeatureVector(minWiseHashPreproc.kGramValue)
minWiseHashPreproc.generateMinHashSimilaritiesForDifferentLengths()
kgramFeatureVectorTime = time.time()
print("Total time -- %s" % (kgramFeatureVectorTime -start_time ))
minWiseHashPreproc.PlotKvsTime()