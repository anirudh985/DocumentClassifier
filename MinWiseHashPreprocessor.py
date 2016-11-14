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


class MinWiseHashPreprocessor:
    baseFileName = 'http://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-'
    baseFileExtension = '.sgm'
    TotalDocs = 0
    listOfDocuments = []

    def __init__(self, kGramValue):
        self.kGramValue = kGramValue
        self.numOfHashFunctions = 10


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

                MinWiseHashPreprocessor.TotalDocs += 1
                MinWiseHashPreprocessor.listOfDocuments.append(title + " " + body)

    def removeTags(self, body):
        return re.sub('<[^<>]+>', '', body)


    def creatingKgramFeatureVector(self, k):
        # max_features=feature_count
        vectorizer = CountVectorizer(ngram_range=(k, k), stop_words='english', analyzer=u'word', binary=True, dtype=np.bool)
        self.kgramFeatureVector = vectorizer.fit_transform(MinWiseHashPreprocessor.listOfDocuments)
        self.maxFeatures = self.kgramFeatureVector.get_shape()[1]

    def generateRandomCoeffs(self):
        randomCoeffList = random.sample(xrange(0, self.maxFeatures), self.numOfHashFunctions)
        return randomCoeffList

    def createMinWiseSignatureMatrix(self):
        coeffA = self.generateRandomCoeffs()
        coeffB = self.generateRandomCoeffs()
        primeNumber = 2145
        self.signatureMatrix = []
        for i in xrange(MinWiseHashPreprocessor.TotalDocs):
            self.signatureMatrix.append(self.generateSingleSignatureMatrixRow(i, coeffA, coeffB, primeNumber))

    def generateSingleSignatureMatrixRow(self, docNumber, coeffA, coeffB, primeNumber):
        return map(self.generateMinwiseHash, zip(coeffA, coeffB, [primeNumber]*self.numOfHashFunctions, [docNumber]*self.numOfHashFunctions))

    def generateMinwiseHash(self, arg):
        minValue = sys.maxint
        for i in self.kgramFeatureVector[arg[3]].nonzero()[1]:
            hashValue = ((arg[0]*i+ arg[1]) % arg[2]) % self.maxFeatures
            if(hashValue < minValue):
                minValue = hashValue
        return minValue

minWiseHashPreproc = MinWiseHashPreprocessor(int(sys.argv[1]))
start_time = time.time()
# minWiseHashPreproc.init_file_parsing()
fileParsing_time = time.time()
print("Time taken for file parsing -- %s" % (fileParsing_time - start_time))
minWiseHashPreproc.creatingKgramFeatureVector(minWiseHashPreproc.kGramValue)
print("Time taken for creating k gram feature vector -- %s " % (time.time() - fileParsing_time))