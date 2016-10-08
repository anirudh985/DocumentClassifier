#!/usr/bin/python

from __future__ import division
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from itertools import izip
from nltk.stem.snowball import SnowballStemmer
import re
import urllib2
import sys
import math
import time

class Preprocessor:
	colIndex = 0
	wordCorpusToColIndexMap = {}
	wordToNumDocsMap = {}
	articleIdTopicsClassLabelMap = {}
	articleIdPlacesClassLabelMap = {}
	nextPositionInFile = 0
	TotalDocs = 0
	stemmer = SnowballStemmer("english")
	baseFileName = 'http://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-'
	baseFileExtension = '.sgm'

	def __init__(self, doParse):
		if(doParse == 'Y' or doParse == 'y'):
			self.fileFeatureVector = open("./BagOfWords", "w+")
			self.file_word_corpus_position = open('./wordPosition','w+')
			self.file_word_numDocs = open('./wordNumDocs', 'w+')
			self.file_topics_classlabels = open('./articleTopicsClassLabels', 'w+')
			self.file_places_classlabels = open('./articlePlacesClassLabels', 'w+')
			self.fileArticleSeekPosition = open("./ArticleIdSeekPosition", "w+")
		# else:
		# 	self.fileFeatureVector = open("./BagOfWords", "r")
		# 	self.file_word_corpus_position = open('./wordPosition', 'r')
		# 	self.file_word_numDocs = open('./wordNumDocs','r')
		# 	self.fileArticleSeekPosition = open("./ArticleIdSeekPosition", "r")
			# print "process started"

	def init_file_parsing(self):
		self.stopWords = self.createStopWords()
		for a in range(0,22):
			file_name = self.baseFileName + str(a).zfill(3) + self.baseFileExtension
			print file_name
			file_handle = urllib2.urlopen(file_name)
			file_handle.readline()
			xml_content =  '<a>'
			xml_content += file_handle.read()
			xml_content += '</a>'
			soup = BeautifulSoup(xml_content, "xml")
			for article in soup.find_all('REUTERS'):
				articleId = article['NEWID']
				title=''
				body=''
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
				if(title == '' and body == ''):
					continue

				if article.TOPICS:
					topicsClassLabels = ""
					for topic in article.TOPICS.find_all('D'):
						topicsClassLabels += topic.string + ","
					topicsClassLabels = topicsClassLabels[:-1]
					Preprocessor.articleIdTopicsClassLabelMap[articleId] = topicsClassLabels

				if article.PLACES:
					placeClassLabels = ""
					for place in article.PLACES.find_all('D'):
						placeClassLabels += place.string + ","
					placeClassLabels = placeClassLabels[:-1]
					Preprocessor.articleIdPlacesClassLabelMap[articleId] = placeClassLabels

				Preprocessor.TotalDocs += 1
				self.storeToFiles(self.myTokenizer(title, body), articleId)
				# self.storeFeatureVector(self.myTokenizer(title, body), articleId)
		self.fileFeatureVector.close()
		self.dump_wordcorpus_columnindex_map()
		self.file_word_corpus_position.close()
		self.dump_word_to_numDocs_map()
		self.file_word_numDocs.close()
		self.dump_articleId_TopicsClassLabels_map()
		self.file_topics_classlabels.close()
		self.dump_articleId_PlacesClassLabels_map()
		self.file_places_classlabels.close()


		self.fileArticleSeekPosition.close()

	def createStopWords(self):
		stopWordsList = stopwords.words('english')
		return dict((word, i) for (word, i) in izip(stopWordsList, range(1,len(stopWordsList)+1)))

	def update_word_to_doc_map(self,articleWiseWordFreqMap):
		for key, value in articleWiseWordFreqMap.iteritems():
			if key not in Preprocessor.wordToNumDocsMap:
				Preprocessor.wordToNumDocsMap[key] = 1
			else:
				Preprocessor.wordToNumDocsMap[key] += 1


	def myTokenizer(self, title, body):
		articleWiseWordFreqMap = {}
		listOfTitleWords = re.findall("\w+", title)
		listOfBodyWords = re.findall("\w+", body)
		titleWeight = 5
		bodyTextWeight = 1
		self.processWordsAndAddToFreqMap(listOfTitleWords, titleWeight, articleWiseWordFreqMap)
		self.processWordsAndAddToFreqMap(listOfBodyWords, bodyTextWeight, articleWiseWordFreqMap)
		self.update_word_to_doc_map(articleWiseWordFreqMap)
		return articleWiseWordFreqMap

	def processWordsAndAddToFreqMap(self, listOfWords, weightOfWord, articleWiseWordFreqMap):
		for word in listOfWords:
			word_stem = Preprocessor.stemmer.stem(word)
			if self.isValidWord(word_stem):
				self.addToCorpus(word_stem.lower())
				# If Feature Vector is a map (position, count), get the position from the above line
				self.addToFreqMap(word_stem.lower(), articleWiseWordFreqMap, weightOfWord)

	def containsDigits(self, word):
		return len(re.findall('\d+', word)) != 0

	def isValidWord(self, word):
		return len(word) > 2 and word.lower() not in self.stopWords and not self.containsDigits(word)

	def addToFreqMap(self, word, dict, weight):
		if word not in dict:
			dict[word] = weight
		else:
			dict[word] += weight

	def addToCorpus(self, word):
		if word not in Preprocessor.wordCorpusToColIndexMap:
			Preprocessor.wordCorpusToColIndexMap[word] = Preprocessor.colIndex
			Preprocessor.colIndex += 1
		return Preprocessor.wordCorpusToColIndexMap[word]

	def removeTags(self,body):
		return re.sub('<[^<>]+>', '', body)

	def storeToFiles(self,articleWiseWordFreqMap, articleId):
		self.storeArticleSeekPosition(articleId)
		Preprocessor.nextPositionInFile = self.storeFeatureVector(articleWiseWordFreqMap, articleId)

	def storeArticleSeekPosition(self,articleId):
	 	self.fileArticleSeekPosition.write(str(articleId)+"-"+str(Preprocessor.nextPositionInFile)+"\n")

	def storeFeatureVector(self,articleWiseWordFreqMap, articleId):
		outputString = str(articleId)+"@"
		for key, value in articleWiseWordFreqMap.iteritems():
			outputString += key+'-'+str(value)+'\t'
		if(outputString[-1] != '@'):
			outputString = outputString[:-1]
		outputString += '\n'
		self.fileFeatureVector.write(outputString)
		return self.fileFeatureVector.tell()

	def dumpFile(self, someMap, filehandle, outerSplitSeparator, innerSplitSeparator = '-', headerLine = None):
		if(headerLine != None):
			filehandle.write(str(headerLine)+"\n")
		outputString = ""
		for key, value in someMap.iteritems():
			outputString += key + innerSplitSeparator + str(value) + outerSplitSeparator
		outputString = outputString[:-1]
		filehandle.write(outputString)

	def dump_wordcorpus_columnindex_map(self):
		self.dumpFile(Preprocessor.wordCorpusToColIndexMap, self.file_word_corpus_position, outerSplitSeparator="\t", headerLine=Preprocessor.TotalDocs)

	def dump_word_to_numDocs_map(self):
		self.dumpFile(Preprocessor.wordToNumDocsMap, self.file_word_numDocs, outerSplitSeparator="\t", headerLine=Preprocessor.TotalDocs)

	def dump_articleId_TopicsClassLabels_map(self):
		self.dumpFile(Preprocessor.articleIdTopicsClassLabelMap, self.file_topics_classlabels, outerSplitSeparator="\t", innerSplitSeparator="@")

	def dump_articleId_PlacesClassLabels_map(self):
		self.dumpFile(Preprocessor.articleIdPlacesClassLabelMap, self.file_places_classlabels, outerSplitSeparator="\t", innerSplitSeparator="@")

	def loadLineByLineFileIntoMap(self, filePath, someMap, splitSeparator):
		with open(filePath) as fileobject:
			# headerValue = int(fileobject.readline())
			for line in fileobject:
				(key, value) = line.split(splitSeparator)
				someMap[key] = int(value)
		# return headerValue

	def loadSingleLineFileIntoMap(self, filePath, someMap, outerSplitSeparator, innerSplitSeparator, methodToConvertValue = int, headerIncluded=True):
		with open(filePath) as fileobject:
			headerValue = -1
			if(headerIncluded):
				headerValue = int(fileobject.readline())
			entireData = fileobject.readline()
		for pair in entireData.split(outerSplitSeparator):
			(key, value) = pair.split(innerSplitSeparator)
			someMap[key] = methodToConvertValue(value)
		return headerValue

	def loadStoredMaps(self):
		self.loadSingleLineFileIntoMap("./wordPosition", Preprocessor.wordCorpusToColIndexMap, '\t', '-')
		Preprocessor.TotalDocs = self.loadSingleLineFileIntoMap("./wordNumDocs", Preprocessor.wordToNumDocsMap, '\t', '-')


	def construct_feature_vetor_matrix(self, doNotLoadStoredMaps = 'Y', isFeatureTfIdf = False):
		if not (doNotLoadStoredMaps == 'Y' or doNotLoadStoredMaps == 'y'):
			self.loadStoredMaps()
		if isFeatureTfIdf:
			fileName="./FinalFeatureVectorTFIDF"
		else:
			fileName = "./FinalFeatureVector"
		self.fileFinalFeatureVector = open(fileName, "w+")
		corpus_length = len(Preprocessor.wordCorpusToColIndexMap)
		with open("./BagOfWords") as fileobject:
			for line in fileobject:
				self.dump_single_feature_vector(self.construct_single_feature_vector(line, corpus_length, isFeatureTfIdf))
		self.fileFinalFeatureVector.close()


	def construct_single_feature_vector(self, line, featureVectorLength, isFeatureTfIdf=False):
		vector = [0] * featureVectorLength
		(articleId, wordsTabSeparated) = line.split('@')
		if(wordsTabSeparated != None and wordsTabSeparated != ""):
			words = wordsTabSeparated.split('\t')
			if (len(words) > 1):
				for wordFreqPair in words:
					(word, wordFreq) = wordFreqPair.split('-')
					if word in Preprocessor.wordCorpusToColIndexMap and Preprocessor.wordCorpusToColIndexMap[word] < featureVectorLength:
						if isFeatureTfIdf:
							tf = int(wordFreq)
							idf = 0
							if Preprocessor.TotalDocs > 0 and Preprocessor.wordToNumDocsMap[word] > 0:
								idf = math.log(Preprocessor.TotalDocs / Preprocessor.wordToNumDocsMap[word])
							vector[Preprocessor.wordCorpusToColIndexMap[word]] = tf * idf
						else:
							vector[Preprocessor.wordCorpusToColIndexMap[word]] = int(wordFreq)
		return vector


	# you need to open 'self.fileFinalFeatureVector' before calling this method
	def dump_single_feature_vector(self, vector):
		outputString = ''
		for column in vector:
			outputString += str(column)
			outputString += '\t'
		outputString = outputString[:-1]
		outputString += '\n'
		self.fileFinalFeatureVector.write(outputString)



	def construct_feature_vetor_matrix_for_naive_bayes(self, isFeatureTfIdf=False):
		self.Y_Labels = []
		self.X = []
		self.loadStoredMaps()
		if isFeatureTfIdf:
			fileName = "./FinalFeatureVectorNaiveBayesTFIDF"
		else:
			fileName = "./FinalFeatureVectorNaiveBayes"
		self.fileFinalFeatureVector = open(fileName, "w+")
		corpus_length = len(Preprocessor.wordCorpusToColIndexMap)
		fileBagOfWords = open("./BagOfWords", "r")

		#getting articleIdSeekPosition into a map
		articleIdSeekPosition = {}
		self.loadLineByLineFileIntoMap("./ArticleIdSeekPosition", articleIdSeekPosition, '-')

		#loading topics file into map
		self.loadSingleLineFileIntoMap("./articleTopicsClassLabels", Preprocessor.articleIdTopicsClassLabelMap, '\t', '@', methodToConvertValue=str, headerIncluded=False)

		#prepare the feature vector and class label vector
		with open('./articleTopicsClassLabels') as fileobject:
			entireData = fileobject.readline()
			for pair in entireData.split('\t'):
				(key, value) = pair.split('@')
				if (value != None and value != ""):
					topics = value.split(',')
					self.Y_Labels.append(topics)

					#construct the featurevector here and add it to the new file
					fileBagOfWords.seek(articleIdSeekPosition[key])
					self.X.append(self.construct_single_feature_vector(fileBagOfWords.readline(), corpus_length, isFeatureTfIdf))
					if(len(self.X) == 1000):
						break
					# else:
					# 	print key

		fileBagOfWords.close()






# newPrep = Preprocessor(sys.argv[1])
# newPrep = Preprocessor('Y')
# start_time = time.time()
# # if(sys.argv[1] == 'Y' or sys.argv[1] == 'y'):
# newPrep.init_file_parsing()
# # generateTfIdf = (sys.argv[2] == 'Y') or (sys.argv[2] == 'y')
# # newPrep.construct_feature_vetor_matrix(sys.argv[1],generateTfIdf)
# newPrep.construct_feature_vetor_matrix('Y', False)
# print("--- %s seconds ---" % (time.time() - start_time))
