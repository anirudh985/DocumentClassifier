#!/usr/bin/python

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from itertools import izip
from nltk.stem.snowball import SnowballStemmer
import re
import urllib2
import sys
import math
import time
from __future__ import division

class Preprocessor:
	colIndex = 0
	wordCorpusToColIndexMap = {}
	wordToNumDocsMap = {}
	nextPositionInFile = 0
	TotalDocs = 0
	stemmer = SnowballStemmer("english")
	baseFileName = 'http://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-'
	baseFileExtension = '.sgm'
	# stop_words = set(stopwords.words('english'))
	# stopWords = {'the': 1, 'for': 1, 'can': 1, 'could': 1, 'would': 1,'they': 1, 'there': 1, 'and': 1}

	def __init__(self):
		if(sys.argv[1] == 'Y' or sys.argv[1] == 'y'):
			self.fileFeatureVector = open("./FeatureVector", "r")
			self.file_word_corpus_position = open('./wordPosition','r')
			self.file_word_numDocs = open('./wordNumDocs', 'r')
		else:
			self.fileFeatureVector = open("./FeatureVector", "w+")
			self.file_word_corpus_position = open('./wordPosition', 'w+')
			self.file_word_numDocs = open('./wordNumDocs','w+')
		# self.fileArticleSeekPosition = open("./Article-SeekPosition", "w+")

	def init_file_parsing(self):
		self.stopWords = self.createStopWords()
		for a in range(0,21):
			file_name = self.baseFileName + str(a).zfill(3) + self.baseFileExtension
			# file_name = '/Users/kalyan/Downloads/reut2-000.sgm'
			print file_name
			file_handle = urllib2.urlopen(file_name)
			file_handle.readline()
			xml_content =  '<a>'
			xml_content += file_handle.read()
			xml_content += '</a>'
			#print xml_content
			soup = BeautifulSoup(xml_content, "xml")
			#print soup.prettify()
			for article in soup.find_all('REUTERS'):
				if article.TITLE:
					#print article.TITLE.string
					title = self.removeTags(article.TITLE.string)
					#print self.myTokenizer(title, self.stopWords)
				else:
					title = ''
				if article.BODY:
					body = self.removeTags(article.BODY.string)
				else:
					body = ''
					# print self.myTokenizer(body, self.stopWords)
				articleId = article['NEWID']
				Preprocessor.TotalDocs += 1
				# self.storeToFiles(self.myTokenizer(title, body), articleId)
				self.storeFeatureVector(self.myTokenizer(title, body))
		self.dump_wordcorpus_columnindex_map()
		self.dump_word_to_numDocs_map()
		self.file_word_corpus_position.close()
		self.fileFeatureVector.close()
		# self.fileArticleSeekPosition.close()

	def createStopWords(self):
		# stopWordsList = set(stopwords.words('english'))
		stopWordsList = stopwords.words('english')
		# return stopWordsList
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

	# def storeToFiles(self,articleWiseWordFreqMap, articleId):
	# 	self.storeArticleSeekPosition(articleId)
	# 	Preprocessor.nextPositionInFile = self.storeFeatureVector(articleWiseWordFreqMap)
    #
	# def storeArticleSeekPosition(self,articleId):
	# 	self.fileArticleSeekPosition.write(str(articleId)+"-"+str(Preprocessor.nextPositionInFile)+"\n")

	def storeFeatureVector(self,articleWiseWordFreqMap):
		outputString = ""
		for key, value in articleWiseWordFreqMap.iteritems():
			outputString += key+'-'+str(value)+'\t'
		outputString = outputString[:-1]
		outputString += '\n'
		self.fileFeatureVector.write(outputString)
		# return self.fileFeatureVector.tell()

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

	def loadLineByLineFileIntoMap(self, filePath, someMap, splitSeparator):
		with open(filePath) as fileobject:
			headerValue = int(fileobject.readline())
			for line in fileobject:
				(key, value) = line.split(splitSeparator)
				someMap[key] = int(value)
		return headerValue

	def loadSingleLineFileIntoMap(self, filePath, someMap, outerSplitSeparator, innerSplitSeparator):
		with open(filePath) as fileobject:
			headerValue = int(fileobject.readline())
			entireData = fileobject.readline()
		for pair in entireData.split(outerSplitSeparator):
			(key, value) = pair.split(innerSplitSeparator)
			someMap[key] = int(value)
		return headerValue

	def loadStoredMaps(self):
		self.loadSingleLineFileIntoMap("./wordPosition", Preprocessor.wordCorpusToColIndexMap, '\t', '-')
		Preprocessor.TotalDocs = self.loadSingleLineFileIntoMap("./wordNumDocs", Preprocessor.wordToNumDocsMap, '\t', '-')


	def construct_feature_vetor_matrix(self, doNotLoadStoredMaps = 'Y', isFeatureTfIdf = 0):
		if not (doNotLoadStoredMaps == 'Y' or doNotLoadStoredMaps == 'y'):
			self.loadStoredMaps()
		if isFeatureTfIdf:
			fileName="./FinalFeatureVectorTFIDF"
		else:
			fileName = "./FinalFeatureVector"
		self.fileFinalFeatureVector = open(fileName, "w+")
		corpus_length = len(Preprocessor.wordCorpusToColIndexMap)
		with open("./FeatureVector") as fileobject:
			for line in fileobject:
				vector = [0] * corpus_length
				words = line.split('\t')
				if(len(words) > 1):
					for wordFreqPair in words:
						(word, wordFreq) = wordFreqPair.split('-')
						if word in Preprocessor.wordCorpusToColIndexMap and Preprocessor.wordCorpusToColIndexMap[word] < corpus_length:
							if isFeatureTfIdf:
								tf = wordFreq
								idf = 0
								if Preprocessor.TotalDocs > 0 and Preprocessor.wordToNumDocsMap[word] > 0:
									idf = math.log(Preprocessor.TotalDocs/Preprocessor.wordToNumDocsMap[word])
								vector[Preprocessor.wordCorpusToColIndexMap[word]] = tf * idf
							else:
								vector[Preprocessor.wordCorpusToColIndexMap[word]] = wordFreq
						# else:
						# 	print (word, wordFreq)
				outputString = ''
				for column in vector:
					outputString += str(column)
					outputString += '\t'
				outputString = outputString[:-1]
				outputString+= '\n'
				self.fileFinalFeatureVector.write(outputString)
				# self.fileFinalFeatureVector.write('\n')
		self.fileFinalFeatureVector.close()




newPrep = Preprocessor()
start_time = time.time()
if(sys.argv[1] == 'Y' or sys.argv[1] == 'y'):
	newPrep.init_file_parsing()
generateTfIdf = (sys.argv[2] == 'Y') or (sys.argv[2] == 'y')
newPrep.construct_feature_vetor_matrix(sys.argv[1],generateTfIdf)
print("--- %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# newPrep.init_file_parsing()
# generateTfIdf = 0
# newPrep.construct_feature_vetor_matrix('Y',generateTfIdf)
# print("--- %s seconds ---" % (time.time() - start_time))
