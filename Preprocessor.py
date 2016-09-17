#!/usr/bin/python

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
#from itertools import izip
from nltk.stem.snowball import SnowballStemmer
import re
import urllib2
import sys
import math

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
		self.fileFeatureVector = open("./FeatureVector", "w+")
		self.file_word_corpus_position = open('./wordPosition','w+')
		self.fileArticleSeekPosition = open("./Article-SeekPosition", "w+")

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
				self.storeToFiles(self.myTokenizer(title, body), articleId)
		self.dump_wordcorpus_columnindex_map()
		self.file_word_corpus_position.close()
		self.fileFeatureVector.close()
		self.fileArticleSeekPosition.close()

	def createStopWords(self):
		stopWordsList = set(stopwords.words('english'))
		return stopWordsList
			#dict((word, i) for (word, i) in izip(stopWordsList, range(1,len(stopWordsList)+1)))

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
		Preprocessor.nextPositionInFile = self.storeFeatureVector(articleWiseWordFreqMap)

	def storeArticleSeekPosition(self,articleId):
		self.fileArticleSeekPosition.write(str(articleId)+"-"+str(Preprocessor.nextPositionInFile)+"\n")

	def storeFeatureVector(self,articleWiseWordFreqMap):
		outputString = ""
		for key, value in articleWiseWordFreqMap.iteritems():
			outputString += key+'-'+str(value)+'\t'
		outputString = outputString[:-1]
		outputString += '\n'
		self.fileFeatureVector.write(outputString)
		return self.fileFeatureVector.tell()

	def dump_wordcorpus_columnindex_map(self):
		for key, value in Preprocessor.wordCorpusToColIndexMap.iteritems():
			self.file_word_corpus_position.write(key+"-"+str(value)+"\t")
		self.file_word_corpus_position.write('\n')

	def load_wordcorpus_columnindex_map(self):
		with open("./wordPosition") as fileobject:
			for line in fileobject:
				(key, value) = line.split('-')
				Preprocessor.wordCorpusToColIndexMap[key] = int(value)
				if key not in Preprocessor.wordToNumDocsMap.keys:
					Preprocessor.wordToNumDocsMap[key] = 1
				else:
					Preprocessor.wordToNumDocsMap[key] += 1
			Preprocessor.TotalDocs += 1

	def construct_feature_vetor_matrix(self, doLoadWordCorpus = 'N', isFeatureTfIdf = 0):
		if doLoadWordCorpus != 'Y' and doLoadWordCorpus != 'y':
			self.load_wordcorpus_columnindex_map()
		self.fileFinalFeatureVector = open("./FinalFeatureVector", "w+")
		corpus_length = len(Preprocessor.wordCorpusToColIndexMap)
		with open("./FeatureVector") as fileobject:
			for line in fileobject:
				vector = [0] * corpus_length
				words = line.split('\t')
				for word in words:
					wordRoots = word.split('-')
					if wordRoots[0] in Preprocessor.wordCorpusToColIndexMap and Preprocessor.wordCorpusToColIndexMap[wordRoots[0]] < corpus_length:
						if isFeatureTfIdf:
							tf = wordRoots[1]
							idf = 0
							if Preprocessor.TotalDocs > 0 and Preprocessor.wordToNumDocsMap[wordRoots[0]] > 0:
								idf = math.log(Preprocessor.TotalDocs/Preprocessor.wordToNumDocsMap[wordRoots[0]])
							vector[Preprocessor.wordCorpusToColIndexMap[wordRoots[0]]] = tf * idf
						else:
							vector[Preprocessor.wordCorpusToColIndexMap[wordRoots[0]]] = wordRoots[1]
					else:
						print wordRoots
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
if(sys.argv[1] == 'Y' or sys.argv[1] == 'y'):
	newPrep.init_file_parsing()
generateTfIdf = (sys.argv[2] == 'Y') or (sys.argv[2] == 'y')
newPrep.construct_feature_vetor_matrix(sys.argv[1],generateTfIdf)
# newPrep.construct_feature_vetor_matrix()
	# print len(tokenizedWords)

	# myTokenizer(abcd, stopWords, tokenizedWords)
	# print tokenizedWords
