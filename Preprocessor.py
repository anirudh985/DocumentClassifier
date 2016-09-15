#!/usr/bin/python

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
#from itertools import izip
from nltk.stem.snowball import SnowballStemmer
import re
import urllib2

class Preprocessor:
	colIndex = 0
	wordCorpusToColIndexMap = {}
	nextPositionInFile = 0
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
		stopWords = self.createStopWords()
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
				self.storeToFiles(self.myTokenizer(title + " " + body, stopWords), articleId)
		self.dump_wordcorpus_colunmindex_map()
		self.file_word_corpus_position.close()
		self.fileFeatureVector.close()
		self.fileArticleSeekPosition.close()
		self.construct_feature_vetor_matrix()

	def createStopWords(self):
		stopWordsList = set(stopwords.words('english'))
		return stopWordsList
			#dict((word, i) for (word, i) in izip(stopWordsList, range(1,len(stopWordsList)+1)))

	def myTokenizer(self, text, stopWords):
		articleWiseWordFreqMap = {}
		listOfWords = re.findall("\w+", text)
		for word in listOfWords:
			word_stem = Preprocessor.stemmer.stem(word)
			if self.isValidWord(word_stem, stopWords):
				self.addToCorpus(word_stem.lower())
				# If Feature Vector is a map (position, count), get the position from the above line
				self.addToFreqMap(word_stem.lower(), articleWiseWordFreqMap)
		return articleWiseWordFreqMap

	def containsDigits(self, word):
		return len(re.findall('\d+', word)) != 0

	def isValidWord(self, word, stopWords):
		return len(word) > 2 and word.lower() not in stopWords and not self.containsDigits(word)

	def addToFreqMap(self, word, dict):
		if word not in dict:
			dict[word] = 1
		else:
			dict[word] += 1

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
		for key, value in articleWiseWordFreqMap.iteritems():
			self.fileFeatureVector.write(key+"-"+str(value)+"\t")
		self.fileFeatureVector.write("\n")
		return self.fileFeatureVector.tell()

	def dump_wordcorpus_colunmindex_map(self):
		for key, value in Preprocessor.wordCorpusToColIndexMap.iteritems():
			self.file_word_corpus_position.write(key+"-"+str(value)+"\t")
		self.file_word_corpus_position.write('\n')

	def construct_feature_vetor_matrix(self):
		self.fileFinalFeatureVector = open("./FinalFeatureVector", "w+")
		corpus_length = len(Preprocessor.wordCorpusToColIndexMap)
		with open("./FeatureVector") as fileobject:
			for line in fileobject:
				vector = [0] * corpus_length
				words = line.split('\t')
				for word in words:
					wordRoots = word.split('-')
					if wordRoots[0] in Preprocessor.wordCorpusToColIndexMap and Preprocessor.wordCorpusToColIndexMap[wordRoots[0]] < corpus_length:
						vector[Preprocessor.wordCorpusToColIndexMap[wordRoots[0]]] = wordRoots[1]
					else:
						print wordRoots
				outputString = ''
				for column in vector:
					outputString += str(column)
					outputString += '\t'
				self.fileFinalFeatureVector.write(outputString)
				self.fileFinalFeatureVector.write('\n')
		self.fileFinalFeatureVector.close();




newPrep = Preprocessor()
newPrep.init_file_parsing()
	# print len(tokenizedWords)

	# myTokenizer(abcd, stopWords, tokenizedWords)
	# print tokenizedWords
