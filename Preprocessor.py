#!/usr/bin/python

from bs4 import BeautifulSoup
import re


class Preprocessor:
	colIndex = 0
	wordCorpusToColIndexMap = {}
	nextPositionInFile = 0
	fileFeatureVector = open("/home/aj/dev/tryouts/FeatureVector", "w+")
	fileArticleSeekPosition = open("/home/aj/dev/tryouts/Article-SeekPosition", "w+")

	baseFileName = 'http://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-'
	baseFileExtension = '.sgm'
	stopWords = {'the': 1, 'for': 1, 'can': 1, 'could': 1, 'would': 1,'they': 1, 'there': 1, 'and': 1}

	def __init__(self):
		a = 10

	def init_file_parsing(self):
		for a in range(0,1):
			file_name = self.baseFileName + str(a).zfill(3) + self.baseFileExtension
			file_name = '/Users/kalyan/Downloads/reut2-000.sgm'
			print file_name
			file_handle = open(file_name,'r')
			file_handle.readline()
			xml_content =  '<a>'
			xml_content += file_handle.read()
			xml_content += '</a>'
			#print xml_content
			soup = BeautifulSoup(xml_content, "xml")
			#print soup.prettify()
			for article in soup.find_all('REUTERS'):
				if article.TITLE:
					print article.TITLE.string;
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
				self.storeToFiles(self.myTokenizer(title + " " + body, Preprocessor.stopWords), articleId)
		Preprocessor.fileFeatureVector.close()
		Preprocessor.fileArticleSeekPosition.close()

	def myTokenizer(self, text, stopWords):
		articleWiseWordFreqMap = {}
		listOfWords = re.findall("\w+", text)
		for word in listOfWords:
			if (self.isValidWord(word, stopWords)):
				self.addToCorpus(word.lower())
				# If Feature Vector is a map (position, count), get the position from the above line
				self.addToFreqMap(word.lower(), articleWiseWordFreqMap)
		return articleWiseWordFreqMap

	def containsDigits(self, word):
		return len(re.findall('\d+', word)) != 0

	def isValidWord(self, word, stopWords):
		return len(word) > 2 and word.lower() not in stopWords and not self.containsDigits(word)

	def addToFreqMap(self, word, dict):
		if (word not in dict):
			dict[word] = 1
		else:
			dict[word] += 1

	def addToCorpus(self, word):
		if (word not in Preprocessor.wordCorpusToColIndexMap):
			Preprocessor.wordCorpusToColIndexMap[word] = Preprocessor.colIndex
			Preprocessor.colIndex += 1
		return Preprocessor.wordCorpusToColIndexMap[word]

	def removeTags(self,body):
		return re.sub('<[^<>]+>', '', body)

	def storeToFiles(self,articleWiseWordFreqMap, articleId):
		Preprocessor.storeArticleSeekPosition(articleId)
		nextPositionInFile = self.storeFeatureVector(articleWiseWordFreqMap)

	def storeArticleSeekPosition(self,articleId):
		Preprocessor.fileArticleSeekPosition.write(str(articleId)+"-"+str(Preprocessor.nextPositionInFile)+"\n")

	def storeFeatureVector(self,articleWiseWordFreqMap):
		for key, value in articleWiseWordFreqMap.iteritems():
			Preprocessor.fileFeatureVector.write(key+"-"+str(value)+"\t")
		nextPositionInFile = Preprocessor.fileFeatureVector.tell()


newPrep = Preprocessor()
newPrep.init_file_parsing()
	# print len(tokenizedWords)

	# myTokenizer(abcd, stopWords, tokenizedWords)
	# print tokenizedWords
