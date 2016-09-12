#!/usr/bin/python

from bs4 import BeautifulSoup
import re


class Preprocessor:

	colIndex = 0
	wordCorpusToColIndexMap = {}

	nextPositionInFile = 0

	fileFeatureVector = open("/home/aj/dev/tryouts/FeatureVector", "w+")
	fileArticleSeekPosition = open("/home/aj/dev/tryouts/Article-SeekPosition", "w+")


	try1File = open("/home/aj/dev/tryouts/try1")

	soup = BeautifulSoup(try1File, "xml")

	stopWords = {"the":1, "for":1, "can":1, "could":1, "would":1,
				 "they":1, "there":1, "and":1}

	def __init__(self):
		a = 10

	def myTokenizer(self,text, stopWords):
		articleWiseWordFreqMap = {}
		listOfWords = re.findall("\w+", text)
		for word in listOfWords:
			if(self.isValidWord(word, stopWords)):
				self.addToCorpus(word.lower())
				# If Feature Vector is a map (position, count), get the position from the above line
				self.addToFreqMap(word.lower(), articleWiseWordFreqMap)
		return articleWiseWordFreqMap

	def containsDigits(self,word):
		return len(re.findall('\d+', word)) != 0

	def isValidWord(self,word, stopWords):
		return len(word) > 2 and word.lower() not in stopWords and not self.containsDigits(word)

	def addToFreqMap(self,word, dict):
		if(word not in dict):
			dict[word] = 1
		else:
			dict[word] += 1

	def addToCorpus(self,word):
		if(word not in Preprocessor.wordCorpusToColIndexMap):
			Preprocessor.wordCorpusToColIndexMap[word] = Preprocessor.colIndex
			Preprocessor.colIndex+=1
		return Preprocessor.wordCorpusToColIndexMap[word]

	def removeTags(body):
		return re.sub('<[^<>]+>', '', body)

	def storeToFiles(articleWiseWordFreqMap, articleId):
		storeArticleSeekPosition(articleId)
		nextPositionInFile = storeFeatureVector(articleWiseWordFreqMap)

	def storeArticleSeekPosition(articleId):
		fileArticleSeekPosition.write(str(articleId)+"-"+str(nextPositionInFile)+"\n")

	def storeFeatureVector(articleWiseWordFreqMap):
		for key, value in articleWiseWordFreqMap.iteritems():
			fileFeatureVector.write(key+"-"+str(value)+"\t")
		nextPositionInFile = fileFeatureVector.tell()	




p = Preprocessor()

for article in p.soup.find_all("REUTERS"):
	title = removeTags(article.TITLE.string)
	body = removeTags(article.BODY.string)
	articleId = article['NEWID']
	p.storeToFiles(myTokenizer(title + " " + body, stopWords), articleId)
	# myTokenizer(body, stopWords)

p.fileFeatureVector.close()
p.fileArticleSeekPosition.close()
#print len(tokenizedWords)

# myTokenizer(abcd, stopWords, tokenizedWords)
# print tokenizedWords	