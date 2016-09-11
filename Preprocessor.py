#!/usr/bin/python

from bs4 import BeautifulSoup
import re

class Preprocessor:

	colIndex = 0
	wordCorpusToColIndexMap = {}


	try1File = open("/home/aj/dev/tryouts/try1")

	soup = BeautifulSoup(try1File, "xml")

	stopWords = {"the":1, "for":1, "can":1, "could":1, "would":1,
				 "they":1, "there":1, "and":1}

	def myTokenizer(text, stopWords, tokenizedWords):
		articleWiseWordFreqMap = {}
		listOfWords = re.findall("\w+", text)
		for word in listOfWords:
			if(isValidWord(word, stopWords)):
				addToCorpus(word.lower())
				# If Feature Vector is a map (position, count), get the position from the above line
				addToFreqMap(word.lower(), articleWiseWordFreqMap)
				
		return articleWiseWordFreqMap

	def containsDigits(word):
		return len(re.findall('\d+', word)) != 0

	def isValidWord(word, stopWords):
		return len(word) > 2 and word.lower() not in stopWords and not containsDigits(word)

	def addToDict(word, dict):
		if(word not in dict):
			dict[word] = 1
		else:
			dict[word] += 1
		# return colIndex

	def addToCorpus(word):
		if(word not in wordCorpusToColIndexMap):
			wordCorpusToColIndexMap[word] = colIndex
			colIndex++
		return wordCorpusToColIndexMap[word]

	def removeTags(body):
		return re.sub('<[^<>]+>', '', body)

	abcd = """Stop by Tech Hub, 247.565 Ohio State's official technology store for exclusive
			discounts on computers, tablets and software. A portion of Tech Hub sales
			go back to student programs"""



	for article in soup.find_all("REUTERS"):
		title = removeTags(article.TITLE.string)
		body = removeTags(article.BODY.string)
		myTokenizer(title, stopWords, tokenizedWords)
		myTokenizer(body, stopWords, tokenizedWords)

	print len(tokenizedWords)

	# myTokenizer(abcd, stopWords, tokenizedWords)
	# print tokenizedWords	

