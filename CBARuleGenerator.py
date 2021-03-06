from bs4 import BeautifulSoup
import re
import urllib2
import random
import time
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from itertools import izip, imap, ifilter, chain
from subprocess import check_call
import sys

class CBAClassifier:
    baseFileName = 'http://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-'
    baseFileExtension = '.sgm'

    def __init__(self, split):
        self.inputFileToCBAClassifier = open("./inputToCBAClassifier", "w+")
        self.testDataFileToCBAClassifier = open("./testInputToCBAClassifier", "w+")
        self.appearancesTextFile = open("./appearances.txt", "w+")
        self.testDataTopicsFile = open("./testDataRules", "w+")
        self.setOfTopics = set()
        self.inputToCBAClassifier = []
        self.testDataToCBAClassifier = []
        self.testDataOriginalClasses = []
        self.stemmer = SnowballStemmer("english")
        self.stopWords = self.createStopWords()
        self.split = split
        self.TotalDocs = 0

    def init_file_parsing(self):
        randomArticleIds = random.sample(xrange(0, 22000), (float(self.split)/100)*22000)
        for a in range(0, 22):
        # randomArticleIds = random.sample(xrange(0, 1000), 800)
        # for a in range(0, 1):
            file_name = self.baseFileName + str(a).zfill(3) + self.baseFileExtension
            print file_name
            file_handle = urllib2.urlopen(file_name)
            file_handle.readline()
            xml_content = '<a>'
            xml_content += file_handle.read()
            xml_content += '</a>'
            soup = BeautifulSoup(xml_content, "xml")
            for article in soup.find_all('REUTERS'):
                if article.TOPICS:
                    # topicsListIterator = self.topicsGenerator(article.TOPICS.find_all('D'))
                    topicsList = ["t_"+topic.string for topic in article.TOPICS.find_all('D')]
                    if(len(topicsList) == 0):
                        continue
                    self.setOfTopics = self.setOfTopics.union(topicsList)
                else:
                    continue

                articleId = int(article['NEWID'])
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

                self.TotalDocs += 1
                if (articleId in randomArticleIds):
                    self.inputToCBAClassifier.append(reduce(chain, [self.myTokenizer(body + " " + title), iter(topicsList)]))
                else:
                    self.testDataToCBAClassifier.append(self.myTokenizer(body + " " + title))
                    self.testDataOriginalClasses.append(topicsList)
        self.inputFileToCBAClassifier.write('\n'.join(imap(lambda x: ' '.join(x), self.inputToCBAClassifier)))
        self.testDataFileToCBAClassifier.write('\n'.join(imap(lambda x: ' '.join(x), self.testDataToCBAClassifier)))
        self.appearancesTextFile.write('antecedent\n' + '\n'.join(imap(lambda x: x + " consequent", iter(self.setOfTopics))))
        self.testDataTopicsFile.write('\n'.join(imap(lambda x: ' '.join(x), self.testDataOriginalClasses)))
        self.inputFileToCBAClassifier.close()
        self.testDataFileToCBAClassifier.close()
        self.appearancesTextFile.close()
        self.testDataTopicsFile.close()


    def removeTags(self, body):
        return re.sub('<[^<>]+>', '', body)

    def createStopWords(self):
        stopWordsList = stopwords.words('english')
        stopWordsList.append('reuter')
        stopWordsList.append('said')
        return dict((word, i) for (word, i) in izip(stopWordsList, range(1, len(stopWordsList) + 1)))

    def containsDigits(self, word):
        return len(re.findall('\d+', word)) != 0

    def isValidWord(self, word):
        return len(word) > 2 and word.lower() not in self.stopWords and not self.containsDigits(word)

    def myTokenizer(self, text):
        listOfWords = re.findall("\w+", text)
        return ifilter(self.isValidWord, imap(lambda x : self.stemmer.stem(x).lower(), listOfWords))

    def topicsGenerator(self, topicsList):
        for topic in topicsList:
            yield "t_"+topic.string


split = int(sys.argv[3])
support = "-s" + sys.argv[1]
confidence = "-c" + sys.argv[2]
cbaClassifier = CBAClassifier(split)
start_time = time.time()
cbaClassifier.init_file_parsing()
parsing_end_time = time.time()
print("Time taken to generate input Files ----- %s" %(parsing_end_time - start_time))
rulesFile = open("./rules.txt", "w+")
returnValue = check_call(['./apriori', '-tr', support, confidence, '-Rappearances.txt', 'inputToCBAClassifier', './rulesFile'])
print("\nTime taken to generate rules ----- %s" %(time.time() - parsing_end_time))
if(returnValue == 0):
    print "Success"
else:
    print "Failed"