from bs4 import BeautifulSoup
import re
import urllib2
import random
import time
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from itertools import izip, imap, ifilter, chain
from subprocess import check_call

parsing_end_time = time.time()
#rulesFile = open("./rules.txt", "w+")
returnValue = check_call(['./apriori', '-tr', '-s5', '-c10', '-Rappearances.txt', 'inputToCBAClassifier', './rulesFile'])
print("\nTime taken to generate rules ----- %s" %(time.time() - parsing_end_time))
if(returnValue == 0):
    print "Success"
else:
    print "Failed"
def parseRulesAndSort():
    with open("./rulesFile") as fileObject:
        for line in fileObject:
            rule = line.split(" ")
            


# cbaClassifier.predictClasses()