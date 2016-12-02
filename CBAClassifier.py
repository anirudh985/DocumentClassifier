import time
from subprocess import check_call

parsing_end_time = time.time()
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
